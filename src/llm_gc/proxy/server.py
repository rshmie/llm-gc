"""The HTTP proxy: GC on the request path.

An app points its SDK here instead of at the provider and changes nothing else:

    client sends the full conversation
      -> the session is reconciled with it: continued, rewound, or started over
      -> turns it has not seen are recorded
      -> forward the recap, then the client's own recent turns and the new one
      -> return the provider's answer unchanged
      -> after responding, a pass ages the session; a cut changes the next request

Any failure in our own machinery forwards the client's original request untouched.
See doc/design/proxy/overview.md.
"""

import asyncio
import contextlib
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, replace
from typing import Any

import httpx2
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from llm_gc.config.constants import APP_VERSION
from llm_gc.engine.gc_result import GCResult
from llm_gc.engine.gc_service import GcService
from llm_gc.engine.gc_status import GCStatus
from llm_gc.events import Event, EventType
from llm_gc.exceptions import ConfigError
from llm_gc.llm.metering import ModelUsage
from llm_gc.models import ClientOrigin, Message
from llm_gc.models.message import tool_result_ids
from llm_gc.models.generation_snapshot import GenerationSnapshot
from llm_gc.proxy.anthropic_translation import (
    Instruction,
    carry_instructions,
    extract_instructions,
    ReplyOutline,
    StreamOutlineReader,
    ThinkingSizer,
    cache_lifetime_s,
    extract_messages,
    fixed_prompt_tokens,
    forwardable_headers,
    instruction_tokens,
    read_reply,
    refusal_reason,
    reply_text,
    rewrite_messages,
    sends_thinking,
    sent_thinking_tokens,
)
from llm_gc.proxy.cache_watch import CacheWatch
from llm_gc.proxy.model_windows import ModelWindow, ModelWindows, WindowLookup
from llm_gc.proxy.session_identity import SESSION_HEADER, chain_hashes, client_origin
from llm_gc.proxy.reply_thinking import ReplyThinking
from llm_gc.proxy.request_log import RequestLog, RequestRecord, newest_kind
from llm_gc.proxy.subagent_links import SpawnLinker
from llm_gc.proxy.traffic import TrafficLedger, token_rate

logger = logging.getLogger(__name__)

ANTHROPIC_UPSTREAM = "https://api.anthropic.com"
MESSAGES_PATH = "/v1/messages"
DEFAULT_PROXY_PORT = 9900

# Generous, and a ceiling rather than a target. A long generation legitimately
# takes minutes, and a proxy that times out before the provider does turns a slow
# success into a failure the client cannot distinguish from a real one.
DEFAULT_UPSTREAM_TIMEOUT_S = 600.0

# The provider's own request limit. A larger body would be rejected upstream
# anyway, so refusing it here costs no user anything, and it stops one request from
# making the proxy buffer an unbounded body.
MAX_REQUEST_BYTES = 32 * 1024 * 1024

# A streamed reply longer than this is relayed whole but not kept to show on the
# dashboard: the copy is held until the stream ends, once per request in flight.
_REPLY_KEPT_MAX_BYTES = 4 * 1024 * 1024

# A model name longer than this is not a model name; it is not kept or shown.
_MODEL_NAME_MAX_CHARS = 128

# Answers that say the upstream is under pressure: rate limited, or overloaded. LLM-GC's own model
# calls share it with the client, so they pause rather than add to it.
_PRESSURE_STATUSES = {429: "rate-limited", 529: "said it was overloaded on"}
# The provider's answer to a request it will not run as sent. Decided before any output, so nothing was billed.
_REFUSED_AS_INVALID = 400

# What to do with a reply once it ends: its outline, and the prompts its tool calls gave subagents.
OutlineHandler = Callable[[ReplyOutline, Mapping[str, str]], None]

# Response headers that describe the upstream connection, not its content.
# Returning them would have us promise an encoding or a length that no longer
# applies to the body we hand back.
_RESPONSE_HOP_BY_HOP = frozenset({
    "content-length", "content-encoding", "transfer-encoding", "connection",
    "keep-alive", "upgrade", "trailer",
})

# What the log says about a failed answer. The client sees the upstream's own error; these
# point whoever runs LLM-GC at the usual cause on its side.
_ERROR_STATUS_HINT = "the client gets the upstream's error unchanged"
_CREDENTIAL_REFUSED_HINT = ("the upstream refused the client's credential. Behind a gateway, check that "
                            "`upstream` is the gateway's address")
_ERROR_STATUS_HINTS = {
    401: _CREDENTIAL_REFUSED_HINT,
    403: _CREDENTIAL_REFUSED_HINT,
    502: "the upstream failed or could not be reached",
}


class ProxyMetrics:
    """Counters for `GET /stats`, and the measured traffic behind them. No conversation content."""

    def __init__(self) -> None:
        self.requests = 0
        self.gc_applied = 0
        self.gc_bypassed = 0
        # Reconciliations with the client's history. Each restart or seed costs one
        # uncompressed request, so they are counted rather than hidden.
        self.history_rewinds = 0
        self.history_restarts = 0
        self.sessions_seeded = 0
        # Requests the provider refused as LLM-GC rewrote them; each was sent again as the client wrote it.
        self.rewrites_refused = 0
        self.traffic = TrafficLedger()
        self.request_log = RequestLog()
        self.cache_watch = CacheWatch()

    def as_dict(self) -> dict[str, Any]:
        total = self.traffic.total
        return {
            "requests": self.requests,
            "gc_applied": self.gc_applied,
            "gc_bypassed": self.gc_bypassed,
            "tokens_from_client": total.tokens_from_client,
            "tokens_forwarded": total.tokens_forwarded,
            "tokens_saved": total.tokens_saved,
            # None, not 1.0: a ratio with no requests behind it is not a measurement.
            "ratio": round(total.tokens_forwarded / total.tokens_from_client, 3) if total.tokens_from_client else None,
            "provider_usage": asdict(total.provider),
            "collection_cache_write_tokens": total.collection_cache_write_tokens,
            "summaries": asdict(total.summaries),
            "history_rewinds": self.history_rewinds,
            "history_restarts": self.history_restarts,
            "sessions_seeded": self.sessions_seeded,
            "rewrites_refused": self.rewrites_refused,
            **self.traffic.as_dict(),
        }


def create_app(
    gc_service: GcService,
    *,
    upstream_base_url: str = ANTHROPIC_UPSTREAM,
    http_client: httpx2.AsyncClient | None = None,
    upstream_timeout_s: float = DEFAULT_UPSTREAM_TIMEOUT_S,
    schedule: Callable[[Awaitable[None]], None] | None = None,
    on_client_history: Callable[[str, list[Message], ClientOrigin | None], None] | None = None,
    on_reply: Callable[[str, Message, str], None] | None = None,
    window_lookup: WindowLookup | None = None,
    collect_subagents: bool = False,
    link_subagents: bool = False,
) -> FastAPI:
    """Wire the proxy around an already-built engine.

    Args:
        gc_service: The engine, injected so the app has no opinion about scorers,
            thresholds, or which compactor is wired in.
        upstream_base_url: Where to forward.
        http_client: An existing client to borrow; tests pass a `MockTransport`.
            When None, the app owns one and closes it on shutdown.
        upstream_timeout_s: Ceiling on the forwarded call.
        schedule: How to run post-response aging. Injectable so a test can drive it
            to completion instead of racing a fire-and-forget task.
        on_client_history: Called with each request's session, the conversation
            the client sent, and the client session it names, before forwarding.
            How the dashboard sees what was said; must be fast, and a failure in it
            never affects the request.
        on_reply: Called with the session, the turn being answered and the reply's
            text, once a reply has finished reaching the client. How the dashboard
            shows the newest reply before the next request records it; the same
            rules apply.
        window_lookup: Asks the upstream for a model's context window, with LLM-GC's
            own credential. Without it, every conversation's recap is sized for the
            configured window.
        collect_subagents: Collect the conversations a client marks as a subagent's.
            Off, they are recorded, shown and forwarded as sent, never collected.
        link_subagents: Link each subagent's first request to the tool call that
            started it. Holds each spawned prompt in memory until matched, so it is on
            only when something will show the link.
    """
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Sessions hold conversation content; without the sweeper an idle one would
        # live as long as the process.
        idle_sweeper = asyncio.create_task(gc_service.session_manager.run_idle_sweeper())
        try:
            yield
        finally:
            idle_sweeper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await idle_sweeper
            # Only a client we created is ours to close; closing a caller's pooled
            # client would break its other users. Same ownership rule as the llm/
            # adapters.
            if app.state.owns_http_client:
                await app.state.http_client.aclose()

    app = FastAPI(title="LLM-GC Proxy", version=APP_VERSION, lifespan=lifespan)
    app.state.gc_service = gc_service
    app.state.metrics = ProxyMetrics()
    app.state.upstream_base_url = upstream_base_url.rstrip("/")
    app.state.upstream_timeout_s = upstream_timeout_s
    app.state.http_client = http_client or httpx2.AsyncClient()
    app.state.owns_http_client = http_client is None
    app.state.schedule = schedule if schedule is not None else _default_schedule
    app.state.on_client_history = on_client_history
    app.state.on_reply = on_reply
    app.state.model_windows = ModelWindows(gc_service.gc_config.context_window, window_lookup, app.state.schedule)
    app.state.collect_subagents = collect_subagents
    app.state.spawn_linker = SpawnLinker() if link_subagents else None
    app.state.reply_thinking = ReplyThinking()
    gc_service.event_bus.subscribe(EventType.GC_FINISHED, lambda event: _count_collection(app, event))

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": APP_VERSION}

    @app.get("/stats")
    def stats() -> dict[str, Any]:
        return app.state.metrics.as_dict()

    @app.post(MESSAGES_PATH)
    async def messages(request: Request) -> Response:
        """Anthropic Messages, with the conversation replaced."""
        return await _handle(app, request)

    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    async def pass_through(request: Request) -> Response:
        """Any other Anthropic endpoint, forwarded untouched: token counting, model listing.

        LLM-GC rewrites only the conversation sent to Messages. A client such as Claude
        Code calls other endpoints through the same base URL, and a 404 from here would
        fail it because of LLM-GC. Registered after Messages, which matches first.
        """
        body = await _read_body(request)
        if body is None:
            return _too_large()
        return await _forward(app, body, forwardable_headers(dict(request.headers)), _target(request),
                              method=request.method)

    return app


def _default_schedule(coro: Awaitable[None]) -> None:
    """Run post-response work without making the client wait for it.

    The task reference is held until it finishes: a `create_task` result nobody
    keeps can be garbage-collected mid-flight, stopping the work with no exception.
    """
    task = asyncio.ensure_future(coro)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)


_BACKGROUND_TASKS: set[Any] = set()


async def _handle(app: FastAPI, request: Request) -> Response:
    """One proxied request, end to end."""
    metrics: ProxyMetrics = app.state.metrics
    metrics.requests += 1

    body = await _read_body(request)
    if body is None:
        return _too_large()
    headers = forwardable_headers(dict(request.headers))
    target = _target(request)

    try:
        payload = _parse_json(body)
        client_messages = extract_messages(payload, size_thinking=_thinking_sizer(app, payload))
        instructions = extract_instructions(payload)
    except (ConfigError, ValueError) as err:
        # The provider is the authority on its own API and may accept something we
        # cannot parse, so forward untouched rather than reject.
        # The reason names the request's structure - a role, a missing field - never its content.
        logger.info("forwarding a request LLM-GC cannot read, untouched: %s", err, extra={"reason": str(err)})
        metrics.gc_bypassed += 1
        return await _forward(app, body, headers, target)

    streaming = payload.get("stream") is True
    history, new_turn = client_messages[:-1], client_messages[-1]
    # The system prompt, the tools and the instructions count on both sides of the ratio:
    # they are sent, whole, either way. Thinking counts too, because the provider bills it as input.
    fixed_tokens = fixed_prompt_tokens(payload)
    client_tokens = (sum(message.token_count for message in client_messages)
                     + sent_thinking_tokens(client_messages)
                     + instruction_tokens(client_messages)
                     + sum(instruction.token_count for instruction in instructions) + fixed_tokens)
    client_sends_thinking = sends_thinking(client_messages)

    # Everything LLM-GC does before forwarding sits inside this one guard, so no
    # failure of ours can stop the conversation: the worst case is no GC.
    session_id: str | None = None
    origin: ClientOrigin | None = None
    window: ModelWindow | None = None
    collected = True
    hashes: list[str] | None = None
    try:
        origin = client_origin(request.headers)
        # One pass serves the session lookup, the claim of a first request and the request log.
        hashes = chain_hashes(client_messages)
        session_id = _record_client_history(app, history, hashes[:-1], request.headers.get(SESSION_HEADER),
                                            origin)
        window = app.state.model_windows.window_for(_model_name(payload))
        collected = _collected(app, origin)
        if session_id is not None:
            _observe_history(app, session_id, client_messages, origin)
            app.state.gc_service.use_context_window(session_id, window.tokens)
            app.state.gc_service.set_collection_enabled(session_id, collected)
        forwarded = await _apply_gc(app, session_id, history, new_turn, payload, body, instructions)
    except Exception as err:
        logger.exception("GC failed on the request path; forwarding the original",
                         extra={"session_id": session_id, "error_type": type(err).__name__})
        metrics.gc_bypassed += 1
        # Counted as forwarded whole, so a bypass does not look free in /stats.
        on_outline, _ = _measure_request(app, session_id, origin, payload, client_messages, hashes, client_tokens,
                                         client_tokens, window, collected, sends_thinking=client_sends_thinking,
                                         carries_front=False)
        response = await _send(app, body, headers, streaming, target, on_outline=on_outline)
        _log_answer(response.status_code, payload, client_tokens, client_tokens, session_id)
        return response

    metrics.gc_applied += 1
    forward_body = forwarded.body
    forwarded_tokens = forwarded.tokens + fixed_tokens
    on_outline, _ = _measure_request(app, session_id, origin, payload, client_messages, hashes, client_tokens,
                                     forwarded_tokens, window, collected, sends_thinking=forwarded.sends_thinking)
    response = await _send(app, forward_body, headers, streaming, target,
                           on_complete=_reply_observer(app, session_id, new_turn, streaming), on_outline=on_outline)
    if response.status_code == _REFUSED_AS_INVALID and forward_body is not body:
        # The provider refused the request as LLM-GC rewrote it. The client's own request is what it
        # would have sent without LLM-GC, so that goes in its place: never worse than no GC.
        _log_refused_rewrite(response, payload, session_id)
        metrics.rewrites_refused += 1
        forwarded_tokens = client_tokens
        on_outline, record = _measure_request(app, session_id, origin, payload, client_messages, hashes, client_tokens,
                                              client_tokens, window, collected, sends_thinking=client_sends_thinking,
                                              carries_front=False)
        if record is not None:
            record.resent_as_written = True
        response = await _send(app, body, headers, streaming, target,
                               on_complete=_reply_observer(app, session_id, new_turn, streaming), on_outline=on_outline)
    _log_answer(response.status_code, payload, client_tokens, forwarded_tokens, session_id)

    # Aging prepares the next request from the conversation recorded so far; it
    # never depends on this reply, which the next request will show us exactly.
    if session_id is not None and collected:
        _never_blocking(lambda: app.state.schedule(_age(app, session_id)))
    return response


def _log_refused_rewrite(response: Response, payload: dict[str, Any], session_id: str | None) -> None:
    """Say why the provider refused a request LLM-GC rewrote: the one refusal LLM-GC may have caused.

    Nobody else records it: the client never sees it, since its own request goes in its place.
    """
    reason = refusal_reason(bytes(response.body))
    where = f" in {session_id}" if session_id is not None else ""
    logger.warning("%s refused the request%s as LLM-GC rewrote it (%s); sent again as the client wrote it",
                   _model_name(payload) or "?", where, reason,
                   extra={"session_id": session_id, "status": response.status_code, "reason": reason})


def _log_answer(status: int, payload: dict[str, Any], from_client: int, forwarded: int,
                session_id: str | None) -> None:
    """One line per conversation request, logged once the upstream has answered.

    After forwarding rather than before, so the line says whether the request worked: one
    the upstream refused must not read like one that went through. The text carries the
    counts, because the console format shows only the message; `extra` carries them for a
    structured handler.
    """
    name = _model_name(payload) or "?"
    extra = {"session_id": session_id, "status": status, "model": name,
             "tokens_from_client": from_client, "tokens_forwarded": forwarded}
    # Names the conversation, as the dashboard's picker does; a first turn belongs to none yet.
    where = f" in {session_id}" if session_id is not None else ""
    if status < 400:
        share = f"{forwarded / from_client:.0%}" if from_client else "-"
        logger.info("%s -> %d%s: client sent %s tokens, LLM-GC forwarded %s (%s)",
                    name, status, where, f"{from_client:,}", f"{forwarded:,}", share, extra=extra)
    else:
        logger.warning("%s -> %d%s: %s", name, status, where, _ERROR_STATUS_HINTS.get(status, _ERROR_STATUS_HINT),
                       extra=extra)


def _model_name(payload: dict[str, Any]) -> str | None:
    """The model the request names. It comes from the client, so only a short printable string is taken."""
    model = payload.get("model")
    return model if isinstance(model, str) and model.isprintable() and len(model) <= _MODEL_NAME_MAX_CHARS else None


def _collected(app: FastAPI, origin: ClientOrigin | None) -> bool:
    """Whether this request's conversation is collected: every one, except a subagent's unless asked.

    A subagent usually ends soon after a collection could first help it, so the
    collection is paid for and seldom sent; and its answer is only as good as what it
    read, which no reader of the main conversation sees. The client's header only
    turns collection off: without it, a conversation is collected as any other.
    """
    return origin is None or origin.agent_id is None or app.state.collect_subagents


def _measure_request(app: FastAPI, session_id: str | None, origin: ClientOrigin | None, payload: dict[str, Any],
                     client_messages: Sequence[Message], hashes: Sequence[str] | None, from_client: int,
                     forwarded: int, window: ModelWindow | None, collected: bool, *, sends_thinking: bool,
                     carries_front: bool = True) -> tuple[OutlineHandler | None, RequestRecord | None]:
    """Count a request in the ledger and open its record; return what to do with its reply's outline, and the record.

    `sends_thinking`: the forwarded request still contains thinking blocks.
    `carries_front` False: the request goes as the client wrote it, so it carries no collection.
    Never fails the request: a failure here costs the counts, not the answer.
    """
    metrics: ProxyMetrics = app.state.metrics
    try:
        hashes = hashes if hashes is not None else chain_hashes(client_messages)
        model = _model_name(payload)
        # Read before the ledger counts this request, which marks them sent.
        carried = metrics.traffic.collections_waiting(session_id) if session_id is not None and carries_front else ()
        counted = metrics.traffic.record_request(conversation_id=session_id, origin=origin, model=model,
                                                 opening=hashes[0], tokens_from_client=from_client,
                                                 tokens_forwarded=forwarded, context_window=window,
                                                 collected=collected, carries_front=carries_front)
        record = metrics.request_log.begin(conversation_id=session_id, origin=origin, hashes=hashes,
                                           newest=newest_kind(client_messages[-1]), model=model,
                                           tokens_from_client=from_client, tokens_forwarded=forwarded,
                                           collected=collected, switch_reasons=carried)
        linker: SpawnLinker | None = app.state.spawn_linker
        if linker is not None and origin is not None and origin.agent_id is not None and record.position == 1:
            first = client_messages[0]
            linker.offer_opening(origin.session_id, record, "\n".join((first.content, *first.instructions)))
        # The cache's lifetime counts from when a request starts, which is now: it is forwarded next.
        started_at, lifetime_s = metrics.cache_watch.now(), cache_lifetime_s(payload)
    except Exception:
        logger.exception("counting the request failed; the request is unaffected", extra={"session_id": session_id})
        return None, None

    def close(outline: ReplyOutline, prompts: Mapping[str, str]) -> None:
        if (pressure := _PRESSURE_STATUSES.get(outline.status)) is not None:
            app.state.gc_service.pause_model_calls(outline.retry_after_s,
                                                   f"the upstream {pressure} a request from the client")
        if carried and outline.status != 200 and session_id is not None:
            # A refused request reached no model, so the collections it carried still wait for one that does.
            record.switch_reasons = ()
            metrics.traffic.restore_waiting(session_id, carried)
        if outline.usage is not None:
            TrafficLedger.record_usage(counted, outline.usage, forwarded, carried_collection=bool(carried))
            # Only a request without thinking gives a clean rate: the count of thinking is itself an estimate.
            rate = token_rate(forwarded, outline.usage) if not sends_thinking else None
            if rate is not None:
                app.state.reply_thinking.observe_rate(model, rate)
            if session_id is not None:
                metrics.cache_watch.observe(session_id, started_at=started_at, lifetime_s=lifetime_s,
                                            usage=outline.usage)
                # The window is in the model's tokens; this is how LLM-GC sets its own counts against it.
                if rate is not None:
                    app.state.gc_service.use_token_rate(session_id, rate)
            if outline.finished and outline.thinking_key is not None:
                # The thinking's size is worked out when the client sends this reply back.
                app.state.reply_thinking.remember(outline.thinking_key, output_tokens=outline.usage.output_tokens,
                                                  model=model)
        metrics.request_log.finish(record, outline)
        if linker is not None and origin is not None and prompts:
            linker.offer_calls(origin.session_id, record, prompts)

    return (lambda outline, prompts: _never_blocking(close, outline, prompts)), record


def _thinking_sizer(app: FastAPI, payload: dict[str, Any]) -> ThinkingSizer:
    """Return a function that sizes the thinking in this request's turns, for the model it names."""
    reply_thinking: ReplyThinking = app.state.reply_thinking
    model = _model_name(payload)
    return lambda keys, visible_tokens: reply_thinking.tokens(keys, visible_tokens, model)


def _count_collection(app: FastAPI, event: Event) -> None:
    """Count a collection, why it was switched in, and what its own model calls spent, against its conversation."""
    result: GCResult = event.data["gc_result"]
    traffic: TrafficLedger = app.state.metrics.traffic
    traffic.record_collection(event.data["session_id"], completed=result.status is GCStatus.COMPLETED,
                              spent=ModelUsage(calls=result.summary_calls, input_tokens=result.summary_input_tokens,
                                               output_tokens=result.summary_output_tokens),
                              switch_reason=result.switch_reason.value if result.switch_reason is not None else None)


def _record_client_history(app: FastAPI, history: list[Message], hashes: Sequence[str], header_value: str | None,
                           origin: ClientOrigin | None) -> str | None:
    """Find the session this history belongs to and record what it has not seen.

    The client's history is the truth: it holds every earlier reply exactly as the
    user saw it, and nothing the user never saw. So the session is reconciled to
    it on every request - continued, rewound, or started over - rather than built
    from responses. `hashes` are the history's chain hashes. Returns None for a first
    turn, which has no history to keep.
    """
    if not history:
        return None

    gc_service: GcService = app.state.gc_service
    manager = gc_service.session_manager
    metrics: ProxyMetrics = app.state.metrics

    if header_value and header_value.strip():
        state = manager.get_or_create(header_value.strip())
        recorded = state.snapshot.lineage
        agreed = _agreed_length(recorded, hashes)
        if agreed < len(recorded):
            # Regenerate, edit, or a retry of an earlier point: the client's history
            # left ours at turn `agreed`.
            rewound = state.snapshot.rewound_to(agreed)
            if rewound is not None:
                metrics.history_rewinds += 1
                state = manager.replace(state, rewound)
            else:
                # The abandoned branch was already summarised or archived, and
                # what was derived from it cannot be separated from what came
                # before. Starting over costs one uncompressed request; keeping it
                # could put the abandoned branch in the model's memory.
                metrics.history_restarts += 1
                state = manager.replace(state, GenerationSnapshot())
                agreed = 0
            logger.info("client history diverged from the session; reconciled",
                        extra={"session_id": state.session_id, "at_turn": agreed})
    else:
        found = manager.find_by_lineage(hashes)
        if found is not None:
            state, agreed = found
        else:
            # No conversation we know continues into this one: a first exchange, a
            # proxy restart, or an edited history. Seed from what the client sent.
            metrics.sessions_seeded += 1
            state = manager.get_or_create(f"s_{uuid.uuid4().hex[:16]}")
            agreed = 0
            _log_conversation_started(state.session_id, len(history), origin)

    if agreed < len(history):
        gc_service.gc_record(state.session_id, history[agreed:], hashes[agreed:])
    return state.session_id


def _log_conversation_started(session_id: str, turns: int, origin: ClientOrigin | None) -> None:
    """Say why a conversation appeared: nothing in progress continues the history it came with.

    A client session can start several - one per sub-agent, and another after it
    compacts or rewinds - so the line names the client session and agent it came from.
    """
    if origin is None:
        source = "no client session named"
    else:
        source = f"client session {origin.session_id}, " + (
            f"sub-agent {origin.agent_id}" if origin.agent_id is not None else "main conversation")
    logger.info("conversation %s started: no conversation in progress continues its %d-turn history (%s)",
                session_id, turns, source,
                extra={"session_id": session_id, "history_turns": turns,
                       "client_session": origin.session_id if origin is not None else None,
                       "client_agent": origin.agent_id if origin is not None else None})


def _observe_history(app: FastAPI, session_id: str, client_messages: list[Message],
                     origin: ClientOrigin | None) -> None:
    """Hand the conversation to an observer, if one is attached. Never fails the request."""
    observer = app.state.on_client_history
    if observer is None:
        return
    try:
        observer(session_id, client_messages, origin)
    except Exception:
        logger.exception("history observer failed; the request is unaffected", extra={"session_id": session_id})


def _agreed_length(recorded: Sequence[str], hashes: Sequence[str]) -> int:
    """How many leading turns two conversations share, from their chain hashes."""
    agreed = 0
    for ours, theirs in zip(recorded, hashes):
        if ours != theirs:
            break
        agreed += 1
    return agreed


async def _apply_gc(
    app: FastAPI,
    session_id: str | None,
    history: list[Message],
    new_turn: Message,
    payload: dict[str, Any],
    original_body: bytes,
    instructions: Sequence[Instruction] = (),
) -> "_Forwarded":
    """The body to forward, and its conversation's token count.

    When the collected context is exactly the client's history - nothing has been
    aged yet - the client's own bytes are forwarded untouched, so images, tool
    calls, thinking blocks and the provider's prompt cache are all preserved. The
    client's instructions are never condensed: each goes back in place.
    """
    sent_whole = sum(instruction.token_count for instruction in instructions)
    client_tokens = (sum(message.token_count for message in history) + new_turn.token_count
                     + sent_thinking_tokens([*history, new_turn])
                     + instruction_tokens([*history, new_turn]) + sent_whole)
    whole = _Forwarded(original_body, client_tokens, sends_thinking([*history, new_turn]))
    if session_id is None:
        return whole

    gc_service: GcService = app.state.gc_service
    metrics: ProxyMetrics = app.state.metrics
    # A staged cut is switched in, or not, as this request is built: then it is the first to carry it.
    # Whether its cache would serve this request decides whether switching costs a rewrite, and
    # a request answering tool calls never carries a new front.
    gc_service.gc_switch(session_id, metrics.cache_watch.state(session_id),
                         tool_round_open=bool(tool_result_ids(new_turn)))
    collected = await gc_service.gc_collect(session_id)
    if [message.turn_index for message in collected] == list(range(len(history))):
        return whole

    # No await since the collect, so this describes the front being forwarded.
    valid_from = gc_service.mark_front_sent(session_id, len(history))
    # Recent turns go as the client sent them in this request, not as recorded when
    # they arrived: the client moves its cache markers onto the newest turns, and
    # markers kept from earlier requests would pile up past the provider's limit.
    sent_now = {message.turn_index: message for message in history}
    # The front stands in for every turn before the first one sent as written; the
    # instructions in those turns open it, word for word, as this request has them.
    first_sent = next((message.turn_index for message in collected if message.turn_index >= 0), len(history))
    stood_in = [message for message in history if message.turn_index < first_sent]
    forwarded = [carry_instructions(message, stood_in) if message.turn_index < 0
                 else sent_now.get(message.turn_index, message) for message in collected] + [new_turn]
    body = rewrite_messages(payload, forwarded, strip_thinking_before=valid_from, instructions=instructions)
    forwarded_tokens = (sum(message.token_count for message in forwarded)
                        + sent_thinking_tokens(forwarded, strip_thinking_before=valid_from)
                        + instruction_tokens(forwarded) + sent_whole)
    return _Forwarded(_dump_json(body), forwarded_tokens, sends_thinking(forwarded, strip_thinking_before=valid_from))


@dataclass(frozen=True, slots=True)
class _Forwarded:
    """The request body LLM-GC forwards, as `_apply_gc` built it."""

    body: bytes
    tokens: int              # its conversation's tokens, thinking included
    sends_thinking: bool     # whether any thinking block is left in it


async def _age(app: FastAPI, session_id: str) -> None:
    """Age the session after the response. Detached, so failures are logged here."""
    try:
        gc_service: GcService = app.state.gc_service
        await gc_service.gc_age(session_id)
    except Exception:
        logger.exception("post-response aging failed", extra={"session_id": session_id})


def _never_blocking(action: Callable[..., None], *args: Any) -> None:
    """Run post-response bookkeeping that must never cost the client their answer.

    By this point the provider has answered; a failure here is logged, and the
    response the user is waiting on is unaffected.
    """
    try:
        action(*args)
    except Exception:
        logger.exception("post-response bookkeeping failed; the client's response is unaffected")


def _target(request: Request) -> str:
    """The path and query the client asked for, to ask the provider for the same."""
    return f"{request.url.path}?{request.url.query}" if request.url.query else request.url.path


async def _send(app: FastAPI, body: bytes, headers: dict[str, str], streaming: bool, target: str,
                on_complete: Callable[[bytes], None] | None = None,
                on_outline: OutlineHandler | None = None) -> Response:
    if streaming:
        return await _forward_stream(app, body, headers, target, on_complete, on_outline)
    return await _forward(app, body, headers, target, on_complete=on_complete, on_outline=on_outline)


def _reply_observer(app: FastAPI, session_id: str | None, answering: Message,
                    streaming: bool) -> Callable[[bytes], None] | None:
    """What to do with a finished reply's bytes: show its text, when an observer is attached.

    Read only for showing: the record still learns the reply from the next request,
    which says whether the client kept it. Never fails the response.
    """
    observer = app.state.on_reply
    if observer is None or session_id is None:
        return None

    def observe(raw: bytes) -> None:
        try:
            text = reply_text(raw, streamed=streaming)
            if text is not None:
                observer(session_id, answering, text)
        except Exception:
            logger.exception("reply observer failed; the response is unaffected", extra={"session_id": session_id})

    return observe


async def _forward(app: FastAPI, body: bytes, headers: dict[str, str], target: str, *,
                   method: str = "POST", on_complete: Callable[[bytes], None] | None = None,
                   on_outline: OutlineHandler | None = None) -> Response:
    """Send the request upstream and hand the answer back verbatim.

    Status codes and error bodies pass through unchanged: the client's error
    handling was written against the provider's format, not ours.
    """
    client: httpx2.AsyncClient = app.state.http_client
    try:
        upstream = await client.send(_upstream_request(app, body, headers, target, method))
    except httpx2.RequestError as err:
        return _unreachable(err, on_outline)
    if on_complete is not None and upstream.status_code == 200:
        on_complete(upstream.content)
    if on_outline is not None:
        outline, prompts = read_reply(upstream.status_code, upstream.content, read_prompts=_reads_prompts(app))
        on_outline(_with_retry_after(outline, upstream.headers), prompts)
    return _verbatim(upstream, upstream.content)


async def _forward_stream(app: FastAPI, body: bytes, headers: dict[str, str], target: str,
                          on_complete: Callable[[bytes], None] | None = None,
                          on_outline: OutlineHandler | None = None) -> Response:
    """Relay a streamed reply chunk by chunk, as it arrives.

    With `on_complete`, each chunk is also kept, and the whole stream is handed
    over once the client has it all: one append per chunk on the way through.
    With `on_outline`, the reply's outline is read as it passes, and handed over when
    it ends - also when it ends early, since the prompt was billed anyway.
    """
    client: httpx2.AsyncClient = app.state.http_client
    try:
        upstream = await client.send(_upstream_request(app, body, headers, target, "POST"), stream=True)
    except httpx2.RequestError as err:
        return _unreachable(err, on_outline)

    # An error status arrives as a JSON body, not a stream; pass it through whole.
    if upstream.status_code != 200:
        try:
            content = await upstream.aread()
        finally:
            await upstream.aclose()
        if on_outline is not None:
            outline, prompts = read_reply(upstream.status_code, content)
            on_outline(_with_retry_after(outline, upstream.headers), prompts)
        return _verbatim(upstream, content)

    async def relay() -> AsyncIterator[bytes]:
        kept: list[bytes] | None = [] if on_complete is not None else None
        outline_reader = StreamOutlineReader(read_prompts=_reads_prompts(app)) if on_outline is not None else None
        size = 0
        try:
            # Decoded bytes: we strip `content-encoding`, so relaying raw gzip would
            # hand the client compressed bytes labelled as plain text.
            async for chunk in upstream.aiter_bytes():
                if outline_reader is not None:
                    outline_reader.feed(chunk)
                yield chunk
                if kept is not None:
                    size += len(chunk)
                    if size > _REPLY_KEPT_MAX_BYTES:
                        kept = None
                    else:
                        kept.append(chunk)
            if on_complete is not None and kept is not None:
                on_complete(b"".join(kept))
        except httpx2.TransportError as err:
            logger.warning("provider stream dropped mid-response", extra={"error_type": type(err).__name__})
            raise
        finally:
            # Also runs on client disconnect, when the server closes this generator.
            if on_outline is not None and outline_reader is not None:
                on_outline(outline_reader.outline(), outline_reader.spawn_prompts())
            await upstream.aclose()

    return StreamingResponse(
        relay(),
        status_code=upstream.status_code,
        headers=_response_headers(upstream),
        media_type=upstream.headers.get("content-type", "text/event-stream"),
    )


def _with_retry_after(outline: ReplyOutline, headers: httpx2.Headers) -> ReplyOutline:
    """The outline of a refusal, with the wait the upstream asked for, in seconds, when it gave one."""
    if outline.status not in _PRESSURE_STATUSES:
        return outline
    try:
        seconds = float(headers.get("retry-after", ""))
    except ValueError:
        return outline
    return replace(outline, retry_after_s=seconds) if seconds > 0 else outline


def _reads_prompts(app: FastAPI) -> bool:
    return app.state.spawn_linker is not None


def _upstream_request(app: FastAPI, body: bytes, headers: dict[str, str], target: str,
                      method: str) -> httpx2.Request:
    client: httpx2.AsyncClient = app.state.http_client
    return client.build_request(
        method,
        f"{app.state.upstream_base_url}{target}",
        content=body or None,
        headers={**headers, "content-type": "application/json"} if body else headers,
        timeout=app.state.upstream_timeout_s,
    )


def _verbatim(upstream: httpx2.Response, content: bytes) -> Response:
    return Response(
        content=content,
        status_code=upstream.status_code,
        headers=_response_headers(upstream),
        media_type=upstream.headers.get("content-type", "application/json"),
    )


def _response_headers(upstream: httpx2.Response) -> dict[str, str]:
    return {k: v for k, v in upstream.headers.items() if k.lower() not in _RESPONSE_HOP_BY_HOP}


def _unreachable(err: httpx2.RequestError, on_outline: OutlineHandler | None = None) -> Response:
    # 502: the failure is upstream of us, not the client's mistake.
    logger.warning("could not reach the provider", extra={"error_type": type(err).__name__})
    if on_outline is not None:
        on_outline(ReplyOutline(status=502), {})
    return Response(
        content=_dump_json_bytes(
            {"error": {"type": "api_error", "message": f"llm-gc could not reach the provider: {type(err).__name__}"}}
        ),
        status_code=502,
        media_type="application/json",
    )


async def _read_body(request: Request) -> bytes | None:
    """The request body, or None once it passes `MAX_REQUEST_BYTES`.

    Checked while reading rather than trusted from `content-length`, which a client
    may omit or misstate.
    """
    received = bytearray()
    async for chunk in request.stream():
        received += chunk
        if len(received) > MAX_REQUEST_BYTES:
            return None
    return bytes(received)


def _too_large() -> Response:
    return Response(
        content=_dump_json_bytes({"type": "error", "error": {
            "type": "request_too_large",
            "message": f"request body exceeds {MAX_REQUEST_BYTES} bytes, the provider's own limit"}}),
        status_code=413,
        media_type="application/json",
    )


def _parse_json(body: bytes) -> dict[str, Any]:
    import json

    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError("request body is not a JSON object")
    return parsed


def _dump_json(payload: dict[str, Any]) -> bytes:
    return _dump_json_bytes(payload)


def _dump_json_bytes(payload: dict[str, Any]) -> bytes:
    import json

    return json.dumps(payload).encode("utf-8")
