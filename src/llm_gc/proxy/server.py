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
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from typing import Any

import httpx2
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from llm_gc.config.constants import APP_VERSION
from llm_gc.engine.gc_service import GcService
from llm_gc.exceptions import ConfigError
from llm_gc.models import Message
from llm_gc.models.generation_snapshot import GenerationSnapshot
from llm_gc.proxy.anthropic_translation import (
    Instruction,
    carry_instructions,
    extract_instructions,
    extract_messages,
    forwardable_headers,
    instruction_tokens,
    reply_text,
    rewrite_messages,
)
from llm_gc.proxy.session_identity import SESSION_HEADER, chain_hashes

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

# Response headers that describe the upstream connection, not its content.
# Returning them would have us promise an encoding or a length that no longer
# applies to the body we hand back.
_RESPONSE_HOP_BY_HOP = frozenset({
    "content-length", "content-encoding", "transfer-encoding", "connection",
    "keep-alive", "upgrade", "trailer",
})


class ProxyMetrics:
    """Aggregate counters for `GET /stats`. No conversation content."""

    def __init__(self) -> None:
        self.requests = 0
        self.gc_applied = 0
        self.gc_bypassed = 0
        self.tokens_in_client_history = 0
        self.tokens_forwarded = 0
        # Reconciliations with the client's history. Each restart or seed costs one
        # uncompressed request, so they are counted rather than hidden.
        self.history_rewinds = 0
        self.history_restarts = 0
        self.sessions_seeded = 0

    def as_dict(self) -> dict[str, Any]:
        saved = self.tokens_in_client_history - self.tokens_forwarded
        return {
            "requests": self.requests,
            "gc_applied": self.gc_applied,
            "gc_bypassed": self.gc_bypassed,
            "tokens_in_client_history": self.tokens_in_client_history,
            "tokens_forwarded": self.tokens_forwarded,
            "tokens_saved": saved,
            "history_rewinds": self.history_rewinds,
            "history_restarts": self.history_restarts,
            "sessions_seeded": self.sessions_seeded,
            # None, not 1.0: a ratio with no requests behind it is not a measurement.
            "ratio": (
                round(self.tokens_forwarded / self.tokens_in_client_history, 3)
                if self.tokens_in_client_history
                else None
            ),
        }


def create_app(
    gc_service: GcService,
    *,
    upstream_base_url: str = ANTHROPIC_UPSTREAM,
    http_client: httpx2.AsyncClient | None = None,
    upstream_timeout_s: float = DEFAULT_UPSTREAM_TIMEOUT_S,
    schedule: Callable[[Awaitable[None]], None] | None = None,
    on_client_history: Callable[[str, list[Message]], None] | None = None,
    on_reply: Callable[[str, Message, str], None] | None = None,
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
        on_client_history: Called with each request's session and the conversation
            the client sent, before forwarding. How the dashboard sees what was
            said; must be fast, and a failure in it never affects the request.
        on_reply: Called with the session, the turn being answered and the reply's
            text, once a reply has finished reaching the client. How the dashboard
            shows the newest reply before the next request records it; the same
            rules apply.
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
        client_messages = extract_messages(payload)
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
    # Instructions count on both sides of the ratio: they are sent, whole, either way.
    client_history_tokens = (sum(message.token_count for message in client_messages)
                             + instruction_tokens(client_messages)
                             + sum(instruction.token_count for instruction in instructions))
    metrics.tokens_in_client_history += client_history_tokens

    # Everything LLM-GC does before forwarding sits inside this one guard, so no
    # failure of ours can stop the conversation: the worst case is no GC.
    session_id: str | None = None
    try:
        session_id = _record_client_history(app, history, request.headers.get(SESSION_HEADER))
        if session_id is not None:
            _observe_history(app, session_id, client_messages)
        forward_body, forwarded_tokens = await _apply_gc(app, session_id, history, new_turn, payload, body,
                                                         instructions)
        logger.info("proxying", extra={"session_id": session_id, "turns_from_client": len(client_messages),
                                       "tokens_from_client": client_history_tokens,
                                       "tokens_forwarded": forwarded_tokens})
    except Exception as err:
        logger.exception("GC failed on the request path; forwarding the original",
                         extra={"session_id": session_id, "error_type": type(err).__name__})
        metrics.gc_bypassed += 1
        # Counted as forwarded, so a bypass does not look free in /stats.
        metrics.tokens_forwarded += client_history_tokens
        return await _send(app, body, headers, streaming, target)

    metrics.gc_applied += 1
    metrics.tokens_forwarded += forwarded_tokens
    response = await _send(app, forward_body, headers, streaming, target,
                           on_complete=_reply_observer(app, session_id, new_turn, streaming))

    # Aging prepares the next request from the conversation recorded so far; it
    # never depends on this reply, which the next request will show us exactly.
    if session_id is not None:
        _never_blocking(lambda: app.state.schedule(_age(app, session_id)))
    return response


def _record_client_history(app: FastAPI, history: list[Message], header_value: str | None) -> str | None:
    """Find the session this history belongs to and record what it has not seen.

    The client's history is the truth: it holds every earlier reply exactly as the
    user saw it, and nothing the user never saw. So the session is reconciled to
    it on every request - continued, rewound, or started over - rather than built
    from responses. Returns None for a first turn, which has no history to keep.
    """
    if not history:
        return None

    gc_service: GcService = app.state.gc_service
    manager = gc_service.session_manager
    metrics: ProxyMetrics = app.state.metrics
    hashes = chain_hashes(history)

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

    if agreed < len(history):
        gc_service.gc_record(state.session_id, history[agreed:], hashes[agreed:])
    return state.session_id


def _observe_history(app: FastAPI, session_id: str, client_messages: list[Message]) -> None:
    """Hand the conversation to an observer, if one is attached. Never fails the request."""
    observer = app.state.on_client_history
    if observer is None:
        return
    try:
        observer(session_id, client_messages)
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
) -> tuple[bytes, int]:
    """The body to forward, and its conversation's token count.

    When the collected context is exactly the client's history - nothing has been
    aged yet - the client's own bytes are forwarded untouched, so images, tool
    calls, thinking blocks and the provider's prompt cache are all preserved. The
    client's instructions are never condensed: each goes back in place.
    """
    sent_whole = sum(instruction.token_count for instruction in instructions)
    client_tokens = (sum(message.token_count for message in history) + new_turn.token_count
                     + instruction_tokens([*history, new_turn]) + sent_whole)
    if session_id is None:
        return original_body, client_tokens

    gc_service: GcService = app.state.gc_service
    collected = await gc_service.gc_collect(session_id)
    if [message.turn_index for message in collected] == list(range(len(history))):
        return original_body, client_tokens

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
    return _dump_json(body), sum(m.token_count for m in forwarded) + instruction_tokens(forwarded) + sent_whole


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
                on_complete: Callable[[bytes], None] | None = None) -> Response:
    if streaming:
        return await _forward_stream(app, body, headers, target, on_complete)
    return await _forward(app, body, headers, target, on_complete=on_complete)


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
                   method: str = "POST", on_complete: Callable[[bytes], None] | None = None) -> Response:
    """Send the request upstream and hand the answer back verbatim.

    Status codes and error bodies pass through unchanged: the client's error
    handling was written against the provider's format, not ours.
    """
    client: httpx2.AsyncClient = app.state.http_client
    try:
        upstream = await client.send(_upstream_request(app, body, headers, target, method))
    except httpx2.RequestError as err:
        return _unreachable(err)
    if on_complete is not None and upstream.status_code == 200:
        on_complete(upstream.content)
    return _verbatim(upstream, upstream.content)


async def _forward_stream(app: FastAPI, body: bytes, headers: dict[str, str], target: str,
                          on_complete: Callable[[bytes], None] | None = None) -> Response:
    """Relay a streamed reply chunk by chunk, as it arrives.

    With `on_complete`, each chunk is also kept, and the whole stream is handed
    over once the client has it all: one append per chunk on the way through.
    """
    client: httpx2.AsyncClient = app.state.http_client
    try:
        upstream = await client.send(_upstream_request(app, body, headers, target, "POST"), stream=True)
    except httpx2.RequestError as err:
        return _unreachable(err)

    # An error status arrives as a JSON body, not a stream; pass it through whole.
    if upstream.status_code != 200:
        try:
            content = await upstream.aread()
        finally:
            await upstream.aclose()
        return _verbatim(upstream, content)

    async def relay() -> AsyncIterator[bytes]:
        kept: list[bytes] | None = [] if on_complete is not None else None
        size = 0
        try:
            # Decoded bytes: we strip `content-encoding`, so relaying raw gzip would
            # hand the client compressed bytes labelled as plain text.
            async for chunk in upstream.aiter_bytes():
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
            await upstream.aclose()

    return StreamingResponse(
        relay(),
        status_code=upstream.status_code,
        headers=_response_headers(upstream),
        media_type=upstream.headers.get("content-type", "text/event-stream"),
    )


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


def _unreachable(err: httpx2.RequestError) -> Response:
    # 502: the failure is upstream of us, not the client's mistake.
    logger.warning("could not reach the provider", extra={"error_type": type(err).__name__})
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
