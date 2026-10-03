"""The HTTP proxy: GC on the request path.

An app points its SDK here instead of at the provider and changes nothing else.
Each request's conversation is replaced by the garbage-collected version:

    client sends the full conversation
      -> its last message is the new turn
      -> gc_collect(session, query=new turn) returns OUR maintained context
      -> forward [our context] + [new turn]
      -> return the provider's answer unchanged
      -> after responding, record both turns with gc_update()

The client's history is ignored; we send ours. Any failure in our own machinery
forwards the client's original request untouched. See doc/design/ for rationale.
"""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx2
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from llm_gc.config.constants import APP_VERSION
from llm_gc.engine.gc_service import GcService
from llm_gc.exceptions import ConfigError
from llm_gc.models import Message
from llm_gc.proxy.anthropic_stream import StreamedReply
from llm_gc.proxy.anthropic_translation import extract_messages, forwardable_headers, rewrite_messages
from llm_gc.proxy.session_identity import SESSION_HEADER, resolve_session_id
from llm_gc.utils import count_tokens

logger = logging.getLogger(__name__)

ANTHROPIC_UPSTREAM = "https://api.anthropic.com"
MESSAGES_PATH = "/v1/messages"
DEFAULT_PROXY_PORT = 9900

# Generous, and a ceiling rather than a target. A long generation legitimately
# takes minutes, and a proxy that times out before the provider does turns a slow
# success into a failure the client cannot distinguish from a real one.
DEFAULT_UPSTREAM_TIMEOUT_S = 600.0

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
        # Streams that reached the client but whose reply the engine never heard.
        self.streams_unrecorded = 0

    def as_dict(self) -> dict[str, Any]:
        saved = self.tokens_in_client_history - self.tokens_forwarded
        return {
            "requests": self.requests,
            "gc_applied": self.gc_applied,
            "gc_bypassed": self.gc_bypassed,
            "tokens_in_client_history": self.tokens_in_client_history,
            "tokens_forwarded": self.tokens_forwarded,
            "tokens_saved": saved,
            "streams_unrecorded": self.streams_unrecorded,
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
    """
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
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

    return app


def _default_schedule(coro: Awaitable[None]) -> None:
    """Run post-response work without making the client wait for it.

    The task reference is held until it finishes: a `create_task` result nobody
    keeps can be garbage-collected mid-flight, stopping the work with no exception.
    """
    import asyncio

    task = asyncio.ensure_future(coro)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)


_BACKGROUND_TASKS: set[Any] = set()


async def _handle(app: FastAPI, request: Request) -> Response:
    """One proxied request, end to end."""
    metrics: ProxyMetrics = app.state.metrics
    metrics.requests += 1

    body = await request.body()
    headers = forwardable_headers(dict(request.headers))

    try:
        payload = _parse_json(body)
        client_messages = extract_messages(payload)
    except (ConfigError, ValueError) as err:
        # The provider is the authority on its own API and may accept something we
        # cannot parse, so forward untouched rather than reject.
        logger.info("forwarding unprocessable request untouched", extra={"reason": str(err)})
        metrics.gc_bypassed += 1
        return await _forward(app, body, headers)

    streaming = payload.get("stream") is True
    new_turn = client_messages[-1]
    client_history_tokens = sum(message.token_count for message in client_messages)
    metrics.tokens_in_client_history += client_history_tokens

    # Everything LLM-GC does before forwarding sits inside this one guard, so no
    # failure of ours can stop the conversation: the worst case is no GC.
    session_id: str | None = None
    try:
        session_id = resolve_session_id(client_messages, request.headers.get(SESSION_HEADER))
        forward_body, forwarded_messages = await _apply_gc(app, session_id, client_messages, payload, new_turn)
        logger.info(
            "proxying with a collected context",
            extra={
                "session_id": session_id,
                "turns_from_client": len(client_messages),
                "turns_forwarded": len(forwarded_messages),
                "tokens_from_client": client_history_tokens,
                "tokens_forwarded": sum(message.token_count for message in forwarded_messages),
            },
        )
    except Exception as err:
        logger.exception("GC failed on the request path; forwarding the original",
                         extra={"session_id": session_id, "error_type": type(err).__name__})
        metrics.gc_bypassed += 1
        # Counted as forwarded, so a bypass does not look free in /stats.
        metrics.tokens_forwarded += client_history_tokens
        # Still recorded when the session is known: the exchange really happened,
        # and skipping it would leave a hole every later context is built over.
        return await _forward_and_record(app, body, headers, streaming, session_id, new_turn)

    metrics.gc_applied += 1
    metrics.tokens_forwarded += sum(message.token_count for message in forwarded_messages)
    return await _forward_and_record(app, _dump_json(forward_body), headers, streaming, session_id, new_turn)


async def _forward_and_record(
    app: FastAPI,
    body: bytes,
    headers: dict[str, str],
    streaming: bool,
    session_id: str | None,
    new_turn: Message,
) -> Response:
    """Forward, then record the exchange only if the provider answered it.

    Recording only a successful exchange keeps session state to conversations the
    client actually had: a failed one will be retried, and recording it would put
    the turn in twice.
    """
    metrics: ProxyMetrics = app.state.metrics

    if streaming:
        def record_stream(reply: StreamedReply) -> None:
            if session_id is None:
                return
            if reply.completed and not reply.errored:
                app.state.schedule(_age(app, session_id, new_turn, reply.text or None))
                return
            metrics.streams_unrecorded += 1
            logger.warning("streamed reply not recorded", extra={
                "session_id": session_id, "completed": reply.completed, "errored": reply.errored,
                "abandoned": reply.abandoned, "malformed_events": reply.malformed_events,
            })

        return await _forward_stream(app, body, headers, on_end=lambda reply: _never_blocking(record_stream, reply))

    response = await _forward(app, body, headers)
    if response.status_code == 200 and session_id is not None:
        _never_blocking(
            lambda: app.state.schedule(_age(app, session_id, new_turn, _reply_text(bytes(response.body))))
        )
    return response


def _never_blocking(action: Callable[..., None], *args: Any) -> None:
    """Run post-response bookkeeping that must never cost the client their answer.

    By this point the provider has answered; a failure here loses one recorded
    turn, which is logged, rather than the response the user is waiting on.
    """
    try:
        action(*args)
    except Exception:
        logger.exception("post-response recording failed; the client's response is unaffected")


async def _apply_gc(
    app: FastAPI,
    session_id: str,
    client_messages: list[Message],
    payload: dict[str, Any],
    new_turn: Message,
) -> tuple[dict[str, Any], list[Message]]:
    """Replace the client's history with ours, keeping their newest turn.

    The new turn is appended because aging runs after the response, so it is not
    in the maintained context yet. On a cold session nothing is maintained and the
    client's own messages are forwarded.
    """
    gc_service: GcService = app.state.gc_service
    collected = await gc_service.gc_collect(session_id, query=new_turn.content)

    # A duplicated final turn is a confusing prompt, and the check is one compare.
    if collected and collected[-1].content == new_turn.content:
        forwarded = collected
    else:
        forwarded = collected + [new_turn]

    return rewrite_messages(payload, forwarded), forwarded


async def _age(app: FastAPI, session_id: str, new_turn: Message, reply_text: str | None) -> None:
    """Record both halves of the exchange after the client has their answer.

    Runs detached, so exceptions are logged rather than left to vanish unretrieved.
    """
    try:
        gc_service: GcService = app.state.gc_service
        await gc_service.gc_update(session_id, new_turn)

        if reply_text is not None:
            reply = Message(role="assistant", content=reply_text, token_count=count_tokens(reply_text),
                            turn_index=new_turn.turn_index + 1)
            await gc_service.gc_update(session_id, reply)
    except Exception:
        logger.exception("post-response aging failed", extra={"session_id": session_id})


def _reply_text(body: bytes) -> str | None:
    """Read the assistant's text out of a non-streamed reply, or None.

    None rather than raising: the client already has their answer, and a missing
    reply costs one turn of scoring accuracy.
    """
    try:
        payload = _parse_json(body)
    except ValueError:
        return None

    blocks = payload.get("content")
    if not isinstance(blocks, list):
        return None

    text = "".join(
        block["text"]
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
    )
    return text or None


async def _forward(app: FastAPI, body: bytes, headers: dict[str, str]) -> Response:
    """Send the request upstream and hand the answer back verbatim.

    Status codes and error bodies pass through unchanged: the client's error
    handling was written against the provider's format, not ours.
    """
    client: httpx2.AsyncClient = app.state.http_client
    try:
        upstream = await client.send(_upstream_request(app, body, headers))
    except httpx2.RequestError as err:
        return _unreachable(err)
    return _verbatim(upstream, upstream.content)


async def _forward_stream(
    app: FastAPI,
    body: bytes,
    headers: dict[str, str],
    *,
    on_end: Callable[[StreamedReply], None] | None,
) -> Response:
    """Relay a streamed reply chunk by chunk, observing it on the way past.

    `on_end` runs only when the upstream stream ran to its end. A client that
    disconnects, or an upstream that drops mid-stream, never reaches it.
    """
    client: httpx2.AsyncClient = app.state.http_client
    try:
        upstream = await client.send(_upstream_request(app, body, headers), stream=True)
    except httpx2.RequestError as err:
        return _unreachable(err)

    # An error status arrives as a JSON body, not a stream; pass it through whole.
    if upstream.status_code != 200:
        try:
            content = await upstream.aread()
        finally:
            await upstream.aclose()
        return _verbatim(upstream, content)

    reply = StreamedReply()

    async def relay() -> AsyncIterator[bytes]:
        observing = True
        try:
            # Decoded bytes: we strip `content-encoding`, so relaying raw gzip would
            # hand the client compressed bytes labelled as plain text.
            async for chunk in upstream.aiter_bytes():
                if observing:
                    try:
                        reply.feed(chunk)
                    except Exception:
                        # A bug in the observer stops observation, never the relay.
                        # The reply ends incomplete, so it is counted as unrecorded.
                        logger.exception("stream observer failed; relaying without it")
                        observing = False
                yield chunk
        except httpx2.TransportError as err:
            logger.warning("provider stream dropped mid-response", extra={"error_type": type(err).__name__})
            raise
        finally:
            # Also runs on client disconnect, when the server closes this generator.
            await upstream.aclose()
        if on_end is not None:
            on_end(reply)

    return StreamingResponse(
        relay(),
        status_code=upstream.status_code,
        headers=_response_headers(upstream),
        media_type=upstream.headers.get("content-type", "text/event-stream"),
    )


def _upstream_request(app: FastAPI, body: bytes, headers: dict[str, str]) -> httpx2.Request:
    client: httpx2.AsyncClient = app.state.http_client
    return client.build_request(
        "POST",
        f"{app.state.upstream_base_url}{MESSAGES_PATH}",
        content=body,
        headers={**headers, "content-type": "application/json"},
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
