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

from llm_gc.config.constants import APP_VERSION
from llm_gc.engine.gc_service import GcService
from llm_gc.exceptions import ConfigError
from llm_gc.models import Message
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

    def as_dict(self) -> dict[str, Any]:
        saved = self.tokens_in_client_history - self.tokens_forwarded
        return {
            "requests": self.requests,
            "gc_applied": self.gc_applied,
            "gc_bypassed": self.gc_bypassed,
            "tokens_in_client_history": self.tokens_in_client_history,
            "tokens_forwarded": self.tokens_forwarded,
            "tokens_saved": saved,
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

    session_id = resolve_session_id(client_messages, request.headers.get(SESSION_HEADER))
    new_turn = client_messages[-1]
    client_history_tokens = sum(message.token_count for message in client_messages)
    metrics.tokens_in_client_history += client_history_tokens

    try:
        forward_body, forwarded_messages = await _apply_gc(app, session_id, client_messages, payload, new_turn)
        metrics.gc_applied += 1
    except Exception as err:
        # First do no harm: a 500 here is worse than forwarding what they sent.
        logger.exception("GC failed on the request path; forwarding the original",
                         extra={"session_id": session_id, "error_type": type(err).__name__})
        metrics.gc_bypassed += 1
        # Counted as forwarded, so a bypass does not look free in /stats.
        metrics.tokens_forwarded += client_history_tokens
        return await _forward(app, body, headers)

    metrics.tokens_forwarded += sum(message.token_count for message in forwarded_messages)
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

    response = await _forward(app, _dump_json(forward_body), headers)

    # Only record a successful exchange: otherwise session state describes a
    # conversation the client never had, and their retry records the turn twice.
    if response.status_code == 200:
        app.state.schedule(_age(app, session_id, new_turn, response))

    return response


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


async def _age(app: FastAPI, session_id: str, new_turn: Message, response: Response) -> None:
    """Record both halves of the exchange after the client has their answer.

    Runs detached, so exceptions are logged rather than left to vanish unretrieved.
    """
    try:
        gc_service: GcService = app.state.gc_service
        await gc_service.gc_update(session_id, new_turn)

        reply = _assistant_turn(response, new_turn.turn_index + 1)
        if reply is not None:
            await gc_service.gc_update(session_id, reply)
    except Exception:
        logger.exception("post-response aging failed", extra={"session_id": session_id})


def _assistant_turn(response: Response, turn_index: int) -> Message | None:
    """Read the provider's reply back out as a `Message`, or None.

    None rather than raising: the client already has their answer, and a missing
    reply costs one turn of scoring accuracy.
    """
    try:
        payload = _parse_json(bytes(response.body))
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
    if not text:
        return None

    return Message(role="assistant", content=text, token_count=count_tokens(text), turn_index=turn_index)


async def _forward(app: FastAPI, body: bytes, headers: dict[str, str]) -> Response:
    """Send the request upstream and hand the answer back verbatim.

    Status codes and error bodies pass through unchanged: the client's error
    handling was written against the provider's format, not ours.
    """
    client: httpx2.AsyncClient = app.state.http_client
    try:
        upstream = await client.post(
            f"{app.state.upstream_base_url}{MESSAGES_PATH}",
            content=body,
            headers={**headers, "content-type": "application/json"},
            timeout=app.state.upstream_timeout_s,
        )
    except httpx2.RequestError as err:
        # 502: the failure is upstream of us, not the client's mistake.
        logger.warning("could not reach the provider", extra={"error_type": type(err).__name__})
        return Response(
            content=_dump_json_bytes(
                {"error": {"type": "api_error", "message": f"llm-gc could not reach the provider: {type(err).__name__}"}}
            ),
            status_code=502,
            media_type="application/json",
        )

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers={k: v for k, v in upstream.headers.items() if k.lower() not in _RESPONSE_HOP_BY_HOP},
        media_type=upstream.headers.get("content-type", "application/json"),
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
