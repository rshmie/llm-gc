"""The ID of the HTTP request being handled. Application code.

Each request the proxy receives gets one ID, held for as long as the request runs. The
log lines written while handling it, its rows on the dashboard's Requests tab and any
error body LLM-GC writes itself all carry the same ID, so a row that looks wrong can be
followed to its log lines.

The ID is held in a `ContextVar` rather than passed down as an argument: the engine's
log lines are written deep inside the library, which knows nothing about HTTP requests.
"""

from contextvars import ContextVar
from uuid import uuid4

from starlette.types import ASGIApp, Receive, Scope, Send

# 48 random bits: among the 10,000 records the request log keeps, the chance that two
# share an ID is about one in five million, and 12 characters still fit in a log line.
REQUEST_ID_CHARS = 12

_current: ContextVar[str | None] = ContextVar("llm_gc_request_id", default=None)


def new_request_id() -> str:
    return uuid4().hex[:REQUEST_ID_CHARS]


def current_request_id() -> str | None:
    """The ID of the request being handled, or None outside one: at startup, or in the idle sweeper.

    Work a request schedules, such as the aging pass after its answer, keeps that request's ID:
    an asyncio task starts with a copy of the context it was created in.
    """
    return _current.get()


class RequestIdMiddleware:
    """Gives each HTTP request a new ID for as long as it runs.

    A plain ASGI middleware rather than code at the top of each route, because a streamed
    reply is relayed after the route has returned: the ID must stay set until the last
    chunk has been sent, so that a dropped stream is logged under its request.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        token = _current.set(new_request_id())
        try:
            await self.app(scope, receive, send)
        finally:
            _current.reset(token)
