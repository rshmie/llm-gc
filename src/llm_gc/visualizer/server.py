"""The live dashboard: every session the engine is serving, as it happens.

A read-only view. Each value is produced by an engine, monitor or proxy component
and only displayed here, so a wrong number is wrong in the component that produced
it. It shows conversation content, so it is opt-in and must only ever listen on
localhost; see doc/design/visibility/overview.md.
"""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from llm_gc.config.constants import APP_VERSION
from llm_gc.engine import GcService
from llm_gc.monitoring import ContextHealth, ContextHealthMonitor
from llm_gc.monitoring.knowledge_state import KnowledgeState, project_knowledge_state

_STATIC_DIR = Path(__file__).parent / "static"


class SessionSummary(BaseModel):
    session_id: str
    version: int                  # changes whenever the session's content does
    recorded_turns: int
    cuts: int
    aging_in_flight: bool
    last_activity: datetime


class DashboardOverview(BaseModel):
    sessions: list[SessionSummary]   # most recently active first
    config: dict[str, Any]
    proxy: dict[str, Any] | None     # None when the dashboard is not running beside a proxy


def create_dashboard_app(
    gc_service: GcService,
    monitor: ContextHealthMonitor,
    read_proxy_stats: Callable[[], dict[str, Any]] | None = None,
) -> FastAPI:
    """The dashboard over a running engine.

    Args:
        gc_service: The engine whose sessions are shown; read, never written.
        monitor: Subscribed to the same bus as `gc_service`.
        read_proxy_stats: The proxy's counters, passed as a function so the
            dashboard does not depend on the proxy package.

    Every route is `async` on purpose: FastAPI runs plain `def` routes in a thread
    pool, and the session store and monitor are only safe to read from the event
    loop that writes them.
    """
    app = FastAPI(title="LLM-GC Dashboard", version=APP_VERSION)
    config = gc_service.gc_config

    @app.get("/", include_in_schema=False)
    async def serve_page() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    @app.get("/api/overview")
    async def overview() -> DashboardOverview:
        store = gc_service.session_manager.store
        states = [state for session_id in store.active_session_ids()
                  if (state := store.get(session_id)) is not None]
        sessions = [
            SessionSummary(session_id=state.session_id, version=state.snapshot.version,
                           recorded_turns=state.snapshot.recorded_turn_count, cuts=state.snapshot.cut_count,
                           aging_in_flight=state.aging_in_flight, last_activity=state.last_updated_at)
            for state in sorted(states, key=lambda state: state.last_updated_at, reverse=True)
        ]
        return DashboardOverview(
            sessions=sessions,
            config={"context_window": config.context_window, "min_cut_tokens": config.min_cut_tokens,
                    "last_n_turns_to_keep": config.last_n_turns_to_keep, "keep_threshold": config.keep_threshold,
                    "archive_threshold": config.archive_threshold, "max_memory_tokens": config.max_memory_tokens},
            proxy=read_proxy_stats() if read_proxy_stats is not None else None,
        )

    @app.get("/api/sessions/{session_id}", response_model=KnowledgeState)
    async def session_knowledge(session_id: str) -> KnowledgeState | JSONResponse:
        inspection = gc_service.inspect(session_id)
        if inspection is None:
            return _session_not_found(session_id)
        return project_knowledge_state(inspection)

    @app.get("/api/sessions/{session_id}/activity", response_model=ContextHealth)
    async def session_activity(session_id: str) -> ContextHealth | JSONResponse:
        if gc_service.session_manager.store.get(session_id) is None:
            return _session_not_found(session_id)
        return monitor.get_snapshot(session_id)

    return app


def _session_not_found(session_id: str) -> JSONResponse:
    # Sessions close when idle, so a dashboard left open will ask for one that has gone.
    return JSONResponse(status_code=404, content={"error": {
        "code": "SESSION_NOT_FOUND", "message": f"no live session {session_id!r}; it may have been closed"}})
