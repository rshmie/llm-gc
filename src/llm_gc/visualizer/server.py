"""The live dashboard: every session the engine is serving, as it happens.

Each value is produced by an engine, monitor, proxy or transcript component and
only displayed here, so a wrong number is wrong in the component that produced it.
Its one action is the developer's manual cut. It shows conversation content, so it
is opt-in and must only ever listen on localhost; see
doc/design/visibility/overview.md.
"""

from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

from llm_gc.config.constants import APP_VERSION
from llm_gc.engine import AbandonedCut, AgingDegradation, CutOutcome, GcService, GCStatus
from llm_gc.engine.pause import Pause
from llm_gc.engine.switch_policy import CacheReading, CacheState
from llm_gc.monitoring import ContextHealth, ContextHealthMonitor
from llm_gc.monitoring.knowledge_state import CacheReadingState, KnowledgeState, project_knowledge_state
from llm_gc.session import SessionState
from llm_gc.visualizer.transcript import TranscriptRecorder, TranscriptTurn

_STATIC_DIR = Path(__file__).parent / "static"
# Without it the browser guesses a cache lifetime and keeps running an old dashboard
# after the files change; with it, each load revalidates by ETag, which is cheap.
_REVALIDATE = {"Cache-Control": "no-cache"}


class _RevalidatedStaticFiles(StaticFiles):
    def file_response(self, *args: Any, **kwargs: Any) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers.update(_REVALIDATE)
        return response

# Only localhost names are served. A page on another site can make the browser
# resolve its own hostname to 127.0.0.1 (DNS rebinding) and read the responses as
# same-origin; checking the Host header defeats that.
LOCAL_HOSTS = ["127.0.0.1", "localhost"]

# Required on the one route that acts. A page on another site cannot send a custom
# header without a CORS preflight, which this server never grants.
ACTION_HEADER = "x-llm-gc-dashboard"

# Longest opening line shown for a conversation in the session picker.
SESSION_TITLE_MAX_CHARS = 72


class SessionSummary(BaseModel):
    session_id: str
    title: str | None             # its opening words, when the transcript was captured
    client_session: str | None    # the client's own session it came from, when the client names one
    agent: str | None             # the client's sub-agent it belongs to; None for a main conversation
    started_at: datetime
    version: int                  # changes whenever the session's content does
    recorded_turns: int
    cuts: int
    aging_in_flight: bool
    last_activity: datetime
    newest_reply: int | None = None   # the turn index of the reply being shown, which arrives without a new version
    # The provider's cache for it as the proxy reads it now; None without a proxy. Polled with the overview because
    # it changes with time, not with the conversation.
    cache: CacheReadingState | None = None


class Transcript(BaseModel):
    captured: bool                # False when no transcript was recorded for the session
    turns: list[TranscriptTurn]
    total_tokens: int             # what the client's conversation costs sent whole
    recorded_tokens: int          # the same, for only the turns the session has recorded -
                                  # what to compare with the context the model is sent
    reply: TranscriptTurn | None = None   # the newest reply, shown until the next request records it


class ManualCutResult(BaseModel):
    status: GCStatus
    tokens_before: int
    tokens_after: int
    outcome: CutOutcome | None            # set when a cut was made: where its turns ended up
    degradations: list[AgingDegradation]
    abandoned_cut: AbandonedCut | None    # set when the cut would not have shrunk the context
    duration_ms: float


class DashboardOverview(BaseModel):
    sessions: list[SessionSummary]   # most recently active first
    config: dict[str, Any]
    proxy: dict[str, Any] | None     # None when the dashboard is not running beside a proxy
    pause: Pause | None = None       # set while LLM-GC's own model calls yield to the upstream's rate limit


def create_dashboard_app(
    gc_service: GcService,
    monitor: ContextHealthMonitor,
    read_proxy_stats: Callable[[], dict[str, Any]] | None = None,
    transcripts: TranscriptRecorder | None = None,
    read_trace: Callable[[str], dict[str, Any] | None] | None = None,
    read_cache: Callable[[str], CacheReading] | None = None,
    read_tool_round: Callable[[str], bool] | None = None,
    allowed_hosts: list[str] | None = None,
) -> FastAPI:
    """The dashboard over a running engine.

    Args:
        gc_service: The engine whose sessions are shown.
        monitor: Subscribed to the same bus as `gc_service`.
        read_proxy_stats: The proxy's counters, passed as a function so the
            dashboard does not depend on the proxy package.
        transcripts: What each client last sent, fed by the proxy. Without it the
            dashboard still shows what the model receives, but not what was said.
        read_trace: A client session's trace from the proxy's request log, passed as
            a function for the same reason as `read_proxy_stats`.
        read_cache: A conversation's prompt cache as the proxy reads it, for the verdict
            on a staged cut. Without it the verdict is for a cold cache, which is what a
            conversation with no proxy reading its cache meets.
        read_tool_round: Whether a conversation's next request will answer tool calls, as
            the proxy's request log tells. No cut is switched in while one is open; without
            a proxy, none ever is.
        allowed_hosts: Host names served; localhost by default.

    Every route is `async` on purpose: FastAPI runs plain `def` routes in a thread
    pool, and the session store and monitor are only safe to touch from the event
    loop that writes them.
    """
    app = FastAPI(title="LLM-GC Dashboard", version=APP_VERSION)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts or LOCAL_HOSTS)
    config = gc_service.gc_config

    @app.get("/", include_in_schema=False)
    async def serve_page() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html", headers=_REVALIDATE)

    # The page's styles and scripts, shipped inside the package so no build step is needed.
    app.mount("/static", _RevalidatedStaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/api/overview")
    async def overview() -> DashboardOverview:
        store = gc_service.session_manager.store
        states = [state for session_id in store.active_session_ids()
                  if (state := store.get(session_id)) is not None]
        sessions = [_summarise(state, transcripts, read_cache)
                    for state in sorted(states, key=lambda state: state.last_updated_at, reverse=True)]
        return DashboardOverview(
            sessions=sessions,
            config={"context_window": config.context_window, "min_cut_tokens": config.min_cut_tokens,
                    "last_n_turns_to_keep": config.last_n_turns_to_keep, "keep_threshold": config.keep_threshold,
                    "archive_threshold": config.archive_threshold, "level_margin": config.level_margin,
                    "max_recap_tokens": config.max_recap_tokens, "max_recap_ratio": config.max_recap_ratio,
                    "min_saving_share": config.min_saving_share,
                    "max_stale_ratio": config.max_stale_ratio,
                    # Whose turns facts are read from: without a model, only the user's.
                    "facts_read_from": sorted(gc_service.generational_memory.fact_reader.speakers)},
            proxy=read_proxy_stats() if read_proxy_stats is not None else None,
            pause=gc_service.model_calls.current(),
        )

    @app.get("/api/sessions/{session_id}", response_model=KnowledgeState)
    async def session_knowledge(session_id: str) -> KnowledgeState | JSONResponse:
        cache = read_cache(session_id) if read_cache is not None else None
        inspection = gc_service.inspect(session_id, cache=cache.state if cache is not None else CacheState.COLD,
                                        tool_round_open=_round_open(read_tool_round, session_id))
        if inspection is None:
            return _session_not_found(session_id)
        return project_knowledge_state(inspection, cache)

    @app.get("/api/sessions/{session_id}/activity", response_model=ContextHealth)
    async def session_activity(session_id: str) -> ContextHealth | JSONResponse:
        if gc_service.session_manager.store.get(session_id) is None:
            return _session_not_found(session_id)
        return monitor.get_snapshot(session_id)

    @app.get("/api/sessions/{session_id}/transcript", response_model=Transcript)
    async def session_transcript(session_id: str) -> Transcript | JSONResponse:
        state = gc_service.session_manager.store.get(session_id)
        if state is None:
            if transcripts is not None:
                transcripts.forget(session_id)   # its session closed; so does its copy
            return _session_not_found(session_id)
        turns = transcripts.get(session_id) if transcripts is not None else None
        recorded = state.snapshot.recorded_turn_count
        return Transcript(captured=turns is not None, turns=list(turns or ()),
                          total_tokens=sum(turn.tokens for turn in turns or ()),
                          recorded_tokens=sum(turn.tokens for turn in turns or () if turn.turn_index < recorded),
                          reply=transcripts.reply(session_id) if transcripts is not None else None)

    @app.get("/api/trace/{client_session}")
    async def session_trace(client_session: str) -> JSONResponse:
        """One client session as its turns, their requests and the subagents each started."""
        trace = read_trace(client_session) if read_trace is not None else None
        if trace is None:
            return JSONResponse(status_code=404, content={"error": {
                "code": "TRACE_NOT_FOUND", "message": f"no requests named client session {client_session!r}"}})
        return JSONResponse(_with_words(trace, transcripts))

    @app.post("/api/sessions/{session_id}/gc", response_model=ManualCutResult)
    async def run_gc_now(session_id: str, request: Request) -> ManualCutResult | JSONResponse:
        """The developer's lever: prepare a cut without waiting for the batch minimum, and switch it in now.

        The switch policy is waived; every other rule of a cut still applies. It reports
        the collection when one was switched in, and the pass otherwise, the same way an
        automatic one is reported.
        """
        if request.headers.get(ACTION_HEADER) != "1":
            return JSONResponse(status_code=403, content={"error": {
                "code": "ACTION_HEADER_REQUIRED", "message": f"send {ACTION_HEADER}: 1 to act from the dashboard"}})
        if gc_service.session_manager.store.get(session_id) is None:
            return _session_not_found(session_id)
        prepared = await gc_service.gc_age(session_id, ignore_batch_minimum=True)
        # Mid tool round the switch is held back and made by the first request after the round.
        switched = gc_service.gc_switch(session_id, CacheState.COLD, requested=True,
                                        tool_round_open=_round_open(read_tool_round, session_id))
        result = switched if switched is not None else prepared
        return ManualCutResult(status=result.status, tokens_before=result.tokens_before,
                               tokens_after=result.tokens_in_final, outcome=result.outcome,
                               degradations=result.degradations, abandoned_cut=prepared.abandoned_cut,
                               duration_ms=prepared.duration_ms + (switched.duration_ms if switched else 0.0))

    return app


def _round_open(read_tool_round: Callable[[str], bool] | None, session_id: str) -> bool:
    return read_tool_round(session_id) if read_tool_round is not None else False


def _summarise(state: SessionState, transcripts: TranscriptRecorder | None,
               read_cache: Callable[[str], CacheReading] | None) -> SessionSummary:
    origin = transcripts.origin(state.session_id) if transcripts is not None else None
    cache = read_cache(state.session_id) if read_cache is not None else None
    return SessionSummary(session_id=state.session_id,
                          title=_opening_words(transcripts.get(state.session_id) if transcripts is not None else None),
                          client_session=origin.session_id if origin is not None else None,
                          agent=origin.agent_id if origin is not None else None,
                          started_at=state.created_at,
                          version=state.snapshot.version,
                          recorded_turns=state.snapshot.recorded_turn_count, cuts=state.snapshot.cut_count,
                          aging_in_flight=state.aging_in_flight, last_activity=state.last_updated_at,
                          newest_reply=_reply_index(transcripts, state.session_id),
                          cache=CacheReadingState(state=cache.state, uses_cache=cache.uses_cache, idle_s=cache.idle_s,
                                                  lifetime_s=cache.lifetime_s) if cache is not None else None)


def _reply_index(transcripts: TranscriptRecorder | None, session_id: str) -> int | None:
    reply = transcripts.reply(session_id) if transcripts is not None else None
    return reply.turn_index if reply is not None else None


def _opening_words(turns: Sequence[TranscriptTurn] | None) -> str | None:
    """How a person recognises a conversation: the first words they wrote in it, on one line and shortened."""
    first = next((turn.person_text for turn in turns or ()
                  if turn.role == "user" and not turn.relayed and turn.person_text.strip()), None)
    return _one_line(first) if first is not None else None


def _one_line(text: str) -> str:
    line = " ".join(text.split())
    return line if len(line) <= SESSION_TITLE_MAX_CHARS else line[:SESSION_TITLE_MAX_CHARS - 1].rstrip() + "…"


def _with_words(trace: dict[str, Any], transcripts: TranscriptRecorder | None) -> dict[str, Any]:
    """The trace with what a person recognises it by: each turn's message, each subagent's task.

    From the transcripts the dashboard already holds, so only while a conversation is
    live; a closed one's words are gone, and its turns show without them.
    """
    def said(conversation_id: str | None, position: int) -> str | None:
        turns = transcripts.get(conversation_id) if transcripts is not None and conversation_id else None
        text = next((turn.person_text for turn in turns or () if turn.turn_index == position - 1), None)
        return _one_line(text) if text and text.strip() else None

    def tasked(node: dict[str, Any]) -> dict[str, Any]:
        conversation = node["conversation_id"]
        turns = transcripts.get(conversation) if transcripts is not None and conversation else None
        return {**node, "words": _opening_words(turns)}

    return {**trace, "subagents_before": [tasked(node) for node in trace["subagents_before"]],
            "turns": [{**turn, "words": said(turn["conversation_id"], turn["position"]),
                       "subagents": [tasked(node) for node in turn["subagents"]]} for turn in trace["turns"]]}


def _session_not_found(session_id: str) -> JSONResponse:
    # Sessions close when idle, so a dashboard left open will ask for one that has gone.
    return JSONResponse(status_code=404, content={"error": {
        "code": "SESSION_NOT_FOUND", "message": f"no live session {session_id!r}; it may have been closed"}})
