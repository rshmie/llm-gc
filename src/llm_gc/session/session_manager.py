import asyncio
import logging
from collections.abc import Sequence
from datetime import datetime, timezone

from llm_gc.config.constants import DEFAULT_MAX_SESSIONS, DEFAULT_SESSION_SWEEP_INTERVAL_S, DEFAULT_SESSION_TIMEOUT_S
from llm_gc.events import Event, EventBus, EventType
from llm_gc.models.generation_snapshot import GenerationSnapshot
from llm_gc.session.session_state import SessionState
from llm_gc.session.session_store import SessionStore

logger = logging.getLogger(__name__)


class SessionManager:
    """Owns session lifecycle: lookup, close, and idle expiry.

    There is no per-session lock. Session data lives in an immutable snapshot that
    writers replace in one synchronous step, so a reader can never see a partial
    update and never has to wait for a writer. See doc/design/session/overview.md.

    Each close is announced on `event_bus` as `SESSION_CLOSED`, so a component that
    keeps its own copy of a session's data can drop it with the session.
    """

    def __init__(
        self,
        store: SessionStore | None = None,
        session_timeout_s: float = DEFAULT_SESSION_TIMEOUT_S,
        sweep_interval_s: float = DEFAULT_SESSION_SWEEP_INTERVAL_S,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        event_bus: EventBus | None = None,
    ) -> None:
        if max_sessions < 1:
            raise ValueError(f"max_sessions must be at least 1, got {max_sessions}")
        self._store = store if store is not None else SessionStore()
        self._session_timeout_s = session_timeout_s
        self._sweep_interval_s = sweep_interval_s
        self._max_sessions = max_sessions
        self._event_bus = event_bus
        # Chain hash of each session's last recorded turn -> session id, so a
        # request can be matched to the conversation it continues by content.
        self._sessions_by_tip: dict[str, str] = {}
        self._tip_of: dict[str, str] = {}

    @property
    def store(self) -> SessionStore:
        return self._store

    def get_or_create(self, session_id: str) -> SessionState:
        """The live state for a session, created on first use, with its idle clock reset.

        A new session past `max_sessions` closes the least recently active other one.
        """
        is_new = self._store.get(session_id) is None
        state = self._store.get_or_create(session_id)
        state.touch()
        if is_new:
            self._close_beyond_cap(keep=session_id)
        return state

    def is_live(self, state: SessionState) -> bool:
        """Whether `state` is still the session the store holds for its id.

        False once the session was closed, including when a new conversation has
        since reopened the same id. A writer that started before the close checks
        this before committing, so closed sessions stay closed.
        """
        return self._store.get(state.session_id) is state

    def find_by_lineage(self, hashes: Sequence[str]) -> tuple[SessionState, int] | None:
        """The session whose recorded conversation is the longest prefix of `hashes`.

        Returns the session and how many of the turns it has already recorded, or
        None if no session's last recorded turn matches any prefix. A match means
        the two conversations are identical up to that point, so nothing a session
        holds can belong to a different conversation.
        """
        for length in range(len(hashes), 0, -1):
            session_id = self._sessions_by_tip.get(hashes[length - 1])
            state = self._store.get(session_id) if session_id is not None else None
            if state is not None and state.snapshot.tip == hashes[length - 1]:
                return state, length
        return None

    def retip(self, state: SessionState) -> None:
        """Re-index a session after its recorded conversation changed."""
        stale = self._tip_of.pop(state.session_id, None)
        if stale is not None and self._sessions_by_tip.get(stale) == state.session_id:
            del self._sessions_by_tip[stale]
        tip = state.snapshot.tip
        if tip is not None:
            self._sessions_by_tip[tip] = state.session_id
            self._tip_of[state.session_id] = tip

    def replace(self, state: SessionState, snapshot: GenerationSnapshot) -> SessionState:
        """Start the session over from `snapshot`, under the same id, as a new object.

        A new object rather than an edit, so an aging pass still running on the old
        one finds it no longer live and discards its result instead of committing
        changes planned against a conversation that has since been rewound.
        """
        fresh = SessionState(
            session_id=state.session_id,
            snapshot=snapshot,
            context_window=state.context_window,
            token_rate=state.token_rate,
            collection_enabled=state.collection_enabled,
        )
        self._store.put(fresh)
        self.retip(fresh)
        return fresh

    def _close_beyond_cap(self, keep: str) -> None:
        held = [
            state
            for session_id in self._store.active_session_ids()
            if session_id != keep and (state := self._store.get(session_id)) is not None
        ]
        excess = len(held) + 1 - self._max_sessions
        for state in sorted(held, key=lambda s: s.last_updated_at)[: max(0, excess)]:
            self.close_session(state.session_id)
            logger.info(
                "closed the least recently active session to stay within the cap",
                extra={"session_id": state.session_id, "max_sessions": self._max_sessions},
            )

    def close_session(self, session_id: str) -> None:
        """Drop a session and all of its data, and announce it. A second close is a no-op."""
        closed = self._store.remove(session_id)
        stale = self._tip_of.pop(session_id, None)
        if stale is not None and self._sessions_by_tip.get(stale) == session_id:
            del self._sessions_by_tip[stale]
        # Announced once the session is gone, so a listener never finds it still held.
        if closed is not None and self._event_bus is not None:
            self._event_bus.emit(Event(event_type=EventType.SESSION_CLOSED, data={"session_id": session_id}))

    def sweep_once(self, now: datetime | None = None) -> list[str]:
        """Close every session idle longer than `session_timeout_s`; return their ids.

        `now` is injectable so tests do not wait real time.
        """
        current_time = now if now is not None else datetime.now(timezone.utc)
        closed: list[str] = []
        for session_id in self._store.active_session_ids():
            state = self._store.get(session_id)
            if state is None:
                continue
            if (current_time - state.last_updated_at).total_seconds() >= self._session_timeout_s:
                self.close_session(session_id)
                closed.append(session_id)
        return closed

    async def run_idle_sweeper(self) -> None:
        """Sweep for idle sessions every `sweep_interval_s` until cancelled.

        One failed pass is logged and the loop continues. Cancellation is a
        `BaseException`, so it passes the guard and ends the loop.
        """
        while True:
            await asyncio.sleep(self._sweep_interval_s)
            try:
                closed = self.sweep_once()
                if closed:
                    logger.info("Idle sweep closed sessions", extra={"closed_count": len(closed)})
            except Exception:
                logger.exception("Idle sweep pass failed; continuing")
