import asyncio
import logging
from datetime import datetime, timezone

from llm_gc.config.constants import DEFAULT_SESSION_SWEEP_INTERVAL_S, DEFAULT_SESSION_TIMEOUT_S
from llm_gc.session.session_state import SessionState
from llm_gc.session.session_store import SessionStore

logger = logging.getLogger(__name__)


class SessionManager:
    """Owns session lifecycle: lookup, close, and idle expiry.

    There is no per-session lock. Session data lives in an immutable snapshot that
    writers replace in one synchronous step, so a reader can never see a partial
    update and never has to wait for a writer. See doc/design/session/overview.md.
    """

    def __init__(
        self,
        store: SessionStore | None = None,
        session_timeout_s: float = DEFAULT_SESSION_TIMEOUT_S,
        sweep_interval_s: float = DEFAULT_SESSION_SWEEP_INTERVAL_S,
    ) -> None:
        self._store = store if store is not None else SessionStore()
        self._session_timeout_s = session_timeout_s
        self._sweep_interval_s = sweep_interval_s

    @property
    def store(self) -> SessionStore:
        return self._store

    def get_or_create(self, session_id: str) -> SessionState:
        """The live state for a session, created on first use, with its idle clock reset."""
        state = self._store.get_or_create(session_id)
        state.touch()
        return state

    def is_live(self, state: SessionState) -> bool:
        """Whether `state` is still the session the store holds for its id.

        False once the session was closed, including when a new conversation has
        since reopened the same id. A writer that started before the close checks
        this before committing, so closed sessions stay closed.
        """
        return self._store.get(state.session_id) is state

    def close_session(self, session_id: str) -> None:
        """Drop a session and all of its data. A second close is a no-op."""
        self._store.remove(session_id)

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
