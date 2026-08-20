import logging

from llm_gc.session.session_state import SessionState

logger = logging.getLogger(__name__)


class SessionStore:
    """In-memory store of active sessions, keyed by session id.

    The single source of truth for "which sessions exist and what is their
    state". It hands out SessionState objects by id, creating one the first
    time an id is seen.

    Concurrency note: the methods here do no `await` between reading and writing
    `_sessions`, so under a single asyncio event loop they run to completion
    without another coroutine interleaving. The per-session locking that guards
    the read-modify-write of a session's *contents* lives one layer up, in the
    session manager, not here.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def get_or_create(self, session_id: str) -> SessionState:
        """Return the state for `session_id`, creating a fresh one if unseen."""
        state = self._sessions.get(session_id)
        if state is None:
            state = SessionState(session_id=session_id)
            self._sessions[session_id] = state
            logger.info("Created new session", extra={"session_id": session_id})
        else:
            logger.debug("Reusing existing session", extra={"session_id": session_id})
        return state

    def get(self, session_id: str) -> SessionState | None:
        """Return the state for `session_id`, or None if it does not exist."""
        return self._sessions.get(session_id)

    def remove(self, session_id: str) -> SessionState | None:
        """Drop a session, returning its state if it was present.

        Called on session close / idle timeout to release in-memory state.
        Returns None (rather than raising) when the id is already gone, so a
        double close is a harmless no-op.
        """
        return self._sessions.pop(session_id, None)

    def active_session_ids(self) -> list[str]:
        """A snapshot list of the currently-tracked session ids.

        A copy, not a live view: the caller can iterate it while sessions are
        being added or removed without tripping over a mutating dict.
        """
        return list(self._sessions)

    def count(self) -> int:
        return len(self._sessions)

