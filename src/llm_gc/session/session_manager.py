import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from llm_gc.config.constants import DEFAULT_SESSION_LOCK_TIMEOUT_MS
from llm_gc.session.session_state import SessionState
from llm_gc.session.session_store import SessionStore

logger = logging.getLogger(__name__)


class SessionLockTimeout(Exception):
    """Raised when a session's lock could not be acquired within the budget.

    This is an operational error, not a bug: it means another coroutine held
    the session's lock longer than `session_lock_timeout_ms`. The caller is
    expected to catch it and decide a fallback, not to crash.
    """

    def __init__(self, session_id: str, timeout_ms: float) -> None:
        super().__init__(f"Could not acquire lock for session {session_id!r} within {timeout_ms}ms")
        self.session_id = session_id
        self.timeout_ms = timeout_ms


class SessionManager:
    """Coordinates concurrent access to per-session state under one event loop.

    Wraps a SessionStore and adds the concurrency contract: all access to one
    session's state happens under a per-session asyncio.Lock, so a
    read-modify-write that spans an `await` cannot interleave with another
    coroutine touching the same session. Different sessions each get their own
    lock and never block each other.

    The lock exists because `gc.update` (and later `gc.collect`) load state, do
    async work, then write state back — and the gap between the load and the
    write contains `await` points. Without the lock, two updates for the same
    session could both read the same starting state and one would silently
    overwrite the other's work. The lock closes that window.
    """

    def __init__(
        self,
        store: SessionStore | None = None,
        lock_timeout_ms: float = DEFAULT_SESSION_LOCK_TIMEOUT_MS,
    ) -> None:
        self._store = store if store is not None else SessionStore()
        self._locks: dict[str, asyncio.Lock] = {}
        self._lock_timeout_ms = lock_timeout_ms

    @property
    def store(self) -> SessionStore:
        """The underlying store. Read-only access for callers that only need to
        inspect session existence (e.g. the idle sweeper listing active ids)."""
        return self._store

    def _lock_for(self, session_id: str) -> asyncio.Lock:
        """Return the lock for a session, creating it on first use.

        Safe check-then-set: there is no `await` between the get and the set, so
        no other coroutine can interleave and create a second, competing lock
        for the same id.
        """
        lock = self._locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_id] = lock
        return lock

    @asynccontextmanager
    async def acquire(self, session_id: str) -> AsyncIterator[SessionState]:
        """Hold a session's lock for the duration of a `with` block.

            async with manager.acquire("s1") as state:
                state.messages.append(msg)   # exclusive: no other coroutine
                                             # can be inside "s1" right now

        Waits at most `lock_timeout_ms` for the lock. On timeout, raises
        SessionLockTimeout instead of hanging forever, so one stuck holder can't
        freeze every later access to the same session. The lock is always
        released on exit, including when the body raises.
        """
        lock = self._lock_for(session_id)
        try:
            async with asyncio.timeout(self._lock_timeout_ms / 1000):
                await lock.acquire()
        except TimeoutError as e:
            logger.warning(
                "Session lock acquisition timed out",
                extra={"session_id": session_id, "timeout_ms": self._lock_timeout_ms},
            )
            raise SessionLockTimeout(session_id, self._lock_timeout_ms) from e

        try:
            state = self._store.get_or_create(session_id)
            state.touch()
            yield state
        finally:
            lock.release()

