import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from llm_gc.config.constants import (
    DEFAULT_SESSION_CLOSE_DRAIN_MS,
    DEFAULT_SESSION_LOCK_TIMEOUT_MS,
    DEFAULT_SESSION_SWEEP_INTERVAL_S,
    DEFAULT_SESSION_TIMEOUT_S,
)
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
        session_timeout_s: float = DEFAULT_SESSION_TIMEOUT_S,
        sweep_interval_s: float = DEFAULT_SESSION_SWEEP_INTERVAL_S,
        close_drain_ms: float = DEFAULT_SESSION_CLOSE_DRAIN_MS,
    ) -> None:
        self._store = store if store is not None else SessionStore()
        self._locks: dict[str, asyncio.Lock] = {}
        self._lock_timeout_ms = lock_timeout_ms
        self._session_timeout_s = session_timeout_s
        self._sweep_interval_s = sweep_interval_s
        self._close_drain_ms = close_drain_ms

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

    async def close_session(self, session_id: str) -> None:
        """Close a session: drain any in-flight work, drop its state, keep its lock.

        Acquires the session's lock first (draining an in-flight update, up to
        `close_drain_ms`), then removes the state from the store. The lock object
        is deliberately KEPT in the registry: every `acquire` for this id must
        return the *same* lock, so a request that arrives during or after the
        close serialises on it and can never end up holding a second, competing
        lock for the same session. A later request re-creates fresh state through
        that lock - a reopen. The kept lock is a small, bounded leak (one per
        session id ever seen); bounded lock cleanup is a future refinement.

        Removing the lock instead would reintroduce the two-lock race: a waiter
        queued on the old lock and a newcomer that mints a fresh one would both
        run inside the session at once. See doc/design/session/overview.md.
        """
        lock = self._lock_for(session_id)
        try:
            async with asyncio.timeout(self._close_drain_ms / 1000):
                await lock.acquire()
        except TimeoutError:
            # An operation held the lock past the drain window. Close anyway: it
            # keeps its own reference to the state object and finishes on it;
            # dropping the store entry just means the next request reopens.
            logger.warning("close_session drain timed out; closing without waiting further",
                           extra={"session_id": session_id, "drain_ms": self._close_drain_ms})
            self._store.remove(session_id)
            return
        try:
            self._store.remove(session_id)
        finally:
            lock.release()

    async def sweep_once(self, now: datetime | None = None) -> list[str]:
        """One idle-timeout pass: close every session whose last activity is older
        than `session_timeout_s`. Returns the ids closed.

        Iterates a *snapshot* of the active ids (a copy), so sessions added or
        removed during the pass don't disturb the iteration. `now` is injectable
        so tests don't have to wait real time.
        """
        current_time = now if now is not None else datetime.now(timezone.utc)
        closed: list[str] = []
        for session_id in self._store.active_session_ids():
            state = self._store.get(session_id)
            if state is None:
                continue  # already closed between the snapshot and now
            idle_seconds = (current_time - state.last_updated_at).total_seconds()
            if idle_seconds >= self._session_timeout_s:
                await self.close_session(session_id)
                closed.append(session_id)
        return closed

    async def run_idle_sweeper(self) -> None:
        """Long-running background task: sweep for idle sessions every
        `sweep_interval_s`, until cancelled.

        The caller (the proxy) starts it with `asyncio.create_task` and stops it
        with `task.cancel()`. `CancelledError` is a `BaseException`, not an
        `Exception`, so it is not caught by the guard below - it propagates and
        ends the loop cleanly. One failed pass is logged and the loop continues,
        so a transient error never kills the sweeper.
        """
        while True:
            await asyncio.sleep(self._sweep_interval_s)
            try:
                closed = await self.sweep_once()
                if closed:
                    logger.info("Idle sweep closed sessions", extra={"closed_count": len(closed)})
            except Exception:
                logger.exception("Idle sweep pass failed; continuing")

