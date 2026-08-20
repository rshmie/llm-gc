import asyncio

import pytest

from llm_gc.models import Message
from llm_gc.session import SessionLockTimeout, SessionManager, SessionState, SessionStore


def _msg(content: str = "x") -> Message:
    return Message(role="user", content=content)


class TestLockSerialization:
    """The core reason SessionManager exists: serialize a read-modify-write that
    spans an `await`, so two coroutines on the same session can't lose an update.
    """

    def test_without_a_lock_two_writers_lose_an_update(self):
        """Baseline: this is the bug the lock prevents. Two coroutines each read
        the message count, `await` (letting the other run), then append based on
        the count they read. Because they interleave at the await, they both read
        the same starting length and one append is lost."""
        store = SessionStore()

        async def racy_append():
            state = store.get_or_create("s1")
            count_before = len(state.messages)
            await asyncio.sleep(0)  # yield point — the other coroutine runs here
            # Rebuild the list from the stale count, dropping any concurrent append.
            state.messages = state.messages[:count_before] + [_msg()]

        async def scenario():
            await asyncio.gather(racy_append(), racy_append())

        asyncio.run(scenario())
        # Two appends happened, but one was clobbered: only one survives.
        assert len(store.get("s1").messages) == 1

    def test_with_the_lock_both_writers_survive(self):
        """The fix: the same racy body, but run under `manager.acquire`. The lock
        forces the two coroutines to run their read-modify-write one after the
        other, so both appends land."""
        manager = SessionManager()

        async def guarded_append():
            async with manager.acquire("s1") as state:
                count_before = len(state.messages)
                await asyncio.sleep(0)  # yield point — but the lock is held
                state.messages = state.messages[:count_before] + [_msg()]

        async def scenario():
            await asyncio.gather(guarded_append(), guarded_append())

        asyncio.run(scenario())
        assert len(manager.store.get("s1").messages) == 2


class TestAcquireContract:
    def test_acquire_yields_the_same_state_object_across_calls(self):
        manager = SessionManager()

        async def scenario():
            async with manager.acquire("s1") as first:
                first.messages.append(_msg("hello"))
            async with manager.acquire("s1") as second:
                return first is second, len(second.messages)

        same, count = asyncio.run(scenario())
        assert same is True
        assert count == 1

    def test_acquire_touches_the_session(self):
        manager = SessionManager()

        async def scenario():
            before = SessionState(session_id="s1").created_at
            async with manager.acquire("s1") as state:
                return state.last_updated_at >= before

        assert asyncio.run(scenario()) is True

    def test_lock_is_released_even_when_the_body_raises(self):
        """The `finally: lock.release()` matters: if a body raises while holding
        the lock and the lock were not released, every later acquire on that
        session would deadlock. Here the second acquire must still succeed."""
        manager = SessionManager()

        async def scenario():
            with pytest.raises(ValueError):
                async with manager.acquire("s1"):
                    raise ValueError("boom")
            # If the lock leaked, this would block until the timeout and raise
            # SessionLockTimeout instead of succeeding.
            async with manager.acquire("s1") as state:
                return state.session_id

        assert asyncio.run(scenario()) == "s1"

    def test_different_sessions_do_not_block_each_other(self):
        """Distinct sessions have distinct locks, so holding one open must not
        stop another session from being acquired concurrently."""
        manager = SessionManager()
        order: list[str] = []

        async def hold(session_id: str, hold_time: float):
            async with manager.acquire(session_id):
                order.append(f"enter-{session_id}")
                await asyncio.sleep(hold_time)
                order.append(f"exit-{session_id}")

        async def scenario():
            # s1 holds for a beat; s2 should enter without waiting for s1 to exit.
            await asyncio.gather(hold("s1", 0.05), hold("s2", 0.0))

        asyncio.run(scenario())
        # s2 entered and exited before s1 exited → they did not serialize.
        assert order.index("enter-s2") < order.index("exit-s1")


class TestLockTimeout:
    def test_acquire_times_out_when_the_lock_is_held_too_long(self):
        """A holder that keeps the lock past the budget makes the next acquirer
        give up with SessionLockTimeout rather than wait forever."""
        manager = SessionManager(lock_timeout_ms=20)

        async def scenario():
            async def hold_forever():
                async with manager.acquire("s1"):
                    await asyncio.sleep(0.5)  # far longer than the 20ms budget

            holder = asyncio.create_task(hold_forever())
            await asyncio.sleep(0.01)  # let the holder grab the lock first
            try:
                with pytest.raises(SessionLockTimeout) as excinfo:
                    async with manager.acquire("s1"):
                        pass
                assert excinfo.value.session_id == "s1"
            finally:
                holder.cancel()

        asyncio.run(scenario())
