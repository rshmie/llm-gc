"""What the hot path sees while an aging pass is waiting on its model call.

The compactor here blocks on a gate, so a test can act at the exact moment a real
compaction would be in flight: after the pass has planned, before it commits.
"""

import asyncio

from llm_gc.config import GCConfig
from llm_gc.engine.compaction import BaseCompactor, CompactionResult
from llm_gc.engine.compaction.compaction_strategy import CompactionStrategy
from llm_gc.engine.gc_service import GcService
from llm_gc.engine.gc_status import GCStatus
from llm_gc.engine.generations import GenerationalMemory
from llm_gc.events import Event, EventBus, EventType
from llm_gc.extraction import PatternFactReader
from llm_gc.models import Message
from llm_gc.session import SessionManager
from tests.engine.support import COMPACTED_SCORE, KEPT_SCORE, ScriptedSpanScorer, turn

SESSION = "s"


class _GatedCompactor(BaseCompactor):
    """Summarises only once `gate` is opened; `started` fires when it is waiting."""

    COMPACTION_STRATEGY = CompactionStrategy.NOOP

    def __init__(self, event_bus: EventBus) -> None:
        super().__init__(event_bus=event_bus)
        self.started = asyncio.Event()
        self.gate = asyncio.Event()

    async def compact(self, messages):
        self.started.set()
        await self.gate.wait()
        summary = Message(
            role="assistant",
            content=f"[Compacted] turns {messages[0].turn_index}-",
            token_count=1,
            turn_index=messages[0].turn_index,
        )
        return CompactionResult(
            summary=summary,
            original_token_count=sum(m.token_count for m in messages),
            compaction_strategy=self.COMPACTION_STRATEGY,
        )


def _service():
    bus = EventBus()
    compactor = _GatedCompactor(event_bus=bus)
    service = GcService(
        session_manager=SessionManager(),
        generational_memory=GenerationalMemory(fact_reader=PatternFactReader(), compactor=compactor),
        span_scorer=ScriptedSpanScorer({0: COMPACTED_SCORE, 1: COMPACTED_SCORE, 2: COMPACTED_SCORE}),  # type: ignore[arg-type]
        event_bus=bus,
        gc_config=GCConfig(context_window=1000, min_cut_tokens=0, last_n_messages_to_keep=0, archive_threshold=0.0),
    )
    return service, compactor


def _turn(index: int) -> Message:
    # Sized like a real turn, so a summary is worth buying and a cut pays for itself.
    return turn(index)


async def _start_aging(service: GcService, compactor: _GatedCompactor) -> asyncio.Task:
    """Record turns 0-2, then begin a pass that will block inside the compactor."""
    service.session_manager.get_or_create(SESSION)
    for index in (0, 1):
        state = service.session_manager.get_or_create(SESSION)
        state.snapshot = state.snapshot.with_turn(_turn(index))
    task = asyncio.create_task(service.gc_update(SESSION, _turn(2)))
    await asyncio.wait_for(compactor.started.wait(), timeout=1)
    return task


def _run(coro):
    """A gated test that deadlocks fails in two seconds instead of hanging the suite."""

    async def bounded():
        return await asyncio.wait_for(coro, timeout=2)

    return asyncio.run(bounded())


def _indices(messages: list[Message]) -> list[int]:
    return [m.turn_index for m in messages]


class TestTheHotPathNeverWaits:
    def test_collect_returns_immediately_while_aging_is_in_flight(self):
        """Before this design, collect waited on the session lock for the whole
        model call, then gave up after 100ms and the proxy bypassed GC."""
        service, compactor = _service()

        async def scenario():
            task = await _start_aging(service, compactor)
            context = await asyncio.wait_for(service.gc_collect(SESSION), timeout=0.05)
            compactor.gate.set()
            await task
            return context

        assert _indices(_run(scenario())) == [0, 1, 2], "the pre-pass context, whole and verbatim"

    def test_a_turn_recorded_mid_pass_is_visible_at_once_and_survives_the_commit(self):
        """The stale-context defect the snapshot design was chosen to prevent."""
        service, compactor = _service()

        async def scenario():
            task = await _start_aging(service, compactor)
            await service.gc_update(SESSION, _turn(3))
            during = await service.gc_collect(SESSION)
            compactor.gate.set()
            await task
            return during, await service.gc_collect(SESSION)

        during, after = _run(scenario())
        assert 3 in _indices(during), "recorded turns must not wait for aging to finish"
        assert _indices(after) == [-1, 3], "turns 0-2 are in the front; turn 3 kept"
        assert "[Summarised]" in after[0].content


class TestOnePassAtATime:
    def test_a_second_trigger_while_aging_is_reported_and_skipped(self):
        service, compactor = _service()
        finished: list[Event] = []
        service.event_bus.subscribe(EventType.GC_FINISHED, finished.append)

        async def scenario():
            task = await _start_aging(service, compactor)
            await service.gc_update(SESSION, _turn(3))
            compactor.gate.set()
            await task

        _run(scenario())
        statuses = [event.data["report"].status for event in finished]
        # The running pass stages its cut, and switching it in is a collection of its own.
        assert statuses == [GCStatus.BYPASSED_AGING_IN_FLIGHT, GCStatus.STAGED, GCStatus.COMPLETED]

    def test_the_in_flight_flag_is_cleared_even_when_the_pass_fails(self):
        """Otherwise one failed pass would disable aging for the session forever."""
        service, compactor = _service()

        async def failing(messages):
            raise RuntimeError("provider down")

        compactor.compact = failing  # type: ignore[method-assign]
        _run(service.gc_update(SESSION, _turn(0)))
        assert service.session_manager.store.get(SESSION).aging_in_flight is False


class TestAClosedSessionStaysClosed:
    def test_a_pass_finishing_after_close_writes_nothing(self):
        service, compactor = _service()
        finished: list[Event] = []
        service.event_bus.subscribe(EventType.GC_FINISHED, finished.append)

        async def scenario():
            task = await _start_aging(service, compactor)
            service.session_manager.close_session(SESSION)
            compactor.gate.set()
            await task

        _run(scenario())
        assert service.session_manager.store.get(SESSION) is None
        assert finished[-1].data["report"].status == GCStatus.DISCARDED_SESSION_CLOSED

    def test_a_reopened_id_does_not_inherit_the_old_conversations_pass(self):
        service, compactor = _service()

        async def scenario():
            task = await _start_aging(service, compactor)
            service.session_manager.close_session(SESSION)
            # A new conversation, same id. The old pass has scored already, so its turn 0
            # is kept from here on: any summary that appears came from the old pass.
            service.span_scorer.scores[0] = KEPT_SCORE  # type: ignore[attr-defined]
            reopened = asyncio.create_task(service.gc_update(SESSION, _turn(0)))
            compactor.gate.set()
            await asyncio.gather(task, reopened)
            return service.session_manager.store.get(SESSION).snapshot

        snapshot = _run(scenario())
        assert snapshot.old == (), "the old conversation's summary must not land in the new one"
        assert _indices(list(snapshot.young)) == [0]


class TestEventsDescribeCommittedChanges:
    def test_promotion_is_announced_after_the_commit_with_its_session(self):
        service, compactor = _service()
        promoted: list[tuple[Event, int]] = []

        def on_promoted(event: Event) -> None:
            promoted.append((event, len(service.session_manager.store.get(SESSION).snapshot.old)))

        service.event_bus.subscribe(EventType.MESSAGE_PROMOTED_TO_OLD_GEN, on_promoted)

        async def scenario():
            task = await _start_aging(service, compactor)
            assert promoted == [], "nothing is announced while the pass is still planning"
            compactor.gate.set()
            await task

        _run(scenario())
        event, old_gen_size_when_announced = promoted[0]
        assert event.data["session_id"] == SESSION
        committed = len(service.session_manager.store.get(SESSION).snapshot.old)
        assert old_gen_size_when_announced == committed == 3, (
            "a subscriber reading state sees the change it was told about"
        )


class TestACancelledPassLeavesNothingBehind:
    def test_cancellation_mid_compaction_clears_the_flag_and_writes_nothing(self):
        """Cancellation lands only at an `await`, and the commit contains none, so a
        pass is either wholly committed or not at all. `finally` clears the flag,
        or aging would stay disabled for the session forever."""
        service, compactor = _service()
        finished: list[Event] = []
        service.event_bus.subscribe(EventType.GC_FINISHED, finished.append)

        async def scenario():
            task = await _start_aging(service, compactor)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("cancellation must propagate, not be swallowed as a failure")
            return service.session_manager.store.get(SESSION)

        state = _run(scenario())
        assert state.aging_in_flight is False
        assert _indices(list(state.snapshot.young)) == [0, 1, 2], "the recorded turn stays; nothing aged"
        assert state.snapshot.old == ()
        assert finished == [], "a cancelled pass reports nothing: it stopped, it did not fail"
