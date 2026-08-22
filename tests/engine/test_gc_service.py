import asyncio

from llm_gc.config import GCConfig
from llm_gc.engine.compaction import BaseCompactor, NoOpCompactor
from llm_gc.engine.compaction.compaction_strategy import CompactionStrategy
from llm_gc.engine.gc_service import GcService
from llm_gc.engine.gc_status import GCStatus
from llm_gc.engine.generations.generational_memory import GenerationalMemory
from llm_gc.engine.generations.permanent_generation import PermanentGeneration
from llm_gc.engine.sweep import SweepClassification, SweepEntry, SweepResult
from llm_gc.events import Event, EventBus, EventType
from llm_gc.extraction import KnowledgeExtractor
from llm_gc.models import Message
from llm_gc.session import SessionManager


class _StubScorer:
    """The stub sweeper ignores scores, so this can return anything."""

    def score(self, message, conversation):
        return None


class _StubSweeper:
    """Classifies turns by a caller-supplied {turn_index: classification} map,
    defaulting to KEEP. Lets a test place a specific turn in COMPACT or ARCHIVE
    without depending on the real scorers."""

    def __init__(self) -> None:
        self.classifications: dict[int, SweepClassification] = {}

    def sweep(self, conversation, relevance_scores):
        entries = [
            SweepEntry(
                message=m,
                classification=self.classifications.get(m.turn_index, SweepClassification.KEEP),
                relevance_score=0.5,
                turn_index=m.turn_index,
            )
            for m in conversation
        ]
        counts = {c: sum(1 for e in entries if e.classification == c) for c in SweepClassification}
        return SweepResult(sweep_entries=entries, classification_counts=counts, processing_time_ms=0.0,
                           total_keep_tokens=0, total_compact_tokens=0, total_archive_tokens=0)


class _RaisingScorer:
    """A scorer that always fails - drives gc_update's MARK-stage failure path,
    where nothing has been aged yet and the whole pass must be reported bypassed."""

    def score(self, message, conversation):
        raise RuntimeError("scoring failed")


class _RaisingCompactor(BaseCompactor):
    """A compactor that always fails - stands in for a Phase-7 LLMCompactor whose
    API call errors, so a test can drive gc_update's failure path."""

    COMPACTION_STRATEGY = CompactionStrategy.NOOP

    def compact(self, messages):
        raise RuntimeError("compaction failed")


def _msg(turn_index: int) -> Message:
    return Message(role="user", content=f"turn {turn_index}", token_count=10, turn_index=turn_index)


# gc_threshold=0.0 means "always age": tokens_before < 0 is never true, so the
# pressure gate never fires. The aging tests below are about *what* ages, not
# about *when*; the gate gets its own class.
_ALWAYS_AGE = GCConfig(context_window=1000, gc_threshold=0.0)


def _make_service(compactor: BaseCompactor | None = None, scorer=None,
                  gc_config: GCConfig | None = None) -> tuple[GcService, _StubSweeper]:
    bus = EventBus()
    generational_memory = GenerationalMemory(
        event_bus=bus,
        knowledge_extractor=KnowledgeExtractor(event_bus=bus),
        permanent_generation=PermanentGeneration(event_bus=bus),
        compactor=compactor if compactor is not None else NoOpCompactor(event_bus=bus),
    )
    sweeper = _StubSweeper()
    service = GcService(session_manager=SessionManager(), generational_memory=generational_memory,
                        relevance_scorer=scorer if scorer is not None else _StubScorer(),
                        sweeper=sweeper, event_bus=bus,
                        gc_config=gc_config if gc_config is not None else _ALWAYS_AGE)
    return service, sweeper


def _capture_gc_finished(service: GcService) -> list[Event]:
    """Collect every GC_FINISHED the service emits, in order."""
    captured: list[Event] = []
    service.event_bus.subscribe(EventType.GC_FINISHED, captured.append)
    return captured


class TestGcUpdateActsOnClassification:
    def test_compact_turn_is_summarised_and_leaves_young(self):
        service, sweeper = _make_service()

        async def scenario():
            await service.gc_update("s1", _msg(1))              # turn 1 KEEP (default)
            sweeper.classifications = {1: SweepClassification.COMPACT}
            await service.gc_update("s1", _msg(2))              # turn 1 cools to COMPACT
            state = service.session_manager.store.get("s1")
            return [m.turn_index for m in state.messages], len(service.generational_memory.get_old_gen())

        young, old_gen_count = asyncio.run(scenario())
        assert young == [2]          # turn 1 left the desk; turn 2 stays
        assert old_gen_count == 1    # turn 1's summary is filed in old gen

    def test_archive_turn_is_extracted_and_leaves_young(self):
        service, sweeper = _make_service()

        async def scenario():
            await service.gc_update("s1", _msg(1))
            sweeper.classifications = {1: SweepClassification.ARCHIVE}
            await service.gc_update("s1", _msg(2))
            state = service.session_manager.store.get("s1")
            return [m.turn_index for m in state.messages], len(service.generational_memory.get_permanent_gen())

        young, permanent_count = asyncio.run(scenario())
        assert young == [2]              # turn 1 left the desk
        assert permanent_count >= 1      # its facts (or a raw fallback) went to permanent gen

    def test_keep_turns_all_stay_young(self):
        service, _ = _make_service()

        async def scenario():
            await service.gc_update("s1", _msg(1))
            await service.gc_update("s1", _msg(2))
            state = service.session_manager.store.get("s1")
            return [m.turn_index for m in state.messages]

        # Nothing classified COMPACT/ARCHIVE, so nothing departs.
        assert asyncio.run(scenario()) == [1, 2]

    def test_old_gen_stays_empty_while_everything_is_kept(self):
        service, _ = _make_service()

        async def scenario():
            await service.gc_update("s1", _msg(1))
            return service.generational_memory.get_old_gen()

        assert asyncio.run(scenario()) == []

    def test_produce_failure_leaves_turn_in_young_and_does_not_propagate(self):
        service, sweeper = _make_service(compactor=_RaisingCompactor(event_bus=EventBus()))

        async def scenario():
            await service.gc_update("s1", _msg(1))              # turn 1 KEEP
            sweeper.classifications = {1: SweepClassification.COMPACT}
            await service.gc_update("s1", _msg(2))              # turn 1 COMPACT -> promotion raises
            state = service.session_manager.store.get("s1")
            return [m.turn_index for m in state.messages], service.generational_memory.get_old_gen()

        # gc_update must NOT re-raise (asyncio.run would surface it), and the failed
        # turn must stay in young with nothing written - no loss, no duplicate.
        young, old_gen = asyncio.run(scenario())
        assert young == [1, 2]
        assert old_gen == []


class TestGcCollectAssembly:
    def test_assembles_old_gen_summary_then_young_turn_in_order(self):
        service, sweeper = _make_service()

        async def scenario():
            await service.gc_update("s1", _msg(1))
            sweeper.classifications = {1: SweepClassification.COMPACT}
            await service.gc_update("s1", _msg(2))  # turn 1 -> old-gen summary, turn 2 stays young
            return await service.gc_collect("s1")

        assembled = asyncio.run(scenario())
        # old-gen summary (covers turn 1) first, then the young turn 2 - chronological.
        assert [m.turn_index for m in assembled] == [1, 2]
        assert assembled[0].role == "assistant"
        assert "Compacted" in assembled[0].content
        assert assembled[1].content == "turn 2"

    def test_collect_on_empty_session_returns_empty(self):
        service, _ = _make_service()
        assert asyncio.run(service.gc_collect("s1")) == []


class TestGcUpdateReportsOnTheBus:
    """gc_update is the pass that does the real GC work, so it is the pass the
    health monitor has to be able to see. Before this, the session path emitted
    no GC_FINISHED at all and the dashboard stayed blank on a real conversation."""

    def test_update_emits_gc_finished_carrying_the_session_id(self):
        service, _ = _make_service()
        captured = _capture_gc_finished(service)

        asyncio.run(service.gc_update("s1", _msg(1)))

        assert len(captured) == 1
        assert captured[0].data["session_id"] == "s1"
        assert captured[0].data["gc_result"].status == GCStatus.COMPLETED

    def test_collect_emits_nothing(self):
        service, _ = _make_service()
        captured = _capture_gc_finished(service)

        asyncio.run(service.gc_collect("s1"))

        # collect only reads. A read reporting itself as a "GC run" would
        # overwrite the last real run's breakdown with a no-op.
        assert captured == []

    def test_final_messages_equal_what_collect_would_return(self):
        """The invariant that keeps the dashboard honest: what update *reports*
        as the resulting context is exactly what collect *sends*."""
        service, sweeper = _make_service()
        captured = _capture_gc_finished(service)

        async def scenario():
            await service.gc_update("s1", _msg(1))
            sweeper.classifications = {1: SweepClassification.COMPACT}
            await service.gc_update("s1", _msg(2))   # turn 1 -> old-gen summary
            return await service.gc_collect("s1")

        assembled = asyncio.run(scenario())
        reported = captured[-1].data["gc_result"].final_messages
        assert [m.turn_index for m in reported] == [m.turn_index for m in assembled]
        assert [m.content for m in reported] == [m.content for m in assembled]

    def test_tokens_in_final_counts_old_gen_not_just_young(self):
        """A budget meter that omitted old-gen summaries would under-report the
        real prompt size - in the reassuring direction, which is the one kind of
        wrong this project refuses to ship."""
        service, sweeper = _make_service()
        captured = _capture_gc_finished(service)

        async def scenario():
            await service.gc_update("s1", _msg(1))
            sweeper.classifications = {1: SweepClassification.COMPACT}
            await service.gc_update("s1", _msg(2))
            state = service.session_manager.store.get("s1")
            return state.messages

        young = asyncio.run(scenario())
        gc_result = captured[-1].data["gc_result"]
        young_tokens = sum(m.token_count for m in young)
        old_gen_tokens = sum(m.token_count for m in service.generational_memory.get_old_gen())
        assert old_gen_tokens > 0                                    # there is something to omit
        assert gc_result.tokens_in_final == young_tokens + old_gen_tokens

    def test_mark_failure_is_reported_bypassed_and_never_escapes(self):
        """gc_update is fire-and-forget: an exception that escapes it dies as an
        unretrieved task exception, invisible to logs and dashboard alike."""
        service, _ = _make_service(scorer=_RaisingScorer())
        captured = _capture_gc_finished(service)

        asyncio.run(service.gc_update("s1", _msg(1)))                # must not raise

        gc_result = captured[0].data["gc_result"]
        assert gc_result.status == GCStatus.BYPASSED_ON_ERROR
        assert gc_result.failure_stage == "score"
        assert "scoring failed" in gc_result.failure_reason
        # First do no harm: the new turn is still there, nothing was aged.
        assert [m.turn_index for m in gc_result.final_messages] == [1]
        assert service.generational_memory.get_old_gen() == []


class TestAgingIsPressureTriggered:
    """Aging costs something: it replaces verbatim text with a summary, and once
    the compactor is a real LLM it costs a call. Below the threshold there is no
    pressure to relieve, so the right amount of aging is none."""

    def test_below_threshold_nothing_ages_and_the_run_is_reported_bypassed(self):
        # Threshold is 500 tokens; two 10-token turns are nowhere near it.
        service, sweeper = _make_service(gc_config=GCConfig(context_window=1000, gc_threshold=0.5))
        captured = _capture_gc_finished(service)

        async def scenario():
            await service.gc_update("s1", _msg(1))
            sweeper.classifications = {1: SweepClassification.COMPACT}
            await service.gc_update("s1", _msg(2))     # would age turn 1 under pressure
            state = service.session_manager.store.get("s1")
            return [m.turn_index for m in state.messages]

        young = asyncio.run(scenario())
        assert young == [1, 2]                                   # turn 1 kept verbatim
        assert service.generational_memory.get_old_gen() == []
        assert captured[-1].data["gc_result"].status == GCStatus.BYPASSED_BELOW_THRESHOLD

    def test_crossing_the_threshold_starts_aging(self):
        # Threshold is 25 tokens; each turn is 10, so the third turn crosses it.
        service, sweeper = _make_service(gc_config=GCConfig(context_window=100, gc_threshold=0.25))
        captured = _capture_gc_finished(service)
        sweeper.classifications = {1: SweepClassification.COMPACT}

        async def scenario():
            await service.gc_update("s1", _msg(1))    # 10 tokens - under
            await service.gc_update("s1", _msg(2))    # 20 tokens - under
            await service.gc_update("s1", _msg(3))    # 30 tokens - over, ages
            state = service.session_manager.store.get("s1")
            return [m.turn_index for m in state.messages]

        young = asyncio.run(scenario())
        statuses = [e.data["gc_result"].status for e in captured]
        assert statuses[:2] == [GCStatus.BYPASSED_BELOW_THRESHOLD, GCStatus.BYPASSED_BELOW_THRESHOLD]
        assert statuses[2] == GCStatus.COMPLETED
        assert young == [2, 3]                                   # turn 1 finally aged out
        assert len(service.generational_memory.get_old_gen()) == 1

    def test_a_bypassed_update_still_records_the_new_turn(self):
        """Bypassing aging must not bypass bookkeeping - the turn is part of the
        conversation whether or not there is pressure to act on it."""
        service, _ = _make_service(gc_config=GCConfig(context_window=1000, gc_threshold=0.5))
        captured = _capture_gc_finished(service)

        asyncio.run(service.gc_update("s1", _msg(1)))

        assert [m.turn_index for m in service.session_manager.store.get("s1").messages] == [1]
        assert [m.turn_index for m in captured[0].data["gc_result"].final_messages] == [1]
