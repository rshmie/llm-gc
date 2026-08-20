import asyncio

from llm_gc.engine.compaction import BaseCompactor, NoOpCompactor
from llm_gc.engine.compaction.compaction_strategy import CompactionStrategy
from llm_gc.engine.gc_service import GcService
from llm_gc.engine.generations.generational_memory import GenerationalMemory
from llm_gc.engine.generations.permanent_generation import PermanentGeneration
from llm_gc.engine.sweep import SweepClassification, SweepEntry, SweepResult
from llm_gc.events import EventBus
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
        counts = {SweepClassification.KEEP: 0, SweepClassification.COMPACT: 0, SweepClassification.ARCHIVE: 0}
        return SweepResult(sweep_entries=entries, classification_counts=counts, processing_time_ms=0.0,
                           total_keep_tokens=0, total_compact_tokens=0, total_archive_tokens=0)


class _RaisingCompactor(BaseCompactor):
    """A compactor that always fails - stands in for a Phase-7 LLMCompactor whose
    API call errors, so a test can drive gc_update's failure path."""

    COMPACTION_STRATEGY = CompactionStrategy.NOOP

    def compact(self, messages):
        raise RuntimeError("compaction failed")


def _msg(turn_index: int) -> Message:
    return Message(role="user", content=f"turn {turn_index}", token_count=10, turn_index=turn_index)


def _make_service(compactor: BaseCompactor | None = None) -> tuple[GcService, _StubSweeper]:
    bus = EventBus()
    generational_memory = GenerationalMemory(
        event_bus=bus,
        knowledge_extractor=KnowledgeExtractor(event_bus=bus),
        permanent_generation=PermanentGeneration(event_bus=bus),
        compactor=compactor if compactor is not None else NoOpCompactor(event_bus=bus),
    )
    sweeper = _StubSweeper()
    service = GcService(session_manager=SessionManager(), generational_memory=generational_memory,
                        relevance_scorer=_StubScorer(), sweeper=sweeper)
    return service, sweeper


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
