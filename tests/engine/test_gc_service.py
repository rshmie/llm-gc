import asyncio

from llm_gc.engine.compaction import NoOpCompactor
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
    """Classifies every turn with one caller-controlled value, so a test can
    force a turn from KEEP to COMPACT across two update passes."""

    def __init__(self) -> None:
        self.classification = SweepClassification.KEEP

    def sweep(self, conversation, relevance_scores):
        entries = [
            SweepEntry(message=m, classification=self.classification, relevance_score=0.5, turn_index=m.turn_index)
            for m in conversation
        ]
        counts = {SweepClassification.KEEP: 0, SweepClassification.COMPACT: 0, SweepClassification.ARCHIVE: 0}
        return SweepResult(sweep_entries=entries, classification_counts=counts, processing_time_ms=0.0,
                           total_keep_tokens=0, total_compact_tokens=0, total_archive_tokens=0)


def _msg(turn_index: int) -> Message:
    return Message(role="user", content=f"turn {turn_index}", token_count=10, turn_index=turn_index)


def _make_service(sweeper: _StubSweeper) -> GcService:
    bus = EventBus()
    generational_memory = GenerationalMemory(
        event_bus=bus,
        knowledge_extractor=KnowledgeExtractor(event_bus=bus),
        permanent_generation=PermanentGeneration(event_bus=bus),
        compactor=NoOpCompactor(event_bus=bus),
    )
    # gc_collector is unused by gc_update, so a stub None keeps the test focused.
    return GcService(gc_collector=None, session_manager=SessionManager(),
                     generational_memory=generational_memory, relevance_scorer=_StubScorer(), sweeper=sweeper)


class TestGcUpdatePromotion:
    def test_turn_that_cools_keep_to_compact_is_promoted(self):
        sweeper = _StubSweeper()
        service = _make_service(sweeper)

        async def scenario():
            # Pass 1: turn 1 arrives and is KEEP. Nothing has cooled yet.
            sweeper.classification = SweepClassification.KEEP
            await service.gc_update("s1", _msg(1))
            assert service.generational_memory.get_old_gen() == []

            # Pass 2: the topic moves on; turn 1 is now COMPACT. It cooled → promote.
            sweeper.classification = SweepClassification.COMPACT
            await service.gc_update("s1", _msg(2))
            return service.generational_memory.get_old_gen()

        old_gen = asyncio.run(scenario())
        # Turn 1 cooled and was promoted; turn 2 is newly-seen so it is NOT a
        # promotion this pass — exactly one summary in old gen.
        assert len(old_gen) == 1

    def test_first_pass_promotes_nothing(self):
        sweeper = _StubSweeper()
        sweeper.classification = SweepClassification.COMPACT  # even COMPACT on first sight
        service = _make_service(sweeper)

        async def scenario():
            await service.gc_update("s1", _msg(1))
            return service.generational_memory.get_old_gen()

        # No prior classification exists, so nothing can be detected as "cooled".
        assert asyncio.run(scenario()) == []

    def test_stable_keep_is_not_a_promotion(self):
        sweeper = _StubSweeper()
        sweeper.classification = SweepClassification.KEEP
        service = _make_service(sweeper)

        async def scenario():
            await service.gc_update("s1", _msg(1))
            await service.gc_update("s1", _msg(2))
            return service.generational_memory.get_old_gen()

        # KEEP → KEEP is no transition, so nothing is promoted.
        assert asyncio.run(scenario()) == []

    def test_baseline_is_recorded_for_every_turn(self):
        sweeper = _StubSweeper()
        service = _make_service(sweeper)

        async def scenario():
            await service.gc_update("s1", _msg(1))
            state = service.session_manager.store.get("s1")
            return dict(state.last_classifications)

        recorded = asyncio.run(scenario())
        assert recorded == {1: SweepClassification.KEEP}
