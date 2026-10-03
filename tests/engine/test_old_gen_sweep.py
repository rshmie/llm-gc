"""A turn's journey has to end somewhere other than old gen.

`_old_gen` was append-only, so the assembled context grew by one summary per aged
run forever. Old gen was the leak that young gen had been fixed to avoid: measured
over 200 turns it reached 72 summaries and 2226 tokens against a 700-token window,
and the compression ratio got *worse* as the conversation went on.
"""

import asyncio

from llm_gc.config import GCConfig
from llm_gc.engine import DegradationReason
from llm_gc.engine.compaction import LLMCompactor
from llm_gc.engine.gc_service import GcService
from llm_gc.engine.generations import GenerationalMemory
from llm_gc.engine.generations.permanent_generation import PermanentGeneration
from llm_gc.engine.sweep import Sweeper, ThresholdSweepStrategy
from llm_gc.events import EventBus, EventType
from llm_gc.extraction import KnowledgeExtractor
from llm_gc.llm import FakeProvider, LLMResponse, StopReason
from llm_gc.models import Message
from llm_gc.scoring import RecencyScorer, RelevanceScorer, RelevanceType
from llm_gc.session import SessionManager
from llm_gc.utils import count_tokens

SESSION = "s"

SUMMARY = LLMResponse(
    text="Earlier turns covered the storage decision and the retention window in brief.",
    stop_reason=StopReason.COMPLETE,
    input_tokens=0,
    output_tokens=14,
    model="scripted",
    provider="offline",
)


def _build(*, archive_threshold: float = 0.25, memory_cls=GenerationalMemory):
    bus = EventBus()
    config = GCConfig(
        context_window=700,
        gc_threshold=0.5,
        keep_threshold=0.55,
        archive_threshold=archive_threshold,
        min_compactable_tokens=10,
        last_n_turns_to_keep=3,
    )
    memory = memory_cls(
        event_bus=bus,
        knowledge_extractor=KnowledgeExtractor(event_bus=bus),
        permanent_generation=PermanentGeneration(event_bus=bus),
        compactor=LLMCompactor(event_bus=bus, provider=FakeProvider(responses=[SUMMARY] * 400)),
    )
    service = GcService(
        session_manager=SessionManager(),
        generational_memory=memory,
        relevance_scorer=RelevanceScorer(
            scorers=[RecencyScorer(decay_rate=0.15)], weights={RelevanceType.RECENCY: 1.0}, event_bus=bus
        ),
        sweeper=Sweeper(
            sweeper_strategy=ThresholdSweepStrategy(gc_config=config), gc_config=config, event_bus=bus
        ),
        event_bus=bus,
        gc_config=config,
    )
    return service, memory


def _converse(service: GcService, turns: int) -> None:
    async def scenario():
        for index in range(turns):
            text = (
                f"the topic{index % 8} is version {index} and we walked through the implications "
                f"for throughput and retention at some length this morning."
            )
            await service.gc_update(
                SESSION,
                Message(
                    role="user" if index % 2 == 0 else "assistant",
                    content=text,
                    token_count=count_tokens(text),
                    turn_index=index,
                ),
            )

    asyncio.run(scenario())


class TestOldGenIsBounded:
    def test_it_stops_growing(self):
        """The measurement this whole pass exists for. Not "grows slower" —
        *constant*, regardless of how long the conversation runs."""
        service, memory = _build()

        _converse(service, 80)
        at_80 = len(memory.get_old_gen())
        _converse(service, 80)
        at_160 = len(memory.get_old_gen())

        assert at_80 > 0, "summaries must actually be accumulating for this to mean anything"
        assert at_160 <= at_80

    def test_the_assembled_context_stops_growing(self):
        """Old gen being bounded is only interesting because the context is. A
        bounded old gen feeding an unbounded prompt would prove nothing."""
        service, _ = _build()

        _converse(service, 80)
        early = sum(m.token_count for m in asyncio.run(service.gc_collect(SESSION)))
        _converse(service, 120)
        late = sum(m.token_count for m in asyncio.run(service.gc_collect(SESSION)))

        assert late <= early * 1.2, f"context grew from {early} to {late}"

    def test_without_the_sweep_it_grows_without_bound(self):
        """The control. `archive_threshold=0` makes nothing ever cold enough to
        evict, which is exactly the behaviour this pass replaced — so if this test
        ever passes the one above, the measurement is not measuring the sweep."""
        service, memory = _build(archive_threshold=0.0)

        _converse(service, 80)
        at_80 = len(memory.get_old_gen())
        _converse(service, 80)
        at_160 = len(memory.get_old_gen())

        assert at_160 > at_80


class TestWhereTheSummariesGo:
    def test_an_evicted_summary_becomes_permanent_gen_content(self):
        """Archived, not deleted. The saving is prompt *position*: an old-gen
        summary is in every prompt, its facts only in a prompt that needs them."""
        service, memory = _build()

        _converse(service, 120)

        assert memory.get_permanent_gen(), "evicted summaries must leave something behind"

    def test_eviction_is_announced_as_an_archive_from_old_gen(self):
        """Both hops are archives, but one ends a turn's first hop and the other its
        last. A transitions timeline that conflated them would show a turn archived
        twice."""
        service, _ = _build()
        archived = []
        service.event_bus.subscribe(EventType.MESSAGE_ARCHIVED, archived.append)

        _converse(service, 120)

        sources = {event.data["source_generation"] for event in archived}
        assert "old" in sources
        assert "young" in sources

    def test_a_young_turn_archive_defaults_to_young(self):
        """Asserted on the call rather than inferred from a conversation: by turn 40
        old-gen eviction is already running, so a whole-run assertion that every
        archive is young is simply false. The default is what matters — the
        stateless composer path archives without passing the argument."""
        _, memory = _build()
        archived = []
        memory.event_bus.subscribe(EventType.MESSAGE_ARCHIVED, archived.append)

        memory.archive_message(
            Message(role="user", content="the database is postgres", token_count=6, turn_index=1)
        )

        assert [event.data["source_generation"] for event in archived] == ["young"]


class TestEvictionIsAllOrNothing:
    def test_a_failed_archive_leaves_the_summary_in_old_gen(self):
        """Archive first, remove second. Removing first would lose the summary
        outright if archiving then failed — and unlike a turn, a summary has no
        original left to fall back to."""

        class _ArchiveFailingMemory(GenerationalMemory):
            def archive_message(self, message, source_generation="young"):
                if source_generation == "old":
                    raise RuntimeError("permanent generation is unavailable")
                super().archive_message(message, source_generation)

        service, memory = _build(memory_cls=_ArchiveFailingMemory)

        _converse(service, 120)

        assert memory.get_old_gen(), "nothing may be lost when archiving fails"

    def test_a_failed_eviction_is_reported_as_a_degradation(self):
        """A pass that fell short says so. Without this the old-gen count would
        quietly stop shrinking and nothing would explain why."""

        class _ArchiveFailingMemory(GenerationalMemory):
            def archive_message(self, message, source_generation="young"):
                if source_generation == "old":
                    raise RuntimeError("permanent generation is unavailable")
                super().archive_message(message, source_generation)

        service, _ = _build(memory_cls=_ArchiveFailingMemory)
        degraded = []
        service.event_bus.subscribe(EventType.AGING_DEGRADED, degraded.append)

        _converse(service, 120)

        reasons = {event.data["degradation"].reason for event in degraded}
        assert DegradationReason.ARCHIVE_FAILED in reasons

    def test_evicting_a_stale_summary_crashes_loudly(self):
        """A caller working from a copy that no longer reflects old gen is a
        programmer error, so it raises rather than silently doing nothing."""
        _, memory = _build()
        ghost = Message(role="assistant", content="never filed", token_count=3, turn_index=99)

        try:
            memory.evict_from_old_gen(ghost)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for a summary not in old gen")


class TestScoringIsAgainstTheRealConversation:
    def test_the_coldest_summary_is_not_permanently_safe(self):
        """Summaries scored among themselves would make the oldest surviving one
        always look recent, so the coldest would never be evicted — the pass would
        appear to work while bounding nothing. This asserts the turn indices in old
        gen keep moving forward."""
        service, memory = _build()

        _converse(service, 80)
        oldest_early = min(m.turn_index for m in memory.get_old_gen())
        _converse(service, 120)
        oldest_late = min(m.turn_index for m in memory.get_old_gen())

        assert oldest_late > oldest_early
