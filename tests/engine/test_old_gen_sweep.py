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


def _old_gen(service: GcService) -> list[Message]:
    return [entry.message for entry in service.session_manager.store.get(SESSION).snapshot.old]


def _facts(service: GcService) -> list:
    return service.session_manager.store.get(SESSION).snapshot.active_knowledge


def _build(*, archive_threshold: float = 0.25, memory_cls=GenerationalMemory):
    bus = EventBus()
    config = GCConfig(
        context_window=700,
        keep_threshold=0.55,
        archive_threshold=archive_threshold,
        min_compactable_tokens=10,
        last_n_turns_to_keep=3,
        min_cut_tokens=60,
    )
    memory = memory_cls(
        knowledge_extractor=KnowledgeExtractor(event_bus=bus),
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


def _converse(service: GcService, turns: int, start: int = 0) -> None:
    async def scenario():
        for index in range(start, start + turns):
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
        at_80 = len(_old_gen(service))
        _converse(service, 80, start=80)
        at_160 = len(_old_gen(service))

        assert at_80 > 0, "summaries must actually be accumulating for this to mean anything"
        assert at_160 <= at_80

    def test_the_assembled_context_stops_growing(self):
        """Old gen being bounded is only interesting because the context is. A
        bounded old gen feeding an unbounded prompt would prove nothing."""
        service, _ = _build()

        _converse(service, 80)
        early = sum(m.token_count for m in asyncio.run(service.gc_collect(SESSION)))
        _converse(service, 120, start=80)
        late = sum(m.token_count for m in asyncio.run(service.gc_collect(SESSION)))

        assert late <= early * 1.2, f"context grew from {early} to {late}"

    def test_without_the_sweep_it_grows_without_bound(self):
        """The control. `archive_threshold=0` makes nothing ever cold enough to
        evict, which is exactly the behaviour this pass replaced — so if this test
        ever passes the one above, the measurement is not measuring the sweep."""
        service, memory = _build(archive_threshold=0.0)

        _converse(service, 80)
        at_80 = len(_old_gen(service))
        _converse(service, 80, start=80)
        at_160 = len(_old_gen(service))

        assert at_160 > at_80


class TestWhereTheSummariesGo:
    def test_an_evicted_summary_becomes_permanent_gen_content(self):
        """Archived, not deleted. The saving is prompt *position*: an old-gen
        summary is in every prompt, its facts only in a prompt that needs them."""
        service, memory = _build()

        _converse(service, 120)

        assert _facts(service), "evicted summaries must leave something behind"

    def test_what_an_evicted_summary_leaves_is_marked_as_its_summary(self):
        """Its text is LLM-GC's summary, not the turns' own words, and an excerpt of it
        must not be shown as if it were the message."""
        service, _ = _build()

        _converse(service, 120)

        from_summaries = [fact for fact in _facts(service) if fact.from_summary]
        assert from_summaries
        assert all(len(fact.covered_turns) >= 1 for fact in from_summaries)

    def test_eviction_is_announced_as_an_archive_from_old_gen(self):
        """Both hops are archives, but one ends a turn's first hop and the other its
        last. A transitions timeline that conflated them would show a turn archived
        twice.

        A wide archive band, so young turns decay into ARCHIVE before a cut compacts
        them and both kinds of archive occur in one conversation."""
        service, _ = _build(archive_threshold=0.45)
        archived = []
        service.event_bus.subscribe(EventType.MESSAGE_ARCHIVED, archived.append)

        _converse(service, 120)

        sources = {event.data["source_generation"] for event in archived}
        assert "old" in sources
        assert "young" in sources

class TestEvictionIsAllOrNothing:
    def test_a_failed_archive_leaves_the_summary_in_old_gen(self):
        """Archive first, remove second. Removing first would lose the summary
        outright if archiving then failed — and unlike a turn, a summary has no
        original left to fall back to."""

        class _ArchiveFailingMemory(GenerationalMemory):
            def extract_knowledge(self, message):
                # Archiving an old-gen summary extracts from what the summary says.
                if message.content == SUMMARY.text:
                    raise RuntimeError("permanent generation is unavailable")
                return super().extract_knowledge(message)

        service, memory = _build(memory_cls=_ArchiveFailingMemory)

        _converse(service, 120)

        assert _old_gen(service), "nothing may be lost when archiving fails"

    def test_a_failed_eviction_is_reported_as_a_degradation(self):
        """A pass that fell short says so. Without this the old-gen count would
        quietly stop shrinking and nothing would explain why."""

        class _ArchiveFailingMemory(GenerationalMemory):
            def extract_knowledge(self, message):
                # Archiving an old-gen summary extracts from what the summary says.
                if message.content == SUMMARY.text:
                    raise RuntimeError("permanent generation is unavailable")
                return super().extract_knowledge(message)

        service, _ = _build(memory_cls=_ArchiveFailingMemory)
        degraded = []
        service.event_bus.subscribe(EventType.AGING_DEGRADED, degraded.append)

        _converse(service, 120)

        reasons = {event.data["degradation"].reason for event in degraded}
        assert DegradationReason.ARCHIVE_FAILED in reasons

class TestScoringIsAgainstTheRealConversation:
    def test_the_coldest_summary_is_not_permanently_safe(self):
        """Summaries scored among themselves would make the oldest surviving one
        always look recent, so the coldest would never be evicted — the pass would
        appear to work while bounding nothing. This asserts the turn indices in old
        gen keep moving forward."""
        service, memory = _build()

        _converse(service, 80)
        oldest_early = min(m.turn_index for m in _old_gen(service))
        _converse(service, 120, start=80)
        oldest_late = min(m.turn_index for m in _old_gen(service))

        assert oldest_late > oldest_early


class TestArchivedEntriesAreReadAsStated:
    """Found on the live dashboard: a short turn stating a decision is too small to
    summarise, so a cut quotes it; when the quote went cold it was archived by
    parsing the quote as shown to the model, marker included. "[Quoted exactly]
    (turn 2, user) The database is postgres" yields no fact, so it was filed as a
    raw excerpt, ranked last for the front, and dropped from the prompt."""

    def test_a_quoted_decision_is_filed_as_a_fact_when_it_goes_cold(self):
        service, _ = _build()

        async def scenario():
            await service.gc_update(SESSION, Message(role="user", content="the database is postgres.",
                                                     token_count=count_tokens("the database is postgres."),
                                                     turn_index=0))
            for index in range(1, 80):
                text = (f"we walked through scenario {index} for throughput and retention at some length "
                        f"and compared it against what the pool does when the replica goes away.")
                await service.gc_update(SESSION, Message(role="user" if index % 2 == 0 else "assistant",
                                                         content=text, token_count=count_tokens(text),
                                                         turn_index=index))

        asyncio.run(scenario())
        knowledge = service.session_manager.store.get(SESSION).snapshot.knowledge

        database = [entry for entry in knowledge if entry.topic_label == "database"]
        assert database and database[0].content.startswith("postgres")
        assert database[0].message_turn == 0

    def test_nothing_filed_carries_a_front_marker(self):
        service, _ = _build()
        _converse(service, 120)

        knowledge = service.session_manager.store.get(SESSION).snapshot.knowledge
        assert knowledge
        assert not any("[Quoted exactly]" in entry.content or "[Compacted" in entry.content for entry in knowledge)
