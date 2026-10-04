"""The read-only session view the dashboard is built on.

Two properties matter more than any single field. The view must describe what the
engine actually does - the same rendering as the prompt, the same marking step as
a real pass. And looking must change nothing: no session created, no idle clock
reset, nothing announced on the bus.
"""

import asyncio

from llm_gc.config import GCConfig
from llm_gc.engine.compaction import LLMCompactor
from llm_gc.engine.gc_service import GcService
from llm_gc.engine.gc_status import GCStatus
from llm_gc.engine.generations import GenerationalMemory
from llm_gc.engine.sweep import Sweeper, ThresholdSweepStrategy
from llm_gc.events import Event, EventBus, EventType
from llm_gc.extraction import KnowledgeExtractor
from llm_gc.llm import FakeProvider, LLMResponse, StopReason
from llm_gc.models import KnowledgeEntry, Message
from llm_gc.models.generation_snapshot import EntryKind, FrontEntry, GenerationSnapshot
from llm_gc.models.knowledge_entry import KnowledgeType
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


def _build(**overrides) -> GcService:
    """The real engine graph, scaled so cuts happen within a short conversation."""
    bus = EventBus()
    config = GCConfig(**{
        "context_window": 700, "keep_threshold": 0.55, "archive_threshold": 0.25,
        "min_compactable_tokens": 10, "last_n_turns_to_keep": 3, "min_cut_tokens": 60, **overrides,
    })
    return GcService(
        session_manager=SessionManager(),
        generational_memory=GenerationalMemory(
            knowledge_extractor=KnowledgeExtractor(event_bus=bus),
            compactor=LLMCompactor(event_bus=bus, provider=FakeProvider(responses=[SUMMARY] * 400)),
        ),
        relevance_scorer=RelevanceScorer(
            scorers=[RecencyScorer(decay_rate=0.15)], weights={RelevanceType.RECENCY: 1.0}, event_bus=bus
        ),
        sweeper=Sweeper(sweeper_strategy=ThresholdSweepStrategy(gc_config=config), gc_config=config, event_bus=bus),
        event_bus=bus,
        gc_config=config,
    )


def _turn(index: int) -> Message:
    text = (f"the topic{index % 8} is version {index} and we walked through the implications "
            f"for throughput and retention at some length this morning.")
    return Message(role="user" if index % 2 == 0 else "assistant", content=text,
                   token_count=count_tokens(text), turn_index=index)


def _record(service: GcService, index: int) -> None:
    service.gc_record(SESSION, [_turn(index)], [f"h{index}"])


def _converse(service: GcService, turns: int) -> list[Event]:
    """Record and age turn by turn, as the proxy does; returns every GC_FINISHED."""
    finished: list[Event] = []
    service.event_bus.subscribe(EventType.GC_FINISHED, finished.append)

    async def scenario() -> None:
        for index in range(turns):
            _record(service, index)
            await service.gc_age(SESSION)

    asyncio.run(scenario())
    return finished


def _snapshot(service: GcService) -> GenerationSnapshot:
    return service.session_manager.store.get(SESSION).snapshot


class TestCoverage:
    def test_a_summary_records_every_turn_of_its_run(self):
        service = _build()
        _converse(service, 12)

        summaries = [entry for entry in _snapshot(service).old if entry.kind is EntryKind.SUMMARY]
        assert summaries, "the conversation is long enough to have been cut"
        for entry in summaries:
            assert entry.message.turn_index == entry.source_turns[0]
            assert list(entry.source_turns) == list(range(entry.source_turns[0], entry.source_turns[-1] + 1))

    def test_every_recorded_turn_is_somewhere(self):
        """The dashboard reports a turn that is neither verbatim nor in the front as
        reduced to facts. That is only true if every departure leaves a trace:
        archiving always files at least one entry, and an archived summary passes
        its coverage on to its facts."""
        service = _build()
        _converse(service, 80)
        snapshot = _snapshot(service)

        accounted = {message.turn_index for message in snapshot.young}
        accounted |= {turn for entry in snapshot.old for turn in entry.source_turns}
        accounted |= {turn for entry in snapshot.knowledge for turn in entry.covered_turns}

        assert accounted == set(range(snapshot.recorded_turn_count))
        assert any(len(entry.source_turns) > 1 for entry in snapshot.knowledge), \
            "over 80 turns some summary has gone cold and been archived"

    def test_cuts_are_counted_when_committed_not_when_attempted(self):
        service = _build()
        finished = _converse(service, 30)

        committed = sum(1 for event in finished if event.data["gc_result"].status is GCStatus.COMPLETED
                        and event.data["gc_result"].tokens_saved > 0)
        assert _snapshot(service).cut_count == committed > 0
        assert len(finished) > committed, "most passes find no batch worth a cut"


class TestLookingChangesNothing:
    def test_an_unknown_session_is_reported_absent_and_not_created(self):
        service = _build()
        assert service.inspect("nobody") is None
        assert service.session_manager.store.get("nobody") is None

    def test_the_idle_clock_is_not_reset(self):
        service = _build()
        _record(service, 0)
        before = service.session_manager.store.get(SESSION).last_updated_at

        service.inspect(SESSION)

        assert service.session_manager.store.get(SESSION).last_updated_at == before

    def test_nothing_is_announced(self):
        """A preview reported as a sweep would have every subscriber record work
        that never happened."""
        service = _build()
        _converse(service, 10)
        heard: list[Event] = []
        for event_type in EventType:
            service.event_bus.subscribe(event_type, heard.append)

        service.inspect(SESSION)

        assert heard == []

    def test_a_real_pass_still_announces_after_a_preview(self):
        """The preview silences a copy. Silencing the shared scorer instead would
        pass the test above and quietly blind every subscriber from then on."""
        service = _build()
        _record(service, 0)
        service.inspect(SESSION)
        heard: list[Event] = []
        service.event_bus.subscribe(EventType.SWEEP_COMPLETED, heard.append)

        asyncio.run(service.gc_age(SESSION))

        assert len(heard) == 1


class TestTheViewIsTheEngine:
    def test_the_context_is_what_collect_returns(self):
        service = _build()
        _converse(service, 20)

        inspection = service.inspect(SESSION)

        assert inspection.context == asyncio.run(service.gc_collect(SESSION))
        assert inspection.context_tokens == sum(m.token_count for m in inspection.context)

    def test_the_preview_predicts_the_next_pass(self):
        """The preview is the pass's own marking step, so its verdict must match
        the pass that follows - turn after turn, across cuts."""
        service = _build()
        finished: list[Event] = []
        service.event_bus.subscribe(EventType.GC_FINISHED, finished.append)
        cuts_checked = 0

        async def scenario() -> None:
            nonlocal cuts_checked
            for index in range(40):
                _record(service, index)
                preview = service.inspect(SESSION).next_cut
                young_before = {m.turn_index for m in _snapshot(service).young}
                await service.gc_age(SESSION)
                result = finished[-1].data["gc_result"]
                if not preview.batch_ready:
                    assert result.status is GCStatus.BYPASSED_BATCH_TOO_SMALL, index
                elif not result.degradations and result.tokens_saved > 0:
                    departed = young_before - {m.turn_index for m in _snapshot(service).young}
                    assert departed == {e.turn_index for e in preview.entries[:preview.cut]}, index
                    cuts_checked += 1

        asyncio.run(scenario())
        assert cuts_checked > 0

    def test_the_front_reports_exactly_the_facts_it_carries(self):
        """Facts are chosen by importance and then cut to the memory budget. The
        dashboard marks each fact as sent or stored-only, so the facts the front
        reports must be precisely the ones whose text the model receives."""
        service = _build(context_window=2000, max_memory_ratio=0.03)   # room for a few facts
        facts = tuple(
            KnowledgeEntry(message_turn=i, content=f"value{i}x{i}", topic_label=f"subject{i}",
                           knowledge_type=KnowledgeType.DECISION)
            for i in range(12)
        )
        quote = FrontEntry(message=Message(role="user", content="[Quoted exactly] hello", token_count=5,
                                           turn_index=0), kind=EntryKind.QUOTE, source_turns=(0,), text="hello")
        service.session_manager.get_or_create(SESSION).snapshot = GenerationSnapshot(old=(quote,), knowledge=facts)

        front = service.inspect(SESSION).front

        assert 0 < len(front.facts) < len(facts)
        for fact in facts:
            assert (fact.content in front.message.content) == (fact in front.facts), fact.content
