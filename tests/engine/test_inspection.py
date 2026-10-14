"""The read-only session view the dashboard is built on.

Two properties matter more than any single field. The view must describe what the
engine actually does - the same rendering as the prompt, the same marking step as
a real pass. And looking must change nothing: no session created, no idle clock
reset, nothing announced on the bus.
"""

import asyncio

from llm_gc.config import GCConfig
from llm_gc.engine import build_gc_service
from llm_gc.engine.gc_service import GcService
from llm_gc.engine.gc_status import GCStatus
from llm_gc.events import Event, EventType
from llm_gc.extraction import PatternFactReader
from llm_gc.llm import FakeProvider, LLMResponse, StopReason
from llm_gc.models import KnowledgeEntry, Message
from llm_gc.models.generation_snapshot import DetailLevel, GenerationSnapshot, Span
from llm_gc.models.knowledge_entry import KnowledgeType
from llm_gc.utils import count_tokens

from tests.engine.support import collect

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
    """The real engine graph, scaled so collections happen within a short conversation."""
    config = GCConfig(**{"context_window": 700, "keep_threshold": 0.55, "archive_threshold": 0.25,
                         "last_n_turns_to_keep": 3, "min_cut_tokens": 60, **overrides})
    return build_gc_service(config, summary_provider=FakeProvider(responses=[SUMMARY] * 400),
                            fact_reader=PatternFactReader())


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
            await collect(service, SESSION)

    asyncio.run(scenario())
    return finished


def _snapshot(service: GcService) -> GenerationSnapshot:
    return service.session_manager.store.get(SESSION).snapshot


class TestCoverage:
    def test_a_summary_records_every_turn_of_its_run(self):
        service = _build()
        _converse(service, 30)

        summaries = [span for span in _snapshot(service).old if span.summary is not None]
        assert summaries, "the conversation is long enough to have been cut"
        for span in summaries:
            assert span.summary is not None and span.summary.turn_index == span.first_turn

    def test_every_recorded_turn_is_kept_exactly_once(self):
        """Nothing recorded is destroyed: each turn is in the tail or in exactly one
        span, and the span holds the turn as it was recorded - archived ones too."""
        service = _build()
        _converse(service, 80)
        snapshot = _snapshot(service)

        held = list(snapshot.young) + [turn for span in snapshot.old for turn in span.turns]
        assert sorted(turn.turn_index for turn in held) == list(range(80))
        assert all(turn == _turn(turn.turn_index) for turn in held)
        condensed = (DetailLevel.ARCHIVED, DetailLevel.OUT_OF_VIEW)
        assert any(span.level in condensed and len(span.turns) > 1 for span in snapshot.old), \
            "over 80 turns on a small budget some span has been archived or folded, and still holds every turn"

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
        """A preview reported as a collection would have every subscriber record work
        that never happened."""
        service = _build()
        _converse(service, 10)
        heard: list[Event] = []
        for event_type in EventType:
            service.event_bus.subscribe(event_type, heard.append)

        service.inspect(SESSION)

        assert heard == []

    def test_a_real_collection_still_announces_after_previews(self):
        """Previews are silent because planning announces nothing, not because a
        component was silenced - which would blind every subscriber from then on."""
        service = _build()
        heard: list[Event] = []
        service.event_bus.subscribe(EventType.KNOWLEDGE_ENTRY_ADDED, heard.append)

        async def scenario() -> None:
            for index in range(20):
                _record(service, index)
                service.inspect(SESSION)
                await collect(service, SESSION)

        asyncio.run(scenario())
        assert heard, "collections after previews still file and announce facts"


class TestTheViewIsTheEngine:
    def test_the_context_is_what_collect_returns(self):
        service = _build()
        _converse(service, 20)

        inspection = service.inspect(SESSION)

        assert inspection.context == asyncio.run(service.gc_collect(SESSION))
        assert inspection.context_tokens == sum(m.token_count for m in inspection.context)

    def test_the_preview_predicts_the_next_pass(self):
        """The preview is the collection's own planning step, so its verdict must
        match the pass that follows - turn after turn, across collections."""
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
                await collect(service, SESSION)
                result = finished[-1].data["gc_result"]
                if not preview.batch_ready:
                    assert result.status is GCStatus.BYPASSED_BATCH_TOO_SMALL, index
                elif not result.degradations and result.tokens_saved > 0:
                    departed = young_before - {m.turn_index for m in _snapshot(service).young}
                    assert departed == set(preview.turns), index
                    cuts_checked += 1

        asyncio.run(scenario())
        assert cuts_checked > 0

    def test_the_estimated_saving_is_the_collections_own_first_check(self):
        """A collection checks its saving at estimated summary sizes before paying for
        any summary, and skips a cut that would not pay for itself. The preview reports
        that same estimate, so a cut it shows as not paying is one the collection skips,
        by exactly the margin shown."""
        service = _build(min_cut_tokens=0)
        finished: list[Event] = []
        service.event_bus.subscribe(EventType.GC_FINISHED, finished.append)
        checked = []

        async def scenario() -> None:
            for index in range(30):
                _record(service, index)
                preview = service.inspect(SESSION).next_cut
                await collect(service, SESSION)
                if preview.batch_ready:
                    checked.append((preview.estimated_saving, finished[-1].data["gc_result"]))

        asyncio.run(scenario())
        assert any(saving <= 0 for saving, _ in checked), "a cut that does not pay was previewed"
        assert any(result.status is GCStatus.COMPLETED for _, result in checked), "and one that does"
        for saving, result in checked:
            if saving <= 0:
                assert result.status is GCStatus.BYPASSED_NO_SAVING
                assert result.abandoned_cut.context_tokens_after == result.tokens_before - saving

    def test_the_front_reports_exactly_the_facts_it_carries(self):
        """Facts reach the model on their archived span's line; a kept or compacted
        span's words carry its facts already. The dashboard marks each fact as sent
        or stored-only, so the facts the front reports must be precisely the ones
        whose text the model receives."""
        service = _build()
        facts = tuple(KnowledgeEntry(message_turn=i, content=f"value{i}x{i}", topic_label=f"subject{i}",
                                     knowledge_type=KnowledgeType.DECISION) for i in range(4))
        archived = Span(turns=(Message(role="user", content="first", token_count=1, turn_index=0),
                               Message(role="assistant", content="second", token_count=1, turn_index=1)),
                        level=DetailLevel.ARCHIVED, index_line="first")
        kept = Span(turns=(Message(role="user", content="third", token_count=1, turn_index=2),
                           Message(role="assistant", content="fourth", token_count=1, turn_index=3)),
                    level=DetailLevel.KEPT)
        service.session_manager.get_or_create(SESSION).snapshot = GenerationSnapshot(old=(archived, kept),
                                                                                    knowledge=facts)

        front = service.inspect(SESSION).front

        assert front.facts == facts[:2]
        for fact in facts:
            assert (fact.content in front.message.content) == (fact in front.facts), fact.content
