"""A collection that fell short has to say so.

These pin the project's own premise applied to its own internals: a span made
without its summary, or without its facts, is still made - nothing is lost - but
a consumer that could not see why would describe a front that is larger, or
thinner, than it claims.
"""

import asyncio

import pytest

from llm_gc.config import GCConfig
from llm_gc.engine import DegradationReason, GCStatus
from llm_gc.engine.compaction import BaseCompactor, CompactionStrategy, LLMCompactor
from llm_gc.events import EventBus, EventType
from llm_gc.exceptions import LLMTimeoutError
from llm_gc.llm import FakeProvider, LLMResponse, StopReason
from llm_gc.models.generation_snapshot import DetailLevel
from llm_gc.monitoring import ContextHealthMonitor
from tests.engine.support import ARCHIVED_SCORE, COMPACTED_SCORE, make_service, turn

SESSION = "s"
# Turn 0 earns a summary, turn 1 its index line, turn 2 stays in the tail: the
# archived span keeps the collection worth making whatever happens to the summary.
SCORES = {0: COMPACTED_SCORE, 1: ARCHIVED_SCORE}

_TRUNCATED = LLMResponse(text="tail-less", stop_reason=StopReason.TRUNCATED, input_tokens=50, output_tokens=2,
                         model="scripted", provider="offline")


class _NonLLMGCFailingCompactor(BaseCompactor):
    """Raises something that is not one of ours, carrying conversation text.

    A third-party compactor could do exactly this, which is why the recorded
    `detail` must not come from an arbitrary exception's message.
    """

    COMPACTION_STRATEGY = CompactionStrategy.NOOP

    async def compact(self, messages):
        raise RuntimeError(f"failed while summarising: {messages[0].content}")


def _summariser(response: LLMResponse | Exception) -> LLMCompactor:
    return LLMCompactor(event_bus=EventBus(), provider=FakeProvider(responses=[response] * 20))


def _run(service, turns: int = 3) -> list:
    """Record turns one by one, a pass after each; every GCResult, in order."""
    results = []
    service.event_bus.subscribe(EventType.GC_FINISHED, lambda event: results.append(event.data["report"]))

    async def scenario():
        for index in range(turns):
            await service.gc_update(SESSION, turn(index))

    asyncio.run(scenario())
    return results


def _degradations(results) -> list:
    return [degradation for result in results for degradation in result.degradations]


class TestACleanPassReportsNothing:
    def test_a_successful_collection_has_no_degradations(self):
        service, _, _ = make_service(scores=SCORES)
        results = _run(service)

        assert results[-1].status is GCStatus.COMPLETED
        assert _degradations(results) == []

    def test_a_bypassed_pass_has_no_degradations(self):
        service, _, _ = make_service(GCConfig(min_cut_tokens=10_000), scores=SCORES)
        assert _degradations(_run(service)) == []


class TestASpanMadeWithoutItsSummaryIsRecorded:
    def test_the_collection_still_completes_and_says_what_fell_short(self):
        """Status answers "did the pass run"; degradation answers "did everything it
        intended happen". A fourth status would make every `== COMPLETED` check
        silently stop matching."""
        service, _, _ = make_service(scores=SCORES, compactor=_summariser(_TRUNCATED))
        results = _run(service)

        assert results[-1].status is GCStatus.COMPLETED
        assert (DegradationReason.COMPACTION_REFUSED, [0]) in [(d.reason, d.turn_indices)
                                                                for d in results[-1].degradations]

    def test_the_span_is_kept_whole_rather_than_lost(self):
        """First do no harm: the front is correct, only larger than the plan wanted."""
        service, _, _ = make_service(scores=SCORES, compactor=_summariser(_TRUNCATED))
        _run(service)

        span = service.session_manager.store.get(SESSION).snapshot.old[0]
        assert span.level is DetailLevel.KEPT and span.turns == (turn(0),)

    def test_a_refusal_and_a_provider_failure_are_told_apart(self):
        """Different next action: a refusal means the run or the cap was wrong, a
        failure means check the provider."""
        refused, _, _ = make_service(scores=SCORES, compactor=_summariser(_TRUNCATED))
        failed, _, _ = make_service(scores=SCORES, compactor=_summariser(LLMTimeoutError("no answer in time")))

        assert DegradationReason.COMPACTION_REFUSED in {d.reason for d in _degradations(_run(refused))}
        assert DegradationReason.COMPACTION_FAILED in {d.reason for d in _degradations(_run(failed))}


class TestASpanMadeWithoutItsFactsIsRecorded:
    def test_an_extraction_failure_is_recorded_and_the_span_is_still_made(self):
        class _FailingReader:
            async def read(self, spans, on_record):
                raise RuntimeError("extraction unavailable")

        service, _, _ = make_service(scores=SCORES, fact_reader=_FailingReader())  # type: ignore[arg-type]
        results = _run(service)

        assert DegradationReason.ARCHIVE_FAILED in {d.reason for d in _degradations(results)}
        assert [span.first_turn for span in service.session_manager.store.get(SESSION).snapshot.old] == [0, 1]


class TestAFrontThatCannotFitIsRecorded:
    def test_folding_everything_and_still_not_fitting_is_reported(self):
        tiny = GCConfig(context_window=200, max_recap_ratio=0.3, min_cut_tokens=0, last_n_turns_to_keep=1)
        service, _, _ = make_service(tiny, scores=SCORES)

        results = _run(service)

        assert DegradationReason.RECAP_OVER_BUDGET in {d.reason for d in _degradations(results)}


class TestThePayloadCarriesNoConversationContent:
    def test_an_unknown_exceptions_message_is_not_recorded(self):
        """Event payloads reach the dashboard over HTTP. Our own errors are content
        free by convention; a third-party compactor's are not a boundary we control."""
        service, _, _ = make_service(scores=SCORES, compactor=_NonLLMGCFailingCompactor(event_bus=EventBus()))
        degradation = _degradations(_run(service))[0]

        assert degradation.detail == "RuntimeError"
        assert degradation.error_code is None

    def test_our_own_error_message_is_recorded_because_it_is_content_free(self):
        service, _, _ = make_service(scores=SCORES, compactor=_summariser(LLMTimeoutError("no answer in time")))
        degradation = _degradations(_run(service))[0]

        assert "no answer in time" in degradation.detail
        assert degradation.error_code is not None


class TestTheEventIsEmittedAsItHappens:
    def test_one_event_per_degradation_carrying_the_session(self):
        service, _, _ = make_service(scores=SCORES, compactor=_summariser(_TRUNCATED))
        seen = []
        service.event_bus.subscribe(EventType.AGING_DEGRADED, seen.append)

        results = _run(service)

        # Counted on the passes that met them: a collection switched in repeats its preparation's.
        passes = [result for result in results if result.status is not GCStatus.COMPLETED]
        assert len(seen) == len(_degradations(passes)) > 0
        assert {event.data["session_id"] for event in seen} == {SESSION}

    def test_a_clean_collection_emits_nothing(self):
        service, _, _ = make_service(scores=SCORES)
        seen = []
        service.event_bus.subscribe(EventType.AGING_DEGRADED, seen.append)

        _run(service)

        assert seen == []


class TestTheDashboardCanSeeIt:
    @staticmethod
    def _breakdown(service):
        monitor = ContextHealthMonitor(event_bus=service.event_bus, gc_config=service.gc_config,
                                       read_generations=lambda _session_id: service.session_manager.store.get(
                                           SESSION).snapshot)
        _run(service)
        return monitor.get_snapshot(SESSION).gc_breakdown

    def test_the_health_snapshot_reports_the_degradation(self):
        breakdown = self._breakdown(make_service(scores=SCORES, compactor=_summariser(_TRUNCATED))[0])
        assert breakdown is not None and breakdown.is_degraded is True

    def test_a_clean_snapshot_is_not_degraded(self):
        breakdown = self._breakdown(make_service(scores=SCORES)[0])
        assert breakdown is not None and breakdown.is_degraded is False


class TestTheRecordIsImmutable:
    def test_a_degradation_cannot_be_edited_after_the_fact(self):
        service, _, _ = make_service(scores=SCORES, compactor=_summariser(_TRUNCATED))
        degradation = _degradations(_run(service))[0]

        with pytest.raises(Exception):
            degradation.detail = "something else"
