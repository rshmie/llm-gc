"""A completed pass that fell short has to say so.

These pin the project's own premise applied to its own internals. `GCResult`'s
counts are the *sweeper's decision*: a pass whose compactions were all refused
reports a non-zero `compacted_count` and files nothing. Without the records tested
here, a consumer would render compactions that never happened — a confident-looking
falsehood about what the model is being sent, which is the worst failure this
system can produce.
"""

import asyncio

import pytest

from llm_gc.config import GCConfig
from llm_gc.engine import DegradationReason, GCStatus
from llm_gc.engine.compaction import BaseCompactor, CompactionStrategy, LLMCompactor, NoOpCompactor
from llm_gc.engine.gc_service import GcService
from llm_gc.engine.generations import GenerationalMemory
from llm_gc.engine.generations.permanent_generation import PermanentGeneration
from llm_gc.engine.sweep import Sweeper, ThresholdSweepStrategy
from llm_gc.events import EventBus, EventType
from llm_gc.exceptions import LLMTimeoutError
from llm_gc.extraction import KnowledgeExtractor
from llm_gc.llm import FakeProvider, LLMResponse, StopReason
from llm_gc.models import Message
from llm_gc.monitoring import ContextHealthMonitor
from llm_gc.scoring import RecencyScorer, RelevanceScorer, RelevanceType
from llm_gc.session import SessionManager
from llm_gc.utils import count_tokens

SESSION = "s"
TURN_COUNT = 10

# Aggressive enough that GC engages after a few turns and keeps engaging, so the
# aging path runs repeatedly rather than once.
CONFIG = GCConfig(
    context_window=200,
    gc_threshold=0.3,
    keep_threshold=0.6,
    archive_threshold=0.05,
    min_compactable_tokens=5,
    last_n_turns_to_keep=2,
)

# Under CONFIG, cooling turns are reclassified COMPACT and leave the young
# generation before they ever decay into ARCHIVE range, so the archive path is
# unreachable. Raising archive_threshold sends mid-score turns straight to ARCHIVE
# instead. Worth stating: the first draft of the archive test below failed for
# exactly this reason, and the config was the bug, not the engine.
ARCHIVING_CONFIG = CONFIG.model_copy(update={"keep_threshold": 0.9, "archive_threshold": 0.5})


def _response(text: str, *, stop_reason: StopReason = StopReason.COMPLETE) -> LLMResponse:
    return LLMResponse(
        text=text,
        stop_reason=stop_reason,
        input_tokens=50,
        output_tokens=count_tokens(text) or 1,
        model="scripted",
        provider="offline",
    )


class _ArchiveFailingMemory(GenerationalMemory):
    """Archiving raises. Stands in for extraction or storage breaking."""

    def archive_message(self, message: Message) -> None:
        raise RuntimeError("permanent generation is unavailable")


class _NonLLMGCFailingCompactor(BaseCompactor):
    """Raises something that is not one of ours, carrying conversation text.

    A third-party compactor could do exactly this, which is why the recorded
    `detail` must not come from an arbitrary exception's message.
    """

    COMPACTION_STRATEGY = CompactionStrategy.NOOP

    async def compact(self, messages):
        raise RuntimeError(f"failed while summarising: {messages[0].content}")


def _build(compactor: BaseCompactor | None = None, memory_cls=GenerationalMemory, config: GCConfig = CONFIG):
    bus = EventBus()
    memory = memory_cls(
        event_bus=bus,
        knowledge_extractor=KnowledgeExtractor(event_bus=bus),
        permanent_generation=PermanentGeneration(event_bus=bus),
        compactor=compactor if compactor is not None else NoOpCompactor(event_bus=bus),
    )
    service = GcService(
        session_manager=SessionManager(),
        generational_memory=memory,
        relevance_scorer=RelevanceScorer(
            scorers=[RecencyScorer(decay_rate=0.4)], weights={RelevanceType.RECENCY: 1.0}, event_bus=bus
        ),
        sweeper=Sweeper(
            sweeper_strategy=ThresholdSweepStrategy(gc_config=config), gc_config=config, event_bus=bus
        ),
        event_bus=bus,
        gc_config=config,
    )
    return service, memory, bus


def _run(service, turns: int = TURN_COUNT) -> list:
    """Feed a conversation and return every GCResult the pass emitted."""
    results = []

    def collect(event):
        results.append(event.data["gc_result"])

    service.event_bus.subscribe(EventType.GC_FINISHED, collect)

    async def scenario():
        for index in range(turns):
            text = f"Turn {index}: we discussed the storage layer and the retention window at length today."
            await service.gc_update(
                SESSION, Message(role="user", content=text, token_count=count_tokens(text), turn_index=index)
            )

    asyncio.run(scenario())
    return results


def _llm_compactor(bus: EventBus, response: LLMResponse | Exception):
    # One script entry per possible call; FakeProvider raises when exhausted, so a
    # generous count is what keeps an over-calling bug visible rather than hidden.
    return LLMCompactor(event_bus=bus, provider=FakeProvider(responses=[response] * (TURN_COUNT * 2)))


class TestACleanPassReportsNothing:
    def test_a_successful_pass_has_no_degradations(self):
        """The common case, pinned so the field cannot become noise that consumers
        learn to ignore."""
        service, _, _ = _build()
        results = _run(service)

        completed = [r for r in results if r.status is GCStatus.COMPLETED]
        assert completed, "the conversation must cross the threshold for this test to mean anything"
        assert all(r.degradations == [] for r in completed)

    def test_a_bypassed_pass_has_no_degradations(self):
        service, _, _ = _build()
        results = _run(service, turns=2)

        assert all(r.status is GCStatus.BYPASSED_BELOW_THRESHOLD for r in results)
        assert all(r.degradations == [] for r in results)


class TestARefusedCompactionIsRecorded:
    def test_the_pass_still_completes(self):
        """Status answers "did the pass run"; degradation answers "did everything it
        intended happen". Fusing them into a fourth status value would make every
        existing `== COMPLETED` check silently stop matching."""
        bus = EventBus()
        service, _, _ = _build(_llm_compactor(bus, _response("tail-less", stop_reason=StopReason.TRUNCATED)))
        results = _run(service)

        degraded = [r for r in results if r.degradations]
        assert degraded
        assert all(r.status is GCStatus.COMPLETED for r in degraded)

    def test_the_reason_distinguishes_a_refusal_from_a_failure(self):
        """Different operator action: a refusal means the run or the cap was wrong
        and the next pass tries a different run; a provider failure means check the
        provider."""
        bus = EventBus()
        service, _, _ = _build(_llm_compactor(bus, _response("tail-less", stop_reason=StopReason.TRUNCATED)))
        results = _run(service)

        degradation = next(d for r in results for d in r.degradations)
        assert degradation.reason is DegradationReason.COMPACTION_REFUSED
        assert degradation.error_code == "COMPACTION_REFUSED"

    def test_it_names_the_turns_that_stayed_verbatim(self):
        """A list, not a count: a consumer showing *which* turns were affected
        cannot recover identity from a number."""
        bus = EventBus()
        service, _, _ = _build(_llm_compactor(bus, _response("tail-less", stop_reason=StopReason.TRUNCATED)))
        results = _run(service)

        degradation = next(d for r in results for d in r.degradations)
        assert degradation.turn_indices
        assert all(isinstance(index, int) for index in degradation.turn_indices)

    def test_nothing_is_filed_in_old_gen(self):
        bus = EventBus()
        service, memory, _ = _build(_llm_compactor(bus, _response("tail-less", stop_reason=StopReason.TRUNCATED)))
        _run(service)

        assert memory.get_old_gen() == []

    def test_the_refused_turns_stay_young_and_verbatim(self):
        """First do no harm: the context is correct, just larger than intended.
        A refusal that still dropped the turns would be the one unrecoverable
        outcome."""
        bus = EventBus()
        service, _, _ = _build(_llm_compactor(bus, _response("tail-less", stop_reason=StopReason.TRUNCATED)))
        _run(service)

        final = asyncio.run(service.gc_collect(SESSION))
        assert final, "the turns must still be there"
        assert all("[Compacted" not in message.content for message in final)

    def test_the_run_grows_across_passes_which_is_the_real_retry(self):
        """`CompactionRefused.retryable` does not mean "repeat this call" — the
        identical call reproduces the identical refusal. It means a later attempt
        can differ, and on the session path it does for free: the refused run stays
        young, so the next pass re-sweeps it with more turns cooled."""
        bus = EventBus()
        service, _, _ = _build(_llm_compactor(bus, _response("tail-less", stop_reason=StopReason.TRUNCATED)))
        seen: list[list[int]] = []
        service.event_bus.subscribe(EventType.AGING_DEGRADED, lambda e: seen.append(e.data["degradation"].turn_indices))

        _run(service)

        assert len(seen) >= 2, "need at least two degraded passes to observe growth"
        assert len(seen[-1]) > len(seen[0]), f"run should grow across passes, got {seen}"


class TestAFailedProviderCallIsRecorded:
    def test_a_provider_error_is_a_failure_not_a_refusal(self):
        bus = EventBus()
        service, _, _ = _build(_llm_compactor(bus, LLMTimeoutError("provider timed out", provider="offline")))
        results = _run(service)

        degradation = next(d for r in results for d in r.degradations)
        assert degradation.reason is DegradationReason.COMPACTION_FAILED
        assert degradation.error_code == "LLM_TIMEOUT"


class TestAFailedArchiveIsRecorded:
    def test_archive_failure_is_recorded_and_the_turn_stays(self):
        """The safe direction. A failed archive that still dropped the turn would
        lose it outright — extraction never ran, so nothing stands in for it."""
        service, _, _ = _build(memory_cls=_ArchiveFailingMemory, config=ARCHIVING_CONFIG)
        results = _run(service)

        degradations = [d for r in results for d in r.degradations]
        assert degradations
        assert all(d.reason is DegradationReason.ARCHIVE_FAILED for d in degradations)

        final = asyncio.run(service.gc_collect(SESSION))
        assert len(final) == TURN_COUNT, "no turn may be lost when archiving fails"


class TestThePayloadCarriesNoConversationContent:
    def test_an_unknown_exceptions_message_is_not_recorded(self):
        """Event payloads reach the dashboard over HTTP (CLAUDE.md §11). Our own
        `LLMGCError` messages are content-free by convention; a third-party
        compactor's are not a boundary we control, so only the type name is kept."""
        bus = EventBus()
        service, _, _ = _build(_NonLLMGCFailingCompactor(event_bus=bus))
        results = _run(service)

        degradations = [d for r in results for d in r.degradations]
        assert degradations
        for degradation in degradations:
            assert degradation.detail == "RuntimeError"
            assert "storage layer" not in degradation.detail
            assert degradation.error_code is None

    def test_our_own_error_message_is_recorded_because_it_is_content_free(self):
        bus = EventBus()
        service, _, _ = _build(_llm_compactor(bus, _response("tail-less", stop_reason=StopReason.TRUNCATED)))
        results = _run(service)

        degradation = next(d for r in results for d in r.degradations)
        assert "incomplete summary" in degradation.detail
        assert "storage layer" not in degradation.detail


class TestTheEventIsEmittedAsItHappens:
    def test_one_event_per_degradation(self):
        """Emitted separately from GC_FINISHED so a subscriber gets the detail
        without waiting for the pass to end."""
        bus = EventBus()
        service, _, _ = _build(_llm_compactor(bus, _response("tail-less", stop_reason=StopReason.TRUNCATED)))
        seen = []
        service.event_bus.subscribe(EventType.AGING_DEGRADED, seen.append)

        results = _run(service)
        total_recorded = sum(len(r.degradations) for r in results)

        assert len(seen) == total_recorded > 0

    def test_the_event_carries_the_session_id(self):
        """One monitor may watch several sessions; a degradation with no session
        identity cannot be attributed to a conversation."""
        bus = EventBus()
        service, _, _ = _build(_llm_compactor(bus, _response("tail-less", stop_reason=StopReason.TRUNCATED)))
        seen = []
        service.event_bus.subscribe(EventType.AGING_DEGRADED, seen.append)

        _run(service)
        assert all(event.data["session_id"] == SESSION for event in seen)

    def test_a_clean_pass_emits_nothing(self):
        service, _, _ = _build()
        seen = []
        service.event_bus.subscribe(EventType.AGING_DEGRADED, seen.append)

        _run(service)
        assert seen == []


class TestTheDashboardCanSeeIt:
    def test_the_health_snapshot_reports_the_degradation(self):
        """The end of the chain this whole change exists for: a dashboard rendering
        `compact_count` without reading this would draw compactions that never
        happened."""
        bus = EventBus()
        compactor = _llm_compactor(bus, _response("tail-less", stop_reason=StopReason.TRUNCATED))
        service, memory, service_bus = _build(compactor)
        monitor = ContextHealthMonitor(event_bus=service.event_bus, gc_config=CONFIG, generational_memory=memory)

        _run(service)
        breakdown = monitor.get_snapshot().gc_breakdown

        assert breakdown is not None
        assert breakdown.is_degraded is True
        assert breakdown.compact_count > 0, "the sweeper intended compactions"
        assert memory.get_old_gen() == [], "and none were filed - which is the discrepancy"

    def test_a_clean_snapshot_is_not_degraded(self):
        service, memory, _ = _build()
        monitor = ContextHealthMonitor(event_bus=service.event_bus, gc_config=CONFIG, generational_memory=memory)

        _run(service)
        breakdown = monitor.get_snapshot().gc_breakdown

        assert breakdown is not None
        assert breakdown.is_degraded is False


class TestTheRecordIsImmutable:
    def test_a_degradation_cannot_be_edited_after_the_fact(self):
        bus = EventBus()
        service, _, _ = _build(_llm_compactor(bus, _response("tail-less", stop_reason=StopReason.TRUNCATED)))
        results = _run(service)
        degradation = next(d for r in results for d in r.degradations)

        with pytest.raises(Exception):
            degradation.detail = "something else"
