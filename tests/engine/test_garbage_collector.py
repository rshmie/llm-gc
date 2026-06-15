from unittest.mock import MagicMock

import pytest

from llm_gc.config import GCConfig
from llm_gc.engine import ContextComposer, GarbageCollector, GCResult, GCStatus
from llm_gc.engine.sweep import SweepClassification, SweepEntry, SweepResult, Sweeper
from llm_gc.events import EventBus, EventType
from llm_gc.models import Message
from llm_gc.scoring import RelevanceScorer
from llm_gc.scoring.relevance_scorer_result import RelevanceScorerResult


# ---- Helper factories -------------------------------------------------------
#
# Same pattern as test_context_composer.py: small builders that take only the
# fields a given test cares about and fill the rest with sensible defaults.
# A test for "did the orchestrator emit one event?" should not be cluttered
# with token counts and turn indices.


def _make_message(content: str = "msg", token_count: int = 10, role: str = "user", turn_index: int = 0) -> Message:
    return Message(role=role, content=content, token_count=token_count, turn_index=turn_index)


def _make_messages_above_threshold(config: GCConfig, count: int = 4) -> list[Message]:
    """Build a message list whose total tokens *exceed* the GC engagement threshold.

    Uses one big token_count rather than many small messages so the threshold
    is comfortably exceeded regardless of small config tweaks in tests.
    """
    threshold_tokens = int(config.context_window * config.gc_threshold)
    per_message = (threshold_tokens // count) + 1  # +1 so total > threshold
    return [_make_message(content=f"m{i}", token_count=per_message, turn_index=i) for i in range(count)]


def _make_messages_below_threshold(count: int = 3) -> list[Message]:
    return [_make_message(content=f"m{i}", token_count=5, turn_index=i) for i in range(count)]


def _make_scorer_result(message: Message, score: float = 0.5) -> RelevanceScorerResult:
    return RelevanceScorerResult(message=message, combined_score=score, scorer_results=[])


def _make_sweep_entry(message: Message, classification: SweepClassification = SweepClassification.KEEP) -> SweepEntry:
    return SweepEntry(message=message, classification=classification, relevance_score=0.5, turn_index=message.turn_index)


def _make_sweep_result(messages: list[Message]) -> SweepResult:
    """Build a SweepResult that classifies every message as KEEP.

    Default for the orchestrator tests because the orchestrator's contract is
    pipeline-level — *what* the sweeper decides is the sweeper's contract,
    tested elsewhere.
    """
    entries = [_make_sweep_entry(m, SweepClassification.KEEP) for m in messages]
    counts = {SweepClassification.KEEP: len(entries), SweepClassification.COMPACT: 0, SweepClassification.ARCHIVE: 0}
    return SweepResult(
        sweep_entries=entries,
        classification_counts=counts,
        processing_time_ms=1.0,
        total_keep_tokens=sum(m.token_count for m in messages),
        total_compact_tokens=0,
        total_archive_tokens=0,
    )


def _make_collector(
    event_bus: EventBus | None = None,
    config: GCConfig | None = None,
    scorer: RelevanceScorer | MagicMock | None = None,
    sweeper: Sweeper | MagicMock | None = None,
    composer: ContextComposer | MagicMock | None = None,
) -> GarbageCollector:
    """Construct a GarbageCollector with mock components by default.

    Each dependency is replaced with a MagicMock unless explicitly passed.
    Tests that care about a specific component's wiring inject it; tests that
    care only about a different concern accept the default mock.

    This is the dependency-injection payoff: orchestrator behaviour is
    testable in isolation without standing up real scorers, embedding
    models, or generational memory.
    """
    return GarbageCollector(
        event_bus=event_bus if event_bus is not None else EventBus(),
        gc_config=config if config is not None else GCConfig(),
        relevance_scorer=scorer if scorer is not None else MagicMock(spec=RelevanceScorer),
        sweeper=sweeper if sweeper is not None else MagicMock(spec=Sweeper),
        context_composer=composer if composer is not None else MagicMock(spec=ContextComposer),
    )


def _wire_happy_path_mocks(scorer: MagicMock, sweeper: MagicMock, composer: MagicMock, messages: list[Message]) -> None:
    """Configure the three downstream mocks to behave like a successful pipeline.

    - scorer.score(msg, conversation) returns a RelevanceScorerResult per call
    - sweeper.sweep(messages, scores) returns a SweepResult that classifies all KEEP
    - composer.compose(sweep_result) echoes the messages straight back

    Tests that need a specific downstream behaviour (failure, different sweep
    shape) override after calling this.
    """
    scorer.score.side_effect = lambda msg, conv: _make_scorer_result(msg)
    sweeper.sweep.return_value = _make_sweep_result(messages)
    composer.compose.return_value = list(messages)


# ---- COMPLETED status -------------------------------------------------------


class TestCollectCompleted:
    """The full happy path: input is above the GC threshold, every component
    succeeds, the orchestrator returns COMPLETED and forwards the composed
    list."""

    def test_returns_status_completed(self):
        config = GCConfig()
        messages = _make_messages_above_threshold(config)
        scorer, sweeper, composer = MagicMock(spec=RelevanceScorer), MagicMock(spec=Sweeper), MagicMock(spec=ContextComposer)
        _wire_happy_path_mocks(scorer, sweeper, composer, messages)
        gc = _make_collector(config=config, scorer=scorer, sweeper=sweeper, composer=composer)

        result = gc.collect(messages)

        assert result.status == GCStatus.COMPLETED

    def test_calls_scorer_once_per_message(self):
        config = GCConfig()
        messages = _make_messages_above_threshold(config, count=4)
        scorer, sweeper, composer = MagicMock(spec=RelevanceScorer), MagicMock(spec=Sweeper), MagicMock(spec=ContextComposer)
        _wire_happy_path_mocks(scorer, sweeper, composer, messages)
        gc = _make_collector(config=config, scorer=scorer, sweeper=sweeper, composer=composer)

        gc.collect(messages)

        assert scorer.score.call_count == 4

    def test_calls_sweeper_once_with_messages_and_scores(self):
        config = GCConfig()
        messages = _make_messages_above_threshold(config)
        scorer, sweeper, composer = MagicMock(spec=RelevanceScorer), MagicMock(spec=Sweeper), MagicMock(spec=ContextComposer)
        _wire_happy_path_mocks(scorer, sweeper, composer, messages)
        gc = _make_collector(config=config, scorer=scorer, sweeper=sweeper, composer=composer)

        gc.collect(messages)

        sweeper.sweep.assert_called_once()
        passed_messages, passed_scores = sweeper.sweep.call_args.args
        assert passed_messages == messages
        assert len(passed_scores) == len(messages)

    def test_calls_composer_once_with_sweep_result(self):
        config = GCConfig()
        messages = _make_messages_above_threshold(config)
        scorer, sweeper, composer = MagicMock(spec=RelevanceScorer), MagicMock(spec=Sweeper), MagicMock(spec=ContextComposer)
        _wire_happy_path_mocks(scorer, sweeper, composer, messages)
        gc = _make_collector(config=config, scorer=scorer, sweeper=sweeper, composer=composer)

        gc.collect(messages)

        composer.compose.assert_called_once_with(sweeper.sweep.return_value)

    def test_final_messages_match_composer_output(self):
        config = GCConfig()
        messages = _make_messages_above_threshold(config)
        scorer, sweeper, composer = MagicMock(spec=RelevanceScorer), MagicMock(spec=Sweeper), MagicMock(spec=ContextComposer)
        _wire_happy_path_mocks(scorer, sweeper, composer, messages)
        composed = [_make_message(content="composed", token_count=5, turn_index=99)]
        composer.compose.return_value = composed
        gc = _make_collector(config=config, scorer=scorer, sweeper=sweeper, composer=composer)

        result = gc.collect(messages)

        assert result.final_messages == composed

    def test_failure_fields_are_none_on_success(self):
        config = GCConfig()
        messages = _make_messages_above_threshold(config)
        scorer, sweeper, composer = MagicMock(spec=RelevanceScorer), MagicMock(spec=Sweeper), MagicMock(spec=ContextComposer)
        _wire_happy_path_mocks(scorer, sweeper, composer, messages)
        gc = _make_collector(config=config, scorer=scorer, sweeper=sweeper, composer=composer)

        result = gc.collect(messages)

        assert result.failure_reason is None
        assert result.failure_stage is None

    def test_sweep_result_carried_in_gc_result(self):
        config = GCConfig()
        messages = _make_messages_above_threshold(config)
        scorer, sweeper, composer = MagicMock(spec=RelevanceScorer), MagicMock(spec=Sweeper), MagicMock(spec=ContextComposer)
        _wire_happy_path_mocks(scorer, sweeper, composer, messages)
        gc = _make_collector(config=config, scorer=scorer, sweeper=sweeper, composer=composer)

        result = gc.collect(messages)

        assert result.sweep_result is sweeper.sweep.return_value

    def test_classification_counts_are_propagated(self):
        config = GCConfig()
        messages = _make_messages_above_threshold(config, count=5)
        scorer, sweeper, composer = MagicMock(spec=RelevanceScorer), MagicMock(spec=Sweeper), MagicMock(spec=ContextComposer)
        _wire_happy_path_mocks(scorer, sweeper, composer, messages)
        # Override the sweep result to have a mix of classifications.
        custom_entries = [
            _make_sweep_entry(messages[0], SweepClassification.KEEP),
            _make_sweep_entry(messages[1], SweepClassification.KEEP),
            _make_sweep_entry(messages[2], SweepClassification.COMPACT),
            _make_sweep_entry(messages[3], SweepClassification.COMPACT),
            _make_sweep_entry(messages[4], SweepClassification.ARCHIVE),
        ]
        sweeper.sweep.return_value = SweepResult(
            sweep_entries=custom_entries,
            classification_counts={
                SweepClassification.KEEP: 2,
                SweepClassification.COMPACT: 2,
                SweepClassification.ARCHIVE: 1,
            },
            processing_time_ms=1.0,
            total_keep_tokens=20,
            total_compact_tokens=20,
            total_archive_tokens=10,
        )
        gc = _make_collector(config=config, scorer=scorer, sweeper=sweeper, composer=composer)

        result = gc.collect(messages)

        assert result.kept_count == 2
        assert result.compacted_count == 2
        assert result.archived_count == 1


# ---- BYPASSED_BELOW_THRESHOLD status ----------------------------------------


class TestCollectBypassedBelowThreshold:
    """The fast-path skip: total input tokens are below the engagement threshold,
    so the pipeline does not run at all and the input list is returned verbatim."""

    def test_returns_status_bypassed_below_threshold(self):
        gc = _make_collector()

        result = gc.collect(_make_messages_below_threshold())

        assert result.status == GCStatus.BYPASSED_BELOW_THRESHOLD

    def test_final_messages_is_input_verbatim(self):
        messages = _make_messages_below_threshold()
        gc = _make_collector()

        result = gc.collect(messages)

        assert result.final_messages == messages

    def test_does_not_call_scorer(self):
        scorer = MagicMock(spec=RelevanceScorer)
        gc = _make_collector(scorer=scorer)

        gc.collect(_make_messages_below_threshold())

        scorer.score.assert_not_called()

    def test_does_not_call_sweeper(self):
        sweeper = MagicMock(spec=Sweeper)
        gc = _make_collector(sweeper=sweeper)

        gc.collect(_make_messages_below_threshold())

        sweeper.sweep.assert_not_called()

    def test_does_not_call_composer(self):
        composer = MagicMock(spec=ContextComposer)
        gc = _make_collector(composer=composer)

        gc.collect(_make_messages_below_threshold())

        composer.compose.assert_not_called()

    def test_sweep_result_is_none(self):
        gc = _make_collector()

        result = gc.collect(_make_messages_below_threshold())

        assert result.sweep_result is None

    def test_failure_fields_are_none(self):
        gc = _make_collector()

        result = gc.collect(_make_messages_below_threshold())

        assert result.failure_reason is None
        assert result.failure_stage is None

    def test_tokens_in_final_equals_tokens_before(self):
        messages = _make_messages_below_threshold()
        gc = _make_collector()

        result = gc.collect(messages)

        assert result.tokens_before == result.tokens_in_final
        assert result.tokens_saved == 0


# ---- BYPASSED_ON_ERROR status -----------------------------------------------


class TestCollectBypassedOnError:
    """When any pipeline stage raises, the orchestrator must:
    - never re-raise to the caller (first-do-no-harm contract)
    - return original messages so the LLM call can still proceed
    - record which stage failed so the dashboard / logs can reason about it
    """

    def test_scorer_failure_returns_status_bypassed_on_error(self):
        config = GCConfig()
        messages = _make_messages_above_threshold(config)
        scorer = MagicMock(spec=RelevanceScorer)
        scorer.score.side_effect = RuntimeError("scorer broke")
        gc = _make_collector(config=config, scorer=scorer)

        result = gc.collect(messages)

        assert result.status == GCStatus.BYPASSED_ON_ERROR

    def test_scorer_failure_records_failure_stage_score(self):
        config = GCConfig()
        messages = _make_messages_above_threshold(config)
        scorer = MagicMock(spec=RelevanceScorer)
        scorer.score.side_effect = RuntimeError("scorer broke")
        gc = _make_collector(config=config, scorer=scorer)

        result = gc.collect(messages)

        assert result.failure_stage == "score"

    def test_sweeper_failure_records_failure_stage_sweep(self):
        config = GCConfig()
        messages = _make_messages_above_threshold(config)
        scorer, sweeper, composer = MagicMock(spec=RelevanceScorer), MagicMock(spec=Sweeper), MagicMock(spec=ContextComposer)
        _wire_happy_path_mocks(scorer, sweeper, composer, messages)
        sweeper.sweep.side_effect = RuntimeError("sweeper broke")
        gc = _make_collector(config=config, scorer=scorer, sweeper=sweeper, composer=composer)

        result = gc.collect(messages)

        assert result.failure_stage == "sweep"

    def test_composer_failure_records_failure_stage_compose(self):
        config = GCConfig()
        messages = _make_messages_above_threshold(config)
        scorer, sweeper, composer = MagicMock(spec=RelevanceScorer), MagicMock(spec=Sweeper), MagicMock(spec=ContextComposer)
        _wire_happy_path_mocks(scorer, sweeper, composer, messages)
        composer.compose.side_effect = RuntimeError("composer broke")
        gc = _make_collector(config=config, scorer=scorer, sweeper=sweeper, composer=composer)

        result = gc.collect(messages)

        assert result.failure_stage == "compose"

    def test_failure_reason_carries_exception_message(self):
        config = GCConfig()
        messages = _make_messages_above_threshold(config)
        scorer = MagicMock(spec=RelevanceScorer)
        scorer.score.side_effect = RuntimeError("very specific message")
        gc = _make_collector(config=config, scorer=scorer)

        result = gc.collect(messages)

        assert "very specific message" in result.failure_reason

    def test_collect_does_not_raise_when_scorer_throws(self):
        config = GCConfig()
        messages = _make_messages_above_threshold(config)
        scorer = MagicMock(spec=RelevanceScorer)
        scorer.score.side_effect = RuntimeError("boom")
        gc = _make_collector(config=config, scorer=scorer)

        # The whole point of the orchestrator's error contract: the caller
        # (proxy, benchmark, CLI) must never see the exception. If this
        # assertion ever fails, the first-do-no-harm contract is broken.
        try:
            gc.collect(messages)
        except Exception as exc:  # noqa: BLE001 — intentionally broad
            pytest.fail(f"collect() raised {type(exc).__name__}: {exc}")

    def test_final_messages_is_original_input_when_failed(self):
        config = GCConfig()
        messages = _make_messages_above_threshold(config)
        scorer = MagicMock(spec=RelevanceScorer)
        scorer.score.side_effect = RuntimeError("boom")
        gc = _make_collector(config=config, scorer=scorer)

        result = gc.collect(messages)

        # The fallback contract: if GC fails, the proxy still has a
        # forwardable list — the original messages.
        assert result.final_messages == messages

    def test_sweep_result_is_none_when_failed_before_sweep(self):
        config = GCConfig()
        messages = _make_messages_above_threshold(config)
        scorer = MagicMock(spec=RelevanceScorer)
        scorer.score.side_effect = RuntimeError("scorer broke")
        gc = _make_collector(config=config, scorer=scorer)

        result = gc.collect(messages)

        assert result.sweep_result is None


# ---- gc_run_id and timing invariants ----------------------------------------


class TestCollectRunIdentity:
    """Cross-cutting invariants that hold regardless of the terminal status."""

    def test_each_call_produces_a_unique_run_id(self):
        config = GCConfig()
        messages = _make_messages_above_threshold(config)
        scorer, sweeper, composer = MagicMock(spec=RelevanceScorer), MagicMock(spec=Sweeper), MagicMock(spec=ContextComposer)
        _wire_happy_path_mocks(scorer, sweeper, composer, messages)
        gc = _make_collector(config=config, scorer=scorer, sweeper=sweeper, composer=composer)

        result1 = gc.collect(messages)
        result2 = gc.collect(messages)

        assert result1.gc_run_id != result2.gc_run_id

    def test_run_id_is_non_empty_for_completed(self):
        config = GCConfig()
        messages = _make_messages_above_threshold(config)
        scorer, sweeper, composer = MagicMock(spec=RelevanceScorer), MagicMock(spec=Sweeper), MagicMock(spec=ContextComposer)
        _wire_happy_path_mocks(scorer, sweeper, composer, messages)
        gc = _make_collector(config=config, scorer=scorer, sweeper=sweeper, composer=composer)

        result = gc.collect(messages)

        assert result.gc_run_id
        assert isinstance(result.gc_run_id, str)

    def test_run_id_is_non_empty_for_bypassed(self):
        gc = _make_collector()

        result = gc.collect(_make_messages_below_threshold())

        assert result.gc_run_id
        assert isinstance(result.gc_run_id, str)

    def test_run_id_is_non_empty_for_failed(self):
        config = GCConfig()
        messages = _make_messages_above_threshold(config)
        scorer = MagicMock(spec=RelevanceScorer)
        scorer.score.side_effect = RuntimeError("boom")
        gc = _make_collector(config=config, scorer=scorer)

        result = gc.collect(messages)

        assert result.gc_run_id
        assert isinstance(result.gc_run_id, str)

    def test_duration_ms_is_non_negative(self):
        config = GCConfig()
        messages = _make_messages_above_threshold(config)
        scorer, sweeper, composer = MagicMock(spec=RelevanceScorer), MagicMock(spec=Sweeper), MagicMock(spec=ContextComposer)
        _wire_happy_path_mocks(scorer, sweeper, composer, messages)
        gc = _make_collector(config=config, scorer=scorer, sweeper=sweeper, composer=composer)

        result = gc.collect(messages)

        assert result.duration_ms >= 0


# ---- Event emission ---------------------------------------------------------


class TestCollectEventEmission:
    """The orchestrator must emit exactly one terminal GC_FINISHED event per
    call, regardless of which status was reached. The dashboard, logger, and
    benchmarks all rely on this single-event contract."""

    def test_emits_one_gc_finished_event_on_success(self):
        received = []
        bus = EventBus()
        bus.subscribe(EventType.GC_FINISHED, lambda e: received.append(e))
        config = GCConfig()
        messages = _make_messages_above_threshold(config)
        scorer, sweeper, composer = MagicMock(spec=RelevanceScorer), MagicMock(spec=Sweeper), MagicMock(spec=ContextComposer)
        _wire_happy_path_mocks(scorer, sweeper, composer, messages)
        gc = _make_collector(event_bus=bus, config=config, scorer=scorer, sweeper=sweeper, composer=composer)

        gc.collect(messages)

        assert len(received) == 1

    def test_emits_one_gc_finished_event_on_bypass(self):
        received = []
        bus = EventBus()
        bus.subscribe(EventType.GC_FINISHED, lambda e: received.append(e))
        gc = _make_collector(event_bus=bus)

        gc.collect(_make_messages_below_threshold())

        assert len(received) == 1

    def test_emits_one_gc_finished_event_on_error(self):
        received = []
        bus = EventBus()
        bus.subscribe(EventType.GC_FINISHED, lambda e: received.append(e))
        config = GCConfig()
        messages = _make_messages_above_threshold(config)
        scorer = MagicMock(spec=RelevanceScorer)
        scorer.score.side_effect = RuntimeError("boom")
        gc = _make_collector(event_bus=bus, config=config, scorer=scorer)

        gc.collect(messages)

        assert len(received) == 1

    def test_event_payload_carries_gc_result(self):
        received = []
        bus = EventBus()
        bus.subscribe(EventType.GC_FINISHED, lambda e: received.append(e))
        config = GCConfig()
        messages = _make_messages_above_threshold(config)
        scorer, sweeper, composer = MagicMock(spec=RelevanceScorer), MagicMock(spec=Sweeper), MagicMock(spec=ContextComposer)
        _wire_happy_path_mocks(scorer, sweeper, composer, messages)
        gc = _make_collector(event_bus=bus, config=config, scorer=scorer, sweeper=sweeper, composer=composer)

        returned = gc.collect(messages)

        emitted_result = received[0].data["gc_result"]
        # The reactive (event) and imperative (return) views of the same run
        # carry the same GCResult — by reference. This is the single-source
        # invariant the dashboard depends on.
        assert emitted_result is returned

    def test_event_run_id_matches_returned_run_id(self):
        received = []
        bus = EventBus()
        bus.subscribe(EventType.GC_FINISHED, lambda e: received.append(e))
        config = GCConfig()
        messages = _make_messages_above_threshold(config)
        scorer, sweeper, composer = MagicMock(spec=RelevanceScorer), MagicMock(spec=Sweeper), MagicMock(spec=ContextComposer)
        _wire_happy_path_mocks(scorer, sweeper, composer, messages)
        gc = _make_collector(event_bus=bus, config=config, scorer=scorer, sweeper=sweeper, composer=composer)

        returned = gc.collect(messages)

        assert received[0].data["gc_run_id"] == returned.gc_run_id


# ---- Threshold boundary -----------------------------------------------------


class TestCollectThresholdBoundary:
    """The threshold is `context_window * gc_threshold`. Tests pin the exact
    inequality so a refactor that flips < to <= gets caught."""

    def test_strictly_below_threshold_bypasses(self):
        # context_window=100, gc_threshold=0.7 → threshold=70
        # Total tokens=69 (strictly below) → bypass
        config = GCConfig(context_window=100, gc_threshold=0.7)
        messages = [_make_message(token_count=23, turn_index=i) for i in range(3)]  # 69 total
        gc = _make_collector(config=config)

        result = gc.collect(messages)

        assert result.status == GCStatus.BYPASSED_BELOW_THRESHOLD

    def test_exactly_at_threshold_runs_pipeline(self):
        # context_window=100, gc_threshold=0.7 → threshold=70
        # Total tokens=70 → at threshold, should run (since bypass is `<`)
        config = GCConfig(context_window=100, gc_threshold=0.7)
        messages = [_make_message(token_count=35, turn_index=i) for i in range(2)]  # 70 total
        scorer, sweeper, composer = MagicMock(spec=RelevanceScorer), MagicMock(spec=Sweeper), MagicMock(spec=ContextComposer)
        _wire_happy_path_mocks(scorer, sweeper, composer, messages)
        gc = _make_collector(config=config, scorer=scorer, sweeper=sweeper, composer=composer)

        result = gc.collect(messages)

        assert result.status == GCStatus.COMPLETED

    def test_above_threshold_runs_pipeline(self):
        config = GCConfig(context_window=100, gc_threshold=0.7)
        messages = [_make_message(token_count=50, turn_index=i) for i in range(2)]  # 100 total
        scorer, sweeper, composer = MagicMock(spec=RelevanceScorer), MagicMock(spec=Sweeper), MagicMock(spec=ContextComposer)
        _wire_happy_path_mocks(scorer, sweeper, composer, messages)
        gc = _make_collector(config=config, scorer=scorer, sweeper=sweeper, composer=composer)

        result = gc.collect(messages)

        assert result.status == GCStatus.COMPLETED


# ---- Type-of-return invariant -----------------------------------------------


class TestCollectReturnType:
    def test_returns_gc_result_on_success(self):
        config = GCConfig()
        messages = _make_messages_above_threshold(config)
        scorer, sweeper, composer = MagicMock(spec=RelevanceScorer), MagicMock(spec=Sweeper), MagicMock(spec=ContextComposer)
        _wire_happy_path_mocks(scorer, sweeper, composer, messages)
        gc = _make_collector(config=config, scorer=scorer, sweeper=sweeper, composer=composer)

        assert isinstance(gc.collect(messages), GCResult)

    def test_returns_gc_result_on_bypass(self):
        gc = _make_collector()

        assert isinstance(gc.collect(_make_messages_below_threshold()), GCResult)

    def test_returns_gc_result_on_error(self):
        config = GCConfig()
        messages = _make_messages_above_threshold(config)
        scorer = MagicMock(spec=RelevanceScorer)
        scorer.score.side_effect = RuntimeError("boom")
        gc = _make_collector(config=config, scorer=scorer)

        assert isinstance(gc.collect(messages), GCResult)
