from llm_gc.config import GCConfig
from llm_gc.engine.sweep import (
    Sweeper, ThresholdSweepStrategy, SweepClassification
)
from llm_gc.events import EventBus, EventType
from llm_gc.models import Message
from llm_gc.scoring import RelevanceScorerResult

def _make_message(content: str, token_count: int, role: str = "user", turn_index: int = 0) -> Message:
    return Message(role=role, content=content, token_count=token_count, turn_index=turn_index)

def _make_relevance_result(message: Message, combined_score: float) -> RelevanceScorerResult:
    return RelevanceScorerResult(message=message, combined_score=combined_score, scorer_results=[])

def _make_sweeper(keep_threshold: float = 0.7, min_compactable_tokens: int = 50,
                  last_n_turns_to_keep: int = 5, event_bus: EventBus | None = None) -> Sweeper:
    config = GCConfig(keep_threshold=keep_threshold, min_compactable_tokens=min_compactable_tokens,
                      last_n_turns_to_keep=last_n_turns_to_keep)
    strategy = ThresholdSweepStrategy(gc_config=config)
    bus = event_bus if event_bus is not None else EventBus()
    return Sweeper(sweeper_strategy=strategy, gc_config=config, event_bus=bus)

class TestSweeperOverrideLastNTurns:
    def test_last_n_turns_are_always_kept(self):
        sweeper = _make_sweeper(keep_threshold=0.7, last_n_turns_to_keep=3)
        messages = [
            _make_message("Old message 1", token_count=100, turn_index=0),
            _make_message("Old message 2", token_count=100, turn_index=1),
            _make_message("Old message 3", token_count=100, turn_index=2),
            _make_message("Recent message 1", token_count=100, turn_index=3),
            _make_message("Recent message 2", token_count=100, turn_index=4),
            _make_message("Recent message 3", token_count=100, turn_index=5),
        ]
        scores = [
            _make_relevance_result(messages[0], 0.1),
            _make_relevance_result(messages[1], 0.2),
            _make_relevance_result(messages[2], 0.15),
            _make_relevance_result(messages[3], 0.1),
            _make_relevance_result(messages[4], 0.05),
            _make_relevance_result(messages[5], 0.1),
        ]

        result = sweeper.sweep(messages, scores)

        assert result.sweep_entries[3].classification == SweepClassification.KEEP
        assert result.sweep_entries[4].classification == SweepClassification.KEEP
        assert result.sweep_entries[5].classification == SweepClassification.KEEP

    def test_last_n_override_sets_override_applied_true(self):
        sweeper = _make_sweeper(keep_threshold=0.7, last_n_turns_to_keep=2)
        messages = [
            _make_message("Old message", token_count=100, turn_index=0),
            _make_message("Recent 1", token_count=100, turn_index=1),
            _make_message("Recent 2", token_count=100, turn_index=2),
        ]
        scores = [
            _make_relevance_result(messages[0], 0.1),
            _make_relevance_result(messages[1], 0.1),
            _make_relevance_result(messages[2], 0.1),
        ]

        result = sweeper.sweep(messages, scores)

        assert result.sweep_entries[0].override_applied is False
        assert result.sweep_entries[1].override_applied is True
        assert result.sweep_entries[2].override_applied is True

    def test_last_n_override_sets_override_reason(self):
        sweeper = _make_sweeper(keep_threshold=0.7, last_n_turns_to_keep=1)
        messages = [
            _make_message("Old", token_count=100, turn_index=0),
            _make_message("Recent", token_count=100, turn_index=1),
        ]
        scores = [
            _make_relevance_result(messages[0], 0.1),
            _make_relevance_result(messages[1], 0.1),
        ]

        result = sweeper.sweep(messages, scores)

        assert result.sweep_entries[1].override_reason is not None
        assert "last_n_turns" in result.sweep_entries[1].override_reason.lower()

    def test_messages_before_last_n_can_be_compacted(self):
        sweeper = _make_sweeper(keep_threshold=0.7, last_n_turns_to_keep=2)
        messages = [
            _make_message("Old low relevance message", token_count=100, turn_index=0),
            _make_message("Recent 1", token_count=100, turn_index=1),
            _make_message("Recent 2", token_count=100, turn_index=2),
        ]
        scores = [
            _make_relevance_result(messages[0], 0.4),
            _make_relevance_result(messages[1], 0.4),
            _make_relevance_result(messages[2], 0.4),
        ]

        result = sweeper.sweep(messages, scores)

        assert result.sweep_entries[0].classification == SweepClassification.COMPACT

    def test_last_n_zero_means_no_override(self):
        sweeper = _make_sweeper(keep_threshold=0.7, last_n_turns_to_keep=0)
        messages = [
            _make_message("Message 1", token_count=100, turn_index=0),
            _make_message("Message 2", token_count=100, turn_index=1),
        ]
        scores = [
            _make_relevance_result(messages[0], 0.4),
            _make_relevance_result(messages[1], 0.5),
        ]

        result = sweeper.sweep(messages, scores)

        assert result.sweep_entries[0].classification == SweepClassification.COMPACT
        assert result.sweep_entries[1].classification == SweepClassification.COMPACT
        assert all(e.override_applied is False for e in result.sweep_entries)

class TestSweeperOverrideSystemPrompt:
    def test_system_message_is_always_kept(self):
        sweeper = _make_sweeper(keep_threshold=0.7, last_n_turns_to_keep=0)
        messages = [
            _make_message("You are a helpful assistant.", token_count=100, role="system", turn_index=0),
            _make_message("Hello", token_count=100, role="user", turn_index=1),
        ]
        scores = [
            _make_relevance_result(messages[0], 0.1),
            _make_relevance_result(messages[1], 0.1),
        ]

        result = sweeper.sweep(messages, scores)

        assert result.sweep_entries[0].classification == SweepClassification.KEEP
        assert result.sweep_entries[0].override_applied is True

    def test_system_message_override_reason_mentions_system(self):
        sweeper = _make_sweeper(keep_threshold=0.7, last_n_turns_to_keep=0)
        messages = [
            _make_message("You are a coding assistant.", token_count=100, role="system", turn_index=0),
        ]
        scores = [_make_relevance_result(messages[0], 0.05)]

        result = sweeper.sweep(messages, scores)

        assert "system" in result.sweep_entries[0].override_reason.lower()

    def test_system_message_in_middle_of_conversation_is_kept(self):
        sweeper = _make_sweeper(keep_threshold=0.7, last_n_turns_to_keep=0)
        messages = [
            _make_message("User message", token_count=100, role="user", turn_index=0),
            _make_message("System instruction", token_count=100, role="system", turn_index=1),
            _make_message("Another user message", token_count=100, role="user", turn_index=2),
        ]
        scores = [
            _make_relevance_result(messages[0], 0.1),
            _make_relevance_result(messages[1], 0.05),
            _make_relevance_result(messages[2], 0.1),
        ]

        result = sweeper.sweep(messages, scores)

        assert result.sweep_entries[1].classification == SweepClassification.KEEP
        assert result.sweep_entries[1].override_applied is True

class TestSweeperSweepResult:
    def test_classification_counts_are_correct(self):
        sweeper = _make_sweeper(keep_threshold=0.7, last_n_turns_to_keep=0)
        messages = [
            _make_message("High relevance", token_count=100, turn_index=0),
            _make_message("Low relevance long", token_count=100, turn_index=1),
            _make_message("Low relevance long 2", token_count=100, turn_index=2),
            _make_message("Very low relevance", token_count=100, turn_index=3),
            _make_message("High relevance 2", token_count=100, turn_index=4),
        ]
        scores = [
            _make_relevance_result(messages[0], 0.9),
            _make_relevance_result(messages[1], 0.4),
            _make_relevance_result(messages[2], 0.5),
            _make_relevance_result(messages[3], 0.1),
            _make_relevance_result(messages[4], 0.8),
        ]

        result = sweeper.sweep(messages, scores)

        assert result.classification_counts[SweepClassification.KEEP] == 2
        assert result.classification_counts[SweepClassification.COMPACT] == 2
        assert result.classification_counts[SweepClassification.ARCHIVE] == 1

    def test_token_totals_are_correct(self):
        sweeper = _make_sweeper(keep_threshold=0.7, last_n_turns_to_keep=0)
        messages = [
            _make_message("Kept message", token_count=150, turn_index=0),
            _make_message("Compacted message", token_count=200, turn_index=1),
            _make_message("Archived message", token_count=120, turn_index=2),
            _make_message("Another kept", token_count=80, turn_index=3),
        ]
        scores = [
            _make_relevance_result(messages[0], 0.9),
            _make_relevance_result(messages[1], 0.5),
            _make_relevance_result(messages[2], 0.1),
            _make_relevance_result(messages[3], 0.8),
        ]

        result = sweeper.sweep(messages, scores)

        assert result.total_keep_tokens == 230
        assert result.total_compact_tokens == 200
        assert result.total_archive_tokens == 120

    def test_token_totals_include_overridden_messages(self):
        sweeper = _make_sweeper(keep_threshold=0.7, last_n_turns_to_keep=2)
        messages = [
            _make_message("Old compactable", token_count=100, turn_index=0),
            _make_message("Recent overridden 1", token_count=150, turn_index=1),
            _make_message("Recent overridden 2", token_count=200, turn_index=2),
        ]
        scores = [
            _make_relevance_result(messages[0], 0.4),
            _make_relevance_result(messages[1], 0.1),
            _make_relevance_result(messages[2], 0.1),
        ]

        result = sweeper.sweep(messages, scores)

        assert result.total_keep_tokens == 350
        assert result.total_compact_tokens == 100

    def test_processing_time_is_non_negative(self):
        sweeper = _make_sweeper()
        messages = [_make_message("Test", token_count=100)]
        scores = [_make_relevance_result(messages[0], 0.5)]

        result = sweeper.sweep(messages, scores)

        assert result.processing_time_ms >= 0

    def test_empty_conversation_returns_zero_counts(self):
        sweeper = _make_sweeper(last_n_turns_to_keep=0)

        result = sweeper.sweep([], [])

        assert result.classification_counts[SweepClassification.KEEP] == 0
        assert result.classification_counts[SweepClassification.COMPACT] == 0
        assert result.total_keep_tokens == 0
        assert result.total_compact_tokens == 0
        assert result.sweep_entries == []

class TestSweeperEventEmission:
    def test_sweep_completed_event_is_emitted(self):
        emitted_events = []
        bus = EventBus()
        bus.subscribe(EventType.SWEEP_COMPLETED, lambda event: emitted_events.append(event))
        sweeper = _make_sweeper(event_bus=bus, last_n_turns_to_keep=0)
        messages = [_make_message("Test message", token_count=100)]
        scores = [_make_relevance_result(messages[0], 0.5)]

        sweeper.sweep(messages, scores)

        assert len(emitted_events) == 1
        assert emitted_events[0].event_type == EventType.SWEEP_COMPLETED

    def test_sweep_event_contains_entries_and_counts(self):
        emitted_events = []
        bus = EventBus()
        bus.subscribe(EventType.SWEEP_COMPLETED, lambda event: emitted_events.append(event))
        sweeper = _make_sweeper(event_bus=bus, last_n_turns_to_keep=0)
        messages = [
            _make_message("Msg 1", token_count=100, turn_index=0),
            _make_message("Msg 2", token_count=100, turn_index=1),
        ]
        scores = [
            _make_relevance_result(messages[0], 0.9),
            _make_relevance_result(messages[1], 0.2),
        ]

        sweeper.sweep(messages, scores)

        event_data = emitted_events[0].data
        assert "sweep_entries" in event_data
        assert "classification_counts" in event_data
        assert "current_message_turn" in event_data
        assert event_data["current_message_turn"] == 2

    def test_no_crash_when_event_bus_is_none(self):
        config = GCConfig(keep_threshold=0.7, min_compactable_tokens=50, last_n_turns_to_keep=0)
        strategy = ThresholdSweepStrategy(gc_config=config)
        sweeper = Sweeper(sweeper_strategy=strategy, gc_config=config, event_bus=None)
        messages = [_make_message("Test", token_count=100)]
        scores = [_make_relevance_result(messages[0], 0.5)]

        result = sweeper.sweep(messages, scores)

        assert result is not None

class TestSweeperOverridePriority:
    def test_system_message_in_last_n_gets_system_override(self):
        sweeper = _make_sweeper(keep_threshold=0.7, last_n_turns_to_keep=5)
        messages = [
            _make_message("System prompt", token_count=100, role="system", turn_index=0),
        ]
        scores = [_make_relevance_result(messages[0], 0.1)]

        result = sweeper.sweep(messages, scores)

        assert result.sweep_entries[0].override_applied is True
        assert "last_n_turns" in result.sweep_entries[0].override_reason.lower()

    def test_high_score_message_in_last_n_still_gets_override(self):
        sweeper = _make_sweeper(keep_threshold=0.7, last_n_turns_to_keep=2)
        messages = [
            _make_message("Old", token_count=100, turn_index=0),
            _make_message("Recent high score", token_count=100, turn_index=1),
            _make_message("Recent high score 2", token_count=100, turn_index=2),
        ]
        scores = [
            _make_relevance_result(messages[0], 0.9),
            _make_relevance_result(messages[1], 0.95),
            _make_relevance_result(messages[2], 0.85),
        ]

        result = sweeper.sweep(messages, scores)

        assert result.sweep_entries[0].override_applied is False
        assert result.sweep_entries[0].classification == SweepClassification.KEEP
        assert result.sweep_entries[1].override_applied is True
        assert result.sweep_entries[2].override_applied is True