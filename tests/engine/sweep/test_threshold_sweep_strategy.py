import pytest

from llm_gc.config import GCConfig
from llm_gc.config.constants import DEFAULT_MIN_COMPACTABLE_TOKENS, DEFAULT_KEEP_THRESHOLD, DEFAULT_ARCHIVE_THRESHOLD
from llm_gc.engine.sweep import ThresholdSweepStrategy, SweepClassification
from llm_gc.models import Message
from llm_gc.scoring import RelevanceScorerResult

def _make_message(content: str, token_count: int, role: str = "user", turn_index: int = 0) -> Message:
    return Message(role=role, content=content, token_count=token_count, turn_index=turn_index)

def _make_relevance_result(message: Message, combined_score: float) -> RelevanceScorerResult:
    return RelevanceScorerResult(message=message, combined_score=combined_score, scorer_results=[])

class TestThresholdClassification:
    def test_message_above_keep_threshold_is_kept(self):
        config = GCConfig(keep_threshold=0.7, min_compactable_tokens=50)
        strategy = ThresholdSweepStrategy(gc_config=config)
        msg = _make_message("This is a relevant message with enough tokens", token_count=100)
        scores = [_make_relevance_result(msg, 0.85)]

        entries = strategy.sweep([msg], scores)

        assert len(entries) == 1
        assert entries[0].classification == SweepClassification.KEEP

    def test_message_exactly_at_keep_threshold_is_kept(self):
        config = GCConfig(keep_threshold=0.7, min_compactable_tokens=50)
        strategy = ThresholdSweepStrategy(gc_config=config)
        msg = _make_message("Message at threshold boundary", token_count=100)
        scores = [_make_relevance_result(msg, 0.7)]

        entries = strategy.sweep([msg], scores)

        assert entries[0].classification == SweepClassification.KEEP

    def test_message_below_threshold_with_enough_tokens_is_compacted(self):
        config = GCConfig(keep_threshold=0.7, min_compactable_tokens=50)
        strategy = ThresholdSweepStrategy(gc_config=config)
        msg = _make_message("This message has low relevance but is long enough to compact", token_count=80)
        scores = [_make_relevance_result(msg, 0.3)]

        entries = strategy.sweep([msg], scores)

        assert entries[0].classification == SweepClassification.COMPACT

    def test_message_below_threshold_but_too_short_is_kept(self):
        config = GCConfig(keep_threshold=0.7, min_compactable_tokens=DEFAULT_MIN_COMPACTABLE_TOKENS)
        strategy = ThresholdSweepStrategy(gc_config=config)
        msg = _make_message("ok", token_count=2)
        scores = [_make_relevance_result(msg, 0.1)]

        entries = strategy.sweep([msg], scores)

        assert entries[0].classification == SweepClassification.KEEP

    def test_message_below_threshold_at_exact_min_tokens_boundary(self):
        config = GCConfig(keep_threshold=0.7, min_compactable_tokens=50)
        strategy = ThresholdSweepStrategy(gc_config=config)
        msg = _make_message("Exactly at token boundary", token_count=50)
        scores = [_make_relevance_result(msg, 0.4)]

        entries = strategy.sweep([msg], scores)

        assert entries[0].classification == SweepClassification.KEEP

    def test_message_below_threshold_above_min_tokens_is_compacted(self):
        config = GCConfig(keep_threshold=0.7, min_compactable_tokens=50)
        strategy = ThresholdSweepStrategy(gc_config=config)
        msg = _make_message("Just above the token boundary", token_count=51)
        scores = [_make_relevance_result(msg, 0.4)]

        entries = strategy.sweep([msg], scores)

        assert entries[0].classification == SweepClassification.COMPACT

class TestThresholdMultipleMessages:
    def test_mixed_conversation_classifies_correctly(self):
        config = GCConfig(keep_threshold=0.7, archive_threshold=0.3, min_compactable_tokens=50)
        strategy = ThresholdSweepStrategy(gc_config=config)
        messages = [
            _make_message("High relevance message", token_count=100, turn_index=0),
            _make_message("Medium relevance, long message", token_count=200, turn_index=1),
            _make_message("Very low relevance, short message", token_count=10, turn_index=2),
            _make_message("Low relevance, long message", token_count=200, turn_index=3),
            _make_message("Another high relevance", token_count=150, turn_index=4),
        ]
        scores = [
            _make_relevance_result(messages[0], 0.9),
            _make_relevance_result(messages[1], 0.5),
            _make_relevance_result(messages[2], 0.1),
            _make_relevance_result(messages[3], 0.2),
            _make_relevance_result(messages[4], 0.75),
        ]

        entries = strategy.sweep(messages, scores)

        assert entries[0].classification == SweepClassification.KEEP
        assert entries[1].classification == SweepClassification.COMPACT
        assert entries[2].classification == SweepClassification.KEEP  # too short to compact
        assert entries[3].classification == SweepClassification.ARCHIVE
        assert entries[4].classification == SweepClassification.KEEP

    def test_all_messages_above_threshold(self):
        config = GCConfig(keep_threshold=0.7, min_compactable_tokens=50)
        strategy = ThresholdSweepStrategy(gc_config=config)
        messages = [
            _make_message("First", token_count=100, turn_index=0),
            _make_message("Second", token_count=100, turn_index=1),
            _make_message("Third", token_count=100, turn_index=2),
        ]
        scores = [
            _make_relevance_result(messages[0], 0.8),
            _make_relevance_result(messages[1], 0.9),
            _make_relevance_result(messages[2], 0.75),
        ]

        entries = strategy.sweep(messages, scores)

        assert all(e.classification == SweepClassification.KEEP for e in entries)

    def test_all_messages_below_threshold_and_long_enough(self):
        config = GCConfig(keep_threshold=0.7, min_compactable_tokens=50)
        strategy = ThresholdSweepStrategy(gc_config=config)
        messages = [
            _make_message("First long message", token_count=100, turn_index=0),
            _make_message("Second long message", token_count=200, turn_index=1),
            _make_message("Third long message", token_count=150, turn_index=2),
        ]
        scores = [
            _make_relevance_result(messages[0], 0.32),
            _make_relevance_result(messages[1], 0.31),
            _make_relevance_result(messages[2], 0.4),
        ]

        entries = strategy.sweep(messages, scores)

        assert all(e.classification == SweepClassification.COMPACT for e in entries)

    def test_empty_conversation_returns_empty_list(self):
        config = GCConfig(keep_threshold=0.7, min_compactable_tokens=50)
        strategy = ThresholdSweepStrategy(gc_config=config)

        entries = strategy.sweep([], [])

        assert entries == []

class TestThresholdEntryFields:
    def test_entry_contains_correct_relevance_score(self):
        config = GCConfig(keep_threshold=0.7, min_compactable_tokens=50)
        strategy = ThresholdSweepStrategy(gc_config=config)
        msg = _make_message("Test message", token_count=100, turn_index=3)
        scores = [_make_relevance_result(msg, 0.45)]

        entries = strategy.sweep([msg], scores)

        assert entries[0].relevance_score == pytest.approx(0.45)

    def test_entry_contains_correct_turn_index(self):
        config = GCConfig(keep_threshold=0.7, min_compactable_tokens=50)
        strategy = ThresholdSweepStrategy(gc_config=config)
        msg = _make_message("Test message", token_count=100, turn_index=7)
        scores = [_make_relevance_result(msg, 0.8)]

        entries = strategy.sweep([msg], scores)

        assert entries[0].turn_index == 7

    def test_entry_override_is_always_false(self):
        config = GCConfig(keep_threshold=0.7, min_compactable_tokens=50)
        strategy = ThresholdSweepStrategy(gc_config=config)
        msg = _make_message("Test message", token_count=100)
        scores = [_make_relevance_result(msg, 0.3)]

        entries = strategy.sweep([msg], scores)

        assert entries[0].override_applied is False
        assert entries[0].override_reason is None

    def test_entry_contains_original_message(self):
        config = GCConfig(keep_threshold=0.7, min_compactable_tokens=50)
        strategy = ThresholdSweepStrategy(gc_config=config)
        msg = _make_message("Original content here", token_count=60, role="assistant")
        scores = [_make_relevance_result(msg, 0.5)]

        entries = strategy.sweep([msg], scores)

        assert entries[0].message == msg
        assert entries[0].message.content == "Original content here"
        assert entries[0].message.role == "assistant"

class TestThresholdConfigVariations:
    def test_low_keep_threshold_keeps_more(self):
        config = GCConfig(keep_threshold=0.3, min_compactable_tokens=50)
        strategy = ThresholdSweepStrategy(gc_config=config)
        msg = _make_message("Medium relevance message", token_count=100)
        scores = [_make_relevance_result(msg, 0.5)]

        entries = strategy.sweep([msg], scores)

        assert entries[0].classification == SweepClassification.KEEP

    def test_high_keep_threshold_compacts_more(self):
        config = GCConfig(keep_threshold=0.9, min_compactable_tokens=50)
        strategy = ThresholdSweepStrategy(gc_config=config)
        msg = _make_message("Medium relevance message", token_count=100)
        scores = [_make_relevance_result(msg, 0.8)]

        entries = strategy.sweep([msg], scores)

        assert entries[0].classification == SweepClassification.COMPACT

    def test_high_min_compactable_tokens_keeps_more(self):
        config = GCConfig(keep_threshold=DEFAULT_KEEP_THRESHOLD, min_compactable_tokens=200)
        strategy = ThresholdSweepStrategy(gc_config=config)
        msg = _make_message("Low score but not long enough for high threshold", token_count=150)
        scores = [_make_relevance_result(msg, 0.3)]

        entries = strategy.sweep([msg], scores)

        assert entries[0].classification == SweepClassification.KEEP

    def test_zero_min_compactable_tokens_compacts_everything_below_threshold(self):
        config = GCConfig(keep_threshold=DEFAULT_KEEP_THRESHOLD, archive_threshold=DEFAULT_ARCHIVE_THRESHOLD, min_compactable_tokens=0)
        strategy = ThresholdSweepStrategy(gc_config=config)
        msg = _make_message("ok", token_count=2)
        scores = [_make_relevance_result(msg, 0.45)]

        entries = strategy.sweep([msg], scores)

        assert entries[0].classification == SweepClassification.COMPACT


class TestThresholdArchiveClassification:
    def test_message_below_archive_threshold_is_archived(self):
        config = GCConfig(keep_threshold=0.7, archive_threshold=0.3, min_compactable_tokens=50)
        strategy = ThresholdSweepStrategy(gc_config=config)
        msg = _make_message("Very old irrelevant message", token_count=100)
        scores = [_make_relevance_result(msg, 0.1)]

        entries = strategy.sweep([msg], scores)

        assert entries[0].classification == SweepClassification.ARCHIVE

    def test_message_exactly_at_archive_threshold_is_compacted(self):
        config = GCConfig(keep_threshold=0.7, archive_threshold=0.3, min_compactable_tokens=50)
        strategy = ThresholdSweepStrategy(gc_config=config)
        msg = _make_message("Borderline message", token_count=100)
        scores = [_make_relevance_result(msg, 0.3)]

        entries = strategy.sweep([msg], scores)

        assert entries[0].classification == SweepClassification.COMPACT

    def test_message_just_below_archive_threshold_is_archived(self):
        config = GCConfig(keep_threshold=0.7, archive_threshold=0.3, min_compactable_tokens=50)
        strategy = ThresholdSweepStrategy(gc_config=config)
        msg = _make_message("Just below archive boundary", token_count=100)
        scores = [_make_relevance_result(msg, 0.29)]

        entries = strategy.sweep([msg], scores)

        assert entries[0].classification == SweepClassification.ARCHIVE

    def test_short_message_below_archive_threshold_is_still_kept(self):
        config = GCConfig(keep_threshold=0.7, archive_threshold=0.3, min_compactable_tokens=50)
        strategy = ThresholdSweepStrategy(gc_config=config)
        msg = _make_message("ok", token_count=5)
        scores = [_make_relevance_result(msg, 0.1)]

        entries = strategy.sweep([msg], scores)

        assert entries[0].classification == SweepClassification.KEEP

    def test_custom_archive_threshold(self):
        config = GCConfig(keep_threshold=0.7, archive_threshold=0.5, min_compactable_tokens=50)
        strategy = ThresholdSweepStrategy(gc_config=config)
        msg = _make_message("Medium score message", token_count=100)
        scores = [_make_relevance_result(msg, 0.4)]

        entries = strategy.sweep([msg], scores)

        assert entries[0].classification == SweepClassification.ARCHIVE
