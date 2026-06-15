from llm_gc.models import Message
from llm_gc.scoring.recency_scorer import RecencyScorer
from llm_gc.scoring.scorer_result import ScorerResult

def test_most_recent_message_scored_highest():
    scorer = RecencyScorer()
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi"),
        Message(role="user", content="Latest"),
    ]
    result = scorer.score(messages[-1], messages)
    assert result.score == 1.0


def test_oldest_message_scored_lowest():
    scorer = RecencyScorer()
    messages = [
        Message(role="user", content="First"),
        Message(role="assistant", content="Second"),
        Message(role="user", content="Third"),
    ]
    oldest = scorer.score(messages[0], messages)
    newest = scorer.score(messages[-1], messages)
    assert oldest.score < newest.score


def test_score_decreases_with_age():
    scorer = RecencyScorer()
    messages = [
        Message(role="user", content="First"),
        Message(role="assistant", content="Second"),
        Message(role="user", content="Third"),
    ]
    oldest = scorer.score(messages[0], messages)
    middle = scorer.score(messages[1], messages)
    newest = scorer.score(messages[-1], messages)
    assert oldest.score < middle.score < newest.score


def test_custom_decay_rate_gives_lower_scores():
    low_decay = RecencyScorer(decay_rate=0.1)
    high_decay = RecencyScorer(decay_rate=0.5)
    messages = [
        Message(role="user", content="First"),
        Message(role="assistant", content="Second"),
        Message(role="user", content="Third"),
    ]
    low_result = low_decay.score(messages[0], messages)
    high_result = high_decay.score(messages[0], messages)
    assert high_result.score < low_result.score


def test_single_message_conversation():
    scorer = RecencyScorer()
    messages = [Message(role="user", content="Only message")]
    result = scorer.score(messages[0], messages)
    assert result.score == 1.0


def test_signals_dict_contains_raw_values():
    scorer = RecencyScorer(decay_rate=0.3)
    messages = [
        Message(role="user", content="First"),
        Message(role="assistant", content="Second"),
        Message(role="user", content="Third"),
    ]
    result = scorer.score(messages[0], messages)
    assert isinstance(result, ScorerResult)
    assert "age" in result.signals
    assert result.signals["age"] == 2
    assert "decay_rate" in result.signals
    assert result.signals["decay_rate"] == 0.3
