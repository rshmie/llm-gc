import pytest
from llm_gc.models import Message
from llm_gc.scoring.decision_scorer import DecisionScorer

@pytest.fixture
def scorer():
    return DecisionScorer()

def test_high_decision_count_scores_high(scorer):
    msg = Message(role="user", content="Let's go with option A. I've decided we should use Python. The final choice is REST.")
    result = scorer.score(msg, [])
    assert result.score >= 0.7

def test_no_decisions_scores_zero(scorer):
    msg = Message(role="user", content="hello world this is just a normal message with nothing special")
    result = scorer.score(msg, [])
    assert result.score == 0.0

def test_score_capped_at_one():
    scorer = DecisionScorer(max_decision=1)
    msg = Message(role="user", content="Let's go with A. I've decided on B. The final choice is C.")
    result = scorer.score(msg, [])
    assert result.score == 1.0

def test_custom_max_decision_changes_score():
    msg = Message(role="user", content="Let's go with option A. I've decided on B.")
    result_low = DecisionScorer(max_decision=2).score(msg, [])
    result_high = DecisionScorer(max_decision=10).score(msg, [])
    assert result_low.score > result_high.score

def test_signals_contain_expected_keys(scorer):
    msg = Message(role="user", content="Let's do it.")
    result = scorer.score(msg, [])
    assert "decision_signal_count" in result.signals
    assert "decision_score" in result.signals

def test_signals_with_mixed_case(scorer):
    msg = Message(role="user", content="Let's GO with option A.")
    result = scorer.score(msg, [])
    assert result.signals["decision_signal_count"] > 0

def test_score_between_zero_and_one(scorer):
    msg = Message(role="user", content="I think we should decide to go with plan B.")
    result = scorer.score(msg, [])
    assert 0.0 <= result.score <= 1.0

def test_score_when_message_is_empty(scorer):
    msg = Message(role="user", content="")
    result = scorer.score(msg, [])
    assert result.score == 0.0
    assert result.signals["decision_signal_count"] == 0