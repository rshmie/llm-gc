import pytest
from llm_gc.models import Message
from llm_gc.scoring.reference_scorer import ReferenceScorer

@pytest.fixture
def scorer():
    return ReferenceScorer()

def test_referenced_message_scores_high(scorer):
    msg1 = Message(role="user", content="We should use Python for this project.")
    msg2 = Message(role="assistant", content="As you mentioned earlier, Python is a good choice.")
    msg3 = Message(role="user", content="Referring back to that, I agree.")
    result = scorer.score(msg1, [msg1, msg2, msg3])
    assert result.score > 0.0

def test_unreferenced_message_scores_zero(scorer):
    msg1 = Message(role="user", content="Hello world.")
    msg2 = Message(role="assistant", content="Hi there, how can I help?")
    result = scorer.score(msg1, [msg1, msg2])
    assert result.score == 0.0

def test_score_capped_at_one():
    scorer = ReferenceScorer(max_references=1)
    msg1 = Message(role="user", content="Use Python.")
    msg2 = Message(role="assistant", content="As mentioned earlier, yes.")
    msg3 = Message(role="user", content="As I said before, confirmed.")
    result = scorer.score(msg1, [msg1, msg2, msg3])
    assert result.score <= 1.0

def test_custom_max_references_changes_score():
    msg1 = Message(role="user", content="Use Python.")
    msg2 = Message(role="assistant", content="As mentioned earlier, good idea.")
    convo = [msg1, msg2]
    result_low = ReferenceScorer(max_references=1).score(msg1, convo)
    result_high = ReferenceScorer(max_references=10).score(msg1, convo)
    assert result_low.score >= result_high.score

def test_signals_contain_expected_keys(scorer):
    msg1 = Message(role="user", content="Hello.")
    result = scorer.score(msg1, [msg1])
    assert "reference_count" in result.signals
    assert "reference_score" in result.signals

def test_score_between_zero_and_one(scorer):
    msg1 = Message(role="user", content="Some message.")
    msg2 = Message(role="assistant", content="As you mentioned, sure.")
    result = scorer.score(msg1, [msg1, msg2])
    assert 0.0 <= result.score <= 1.

def test_score_when_message_is_empty(scorer):
    msg1 = Message(role="user", content="")
    msg2 = Message(role="assistant", content="")
    result = scorer.score(msg1, [msg1, msg2])
    assert result.score == 0.0
    assert result.signals["reference_count"] == 0

def test_score_with_multiple_references(scorer):
    msg1 = Message(role="user", content="Use Python.")
    msg2 = Message(role="assistant", content="As mentioned earlier, good idea.")
    msg3 = Message(role="user", content="Referring back to that, I agree.")
    convo = [msg1, msg2, msg3]
    result = scorer.score(msg1, convo)
    assert result.signals["reference_count"] == 2

def test_score_when_message_is_last_in_conversation(scorer):
    msg1 = Message(role="user", content="Use Python.")
    result = scorer.score(msg1, [msg1])
    assert result.score == 0.0
    assert result.signals["reference_count"] == 0