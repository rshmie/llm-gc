import pytest
from llm_gc.models import Message
from llm_gc.scoring.density_scorer import DensityScorer

@pytest.fixture
def scorer():
    return DensityScorer()

@pytest.fixture
def high_density_msg():
    return Message(role="user", content="Check `Python` 3.12 at https://python.org NASA 42 API")

@pytest.fixture
def low_density_msg():
    return Message(role="user", content="the thing is that we should probably just go ahead and do the stuff that we were talking about earlier today")

def test_high_density_scores_high(scorer, high_density_msg):
    result = scorer.score(high_density_msg, [])
    assert result.score > 0.5

def test_low_density_scores_low(scorer, low_density_msg):
    result = scorer.score(low_density_msg, [])
    assert result.score < 0.3

def test_score_capped_at_one():
    scorer = DensityScorer(max_density=0.01)
    msg = Message(role="user", content="NASA API 42 `code`")
    result = scorer.score(msg, [])
    assert result.score == 1.0

def test_custom_max_density_changes_score(high_density_msg):
    score1 = DensityScorer(max_density=0.5).score(high_density_msg, []).score
    score2 = DensityScorer(max_density=1.0).score(high_density_msg, []).score
    assert score1 > score2

def test_signals_dict_contains_expected_keys(scorer, high_density_msg):
    result = scorer.score(high_density_msg, [])
    assert set(result.signals.keys()) == {"signal_count", "token_counts", "density_ratio", "density_score"}

def test_score_between_zero_and_one(scorer, low_density_msg, high_density_msg):
    for msg in [low_density_msg, high_density_msg]:
        result = scorer.score(msg, [])
        assert 0.0 <= result.score <= 1.0

def test_score_when_token_count_is_zero(scorer):
    msg = Message(role="user", content="")
    result = scorer.score(msg, [])
    assert result.score == 0.0
    assert result.signals["token_counts"] == 0