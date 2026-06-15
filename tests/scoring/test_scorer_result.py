import pytest

from llm_gc.scoring.relevance_type import RelevanceType
from llm_gc.scoring.scorer_result import ScorerResult

def test_valid_scorer_result():
    reason = "The document is recent and relevant to the query."
    signals = {"turn_age": "0.5"}
    assert ScorerResult(score=0.9, scorer_name=RelevanceType.RECENCY, reason=reason, signals=signals)

def test_score_below_zero_raises_error():
    with pytest.raises(ValueError):
        ScorerResult(score=-0.1, scorer_name=RelevanceType.RECENCY, reason="Invalid score", signals={})

def test_score_above_one_raises_error():
    with pytest.raises(ValueError):
        ScorerResult(score=1.1, scorer_name=RelevanceType.RECENCY, reason="Invalid score", signals={})