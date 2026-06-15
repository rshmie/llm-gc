import pytest

from llm_gc.scoring.relevance_type import RelevanceType
from llm_gc.scoring.base_scorer import BaseScorer

def test_cannot_instantiate_base_scorer():
    with pytest.raises(TypeError):
        BaseScorer()

def test_subclass_without_score_raises_error():
    class IncompleteScorer(BaseScorer):
        pass

    with pytest.raises(TypeError):
        IncompleteScorer()

def test_subclass_with_score_works():
    class RelevanceScorer(BaseScorer):
        def score(self, message, conversation):
            return None

    scorer = RelevanceScorer(scorer_name=RelevanceType.RECENCY)
    assert scorer.scorer_name == RelevanceType.RECENCY