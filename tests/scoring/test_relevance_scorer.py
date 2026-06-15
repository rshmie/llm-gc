import pytest

from llm_gc.events import EventBus, EventType
from llm_gc.models import Message
from llm_gc.scoring.base_scorer import BaseScorer
from llm_gc.scoring.relevance_type import RelevanceType
from llm_gc.scoring.scorer_result import ScorerResult
from llm_gc.scoring.relevance_scorer import RelevanceScorer

class FakeScorer(BaseScorer):
    """A fake scorer that always returns a fixed score."""

    def __init__(self, scorer_name: RelevanceType, fixed_score: float):
        super().__init__(scorer_name=scorer_name)
        self.fixed_score = fixed_score

    def score(self, message: Message, conversation: list[Message]) -> ScorerResult:
        return ScorerResult(
            score=self.fixed_score,
            scorer_name=self.scorer_name,
            reason=f"Fixed score: {self.fixed_score}",
            signals={"fixed_score": self.fixed_score},
        )

@pytest.fixture
def default_weights():
    return {
        RelevanceType.RECENCY: 0.30,
        RelevanceType.SIMILARITY: 0.30,
        RelevanceType.DENSITY: 0.15,
        RelevanceType.DECISION: 0.15,
        RelevanceType.REFERENCE: 0.10,
    }

@pytest.fixture
def all_scorers_max():
    """All scorers return 1.0."""
    return [
        FakeScorer(RelevanceType.RECENCY, 1.0),
        FakeScorer(RelevanceType.SIMILARITY, 1.0),
        FakeScorer(RelevanceType.DENSITY, 1.0),
        FakeScorer(RelevanceType.DECISION, 1.0),
        FakeScorer(RelevanceType.REFERENCE, 1.0),
    ]

@pytest.fixture
def all_scorers_zero():
    """All scorers return 0.0."""
    return [
        FakeScorer(RelevanceType.RECENCY, 0.0),
        FakeScorer(RelevanceType.SIMILARITY, 0.0),
        FakeScorer(RelevanceType.DENSITY, 0.0),
        FakeScorer(RelevanceType.DECISION, 0.0),
        FakeScorer(RelevanceType.REFERENCE, 0.0),
    ]

@pytest.fixture
def conversation():
    return [
        Message(role="user", content="How do I set up the database?"),
        Message(role="assistant", content="You can use PostgreSQL with SQLAlchemy."),
    ]

def test_all_scorers_max_gives_combined_one(all_scorers_max, default_weights, conversation):
    scorer = RelevanceScorer(all_scorers_max, default_weights)
    result = scorer.score(conversation[0], conversation)
    assert result.combined_score == pytest.approx(1.0)

def test_all_scorers_zero_gives_combined_zero(all_scorers_zero, default_weights, conversation):
    scorer = RelevanceScorer(all_scorers_zero, default_weights)
    result = scorer.score(conversation[0], conversation)
    assert result.combined_score == pytest.approx(0.0)

def test_weighted_combination_is_correct(default_weights, conversation):
    scorers = [
        FakeScorer(RelevanceType.RECENCY, 0.8),
        FakeScorer(RelevanceType.SIMILARITY, 0.6),
        FakeScorer(RelevanceType.DENSITY, 0.4),
        FakeScorer(RelevanceType.DECISION, 1.0),
        FakeScorer(RelevanceType.REFERENCE, 0.2),
    ]
    scorer = RelevanceScorer(scorers, default_weights)
    result = scorer.score(conversation[0], conversation)
    # (0.30*0.8 + 0.30*0.6 + 0.15*0.4 + 0.15*1.0 + 0.10*0.2) / 1.0
    expected = (0.24 + 0.18 + 0.06 + 0.15 + 0.02) / 1.0
    assert result.combined_score == pytest.approx(expected)

def test_normalization_when_weights_dont_sum_to_one(conversation):
    weights = {
        RelevanceType.RECENCY: 0.5,
        RelevanceType.SIMILARITY: 0.5,
    }
    scorers = [
        FakeScorer(RelevanceType.RECENCY, 1.0),
        FakeScorer(RelevanceType.SIMILARITY, 1.0),
    ]
    scorer = RelevanceScorer(scorers, weights)
    result = scorer.score(conversation[0], conversation)
    # (0.5*1.0 + 0.5*1.0) / (0.5 + 0.5) = 1.0
    assert result.combined_score == pytest.approx(1.0)

def test_normalization_prevents_score_above_one(conversation):
    weights = {
        RelevanceType.RECENCY: 0.6,
        RelevanceType.SIMILARITY: 0.6,
    }
    scorers = [
        FakeScorer(RelevanceType.RECENCY, 1.0),
        FakeScorer(RelevanceType.SIMILARITY, 1.0),
    ]
    scorer = RelevanceScorer(scorers, weights)
    result = scorer.score(conversation[0], conversation)
    # (0.6*1.0 + 0.6*1.0) / (0.6 + 0.6) = 1.0 (normalized)
    assert result.combined_score == pytest.approx(1.0)

def test_scorer_with_no_weight_contributes_zero(default_weights, conversation):
    scorers = [
        FakeScorer(RelevanceType.RECENCY, 1.0),
        FakeScorer(RelevanceType.SIMILARITY, 1.0),
    ]
    # Only recency has a weight, similarity has a weight, others not in scorers list
    weights = {RelevanceType.RECENCY: 0.5}
    scorer = RelevanceScorer(scorers, weights)
    result = scorer.score(conversation[0], conversation)
    # (0.5*1.0 + 0.0*1.0) / 0.5 = 1.0
    assert result.combined_score == pytest.approx(1.0)

def test_result_contains_all_individual_scorer_results(all_scorers_max, default_weights, conversation):
    scorer = RelevanceScorer(all_scorers_max, default_weights)
    result = scorer.score(conversation[0], conversation)
    assert len(result.scorer_results) == 5
    scorer_names = {r.scorer_name for r in result.scorer_results}
    assert scorer_names == {
        RelevanceType.RECENCY,
        RelevanceType.SIMILARITY,
        RelevanceType.DENSITY,
        RelevanceType.DECISION,
        RelevanceType.REFERENCE,
    }

def test_result_contains_correct_message(all_scorers_max, default_weights, conversation):
    scorer = RelevanceScorer(all_scorers_max, default_weights)
    msg = conversation[0]
    result = scorer.score(msg, conversation)
    assert result.message == msg

def test_combined_score_between_zero_and_one(default_weights, conversation):
    scorers = [
        FakeScorer(RelevanceType.RECENCY, 0.5),
        FakeScorer(RelevanceType.SIMILARITY, 0.3),
        FakeScorer(RelevanceType.DENSITY, 0.7),
        FakeScorer(RelevanceType.DECISION, 0.2),
        FakeScorer(RelevanceType.REFERENCE, 0.9),
    ]
    scorer = RelevanceScorer(scorers, default_weights)
    result = scorer.score(conversation[0], conversation)
    assert 0.0 <= result.combined_score <= 1.0


def test_event_emitted_when_event_bus_provided(default_weights, conversation):
    emitted_events = []
    bus = EventBus()
    bus.subscribe(EventType.MESSAGE_RELEVANCE_SCORED, lambda event: emitted_events.append(event))

    scorers = [
        FakeScorer(RelevanceType.RECENCY, 0.8),
        FakeScorer(RelevanceType.DENSITY, 0.5),
    ]
    scorer = RelevanceScorer(scorers, default_weights, event_bus=bus)
    scorer.score(conversation[0], conversation)

    assert len(emitted_events) == 1
    event = emitted_events[0]
    assert event.event_type == EventType.MESSAGE_RELEVANCE_SCORED
    assert "combined_score" in event.data
    assert "scorer_results" in event.data
    assert "message" in event.data


def test_no_event_emitted_when_no_event_bus(default_weights, conversation):
    scorers = [FakeScorer(RelevanceType.RECENCY, 0.5)]
    scorer = RelevanceScorer(scorers, default_weights)
    # Should not raise - just silently skips emission
    result = scorer.score(conversation[0], conversation)
    assert result.combined_score >= 0.0