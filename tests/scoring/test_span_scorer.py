"""Span scoring: liveness, relevance to now, and whether every detail level stays reachable."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from llm_gc.config import GCConfig
from llm_gc.config.constants import DEFAULT_LEVEL_MARGIN
from llm_gc.exceptions import ConfigError
from llm_gc.models import Message
from llm_gc.scoring import DecisionScorer, DensityScorer, LexicalRelevance, RelevanceType, SpanScorer


def _turn(index: int, text: str, role: str = "user") -> Message:
    return Message(role=role, content=text, token_count=len(text.split()), turn_index=index)


class _FixedRelevance:
    """Relevance supplied by the test, so liveness can be checked without word matching."""

    def __init__(self, *values: float) -> None:
        self.values = list(values)

    def score(self, spans, now):
        return self.values[: len(spans)]


def _scorer(*relevance: float) -> SpanScorer:
    return SpanScorer([DecisionScorer(), DensityScorer()], relevance=_FixedRelevance(*relevance))


class TestLexicalRelevance:
    relevance = LexicalRelevance()

    def test_a_span_that_discusses_the_subject_outranks_one_that_names_it(self):
        discusses = "The hold lasts seven minutes. When a hold expires the seat is released; a hold renews once."
        names = "Payments go through Stripe, and a hold on the card is placed at checkout."
        assert self.relevance.score([discusses, names], "how long does a hold last")[0] > \
            self.relevance.score([discusses, names], "how long does a hold last")[1]

    def test_a_word_every_span_uses_counts_for_less_than_a_rare_one(self):
        """"ticket" is in every span of a ticketing conversation; "refund" is in one."""
        spans = ["ticket pricing and fees", "ticket refund policy", "ticket scanning at the gate"]
        pricing, refund, _ = self.relevance.score(spans, "ticket refund")
        # Unweighted, sharing one of two words would score exactly half of sharing both.
        assert refund > 2 * pricing, "sharing only the common word is a weak match"

    def test_a_passing_match_stays_weak(self):
        """One incidental word out of several asked must not read as relevant."""
        scores = self.relevance.score(["we discussed the rollout of the mobile app", "seat maps and pricing"],
                                      "phase the rollout, the biggest risks, and who signs off")
        assert max(scores) < 0.2

    def test_nothing_to_match_scores_zero(self):
        assert self.relevance.score(["the database is postgres"], "ok, thanks!") == [0.0]
        assert self.relevance.score([], "the database") == []

    @given(spans=st.lists(st.text(alphabet="abc de", max_size=40), max_size=6),
           now=st.text(alphabet="abc de", max_size=40))
    def test_scores_stay_between_zero_and_one(self, spans, now):
        assert all(0.0 <= value <= 1.0 for value in self.relevance.score(spans, now))


class TestLiveness:
    def test_a_span_is_as_live_as_the_larger_of_recency_and_relevance(self):
        """Not their sum: a summed recency would cap what an old span could ever score."""
        old_but_discussed, recent_but_unrelated = _scorer(0.9, 0.0).score(
            [[_turn(0, "first")], [_turn(199, "second")]], now=[], newest_turn=200)

        assert old_but_discussed.liveness == pytest.approx(0.9)
        assert old_but_discussed.parts[RelevanceType.RECENCY] < 0.01
        assert recent_but_unrelated.liveness == pytest.approx(recent_but_unrelated.parts[RelevanceType.RECENCY])

    def test_recency_is_measured_from_the_spans_last_turn(self):
        [score] = _scorer(0.0).score([[_turn(3, "a"), _turn(4, "b"), _turn(9, "c")]], now=[], newest_turn=9)
        assert score.parts[RelevanceType.RECENCY] == pytest.approx(1.0)

    def test_importance_reads_the_whole_span(self):
        """A decision stated in the answer, not the question, still counts."""
        [score] = _scorer(0.0).score([[_turn(0, "Which database?"),
                                       _turn(1, "We decided to go with postgres, settled on it.", "assistant")]],
                                     now=[], newest_turn=1)
        assert score.parts[RelevanceType.DECISION] > 0


class TestEveryLevelStaysReachable:
    """The defect this guards against: with recency weighted on its own, no turn old
    enough to be collected could ever score into the keep band."""

    margin = DEFAULT_LEVEL_MARGIN
    config = GCConfig()

    def test_an_old_span_can_reach_the_top_band(self):
        scorer = SpanScorer([DecisionScorer(), DensityScorer()])
        assert scorer.best_score_however_old >= self.config.keep_threshold + self.margin

    def test_an_old_span_the_conversation_returns_to_scores_into_the_top_band(self):
        discussed = [_turn(0, "We decided to go with Postgres 16 on AWS; the plan is RDS in eu-west-1."),
                     _turn(1, "Settled on Postgres 16, RDS, eu-west-1, 3 replicas, 30-day PITR.", "assistant")]
        [score] = _scorer(1.0).score([discussed], now=[], newest_turn=200)
        assert score.combined >= self.config.keep_threshold + self.margin

    def test_an_old_span_nobody_mentions_falls_below_the_bottom_band(self):
        [score] = _scorer(0.0).score([[_turn(0, "sure, sounds good to me")]], now=[], newest_turn=200)
        assert score.combined < self.config.archive_threshold - self.margin

    @given(relevance=st.floats(0, 1), ages=st.integers(0, 500))
    def test_combined_scores_stay_on_the_band_scale(self, relevance, ages):
        [score] = _scorer(relevance).score([[_turn(0, "We decided to use Redis 7 for the cache.")]],
                                           now=[], newest_turn=ages)
        assert 0.0 <= score.combined <= 1.0


class TestASwappedRelevanceIsHeldToItsContract:
    def test_a_missing_score_is_an_error_not_a_dropped_span(self):
        with pytest.raises(ValueError):
            _scorer(0.5).score([[_turn(0, "first")], [_turn(1, "second")]], now=[], newest_turn=1)


class TestConfiguration:
    def test_a_scorer_without_a_weight_is_refused(self):
        with pytest.raises(ConfigError, match="needs a weight"):
            SpanScorer([DecisionScorer()], importance_weights={})

    def test_a_weight_without_a_scorer_is_refused(self):
        with pytest.raises(ConfigError, match="needs a weight"):
            SpanScorer([], importance_weights={RelevanceType.DECISION: 0.2})

    def test_weights_must_add_up_to_something(self):
        with pytest.raises(ConfigError, match="more than zero"):
            SpanScorer([], liveness_weight=0.0, importance_weights={})
