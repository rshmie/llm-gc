"""The planner: one level per span, by score, under the budget, folding only as a last resort."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from llm_gc.engine.generations.planning import Bands, LevelOptions, plan_levels
from llm_gc.exceptions import ConfigError
from llm_gc.models.generation_snapshot import DetailLevel

KEPT, COMPACTED, ARCHIVED, OUT = DetailLevel.KEPT, DetailLevel.COMPACTED, DetailLevel.ARCHIVED, DetailLevel.OUT_OF_VIEW
BANDS = Bands(keep=0.7, archive=0.3, margin=0.05)
ROOMY = 10**9
_RANK = {OUT: 0, ARCHIVED: 1, COMPACTED: 2, KEPT: 3}


def _span(score: float, current: DetailLevel | None = None, kept: int = 1000, compacted: int | None = 300,
          archived: int = 40) -> LevelOptions:
    return LevelOptions(score=score, current=current, kept_tokens=kept, compacted_tokens=compacted,
                        archived_tokens=archived)


def _levels(spans, budget=ROOMY, fold_line=10):
    return list(plan_levels(spans, BANDS, budget, fold_line).levels)


class TestTheScoreDecidesTheLevel:
    @pytest.mark.parametrize("score,level", [(0.9, KEPT), (0.7, KEPT), (0.5, COMPACTED), (0.3, COMPACTED), (0.1, ARCHIVED)])
    def test_a_new_span_takes_the_band_its_score_falls_in(self, score, level):
        assert _levels([_span(score)]) == [level]

    @pytest.mark.parametrize("current,score,level", [
        (COMPACTED, 0.72, COMPACTED), (COMPACTED, 0.76, KEPT),     # up only past the margin
        (KEPT, 0.68, KEPT), (KEPT, 0.64, COMPACTED),               # down only past the margin
        (ARCHIVED, 0.32, ARCHIVED), (ARCHIVED, 0.36, COMPACTED),
        (ARCHIVED, 0.9, KEPT),                                     # a span the conversation returns to
    ])
    def test_a_span_moves_only_once_its_score_clears_a_line_by_the_margin(self, current, score, level):
        assert _levels([_span(score, current)]) == [level]

    @pytest.mark.parametrize("score,level", [(0.5, COMPACTED), (0.32, ARCHIVED), (0.8, KEPT)])
    def test_a_folded_span_returns_to_what_its_score_earns_moving_up(self, score, level):
        """Folding answers the budget, not the score: with room again, the score decides."""
        assert _levels([_span(score, OUT)]) == [level]


class TestALevelIsUsedOnlyIfItIsCheaper:
    def test_a_span_without_a_summary_is_kept_rather_than_archived(self):
        """Losing detail is never the result of a failed summary; the budget lowers it if it must."""
        assert _levels([_span(0.5, compacted=None)]) == [KEPT]

    def test_a_span_shorter_than_its_index_line_stays_kept(self):
        assert _levels([_span(0.0, kept=12, compacted=None, archived=30)]) == [KEPT]

    def test_a_summary_longer_than_the_turns_is_not_used(self):
        assert _levels([_span(0.5, kept=200, compacted=260)]) == [KEPT]


class TestTheBudgetCapsTheTotal:
    def test_the_coldest_span_is_lowered_first(self):
        spans = [_span(0.9), _span(0.8)]                  # both kept: 2000 tokens
        assert _levels(spans, budget=1400) == [KEPT, COMPACTED]

    def test_on_equal_scores_the_older_span_is_lowered(self):
        assert _levels([_span(0.8), _span(0.8)], budget=1400) == [COMPACTED, KEPT]

    def test_the_coldest_goes_all_the_way_down_before_a_warmer_span_loses_detail(self):
        spans = [_span(0.9), _span(0.8)]
        assert _levels(spans, budget=1050) == [KEPT, ARCHIVED]

    def test_reaching_the_budget_never_raises_a_level(self):
        assert _levels([_span(0.1)], budget=ROOMY) == [ARCHIVED]

    def test_the_oldest_are_folded_only_when_nothing_can_go_lower(self):
        spans = [_span(0.9), _span(0.1), _span(0.1)]      # 1000 + 40 + 40 at their bands
        plan = plan_levels(spans, BANDS, budget=100, fold_line_tokens=10)
        assert list(plan.levels) == [OUT, ARCHIVED, ARCHIVED]
        assert plan.folded == 1 and plan.tokens == 10 + 40 + 40
        assert not plan.over_budget

    def test_a_budget_too_small_even_for_the_fold_line_is_reported(self):
        plan = plan_levels([_span(0.1)], BANDS, budget=5, fold_line_tokens=10)
        assert plan.levels == (OUT,) and plan.over_budget


class TestBandsAreValidated:
    @pytest.mark.parametrize("keep,archive,margin", [(0.3, 0.3, 0.0), (0.7, 0.3, -0.01), (0.7, 0.3, 0.2)],
                             ids=["no-gap", "negative-margin", "margin-swallows-the-band"])
    def test_bands_that_could_not_work_are_refused(self, keep, archive, margin):
        with pytest.raises(ConfigError):
            Bands(keep=keep, archive=archive, margin=margin)


# Properties over generated spans: the rules the locked design states, checked on any input.

_level = st.sampled_from([None, KEPT, COMPACTED, ARCHIVED, OUT])
_spans = st.lists(st.builds(LevelOptions, score=st.floats(0, 1), current=_level,
                            kept_tokens=st.integers(1, 3000), compacted_tokens=st.none() | st.integers(1, 3000),
                            archived_tokens=st.integers(1, 300)), max_size=12)


def _at_floor(span: LevelOptions, level: DetailLevel) -> bool:
    poorer = [other for other in (COMPACTED, ARCHIVED) if _RANK[other] < _RANK[level]]
    return level is OUT or not any(span.worthwhile(other) and span.cost(other) < span.cost(level) for other in poorer)


class TestPlannerProperties:
    @given(spans=_spans, budget=st.integers(0, 20_000), fold_line=st.integers(1, 30))
    def test_every_span_gets_one_level_and_the_total_is_what_the_plan_says(self, spans, budget, fold_line):
        plan = plan_levels(spans, BANDS, budget, fold_line)
        assert len(plan.levels) == len(spans)
        expected = sum(span.cost(level) for span, level in zip(spans, plan.levels)) + (fold_line if plan.folded else 0)
        assert plan.tokens == expected
        assert plan.tokens <= budget or (plan.over_budget and plan.folded == len(spans))

    @given(spans=_spans, budget=st.integers(0, 20_000))
    def test_every_level_shown_is_one_worth_showing(self, spans, budget):
        plan = plan_levels(spans, BANDS, budget, 10)
        assert all(level is OUT or span.worthwhile(level) for span, level in zip(spans, plan.levels))

    @given(spans=_spans, budget=st.integers(0, 20_000))
    def test_folding_takes_the_oldest_spans_together(self, spans, budget):
        plan = plan_levels(spans, BANDS, budget, 10)
        assert all(level is OUT for level in plan.levels[: plan.folded])
        assert not any(level is OUT for level in plan.levels[plan.folded:])

    @given(spans=_spans, budget=st.integers(0, 20_000))
    def test_the_budget_only_ever_lowers_and_only_when_needed(self, spans, budget):
        unconstrained = plan_levels(spans, BANDS, ROOMY, 10)
        plan = plan_levels(spans, BANDS, budget, 10)
        assert all(_RANK[level] <= _RANK[free] for level, free in zip(plan.levels, unconstrained.levels))
        if unconstrained.tokens <= budget:
            assert plan.levels == unconstrained.levels and not plan.lowered and not plan.folded

    @given(spans=_spans, budget=st.integers(0, 20_000))
    def test_a_span_is_lowered_only_once_every_colder_span_is_at_its_floor(self, spans, budget):
        plan = plan_levels(spans, BANDS, budget, 10)
        for index in plan.lowered:
            colder = [j for j, span in enumerate(spans) if (span.score, j) < (spans[index].score, index)]
            assert all(_at_floor(spans[j], plan.levels[j]) for j in colder)

    @given(spans=_spans)
    def test_replanning_unchanged_scores_changes_nothing(self, spans):
        """No flapping: a span that settled stays settled until its score moves."""
        first = plan_levels(spans, BANDS, ROOMY, 10)
        again = [LevelOptions(score=s.score, current=level, kept_tokens=s.kept_tokens,
                              compacted_tokens=s.compacted_tokens, archived_tokens=s.archived_tokens)
                 for s, level in zip(spans, first.levels)]
        assert plan_levels(again, BANDS, ROOMY, 10).levels == first.levels
