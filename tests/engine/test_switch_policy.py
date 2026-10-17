"""When a staged cut becomes the prompt: cold switches are free, warm ones must repay their
rewrite, and the model's stale weight has a hard bound."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from llm_gc.engine.switch_policy import CacheState, SwitchReason, decide_switch

SWITCH, WAIT = True, False


def _decide(*, saving=10_000, rewritten=20_000, stale=9_000, bound=20_000, share=1.0, cache=CacheState.WARM,
            requested=False, tool_round_open=False):
    """A warm burst by default: a 10k saving against 20k rewritten, 9k stale of a 20k bound."""
    return decide_switch(saving_tokens=saving, rewritten_tokens=rewritten, stale_tokens=stale,
                         stale_bound_tokens=bound, min_saving_share=share, cache=cache, requested=requested,
                         tool_round_open=tool_round_open)


class TestAWarmBurst:
    def test_a_cut_that_does_not_repay_its_rewrite_waits(self):
        """Saves 0.3x what it would make the provider write again: not worth breaking the cache."""
        decision = _decide(saving=6_000, rewritten=20_000, stale=9_000, bound=20_000)
        assert (decision.switch, decision.reason) == (WAIT, SwitchReason.WAITING_FOR_PAYBACK)

    def test_a_cut_that_repays_its_rewrite_switches(self):
        decision = _decide(saving=25_000, rewritten=20_000)
        assert (decision.switch, decision.reason) == (SWITCH, SwitchReason.REPAYS_REWRITE)

    def test_a_saving_exactly_the_share_of_the_rewrite_switches(self):
        assert _decide(saving=20_000, rewritten=20_000, share=1.0).switch is SWITCH
        assert _decide(saving=19_999, rewritten=20_000, share=1.0).switch is WAIT

    def test_stale_weight_at_the_bound_forces_a_switch_that_does_not_repay(self):
        decision = _decide(saving=6_000, rewritten=20_000, stale=21_000, bound=20_000)
        assert (decision.switch, decision.reason) == (SWITCH, SwitchReason.STALE_WEIGHT_AT_BOUND)

    def test_the_bound_is_inclusive(self):
        assert _decide(saving=6_000, stale=20_000, bound=20_000).switch is SWITCH
        assert _decide(saving=6_000, stale=19_999, bound=20_000).switch is WAIT


class TestAColdCache:
    def test_any_saving_switches_because_the_rewrite_is_free(self):
        decision = _decide(saving=1, rewritten=1_000_000, stale=0, cache=CacheState.COLD)
        assert (decision.switch, decision.reason) == (SWITCH, SwitchReason.CACHE_COLD)

    def test_the_same_cut_waits_when_warm(self):
        assert _decide(saving=1, rewritten=1_000_000, stale=0, cache=CacheState.WARM).switch is WAIT


class TestTheDeveloperAsks:
    def test_run_gc_now_switches_a_cut_that_would_wait(self):
        decision = _decide(saving=1, rewritten=1_000_000, stale=0, requested=True)
        assert (decision.switch, decision.reason) == (SWITCH, SwitchReason.REQUESTED)

    def test_even_then_a_cut_that_grows_the_prompt_is_not_switched(self):
        decision = _decide(saving=-5, requested=True)
        assert (decision.switch, decision.reason) == (WAIT, SwitchReason.NO_SAVING)


class TestTheReasonIsTheCheapestThatApplies:
    def test_cold_beats_repaying_and_overdue(self):
        assert _decide(saving=50_000, stale=99_000, cache=CacheState.COLD).reason is SwitchReason.CACHE_COLD

    def test_repaying_beats_overdue(self):
        assert _decide(saving=50_000, stale=99_000).reason is SwitchReason.REPAYS_REWRITE

    def test_requested_beats_cold(self):
        assert _decide(cache=CacheState.COLD, requested=True).reason is SwitchReason.REQUESTED


class TestNeverWaitsForever:
    def test_a_cut_that_grows_the_prompt_never_switches_whatever_else_holds(self):
        for saving in (0, -1, -50_000):
            decision = _decide(saving=saving, stale=99_000, cache=CacheState.COLD)
            assert (decision.switch, decision.reason) == (WAIT, SwitchReason.NO_SAVING)

    def test_a_zero_share_collects_as_soon_as_there_is_a_saving(self):
        assert _decide(saving=1, rewritten=1_000_000, share=0.0).switch is SWITCH

    def test_a_zero_bound_collects_as_soon_as_there_is_a_saving(self):
        assert _decide(saving=1, rewritten=1_000_000, stale=0, bound=0).switch is SWITCH


class TestBadInput:
    @pytest.mark.parametrize("field", ["rewritten", "stale", "bound", "share"])
    def test_a_negative_measure_is_a_programming_error(self, field):
        with pytest.raises(ValueError, match="cannot be negative"):
            _decide(**{field: -1})


_tokens = st.integers(min_value=0, max_value=10_000_000)
_savings = st.integers(min_value=-1_000_000, max_value=10_000_000)
_shares = st.floats(min_value=0.0, max_value=5.0, allow_nan=False)
_caches = st.sampled_from(CacheState)


class TestInvariants:
    @given(saving=_savings, rewritten=_tokens, stale=_tokens, bound=_tokens, share=_shares, cache=_caches,
           requested=st.booleans())
    def test_the_prompt_is_never_made_larger(self, saving, rewritten, stale, bound, share, cache, requested):
        decision = _decide(saving=saving, rewritten=rewritten, stale=stale, bound=bound, share=share, cache=cache,
                           requested=requested)
        assert not decision.switch or saving > 0

    @given(saving=st.integers(min_value=1, max_value=10_000_000), rewritten=_tokens, stale=_tokens, bound=_tokens,
           share=_shares)
    def test_a_cold_cache_with_a_saving_always_switches(self, saving, rewritten, stale, bound, share):
        assert _decide(saving=saving, rewritten=rewritten, stale=stale, bound=bound, share=share,
                       cache=CacheState.COLD).switch is SWITCH

    @given(saving=st.integers(min_value=1, max_value=10_000_000), rewritten=_tokens, bound=_tokens, share=_shares,
           cache=_caches, extra=_tokens)
    def test_stale_weight_at_the_bound_always_switches_a_cut_with_a_saving(self, saving, rewritten, bound, share, cache,
                                                                         extra):
        assert _decide(saving=saving, rewritten=rewritten, stale=bound + extra, bound=bound, share=share,
                       cache=cache).switch is SWITCH

    @given(saving=_savings, rewritten=_tokens, stale=_tokens, bound=_tokens, share=_shares)
    def test_a_cold_cache_only_ever_makes_switching_easier(self, saving, rewritten, stale, bound, share):
        warm = _decide(saving=saving, rewritten=rewritten, stale=stale, bound=bound, share=share,
                       cache=CacheState.WARM)
        cold = _decide(saving=saving, rewritten=rewritten, stale=stale, bound=bound, share=share,
                       cache=CacheState.COLD)
        assert cold.switch or not warm.switch

    @given(saving=_savings, rewritten=_tokens, stale=_tokens, bound=_tokens, share=_shares, cache=_caches,
           more=_tokens)
    def test_more_stale_weight_never_turns_a_switch_into_a_wait(self, saving, rewritten, stale, bound, share, cache,
                                                                 more):
        before = _decide(saving=saving, rewritten=rewritten, stale=stale, bound=bound, share=share, cache=cache)
        after = _decide(saving=saving, rewritten=rewritten, stale=stale + more, bound=bound, share=share, cache=cache)
        assert after.switch or not before.switch

    @given(saving=_savings, rewritten=_tokens, stale=_tokens, bound=_tokens, share=_shares, cache=_caches,
           more=_tokens)
    def test_a_larger_saving_never_turns_a_switch_into_a_wait(self, saving, rewritten, stale, bound, share, cache,
                                                               more):
        before = _decide(saving=saving, rewritten=rewritten, stale=stale, bound=bound, share=share, cache=cache)
        after = _decide(saving=saving + more, rewritten=rewritten, stale=stale, bound=bound, share=share, cache=cache)
        assert after.switch or not before.switch

    @given(saving=st.integers(min_value=1, max_value=10_000_000), rewritten=_tokens, stale=_tokens, bound=_tokens,
           share=_shares, cache=_caches)
    def test_a_cut_with_a_saving_waits_only_when_warm_unrepaid_and_under_the_bound(self, saving, rewritten, stale,
                                                                                  bound, share, cache):
        """The converse of every arm: if nothing was forced, each arm must really have failed."""
        decision = _decide(saving=saving, rewritten=rewritten, stale=stale, bound=bound, share=share, cache=cache)
        if not decision.switch:
            assert cache is CacheState.WARM
            assert saving < share * rewritten
            assert stale < bound


class TestAnOpenToolRound:
    """A reply whose tool calls are being answered must go back with its thinking, which a new front invalidates."""

    @pytest.mark.parametrize("would_switch", [{"requested": True}, {"cache": CacheState.COLD}, {"saving": 25_000},
                                              {"saving": 6_000, "stale": 21_000}],
                             ids=["requested", "cold", "repays", "at_bound"])
    def test_it_holds_back_every_switch_even_a_requested_one(self, would_switch):
        assert _decide(**would_switch).switch is SWITCH
        decision = _decide(tool_round_open=True, **would_switch)
        assert (decision.switch, decision.reason) == (WAIT, SwitchReason.TOOL_ROUND_OPEN)
