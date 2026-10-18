"""Each model's rate between its token count and LLM-GC's, weighted by the size of the requests that measured it."""

import pytest

from llm_gc.proxy.token_rates import TokenRates

MODEL = "claude-opus-5-5"


class TestTheModelsRate:
    def test_each_request_weighs_by_its_size(self):
        """Live run: an 843-token side request replaced the rate measured on the conversation's
        55k-token first request, and every reply after it had its thinking converted at 1.55, not 1.48."""
        rates = TokenRates()
        rates.observe(MODEL, provider_tokens=81_326, forwarded_tokens=54_945)
        rates.observe(MODEL, provider_tokens=1_304, forwarded_tokens=843)

        assert rates.rate(MODEL) == pytest.approx(1.4812, abs=1e-4)

    def test_each_model_has_its_own(self):
        rates = TokenRates()
        rates.observe(MODEL, provider_tokens=1_500, forwarded_tokens=1_000)
        rates.observe("claude-sonnet-5-5", provider_tokens=1_200, forwarded_tokens=1_000)

        assert rates.rate(MODEL) == 1.5
        assert rates.rate("claude-sonnet-5-5") == 1.2
        assert rates.rate("never seen") is None

    @pytest.mark.parametrize("provider_tokens, forwarded_tokens", [(0, 1_000), (1_000, 0)])
    def test_a_count_of_zero_is_a_caller_bug(self, provider_tokens, forwarded_tokens):
        with pytest.raises(ValueError):
            TokenRates().observe(MODEL, provider_tokens=provider_tokens, forwarded_tokens=forwarded_tokens)
