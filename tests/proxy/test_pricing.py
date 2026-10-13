"""What a request costs at list prices, from the usage the provider reported."""

import pytest

from llm_gc.llm import ModelUsage
from llm_gc.proxy.pricing import cache_write_cost, calls_cost, price_of, request_cost
from llm_gc.proxy.traffic import ProviderUsage

OPUS = price_of("claude-opus-5-5")


def test_a_request_is_priced_by_each_kind_of_token():
    """Opus 5.5: $4 input, $20 output, $0.20 cache read, cache writes 1.25 times input."""
    usage = ProviderUsage(replies=1, input_tokens=1_000, cache_read_input_tokens=80_000,
                          cache_creation_input_tokens=2_000, output_tokens=1_000)

    assert request_cost(usage, OPUS) == pytest.approx(0.004 + 0.016 + 0.010 + 0.020)


def test_a_cache_write_meant_to_last_an_hour_costs_twice_the_input_price():
    usage = ProviderUsage(replies=1, cache_creation_input_tokens=26_000, cache_creation_1h_input_tokens=6_000)

    assert cache_write_cost(usage, OPUS) == pytest.approx((20_000 * 1.25 + 6_000 * 2.0) * 4 / 1_000_000)


def test_a_model_the_list_does_not_know_is_not_priced():
    assert price_of("gateway-renamed-sonnet") is None and price_of(None) is None


def test_the_long_window_suffix_does_not_change_the_price():
    assert price_of("claude-opus-5-5[1m]") == OPUS


def test_a_model_priced_higher_for_long_prompts_is_priced_by_the_prompts_size():
    haiku = price_of("claude-haiku-5-5")
    short = ProviderUsage(replies=1, cache_read_input_tokens=50_000)
    long = ProviderUsage(replies=1, cache_read_input_tokens=150_000)

    assert request_cost(short, haiku) == pytest.approx(50_000 * 0.01 / 1_000_000)
    assert request_cost(long, haiku) == pytest.approx(150_000 * 0.05 / 1_000_000)


def test_llm_gcs_own_calls_are_priced_without_a_cache():
    assert calls_cost(ModelUsage(calls=3, input_tokens=90_000, output_tokens=8_000),
                      price_of("claude-sonnet-5")) == pytest.approx(0.18 + 0.08)


def test_the_providers_one_hour_writes_are_read_from_its_usage():
    usage = ProviderUsage.from_reported({"input_tokens": 3, "cache_creation_input_tokens": 900,
                                         "cache_creation": {"ephemeral_5m_input_tokens": 600,
                                                            "ephemeral_1h_input_tokens": 300}})
    assert (usage.cache_creation_input_tokens, usage.cache_creation_1h_input_tokens) == (900, 300)
