"""What a request costs at the provider's list prices, from the usage it reported. Application code.

An estimate, said to be one wherever it is shown: list prices, per million tokens, as of
2026-10; a discount, a batch, fast mode or a gateway's own pricing is not known here. A model
not in the table is not priced, never guessed at. Cache writes are priced by how long they
last, as the reply reports them: 1.25 times the input price for five minutes, twice it for
an hour.
"""

from dataclasses import dataclass

from llm_gc.llm.metering import ModelUsage
from llm_gc.proxy.traffic import ProviderUsage

MILLION = 1_000_000
CACHE_WRITE_5M = 1.25
CACHE_WRITE_1H = 2.0


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """Dollars per million tokens. A model priced higher above a prompt size names that size and those prices."""

    input: float
    output: float
    cache_read: float
    long_prompt: int | None = None            # prompt tokens above which `long` applies
    long: "ModelPrice | None" = None


# Anthropic's list prices, 2026-10. Cache reads are a tenth of input except where noted.
PRICES: dict[str, ModelPrice] = {
    "claude-fable-5-1": ModelPrice(10.0, 50.0, 0.25),
    "claude-mythos-5-1": ModelPrice(10.0, 50.0, 0.25),
    "claude-fable-5": ModelPrice(10.0, 50.0, 1.0),
    "claude-opus-5-5": ModelPrice(4.0, 20.0, 0.20),
    "claude-opus-5": ModelPrice(5.0, 25.0, 0.50),
    "claude-opus-4-8": ModelPrice(5.0, 25.0, 0.50),
    "claude-opus-4-7": ModelPrice(5.0, 25.0, 0.50),
    "claude-opus-4-6": ModelPrice(5.0, 25.0, 0.50),
    "claude-sonnet-5-5": ModelPrice(2.0, 10.0, 0.20),
    "claude-sonnet-5": ModelPrice(2.0, 10.0, 0.20),
    "claude-sonnet-4-6": ModelPrice(3.0, 15.0, 0.30),
    "claude-haiku-5-5": ModelPrice(0.10, 0.50, 0.01, long_prompt=100_000, long=ModelPrice(0.50, 2.50, 0.05)),
    "claude-haiku-4-5": ModelPrice(1.0, 5.0, 0.10),
}
# Suffixes a client adds to a model's name that do not change its price: Claude Code's
# "[1m]" for the long window.
_NAME_SUFFIXES = ("[1m]",)


def price_of(model: str | None) -> ModelPrice | None:
    """The model's list price, or None when it is not one this table knows."""
    if model is None:
        return None
    for suffix in _NAME_SUFFIXES:
        model = model.removesuffix(suffix)
    return PRICES.get(model)


def request_cost(usage: ProviderUsage, price: ModelPrice) -> float:
    """Dollars for one reply's usage, at the price for its prompt's size."""
    tier = _for_prompt(usage, price)
    return ((usage.input_tokens * tier.input + usage.output_tokens * tier.output
             + usage.cache_read_input_tokens * tier.cache_read) / MILLION + cache_write_cost(usage, price))


def cache_write_cost(usage: ProviderUsage, price: ModelPrice) -> float:
    """Dollars for the tokens a reply wrote to the cache, priced by how long they last."""
    tier = _for_prompt(usage, price)
    hour = min(usage.cache_creation_1h_input_tokens, usage.cache_creation_input_tokens)
    minutes = usage.cache_creation_input_tokens - hour
    return (minutes * CACHE_WRITE_5M + hour * CACHE_WRITE_1H) * tier.input / MILLION


def _for_prompt(usage: ProviderUsage, price: ModelPrice) -> ModelPrice:
    prompt = usage.input_tokens + usage.cache_read_input_tokens + usage.cache_creation_input_tokens
    if price.long is not None and price.long_prompt is not None and prompt > price.long_prompt:
        return price.long
    return price


def calls_cost(usage: ModelUsage, price: ModelPrice) -> float:
    """Dollars for LLM-GC's own model calls, which use no cache."""
    return (usage.input_tokens * price.input + usage.output_tokens * price.output) / MILLION
