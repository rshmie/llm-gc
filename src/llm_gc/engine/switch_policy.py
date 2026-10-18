"""When a prepared cut becomes the prompt: the switch policy.

A pass prepares a cut without changing what the model is sent; switching to it is what
changes the prompt, and from the front onwards that misses the provider's cache. This
module decides, as a request is built, whether to switch now. It is a pure function over
numbers the caller measured, so it knows nothing about providers, prices or clocks: the
caller turns a vendor's usage and the time since its last request into a `CacheState`.

The rationale, the zones and the rejected alternatives are in
doc/design/generational-memory/when-to-collect.md.
"""

from dataclasses import dataclass
from enum import Enum


class CacheState(Enum):
    """Whether the provider's cache would serve this conversation's next request."""

    WARM = "warm"
    # The request misses the cache whatever happens: the conversation does not use it, or
    # has been idle for longer than its entries live. A switch costs nothing then.
    COLD = "cold"


@dataclass(frozen=True, slots=True)
class CacheReading:
    """A conversation's cache state, with what it was read from, for showing to a person."""

    state: CacheState
    uses_cache: bool | None = None  # whether any reply used the cache; None before one was answered
    idle_s: float | None = None  # since its newest answered request began
    lifetime_s: float | None = None  # how long that request asked the provider to keep its entry


class SwitchReason(Enum):
    """Why a cut was switched in, or why it was not. Stable: the dashboard and events carry it."""

    # Switching would not shrink the prompt, so the cut waits whatever else holds.
    NO_SAVING = "no_saving"
    REQUESTED = "requested"
    CACHE_COLD = "cache_cold"
    REPAYS_REWRITE = "repays_rewrite"
    # Forced for the model's health, not for the bill: the one automatic switch that can cost more.
    STALE_WEIGHT_AT_BOUND = "stale_weight_at_bound"
    WAITING_FOR_PAYBACK = "waiting_for_payback"
    # The request answers the previous reply's tool calls. That reply must go back with its thinking,
    # which a new front would invalidate, so nothing is switched in until the round ends.
    TOOL_ROUND_OPEN = "tool_round_open"


@dataclass(frozen=True, slots=True)
class SwitchDecision:
    switch: bool
    reason: SwitchReason


def decide_switch(
    *,
    saving_tokens: int,
    rewritten_tokens: int,
    stale_tokens: int,
    stale_bound_tokens: int,
    min_saving_share: float,
    cache: CacheState,
    requested: bool = False,
    tool_round_open: bool = False,
) -> SwitchDecision:
    """Whether to switch a staged cut into the request being built.

    - `saving_tokens`: what the switch removes from the prompt, on the real summaries,
      including the model's earlier thinking it stops sending. Negative when the cut
      would grow it.
    - `rewritten_tokens`: what the provider must cache again after the switch: the messages,
      and any part of the system prompt and tools the client's cache markers do not cover.
      The rest of the system prompt and tools is read either way and is left out.
    - `stale_tokens`: what the model carries that the plan has condensed, thinking
      included: the measure of its health debt. `stale_bound_tokens` is the most it may carry.
    - `min_saving_share`: when warm, how many times `rewritten_tokens` the saving must be.
    - `requested`: the developer asked for a collection now.
    - `tool_round_open`: the request continues a tool round. It holds every switch back,
      a requested one too: the provider refuses a reply's thinking once the history
      before it has changed, and a reply whose tool calls are being answered must keep it.

    The checks run in order and the first that holds decides, so the reason is the cheapest
    one that applies: a switch labelled `STALE_WEIGHT_AT_BOUND` was warm, did not repay
    its rewrite and was forced. A zero `min_saving_share` or `stale_bound_tokens` switches
    as soon as there is a saving.
    """
    for name, value in (
        ("rewritten_tokens", rewritten_tokens),
        ("stale_tokens", stale_tokens),
        ("stale_bound_tokens", stale_bound_tokens),
        ("min_saving_share", min_saving_share),
    ):
        if value < 0:
            raise ValueError(f"{name} cannot be negative, got {value}")

    if tool_round_open:
        return SwitchDecision(False, SwitchReason.TOOL_ROUND_OPEN)
    if saving_tokens <= 0:
        return SwitchDecision(False, SwitchReason.NO_SAVING)
    if requested:
        return SwitchDecision(True, SwitchReason.REQUESTED)
    if cache is CacheState.COLD:
        return SwitchDecision(True, SwitchReason.CACHE_COLD)
    if saving_tokens >= min_saving_share * rewritten_tokens:
        return SwitchDecision(True, SwitchReason.REPAYS_REWRITE)
    if stale_tokens >= stale_bound_tokens:
        return SwitchDecision(True, SwitchReason.STALE_WEIGHT_AT_BOUND)
    return SwitchDecision(False, SwitchReason.WAITING_FOR_PAYBACK)
