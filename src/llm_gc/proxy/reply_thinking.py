"""Remembers how much hidden thinking each reply contained, so its turn can be sized later.

When a model thinks before answering, the client sends that thinking back with every
later request, and the provider bills it as input. LLM-GC cannot count it from text,
because the thinking is encrypted. So the size is estimated once, from the reply's usage,
and looked up by key when the turn comes back.

The provider's output count is converted to LLM-GC's count with a rate measured only on
requests that carried no thinking. If requests with thinking were used, the estimate would
feed its own rate, and with a model that does not bill old thinking both would grow
without limit.

This module is provider-neutral: keys are opaque strings made by the provider's
translation module. It is used only from the event loop, so it needs no lock.
"""

from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass

# Maximum number of replies remembered. When full, the least recently used entry is
# evicted; its turn then counts as having no thinking. One entry is about 300 bytes.
REPLIES_REMEMBERED = 20_000
# Maximum number of models whose rate is kept. Model names come from clients, so this is bounded.
MODELS_REMEMBERED = 64


@dataclass(frozen=True, slots=True)
class _Measured:
    output_tokens: int    # the reply's total output as the provider counted it, thinking included
    token_rate: float     # provider tokens per LLM-GC token for its model, when it was measured
    model: str | None     # the model that wrote the reply


class ReplyThinking:
    """An LRU cache from a reply's thinking key to the reply's measured output."""

    def __init__(self, capacity: int = REPLIES_REMEMBERED) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be at least 1, got {capacity}")
        self._replies: OrderedDict[str, _Measured] = OrderedDict()
        self._rates: OrderedDict[str | None, float] = OrderedDict()
        self._capacity = capacity

    def observe_rate(self, model: str | None, token_rate: float) -> None:
        """Record a model's latest token rate, measured on a request that carried no thinking."""
        self._rates[model] = token_rate
        self._rates.move_to_end(model)
        while len(self._rates) > MODELS_REMEMBERED:
            self._rates.popitem(last=False)

    def rate(self, model: str | None) -> float | None:
        """Return the model's latest token rate, or None if none has been measured."""
        return self._rates.get(model)

    def remember(self, key: str, *, output_tokens: int, model: str | None) -> None:
        """Store a finished reply's output under the key of its first thinking block.

        Skipped if no rate is known for the model yet, because the output cannot be converted.
        """
        rate = self._rates.get(model)
        if rate is None:
            return
        self._replies[key] = _Measured(output_tokens, rate, model)
        self._replies.move_to_end(key)
        while len(self._replies) > self._capacity:
            self._replies.popitem(last=False)

    def tokens(self, keys: Iterable[str], visible_tokens: int, model: str | None) -> int:
        """Estimate the thinking tokens a turn carries, in LLM-GC's count.

        The estimate is the reply's output converted to LLM-GC's count, minus the turn's
        visible text (`visible_tokens`). Returns 0 if the reply was never measured, or if
        `model` is not the model that wrote it: another model may not be able to read the
        thinking, and the provider does not bill thinking it drops.
        """
        found = [measured for key in keys if (measured := self._lookup(key)) is not None]
        output = sum(measured.output_tokens / measured.token_rate for measured in found if measured.model == model)
        return max(0, round(output) - visible_tokens) if output else 0

    def __len__(self) -> int:
        return len(self._replies)

    def _lookup(self, key: str) -> _Measured | None:
        measured = self._replies.get(key)
        if measured is not None:
            # Mark as recently used: an active conversation sends its replies back on every request.
            self._replies.move_to_end(key)
        return measured
