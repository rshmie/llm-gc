"""How many of a model's tokens make one of LLM-GC's, measured per model. Application code.

LLM-GC counts with its own tokenizer, and each model counts with its own, so the two
counts of the same prompt differ by a rate that is measured, never assumed. It converts
what the provider reports into LLM-GC's count: the thinking in a reply, the size of an
image, and a conversation's window until it has measured its own rate.

Only a request LLM-GC counts exactly measures it: no thinking, no images, no documents.
Thinking is counted by estimate, and an image or a document counts as a placeholder in
LLM-GC's count but at its real size in the provider's, so either would skew the rate.

The rate is all those requests' provider tokens over all their LLM-GC tokens, so each
request weighs by its size. A small side request, such as a title the client asks for,
has a different mix of text, so it barely moves a rate measured on whole conversations.

Used only from the event loop, so it needs no lock.
"""

from collections import OrderedDict

# Maximum number of models whose rate is kept. Model names come from clients, so this is bounded.
MODELS_REMEMBERED = 64


class TokenRates:
    """Each model's rate: provider tokens per LLM-GC token, over the requests that measured it."""

    def __init__(self) -> None:
        # Per model: (provider tokens, LLM-GC tokens), summed over the requests that measured it.
        self._measured: OrderedDict[str | None, tuple[int, int]] = OrderedDict()

    def observe(self, model: str | None, *, provider_tokens: int, forwarded_tokens: int) -> None:
        """Add one request to its model's rate. Only for a request LLM-GC counts exactly."""
        if provider_tokens <= 0 or forwarded_tokens <= 0:
            raise ValueError(f"a rate is measured on two positive counts, not {provider_tokens} and {forwarded_tokens}")
        provider, forwarded = self._measured.get(model, (0, 0))
        self._measured[model] = (provider + provider_tokens, forwarded + forwarded_tokens)
        self._measured.move_to_end(model)
        while len(self._measured) > MODELS_REMEMBERED:
            self._measured.popitem(last=False)

    def rate(self, model: str | None) -> float | None:
        """Return the model's token rate, or None if no request has measured it."""
        measured = self._measured.get(model)
        return measured[0] / measured[1] if measured is not None else None
