"""How many tokens each image or document a client sends costs its model. Application code.

LLM-GC's own count reads an attachment as a placeholder, but the provider bills it at its
real size: a screenshot costs a recent model a few thousand tokens. So each attachment is
counted once with the provider's token-counting endpoint, with LLM-GC's own credential,
off the request path, and remembered by a key made from its bytes. The endpoint is free,
and its rate limit is apart from the one the client's requests share.

A size depends on the model, so it is kept per model. Until the count comes back, and when
none can - an attachment given by URL or file, an upstream without the endpoint, no
credential of LLM-GC's own - the attachment is unsized and counts as its placeholder.

This module is provider-neutral: keys are opaque strings and blocks are passed through
to the counter as the client sent them. It is used only from the event loop, so it needs
no lock.
"""

import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

# Counts the input tokens of one user message made of these content blocks, on a model.
AttachmentCounter = Callable[[str, list[dict[str, Any]]], Awaitable[int | None]]

# Sizes remembered, failures included. When full, the least recently used goes first, and an
# attachment sent again after that is counted again. One entry is about 200 bytes.
SIZES_REMEMBERED = 4_096
# Counts asked at once. Beyond it, an attachment is asked about on a later request.
COUNTS_IN_FLIGHT = 8
# Counted alone and with each attachment: what is left after subtracting it is the attachment's
# own size, without the tokens the provider adds to every message.
_BASELINE_BLOCK: dict[str, Any] = {"type": "text", "text": "."}


class AttachmentSizes:
    """An LRU cache from (model, attachment key) to the attachment's size in the model's tokens."""

    def __init__(
        self,
        counter: AttachmentCounter | None,
        schedule: Callable[[Awaitable[None]], None],
        capacity: int = SIZES_REMEMBERED,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be at least 1, got {capacity}")
        self._counter = counter
        self._schedule = schedule
        self._capacity = capacity
        self._sizes: OrderedDict[tuple[str, str], int | None] = OrderedDict()  # None: asked, and none came back
        self._baselines: dict[str, int] = {}
        self._asking: set[tuple[str, str]] = set()

    def size(self, model: str | None, key: str, block: dict[str, Any]) -> int | None:
        """The attachment's size in the model's tokens, or None while it is unsized.

        Never waits: an attachment not seen before on this model is counted in the background,
        and sized on a later request.
        """
        if model is None or self._counter is None:
            return None
        entry = (model, key)
        if entry in self._sizes:
            # An active conversation sends its attachments on every request, so they stay.
            self._sizes.move_to_end(entry)
            return self._sizes[entry]
        if entry not in self._asking and len(self._asking) < COUNTS_IN_FLIGHT:
            self._asking.add(entry)
            self._schedule(self._count(entry, block, self._counter))
        return None

    async def _count(self, entry: tuple[str, str], block: dict[str, Any], counter: AttachmentCounter) -> None:
        model = entry[0]
        tokens: int | None = None
        try:
            baseline = self._baselines.get(model)
            if baseline is None:
                baseline = await counter(model, [_BASELINE_BLOCK])
            with_block = await counter(model, [block, _BASELINE_BLOCK])
        except Exception as err:
            # The reason's type only: an upstream's error text can echo the request.
            logger.warning(
                "could not count an attachment's tokens on %s (%s); it counts as its placeholder",
                model,
                type(err).__name__,
                extra={"model": model},
            )
        else:
            if baseline is not None:
                self._baselines[model] = baseline
                if with_block is not None and with_block > baseline:
                    tokens = with_block - baseline
            if tokens is None:
                logger.warning(
                    "the upstream gave no token count for an attachment on %s; it counts as its placeholder",
                    model,
                    extra={"model": model},
                )
        finally:
            self._asking.discard(entry)
        self._sizes[entry] = tokens
        self._sizes.move_to_end(entry)
        while len(self._sizes) > self._capacity:
            self._sizes.popitem(last=False)
