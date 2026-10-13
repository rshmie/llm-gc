"""Counting what LLM-GC's own model calls spend, one measurement at a time.

A collection calls the summary model from several places, some of them concurrently: the
compactor for each span, the fact reader for each batch. Rather than thread a counter
through each, the provider is wrapped once, and every call adds its usage to the
measurement open in the calling task's context.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

from llm_gc.llm.provider import LLMProvider, LLMResponse
from llm_gc.models import Message


@dataclass(slots=True)
class ModelUsage:
    """Calls made and tokens spent, as the provider reported them."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: "ModelUsage") -> None:
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens


_open_measurement: ContextVar[ModelUsage | None] = ContextVar("llm_gc_open_measurement", default=None)


@contextmanager
def measure_model_usage() -> Iterator[ModelUsage]:
    """Count every metered call made in this context until the block ends.

    Tasks created inside the block copy the context, and with it this same object,
    so calls they make land in the same count. Measurements do not nest: an inner
    block counts its own calls, and the outer one does not see them.
    """
    usage = ModelUsage()
    token = _open_measurement.set(usage)
    try:
        yield usage
    finally:
        _open_measurement.reset(token)


class MeteredProvider:
    """An `LLMProvider` that adds each finished call's usage to the open measurement, if any.

    A call that raises spent tokens the provider did not report, so it adds nothing.
    """

    def __init__(self, inner: LLMProvider) -> None:
        self.inner = inner
        # The wrapped provider's name; a stand-in that has none is named by its class.
        self.name: str = getattr(inner, "name", type(inner).__name__)

    async def generate(self, *, messages: list[Message], system: str | None = None, max_output_tokens: int,
                       timeout_s: float) -> LLMResponse:
        response = await self.inner.generate(messages=messages, system=system,
                                              max_output_tokens=max_output_tokens, timeout_s=timeout_s)
        usage = _open_measurement.get()
        if usage is not None:
            usage.add(ModelUsage(calls=1, input_tokens=response.input_tokens, output_tokens=response.output_tokens))
        return response


if TYPE_CHECKING:
    # mypy verifies conformance to the Protocol; nothing runs at runtime. See the
    # note in fake_provider.py for why this line is the only enforcement there is.
    from llm_gc.llm.fake_provider import FakeProvider

    _conformance: LLMProvider = MeteredProvider(FakeProvider())
