"""An in-memory `LLMProvider` for tests — ours, and library users'.

Shipped in the package rather than confined to `tests/` because anyone importing
LLM-GC needs a way to exercise their code without calling a real model, and a
fake they have to write themselves will be more forgiving than this one.

The guiding constraint is that this fake must not be kinder than a real provider.
It validates requests the way a real client does, it reports usage from the real
tokenizer, and a script that runs out raises instead of quietly repeating itself.
Every one of those would otherwise be a way for a green suite to hide a bug.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from llm_gc.exceptions import LLMError
from llm_gc.llm.provider import LLMProvider, LLMResponse, StopReason, validate_generate_request
from llm_gc.models import Message
from llm_gc.utils import count_tokens

FAKE_PROVIDER_NAME = "fake"
FAKE_MODEL_NAME = "fake-model-1"

#: Returned when the fake was constructed without a script. Deliberately boring:
#: a test that cares what the model said scripts it, and one that does not should
#: not be silently depending on this text.
DEFAULT_FAKE_TEXT = "[fake summary]"


@dataclass(frozen=True)
class GenerateCall:
    """One recorded invocation, so tests can assert on what was *sent*.

    A frozen dataclass rather than a Pydantic model: this is built from arguments
    we already control, not parsed from an untrusted boundary, so validation would
    cost something and buy nothing. `LLMResponse` is Pydantic for the opposite
    reason — it is built from provider JSON.
    """

    messages: tuple[Message, ...]
    system: str | None
    max_output_tokens: int
    timeout_s: float


class FakeProviderExhausted(LLMError):
    """The scripted responses ran out but another call arrived.

    An error rather than a repeat of the last response, because over-calling is
    almost always the bug under test: a retry policy that does not stop, a loop
    that re-compacts, a caller that issues one request per message where it meant
    to batch. Repeating the last response would make every one of those pass.
    """

    code = "FAKE_PROVIDER_EXHAUSTED"


@dataclass
class FakeProvider:
    """A scripted `LLMProvider`.

    Two modes, and which one applies depends only on whether `responses` was given:

    - **Unscripted** (the default) — every call returns a generic complete
      response. For tests that need *a* provider to exist but do not care what it
      says.
    - **Scripted** — `responses` is consumed in order. An entry that is an
      exception is raised instead of returned, so success and failure are scripted
      through one mechanism (the same idea as `unittest.mock`'s `side_effect`).
      Running out raises `FakeProviderExhausted`.

    Attributes:
        responses: The script, consumed left to right. `None` means unscripted.
        calls: Every invocation, in order, for assertions about what was sent.
    """

    responses: list[LLMResponse | Exception] | None = None
    calls: list[GenerateCall] = field(default_factory=list)
    name: str = FAKE_PROVIDER_NAME

    # Bookkeeping, not configuration: init=False keeps it out of the constructor so
    # a test cannot start a script half-consumed by accident.
    _next_index: int = field(default=0, init=False, repr=False)

    async def generate(
        self,
        *,
        messages: list[Message],
        system: str | None = None,
        max_output_tokens: int,
        timeout_s: float,
    ) -> LLMResponse:
        """Record the call, then return or raise the next scripted item.

        Validation runs *before* recording, matching a real client: a request that
        never goes out should not appear in the call log as though it did.
        """
        validate_generate_request(messages, max_output_tokens, timeout_s)

        self.calls.append(
            GenerateCall(
                messages=tuple(messages),
                system=system,
                max_output_tokens=max_output_tokens,
                timeout_s=timeout_s,
            )
        )

        if self.responses is None:
            return self._default_response(messages, system)

        if self._next_index >= len(self.responses):
            raise FakeProviderExhausted(
                f"FakeProvider was scripted with {len(self.responses)} response(s) "
                f"but received call {self._next_index + 1}",
                provider=self.name,
            )

        scripted = self.responses[self._next_index]
        self._next_index += 1
        if isinstance(scripted, Exception):
            raise scripted
        return scripted

    def _default_response(self, messages: list[Message], system: str | None) -> LLMResponse:
        """Build a plausible response for the unscripted mode.

        Usage comes from the real tokenizer rather than a constant, so a test
        asserting on token accounting is measuring something. A fake that always
        reported zero would let a cost-tracking bug pass every time.
        """
        prompt_text = "\n".join(message.content for message in messages)
        if system:
            prompt_text = f"{system}\n{prompt_text}"

        return LLMResponse(
            text=DEFAULT_FAKE_TEXT,
            stop_reason=StopReason.COMPLETE,
            input_tokens=count_tokens(prompt_text),
            output_tokens=count_tokens(DEFAULT_FAKE_TEXT),
            model=FAKE_MODEL_NAME,
            provider=self.name,
        )


if TYPE_CHECKING:
    # Structural-conformance check. Nothing runs here — the block is erased at
    # runtime — but mypy verifies that FakeProvider actually satisfies the
    # Protocol, and reports the exact signature diff if it stops doing so.
    #
    # This is the *only* enforcement a Protocol gets. Python does not check type
    # annotations at runtime, and LLMProvider is deliberately not
    # @runtime_checkable, so without this line a fake whose signature drifted from
    # the interface would sail through every test that uses it. Every
    # implementation in this package carries one.
    _conformance: LLMProvider = FakeProvider()
