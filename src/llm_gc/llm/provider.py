"""The surface the engine uses to call a model, and the result it gets back.

Deliberately smaller than any vendor's API: the engine needs messages in, text
plus usage out, under a timeout. Everything a provider offers beyond that is the
adapter's business, not the engine's.

See doc/design/llm-provider.md for more details.
"""

from enum import Enum
from typing import Protocol

from pydantic import BaseModel, Field

from llm_gc.exceptions import ConfigError
from llm_gc.models import Message


class StopReason(Enum):
    """Why the model stopped generating, in LLM-GC's own vocabulary.

    Normalised rather than passed through, because providers agree on different
    words: Anthropic says `end_turn` / `max_tokens` / `stop_sequence`, OpenAI says
    `stop` / `length` / `content_filter`. A caller asking the one question that
    matters here — *is this output complete?* — would otherwise have to know both
    vocabularies, which is exactly the coupling the adapters exist to absorb.

    A compacted summary that stopped at the output cap has silently dropped the tail of
    the run it was summarising and still looks like a successful call - that has to be visible in the type.
    """

    COMPLETE = "complete"
    """The model finished on its own. The output is whole."""

    TRUNCATED = "truncated"
    """The output hit `max_output_tokens` and is cut off mid-thought. Callers
    must treat the text as incomplete: for a compactor that means refusing the
    summary or retrying with a smaller run, never filing it as if it were whole."""

    FILTERED = "filtered"
    """The provider suppressed the output on content-policy grounds. Distinct
    from TRUNCATED because retrying with a smaller input will not help."""

    OTHER = "other"
    """A stop reason this version does not recognise. Present so an unknown value
    from a provider degrades to 'we don't know' rather than being silently
    mapped onto COMPLETE, which would assert something we cannot support."""


class LLMResponse(BaseModel):
    """One successful generation, normalised across providers.

    Frozen: a record of something that already happened has no correct reason to
    be edited after the fact.

    The field list follows one rule — it carries **what only the provider knows**.
    Token usage and the stop reason qualify: the caller cannot derive either, and
    guessing them with a local tokenizer would report our estimate as the
    provider's fact. Call duration deliberately does not, because the caller can
    time the call itself just as accurately, and every field here is one more
    thing an external implementation has to populate honestly.
    """

    model_config = {"frozen": True}

    text: str
    """The generated content, already unwrapped from whatever envelope the
    provider used (Anthropic's content-block list, OpenAI's choices array)."""

    stop_reason: StopReason
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)

    model: str
    """The model that actually served the request, as the provider reported it —
    not the one we asked for. Providers route and alias, and provenance should
    record what answered, not what was requested."""

    provider: str
    """Short adapter name (`"anthropic"`, `"openai"`, `"fake"`), so a caller
    holding several providers can attribute a response without inspecting types."""

    @property
    def is_complete(self) -> bool:
        """Whether the text can be trusted as a whole answer.

        A named property rather than `response.stop_reason == StopReason.COMPLETE`
        at each call site: this is the check most likely to be forgotten, and the
        one whose absence fails silently.
        """
        return self.stop_reason is StopReason.COMPLETE


class LLMProvider(Protocol):
    """The contract every model client satisfies.

    A `Protocol` (structural typing), not an ABC: a library user who already has
    their own client wrapper should be able to pass it in because it has the right
    shape, without importing our base class and inheriting from it. There is no
    shared implementation here to inherit anyway — only a shape. Contrast
    `BaseCompactor`, which *is* an ABC and correctly so, because it carries real
    shared code (`_format_summary_marker`) that every compactor should reuse.

    Not `@runtime_checkable` on purpose: that would make `isinstance` succeed for
    any object with a `generate` attribute of any signature at all, which is worse
    than no check because it reads like verification and isn't one.
    """

    name: str
    """Short identifier the adapter stamps onto responses and errors."""

    async def generate(
        self,
        *,
        messages: list[Message],
        system: str | None = None,
        max_output_tokens: int,
        timeout_s: float,
    ) -> LLMResponse:
        """Generate one completion.

        Async because the production caller is: `GcService.gc_update` is a
        coroutine, and a blocking HTTP call inside it stalls the event loop — and
        therefore every other session — for the multi-second duration of a model
        call.

        Keyword-only throughout, so the parameter *names* are the contract rather
        than their order. An external implementation must not break because we
        inserted an argument.

        Args:
            messages: The conversation to send, in order. Must be non-empty and
                must contain no `system`-role messages — system content goes in
                `system`, because Anthropic takes it as a top-level field and has
                no system role to put it in. `token_count` and `turn_index` are
                LLM-GC bookkeeping and are ignored by every adapter.
            system: System instructions, or None.
            max_output_tokens: Hard cap on generated tokens. Required, not
                optional with a default: Anthropic mandates it, and an
                unbounded generation is a cost incident waiting to happen. If the
                model hits this cap the response comes back `TRUNCATED`.
            timeout_s: Wall-clock budget for the whole call. Required and
                undefaulted by design — a default timeout is the one every caller
                silently accepts and nobody tunes, and the right budget for
                summarising fifty turns is nothing like the right budget for a
                one-line classification.

        There is deliberately no `temperature` parameter. It was in an earlier
        draft of this Protocol and had to come out: current Claude models reject
        `temperature` with a 400 (it survives only on Opus 4.6 / Sonnet 4.6 /
        Haiku 4.5), while OpenAI accepts it everywhere. A parameter one adapter
        must either reject or silently discard is not a shared contract — and
        silently discarding it would be the dishonest option, since the caller
        would believe it had asked for determinism it never got. Determinism for
        compaction comes from the prompt instead, and on Anthropic from
        `output_config.effort`, which has no OpenAI equivalent and therefore
        belongs to that adapter rather than to this interface.

        Returns:
            A validated `LLMResponse`. Check `is_complete` before trusting `text`.

        Raises:
            LLMTimeoutError: The call exceeded `timeout_s`.
            LLMRateLimitError: The provider refused on rate or spend limits.
            LLMAuthError: Credentials were rejected.
            LLMContextOverflowError: The request exceeded the model's context window.
            LLMResponseError: The provider answered with something unusable.

            Every adapter raises from this set and nothing outside it. A caller
            that has to add a provider-specific `except` clause has found a bug in
            the adapter, not a gap in this list.
        """
        ...


def validate_generate_request(messages: list[Message], max_output_tokens: int, timeout_s: float) -> None:
    """Enforce the preconditions `LLMProvider.generate` documents.

    Shared by every implementation, including the fake.
    A fake that accepts input a real client would reject turns the test suite into
    a document about a system nobody is running. If it is going to blow up against
    Anthropic, it has to blow up here first.

    Raises:
        ConfigError: A precondition is violated. `ConfigError` rather than an
            `LLMError` because nothing went wrong with the *provider* — the caller
            built a request that no provider could serve, which is a bug on our
            side of the wire and should never reach the network.
    """
    if not messages:
        raise ConfigError("generate() requires at least one message")

    system_role_indices = [i for i, message in enumerate(messages) if message.role == "system"]
    if system_role_indices:
        # Anthropic has no system role in its messages array at all, so passing one
        # through is unrepresentable rather than merely unusual. Caught here so the
        # caller learns it at every provider, not just the one that happens to reject it.
        raise ConfigError(
            f"system-role messages are not allowed in `messages` (found at index "
            f"{system_role_indices[0]}); pass system content in the `system` argument"
        )

    if max_output_tokens <= 0:
        raise ConfigError(f"max_output_tokens must be positive, got {max_output_tokens}")

    if timeout_s <= 0:
        raise ConfigError(f"timeout_s must be positive, got {timeout_s}")
