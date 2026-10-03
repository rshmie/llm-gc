"""The interface the engine uses to call a model, and the result it gets back.

Deliberately smaller than any vendor API. See doc/design/llm-provider.md.
"""

from enum import Enum
from typing import Protocol

from pydantic import BaseModel, Field

from llm_gc.exceptions import ConfigError
from llm_gc.models import Message


class StopReason(Enum):
    """Why generation stopped, normalised across providers.

    Providers disagree on wording (`end_turn`/`max_tokens` vs
    `stop`/`length`/`content_filter`). The question callers need answered is
    whether the output is complete.
    """

    COMPLETE = "complete"
    TRUNCATED = "truncated"
    """Hit `max_output_tokens`; the text is cut off and must not be trusted."""

    FILTERED = "filtered"
    """Suppressed on content policy. Retrying smaller will not help."""

    OTHER = "other"
    """Unrecognised value. Never mapped to COMPLETE, which would assert
    something we cannot support."""


class LLMResponse(BaseModel):
    """One successful generation, normalised across providers.

    Carries only what the provider alone knows: a caller can time the call
    itself, so duration is deliberately absent.
    """

    model_config = {"frozen": True}

    text: str
    stop_reason: StopReason
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)

    model: str
    """The model that actually answered, as reported — providers route and alias."""

    provider: str
    """Short adapter name (`"anthropic"`, `"openai"`, `"fake"`)."""

    @property
    def is_complete(self) -> bool:
        """Whether `text` can be trusted as a whole answer."""
        return self.stop_reason is StopReason.COMPLETE


class LLMProvider(Protocol):
    """The contract every model client satisfies.

    Structural typing, so a library user's existing client wrapper fits without
    inheriting from us. Deliberately not `@runtime_checkable`: that only checks
    attribute names, which reads like verification without being any. Each
    implementation carries a `TYPE_CHECKING` conformance assertion instead.
    """

    name: str

    async def generate(
        self,
        *,
        messages: list[Message],
        system: str | None = None,
        max_output_tokens: int,
        timeout_s: float,
    ) -> LLMResponse:
        """Generate one completion.

        Keyword-only, so parameter names rather than their order are the contract.
        There is no `temperature`: current Claude models reject it while OpenAI
        accepts it, so no adapter could honour it honestly.

        Args:
            messages: The conversation, in order. Non-empty, and no `system`-role
                entries — system content goes in `system`, because Anthropic has
                no system role to put it in.
            system: System instructions, or None.
            max_output_tokens: Hard cap on generated tokens. Required: Anthropic
                mandates it and an uncapped generation is a cost incident.
            timeout_s: Wall-clock budget. Required, because a default timeout is
                the one nobody tunes.

        Returns:
            A validated `LLMResponse`. Check `is_complete` before trusting `text`.

        Raises:
            LLMTimeoutError, LLMConnectionError, LLMRateLimitError, LLMAuthError,
            LLMContextOverflowError, LLMResponseError: Adapters raise from this set
            and nothing outside it.
        """
        ...


def validate_generate_request(messages: list[Message], max_output_tokens: int, timeout_s: float) -> None:
    """Enforce `generate`'s documented preconditions.

    Shared by every implementation including the fake, so a request that would
    fail against a real provider fails in tests too.

    Raises:
        ConfigError: A precondition is violated. Not an `LLMError` — the caller
            built a request no provider could serve, which is a bug on our side.
    """
    if not messages:
        raise ConfigError("generate() requires at least one message")

    system_role_indices = [i for i, message in enumerate(messages) if message.role == "system"]
    if system_role_indices:
        raise ConfigError(
            f"system-role messages are not allowed in `messages` (found at index "
            f"{system_role_indices[0]}); pass system content in the `system` argument"
        )

    if max_output_tokens <= 0:
        raise ConfigError(f"max_output_tokens must be positive, got {max_output_tokens}")

    if timeout_s <= 0:
        raise ConfigError(f"timeout_s must be positive, got {timeout_s}")
