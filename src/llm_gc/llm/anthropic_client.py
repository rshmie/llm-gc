"""An `LLMProvider` backed by Anthropic's Messages API, over raw HTTP.

Raw `httpx2` rather than the `anthropic` SDK, per doc/design/llm-provider.md: the
surface we need is one non-streaming call, and a library should not oblige its users
to install a vendor SDK to get a working compactor. The cost we accept in exchange
is owning the wire format ourselves, which is why the envelope below is parsed by
Pydantic rather than by indexing into dicts.

Everything provider-specific about Anthropic lives in this file: the request shape,
the response envelope, the stop-reason vocabulary, and the error taxonomy.
"""

import logging
import re
from typing import TYPE_CHECKING, Any, Self

import httpx2
from pydantic import BaseModel, ValidationError

from llm_gc.exceptions import (
    ConfigError,
    LLMAuthError,
    LLMConnectionError,
    LLMContextOverflowError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)
from llm_gc.llm.provider import LLMProvider, LLMResponse, StopReason, validate_generate_request
from llm_gc.models import Message

logger = logging.getLogger(__name__)

PROVIDER_NAME = "anthropic"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
MESSAGES_PATH = "/v1/messages"

# Anthropic pins breaking changes behind this header rather than a URL version, so it
# is required on every request. Bumping it is a deliberate migration, never a default.
ANTHROPIC_API_VERSION = "2023-06-01"

# The model the *compactor* calls, which is not the model being proxied: `GCConfig`
# describes the conversation's own model and its context window. Keeping them separate
# is what lets a deployment proxy an expensive model while summarising with a cheaper
# one. Sonnet over Haiku here because a lossy summary defeats the entire purpose of
# compaction, and because Haiku's smaller context window would cap how large a run
# can be summarised in one call. Cost is the reason to revisit it, deliberately.
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"

# Anthropic's own words for why generation stopped, mapped onto ours. Anything absent
# from this table becomes StopReason.OTHER rather than being assumed complete.
_STOP_REASONS: dict[str, StopReason] = {
    "end_turn": StopReason.COMPLETE,
    "stop_sequence": StopReason.COMPLETE,
    "max_tokens": StopReason.TRUNCATED,
    "refusal": StopReason.FILTERED,
}

# Anthropic signals context overflow as a 400 whose message names the limit, with no
# distinct error type to match on. Substring matching is fragile by nature, so it is
# kept narrow and falls through to a plain LLMResponseError when it does not hit:
# mislabelling a generic 400 as an overflow would send the compactor off splitting a
# run that was never too long.
_OVERFLOW_MARKERS = ("prompt is too long", "exceeds the maximum", "context window")
_OVERFLOW_NUMBERS = re.compile(r"(\d[\d,]*)\s*tokens?\D+(\d[\d,]*)")

_RETRYABLE_STATUSES = frozenset({408, 409, 500, 502, 503, 504, 529})


class _Usage(BaseModel):
    """Anthropic's token accounting. Extra fields (cache counters) are ignored."""

    input_tokens: int
    output_tokens: int


class _ContentBlock(BaseModel):
    type: str
    text: str | None = None


class _MessagesResponse(BaseModel):
    """The Messages API success envelope, validated rather than indexed.

    Only the fields we consume are declared; Pydantic ignores the rest by default,
    so Anthropic adding a field cannot break us. Declaring them at all is what turns
    a malformed body into one `ValidationError` at the boundary instead of a
    `KeyError` or a `None` surfacing three layers away.
    """

    model: str
    content: list[_ContentBlock]
    usage: _Usage
    stop_reason: str | None = None


class AnthropicClient:
    """Calls Anthropic's Messages API and returns a normalised `LLMResponse`.

    Holds a long-lived `httpx2.AsyncClient` so repeated compactions reuse the
    connection: a fresh TLS handshake per call would add a hundred milliseconds or
    more to every promotion, and promotions happen throughout a session.

    Either close it explicitly with `aclose()` or use it as an async context
    manager. An injected client is never closed by us — whoever created it owns its
    lifetime, and closing someone else's pooled client would break its other users.
    """

    name = PROVIDER_NAME

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        base_url: str = ANTHROPIC_BASE_URL,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        """Build a client.

        Args:
            api_key: The Anthropic API key. Validated here rather than on first use,
                so a misconfigured deployment fails at startup instead of on the
                first compaction under load.
            model: The model to summarise with. See `DEFAULT_ANTHROPIC_MODEL`.
            base_url: Overridable for a proxy or a test double.
            http_client: An existing client to borrow, chiefly so tests can supply an
                `httpx2.MockTransport` and exercise the real parsing and error
                mapping without a network. When None, we own one and close it.

        Raises:
            ConfigError: The API key is missing or blank.
        """
        if not api_key or not api_key.strip():
            raise ConfigError("AnthropicClient requires a non-empty api_key")

        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._http_client = http_client or httpx2.AsyncClient()
        # Only a client we created is ours to close.
        self._owns_http_client = http_client is None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Release the connection pool, if this client owns it."""
        if self._owns_http_client:
            await self._http_client.aclose()

    async def generate(
        self,
        *,
        messages: list[Message],
        system: str | None = None,
        max_output_tokens: int,
        timeout_s: float,
    ) -> LLMResponse:
        """Generate one completion. See `LLMProvider.generate` for the contract."""
        validate_generate_request(messages, max_output_tokens, timeout_s)

        try:
            http_response = await self._http_client.post(
                f"{self._base_url}{MESSAGES_PATH}",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": ANTHROPIC_API_VERSION,
                    "content-type": "application/json",
                },
                json=self._build_payload(messages, system, max_output_tokens),
                timeout=timeout_s,
            )
        except httpx2.TimeoutException as err:
            # The message deliberately names the budget and not the request: an
            # exception string reaches logs, and the prompt is user data.
            raise LLMTimeoutError(
                f"anthropic did not respond within {timeout_s}s", provider=self.name
            ) from err
        except httpx2.RequestError as err:
            raise LLMConnectionError(
                f"could not reach anthropic: {type(err).__name__}", provider=self.name
            ) from err

        if http_response.status_code != 200:
            raise self._as_error(http_response)

        return self._parse(http_response)

    def _build_payload(
        self, messages: list[Message], system: str | None, max_output_tokens: int
    ) -> dict[str, Any]:
        """Translate our vocabulary into Anthropic's request body.

        Two translations happen here. `system` becomes a top-level field rather than
        a message, because Anthropic has no system role in its messages array. And
        `Message` is reduced to role and content: `token_count` and `turn_index` are
        LLM-GC bookkeeping that would be rejected as unknown fields.

        `max_tokens` is Anthropic's name for the output cap, not for a context
        window — the same collision that got `GCConfig.max_tokens` renamed to
        `context_window`.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_output_tokens,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
        }
        if system:
            payload["system"] = system
        return payload

    def _parse(self, http_response: httpx2.Response) -> LLMResponse:
        """Validate the success envelope and normalise it."""
        try:
            parsed = _MessagesResponse.model_validate(http_response.json())
        except (ValidationError, ValueError) as err:
            raise LLMResponseError(
                "anthropic returned a body that is not a valid Messages response",
                provider=self.name,
                status_code=http_response.status_code,
                # A 200 whose body does not parse is more likely a proxy or a partial
                # write than a permanent fault, so one more attempt is worth it.
                retryable=True,
            ) from err

        # `content` is a list of blocks, and only text blocks carry text — a thinking
        # or tool-use block has none. Joining the text blocks rather than taking
        # content[0] is what keeps this correct if a non-text block ever leads.
        text = "".join(block.text or "" for block in parsed.content if block.type == "text")

        stop_reason = _STOP_REASONS.get(parsed.stop_reason or "", StopReason.OTHER)

        if not text and stop_reason is not StopReason.FILTERED:
            # A filtered response legitimately has no text. Anything else with none
            # means we asked for a completion and did not get one.
            raise LLMResponseError(
                f"anthropic returned no text content (stop_reason={parsed.stop_reason!r})",
                provider=self.name,
                status_code=200,
                retryable=True,
            )

        if parsed.stop_reason and parsed.stop_reason not in _STOP_REASONS:
            # Worth a line in the log: it means this table needs a new row, and until
            # it gets one the caller is told OTHER and will not trust the output.
            logger.warning(
                "unrecognised anthropic stop_reason, reporting as OTHER",
                extra={"stop_reason": parsed.stop_reason, "provider": self.name},
            )

        return LLMResponse(
            text=text,
            stop_reason=stop_reason,
            input_tokens=parsed.usage.input_tokens,
            output_tokens=parsed.usage.output_tokens,
            model=parsed.model,
            provider=self.name,
        )

    def _as_error(self, http_response: httpx2.Response) -> Exception:
        """Map a failing status onto LLM-GC's taxonomy.

        Returns the exception rather than raising it, so the call site reads
        `raise self._as_error(...)` and every path out of `generate` is visibly a
        raise. The provider's own message is quoted because it is diagnostic, but
        nothing from the request is echoed back.
        """
        status = http_response.status_code
        detail = self._error_message(http_response)

        if status in (401, 403):
            return LLMAuthError(f"anthropic rejected our credentials ({status}): {detail}", provider=self.name)

        if status == 429:
            return LLMRateLimitError(
                f"anthropic rate limited us (429): {detail}",
                provider=self.name,
                retry_after_s=self._retry_after(http_response),
            )

        if status == 400 and any(marker in detail.lower() for marker in _OVERFLOW_MARKERS):
            requested, limit = self._overflow_numbers(detail)
            return LLMContextOverflowError(
                f"anthropic refused the request as too long: {detail}",
                provider=self.name,
                requested_tokens=requested,
                limit_tokens=limit,
            )

        return LLMResponseError(
            f"anthropic returned {status}: {detail}",
            provider=self.name,
            status_code=status,
            # Only the adapter knows which statuses are worth repeating. A 503 is; a
            # 400 never will be, and retrying it spends the budget on a certainty.
            retryable=status in _RETRYABLE_STATUSES,
        )

    @staticmethod
    def _error_message(http_response: httpx2.Response) -> str:
        """Pull the human-readable detail out of an error body, defensively.

        An error response is the least reliable thing a provider sends — it can come
        from a load balancer as HTML, or be empty. Every failure here degrades to a
        short placeholder, because an exception raised while building an exception
        loses the original status entirely.
        """
        try:
            body = http_response.json()
        except ValueError:
            return http_response.text[:200] or "<empty body>"

        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                # Bound to a local and narrowed here rather than indexed twice:
                # `json()` is typed Any, so an isinstance check on the dict says
                # nothing about the type of what comes out of it.
                message = error.get("message")
                if isinstance(message, str):
                    return message
        return str(body)[:200]

    @staticmethod
    def _retry_after(http_response: httpx2.Response) -> float | None:
        """Read `Retry-After`, which the retry policy prefers over its own backoff.

        Anthropic sends it as seconds. The HTTP-date form is legal but unused here,
        so a non-numeric value yields None and our schedule applies instead of a
        wrong number.
        """
        raw = http_response.headers.get("retry-after")
        if raw is None:
            return None
        try:
            return float(raw)
        except ValueError:
            logger.debug("unparseable retry-after header", extra={"retry_after": raw})
            return None

    @staticmethod
    def _overflow_numbers(detail: str) -> tuple[int | None, int | None]:
        """Best-effort read of "N tokens > M maximum" from an overflow message.

        Both values are optional on `LLMContextOverflowError` precisely because this
        is scraped from prose. A caller that needs them must handle None; one that
        only needs to know it overflowed does not care.
        """
        match = _OVERFLOW_NUMBERS.search(detail)
        if not match:
            return None, None
        return int(match.group(1).replace(",", "")), int(match.group(2).replace(",", ""))


if TYPE_CHECKING:
    # mypy verifies conformance to the Protocol; nothing runs at runtime. See the
    # note in fake_provider.py for why this line is the only enforcement there is.
    _conformance: LLMProvider = AnthropicClient(api_key="x")
