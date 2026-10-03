"""An `LLMProvider` backed by OpenAI's Chat Completions API, over raw HTTP.

Raw `httpx2` rather than the `openai` SDK, for the same reason `anthropic_client`
does it: two adapters ship, and obliging every library user to install two vendor
SDKs to get a working compactor is a poor trade for one non-streaming call.

This file exists as much to *test* `LLMProvider` as to use it. An interface
designed while looking at a single vendor takes that vendor's shape, and no amount
of reading the interface reveals it — only a second implementation does. Three
things it found are called out where they occur: the output-cap parameter name,
the `developer`/`system` role drift, and a 429 that is permanent.

Everything provider-specific about OpenAI lives here: the request shape, the
response envelope, the `finish_reason` vocabulary, and the error taxonomy.
"""

import logging
from dataclasses import dataclass
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

PROVIDER_NAME = "openai"
OPENAI_BASE_URL = "https://api.openai.com"
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"

# OpenAI versions by model name and by URL path, not by a header. There is no
# equivalent of Anthropic's `anthropic-version`, so this adapter sends one fewer
# header — the first asymmetry, and a harmless one.

# The middle tier, matching the Sonnet-not-Haiku reasoning in `anthropic_client`: a
# summary that loses what mattered defeats the point of compacting, and the cheapest
# tier is the one most likely to produce one. Cost is the reason to override it, and
# a deployment can, per call site. Verify the id against OpenAI's current model list
# before shipping — vendor model names churn faster than this file will.
DEFAULT_OPENAI_MODEL = "gpt-5-mini"

# OpenAI's `finish_reason` vocabulary mapped onto ours. Note how little it overlaps
# with Anthropic's table even though both describe the same four outcomes: this
# non-overlap is the whole argument for normalising stop reasons in the adapter
# instead of passing the vendor's string through to callers.
_FINISH_REASONS: dict[str, StopReason] = {
    "stop": StopReason.COMPLETE,
    "length": StopReason.TRUNCATED,
    "content_filter": StopReason.FILTERED,
    # `tool_calls` means the model chose to call a tool instead of answering. We
    # never offer tools, so seeing it means the request was not what we think it
    # was. OTHER, not COMPLETE: the text is not an answer to what we asked.
    "tool_calls": StopReason.OTHER,
    "function_call": StopReason.OTHER,
}

# Unlike Anthropic, OpenAI reports context overflow with a machine-readable `code`
# on the error body, so this adapter matches on an identifier instead of scraping
# English prose. That difference is worth noticing: the fragile substring match in
# `anthropic_client` is not a style choice, it is the best available signal there.
_OVERFLOW_CODES = frozenset({"context_length_exceeded", "string_above_max_length"})

# A 429 from OpenAI is two different events wearing one status code. Hitting a
# requests-per-minute ceiling clears in seconds; exhausting the account's quota
# never clears without someone adding funds. Retrying the second one burns the whole
# retry budget to arrive at a certainty, and delays the one error message that tells
# an operator what to actually fix.
_PERMANENT_RATE_LIMIT_TYPES = frozenset({"insufficient_quota"})

_RETRYABLE_STATUSES = frozenset({408, 409, 500, 502, 503, 504, 529})


@dataclass(frozen=True)
class _ErrorDetail:
    """The three fields we can use out of an OpenAI error body.

    A `dataclass`, not a Pydantic model, and the distinction is deliberate: Pydantic
    is for parsing a boundary we do not control, which is why `_ChatCompletion`
    below is one. This object is built *by us* after defensively digging through an
    untrusted body, with every field already guaranteed to be a `str`. Validating
    our own output would be ceremony.
    """

    message: str
    type: str
    code: str


class _ChatMessage(BaseModel):
    """`content` is nullable: a filtered or tool-call completion carries no text."""

    content: str | None = None


class _Choice(BaseModel):
    message: _ChatMessage
    finish_reason: str | None = None


class _Usage(BaseModel):
    """OpenAI's token accounting.

    Both fields are required rather than defaulted to 0. Usage is the one thing only
    the provider knows, and a 0 would be recorded as "this call was free" — a quiet
    lie in exactly the place the project promises honesty.
    """

    prompt_tokens: int
    completion_tokens: int


class _ChatCompletion(BaseModel):
    """The Chat Completions success envelope, validated rather than indexed.

    Only the fields we consume are declared; Pydantic ignores the rest, so OpenAI
    adding one cannot break us. Declaring them is what turns a malformed body into a
    single `ValidationError` at the boundary instead of an `IndexError` on
    `choices[0]` or a `None` surfacing three layers away.
    """

    model: str
    choices: list[_Choice]
    usage: _Usage


class OpenAIClient:
    """Calls OpenAI's Chat Completions API and returns a normalised `LLMResponse`.

    Holds a long-lived `httpx2.AsyncClient` so repeated compactions reuse the
    connection. Either close it with `aclose()` or use it as an async context
    manager. An injected client is never closed by us — whoever created it owns its
    lifetime, and closing someone else's pooled client breaks its other users.
    """

    name = PROVIDER_NAME

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_OPENAI_MODEL,
        base_url: str = OPENAI_BASE_URL,
        organization: str | None = None,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        """Build a client.

        Args:
            api_key: The OpenAI API key. Validated here rather than on first use, so
                a misconfigured deployment fails at startup instead of on the first
                compaction under load.
            model: The model to summarise with. See `DEFAULT_OPENAI_MODEL`.
            base_url: Overridable for a gateway, Azure-style endpoint, or a test
                double.
            organization: Optional `OpenAI-Organization` header, for accounts whose
                billing is split across organisations. Adapter-specific on purpose —
                it has no Anthropic equivalent, so it belongs here and not on
                `LLMProvider`, exactly as `temperature` did not belong there.
            http_client: An existing client to borrow, chiefly so tests can supply an
                `httpx2.MockTransport` and exercise the real parsing and error
                mapping without a network. When None, we own one and close it.

        Raises:
            ConfigError: The API key is missing or blank.
        """
        if not api_key or not api_key.strip():
            raise ConfigError("OpenAIClient requires a non-empty api_key")

        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._organization = organization
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
                f"{self._base_url}{CHAT_COMPLETIONS_PATH}",
                headers=self._build_headers(),
                json=self._build_payload(messages, system, max_output_tokens),
                timeout=timeout_s,
            )
        except httpx2.TimeoutException as err:
            # The message names the budget, never the request: an exception string
            # reaches logs, and the prompt is the user's conversation.
            raise LLMTimeoutError(f"openai did not respond within {timeout_s}s", provider=self.name) from err
        except httpx2.RequestError as err:
            raise LLMConnectionError(f"could not reach openai: {type(err).__name__}", provider=self.name) from err

        if http_response.status_code != 200:
            raise self._as_error(http_response)

        return self._parse(http_response)

    def _build_headers(self) -> dict[str, str]:
        """Assemble request headers.

        Bearer auth, where Anthropic uses its own `x-api-key`. Both are secrets, and
        neither is ever logged — CLAUDE.md §4 forbids logging request headers at all,
        precisely because this is where keys live.
        """
        headers = {
            "authorization": f"Bearer {self._api_key}",
            "content-type": "application/json",
        }
        if self._organization:
            headers["openai-organization"] = self._organization
        return headers

    def _build_payload(self, messages: list[Message], system: str | None, max_output_tokens: int) -> dict[str, Any]:
        """Translate our vocabulary into OpenAI's request body.

        Two differences from Anthropic, both absorbed here so callers never see
        them:

        **System content is a message, not a field.** OpenAI has a `system` role in
        the messages array; Anthropic has no system role at all. `LLMProvider` takes
        `system` as its own argument because that is the only shape *both* can
        express — the narrower of two vendors wins when picking an interface,
        because the wider one can always be reached from it and not the reverse.
        (Newer OpenAI reasoning models prefer the role name `developer`, with
        `system` still accepted and treated as an alias. We send `system`: it works
        on every model including the older ones, and switching would gain nothing.)

        **The output cap is `max_completion_tokens`.** Anthropic calls it
        `max_tokens`; so did OpenAI, until reasoning models made the old name wrong
        — a reasoning model's hidden thinking also consumes the cap, so the field
        was renamed and `max_tokens` is now rejected by those models. This is the
        second time this concept has needed renaming in this project, after
        `GCConfig.max_tokens` became `context_window`. Three different meanings have
        worn that one name.

        One consequence worth knowing: on a reasoning model, a small cap can be
        consumed entirely by thinking, returning empty content with
        `finish_reason: "length"`. That surfaces honestly as `TRUNCATED` and the
        compactor refuses the summary, which is the right outcome — but the fix is a
        larger cap, not a retry.
        """
        wire_messages: list[dict[str, str]] = []
        if system:
            wire_messages.append({"role": "system", "content": system})
        # `token_count` and `turn_index` are LLM-GC bookkeeping; OpenAI rejects
        # unknown keys inside a message object, so they are dropped rather than
        # passed through.
        wire_messages.extend({"role": message.role, "content": message.content} for message in messages)

        return {
            "model": self._model,
            "max_completion_tokens": max_output_tokens,
            "messages": wire_messages,
        }

    def _parse(self, http_response: httpx2.Response) -> LLMResponse:
        """Validate the success envelope and normalise it."""
        try:
            parsed = _ChatCompletion.model_validate(http_response.json())
        except (ValidationError, ValueError) as err:
            raise LLMResponseError(
                "openai returned a body that is not a valid chat completion",
                provider=self.name,
                status_code=http_response.status_code,
                # A 200 whose body does not parse is more likely a gateway or a
                # partial write than a permanent fault, so one more attempt is worth
                # it.
                retryable=True,
            ) from err

        if not parsed.choices:
            # Schema-valid and useless. An empty list survives validation because
            # `list[_Choice]` permits it, so the emptiness has to be caught here.
            raise LLMResponseError(
                "openai returned no choices",
                provider=self.name,
                status_code=200,
                retryable=True,
            )

        # We ask for one completion and read the first. Taking choices[0] is safe
        # only because `n` is never set above its default of 1; if that ever
        # changes, this line is the one that has to change with it.
        choice = parsed.choices[0]
        text = choice.message.content or ""
        stop_reason = _FINISH_REASONS.get(choice.finish_reason or "", StopReason.OTHER)

        if not text and stop_reason is not StopReason.FILTERED:
            # A filtered completion legitimately has no text. Anything else with
            # none means we asked for a completion and did not get one.
            raise LLMResponseError(
                f"openai returned empty content (finish_reason={choice.finish_reason!r})",
                provider=self.name,
                status_code=200,
                retryable=True,
            )

        if choice.finish_reason and choice.finish_reason not in _FINISH_REASONS:
            # Worth a log line: it means the table needs a new row, and until it
            # gets one the caller is told OTHER and will not trust the output.
            logger.warning(
                "unrecognised openai finish_reason, reporting as OTHER",
                extra={"finish_reason": choice.finish_reason, "provider": self.name},
            )

        return LLMResponse(
            text=text,
            stop_reason=stop_reason,
            input_tokens=parsed.usage.prompt_tokens,
            output_tokens=parsed.usage.completion_tokens,
            model=parsed.model,
            provider=self.name,
        )

    def _as_error(self, http_response: httpx2.Response) -> Exception:
        """Map a failing status onto LLM-GC's taxonomy.

        Returns the exception rather than raising it, so the call site reads
        `raise self._as_error(...)` and every path out of `generate` is visibly a
        raise. The provider's own message is quoted because it is diagnostic;
        nothing from the request is echoed back.
        """
        status = http_response.status_code
        detail = self._error_detail(http_response)

        # Checked before the status branches, because overflow arrives as a 400 and
        # a plain 400 must not be mistaken for it. OpenAI gives us a code, so unlike
        # Anthropic this is an exact match rather than a prose search.
        if detail.code in _OVERFLOW_CODES:
            return LLMContextOverflowError(
                f"openai refused the request as too long: {detail.message}",
                provider=self.name,
                # OpenAI states both numbers in prose with no stable shape, and a
                # wrong number here would send the compactor splitting runs by a
                # figure it invented. None is the honest answer; both fields are
                # optional for exactly this reason.
                requested_tokens=None,
                limit_tokens=None,
            )

        if status in (401, 403):
            return LLMAuthError(
                f"openai rejected our credentials ({status}): {detail.message}", provider=self.name
            )

        if status == 429:
            permanent = detail.type in _PERMANENT_RATE_LIMIT_TYPES or detail.code in _PERMANENT_RATE_LIMIT_TYPES
            return LLMRateLimitError(
                f"openai rate limited us (429): {detail.message}",
                provider=self.name,
                retry_after_s=self._retry_after(http_response),
                # The finding this adapter produced: the class default is True, and
                # a quota-exhausted 429 is the case that needs to say otherwise.
                retryable=not permanent,
            )

        return LLMResponseError(
            f"openai returned {status}: {detail.message}",
            provider=self.name,
            status_code=status,
            # Only the adapter knows which statuses are worth repeating. A 503 is; a
            # 400 never will be, and retrying it spends the budget on a certainty.
            retryable=status in _RETRYABLE_STATUSES,
        )

    @staticmethod
    def _error_detail(http_response: httpx2.Response) -> _ErrorDetail:
        """Pull message, type and code out of an error body, defensively.

        An error response is the least reliable thing a provider sends — it can
        arrive as HTML from a load balancer, or be empty. Every failure path here
        degrades to a placeholder, because an exception raised while building an
        exception loses the original status entirely.

        `type` and `code` default to `""` rather than None so the caller can test
        membership in a frozenset without a None check first; `""` is in no set.
        """
        try:
            body = http_response.json()
        except ValueError:
            return _ErrorDetail(message=http_response.text[:200] or "<empty body>", type="", code="")

        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                # Each value is bound to a local and narrowed separately: `json()`
                # is typed `Any`, so knowing the container is a dict says nothing
                # about what comes out of it.
                message = error.get("message")
                error_type = error.get("type")
                error_code = error.get("code")
                return _ErrorDetail(
                    message=message if isinstance(message, str) else str(body)[:200],
                    type=error_type if isinstance(error_type, str) else "",
                    code=error_code if isinstance(error_code, str) else "",
                )
        return _ErrorDetail(message=str(body)[:200], type="", code="")

    @staticmethod
    def _retry_after(http_response: httpx2.Response) -> float | None:
        """Read `Retry-After`, which the retry policy prefers over its own backoff.

        OpenAI also sends `x-ratelimit-reset-requests` (as a duration string like
        `"6m0s"`), deliberately not read here: `Retry-After` is the standard header
        and parsing a second, vendor-shaped one adds a failure mode for information
        we already have. A non-numeric value yields None and our own schedule
        applies instead of a wrong number.
        """
        raw = http_response.headers.get("retry-after")
        if raw is None:
            return None
        try:
            return float(raw)
        except ValueError:
            logger.debug("unparseable retry-after header", extra={"retry_after": raw})
            return None


if TYPE_CHECKING:
    # mypy verifies conformance to the Protocol; nothing runs at runtime. See the
    # note in fake_provider.py for why this line is the only enforcement there is.
    _conformance: LLMProvider = OpenAIClient(api_key="x")
