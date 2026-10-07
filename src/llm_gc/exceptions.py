"""The exception hierarchy for LLM-GC.

Everything derives from `LLMGCError`, so a caller can catch the whole library or
narrow to a category. Two conventions: retryability is carried by the error rather
than inferred by the caller, and messages never carry credentials or prompt text
(they reach logs, and after translation, HTTP responses).

Raise sites chain the cause (`raise LLMTimeoutError(...) from err`).
"""


class LLMGCError(Exception):
    """Base class for every error LLM-GC raises.

    Attributes:
        code: Stable, protocol-agnostic identifier (e.g. `"LLM_TIMEOUT"`) that the
            HTTP boundary maps to a response code from one table. Deliberately not
            an HTTP status — the library must stay usable without HTTP.
        retryable: Whether retrying the identical operation could succeed.
            Per-class default, overridden per-instance where a category spans both.
    """

    code: str = "LLM_GC_ERROR"
    retryable: bool = False


class ConfigError(LLMGCError):
    """Configuration is missing, malformed, or inconsistent.

    Raised at construction time, not on the request path: a client built without
    an API key should fail at startup, not on the first call under load.
    """

    code = "CONFIG_ERROR"


class ProxyUnavailableError(LLMGCError):
    """A running LLM-GC proxy could not be reached, or did not answer like one.

    Raised by the command line when it asks a proxy about itself. Retryable: the
    proxy may simply not have finished starting.
    """

    code = "PROXY_UNAVAILABLE"
    retryable = True


class CompactionRefused(LLMGCError):
    """A compactor declined to file the summary it was handed.

    Not a provider failure — the call succeeded. The summary was truncated, empty,
    or no shorter than the run it would replace, so it must not stand in for those
    turns. Callers leave the run verbatim for the next pass.

    `retryable` means a *later* attempt can succeed, not that repeating this call
    will: a truncated summary wants a smaller run or a larger cap, while one that
    came back no shorter wants a larger run to amortise the provenance marker.

    Attributes:
        original_token_count: Tokens in the run being compacted.
        summary_token_count: Tokens in the rejected summary, when there was one.
    """

    code = "COMPACTION_REFUSED"
    retryable = True

    def __init__(
        self,
        message: str,
        *,
        original_token_count: int | None = None,
        summary_token_count: int | None = None,
    ) -> None:
        super().__init__(message)
        self.original_token_count = original_token_count
        self.summary_token_count = summary_token_count


class LLMError(LLMGCError):
    """Base class for failures of an outbound call to a model provider.

    Attributes:
        provider: Short name of the provider that failed, so a caller holding
            several can report which one without unwrapping the cause.
    """

    code = "LLM_ERROR"

    def __init__(self, message: str, *, provider: str | None = None) -> None:
        super().__init__(message)
        self.provider = provider


class LLMTimeoutError(LLMError):
    """The provider did not respond within the caller's timeout.

    Always retryable: a timeout says nothing about the request's validity.
    """

    code = "LLM_TIMEOUT"
    retryable = True


class LLMConnectionError(LLMError):
    """The request never reached the provider — DNS, TLS, or a refused connection."""

    code = "LLM_CONNECTION_FAILED"
    retryable = True


class LLMRateLimitError(LLMError):
    """The provider refused the call on a rate or spend limit.

    Attributes:
        retry_after_s: The provider's own advice, from `Retry-After`. Preferred
            over our backoff schedule — the provider knows when its window resets.
        retryable: True by default, but overridable: an exhausted account quota
            arrives as a 429 like any other and no wait clears it. Only the adapter
            that saw the error type can tell the two apart.
    """

    code = "LLM_RATE_LIMITED"
    retryable = True

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        retry_after_s: float | None = None,
        retryable: bool = True,
    ) -> None:
        super().__init__(message, provider=provider)
        self.retry_after_s = retry_after_s
        self.retryable = retryable


class LLMAuthError(LLMError):
    """The provider rejected our credentials (401/403).

    Never retryable, and separate from `LLMResponseError` for that reason: the same
    key will be rejected three times while the operator waits for the one message
    that tells them what to fix.
    """

    code = "LLM_AUTH_FAILED"


class LLMContextOverflowError(LLMError):
    """The request exceeded the model's context window.

    Its own type because it is actionable: the compactor should split the run and
    retry, which differs from backing off and repeating the identical call.

    Attributes:
        requested_tokens: Tokens in the rejected request, when the provider says.
        limit_tokens: The model's limit, when the provider says.
    """

    code = "LLM_CONTEXT_OVERFLOW"

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        requested_tokens: int | None = None,
        limit_tokens: int | None = None,
    ) -> None:
        super().__init__(message, provider=provider)
        self.requested_tokens = requested_tokens
        self.limit_tokens = limit_tokens


class LLMResponseError(LLMError):
    """The provider answered, but not with a usable completion.

    Covers a failing status, a body that is not the expected JSON, a missing field,
    and output failing validation — one category because the caller's options are
    identical: retry or degrade visibly.

    Attributes:
        status_code: The HTTP status, when the failure came from one.
        retryable: Set per-instance, because this category spans both: 503 is worth
            repeating and 400 never will be, and only the adapter saw the status.
    """

    code = "LLM_BAD_RESPONSE"

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message, provider=provider)
        self.status_code = status_code
        self.retryable = retryable
