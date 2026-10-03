"""The exception hierarchy for LLM-GC.

Every error this library raises derives from `LLMGCError`, so a caller can catch
the whole library with one `except` clause, or narrow to a category when it wants
to react differently to a timeout than to a bad API key.

Two conventions run through this module:

- **Retryability is carried by the error, not inferred by the caller.** See
  `LLMGCError.retryable`.
- **Errors never carry credentials.** An exception message ends up in logs and,
  after translation, in HTTP responses. Adapters put the status code and the
  provider name in the message; they never put the request headers or the API
  key in it.

Raise sites chain the original cause (`raise LLMTimeoutError(...) from err`) so
the provider's own traceback survives for debugging without leaking into the
message.
"""


class LLMGCError(Exception):
    """Base class for every error raised by LLM-GC.

    Attributes:
        code: A stable, protocol-agnostic identifier for this error category
            (e.g. `"LLM_TIMEOUT"`). It exists so the HTTP boundary can translate
            errors into `{"error": {"code": ...}}` responses from one mapping
            table instead of a chain of `isinstance` checks, and so the code can
            be documented and kept stable while the human-readable message is
            free to change. Deliberately *not* an HTTP status: statuses are the
            proxy's concern, and the library must stay usable by callers who
            never speak HTTP.
        retryable: Whether retrying the identical operation could plausibly
            succeed. Declared per-class as a default and overridden per-instance
            where the same category spans both (see `LLMResponseError`).
    """

    code: str = "LLM_GC_ERROR"
    retryable: bool = False


class ConfigError(LLMGCError):
    """Configuration is missing, malformed, or internally inconsistent.

    A programmer- or operator-facing error, raised at construction time rather
    than on the request path: a client built without an API key should fail when
    it is built, not on the first compaction under load.
    """

    code = "CONFIG_ERROR"


class SessionLockTimeout(LLMGCError):
    """Raised when a session's lock could not be acquired within the budget.

    This is an operational error, not a bug: it means another coroutine held
    the session's lock longer than `session_lock_timeout_ms`. The caller is
    expected to catch it and decide a fallback, not to crash.
    """

    code = "SESSION_LOCK_TIMEOUT"
    # The holder is expected to release shortly, so a later attempt may well win
    # the lock. Whether to retry is still the caller's call: on the live request
    # path, proceeding on stale state beats waiting twice.
    retryable = True

    def __init__(self, session_id: str, timeout_ms: float) -> None:
        super().__init__(f"Could not acquire lock for session {session_id!r} within {timeout_ms}ms")
        self.session_id = session_id
        self.timeout_ms = timeout_ms


class CompactionRefused(LLMGCError):
    """A compactor declined to file the summary it was handed.

    Not a provider failure — the call succeeded and returned something. It is the
    compactor judging that what came back must not stand in for the turns it
    would replace: output that stopped at the token cap and is therefore missing
    its tail, output that is empty, or a "summary" no shorter than its input.

    Raised rather than returned, because the caller's correct response is to do
    nothing. `GcService._age_out_turns` catches it per turn and leaves the turn in
    the young generation, verbatim, for the next pass — so a refusal costs one
    wasted call and loses nothing. Returning a degraded `CompactionResult`
    instead would file the bad summary and delete the originals, which is the one
    outcome this project must never produce silently.

    Attributes:
        original_token_count: Tokens in the run that was being compacted.
        summary_token_count: Tokens in the rejected summary, when there was one.
    """

    code = "COMPACTION_REFUSED"
    # Retryable means "a later attempt can succeed", NOT "repeat this call". The
    # identical call mostly reproduces the identical refusal, and the useful change
    # differs by case: a truncated summary wants a *smaller* run or a larger cap,
    # while one that came back no shorter than its input wants a *larger* run, so
    # the fixed-size provenance marker is amortised over more turns. The session
    # path gets the second for free — a refused run stays in the young generation,
    # and the next pass re-sweeps it with more turns cooled, so the run grows.
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

    Callers that only care "did the model answer?" catch this. The subclasses
    exist for the callers that need to tell a rate limit from a bad key, and for
    the retry policy, which needs to tell a wasted attempt from a useful one.

    Attributes:
        provider: Short name of the provider the call went to (`"anthropic"`,
            `"openai"`, `"fake"`). Populated by the adapter so a caller holding
            several providers can report which one failed without unwrapping the
            cause.
    """

    code = "LLM_ERROR"

    def __init__(self, message: str, *, provider: str | None = None) -> None:
        super().__init__(message)
        self.provider = provider


class LLMTimeoutError(LLMError):
    """The provider did not respond within the caller-supplied timeout.

    Always retryable: a timeout carries no information about the request's
    validity, only about how busy the provider was.
    """

    code = "LLM_TIMEOUT"
    retryable = True


class LLMConnectionError(LLMError):
    """The request never reached the provider — DNS, TLS, or a refused connection.
    Retryable — transient network faults are the common case.
    """

    code = "LLM_CONNECTION_FAILED"
    retryable = True


class LLMRateLimitError(LLMError):
    """The provider rejected the call for exceeding a rate or spend limit.

    Attributes:
        retry_after_s: The provider's own advice on how long to wait, parsed
            from the `Retry-After` header when present. The retry policy honours
            this in preference to its own backoff schedule — the provider knows
            when its limit window resets and we are guessing.
        retryable: True by default, because a rate limit is a window that
            reopens. Overridable per-instance because one case genuinely is not:
            an exhausted account quota arrives from OpenAI as a 429 like any
            other, and no amount of waiting clears it. Only the adapter that saw
            the error type can tell the two apart, so the adapter says so here
            rather than leaving every retry policy to re-derive it.
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
        # Shadows the class attribute for this instance only.
        self.retryable = retryable


class LLMAuthError(LLMError):
    """The provider rejected our credentials (HTTP 401/403).

    Never retryable, and separated from `LLMResponseError` for that reason: the
    same key will be rejected three times in a row, so retrying spends the budget
    to arrive at a certainty while delaying the one error message that tells an
    operator what to actually fix.
    """

    code = "LLM_AUTH_FAILED"


class LLMContextOverflowError(LLMError):
    """The request exceeded the model's context window.

    Its own type because for this project it is an actionable signal rather than
    a generic failure: the compactor asking a model to summarise a run larger
    than the model can read should split the run and try again, which is a
    different response from backing off and repeating the identical call.

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
    """The provider answered, but the answer is not a usable completion.

    Covers a failing status code, a body that is not the JSON we expect, a
    missing field, and output that fails schema validation. These are one
    category because the caller's options are identical in every case: retry or
    degrade visibly.

    Retryability is set per-instance rather than per-class, because this category
    spans both: HTTP 503 is worth repeating and HTTP 400 is not, and
    only the adapter that saw the status code can tell them apart. A retry policy
    matching on type alone would have to re-derive that from information the
    exception never carried.

    Attributes:
        status_code: The HTTP status, when the failure came from one.
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
        # Shadows the class attribute for this instance only; the class default
        # stays False so an adapter must opt in to retrying deliberately.
        self.retryable = retryable
