import inspect

import pytest

from llm_gc import exceptions
from llm_gc.exceptions import (
    ConfigError,
    LLMAuthError,
    LLMContextOverflowError,
    LLMError,
    LLMGCError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    SessionLockTimeout,
)


def _exception_classes() -> list[type[BaseException]]:
    """Every exception class defined in the exceptions module itself.

    Introspected rather than listed by hand so a newly added exception is covered
    by the contract tests below without anyone remembering to update this file.
    """
    return [
        obj
        for _, obj in inspect.getmembers(exceptions, inspect.isclass)
        if issubclass(obj, BaseException) and obj.__module__ == exceptions.__name__
    ]


class TestTheCatchAllContract:
    """The point of a single base class: a caller wrapping LLM-GC can catch the
    whole library without enumerating its errors, and without accidentally
    swallowing unrelated bugs the way a bare `except Exception` would.
    """

    def test_every_exception_derives_from_the_base(self):
        """If one escapes the hierarchy, a caller's `except LLMGCError` silently
        stops covering it — and they find out in production, not at import."""
        for cls in _exception_classes():
            assert issubclass(cls, LLMGCError), f"{cls.__name__} is outside the hierarchy"

    def test_catching_the_base_catches_a_provider_failure(self):
        with pytest.raises(LLMGCError):
            raise LLMTimeoutError("provider took too long", provider="anthropic")

    def test_catching_llm_error_catches_every_provider_failure_but_not_config(self):
        """`LLMError` is the useful middle rung: 'the model call failed' without
        caring how. Config problems are deliberately outside it — they are fixed
        by an operator, not by retrying or degrading."""
        for cls in (LLMTimeoutError, LLMRateLimitError, LLMAuthError, LLMContextOverflowError, LLMResponseError):
            assert issubclass(cls, LLMError)
        assert not issubclass(ConfigError, LLMError)


class TestErrorCodes:
    """`code` is the stable identifier the HTTP boundary maps to a status and
    documents publicly. Its invariants matter more than they look.
    """

    def test_every_code_is_unique(self):
        """A duplicate does not raise anything — it silently merges two failure
        categories into one response, so an operator reading `LLM_BAD_RESPONSE`
        cannot tell which of two things went wrong. Uniqueness has to be pinned
        because nothing else enforces it."""
        codes = [cls.code for cls in _exception_classes()]
        duplicates = {code for code in codes if codes.count(code) > 1}
        assert not duplicates, f"error codes are not unique: {duplicates}"

    def test_every_code_is_a_non_empty_upper_snake_identifier(self):
        """Codes appear verbatim in API responses and documentation. Pinning the
        shape stops a lowercase or spaced code slipping into what is a public,
        stable contract."""
        for cls in _exception_classes():
            assert cls.code, f"{cls.__name__} has an empty code"
            assert cls.code.isupper(), f"{cls.__name__}'s code is not upper case"
            assert cls.code.replace("_", "").isalnum(), f"{cls.__name__}'s code has unexpected characters"


class TestRetryability:
    """The retry policy asks the error whether repeating the call is worth an
    attempt. These defaults are what it reads.
    """

    def test_timeouts_and_rate_limits_are_retryable(self):
        """Neither says anything is wrong with the request — only that the
        provider was busy."""
        assert LLMTimeoutError("timed out").retryable is True
        assert LLMRateLimitError("slow down").retryable is True

    def test_auth_failures_are_never_retryable(self):
        """The same key will be rejected three more times. Retrying spends the
        budget to reach a certainty and delays the error an operator can act on."""
        assert LLMAuthError("bad key").retryable is False

    def test_context_overflow_is_not_retryable(self):
        """Repeating an identical oversized request cannot succeed. The caller's
        useful move is to split the input, which is a different action."""
        assert LLMContextOverflowError("too long").retryable is False

    def test_response_errors_default_to_not_retryable(self):
        """Safe default: an adapter opts in to retrying deliberately, having seen
        the status code. Defaulting the other way would retry every malformed
        body and every HTTP 400."""
        assert LLMResponseError("unparseable body").retryable is False

    def test_a_retryable_response_error_does_not_leak_to_other_instances(self):
        """The per-instance flag has to shadow the class attribute, not reassign
        it. Written as a class-level assignment, one retryable 503 would silently
        make every later 400 retryable — a bug that only shows up as wasted retry
        budget on requests that can never succeed."""
        transient = LLMResponseError("upstream 503", status_code=503, retryable=True)
        permanent = LLMResponseError("upstream 400", status_code=400)

        assert transient.retryable is True
        assert permanent.retryable is False
        assert LLMResponseError.retryable is False

    def test_a_rate_limit_can_be_declared_permanent_without_changing_the_default(self):
        """The same shadowing, in the opposite direction. OpenAI reports an exhausted
        account quota as a 429 like any other, and retrying it never succeeds — so
        the adapter has to be able to say so. The class default stays True, because
        a rate limit is a window that reopens unless something tells us otherwise."""
        quota_exhausted = LLMRateLimitError("insufficient quota", retryable=False)
        throttled = LLMRateLimitError("slow down")

        assert quota_exhausted.retryable is False
        assert throttled.retryable is True
        assert LLMRateLimitError.retryable is True


class TestFailuresCarryTheirContext:
    """Each subclass exists to carry what its caller needs to act. These pin that
    the detail actually survives onto the instance.
    """

    def test_a_provider_failure_names_its_provider(self):
        """A caller holding two providers can report which one failed without
        unwrapping the cause."""
        assert LLMTimeoutError("timed out", provider="openai").provider == "openai"

    def test_provider_is_optional(self):
        """The fake and any caller-side raise site may not have one."""
        assert LLMTimeoutError("timed out").provider is None

    def test_rate_limit_carries_the_providers_own_advice(self):
        """`Retry-After` beats our backoff schedule: the provider knows when its
        limit window resets and we are guessing."""
        assert LLMRateLimitError("slow down", retry_after_s=12.0).retry_after_s == 12.0
        assert LLMRateLimitError("slow down").retry_after_s is None

    def test_context_overflow_carries_the_numbers_when_the_provider_gives_them(self):
        """These are what a caller needs to decide how far to split the run."""
        err = LLMContextOverflowError("too long", requested_tokens=210_000, limit_tokens=200_000)
        assert err.requested_tokens == 210_000
        assert err.limit_tokens == 200_000

    def test_response_error_carries_the_status_code(self):
        assert LLMResponseError("upstream 500", status_code=500).status_code == 500


class TestChaining:
    """`raise ... from err` is the convention at every adapter raise site."""

    def test_the_original_cause_survives_translation(self):
        """The provider's own traceback has to stay reachable for debugging, even
        though our message deliberately does not repeat its contents."""
        original = ValueError("connection reset")
        with pytest.raises(LLMTimeoutError) as excinfo:
            try:
                raise original
            except ValueError as err:
                raise LLMTimeoutError("provider timed out", provider="anthropic") from err

        assert excinfo.value.__cause__ is original


class TestSessionLockTimeout:
    """Moved here from `session_manager`, where it inherited bare `Exception`."""

    def test_it_is_now_inside_the_hierarchy(self):
        """Regression guard on the move: before it lived here, a caller's
        `except LLMGCError` would not have caught a lock timeout."""
        assert issubclass(SessionLockTimeout, LLMGCError)

    def test_the_message_names_the_session_and_the_budget(self):
        """Both are what an operator needs to tell a contended session from a
        budget set too low."""
        err = SessionLockTimeout("session-7", 100)
        assert "session-7" in str(err)
        assert "100" in str(err)
        assert err.session_id == "session-7"
        assert err.timeout_ms == 100
