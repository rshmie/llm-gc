import asyncio

import pytest

from llm_gc.exceptions import ConfigError, LLMRateLimitError
from llm_gc.llm import FakeProvider, FakeProviderExhausted, LLMResponse, StopReason
from llm_gc.llm.fake_provider import DEFAULT_FAKE_TEXT
from llm_gc.models import Message


def _msgs(*contents: str) -> list[Message]:
    return [Message(role="user", content=c, turn_index=i) for i, c in enumerate(contents)]


def _call(provider: FakeProvider, **overrides) -> LLMResponse:
    kwargs = {"messages": _msgs("summarise this"), "max_output_tokens": 100, "timeout_s": 5.0}
    kwargs.update(overrides)
    return asyncio.run(provider.generate(**kwargs))


def _response(**overrides) -> LLMResponse:
    fields = {
        "text": "scripted",
        "stop_reason": StopReason.COMPLETE,
        "input_tokens": 10,
        "output_tokens": 3,
        "model": "fake-model-1",
        "provider": "fake",
    }
    fields.update(overrides)
    return LLMResponse(**fields)


class TestTheFakeIsNotKinderThanReality:
    """The central constraint on this object. Every rule a real client enforces has
    to be enforced here too — otherwise a test suite that is green against the fake
    says nothing about whether the code works against Anthropic.
    """

    def test_an_empty_conversation_is_rejected(self):
        with pytest.raises(ConfigError):
            _call(FakeProvider(), messages=[])

    def test_a_system_role_message_in_the_list_is_rejected(self):
        """Anthropic has no system role in its messages array, so this is
        unrepresentable rather than merely unusual. If the fake accepted it, the
        bug would surface only when a real client was first wired in."""
        messages = [Message(role="system", content="you are a summariser"), *_msgs("hi")]
        with pytest.raises(ConfigError) as excinfo:
            _call(FakeProvider(), messages=messages)
        assert "system" in str(excinfo.value)

    def test_the_rejection_points_at_the_offending_message(self):
        """A conversation can be long. Saying which index is wrong is the
        difference between a two-second fix and a bisect."""
        messages = [*_msgs("a", "b"), Message(role="system", content="oops")]
        with pytest.raises(ConfigError) as excinfo:
            _call(FakeProvider(), messages=messages)
        assert "index 2" in str(excinfo.value)

    def test_a_nonsensical_output_cap_is_rejected(self):
        with pytest.raises(ConfigError):
            _call(FakeProvider(), max_output_tokens=0)

    def test_a_nonsensical_timeout_is_rejected(self):
        with pytest.raises(ConfigError):
            _call(FakeProvider(), timeout_s=0)

    def test_a_rejected_request_is_not_recorded_as_a_call(self):
        """It never went out. Logging it would let a test assert that a request
        was made when the opposite happened — and the assertion would pass."""
        provider = FakeProvider()
        with pytest.raises(ConfigError):
            _call(provider, messages=[])
        assert provider.calls == []


class TestUnscriptedMode:
    """For tests that need a provider to exist but do not care what it says."""

    def test_it_returns_a_complete_response(self):
        response = _call(FakeProvider())
        assert response.text == DEFAULT_FAKE_TEXT
        assert response.is_complete is True
        assert response.provider == "fake"

    def test_usage_comes_from_the_real_tokenizer(self):
        """A fake that always reported zero usage would let a cost-tracking bug
        pass every test. Longer input has to cost more, or the number is decoration."""
        short = _call(FakeProvider(), messages=_msgs("hi"))
        long = _call(FakeProvider(), messages=_msgs("hi " * 200))
        assert short.input_tokens > 0
        assert long.input_tokens > short.input_tokens

    def test_the_system_prompt_counts_toward_input_tokens(self):
        """It is sent to the model and billed, so omitting it would understate
        every compaction's cost by the length of a prompt we control."""
        without = _call(FakeProvider())
        with_system = _call(FakeProvider(), system="You are a careful summariser of technical conversations.")
        assert with_system.input_tokens > without.input_tokens

    def test_it_can_be_called_indefinitely(self):
        provider = FakeProvider()
        for _ in range(5):
            _call(provider)
        assert len(provider.calls) == 5


class TestScriptedMode:
    """One mechanism scripts both success and failure, the way `side_effect` does."""

    def test_responses_are_consumed_in_order(self):
        provider = FakeProvider(responses=[_response(text="first"), _response(text="second")])
        assert _call(provider).text == "first"
        assert _call(provider).text == "second"

    def test_an_exception_in_the_script_is_raised(self):
        """This is how a test drives the retry policy: script a 429 followed by a
        success and assert the caller recovered."""
        provider = FakeProvider(
            responses=[LLMRateLimitError("slow down", retry_after_s=0.0), _response(text="recovered")]
        )
        with pytest.raises(LLMRateLimitError):
            _call(provider)
        assert _call(provider).text == "recovered"

    def test_over_calling_a_script_raises_instead_of_repeating(self):
        """The behaviour that earns this class its keep. Over-calling is almost
        always the bug under test — a retry loop that does not terminate, a caller
        issuing one request per message where it meant to batch. Repeating the last
        response would make every one of those pass silently."""
        provider = FakeProvider(responses=[_response()])
        _call(provider)
        with pytest.raises(FakeProviderExhausted) as excinfo:
            _call(provider)
        assert "call 2" in str(excinfo.value)

    def test_exhaustion_is_catchable_as_an_llm_error(self):
        """It arrives where provider failures arrive, so a caller with a blanket
        `except LLMError` does not accidentally swallow it as something else."""
        from llm_gc.exceptions import LLMError

        provider = FakeProvider(responses=[])
        with pytest.raises(LLMError):
            _call(provider)

    def test_the_script_cannot_be_started_half_consumed(self):
        """`_next_index` is bookkeeping, not configuration."""
        with pytest.raises(TypeError):
            FakeProvider(responses=[_response()], _next_index=1)


class TestCallRecording:
    """Tests assert on what was *sent* at least as often as on what came back —
    that the right system prompt went out, that the timeout was the one configured.
    """

    def test_it_records_every_argument(self):
        provider = FakeProvider()
        _call(provider, system="be brief", max_output_tokens=512, timeout_s=20.0, temperature=0.2)

        call = provider.calls[0]
        assert call.system == "be brief"
        assert call.max_output_tokens == 512
        assert call.timeout_s == 20.0
        assert call.temperature == 0.2

    def test_the_recorded_messages_are_a_snapshot_not_a_reference(self):
        """The subtle one. Stored by reference, a caller that reuses and mutates
        its message list after the call would retroactively rewrite history — and
        an assertion about what was sent would describe the list's *current*
        contents, passing or failing for reasons unrelated to the call."""
        provider = FakeProvider()
        messages = _msgs("original")
        _call(provider, messages=messages)

        messages.append(Message(role="user", content="added afterwards"))
        messages[0] = Message(role="user", content="rewritten afterwards")

        assert len(provider.calls[0].messages) == 1
        assert provider.calls[0].messages[0].content == "original"
