import asyncio

import pytest
from pydantic import ValidationError

from llm_gc.llm import LLMProvider, LLMResponse, StopReason
from llm_gc.models import Message


def _response(**overrides) -> LLMResponse:
    fields = {
        "text": "a summary",
        "stop_reason": StopReason.COMPLETE,
        "input_tokens": 120,
        "output_tokens": 18,
        "model": "claude-sonnet-4-6",
        "provider": "anthropic",
    }
    fields.update(overrides)
    return LLMResponse(**fields)


class TestStopReasonNormalisation:
    """Providers disagree on the words for 'why it stopped'. These values are
    ours, and they are serialized to the dashboard, so they are a contract.
    """

    def test_the_vocabulary_is_stable(self):
        assert StopReason.COMPLETE.value == "complete"
        assert StopReason.TRUNCATED.value == "truncated"
        assert StopReason.FILTERED.value == "filtered"
        assert StopReason.OTHER.value == "other"

    def test_there_is_an_escape_hatch_for_unknown_reasons(self):
        """A provider can add a stop reason tomorrow. Without OTHER, an adapter
        would have to map the unfamiliar value onto one of the known ones — and
        the tempting default is COMPLETE, which asserts the output is whole when
        we have no idea whether it is."""
        assert StopReason.OTHER in StopReason


class TestIsComplete:
    """The single most forgettable check on this object, which is why it is a
    named property and not a comparison repeated at each call site.
    """

    def test_a_finished_generation_is_complete(self):
        assert _response(stop_reason=StopReason.COMPLETE).is_complete is True

    def test_a_truncated_generation_is_not_complete(self):
        """The load-bearing case. A summary that hit the output cap has dropped
        the tail of the run it was summarising, while the HTTP call succeeded and
        the text field looks perfectly reasonable. Nothing else in the system will
        notice."""
        assert _response(stop_reason=StopReason.TRUNCATED).is_complete is False

    def test_a_filtered_or_unknown_generation_is_not_complete(self):
        """Neither can be asserted whole, so neither may report itself as such."""
        assert _response(stop_reason=StopReason.FILTERED).is_complete is False
        assert _response(stop_reason=StopReason.OTHER).is_complete is False


class TestUsageIsReportedHonestly:
    """Token counts feed cost tracking, so the ways they can quietly become
    wrong matter more than the ways they can be right.
    """

    def test_usage_has_no_default(self):
        """A provider that reported no usage must force the adapter to decide what
        to do. Defaulting to 0 would record 'this call was free', and the error
        compounds silently across every call in a session."""
        with pytest.raises(ValidationError):
            LLMResponse(
                text="x", stop_reason=StopReason.COMPLETE, model="m", provider="p", output_tokens=1
            )  # input_tokens omitted

    def test_negative_usage_is_rejected(self):
        """Not reachable from a well-behaved provider — which is the point. It
        would mean an adapter subtracted or mis-parsed, and the bound catches that
        at the boundary rather than downstream in a cost total."""
        with pytest.raises(ValidationError):
            _response(input_tokens=-1)
        with pytest.raises(ValidationError):
            _response(output_tokens=-1)

    def test_zero_usage_is_allowed(self):
        """Legitimately possible: a filtered response can generate nothing."""
        assert _response(output_tokens=0, stop_reason=StopReason.FILTERED).output_tokens == 0


class TestTheResponseIsARecord:
    def test_it_is_frozen(self):
        """It describes something that already happened. Editing it after the fact
        would let a caller launder its own value into what reads as the provider's
        report."""
        with pytest.raises(ValidationError):
            _response().text = "rewritten"

    def test_it_records_which_model_answered(self):
        """Providers route and alias, so provenance records what served the
        request, not what was asked for."""
        assert _response(model="claude-sonnet-4-6-20260514").model == "claude-sonnet-4-6-20260514"


class TestTheProtocolIsImplementable:
    """Structural conformance is mypy's job, not pytest's — this pins the part
    mypy cannot see: that the shape actually works when awaited.
    """

    def test_a_conforming_implementation_can_be_awaited(self):
        class StubProvider:
            name = "stub"

            async def generate(
                self,
                *,
                messages: list[Message],
                system: str | None = None,
                max_output_tokens: int,
                timeout_s: float,
            ) -> LLMResponse:
                return _response(text=f"saw {len(messages)} messages", provider=self.name)

        provider: LLMProvider = StubProvider()
        result = asyncio.run(
            provider.generate(messages=[Message(role="user", content="hi")], max_output_tokens=100, timeout_s=5.0)
        )

        assert result.text == "saw 1 messages"
        assert result.provider == "stub"

    def test_timeout_and_max_output_tokens_are_required_of_callers(self):
        """Both are undefaulted on purpose: a forgettable timeout is one nobody
        tunes, and an uncapped generation is a cost incident. A caller that omits
        either should fail loudly at the call, not fall back to a house default."""

        class StubProvider:
            name = "stub"

            async def generate(
                self,
                *,
                messages: list[Message],
                system: str | None = None,
                max_output_tokens: int,
                timeout_s: float,
            ) -> LLMResponse:
                return _response()

        with pytest.raises(TypeError):
            asyncio.run(StubProvider().generate(messages=[], max_output_tokens=10))  # no timeout_s
