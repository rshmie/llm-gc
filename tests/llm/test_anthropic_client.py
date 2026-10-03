import asyncio
import json

import httpx2
import pytest

from llm_gc.exceptions import (
    ConfigError,
    LLMAuthError,
    LLMConnectionError,
    LLMContextOverflowError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)
from llm_gc.llm import StopReason
from llm_gc.llm.anthropic_client import ANTHROPIC_API_VERSION, AnthropicClient
from llm_gc.models import Message

# A minimal valid Messages API success envelope. Tests override the parts they care
# about so each one states only its own subject.
OK_BODY = {
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5",
    "content": [{"type": "text", "text": "a summary"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 120, "output_tokens": 18},
}


def _body(**overrides) -> dict:
    return {**OK_BODY, **overrides}


def _client(handler, **kwargs) -> AnthropicClient:
    """An AnthropicClient whose transport is a scripted handler.

    MockTransport intercepts below httpx2's request machinery, so everything this
    class actually does — payload construction, header assembly, envelope validation,
    error mapping — runs for real. Mocking the client's own methods instead would
    test nothing but the mock.
    """
    transport = httpx2.MockTransport(handler)
    return AnthropicClient(
        api_key=kwargs.pop("api_key", "sk-test"),
        http_client=httpx2.AsyncClient(transport=transport),
        **kwargs,
    )


def _respond(status: int = 200, body: dict | str | None = None, headers: dict | None = None):
    """Build a handler that always answers the same way, and records the request."""
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        payload = OK_BODY if body is None else body
        if isinstance(payload, str):
            return httpx2.Response(status, text=payload, headers=headers or {})
        return httpx2.Response(status, json=payload, headers=headers or {})

    handler.seen = seen  # type: ignore[attr-defined]
    return handler


def _call(client: AnthropicClient, **overrides):
    kwargs = {
        "messages": [Message(role="user", content="summarise this", token_count=5, turn_index=3)],
        "max_output_tokens": 512,
        "timeout_s": 10.0,
    }
    kwargs.update(overrides)
    return asyncio.run(client.generate(**kwargs))


class TestTheRequestShape:
    """The first of the three translations this adapter owns."""

    def test_system_goes_top_level_not_into_messages(self):
        """Anthropic has no system role in its messages array, so a system prompt
        placed there would be rejected. This is the difference the Protocol's
        separate `system` parameter exists to absorb."""
        handler = _respond()
        _call(_client(handler), system="be concise")

        sent = json.loads(handler.seen[0].content)
        assert sent["system"] == "be concise"
        assert all(m["role"] != "system" for m in sent["messages"])

    def test_system_is_omitted_entirely_when_absent(self):
        """Sending `"system": null` is not the same as not sending it."""
        handler = _respond()
        _call(_client(handler))
        assert "system" not in json.loads(handler.seen[0].content)

    def test_llm_gc_bookkeeping_is_not_sent(self):
        """`token_count` and `turn_index` are ours. Anthropic would reject them as
        unknown fields, so the payload must carry role and content only."""
        handler = _respond()
        _call(_client(handler))

        sent_message = json.loads(handler.seen[0].content)["messages"][0]
        assert set(sent_message) == {"role", "content"}

    def test_the_output_cap_is_sent_as_max_tokens(self):
        """Anthropic's `max_tokens` means the output cap, not a context window — the
        same name collision that got GCConfig.max_tokens renamed to context_window."""
        handler = _respond()
        _call(_client(handler), max_output_tokens=256)
        assert json.loads(handler.seen[0].content)["max_tokens"] == 256

    def test_the_configured_model_is_sent(self):
        handler = _respond()
        _call(_client(handler, model="claude-haiku-4-5"))
        assert json.loads(handler.seen[0].content)["model"] == "claude-haiku-4-5"

    def test_required_headers_are_present(self):
        """`anthropic-version` is not optional: Anthropic pins breaking changes
        behind it rather than behind a URL version."""
        handler = _respond()
        _call(_client(handler))

        headers = handler.seen[0].headers
        assert headers["x-api-key"] == "sk-test"
        assert headers["anthropic-version"] == ANTHROPIC_API_VERSION

    def test_no_temperature_is_sent(self):
        """Current Claude models reject `temperature` with a 400. The Protocol has no
        such parameter for exactly this reason; this pins that we never add one back
        by accident."""
        handler = _respond()
        _call(_client(handler))
        assert "temperature" not in json.loads(handler.seen[0].content)


class TestParsingTheSuccessEnvelope:
    def test_text_is_unwrapped_from_the_content_block_list(self):
        result = _call(_client(_respond()))
        assert result.text == "a summary"

    def test_several_text_blocks_are_joined(self):
        body = _body(content=[{"type": "text", "text": "first "}, {"type": "text", "text": "second"}])
        assert _call(_client(_respond(body=body))).text == "first second"

    def test_a_leading_non_text_block_does_not_break_extraction(self):
        """A thinking block carries no text. Taking content[0].text would return None
        here; filtering by type is what makes this robust to block order."""
        body = _body(content=[{"type": "thinking"}, {"type": "text", "text": "the summary"}])
        assert _call(_client(_respond(body=body))).text == "the summary"

    def test_usage_is_carried_through(self):
        result = _call(_client(_respond()))
        assert (result.input_tokens, result.output_tokens) == (120, 18)

    def test_the_model_reported_is_the_one_that_answered(self):
        """Not the one we asked for. Providers route and alias, and provenance should
        record what served the request."""
        body = _body(model="claude-sonnet-5-20260514")
        result = _call(_client(_respond(body=body), model="claude-sonnet-5"))
        assert result.model == "claude-sonnet-5-20260514"

    def test_provider_is_stamped(self):
        assert _call(_client(_respond())).provider == "anthropic"

    @pytest.mark.parametrize(
        ("anthropic_reason", "expected"),
        [
            ("end_turn", StopReason.COMPLETE),
            ("stop_sequence", StopReason.COMPLETE),
            ("max_tokens", StopReason.TRUNCATED),
            ("refusal", StopReason.FILTERED),
        ],
    )
    def test_stop_reasons_are_normalised(self, anthropic_reason, expected):
        body = _body(stop_reason=anthropic_reason)
        assert _call(_client(_respond(body=body))).stop_reason is expected

    def test_max_tokens_becomes_truncated_and_therefore_not_complete(self):
        """The case that matters most. HTTP 200, plausible text, and a summary that
        has silently lost its tail. `is_complete` is what stops the compactor filing
        it as whole."""
        body = _body(stop_reason="max_tokens", content=[{"type": "text", "text": "the run covered pool sizing and"}])
        result = _call(_client(_respond(body=body)))
        assert result.stop_reason is StopReason.TRUNCATED
        assert result.is_complete is False

    def test_an_unknown_stop_reason_becomes_other_not_complete(self):
        """A provider can add a stop reason tomorrow. Mapping the unfamiliar value to
        COMPLETE would assert the output is whole with no basis for it."""
        body = _body(stop_reason="some_future_reason")
        result = _call(_client(_respond(body=body)))
        assert result.stop_reason is StopReason.OTHER
        assert result.is_complete is False

    def test_a_missing_stop_reason_becomes_other(self):
        body = _body(stop_reason=None)
        assert _call(_client(_respond(body=body))).stop_reason is StopReason.OTHER

    def test_a_filtered_response_may_legitimately_have_no_text(self):
        body = _body(stop_reason="refusal", content=[], usage={"input_tokens": 40, "output_tokens": 0})
        result = _call(_client(_respond(body=body)))
        assert result.text == ""
        assert result.stop_reason is StopReason.FILTERED

    def test_empty_text_without_a_refusal_is_an_error(self):
        """We asked for a completion and did not get one. Returning an empty summary
        would have the compactor file a promotion that threw the turns away."""
        body = _body(content=[], stop_reason="end_turn")
        with pytest.raises(LLMResponseError):
            _call(_client(_respond(body=body)))

    def test_a_body_missing_required_fields_is_a_response_error(self):
        """Validated at the boundary, so a malformed envelope surfaces here rather
        than as a KeyError or a None three layers downstream."""
        with pytest.raises(LLMResponseError) as excinfo:
            _call(_client(_respond(body={"id": "msg_01"})))
        assert excinfo.value.retryable is True

    def test_a_non_json_200_is_a_response_error(self):
        """A 200 whose body is HTML is usually a proxy in the path, which is worth
        one more attempt."""
        with pytest.raises(LLMResponseError) as excinfo:
            _call(_client(_respond(body="<html>gateway</html>")))
        assert excinfo.value.retryable is True


class TestErrorMapping:
    """The second translation, and the one the design doc calls the dangerous one:
    what actually couples code to a provider is the failure path.
    """

    @pytest.mark.parametrize("status", [401, 403])
    def test_credential_rejection_is_an_auth_error_and_never_retryable(self, status):
        """The same key will be rejected again. Retrying spends the budget to reach a
        certainty while delaying the one message an operator can act on."""
        body = {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}
        with pytest.raises(LLMAuthError) as excinfo:
            _call(_client(_respond(status=status, body=body)))
        assert excinfo.value.retryable is False

    def test_429_is_a_rate_limit_carrying_the_providers_own_advice(self):
        """`Retry-After` beats our backoff: the provider knows when its window
        resets and we are guessing."""
        body = {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}
        with pytest.raises(LLMRateLimitError) as excinfo:
            _call(_client(_respond(status=429, body=body, headers={"retry-after": "7"})))

        assert excinfo.value.retryable is True
        assert excinfo.value.retry_after_s == 7.0

    def test_a_missing_retry_after_leaves_the_policy_to_its_own_schedule(self):
        body = {"type": "error", "error": {"message": "slow down"}}
        with pytest.raises(LLMRateLimitError) as excinfo:
            _call(_client(_respond(status=429, body=body)))
        assert excinfo.value.retry_after_s is None

    def test_an_unparseable_retry_after_is_ignored_rather_than_guessed(self):
        """The HTTP-date form is legal. Reporting a wrong number would be worse than
        reporting none, because the policy would trust it."""
        body = {"type": "error", "error": {"message": "slow down"}}
        headers = {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}
        with pytest.raises(LLMRateLimitError) as excinfo:
            _call(_client(_respond(status=429, body=body, headers=headers)))
        assert excinfo.value.retry_after_s is None

    def test_context_overflow_is_recognised_and_its_numbers_read(self):
        """Its own type because it is actionable: the compactor should split the run,
        which is a different response from repeating the identical call."""
        body = {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "prompt is too long: 250,000 tokens > 200,000 maximum"},
        }
        with pytest.raises(LLMContextOverflowError) as excinfo:
            _call(_client(_respond(status=400, body=body)))

        assert excinfo.value.requested_tokens == 250_000
        assert excinfo.value.limit_tokens == 200_000
        assert excinfo.value.retryable is False

    def test_an_overflow_without_readable_numbers_still_classifies(self):
        """The numbers are scraped from prose, so they are optional. Knowing it
        overflowed is the part the caller cannot do without."""
        body = {"type": "error", "error": {"message": "prompt is too long"}}
        with pytest.raises(LLMContextOverflowError) as excinfo:
            _call(_client(_respond(status=400, body=body)))
        assert excinfo.value.requested_tokens is None

    def test_an_ordinary_400_is_not_mistaken_for_an_overflow(self):
        """Substring matching is fragile, so it is kept narrow. Mislabelling a plain
        400 would send the compactor off splitting a run that was never too long."""
        body = {"type": "error", "error": {"message": "messages: unexpected role 'bot'"}}
        with pytest.raises(LLMResponseError) as excinfo:
            _call(_client(_respond(status=400, body=body)))

        assert not isinstance(excinfo.value, LLMContextOverflowError)
        assert excinfo.value.retryable is False
        assert excinfo.value.status_code == 400

    @pytest.mark.parametrize("status", [500, 502, 503, 529])
    def test_server_errors_are_retryable(self, status):
        body = {"type": "error", "error": {"message": "overloaded"}}
        with pytest.raises(LLMResponseError) as excinfo:
            _call(_client(_respond(status=status, body=body)))
        assert excinfo.value.retryable is True

    def test_a_404_is_not_retryable(self):
        body = {"type": "error", "error": {"message": "model not found"}}
        with pytest.raises(LLMResponseError) as excinfo:
            _call(_client(_respond(status=404, body=body)))
        assert excinfo.value.retryable is False

    def test_an_html_error_body_does_not_crash_the_mapper(self):
        """An error response is the least reliable thing a provider sends — it can
        come from a load balancer as HTML. An exception raised while building an
        exception would lose the status entirely, which is the real failure."""
        with pytest.raises(LLMResponseError) as excinfo:
            _call(_client(_respond(status=502, body="<html>502 Bad Gateway</html>")))
        assert excinfo.value.status_code == 502

    def test_an_empty_error_body_does_not_crash_the_mapper(self):
        with pytest.raises(LLMResponseError) as excinfo:
            _call(_client(_respond(status=503, body="")))
        assert excinfo.value.status_code == 503

    def test_the_message_does_not_echo_the_prompt(self):
        """Exception strings reach logs, and conversation content is user data."""
        body = {"type": "error", "error": {"message": "bad request"}}
        with pytest.raises(LLMResponseError) as excinfo:
            _call(
                _client(_respond(status=400, body=body)),
                messages=[Message(role="user", content="my secret passphrase is hunter2")],
            )
        assert "hunter2" not in str(excinfo.value)


class TestNetworkFailures:
    def test_a_timeout_becomes_an_llm_timeout_error(self):
        def handler(request):
            raise httpx2.ReadTimeout("too slow", request=request)

        with pytest.raises(LLMTimeoutError) as excinfo:
            _call(_client(handler), timeout_s=2.5)

        assert excinfo.value.retryable is True
        assert "2.5" in str(excinfo.value)

    def test_a_refused_connection_becomes_a_connection_error(self):
        """Distinct from both neighbours: the provider did not answer badly and was
        not slow — we never reached it. For an operator that is the difference
        between checking their own network and checking a status page."""
        def handler(request):
            raise httpx2.ConnectError("connection refused", request=request)

        with pytest.raises(LLMConnectionError) as excinfo:
            _call(_client(handler))

        assert excinfo.value.retryable is True
        assert excinfo.value.provider == "anthropic"

    def test_the_original_cause_survives(self):
        """The provider's own traceback stays reachable for debugging even though our
        message deliberately does not repeat its contents."""
        original = httpx2.ConnectError("dns failure")

        def handler(request):
            raise original

        with pytest.raises(LLMConnectionError) as excinfo:
            _call(_client(handler))
        assert excinfo.value.__cause__ is original


class TestConfigurationAndValidation:
    def test_a_missing_api_key_fails_at_construction(self):
        """Not on the first compaction under load. A misconfigured deployment should
        fail while someone is still watching it start."""
        with pytest.raises(ConfigError):
            AnthropicClient(api_key="")

    def test_a_blank_api_key_fails_at_construction(self):
        with pytest.raises(ConfigError):
            AnthropicClient(api_key="   ")

    def test_the_shared_preconditions_are_enforced(self):
        """Same validation the fake runs, which is the point: a request rejected in
        tests has to be rejected here too, or the suite is lying."""
        handler = _respond()
        client = _client(handler)
        with pytest.raises(ConfigError):
            _call(client, messages=[Message(role="system", content="you are a summariser")])

    def test_an_invalid_request_is_never_put_on_the_wire(self):
        handler = _respond()
        client = _client(handler)
        with pytest.raises(ConfigError):
            _call(client, messages=[])
        assert handler.seen == []


class TestConnectionLifetime:
    def test_an_injected_client_is_not_closed_by_us(self):
        """Whoever created it owns its lifetime. Closing a caller's pooled client
        would break every other user of it — and the symptom would appear somewhere
        else entirely."""
        borrowed = httpx2.AsyncClient(transport=httpx2.MockTransport(_respond()))
        client = AnthropicClient(api_key="sk-test", http_client=borrowed)

        asyncio.run(client.aclose())
        assert borrowed.is_closed is False
        asyncio.run(borrowed.aclose())

    def test_a_client_we_created_is_closed(self):
        client = AnthropicClient(api_key="sk-test")
        asyncio.run(client.aclose())
        assert client._http_client.is_closed is True

    def test_it_works_as_an_async_context_manager(self):
        async def scenario():
            async with AnthropicClient(api_key="sk-test") as client:
                inner = client._http_client
                assert inner.is_closed is False
            return inner.is_closed

        assert asyncio.run(scenario()) is True
