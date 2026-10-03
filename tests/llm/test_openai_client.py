import asyncio

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
from llm_gc.llm.openai_client import OpenAIClient
from llm_gc.models import Message

# A minimal valid Chat Completions success envelope. Tests override the parts they
# care about so each one states only its own subject.
OK_BODY = {
    "id": "chatcmpl-01",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-5-mini-2026-01-01",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "a summary"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 120, "completion_tokens": 18, "total_tokens": 138},
}


def _body(**overrides) -> dict:
    return {**OK_BODY, **overrides}


def _choice(**overrides) -> list[dict]:
    """One choice with the given fields replaced, for the many single-choice cases."""
    base = dict(OK_BODY["choices"][0])  # type: ignore[arg-type]
    base.update(overrides)
    return [base]


def _client(handler, **kwargs) -> OpenAIClient:
    """An OpenAIClient whose transport is a scripted handler.

    MockTransport intercepts below httpx2's request machinery, so everything this
    class actually does — payload construction, header assembly, envelope validation,
    error mapping — runs for real. Mocking the client's own methods instead would
    test nothing but the mock.
    """
    transport = httpx2.MockTransport(handler)
    return OpenAIClient(
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


def _error_body(message: str = "something went wrong", *, type_: str = "invalid_request_error", code=None) -> dict:
    return {"error": {"message": message, "type": type_, "param": None, "code": code}}


def _call(client: OpenAIClient, **overrides):
    kwargs = {
        "messages": [Message(role="user", content="summarise this", token_count=5, turn_index=3)],
        "max_output_tokens": 512,
        "timeout_s": 10.0,
    }
    kwargs.update(overrides)
    return asyncio.run(client.generate(**kwargs))


def _sent(handler) -> dict:
    import json

    return json.loads(handler.seen[0].content)


class TestTheRequestShape:
    """The first of the three translations this adapter owns — and the half of the
    interface where OpenAI and Anthropic disagree most."""

    def test_system_becomes_a_message_not_a_top_level_field(self):
        """The mirror image of the Anthropic adapter, from the same `system=`
        argument. This divergence is the reason `LLMProvider` takes `system`
        separately instead of letting callers position it themselves."""
        handler = _respond()
        _call(_client(handler), system="you are a summariser")

        payload = _sent(handler)
        assert "system" not in payload
        assert payload["messages"][0] == {"role": "system", "content": "you are a summariser"}

    def test_system_message_precedes_the_conversation(self):
        handler = _respond()
        _call(_client(handler), system="rules")

        roles = [message["role"] for message in _sent(handler)["messages"]]
        assert roles == ["system", "user"]

    def test_no_system_message_is_added_when_absent(self):
        handler = _respond()
        _call(_client(handler))

        assert [message["role"] for message in _sent(handler)["messages"]] == ["user"]

    def test_llm_gc_bookkeeping_is_not_sent(self):
        """`token_count` and `turn_index` are ours. OpenAI rejects unknown keys
        inside a message object, so leaking them is a 400 on every call."""
        handler = _respond()
        _call(_client(handler))

        assert _sent(handler)["messages"][0] == {"role": "user", "content": "summarise this"}

    def test_the_output_cap_is_sent_as_max_completion_tokens(self):
        """Not `max_tokens`. Copying Anthropic's key here would work on older models
        and fail on every reasoning model, which is the worst kind of bug: it passes
        in whatever the author happened to test against."""
        handler = _respond()
        _call(_client(handler), max_output_tokens=256)

        payload = _sent(handler)
        assert payload["max_completion_tokens"] == 256
        assert "max_tokens" not in payload

    def test_the_configured_model_is_sent(self):
        handler = _respond()
        _call(_client(handler, model="gpt-5"))

        assert _sent(handler)["model"] == "gpt-5"

    def test_bearer_auth_is_used(self):
        handler = _respond()
        _call(_client(handler, api_key="sk-secret"))

        headers = handler.seen[0].headers
        assert headers["authorization"] == "Bearer sk-secret"
        assert headers["content-type"] == "application/json"

    def test_no_anthropic_version_header_leaks_across(self):
        """A copy-paste check with a real consequence: OpenAI ignores it, but a
        header named after another vendor in our requests is a sign the adapters
        have started sharing things they should not."""
        handler = _respond()
        _call(_client(handler))

        assert "anthropic-version" not in handler.seen[0].headers

    def test_the_organization_header_is_sent_only_when_configured(self):
        with_org = _respond()
        _call(_client(with_org, organization="org-123"))
        assert with_org.seen[0].headers["openai-organization"] == "org-123"

        without_org = _respond()
        _call(_client(without_org))
        assert "openai-organization" not in without_org.seen[0].headers

    def test_no_temperature_is_sent(self):
        """Not on the interface, so not on the wire — even though OpenAI would
        accept it. An adapter that quietly added it would make the two providers
        behave differently through one interface."""
        handler = _respond()
        _call(_client(handler))

        assert "temperature" not in _sent(handler)

    def test_the_request_goes_to_chat_completions(self):
        handler = _respond()
        _call(_client(handler, base_url="https://gateway.internal/"))

        assert str(handler.seen[0].url) == "https://gateway.internal/v1/chat/completions"


class TestParsingTheSuccessEnvelope:
    def test_text_is_unwrapped_from_the_first_choice(self):
        assert _call(_client(_respond())).text == "a summary"

    def test_usage_is_carried_through_under_our_names(self):
        """OpenAI says prompt/completion where Anthropic says input/output. Callers
        see one vocabulary."""
        response = _call(_client(_respond()))
        assert response.input_tokens == 120
        assert response.output_tokens == 18

    def test_the_model_reported_is_the_one_that_answered(self):
        """We asked for `gpt-5-mini`; the dated snapshot that served it is what goes
        into provenance. Providers route and alias."""
        response = _call(_client(_respond(), model="gpt-5-mini"))
        assert response.model == "gpt-5-mini-2026-01-01"

    def test_provider_is_stamped(self):
        assert _call(_client(_respond())).provider == "openai"

    def test_missing_usage_fails_rather_than_reporting_a_free_call(self):
        """Required fields, not defaulted to 0. A 0 would be filed as "this call cost
        nothing", which is a quiet lie in the one place the project promises
        honesty."""
        body = _body()
        del body["usage"]
        with pytest.raises(LLMResponseError):
            _call(_client(_respond(body=body)))

    @pytest.mark.parametrize(
        "openai_reason, expected",
        [
            ("stop", StopReason.COMPLETE),
            ("length", StopReason.TRUNCATED),
            ("content_filter", StopReason.FILTERED),
            ("tool_calls", StopReason.OTHER),
        ],
    )
    def test_finish_reasons_are_normalised(self, openai_reason, expected):
        body = _body(choices=_choice(finish_reason=openai_reason, message={"content": "text"}))
        assert _call(_client(_respond(body=body))).stop_reason is expected

    def test_length_becomes_truncated_and_therefore_not_complete(self):
        """The quiet failure this whole translation exists for: HTTP 200, a
        well-formed body, and a summary missing its end. Nothing in the transport
        layer flags it."""
        body = _body(choices=_choice(finish_reason="length", message={"content": "a summary that stops mid-"}))
        response = _call(_client(_respond(body=body)))

        assert response.stop_reason is StopReason.TRUNCATED
        assert response.is_complete is False

    def test_an_unknown_finish_reason_becomes_other_not_complete(self):
        body = _body(choices=_choice(finish_reason="some_new_reason"))
        response = _call(_client(_respond(body=body)))

        assert response.stop_reason is StopReason.OTHER
        assert response.is_complete is False

    def test_a_missing_finish_reason_becomes_other(self):
        body = _body(choices=_choice(finish_reason=None))
        assert _call(_client(_respond(body=body))).stop_reason is StopReason.OTHER

    def test_a_filtered_completion_may_legitimately_have_no_text(self):
        """`content: null` with a content filter is a real, correct response. It must
        not be mistaken for a malformed body."""
        body = _body(choices=_choice(finish_reason="content_filter", message={"content": None}))
        response = _call(_client(_respond(body=body)))

        assert response.text == ""
        assert response.stop_reason is StopReason.FILTERED

    def test_null_content_never_becomes_the_string_none(self):
        """The subtle version of the bug above: `str(None)` is `"None"`, so a sloppy
        unwrap files a four-character summary that reads as a word."""
        body = _body(choices=_choice(finish_reason="content_filter", message={"content": None}))
        assert _call(_client(_respond(body=body))).text == ""

    def test_empty_content_without_a_filter_is_an_error(self):
        """We asked for a completion and did not get one. Returning an empty summary
        would file nothing in place of a run of turns and report success."""
        body = _body(choices=_choice(finish_reason="stop", message={"content": ""}))
        with pytest.raises(LLMResponseError) as excinfo:
            _call(_client(_respond(body=body)))
        assert excinfo.value.retryable is True

    def test_an_empty_choices_list_is_an_error_not_an_index_error(self):
        """Schema-valid and useless: `list[_Choice]` permits empty, so Pydantic lets
        it through and `choices[0]` would raise an IndexError from inside a parser —
        an exception no caller of `generate` is told to catch."""
        with pytest.raises(LLMResponseError) as excinfo:
            _call(_client(_respond(body=_body(choices=[]))))
        assert "no choices" in str(excinfo.value)

    def test_a_body_missing_required_fields_is_a_response_error(self):
        body = _body()
        del body["model"]
        with pytest.raises(LLMResponseError) as excinfo:
            _call(_client(_respond(body=body)))
        assert excinfo.value.retryable is True

    def test_a_non_json_200_is_a_response_error(self):
        """A gateway can return 200 with an HTML error page."""
        with pytest.raises(LLMResponseError):
            _call(_client(_respond(body="<html>gateway</html>")))


class TestErrorMapping:
    """The second translation, and the one that actually ties code to a vendor."""

    @pytest.mark.parametrize("status", [401, 403])
    def test_credential_rejection_is_an_auth_error_and_never_retryable(self, status):
        handler = _respond(status, body=_error_body("Incorrect API key provided", type_="invalid_request_error"))
        with pytest.raises(LLMAuthError) as excinfo:
            _call(_client(handler))

        assert excinfo.value.retryable is False
        assert excinfo.value.provider == "openai"

    def test_429_is_a_rate_limit_carrying_the_providers_own_advice(self):
        handler = _respond(
            429,
            body=_error_body("Rate limit reached", type_="rate_limit_exceeded", code="rate_limit_exceeded"),
            headers={"retry-after": "12"},
        )
        with pytest.raises(LLMRateLimitError) as excinfo:
            _call(_client(handler))

        assert excinfo.value.retry_after_s == 12.0
        assert excinfo.value.retryable is True

    def test_an_exhausted_quota_is_a_429_that_is_not_retryable(self):
        """The finding this adapter produced. OpenAI reports "out of credit" with the
        same status as "slow down", and the retry policy's whole budget would go on a
        permanent failure — delaying the one error message that tells an operator
        what to fix. Only the adapter can tell them apart."""
        handler = _respond(
            429,
            body=_error_body("You exceeded your current quota", type_="insufficient_quota", code="insufficient_quota"),
        )
        with pytest.raises(LLMRateLimitError) as excinfo:
            _call(_client(handler))

        assert excinfo.value.retryable is False

    def test_the_rate_limit_class_default_is_still_retryable(self):
        """The per-instance override must not have flipped the default: a 429 with an
        unrecognised type is a window that reopens until something says otherwise."""
        handler = _respond(429, body=_error_body("Rate limit reached", type_="rate_limit_exceeded"))
        with pytest.raises(LLMRateLimitError) as excinfo:
            _call(_client(handler))

        assert excinfo.value.retryable is True

    def test_a_missing_retry_after_leaves_the_policy_to_its_own_schedule(self):
        handler = _respond(429, body=_error_body("Rate limit reached", type_="rate_limit_exceeded"))
        with pytest.raises(LLMRateLimitError) as excinfo:
            _call(_client(handler))

        assert excinfo.value.retry_after_s is None

    def test_an_unparseable_retry_after_is_ignored_rather_than_guessed(self):
        handler = _respond(
            429,
            body=_error_body("Rate limit reached", type_="rate_limit_exceeded"),
            headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"},
        )
        with pytest.raises(LLMRateLimitError) as excinfo:
            _call(_client(handler))

        assert excinfo.value.retry_after_s is None

    def test_context_overflow_is_recognised_from_the_error_code(self):
        """The one place OpenAI is easier to adapt than Anthropic: a machine-readable
        code instead of a prose search. Its own exception type because the
        compactor's response is to split the run, not to repeat the call."""
        handler = _respond(
            400,
            body=_error_body(
                "This model's maximum context length is 400000 tokens",
                type_="invalid_request_error",
                code="context_length_exceeded",
            ),
        )
        with pytest.raises(LLMContextOverflowError) as excinfo:
            _call(_client(handler))

        assert excinfo.value.retryable is False
        assert excinfo.value.provider == "openai"

    def test_overflow_numbers_are_not_invented(self):
        """None rather than a scraped guess. A wrong limit would send the compactor
        splitting runs by a figure it made up."""
        handler = _respond(400, body=_error_body("too long", code="context_length_exceeded"))
        with pytest.raises(LLMContextOverflowError) as excinfo:
            _call(_client(handler))

        assert excinfo.value.requested_tokens is None
        assert excinfo.value.limit_tokens is None

    def test_an_ordinary_400_is_not_mistaken_for_an_overflow(self):
        """Over-matching here is the expensive direction: the compactor would start
        splitting a run that was never too long, and keep failing in halves."""
        handler = _respond(400, body=_error_body("Invalid value for 'model'", code="model_not_found"))
        with pytest.raises(LLMResponseError) as excinfo:
            _call(_client(handler))

        assert not isinstance(excinfo.value, LLMContextOverflowError)
        assert excinfo.value.retryable is False
        assert excinfo.value.status_code == 400

    @pytest.mark.parametrize("status", [500, 502, 503, 504])
    def test_server_errors_are_retryable(self, status):
        handler = _respond(status, body=_error_body("The server had an error"))
        with pytest.raises(LLMResponseError) as excinfo:
            _call(_client(handler))

        assert excinfo.value.retryable is True

    def test_a_404_is_not_retryable(self):
        handler = _respond(404, body=_error_body("The model does not exist", code="model_not_found"))
        with pytest.raises(LLMResponseError) as excinfo:
            _call(_client(handler))

        assert excinfo.value.retryable is False

    def test_an_html_error_body_does_not_crash_the_mapper(self):
        """Error responses are the least reliable thing a provider sends — this one
        came from a load balancer. An exception raised while building an exception
        loses the status entirely."""
        handler = _respond(503, body="<html>502 Bad Gateway</html>")
        with pytest.raises(LLMResponseError) as excinfo:
            _call(_client(handler))

        assert excinfo.value.status_code == 503

    def test_an_empty_error_body_does_not_crash_the_mapper(self):
        handler = _respond(500, body="")
        with pytest.raises(LLMResponseError):
            _call(_client(handler))

    def test_an_error_body_with_no_error_object_does_not_crash_the_mapper(self):
        handler = _respond(500, body={"detail": "upstream failure"})
        with pytest.raises(LLMResponseError) as excinfo:
            _call(_client(handler))

        assert excinfo.value.status_code == 500

    def test_the_message_does_not_echo_the_prompt(self):
        """Exception strings reach logs, and the prompt is the user's conversation.
        CLAUDE.md §4: conversation content is not logged by default."""
        handler = _respond(500, body=_error_body("internal error"))
        with pytest.raises(LLMResponseError) as excinfo:
            _call(
                _client(handler),
                messages=[Message(role="user", content="my bank password is hunter2")],
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
        def handler(request):
            raise httpx2.ConnectError("connection refused", request=request)

        with pytest.raises(LLMConnectionError) as excinfo:
            _call(_client(handler))

        assert excinfo.value.retryable is True
        assert excinfo.value.provider == "openai"

    def test_the_original_cause_survives(self):
        original = httpx2.ConnectError("dns failure")

        def handler(request):
            raise original

        with pytest.raises(LLMConnectionError) as excinfo:
            _call(_client(handler))
        assert excinfo.value.__cause__ is original


class TestConfigurationAndValidation:
    def test_a_missing_api_key_fails_at_construction(self):
        with pytest.raises(ConfigError):
            OpenAIClient(api_key="")

    def test_a_blank_api_key_fails_at_construction(self):
        with pytest.raises(ConfigError):
            OpenAIClient(api_key="   ")

    def test_the_shared_preconditions_are_enforced(self):
        """OpenAI would happily accept a system-role message in the list. We reject
        it anyway: the interface's rules have to be the same at both adapters, or
        code written against one silently breaks on the other."""
        handler = _respond()
        with pytest.raises(ConfigError):
            _call(_client(handler), messages=[Message(role="system", content="you are a summariser")])

    def test_an_invalid_request_is_never_put_on_the_wire(self):
        handler = _respond()
        with pytest.raises(ConfigError):
            _call(_client(handler), messages=[])
        assert handler.seen == []


class TestConnectionLifetime:
    def test_an_injected_client_is_not_closed_by_us(self):
        borrowed = httpx2.AsyncClient(transport=httpx2.MockTransport(_respond()))
        client = OpenAIClient(api_key="sk-test", http_client=borrowed)

        asyncio.run(client.aclose())
        assert borrowed.is_closed is False
        asyncio.run(borrowed.aclose())

    def test_a_client_we_created_is_closed(self):
        client = OpenAIClient(api_key="sk-test")
        asyncio.run(client.aclose())
        assert client._http_client.is_closed is True

    def test_it_works_as_an_async_context_manager(self):
        async def scenario():
            async with OpenAIClient(api_key="sk-test") as client:
                inner = client._http_client
                assert inner.is_closed is False
            return inner.is_closed

        assert asyncio.run(scenario()) is True


class TestTheAbstractionHolds:
    """The reason this file exists. These tests do not care which provider answered —
    they assert that the two adapters are interchangeable through the interface."""

    def test_both_adapters_return_the_same_shape_from_different_envelopes(self):
        from llm_gc.llm.anthropic_client import AnthropicClient

        anthropic_body = {
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "a summary"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 120, "output_tokens": 18},
        }

        def anthropic_handler(request):
            return httpx2.Response(200, json=anthropic_body)

        anthropic = AnthropicClient(
            api_key="sk-a", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(anthropic_handler))
        )
        openai = _client(_respond())

        from_anthropic = _call(anthropic)
        from_openai = _call(openai)

        # Same text, same usage, same stop reason — from two bodies with no field in
        # common. If the interface had taken one vendor's shape, one of these would
        # need a provider-specific read at the call site.
        assert from_anthropic.text == from_openai.text == "a summary"
        assert (from_anthropic.input_tokens, from_anthropic.output_tokens) == (120, 18)
        assert (from_openai.input_tokens, from_openai.output_tokens) == (120, 18)
        assert from_anthropic.stop_reason is from_openai.stop_reason is StopReason.COMPLETE
        assert from_anthropic.is_complete and from_openai.is_complete
        # The one field that must differ, because provenance is the point.
        assert from_anthropic.provider != from_openai.provider

    def test_a_caller_can_hold_either_behind_the_protocol(self):
        """Written against `LLMProvider`, with no mention of a vendor. This is what
        `LLMCompactor` will look like."""
        from llm_gc.llm import FakeProvider, LLMProvider

        async def compact(provider: LLMProvider) -> str:
            response = await provider.generate(
                messages=[Message(role="user", content="summarise")],
                system="be brief",
                max_output_tokens=64,
                timeout_s=5.0,
            )
            return response.text if response.is_complete else ""

        providers: list[LLMProvider] = [_client(_respond()), FakeProvider()]
        results = [asyncio.run(compact(provider)) for provider in providers]

        assert all(result for result in results)
