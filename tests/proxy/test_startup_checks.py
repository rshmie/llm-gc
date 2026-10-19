"""The summariser check, against the real client and its error mapping, with no network."""

import asyncio
import json

import httpx2
import pytest

from llm_gc.exceptions import ConfigError, LLMAuthError, LLMRequestRefusedError
from llm_gc.llm import AnthropicClient
from llm_gc.proxy.startup_checks import CHECK_OUTPUT_TOKENS, check_summariser

ANSWER = {
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "model": "gateway-sonnet",
    "content": [{"type": "text", "text": "OK"}],
    "stop_reason": "max_tokens",
    "usage": {"input_tokens": 9, "output_tokens": 1},
}


def _check(handler) -> str | None:
    client = AnthropicClient(
        api_key="sk-test",
        model="gateway-sonnet",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    return asyncio.run(check_summariser(client))


def _status(status: int, message: str):
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": message}}
    return lambda request: httpx2.Response(status, json=body)


def test_an_answer_passes_and_asks_for_one_token_of_the_configured_model():
    seen: list[httpx2.Request] = []

    def answer(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json=ANSWER)

    assert _check(answer) is None
    body = json.loads(seen[0].content)
    assert (body["model"], body["max_tokens"]) == ("gateway-sonnet", CHECK_OUTPUT_TOKENS)


def test_a_model_the_upstream_does_not_offer_stops_startup_with_its_own_words():
    """The case that failed silently at every collection: a gateway that names models its own way."""
    with pytest.raises(ConfigError, match="invalid model specified") as excinfo:
        _check(_status(400, "invalid model specified: claude-sonnet-5"))
    assert isinstance(excinfo.value.__cause__, LLMRequestRefusedError)


def test_an_account_out_of_credit_stops_startup_with_its_own_words():
    """Else every collection fails for the same reason, and only the dashboard says so."""
    body = {"type": "error", "error": {"type": "billing_error", "message": "Your credit balance is too low."}}
    with pytest.raises(ConfigError, match="credit balance is too low") as excinfo:
        _check(lambda request: httpx2.Response(402, json=body))
    assert isinstance(excinfo.value.__cause__, LLMRequestRefusedError)


def test_a_refused_credential_stops_startup():
    with pytest.raises(ConfigError, match="credential") as excinfo:
        _check(_status(401, "invalid x-api-key"))
    assert isinstance(excinfo.value.__cause__, LLMAuthError)


@pytest.mark.parametrize("status", [429, 503, 529])
def test_a_busy_upstream_warns_and_lets_the_proxy_start(status):
    """It may pass; refusing to start would make a provider's bad minute an outage."""
    warning = _check(_status(status, "overloaded"))
    assert warning is not None and "unchecked" in warning


def test_an_unreachable_upstream_warns_and_lets_the_proxy_start():
    def unreachable(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("no route to host")

    warning = _check(unreachable)
    assert warning is not None and "unchecked" in warning
