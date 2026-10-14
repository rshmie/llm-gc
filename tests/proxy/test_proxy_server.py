"""The proxy, driven through its real routes with no network.

`MockTransport` intercepts below httpx2's request machinery, so parsing, session
resolution, GC, header filtering and forwarding all run for real. The one thing
stubbed is the socket.
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta, timezone

import httpx2
import pytest
from fastapi.testclient import TestClient

from llm_gc.config import GCConfig
from llm_gc.engine.switch_policy import CacheState, SwitchReason
from llm_gc.engine import build_gc_service
from llm_gc.engine.generations.front import FRONT_HEADER
from llm_gc.events import EventBus, EventType
from llm_gc.llm import FakeProvider, LLMResponse, StopReason
from llm_gc.models import ClientOrigin
from llm_gc.proxy import SESSION_HEADER, create_app
from llm_gc.proxy.anthropic_translation import fixed_prompt_tokens
from llm_gc.proxy.cache_watch import CacheWatch
from llm_gc.proxy.session_identity import CLIENT_AGENT_HEADER, CLIENT_SESSION_HEADER
from llm_gc.session import SessionManager

API_KEY = "sk-ant-super-secret"

UPSTREAM_OK = {
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5",
    "content": [{"type": "text", "text": "Postgres, partitioned by day."}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 120, "output_tokens": 18},
}

SUMMARY = LLMResponse(
    text="Earlier turns settled the storage layer and the retention window.",
    stop_reason=StopReason.COMPLETE,
    input_tokens=0,
    output_tokens=12,
    model="scripted",
    provider="offline",
)


def _engine(**config_overrides):
    """The real engine graph, with a scripted provider in place of a model."""
    base = {
        "context_window": 200,
        "keep_threshold": 0.55,
        "archive_threshold": 0.25,
        "last_n_turns_to_keep": 2,
        # Scaled to the 200-token window: a few turns' worth is worth a cut.
        "min_cut_tokens": 30,
    }
    base.update(config_overrides)
    service = build_gc_service(GCConfig(**base), summary_provider=FakeProvider(responses=[SUMMARY] * 200))
    return service, service.generational_memory


def _proxy(upstream=None, *, gc_service=None, on_reply=None, window_lookup=None, link_subagents=False,
           **engine_kwargs):
    """A proxy whose upstream is a scripted handler, with aging run synchronously.

    `schedule` is injected to run the post-response work to completion inline.
    Aging is fire-and-forget in production, and a test that asserted on session
    state after a bare `create_task` would be racing it.
    """
    service, memory = (gc_service, None) if gc_service is not None else _engine(**engine_kwargs)
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        if upstream is None:
            return httpx2.Response(200, json=UPSTREAM_OK)
        return upstream(request)

    app = create_app(
        service,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        schedule=lambda coro: asyncio.get_event_loop().run_until_complete(coro)
        if False
        else _run_now(coro),
        on_reply=on_reply,
        window_lookup=window_lookup,
        link_subagents=link_subagents,
    )
    app.state.seen = seen
    return TestClient(app), service, memory, seen


def _run_now(coro):
    """Drive a coroutine to completion from inside a running loop's thread.

    TestClient runs the app in its own loop; `asyncio.run` here would complain
    about a loop already running, so the coroutine is stepped manually. Crude, and
    only acceptable in a test: it makes post-response aging observable instead of
    a race.
    """
    try:
        coro.send(None)
    except StopIteration:
        return
    # A real await would suspend; the engine's awaits all complete without
    # yielding to I/O here (the provider is in-memory), so arriving at this point
    # means something genuinely blocked and the test should say so loudly.
    raise AssertionError("post-response aging suspended on real I/O; the test cannot observe it")


def _body(*turns: tuple[str, str], **extra) -> dict:
    payload = {
        "model": "claude-sonnet-5",
        "max_tokens": 512,
        "messages": [{"role": role, "content": content} for role, content in turns],
    }
    payload.update(extra)
    return payload


def _post(client, payload, headers=None):
    return client.post(
        "/v1/messages",
        json=payload,
        headers={"x-api-key": API_KEY, "anthropic-version": "2023-06-01", **(headers or {})},
    )


def _sent(seen) -> dict:
    return json.loads(seen[-1].content)


class TestOtherEndpointsPassThrough:
    """A client such as Claude Code calls more than Messages through the same base URL."""

    def test_an_endpoint_llm_gc_does_not_rewrite_is_forwarded_untouched(self):
        client, service, _, seen = _proxy(lambda request: httpx2.Response(200, json={"input_tokens": 42}))
        body = _body(("user", "how many tokens is this?"))

        response = client.post("/v1/messages/count_tokens?beta=true", json=body,
                               headers={"x-api-key": API_KEY, "anthropic-beta": "token-counting"})

        assert response.status_code == 200 and response.json() == {"input_tokens": 42}
        [sent] = seen
        assert (sent.method, sent.url.path, sent.url.params.get("beta")) == ("POST", "/v1/messages/count_tokens", "true")
        assert json.loads(sent.content) == body, "the body is not rewritten"
        assert (sent.headers["x-api-key"], sent.headers["anthropic-beta"]) == (API_KEY, "token-counting")
        assert service.session_manager.store.count() == 0, "nothing is recorded as a conversation"

    def test_a_get_is_forwarded_as_a_get_without_a_body(self):
        client, _, _, seen = _proxy(lambda request: httpx2.Response(200, json={"data": []}))

        assert client.get("/v1/models", headers={"x-api-key": API_KEY}).json() == {"data": []}
        assert (seen[0].method, seen[0].content) == ("GET", b"")

    def test_the_providers_error_passes_through_unchanged(self):
        client, _, _, _ = _proxy(lambda request: httpx2.Response(404, json={"type": "error"}))

        assert client.get("/v1/no-such-thing").status_code == 404

    def test_messages_keeps_its_query_string(self):
        client, _, _, seen = _proxy()

        client.post("/v1/messages?beta=true", json=_body(("user", "hello")), headers={"x-api-key": API_KEY})

        assert seen[0].url.path == "/v1/messages" and seen[0].url.params.get("beta") == "true"


class TestItForwardsAndAnswers:
    def test_a_request_is_forwarded_upstream(self):
        client, _, _, seen = _proxy()
        response = _post(client, _body(("user", "which database are we using?")))

        assert response.status_code == 200
        assert len(seen) == 1
        assert str(seen[0].url).endswith("/v1/messages")

    def test_the_providers_answer_is_returned_unchanged(self):
        """A client's own parsing is written against the provider's format. The
        proxy is not a translation layer."""
        client, _, _, _ = _proxy()
        response = _post(client, _body(("user", "hello")))

        assert response.json() == UPSTREAM_OK

    def test_a_provider_error_is_passed_through_verbatim(self):
        """Including the status code. Translating a 429 into something of our own
        would break the client's retry logic, which was written for the provider."""
        error = {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}
        client, _, _, _ = _proxy(lambda request: httpx2.Response(429, json=error))

        response = _post(client, _body(("user", "hello")))

        assert response.status_code == 429
        assert response.json() == error

    def test_an_unreachable_provider_is_a_502(self):
        """The failure is upstream of us, and the client should not read it as
        their own mistake."""

        def refuse(request):
            raise httpx2.ConnectError("connection refused", request=request)

        client, _, _, _ = _proxy(refuse)
        response = _post(client, _body(("user", "hello")))

        assert response.status_code == 502
        assert "could not reach" in response.json()["error"]["message"]

    def test_health_and_stats_are_served(self):
        client, _, _, _ = _proxy()
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/stats").json()["requests"] == 0


class TestTheClientsHistoryIsReplaced:
    def test_the_client_fields_are_passed_through_untouched(self):
        """The proxy changes the conversation and nothing else. Normalising unknown
        fields would break a client the day the provider adds one."""
        client, _, _, seen = _proxy()
        _post(client, _body(("user", "hi"), system="be terse", temperature=0.5, some_future_field=7))

        sent = _sent(seen)
        assert sent["system"] == "be terse"
        assert sent["temperature"] == 0.5
        assert sent["some_future_field"] == 7
        assert sent["model"] == "claude-sonnet-5"

    def test_a_cold_session_forwards_what_the_client_sent(self):
        """The correct cold start: the proxy cannot compress a conversation it has
        not seen."""
        client, _, _, seen = _proxy()
        _post(client, _body(("user", "first ever turn")))

        assert [m["content"] for m in _sent(seen)["messages"]] == ["first ever turn"]

    def test_a_warm_session_forwards_our_context_not_theirs(self):
        """The entire point of the proxy. The client resends everything; we send
        our maintained version, which has been through compression."""
        client, _, _, seen = _proxy()

        conversation: list[tuple[str, str]] = []
        for index in range(10):
            conversation.append(("user", f"turn {index}: we discussed the storage layer at length today"))
            _post(client, _body(*conversation))
            conversation.append(("assistant", UPSTREAM_OK["content"][0]["text"]))

        forwarded = len(_sent(seen)["messages"])
        # `conversation` now ends with the trailing assistant reply, so the client's
        # last request carried one fewer turn than its current length.
        sent_by_client = len(conversation) - 1
        assert forwarded < sent_by_client, f"forwarded {forwarded} of {sent_by_client}"

    def test_a_conversation_carrying_instructions_is_collected_and_keeps_every_instruction(self):
        """Claude Code puts its own instructions inside the conversation, as system
        messages. Unreadable, every request was forwarded untouched and nothing was
        ever collected; read, the conversation is collected and every instruction is
        still sent, word for word, exactly once."""
        client, _, _, seen = _proxy()
        messages: list[dict] = []
        for index in range(10):
            messages.append({"role": "user", "content": f"turn {index}: we discussed the storage layer at length today"})
            messages.append({"role": "system", "content": f"instruction {index}"})
            _post(client, {"model": "claude-sonnet-5", "max_tokens": 512, "messages": messages})
            messages.append({"role": "assistant", "content": UPSTREAM_OK["content"][0]["text"]})

        sent = _sent(seen)["messages"]
        assert client.get("/stats").json()["gc_bypassed"] == 0
        assert any(_is_front(message) for message in sent), "the conversation was collected"
        instructions = [m["content"] for m in sent if m["role"] == "system"]
        assert instructions == [f"instruction {index}" for index in range(10)], "each once, in order"

    def test_instructions_inside_a_message_reach_the_model_word_for_word_after_a_cut(self):
        """Claude Code wraps CLAUDE.md, memory and context in <system-reminder> blocks
        inside the user's first message. Summarised with that message, the model read
        a paraphrase of them, or nothing; lifted out, they open the recap word for
        word, once each, and no summary is written from them."""
        client, service, _, seen = _proxy()
        reminder = "<system-reminder>Project rule: never commit to main.</system-reminder>"
        messages: list[dict] = []
        for index in range(10):
            text = f"turn {index}: we discussed the storage layer at length today"
            messages.append({"role": "user", "content": [{"type": "text", "text": reminder}, {"type": "text", "text": text}]})
            _post(client, {"model": "claude-sonnet-5", "max_tokens": 512, "messages": messages})
            messages.append({"role": "assistant", "content": UPSTREAM_OK["content"][0]["text"]})

        sent = _sent(seen)["messages"]
        front = next(message for message in sent if _is_front(message))
        assert front["content"][0] == {"type": "text", "text": reminder}, "said once, ahead of the recap"
        assert sum(json.dumps(message).count("Project rule") for message in sent) == 1 + sum(
            1 for message in sent if not _is_front(message) and message["role"] == "user"), "once in the recap, as sent in each recent turn"
        summariser = service.generational_memory.compactor.provider.inner
        assert summariser.calls and not any("Project rule" in str(call.messages) for call in summariser.calls)

    def test_a_client_moving_its_cache_markers_stays_one_conversation_within_the_limit(self):
        """Claude Code marks the newest turns for caching on every request. The
        conversation must stay one session, and after a cut the forwarded turns must
        carry the markers where this request put them - recorded ones would pile up
        past the provider's limit of four."""
        client, service, _, seen = _proxy()
        conversation: list[tuple[str, str]] = []
        for index in range(10):
            conversation.append(("user", f"turn {index}: we discussed the storage layer at length today"))
            messages = [{"role": role, "content": [{"type": "text", "text": text}]} for role, text in conversation]
            for message in messages[-2:]:
                message["content"][-1]["cache_control"] = {"type": "ephemeral"}
            _post(client, {"model": "claude-sonnet-5", "max_tokens": 512, "messages": messages})
            conversation.append(("assistant", UPSTREAM_OK["content"][0]["text"]))

        stats = client.get("/stats").json()
        assert service.session_manager.store.count() == 1 and stats["history_restarts"] == 0
        sent = _sent(seen)["messages"]
        assert any(_is_front(message) for message in sent), "the conversation was collected"
        marked = [m for m in sent if isinstance(m["content"], list) and any("cache_control" in b for b in m["content"])]
        assert len(marked) == 2, "exactly the markers this request carried"

    def test_the_newest_turn_is_always_forwarded(self):
        """Whatever GC decides about history, the question being asked now has to
        reach the model."""
        client, _, _, seen = _proxy()
        for index in range(10):
            _post(client, _body(*[("user", f"turn {i} about the storage layer") for i in range(index + 1)]))

        _post(client, _body(*[("user", f"turn {i} about the storage layer") for i in range(10)],
                            ("user", "WHICH DATABASE?")))

        assert _sent(seen)["messages"][-1]["content"] == "WHICH DATABASE?"

    def test_the_newest_turn_is_not_sent_twice(self):
        """A prompt ending in a duplicated question is a confusing prompt, and the
        duplication would be invisible in anything but the wire body."""
        client, _, _, seen = _proxy()
        _post(client, _body(("user", "only turn")))

        contents = [m["content"] for m in _sent(seen)["messages"]]
        assert contents.count("only turn") == 1


class TestFindingTheConversation:
    """Each test here is a defect the proxy shipped with: the request reached the
    provider without the conversation it belonged to."""

    def test_a_first_turn_is_forwarded_untouched_and_starts_no_session(self):
        client, service, _, seen = _proxy()
        _post(client, _body(("user", "hello")))

        assert [m["content"] for m in _sent(seen)["messages"]] == ["hello"]
        assert service.session_manager.store.count() == 0

    def test_without_a_header_the_second_request_keeps_its_history(self):
        """The fingerprint gave request 1 and request 2 different sessions, and the
        cold second session forwarded only the newest turn."""
        client, _, _, seen = _proxy()
        _post(client, _body(("user", "we chose postgres")))
        _post(client, _body(("user", "we chose postgres"), ("assistant", "Noted."), ("user", "which database?")))

        assert [m["content"] for m in _sent(seen)["messages"]] == ["we chose postgres", "Noted.", "which database?"]

    def test_a_restarted_proxy_does_not_forget_a_conversation_in_progress(self):
        """A cold session forwarded only the newest turn: after a restart, every
        ongoing conversation reached the model with no history at all."""
        client, _, _, seen = _proxy()
        history = [("user", "we chose postgres"), ("assistant", "Noted."), ("user", "retention is 90 days"),
                   ("assistant", "Noted."), ("user", "which database did we choose?")]
        _post(client, _body(*history))

        assert len(_sent(seen)["messages"]) == len(history)

    def test_instructions_rewritten_in_an_earlier_turn_keep_the_conversation_in_its_session(self):
        """Claude Code rewrites the instructions in its first message between requests;
        hashed with them, one conversation was seeded as a second session, which
        collected and read facts again from nothing."""
        client, service, _, seen = _proxy()

        def opening(notes: str) -> dict:
            return {"role": "user", "content": [{"type": "text", "text": f"<system-reminder>{notes}</system-reminder>"},
                                                {"type": "text", "text": "we chose postgres"}]}

        later = [{"role": "assistant", "content": "Noted."}, {"role": "user", "content": "and retention?"}]
        _post(client, _body(messages=[opening("date"), *later]))
        _post(client, _body(messages=[opening("CLAUDE.md, memory, date"), *later,
                                      {"role": "assistant", "content": "90 days."}, {"role": "user", "content": "thanks"}]))

        assert service.session_manager.store.count() == 1
        assert client.get("/stats").json()["sessions_seeded"] == 1
        assert "CLAUDE.md, memory, date" in _sent(seen)["messages"][0]["content"][0]["text"]

    def test_regenerate_does_not_send_the_answer_being_replaced(self):
        """Regenerate re-sends the conversation without its last reply. Recording
        the reply from the response made the proxy forward it anyway, as if the user
        had asked the same question twice."""
        client, service, _, seen = _proxy()
        opening = [("user", "we chose postgres"), ("assistant", "Noted.")]
        _post(client, _body(*opening, ("user", "which database?")), headers={SESSION_HEADER: "s"})
        _post(client, _body(*opening, ("user", "which database?")), headers={SESSION_HEADER: "s"})

        assert [m["content"] for m in _sent(seen)["messages"]] == ["we chose postgres", "Noted.", "which database?"]
        assert len(service.session_manager.store.get("s").snapshot.young) == 2

    def test_an_edited_turn_rewinds_the_session_to_the_edit(self):
        client, service, _, seen = _proxy()
        h = {SESSION_HEADER: "s"}
        _post(client, _body(("user", "use postgres"), ("assistant", "Noted."), ("user", "and retention?")), headers=h)
        _post(client, _body(("user", "use mysql"), ("assistant", "Noted."), ("user", "and retention?")), headers=h)

        sent = [m["content"] for m in _sent(seen)["messages"]]
        assert sent == ["use mysql", "Noted.", "and retention?"]
        assert "use postgres" not in [m.content for m in service.session_manager.store.get("s").snapshot.young]
        assert client.get("/stats").json()["history_rewinds"] == 1

    def test_an_explicit_session_header_is_honoured(self):
        client, service, _, _ = _proxy()
        _post(client, _body(("user", "a"), ("assistant", "b"), ("user", "c")), headers={SESSION_HEADER: "chat-42"})
        assert service.session_manager.store.get("chat-42") is not None

    def test_different_conversations_get_different_sessions(self):
        client, service, _, _ = _proxy()
        _post(client, _body(("user", "about databases"), ("assistant", "ok"), ("user", "go on")))
        _post(client, _body(("user", "about holidays"), ("assistant", "ok"), ("user", "go on")))
        assert service.session_manager.store.count() == 2

    def test_the_session_id_carries_no_conversation_text(self):
        client, service, _, _ = _proxy()
        _post(client, _body(("user", "my secret project codename is BLUEBIRD"), ("assistant", "ok"), ("user", "?")))
        [session_id] = service.session_manager.store.active_session_ids()
        assert "BLUEBIRD" not in session_id


class TestTheRecordIsTheClientsHistory:
    """The client's next request holds every earlier reply exactly as the user saw
    it, and nothing the user never saw. Recording from it, rather than from the
    response, is what keeps the record and the conversation the same thing."""

    def test_both_halves_of_an_exchange_are_recorded_from_the_next_request(self):
        client, service, _, _ = _proxy()
        h = {SESSION_HEADER: "s"}
        _post(client, _body(("user", "which database?")), headers=h)
        _post(client, _body(("user", "which database?"), ("assistant", "Postgres, partitioned by day."),
                            ("user", "why?")), headers=h)

        young = service.session_manager.store.get("s").snapshot.young
        assert [(m.role, m.content) for m in young] == [("user", "which database?"),
                                                        ("assistant", "Postgres, partitioned by day.")]

    def test_a_failed_request_retried_records_its_history_once(self):
        """Recording on the first attempt would put the history in twice when the
        client retries."""
        attempts = iter([httpx2.Response(500, json={"error": "boom"}), httpx2.Response(200, json=UPSTREAM_OK)])
        client, service, _, _ = _proxy(lambda request: next(attempts))
        body = _body(("user", "a"), ("assistant", "b"), ("user", "c"))
        _post(client, body, headers={SESSION_HEADER: "s"})
        _post(client, body, headers={SESSION_HEADER: "s"})

        assert [m.content for m in service.session_manager.store.get("s").snapshot.young] == ["a", "b"]

    def test_a_tool_call_reply_is_recorded_exactly_as_sent(self):
        """Parsed out of a response, a tool-call reply had no text and was lost."""
        tool_use = [{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}]
        tool_result = [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "18C"}]
        client, service, _, _ = _proxy()
        client.post("/v1/messages", json={"model": "m", "max_tokens": 10, "messages": [
            {"role": "user", "content": "weather in paris?"},
            {"role": "assistant", "content": tool_use},
            {"role": "user", "content": tool_result}]}, headers={SESSION_HEADER: "s"})

        assert service.session_manager.store.get("s").snapshot.young[1].raw_content == tool_use

    def test_an_unaged_conversation_is_forwarded_byte_for_byte(self):
        """Nothing aged means nothing to change, so the client's own bytes go out:
        images, tool calls and the provider's prompt cache survive intact."""
        image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}
        payload = {"model": "m", "max_tokens": 10, "messages": [
            {"role": "user", "content": [{"type": "text", "text": "what is this?"}, image]},
            {"role": "assistant", "content": "A diagram."},
            {"role": "user", "content": "explain it"}]}
        client, _, _, seen = _proxy()
        raw = json.dumps(payload).encode()
        client.post("/v1/messages", content=raw, headers={"content-type": "application/json", SESSION_HEADER: "s"})

        assert seen[-1].content == raw


class TestItNeverLeaksCredentials:
    def test_the_api_key_is_forwarded_untouched(self):
        """Forwarded only. Never stored, never logged, never in an event."""
        client, _, _, seen = _proxy()
        _post(client, _body(("user", "hello")))

        assert seen[0].headers["x-api-key"] == API_KEY

    def test_the_key_is_not_in_the_stats_surface(self):
        client, _, _, _ = _proxy()
        _post(client, _body(("user", "hello")))

        assert API_KEY not in json.dumps(client.get("/stats").json())

    def test_the_key_is_not_held_in_session_state(self):
        client, service, _, _ = _proxy()
        _post(client, _body(("user", "hello"), ("assistant", "hi"), ("user", "?")), headers={SESSION_HEADER: "s"})

        state = service.session_manager.store.get("s")
        assert state is not None
        assert API_KEY not in repr(state)


class TestEachAnswerIsLogged:
    """One line per conversation request, written once the upstream has answered."""

    @staticmethod
    def _answers(caplog) -> list[logging.LogRecord]:
        return [record for record in caplog.records
                if record.name == "llm_gc.proxy.server" and hasattr(record, "status")]

    def test_a_success_says_the_model_the_status_and_the_tokens_in_against_out(self, caplog):
        caplog.set_level(logging.INFO, logger="llm_gc.proxy.server")
        client, _, _, _ = _proxy()
        _post(client, _body(("user", "which database are we using?")))

        [record] = self._answers(caplog)
        assert record.levelno == logging.INFO
        message = record.getMessage()
        assert message.startswith("claude-sonnet-5 -> 200: client sent ")
        assert "LLM-GC forwarded" in message
        assert record.tokens_from_client >= record.tokens_forwarded > 0

    def test_a_refused_credential_is_a_warning_that_says_so(self, caplog):
        """A request the upstream refused must not read like one that went through."""
        error = {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}
        client, _, _, _ = _proxy(lambda request: httpx2.Response(401, json=error))
        _post(client, _body(("user", "hello")))

        [record] = self._answers(caplog)
        assert record.levelno == logging.WARNING
        assert "-> 401" in record.getMessage() and "refused the client's credential" in record.getMessage()

    def test_a_refused_stream_is_logged_too(self, caplog):
        client, _, _, _ = _proxy(lambda request: httpx2.Response(429, json={"type": "error"}))
        _post(client, _body(("user", "hello"), stream=True))

        [record] = self._answers(caplog)
        assert record.status == 429 and "upstream's error unchanged" in record.getMessage()

    def test_a_model_name_that_could_forge_a_log_line_is_not_written(self, caplog):
        caplog.set_level(logging.INFO, logger="llm_gc.proxy.server")
        client, _, _, _ = _proxy()
        payload = _body(("user", "hello"))
        payload["model"] = "x\nINFO llm_gc: forged"
        _post(client, payload)

        [record] = self._answers(caplog)
        assert record.getMessage().startswith("? -> 200")

    def test_the_client_key_never_reaches_the_log(self, caplog):
        caplog.set_level(logging.DEBUG)
        client, _, _, _ = _proxy(lambda request: httpx2.Response(401, json={"type": "error"}))
        _post(client, _body(("user", "hello")))

        assert API_KEY not in caplog.text


CLIENT_SESSION = "c7a78c2f-d25c-44e4-af6f-dba30717fb9b"


class TestClientSessions:
    """One Claude Code session holds several conversations: its main one and one per sub-agent."""

    def _proxy_seeing_origins(self):
        observed: list[tuple[str, ClientOrigin | None]] = []
        service, _ = _engine()
        app = create_app(service, http_client=httpx2.AsyncClient(
            transport=httpx2.MockTransport(lambda request: httpx2.Response(200, json=UPSTREAM_OK))),
            schedule=_run_now, on_client_history=lambda session_id, messages, origin: observed.append((session_id, origin)))
        return TestClient(app), observed

    def test_a_sub_agent_is_its_own_conversation_labelled_with_the_client_session(self):
        client, observed = self._proxy_seeing_origins()
        main = {CLIENT_SESSION_HEADER: CLIENT_SESSION}
        agent = {CLIENT_SESSION_HEADER: CLIENT_SESSION, CLIENT_AGENT_HEADER: "a1c50dab6bf4df2c1"}

        _post(client, _body(("user", "find the TODOs"), ("assistant", "starting an agent"), ("user", "go")), main)
        _post(client, _body(("user", "search for TODO"), ("assistant", "grep"), ("user", "results")), agent)

        (main_id, main_origin), (agent_id, agent_origin) = observed
        assert main_id != agent_id
        assert main_origin == ClientOrigin(session_id=CLIENT_SESSION)
        assert agent_origin == ClientOrigin(session_id=CLIENT_SESSION, agent_id="a1c50dab6bf4df2c1")

    def test_the_client_session_never_joins_two_conversations(self):
        """It labels; content decides. After a compaction the client sends a history that
        continues nothing recorded, under the same client session: a new conversation."""
        client, observed = self._proxy_seeing_origins()
        headers = {CLIENT_SESSION_HEADER: CLIENT_SESSION}

        _post(client, _body(("user", "the database is postgres"), ("assistant", "noted"), ("user", "next")), headers)
        _post(client, _body(("user", "This session is being continued: a summary"), ("assistant", "ok"),
                            ("user", "go on")), headers)

        assert len({session_id for session_id, _ in observed}) == 2

    def test_a_new_conversation_is_logged_with_why_and_where_it_came_from(self, caplog):
        caplog.set_level(logging.INFO, logger="llm_gc.proxy.server")
        client, observed = self._proxy_seeing_origins()
        headers = {CLIENT_SESSION_HEADER: CLIENT_SESSION, CLIENT_AGENT_HEADER: "a545ab218c80b9b63"}

        _post(client, _body(("user", "list the files"), ("assistant", "ls"), ("user", "three files")), headers)

        [started] = [r for r in caplog.records if hasattr(r, "history_turns")]
        session_id, _ = observed[0]
        assert started.getMessage() == (
            f"conversation {session_id} started: no conversation in progress continues its 2-turn history "
            f"(client session {CLIENT_SESSION}, sub-agent a545ab218c80b9b63)")

    def test_a_continuing_request_starts_nothing_and_its_answer_names_the_conversation(self, caplog):
        caplog.set_level(logging.INFO, logger="llm_gc.proxy.server")
        client, observed = self._proxy_seeing_origins()
        turns = [("user", "hello"), ("assistant", "hi"), ("user", "which database?")]

        _post(client, _body(*turns))
        _post(client, _body(*turns, ("assistant", "postgres"), ("user", "and the cache?")))

        started = [r for r in caplog.records if hasattr(r, "history_turns")]
        answers = [r for r in caplog.records if hasattr(r, "status")]
        session_id, _ = observed[-1]
        assert len(started) == 1
        assert "(no client session named)" in started[0].getMessage()
        assert all(f"-> 200 in {session_id}: " in answer.getMessage() for answer in answers)


class TestHeaderHygiene:
    def test_content_length_is_not_forwarded(self):
        """The body is rewritten after headers are built, so any value set here is
        wrong by exactly the number of tokens GC saved — and a wrong content-length
        surfaces as a truncated body a long way from its cause."""
        client, _, _, seen = _proxy()
        _post(client, _body(("user", "hello")), headers={"content-length": "999"})

        assert "content-length" not in {k.lower() for k in seen[0].headers} or seen[
            0
        ].headers.get("content-length") != "999"

    def test_the_proxys_own_host_header_is_not_forwarded(self):
        client, _, _, seen = _proxy()
        _post(client, _body(("user", "hello")))

        assert seen[0].headers.get("host") != "testserver"

    def test_provider_headers_are_forwarded(self):
        client, _, _, seen = _proxy()
        _post(client, _body(("user", "hello")))

        assert seen[0].headers["anthropic-version"] == "2023-06-01"


class TestFirstDoNoHarm:
    def test_a_malformed_body_is_forwarded_for_the_provider_to_judge(self):
        """The provider is the authority on its own API — it may accept something
        this version does not understand."""
        client, _, _, seen = _proxy()
        response = client.post("/v1/messages", content=b"{not json", headers={"x-api-key": API_KEY})

        assert response.status_code == 200
        assert seen[0].content == b"{not json"

    def test_a_body_with_no_messages_is_forwarded_untouched(self):
        client, _, _, seen = _proxy()
        client.post("/v1/messages", json={"model": "x", "max_tokens": 1}, headers={"x-api-key": API_KEY})

        assert json.loads(seen[0].content) == {"model": "x", "max_tokens": 1}

    def test_a_gc_failure_does_not_become_a_500(self):
        """A proxy that returns 500 because its optimiser had a bad day is worse
        than no proxy. It must not break an app that worked before it was
        installed."""

        class _BrokenService:
            session_manager = SessionManager()
            event_bus = EventBus()
            gc_config = GCConfig()

            async def gc_collect(self, session_id, query=None):
                raise RuntimeError("engine exploded")

            async def gc_update(self, session_id, new_turn):
                raise RuntimeError("engine exploded")

        client, _, _, seen = _proxy(gc_service=_BrokenService())
        response = _post(client, _body(("user", "hello")))

        assert response.status_code == 200
        assert json.loads(seen[0].content)["messages"][0]["content"] == "hello"

    def test_a_gc_failure_is_counted_as_a_bypass(self):
        class _BrokenService:
            session_manager = SessionManager()
            event_bus = EventBus()
            gc_config = GCConfig()

            async def gc_collect(self, session_id, query=None):
                raise RuntimeError("engine exploded")

            def gc_record(self, session_id, turns, lineage):
                raise RuntimeError("engine exploded")

        client, _, _, _ = _proxy(gc_service=_BrokenService())
        _post(client, _body(("user", "hello"), ("assistant", "hi"), ("user", "?")))

        stats = client.get("/stats").json()
        assert stats["gc_bypassed"] == 1
        assert stats["gc_applied"] == 0


class TestTheStatsSurface:
    def test_it_reports_what_was_saved(self):
        """The conversation alternates, because a real client resends the whole
        thing including the replies it received.

        An earlier version of this test sent twelve consecutive user turns and
        reported tokens_saved as -301 — not a proxy bug. Our session correctly
        records both halves of every exchange, so a user-only "client history" is
        half the real conversation and the comparison is meaningless. The metric is
        only interpretable against what a client actually sends."""
        client, _, _, _ = _proxy()

        conversation: list[tuple[str, str]] = []
        for index in range(10):
            conversation.append(("user", f"turn {index}: we discussed the storage layer at length today"))
            _post(client, _body(*conversation))
            # The reply the client would have received, which it resends next time.
            conversation.append(("assistant", UPSTREAM_OK["content"][0]["text"]))

        stats = client.get("/stats").json()
        assert stats["requests"] == 10
        assert stats["tokens_saved"] > 0, stats
        assert 0 < stats["ratio"] < 1, stats

    def test_the_ratio_is_none_before_any_traffic(self):
        """A ratio with no requests behind it is not a measurement, and 1.0 would
        read as "we saved nothing" rather than "we have not run"."""
        client, _, _, _ = _proxy()
        assert client.get("/stats").json()["ratio"] is None

    def test_a_bypassed_request_is_not_counted_as_free(self):
        class _BrokenService:
            session_manager = SessionManager()
            event_bus = EventBus()
            gc_config = GCConfig()

            async def gc_collect(self, session_id, query=None):
                raise RuntimeError("engine exploded")

            async def gc_update(self, session_id, new_turn):
                raise RuntimeError("engine exploded")

        client, _, _, _ = _proxy(gc_service=_BrokenService())
        _post(client, _body(("user", "a turn with some real length to it here")))

        stats = client.get("/stats").json()
        assert stats["tokens_forwarded"] == stats["tokens_from_client"]
        assert stats["tokens_saved"] == 0


class TestContentBlocks:
    def test_a_block_list_is_flattened_for_scoring(self):
        client, service, _, _ = _proxy()
        client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-5",
                "max_tokens": 512,
                "messages": [{"role": "user", "content": [{"type": "text", "text": "which database?"}]},
                             {"role": "assistant", "content": "Postgres."},
                             {"role": "user", "content": "why?"}],
            },
            headers={"x-api-key": API_KEY, SESSION_HEADER: "s"},
        )

        state = service.session_manager.store.get("s")
        assert state is not None
        assert state.snapshot.young[0].content == "which database?"

    def test_a_non_text_block_does_not_break_the_request(self):
        client, _, _, seen = _proxy()
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-5",
                "max_tokens": 512,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "data": "aaa"}},
                            {"type": "text", "text": "what is this?"},
                        ],
                    }
                ],
            },
            headers={"x-api-key": API_KEY},
        )

        assert response.status_code == 200
        assert len(seen) == 1


@pytest.mark.parametrize("role", ["system", "tool", ""])
def test_an_unsupported_role_is_forwarded_for_the_provider_to_judge(role):
    client, _, _, seen = _proxy()
    client.post(
        "/v1/messages",
        json={"model": "x", "max_tokens": 1, "messages": [{"role": role, "content": "hi"}]},
        headers={"x-api-key": API_KEY},
    )

    assert json.loads(seen[0].content)["messages"][0]["role"] == role


class TestTheRecapBudgetCannotInvertTheSaving:
    """A regression guard with a measured origin.

    With an absolute memory budget, `max_memory_tokens=100` against a 200-token
    window made the proxy forward 21% MORE than the client sent: the injected block
    consumed the whole saving and then some, silently. The recap's budget is a
    share of the window, and a collection must pay for itself, so at no share can
    the proxy forward more than the client sent.
    """

    @staticmethod
    def _savings(**config) -> dict:
        client, _, _, _ = _proxy(**config)
        conversation: list[tuple[str, str]] = []
        for index in range(10):
            conversation.append(("user", f"turn {index}: we discussed the storage layer at length today"))
            _post(client, _body(*conversation))
            conversation.append(("assistant", UPSTREAM_OK["content"][0]["text"]))
        return client.get("/stats").json()

    @pytest.mark.parametrize("ratio", [0.0, 0.3, 0.6, 1.0])
    def test_no_share_of_the_window_makes_the_proxy_forward_more_than_it_was_sent(self, ratio):
        stats = self._savings(context_window=200, max_recap_ratio=ratio)
        assert stats["tokens_saved"] >= 0, stats

    def test_on_a_small_window_the_budget_bounds_the_front_and_saves(self):
        """Turns of a dozen tokens are not worth condensing on their own merits; a
        200-token window leaves the budget no choice, and folding pays."""
        stats = self._savings(context_window=200, max_recap_ratio=0.3)
        assert stats["tokens_saved"] > 0, stats

    def test_the_budget_is_derived_from_the_window(self):
        from llm_gc.config import GCConfig

        assert GCConfig(context_window=1000, max_recap_ratio=0.3).max_recap_tokens == 300
        assert GCConfig(context_window=200, max_recap_ratio=0.3).max_recap_tokens == 60
        # One source of truth: there is no stored absolute to drift from the window.
        assert GCConfig(context_window=128_000, max_recap_ratio=0.0).max_recap_tokens == 0


def _sse(*texts: str, stop: bool = True, error: bool = False) -> bytes:
    def event(payload: dict) -> str:
        return f"event: {payload['type']}\ndata: {json.dumps(payload)}\n\n"

    out = event({"type": "message_start", "message": {"id": "m", "role": "assistant", "content": []}})
    out += "".join(
        event({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": t}}) for t in texts
    )
    if error:
        out += event({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})
    if stop:
        out += event({"type": "message_stop"})
    return out.encode()


def _streaming_upstream(body: bytes):
    return lambda request: httpx2.Response(200, content=body, headers={"content-type": "text/event-stream"})


class TestStreaming:
    def test_the_stream_reaches_the_client_byte_for_byte(self):
        body = _sse("Postgres, ", "partitioned by day.")
        client, _, _, _ = _proxy(_streaming_upstream(body))
        response = _post(client, _body(("user", "which database?"), stream=True))
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.content == body

    def test_a_finished_reply_is_shown_but_not_recorded(self):
        """The dashboard showed the newest message without its answer until the user
        wrote again. The reply is read for showing only; the record still waits for
        the next request, which says whether the client kept it."""
        body = _sse("Postgres, ", "partitioned by day.")
        shown = []
        client, service, _, _ = _proxy(_streaming_upstream(body), on_reply=lambda *args: shown.append(args))
        opening = [("user", "we need a database"), ("assistant", "Noted.")]

        response = _post(client, _body(*opening, ("user", "which database?"), stream=True), headers={SESSION_HEADER: "s"})

        assert response.content == body
        [(session_id, answering, text)] = shown
        assert (session_id, answering.turn_index, answering.content, text) == (
            "s", 2, "which database?", "Postgres, partitioned by day.")
        assert service.session_manager.store.get("s").snapshot.recorded_turn_count == 2

    def test_a_reply_cut_off_or_refused_is_not_shown(self):
        shown = []
        cut_off, _, _, _ = _proxy(_streaming_upstream(_sse("Postg", stop=False)), on_reply=lambda *a: shown.append(a))
        _post(cut_off, _body(("user", "a"), ("assistant", "b"), ("user", "c"), stream=True))
        refused, _, _, _ = _proxy(lambda request: httpx2.Response(529, json={"type": "error"}),
                                  on_reply=lambda *a: shown.append(a))
        _post(refused, _body(("user", "a"), ("assistant", "b"), ("user", "c")))

        assert shown == []

    def test_a_failing_reply_observer_never_reaches_the_client(self):
        def broken(*_):
            raise RuntimeError("dashboard bug")
        body = _sse("Postgres.")
        client, _, _, _ = _proxy(_streaming_upstream(body), on_reply=broken)
        answered, _, _, _ = _proxy(on_reply=broken)

        assert _post(client, _body(("user", "a"), ("assistant", "b"), ("user", "c"), stream=True)).content == body
        assert _post(answered, _body(("user", "a"), ("assistant", "b"), ("user", "c"))).json() == UPSTREAM_OK

    def test_a_streamed_reply_is_recorded_from_the_next_request(self):
        """The defect this suite once existed for: a streamed reply was never
        recorded. Now no reply is parsed; the client's next request carries it."""
        client, service, _, _ = _proxy(_streaming_upstream(_sse("Postgres.")))
        h = {SESSION_HEADER: "s"}
        _post(client, _body(("user", "which database?"), stream=True), headers=h)
        _post(client, _body(("user", "which database?"), ("assistant", "Postgres."), ("user", "why?"), stream=True),
              headers=h)

        assert [m.role for m in service.session_manager.store.get("s").snapshot.young] == ["user", "assistant"]

    def test_an_error_status_on_a_stream_request_passes_through_verbatim(self):
        error = {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}
        client, _, _, _ = _proxy(lambda request: httpx2.Response(429, json=error))
        response = _post(client, _body(("user", "which database?"), stream=True))

        assert response.status_code == 429
        assert response.json() == error

    def test_a_gc_failure_still_streams_the_original_request(self, monkeypatch):
        body = _sse("fine")
        client, service, _, seen = _proxy(_streaming_upstream(body))

        async def broken(*args, **kwargs):
            raise RuntimeError("engine bug")

        monkeypatch.setattr(service, "gc_collect", broken)
        response = _post(client, _body(("user", "a"), ("assistant", "b"), ("user", "c"), stream=True))
        assert response.content == body
        assert _sent(seen)["stream"] is True


class TestStreamingIsIncremental:
    """Driven at the ASGI level: TestClient reads the whole body before returning,
    so it cannot tell a relayed stream from a buffered one."""

    def test_the_first_chunk_reaches_the_client_before_the_upstream_finishes(self):
        first, rest = _sse("Postgres", stop=False), _sse(", partitioned by day.")

        async def scenario():
            gate = asyncio.Event()

            async def upstream_body():
                yield first
                await gate.wait()
                yield rest

            client, _, _, _ = _proxy(lambda request: httpx2.Response(
                200, content=upstream_body(), headers={"content-type": "text/event-stream"}))
            sent: list[dict] = []
            first_body = asyncio.Event()

            async def send(message):
                sent.append(message)
                if message["type"] == "http.response.body" and message.get("body"):
                    first_body.set()

            payload = json.dumps(_body(("user", "which database?"), stream=True)).encode()
            received = iter([{"type": "http.request", "body": payload, "more_body": False}])

            async def receive():
                try:
                    return next(received)
                except StopIteration:
                    await asyncio.Event().wait()  # the client never disconnects

            scope = {"type": "http", "method": "POST", "path": "/v1/messages", "headers": [
                (b"content-type", b"application/json"), (b"x-api-key", API_KEY.encode())],
                "query_string": b"", "http_version": "1.1", "scheme": "http", "server": ("t", 80)}

            task = asyncio.create_task(client.app(scope, receive, send))
            await asyncio.wait_for(first_body.wait(), timeout=2)
            delivered_early = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
            gate.set()
            await asyncio.wait_for(task, timeout=2)
            return delivered_early

        assert asyncio.run(scenario()) == _sse("Postgres", stop=False)


class TestLlmGcNeverBlocksTheConversation:
    """The rule: a failure anywhere in LLM-GC costs, at worst, GC or one recorded
    turn. It never costs the user their response."""

    def test_a_session_resolution_failure_forwards_the_original(self, monkeypatch):
        def broken(*args, **kwargs):
            raise RuntimeError("bug")

        monkeypatch.setattr("llm_gc.proxy.server._record_client_history", broken)
        client, _, _, _ = _proxy()
        response = _post(client, _body(("user", "a"), ("assistant", "b"), ("user", "which database?")))
        assert response.status_code == 200
        assert response.json() == UPSTREAM_OK

    @pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
    def test_a_recording_failure_after_the_answer_does_not_lose_it(self, stream):
        """The provider answered and billed for it; our bookkeeping must not
        discard it. On a stream, a raise here would abort the connection before
        its final frame, after every byte had been sent."""
        body = _sse("Postgres.")
        client, _, _, _ = _proxy(_streaming_upstream(body) if stream else None)

        def broken(coro):
            coro.close()
            raise RuntimeError("scheduler down")

        client.app.state.schedule = broken
        response = _post(client, _body(("user", "a"), ("assistant", "b"), ("user", "c"), stream=stream))
        assert response.status_code == 200
        assert response.content == body if stream else response.json() == UPSTREAM_OK


class TestOneUsersConversationNeverReachesAnother:
    def test_a_fact_from_one_session_is_never_injected_into_another(self):
        """Found by probing, not by any existing test: with a shared permanent
        generation, Bob's prompt carried Alice's facts labelled as *his* memory.
        Every earlier test drove one session, and a small memory budget had
        hidden the block that carried them."""
        client, _, _, seen = _proxy(context_window=400, max_recap_ratio=0.4)
        alice: list[tuple[str, str]] = []
        for i in range(14):
            alice.append(("user", f"Alice question {i}: the patient records database stores diagnoses for clinic {i}"))
            _post(client, _body(*alice), headers={SESSION_HEADER: "alice"})
            alice.append(("assistant", "Postgres, partitioned by day."))
        alice_front = json.dumps(_sent(seen)["messages"][0])
        assert "condensed by LLM-GC" in alice_front and "patient" in alice_front, \
            "precondition: Alice's own condensed turns are in her front"

        bob_turns = [("user", "Bob here: help me plan a birthday party"), ("assistant", "Sure.")]
        _post(client, _body(*bob_turns[:1]), headers={SESSION_HEADER: "bob"})
        _post(client, _body(*bob_turns, ("user", "which database should store the party guest records?")),
              headers={SESSION_HEADER: "bob"})

        forwarded_for_bob = json.dumps(_sent(seen)["messages"])
        assert "patient" not in forwarded_for_bob
        assert "clinic" not in forwarded_for_bob


def _converse_through(client, seen, turns: int, *, extra_content=None, header="s"):
    """Drive a conversation the way a client does: full history, replies included.

    Returns the messages forwarded on each request. `extra_content(i)` may return a
    content-block list to send for user turn i instead of plain text.
    """
    history: list[dict] = []
    forwarded = []
    for i in range(turns):
        content = extra_content(i) if extra_content else None
        history.append({"role": "user", "content": content or
                        f"question {i}: we went over the storage layer, retention, and the rollout plan again"})
        client.post("/v1/messages", json={"model": "m", "max_tokens": 64, "messages": history},
                    headers={"x-api-key": API_KEY, SESSION_HEADER: header})
        forwarded.append(json.loads(seen[-1].content)["messages"])
        history.append({"role": "assistant", "content": UPSTREAM_OK["content"]})
    return forwarded


def _is_front(message: dict) -> bool:
    """The recap message: its text, or its last text block once instructions open it."""
    content = message["content"]
    text = content if isinstance(content, str) else (content[-1].get("text", "") if content else "")
    return text.startswith(FRONT_HEADER)


class TestTheCutShape:
    """Every change LLM-GC makes to a prompt happens at its front, at a cut; between
    cuts the forwarded conversation only grows at the end."""

    def test_after_a_cut_the_conversation_opens_with_a_user_turn(self):
        """A rewritten history that opened with an assistant turn was rejected by
        the provider - the user's request failing because of LLM-GC."""
        client, _, _, seen = _proxy()
        forwarded = _converse_through(client, seen, 16)

        cut_requests = [messages for messages in forwarded if _is_front(messages[0])]
        assert cut_requests, "the conversation must be long enough to cut"
        assert all(messages[0]["role"] == "user" for messages in forwarded)

    def test_between_cuts_each_request_extends_the_last(self):
        """Append-only between cuts is what lets the provider cache the prompt and
        keeps the model's thinking blocks valid."""
        client, _, _, seen = _proxy()
        forwarded = _converse_through(client, seen, 16)

        extended = 0
        for earlier, later in zip(forwarded, forwarded[1:]):
            if earlier[0] == later[0]:          # same front: no cut in between
                assert later[: len(earlier)] == earlier
                extended += 1
        assert extended >= 3, "the test must observe several requests between cuts"

    def test_tail_turns_are_forwarded_as_the_clients_original_content(self):
        image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}

        def with_image(i):
            return [{"type": "text", "text": f"question {i}: what does this diagram show about the rollout?"},
                    image] if i == 14 else None

        client, _, _, seen = _proxy()
        forwarded = _converse_through(client, seen, 16, extra_content=with_image)

        last = forwarded[-1]
        assert _is_front(last[0]), "the image turn must sit in the tail of a cut conversation"
        assert any(isinstance(m["content"], list) and image in m["content"] for m in last)

    def test_no_request_replays_thinking_produced_against_another_front(self):
        """Every thinking block is bound to the exact prompt before it, and current
        models reject a request that replays one whose prompt has since changed.

        The reply to the request during which a cut is planned is produced against
        the old front, and the client sends it back on the next request - the first
        to carry the new front. Stripping only turns recorded before the commit kept
        that reply's thinking, and the provider answered the user with a 400.
        """
        client, _, _, seen = _proxy(upstream=_front_binding_provider)
        history: list[dict] = []
        statuses = []
        for i in range(30):
            history.append({"role": "user", "content":
                            f"question {i}: we went over the storage layer, retention, and the rollout plan again"})
            response = client.post("/v1/messages", json={"model": "m", "max_tokens": 64, "messages": history},
                                   headers={"x-api-key": API_KEY})
            statuses.append(response.status_code)
            if response.status_code != 200:
                break
            history.append({"role": "assistant", "content": response.json()["content"]})

        forwarded = [json.loads(request.content)["messages"] for request in seen]
        assert statuses == [200] * 30, f"rejected at requests {[i for i, s in enumerate(statuses) if s != 200]}"
        assert sum(1 for i, m in enumerate(forwarded[1:], 1) if m[0] != forwarded[i - 1][0]) >= 2, "two cuts at least"
        assert any(_is_front(m[0]) and _thinking_blocks(m) for m in forwarded), \
            "replies produced against the current front keep their thinking"

    def test_a_cut_that_commits_after_the_next_request_left_is_still_safe(self):
        """Compaction is a model call, so a pass can commit after the next request
        has already been forwarded on the old front. That request's reply predates
        the new front too, and the commit cannot know it exists."""
        service, _ = _engine()
        pending: list = []
        seen: list[httpx2.Request] = []

        def provider(request: httpx2.Request) -> httpx2.Response:
            seen.append(request)
            return _front_binding_provider(request)

        client = TestClient(create_app(service, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(provider)),
                                       schedule=pending.append))
        history: list[dict] = []
        try:
            for i in range(30):
                history.append({"role": "user", "content":
                                f"question {i}: we went over the storage layer, retention, and the rollout plan again"})
                response = client.post("/v1/messages", json={"model": "m", "max_tokens": 64, "messages": history},
                                       headers={"x-api-key": API_KEY})
                assert response.status_code == 200, f"request {i} was rejected"
                history.append({"role": "assistant", "content": response.json()["content"]})
                while len(pending) > 1:          # each pass runs one request late
                    _run_now(pending.pop(0))
        finally:
            for coro in pending:
                coro.close()
        assert service.session_manager.store.get(service.session_manager.store.active_session_ids()[0]).snapshot.cut_count >= 2


def _thinking_blocks(messages: list[dict]) -> list[dict]:
    return [block for message in messages if isinstance(message["content"], list)
            for block in message["content"] if block.get("type") == "thinking"]


def _front_binding_provider(request: httpx2.Request) -> httpx2.Response:
    """A provider that binds each reply's thinking to the front it was produced
    against, and rejects a request replaying a block bound to a different one.

    The real check binds to the whole prefix; between cuts the proxy only appends,
    so the front is the part of the prefix a cut can change.
    """
    messages = json.loads(request.content)["messages"]
    front = messages[0]["content"] if _is_front(messages[0]) else ""
    binding = hashlib.sha256(front.encode()).hexdigest()[:16]
    if any(block["signature"] != binding for block in _thinking_blocks(messages)):
        return httpx2.Response(400, json={"type": "error", "error": {
            "type": "invalid_request_error",
            "message": "Invalid `signature` in `thinking` block. The block is bound to a different conversation."}})
    return httpx2.Response(200, json={**UPSTREAM_OK, "content": [
        {"type": "thinking", "thinking": "", "signature": binding}, {"type": "text", "text": "Postgres."}]})


class TestRequestSize:
    def test_a_body_over_the_providers_limit_is_refused_without_forwarding(self, monkeypatch):
        monkeypatch.setattr("llm_gc.proxy.server.MAX_REQUEST_BYTES", 1024)
        client, _, _, seen = _proxy()
        response = client.post("/v1/messages", content=b"x" * 2048, headers={"content-type": "application/json"})
        assert response.status_code == 413
        assert response.json()["error"]["type"] == "request_too_large"
        assert seen == []

    def test_a_body_under_the_limit_is_forwarded(self, monkeypatch):
        monkeypatch.setattr("llm_gc.proxy.server.MAX_REQUEST_BYTES", 4096)
        client, _, _, seen = _proxy()
        assert _post(client, _body(("user", "hello"))).status_code == 200
        assert len(seen) == 1


class TestIdleSessionsAreClosed:
    def test_the_proxy_closes_sessions_that_go_idle(self):
        """Sessions hold conversation content. The sweeper existed but nothing
        started it, so a serving proxy kept every conversation for its lifetime."""
        service, _ = _engine()
        service.session_manager = SessionManager(session_timeout_s=0, sweep_interval_s=0.01)
        client, _, _, _ = _proxy(gc_service=service)

        with client:   # entering the client runs the app's lifespan
            _post(client, _body(("user", "hello"), ("assistant", "hi"), ("user", "which database?")))
            assert client.get("/stats").json()["sessions_seeded"] == 1
            deadline = time.monotonic() + 2
            while service.session_manager.store.count() and time.monotonic() < deadline:
                time.sleep(0.01)

        assert service.session_manager.store.count() == 0


class TestTheHistoryObserver:
    """How the dashboard sees what was said: the engine keeps only what the model is sent."""

    def _proxy_with(self, observer):
        service, _ = _engine()
        seen: list[httpx2.Request] = []

        def handler(request):
            seen.append(request)
            return httpx2.Response(200, json=UPSTREAM_OK)

        app = create_app(service, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
                         schedule=_run_now, on_client_history=observer)
        return TestClient(app), seen

    def test_it_receives_the_whole_conversation_the_client_sent(self):
        observed = []
        client, _ = self._proxy_with(lambda session_id, messages, origin: observed.append((session_id, messages)))

        _post(client, _body(("user", "the database is postgres"), ("assistant", "noted"), ("user", "and the cache?")))

        session_id, messages = observed[-1]
        assert session_id.startswith("s_")
        assert [(m.role, m.content) for m in messages] == [
            ("user", "the database is postgres"), ("assistant", "noted"), ("user", "and the cache?")]

    def test_a_failing_observer_never_costs_the_request(self):
        def broken(session_id, messages, origin):
            raise RuntimeError("dashboard bug")

        client, seen = self._proxy_with(broken)
        response = _post(client, _body(("user", "hello"), ("assistant", "hi"), ("user", "which database?")))

        assert response.status_code == 200
        assert len(seen) == 1
        assert client.get("/stats").json()["gc_bypassed"] == 0, "not even GC is given up for it"


class TestTrafficIsMeasured:
    """What `/stats` reports happened: each request whole, the provider's own usage, grouped as the client ran it."""

    TOOLS = [{"name": "Read", "description": "Reads a file from the local filesystem. " * 20,
              "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}]

    @staticmethod
    def _converse(client, turns: int, headers=None, **extra) -> dict:
        conversation: list[tuple[str, str]] = []
        for index in range(turns):
            conversation.append(("user", f"turn {index}: we discussed the storage layer at length today"))
            _post(client, _body(*conversation, **extra), headers=headers)
            conversation.append(("assistant", UPSTREAM_OK["content"][0]["text"]))
        return client.get("/stats").json()

    def test_the_system_prompt_and_tools_count_on_both_sides_and_save_nothing(self):
        """Left out, the share saved was measured against only the part GC can shrink, and read too large."""
        bare = self._converse(_proxy()[0], 10)
        whole = self._converse(_proxy()[0], 10, system="You are a careful assistant. " * 40, tools=self.TOOLS)

        fixed = fixed_prompt_tokens({"system": "You are a careful assistant. " * 40, "tools": self.TOOLS})
        assert whole["tokens_from_client"] - bare["tokens_from_client"] == 10 * fixed
        assert whole["tokens_forwarded"] - bare["tokens_forwarded"] == 10 * fixed
        assert whole["tokens_saved"] == bare["tokens_saved"] > 0
        assert whole["ratio"] > bare["ratio"]

    def test_the_providers_usage_is_added_up_from_every_reply(self):
        stats = self._converse(_proxy()[0], 3)

        assert stats["provider_usage"]["replies"] == 3
        assert stats["provider_usage"]["input_tokens"] == 3 * UPSTREAM_OK["usage"]["input_tokens"]
        assert stats["provider_usage"]["output_tokens"] == 3 * UPSTREAM_OK["usage"]["output_tokens"]

    def test_a_streamed_replys_usage_is_read_as_it_passes(self):
        def event(payload: dict) -> str:
            return f"event: {payload['type']}\ndata: {json.dumps(payload)}\n\n"

        body = (event({"type": "message_start", "message": {"usage": {"input_tokens": 7,
                                                                      "cache_read_input_tokens": 4000,
                                                                      "output_tokens": 1}}})
                + event({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}})
                + event({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 61}})
                + event({"type": "message_stop"})).encode()
        client, _, _, _ = _proxy(_streaming_upstream(body))

        response = _post(client, _body(("user", "which database?"), stream=True))

        assert response.content == body
        usage = client.get("/stats").json()["provider_usage"]
        assert (usage["input_tokens"], usage["cache_read_input_tokens"], usage["output_tokens"]) == (7, 4000, 61)

    def test_conversations_are_grouped_under_the_client_session_that_ran_them(self):
        client, _, _, _ = _proxy()
        self._converse(client, 4, headers={CLIENT_SESSION_HEADER: "cc-1"})
        self._converse(client, 3, headers={CLIENT_SESSION_HEADER: "cc-1", CLIENT_AGENT_HEADER: "agent-a"})

        [session] = client.get("/stats").json()["client_sessions"]
        by_agent = {conversation["agent"]: conversation for conversation in session["conversations"]}
        assert set(by_agent) == {None, "agent-a"}
        # Each conversation's first request, which belonged to none yet, was claimed by it.
        assert by_agent[None]["traffic"]["requests"] == 4
        assert by_agent["agent-a"]["traffic"]["requests"] == 3
        assert session["outside_conversations"]["requests"] == 0
        assert session["traffic"]["requests"] == 7
        assert by_agent[None]["model"] == "claude-sonnet-5"
        assert API_KEY not in json.dumps(session)


class TestEachConversationIsSizedForItsModel:
    def test_the_window_the_upstream_reports_sizes_the_recap_and_is_shown_as_reported(self):
        asked: list[str] = []

        async def lookup(model: str) -> int | None:
            asked.append(model)
            return 1_000_000

        client, service, _, _ = _proxy(window_lookup=lookup)
        conversation = [("user", "we need a database"), ("assistant", "Noted."), ("user", "which one?")]
        _post(client, _body(*conversation), headers={CLIENT_SESSION_HEADER: "cc-1"})
        _post(client, _body(*conversation, ("assistant", "Postgres."), ("user", "and backups?")),
              headers={CLIENT_SESSION_HEADER: "cc-1"})

        [state_id] = service.session_manager.store.active_session_ids()
        assert service.session_manager.store.get(state_id).context_window == 1_000_000
        [conversation_stats] = client.get("/stats").json()["client_sessions"][0]["conversations"]
        assert conversation_stats["context_window"] == {"tokens": 1_000_000, "reported": True}
        assert asked == ["claude-sonnet-5"]

    def test_without_a_lookup_the_configured_window_is_used_and_said_to_be_assumed(self):
        client, service, _, _ = _proxy()
        conversation = [("user", "we need a database"), ("assistant", "Noted."), ("user", "which one?")]
        _post(client, _body(*conversation), headers={CLIENT_SESSION_HEADER: "cc-1"})

        [conversation_stats] = client.get("/stats").json()["client_sessions"][0]["conversations"]
        assert conversation_stats["context_window"] == {"tokens": service.gc_config.context_window, "reported": False}


class TestSubagentsAreRecordedNotCollected:
    """A subagent usually ends before a collection pays for itself, and its answer is only as good as what it read."""

    @staticmethod
    def _converse(client, headers, turns: int = 10) -> list[dict]:
        sent = []
        conversation: list[tuple[str, str]] = []
        for index in range(turns):
            conversation.append(("user", f"turn {index}: we discussed the storage layer at length today"))
            payload = _body(*conversation)
            _post(client, payload, headers=headers)
            sent.append(payload)
            conversation.append(("assistant", UPSTREAM_OK["content"][0]["text"]))
        return sent

    def test_a_subagents_conversation_is_forwarded_as_sent_and_never_collected(self):
        client, service, _, seen = _proxy()
        sent = self._converse(client, {CLIENT_SESSION_HEADER: "cc-1", CLIENT_AGENT_HEADER: "agent-a"})

        assert [json.loads(request.content) for request in seen] == sent
        [session_id] = service.session_manager.store.active_session_ids()
        assert service.session_manager.store.get(session_id).snapshot.cut_count == 0
        [conversation] = client.get("/stats").json()["client_sessions"][0]["conversations"]
        assert conversation["collected"] is False and conversation["collections"] == 0

    def test_the_main_conversation_beside_it_is_collected(self):
        client, service, _, _ = _proxy()
        self._converse(client, {CLIENT_SESSION_HEADER: "cc-1"})

        [session_id] = service.session_manager.store.active_session_ids()
        assert service.session_manager.store.get(session_id).snapshot.cut_count > 0

    def test_asked_to_the_proxy_collects_subagents_too(self):
        service, _ = _engine()
        app = create_app(service, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(
            lambda request: httpx2.Response(200, json=UPSTREAM_OK))), schedule=_run_now, collect_subagents=True)
        client = TestClient(app)
        self._converse(client, {CLIENT_SESSION_HEADER: "cc-1", CLIENT_AGENT_HEADER: "agent-a"})

        [session_id] = service.session_manager.store.active_session_ids()
        assert service.session_manager.store.get(session_id).snapshot.cut_count > 0


class TestEachRequestIsRecorded:
    """Shaped like a live Claude Code trace: a first request that spawns two subagents, a side request the
    conversation never continues, a subagent's relayed report. Structure is recorded, never content."""

    ASK = "Use two subagents: count the test files, and summarise the README."
    SPAWN = {**UPSTREAM_OK, "stop_reason": "tool_use", "content": [
        {"type": "tool_use", "id": "toolu_a", "name": "Agent", "input": {"prompt": "count the test files"}},
        {"type": "tool_use", "id": "toolu_b", "name": "Agent", "input": {"prompt": "summarise the README"}}]}

    def _upstream(self, request: httpx2.Request) -> httpx2.Response:
        messages = json.loads(request.content)["messages"]
        return httpx2.Response(200, json=self.SPAWN if messages[-1]["content"] == self.ASK else UPSTREAM_OK)

    def test_a_sessions_requests_are_recorded_and_settled(self):
        client, _, _, _ = _proxy(self._upstream)
        main = {CLIENT_SESSION_HEADER: "cc-1"}
        started = [{"role": "user", "content": self.ASK},
                   {"role": "assistant", "content": self.SPAWN["content"]},
                   {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_a", "content": "started"},
                                                {"type": "tool_result", "tool_use_id": "toolu_b", "content": "started"}]},
                   {"role": "assistant", "content": "Both are running."}]

        _post(client, {"model": "claude-opus-5-5", "max_tokens": 64, "messages": started[:1]}, headers=main)
        _post(client, _body(("user", "count the test files"), max_tokens=64),
              headers={**main, CLIENT_AGENT_HEADER: "agent-a"})
        _post(client, {"model": "claude-opus-5-5", "max_tokens": 64, "messages": started[:3]}, headers=main)
        _post(client, {"model": "claude-opus-5-5", "max_tokens": 64,
                       "messages": [*started, {"role": "user", "content": "Suggest the user's next message."}]},
              headers=main)
        reported = [*started, {"role": "user", "content": "<agent-message>42 test files.</agent-message>"}]
        _post(client, {"model": "claude-opus-5-5", "max_tokens": 64, "messages": reported}, headers=main)
        log = client.app.state.metrics.request_log
        # The side request and the report hold the same position: until a later request carries
        # one of them on, neither is known to be kept.
        assert (log.records()[3].kept, log.records()[4].kept) == (None, None)

        _post(client, {"model": "claude-opus-5-5", "max_tokens": 64,
                       "messages": [*reported, {"role": "assistant", "content": "42 test files."},
                                    {"role": "user", "content": "Thanks."}]}, headers=main)

        first, agent, results, side, report, thanks = log.records()
        assert [call.name for call in first.outline.tool_calls] == ["Agent", "Agent"]
        assert len({r.conversation_id for r in (first, results, side, report, thanks)}) == 1
        assert (first.kept, results.kept, side.kept, report.kept, thanks.kept) == (True, True, False, True, None)
        assert [r.newest for r in (first, results, side, report)] == ["typed", "tool_results", "typed", "relayed"]
        assert agent.origin.agent_id == "agent-a" and agent.conversation_id is None
        assert all(r.outline is not None and r.outline.status == 200 and r.ended_at is not None for r in log.records())
        recorded = repr(log.records())
        assert "test files" not in recorded and "Suggest" not in recorded

    def test_a_streamed_replys_tool_calls_and_end_are_recorded(self):
        def event(payload: dict) -> str:
            return f"event: {payload['type']}\ndata: {json.dumps(payload)}\n\n"

        body = (event({"type": "message_start", "message": {"usage": {"input_tokens": 7, "output_tokens": 1}}})
                + event({"type": "content_block_start", "index": 0,
                         "content_block": {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {}}})
                + event({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}})
                + event({"type": "message_stop"})).encode()
        client, _, _, _ = _proxy(_streaming_upstream(body))

        _post(client, _body(("user", "read the config"), stream=True))

        [record] = client.app.state.metrics.request_log.records()
        assert record.outline.finished and record.outline.stop_reason == "tool_use"
        assert [call.name for call in record.outline.tool_calls] == ["Read"]

    def test_an_unreachable_upstream_closes_the_record_as_a_502(self):
        def refuse(request):
            raise httpx2.ConnectError("refused")

        client, _, _, _ = _proxy(refuse)
        _post(client, _body(("user", "hello")))

        [record] = client.app.state.metrics.request_log.records()
        assert record.outline.status == 502 and not record.outline.finished


class TestEachSubagentIsLinkedToItsCall:
    PROMPT_A = "Count the test files under tests/ and report the total."
    PROMPT_B = "Summarise README.md in three lines for the user."
    ASK = "Use two subagents: count the test files, and summarise the README."

    def _upstream(self, request: httpx2.Request) -> httpx2.Response:
        if json.loads(request.content)["messages"][-1]["content"] != self.ASK:
            return httpx2.Response(200, json=UPSTREAM_OK)
        return httpx2.Response(200, json={**UPSTREAM_OK, "stop_reason": "tool_use", "content": [
            {"type": "tool_use", "id": "toolu_a", "name": "Agent", "input": {"prompt": self.PROMPT_A}},
            {"type": "tool_use", "id": "toolu_b", "name": "Agent", "input": {"prompt": self.PROMPT_B}}]})

    def _run(self, link_subagents: bool):
        client, _, _, _ = _proxy(self._upstream, link_subagents=link_subagents)
        main = {CLIENT_SESSION_HEADER: "cc-1"}
        _post(client, _body(("user", self.ASK)), headers=main)
        for agent, prompt in (("agent-b", self.PROMPT_B), ("agent-a", self.PROMPT_A)):
            opening = [{"type": "text", "text": "<system-reminder>Repository context.</system-reminder>"},
                       {"type": "text", "text": prompt}]
            _post(client, {"model": "claude-sonnet-5", "max_tokens": 64, "messages": [{"role": "user", "content": opening}]},
                  headers={**main, CLIENT_AGENT_HEADER: agent})
        return client.app.state.metrics.request_log.records()

    def test_each_subagents_first_request_names_the_call_that_started_it(self):
        spawner, readme, count = self._run(link_subagents=True)

        assert (count.spawned_by.request_id, count.spawned_by.call_id, count.spawned_by.tool) == (
            spawner.request_id, "toolu_a", "Agent")
        assert readme.spawned_by.call_id == "toolu_b"
        assert self.PROMPT_A not in repr((spawner, readme, count))

    def test_without_linking_no_prompt_is_read_and_nothing_is_linked(self):
        records = self._run(link_subagents=False)

        assert all(record.spawned_by is None for record in records)


class TestLlmGcYieldsWhenTheUpstreamPushesBack:
    """A 429 or 529 on a client's request pauses LLM-GC's own model calls; the client's requests go through."""

    REFUSED = {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}

    def _client(self, refuse_first: int, status: int, headers=None):
        answered = {"n": 0}

        def upstream(request):
            answered["n"] += 1
            if answered["n"] <= refuse_first:
                return httpx2.Response(status, json=self.REFUSED, headers=headers or {})
            return httpx2.Response(200, json=UPSTREAM_OK)

        return _proxy(upstream)

    @staticmethod
    def _converse(client, turns: int = 10) -> list:
        conversation: list[tuple[str, str]] = []
        responses = []
        for index in range(turns):
            conversation.append(("user", f"turn {index}: we discussed the storage layer at length today"))
            responses.append(_post(client, _body(*conversation)))
            conversation.append(("assistant", UPSTREAM_OK["content"][0]["text"]))
        return responses

    def test_a_rate_limited_request_pauses_llm_gc_for_as_long_as_the_upstream_asked(self):
        client, service, _, _ = self._client(refuse_first=1, status=429, headers={"retry-after": "30"})
        before = datetime.now(timezone.utc)

        response = _post(client, _body(("user", "hello")))

        assert response.status_code == 429 and response.json() == self.REFUSED, "the client gets the upstream's answer"
        pause = service.model_calls.current()
        assert pause is not None and "rate-limited a request from the client" in pause.reason
        assert timedelta(seconds=29) < pause.until - before < timedelta(seconds=32)

    def test_while_paused_requests_go_through_and_nothing_is_collected(self):
        client, service, _, _ = self._client(refuse_first=1, status=529)
        _post(client, _body(("user", "hello")))

        responses = self._converse(client)

        assert all(response.status_code == 200 for response in responses)
        [session_id] = service.session_manager.store.active_session_ids()
        assert service.session_manager.store.get(session_id).snapshot.cut_count == 0


class TestTheCacheIsWatched:
    """From each answered request, the proxy learns whether its conversation's cache would serve the next one."""

    CACHE_READ = {"input_tokens": 12, "cache_read_input_tokens": 54_000, "cache_creation_input_tokens": 1_700,
                  "output_tokens": 18}

    @staticmethod
    def _answered(usage: dict, *, status: int = 200, ttl: str | None = None, took_s: float = 0.0):
        """One request answered with `usage`, its reply taking `took_s` on the watch's clock."""
        reply = {**UPSTREAM_OK, "usage": usage}
        clock = [1_000.0]

        def upstream(request: httpx2.Request) -> httpx2.Response:
            clock[0] += took_s
            return httpx2.Response(status, json=reply)

        client, service, _, _ = _proxy(upstream)
        watch = client.app.state.metrics.cache_watch = CacheWatch(clock=lambda: clock[0])
        marker = {"type": "ephemeral", **({"ttl": ttl} if ttl is not None else {})}
        _post(client, _body(("user", "about databases"), ("assistant", "ok"), ("user", "go on"),
                            cache_control=marker))
        [session_id] = service.session_manager.store.active_session_ids()
        return watch, clock, session_id

    def test_a_reply_that_read_the_cache_keeps_the_conversation_warm_until_its_lifetime_passes(self):
        watch, clock, session_id = self._answered(self.CACHE_READ)
        clock[0] += 60
        assert watch.state(session_id) is CacheState.WARM
        clock[0] += 300
        assert watch.state(session_id) is CacheState.COLD

    def test_the_lifetime_counts_from_when_the_request_was_forwarded_not_when_its_reply_ended(self):
        watch, clock, session_id = self._answered(self.CACHE_READ, took_s=240.0)
        clock[0] = 1_000.0 + 310
        assert watch.state(session_id) is CacheState.COLD

    def test_the_lifetime_the_request_asked_for_is_the_one_used(self):
        watch, clock, session_id = self._answered(self.CACHE_READ, ttl="1h")
        clock[0] += 20 * 60
        assert watch.state(session_id) is CacheState.WARM

    def test_a_conversation_whose_replies_use_no_cache_is_cold(self):
        watch, _, session_id = self._answered(UPSTREAM_OK["usage"])
        assert watch.state(session_id) is CacheState.COLD

    def test_a_refused_request_tells_nothing_about_the_cache(self):
        """A 429 never reached the cache: the conversation stays unknown, which counts as warm."""
        watch, clock, session_id = self._answered(self.CACHE_READ, status=429)
        clock[0] += 3600
        assert watch.state(session_id) is CacheState.WARM


class TestAPreparedCutWaitsForTheRightRequest:
    """With the cache warm, a prepared cut waits; a cold cache or the stale bound switches it in."""

    CACHED = {**UPSTREAM_OK, "usage": {"input_tokens": 12, "cache_read_input_tokens": 54_000,
                                        "cache_creation_input_tokens": 1_700, "output_tokens": 18}}
    # No saving repays a thousandfold, so on a warm cache only the bound could switch a cut in. On the
    # 200-token window, eight exchanges stage a cut of about 156 stale tokens: under a bound of the
    # whole window, over one of half of it.
    PATIENT = {"min_saving_share": 1000.0}

    def _client(self, **config):
        client, service, _, seen = _proxy(lambda request: httpx2.Response(200, json=self.CACHED), **config)
        clock = [1_000.0]
        client.app.state.metrics.cache_watch = CacheWatch(clock=lambda: clock[0])
        return client, service, seen, clock

    @staticmethod
    def _ask(client, seen, history: list[dict]) -> list[dict]:
        """One more exchange of the same conversation; returns the messages forwarded for it."""
        history.append({"role": "user", "content": f"question {len(history) // 2}: we went over the storage layer, "
                                                   "retention, and the rollout plan again"})
        client.post("/v1/messages", json={"model": "m", "max_tokens": 64, "messages": history},
                    headers={"x-api-key": API_KEY, SESSION_HEADER: "s"})
        history.append({"role": "assistant", "content": UPSTREAM_OK["content"]})
        return json.loads(seen[-1].content)["messages"]

    def test_on_a_warm_cache_a_cut_that_would_not_repay_its_rewrite_waits(self):
        client, service, seen, _ = self._client(max_stale_ratio=1.0, **self.PATIENT)
        history: list[dict] = []
        forwarded = [self._ask(client, seen, history) for _ in range(8)]

        assert not any(_is_front(message) for request in forwarded for message in request), "sent as written"
        assert service.session_manager.store.get("s").snapshot.staged is not None, "prepared, waiting"

    def test_the_waiting_cut_is_switched_in_once_the_cache_goes_cold(self):
        client, service, seen, clock = self._client(max_stale_ratio=1.0, **self.PATIENT)
        history: list[dict] = []
        for _ in range(8):
            self._ask(client, seen, history)

        clock[0] += 3600   # the person steps away; the provider's entry expires
        forwarded = self._ask(client, seen, history)

        assert _is_front(forwarded[0]), "the cold request carries the collection"
        snapshot = service.session_manager.store.get("s").snapshot
        assert snapshot.staged is None and snapshot.cut_count == 1

    def test_on_a_warm_cache_the_stale_bound_switches_it_in_for_the_models_sake(self):
        client, service, seen, _ = self._client(max_stale_ratio=0.5, **self.PATIENT)
        reasons = []
        service.event_bus.subscribe(EventType.GC_FINISHED, lambda event: reasons.append(event.data["gc_result"].switch_reason))
        history: list[dict] = []
        forwarded = [self._ask(client, seen, history) for _ in range(8)]

        assert any(_is_front(message) for message in forwarded[-1])
        assert SwitchReason.STALE_WEIGHT_AT_BOUND in reasons and SwitchReason.CACHE_COLD not in reasons


class TestTheNewestRequestIsCountedBothWays:
    def test_llm_gcs_count_and_the_providers_are_kept_for_the_same_request(self):
        usage = {"input_tokens": 40, "cache_read_input_tokens": 900, "cache_creation_input_tokens": 60, "output_tokens": 7}
        client, service, _, _ = _proxy(lambda request: httpx2.Response(200, json={**UPSTREAM_OK, "usage": usage}))
        _post(client, _body(("user", "about databases"), ("assistant", "ok"), ("user", "go on")))
        [session_id] = service.session_manager.store.active_session_ids()

        stats = client.get("/stats").json()
        conversation = next(c for c in stats["other_conversations"] if c["conversation_id"] == session_id)
        last = conversation["traffic"]["last_reply"]

        assert last["tokens_forwarded"] == conversation["traffic"]["tokens_forwarded"], "its only request"
        assert last["provider_prompt_tokens"] == 40 + 900 + 60

