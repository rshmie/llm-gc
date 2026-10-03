"""The proxy, driven through its real routes with no network.

`MockTransport` intercepts below httpx2's request machinery, so parsing, session
resolution, GC, header filtering and forwarding all run for real. The one thing
stubbed is the socket.
"""

import asyncio
import json

import httpx2
import pytest
from fastapi.testclient import TestClient

from llm_gc.config import GCConfig
from llm_gc.engine.compaction import LLMCompactor
from llm_gc.engine.gc_service import GcService
from llm_gc.engine.generations import GenerationalMemory
from llm_gc.engine.generations.permanent_generation import PermanentGeneration
from llm_gc.engine.sweep import Sweeper, ThresholdSweepStrategy
from llm_gc.events import EventBus
from llm_gc.extraction import KnowledgeExtractor
from llm_gc.llm import FakeProvider, LLMResponse, StopReason
from llm_gc.proxy import SESSION_HEADER, create_app
from llm_gc.scoring import RecencyScorer, RelevanceScorer, RelevanceType
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
    bus = EventBus()
    base = {
        "context_window": 200,
        "gc_threshold": 0.4,
        "keep_threshold": 0.55,
        "archive_threshold": 0.25,
        "min_compactable_tokens": 10,
        "last_n_turns_to_keep": 2,
        "max_memory_ratio": 0.1,
        "max_memory_facts": 5,
    }
    base.update(config_overrides)
    config = GCConfig(**base)
    memory = GenerationalMemory(
        event_bus=bus,
        knowledge_extractor=KnowledgeExtractor(event_bus=bus),
        permanent_generation=PermanentGeneration(event_bus=bus),
        compactor=LLMCompactor(event_bus=bus, provider=FakeProvider(responses=[SUMMARY] * 200)),
    )
    return GcService(
        session_manager=SessionManager(),
        generational_memory=memory,
        relevance_scorer=RelevanceScorer(
            scorers=[RecencyScorer(decay_rate=0.3)], weights={RelevanceType.RECENCY: 1.0}, event_bus=bus
        ),
        sweeper=Sweeper(
            sweeper_strategy=ThresholdSweepStrategy(gc_config=config), gc_config=config, event_bus=bus
        ),
        event_bus=bus,
        gc_config=config,
    ), memory


def _proxy(upstream=None, *, gc_service=None, **engine_kwargs):
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


class TestSessionIdentity:
    def test_an_explicit_session_header_is_honoured(self):
        """A client that knows its own conversation identity is authoritative, and
        nothing we infer can beat it."""
        client, service, _, _ = _proxy()
        _post(client, _body(("user", "hello")), headers={SESSION_HEADER: "my-session"})

        assert service.session_manager.store.get("my-session") is not None

    def test_without_a_header_the_opening_turns_are_fingerprinted(self):
        client, service, _, _ = _proxy()
        _post(client, _body(("user", "a distinctive opening line")))

        ids = service.session_manager.store.active_session_ids()
        assert len(ids) == 1
        assert ids[0].startswith("fp_")

    def test_the_same_conversation_resolves_to_one_session(self):
        """The property a fingerprint exists for: stable as the conversation grows."""
        client, service, _, _ = _proxy()
        _post(client, _body(("user", "opening line"), ("assistant", "reply")))
        _post(client, _body(("user", "opening line"), ("assistant", "reply"), ("user", "more")))

        assert service.session_manager.store.count() == 1

    def test_different_conversations_get_different_sessions(self):
        client, service, _, _ = _proxy()
        _post(client, _body(("user", "conversation about databases")))
        _post(client, _body(("user", "conversation about scheduling")))

        assert service.session_manager.store.count() == 2

    def test_the_session_id_carries_no_conversation_text(self):
        """A session id reaches logs and events, and conversation content is user
        data. A hash identifies without carrying what was said."""
        client, service, _, _ = _proxy()
        _post(client, _body(("user", "my bank password is hunter2")))

        assert all("hunter2" not in sid for sid in service.session_manager.store.active_session_ids())


class TestPostResponseAging:
    def test_both_halves_of_the_exchange_are_recorded(self):
        """Recording only the user's turn would leave the engine scoring a
        conversation in which the assistant never spoke — and half of what a
        conversation decides is in the replies."""
        client, service, _, _ = _proxy()
        _post(client, _body(("user", "which database?")), headers={SESSION_HEADER: "s"})

        state = service.session_manager.store.get("s")
        assert state is not None
        roles = [m.role for m in state.messages]
        assert roles == ["user", "assistant"]
        assert "Postgres" in state.messages[-1].content

    def test_nothing_is_recorded_when_the_provider_fails(self):
        """Session state would otherwise describe a conversation that never
        happened — and the client will retry, recording the turn twice.

        The session *object* does exist: `gc_collect` runs before forwarding and
        creates it. What must be empty is its turn list."""
        client, service, _, _ = _proxy(lambda request: httpx2.Response(500, json={"error": "boom"}))
        _post(client, _body(("user", "which database?")), headers={SESSION_HEADER: "s"})

        state = service.session_manager.store.get("s")
        assert state is not None, "gc_collect creates the session before we forward"
        assert state.messages == [], "but a failed exchange records no turns"

    def test_an_unparseable_reply_still_records_the_user_turn(self):
        """A 200 we cannot read costs one turn of scoring accuracy, not the turn
        the user actually sent."""
        client, service, _, _ = _proxy(lambda request: httpx2.Response(200, text="not json"))
        _post(client, _body(("user", "which database?")), headers={SESSION_HEADER: "s"})

        state = service.session_manager.store.get("s")
        assert state is not None
        assert [m.role for m in state.messages] == ["user"]


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
        _post(client, _body(("user", "hello")), headers={SESSION_HEADER: "s"})

        state = service.session_manager.store.get("s")
        assert state is not None
        assert API_KEY not in repr(state)


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

            async def gc_collect(self, session_id, query=None):
                raise RuntimeError("engine exploded")

            async def gc_update(self, session_id, new_turn):
                raise RuntimeError("engine exploded")

        client, _, _, _ = _proxy(gc_service=_BrokenService())
        _post(client, _body(("user", "hello")))

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

            async def gc_collect(self, session_id, query=None):
                raise RuntimeError("engine exploded")

            async def gc_update(self, session_id, new_turn):
                raise RuntimeError("engine exploded")

        client, _, _, _ = _proxy(gc_service=_BrokenService())
        _post(client, _body(("user", "a turn with some real length to it here")))

        stats = client.get("/stats").json()
        assert stats["tokens_forwarded"] == stats["tokens_in_client_history"]
        assert stats["tokens_saved"] == 0


class TestContentBlocks:
    def test_a_block_list_is_flattened_for_scoring(self):
        client, service, _, _ = _proxy()
        client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-5",
                "max_tokens": 512,
                "messages": [{"role": "user", "content": [{"type": "text", "text": "which database?"}]}],
            },
            headers={"x-api-key": API_KEY, SESSION_HEADER: "s"},
        )

        state = service.session_manager.store.get("s")
        assert state is not None
        assert state.messages[0].content == "which database?"

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


class TestTheMemoryBudgetCannotInvertTheSaving:
    """A regression guard with a measured origin.

    With an absolute memory budget, `max_memory_tokens=100` against a 200-token
    window made the proxy forward 21% MORE than the client sent: the injected block
    consumed the whole saving and then some, silently. The budget is now a fraction
    of the window, so the same mistake is not expressible.
    """

    @staticmethod
    def _savings(ratio: float) -> dict:
        client, _, _, _ = _proxy(max_memory_ratio=ratio)
        conversation: list[tuple[str, str]] = []
        for index in range(10):
            conversation.append(("user", f"turn {index}: we discussed the storage layer at length today"))
            _post(client, _body(*conversation))
            conversation.append(("assistant", UPSTREAM_OK["content"][0]["text"]))
        return client.get("/stats").json()

    def test_the_default_proportion_still_saves(self):
        stats = self._savings(0.1)
        assert stats["tokens_saved"] > 0, stats

    def test_even_a_generous_proportion_still_saves(self):
        """The property the ratio buys: memory scales with the window, so a
        generous share of a window is still a share of it."""
        stats = self._savings(0.25)
        assert stats["tokens_saved"] > 0, stats

    def test_the_budget_is_derived_from_the_window(self):
        from llm_gc.config import GCConfig

        assert GCConfig(context_window=1000, max_memory_ratio=0.1).max_memory_tokens == 100
        assert GCConfig(context_window=200, max_memory_ratio=0.1).max_memory_tokens == 20
        # One source of truth: there is no stored absolute to drift from the window.
        assert GCConfig(context_window=128_000, max_memory_ratio=0.0).max_memory_tokens == 0


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

    def test_both_halves_of_a_streamed_exchange_are_recorded(self):
        """The defect this suite exists for: before streaming was handled, the
        reply was dropped and the engine scored a conversation with a silent
        assistant."""
        client, service, _, _ = _proxy(_streaming_upstream(_sse("Postgres, ", "partitioned by day.")))
        _post(client, _body(("user", "which database?"), stream=True), headers={SESSION_HEADER: "s"})

        state = service.session_manager.store.get("s")
        assert [m.role for m in state.messages] == ["user", "assistant"]
        assert state.messages[-1].content == "Postgres, partitioned by day."

    @pytest.mark.parametrize("broken", [_sse("half an", stop=False), _sse("partial", error=True)],
                             ids=["truncated", "provider-error-event"])
    def test_an_unfinished_stream_records_nothing_and_says_so(self, broken):
        """The client will retry it; recording it would put the turn in twice."""
        client, service, _, _ = _proxy(_streaming_upstream(broken))
        _post(client, _body(("user", "which database?"), stream=True), headers={SESSION_HEADER: "s"})

        assert service.session_manager.store.get("s").messages == []
        assert client.get("/stats").json()["streams_unrecorded"] == 1

    def test_an_error_status_on_a_stream_request_passes_through_verbatim(self):
        error = {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}
        client, service, _, _ = _proxy(lambda request: httpx2.Response(429, json=error))
        response = _post(client, _body(("user", "which database?"), stream=True), headers={SESSION_HEADER: "s"})

        assert response.status_code == 429
        assert response.json() == error
        assert service.session_manager.store.get("s").messages == []

    def test_a_gc_failure_still_streams_the_original_request(self, monkeypatch):
        body = _sse("fine")
        client, service, _, seen = _proxy(_streaming_upstream(body))

        async def broken(*args, **kwargs):
            raise RuntimeError("engine bug")

        monkeypatch.setattr(service, "gc_collect", broken)
        response = _post(client, _body(("user", "which database?"), stream=True))
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

        monkeypatch.setattr("llm_gc.proxy.server.resolve_session_id", broken)
        client, _, _, _ = _proxy()
        response = _post(client, _body(("user", "which database?")))
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
        response = _post(client, _body(("user", "which database?"), stream=stream))
        assert response.status_code == 200
        assert response.content == body if stream else response.json() == UPSTREAM_OK

    def test_an_observer_failure_does_not_cut_the_stream(self, monkeypatch):
        def broken(self, chunk):
            raise RuntimeError("parser bug")

        monkeypatch.setattr("llm_gc.proxy.server.StreamedReply.feed", broken)
        body = _sse("Postgres, ", "partitioned by day.")
        client, service, _, _ = _proxy(_streaming_upstream(body))
        response = _post(client, _body(("user", "which database?"), stream=True), headers={SESSION_HEADER: "s"})

        assert response.content == body
        assert service.session_manager.store.get("s").messages == []
        assert client.get("/stats").json()["streams_unrecorded"] == 1
