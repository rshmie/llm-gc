"""The dashboard over live sessions: read-only, per session, and honest about absence."""

import asyncio
from pathlib import Path

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from llm_gc.config import GCConfig
from llm_gc.engine import build_gc_service
from llm_gc.engine.gc_service import GcService
from llm_gc.llm import FakeProvider, LLMResponse, StopReason
from llm_gc.models import Message
from llm_gc.monitoring import ContextHealthMonitor
from llm_gc.utils import count_tokens
from llm_gc.visualizer import TranscriptRecorder, create_dashboard_app

SUMMARY = LLMResponse(text="Earlier turns covered the storage decision and the retention window in brief.",
                      stop_reason=StopReason.COMPLETE, input_tokens=0, output_tokens=14,
                      model="scripted", provider="offline")


def _engine() -> tuple[GcService, ContextHealthMonitor]:
    config = GCConfig(context_window=700, keep_threshold=0.55, archive_threshold=0.25, last_n_turns_to_keep=3,
                      min_cut_tokens=60)
    service = build_gc_service(config, summary_provider=FakeProvider(responses=[SUMMARY] * 400))
    bus = service.event_bus
    store = service.session_manager.store
    monitor = ContextHealthMonitor(
        event_bus=bus, gc_config=config,
        read_generations=lambda session_id: (state.snapshot if session_id is not None
                                             and (state := store.get(session_id)) is not None else None))
    return service, monitor


def _converse(service: GcService, session_id: str, turns: int) -> None:
    async def scenario() -> None:
        for index in range(turns):
            text = f"the topic{index % 8} is version {index} and we walked through the implications at length."
            turn = Message(role="user" if index % 2 == 0 else "assistant", content=text,
                           token_count=count_tokens(text), turn_index=index)
            service.gc_record(session_id, [turn], [f"{session_id}-{index}"])
            await service.gc_age(session_id)

    asyncio.run(scenario())


def _client(service: GcService, monitor: ContextHealthMonitor, **kwargs) -> TestClient:
    return TestClient(create_dashboard_app(service, monitor, **kwargs), base_url="http://127.0.0.1")


class TestThePage:
    def test_serves_the_dashboard(self):
        response = _client(*_engine()).get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_serves_its_styles_and_scripts(self):
        client = _client(*_engine())
        assert "text/css" in client.get("/static/css/dashboard.css").headers["content-type"]
        assert "javascript" in client.get("/static/js/main.js").headers["content-type"]

    def test_the_browser_revalidates_the_page_and_its_scripts(self):
        """Left to guess a cache lifetime, a browser kept running the old scripts after
        they changed, and an edited dashboard looked unedited."""
        client = _client(*_engine())
        for path in ("/", "/static/js/main.js", "/static/css/dashboard.css"):
            assert client.get(path).headers["cache-control"] == "no-cache", path

    def test_its_one_action_is_the_manual_cut(self):
        app = create_dashboard_app(*_engine())
        writes = [route.path for route in app.routes if isinstance(route, APIRoute) and route.methods - {"GET"}]
        assert writes == ["/api/sessions/{session_id}/gc"]

    def test_it_answers_only_to_localhost_names(self):
        """A page on another site can point its own hostname at 127.0.0.1 and read
        the dashboard as same-origin - every conversation the proxy holds."""
        service, monitor = _engine()
        rebound = TestClient(create_dashboard_app(service, monitor), base_url="http://attacker.example")
        assert rebound.get("/api/overview").status_code == 400
        assert _client(service, monitor).get("/api/overview").status_code == 200


class TestOverview:
    def test_lists_sessions_most_recent_first_with_their_counts(self):
        service, monitor = _engine()
        _converse(service, "older", 4)
        _converse(service, "newer", 30)

        sessions = _client(service, monitor).get("/api/overview").json()["sessions"]

        assert [s["session_id"] for s in sessions] == ["newer", "older"]
        assert sessions[0]["recorded_turns"] == 30
        assert sessions[0]["cuts"] > 0
        assert sessions[1]["cuts"] == 0

    def test_a_conversation_is_named_by_its_opening_words(self):
        """Session ids are hashes; a person finds their conversation by what they said first."""
        service, monitor = _engine()
        _converse(service, "named", 2)
        _converse(service, "unnamed", 2)
        recorder = TranscriptRecorder()
        opening = "We need to decide on the storage layer\nfor the new ingest service " + "before Friday " * 10
        recorder.record("named", [Message(role="assistant", content="hello", turn_index=0),
                                  Message(role="user", content=opening, turn_index=1)])

        sessions = _client(service, monitor, transcripts=recorder).get("/api/overview").json()["sessions"]
        titles = {s["session_id"]: s["title"] for s in sessions}

        assert titles["named"].startswith("We need to decide on the storage layer for the new ingest service")
        assert titles["named"].endswith("…")
        assert len(titles["named"]) <= 72
        assert titles["unnamed"] is None

    def test_reports_the_tunable_settings(self):
        config = _client(*_engine()).get("/api/overview").json()["config"]
        assert config["min_cut_tokens"] == 60
        assert config["last_n_turns_to_keep"] == 3

    def test_proxy_counters_appear_only_when_a_proxy_supplies_them(self):
        service, monitor = _engine()
        assert _client(service, monitor).get("/api/overview").json()["proxy"] is None
        stats = _client(service, monitor, read_proxy_stats=lambda: {"requests": 7}).get("/api/overview").json()
        assert stats["proxy"] == {"requests": 7}


class TestOneSession:
    def test_reports_where_every_turn_is(self):
        service, monitor = _engine()
        _converse(service, "s", 30)

        state = _client(service, monitor).get("/api/sessions/s").json()

        assert len(state["turns"]) == 30
        assert {t["fidelity"] for t in state["turns"]} >= {"verbatim", "summarised"}
        assert state["front_text"] is not None

    def test_activity_is_the_sessions_own(self):
        service, monitor = _engine()
        _converse(service, "busy", 30)
        _converse(service, "quiet", 2)
        client = _client(service, monitor)

        busy = client.get("/api/sessions/busy/activity").json()
        quiet = client.get("/api/sessions/quiet/activity").json()

        assert busy["generation_lifecycle"]["recent_transitions"]
        assert quiet["generation_lifecycle"]["recent_transitions"] == []

    def test_an_unknown_session_is_a_structured_404_and_is_not_created(self):
        service, monitor = _engine()
        client = _client(service, monitor)

        for path in ("/api/sessions/gone", "/api/sessions/gone/activity"):
            response = client.get(path)
            assert response.status_code == 404
            assert response.json()["error"]["code"] == "SESSION_NOT_FOUND"
        assert service.session_manager.store.get("gone") is None

    def test_watching_a_session_does_not_keep_it_alive(self):
        """The idle sweeper closes sessions by last activity. A dashboard left open
        on a session must not count as activity, or no watched session would ever close."""
        service, monitor = _engine()
        _converse(service, "s", 3)
        before = service.session_manager.store.get("s").last_updated_at

        client = _client(service, monitor)
        client.get("/api/sessions/s")
        client.get("/api/sessions/s/activity")
        client.get("/api/overview")

        assert service.session_manager.store.get("s").last_updated_at == before


class TestRunGcNow:
    def test_it_cuts_without_waiting_for_the_batch_minimum(self):
        service, monitor = _engine()
        service.gc_config.min_cut_tokens = 100_000      # the automatic pass would never cut
        _converse(service, "s", 20)
        assert service.session_manager.store.get("s").snapshot.cut_count == 0

        response = _client(service, monitor).post("/api/sessions/s/gc", headers={"x-llm-gc-dashboard": "1"})

        assert response.status_code == 200
        assert response.json()["status"] == "completed"
        assert response.json()["tokens_after"] < response.json()["tokens_before"]
        assert service.session_manager.store.get("s").snapshot.cut_count == 1
        outcome = response.json()["outcome"]
        assert outcome["quoted"] + outcome["summarised"] + outcome["archived"] > 0, "it says where the turns went"

    def test_it_is_refused_without_the_action_header(self):
        """Another site's page can make the browser send a plain POST to localhost;
        it cannot add a custom header without a preflight this server never grants."""
        service, monitor = _engine()
        _converse(service, "s", 20)
        before = service.session_manager.store.get("s").snapshot

        response = _client(service, monitor).post("/api/sessions/s/gc")

        assert response.status_code == 403
        assert service.session_manager.store.get("s").snapshot is before, "nothing ran"

    def test_an_unknown_session_is_not_created(self):
        service, monitor = _engine()
        response = _client(service, monitor).post("/api/sessions/gone/gc", headers={"x-llm-gc-dashboard": "1"})
        assert response.status_code == 404
        assert service.session_manager.store.get("gone") is None


class TestTranscript:
    def test_it_returns_what_the_client_last_sent(self):
        service, monitor = _engine()
        _converse(service, "s", 3)
        recorder = TranscriptRecorder()
        recorder.record("s", [Message(role="user", content="the database is postgres", token_count=5, turn_index=0),
                              Message(role="assistant", content="noted", token_count=1, turn_index=1)])

        transcript = _client(service, monitor, transcripts=recorder).get("/api/sessions/s/transcript").json()

        assert transcript["captured"] is True
        assert [(t["turn_index"], t["role"], t["text"]) for t in transcript["turns"]] == [
            (0, "user", "the database is postgres"), (1, "assistant", "noted")]
        assert transcript["total_tokens"] == 6

    def test_tokens_are_compared_over_the_same_turns(self):
        """The client's latest message reaches the session only with the next request.
        Comparing the whole transcript with the context the model is sent showed a
        saving equal to that message - a saving that never happened."""
        service, monitor = _engine()
        _converse(service, "s", 2)                     # the session has recorded turns 0 and 1
        recorder = TranscriptRecorder()
        recorder.record("s", [Message(role="user", content="a", token_count=5, turn_index=0),
                              Message(role="assistant", content="b", token_count=7, turn_index=1),
                              Message(role="user", content="c", token_count=11, turn_index=2)])

        transcript = _client(service, monitor, transcripts=recorder).get("/api/sessions/s/transcript").json()

        assert transcript["total_tokens"] == 23
        assert transcript["recorded_tokens"] == 12

    def test_without_a_recorder_it_says_nothing_was_captured(self):
        service, monitor = _engine()
        _converse(service, "s", 3)
        transcript = _client(service, monitor).get("/api/sessions/s/transcript").json()
        assert transcript == {"captured": False, "turns": [], "total_tokens": 0, "recorded_tokens": 0}

    def test_a_closed_sessions_copy_is_dropped(self):
        service, monitor = _engine()
        recorder = TranscriptRecorder()
        recorder.record("gone", [Message(role="user", content="secret", turn_index=0)])

        assert _client(service, monitor, transcripts=recorder).get("/api/sessions/gone/transcript").status_code == 404
        assert recorder.get("gone") is None


def test_the_page_sets_children_only_through_its_helper():
    """The DOM's own replaceChildren prints an absent part as "null" and a nested
    list as "[object HTMLDivElement]" - which reached the screen twice. Only the
    page's setChildren, which skips and flattens, may call it."""
    static = Path(__file__).parents[2] / "src/llm_gc/visualizer/static"
    page = "\n".join(path.read_text() for path in sorted(static.glob("js/*.js")))
    calls = [line.strip() for line in page.splitlines() if ".replaceChildren(" in line and not line.strip().startswith("//")]
    assert calls == ["export function setChildren(target, ...children) { target.replaceChildren(...nodes(children)); }"]
