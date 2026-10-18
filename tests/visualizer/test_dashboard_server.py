"""The dashboard over live sessions: read-only, per session, and honest about absence."""

import asyncio
from pathlib import Path

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from llm_gc.config import GCConfig
from llm_gc.engine import build_gc_service
from llm_gc.engine.gc_service import GcService
from llm_gc.engine.switch_policy import CacheReading, CacheState
from llm_gc.llm import FakeProvider, LLMResponse, StopReason
from llm_gc.models import ClientOrigin, Message
from llm_gc.monitoring import ContextHealthMonitor
from llm_gc.utils import count_tokens
from llm_gc.visualizer import TranscriptRecorder, create_dashboard_app

from tests.engine.support import collect
from tests.models.support import NOTICE, SKILL_RULES, turn_with_client_text

SUMMARY = LLMResponse(text="Earlier turns covered the storage decision and the retention window in brief.",
                      stop_reason=StopReason.COMPLETE, input_tokens=0, output_tokens=14,
                      model="scripted", provider="offline")


def _engine(*, model: bool = True) -> tuple[GcService, ContextHealthMonitor]:
    config = GCConfig(context_window=700, keep_threshold=0.55, archive_threshold=0.25, last_n_turns_to_keep=3,
                      min_cut_tokens=60)
    service = build_gc_service(config, summary_provider=FakeProvider(responses=[SUMMARY] * 400) if model else None)
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
            text = (f"the topic{index % 8} is version {index}. "
                    + "We walked through the implications for throughput and retention at length. " * 4)
            turn = Message(role="user" if index % 2 == 0 else "assistant", content=text,
                           token_count=count_tokens(text), turn_index=index)
            service.gc_record(session_id, [turn], [f"{session_id}-{index}"])
            await collect(service, session_id)

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

    def test_a_conversation_is_named_by_the_words_the_person_wrote_not_by_claude_codes(self):
        """A conversation opened after a local command starts with the caveat Claude Code writes."""
        service, monitor = _engine()
        _converse(service, "named", 2)
        recorder = TranscriptRecorder()
        caveat = "<local-command-caveat>Caveat: generated by local commands.</local-command-caveat>"
        recorder.record("named", [turn_with_client_text(0, (True, caveat), (False, "Which runner is faster?"))])

        sessions = _client(service, monitor, transcripts=recorder).get("/api/overview").json()["sessions"]

        assert sessions[0]["title"] == "Which runner is faster?"

    def test_each_conversation_says_which_client_session_and_agent_it_came_from(self):
        """What lets the picker group a Claude Code session's main conversation with its sub-agents."""
        service, monitor = _engine()
        for session_id in ("main", "agent", "other"):
            _converse(service, session_id, 2)
        recorder = TranscriptRecorder()
        turn = [Message(role="user", content="hello", turn_index=0)]
        recorder.record("main", turn, ClientOrigin(session_id="c7a78c2f"))
        recorder.record("agent", turn, ClientOrigin(session_id="c7a78c2f", agent_id="a1c50dab"))
        recorder.record("other", turn)

        sessions = _client(service, monitor, transcripts=recorder).get("/api/overview").json()["sessions"]
        origins = {s["session_id"]: (s["client_session"], s["agent"]) for s in sessions}

        assert origins == {"main": ("c7a78c2f", None), "agent": ("c7a78c2f", "a1c50dab"), "other": (None, None)}
        assert all(s["started_at"] <= s["last_activity"] for s in sessions)

    def test_reports_the_tunable_settings(self):
        config = _client(*_engine()).get("/api/overview").json()["config"]
        assert config["min_cut_tokens"] == 60
        assert config["last_n_turns_to_keep"] == 3

    def test_says_whose_turns_facts_are_read_from(self):
        # A model reads facts from both sides; the patterns used without one read only the user's,
        # and the page says so rather than showing the assistant as having settled nothing.
        assert _client(*_engine()).get("/api/overview").json()["config"]["facts_read_from"] == ["assistant", "user"]
        assert _client(*_engine(model=False)).get("/api/overview").json()["config"]["facts_read_from"] == ["user"]

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
        fidelities = {t["fidelity"] for t in state["turns"]}
        assert "verbatim" in fidelities and fidelities - {"verbatim"}, "the tail, and turns condensed before it"
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
        assert outcome["kept"] + outcome["compacted"] + outcome["archived"] + outcome["folded"], \
            "it names where the turns went"

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

    def test_a_turn_claude_code_wrote_part_of_comes_with_its_parts_by_writer(self):
        """Sent as text, not as character ranges, which JavaScript would count differently."""
        service, monitor = _engine()
        _converse(service, "s", 2)
        recorder = TranscriptRecorder()
        recorder.record("s", [turn_with_client_text(0, (False, "Run the audit 🙂"), (True, SKILL_RULES)),
                              Message(role="assistant", content="ok", token_count=1, turn_index=1),
                              turn_with_client_text(2, (True, NOTICE))])

        turns = _client(service, monitor, transcripts=recorder).get("/api/sessions/s/transcript").json()["turns"]

        assert turns[0]["parts"] == [{"by_client": False, "text": "Run the audit 🙂"},
                                     {"by_client": True, "text": SKILL_RULES}]
        assert turns[1]["parts"] is None
        assert turns[2]["parts"] == [{"by_client": True, "text": NOTICE}]

    def test_instructions_are_sized_as_the_client_sends_them_now(self):
        """The record keeps a turn as first seen; Claude Code rewrites the instructions in
        earlier turns, and the tag showed 263 tokens while 4,303 went with each request."""
        service, monitor = _engine()
        _converse(service, "s", 2)
        recorder = TranscriptRecorder()
        rewritten = "<system-reminder>CLAUDE.md, memory, date</system-reminder>"
        recorder.record("s", [Message(role="user", content="hello", token_count=1, turn_index=0, instructions=(rewritten,)),
                              Message(role="assistant", content="hi", token_count=1, turn_index=1)])

        transcript = _client(service, monitor, transcripts=recorder).get("/api/sessions/s/transcript").json()

        assert [t["instruction_tokens"] for t in transcript["turns"]] == [count_tokens(rewritten), 0]

    def test_the_newest_reply_is_shown_until_the_next_request_records_it(self):
        service, monitor = _engine()
        _converse(service, "s", 2)
        recorder = TranscriptRecorder()
        asked = Message(role="user", content="which database?", token_count=3, turn_index=2)
        recorder.record("s", [Message(role="user", content="a", token_count=1, turn_index=0),
                              Message(role="assistant", content="b", token_count=1, turn_index=1), asked])

        recorder.record_reply("s", asked, "Postgres.")
        client = _client(service, monitor, transcripts=recorder)
        reply = client.get("/api/sessions/s/transcript").json()["reply"]
        assert (reply["turn_index"], reply["role"], reply["text"]) == (3, "assistant", "Postgres.")
        # It arrives without changing the session's version, so the polled list says it is there.
        assert client.get("/api/overview").json()["sessions"][0]["newest_reply"] == 3

        recorder.record("s", [Message(role="user", content="a", token_count=1, turn_index=0)])
        assert recorder.reply("s") is None, "the next request carries it, or the client moved on without it"

    def test_a_late_reply_to_an_earlier_message_is_not_shown_under_the_newest(self):
        """A client can have several requests in flight; the slower one finishing last
        put its answer under a question it did not answer."""
        recorder = TranscriptRecorder()
        earlier = Message(role="user", content="which database?", token_count=3, turn_index=0)
        recorder.record("s", [earlier, Message(role="assistant", content="Postgres.", token_count=2, turn_index=1),
                              Message(role="user", content="and the cache?", token_count=4, turn_index=2)])

        recorder.record_reply("s", earlier, "Postgres.")
        recorder.record_reply("s", Message(role="user", content="suggest a prompt", token_count=3, turn_index=2), "x")

        assert recorder.reply("s") is None

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
        assert transcript == {"captured": False, "turns": [], "total_tokens": 0, "recorded_tokens": 0, "reply": None}

    def test_a_request_that_names_no_client_session_keeps_the_one_named_before(self):
        recorder = TranscriptRecorder()
        recorder.record("s", [Message(role="user", content="a", turn_index=0)], ClientOrigin(session_id="c7a78c2f"))
        recorder.record("s", [Message(role="user", content="a", turn_index=0)])

        assert recorder.origin("s") == ClientOrigin(session_id="c7a78c2f")
        recorder.forget("s")
        assert recorder.origin("s") is None

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


class TestTheSessionTrace:
    TRACE = {"client_session": "cc-1", "before": [], "subagents_before": [],
             "turns": [{"index": 1, "conversation_id": "s_main", "position": 3, "requests": [],
                        "subagents": [{"agent": "a1", "conversation_id": "s_a1", "requests": []}]}]}

    def _client(self, transcripts):
        service, monitor = _engine()
        return _client(service, monitor, transcripts=transcripts,
                       read_trace=lambda client_session: self.TRACE if client_session == "cc-1" else None)

    def test_each_turn_and_subagent_carries_the_words_a_person_recognises_it_by(self):
        transcripts = TranscriptRecorder()
        transcripts.record("s_main", [Message(role="user", content="first", turn_index=0),
                                      Message(role="assistant", content="ok", turn_index=1),
                                      Message(role="user", content="Use two   subagents,\nplease.", turn_index=2)])
        transcripts.record("s_a1", [Message(role="user", content="Count the test files.", turn_index=0)])

        trace = self._client(transcripts).get("/api/trace/cc-1").json()

        [turn] = trace["turns"]
        assert turn["words"] == "Use two subagents, please."
        assert turn["subagents"][0]["words"] == "Count the test files."

    def test_a_turn_is_known_by_the_words_the_person_wrote_not_by_claude_codes(self):
        """A message typed after a local command opens with the caveat Claude Code writes."""
        transcripts = TranscriptRecorder()
        caveat = "<local-command-caveat>Caveat: generated by local commands.</local-command-caveat>"
        transcripts.record("s_main", [Message(role="user", content="first", turn_index=0),
                                      Message(role="assistant", content="ok", turn_index=1),
                                      turn_with_client_text(2, (True, caveat), (False, "Now run the tests."))])

        trace = self._client(transcripts).get("/api/trace/cc-1").json()

        assert trace["turns"][0]["words"] == "Now run the tests."

    def test_a_closed_conversation_shows_without_its_words(self):
        trace = self._client(TranscriptRecorder()).get("/api/trace/cc-1").json()
        assert trace["turns"][0]["words"] is None

    def test_an_unknown_client_session_is_a_404(self):
        response = self._client(TranscriptRecorder()).get("/api/trace/cc-gone")
        assert response.status_code == 404 and response.json()["error"]["code"] == "TRACE_NOT_FOUND"


def test_the_overview_says_while_llm_gc_is_yielding_to_a_rate_limit():
    service, monitor = _engine()
    client = _client(service, monitor)
    assert client.get("/api/overview").json()["pause"] is None

    service.pause_model_calls(60, "the upstream rate-limited a request from the client")

    pause = client.get("/api/overview").json()["pause"]
    assert pause["reason"] == "the upstream rate-limited a request from the client" and pause["until"]


class TestAPreparedCutIsShownWithWhatDecidesIt:
    @staticmethod
    def _staged(service: GcService, session_id: str, turns: int) -> None:
        """Record turns and prepare a cut, without switching it in."""
        async def scenario() -> None:
            for index in range(turns):
                text = f"the topic{index % 8} is version {index}. " + "We walked through throughput at length. " * 4
                turn = Message(role="user" if index % 2 == 0 else "assistant", content=text,
                               token_count=count_tokens(text), turn_index=index)
                service.gc_record(session_id, [turn], [f"{session_id}-{index}"])
            await service.gc_age(session_id, ignore_batch_minimum=True)

        asyncio.run(scenario())

    def test_the_verdict_is_for_the_cache_the_proxy_reads(self):
        service, monitor = _engine()
        self._staged(service, "s", 12)
        reading = CacheReading(CacheState.WARM, uses_cache=True, idle_s=40.0, lifetime_s=300.0)
        body = _client(service, monitor, read_cache=lambda session_id: reading).get("/api/sessions/s").json()

        assert body["cache"] == {"state": "warm", "uses_cache": True, "idle_s": 40.0, "lifetime_s": 300.0}
        staged = body["staged"]
        assert staged is not None and staged["switches"] == (staged["reason"] != "waiting_for_payback")
        assert staged["rewritten_tokens"] > 0 and staged["stale_bound_tokens"] > 0

    def test_without_a_proxy_reading_the_cache_the_verdict_is_for_a_cold_one(self):
        service, monitor = _engine()
        self._staged(service, "s", 12)
        body = _client(service, monitor).get("/api/sessions/s").json()

        assert body["cache"] is None
        assert body["staged"]["switches"] is True and body["staged"]["reason"] == "cache_cold"

