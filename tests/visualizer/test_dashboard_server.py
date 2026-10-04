"""The dashboard over live sessions: read-only, per session, and honest about absence."""

import asyncio

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from llm_gc.config import GCConfig
from llm_gc.engine.compaction import LLMCompactor
from llm_gc.engine.gc_service import GcService
from llm_gc.engine.generations import GenerationalMemory
from llm_gc.engine.sweep import Sweeper, ThresholdSweepStrategy
from llm_gc.events import EventBus
from llm_gc.extraction import KnowledgeExtractor
from llm_gc.llm import FakeProvider, LLMResponse, StopReason
from llm_gc.models import Message
from llm_gc.monitoring import ContextHealthMonitor
from llm_gc.scoring import RecencyScorer, RelevanceScorer, RelevanceType
from llm_gc.session import SessionManager
from llm_gc.utils import count_tokens
from llm_gc.visualizer import create_dashboard_app

SUMMARY = LLMResponse(text="Earlier turns covered the storage decision and the retention window in brief.",
                      stop_reason=StopReason.COMPLETE, input_tokens=0, output_tokens=14,
                      model="scripted", provider="offline")


def _engine() -> tuple[GcService, ContextHealthMonitor]:
    bus = EventBus()
    config = GCConfig(context_window=700, keep_threshold=0.55, archive_threshold=0.25, min_compactable_tokens=10,
                      last_n_turns_to_keep=3, min_cut_tokens=60)
    service = GcService(
        session_manager=SessionManager(),
        generational_memory=GenerationalMemory(
            knowledge_extractor=KnowledgeExtractor(event_bus=bus),
            compactor=LLMCompactor(event_bus=bus, provider=FakeProvider(responses=[SUMMARY] * 400))),
        relevance_scorer=RelevanceScorer(scorers=[RecencyScorer(decay_rate=0.15)],
                                         weights={RelevanceType.RECENCY: 1.0}, event_bus=bus),
        sweeper=Sweeper(sweeper_strategy=ThresholdSweepStrategy(gc_config=config), gc_config=config, event_bus=bus),
        event_bus=bus, gc_config=config)
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
    return TestClient(create_dashboard_app(service, monitor, **kwargs))


class TestThePage:
    def test_serves_the_dashboard(self):
        response = _client(*_engine()).get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_every_route_is_read_only(self):
        """It shows what the engine did; it never asks the engine to do anything."""
        app = create_dashboard_app(*_engine())
        methods = {method for route in app.routes if isinstance(route, APIRoute) for method in route.methods}
        assert methods == {"GET"}


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
