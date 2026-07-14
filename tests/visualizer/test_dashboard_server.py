from fastapi.testclient import TestClient

from llm_gc.visualizer.server import create_app


def _make_client() -> TestClient:
    return TestClient(create_app())


class TestDashboardEndpoints:
    def test_serves_dashboard_page(self):
        response = _make_client().get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Run GC now" in response.text

    def test_health_before_any_run_reports_not_measured(self):
        snapshot = _make_client().get("/health").json()
        assert snapshot["token_budget"] is None
        assert snapshot["gc_breakdown"] is None
        assert snapshot["transformation_ratio"] is None
        assert snapshot["generation_lifecycle"]["young_gen_count"] == 0
        assert snapshot["generation_lifecycle"]["recent_transitions"] == []

    def test_config_exposes_the_meter_inputs(self):
        config = _make_client().get("/config").json()
        assert config["context_window"] > 0
        assert 0.0 <= config["gc_threshold"] <= 1.0

    def test_run_gc_returns_summary_and_health_reflects_it(self):
        client = _make_client()
        run = client.post("/gc/run").json()
        assert run["status"] in {"completed", "bypassed_below_threshold", "bypassed_on_error"}
        assert run["turns_added"] > 0
        assert run["conversation_length"] > 0
        assert run["tokens_saved"] == run["tokens_before"] - run["tokens_in_final"]

        snapshot = client.get("/health").json()
        assert snapshot["gc_breakdown"] is not None
        assert snapshot["gc_breakdown"]["gc_run_id"] == run["gc_run_id"]

    def test_repeated_runs_accumulate_lifecycle_activity(self):
        client = _make_client()
        for _ in range(6):
            assert client.post("/gc/run").status_code == 200

        snapshot = client.get("/health").json()
        lifecycle = snapshot["generation_lifecycle"]
        # Six batches far exceed the demo window: GC must have completed at
        # least once, archived something, and extracted knowledge from it.
        assert snapshot["gc_breakdown"]["gc_status"] == "completed"
        assert lifecycle["permanent_gen_count"] > 0
        assert any(t["transition_type"] == "archived" for t in lifecycle["recent_transitions"])
        assert snapshot["token_budget"]["current_msg_turn_index"] >= 20
