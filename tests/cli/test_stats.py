"""`llm-gc stats`: asking a running proxy for its counters, and not overstating them."""

import json
import re
from pathlib import Path

import httpx2
import pytest
from click.testing import CliRunner

from llm_gc.cli.main import build_cli
from llm_gc.cli.stats import ProxyStats, fetch_proxy_stats, render_stats
from llm_gc.exceptions import ProxyUnavailableError
from llm_gc.proxy import ProxyMetrics

URL = "http://127.0.0.1:9900/stats"


def _counters(**overrides) -> dict:
    base = {"requests": 12, "gc_applied": 11, "gc_bypassed": 1, "tokens_in_client_history": 21_666,
            "tokens_forwarded": 4_310, "tokens_saved": 17_356, "history_rewinds": 0, "history_restarts": 1,
            "sessions_seeded": 0, "ratio": 0.199}
    return {**base, **overrides}


def _transport(handler) -> httpx2.MockTransport:
    return httpx2.MockTransport(handler)


def test_the_proxys_real_counters_validate_as_proxy_stats():
    """Pins the contract between what the proxy serves and what the CLI reads: a renamed counter fails here."""
    assert ProxyStats.model_validate(ProxyMetrics().as_dict()).requests == 0


@pytest.fixture
def no_color(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")


def _report(**overrides) -> str:
    return render_stats(ProxyStats(**_counters(**overrides)), "http://127.0.0.1:9900")


def test_a_proxy_that_has_seen_nothing_reports_that_instead_of_zeros():
    output = render_stats(ProxyStats.model_validate(ProxyMetrics().as_dict()), "http://127.0.0.1:9900")
    assert "No requests yet" in output and "tokens" not in output.lower()


def test_the_report_gives_each_counter_and_what_it_does_not_claim(no_color):
    output = _report()
    assert "21,666  the client sent" in output
    assert "4,310  LLM-GC forwarded (20%)" in output and "17,356  saved (80%)" in output
    assert "passed on as sent (unreadable, or an error in LLM-GC)" in output
    assert "not what the provider bills" in output


def test_no_saving_yet_says_so_rather_than_showing_a_bare_zero(no_color):
    assert "0  saved (nothing condensed yet)" in _report(tokens_forwarded=21_666, tokens_saved=0, ratio=1.0)


def test_rewinds_and_restarts_show_only_once_they_have_happened(no_color):
    """Only a client sending the session header can cause them, so zeros would be noise for Claude Code."""
    quiet = _report(history_rewinds=0, history_restarts=0)
    assert "rewound" not in quiet and "started over" not in quiet
    busy = _report(history_rewinds=2, history_restarts=1)
    assert "2  rewound" in busy and "1  started over" in busy


def test_every_count_ends_in_the_same_column(no_color):
    """Right-aligned, so magnitudes compare at a glance: 1,843,210 against 702,115."""
    output = _report(tokens_in_client_history=1_843_210, tokens_forwarded=702_115, tokens_saved=1_141_095,
                     requests=9, history_rewinds=3)
    ends = {match.end(1) for line in output.splitlines() if (match := re.search(r"(\d[\d,]*)  \D", line))}
    assert len(ends) == 1


def test_a_valid_answer_is_parsed():
    stats = fetch_proxy_stats(URL, transport=_transport(lambda request: httpx2.Response(200, json=_counters())))
    assert stats.requests == 12 and stats.ratio == 0.199


def test_extra_counters_a_newer_proxy_adds_are_ignored():
    stats = fetch_proxy_stats(URL, transport=_transport(
        lambda request: httpx2.Response(200, json=_counters(a_new_counter=5))))
    assert stats.requests == 12


def test_nothing_listening_says_how_to_start_it():
    def refuse(request):
        raise httpx2.ConnectError("refused")

    with pytest.raises(ProxyUnavailableError, match="llm-gc start"):
        fetch_proxy_stats(URL, transport=_transport(refuse))


def test_an_error_status_is_reported_with_the_status():
    with pytest.raises(ProxyUnavailableError, match="503"):
        fetch_proxy_stats(URL, transport=_transport(lambda request: httpx2.Response(503)))


@pytest.mark.parametrize("body", [b"<html>not json</html>", b'{"unrelated": true}', b"[1, 2]"])
def test_something_else_on_the_port_is_not_mistaken_for_a_proxy(body):
    with pytest.raises(ProxyUnavailableError, match="not like an LLM-GC proxy"):
        fetch_proxy_stats(URL, transport=_transport(lambda request: httpx2.Response(200, content=body)))


class TestCommand:
    @pytest.fixture(autouse=True)
    def isolated(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        for name in ("XDG_CONFIG_HOME", "LLM_GC_CONFIG", "LLM_GC_PORT"):
            monkeypatch.delenv(name, raising=False)
        return tmp_path

    def _run(self, *args: str, fetch=None):
        asked: list[str] = []

        def fake(url: str) -> ProxyStats:
            asked.append(url)
            if fetch is not None:
                return fetch(url)
            return ProxyStats(**_counters())

        result = CliRunner().invoke(build_cli(fetch_stats=fake), ["stats", *args])
        return result, asked

    def test_it_asks_the_default_port_on_localhost_and_prints_the_report(self):
        result, asked = self._run()
        assert result.exit_code == 0, result.output
        assert asked == [URL] and "Requests" in result.output

    def test_the_port_follows_the_config_file(self, isolated):
        (isolated / "llm-gc.toml").write_text("port = 9100\n")
        assert self._run()[1] == ["http://127.0.0.1:9100/stats"]

    def test_a_flag_beats_the_file(self, isolated):
        (isolated / "llm-gc.toml").write_text("port = 9100\n")
        assert self._run("--port", "9200")[1] == ["http://127.0.0.1:9200/stats"]

    def test_json_prints_the_raw_counters(self):
        result, _ = self._run("--json")
        assert json.loads(result.output)["tokens_forwarded"] == 4_310

    def test_an_unreachable_proxy_is_one_clean_line_and_a_failing_exit(self):
        def down(url: str):
            raise ProxyUnavailableError(f"nothing is answering at {url}: is `llm-gc start` running?")

        result, _ = self._run(fetch=down)
        assert result.exit_code == 1 and result.output.strip().startswith("Error: nothing is answering")
