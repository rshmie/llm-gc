"""The `llm-gc start` banner's rows, rendered without a command run."""

from pathlib import Path

import pytest

from llm_gc.cli.banner import render_addresses, render_check, render_header
from llm_gc.cli.settings import ConfigLocation, ProxySettings


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty directory, so no `.env` or LLM_GC_* variable shapes the settings, and colour allowed."""
    monkeypatch.chdir(tmp_path)
    for name in ("NO_COLOR", "LLM_GC_DASHBOARD", "LLM_GC_PORT", "LLM_GC_DASHBOARD_PORT"):
        monkeypatch.delenv(name, raising=False)


def test_a_check_that_could_not_run_is_a_warning_not_a_tick():
    assert "! the upstream could not be reached" in render_check("the upstream could not be reached")
    assert "✓" in render_check(None)


def test_no_color_drops_every_escape_code(monkeypatch):
    settings = ProxySettings(dashboard=True)
    assert "\x1b[" in render_addresses(settings)
    monkeypatch.setenv("NO_COLOR", "1")
    rendered = render_header(settings, None, "0.1.0") + render_check(None) + render_addresses(settings)
    assert "\x1b[" not in rendered


def test_a_config_file_under_home_is_shown_from_tilde(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    location = ConfigLocation(tmp_path / "work" / "llm-gc.toml", "working directory")
    assert "~/work/llm-gc.toml" in render_header(ProxySettings(), location, "0.1.0")


def test_a_config_file_elsewhere_keeps_its_full_path(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    path = Path("/etc/llm-gc/config.toml")
    assert str(path) in render_header(ProxySettings(), ConfigLocation(path, "--config"), "0.1.0")


def test_the_stats_command_names_a_port_that_is_not_the_default(monkeypatch):
    """A port given only as a flag to `start` is one `llm-gc stats` cannot find by itself."""
    monkeypatch.setenv("NO_COLOR", "1")
    assert "llm-gc stats (in another terminal)" in render_addresses(ProxySettings())
    assert "llm-gc stats --port 9950" in render_addresses(ProxySettings(port=9950))
