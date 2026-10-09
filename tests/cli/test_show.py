"""`llm-gc config show`: the effective settings, and the layer each one came from."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from llm_gc.cli.main import build_cli
from llm_gc.cli.settings import flatten_settings, load_settings, settings_origins
from llm_gc.cli.show import render_settings
from llm_gc.llm.factory import DEFAULT_MODELS


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for name in ("XDG_CONFIG_HOME", "LLM_GC_CONFIG", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def _show(path: Path | None = None, **flags) -> str:
    settings = load_settings(path, **flags)
    return render_settings(settings, settings_origins(path, **flags))


def _line(output: str, key: str) -> str:
    return next(line for line in output.splitlines() if line.startswith(f"{key} ") or line.startswith(f"# {key}:"))


def test_everything_is_a_default_when_no_layer_says_otherwise():
    output = _show()
    assert _line(output, "port").endswith("# default")
    assert _line(output, "min_cut_tokens").endswith("# default")


def test_each_layer_is_named_for_the_values_it_supplied(isolated, monkeypatch):
    file = isolated / "c.toml"
    file.write_text("log_level = \"WARNING\"\n[engine]\nmin_cut_tokens = 2000\n")
    (isolated / ".env").write_text("LLM_GC_DASHBOARD_PORT=9500\n")
    monkeypatch.setenv("LLM_GC_PORT", "9200")
    output = _show(file, compactor="noop")
    assert _line(output, "log_level").endswith("# file")
    assert _line(output, "min_cut_tokens").endswith("# file")
    assert _line(output, "dashboard_port").endswith("# .env")
    assert _line(output, "port").endswith("# environment")
    assert _line(output, "compactor").endswith("# flag")
    assert _line(output, "keep_threshold").endswith("# default")


def test_a_blank_credential_copied_from_the_example_shows_as_not_set(isolated):
    # `start` refuses without a credential, so `show` must not claim one is set.
    (isolated / ".env").write_text("ANTHROPIC_API_KEY=\nANTHROPIC_AUTH_TOKEN=tok\nOPENAI_API_KEY=\n")
    output = _show()
    assert "ANTHROPIC_API_KEY: not set" in output
    assert "ANTHROPIC_AUTH_TOKEN: set, from .env" in output
    assert "OPENAI_API_KEY: not set" in output


def test_the_highest_layer_is_the_one_named_when_several_supply_a_value(isolated, monkeypatch):
    file = isolated / "c.toml"
    file.write_text("port = 9100\n")
    (isolated / ".env").write_text("LLM_GC_PORT=9150\n")
    assert _line(_show(file), "port").endswith("# .env")
    monkeypatch.setenv("LLM_GC_PORT", "9200")
    assert _line(_show(file), "port").endswith("# environment")
    assert "port = 9300" in _show(file, port=9300) and _line(_show(file, port=9300), "port").endswith("# flag")


def test_a_credential_is_reported_as_set_with_its_layer_and_its_value_never_appears(isolated, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key-that-must-not-print")
    (isolated / ".env").write_text("OPENAI_API_KEY=other-secret-value\n")
    output = _show()
    assert "ANTHROPIC_API_KEY: set, from environment" in output
    assert "OPENAI_API_KEY: set, from .env" in output
    assert "ANTHROPIC_AUTH_TOKEN: not set" in output
    assert "key-that-must-not-print" not in output and "other-secret-value" not in output


def test_an_unset_value_is_a_comment_because_toml_has_no_null():
    assert "# model: not set" in _show()


def test_an_unset_summary_setting_says_what_is_used_instead():
    """"not set" alone reads as a fault; for these two, leaving them out is the normal setup."""
    output = _show(upstream="https://gateway.example.com")
    assert f"# model: not set: the provider's default, {DEFAULT_MODELS['anthropic']}" in output
    assert "# base_url: not set: follows upstream, https://gateway.example.com" in output
    assert "# base_url: not set: the provider's own address" in _show(summary={"provider": "openai"})


def test_the_output_loads_back_as_the_same_settings(isolated, monkeypatch):
    """So it can be redirected into a config file: every value shown is a value that loads."""
    monkeypatch.setenv("LLM_GC_ENGINE__MIN_CUT_TOKENS", "2500")
    original = load_settings()
    reloaded_file = isolated / "reloaded.toml"
    reloaded_file.write_text(_show())
    monkeypatch.delenv("LLM_GC_ENGINE__MIN_CUT_TOKENS")
    assert load_settings(reloaded_file).model_dump() == original.model_dump()


def test_a_long_value_does_not_run_into_its_comment():
    assert 'upstream = "https://api.anthropic.com"  # default' in _show()


def test_a_flag_is_a_layer_of_its_own():
    assert settings_origins(None, engine={"min_cut_tokens": 5}, port=None) == {"engine.min_cut_tokens": "flag"}


def test_flattening_gives_dotted_names():
    assert flatten_settings({"a": 1, "b": {"c": 2, "d": {"e": 3}}}) == {"a": 1, "b.c": 2, "b.d.e": 3}


def test_the_command_prints_what_start_would_use(isolated):
    (isolated / "llm-gc.toml").write_text("port = 9100\n")
    result = CliRunner().invoke(build_cli(), ["config", "show"])
    assert result.exit_code == 0, result.output
    assert _line(result.output, "port").endswith("# file")


def test_the_command_is_also_a_validity_check(isolated):
    (isolated / "llm-gc.toml").write_text("[engine]\nmin_cut_token = 2000\n")
    result = CliRunner().invoke(build_cli(), ["config", "show"])
    assert result.exit_code == 1 and "engine.min_cut_token" in result.output
