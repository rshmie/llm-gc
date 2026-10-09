"""`llm-gc start`, run for real with the servers and the model replaced by recorders."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from llm_gc.cli.main import build_cli
from llm_gc.cli.settings import ProxySettings
from llm_gc.exceptions import LLMAuthError
from llm_gc.llm import FakeProvider
from llm_gc.llm.factory import DEFAULT_MODELS


class Recorder:
    """Stands in for `serve_proxy` and the provider factory, and remembers what it was given."""

    def __init__(self, provider_script=None, busy_port=None):
        self.served: list[tuple[object, ProxySettings]] = []
        self.providers: list[FakeProvider] = []
        self.probed: list[list[int]] = []
        self._script = provider_script
        self._busy_port = busy_port

    def serve(self, service, settings):
        self.served.append((service, settings))

    def make_provider(self, settings):
        provider = FakeProvider(responses=list(self._script) if self._script is not None else None)
        self.providers.append(provider)
        return provider

    def find_busy(self, ports):
        self.probed.append(ports)
        return self._busy_port if self._busy_port in ports else None


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty directory and home, with no LLM-GC or credential variables to leak in."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("LLM_GC_CONFIG", raising=False)
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def _value(output: str, label: str) -> str:
    """The text of the banner row with this label, after the label."""
    for line in output.splitlines():
        if line.strip().startswith(f"{label} "):
            return line.strip()[len(label):].strip()
    raise AssertionError(f"no {label} row in:\n{output}")


def _run(recorder: Recorder, *args: str):
    cli = build_cli(serve=recorder.serve, make_provider=recorder.make_provider, find_busy=recorder.find_busy)
    return CliRunner().invoke(cli, ["start", *args])


def test_the_noop_compactor_starts_with_no_credential_and_calls_no_model():
    recorder = Recorder()
    result = _run(recorder, "--compactor", "noop")
    assert result.exit_code == 0, result.output
    assert len(recorder.served) == 1 and recorder.providers == []


def test_no_credential_refuses_to_start_and_names_the_ways_out():
    recorder = Recorder()
    result = _run(recorder)
    assert result.exit_code == 1
    assert "no credential" in result.output and "--compactor noop" in result.output
    assert "ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN" in result.output
    assert recorder.served == []


def test_no_credential_names_the_chosen_providers_variable_and_the_env_file_it_reads(isolated):
    result = _run(Recorder(), "--summary-provider", "openai")
    assert "OPENAI_API_KEY" in result.output and "ANTHROPIC" not in result.output
    assert str(isolated / ".env") in result.output.replace("\n", "")


def test_a_credential_starts_it_after_the_upstream_answers_the_check(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    recorder = Recorder()
    result = _run(recorder)
    assert result.exit_code == 0, result.output
    # One client asked the upstream for a token and was closed; a second serves the engine.
    assert len(recorder.providers) == 2
    assert len(recorder.providers[0].calls) == 1 and recorder.providers[1].calls == []


def test_a_credential_the_upstream_refuses_stops_startup_with_its_words(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    recorder = Recorder(provider_script=[LLMAuthError("invalid x-api-key")])
    result = _run(recorder)
    assert result.exit_code == 1
    assert "refused LLM-GC's credential" in result.output and "invalid x-api-key" in result.output
    assert recorder.served == []


def test_a_setting_in_the_file_is_used_when_its_flag_is_not_passed(isolated):
    """The layering bug, end to end: a flag's built-in default must not beat the file."""
    (isolated / "llm-gc.toml").write_text("port = 9100\n[engine]\nmin_cut_tokens = 2000\n")
    recorder = Recorder()
    assert _run(recorder, "--compactor", "noop").exit_code == 0
    settings = recorder.served[0][1]
    assert (settings.port, settings.engine.min_cut_tokens) == (9100, 2000)


def test_a_flag_beats_the_file(isolated):
    (isolated / "llm-gc.toml").write_text("port = 9100\n")
    recorder = Recorder()
    assert _run(recorder, "--compactor", "noop", "--port", "9200").exit_code == 0
    assert recorder.served[0][1].port == 9200


def test_the_banner_says_which_config_file_was_used(isolated):
    (isolated / "llm-gc.toml").write_text("")
    result = _run(Recorder(), "--compactor", "noop")
    assert "llm-gc.toml (working directory)" in result.output


def test_the_banner_says_when_no_file_was_used():
    assert _value(_run(Recorder(), "--compactor", "noop").output, "Config") == "none: built-in defaults"


def test_a_bad_setting_stops_startup_with_the_setting_named(isolated):
    (isolated / "llm-gc.toml").write_text("[engine]\nmin_cut_token = 2000\n")
    recorder = Recorder()
    result = _run(recorder, "--compactor", "noop")
    assert result.exit_code == 1 and "engine.min_cut_token" in result.output
    assert recorder.served == []


def test_a_missing_named_config_file_is_an_error(isolated):
    result = _run(Recorder(), "--compactor", "noop", "--config", str(isolated / "absent.toml"))
    assert result.exit_code == 1 and "config file not found" in result.output


def test_no_credential_value_reaches_the_output(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key-that-must-not-print")
    assert "key-that-must-not-print" not in _run(Recorder()).output


def test_a_passed_check_says_so_beside_the_upstream_and_model_it_used(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    output = _run(Recorder(), "--upstream", "https://gateway.example.com", "--summary-model", "gw-sonnet").output
    assert _value(output, "Upstream") == "https://gateway.example.com"
    assert _value(output, "Summaries") == "gw-sonnet"
    assert _value(output, "Check") == "✓ the upstream accepted LLM-GC's key and model"


def test_the_banner_comes_before_the_check_so_a_refusal_reads_in_context(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    recorder = Recorder(provider_script=[LLMAuthError("invalid x-api-key")])
    output = _run(recorder).output
    assert output.index("Upstream") < output.index("refused LLM-GC's credential")


def test_the_banner_names_the_default_model_rather_than_saying_default(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    output = _run(Recorder()).output
    assert _value(output, "Summaries") == f"{DEFAULT_MODELS['anthropic']} (the default)"


def test_a_busy_port_stops_startup_naming_the_port_and_the_flag_before_any_model_call(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    recorder = Recorder(busy_port=9900)
    result = _run(recorder)
    assert result.exit_code == 1
    assert "port 9900 is already in use" in result.output and "--port" in result.output
    assert recorder.providers == [] and recorder.served == []


def test_a_busy_dashboard_port_names_its_own_flag():
    result = _run(Recorder(busy_port=9901), "--compactor", "noop", "--dashboard")
    assert result.exit_code == 1 and "--dashboard-port" in result.output


def test_the_dashboard_port_is_probed_only_when_the_dashboard_runs():
    recorder = Recorder()
    _run(recorder, "--compactor", "noop")
    assert recorder.probed == [[9900]]


def test_the_dashboard_row_gives_its_address_or_says_how_to_turn_it_on():
    """Off is the default, so the banner must not leave a reader looking for an address that isn't there."""
    assert _value(_run(Recorder(), "--compactor", "noop", "--dashboard").output, "Dashboard") == "http://127.0.0.1:9901"
    assert "--dashboard" in _value(_run(Recorder(), "--compactor", "noop").output, "Dashboard")


def test_summaries_name_where_they_go_only_when_it_is_not_the_upstream(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    output = _run(Recorder(), "--summary-provider", "openai").output
    assert _value(output, "Summaries").endswith("(the default), at openai's own API")


def test_the_noop_compactor_says_summaries_are_off_and_runs_no_check():
    output = _run(Recorder(), "--compactor", "noop").output
    assert _value(output, "Summaries").startswith("off")
    assert "Check" not in output


def test_a_refusal_after_the_banner_is_set_apart_from_it(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    output = _run(Recorder(busy_port=9900)).output
    assert "\n\nError: port 9900 is already in use" in output
