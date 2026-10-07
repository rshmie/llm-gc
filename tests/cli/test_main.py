"""`llm-gc start`, run for real with the servers and the model replaced by recorders."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from llm_gc.cli.main import build_cli
from llm_gc.cli.settings import ProxySettings
from llm_gc.exceptions import LLMAuthError
from llm_gc.llm import FakeProvider


class Recorder:
    """Stands in for `serve_proxy` and the provider factory, and remembers what it was given."""

    def __init__(self, provider_script=None):
        self.served: list[tuple[object, ProxySettings]] = []
        self.providers: list[FakeProvider] = []
        self._script = provider_script

    def serve(self, service, settings):
        self.served.append((service, settings))

    def make_provider(self, settings):
        provider = FakeProvider(responses=list(self._script) if self._script is not None else None)
        self.providers.append(provider)
        return provider


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


def _run(recorder: Recorder, *args: str):
    return CliRunner().invoke(build_cli(serve=recorder.serve, make_provider=recorder.make_provider), ["start", *args])


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
    assert recorder.served == []


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
    assert "none, built-in defaults" in _run(Recorder(), "--compactor", "noop").output


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


def test_a_passed_check_says_so_and_where(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    result = _run(Recorder(), "--upstream", "https://gateway.example.com", "--summary-model", "gw-sonnet")
    assert "check:     gw-sonnet answered at https://gateway.example.com" in result.output


def test_the_banner_comes_before_the_check_so_a_refusal_reads_in_context(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    recorder = Recorder(provider_script=[LLMAuthError("invalid x-api-key")])
    output = _run(recorder).output
    assert output.index("upstream:") < output.index("refused LLM-GC's credential")
