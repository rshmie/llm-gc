"""`llm-gc config init` and `config path`, and the packaged starter file they rely on."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from llm_gc.cli.config_file import read_config_template, write_config_file
from llm_gc.cli.main import build_cli
from llm_gc.cli.settings import load_settings
from llm_gc.exceptions import ConfigError


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("LLM_GC_CONFIG", raising=False)
    return tmp_path


def _run(*args: str):
    return CliRunner().invoke(build_cli(), ["config", *args])


def test_the_packaged_template_loads_and_states_the_real_defaults(isolated):
    """Keeps the template from drifting: every value it shows must be what the code does."""
    path = isolated / "template.toml"
    path.write_text(read_config_template())
    assert load_settings(path).model_dump() == load_settings().model_dump()


def test_init_writes_a_starter_file_in_this_directory(isolated):
    result = _run("init")
    assert result.exit_code == 0, result.output
    assert (isolated / "llm-gc.toml").read_text() == read_config_template()
    assert "llm-gc.toml" in result.output
    assert str(isolated / ".env") in result.output  # where the credential goes instead
    assert "llm-gc start --dashboard" in result.output


def test_init_s_next_step_follows_whether_env_exists(isolated):
    assert "copy .env.example" in _run("init").output
    (isolated / ".env").write_text("ANTHROPIC_API_KEY=k\n")
    assert "llm-gc config show" in _run("init", "--force").output


def test_a_file_init_wrote_is_then_found_and_loadable(isolated):
    _run("init")
    assert "llm-gc.toml (working directory)" in _run("path").output
    load_settings(isolated / "llm-gc.toml")


def test_init_refuses_to_replace_a_file_and_leaves_it_untouched(isolated):
    (isolated / "llm-gc.toml").write_text("port = 1234\n")
    result = _run("init")
    assert result.exit_code == 1 and "--force" in result.output
    assert (isolated / "llm-gc.toml").read_text() == "port = 1234\n"


def test_force_replaces_the_file(isolated):
    (isolated / "llm-gc.toml").write_text("port = 1234\n")
    assert _run("init", "--force").exit_code == 0
    assert (isolated / "llm-gc.toml").read_text() == read_config_template()


def test_init_user_writes_the_per_user_file_and_creates_its_directory(isolated):
    assert _run("init", "--user").exit_code == 0
    assert (isolated / "home" / ".config" / "llm-gc" / "config.toml").read_text() == read_config_template()
    assert not (isolated / "llm-gc.toml").exists()


def test_path_says_when_there_is_no_file_and_where_it_looked_in_order():
    result = _run("path")
    assert result.exit_code == 0 and "No config file" in result.output and "config init" in result.output
    output = result.output
    assert output.index("--config PATH") < output.index("LLM_GC_CONFIG") < output.index("./llm-gc.toml") \
        < output.index(".config/llm-gc/config.toml")


def test_path_names_the_file_and_why(isolated):
    named = isolated / "named.toml"
    named.write_text("")
    result = _run("path", "--config", str(named))
    assert f"{named} (--config)" in result.output


def test_path_reports_a_missing_named_file_as_an_error(isolated):
    result = _run("path", "--config", str(isolated / "absent.toml"))
    assert result.exit_code == 1 and "not found" in result.output


def test_write_never_replaces_without_being_told(isolated):
    target = isolated / "x.toml"
    write_config_file(target, "a")
    with pytest.raises(ConfigError):
        write_config_file(target, "b")
    assert target.read_text() == "a"
