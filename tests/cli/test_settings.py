"""Layering, validation and secret handling of the application's settings."""

from pathlib import Path

import pytest

from llm_gc.cli.settings import create_summary_provider, find_config_file, load_settings
from llm_gc.config.constants import DEFAULT_MIN_CUT_TOKENS, DEFAULT_PROXY_CONTEXT_WINDOW
from llm_gc.exceptions import ConfigError


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run in an empty directory with no LLM-GC or credential variables, so only the test's layers apply."""
    monkeypatch.chdir(tmp_path)
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def _write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text)
    return path


def test_defaults_apply_with_no_file_and_no_environment():
    settings = load_settings()
    assert settings.port == 9900
    assert settings.dashboard is False
    assert settings.engine.min_cut_tokens == DEFAULT_MIN_CUT_TOKENS
    assert settings.engine.context_window == DEFAULT_PROXY_CONTEXT_WINDOW


def test_the_file_overrides_defaults(isolated):
    path = _write(isolated, "c.toml", '[summary]\nmodel = "claude-haiku-4-5"\n[engine]\nmin_cut_tokens = 2000\n')
    settings = load_settings(path)
    assert settings.summary.model == "claude-haiku-4-5"
    assert settings.engine.min_cut_tokens == 2000


def test_the_environment_overrides_the_file(isolated, monkeypatch):
    path = _write(isolated, "c.toml", "port = 9000\n")
    monkeypatch.setenv("LLM_GC_PORT", "9100")
    assert load_settings(path).port == 9100


def test_a_nested_setting_can_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("LLM_GC_ENGINE__MIN_CUT_TOKENS", "3000")
    assert load_settings().engine.min_cut_tokens == 3000


def test_a_flag_overrides_the_environment_and_the_file(isolated, monkeypatch):
    path = _write(isolated, "c.toml", "port = 9000\n")
    monkeypatch.setenv("LLM_GC_PORT", "9100")
    assert load_settings(path, port=9200).port == 9200


def test_a_flag_that_was_not_passed_does_not_mask_the_file(isolated):
    path = _write(isolated, "c.toml", "port = 9000\n[engine]\nmin_cut_tokens = 2000\n")
    settings = load_settings(path, port=None, engine={"min_cut_tokens": None})
    assert settings.port == 9000
    assert settings.engine.min_cut_tokens == 2000


def test_nested_sections_merge_across_layers_instead_of_replacing(isolated):
    path = _write(isolated, "c.toml", "[engine]\nmin_cut_tokens = 2000\n")
    settings = load_settings(path, engine={"last_n_turns_to_keep": 2})
    assert settings.engine.min_cut_tokens == 2000
    assert settings.engine.last_n_turns_to_keep == 2


def test_an_explicit_missing_file_is_an_error(isolated):
    with pytest.raises(ConfigError, match="not found"):
        load_settings(isolated / "absent.toml")


def test_malformed_toml_names_the_file(isolated):
    path = _write(isolated, "c.toml", "port = \n")
    with pytest.raises(ConfigError, match="c.toml.*not valid TOML"):
        load_settings(path)


def test_an_unknown_key_is_refused_and_named(isolated):
    path = _write(isolated, "c.toml", "[engine]\nmin_cut_token = 2000\n")
    with pytest.raises(ConfigError, match="engine.min_cut_token"):
        load_settings(path)


@pytest.mark.parametrize(("text", "expected"), [
    # Appended at the end of the starter file, a line lands in its last section, [engine].
    ('[engine]\nmodel = "gw-sonnet"\n', "engine.model: unknown setting here: `model` goes under [summary]"),
    ('model = "gw-sonnet"\n', "model: unknown setting here: `model` goes under [summary]"),
    ("[summary]\nport = 9100\n", "summary.port: unknown setting here: `port` goes at the top, before any [section]"),
    ("[engine]\nmin_cut_token = 2000\n", "engine.min_cut_token: unknown setting: did you mean `min_cut_tokens`?"),
    ("[engine]\nflux_capacitor = 1\n", "engine.flux_capacitor: unknown setting"),
])
def test_an_unknown_key_says_where_it_belongs_or_what_it_resembles(isolated, text, expected):
    path = _write(isolated, "c.toml", text)
    with pytest.raises(ConfigError) as caught:
        load_settings(path)
    assert expected in str(caught.value)
    assert "Extra inputs" not in str(caught.value)


def test_every_bad_setting_is_reported_at_once(isolated):
    path = _write(isolated, "c.toml", "port = 0\n[engine]\nkeep_threshold = 2\n")
    with pytest.raises(ConfigError) as caught:
        load_settings(path)
    assert "port" in str(caught.value) and "engine.keep_threshold" in str(caught.value)


@pytest.mark.parametrize("upstream", ["gateway.example.com", "ftp://x.example.com"])
def test_an_upstream_that_is_not_an_http_url_is_refused(upstream):
    with pytest.raises(ConfigError, match="http"):
        load_settings(upstream=upstream)


def test_an_upstream_pointing_back_at_the_proxy_is_refused():
    with pytest.raises(ConfigError, match="forwarded back"):
        load_settings(upstream="http://localhost:9900")


def test_the_dashboard_cannot_share_the_proxy_port():
    with pytest.raises(ConfigError, match="both 9900"):
        load_settings(dashboard=True, dashboard_port=9900)


def test_credentials_come_from_the_environment_and_stay_out_of_the_repr(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok-secret-123")
    settings = load_settings()
    assert settings.anthropic_auth_token.get_secret_value() == "tok-secret-123"
    assert "tok-secret-123" not in repr(settings)


def test_credentials_can_come_from_dotenv_beside_unrelated_variables(isolated):
    _write(isolated, ".env", "ANTHROPIC_API_KEY=key-from-dotenv\nSOMETHING_ELSE=1\n")
    assert load_settings().anthropic_api_key.get_secret_value() == "key-from-dotenv"


def test_an_empty_credential_line_is_not_a_credential(isolated, monkeypatch):
    # .env.example ships `ANTHROPIC_API_KEY=` blank; a copy of it must not count as a key.
    _write(isolated, ".env", "ANTHROPIC_API_KEY=\nANTHROPIC_AUTH_TOKEN=\nOPENAI_API_KEY=\n")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "")
    settings = load_settings()
    assert settings.anthropic_api_key is None and settings.anthropic_auth_token is None
    assert not settings.has_summary_credential


def test_an_empty_value_does_not_override_a_lower_layer(isolated, monkeypatch):
    path = _write(isolated, "c.toml", 'upstream = "https://gateway.example.com"\n')
    _write(isolated, ".env", "LLM_GC_UPSTREAM=\n")
    monkeypatch.setenv("LLM_GC_PORT", "")
    settings = load_settings(path)
    assert settings.upstream == "https://gateway.example.com"
    assert settings.port == 9900


@pytest.mark.parametrize("line", ["ANTHROPIC_API_KEY = \"k\"", "anthropic_api_key = \"k\""])
def test_a_credential_in_the_config_file_is_refused_without_echoing_it(isolated, line):
    path = _write(isolated, "c.toml", line + "\n")
    with pytest.raises(ConfigError, match="must not be set in the config file") as caught:
        load_settings(path)
    assert '"k"' not in str(caught.value)


def test_an_unknown_summary_provider_is_refused_and_the_choices_named():
    with pytest.raises(ConfigError, match="summary.provider.*anthropic, openai"):
        load_settings(summary={"provider": "cohere"})


def test_the_summary_provider_is_chosen_by_settings(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    provider = create_summary_provider(load_settings(summary={"provider": "openai"}))
    assert provider.name == "openai"


def test_anthropic_summaries_follow_the_upstream_unless_given_their_own_url(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok")
    provider = create_summary_provider(load_settings(upstream="https://gateway.example.com"))
    assert provider._base_url == "https://gateway.example.com"
    own = create_summary_provider(load_settings(upstream="https://gateway.example.com",
                                                summary={"base_url": "https://other.example.com"}))
    assert own._base_url == "https://other.example.com"


def test_an_openai_summariser_does_not_inherit_the_anthropic_upstream_or_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok-for-claude-code")
    provider = create_summary_provider(load_settings(upstream="https://gateway.example.com",
                                                     summary={"provider": "openai"}))
    assert provider._base_url == "https://api.openai.com"


def test_a_provider_with_no_credential_fails_at_creation_not_on_first_use():
    with pytest.raises(ConfigError):
        create_summary_provider(load_settings(summary={"provider": "openai"}))


def test_has_summary_credential_follows_the_chosen_provider(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert load_settings().has_summary_credential
    assert not load_settings(summary={"provider": "openai"}).has_summary_credential



class TestFindConfigFile:
    """Where the file comes from. The environment, directory and home are passed in, never patched."""

    @pytest.fixture
    def places(self, tmp_path: Path) -> dict[str, Path]:
        cwd, home = tmp_path / "project", tmp_path / "home"
        (home / ".config" / "llm-gc").mkdir(parents=True)
        cwd.mkdir()
        return {"cwd": cwd, "home": home, "user": home / ".config" / "llm-gc" / "config.toml"}

    def _find(self, places, explicit=None, environ=None):
        return find_config_file(explicit, environ=environ or {}, cwd=places["cwd"], home=places["home"])

    def test_no_file_anywhere_means_the_defaults_stand(self, places):
        assert self._find(places) is None

    def test_the_working_directory_is_found_and_says_so(self, places):
        file = _write(places["cwd"], "llm-gc.toml", "")
        found = self._find(places)
        assert (found.path, found.origin) == (file, "working directory")

    def test_the_user_config_is_the_last_resort(self, places):
        user = _write(places["user"].parent, "config.toml", "")
        assert self._find(places).path == user

    def test_the_working_directory_beats_the_user_config_and_the_two_are_never_merged(self, places):
        _write(places["user"].parent, "config.toml", "port = 1\n")
        local = _write(places["cwd"], "llm-gc.toml", "")
        assert self._find(places).path == local

    def test_xdg_config_home_moves_the_user_config(self, places, tmp_path):
        xdg = tmp_path / "xdg"
        (xdg / "llm-gc").mkdir(parents=True)
        file = _write(xdg / "llm-gc", "config.toml", "")
        assert self._find(places, environ={"XDG_CONFIG_HOME": str(xdg)}).path == file

    def test_the_environment_variable_beats_the_working_directory(self, places, tmp_path):
        _write(places["cwd"], "llm-gc.toml", "")
        named = _write(tmp_path, "elsewhere.toml", "")
        found = self._find(places, environ={"LLM_GC_CONFIG": str(named)})
        assert (found.path, found.origin) == (named, "LLM_GC_CONFIG")

    def test_the_flag_beats_the_environment_variable(self, places, tmp_path):
        flagged = _write(tmp_path, "flag.toml", "")
        named = _write(tmp_path, "env.toml", "")
        found = self._find(places, explicit=flagged, environ={"LLM_GC_CONFIG": str(named)})
        assert (found.path, found.origin) == (flagged, "--config")

    def test_a_named_file_that_is_missing_is_an_error_not_a_fallback(self, places, tmp_path):
        _write(places["cwd"], "llm-gc.toml", "")
        with pytest.raises(ConfigError, match="LLM_GC_CONFIG"):
            self._find(places, environ={"LLM_GC_CONFIG": str(tmp_path / "absent.toml")})
        with pytest.raises(ConfigError, match="--config"):
            self._find(places, explicit=tmp_path / "absent.toml")


@pytest.mark.parametrize("how", ["file", "environment", "flag"])
def test_the_proxy_window_default_survives_setting_any_other_engine_key(isolated, monkeypatch, how):
    """A partial [engine] must not fall back to the library's own window: only the keys given are overridden."""
    if how == "file":
        settings = load_settings(_write(isolated, "c.toml", "[engine]\nmin_cut_tokens = 2000\n"))
    elif how == "environment":
        monkeypatch.setenv("LLM_GC_ENGINE__MIN_CUT_TOKENS", "2000")
        settings = load_settings()
    else:
        settings = load_settings(engine={"min_cut_tokens": 2000})
    assert settings.engine.min_cut_tokens == 2000
    assert settings.engine.context_window == DEFAULT_PROXY_CONTEXT_WINDOW
