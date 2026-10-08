"""Settings for the LLM-GC application: one validated object, layered from four sources.

Application code, not library code. The library takes an explicit `GCConfig` and never
reads the environment; this module is where the proxy's own choices (ports, upstream,
summary model) and the engine's tunables meet, so a bad value fails at startup with the
setting named instead of degrading a request later.

Precedence, lowest to highest:

    built-in defaults  <  TOML file  <  `.env` file  <  `LLM_GC_*` environment  <  flags

Credentials come from `.env` or the environment only. A credential in the TOML file is
refused, because that file is meant to be shared and committed.
"""

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from llm_gc.config import GCConfig
from llm_gc.config.constants import DEFAULT_DASHBOARD_PORT, DEFAULT_PROXY_CONTEXT_WINDOW
from llm_gc.exceptions import ConfigError
from llm_gc.llm.factory import PROVIDER_NAMES, create_provider
from llm_gc.llm.provider import LLMProvider
from llm_gc.proxy.server import ANTHROPIC_UPSTREAM, DEFAULT_PROXY_PORT

# Looked for in the working directory when nothing asks for a file; absent is fine.
DEFAULT_CONFIG_FILENAME = "llm-gc.toml"
# Names a config file in the process environment, for a service manager or a container that
# cannot pass a flag. Not read from `.env`: that file is a setting source, not a locator.
CONFIG_PATH_ENV = "LLM_GC_CONFIG"
USER_CONFIG_RELATIVE = Path("llm-gc") / "config.toml"
ENV_FILENAME = ".env"
ENV_PREFIX = "LLM_GC_"
# Joins a nested key in an environment variable: LLM_GC_ENGINE__MIN_CUT_TOKENS.
ENV_NESTED_DELIMITER = "__"

# LLM-GC's own credentials keep the names every Anthropic tool uses, not LLM_GC_*.
_CREDENTIAL_ENV_NAMES = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY")
# Which settings fields hold each provider's credentials, as (api key, bearer token). A new
# provider adds a row here and its field below. The one other place that names a vendor is
# `create_summary_provider`'s rule that only anthropic follows the proxy's upstream.
_CREDENTIAL_FIELDS: dict[str, tuple[str, str | None]] = {
    "anthropic": ("anthropic_api_key", "anthropic_auth_token"),
    "openai": ("openai_api_key", None),
}
# Hosts that reach this machine: an upstream on one of them, at our own port, is a loop.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class _ConfigFileSource(TomlConfigSettingsSource):
    """The TOML source, refusing a credential in the file with a message that says why."""

    def __init__(self, settings_cls: type[BaseSettings], toml_file: Path | None) -> None:
        super().__init__(settings_cls, toml_file=toml_file)
        leaked = [name for name in _CREDENTIAL_ENV_NAMES if name in {key.upper() for key in self.toml_data}]
        if leaked:
            raise ConfigError(
                f"{toml_file}: {', '.join(leaked)} must not be set in the config file, which is meant to be "
                f"shared. Put it in the environment or in {ENV_FILENAME} instead."
            )


class _KnownSettingsDotEnvSource(DotEnvSettingsSource):
    """The `.env` source, keeping only variables that name a setting.

    `.env` is shared with whatever else the user keeps there, so an unrelated variable
    must not fail startup the way an unknown key in the config file does.
    """

    def __call__(self) -> dict[str, Any]:
        known = set(self.settings_cls.model_fields) | set(_CREDENTIAL_ENV_NAMES)
        return {key: value for key, value in super().__call__().items() if key in known}


class EngineSettings(GCConfig):
    """The engine's tunables as the proxy defaults them: `GCConfig`, with the proxy's window.

    A subclass rather than a `default_factory` on the field, because a layer that sets
    only some engine keys makes pydantic build the section from just those keys: a
    factory is bypassed, and the library's own window would fill in the rest.
    """

    context_window: int = Field(default=DEFAULT_PROXY_CONTEXT_WINDOW, gt=0)


class SummarySettings(BaseModel):
    """The model LLM-GC itself calls to summarise spans and read their facts.

    Distinct from the conversation model: that one is the client's, chosen per request.
    Anything left out takes the vendor's own default.
    """

    model_config = ConfigDict(extra="forbid")

    provider: str = "anthropic"
    # Named as the vendor, or a gateway in front of it, names it.
    model: str | None = None
    # Where the summary calls go. For anthropic it defaults to the proxy's `upstream`,
    # because that speaks Anthropic's protocol; any other vendor is a different service.
    base_url: str | None = None

    @field_validator("provider")
    @classmethod
    def _provider_is_known(cls, value: str) -> str:
        if value not in PROVIDER_NAMES:
            raise ValueError(f"unknown provider {value!r}: choose one of {', '.join(PROVIDER_NAMES)}")
        return value


class ProxySettings(BaseSettings):
    """Everything `llm-gc start` needs, validated before anything is started.

    Build it with `load_settings`, not by calling the class: the loader is what binds the
    config file and turns pydantic's errors into `ConfigError`.
    """

    # An unknown key, in the file or an LLM_GC_ variable, fails instead of being ignored
    # while the user believes it took effect. An empty value (`ANTHROPIC_API_KEY=`, as
    # copied from .env.example) is not set: it neither overrides a lower layer nor counts
    # as a credential.
    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter=ENV_NESTED_DELIMITER,
        env_file=ENV_FILENAME,
        env_ignore_empty=True,
        extra="forbid",
    )

    port: int = Field(default=DEFAULT_PROXY_PORT, ge=1, le=65535)
    # The dashboard shows conversation content, so it is off unless asked for.
    dashboard: bool = False
    dashboard_port: int = Field(default=DEFAULT_DASHBOARD_PORT, ge=1, le=65535)
    # Where client requests are forwarded, and where LLM-GC's own summary calls go.
    upstream: str = ANTHROPIC_UPSTREAM
    # "noop" runs no summaries: exchanges are kept or archived but never condensed by a model.
    # It makes no model calls and needs no credential, and is the baseline to measure against.
    compactor: Literal["llm", "noop"] = "llm"
    summary: SummarySettings = Field(default_factory=SummarySettings)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    engine: EngineSettings = Field(default_factory=EngineSettings)

    # `SecretStr` keeps a value out of repr(), logs and error messages. The alias
    # replaces the LLM_GC_ prefix for these two, so they read as ANTHROPIC_API_KEY.
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias=AliasChoices(_CREDENTIAL_ENV_NAMES[0]))
    anthropic_auth_token: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices(_CREDENTIAL_ENV_NAMES[1])
    )
    openai_api_key: SecretStr | None = Field(default=None, validation_alias=AliasChoices(_CREDENTIAL_ENV_NAMES[2]))

    @field_validator("upstream")
    @classmethod
    def _upstream_is_an_http_url(cls, value: str) -> str:
        parts = urlsplit(value)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"must be an http(s) URL, not {value!r}")
        return value

    @model_validator(mode="after")
    def _upstream_is_not_this_process(self) -> "ProxySettings":
        parts = urlsplit(self.upstream)
        if parts.hostname in _LOOPBACK_HOSTS and parts.port in (self.port, self.dashboard_port):
            raise ValueError(f"upstream {self.upstream} is this process: every request would be forwarded back to it")
        return self

    @model_validator(mode="after")
    def _ports_differ(self) -> "ProxySettings":
        if self.dashboard and self.port == self.dashboard_port:
            raise ValueError(f"port and dashboard_port are both {self.port}")
        return self

    @property
    def has_summary_credential(self) -> bool:
        """Whether the chosen summary provider has a credential to authenticate with."""
        return any(getattr(self, field) for field in _CREDENTIAL_FIELDS[self.summary.provider] if field)

    @property
    def summary_credential_names(self) -> tuple[str, ...]:
        """The variables that can hold the chosen summary provider's credential, such as `ANTHROPIC_API_KEY`."""
        # A credential's variable is its field name in capitals: the alias set on each field.
        return tuple(field.upper() for field in _CREDENTIAL_FIELDS[self.summary.provider] if field)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Order the sources, highest precedence first. The file is added by `load_settings`'s subclass."""
        return (init_settings, env_settings, _KnownSettingsDotEnvSource(settings_cls))


@dataclass(frozen=True)
class ConfigLocation:
    """A config file and why it was chosen, so a run can say which file shaped it."""

    path: Path
    origin: str


def find_config_file(
    explicit: Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    home: Path | None = None,
) -> ConfigLocation | None:
    """The config file to load, or None when there is none and the defaults stand.

    Searched in order, and the first found wins; files are never merged, so which one
    applies is never a question of how several combine:

        1. `--config PATH`
        2. `LLM_GC_CONFIG`
        3. `llm-gc.toml` in the working directory
        4. `llm-gc/config.toml` under `XDG_CONFIG_HOME` (default `~/.config`)

    A file that was asked for, by 1 or 2, and is missing is an error. A missing 3 or 4
    is normal.

    The environment, directory and home are parameters so tests state them instead of
    patching the process.

    Raises:
        ConfigError: A file named by `--config` or `LLM_GC_CONFIG` does not exist.
    """
    environ = os.environ if environ is None else environ
    cwd = Path.cwd() if cwd is None else cwd
    home = Path.home() if home is None else home

    if explicit is not None:
        asked_for: tuple[Path, str] | None = (explicit, "--config")
    elif environ.get(CONFIG_PATH_ENV):
        asked_for = (Path(environ[CONFIG_PATH_ENV]), CONFIG_PATH_ENV)
    else:
        asked_for = None
    if asked_for is not None:
        path, origin = asked_for
        if not path.is_file():
            raise ConfigError(f"config file not found: {path} (from {origin})")
        return ConfigLocation(path, origin)

    for path, origin in ((cwd / DEFAULT_CONFIG_FILENAME, "working directory"),
                         (user_config_path(environ=environ, home=home), "user config")):
        if path.is_file():
            return ConfigLocation(path, origin)
    return None


def user_config_path(*, environ: Mapping[str, str] | None = None, home: Path | None = None) -> Path:
    """Where the per-user config file lives, whether or not it exists yet."""
    environ = os.environ if environ is None else environ
    home = Path.home() if home is None else home
    base = Path(environ["XDG_CONFIG_HOME"]) if environ.get("XDG_CONFIG_HOME") else home / ".config"
    return base / USER_CONFIG_RELATIVE


def load_settings(config_path: Path | None = None, **flags: Any) -> ProxySettings:
    """Layer the four sources into validated settings, or raise `ConfigError` saying what is wrong.

    Args:
        config_path: The TOML file to load, as `find_config_file` resolved it, or None
            for no file. This function does no searching of its own.
        **flags: Command-line values. A flag the user did not pass must be left out or
            `None`: a flag's built-in default would otherwise beat the file. Engine
            tunables nest under `engine={...}`.
    """
    if config_path is not None and not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")

    overrides = _drop_unset(flags)

    class _BoundSettings(ProxySettings):
        """`ProxySettings` with this call's file in the source list.

        A subclass rather than a module global, so concurrent loads (and tests) never
        share the path.
        """

        @classmethod
        def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings,
                                       file_secret_settings):  # type: ignore[no-untyped-def]
            file_source = _ConfigFileSource(settings_cls, config_path)
            return (init_settings, env_settings, _KnownSettingsDotEnvSource(settings_cls), file_source)

    try:
        return _BoundSettings(**overrides)
    except tomllib.TOMLDecodeError as err:
        raise ConfigError(f"{config_path}: not valid TOML: {err}") from err
    except ValidationError as err:
        raise ConfigError(_describe_problems(err, config_path)) from err


def settings_origins(config_path: Path | None, **flags: Any) -> dict[str, str]:
    """For each setting some layer supplies, the highest layer that does: `flag`, `environment`, `.env` or `file`.

    A setting no layer mentions is not in the result: it is the built-in default. Dotted
    names, so `engine.min_cut_tokens`; credentials are keyed by their field name.

    Asks the same sources `load_settings` uses what they contain, rather than inferring
    from the finished object, because once layers are merged the origin is gone.
    """
    layers = (  # highest precedence first
        ("flag", _drop_unset(flags)),
        ("environment", EnvSettingsSource(ProxySettings)()),
        (".env", _KnownSettingsDotEnvSource(ProxySettings)()),
        ("file", _ConfigFileSource(ProxySettings, config_path)()),
    )
    origins: dict[str, str] = {}
    for layer, supplied in layers:
        for name in flatten_settings(supplied):
            origins.setdefault(name.lower(), layer)  # lower: a credential's source key is its alias
    return origins


def flatten_settings(tree: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Nested settings as one level of dotted names: `{"engine": {"a": 1}}` becomes `{"engine.a": 1}`."""
    flat: dict[str, Any] = {}
    for key, value in tree.items():
        if isinstance(value, Mapping):
            flat.update(flatten_settings(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def credential_fields() -> frozenset[str]:
    """The settings fields that hold a credential, which a display must never print."""
    return frozenset(name for pair in _CREDENTIAL_FIELDS.values() for name in pair if name)


def _drop_unset(flags: dict[str, Any]) -> dict[str, Any]:
    """Remove `None` values, recursing into nested sections, so an unpassed flag overrides nothing."""
    kept: dict[str, Any] = {}
    for key, value in flags.items():
        if isinstance(value, dict):
            value = _drop_unset(value)
            if not value:
                continue
        if value is not None:
            kept[key] = value
    return kept


def _describe_problems(err: ValidationError, config_path: Path | None) -> str:
    """One line per bad setting, naming it by its dotted path. Never echoes a value that may be a secret."""
    lines = [f"invalid configuration{f' ({config_path})' if config_path else ''}:"]
    for problem in err.errors(include_input=False):
        name = ".".join(str(part) for part in problem["loc"]) or "settings"
        # pydantic prefixes a validator's own message with its mechanism, which says nothing to a user.
        lines.append(f"  {name}: {problem['msg'].removeprefix('Value error, ')}")
    return "\n".join(lines)


def summary_base_url(settings: ProxySettings) -> str | None:
    """Where the summary calls go: the setting, else the upstream for anthropic, else the vendor's own address (None)."""
    chosen = settings.summary
    # Only a provider speaking the proxy's own protocol can share its upstream.
    return chosen.base_url or (settings.upstream if chosen.provider == "anthropic" else None)


def create_summary_provider(settings: ProxySettings) -> LLMProvider:
    """The client for LLM-GC's own model calls, as the settings describe it.

    Raises:
        ConfigError: The chosen provider has no credential, or cannot take the one given.
    """
    chosen = settings.summary
    api_key_field, token_field = _CREDENTIAL_FIELDS[chosen.provider]
    # Each vendor gets only its own credentials: another vendor's, exported for some
    # other tool, must neither be sent to it nor stop it starting.
    return create_provider(
        chosen.provider,
        model=chosen.model,
        base_url=summary_base_url(settings),
        api_key=_reveal(getattr(settings, api_key_field)),
        auth_token=_reveal(getattr(settings, token_field)) if token_field else None,
    )


def _reveal(secret: SecretStr | None) -> str | None:
    return secret.get_secret_value() if secret is not None else None
