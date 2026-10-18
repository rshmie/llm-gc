"""Rendering the effective settings, each with the layer it came from.

The output is TOML, so it can be redirected into a config file: every line that is not a
setting is a comment, and the styling drops out when the output is not a terminal.
"""

import json
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from llm_gc.cli.settings import (
    ProxySettings,
    credential_fields,
    flatten_settings,
    summary_base_url,
    summary_model,
)
from llm_gc.cli.style import style

DEFAULT_ORIGIN = "default"
# Comments line up one column past the longest setting, up to here: a long gateway URL
# must not push every comment off to the right.
_MAX_COMMENT_COLUMN = 50


def render_settings(settings: ProxySettings, origins: Mapping[str, str], *, source: str | None = None) -> str:
    """The settings as TOML, a trailing comment on each value naming its layer.

    A value some layer changed from its default stands out; one that only repeats the
    default is dimmed, so what was actually configured is visible at a glance. A setting
    with no value is a comment, since TOML has no null, saying what applies instead. A
    credential is only ever reported as set or not, never shown.

    Args:
        settings: The settings in effect.
        origins: Each setting's layer, as `settings_origins` returns it.
        source: The config file, for the opening line; None for the built-in defaults.
    """
    secrets = credential_fields()
    values = flatten_settings(settings.model_dump(exclude=set(secrets)))
    defaults = _default_values()
    names = sorted(values, key=lambda n: ("." in n, n.rsplit(".", 1)[0] if "." in n else ""))
    longest = max(len(_assignment(name, value)) for name, value in values.items() if value is not None)
    column = min(longest, _MAX_COMMENT_COLUMN) + 2

    lines = [
        _comment(f"# LLM-GC settings in effect, from {source or 'the built-in defaults'}."),
        _comment("# The comment beside each value says where it came from."),
    ]
    section = ""
    for name in names:
        head, _, key = name.rpartition(".")
        if head != section:
            lines += ["", style(f"[{head}]", bold=True)]
            section = head
        value = values[name]
        if value is None:
            lines.append(_comment(f"# {key}: not set{_what_applies_instead(name, settings)}"))
            continue
        text = _assignment(name, value)
        padding = " " * max(column - len(text), 2)
        origin = origins.get(name, DEFAULT_ORIGIN)
        if origin != DEFAULT_ORIGIN and value != defaults.get(name):
            lines.append(text + padding + style(f"# {origin}", fg="cyan"))
        else:
            note = origin if origin == DEFAULT_ORIGIN else f"{origin}, the default"
            lines.append(text + padding + _comment(f"# {note}"))

    return "\n".join([*lines, "", *_render_credentials(settings, origins)])


def _render_credentials(settings: ProxySettings, origins: Mapping[str, str]) -> list[str]:
    """LLM-GC's own key: each variable the chosen provider reads, then whether `start` will find one."""
    needed = settings.summary_credential_names
    # Another vendor's variable is shown only when set: exported for some other tool, it is
    # worth knowing it is not the one in use.
    shown = [
        name
        for name in sorted(field.upper() for field in credential_fields())
        if name in needed or getattr(settings, name.lower()) is not None
    ]
    width = max(len(name) for name in shown) + 3
    lines = [_comment("# LLM-GC's own key, never shown:")]
    for name in shown:
        if getattr(settings, name.lower()) is None:
            status = _comment("not set")
        else:
            status = style("set", fg="green") + _comment(f", from {origins.get(name.lower(), '?')}")
        if name not in needed:
            status += _comment(f" (not used: provider is {settings.summary.provider})")
        lines.append(_comment(f"#   {name.ljust(width)}") + status)

    provider = settings.summary.provider
    if settings.compactor == "noop":
        lines.append(_comment('# compactor = "noop" makes no model calls, so no key is needed.'))
    elif settings.has_summary_credential:
        lines.append(style(f"# ✓ LLM-GC has its own key for {provider}.", fg="green"))
    else:
        lines.append(
            style(
                f"# ! No key for {provider}, so `llm-gc start` will stop. Put one in .env as {' or '.join(needed)}.",
                fg="yellow",
            )
        )
    return lines


def _what_applies_instead(name: str, settings: ProxySettings) -> str:
    """For a setting left unset, what is used in its place, so "not set" never reads as a fault."""
    if name == "summary.model":
        return f": the provider's default, {summary_model(settings)}"
    if name == "summary.base_url":
        url = summary_base_url(settings)
        return f": follows upstream, {url}" if url else ": the provider's own address"
    return ""


def _default_values() -> dict[str, Any]:
    """Every non-secret setting's built-in default, by dotted name."""
    defaults = {
        name: field.get_default(call_default_factory=True)
        for name, field in ProxySettings.model_fields.items()
        if name not in credential_fields()
    }
    return flatten_settings(
        {name: value.model_dump() if isinstance(value, BaseModel) else value for name, value in defaults.items()}
    )


def _assignment(name: str, value: Any) -> str:
    return f"{name.rpartition('.')[2]} = {_toml_value(value)}"


def _comment(text: str) -> str:
    return style(text, dim=True)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):  # before int: bool is a subclass of int
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)  # a JSON string is a valid TOML basic string
    return repr(value)
