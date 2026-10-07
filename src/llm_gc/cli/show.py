"""Rendering the effective settings, each with the layer it came from."""

import json
from collections.abc import Mapping
from typing import Any

from llm_gc.cli.settings import ProxySettings, credential_fields, flatten_settings

DEFAULT_ORIGIN = "default"
_COMMENT_COLUMN = 36


def render_settings(settings: ProxySettings, origins: Mapping[str, str]) -> str:
    """The settings as TOML, a trailing comment on each value naming its layer.

    The output loads back as the same settings, so it can be redirected into a config file.
    A setting with no value is a comment, since TOML has no null, and a credential is only
    ever reported as set or not, never shown.
    """
    secrets = credential_fields()
    values = flatten_settings(settings.model_dump(exclude=set(secrets)))
    lines = ["# Effective settings, and the layer each value came from."]
    section = ""
    for name in sorted(values, key=lambda n: ("." in n, n.rsplit(".", 1)[0] if "." in n else "")):
        head, _, key = name.rpartition(".")
        if head != section:
            lines += ["", f"[{head}]"]
            section = head
        value = values[name]
        if value is None:
            lines.append(f"# {key}: not set")
            continue
        text = f"{key} = {_toml_value(value)}"
        # Two spaces at least, so a long value never runs into its comment.
        lines.append(f"{text:<{max(_COMMENT_COLUMN, len(text) + 2)}}# {origins.get(name, DEFAULT_ORIGIN)}")

    lines += ["", "# Credentials, never shown:"]
    for field in sorted(secrets):
        is_set = getattr(settings, field) is not None
        lines.append(f"#   {field}: " + (f"set, from {origins.get(field, '?')}" if is_set else "not set"))
    return "\n".join(lines)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):  # before int: bool is a subclass of int
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)  # a JSON string is a valid TOML basic string
    return repr(value)
