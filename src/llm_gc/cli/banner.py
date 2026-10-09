"""The `llm-gc start` banner: what was loaded, what was checked, and where to connect.

Application code. Styled with click, which drops the styling when the output is not a
terminal; `NO_COLOR` (https://no-color.org) turns it off on a terminal too.
"""

import os
from pathlib import Path
from typing import IO, Any

import click

from llm_gc.cli.serve import BIND_HOST
from llm_gc.cli.settings import ConfigLocation, ProxySettings, summary_base_url, summary_model
from llm_gc.proxy.server import DEFAULT_PROXY_PORT

# Wide enough for the longest label, "Dashboard", and the gap after it.
_LABEL_WIDTH = 11
_INDENT = "  "


def render_header(settings: ProxySettings, location: ConfigLocation | None, version: str) -> str:
    """What this run loaded: the file, where requests go, and what summarises them."""
    return "\n".join([
        "",
        _INDENT + _style(f"LLM-GC {version}", bold=True),
        "",
        _row("Config", _describe_config(location)),
        _row("Upstream", settings.upstream),
        _row("Summaries", _describe_summaries(settings)),
    ])


def render_check(warning: str | None) -> str:
    """The startup check's outcome. A refusal never gets here: it stops startup as an error."""
    if warning:
        return _row("Check", _style(f"! {warning}", fg="yellow"))
    return _row("Check", _style("✓", fg="green") + " the upstream accepted LLM-GC's key and model")


def render_addresses(settings: ProxySettings) -> str:
    """Where to connect, and what to do next. A dashboard that is off says how to turn it on."""
    if settings.dashboard:
        dashboard = _style(f"http://{BIND_HOST}:{settings.dashboard_port}", fg="cyan")
    else:
        dashboard = "off " + _style("(start with --dashboard to see your conversations)", dim=True)
    return "\n".join([
        "",
        _row("Proxy", _style(f"http://{BIND_HOST}:{settings.port}", fg="cyan")),
        _row("Dashboard", dashboard),
        _row("Stats", _stats_command(settings) + " " + _style("(in another terminal)", dim=True)),
        "",
        _INDENT + _style("Leave this terminal open while you work. Ctrl-C stops LLM-GC.", dim=True),
        "",
    ])


class StartupRefused(click.ClickException):
    """A reason not to start, found after the banner: set apart from it, its label in red."""

    def show(self, file: IO[Any] | None = None) -> None:
        click.echo(f"\n{_style('Error:', fg='red', bold=True)} {self.format_message()}", file=file, err=file is None)


def _stats_command(settings: ProxySettings) -> str:
    # `llm-gc stats` finds the port in the same settings, except one given only as a flag here.
    return "llm-gc stats" if settings.port == DEFAULT_PROXY_PORT else f"llm-gc stats --port {settings.port}"


def _describe_config(location: ConfigLocation | None) -> str:
    if location is None:
        return "none: " + _style("built-in defaults", dim=True)
    return f"{_tidy_path(location.path)} " + _style(f"({location.origin})", dim=True)


def _describe_summaries(settings: ProxySettings) -> str:
    if settings.compactor == "noop":
        return "off " + _style("(--compactor noop: exchanges are kept or archived, never condensed)", dim=True)
    text = summary_model(settings)
    if not settings.summary.model:
        text += " " + _style("(the default)", dim=True)
    # Named only when it differs: usually the summaries go to the upstream, shown just above.
    where = summary_base_url(settings)
    if where != settings.upstream:
        text += f", at {where}" if where else f", at {settings.summary.provider}'s own API"
    return text


def _tidy_path(path: Path) -> str:
    """The path with the home directory shown as `~`, as a shell would."""
    home = Path.home()
    return f"~/{path.relative_to(home)}" if path.is_relative_to(home) else str(path)


def _row(label: str, value: str) -> str:
    # Padded before styling: escape codes would count towards the width.
    return _INDENT + _style(label.ljust(_LABEL_WIDTH), bold=True) + value


def _style(text: str, **styles: Any) -> str:
    return text if os.environ.get("NO_COLOR") else click.style(text, **styles)
