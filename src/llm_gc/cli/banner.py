"""The `llm-gc start` banner: what was loaded, what was checked, and where to connect. Application code."""

from llm_gc.cli.serve import BIND_HOST
from llm_gc.cli.settings import ConfigLocation, ProxySettings, summary_base_url, summary_model
from llm_gc.cli.style import INDENT, style, tidy_path
from llm_gc.cli.wordmark import render_wordmark
from llm_gc.proxy.server import DEFAULT_PROXY_PORT

# Wide enough for the longest label, "Dashboard", and the gap after it.
_LABEL_WIDTH = 11


def render_header(settings: ProxySettings, location: ConfigLocation | None, version: str,
                  columns: int | None = None) -> str:
    """What this run loaded: the file, where requests go, and what summarises them.

    `columns` is the terminal's width, or None when output is not a terminal; the wordmark is
    drawn only when it fits, and a plain title line stands in for it otherwise.
    """
    wordmark = render_wordmark(columns)
    if wordmark:
        tagline = style("  ·  garbage collection for LLM context", dim=True)
        title = [*wordmark, "", INDENT + style(f"v{version}", bold=True) + tagline]
    else:
        title = [INDENT + style(f"LLM-GC {version}", bold=True)]
    return "\n".join([
        "",
        *title,
        "",
        _row("Config", _describe_config(location)),
        _row("Upstream", settings.upstream),
        _row("Summaries", _describe_summaries(settings)),
    ])


def render_check(warning: str | None) -> str:
    """The startup check's outcome. A refusal never gets here: it stops startup as an error."""
    if warning:
        return _row("Check", style(f"! {warning}", fg="yellow"))
    return _row("Check", style("✓", fg="green") + " the upstream accepted LLM-GC's key and model")


def render_addresses(settings: ProxySettings) -> str:
    """Where to connect, and what to do next. A dashboard that is off says how to turn it on."""
    if settings.dashboard:
        dashboard = style(f"http://{BIND_HOST}:{settings.dashboard_port}", fg="cyan")
    else:
        dashboard = "off " + style("(start with --dashboard to see your conversations)", dim=True)
    return "\n".join([
        "",
        _row("Proxy", style(f"http://{BIND_HOST}:{settings.port}", fg="cyan")),
        _row("Dashboard", dashboard),
        _row("Stats", _stats_command(settings) + " " + style("(in another terminal)", dim=True)),
        "",
        INDENT + style("Leave this terminal open while you work. Ctrl-C stops LLM-GC.", dim=True),
        "",
    ])


def _stats_command(settings: ProxySettings) -> str:
    # `llm-gc stats` finds the port in the same settings, except one given only as a flag here.
    return "llm-gc stats" if settings.port == DEFAULT_PROXY_PORT else f"llm-gc stats --port {settings.port}"


def _describe_config(location: ConfigLocation | None) -> str:
    if location is None:
        return "none: " + style("built-in defaults", dim=True)
    return f"{tidy_path(location.path)} " + style(f"({location.origin})", dim=True)


def _describe_summaries(settings: ProxySettings) -> str:
    if settings.compactor == "noop":
        return "off " + style("(--compactor noop: exchanges are kept or archived, never condensed)", dim=True)
    text = summary_model(settings)
    if not settings.summary.model:
        text += " " + style("(the default)", dim=True)
    # Named only when it differs: usually the summaries go to the upstream, shown just above.
    where = summary_base_url(settings)
    if where != settings.upstream:
        text += f", at {where}" if where else f", at {settings.summary.provider}'s own API"
    return text


def _row(label: str, value: str) -> str:
    # Padded before styling: escape codes would count towards the width.
    return INDENT + style(label.ljust(_LABEL_WIDTH), bold=True) + value
