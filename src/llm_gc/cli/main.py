"""The `llm-gc` command line. Application code: the library never imports it."""

import asyncio
import json
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import click

from llm_gc import __version__
from llm_gc.cli.banner import render_addresses, render_check, render_header
from llm_gc.cli.config_commands import add_config_commands
from llm_gc.cli.logs import configure_logging
from llm_gc.cli.options import config_file_option
from llm_gc.cli.serve import find_busy_port, serve_proxy
from llm_gc.cli.settings import ENV_FILENAME, ProxySettings, create_summary_provider, find_config_file, load_settings
from llm_gc.engine import GcService, build_gc_service
from llm_gc.cli.stats import ProxyStats, fetch_proxy_stats, render_stats
from llm_gc.cli.styled_commands import CommandError, StyledGroup
from llm_gc.exceptions import ConfigError, ProxyUnavailableError
from llm_gc.llm.provider import LLMProvider
from llm_gc.proxy.startup_checks import check_summariser

ServeFn = Callable[[GcService, ProxySettings], None]
ProviderFactory = Callable[[ProxySettings], LLMProvider]
StatsFetcher = Callable[[str], ProxyStats]
PortProbe = Callable[[list[int]], int | None]

_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def build_cli(*, serve: ServeFn = serve_proxy, make_provider: ProviderFactory = create_summary_provider,
              fetch_stats: StatsFetcher = fetch_proxy_stats, find_busy: PortProbe = find_busy_port) -> click.Group:
    """The command group, with its side effects passed in.

    Binding ports, calling a model and calling a running proxy are what a test must not
    do, so they arrive as arguments: `main` uses the real ones, and a test hands in fakes.
    """

    @click.group(cls=StyledGroup)
    @click.version_option(__version__, prog_name="llm-gc", message="%(prog)s %(version)s")
    def cli() -> None:
        """LLM-GC: context garbage collection for LLM conversations, as a proxy.

        \b
        First run, in the LLM-GC folder:
          llm-gc config init         write llm-gc.toml, LLM-GC's settings
          cp .env.example .env       then put LLM-GC's own key in .env
          llm-gc start --dashboard   start the proxy and the dashboard
        """

    @cli.command()
    @config_file_option
    @click.option("--port", type=int, help="Where the proxy listens.")
    @click.option("--dashboard/--no-dashboard", default=None, help="Serve the live dashboard (shows conversations).")
    @click.option("--dashboard-port", type=int, help="Where the dashboard listens.")
    @click.option("--upstream", help="Where requests are forwarded: Anthropic, or a gateway in front of it.")
    @click.option("--compactor", type=click.Choice(["llm", "noop"]), help="noop makes no model calls.")
    @click.option("--summary-provider", help="The vendor LLM-GC summarises with: anthropic or openai.")
    @click.option("--summary-model", help="The model LLM-GC summarises with, as the upstream names it.")
    @click.option("--min-cut-tokens", type=int, help="Tokens a collection must condense.")
    @click.option("--keep-last", type=int, help="Newest turns always sent word for word.")
    @click.option("--context-window", type=int, help="The proxied model's window; sets the recap budget.")
    @click.option("--log-level", type=click.Choice(_LOG_LEVELS, case_sensitive=False),
                  help="How much LLM-GC logs; info prints a line per request.")
    @click.option("--log-format", type=click.Choice(["text", "json"]),
                  help="text for reading in the terminal; json for a program that reads the log.")
    def start(config_path: Path | None, port: int | None, dashboard: bool | None, dashboard_port: int | None,
              upstream: str | None, compactor: str | None, summary_provider: str | None,
              summary_model: str | None, min_cut_tokens: int | None, keep_last: int | None,
              context_window: int | None, log_level: str | None, log_format: str | None) -> None:
        """Start the proxy, and the dashboard with --dashboard.

        Every option is also a setting: a flag you leave out takes its value from the
        environment, then .env, then the config file, then the built-in default.
        """
        try:
            location = find_config_file(config_path)
            settings = load_settings(
                location.path if location else None,
                port=port, dashboard=dashboard, dashboard_port=dashboard_port, upstream=upstream,
                compactor=compactor, log_level=log_level.upper() if log_level else None, log_format=log_format,
                summary={"provider": summary_provider, "model": summary_model},
                engine={"min_cut_tokens": min_cut_tokens, "last_n_turns_to_keep": keep_last,
                        "context_window": context_window},
            )
        except ConfigError as err:
            raise CommandError(str(err)) from err

        # The application chooses how the library's logs are shown. They name sessions and
        # counts, never conversation content.
        configure_logging(settings.log_level, settings.log_format)

        # Said before the check, so a refusal reads against what was being tried.
        columns = shutil.get_terminal_size().columns if sys.stdout.isatty() else None
        click.echo(render_header(settings, location, __version__, columns))
        # Before the check, which spends a model call: a taken port needs no upstream to report.
        _refuse_busy_ports(settings, find_busy)
        try:
            engine_provider = _prepare_summariser(settings, make_provider)
        except ConfigError as err:
            raise CommandError(str(err)) from err
        click.echo(render_addresses(settings))
        serve(build_gc_service(settings.engine, summary_provider=engine_provider), settings)

    @cli.command()
    @config_file_option
    @click.option("--port", type=int, help="The proxy's port (default: from the settings).")
    @click.option("--json", "as_json", is_flag=True, help="Print the raw counters as JSON.")
    def stats(config_path: Path | None, port: int | None, as_json: bool) -> None:
        """Show a running proxy's requests and token savings.

        Asks the proxy on the port the settings name; pass --port for one started with --port.
        """
        try:
            location = find_config_file(config_path)
            settings = load_settings(location.path if location else None, port=port)
            base_url = f"http://127.0.0.1:{settings.port}"
            counters = fetch_stats(f"{base_url}/stats")
        except (ConfigError, ProxyUnavailableError) as err:
            raise CommandError(str(err)) from err
        click.echo(json.dumps(counters.model_dump(mode="json"), indent=2) if as_json else render_stats(counters, base_url))

    add_config_commands(cli)
    return cli


def _refuse_busy_ports(settings: ProxySettings, find_busy: PortProbe) -> None:
    ports = {settings.port: "--port"}
    if settings.dashboard:
        ports[settings.dashboard_port] = "--dashboard-port"
    busy = find_busy(list(ports))
    if busy is not None:
        raise CommandError(
            f"port {busy} is already in use, perhaps by another llm-gc. Stop that one (Ctrl-C in its terminal), "
            f"or choose another port with {ports[busy]}."
        )


def _prepare_summariser(settings: ProxySettings, make_provider: ProviderFactory) -> LLMProvider | None:
    """The engine's summary client, after asking the upstream to accept it; None for the noop compactor.

    Raises:
        ConfigError: There is no credential, or the upstream refuses the model or credential.
    """
    if settings.compactor == "noop":
        return None
    if not settings.has_summary_credential:
        names = " or ".join(settings.summary_credential_names)
        raise ConfigError(
            f"no credential for LLM-GC's own summary calls. Set {names} in {Path.cwd() / ENV_FILENAME}, "
            "or export it. LLM-GC never reuses your client's credential. "
            "Or pass --compactor noop to run without summaries."
        )
    _check_summariser(make_provider(settings))
    # A second client for the engine: the check's was closed on its own event loop, and the
    # engine's first use is on the servers'.
    return make_provider(settings)


def _check_summariser(probe: LLMProvider) -> None:
    async def check() -> str | None:
        try:
            return await check_summariser(probe)
        finally:
            close = getattr(probe, "aclose", None)
            if close is not None:
                await close()

    click.echo(render_check(asyncio.run(check())))


main = build_cli()
