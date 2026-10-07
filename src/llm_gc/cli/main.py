"""The `llm-gc` command line. Application code: the library never imports it."""

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path

import click

from llm_gc import __version__
from llm_gc.cli.serve import serve_proxy
from llm_gc.cli.settings import ProxySettings, create_summary_provider, find_config_file, load_settings
from llm_gc.engine import GcService, build_gc_service
from llm_gc.exceptions import ConfigError
from llm_gc.llm.provider import LLMProvider
from llm_gc.proxy.startup_checks import check_summariser

ServeFn = Callable[[GcService, ProxySettings], None]
ProviderFactory = Callable[[ProxySettings], LLMProvider]

_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def build_cli(*, serve: ServeFn = serve_proxy, make_provider: ProviderFactory = create_summary_provider) -> click.Group:
    """The command group, with its two side effects passed in.

    Binding ports and calling a model are what a test must not do, so they arrive as
    arguments: `main` uses the real ones, and a test hands in recorders.
    """

    @click.group()
    @click.version_option(__version__, prog_name="llm-gc")
    def cli() -> None:
        """LLM-GC: context garbage collection for LLM conversations, as a proxy."""

    @cli.command()
    @click.option("--config", "config_path", type=click.Path(path_type=Path), help="The TOML config file.")
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
    @click.option("--log-level", type=click.Choice(_LOG_LEVELS, case_sensitive=False))
    def start(config_path: Path | None, port: int | None, dashboard: bool | None, dashboard_port: int | None,
              upstream: str | None, compactor: str | None, summary_provider: str | None,
              summary_model: str | None, min_cut_tokens: int | None, keep_last: int | None,
              context_window: int | None, log_level: str | None) -> None:
        """Start the proxy (and the dashboard, with --dashboard).

        Every option is also a setting: a flag you leave out takes its value from the
        environment, then .env, then the config file, then the built-in default.
        """
        try:
            location = find_config_file(config_path)
            settings = load_settings(
                location.path if location else None,
                port=port, dashboard=dashboard, dashboard_port=dashboard_port, upstream=upstream,
                compactor=compactor, log_level=log_level.upper() if log_level else None,
                summary={"provider": summary_provider, "model": summary_model},
                engine={"min_cut_tokens": min_cut_tokens, "last_n_turns_to_keep": keep_last,
                        "context_window": context_window},
            )
            engine_provider = _prepare_summariser(settings, make_provider)
        except ConfigError as err:
            raise click.ClickException(str(err)) from err

        # The application chooses how the library's logs are shown. They name sessions and
        # counts, never conversation content.
        logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                            datefmt="%H:%M:%S")
        logging.getLogger("httpx2").setLevel(logging.WARNING)

        click.echo(f"config:    {f'{location.path} ({location.origin})' if location else 'none, built-in defaults'}")
        click.echo(f"upstream:  {settings.upstream}")
        click.echo(f"compactor: {_describe_compactor(settings)}")
        click.echo(f"\nproxy      http://127.0.0.1:{settings.port}   (stats: /stats)")
        if settings.dashboard:
            click.echo(f"dashboard  http://127.0.0.1:{settings.dashboard_port}")
        click.echo("")
        serve(build_gc_service(settings.engine, summary_provider=engine_provider), settings)

    return cli


def _prepare_summariser(settings: ProxySettings, make_provider: ProviderFactory) -> LLMProvider | None:
    """The engine's summary client, after asking the upstream to accept it; None for the noop compactor.

    Raises:
        ConfigError: There is no credential, or the upstream refuses the model or credential.
    """
    if settings.compactor == "noop":
        return None
    if not settings.has_summary_credential:
        raise ConfigError(
            f"no credential for the {settings.summary.provider} summariser. Set it in the environment or .env "
            "(see .env.example), or pass --compactor noop to run without summaries."
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

    warning = asyncio.run(check())
    if warning:
        click.echo(f"check:     WARNING: {warning}")


def _describe_compactor(settings: ProxySettings) -> str:
    if settings.compactor == "noop":
        return "noop (no summaries: exchanges are kept or archived, never condensed)"
    chosen = settings.summary
    return f"{chosen.provider}, model {chosen.model or 'default'}"


main = build_cli()
