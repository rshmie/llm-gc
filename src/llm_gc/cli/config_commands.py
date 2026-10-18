"""The `llm-gc config` commands: find, create and (later) inspect the config file."""

from pathlib import Path

import click

from llm_gc.cli.config_file import read_config_template, write_config_file
from llm_gc.cli.options import config_file_option
from llm_gc.cli.settings import (
    CONFIG_PATH_ENV,
    DEFAULT_CONFIG_FILENAME,
    ENV_FILENAME,
    ConfigLocation,
    find_config_file,
    load_settings,
    settings_origins,
    user_config_path,
)
from llm_gc.cli.show import render_settings
from llm_gc.cli.style import INDENT, style, tidy_path
from llm_gc.cli.styled_commands import CommandError
from llm_gc.exceptions import ConfigError


def add_config_commands(cli: click.Group) -> None:
    """Attach the `config` group to the root command."""

    @cli.group()
    def config() -> None:
        """Create and inspect LLM-GC's settings.

        Settings live in llm-gc.toml and LLM-GC's own key in .env, both read from the folder
        you start llm-gc in.
        """

    @config.command()
    @config_file_option
    def path(config_path: Path | None) -> None:
        """Say which config file `llm-gc start` would use, and why."""
        try:
            location = find_config_file(config_path)
        except ConfigError as err:
            raise CommandError(str(err)) from err
        click.echo(_describe_location(location))

    @config.command()
    @click.option("--user", is_flag=True, help="Write the per-user file instead of one in this directory.")
    @click.option("--force", is_flag=True, help="Replace the file if it already exists.")
    def init(user: bool, force: bool) -> None:
        """Write a starter llm-gc.toml, every setting commented.

        Each setting is listed at its default and commented out, so the file changes nothing until edited.
        """
        target = user_config_path() if user else Path.cwd() / DEFAULT_CONFIG_FILENAME
        try:
            write_config_file(target, read_config_template(), overwrite=force)
        except ConfigError as err:
            raise CommandError(str(err)) from err
        env_file = Path.cwd() / ENV_FILENAME
        key_step = (
            f"Check that {tidy_path(env_file)} holds LLM-GC's own key: llm-gc config show says"
            if env_file.exists()
            else f"Put LLM-GC's own key in {tidy_path(env_file)} " + style("(copy .env.example)", dim=True)
        )
        click.echo(
            "\n".join(
                [
                    "",
                    f"{INDENT}{style('✓', fg='green')} Wrote {tidy_path(target)} "
                    + style("(LLM-GC's settings, each one commented)", dim=True),
                    "",
                    INDENT + style("Next", bold=True),
                    f"{INDENT}  1. {key_step}",
                    f"{INDENT}  2. llm-gc start --dashboard",
                    "",
                ]
            )
        )

    @config.command()
    @config_file_option
    def show(config_path: Path | None) -> None:
        """Print every setting in effect and where it came from.

        Shows what `llm-gc start` would use with no flags. It validates the configuration
        too, so it also answers "is my config valid?" without starting anything.
        """
        try:
            location = find_config_file(config_path)
            file = location.path if location else None
            settings = load_settings(file)
            source = f"{tidy_path(location.path)} ({location.origin})" if location else None
            click.echo(render_settings(settings, settings_origins(file), source=source))
        except ConfigError as err:
            raise CommandError(str(err)) from err


def _describe_location(location: ConfigLocation | None) -> str:
    """The file `start` would use, or, with none, every place it looked, in the order it looks.

    The full path, not `~`: this command's answer is meant to be copied.
    """
    if location is not None:
        return f"\n{INDENT}{location.path} " + style(f"({location.origin})", dim=True) + "\n"
    places = [
        ("--config PATH", "not given"),
        (CONFIG_PATH_ENV, "not set"),
        (f"./{DEFAULT_CONFIG_FILENAME}", "not found"),
        (tidy_path(user_config_path()), "not found"),
    ]
    width = max(len(place) for place, _ in places) + 3
    return "\n".join(
        [
            "",
            f"{INDENT}No config file, so the built-in defaults apply.",
            "",
            INDENT + style("Looked for, in order:", bold=True),
            *(f"{INDENT}  {place.ljust(width)}" + style(why, dim=True) for place, why in places),
            "",
            f"{INDENT}Create one here with: llm-gc config init",
            "",
        ]
    )
