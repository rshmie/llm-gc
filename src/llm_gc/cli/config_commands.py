"""The `llm-gc config` commands: find, create and (later) inspect the config file."""

from pathlib import Path

import click

from llm_gc.cli.config_file import read_config_template, write_config_file
from llm_gc.cli.options import config_file_option
from llm_gc.cli.settings import (
    DEFAULT_CONFIG_FILENAME, find_config_file, load_settings, settings_origins, user_config_path,
)
from llm_gc.cli.show import render_settings
from llm_gc.exceptions import ConfigError


def add_config_commands(cli: click.Group) -> None:
    """Attach the `config` group to the root command."""

    @cli.group()
    def config() -> None:
        """Find, create and inspect the config file."""

    @config.command()
    @config_file_option
    def path(config_path: Path | None) -> None:
        """Say which config file `llm-gc start` would use, and why."""
        try:
            location = find_config_file(config_path)
        except ConfigError as err:
            raise click.ClickException(str(err)) from err
        if location is None:
            click.echo("no config file: the built-in defaults apply (create one with `llm-gc config init`)")
        else:
            click.echo(f"{location.path}  ({location.origin})")

    @config.command()
    @click.option("--user", is_flag=True, help="Write the per-user file instead of one in this directory.")
    @click.option("--force", is_flag=True, help="Replace the file if it already exists.")
    def init(user: bool, force: bool) -> None:
        """Write a commented starter config file, with every setting at its default."""
        target = user_config_path() if user else Path.cwd() / DEFAULT_CONFIG_FILENAME
        try:
            write_config_file(target, read_config_template(), overwrite=force)
        except ConfigError as err:
            raise click.ClickException(str(err)) from err
        click.echo(f"wrote {target}\nEdit it, then run `llm-gc start`.")

    @config.command()
    @config_file_option
    def show(config_path: Path | None) -> None:
        """Print every effective setting and the layer it came from.

        Shows what `llm-gc start` would use with no flags. It validates the configuration
        too, so it also answers "is my config valid?" without starting anything.
        """
        try:
            location = find_config_file(config_path)
            file = location.path if location else None
            settings = load_settings(file)
            click.echo(render_settings(settings, settings_origins(file)))
        except ConfigError as err:
            raise click.ClickException(str(err)) from err
