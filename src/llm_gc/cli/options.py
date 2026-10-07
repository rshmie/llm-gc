"""Command-line options shared by more than one command."""

from pathlib import Path

import click

config_file_option = click.option(
    "--config", "config_path", type=click.Path(path_type=Path),
    help="The TOML config file (default: the first found; see `llm-gc config path`).",
)
