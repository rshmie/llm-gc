"""Click's command, group and help formatter, in LLM-GC's style. Application code.

The root group is a `StyledGroup`, and Click builds every command and group declared under
it from these classes (`command_class`, `group_class`). A new command therefore gets the
styled help and errors with nothing of its own.
"""

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import IO, Any

import click

from llm_gc.cli.style import style

# What `-h` and `--help` both show, on every command.
_HELP_OPTION_NAMES = ["-h", "--help"]


class StyledHelpFormatter(click.HelpFormatter):
    """Click's help layout with bold headings, flags and commands in cyan, and placeholders dimmed."""

    def write_usage(self, prog: str, args: str = "", prefix: str | None = None) -> None:
        super().write_usage(prog, args, style("Usage:", bold=True) + " " if prefix is None else prefix)

    def write_heading(self, heading: str) -> None:
        super().write_heading(style(heading, bold=True))

    def write_dl(self, rows: Iterable[tuple[str, str]], col_max: int = 30, col_spacing: int = 2) -> None:
        # Click measures the columns without escape codes, so styled terms still line up.
        super().write_dl([(_style_term(term), text) for term, text in rows], col_max, col_spacing)


class StyledContext(click.Context):
    """Every command's context: the styled formatter, and `-h` beside `--help`."""

    formatter_class = StyledHelpFormatter

    def __init__(self, command: click.Command, parent: click.Context | None = None, **kwargs: Any) -> None:
        # Set on the root context only; each subcommand's context inherits it from its parent.
        if parent is None and kwargs.get("help_option_names") is None:
            kwargs["help_option_names"] = _HELP_OPTION_NAMES
        super().__init__(command, parent=parent, **kwargs)


class StyledCommand(click.Command):
    """A command in LLM-GC's style."""

    context_class = StyledContext

    def make_context(self, info_name: str | None, args: list[str], parent: click.Context | None = None,
                     **extra: Any) -> click.Context:
        with _usage_errors_styled():
            return super().make_context(info_name, args, parent=parent, **extra)


class StyledGroup(click.Group):
    """A group in LLM-GC's style, whose commands and subgroups are styled too."""

    context_class = StyledContext
    command_class = StyledCommand
    # `type` means: subgroups are built from this same class.
    group_class = type

    def make_context(self, info_name: str | None, args: list[str], parent: click.Context | None = None,
                     **extra: Any) -> click.Context:
        with _usage_errors_styled():
            return super().make_context(info_name, args, parent=parent, **extra)

    def resolve_command(self, ctx: click.Context,
                        args: list[str]) -> tuple[str | None, click.Command | None, list[str]]:
        # Where "No such command" is raised, after the group's own context is made.
        with _usage_errors_styled():
            return super().resolve_command(ctx, args)


class CommandError(click.ClickException):
    """Every command's failure: set apart from what came before it, its label in red."""

    def show(self, file: IO[Any] | None = None) -> None:
        click.echo(_error_line(self.format_message()), file=file, err=file is None)


class StyledUsageError(click.UsageError):
    """Click's own usage error, such as an unknown command or option, shown like `CommandError`."""

    def show(self, file: IO[Any] | None = None) -> None:
        if self.ctx is not None:
            help_option = max(self.ctx.command.get_help_option_names(self.ctx), key=len)
            hint = style(f"Try '{self.ctx.command_path} {help_option}' for help.", dim=True)
            click.echo(f"{self.ctx.get_usage()}\n{hint}", file=file, err=file is None)
        click.echo(_error_line(self.format_message()), file=file, err=file is None)


@contextmanager
def _usage_errors_styled() -> Iterator[None]:
    """Re-raise Click's usage errors as `StyledUsageError`, keeping their message and context."""
    try:
        yield
    except click.UsageError as err:
        if isinstance(err, StyledUsageError):
            raise
        raise StyledUsageError(err.format_message(), err.ctx) from err


def _error_line(message: str) -> str:
    return f"\n{style('Error:', fg='red', bold=True)} {message}"


def _style_term(term: str) -> str:
    """A command's name in cyan; an option's flags in cyan and its value placeholder dimmed."""
    if not term.startswith("-"):
        return style(term, fg="cyan")
    return " ".join(style(word, fg="cyan") if word.startswith("-") else word if word == "/" else style(word, dim=True)
                    for word in term.split(" "))
