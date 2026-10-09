"""The command line's shared look: every command styled from the root, without code of its own."""

from collections.abc import Iterator

import click
import pytest
from click.testing import CliRunner

from llm_gc.cli.main import build_cli
from llm_gc.cli.styled_commands import StyledCommand, StyledGroup

RED_BOLD = "\x1b[31m\x1b[1m"


def _command_paths(group: click.Group, prefix: tuple[str, ...] = ()) -> Iterator[tuple[str, ...]]:
    """Every command and group under `group`, as the arguments that name it."""
    yield prefix
    for name, command in group.commands.items():
        if isinstance(command, click.Group):
            yield from _command_paths(command, (*prefix, name))
        else:
            yield (*prefix, name)


ALL_COMMANDS = list(_command_paths(build_cli()))


def _invoke(*args: str, color: bool = False) -> click.testing.Result:
    return CliRunner().invoke(build_cli(), list(args), color=color, terminal_width=80)


def test_every_command_and_group_is_built_from_the_styled_classes():
    def check(group: click.Group) -> None:
        assert isinstance(group, StyledGroup)
        for command in group.commands.values():
            if isinstance(command, click.Group):
                check(command)
            else:
                assert isinstance(command, StyledCommand)

    check(build_cli())


def test_a_command_added_later_is_styled_with_nothing_of_its_own():
    """The point of setting the classes on the root: a new command or group inherits them."""
    cli = build_cli()

    @cli.command()
    def later() -> None:
        """A command added after the fact."""

    @cli.group()
    def nested() -> None:
        """A group added after the fact."""

    @nested.command()
    def inner() -> None:
        """A command inside it."""

    assert isinstance(cli.commands["later"], StyledCommand)
    assert isinstance(nested, StyledGroup) and isinstance(nested.commands["inner"], StyledCommand)
    help_text = CliRunner().invoke(cli, ["nested", "inner", "--help"], color=True).output
    assert "\x1b[1mUsage:" in help_text


@pytest.mark.parametrize("path", ALL_COMMANDS, ids=lambda path: " ".join(path) or "llm-gc")
def test_styling_never_changes_the_layout(path):
    """Click pads by visible width, so the coloured help, unstyled, is the plain help exactly."""
    coloured = _invoke(*path, "--help", color=True).output
    assert "\x1b[" in coloured
    assert click.unstyle(coloured) == _invoke(*path, "--help").output


@pytest.mark.parametrize("path", ALL_COMMANDS, ids=lambda path: " ".join(path) or "llm-gc")
def test_h_is_short_for_help_everywhere(path):
    assert _invoke(*path, "-h").output == _invoke(*path, "--help").output


def test_click_s_own_usage_errors_look_like_ours():
    unknown = _invoke("confg", color=True)
    assert unknown.exit_code == 2
    assert f"{RED_BOLD}Error:\x1b[0m No such command 'confg'" in unknown.output
    assert "Try 'cli --help' for help." in click.unstyle(unknown.output)

    bad_option = _invoke("start", "--prot", "1", color=True)
    assert bad_option.exit_code == 2
    assert f"{RED_BOLD}Error:\x1b[0m No such option '--prot'" in bad_option.output


def test_a_command_s_own_error_has_the_same_label(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = _invoke("config", "path", "--config", str(tmp_path / "absent.toml"), color=True)
    assert result.exit_code == 1 and f"{RED_BOLD}Error:\x1b[0m config file not found" in result.output
