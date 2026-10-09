"""Terminal styling shared by the commands' human-readable output.

Styled with click, which drops the styling when the output is not a terminal; `NO_COLOR`
(https://no-color.org) turns it off on a terminal too.
"""

import os
from pathlib import Path
from typing import Any

import click

INDENT = "  "


def style(text: str, **styles: Any) -> str:
    """`click.style`, unless the user has asked for no colour."""
    return text if os.environ.get("NO_COLOR") else click.style(text, **styles)


def tidy_path(path: Path) -> str:
    """The path with the home directory shown as `~`, as a shell would. For reading, not for copying."""
    home = Path.home()
    return f"~/{path.relative_to(home)}" if path.is_relative_to(home) else str(path)
