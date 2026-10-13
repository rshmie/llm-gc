"""The LLM-GC wordmark at the top of `llm-gc start`. Application code.

Coloured with the dashboard's fidelity palette, verbatim to compacted, so the name fades the
way old context does. A terminal that does not advertise 24-bit colour gets the nearest
256-colour steps instead, which every modern terminal shows.
"""

import os

from llm_gc.cli.style import INDENT, style

_Rgb = tuple[int, int, int]

# Rendered once with figlet's "ANSI Shadow" font and kept here, so the CLI needs no figlet.
_ART = """\
██╗     ██╗     ███╗   ███╗       ██████╗  ██████╗
██║     ██║     ████╗ ████║      ██╔════╝ ██╔════╝
██║     ██║     ██╔████╔██║█████╗██║  ███╗██║
██║     ██║     ██║╚██╔╝██║╚════╝██║   ██║██║
███████╗███████╗██║ ╚═╝ ██║      ╚██████╔╝╚██████╗
╚══════╝╚══════╝╚═╝     ╚═╝       ╚═════╝  ╚═════╝"""
_LINES = _ART.splitlines()
_WIDTH = max(len(line) for line in _LINES)

# The dashboard's dark-theme --fid-verbatim, --fid-kept and --fid-compacted, left to right.
_STOPS: tuple[_Rgb, ...] = ((0x9E, 0xC5, 0xF4), (0x55, 0x98, 0xE7), (0x25, 0x6A, 0xBF))
# The channel levels of the 6×6×6 colour cube that 256-colour terminals share.
_CUBE_LEVELS = (0, 95, 135, 175, 215, 255)


def render_wordmark(columns: int | None) -> list[str]:
    """The wordmark's lines; none when there is no terminal (`columns` is None) or it is too narrow."""
    if columns is None or columns < len(INDENT) + _WIDTH:
        return []
    truecolor = os.environ.get("COLORTERM") in ("truecolor", "24bit")
    colours = [_colour_at(column / (_WIDTH - 1), truecolor) for column in range(_WIDTH)]
    return [INDENT + "".join(_paint(char, colours[column]) for column, char in enumerate(line)) for line in _LINES]


def _paint(char: str, colour: _Rgb | int) -> str:
    # Spaces carry no colour, so they are left bare.
    return char if char == " " else style(char, fg=colour)


def _colour_at(position: float, truecolor: bool) -> _Rgb | int:
    # `position` runs 0 to 1 across the stops, mixed linearly between the two either side of it.
    scaled = position * (len(_STOPS) - 1)
    index = min(int(scaled), len(_STOPS) - 2)
    mix = scaled - index
    r, g, b = (round(low + (high - low) * mix) for low, high in zip(_STOPS[index], _STOPS[index + 1], strict=True))
    return (r, g, b) if truecolor else _nearest_cube_colour((r, g, b))


def _nearest_cube_colour(rgb: _Rgb) -> int:
    r, g, b = (min(range(len(_CUBE_LEVELS)), key=lambda level: abs(_CUBE_LEVELS[level] - channel)) for channel in rgb)
    # The cube follows the 16 basic colours, one channel per base-6 digit.
    return 16 + 36 * r + 6 * g + b
