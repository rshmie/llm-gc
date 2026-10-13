"""The `llm-gc start` wordmark: when it is drawn, and how it is coloured."""

import re

import pytest

from llm_gc.cli.banner import render_header
from llm_gc.cli.settings import ProxySettings
from llm_gc.cli.style import INDENT
from llm_gc.cli.wordmark import render_wordmark

_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
_WIDE = 200


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Colour allowed, and no colour depth advertised unless a test sets one."""
    for name in ("NO_COLOR", "COLORTERM"):
        monkeypatch.delenv(name, raising=False)


def _art_width() -> int:
    return max(len(_ESCAPE.sub("", line)) for line in render_wordmark(_WIDE))


def test_no_terminal_means_no_wordmark():
    assert render_wordmark(None) == []
    assert "LLM-GC 0.1.0" in render_header(ProxySettings(), None, "0.1.0")


def test_drawn_only_when_every_line_fits():
    exact = _art_width()
    assert render_wordmark(exact)
    assert render_wordmark(exact - 1) == []
    assert all(len(_ESCAPE.sub("", line)) <= exact for line in render_wordmark(exact))


def test_header_shows_the_version_under_the_wordmark():
    header = _ESCAPE.sub("", render_header(ProxySettings(), None, "0.1.0", _WIDE))
    assert "LLM-GC 0.1.0" not in header
    assert "v0.1.0" in header
    assert "██" in header


def test_gradient_runs_from_the_first_stop_to_the_last(monkeypatch):
    monkeypatch.setenv("COLORTERM", "truecolor")
    first_line = render_wordmark(_WIDE)[0]
    colours = re.findall(r"\x1b\[38;2;(\d+;\d+;\d+)m", first_line)
    assert colours[0] == "158;197;244"
    assert colours[-1] == "37;106;191"


def test_without_truecolor_only_256_colour_codes_are_used():
    rendered = "".join(render_wordmark(_WIDE))
    assert "\x1b[38;5;" in rendered
    assert "\x1b[38;2;" not in rendered


def test_no_color_keeps_the_art_and_drops_every_escape_code(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    lines = render_wordmark(_WIDE)
    assert lines
    assert all(line.startswith(INDENT) for line in lines)
    assert "\x1b[" not in "".join(lines)
