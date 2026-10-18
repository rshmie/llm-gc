"""Prompt texts, loaded from files that ship inside the installed package.

Prompts are code. They are the part of an LLM feature most likely to need
changing, most likely to be changed by someone who is not the original author,
and least likely to be covered by a test — so they live in reviewable `.md` files
rather than as string literals buried in a class, where a diff shows the wording
change on its own line.

Loaded with `importlib.resources`, not with `open(Path(__file__).parent / ...)`.
The difference matters at install time: `__file__` arithmetic assumes the package
is a directory of loose files on disk, which stops being true inside a zipimport,
a frozen bundle, or any other non-filesystem loader. `resources.files()` asks the
*loader* for the file and works in all of them.
"""

import re
from collections.abc import Iterable
from functools import lru_cache
from importlib import resources

COMPACTION_PROMPT = "compaction.md"
"""System prompt for `LLMCompactor`."""

FACTS_PROMPT = "facts.md"
"""System prompt for `LLMFactReader`."""

# Delimiters the prompts name, so a model can tell our instructions from conversation
# content. Conversation text can still contain them, so it is escaped before it is
# placed between them (see `escape_delimiters`).
TRANSCRIPT_OPEN = "<<<TRANSCRIPT"
TRANSCRIPT_CLOSE = "TRANSCRIPT>>>"
FACTS_OPEN = "<<<FACTS"
FACTS_CLOSE = "FACTS>>>"

# The angle brackets of a marker, found by the word they touch, in any letter case. A
# backslash goes between the brackets and the word, so the marker no longer appears.
_MARKER_OPENING = re.compile(r"<<<(?=TRANSCRIPT|FACTS)", re.IGNORECASE)
_MARKER_CLOSING = re.compile(r"(?:(?<=TRANSCRIPT)|(?<=FACTS))>>>", re.IGNORECASE)
# A line that opens the way a transcript line does, with a turn's label.
_TURN_LABEL = re.compile(r"^(\s*)(\[\s*turn\s*\d+\s*\])", re.IGNORECASE)


def escape_delimiters(text: str) -> str:
    """Return conversation text that cannot end the block it is placed in, or start another turn.

    A marker in the text gets a backslash between its brackets and its word
    (`TRANSCRIPT\\>>>`), and a line that opens with a turn's label gets one before
    it (`\\[turn 3]`). Without this, a turn quoting the closing marker would end the
    data early, and a line reading `[turn 3] user: ...` would read as the user's turn.
    The text of this repository contains both, so any session working on LLM-GC
    through its own proxy sends them. The prompts explain the backslash.

    Lines are split the way `str.splitlines` splits them, so a carriage return or
    any other line break a model might read as one counts too.
    """
    text = _MARKER_CLOSING.sub(r"\\>>>", _MARKER_OPENING.sub(r"<<<\\", text))
    return "".join(_TURN_LABEL.sub(r"\1\\\2", line, count=1) for line in text.splitlines(keepends=True))


def transcript_block(turns: Iterable[tuple[int, str, str]]) -> str:
    """Return turns as one delimited block of data, one `[turn N] speaker: text` line each.

    Args:
        turns: Each turn's index, the speaker's label and its text. Only the text is
            escaped: the label is ours.
    """
    lines = "\n".join(f"[turn {index}] {speaker}: {escape_delimiters(text)}" for index, speaker, text in turns)
    return f"{TRANSCRIPT_OPEN}\n{lines}\n{TRANSCRIPT_CLOSE}"


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Read a prompt that ships with the package.

    Cached: a prompt is read once per process and never changes afterwards, and
    compaction happens throughout a session. `lru_cache` on a single `str`
    argument is the whole mechanism — no manual dict, no staleness to reason
    about, because the file cannot change under a running process in any way we
    support.

    Args:
        name: Filename within this package, e.g. `COMPACTION_PROMPT`. Use the
            module constants rather than a literal, so a renamed file is a
            one-line change and a typo is caught at import.

    Returns:
        The file's contents, stripped of trailing whitespace.

    Raises:
        FileNotFoundError: No such prompt ships with the package. Deliberately
            not wrapped in a project exception — a missing prompt file is a
            packaging bug, not an operational condition, and CLAUDE.md §2 says
            programmer errors crash loudly.
    """
    return (resources.files(__package__) / name).read_text(encoding="utf-8").rstrip()
