"""Finding the text Claude Code writes into a user's turn, so it is never read as the person's words.

Claude Code sends some of its own text under the user role, merged into the turn it
belongs to, each piece in an element of its own:

- a command the person ran, recorded as `<command-name>`, `<command-message>` and
  `<command-args>`. This is what they typed, so it stays theirs;
- the text a prompt command or a skill expands to, written right after its command
  with no element around it;
- for a command that runs on the person's machine, `<local-command-caveat>` before it
  and `<local-command-stdout>` (or `-stderr`) after it, when it printed anything;
- `<task-notification>`, when a background task finishes.

Only an element as Claude Code writes it counts: one that opens at the start of a line
and closes at the end of one. A person who names an element in a sentence is talking
about it. The person's text and the client's are told apart for crediting only:
nothing here changes what is sent, kept or condensed.
"""

import re

# What the person ran, which stays theirs.
_COMMAND = frozenset({"command-name", "command-message", "command-args"})
# Written before a command that runs on the person's machine. Such a command expands to nothing.
_CAVEAT = "local-command-caveat"
_ELEMENT = re.compile(
    r"^<(?P<name>command-name|command-message|command-args|local-command-caveat|local-command-stdout"
    r"|local-command-stderr|task-notification)>(?:(?!<(?P=name)>).)*?</(?P=name)>[ \t]*$",
    re.DOTALL | re.MULTILINE,
)


def client_written_ranges(text: str) -> tuple[tuple[int, int], ...]:
    """Return the (start, end) character ranges of a user turn's text that Claude Code wrote, in order.

    A command's expansion runs from the command to the next element, or to the end of
    the turn. A command has none when a caveat came before it, because it ran on the
    person's machine, or when the person's own words came before it, because Claude
    Code writes a command first in a message: the person pasted it.
    """
    ranges: list[tuple[int, int]] = []
    cursor = 0
    local = False  # a caveat came before the command being read
    by_person = False  # the last text read was the person's
    expands = False  # the last element was a command that expands into the text after it
    for element in _ELEMENT.finditer(text):
        if text[cursor : element.start()].strip():
            if expands:
                _add(ranges, cursor, element.start(), text)
            by_person, local, expands = not expands, False, False
        if element["name"] in _COMMAND:
            expands = not local and not by_person
        else:
            _add(ranges, element.start(), element.end(), text)
            by_person, local, expands = False, element["name"] == _CAVEAT, False
        cursor = element.end()
    if expands and text[cursor:].strip():
        _add(ranges, cursor, len(text), text)
    return tuple(ranges)


def _add(ranges: list[tuple[int, int]], start: int, end: int, text: str) -> None:
    """Add a range, joined to the one before it when only blank space lies between them."""
    if ranges and not text[ranges[-1][1] : start].strip():
        ranges[-1] = (ranges[-1][0], end)
    else:
        ranges.append((start, end))
