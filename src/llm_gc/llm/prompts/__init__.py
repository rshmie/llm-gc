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

from functools import lru_cache
from importlib import resources

COMPACTION_PROMPT = "compaction.md"
"""System prompt for `LLMCompactor`."""


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
