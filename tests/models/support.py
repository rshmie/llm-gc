"""User turns written partly by their client, marked the way the proxy marks them."""

from llm_gc.models import Message

# Text Claude Code writes into a person's turn, with invented words.
SKILL_RULES = "Base directory for this skill: /skills/audit\n\nThe database is MySQL. Keep at most 5 rows."
NOTICE = "<task-notification>\n<status>completed</status>\nThe tests passed.\n</task-notification>"
COMMAND = "<command-name>/audit</command-name>\n<command-args>verify</command-args>"


def turn_with_client_text(index: int, *parts: tuple[bool, str], tokens: int = 20) -> Message:
    """Return a user turn made of these parts, each (written by the client, text), joined by blank lines."""
    content = ""
    ranges = []
    for by_client, text in parts:
        content += "\n\n" if content else ""
        if by_client:
            ranges.append((len(content), len(content) + len(text)))
        content += text
    return Message(role="user", content=content, token_count=tokens, turn_index=index, client_written=tuple(ranges))
