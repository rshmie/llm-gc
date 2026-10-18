"""User turns written partly by their client or its tools, marked the way the proxy marks them."""

from llm_gc.models import Message

# Text Claude Code writes into a person's turn, with invented words.
SKILL_RULES = "Base directory for this skill: /skills/audit\n\nThe database is MySQL. Keep at most 5 rows."
NOTICE = "<task-notification>\n<status>completed</status>\nThe tests passed.\n</task-notification>"
COMMAND = "<command-name>/audit</command-name>\n<command-args>verify</command-args>"
# Claude Code's mark that the person stopped a tool round. What they typed follows it.
INTERRUPTED = "[Request interrupted by user]"


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


def turn_with_typed_words(index: int, call_id: str, words: str, *, output: str = "ok", tokens: int = 20) -> Message:
    """Return the turn answering tool call `call_id` that also holds what the person typed after
    interrupting the round: its result, the mark, then their words, divided as the proxy divides it."""
    result = f"[tool result] {output}"
    mark = len(result) + 1
    return Message(
        role="user",
        content=f"{result}\n{INTERRUPTED}\n{words}",
        token_count=tokens,
        turn_index=index,
        raw_content=[
            {"type": "tool_result", "tool_use_id": call_id, "content": output},
            {"type": "text", "text": INTERRUPTED},
            {"type": "text", "text": words},
        ],
        tool_output=((0, len(result)),),
        client_written=((mark, mark + len(INTERRUPTED)),),
    )
