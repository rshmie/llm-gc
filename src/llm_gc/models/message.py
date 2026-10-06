from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in an LLM conversation.

    Represents one turn (user, assistant, or system) with its content,
    token count, and position in the conversation sequence.

    `content` is the turn's text, which is what scoring, compaction and extraction
    read. `raw_content` is the provider's original content exactly as the client
    sent it - images, tool calls, thinking blocks - so a turn forwarded verbatim is
    forwarded byte for byte. It never leaves through serialisation or repr: it can
    hold an attached file, and conversation content must not reach logs or events.
    """
    role: Literal["user", "assistant", "system"]
    content: str
    token_count: int = 0
    turn_index: int = 0
    raw_content: Any = Field(default=None, exclude=True, repr=False)

    def preview(self, max_length: int = 50) -> str:
        if len(self.content) > max_length:
            return self.content[:max_length] + "..."
        return self.content


# The provider's own blocks of a turn: a tool call and the result that answers it
# travel as blocks in `raw_content`, not in the text.
def tool_use_ids(message: Message) -> list[str]:
    return [block["id"] for block in _blocks(message) if block.get("type") == "tool_use" and "id" in block]


def tool_result_ids(message: Message) -> list[str]:
    return [block["tool_use_id"] for block in _blocks(message)
            if block.get("type") == "tool_result" and "tool_use_id" in block]


def _blocks(message: Message) -> Iterable[dict]:
    if isinstance(message.raw_content, list):
        return (block for block in message.raw_content if isinstance(block, dict))
    return ()
