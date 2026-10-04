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
