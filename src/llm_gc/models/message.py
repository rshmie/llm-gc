from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in an LLM conversation.

    Represents one turn (user, assistant, or system) with its content,
    token count, and position in the conversation sequence.

    `content` is the turn as text - its words, with any tool calls and tool results
    written out - which is what sizing, scoring, compaction, extraction and quoting
    read. `raw_content` is the provider's original content exactly as the client
    sent it - images, tool calls, thinking blocks - so a turn forwarded verbatim is
    forwarded byte for byte. It never leaves through serialisation or repr: it can
    hold an attached file, and conversation content must not reach logs or events.

    `instructions` is text the client placed inside the turn for the model rather
    than as conversation - Claude Code's `<system-reminder>` blocks, for one. It is
    not in `content`, so the engine never condenses it; whoever forwards a collected
    turn delivers it word for word. Excluded from serialisation for the same reason
    as `raw_content`.

    `relayed` marks a user-role turn the client wrote to relay another agent's
    report - a background subagent's findings - rather than words the person typed.
    It is conversation, condensed like any turn, but never read as the person's.

    `thinking_tokens` is an estimate of the model's hidden thinking inside `raw_content`,
    in the same units as `token_count`. The provider bills it as input whenever the turn
    is sent with its thinking, but it has no readable text. It only counts toward prompt
    cost; scoring and summarising ignore it. It is 0 when unknown.
    """
    role: Literal["user", "assistant", "system"]
    content: str
    token_count: int = 0
    turn_index: int = 0
    raw_content: Any = Field(default=None, exclude=True, repr=False)
    instructions: tuple[str, ...] = Field(default=(), exclude=True, repr=False)
    relayed: bool = False
    thinking_tokens: int = Field(default=0, ge=0)

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


# How a quote or a transcript names a relayed report's speaker.
RELAYED_SPEAKER = "subagent report"


def said_by_user(message: Message) -> bool:
    """Whether the turn is words the person wrote: a user turn that is neither tool results nor a relayed report.

    What opens an exchange, what scoring reads as the subject now, and whose facts are the user's.
    """
    return message.role == "user" and not message.relayed and not tool_result_ids(message)


def speaker_label(message: Message) -> str:
    """The turn's speaker as a quote or a transcript names it."""
    return RELAYED_SPEAKER if message.relayed else message.role


def _blocks(message: Message) -> Iterable[dict[str, Any]]:
    if isinstance(message.raw_content, list):
        return (block for block in message.raw_content if isinstance(block, dict))
    return ()
