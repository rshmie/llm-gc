from collections.abc import Iterable
from typing import Any, Literal, NamedTuple, Self

from pydantic import BaseModel, Field, model_validator


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

    `client_written` marks the parts of a user turn's `content` that the client wrote
    rather than the person: the text a command expands to, a local command's output,
    a notice that a background task finished. Each part is a (start, end) range of
    characters, in order. Like a relayed turn, they are conversation, condensed with
    the rest of the turn, but never read as the person's words (`person_text`).

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
    client_written: tuple[tuple[int, int], ...] = ()
    thinking_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check_client_written(self) -> Self:
        if self.client_written and self.role != "user":
            raise ValueError("only a user turn holds text the client wrote into it")
        previous_end = 0
        for start, end in self.client_written:
            if not previous_end <= start < end <= len(self.content):
                raise ValueError(f"client_written range ({start}, {end}) is out of order or outside the content")
            previous_end = end
        return self

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


# How a quote or a transcript names a relayed report's speaker, and text the client wrote.
RELAYED_SPEAKER = "subagent report"
CLIENT_SPEAKER = "client"


class TextPart(NamedTuple):
    """A stretch of a turn's text, and whether the client wrote it rather than the turn's speaker."""

    by_client: bool
    text: str


def parts_by_writer(message: Message) -> list[TextPart]:
    """Return the turn's text in order, divided where the client's writing starts and ends.

    A turn the client wrote none of is a single part, its text untouched. Otherwise
    each part is stripped of the blank space around it, and an empty one is left out.
    """
    if not message.client_written:
        return [TextPart(by_client=False, text=message.content)]
    parts: list[TextPart] = []
    cursor = 0
    for start, end in message.client_written:
        parts += [TextPart(False, message.content[cursor:start]), TextPart(True, message.content[start:end])]
        cursor = end
    parts.append(TextPart(False, message.content[cursor:]))
    return [TextPart(part.by_client, stripped) for part in parts if (stripped := part.text.strip())]


def person_text(message: Message) -> str:
    """Return the words the turn's speaker wrote: its text without the parts the client wrote."""
    if not message.client_written:
        return message.content
    return "\n\n".join(part.text for part in parts_by_writer(message) if not part.by_client)


def said_by_user(message: Message) -> bool:
    """Whether the turn holds words the person wrote.

    A user turn that is not tool results or a relayed report, and that the client did
    not write all of. This decides what opens an exchange, what scoring reads as the
    subject now, and whose facts are the user's. For a turn the client wrote part of,
    only `person_text` is the person's.
    """
    return (message.role == "user" and not message.relayed and not tool_result_ids(message)
            and (not message.client_written or bool(person_text(message))))


def speaker_label(message: Message) -> str:
    """The turn's speaker as a quote or a transcript names it.

    A turn the client wrote all of is the client's. A turn it wrote part of is still the
    user's: `parts_by_writer` tells the parts apart.
    """
    if message.relayed:
        return RELAYED_SPEAKER
    if message.client_written and not person_text(message):
        return CLIENT_SPEAKER
    return message.role


def _blocks(message: Message) -> Iterable[dict[str, Any]]:
    if isinstance(message.raw_content, list):
        return (block for block in message.raw_content if isinstance(block, dict))
    return ()
