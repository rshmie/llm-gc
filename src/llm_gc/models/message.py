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

    `tool_output` marks the parts of a turn of tool results that its tools returned,
    as ranges in the same form. It is set only on such a turn that holds other text
    too: when a person interrupts a tool round, the client sends what they type in the
    same turn as the round's results. The person's words are then what neither range
    covers. Without it, a turn of tool results is all tool output.

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
    tool_output: tuple[tuple[int, int], ...] = ()
    thinking_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check_ranges(self) -> Self:
        if (self.client_written or self.tool_output) and self.role != "user":
            raise ValueError("only a user turn holds text the client or its tools wrote into it")
        for name, ranges in (("client_written", self.client_written), ("tool_output", self.tool_output)):
            if list(ranges) != sorted(ranges):
                raise ValueError(f"{name} ranges are out of order")
        # Each character has one writer, so the two never overlap.
        previous_end = 0
        for start, end in sorted(self.client_written + self.tool_output):
            if not previous_end <= start < end <= len(self.content):
                raise ValueError(f"range ({start}, {end}) overlaps another or falls outside the content")
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


# How a quote or a transcript names a relayed report's speaker, text the client wrote, and tool output.
RELAYED_SPEAKER = "subagent report"
CLIENT_SPEAKER = "client"
TOOL_SPEAKER = "tool results"


# Who wrote a part of a turn: its own speaker, the client, or one of the tools whose results it carries.
Writer = Literal["speaker", "client", "tool"]


class TextPart(NamedTuple):
    """A stretch of a turn's text, and who wrote it."""

    writer: Writer
    text: str


def parts_by_writer(message: Message) -> list[TextPart]:
    """Return the turn's text in order, divided where the client's or a tool's writing starts and ends.

    A turn no range divides is a single part, its speaker's, its text untouched. Otherwise
    each part is stripped of the blank space around it, and an empty one is left out.
    """
    if not _divided(message):
        return [TextPart("speaker", message.content)]
    marked: list[tuple[int, int, Writer]] = sorted(
        [(start, end, "client") for start, end in message.client_written]
        + [(start, end, "tool") for start, end in message.tool_output])
    parts: list[TextPart] = []
    cursor = 0
    for start, end, writer in marked:
        parts += [TextPart("speaker", message.content[cursor:start]), TextPart(writer, message.content[start:end])]
        cursor = end
    parts.append(TextPart("speaker", message.content[cursor:]))
    return [TextPart(part.writer, stripped) for part in parts if (stripped := part.text.strip())]


def person_text(message: Message) -> str:
    """Return the words the turn's speaker wrote: its text without the parts the client or a tool wrote."""
    if not _divided(message):
        return message.content
    return "\n\n".join(part.text for part in parts_by_writer(message) if part.writer == "speaker")


def said_by_user(message: Message) -> bool:
    """Whether the turn holds words the person wrote.

    A user turn that is not a relayed report, and whose text is not all the client's
    or its tools'. This decides what scoring reads as the subject now and whose facts
    are the user's; only `person_text` is the person's. A turn of tool results holds
    the person's words only where `tool_output` divides it.
    """
    if message.role != "user" or message.relayed or (tool_result_ids(message) and not message.tool_output):
        return False
    return bool(person_text(message)) if _divided(message) else True


def opens_exchange(message: Message) -> bool:
    """Whether an exchange opens at the turn: the person wrote it, and it answers no tool call.

    What a person types after interrupting a tool round is theirs, but it arrives in the
    turn that carries the round's results, and a tool result stays in the span of the
    call it answers.
    """
    return said_by_user(message) and not tool_result_ids(message)


def speaker_label(message: Message) -> str:
    """The turn's speaker as a quote or a transcript names it.

    A turn holding any of the person's words is the user's: `parts_by_writer` tells
    its parts apart. Otherwise a turn of tool results is the tools', and a turn the
    client wrote all of is the client's.
    """
    if message.relayed:
        return RELAYED_SPEAKER
    if said_by_user(message):
        return message.role
    if tool_result_ids(message):
        return TOOL_SPEAKER
    return CLIENT_SPEAKER if message.client_written else message.role


def _divided(message: Message) -> bool:
    return bool(message.client_written or message.tool_output)


def _blocks(message: Message) -> Iterable[dict[str, Any]]:
    if isinstance(message.raw_content, list):
        return (block for block in message.raw_content if isinstance(block, dict))
    return ()
