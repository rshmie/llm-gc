"""Reading and rewriting an inbound Anthropic Messages request.

Inbound only. `llm/anthropic_client.py` handles the outbound direction, for calls
the engine makes on its own behalf; merging the two would tie the engine to the
web layer.
"""

import json
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from llm_gc.exceptions import ConfigError
from llm_gc.models import Message
from llm_gc.proxy.traffic import ProviderUsage
from llm_gc.utils import count_tokens

logger = logging.getLogger(__name__)

_ROLES = {"user", "assistant"}

# How a block other than text reads in a turn's text. That text is what the engine
# sizes, scores, summarises and quotes, so a tool call or result missing from it
# would be missing from every collected turn. Thinking has no text form: it is
# the model's working, not something said, and no recap carries it.
_TOOL_CALLS = frozenset({"tool_use", "server_tool_use"})
_TOOL_CALL = "[tool call: {name}] {input}"
_TOOL_RESULT = "[tool result] {output}"
_TOOL_ERROR = "[tool result, error] {output}"
_ATTACHMENTS = {"image": "[image]", "document": "[document]"}

# A client can also place instructions inside a turn's text, in this element: Claude
# Code wraps CLAUDE.md, memory, the date and notes about files in it, in the user's
# messages and in tool results. Lifted out of the turn, so it is never condensed.
_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)
_BLANK_RUN = re.compile(r"\n{3,}")

# A client can place instructions inside its conversation, as messages of this role:
# Claude Code adds its environment and its tools' instructions after a user turn.
_INSTRUCTION_ROLE = "system"

# Claude Code relays a background subagent's report into the main conversation as a user
# turn opening with this element: another agent's findings, not words the person wrote.
_RELAYED = re.compile(r"\s*<agent-message\b")

# Distinct system prompts and tool lists whose token counts are remembered: one per client
# setup in use (a main conversation, each kind of subagent), with room to spare.
_REPEATED_TEXTS_CACHED = 64

# The longest unfinished stream line held while waiting for its end. The events an outline
# reads are short; a line longer than this is a content delta, which it never reads.
_STREAM_LINE_MAX_BYTES = 64 * 1024
# The stream events an outline reads, matched in a line before it is parsed.
_OUTLINE_EVENTS = (b'"message_start"', b'"message_delta"', b'"content_block_start"', b'"message_stop"')
# A tool's name or id longer than this is not one; it is not kept or shown.
_OUTLINE_STRING_MAX_CHARS = 128


@dataclass(frozen=True)
class Instruction:
    """A system message the client placed inside its conversation, set aside word for word.

    LLM-GC condenses conversation, never instructions: an instruction is not a turn,
    is never collected, and goes back into every forwarded request. `after_turns` is
    how many conversation turns came before it, which places it again.
    """

    after_turns: int
    raw: dict[str, Any]

    @property
    def token_count(self) -> int:
        return count_tokens(_content_as_text(self.raw.get("content"), -1))

# Hop-by-hop headers describe our connection to the client, not the request's
# content. `content-length` in particular would be wrong by however many tokens
# GC saved, since the body is rewritten after headers are built.
_HOP_BY_HOP = frozenset({
    "host", "content-length", "connection", "keep-alive", "transfer-encoding",
    "upgrade", "proxy-authorization", "proxy-authenticate", "te", "trailer",
    "accept-encoding",
})


def extract_messages(payload: dict[str, Any]) -> list[Message]:
    """Read the conversation out of a Messages API request body.

    Populates `token_count` and `turn_index`, which every downstream component
    scores on.

    Raises:
        ConfigError: The body is not a conversation we can process. Raised rather
            than returning an empty list, which would hand the provider a prompt
            the client never wrote.
    """
    raw_messages = _raw_messages(payload)
    messages: list[Message] = []
    for index, raw in enumerate(raw_messages):
        role = raw.get("role")
        if role == _INSTRUCTION_ROLE:
            continue
        if role not in _ROLES:
            raise ConfigError(f"messages[{index}] has unsupported role {role!r}")

        content, lifted = lift_instructions(_content_as_text(raw.get("content"), index), role)
        messages.append(
            Message(
                role=role,
                content=content,
                token_count=count_tokens(content),
                instructions=lifted,
                relayed=role == "user" and _RELAYED.match(content) is not None,
                # Its place in the conversation, instructions not counted, so a turn's
                # index does not depend on where the client puts its instructions.
                turn_index=len(messages),
                raw_content=raw.get("content"),
            )
        )
    if not messages:
        raise ConfigError("request body has no user or assistant turn")
    return messages


def carry_instructions(front: Message, stood_in: Sequence[Message]) -> Message:
    """The front, opened by the instructions of the turns it stands in for, word for word.

    In order, each said once: a client that repeats an instruction in many turns,
    as Claude Code does after each file it reads, gains nothing from the copies.
    """
    carried = tuple(dict.fromkeys(text for turn in stood_in for text in turn.instructions))
    if not carried:
        return front
    blocks = [{"type": "text", "text": text} for text in (*carried, front.content)]
    return front.model_copy(update={"raw_content": blocks, "instructions": carried})


def instruction_tokens(messages: Iterable[Message]) -> int:
    """Tokens of the instructions these turns carry: sent whole, with or without GC."""
    return sum(count_tokens(text) for message in messages for text in message.instructions)


def lift_instructions(text: str, role: str) -> tuple[str, tuple[str, ...]]:
    """A turn's text without the client's instruction elements, and those elements in order.

    Only the user's side: that is where a client writes, and an assistant reply that
    quotes the element is conversation.
    """
    lifted = tuple(_REMINDER.findall(text)) if role == "user" else ()
    if not lifted:
        return text, ()
    return _BLANK_RUN.sub("\n\n", _REMINDER.sub("", text)).strip(), lifted


def extract_instructions(payload: dict[str, Any]) -> list[Instruction]:
    """The system messages inside a request's conversation, each with its place in it."""
    instructions: list[Instruction] = []
    turns = 0
    for raw in _raw_messages(payload):
        if raw.get("role") == _INSTRUCTION_ROLE:
            instructions.append(Instruction(after_turns=turns, raw=raw))
        else:
            turns += 1
    return instructions


def fixed_prompt_tokens(payload: dict[str, Any]) -> int:
    """Tokens of the parts of a request LLM-GC never changes: the system prompt and the tool definitions.

    They go to the model whole on every request, with or without GC, so they count on
    both sides of every comparison. Left out, a share saved would be measured against
    only the part GC can shrink, and read larger than what the model was spared.
    """
    system = payload.get("system")
    tools = payload.get("tools")
    system_text = system if isinstance(system, str) else (
        "\n".join(block["text"] for block in system if isinstance(block, dict) and isinstance(block.get("text"), str))
        if isinstance(system, list) else "")
    tools_text = json.dumps(tools, ensure_ascii=False) if isinstance(tools, list) and tools else ""
    return _count_repeated(system_text) + _count_repeated(tools_text)


@lru_cache(maxsize=_REPEATED_TEXTS_CACHED)
def _count_repeated(text: str) -> int:
    # Every request of a conversation repeats the same system prompt and tools, often tens of
    # thousands of tokens; counted once each, not once per request.
    return count_tokens(text) if text else 0


def _raw_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ConfigError("request body must contain a non-empty 'messages' array")
    for index, raw in enumerate(raw_messages):
        if not isinstance(raw, dict):
            raise ConfigError(f"messages[{index}] is not an object")
    return raw_messages


def _content_as_text(content: object, index: int) -> str:
    """A turn's content as the text everything downstream reads: its words, tool calls and results.

    The request itself is forwarded with its original blocks; this text is what
    the turn is sized, scored, summarised, quoted and shown by. An attachment reads
    as a placeholder, and counts as one.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = [text for block in content if (text := _block_as_text(block)) is not None]
        skipped = len(content) - len(texts)
        if skipped:
            # Logged without the content: conversation text is user data.
            logger.debug("ignored content blocks with no text form",
                         extra={"message_index": index, "blocks_ignored": skipped})
        return "\n".join(texts)

    raise ConfigError(f"messages[{index}] has unsupported content type {type(content).__name__}")


def _block_as_text(block: object) -> str | None:
    """One block's text form, or None for a block nothing reads, such as thinking."""
    if not isinstance(block, dict):
        return None
    kind = block.get("type")
    if kind == "text" and isinstance(block.get("text"), str):
        return block["text"]
    if kind in _TOOL_CALLS:
        return _TOOL_CALL.format(name=block.get("name", "?"), input=json.dumps(block.get("input"), ensure_ascii=False))
    if kind == "tool_result":
        template = _TOOL_ERROR if block.get("is_error") else _TOOL_RESULT
        return template.format(output=_content_as_text(block.get("content") or "", -1))
    return _ATTACHMENTS.get(kind) if isinstance(kind, str) else None


def reply_text(raw: bytes, *, streamed: bool) -> str | None:
    """A finished reply's text, in the form the turn will be read in once a later request carries it.

    For showing only, never for the record. None when the reply did not finish, or
    is not one: an error body, or a stream that stopped before its end.
    """
    try:
        blocks = _streamed_blocks(raw) if streamed else _answered_blocks(raw)
    except (ValueError, KeyError, TypeError, AttributeError):
        # Malformed or of a shape this reader does not know: nothing is shown, and nothing else changes.
        return None
    return _content_as_text(blocks, -1) if blocks is not None else None


def _answered_blocks(raw: bytes) -> list[Any] | None:
    body = json.loads(raw)
    finished = isinstance(body, dict) and body.get("type") == "message" and body.get("stop_reason")
    return body["content"] if finished and isinstance(body.get("content"), list) else None


def _streamed_blocks(raw: bytes) -> list[Any] | None:
    """The content blocks a server-sent event stream built, if it reached `message_stop`."""
    blocks: dict[int, dict[str, Any]] = {}
    partial_inputs: dict[int, list[str]] = {}
    for line in raw.decode("utf-8").splitlines():
        if not line.startswith("data:"):
            continue
        event = json.loads(line[len("data:"):])
        kind, index = event.get("type"), event.get("index")
        if kind == "content_block_start":
            blocks[index] = dict(event["content_block"])
        elif kind == "content_block_delta":
            delta = event["delta"]
            if delta.get("type") == "text_delta":
                block = blocks.setdefault(index, {"type": "text"})
                block["text"] = block.get("text", "") + delta["text"]
            elif delta.get("type") == "input_json_delta":
                partial_inputs.setdefault(index, []).append(delta["partial_json"])
        elif kind == "message_stop":
            # A tool call's input arrives as pieces of JSON text, whole only at the end.
            for at, pieces in partial_inputs.items():
                blocks[at]["input"] = json.loads("".join(pieces) or "{}")
            return [blocks[at] for at in sorted(blocks)]
    return None


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool the reply asked the client to run: its name and the id its result will answer."""

    name: str
    call_id: str


@dataclass(frozen=True, slots=True)
class ReplyOutline:
    """What a reply did, never what it said: how it ended, the usage it reported, the tools it called."""

    status: int
    finished: bool = False                # it reached its end: a whole body, or a stream's message_stop
    stop_reason: str | None = None
    usage: ProviderUsage | None = None
    tool_calls: tuple[ToolCall, ...] = ()


def reply_outline(status: int, raw: bytes) -> ReplyOutline:
    """The outline of an unstreamed reply. An error body, or one this reader cannot parse, outlines only its status."""
    try:
        body = json.loads(raw)
    except ValueError:
        return ReplyOutline(status=status)
    if status != 200 or not isinstance(body, dict) or body.get("type") != "message":
        return ReplyOutline(status=status)
    usage = body.get("usage")
    content = body.get("content")
    blocks: list[Any] = content if isinstance(content, list) else []
    return ReplyOutline(status=status, finished=True, stop_reason=_plain(body.get("stop_reason")),
                        usage=ProviderUsage.from_reported(usage) if isinstance(usage, dict) else None,
                        tool_calls=tuple(call for block in blocks if (call := _tool_call(block)) is not None))


class StreamOutlineReader:
    """Reads a streamed reply's outline chunk by chunk, as it is relayed.

    Only the events an outline needs are parsed - `message_start` and `message_delta` for
    usage and the stop reason, `content_block_start` for each tool call, `message_stop`
    for the end - and every other line is passed over unread. Only the unfinished last
    line of a chunk is held between chunks.
    """

    def __init__(self) -> None:
        self._partial = b""
        self._usage: dict[str, Any] | None = None
        self._stop_reason: str | None = None
        self._calls: list[ToolCall] = []
        self._finished = False

    def feed(self, chunk: bytes) -> None:
        lines = (self._partial + chunk).split(b"\n")
        self._partial = lines.pop()
        if len(self._partial) > _STREAM_LINE_MAX_BYTES:
            # Not an event line this reader needs: the events it reads are short.
            self._partial = b""
        for line in lines:
            if line.startswith(b"data:") and any(name in line for name in _OUTLINE_EVENTS):
                self._read(line[len(b"data:"):])

    def outline(self) -> ReplyOutline:
        """What the stream showed so far: the prompt's usage once it began, the output as far as it got."""
        return ReplyOutline(status=200, finished=self._finished, stop_reason=self._stop_reason,
                            usage=ProviderUsage.from_reported(self._usage) if self._usage is not None else None,
                            tool_calls=tuple(self._calls))

    def _read(self, data: bytes) -> None:
        try:
            event = json.loads(data)
        except ValueError:
            return
        kind = event.get("type") if isinstance(event, dict) else None
        if kind == "message_start":
            self._merge_usage((event.get("message") or {}).get("usage"))
        elif kind == "message_delta":
            self._merge_usage(event.get("usage"))
            self._stop_reason = _plain((event.get("delta") or {}).get("stop_reason")) or self._stop_reason
        elif kind == "content_block_start" and (call := _tool_call(event.get("content_block"))) is not None:
            self._calls.append(call)
        elif kind == "message_stop":
            self._finished = True

    def _merge_usage(self, reported: object) -> None:
        if isinstance(reported, dict):
            # Later events restate what changed: the output count is a running total, not an increment.
            self._usage = {**(self._usage or {}), **{k: v for k, v in reported.items() if v is not None}}


def _tool_call(block: object) -> ToolCall | None:
    if not isinstance(block, dict) or block.get("type") not in _TOOL_CALLS:
        return None
    name, call_id = _plain(block.get("name")), _plain(block.get("id"))
    return ToolCall(name=name, call_id=call_id) if name and call_id else None


def _plain(value: object) -> str | None:
    """A short printable string from the reply, or None: a name or id that is anything else is not shown."""
    return value if isinstance(value, str) and value.isprintable() and len(value) <= _OUTLINE_STRING_MAX_CHARS else None


def rewrite_messages(payload: dict[str, Any], messages: list[Message], strip_thinking_before: int = 0,
                     instructions: Sequence[Instruction] = ()) -> dict[str, Any]:
    """Return a copy of the body with its conversation replaced.

    A copy, because the original is the fallback if anything downstream fails.
    Every other field passes through untouched, including ones this version has
    never heard of - the proxy changes the conversation and nothing else.

    Turns are sent as the client's original content (images, tool calls), not
    their text. Thinking blocks are removed from turns before
    `strip_thinking_before`: they were produced against a prompt the latest cut
    replaced, and the provider rejects a request that replays them. Removing them
    from a leading run of turns, oldest first, is what keeps later blocks valid.

    Each instruction goes back right after the turn it followed. One that followed a
    turn the front now stands in for goes right after the front, so every instruction
    is sent, word for word, exactly once.
    """
    turns = [message.turn_index for message in messages if message.turn_index >= 0]
    covered = turns[0] if turns else 0     # the turns before this one are what the front stands in for
    leading = [instruction.raw for instruction in instructions if instruction.after_turns <= covered]
    has_front = any(message.turn_index < 0 for message in messages)
    out: list[dict[str, Any]] = [] if has_front else list(leading)
    for message in messages:
        rendered = _render_turn(message, strip_thinking=0 <= message.turn_index < strip_thinking_before)
        if rendered is not None:
            out.append(rendered)
        if message.turn_index < 0:
            out += leading
        else:
            out += [instruction.raw for instruction in instructions
                    if instruction.after_turns == message.turn_index + 1]
    rewritten = dict(payload)
    rewritten["messages"] = out
    return rewritten


_THINKING_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking"})


def _render_turn(message: Message, *, strip_thinking: bool) -> dict[str, Any] | None:
    """One turn as the provider expects it, or None if nothing is left to send."""
    content = message.raw_content if message.raw_content is not None else message.content
    if strip_thinking and isinstance(content, list):
        content = [block for block in content
                   if not (isinstance(block, dict) and block.get("type") in _THINKING_BLOCK_TYPES)]
        if not content:
            return None
    return {"role": message.role, "content": content}


def forwardable_headers(headers: dict[str, str]) -> dict[str, str]:
    """Strip headers describing our connection rather than the request.

    The client's API key passes through untouched and is never stored, logged, or
    copied into an event. This function never logs its input for that reason.
    """
    return {name: value for name, value in headers.items() if name.lower() not in _HOP_BY_HOP}
