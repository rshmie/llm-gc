"""Reading and rewriting an inbound Anthropic Messages request.

Inbound only. `llm/anthropic_client.py` handles the outbound direction, for calls
the engine makes on its own behalf; merging the two would tie the engine to the
web layer.
"""

import hashlib
import json
import logging
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from llm_gc.exceptions import ConfigError
from llm_gc.models import Message
from llm_gc.proxy.client_written import client_written_ranges
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
# Only as the client writes it, a block that opens at the start of a line and closes at the end of
# one: a person who names the element in a sentence is talking about it. Never across a second
# opening tag, so a stray one cannot carry the words before a real element into it.
_REMINDER = re.compile(r"^<system-reminder>(?:(?!<system-reminder>).)*?</system-reminder>[ \t]*$",
                       re.DOTALL | re.MULTILINE)
_BLANK_RUN = re.compile(r"\n{3,}")

# A client can place instructions inside its conversation, as messages of this role:
# Claude Code adds its environment and its tools' instructions after a user turn.
_INSTRUCTION_ROLE = "system"

# Claude Code relays a background subagent's report into the main conversation as a user
# turn opening with this element, alone or after the one line it writes before it: another
# agent's findings, not words the person wrote.
_RELAYED = re.compile(r"\s*(?:Another Claude session sent a message:\s*)?<agent-message\b")

# Distinct system prompts and tool lists whose token counts are remembered: one per client
# setup in use (a main conversation, each kind of subagent), with room to spare.
_REPEATED_TEXTS_CACHED = 64

# The longest unfinished stream line held while waiting for its end. The events an outline
# reads are short; a line longer than this is a content delta, which it never reads.
_STREAM_LINE_MAX_BYTES = 64 * 1024
# The stream events an outline reads, matched in a line before it is parsed.
_OUTLINE_EVENTS = (b'"message_start"', b'"message_delta"', b'"content_block_start"', b'"signature_delta"',
                   b'"message_stop"')
# A tool's name or id longer than this is not one; it is not kept or shown.
_OUTLINE_STRING_MAX_CHARS = 128
# A tool call that starts a subagent gives it a prompt: shorter than this, a string input is a
# path or a flag; longer, its input is not held while the stream is read.
SPAWN_PROMPT_MIN_CHARS = 20
SPAWN_PROMPT_MAX_CHARS = 32_000
# The stream events a spawned prompt is read from, when prompts are read.
_PROMPT_EVENTS = (b'"input_json_delta"', b'"content_block_stop"')

# How long the provider keeps a cache entry when the marker names no lifetime: its documented
# default. A marker may name another as a number and a unit, such as "1h".
DEFAULT_CACHE_LIFETIME_S = 300.0
_CACHE_TTL = re.compile(r"(\d+)([smh])")
_CACHE_TTL_UNIT_S = {"s": 1, "m": 60, "h": 3600}

# Thinking block types, and the field holding the encrypted thinking. That field is unique per
# block and grows with the thinking (about 4 characters per token), so in a stream it can be a
# single line of hundreds of kilobytes. A block's key is a hash of the field's first characters,
# computed the same way for replies and requests.
_THINKING_FIELDS = {"thinking": "signature", "redacted_thinking": "data"}
_THINKING_BLOCK_TYPES = frozenset(_THINKING_FIELDS)
_THINKING_KEY_CHARS = 256
# Finds the field's first characters in the start of a stream line that is too long to keep.
_THINKING_OPENING = re.compile(rb'"(?:signature|data)"\s*:\s*"([^"\\]{1,%d})' % _THINKING_KEY_CHARS)
_THINKING_LINE_MARKS = (b'"signature_delta"', b'"redacted_thinking"')

# Takes a turn's thinking keys and its visible token count; returns its thinking token count.
ThinkingSizer = Callable[[Sequence[str], int], int]


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


def extract_messages(payload: dict[str, Any], size_thinking: ThinkingSizer | None = None) -> list[Message]:
    """Read the conversation out of a Messages API request body.

    Populates `token_count` and `turn_index`, which every downstream component
    scores on. If `size_thinking` is given, it sets `thinking_tokens` on assistant
    turns that contain thinking blocks.

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
        token_count = count_tokens(content)
        keys = thinking_keys(raw.get("content")) if size_thinking is not None and role == "assistant" else ()
        relayed = role == "user" and _RELAYED.match(content) is not None
        # Only in a turn the person may have written: tool output can quote the client's elements.
        typed = role == "user" and not relayed and not _carries_tool_results(raw.get("content"))
        messages.append(
            Message(
                role=role,
                content=content,
                token_count=token_count,
                thinking_tokens=size_thinking(keys, token_count) if size_thinking is not None and keys else 0,
                instructions=lifted,
                relayed=relayed,
                client_written=client_written_ranges(content) if typed else (),
                # Its place in the conversation, instructions not counted, so a turn's
                # index does not depend on where the client puts its instructions.
                turn_index=len(messages),
                raw_content=raw.get("content"),
            )
        )
    if not messages:
        raise ConfigError("request body has no user or assistant turn")
    return messages


def _carries_tool_results(content: object) -> bool:
    return isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") == "tool_result" for block in content)


def thinking_keys(content: object) -> tuple[str, ...]:
    """Return the keys of the thinking blocks in a turn's content, in order."""
    if not isinstance(content, list):
        return ()
    return tuple(key for block in content if (key := _block_thinking_key(block)) is not None)


def _block_thinking_key(block: object) -> str | None:
    if not isinstance(block, dict) or not isinstance(kind := block.get("type"), str):
        return None
    field = _THINKING_FIELDS.get(kind)
    return thinking_key(block.get(field)) if field is not None else None


def thinking_key(opening: object) -> str | None:
    """Return the key for a thinking block's signature (or redacted data), or None if there is none."""
    if not isinstance(opening, str) or not opening:
        return None
    return hashlib.blake2b(opening[:_THINKING_KEY_CHARS].encode(), digest_size=16).hexdigest()


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


def rewrite_overhead_tokens(payload: dict[str, Any], messages: Sequence[Message],
                            instructions: Sequence[Instruction] = ()) -> int:
    """Tokens a collection makes the provider cache again besides the recap and the recent turns' text.

    These are the client's instructions, which LLM-GC sends after the system prompt (each
    counted once, as the recap carries them), and the part of the system prompt and tools
    that the client's cache markers do not cover.
    """
    said_once = dict.fromkeys(text for message in messages for text in message.instructions)
    return (uncached_prefix_tokens(payload) + sum(count_tokens(text) for text in said_once)
            + sum(instruction.token_count for instruction in instructions))


def uncached_prefix_tokens(payload: dict[str, Any]) -> int:
    """Tokens of the tools and system prompt after the client's last cache marker in them; all of them if none.

    The provider caches the prompt up to a marker. Once a collection changes the messages, a
    cached read can reach no further than the last marker before them, so the rest is cached again.
    """
    raw_tools, system = payload.get("tools"), payload.get("system")
    tools: list[Any] = raw_tools if isinstance(raw_tools, list) else []
    system_blocks: list[Any] = system if isinstance(system, list) else [{"type": "text", "text": system}] if system else []
    blocks = [*tools, *system_blocks]                 # the provider's prompt order: tools, then system
    marked = [index for index, block in enumerate(blocks)
              if isinstance(block, dict) and isinstance(block.get("cache_control"), dict)]
    start = marked[-1] + 1 if marked else 0
    later_tools = tools[start:]
    later_system = system_blocks[max(0, start - len(tools)):]
    tools_text = json.dumps(later_tools, ensure_ascii=False) if later_tools else ""
    system_text = "\n".join(block["text"] for block in later_system
                            if isinstance(block, dict) and isinstance(block.get("text"), str))
    return _count_repeated(tools_text) + _count_repeated(system_text)


@lru_cache(maxsize=_REPEATED_TEXTS_CACHED)
def _count_repeated(text: str) -> int:
    # Every request of a conversation repeats the same system prompt and tools, often tens of
    # thousands of tokens; counted once each, not once per request.
    return count_tokens(text) if text else 0


def cache_lifetime_s(payload: dict[str, Any]) -> float:
    """How long the provider keeps this request's cached conversation, in seconds.

    The lifetime its last cache marker names, in prompt order: tools, system, then the
    messages' blocks, then a top-level marker, which the provider applies to the last block.
    The last marker is the entry that covers the conversation, the part a collection makes
    the provider write again. With no marker, or one naming no lifetime, the default.
    """
    blocks: list[object] = []
    for key in ("tools", "system"):
        if isinstance(value := payload.get(key), list):
            blocks.extend(value)
    messages = payload.get("messages")
    for message in messages if isinstance(messages, list) else ():
        if isinstance(message, dict) and isinstance(content := message.get("content"), list):
            blocks.extend(content)
    blocks.append(payload)
    markers = [marker for block in blocks
               if isinstance(block, dict) and isinstance(marker := block.get("cache_control"), dict)]
    ttl = markers[-1].get("ttl") if markers else None
    match = _CACHE_TTL.fullmatch(ttl) if isinstance(ttl, str) else None
    return float(int(match[1]) * _CACHE_TTL_UNIT_S[match[2]]) if match else DEFAULT_CACHE_LIFETIME_S


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
    kind, text = block.get("type"), block.get("text")
    if kind == "text" and isinstance(text, str):
        return text
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


# A refusal's reason is logged up to this length: enough for the provider's sentence, not a payload.
REFUSAL_REASON_MAX_CHARS = 300

@dataclass(frozen=True, slots=True)
class ReplyOutline:
    """What a reply did, never what it said: how it ended, the usage it reported, the tools it called."""

    status: int
    finished: bool = False                # it reached its end: a whole body, or a stream's message_stop
    stop_reason: str | None = None
    usage: ProviderUsage | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    retry_after_s: float | None = None    # the upstream's Retry-After on a refusal, when it gave one
    error_type: str | None = None         # the provider's name for a refusal, such as invalid_request_error
    thinking_key: str | None = None       # key of the reply's first thinking block (a hash, not content)


def read_reply(status: int, raw: bytes, *, read_prompts: bool = False) -> tuple[ReplyOutline, dict[str, str]]:
    """An unstreamed reply's outline and, when asked, the prompt each of its tool calls gave a subagent, by call id.

    An error body, or one this reader cannot parse, outlines only its status. The
    prompts are conversation content, kept apart from the outline so that whoever
    stores an outline never holds them.
    """
    try:
        body = json.loads(raw)
    except ValueError:
        return ReplyOutline(status=status), {}
    if status != 200 or not isinstance(body, dict) or body.get("type") != "message":
        error = body.get("error") if isinstance(body, dict) else None
        return ReplyOutline(status=status, error_type=_plain(error.get("type")) if isinstance(error, dict) else None), {}
    usage = body.get("usage")
    content = body.get("content")
    blocks: list[Any] = content if isinstance(content, list) else []
    calls = [(call, block) for block in blocks if (call := _tool_call(block)) is not None]
    outline = ReplyOutline(status=status, finished=True, stop_reason=_plain(body.get("stop_reason")),
                           usage=ProviderUsage.from_reported(usage) if isinstance(usage, dict) else None,
                           tool_calls=tuple(call for call, _ in calls),
                           thinking_key=next(iter(thinking_keys(blocks)), None))
    prompts = {call.call_id: prompt for call, block in calls
               if read_prompts and (prompt := spawn_prompt(block.get("input"))) is not None}
    return outline, prompts


def spawn_prompt(tool_input: object) -> str | None:
    """The prompt a tool call gives a subagent, or None: a `prompt` string of a prompt's length.

    Read by the input's field, not the tool's name, which Claude Code has changed.
    """
    prompt = tool_input.get("prompt") if isinstance(tool_input, dict) else None
    if not isinstance(prompt, str):
        return None
    prompt = prompt.strip()
    return prompt if SPAWN_PROMPT_MIN_CHARS <= len(prompt) <= SPAWN_PROMPT_MAX_CHARS else None


class StreamOutlineReader:
    """Reads a streamed reply's outline chunk by chunk, as it is relayed.

    Only the events an outline needs are parsed - `message_start` and `message_delta` for
    usage and the stop reason, `content_block_start` for each tool call, the first thinking
    block's signature for its key, `message_stop` for the end - and every other line is
    passed over unread. Only the unfinished last line of a chunk is held between chunks. A
    line too long to hold is dropped, but a signature's key is read from its start first.

    With `read_prompts`, a tool call's input is also gathered while it streams, and its
    prompt for a subagent kept, by call id; an input larger than a prompt can be is let go.
    """

    def __init__(self, *, read_prompts: bool = False) -> None:
        self._partial = b""
        self._usage: dict[str, Any] | None = None
        self._stop_reason: str | None = None
        self._calls: list[ToolCall] = []
        self._thinking_key: str | None = None
        self._finished = False
        self._events = _OUTLINE_EVENTS + (_PROMPT_EVENTS if read_prompts else ())
        self._read_prompts = read_prompts
        self._inputs: dict[int, tuple[str, list[str], int]] = {}   # block index -> call id, input pieces, their size
        self._prompts: dict[str, str] = {}

    def feed(self, chunk: bytes) -> None:
        lines = (self._partial + chunk).split(b"\n")
        self._partial = lines.pop()
        if len(self._partial) > _STREAM_LINE_MAX_BYTES:
            # Too long to keep. Only a thinking signature is this long among the events we read,
            # and its key needs just the start of the line.
            self._read_long_thinking_line(self._partial)
            self._partial = b""
        for line in lines:
            if line.startswith(b"data:") and any(name in line for name in self._events):
                self._read(line[len(b"data:"):])

    def outline(self) -> ReplyOutline:
        """What the stream showed so far: the prompt's usage once it began, the output as far as it got."""
        return ReplyOutline(status=200, finished=self._finished, stop_reason=self._stop_reason,
                            usage=ProviderUsage.from_reported(self._usage) if self._usage is not None else None,
                            tool_calls=tuple(self._calls), thinking_key=self._thinking_key)

    def spawn_prompts(self) -> dict[str, str]:
        """The prompt each tool call gave a subagent, by call id: conversation content, for linking only."""
        return dict(self._prompts)

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
            if self._read_prompts and isinstance(event.get("index"), int):
                self._inputs[event["index"]] = (call.call_id, [], 0)
        elif kind == "content_block_start":
            # A redacted block's data comes in its start event; a thinking signature comes in a delta.
            self._thinking_key = self._thinking_key or _block_thinking_key(event.get("content_block"))
        elif (kind == "content_block_delta" and isinstance(delta := event.get("delta"), dict)
              and delta.get("type") == "signature_delta"):
            self._thinking_key = self._thinking_key or thinking_key(delta.get("signature"))
        elif kind == "content_block_delta" and event.get("index") in self._inputs:
            self._gather(event["index"], event.get("delta"))
        elif kind == "content_block_stop" and event.get("index") in self._inputs:
            call_id, pieces, _ = self._inputs.pop(event["index"])
            try:
                prompt = spawn_prompt(json.loads("".join(pieces) or "{}"))
            except ValueError:
                prompt = None
            if prompt is not None:
                self._prompts[call_id] = prompt
        elif kind == "message_stop":
            self._finished = True

    def _read_long_thinking_line(self, head: bytes) -> None:
        if self._thinking_key is None and head.startswith(b"data:") and any(mark in head for mark in _THINKING_LINE_MARKS):
            # Read from the raw JSON. Safe because the value is base64, which needs no escaping.
            opening = _THINKING_OPENING.search(head)
            self._thinking_key = thinking_key(opening[1].decode("ascii", "replace")) if opening else None

    def _gather(self, index: int, delta: object) -> None:
        piece = delta.get("partial_json") if isinstance(delta, dict) else None
        if delta is None or not isinstance(piece, str):
            return
        call_id, pieces, size = self._inputs[index]
        size += len(piece)
        if size > 2 * SPAWN_PROMPT_MAX_CHARS:
            # Larger than any prompt, even escaped: an edit or a file, not a subagent's task.
            del self._inputs[index]
            return
        pieces.append(piece)
        self._inputs[index] = (call_id, pieces, size)

    def _merge_usage(self, reported: object) -> None:
        if isinstance(reported, dict):
            # Later events restate what changed: the output count is a running total, not an increment.
            self._usage = {**(self._usage or {}), **{k: v for k, v in reported.items() if v is not None}}


def _tool_call(block: object) -> ToolCall | None:
    if not isinstance(block, dict) or block.get("type") not in _TOOL_CALLS:
        return None
    name, call_id = _plain(block.get("name")), _plain(block.get("id"))
    return ToolCall(name=name, call_id=call_id) if name and call_id else None


def refusal_reason(raw: bytes) -> str:
    """What a refusal says was wrong with the request: its error type and message, cut short.

    The provider's message names the request's structure - a message index, a block type, a
    limit - not its content; it is cut short all the same, since it is written to a log.
    """
    try:
        body = json.loads(raw)
    except ValueError:
        return "no readable error"
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return "no readable error"
    kind, message = _plain(error.get("type")) or "error", error.get("message")
    return (f"{kind}: {message}" if isinstance(message, str) else kind)[:REFUSAL_REASON_MAX_CHARS]


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
        rendered = _render_turn(message, strip_thinking=not _keeps_thinking(message, strip_thinking_before))
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


def sent_thinking_tokens(messages: Iterable[Message], strip_thinking_before: int = 0) -> int:
    """Return the thinking tokens these turns still carry after `rewrite_messages` strips thinking."""
    return sum(message.thinking_tokens for message in messages if _keeps_thinking(message, strip_thinking_before))


def sends_thinking(messages: Iterable[Message], strip_thinking_before: int = 0) -> bool:
    """Return whether any thinking block is left in these turns after `rewrite_messages` strips thinking."""
    return any(_keeps_thinking(message, strip_thinking_before) and isinstance(message.raw_content, list)
               and any(map(is_thinking_block, message.raw_content)) for message in messages)


def _keeps_thinking(message: Message, strip_thinking_before: int) -> bool:
    # The front has a negative index and no thinking; turns before `strip_thinking_before` lose theirs.
    return not 0 <= message.turn_index < strip_thinking_before


def is_thinking_block(block: object) -> bool:
    """Return whether a content block is the model's thinking, in either form."""
    return isinstance(block, dict) and isinstance(kind := block.get("type"), str) and kind in _THINKING_BLOCK_TYPES


def _render_turn(message: Message, *, strip_thinking: bool) -> dict[str, Any] | None:
    """One turn as the provider expects it, or None if nothing is left to send."""
    content = message.raw_content if message.raw_content is not None else message.content
    if strip_thinking and isinstance(content, list):
        content = [block for block in content if not is_thinking_block(block)]
        if not content:
            return None
    return {"role": message.role, "content": content}


def forwardable_headers(headers: dict[str, str]) -> dict[str, str]:
    """Strip headers describing our connection rather than the request.

    The client's API key passes through untouched and is never stored, logged, or
    copied into an event. This function never logs its input for that reason.
    """
    return {name: value for name, value in headers.items() if name.lower() not in _HOP_BY_HOP}
