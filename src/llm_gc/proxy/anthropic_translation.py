"""Reading and rewriting an inbound Anthropic Messages request.

Inbound only. `llm/anthropic_client.py` handles the outbound direction, for calls
the engine makes on its own behalf; merging the two would tie the engine to the
web layer.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from llm_gc.exceptions import ConfigError
from llm_gc.models import Message
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

# A client can place instructions inside its conversation, as messages of this role:
# Claude Code adds its environment and its tools' instructions after a user turn.
_INSTRUCTION_ROLE = "system"


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

        content = _content_as_text(raw.get("content"), index)
        messages.append(
            Message(
                role=role,
                content=content,
                token_count=count_tokens(content),
                # Its place in the conversation, instructions not counted, so a turn's
                # index does not depend on where the client puts its instructions.
                turn_index=len(messages),
                raw_content=raw.get("content"),
            )
        )
    if not messages:
        raise ConfigError("request body has no user or assistant turn")
    return messages


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
