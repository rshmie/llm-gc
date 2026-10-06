"""Which conversation a request continues.

A chat completions API is stateless: every request carries the whole
conversation and nothing else says which one it is. So a conversation is
identified by its content - a chain hash per turn - and a request is matched to
the session whose recorded conversation is a prefix of what it sent. See
doc/design/proxy/overview.md.
"""

import hashlib
import json
from collections.abc import Sequence

from llm_gc.models import Message
from llm_gc.proxy.anthropic_translation import lift_instructions

SESSION_HEADER = "x-llm-gc-session"
"""Client-supplied session id. Preferred over matching by content."""


def chain_hashes(messages: Sequence[Message]) -> list[str]:
    """One hash per turn, each covering the whole conversation up to and including it.

    `hashes[k]` equals another conversation's `hashes[k]` only if the two say the
    same through turn k, so a session can be found by matching the hash of the last
    turn it recorded against a prefix of an incoming request. Each turn is hashed
    from its role and its provider content (`raw_content`, falling back to the
    text), in canonical form: key order, a plain string against one text block,
    delivery hints such as `cache_control`, and the client's instructions inside a
    turn do not count as an edit. A client moves its cache markers onto the newest
    turns with every request, and Claude Code rewrites the instructions in earlier
    turns, so a hash that included either saw one conversation as several.
    """
    hashes: list[str] = []
    previous = ""
    for message in messages:
        content = message.raw_content if message.raw_content is not None else message.content
        turn = json.dumps([message.role, _as_said(content, message.role)], sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False)
        previous = hashlib.sha256(f"{previous}\x00{turn}".encode("utf-8")).hexdigest()
        hashes.append(previous)
    return hashes


# Keys that tell the provider how to deliver a turn, not what it says.
_DELIVERY_HINTS = frozenset({"cache_control"})


def _as_said(content: object, role: str) -> object:
    """Content as said: a plain string as its one text block, without delivery hints or instructions, at any depth."""
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    return _said(content, role)


def _said(value: object, role: str) -> object:
    if isinstance(value, dict):
        return {key: _said(item, role) for key, item in value.items() if key not in _DELIVERY_HINTS}
    if isinstance(value, list):
        # A block that held only instructions is gone, not an empty block.
        return [said for item in value if not _is_empty_text(said := _said(item, role))]
    if isinstance(value, str):
        return lift_instructions(value, role)[0]
    return value


def _is_empty_text(block: object) -> bool:
    return isinstance(block, dict) and block.get("type") == "text" and not block.get("text")
