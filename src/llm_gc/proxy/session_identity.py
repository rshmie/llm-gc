"""Which conversation a request continues.

A chat completions API is stateless: every request carries the whole
conversation and nothing else says which one it is. So a conversation is
identified by its content - a chain hash per turn - and a request is matched to
the session whose recorded conversation is a prefix of what it sent. See
doc/design/proxy/overview.md.
"""

import hashlib
import json
import logging
import re
from collections.abc import Mapping, Sequence

from llm_gc.models import ClientOrigin, Message
from llm_gc.proxy.anthropic_translation import is_thinking_block, lift_instructions

logger = logging.getLogger(__name__)

SESSION_HEADER = "x-llm-gc-session"
"""Client-supplied session id, a short plain id. Preferred over matching by content."""

# Claude Code names its own session, and each sub-agent, on every request. They label a
# conversation; they never identify one, since one Claude Code session holds several.
CLIENT_SESSION_HEADER = "x-claude-code-session-id"
CLIENT_AGENT_HEADER = "x-claude-code-agent-id"

# A client's ids reach the log and the dashboard, so only a short plain id is taken.
_CLIENT_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def named_session(headers: Mapping[str, str]) -> str | None:
    """The session a request names with `SESSION_HEADER`, or None when it names none.

    The name follows the same rule as the client's own ids, since it becomes a log field,
    a dashboard label and part of the dashboard's addresses. Any other value is ignored
    with a warning, and the conversation is found by its content instead.
    """
    value = headers.get(SESSION_HEADER)
    session_id = _client_id(value)
    if value is not None and session_id is None:
        # Only its length is logged: the value is what failed the check.
        logger.warning(
            "%s ignored: not 1-64 letters, digits, '_' or '-'; the conversation is matched by its content",
            SESSION_HEADER,
            extra={"header_length": len(value)},
        )
    return session_id


def client_origin(headers: Mapping[str, str]) -> ClientOrigin | None:
    """The client session and agent a request names, or None when it names no session."""
    session_id = _client_id(headers.get(CLIENT_SESSION_HEADER))
    if session_id is None:
        return None
    return ClientOrigin(session_id=session_id, agent_id=_client_id(headers.get(CLIENT_AGENT_HEADER)))


def _client_id(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value if _CLIENT_ID.fullmatch(value) else None


def chain_hashes(messages: Sequence[Message]) -> list[str]:
    """One hash per turn, each covering the whole conversation up to and including it.

    `hashes[k]` equals another conversation's `hashes[k]` only if the two say the
    same through turn k, so a session can be found by matching the hash of the last
    turn it recorded against a prefix of an incoming request. Each turn is hashed
    from its role and its provider content (`raw_content`, falling back to the
    text), in canonical form: key order, a plain string against one text block,
    delivery hints such as `cache_control`, the client's instructions inside a turn,
    and the model's thinking do not count as an edit. A client moves its cache markers
    onto the newest turns with every request, Claude Code rewrites the instructions in
    earlier turns, and it removes earlier thinking when the conversation moves to
    another model, so a hash that included any of them saw one conversation as several.
    """
    hashes: list[str] = []
    previous = ""
    for message in messages:
        content = message.raw_content if message.raw_content is not None else message.content
        turn = json.dumps(
            [message.role, _as_said(content, message.role)], sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        previous = hashlib.sha256(f"{previous}\x00{turn}".encode("utf-8")).hexdigest()
        hashes.append(previous)
    return hashes


# Keys that tell the provider how to deliver a turn, not what it says.
_DELIVERY_HINTS = frozenset({"cache_control"})


def _as_said(content: object, role: str) -> object:
    """Content as said: a plain string as its one text block, without delivery hints, instructions or thinking."""
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    return _said(content, role)


def _said(value: object, role: str) -> object:
    if isinstance(value, dict):
        return {key: _said(item, role) for key, item in value.items() if key not in _DELIVERY_HINTS}
    if isinstance(value, list):
        # A block that held only instructions is gone, not an empty block.
        return [said for item in value if not is_thinking_block(item) and not _is_empty_text(said := _said(item, role))]
    if isinstance(value, str):
        return lift_instructions(value, role)[0]
    return value


def _is_empty_text(block: object) -> bool:
    return isinstance(block, dict) and block.get("type") == "text" and not block.get("text")
