"""Deciding which conversation a request belongs to.

A chat completions API is stateless, so nothing in a request says which remembered
conversation it continues. See doc/design/session/overview.md.
"""

import hashlib

from llm_gc.models import Message

SESSION_HEADER = "x-llm-gc-session"
"""Client-supplied session id. Preferred over any inference."""

_FINGERPRINT_PREFIX_TURNS = 2
"""Leading turns hashed to identify a conversation.

Two, not one: a shared opener ("You are a helpful assistant") would collide every
conversation in an app into one session, interleaving different users' turns. Two,
not many: a longer prefix stops matching once the engine rewrites history.
"""

_FINGERPRINT_LENGTH = 16


def resolve_session_id(messages: list[Message], header_value: str | None = None) -> str:
    """Return the session id for a request, from the header or by fingerprint.

    Args:
        messages: The conversation as the *client* sent it. Not our maintained
            version, which every GC pass rewrites.
        header_value: The client's own session id, if sent.
    """
    if header_value and header_value.strip():
        return header_value.strip()
    return fingerprint_session_id(messages)


def fingerprint_session_id(messages: list[Message]) -> str:
    """Derive a stable id by hashing the conversation's opening turns.

    Hashed rather than stored verbatim because a session id reaches logs and
    events, and conversation content is user data.

    Two known limitations: conversations that genuinely start identically share a
    session, and a conversation whose opening is edited becomes a new one (a cold
    start, not corruption).
    """
    if not messages:
        return "empty"

    prefix = messages[:_FINGERPRINT_PREFIX_TURNS]
    # Roles included, and NUL-separated so parts cannot run together ambiguously.
    material = "\n\x00".join(f"{message.role}\x00{message.content}" for message in prefix)
    return f"fp_{hashlib.sha256(material.encode('utf-8')).hexdigest()[:_FINGERPRINT_LENGTH]}"
