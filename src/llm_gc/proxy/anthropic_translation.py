"""Reading and rewriting an inbound Anthropic Messages request.

Inbound only. `llm/anthropic_client.py` handles the outbound direction, for calls
the engine makes on its own behalf; merging the two would tie the engine to the
web layer.
"""

import logging
from typing import Any

from llm_gc.exceptions import ConfigError
from llm_gc.models import Message
from llm_gc.utils import count_tokens

logger = logging.getLogger(__name__)

_ROLES = {"user", "assistant"}

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
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ConfigError("request body must contain a non-empty 'messages' array")

    messages: list[Message] = []
    for index, raw in enumerate(raw_messages):
        if not isinstance(raw, dict):
            raise ConfigError(f"messages[{index}] is not an object")

        role = raw.get("role")
        if role not in _ROLES:
            raise ConfigError(f"messages[{index}] has unsupported role {role!r}")

        content = _flatten_content(raw.get("content"), index)
        messages.append(
            Message(
                role=role,
                content=content,
                token_count=count_tokens(content),
                turn_index=index,
            )
        )
    return messages


def _flatten_content(content: object, index: int) -> str:
    """Reduce Anthropic's content field to plain text for scoring.

    Non-text blocks (images, tool results) carry no text, so those turns are
    scored on partial content. The request itself is still forwarded intact.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = [
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
        ]
        skipped = len(content) - len(texts)
        if skipped:
            # Logged without the content: conversation text is user data.
            logger.debug("ignored non-text content blocks when scoring",
                         extra={"message_index": index, "blocks_ignored": skipped})
        return "\n".join(texts)

    raise ConfigError(f"messages[{index}] has unsupported content type {type(content).__name__}")


def rewrite_messages(payload: dict[str, Any], messages: list[Message]) -> dict[str, Any]:
    """Return a copy of the body with its conversation replaced.

    A copy, because the original is the fallback if anything downstream fails.
    Every other field passes through untouched, including ones this version has
    never heard of — the proxy changes the conversation and nothing else.
    """
    rewritten = dict(payload)
    rewritten["messages"] = [{"role": message.role, "content": message.content} for message in messages]
    return rewritten


def forwardable_headers(headers: dict[str, str]) -> dict[str, str]:
    """Strip headers describing our connection rather than the request.

    The client's API key passes through untouched and is never stored, logged, or
    copied into an event. This function never logs its input for that reason.
    """
    return {name: value for name, value in headers.items() if name.lower() not in _HOP_BY_HOP}
