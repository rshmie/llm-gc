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

SESSION_HEADER = "x-llm-gc-session"
"""Client-supplied session id. Preferred over matching by content."""


def chain_hashes(messages: Sequence[Message]) -> list[str]:
    """One hash per turn, each covering the whole conversation up to and including it.

    `hashes[k]` equals another conversation's `hashes[k]` only if the two are
    identical through turn k, so a session can be found by matching the hash of
    the last turn it recorded against a prefix of an incoming request. Each turn is
    hashed from its role and its provider content as sent (`raw_content`, falling
    back to the text), in canonical JSON so that key order does not count as an
    edit.
    """
    hashes: list[str] = []
    previous = ""
    for message in messages:
        content = message.raw_content if message.raw_content is not None else message.content
        turn = json.dumps([message.role, content], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        previous = hashlib.sha256(f"{previous}\x00{turn}".encode("utf-8")).hexdigest()
        hashes.append(previous)
    return hashes
