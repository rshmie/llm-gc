"""Rendering retrieved facts into the one message that carries them."""

from llm_gc.models import KnowledgeEntry
from llm_gc.models.knowledge_entry import KnowledgeType
from llm_gc.utils import count_tokens

MEMORY_BLOCK_HEADER = "[Memory from earlier in this conversation, recovered from turns no longer shown verbatim]"
"""Announced, not smuggled in: the model is told these are recovered facts rather
than turns that happened here."""

MEMORY_FACT_MAX_CHARS = 180
"""Per-fact cap, so one long entry cannot consume the whole block. Bites hardest
on `RAW` entries, which are whole turns."""

_TRUNCATION_MARKER = "..."


def fit_facts(entries: list[KnowledgeEntry], max_tokens: int) -> list[KnowledgeEntry]:
    """The leading entries whose rendered block fits within `max_tokens`."""
    if not entries or max_tokens <= 0:
        return []

    # Measured by rendering, not by summing per-line counts: the joining newlines
    # are tokens and a tokenizer's merges differ across a boundary, so summing
    # under-counts and the block would exceed the budget it promised.
    if count_tokens(MEMORY_BLOCK_HEADER) >= max_tokens:
        return []

    kept: list[KnowledgeEntry] = []
    for entry in entries:
        if count_tokens(render_memory_block(kept + [entry])) > max_tokens:
            # Stop here rather than trying the next entry, which would reorder by
            # size and discard the retriever's ranking.
            break
        kept.append(entry)
    return kept


def render_memory_block(entries: list[KnowledgeEntry]) -> str:
    """The block's text as it will be sent: relevance picks, chronology displays."""
    in_order = sorted(entries, key=lambda entry: entry.message_turn)
    body = "\n".join(format_fact(entry) for entry in in_order)
    return f"{MEMORY_BLOCK_HEADER}\n{body}"


def shorten_for_block(content: str) -> str:
    """An entry's content as the block carries it: cut to `MEMORY_FACT_MAX_CHARS`."""
    if len(content) <= MEMORY_FACT_MAX_CHARS:
        return content
    return content[: MEMORY_FACT_MAX_CHARS - len(_TRUNCATION_MARKER)].rstrip() + _TRUNCATION_MARKER


def format_fact(entry: KnowledgeEntry) -> str:
    """One line per fact, carrying its origin turn.

    A `RAW` entry is labelled an excerpt, not a fact: extraction found nothing in
    that turn, so calling it a fact would overstate what is known.
    """
    content = shorten_for_block(entry.content)

    if entry.knowledge_type is KnowledgeType.RAW:
        return f"- (turn {entry.message_turn}) excerpt: {content}"
    return f"- (turn {entry.message_turn}) {entry.topic_label}: {content}"
