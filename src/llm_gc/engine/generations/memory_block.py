"""Rendering retrieved facts into the one message that carries them."""

from llm_gc.models import KnowledgeEntry, Message
from llm_gc.models.knowledge_entry import KnowledgeType
from llm_gc.utils import count_tokens

MEMORY_BLOCK_HEADER = "[Memory from earlier in this conversation, recovered from turns no longer shown verbatim]"
"""Announced, not smuggled in: the model is told these are recovered facts rather
than turns that happened here."""

MEMORY_FACT_MAX_CHARS = 180
"""Per-fact cap, so one long entry cannot consume the whole block. Bites hardest
on `RAW` entries, which are whole turns."""

_TRUNCATION_MARKER = "..."


def format_memory_block(entries: list[KnowledgeEntry], max_tokens: int) -> Message | None:
    """Render ranked entries into one message, within a token budget.

    Args:
        entries: Ranked, most relevant first. Order matters: the budget is spent
            from the front, so the dropped entries are the least relevant.
        max_tokens: Ceiling for the whole block, header included.

    Returns:
        One `Message`, or None when there is nothing to inject or no room — so the
        caller appends nothing rather than a header with no facts under it.
    """
    if not entries or max_tokens <= 0:
        return None

    # Measured by rendering, not by summing per-line counts: the joining newlines
    # are tokens and a tokenizer's merges differ across a boundary, so summing
    # under-counts and the block would exceed the budget it promised.
    if count_tokens(MEMORY_BLOCK_HEADER) >= max_tokens:
        return None

    kept: list[KnowledgeEntry] = []
    content = ""
    for entry in entries:
        candidate = _render(kept + [entry])
        if count_tokens(candidate) > max_tokens:
            # Stop here rather than trying the next entry, which would reorder by
            # size and discard the retriever's ranking.
            break
        kept.append(entry)
        content = candidate

    if not kept:
        return None

    return Message(
        role="user",
        content=content,
        token_count=count_tokens(content),
        # Before every real turn; -1 rather than 0, which may be a live turn.
        turn_index=-1,
    )


def _render(entries: list[KnowledgeEntry]) -> str:
    """Render the block as it will be sent: relevance picks, chronology displays."""
    in_order = sorted(entries, key=lambda entry: entry.message_turn)
    body = "\n".join(_format_fact(entry) for entry in in_order)
    return f"{MEMORY_BLOCK_HEADER}\n{body}"


def _format_fact(entry: KnowledgeEntry) -> str:
    """One line per fact, carrying its origin turn.

    A `RAW` entry is labelled an excerpt, not a fact: extraction found nothing in
    that turn, so calling it a fact would overstate what is known.
    """
    content = entry.content
    if len(content) > MEMORY_FACT_MAX_CHARS:
        content = content[: MEMORY_FACT_MAX_CHARS - len(_TRUNCATION_MARKER)].rstrip() + _TRUNCATION_MARKER

    if entry.knowledge_type is KnowledgeType.RAW:
        return f"- (turn {entry.message_turn}) excerpt: {content}"
    return f"- (turn {entry.message_turn}) {entry.topic_label}: {content}"
