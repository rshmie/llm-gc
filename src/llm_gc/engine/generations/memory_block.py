"""Rendering retrieved facts into the one message that carries them.

This is where the generational model finally closes. A turn archived into
permanent generation has, until this point, been a deletion with a receipt: the
fact is stored and visible on the dashboard, and the model cannot use it. That is
the same information loss as the sliding window this project exists to beat,
reached more slowly — and worse, because the transparency layer reports the fact
as retained.
"""

from llm_gc.models import KnowledgeEntry, Message
from llm_gc.models.knowledge_entry import KnowledgeType
from llm_gc.utils import count_tokens

MEMORY_BLOCK_HEADER = "[Memory from earlier in this conversation, recovered from turns no longer shown verbatim]"
"""Announced, not smuggled in.

The model is told plainly that these are recovered facts rather than turns that
happened here, and a reader of the prompt can see the same thing. Presenting
distilled facts as if they were original conversation is the dishonesty this
project is built against — the compacted-summary marker exists for the identical
reason.
"""

MEMORY_FACT_MAX_CHARS = 180
"""Per-fact cap, so one long entry cannot consume the whole block.

It bites hardest on `RAW` entries, which are whole turns kept verbatim because
extraction found nothing in them. Truncation is marked with an ellipsis rather
than silent, so a clipped fact reads as clipped.
"""

_TRUNCATION_MARKER = "..."


def format_memory_block(entries: list[KnowledgeEntry], max_tokens: int) -> Message | None:
    """Render ranked entries into one message, within a token budget.

    Args:
        entries: Ranked, most relevant first. Order is load-bearing: the budget is
            spent from the front, so the entries that get dropped are the least
            relevant ones.
        max_tokens: Hard cap for the whole block, header included. The injected
            block is real tokens in the real prompt and counts against the context
            budget like any other message, so an uncapped block would grow with the
            session and undo the saving archiving produced.

    Returns:
        One `Message`, or `None` when there is nothing to inject or no room for it.
        `None` rather than an empty message, so the caller appends nothing instead
        of spending tokens on a header with no facts under it.
    """
    if not entries or max_tokens <= 0:
        return None

    # The budget is checked by rendering the candidate block and measuring it,
    # not by summing each line's tokens. Summing under-counts: the newlines that
    # join the lines are tokens of their own, and a tokenizer's merges differ
    # across a boundary from how they fall inside one line. The drift is small but
    # it runs *upward*, so the block would quietly exceed a budget it promised to
    # respect - the same "error in the reassuring direction" the token meter
    # already refuses to make. Re-rendering costs a handful of tokenizations of a
    # short string, bounded by how many facts were retrieved.
    if count_tokens(MEMORY_BLOCK_HEADER) >= max_tokens:
        # No room for the header plus even one fact. Emitting the header alone
        # would spend tokens announcing memory and then show none.
        return None

    kept: list[KnowledgeEntry] = []
    content = ""
    for entry in entries:
        candidate = _render(kept + [entry])
        if count_tokens(candidate) > max_tokens:
            # Stop at the first entry that does not fit rather than skipping it and
            # trying the next. Continuing would silently reorder by size, breaking
            # the relevance ordering the retriever established.
            break
        kept.append(entry)
        content = candidate

    if not kept:
        return None

    return Message(
        role="user",
        content=content,
        token_count=count_tokens(content),
        # Before every surviving turn, because these facts are the oldest
        # information in the context. A negative index keeps it first without
        # colliding with turn 0, which is a real turn that may still be present.
        turn_index=-1,
    )


def _render(entries: list[KnowledgeEntry]) -> str:
    """Render the block exactly as it will be sent.

    Selected by relevance, shown in conversation order. The two orderings do
    different jobs: relevance decides *what* survives the budget, chronology makes
    what survives readable - a fact from turn 2 before one from turn 40. Sorting a
    copy here rather than in place keeps the caller's ranked order intact, which
    the budget loop still depends on.
    """
    in_order = sorted(entries, key=lambda entry: entry.message_turn)
    body = "\n".join(_format_fact(entry) for entry in in_order)
    return f"{MEMORY_BLOCK_HEADER}\n{body}"


def _format_fact(entry: KnowledgeEntry) -> str:
    """One line per fact, carrying its origin turn.

    The turn number is included so a reader - human or model - can tell how far
    back the claim comes from, and so the dashboard and the prompt agree about
    provenance. A `RAW` entry is labelled as a kept excerpt rather than as a
    parsed fact, because that is what it is: extraction found nothing in that
    turn, and dressing it up as a fact would overstate what is known about it.
    """
    content = entry.content
    if len(content) > MEMORY_FACT_MAX_CHARS:
        content = content[: MEMORY_FACT_MAX_CHARS - len(_TRUNCATION_MARKER)].rstrip() + _TRUNCATION_MARKER

    if entry.knowledge_type is KnowledgeType.RAW:
        return f"- (turn {entry.message_turn}) excerpt: {content}"
    return f"- (turn {entry.message_turn}) {entry.topic_label}: {content}"
