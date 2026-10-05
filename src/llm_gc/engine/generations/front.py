"""The front: one message standing in for every turn a cut has removed.

It is rendered from the old generation (each span at its detail level) and the
permanent generation (facts). Both change only when a cut commits, and rendering
is deterministic, so the front's bytes are identical on every request between
cuts - which is what lets the provider cache it and keeps the model's thinking
blocks valid.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from llm_gc.engine.generations.memory_block import fit_facts, render_memory_block
from llm_gc.models import KnowledgeEntry, Message
from llm_gc.models.generation_snapshot import DetailLevel, Span
from llm_gc.utils import count_tokens

FRONT_HEADER = (
    "[Earlier in this conversation, condensed by LLM-GC. The turns before the ones that follow are "
    "summarised, quoted, or recorded as facts below rather than shown in full.]"
)
CONDENSED_HEADER = "[Earlier conversation, in order]"
QUOTE_PREFIX = "[Quoted exactly]"


@dataclass(frozen=True)
class Front:
    """The front as the model receives it, together with what it was rendered from.

    One value, so what a dashboard reports as "in the front" and the bytes the
    proxy forwards cannot disagree.
    """

    message: Message
    facts: tuple[KnowledgeEntry, ...]   # the facts that fit the budget, most important first
    spans: tuple[Span, ...]             # the spans it carries text for, in conversation order


def span_message(span: Span) -> Message | None:
    """What the front carries for one span, or None when it carries only the span's facts.

    The message is scored and measured as part of the front, so it records the
    span's first turn: recency is positional, and a summary of turns 14-19 sits
    where turn 14 did.
    """
    if span.level is DetailLevel.COMPACTED:
        return span.summary
    if span.level is DetailLevel.KEPT:
        content = "\n".join(f"{QUOTE_PREFIX} (turn {turn.turn_index}, {turn.role}) {turn.content}"
                            for turn in span.turns)
        return Message(role=span.turns[0].role, content=content, token_count=count_tokens(content),
                       turn_index=span.first_turn)
    return None


def stated_text(span: Span) -> str:
    """What the front says for a span, without the markers it adds: for a person, or for extraction."""
    if span.level is DetailLevel.COMPACTED and span.summary is not None:
        return span.summary_text if span.summary_text is not None else span.summary.content
    return "\n".join(turn.content for turn in span.turns)


def render_front(old: Sequence[Span], facts: list[KnowledgeEntry], max_fact_tokens: int) -> Front | None:
    """The front for these generations, or None when nothing has been cut.

    A user-role message, because the provider requires the conversation to open
    with one; the turn index of -1 places it before every real turn.
    """
    sections: list[str] = []
    kept_facts = fit_facts(facts, max_fact_tokens)
    if kept_facts:
        sections.append(render_memory_block(kept_facts))
    shown = [(span, message) for span in sorted(old, key=lambda span: span.first_turn)
             if (message := span_message(span)) is not None]
    if shown:
        sections.append(CONDENSED_HEADER + "\n" + "\n".join(message.content for _, message in shown))
    if not sections:
        return None

    content = FRONT_HEADER + "\n\n" + "\n\n".join(sections)
    message = Message(role="user", content=content, token_count=count_tokens(content), turn_index=-1)
    return Front(message=message, facts=tuple(kept_facts), spans=tuple(span for span, _ in shown))
