"""The front: one message standing in for every turn a cut has removed.

It is rendered from the old generation (summaries and quoted turns) and the
permanent generation (facts). Both change only when a cut commits, and rendering
is deterministic, so the front's bytes are identical on every request between
cuts - which is what lets the provider cache it and keeps the model's thinking
blocks valid.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from llm_gc.engine.generations.memory_block import fit_facts, render_memory_block
from llm_gc.models import KnowledgeEntry, Message
from llm_gc.models.generation_snapshot import EntryKind, FrontEntry
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
    entries: tuple[FrontEntry, ...]     # summaries and quotes, in conversation order


def quote_turn(turn: Message) -> FrontEntry:
    """A turn kept word for word inside the front: scored too important to summarise."""
    content = f"{QUOTE_PREFIX} (turn {turn.turn_index}, {turn.role}) {turn.content}"
    message = Message(role=turn.role, content=content, token_count=count_tokens(content), turn_index=turn.turn_index)
    return FrontEntry(message=message, kind=EntryKind.QUOTE, source_turns=(turn.turn_index,), text=turn.content)


def render_front(old: Sequence[FrontEntry], facts: list[KnowledgeEntry], max_fact_tokens: int) -> Front | None:
    """The front for these generations, or None when nothing has been cut.

    A user-role message, because the provider requires the conversation to open
    with one; the turn index of -1 places it before every real turn.
    """
    sections: list[str] = []
    kept_facts = fit_facts(facts, max_fact_tokens)
    if kept_facts:
        sections.append(render_memory_block(kept_facts))
    entries = tuple(sorted(old, key=lambda entry: entry.message.turn_index))
    if entries:
        sections.append(CONDENSED_HEADER + "\n" + "\n".join(entry.message.content for entry in entries))
    if not sections:
        return None

    content = FRONT_HEADER + "\n\n" + "\n\n".join(sections)
    message = Message(role="user", content=content, token_count=count_tokens(content), turn_index=-1)
    return Front(message=message, facts=tuple(kept_facts), entries=entries)
