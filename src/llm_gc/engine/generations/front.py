"""The front: one message standing in for every turn a cut has removed.

It is rendered from the old generation (summaries and quoted turns) and the
permanent generation (facts). Both change only when a cut commits, and rendering
is deterministic, so the front's bytes are identical on every request between
cuts - which is what lets the provider cache it and keeps the model's thinking
blocks valid.
"""

from collections.abc import Sequence

from llm_gc.engine.generations.memory_block import format_memory_block
from llm_gc.models import KnowledgeEntry, Message
from llm_gc.utils import count_tokens

FRONT_HEADER = (
    "[Earlier in this conversation, condensed by LLM-GC. The turns before the ones that follow are "
    "summarised, quoted, or recorded as facts below rather than shown in full.]"
)
CONDENSED_HEADER = "[Earlier conversation, in order]"
QUOTE_PREFIX = "[Quoted exactly]"


def quote_turn(turn: Message) -> Message:
    """A turn kept word for word inside the front: scored too important to summarise."""
    content = f"{QUOTE_PREFIX} (turn {turn.turn_index}, {turn.role}) {turn.content}"
    return Message(role=turn.role, content=content, token_count=count_tokens(content), turn_index=turn.turn_index)


def render_front(old: Sequence[Message], facts: list[KnowledgeEntry], max_fact_tokens: int) -> Message | None:
    """The front for these generations, or None when nothing has been cut.

    A user-role message, because the provider requires the conversation to open
    with one; the turn index of -1 places it before every real turn.
    """
    sections: list[str] = []
    memory = format_memory_block(facts, max_fact_tokens)
    if memory is not None:
        sections.append(memory.content)
    if old:
        entries = sorted(old, key=lambda message: message.turn_index)
        sections.append(CONDENSED_HEADER + "\n" + "\n".join(entry.content for entry in entries))
    if not sections:
        return None

    content = FRONT_HEADER + "\n\n" + "\n\n".join(sections)
    return Message(role="user", content=content, token_count=count_tokens(content), turn_index=-1)
