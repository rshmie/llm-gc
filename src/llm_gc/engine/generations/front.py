"""The front: one message standing in for every turn a cut has removed.

It shows each span of the old generation at its detail level, in conversation
order. It changes only when a collection commits, and rendering is
deterministic, so the front's bytes are identical on every request in between -
which is what lets the provider cache it and keeps the model's thinking blocks
valid.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from itertools import groupby

from llm_gc.models import KnowledgeEntry, Message
from llm_gc.models.generation_snapshot import DetailLevel, Span
from llm_gc.models.knowledge_entry import GENERAL_TOPIC, KnowledgeStatus
from llm_gc.utils import count_tokens

FRONT_HEADER = (
    "[Earlier in this conversation, condensed by LLM-GC. The turns before the ones that follow are shown "
    "below with less detail: quoted exactly, summarised, archived to a line naming them and their facts, "
    "or folded out of view.]"
)
CONDENSED_HEADER = "[Earlier conversation, in order]"
QUOTE_PREFIX = "[Quoted exactly]"
ARCHIVED_PREFIX = "[Archived]"
FOLDED_PREFIX = "[Out of view]"
_SEPARATOR = " · "

FRONT_OVERHEAD_TOKENS = count_tokens(f"{FRONT_HEADER}\n\n{CONDENSED_HEADER}\n")
"""What the front costs before it shows any span: the share of the recap budget no span can use."""


@dataclass(frozen=True)
class Front:
    """The front as the model receives it, together with what it was rendered from.

    One value, so what a dashboard reports as "in the front" and the bytes the
    proxy forwards cannot disagree.
    """

    message: Message
    facts: tuple[KnowledgeEntry, ...]   # the facts it carries, on its archived spans' lines
    spans: tuple[Span, ...]             # the spans it shows word for word or summarised, in order


def span_message(span: Span) -> Message | None:
    """What the front carries for a kept or compacted span; None at the levels that need its facts.

    The message records the span's first turn: recency is positional, and a
    summary of turns 14-19 sits where turn 14 did.
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
    """What the front says for a span, without the markers it adds: for a person to read."""
    if span.level is DetailLevel.COMPACTED and span.summary is not None:
        return span.summary_text if span.summary_text is not None else span.summary.content
    return "\n".join(turn.content for turn in span.turns)


def archived_line(span: Span, facts: Sequence[KnowledgeEntry]) -> str:
    """An archived span's one line: which turns, what was asked, and the facts read from them."""
    heading = f"{ARCHIVED_PREFIX} ({_turns(span.first_turn, span.turns[-1].turn_index)}) {span.index_line}".rstrip()
    stated = [fact.content if fact.topic_label == GENERAL_TOPIC else f"{fact.topic_label}: {fact.content}"
              for fact in facts]
    return _SEPARATOR.join([heading, *stated])


def folded_line(first_turn: int, last_turn: int, spans: int) -> str:
    """The one line standing in for a run of spans folded out of view."""
    exchanges = "exchange" if spans == 1 else "exchanges"
    return (f"{FOLDED_PREFIX} ({_turns(first_turn, last_turn)}) {spans} earlier {exchanges}, "
            "folded to stay within the budget.")


def active_facts_by_turn(knowledge: Iterable[KnowledgeEntry]) -> dict[int, list[KnowledgeEntry]]:
    """The active facts, keyed by the turn each was read from, in the order they were filed."""
    by_turn: dict[int, list[KnowledgeEntry]] = {}
    for fact in knowledge:
        if fact.knowledge_status is KnowledgeStatus.ACTIVE:
            by_turn.setdefault(fact.message_turn, []).append(fact)
    return by_turn


def facts_of(span: Span, facts_by_turn: dict[int, list[KnowledgeEntry]]) -> list[KnowledgeEntry]:
    return [fact for turn in span.turn_indices for fact in facts_by_turn.get(turn, ())]


def level_costs(span: Span, facts: Sequence[KnowledgeEntry]) -> tuple[int, int | None, int]:
    """Tokens the front would spend on the span kept, compacted and archived; compacted is None without a summary."""
    quoted = span_message(replace(span, level=DetailLevel.KEPT))
    assert quoted is not None
    compacted = span.summary.token_count if span.summary is not None else None
    return quoted.token_count, compacted, count_tokens(archived_line(span, facts))


def render_front(old: Sequence[Span], knowledge: Sequence[KnowledgeEntry]) -> Front | None:
    """The front for these generations, or None when nothing has been cut.

    A user-role message, because the provider requires the conversation to open
    with one; the turn index of -1 places it before every real turn. Consecutive
    folded spans share one line.
    """
    facts_by_turn = active_facts_by_turn(knowledge)
    lines: list[str] = []
    shown: list[Span] = []
    carried: list[KnowledgeEntry] = []
    for folded, run in groupby(sorted(old, key=lambda span: span.first_turn),
                               key=lambda span: span.level is DetailLevel.OUT_OF_VIEW):
        spans = list(run)
        if folded:
            lines.append(folded_line(spans[0].first_turn, spans[-1].turns[-1].turn_index, len(spans)))
            continue
        for span in spans:
            message = span_message(span)
            if message is not None:
                lines.append(message.content)
                shown.append(span)
            else:
                facts = facts_of(span, facts_by_turn)
                lines.append(archived_line(span, facts))
                carried.extend(facts)
    if not lines:
        return None

    content = f"{FRONT_HEADER}\n\n{CONDENSED_HEADER}\n" + "\n".join(lines)
    message = Message(role="user", content=content, token_count=count_tokens(content), turn_index=-1)
    return Front(message=message, facts=tuple(carried), spans=tuple(shown))


def _turns(first: int, last: int) -> str:
    return f"turn {first}" if first == last else f"turns {first}–{last}"
