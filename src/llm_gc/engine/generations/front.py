"""The front: one message standing in for every turn a cut has removed.

It shows each span of the old generation at its detail level, in conversation
order. Levels are nested - each shows everything the level below it shows - so
showing a span with more detail never takes anything away. The front changes only
when a collection commits, and rendering is deterministic, so its bytes are
identical on every request in between, which is what lets the provider cache it
and keeps the model's thinking blocks valid.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from itertools import groupby

from llm_gc.models import KnowledgeEntry, Message
from llm_gc.models.generation_snapshot import DetailLevel, Span
from llm_gc.models.knowledge_entry import GENERAL_TOPIC, KnowledgeStatus
from llm_gc.utils import count_tokens

QUOTE_PREFIX = "[Quoted exactly]"
SUMMARISED_PREFIX = "[Summarised]"
ARCHIVED_PREFIX = "[Archived]"
FOLDED_PREFIX = "[Out of view]"

FRONT_HEADER = (
    "[Earlier turns, condensed by LLM-GC: quoted exactly, summarised, archived to an opening line and its facts, "
    "or out of view. If a detail you need is not shown, say so rather than guess.]"
)
CONDENSED_HEADER = "[Earlier conversation, in order]"
_SEPARATOR = " · "

FRONT_OVERHEAD_TOKENS = count_tokens(f"{FRONT_HEADER}\n\n{CONDENSED_HEADER}\n")
"""What the front costs before it shows any span: the share of the recap budget no span can use."""


@dataclass(frozen=True)
class Annotations:
    """What the front adds to a span beyond its own words.

    `facts` are the active facts read from its turns; `replaced` pairs each of its
    facts the conversation later replaced with the fact that replaced it.
    """

    facts: tuple[KnowledgeEntry, ...] = ()
    replaced: tuple[tuple[KnowledgeEntry, KnowledgeEntry], ...] = ()


class KnowledgeIndex:
    """The permanent generation arranged for rendering: each span's annotations by its turns.

    Built once per front, so rendering every span reads the knowledge once.
    """

    def __init__(self, knowledge: Sequence[KnowledgeEntry]) -> None:
        self._active: dict[int, list[KnowledgeEntry]] = {}
        self._replaced: dict[int, list[tuple[KnowledgeEntry, KnowledgeEntry]]] = {}
        for position, fact in enumerate(knowledge):
            if fact.knowledge_status is KnowledgeStatus.ACTIVE:
                self._active.setdefault(fact.message_turn, []).append(fact)
            elif (replacement := _replacement(fact, knowledge[position + 1:])) is not None:
                self._replaced.setdefault(fact.message_turn, []).append((fact, replacement))

    def for_span(self, span: Span) -> Annotations:
        return Annotations(facts=tuple(fact for turn in span.turn_indices for fact in self._active.get(turn, ())),
                           replaced=tuple(pair for turn in span.turn_indices for pair in self._replaced.get(turn, ())))


@dataclass(frozen=True)
class Front:
    """The front as the model receives it, together with what it was rendered from.

    One value, so what a dashboard reports as "in the front" and the bytes the
    proxy forwards cannot disagree.
    """

    message: Message
    entries: tuple[tuple[Span, str], ...]   # every span with lines of its own, and exactly those lines
    facts: tuple[KnowledgeEntry, ...]       # the facts those lines carry


def span_text(span: Span, annotations: Annotations) -> str:
    """What the front carries for a span at its level; a folded span shares a line with its run.

    Raises:
        ValueError: The span is folded, or compacted without a summary.
    """
    if span.level is DetailLevel.OUT_OF_VIEW:
        raise ValueError("a folded span has no line of its own")
    if span.level is DetailLevel.KEPT:
        lines = [f"{QUOTE_PREFIX} (turn {turn.turn_index}, {turn.role}) {turn.content}" for turn in span.turns]
    elif span.level is DetailLevel.COMPACTED:
        if span.summary is None:
            raise ValueError(f"span at turn {span.first_turn} is compacted but has no summary")
        summary = span.summary_text if span.summary_text is not None else span.summary.content
        lines = [_heading(SUMMARISED_PREFIX, span, annotations.facts), summary]
    else:
        lines = [_heading(ARCHIVED_PREFIX, span, annotations.facts)]
    lines += [_replaced_note(old, new) for old, new in annotations.replaced]
    return "\n".join(lines)


def stated_text(span: Span) -> str:
    """What a kept or compacted span says, without the front's markers: for a person to read."""
    if span.level is DetailLevel.COMPACTED and span.summary is not None:
        return span.summary_text if span.summary_text is not None else span.summary.content
    return "\n".join(turn.content for turn in span.turns)


def folded_line(first_turn: int, last_turn: int, spans: int) -> str:
    """The one line standing in for a run of spans folded out of view."""
    exchanges = "exchange" if spans == 1 else "exchanges"
    return (f"{FOLDED_PREFIX} ({_turns(first_turn, last_turn)}) {spans} earlier {exchanges}, "
            "folded to stay within the budget.")


def level_costs(span: Span, annotations: Annotations) -> tuple[int, int | None, int]:
    """Tokens the front would spend on the span kept, compacted and archived; compacted is None without a summary."""
    def cost(level: DetailLevel) -> int:
        return count_tokens(span_text(replace(span, level=level), annotations))

    compacted = cost(DetailLevel.COMPACTED) if span.summary is not None else None
    return cost(DetailLevel.KEPT), compacted, cost(DetailLevel.ARCHIVED)


def render_front(old: Sequence[Span], knowledge: Sequence[KnowledgeEntry]) -> Front | None:
    """The front for these generations, or None when nothing has been cut.

    A user-role message, because the provider requires the conversation to open
    with one; the turn index of -1 places it before every real turn. Consecutive
    folded spans share one line.
    """
    index = KnowledgeIndex(knowledge)
    lines: list[str] = []
    entries: list[tuple[Span, str]] = []
    carried: list[KnowledgeEntry] = []
    for folded, run in groupby(sorted(old, key=lambda span: span.first_turn),
                               key=lambda span: span.level is DetailLevel.OUT_OF_VIEW):
        spans = list(run)
        if folded:
            lines.append(folded_line(spans[0].first_turn, spans[-1].turns[-1].turn_index, len(spans)))
            continue
        for span in spans:
            annotations = index.for_span(span)
            text = span_text(span, annotations)
            lines.append(text)
            entries.append((span, text))
            if span.level is not DetailLevel.KEPT:
                carried.extend(annotations.facts)
    if not lines:
        return None

    content = f"{FRONT_HEADER}\n\n{CONDENSED_HEADER}\n" + "\n".join(lines)
    message = Message(role="user", content=content, token_count=count_tokens(content), turn_index=-1)
    return Front(message=message, entries=tuple(entries), facts=tuple(carried))


def _heading(prefix: str, span: Span, facts: Iterable[KnowledgeEntry]) -> str:
    """A span's first line: which turns, what was said first, and the facts read from them."""
    heading = f"{prefix} ({_turns(span.first_turn, span.turns[-1].turn_index)}) {span.index_line}".rstrip()
    return _SEPARATOR.join([heading, *(_stated(fact) for fact in facts)])


def _replaced_note(old: KnowledgeEntry, new: KnowledgeEntry) -> str:
    return f"↳ Replaced later (turn {new.message_turn}): {_stated(new)}, not {old.content}."


def _stated(fact: KnowledgeEntry) -> str:
    return fact.content if fact.topic_label == GENERAL_TOPIC else f"{fact.topic_label}: {fact.content}"


def _replacement(fact: KnowledgeEntry, later: Sequence[KnowledgeEntry]) -> KnowledgeEntry | None:
    """The fact that superseded `fact`: the next one filed on its topic that could."""
    return next((entry for entry in later if entry.topic_label == fact.topic_label and entry.can_supersede), None)


def _turns(first: int, last: int) -> str:
    return f"turn {first}" if first == last else f"turns {first}–{last}"
