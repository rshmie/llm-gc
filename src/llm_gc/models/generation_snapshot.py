"""One session's three generations, as immutable values.

The old generation keeps every original it was given: a turn leaves the young
generation only into a span, and a span is never removed, only shown at less
detail.

A snapshot is never changed in place; every change produces a new one. A reader
that holds a snapshot therefore has a consistent view of the session for as long
as it likes, across any number of `await`s, with no lock. Writers build the next
snapshot and swap the session's reference in one synchronous step.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum

from llm_gc.models.aging_degradation import AgingDegradation
from llm_gc.models.knowledge_entry import KnowledgeEntry
from llm_gc.models.message import Message
from llm_gc.models.knowledge_entry import KnowledgeStatus


class DetailLevel(Enum):
    """How much of a span the front carries, from most to least."""

    KEPT = "kept"                # the turns, word for word
    COMPACTED = "compacted"      # the span's summary
    ARCHIVED = "archived"        # its opening line and the facts read from it
    OUT_OF_VIEW = "out_of_view"  # folded with its neighbours into one line naming the turns


@dataclass(frozen=True)
class Span:
    """Turns condensed together, holding their originals, and how much of them the front carries.

    The originals are what make a level reversible: a span the front carries as
    facts can be carried word for word again, from the turns themselves rather
    than from anything derived from them.

    `summary` is the compactor's message, marker included, as the front carries
    it; `summary_text` is what it says without the marker. `index_line` names what
    the span is about, for the archived level. `facts_unread` marks a span whose
    facts could not be read when it was made: it has none on record, though its
    turns may state some.

    Raises:
        ValueError: The turns are empty or not consecutive, or the span is
            COMPACTED without a summary. Each is a bug in whatever built it.
    """

    turns: tuple[Message, ...]
    level: DetailLevel
    summary: Message | None = None
    summary_text: str | None = None
    index_line: str = ""
    facts_unread: bool = False

    def __post_init__(self) -> None:
        indices = [turn.turn_index for turn in self.turns]
        if not indices or indices != list(range(indices[0], indices[0] + len(indices))):
            raise ValueError(f"a span needs consecutive turns, got {indices}")
        if self.level is DetailLevel.COMPACTED and self.summary is None:
            raise ValueError(f"span at turn {indices[0]} is compacted but has no summary")

    @property
    def first_turn(self) -> int:
        return self.turns[0].turn_index

    @property
    def turn_indices(self) -> tuple[int, ...]:
        return tuple(turn.turn_index for turn in self.turns)


@dataclass(frozen=True)
class GenerationSnapshot:
    """The young, old and permanent generations of one session at one moment.

    `knowledge` holds every entry ever filed, superseded ones included, because
    the record of what was contradicted is part of what the transparency layer
    reports. `version` increases with every change, so two snapshots can be
    ordered and a future external store can commit with compare-and-set.

    `lineage` holds one chain hash per recorded turn, by position, including turns
    that have since been aged out of `young`: it is the record of which
    conversation this is. Callers that do not identify sessions by content leave
    it empty.
    """

    young: tuple[Message, ...] = ()
    old: tuple[Span, ...] = ()
    knowledge: tuple[KnowledgeEntry, ...] = ()
    version: int = 0
    lineage: tuple[str, ...] = ()
    # The first turn whose thinking blocks were produced against the current front.
    # None from a cut until the first request that carries its front, because until
    # then nothing has been produced against it.
    thinking_valid_from: int | None = 0
    cut_count: int = 0
    # A cut a pass prepared and the request path has not switched in yet. Its turns are
    # still young, and are sent word for word until it is.
    staged: "StagedCut | None" = None

    @property
    def active_knowledge(self) -> list[KnowledgeEntry]:
        return [entry for entry in self.knowledge if entry.knowledge_status == KnowledgeStatus.ACTIVE]

    @property
    def recorded_turn_count(self) -> int:
        """How many turns the session has recorded, aged ones included."""
        if self.lineage:
            return len(self.lineage)
        seen = [message.turn_index for message in self.young]
        seen += [turn for span in self.old for turn in span.turn_indices]
        seen += [entry.message_turn for entry in self.knowledge]
        return max(seen, default=-1) + 1

    @property
    def thinking_sent_from(self) -> int:
        """The first turn whose thinking is still sent with it; earlier turns are sent without theirs.

        From a cut until its front is first sent, this is past every turn recorded so far.
        """
        return self.thinking_valid_from if self.thinking_valid_from is not None else self.recorded_turn_count

    def sent_thinking_tokens(self, turns: Iterable[Message] | None = None) -> int:
        """Estimated thinking tokens still sent with the young turns, or with the given turns."""
        start = self.thinking_sent_from
        return sum(turn.thinking_tokens for turn in (self.young if turns is None else turns) if turn.turn_index >= start)

    @property
    def tip(self) -> str | None:
        """The chain hash of the last recorded turn, or None before any."""
        return self.lineage[-1] if self.lineage else None

    def with_turn(self, turn: Message) -> "GenerationSnapshot":
        return replace(self, young=self.young + (turn,), version=self.version + 1)

    def with_turns(self, turns: Sequence[Message], hashes: Sequence[str]) -> "GenerationSnapshot":
        """Append turns together with their chain hashes, one for one."""
        if len(turns) != len(hashes):
            raise ValueError(f"{len(turns)} turns but {len(hashes)} lineage hashes")
        return replace(self, young=self.young + tuple(turns), lineage=self.lineage + tuple(hashes),
                       version=self.version + 1)

    def rewound_to(self, turn_count: int) -> "GenerationSnapshot | None":
        """The snapshot as it was with only the first `turn_count` turns recorded.

        None when any turn being discarded has already been aged: its summary or
        facts may describe the abandoned branch and cannot be separated from what
        came before, so the caller must start the session over instead.
        """
        young_indices = {message.turn_index for message in self.young}
        if any(index not in young_indices for index in range(turn_count, len(self.lineage))):
            return None
        # A staged cut that holds a discarded turn describes the abandoned branch too.
        staged = self.staged
        if staged is not None and any(index >= turn_count for index in staged.delta.departed_turns):
            staged = None
        # Turns recorded after the rewind are answered with the current front, so their thinking is valid.
        # None stays None: the front has not been sent yet, and its first request sets the turn.
        valid_from = self.thinking_valid_from
        return replace(
            self,
            young=tuple(m for m in self.young if m.turn_index < turn_count),
            lineage=self.lineage[:turn_count],
            version=self.version + 1,
            staged=staged,
            thinking_valid_from=min(valid_from, turn_count) if valid_from is not None else None,
        )


@dataclass(frozen=True)
class AgingDelta:
    """What one aging pass decided to change, computed against an earlier snapshot.

    Expressed as changes, never as replacement lists, so it can be applied to a
    snapshot that has gained turns since the pass began without discarding them.
    Young turns are named by turn index; spans by value, since a caller that
    re-sends a turn index can leave two spans starting at the same turn.
    """

    departed_turns: frozenset[int] = frozenset()
    new_spans: tuple[Span, ...] = ()
    level_changes: tuple[tuple[Span, DetailLevel], ...] = ()
    new_knowledge: tuple[KnowledgeEntry, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not (self.departed_turns or self.new_spans or self.level_changes or self.new_knowledge)


@dataclass(frozen=True)
class StagedCut:
    """A prepared cut waiting to be switched in: the change, and the measures that decide when.

    It holds conversation content - spans, summaries, facts - so it lives in the snapshot,
    the one owner of a session's conversation. Nothing the model is sent changes until it
    is switched in.

    `saving_tokens` is the text its cut removes from the prompt. `stale_tokens` is the
    text of the turns it condenses: what the model carries that has gone quiet. Both
    leave out thinking, which keeps changing until the switch and is measured then.
    `front_tokens` is the front it would render; with the tail at the moment of the switch,
    it is what the provider would have to cache again. `degradations` is what preparing it
    fell short of - a refused summary, facts not read - so the collection that takes effect
    reports them, not only the pass that met them.
    """

    delta: AgingDelta
    saving_tokens: int
    stale_tokens: int
    front_tokens: int
    degradations: tuple[AgingDegradation, ...] = ()


@dataclass(frozen=True)
class Supersession:
    """A newly filed entry and the active entries it contradicted."""

    new_entry: KnowledgeEntry
    superseded: tuple[KnowledgeEntry, ...]


@dataclass(frozen=True)
class AppliedAging:
    snapshot: GenerationSnapshot
    supersessions: tuple[Supersession, ...] = field(default=())


def apply_aging(snapshot: GenerationSnapshot, delta: AgingDelta) -> AppliedAging:
    """Return the snapshot with `delta` applied, and the supersessions it caused.

    Raises:
        ValueError: The delta names a turn or span the snapshot does not hold, or
            a departing turn does not land in exactly one new span. Only one aging
            pass runs per session, so nothing else removes them; reaching this
            means an invariant broke, and merging anyway would silently drop or
            duplicate content.
    """
    young_turns = {message.turn_index for message in snapshot.young}
    if missing := delta.departed_turns - young_turns:
        raise ValueError(f"aging delta removes turns not in the young generation: {sorted(missing)}")
    landed = [turn for span in delta.new_spans for turn in span.turn_indices]
    if sorted(landed) != sorted(delta.departed_turns):
        raise ValueError(f"turns {sorted(delta.departed_turns)} depart but new spans hold {sorted(landed)}: "
                         "every departing turn must land in exactly one span")
    old = list(snapshot.old)
    for span, level in delta.level_changes:
        if span not in old:
            raise ValueError(f"aging delta changes a span not in the old generation: turn {span.first_turn}")
        old[old.index(span)] = replace(span, level=level)

    knowledge, supersessions = file_knowledge(snapshot.knowledge, delta.new_knowledge)
    aged = replace(
        snapshot,
        young=tuple(m for m in snapshot.young if m.turn_index not in delta.departed_turns),
        old=tuple(old) + delta.new_spans,
        knowledge=knowledge,
        version=snapshot.version + 1,
    )
    return AppliedAging(snapshot=aged, supersessions=supersessions)


def file_knowledge(
    knowledge: tuple[KnowledgeEntry, ...], new_entries: tuple[KnowledgeEntry, ...]
) -> tuple[tuple[KnowledgeEntry, ...], tuple[Supersession, ...]]:
    """File entries in order; each entry that can supersede retires active ones on its topic.

    Sequential, so an entry filed earlier in the same batch can be superseded by a
    later one: the conversation changed its mind, and the later statement wins.
    Topic-less entries supersede nothing - see `KnowledgeEntry.can_supersede`. An
    entry that only restates an active one - same topic, same value - is not filed:
    nothing changed, and filing it would tell the model the value was replaced by itself.
    A value the assistant states on a topic the user decided is not filed: only the
    user changes their own decision, and a labelled recommendation beside an
    unlabelled decision reads as the answer. The recommendation stays in its
    exchange's words and summary.
    """
    filed = list(knowledge)
    supersessions: list[Supersession] = []
    for new_entry in new_entries:
        if any(_restates(new_entry, existing) or _contradicts_users_decision(new_entry, existing) for existing in filed):
            continue
        if new_entry.can_supersede:
            superseded = []
            for i, existing in enumerate(filed):
                if existing.topic_label == new_entry.topic_label and existing.knowledge_status == KnowledgeStatus.ACTIVE:
                    filed[i] = replace(existing, knowledge_status=KnowledgeStatus.SUPERSEDED)
                    superseded.append(existing)
            if superseded:
                supersessions.append(Supersession(new_entry=new_entry, superseded=tuple(superseded)))
        filed.append(new_entry)
    return tuple(filed), tuple(supersessions)


def _restates(new: KnowledgeEntry, existing: KnowledgeEntry) -> bool:
    return (existing.knowledge_status is KnowledgeStatus.ACTIVE and existing.topic_label == new.topic_label
            and _same_value(existing.content, new.content))


def _contradicts_users_decision(new: KnowledgeEntry, existing: KnowledgeEntry) -> bool:
    return (new.speaker == "assistant" and existing.speaker == "user" and new.can_supersede
            and existing.knowledge_status is KnowledgeStatus.ACTIVE and existing.topic_label == new.topic_label)


def _same_value(a: str, b: str) -> bool:
    """Equal once case, spacing and closing punctuation are set aside: "MySQL 8." restates "mysql 8"."""
    def plain(value: str) -> str:
        return " ".join(value.split()).rstrip(".!;,").casefold()
    return plain(a) == plain(b)
