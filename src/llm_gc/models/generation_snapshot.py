"""One session's three generations, as immutable values.

A snapshot is never changed in place; every change produces a new one. A reader
that holds a snapshot therefore has a consistent view of the session for as long
as it likes, across any number of `await`s, with no lock. Writers build the next
snapshot and swap the session's reference in one synchronous step.
"""

from dataclasses import dataclass, field, replace

from llm_gc.models.knowledge_entry import KnowledgeEntry
from llm_gc.models.message import Message
from llm_gc.models.knowledge_entry import KnowledgeStatus, KnowledgeType


@dataclass(frozen=True)
class GenerationSnapshot:
    """The young, old and permanent generations of one session at one moment.

    `knowledge` holds every entry ever filed, superseded ones included, because
    the record of what was contradicted is part of what the transparency layer
    reports. `version` increases with every change, so two snapshots can be
    ordered and a future external store can commit with compare-and-set.
    """

    young: tuple[Message, ...] = ()
    old: tuple[Message, ...] = ()
    knowledge: tuple[KnowledgeEntry, ...] = ()
    version: int = 0

    @property
    def active_knowledge(self) -> list[KnowledgeEntry]:
        return [entry for entry in self.knowledge if entry.knowledge_status == KnowledgeStatus.ACTIVE]

    def with_turn(self, turn: Message) -> "GenerationSnapshot":
        return replace(self, young=self.young + (turn,), version=self.version + 1)


@dataclass(frozen=True)
class AgingDelta:
    """What one aging pass decided to change, computed against an earlier snapshot.

    Expressed as things to remove and values to add, never as replacement lists,
    so it can be applied to a snapshot that has gained turns since the pass began
    without discarding them. Young turns are named by turn index; summaries by
    value, since two summaries can share a first turn index.
    """

    departed_turns: frozenset[int] = frozenset()
    new_summaries: tuple[Message, ...] = ()
    evicted_summaries: tuple[Message, ...] = ()
    new_knowledge: tuple[KnowledgeEntry, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not (self.departed_turns or self.new_summaries or self.evicted_summaries or self.new_knowledge)


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
        ValueError: The delta names a turn or summary the snapshot does not hold.
            Only one aging pass runs per session, so nothing else removes them;
            reaching this means that invariant broke, and merging anyway would
            silently drop or duplicate content.
    """
    young_turns = {message.turn_index for message in snapshot.young}
    if missing := delta.departed_turns - young_turns:
        raise ValueError(f"aging delta removes turns not in the young generation: {sorted(missing)}")
    if stale := [s.turn_index for s in delta.evicted_summaries if s not in snapshot.old]:
        raise ValueError(f"aging delta evicts summaries not in the old generation: {sorted(stale)}")

    knowledge, supersessions = file_knowledge(snapshot.knowledge, delta.new_knowledge)
    aged = replace(
        snapshot,
        young=tuple(m for m in snapshot.young if m.turn_index not in delta.departed_turns),
        old=tuple(s for s in snapshot.old if s not in delta.evicted_summaries) + delta.new_summaries,
        knowledge=knowledge,
        version=snapshot.version + 1,
    )
    return AppliedAging(snapshot=aged, supersessions=supersessions)


def file_knowledge(
    knowledge: tuple[KnowledgeEntry, ...], new_entries: tuple[KnowledgeEntry, ...]
) -> tuple[tuple[KnowledgeEntry, ...], tuple[Supersession, ...]]:
    """File entries in order; each non-RAW entry supersedes active ones on its topic.

    Sequential, so an entry filed earlier in the same batch can be superseded by a
    later one: the conversation changed its mind, and the later statement wins.
    RAW entries supersede nothing, since a verbatim fallback states no claim.
    """
    filed = list(knowledge)
    supersessions: list[Supersession] = []
    for new_entry in new_entries:
        if new_entry.knowledge_type != KnowledgeType.RAW:
            superseded = []
            for i, existing in enumerate(filed):
                if existing.topic_label == new_entry.topic_label and existing.knowledge_status == KnowledgeStatus.ACTIVE:
                    filed[i] = replace(existing, knowledge_status=KnowledgeStatus.SUPERSEDED)
                    superseded.append(existing)
            if superseded:
                supersessions.append(Supersession(new_entry=new_entry, superseded=tuple(superseded)))
        filed.append(new_entry)
    return tuple(filed), tuple(supersessions)
