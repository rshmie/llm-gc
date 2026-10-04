"""One session's three generations, as immutable values.

A snapshot is never changed in place; every change produces a new one. A reader
that holds a snapshot therefore has a consistent view of the session for as long
as it likes, across any number of `await`s, with no lock. Writers build the next
snapshot and swap the session's reference in one synchronous step.
"""

from collections.abc import Sequence
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

    `lineage` holds one chain hash per recorded turn, by position, including turns
    that have since been aged out of `young`: it is the record of which
    conversation this is. Callers that do not identify sessions by content leave
    it empty.
    """

    young: tuple[Message, ...] = ()
    old: tuple[Message, ...] = ()
    knowledge: tuple[KnowledgeEntry, ...] = ()
    version: int = 0
    lineage: tuple[str, ...] = ()
    # Turns recorded when the most recent cut committed. Thinking blocks in kept
    # turns before this point were produced against the history the cut replaced.
    cut_turn_count: int = 0

    @property
    def active_knowledge(self) -> list[KnowledgeEntry]:
        return [entry for entry in self.knowledge if entry.knowledge_status == KnowledgeStatus.ACTIVE]

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
        return replace(
            self,
            young=tuple(m for m in self.young if m.turn_index < turn_count),
            lineage=self.lineage[:turn_count],
            version=self.version + 1,
        )


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
