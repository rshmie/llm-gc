"""The snapshot model: immutable values, and a delta that merges onto newer state."""

from dataclasses import FrozenInstanceError

import pytest
from hypothesis import given
from hypothesis import strategies as st

from llm_gc.models.generation_snapshot import (
    AgingDelta,
    DetailLevel,
    GenerationSnapshot,
    Span,
    apply_aging,
    file_knowledge,
)
from llm_gc.models import KnowledgeEntry, Message
from llm_gc.models.knowledge_entry import KnowledgeStatus, KnowledgeType


def _turn(index: int) -> Message:
    return Message(role="user", content=f"turn {index}", token_count=5, turn_index=index)


def _summary(first_turn: int, last_turn: int | None = None, content: str | None = None) -> Span:
    """A compacted span of `first_turn`..`last_turn` (two turns by default)."""
    last = first_turn + 1 if last_turn is None else last_turn
    message = Message(role="assistant", content=content or f"summary from {first_turn}", token_count=3,
                      turn_index=first_turn)
    return Span(turns=tuple(_turn(i) for i in range(first_turn, last + 1)), level=DetailLevel.COMPACTED,
                summary=message, summary_text=message.content)


def _archived(*indices: int) -> tuple[Span, ...]:
    return tuple(Span(turns=(_turn(index),), level=DetailLevel.ARCHIVED) for index in indices)


def _fact(topic: str, content: str, kind: KnowledgeType = KnowledgeType.DECISION) -> KnowledgeEntry:
    return KnowledgeEntry(message_turn=0, content=content, topic_label=topic, knowledge_type=kind)


def _young(*indices: int) -> GenerationSnapshot:
    snapshot = GenerationSnapshot()
    for index in indices:
        snapshot = snapshot.with_turn(_turn(index))
    return snapshot


class TestImmutability:
    def test_a_snapshot_cannot_be_changed_in_place(self):
        with pytest.raises(FrozenInstanceError):
            GenerationSnapshot().young = (_turn(0),)  # type: ignore[misc]

    def test_recording_a_turn_leaves_the_earlier_snapshot_untouched(self):
        """What a reader is holding must not change under it."""
        before = _young(0, 1)
        after = before.with_turn(_turn(2))
        assert [m.turn_index for m in before.young] == [0, 1]
        assert [m.turn_index for m in after.young] == [0, 1, 2]
        assert after.version == before.version + 1


class TestADeltaMergesOntoNewerState:
    def test_turns_recorded_during_the_pass_survive_the_commit(self):
        """The reason the model exists: aging computed against v7 commits onto v8."""
        base = _young(3, 4, 5, 10)
        delta = AgingDelta(departed_turns=frozenset({3, 4, 5}), new_spans=(_summary(3, 5),))
        current = base.with_turn(_turn(11))

        aged = apply_aging(current, delta).snapshot
        assert [m.turn_index for m in aged.young] == [10, 11]
        assert [span.first_turn for span in aged.old] == [3]
        assert aged.version == current.version + 1

    def test_a_level_change_and_a_new_span_in_one_pass(self):
        snapshot = GenerationSnapshot(young=(_turn(5), _turn(6), _turn(7)), old=(_summary(0), _summary(2)))
        delta = AgingDelta(departed_turns=frozenset({5, 6}), new_spans=(_summary(5),),
                           level_changes=((_summary(0), DetailLevel.ARCHIVED),))
        aged = apply_aging(snapshot, delta).snapshot
        assert [(span.first_turn, span.level) for span in aged.old] == [
            (0, DetailLevel.ARCHIVED), (2, DetailLevel.COMPACTED), (5, DetailLevel.COMPACTED)]

    def test_spans_sharing_a_first_turn_index_change_level_individually(self):
        """A caller that re-sends a turn index can leave two spans starting at it.
        Matching by index would change both."""
        first = _summary(4)
        second = _summary(4, content="a different summary")
        aged = apply_aging(GenerationSnapshot(old=(first, second)),
                           AgingDelta(level_changes=((first, DetailLevel.ARCHIVED),))).snapshot
        assert [span.level for span in aged.old] == [DetailLevel.ARCHIVED, DetailLevel.COMPACTED]

    @pytest.mark.parametrize("delta", [
        AgingDelta(departed_turns=frozenset({99}), new_spans=_archived(99)),
        AgingDelta(level_changes=((_summary(99), DetailLevel.ARCHIVED),)),
    ], ids=["unknown-turn", "unknown-span"])
    def test_a_delta_naming_content_that_is_gone_is_a_programmer_error(self, delta):
        """Only one pass ages a session at a time, so this means that rule broke.
        Merging anyway would silently drop or duplicate content."""
        with pytest.raises(ValueError):
            apply_aging(_young(0, 1), delta)

    @given(
        base_turns=st.sets(st.integers(0, 50), max_size=20),
        recorded_during_pass=st.sets(st.integers(51, 100), max_size=10),
        data=st.data(),
    )
    def test_no_turn_is_lost_except_the_ones_the_delta_names(self, base_turns, recorded_during_pass, data):
        departed = data.draw(st.sets(st.sampled_from(sorted(base_turns)))) if base_turns else set()
        base = _young(*sorted(base_turns))
        current = base
        for index in sorted(recorded_during_pass):
            current = current.with_turn(_turn(index))

        delta = AgingDelta(departed_turns=frozenset(departed), new_spans=_archived(*sorted(departed)))
        aged = apply_aging(current, delta).snapshot
        assert {m.turn_index for m in aged.young} == (base_turns | recorded_during_pass) - departed
        assert {span.first_turn for span in aged.old} == departed


class TestNothingRecordedIsDestroyed:
    """A turn leaves the young generation only into a span that holds its original."""

    def test_a_turn_that_departs_without_a_span_is_refused(self):
        with pytest.raises(ValueError, match="exactly one span"):
            apply_aging(_young(0, 1, 2), AgingDelta(departed_turns=frozenset({0, 1}), new_spans=_archived(0)))

    def test_a_turn_in_two_spans_is_refused(self):
        with pytest.raises(ValueError, match="exactly one span"):
            apply_aging(_young(0, 1), AgingDelta(departed_turns=frozenset({0}), new_spans=_archived(0, 0)))

    def test_archiving_a_span_keeps_its_originals_and_summary(self):
        """Archived is a level, not a deletion: the span can be shown in full again."""
        span = _summary(0)
        aged = apply_aging(GenerationSnapshot(old=(span,)),
                           AgingDelta(level_changes=((span, DetailLevel.ARCHIVED),))).snapshot
        assert aged.old[0].turns == span.turns
        assert aged.old[0].summary == span.summary


class TestASpanIsWellFormed:
    @pytest.mark.parametrize("indices", [(), (3, 5), (4, 3)], ids=["empty", "gap", "backwards"])
    def test_its_turns_are_consecutive(self, indices):
        with pytest.raises(ValueError, match="consecutive"):
            Span(turns=tuple(_turn(i) for i in indices), level=DetailLevel.KEPT)

    def test_a_compacted_span_has_a_summary(self):
        with pytest.raises(ValueError, match="no summary"):
            Span(turns=(_turn(0),), level=DetailLevel.COMPACTED)


class TestKnowledgeFiling:
    def test_a_later_decision_supersedes_an_earlier_one_on_its_topic(self):
        knowledge, supersessions = file_knowledge((_fact("database", "postgres"),), (_fact("database", "mysql"),))
        assert [(e.content, e.knowledge_status) for e in knowledge] == [
            ("postgres", KnowledgeStatus.SUPERSEDED), ("mysql", KnowledgeStatus.ACTIVE)]
        assert supersessions[0].superseded[0].content == "postgres"

    def test_supersession_applies_within_one_batch(self):
        knowledge, _ = file_knowledge((), (_fact("database", "postgres"), _fact("database", "mysql")))
        assert [e.content for e in knowledge if e.knowledge_status == KnowledgeStatus.ACTIVE] == ["mysql"]

    def test_superseded_entries_are_kept_but_not_active(self):
        """The record of a contradiction is part of what transparency reports."""
        snapshot = apply_aging(GenerationSnapshot(knowledge=(_fact("db", "postgres"),)),
                               AgingDelta(new_knowledge=(_fact("db", "mysql"),))).snapshot
        assert len(snapshot.knowledge) == 2
        assert [e.content for e in snapshot.active_knowledge] == ["mysql"]


class TestSupersessionRules:
    def test_supersession_does_not_touch_other_topics(self):
        knowledge, _ = file_knowledge(
            (_fact("database", "postgres"), _fact("auth", "jwt")), (_fact("database", "mysql"),))
        auth = [e for e in knowledge if e.topic_label == "auth"]
        assert [e.knowledge_status for e in auth] == [KnowledgeStatus.ACTIVE]

    def test_an_already_superseded_entry_is_not_superseded_again(self):
        knowledge, supersessions = file_knowledge(
            (), (_fact("database", "postgres"), _fact("database", "mysql"), _fact("database", "sqlite")))
        assert [e.knowledge_status for e in knowledge] == [
            KnowledgeStatus.SUPERSEDED, KnowledgeStatus.SUPERSEDED, KnowledgeStatus.ACTIVE]
        assert [len(s.superseded) for s in supersessions] == [1, 1], "each supersedes only the one active entry"

    def test_topic_less_entries_never_supersede_each_other(self):
        """"general" is a shared label too: every decision that names no topic gets it.
        Matched like a topic, "we decided to ship on Friday" retired an unrelated
        earlier decision, which the model then never saw again."""
        knowledge, supersessions = file_knowledge(
            (), (_fact("general", "use microservices"), _fact("general", "ship on Friday")))
        assert supersessions == ()
        assert all(e.knowledge_status == KnowledgeStatus.ACTIVE for e in knowledge)
