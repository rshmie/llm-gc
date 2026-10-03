"""The snapshot model: immutable values, and a delta that merges onto newer state."""

from dataclasses import FrozenInstanceError

import pytest
from hypothesis import given
from hypothesis import strategies as st

from llm_gc.models.generation_snapshot import AgingDelta, GenerationSnapshot, apply_aging, file_knowledge
from llm_gc.models import KnowledgeEntry, Message
from llm_gc.models.knowledge_entry import KnowledgeStatus, KnowledgeType


def _turn(index: int) -> Message:
    return Message(role="user", content=f"turn {index}", token_count=5, turn_index=index)


def _summary(first_turn: int) -> Message:
    return Message(role="assistant", content=f"summary from {first_turn}", token_count=3, turn_index=first_turn)


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
        delta = AgingDelta(departed_turns=frozenset({3, 4, 5}), new_summaries=(_summary(3),))
        current = base.with_turn(_turn(11))

        aged = apply_aging(current, delta).snapshot
        assert [m.turn_index for m in aged.young] == [10, 11]
        assert [s.turn_index for s in aged.old] == [3]
        assert aged.version == current.version + 1

    def test_eviction_and_promotion_in_one_pass(self):
        snapshot = GenerationSnapshot(young=(_turn(5), _turn(6)), old=(_summary(0), _summary(2)))
        delta = AgingDelta(departed_turns=frozenset({5}), new_summaries=(_summary(5),),
                           evicted_summaries=(_summary(0),))
        aged = apply_aging(snapshot, delta).snapshot
        assert [s.turn_index for s in aged.old] == [2, 5]

    def test_summaries_sharing_a_first_turn_index_are_evicted_individually(self):
        """A regenerated turn re-sends its index, so two summaries can share one.
        Evicting by index would remove both."""
        first = _summary(4)
        second = Message(role="assistant", content="a different summary", token_count=3, turn_index=4)
        aged = apply_aging(GenerationSnapshot(old=(first, second)), AgingDelta(evicted_summaries=(first,))).snapshot
        assert aged.old == (second,)

    @pytest.mark.parametrize("delta", [
        AgingDelta(departed_turns=frozenset({99})),
        AgingDelta(evicted_summaries=(_summary(99),)),
    ], ids=["unknown-turn", "unknown-summary"])
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

        aged = apply_aging(current, AgingDelta(departed_turns=frozenset(departed))).snapshot
        assert {m.turn_index for m in aged.young} == (base_turns | recorded_during_pass) - departed


class TestKnowledgeFiling:
    def test_a_later_decision_supersedes_an_earlier_one_on_its_topic(self):
        knowledge, supersessions = file_knowledge((_fact("database", "postgres"),), (_fact("database", "mysql"),))
        assert [(e.content, e.knowledge_status) for e in knowledge] == [
            ("postgres", KnowledgeStatus.SUPERSEDED), ("mysql", KnowledgeStatus.ACTIVE)]
        assert supersessions[0].superseded[0].content == "postgres"

    def test_supersession_applies_within_one_batch(self):
        knowledge, _ = file_knowledge((), (_fact("database", "postgres"), _fact("database", "mysql")))
        assert [e.content for e in knowledge if e.knowledge_status == KnowledgeStatus.ACTIVE] == ["mysql"]

    def test_a_raw_entry_supersedes_nothing(self):
        raw = KnowledgeEntry(message_turn=1, content="verbatim", topic_label="database",
                             knowledge_type=KnowledgeType.RAW)
        knowledge, supersessions = file_knowledge((_fact("database", "postgres"),), (raw,))
        assert supersessions == ()
        assert all(e.knowledge_status == KnowledgeStatus.ACTIVE for e in knowledge)

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

    def test_raw_entries_never_supersede_each_other(self):
        """Verbatim fallbacks share one sentinel label and state no claim. If they
        superseded each other, all but the latest would drop out of the active
        view and the no-information-loss guarantee would leak."""
        raw = [KnowledgeEntry(message_turn=i, content=text, topic_label="__raw__", knowledge_type=KnowledgeType.RAW)
               for i, text in enumerate(["Thanks for the help", "Sounds good to me"])]
        knowledge, supersessions = file_knowledge((), tuple(raw))
        assert supersessions == ()
        assert all(e.knowledge_status == KnowledgeStatus.ACTIVE for e in knowledge)
