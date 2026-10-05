
import pytest
from pydantic import ValidationError

from llm_gc.config import GCConfig
from llm_gc.engine import AgingDegradation, DegradationReason, GCResult, GCStatus
from llm_gc.engine.sweep import SweepClassification, SweepEntry, SweepResult
from llm_gc.events import Event, EventBus, EventType
from llm_gc.models.generation_snapshot import EntryKind, FrontEntry, GenerationSnapshot
from llm_gc.models import KnowledgeEntry, Message
from llm_gc.models.knowledge_entry import KnowledgeType
from llm_gc.monitoring import ContextHealthMonitor
from llm_gc.monitoring.context_health import TransitionType


# ---- Helper factories -------------------------------------------------------
#
# Small builders that take only the fields a given test cares about and fill the
# rest with sensible defaults.


def _make_message(turn_index: int = 0, token_count: int = 10, content: str = "msg") -> Message:
    return Message(role="user", content=content, token_count=token_count, turn_index=turn_index)


def _make_sweep_result(classifications: dict[int, SweepClassification], token_count: int = 10) -> SweepResult:
    """Build a SweepResult from {turn_index: classification}, every message token_count tokens."""
    entries = [SweepEntry(message=_make_message(turn, token_count), classification=classification,
                          relevance_score=0.5, turn_index=turn)
               for turn, classification in classifications.items()]
    counts = {c: sum(1 for v in classifications.values() if v == c) for c in SweepClassification}
    return SweepResult(sweep_entries=entries, classification_counts=counts, processing_time_ms=1.0,
                       total_keep_tokens=counts[SweepClassification.KEEP] * token_count,
                       total_compact_tokens=counts[SweepClassification.COMPACT] * token_count,
                       total_archive_tokens=counts[SweepClassification.ARCHIVE] * token_count)


def _make_completed_result(classifications: dict[int, SweepClassification], token_count: int = 10,
                           gc_run_id: str = "run-1") -> GCResult:
    sweep_result = _make_sweep_result(classifications, token_count)
    final_messages = [e.message for e in sweep_result.sweep_entries
                      if e.classification != SweepClassification.ARCHIVE]
    return GCResult(final_messages=final_messages, gc_run_id=gc_run_id, status=GCStatus.COMPLETED,
                    tokens_before=len(classifications) * token_count,
                    tokens_in_final=sweep_result.total_keep_tokens + sweep_result.total_compact_tokens,
                    kept_count=sweep_result.classification_counts[SweepClassification.KEEP],
                    compacted_count=sweep_result.classification_counts[SweepClassification.COMPACT],
                    archived_count=sweep_result.classification_counts[SweepClassification.ARCHIVE],
                    sweep_result=sweep_result, duration_ms=2.0)


def _make_bypassed_result(messages: list[Message], status: GCStatus = GCStatus.BYPASSED_BATCH_TOO_SMALL,
                          failure_reason: str | None = None, failure_stage: str | None = None) -> GCResult:
    """A bypassed run passes messages through untouched and carries no sweep_result."""
    total = sum(m.token_count for m in messages)
    return GCResult(final_messages=messages, gc_run_id="run-bypass", status=status,
                    failure_reason=failure_reason, failure_stage=failure_stage,
                    tokens_before=total, tokens_in_final=total, kept_count=len(messages),
                    compacted_count=0, archived_count=0, sweep_result=None, duration_ms=1.0)


def _make_knowledge_entry(message_turn: int = 1, topic_label: str = "database",
                          content: str = "postgres") -> KnowledgeEntry:
    return KnowledgeEntry(message_turn=message_turn, content=content, topic_label=topic_label,
                          knowledge_type=KnowledgeType.FACT)


class _Generations:
    """What the monitor reads: the current snapshot of the conversation it describes."""

    def __init__(self) -> None:
        self.snapshot = GenerationSnapshot()


def _make_monitor(max_recent_transitions: int | None = None,
                  context_window: int = 1000) -> tuple[ContextHealthMonitor, EventBus, "_Generations"]:
    """Monitor on a real EventBus, reading a snapshot the test sets directly.

    The bus is cheap and already tested; generations are a plain value, so these
    tests never stand up extractors or a session they don't exercise.
    """
    bus = EventBus()
    gen_memory = _Generations()
    kwargs = {} if max_recent_transitions is None else {"max_recent_transitions": max_recent_transitions}
    monitor = ContextHealthMonitor(event_bus=bus, gc_config=GCConfig(context_window=context_window),
                                   read_generations=lambda _session_id: gen_memory.snapshot, **kwargs)
    return monitor, bus, gen_memory


def _emit_gc_finished(bus: EventBus, gc_result: GCResult, timestamp: float | None = None) -> None:
    kwargs = {} if timestamp is None else {"timestamp": timestamp}
    bus.emit(Event(event_type=EventType.GC_FINISHED,
                   data={"gc_run_id": gc_result.gc_run_id, "gc_result": gc_result}, **kwargs))


def _transitions_of(monitor: ContextHealthMonitor, transition_type: TransitionType) -> list:
    return [t for t in monitor.get_snapshot().generation_lifecycle.recent_transitions
            if t.transition_type == transition_type]


# ---- Promotion recording ------------------------------------------------------


def _emit_promoted(bus: EventBus, source_turn_indices: list[int], timestamp: float | None = None) -> None:
    kwargs = {} if timestamp is None else {"timestamp": timestamp}
    bus.emit(Event(event_type=EventType.MESSAGE_PROMOTED_TO_OLD_GEN,
                   data={"source_turn_indices": source_turn_indices,
                         "messages_compacted": len(source_turn_indices),
                         "tokens_before": 100, "tokens_after": 20}, **kwargs))


class TestPromotionRecording:
    """PROMOTED = generational memory told us it moved a turn into old gen.

    The monitor used to *infer* this by diffing consecutive sweeps for a
    KEEP -> COMPACT change. That inference was both lossy and false-positive:
    it missed a turn first seen as COMPACT (no prior KEEP to diff against), and
    it fired on the stateless collect path, which classifies COMPACT but never
    calls promote_to_old_gen - so old gen stayed empty while the dashboard
    claimed a promotion. These tests pin the replacement: listen for the fact.
    """

    def test_promotion_event_fires_one_transition(self):
        monitor, bus, _ = _make_monitor()
        _emit_promoted(bus, [2])

        promoted = _transitions_of(monitor, TransitionType.PROMOTED)
        assert len(promoted) == 1
        assert promoted[0].turn_index == 2
        assert promoted[0].topic_label is None

    def test_a_promoted_run_fires_one_transition_per_turn(self):
        monitor, bus, _ = _make_monitor()
        # promote_to_old_gen takes a *run* of turns and files one summary; each
        # turn in the run genuinely left young, so each gets its own transition.
        _emit_promoted(bus, [4, 5, 6])

        assert sorted(t.turn_index for t in _transitions_of(monitor, TransitionType.PROMOTED)) == [4, 5, 6]

    def test_classification_change_alone_fires_nothing(self):
        """The regression that matters most: a KEEP -> COMPACT change is a sweep
        *decision*, not a move. Only the producer can say a turn actually left."""
        monitor, bus, _ = _make_monitor()
        _emit_gc_finished(bus, _make_completed_result({2: SweepClassification.KEEP}))
        _emit_gc_finished(bus, _make_completed_result({2: SweepClassification.COMPACT}))

        assert _transitions_of(monitor, TransitionType.PROMOTED) == []

    def test_first_pass_compact_still_records_when_the_move_happens(self):
        """The case the old diff silently missed: a turn appended and classified
        COMPACT in the same pass has no prior KEEP, but it really was promoted."""
        monitor, bus, _ = _make_monitor()
        _emit_gc_finished(bus, _make_completed_result({2: SweepClassification.COMPACT}))
        _emit_promoted(bus, [2])

        assert [t.turn_index for t in _transitions_of(monitor, TransitionType.PROMOTED)] == [2]

    def test_each_promotion_is_recorded_separately(self):
        monitor, bus, _ = _make_monitor()
        _emit_promoted(bus, [1])
        _emit_promoted(bus, [3])

        assert [t.turn_index for t in _transitions_of(monitor, TransitionType.PROMOTED)] == [1, 3]

    def test_bypassed_run_records_no_promotion(self):
        monitor, bus, _ = _make_monitor()
        _emit_gc_finished(bus, _make_bypassed_result([_make_message(turn_index=2)]))

        assert monitor.get_snapshot().generation_lifecycle.recent_transitions == []


# ---- Archive and supersede handlers ------------------------------------------


class TestArchiveAndSupersedeHandlers:
    def test_message_archived_appends_transition(self):
        monitor, bus, _ = _make_monitor()
        bus.emit(Event(event_type=EventType.MESSAGE_ARCHIVED,
                       data={"message": _make_message(turn_index=7), "extracted_knowledge_entries": []},
                       timestamp=111.0))

        archived = _transitions_of(monitor, TransitionType.ARCHIVED)
        assert len(archived) == 1
        assert archived[0].turn_index == 7
        assert archived[0].topic_label is None
        assert archived[0].occurred_at == 111.0

    def test_superseded_appends_one_transition_per_entry(self):
        monitor, bus, _ = _make_monitor()
        old_entries = [_make_knowledge_entry(message_turn=1, topic_label="database"),
                       _make_knowledge_entry(message_turn=3, topic_label="database")]
        bus.emit(Event(event_type=EventType.KNOWLEDGE_ENTRY_SUPERSEDED,
                       data={"new_knowledge_entry": _make_knowledge_entry(message_turn=9, content="mysql"),
                             "superseded_knowledge_entries": old_entries}))

        superseded = _transitions_of(monitor, TransitionType.SUPERSEDED)
        # turn_index points at the *origin turn of the superseded fact*, not the new one.
        assert [(t.turn_index, t.topic_label) for t in superseded] == [(1, "database"), (3, "database")]

    def test_transitions_bounded_to_max_oldest_evicted(self):
        monitor, bus, _ = _make_monitor(max_recent_transitions=3)
        for turn in range(5):
            bus.emit(Event(event_type=EventType.MESSAGE_ARCHIVED,
                           data={"message": _make_message(turn_index=turn), "extracted_knowledge_entries": []}))

        transitions = monitor.get_snapshot().generation_lifecycle.recent_transitions
        assert [t.turn_index for t in transitions] == [2, 3, 4]


# ---- get_snapshot: never ran --------------------------------------------------


class TestSnapshotNeverRan:
    def test_measured_signals_none_lifecycle_empty_but_real(self):
        monitor, _, gen_memory = _make_monitor()
        entry = _make_knowledge_entry()
        gen_memory.snapshot = GenerationSnapshot(knowledge=(entry,))

        snapshot = monitor.get_snapshot()
        assert snapshot.token_budget is None
        assert snapshot.gc_breakdown is None
        assert snapshot.transformation_ratio is None
        # Lifecycle is a standing fact: empty young/old, but permanent storage is real.
        assert snapshot.generation_lifecycle.young_gen_count == 0
        assert snapshot.generation_lifecycle.old_gen_count == 0
        assert snapshot.generation_lifecycle.permanent_gen_entries == [entry]
        assert snapshot.generation_lifecycle.permanent_gen_count == 1


# ---- get_snapshot: completed run ----------------------------------------------


class TestSnapshotCompleted:
    CLASSIFICATIONS = {0: SweepClassification.ARCHIVE, 1: SweepClassification.COMPACT,
                       2: SweepClassification.COMPACT, 3: SweepClassification.KEEP}

    def _snapshot(self, monitor, bus, timestamp: float = 500.0):
        _emit_gc_finished(bus, _make_completed_result(self.CLASSIFICATIONS), timestamp=timestamp)
        return monitor.get_snapshot()

    def test_token_budget_reads_last_result(self):
        monitor, bus, _ = _make_monitor(context_window=1000)
        budget = self._snapshot(monitor, bus).token_budget
        assert budget.current_tokens == 30  # 1 keep + 2 compact, 10 tokens each
        assert budget.context_window == 1000
        assert budget.pressure_ratio == 0.03
        assert budget.current_msg_turn_index == 3  # last sweep entry's turn

    def test_breakdown_copies_counts_tokens_and_identity(self):
        monitor, bus, _ = _make_monitor()
        breakdown = self._snapshot(monitor, bus).gc_breakdown
        assert (breakdown.keep_count, breakdown.compact_count, breakdown.archive_count) == (1, 2, 1)
        assert (breakdown.keep_tokens, breakdown.compact_tokens, breakdown.archive_tokens) == (10, 20, 10)
        assert breakdown.tokens_before == 40
        assert breakdown.tokens_saved == 10  # 40 before - 30 in final
        assert breakdown.gc_run_id == "run-1"
        assert breakdown.gc_status == GCStatus.COMPLETED
        assert breakdown.gc_completed_at == 500.0  # the event's timestamp, not snapshot build time
        assert breakdown.failure_reason is None and breakdown.failure_stage is None
        assert breakdown.newest_turn_index == 3     # places the pass in the conversation

    def test_transformation_ratio_is_transformed_over_total(self):
        monitor, bus, _ = _make_monitor()
        # (compact 20 + archive 10) / (keep 10 + compact 20 + archive 10)
        assert self._snapshot(monitor, bus).transformation_ratio == pytest.approx(30 / 40)

    def test_young_gen_is_derived_from_the_sweep(self):
        monitor, bus, _ = _make_monitor()
        lifecycle = self._snapshot(monitor, bus).generation_lifecycle
        assert lifecycle.young_gen_turn_indices == [3]      # the KEEP entries
        assert lifecycle.young_gen_count == 1
        assert lifecycle.young_gen_tokens == 10

    def test_old_gen_is_read_from_storage_not_from_compact_counts(self):
        """A COMPACT classification is a decision; old gen is where turns actually
        landed. The sweep below classifies two turns COMPACT while storage holds
        one summary from an earlier pass - the signal must report storage."""
        monitor, bus, gen_memory = _make_monitor()
        summary = FrontEntry(message=_make_message(turn_index=7, token_count=5), kind=EntryKind.SUMMARY,
                             source_turns=(7, 8), text="msg")
        gen_memory.snapshot = GenerationSnapshot(old=(summary,))

        lifecycle = self._snapshot(monitor, bus).generation_lifecycle

        assert lifecycle.old_gen_turn_indices == [7]
        assert lifecycle.old_gen_count == 1
        assert lifecycle.old_gen_tokens == 5

    def test_old_gen_reads_empty_on_the_stateless_path(self):
        """The stateless collect path classifies COMPACT but never promotes, so
        old gen really is empty - reporting it as populated was a false claim."""
        monitor, bus, _ = _make_monitor()
        lifecycle = self._snapshot(monitor, bus).generation_lifecycle
        assert lifecycle.old_gen_count == 0
        assert lifecycle.old_gen_turn_indices == []
        assert lifecycle.old_gen_tokens == 0

    def test_zero_token_sweep_yields_no_ratio(self):
        monitor, bus, _ = _make_monitor()
        _emit_gc_finished(bus, _make_completed_result({0: SweepClassification.KEEP}, token_count=0))
        assert monitor.get_snapshot().transformation_ratio is None


# ---- get_snapshot: bypassed runs -----------------------------------------------


class TestSnapshotBypassed:
    def test_below_threshold_breakdown_shows_everything_kept(self):
        monitor, bus, _ = _make_monitor()
        _emit_gc_finished(bus, _make_bypassed_result([_make_message(turn_index=t) for t in range(3)]))

        snapshot = monitor.get_snapshot()
        assert snapshot.gc_breakdown.gc_status == GCStatus.BYPASSED_BATCH_TOO_SMALL
        assert (snapshot.gc_breakdown.keep_count, snapshot.gc_breakdown.compact_count,
                snapshot.gc_breakdown.archive_count) == (3, 0, 0)
        assert (snapshot.gc_breakdown.keep_tokens, snapshot.gc_breakdown.compact_tokens,
                snapshot.gc_breakdown.archive_tokens) == (30, 0, 0)
        assert snapshot.transformation_ratio is None  # no sweep ran - nothing measured

    def test_below_threshold_turn_index_from_final_messages(self):
        monitor, bus, _ = _make_monitor()
        _emit_gc_finished(bus, _make_bypassed_result([_make_message(turn_index=t) for t in range(5)]))

        assert monitor.get_snapshot().token_budget.current_msg_turn_index == 4
        assert monitor.get_snapshot().gc_breakdown.newest_turn_index == 4

    def test_below_threshold_lifecycle_all_young(self):
        monitor, bus, _ = _make_monitor()
        _emit_gc_finished(bus, _make_bypassed_result([_make_message(turn_index=t) for t in range(3)]))

        lifecycle = monitor.get_snapshot().generation_lifecycle
        assert lifecycle.young_gen_turn_indices == [0, 1, 2]
        assert lifecycle.young_gen_tokens == 30
        assert lifecycle.old_gen_count == 0

    def test_on_error_failure_detail_is_visible(self):
        monitor, bus, _ = _make_monitor()
        _emit_gc_finished(bus, _make_bypassed_result([_make_message()], status=GCStatus.BYPASSED_ON_ERROR,
                                                     failure_reason="boom", failure_stage="sweep"))

        breakdown = monitor.get_snapshot().gc_breakdown
        assert breakdown.gc_status == GCStatus.BYPASSED_ON_ERROR
        assert breakdown.failure_reason == "boom"
        assert breakdown.failure_stage == "sweep"

    def test_empty_conversation_has_no_token_budget(self):
        monitor, bus, _ = _make_monitor()
        _emit_gc_finished(bus, _make_bypassed_result([]))

        snapshot = monitor.get_snapshot()
        assert snapshot.token_budget is None  # no turn exists to point at
        assert snapshot.gc_breakdown is not None  # the bypassed pass itself is still visible
        assert snapshot.gc_breakdown.newest_turn_index is None


# ---- Snapshot immutability and freshness ---------------------------------------


class TestSnapshotSemantics:
    def test_snapshot_is_frozen(self):
        monitor, _, _ = _make_monitor()
        snapshot = monitor.get_snapshot()
        with pytest.raises(ValidationError):
            snapshot.transformation_ratio = 0.5

    def test_each_call_builds_a_fresh_snapshot(self):
        monitor, bus, _ = _make_monitor()
        before = monitor.get_snapshot()
        _emit_gc_finished(bus, _make_completed_result({0: SweepClassification.KEEP}))
        after = monitor.get_snapshot()

        assert before is not after
        assert before.gc_breakdown is None  # the old photograph is untouched
        assert after.gc_breakdown is not None


# ---- One monitor, many sessions ------------------------------------------------


def _emit_for(bus: EventBus, session_id: str, event_type: EventType, **data) -> None:
    bus.emit(Event(event_type=event_type, data={"session_id": session_id, **data}))


class TestSessionsAreKeptApart:
    """Every session's events share one bus. The monitor used to keep one last
    pass and one timeline for all of them, so a dashboard watching one user's
    conversation showed whichever session happened to finish a pass last."""

    def test_each_session_sees_only_its_own_pass_and_transitions(self):
        monitor, bus, _ = _make_monitor()
        _emit_for(bus, "alice", EventType.GC_FINISHED, gc_result=_make_completed_result(
            {0: SweepClassification.KEEP}, gc_run_id="alice-run"))
        _emit_for(bus, "bob", EventType.MESSAGE_PROMOTED_TO_OLD_GEN, source_turn_indices=[4])

        alice, bob = monitor.get_snapshot("alice"), monitor.get_snapshot("bob")

        assert alice.gc_breakdown.gc_run_id == "alice-run"
        assert alice.generation_lifecycle.recent_transitions == []
        assert bob.gc_breakdown is None
        assert [t.turn_index for t in bob.generation_lifecycle.recent_transitions] == [4]

    def test_generations_are_read_for_the_session_asked_about(self):
        snapshots = {"alice": GenerationSnapshot(knowledge=(_make_knowledge_entry(topic_label="db"),))}
        monitor = ContextHealthMonitor(event_bus=EventBus(), gc_config=GCConfig(),
                                       read_generations=snapshots.get)

        assert monitor.get_snapshot("alice").generation_lifecycle.permanent_gen_count == 1
        # A session that no longer exists reads as empty, not as an error.
        assert monitor.get_snapshot("gone").generation_lifecycle.permanent_gen_count == 0

    def test_forgetting_a_session_drops_its_history(self):
        monitor, bus, _ = _make_monitor()
        _emit_for(bus, "alice", EventType.MESSAGE_PROMOTED_TO_OLD_GEN, source_turn_indices=[1])

        monitor.forget("alice")

        assert monitor.get_snapshot("alice").generation_lifecycle.recent_transitions == []

    def test_the_quietest_session_is_dropped_past_the_bound(self):
        bus = EventBus()
        monitor = ContextHealthMonitor(event_bus=bus, gc_config=GCConfig(),
                                       read_generations=lambda _session_id: None, max_sessions=2)
        for session_id in ("a", "b"):
            _emit_for(bus, session_id, EventType.MESSAGE_PROMOTED_TO_OLD_GEN, source_turn_indices=[1])
        _emit_for(bus, "a", EventType.MESSAGE_PROMOTED_TO_OLD_GEN, source_turn_indices=[2])   # a is now newest
        _emit_for(bus, "c", EventType.MESSAGE_PROMOTED_TO_OLD_GEN, source_turn_indices=[1])

        assert monitor.get_snapshot("b").generation_lifecycle.recent_transitions == []
        assert len(monitor.get_snapshot("a").generation_lifecycle.recent_transitions) == 2

    def test_the_monitor_keeps_no_conversation_content_from_a_pass(self):
        """A GCResult carries the whole context. Holding it per session would keep
        every conversation's text alive in the monitor long after its session closed."""
        monitor, bus, _ = _make_monitor()
        _emit_for(bus, "alice", EventType.GC_FINISHED, gc_result=_make_completed_result(
            {0: SweepClassification.KEEP, 1: SweepClassification.COMPACT}))

        record = monitor._records["alice"]
        assert not any(isinstance(value, GCResult) for value in vars(record.last_pass).values())


class TestDegradationsOutliveTheirPass:
    def test_a_refusal_stays_visible_after_a_quiet_pass(self):
        monitor, bus, _ = _make_monitor()
        refused = AgingDegradation(reason=DegradationReason.COMPACTION_REFUSED, turn_indices=[3, 4],
                                   detail="summary was not shorter than the run")
        _emit_for(bus, "alice", EventType.AGING_DEGRADED, degradation=refused)
        _emit_for(bus, "alice", EventType.GC_FINISHED, gc_result=_make_bypassed_result(
            [_make_message(turn_index=5)], status=GCStatus.BYPASSED_BATCH_TOO_SMALL))

        snapshot = monitor.get_snapshot("alice")

        assert snapshot.gc_breakdown.is_degraded is False, "the last pass itself was clean"
        assert [r.degradation for r in snapshot.recent_degradations] == [refused]


class TestPassHistory:
    def test_every_pass_is_kept_in_order_per_session(self):
        """The dashboard's activity timeline: what each pass did, not only the last."""
        monitor, bus, _ = _make_monitor()
        for run in ("one", "two"):
            _emit_for(bus, "alice", EventType.GC_FINISHED, gc_result=_make_completed_result(
                {0: SweepClassification.KEEP}, gc_run_id=run))
        _emit_for(bus, "bob", EventType.GC_FINISHED, gc_result=_make_completed_result(
            {0: SweepClassification.KEEP}, gc_run_id="bob-run"))

        assert [p.gc_run_id for p in monitor.get_snapshot("alice").recent_passes] == ["one", "two"]
        assert [p.gc_run_id for p in monitor.get_snapshot("bob").recent_passes] == ["bob-run"]
