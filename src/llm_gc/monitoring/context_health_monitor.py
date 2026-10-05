import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, field

from llm_gc.config import GCConfig
from llm_gc.config.constants import DEFAULT_MAX_RECENT_DEGRADATIONS, DEFAULT_MAX_RECENT_PASSES, \
    DEFAULT_MAX_RECENT_TRANSITIONS, DEFAULT_MONITOR_MAX_SESSIONS
from llm_gc.engine import AgingDegradation, GCResult
from llm_gc.engine.generations.front import span_message
from llm_gc.engine.sweep import SweepClassification, SweepResult
from llm_gc.events import Event, EventBus, EventType
from llm_gc.models import KnowledgeEntry, Message
from llm_gc.models.generation_snapshot import GenerationSnapshot
from llm_gc.monitoring.context_health import (
    GenerationTransition,
    RecordedDegradation,
    TransitionType,
    ContextHealth,
    TokenBudgetSignal,
    GCBreakdownSignal,
    GenerationLifeCycleSignal,
)


@dataclass(frozen=True)
class _PassSignals:
    """What the monitor keeps of one finished pass: its signals, built on arrival.

    Built when GC_FINISHED arrives rather than when a snapshot is read, so the
    monitor never holds the result itself - which carries the conversation - only
    counts, tokens and turn indices.
    """

    token_budget: TokenBudgetSignal | None
    gc_breakdown: GCBreakdownSignal
    transformation_ratio: float | None
    young_turn_indices: list[int]
    young_tokens: int


@dataclass
class _SessionRecord:
    """Everything the monitor has heard about one session."""

    transitions: deque[GenerationTransition]
    degradations: deque[RecordedDegradation]
    passes: deque[GCBreakdownSignal]
    last_pass: _PassSignals | None = field(default=None)


class ContextHealthMonitor:
    """Keeps the event-fed half of each session's health: its last pass and its history.

    Events carry the session they describe, so one monitor serves every session on
    the bus without mixing them. Events with no session - the stateless library
    path - are kept under `None`.
    """

    def __init__(self, event_bus: EventBus, gc_config: GCConfig,
                 read_generations: Callable[[str | None], GenerationSnapshot | None],
                 max_recent_transitions: int = DEFAULT_MAX_RECENT_TRANSITIONS,
                 max_recent_degradations: int = DEFAULT_MAX_RECENT_DEGRADATIONS,
                 max_sessions: int = DEFAULT_MONITOR_MAX_SESSIONS) -> None:
        self.event_bus = event_bus
        self.gc_config = gc_config
        # How to read a session's generations. A function rather than a store,
        # because generations belong to the session; None means it no longer exists.
        self.read_generations = read_generations
        self._max_recent_transitions = max_recent_transitions
        self._max_recent_degradations = max_recent_degradations
        # Least recently heard from first, so the bound evicts sessions that have
        # gone quiet - normally ones already closed.
        self._max_sessions = max_sessions
        self._records: OrderedDict[str | None, _SessionRecord] = OrderedDict()
        event_bus.subscribe(EventType.GC_FINISHED, self._on_gc_finished)
        event_bus.subscribe(EventType.MESSAGE_PROMOTED_TO_OLD_GEN, self._on_message_promoted_to_old_gen)
        event_bus.subscribe(EventType.MESSAGE_ARCHIVED, self._on_message_archived)
        event_bus.subscribe(EventType.KNOWLEDGE_ENTRY_SUPERSEDED, self._on_knowledge_entry_superseded)
        event_bus.subscribe(EventType.AGING_DEGRADED, self._on_aging_degraded)

    def forget(self, session_id: str | None) -> None:
        """Drop everything heard about a session, as when it is closed."""
        self._records.pop(session_id, None)

    def _record_for(self, event: Event) -> _SessionRecord:
        session_id = event.data.get("session_id")
        record = self._records.get(session_id)
        if record is None:
            record = _SessionRecord(transitions=deque(maxlen=self._max_recent_transitions),
                                    degradations=deque(maxlen=self._max_recent_degradations),
                                    passes=deque(maxlen=DEFAULT_MAX_RECENT_PASSES))
            self._records[session_id] = record
            if len(self._records) > self._max_sessions:
                self._records.popitem(last=False)
        else:
            self._records.move_to_end(session_id)
        return record

    def _on_gc_finished(self, event: Event) -> None:
        gc_result: GCResult = event.data["gc_result"]
        young_turn_indices, young_tokens = self._young_generation(gc_result)
        breakdown = self._build_gc_breakdown(gc_result, event.timestamp)
        record = self._record_for(event)
        record.passes.append(breakdown)
        record.last_pass = _PassSignals(
            token_budget=self._build_token_budget(gc_result), gc_breakdown=breakdown,
            transformation_ratio=self._compute_transformation_ratio(gc_result.sweep_result),
            young_turn_indices=young_turn_indices, young_tokens=young_tokens)

    def _on_message_promoted_to_old_gen(self, event: Event) -> None:
        """Record a young -> old promotion, one transition per turn that moved.

        Listens for the fact rather than inferring it. This used to be derived by
        diffing consecutive sweeps for a KEEP -> COMPACT change, which was a
        workaround from when nothing in the pipeline actually moved turns between
        generations. That inference both missed real promotions (a turn first seen
        as COMPACT has no prior KEEP to diff against) and reported promotions that
        never happened (the stateless collect path classifies COMPACT but never
        promotes anything into old gen). The producer knows what
        it moved; the monitor just writes it down.
        """
        transitions = self._record_for(event).transitions
        for turn_index in event.data["source_turn_indices"]:
            transitions.append(
                GenerationTransition(transition_type=TransitionType.PROMOTED, turn_index=turn_index,
                                     topic_label=None, occurred_at=event.timestamp))

    def _on_message_archived(self, event: Event) -> None:
        message: Message = event.data["message"]
        self._record_for(event).transitions.append(
            GenerationTransition(transition_type=TransitionType.ARCHIVED, turn_index=message.turn_index,
                                 topic_label=None, occurred_at=event.timestamp))

    def _on_knowledge_entry_superseded(self, event: Event) -> None:
        superseded_entries: list[KnowledgeEntry] = event.data["superseded_knowledge_entries"]
        transitions = self._record_for(event).transitions
        for superseded_entry in superseded_entries:
            transitions.append(GenerationTransition(transition_type=TransitionType.SUPERSEDED,
                                                                 turn_index=superseded_entry.message_turn,
                                                                 topic_label=superseded_entry.topic_label,
                                                                 occurred_at=event.timestamp))

    def _on_aging_degraded(self, event: Event) -> None:
        """Keep degradations across passes: one quiet pass must not hide a refusal before it."""
        degradation: AgingDegradation = event.data["degradation"]
        self._record_for(event).degradations.append(
            RecordedDegradation(degradation=degradation, occurred_at=event.timestamp))

    def get_snapshot(self, session_id: str | None = None) -> ContextHealth:
        """Build a fresh, immutable ContextHealth for one session.

        Built whole on every call - never cached, never mutated field-by-field -
        so a reader can never observe a torn, half-updated snapshot. token_budget,
        gc_breakdown, and transformation_ratio are None until the session's first
        GC pass has run: an honest "not measured yet" instead of a fabricated zero.
        """
        record = self._records.get(session_id)
        last_pass = record.last_pass if record is not None else None
        transitions = list(record.transitions) if record is not None else []
        degradations = list(record.degradations) if record is not None else []
        passes = list(record.passes) if record is not None else []

        return ContextHealth(
            token_budget=last_pass.token_budget if last_pass is not None else None,
            gc_breakdown=last_pass.gc_breakdown if last_pass is not None else None,
            generation_lifecycle=self._build_generation_lifecycle(last_pass, session_id, transitions),
            transformation_ratio=last_pass.transformation_ratio if last_pass is not None else None,
            recent_degradations=degradations,
            recent_passes=passes,
            snapshot_taken_at=time.time())

    @staticmethod
    def _newest_turn_index(gc_result: GCResult) -> int | None:
        """How far into the conversation the pass ran, or None for an empty one."""
        if gc_result.sweep_result is not None and gc_result.sweep_result.sweep_entries:
            return gc_result.sweep_result.sweep_entries[-1].turn_index
        if gc_result.final_messages:
            # Bypassed run: nothing was swept, so the untouched originals in
            # final_messages are the only source for the current turn index.
            return gc_result.final_messages[-1].turn_index
        return None

    def _build_token_budget(self, gc_result: GCResult) -> TokenBudgetSignal | None:
        current_msg_turn_index = self._newest_turn_index(gc_result)
        if current_msg_turn_index is None:
            return None   # an empty conversation has no turn to point at
        return TokenBudgetSignal(current_tokens=gc_result.tokens_in_final,
                                 context_window=self.gc_config.context_window,
                                 pressure_ratio=gc_result.tokens_in_final / self.gc_config.context_window,
                                 current_msg_turn_index=current_msg_turn_index)

    def _build_gc_breakdown(self, gc_result: GCResult, gc_completed_at: float) -> GCBreakdownSignal:
        if gc_result.sweep_result is not None:
            keep_tokens = gc_result.sweep_result.total_keep_tokens
            compact_tokens = gc_result.sweep_result.total_compact_tokens
            archive_tokens = gc_result.sweep_result.total_archive_tokens
        else:
            # Bypassed run: no sweep measured token totals, but the producer's own
            # claim (kept_count == len(messages)) is that everything stayed untouched.
            keep_tokens = gc_result.tokens_in_final
            compact_tokens = 0
            archive_tokens = 0
        return GCBreakdownSignal(keep_count=gc_result.kept_count, compact_count=gc_result.compacted_count,
                                 archive_count=gc_result.archived_count, keep_tokens=keep_tokens,
                                 compact_tokens=compact_tokens, archive_tokens=archive_tokens,
                                 tokens_before=gc_result.tokens_before, tokens_saved=gc_result.tokens_saved,
                                 gc_run_id=gc_result.gc_run_id, gc_status=gc_result.status,
                                 gc_completed_at=gc_completed_at, duration_ms=gc_result.duration_ms,
                                 failure_reason=gc_result.failure_reason, failure_stage=gc_result.failure_stage,
                                 newest_turn_index=self._newest_turn_index(gc_result),
                                 degradations=gc_result.degradations, outcome=gc_result.outcome,
                                 abandoned_cut=gc_result.abandoned_cut)

    @staticmethod
    def _compute_transformation_ratio(sweep_result: SweepResult | None) -> float | None:
        # A bypassed run has no sweep_result: no sweep ran, so there is no
        # decision profile to report - None, not a fabricated 0.0.
        if sweep_result is None:
            return None
        transformed_tokens = sweep_result.total_compact_tokens + sweep_result.total_archive_tokens
        total_swept_tokens = sweep_result.total_keep_tokens + transformed_tokens
        if total_swept_tokens == 0:
            return None
        return transformed_tokens / total_swept_tokens

    @staticmethod
    def _young_generation(gc_result: GCResult) -> tuple[list[int], int]:
        if gc_result.sweep_result is not None:
            # Young gen is still derived from the last sweep: KEEP is exactly the
            # set of turns that stayed. (One known skew: a turn whose promotion
            # failed is classified COMPACT but stays young, so it is undercounted
            # here until the next pass reclassifies it.)
            return ([entry.turn_index for entry in gc_result.sweep_result.sweep_entries
                     if entry.classification == SweepClassification.KEEP],
                    gc_result.sweep_result.total_keep_tokens)
        # Bypassed run: everything the pass saw stayed live and untransformed.
        return [message.turn_index for message in gc_result.final_messages], gc_result.tokens_in_final

    def _build_generation_lifecycle(self, last_pass: _PassSignals | None, session_id: str | None,
                                    transitions: list[GenerationTransition]) -> GenerationLifeCycleSignal:
        young_gen_turn_indices = last_pass.young_turn_indices if last_pass is not None else []
        young_gen_tokens = last_pass.young_tokens if last_pass is not None else 0

        # Old gen is read from real storage, not derived from COMPACT counts.
        # The derivation reported only the *last pass's* COMPACT turns, so a
        # session that had promoted four turns over four passes still showed
        # "old_gen_count: 1" while generational memory held four summaries - and
        # on the stateless collect path, which classifies COMPACT but never
        # promotes, it reported a populated old gen that was in fact empty. The
        # summaries carry the turn index of the first turn in their run, so the
        # indices still point back at where the content came from.
        generations = self.read_generations(session_id) or GenerationSnapshot()
        old_gen_messages = [message for span in generations.old if (message := span_message(span)) is not None]
        old_gen_turn_indices = [message.turn_index for message in old_gen_messages]
        old_gen_tokens = sum(message.token_count for message in old_gen_messages)

        # Permanent-gen contents and the transitions log are standing facts,
        # real even before any GC pass has ever run.
        permanent_gen_entries = generations.active_knowledge
        return GenerationLifeCycleSignal(young_gen_count=len(young_gen_turn_indices),
                                         old_gen_count=len(old_gen_turn_indices),
                                         permanent_gen_count=len(permanent_gen_entries),
                                         young_gen_tokens=young_gen_tokens, old_gen_tokens=old_gen_tokens,
                                         young_gen_turn_indices=young_gen_turn_indices,
                                         old_gen_turn_indices=old_gen_turn_indices,
                                         permanent_gen_entries=permanent_gen_entries,
                                         recent_transitions=transitions)

