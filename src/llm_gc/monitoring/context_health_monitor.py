import time
from collections import deque

from llm_gc.config import GCConfig
from llm_gc.config.constants import DEFAULT_MAX_RECENT_TRANSITIONS
from llm_gc.engine import GCResult
from llm_gc.engine.generations import GenerationalMemory
from llm_gc.engine.sweep import SweepClassification, SweepResult
from llm_gc.events import Event, EventBus, EventType
from llm_gc.models import KnowledgeEntry, Message
from llm_gc.monitoring.context_health import (
    GenerationTransition,
    TransitionType,
    ContextHealth,
    TokenBudgetSignal,
    GCBreakdownSignal,
    GenerationLifeCycleSignal,
)


class ContextHealthMonitor:
    def __init__(self, event_bus: EventBus, gc_config: GCConfig, generational_memory: GenerationalMemory,
                 max_recent_transitions: int = DEFAULT_MAX_RECENT_TRANSITIONS) -> None:
        self.event_bus = event_bus
        self.gc_config = gc_config
        self.generational_memory = generational_memory
        self._last_gc_result: GCResult | None = None
        self._last_gc_completed_at: float | None = None
        self._last_classifications: dict[int, SweepClassification] = {}
        self._recent_transitions = deque(maxlen=max_recent_transitions)
        event_bus.subscribe(EventType.GC_FINISHED, self._on_gc_finished)
        event_bus.subscribe(EventType.MESSAGE_ARCHIVED, self._on_message_archived)
        event_bus.subscribe(EventType.KNOWLEDGE_ENTRY_SUPERSEDED, self._on_knowledge_entry_superseded)

    def _on_gc_finished(self, event: Event) -> None:
        gc_result: GCResult = event.data["gc_result"]
        self._last_gc_result = gc_result
        self._last_gc_completed_at = event.timestamp

        if gc_result.sweep_result is not None:
            for entry in gc_result.sweep_result.sweep_entries:
                previous = self._last_classifications.get(entry.turn_index)
                if previous == SweepClassification.KEEP and entry.classification == SweepClassification.COMPACT:
                    self._recent_transitions.append(
                        GenerationTransition(transition_type=TransitionType.PROMOTED, turn_index=entry.turn_index,
                                             topic_label=None, occurred_at=event.timestamp))
                self._last_classifications[entry.turn_index] = entry.classification

    def _on_message_archived(self, event: Event) -> None:
        message: Message = event.data["message"]
        self._recent_transitions.append(
            GenerationTransition(transition_type=TransitionType.ARCHIVED, turn_index=message.turn_index,
                                 topic_label=None, occurred_at=event.timestamp))
        self._last_classifications.pop(message.turn_index, None)  # can't transform again once archived

    def _on_knowledge_entry_superseded(self, event: Event) -> None:
        superseded_entries: list[KnowledgeEntry] = event.data["superseded_knowledge_entries"]
        for superseded_entry in superseded_entries:
            self._recent_transitions.append(GenerationTransition(transition_type=TransitionType.SUPERSEDED,
                                                                 turn_index=superseded_entry.message_turn,
                                                                 topic_label=superseded_entry.topic_label,
                                                                 occurred_at=event.timestamp))

    def get_snapshot(self) -> ContextHealth:
        """Build a fresh, immutable ContextHealth from the monitor's current state.

        Built whole on every call - never cached, never mutated field-by-field -
        so a reader can never observe a torn, half-updated snapshot. token_budget,
        gc_breakdown, and transformation_ratio are None until the first GC pass
        has run: an honest "not measured yet" instead of a fabricated zero.
        """
        gc_result = self._last_gc_result

        if gc_result is None:
            token_budget_signal = None
            gc_breakdown_signal = None
            transformation_ratio = None
        else:
            token_budget_signal = self._build_token_budget(gc_result)
            gc_breakdown_signal = self._build_gc_breakdown(gc_result)
            transformation_ratio = self._compute_transformation_ratio(gc_result.sweep_result)

        return ContextHealth(token_budget=token_budget_signal, gc_breakdown=gc_breakdown_signal,
                             generation_lifecycle=self._build_generation_lifecycle(gc_result),
                             transformation_ratio=transformation_ratio, snapshot_taken_at=time.time())

    def _build_token_budget(self, gc_result: GCResult) -> TokenBudgetSignal | None:
        if gc_result.sweep_result is not None:
            current_msg_turn_index = gc_result.sweep_result.sweep_entries[-1].turn_index
        elif gc_result.final_messages:
            # Bypassed run: nothing was swept, so the untouched originals in
            # final_messages are the only source for the current turn index.
            current_msg_turn_index = gc_result.final_messages[-1].turn_index
        else:
            # Bypassed run on an empty conversation - there is no turn to point at.
            return None
        return TokenBudgetSignal(current_tokens=gc_result.tokens_in_final,
                                 context_window=self.gc_config.context_window,
                                 pressure_ratio=gc_result.tokens_in_final / self.gc_config.context_window,
                                 current_msg_turn_index=current_msg_turn_index)

    def _build_gc_breakdown(self, gc_result: GCResult) -> GCBreakdownSignal:
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
        # _last_gc_completed_at is always set alongside _last_gc_result in
        # _on_gc_finished, so it cannot be None here.
        return GCBreakdownSignal(keep_count=gc_result.kept_count, compact_count=gc_result.compacted_count,
                                 archive_count=gc_result.archived_count, keep_tokens=keep_tokens,
                                 compact_tokens=compact_tokens, archive_tokens=archive_tokens,
                                 tokens_before=gc_result.tokens_before, tokens_saved=gc_result.tokens_saved,
                                 gc_run_id=gc_result.gc_run_id, gc_status=gc_result.status,
                                 gc_completed_at=self._last_gc_completed_at, duration_ms=gc_result.duration_ms,
                                 failure_reason=gc_result.failure_reason, failure_stage=gc_result.failure_stage)

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

    def _build_generation_lifecycle(self, gc_result: GCResult | None) -> GenerationLifeCycleSignal:
        young_gen_turn_indices: list[int] = []
        old_gen_turn_indices: list[int] = []
        young_gen_tokens = 0
        old_gen_tokens = 0

        if gc_result is not None and gc_result.sweep_result is not None:
            for entry in gc_result.sweep_result.sweep_entries:
                if entry.classification == SweepClassification.KEEP:
                    young_gen_turn_indices.append(entry.turn_index)
                elif entry.classification == SweepClassification.COMPACT:
                    old_gen_turn_indices.append(entry.turn_index)
            young_gen_tokens = gc_result.sweep_result.total_keep_tokens
            old_gen_tokens = gc_result.sweep_result.total_compact_tokens
        elif gc_result is not None:
            # Bypassed run: everything the pass saw stayed live and untransformed.
            young_gen_turn_indices = [message.turn_index for message in gc_result.final_messages]
            young_gen_tokens = gc_result.tokens_in_final

        # Permanent-gen contents and the transitions log are standing facts,
        # real even before any GC pass has ever run.
        permanent_gen_entries = self.generational_memory.get_permanent_gen()
        return GenerationLifeCycleSignal(young_gen_count=len(young_gen_turn_indices),
                                         old_gen_count=len(old_gen_turn_indices),
                                         permanent_gen_count=len(permanent_gen_entries),
                                         young_gen_tokens=young_gen_tokens, old_gen_tokens=old_gen_tokens,
                                         young_gen_turn_indices=young_gen_turn_indices,
                                         old_gen_turn_indices=old_gen_turn_indices,
                                         permanent_gen_entries=permanent_gen_entries,
                                         recent_transitions=list(self._recent_transitions))

