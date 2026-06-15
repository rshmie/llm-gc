from time import perf_counter

from llm_gc.config import GCConfig
from llm_gc.engine.sweep import BaseSweepStrategy, SweepResult, SweepClassification, SweepEntry
from llm_gc.events import EventBus, EventType, Event
from llm_gc.models import Message
from llm_gc.scoring import RelevanceScorerResult

class Sweeper:
    """Orchestrates the sweep phase of the Mark-Sweep-Compact pipeline.

    Applies universal override rules (system prompt, last N turns),
    delegates classification to the injected strategy, aggregates results
    into a SweepResult, and emits a SWEEP_COMPLETED event.
    """
    def __init__(self, sweeper_strategy: BaseSweepStrategy, gc_config: GCConfig, event_bus: EventBus):
        self.sweeper_strategy = sweeper_strategy
        self.gc_config = gc_config
        self.event_bus = event_bus

    def _apply_override(self, entry: SweepEntry, reason: str):
        entry.classification = SweepClassification.KEEP
        entry.override_applied = True
        entry.override_reason = reason

    def sweep(self, conversation: list[Message], relevance_scores: list[RelevanceScorerResult]) -> SweepResult:
        start_time = perf_counter()
        sweep_entries = self.sweeper_strategy.sweep(conversation, relevance_scores)
        classification_counts = {SweepClassification.KEEP: 0, SweepClassification.COMPACT: 0, SweepClassification.ARCHIVE: 0}
        total_keep_tokens, total_compact_tokens, total_archive_tokens = 0, 0, 0
        for index, entry in enumerate(sweep_entries):
            if index >= len(conversation) - self.gc_config.last_n_turns_to_keep:
                self._apply_override(entry, "Forced KEEP due to last_n_turns_to_keep config from the conversation")
            elif entry.message.role == "system":
                self._apply_override(entry, "Forced KEEP due to system role of the message")

            classification_counts[entry.classification] += 1
            total_keep_tokens += entry.message.token_count if entry.classification == SweepClassification.KEEP else 0
            total_compact_tokens += entry.message.token_count if entry.classification == SweepClassification.COMPACT else 0
            total_archive_tokens += entry.message.token_count if entry.classification == SweepClassification.ARCHIVE else 0

        if self.event_bus is not None:
            self.event_bus.emit(Event(event_type=EventType.SWEEP_COMPLETED, data={"sweep_entries": sweep_entries, "classification_counts": classification_counts,
                                                                                  "current_message_turn": len(conversation)}))
        return SweepResult(
            sweep_entries=sweep_entries,
            classification_counts=classification_counts,
            processing_time_ms=(perf_counter() - start_time) * 1000,
            total_keep_tokens= total_keep_tokens,
            total_compact_tokens= total_compact_tokens,
            total_archive_tokens= total_archive_tokens
        )

