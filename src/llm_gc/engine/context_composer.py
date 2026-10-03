from llm_gc.engine.compaction import BaseCompactor
from llm_gc.engine.generations import GenerationalMemory
from llm_gc.engine.sweep import SweepResult, SweepClassification
from llm_gc.events import EventBus, Event, EventType
from llm_gc.models import Message

class ContextComposer:
    """Single point of responsibility for assembling the cleaned context the LLM provider will receive.
    Takes a SweepResult (each message already classified) and produces a list[Message] ready to ship.

    Handles all three classification outcomes: passes KEEP messages through unchanged,
    delegates compaction (COMPACT) to a BaseCompactor using the appropriate compactor mechanism
    and routes ARCHIVE messages to GenerationalMemory for knowledge extraction.
    This class itself never transforms message content. The purpose is to keep ordering and assembly logic in one place,
    separate from the strategies that decide how to compact or archive.
    """
    def __init__(self, event_bus: EventBus, generational_memory: GenerationalMemory, compactor: BaseCompactor):
        self.event_bus = event_bus
        self.generational_memory = generational_memory
        self.compactor = compactor

    async def compose(self, sweep_result: SweepResult) -> list[Message]:
        """Produce the optimized message list from a sweep result.

        Iterates sweep entries in turn order, grouping adjacent COMPACT messages into compaction runs that are passed to the
        injected BaseCompactor, while preserving relative ordering of all classifications. Emits a CONTEXT_COMPOSED event
        with transformation metrics.
        """
        final_messages: list[Message] = []
        current_compact_run: list[Message] = []
        kept_messages_count, compacted_messages_count, archived_messages_count, compact_runs_count = 0, 0, 0, 0
        for sweep_entry in sweep_result.sweep_entries:
            if current_compact_run and sweep_entry.classification != SweepClassification.COMPACT:
                compaction_result = await self.compactor.compact(current_compact_run)
                final_messages.append(compaction_result.summary)
                compact_runs_count += 1
                current_compact_run = []

            if sweep_entry.classification == SweepClassification.KEEP:
                final_messages.append(sweep_entry.message)
                kept_messages_count += 1
            elif sweep_entry.classification == SweepClassification.COMPACT:
                current_compact_run.append(sweep_entry.message)
                compacted_messages_count += 1
            elif sweep_entry.classification == SweepClassification.ARCHIVE:
                self.generational_memory.archive_message(sweep_entry.message)
                archived_messages_count += 1

        if current_compact_run:
            final_messages.append((await self.compactor.compact(current_compact_run)).summary)
            compact_runs_count += 1

        self.event_bus.emit(Event(event_type=EventType.CONTEXT_COMPOSED,
                                  data={"messages_kept": kept_messages_count, "messages_compacted": compacted_messages_count,
                                        "messages_archived": archived_messages_count, "compact_runs_created": compact_runs_count,
                                        "tokens_before": sweep_result.total_keep_tokens + sweep_result.total_compact_tokens + sweep_result.total_archive_tokens,
                                        "tokens_after": sum(msg.token_count for msg in final_messages),
                                        "final_message_count": len(final_messages)
                                        }))
        return final_messages

