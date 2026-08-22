from llm_gc.engine.compaction import BaseCompactor, CompactionResult
from llm_gc.engine.generations.permanent_generation import PermanentGeneration
from llm_gc.events import EventBus, Event, EventType
from llm_gc.extraction import KnowledgeExtractor
from llm_gc.models import Message, KnowledgeEntry
from llm_gc.models.knowledge_entry import KnowledgeType


class GenerationalMemory:
    def __init__(self, event_bus: EventBus, knowledge_extractor: KnowledgeExtractor, permanent_generation: PermanentGeneration,
                 compactor: BaseCompactor) -> None:
        self.event_bus = event_bus
        self.knowledge_extractor = knowledge_extractor
        self.permanent_generation = permanent_generation
        self.compactor = compactor
        self._old_gen: list[Message] = []

    def archive_message(self, message: Message) -> None:
        extracted_knowledge_entries = self.knowledge_extractor.extract_knowledge(message=message, message_turn=message.turn_index)
        # If extracted entry is empty build verbatim content to prevent information loss
        if not extracted_knowledge_entries:
            extracted_knowledge_entries = [KnowledgeEntry(message_turn=message.turn_index,
                                                          content=message.content,
                                                          topic_label="__raw__",
                                                          knowledge_type=KnowledgeType.RAW)]

        for entry in extracted_knowledge_entries:
            self.permanent_generation.add_knowledge_entry(entry)

        self.event_bus.emit(Event(event_type=EventType.MESSAGE_ARCHIVED, data={"message": message, "extracted_knowledge_entries": extracted_knowledge_entries}))


    def get_permanent_gen(self) -> list[KnowledgeEntry]:
        return self.permanent_generation.get_all_active_entries()

    def get_old_gen(self) -> list[Message]:
        """The old generation's summaries, as a copy.

        A copy, not the live list: callers assemble and sort what they get back,
        and handing out internal storage lets a caller reorder or truncate this
        component's state by accident.
        """
        return list(self._old_gen)

    def promote_to_old_gen(self, compact_run: list[Message]) -> None:
        compaction_result: CompactionResult = self.compactor.compact(compact_run)
        self._old_gen.append(compaction_result.summary)
        self.event_bus.emit(
            Event(
                event_type=EventType.MESSAGE_PROMOTED_TO_OLD_GEN,
                data={
                    # Which turns moved, not just how many. Subscribers that log a
                    # per-turn generational transition (the health monitor) need the
                    # identity of what moved; counts alone cannot name a turn.
                    "source_turn_indices": [message.turn_index for message in compact_run],
                    "messages_compacted": len(compact_run),
                    "tokens_before": compaction_result.original_token_count,
                    "tokens_after": compaction_result.summary.token_count,
                },
            )
        )