from collections import deque

from llm_gc.engine.generations.permanent_generation import PermanentGeneration
from llm_gc.events import EventBus, Event, EventType
from llm_gc.extraction import KnowledgeExtractor
from llm_gc.models import Message, KnowledgeEntry
from llm_gc.models.knowledge_entry import KnowledgeType


class GenerationalMemory:
    def __init__(self, event_bus: EventBus, knowledge_extractor: KnowledgeExtractor, permanent_generation: PermanentGeneration, max_young_gen_size: int) -> None:
        self._young_gen: deque = deque(maxlen=max_young_gen_size)
        self._old_gen: list[Message] = []
        self.event_bus = event_bus
        self.knowledge_extractor = knowledge_extractor
        self.permanent_generation = permanent_generation
        self.max_young_gen_size = max_young_gen_size

    def _promote_to_old_gen(self) -> Message:
        promoted_message = self._young_gen.popleft()
        self._old_gen.append(promoted_message)
        return promoted_message

    def add_message_turn(self, message: Message) -> None:
        if len(self._young_gen) == self.max_young_gen_size:
            promoted_message = self._promote_to_old_gen()
            self.event_bus.emit(Event(event_type=EventType.MESSAGE_PROMOTED_TO_OLD_GEN, data={"promoted_message": promoted_message}))
        self._young_gen.append(message)
        self.event_bus.emit(Event(event_type=EventType.MESSAGE_ADDED_TO_YOUNG_GEN,
                                  data={"message": message, "current_young_gen_size": len(self._young_gen)}))

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

        self._old_gen.remove(message)
        self.event_bus.emit(Event(event_type=EventType.MESSAGE_ARCHIVED, data={"message": message, "extracted_knowledge_entries": extracted_knowledge_entries}))

    def get_young_gen(self) -> list[Message]:
        return list(self._young_gen)

    def get_old_gen(self) -> list[Message]:
        return list(self._old_gen)

    def get_permanent_gen(self) -> list[KnowledgeEntry]:
        return self.permanent_generation.get_all_active_entries()