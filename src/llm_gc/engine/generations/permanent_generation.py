from dataclasses import replace

from llm_gc.events import EventBus, Event, EventType
from llm_gc.models import KnowledgeEntry
from llm_gc.models.knowledge_entry import KnowledgeStatus

class PermanentGeneration:
    """Stores extracted knowledge entries from archived conversation turns.

    Handles contradiction detection (same topic_label = supersession)
    and emits events for the Context Visualizer.
    """

    def __init__(self, event_bus: EventBus):
        self._knowledge_entries: dict[str, list[KnowledgeEntry]] = {}
        self.event_bus = event_bus

    def add_knowledge_entry(self, knowledge_entry: KnowledgeEntry) -> None:
        superseded_entries = self._handle_knowledge_contradiction(knowledge_entry)
        self._knowledge_entries.setdefault(knowledge_entry.topic_label, []).append(knowledge_entry)
        self.event_bus.emit(Event(event_type=EventType.KNOWLEDGE_ENTRY_ADDED, data={"knowledge_entry": knowledge_entry}))
        if superseded_entries:
            self.event_bus.emit(Event(event_type=EventType.KNOWLEDGE_ENTRY_SUPERSEDED,
                                      data={"new_knowledge_entry": knowledge_entry, "superseded_knowledge_entries": superseded_entries}))

    def get_by_topic(self, topic_label: str) -> list[KnowledgeEntry]:
        return self._knowledge_entries.get(topic_label, [])

    def get_all_active_entries(self) -> list[KnowledgeEntry]:
        active_entries = []
        for entries in self._knowledge_entries.values():
            active_entries.extend([entry for entry in entries if entry.knowledge_status == KnowledgeStatus.ACTIVE])
        return active_entries

    def _handle_knowledge_contradiction(self, current_entry: KnowledgeEntry) -> list[KnowledgeEntry]:
        existing_knowledge_entries = self._knowledge_entries.get(current_entry.topic_label, [])
        superseded_entries = []
        for i, knowledge_entry in enumerate(existing_knowledge_entries):
            if knowledge_entry.knowledge_status == KnowledgeStatus.ACTIVE:
                existing_knowledge_entries[i] = replace(knowledge_entry, knowledge_status=KnowledgeStatus.SUPERSEDED)
                superseded_entries.append(knowledge_entry)
        return superseded_entries
