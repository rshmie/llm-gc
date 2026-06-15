import re

from llm_gc.events import EventBus, Event, EventType
from llm_gc.models import KnowledgeEntry, Message
from llm_gc.models.knowledge_entry import KnowledgeType


class KnowledgeExtractor:
    EXTRACTION_PATTERNS = {
        KnowledgeType.FACT: [
            re.compile(r"(?:the\s+)?(?P<topic>[\w\s]+?)\s+(?:is|are)\s+(?P<content>.+)", re.IGNORECASE),
            re.compile(r"(?:we(?:'re|\s+are)\s+)?using\s+(?P<content>\w+)\s+for\s+(?P<topic>[\w\s]+)",
                       re.IGNORECASE),
            re.compile(r"(?P<topic>[\w\s]+?)\s+runs?\s+on\s+(?P<content>.+)", re.IGNORECASE),
        ],
        KnowledgeType.DECISION: [
            re.compile(r"(?:we\s+)?decided\s+to\s+(?P<content>.+?)(?:\s+for\s+(?P<topic>[\w\s]+))?$",
                       re.IGNORECASE),
            re.compile(r"let(?:'s|\s+us)\s+go\s+with\s+(?P<content>.+?)(?:\s+for\s+(?P<topic>[\w\s]+))?$",
                       re.IGNORECASE),
            re.compile(r"(?:we\s+)?chose\s+(?P<content>.+?)\s+over\s+.+?(?:\s+for\s+(?P<topic>[\w\s]+))?$",
                       re.IGNORECASE),
        ],
        KnowledgeType.PREFERENCE: [
            re.compile(r"(?:i|we)\s+prefer\s+(?P<content>.+?)(?:\s+for\s+(?P<topic>[\w\s]+))?$", re.IGNORECASE),
            re.compile(
                r"(?:i|we)(?:'d|\s+would)\s+rather\s+(?:use\s+)?(?P<content>.+?)(?:\s+for\s+(?P<topic>[\w\s]+))?$",
                re.IGNORECASE),
        ],
    }

    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus

    def extract_knowledge(self, message: Message, message_turn: int) -> list[KnowledgeEntry]:
        knowledge_entries = []
        for knowledge_type, patterns in self.EXTRACTION_PATTERNS.items():
            for pattern in patterns:
                match = pattern.search(message.content)
                if match:
                    topic_label = match.group("topic") or "general"
                    content = match.group("content").strip()
                    knowledge_entries.append(KnowledgeEntry(
                        message_turn=message_turn,
                        content=content,
                        topic_label=topic_label,
                        knowledge_type=knowledge_type,
                    ))
        if knowledge_entries:
            self.event_bus.emit(Event(event_type=EventType.KNOWLEDGE_ENTRIES_EXTRACTED, data={"entries": knowledge_entries, "message_turn": message_turn}))
        return knowledge_entries