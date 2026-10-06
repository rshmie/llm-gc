from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class KnowledgeType(Enum):
    """Semantic classification of extracted knowledge."""
    FACT = "fact"
    DECISION = "decision"
    PREFERENCE = "preference"

# The label of an entry whose statement names no topic: "we decided to ship on Friday".
GENERAL_TOPIC = "general"

class KnowledgeStatus(Enum):
    """Lifecycle status of a knowledge entry in permanent generation."""
    ACTIVE = "active"
    SUPERSEDED = "superseded"

@dataclass(frozen=True)
class KnowledgeEntry:
    """One fact, decision or preference read from a turn, filed in the permanent generation.

    Read from the turn's own words when its span is made, and kept for the life of
    the session. Immutable - supersession creates a new copy via
    dataclasses.replace(). `message_turn` is the turn it was read from.
    """
    message_turn: int
    content: str
    topic_label: str
    knowledge_type: KnowledgeType
    knowledge_status: KnowledgeStatus = KnowledgeStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def can_supersede(self) -> bool:
        """Whether filing this entry retires the active entries on its topic.

        An entry with no topic cannot be matched to the claim it replaces: matched
        on "general", every topic-less decision would retire every earlier one,
        however unrelated.
        """
        return self.topic_label != GENERAL_TOPIC

