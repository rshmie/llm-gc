from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class KnowledgeType(Enum):
    """Semantic classification of extracted knowledge."""
    FACT = "fact"
    DECISION = "decision"
    PREFERENCE = "preference"
    RAW = "raw"

class KnowledgeStatus(Enum):
    """Lifecycle status of a knowledge entry in permanent generation."""
    ACTIVE = "active"
    SUPERSEDED = "superseded"

@dataclass(frozen=True)
class KnowledgeEntry:
    """An extracted unit of knowledge from a conversation turn.

    Stored in the permanent generation after a turn is archived.
    Immutable - supersession creates a new copy via dataclasses.replace().
    """
    message_turn: int
    content: str
    topic_label: str
    knowledge_type: KnowledgeType
    knowledge_status: KnowledgeStatus = KnowledgeStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)

