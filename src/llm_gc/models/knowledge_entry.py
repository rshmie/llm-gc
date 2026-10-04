from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class KnowledgeType(Enum):
    """Semantic classification of extracted knowledge."""
    FACT = "fact"
    DECISION = "decision"
    PREFERENCE = "preference"
    RAW = "raw"

# The label of an entry whose statement names no topic: "we decided to ship on Friday".
GENERAL_TOPIC = "general"

class KnowledgeStatus(Enum):
    """Lifecycle status of a knowledge entry in permanent generation."""
    ACTIVE = "active"
    SUPERSEDED = "superseded"

@dataclass(frozen=True)
class KnowledgeEntry:
    """An extracted unit of knowledge from a conversation turn.

    Stored in the permanent generation after a turn is archived.
    Immutable - supersession creates a new copy via dataclasses.replace().

    `message_turn` is the turn the entry is labelled with. `source_turns` names
    every turn it was derived from when that is more than one - facts extracted
    from a summary of turns 14-19 are labelled 14 and come from all six. Empty
    means the entry comes from `message_turn` alone.
    """
    message_turn: int
    content: str
    topic_label: str
    knowledge_type: KnowledgeType
    knowledge_status: KnowledgeStatus = KnowledgeStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    source_turns: tuple[int, ...] = ()

    @property
    def covered_turns(self) -> tuple[int, ...]:
        """Every turn this entry was derived from."""
        return self.source_turns or (self.message_turn,)

    @property
    def can_supersede(self) -> bool:
        """Whether filing this entry retires the active entries on its topic.

        A verbatim excerpt claims nothing, and an entry with no topic cannot be
        matched to the claim it replaces: matched on "general", every topic-less
        decision retired every earlier one, however unrelated.
        """
        return self.knowledge_type is not KnowledgeType.RAW and self.topic_label != GENERAL_TOPIC

