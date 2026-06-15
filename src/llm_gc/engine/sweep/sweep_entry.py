from pydantic import BaseModel

from llm_gc.engine.sweep.sweep_classification import SweepClassification
from llm_gc.models import Message

class SweepEntry(BaseModel):
    """Per-message classification result produced by the sweep phase.

    Pairs a message with its assigned classification, relevance score,
    and metadata about whether an override rule was applied.
    """
    message: Message
    classification: SweepClassification
    relevance_score: float
    turn_index: int
    override_applied: bool = False
    override_reason: str | None = None
