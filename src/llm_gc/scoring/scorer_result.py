from pydantic import BaseModel, Field
from llm_gc.scoring.relevance_type import RelevanceType

class ScorerResult(BaseModel):
    """Output of a single scorer's evaluation of one message.

    Holds the normalized score, which scorer produced it, a human-readable
    reason, and raw signals for the Context Visualizer.
    """
    score: float = Field(ge=0.0, le=1.0)
    scorer_name: RelevanceType
    reason: str
    signals: dict[str, str | float | int]