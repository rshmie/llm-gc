from pydantic import BaseModel

from llm_gc.models.message import Message
from llm_gc.scoring.scorer_result import ScorerResult

class RelevanceScorerResult(BaseModel):
    """Combined relevance result for a single message after all scorers have run.

    Holds the weighted combined score (used by the Sweeper for keep/compress/remove decisions)
    and the individual scorer results (used by the Context Visualizer for signal breakdown).
    """
    message: Message
    combined_score: float
    scorer_results: list[ScorerResult]
