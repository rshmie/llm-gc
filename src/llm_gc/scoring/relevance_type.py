from enum import Enum

class RelevanceType(Enum):
    """The measures a span's score is combined from.

    Recency and similarity (relevance to what is being discussed now) make up its
    liveness; decision and density its importance.
    """
    RECENCY = "recency"
    SIMILARITY = "similarity"
    DENSITY = "density"
    DECISION = "decision"
