from enum import Enum

class RelevanceType(Enum):
    """The five dimensions of relevance measured by the scoring pipeline.

    Each value corresponds to one scorer that evaluates a different
    aspect of a message's importance to the conversation.
    """
    RECENCY = "recency"
    SIMILARITY = "similarity"
    DENSITY = "density"
    DECISION = "decision"
    REFERENCE = "reference"