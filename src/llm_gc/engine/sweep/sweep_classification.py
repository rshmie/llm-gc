from enum import Enum

class SweepClassification(Enum):
    """Action classifications assigned to messages by the Sweeper.

    KEEP - message stays in context as-is, no modification.
    COMPACT - message is passed to the Compactor for compression.
    ARCHIVE - message is removed from context but stored in long-term storage for potential future retrieval.
    """
    KEEP = "keep"
    COMPACT = "compact"
    ARCHIVE = "archive"