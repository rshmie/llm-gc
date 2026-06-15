from enum import Enum


class CompactionStrategy(Enum):
    """Identifies which compactor produced a CompactionResult.

        Values are surfaced to the dashboard, post-session report, and any
        downstream consumer that needs to render compaction provenance.
    """

    NOOP = "noop"