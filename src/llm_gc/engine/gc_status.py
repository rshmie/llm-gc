from enum import Enum


class GCStatus(Enum):
    COMPLETED = "completed"
    BYPASSED_ON_ERROR = "bypassed_on_error"
    BYPASSED_BELOW_THRESHOLD = "bypassed_below_threshold"
    # The turn is recorded; pressure persists, so the next turn triggers aging again.
    BYPASSED_AGING_IN_FLIGHT = "bypassed_aging_in_flight"
    # The turns classified for compaction do not yet add up to a batch worth a cut.
    BYPASSED_BATCH_TOO_SMALL = "bypassed_batch_too_small"
    # The session was closed while this pass was running; its result was dropped
    # so a closed session stays closed.
    DISCARDED_SESSION_CLOSED = "discarded_session_closed"
