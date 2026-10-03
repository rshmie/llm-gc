from enum import Enum


class GCStatus(Enum):
    COMPLETED = "completed"
    BYPASSED_ON_ERROR = "bypassed_on_error"
    BYPASSED_BELOW_THRESHOLD = "bypassed_below_threshold"
    # The turn is recorded; pressure persists, so the next turn triggers aging again.
    BYPASSED_AGING_IN_FLIGHT = "bypassed_aging_in_flight"
    # The session was closed while this pass was running; its result was dropped
    # so a closed session stays closed.
    DISCARDED_SESSION_CLOSED = "discarded_session_closed"
