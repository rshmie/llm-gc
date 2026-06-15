from enum import Enum


class GCStatus(Enum):
    COMPLETED = "completed"
    BYPASSED_ON_ERROR = "bypassed_on_error"
    BYPASSED_BELOW_THRESHOLD = "bypassed_below_threshold"
