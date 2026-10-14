from enum import Enum


class GCStatus(Enum):
    # A collection was switched in: the prompt changed.
    COMPLETED = "completed"
    # A pass prepared a cut and staged it; the prompt is unchanged until the switch policy switches it in.
    STAGED = "staged"
    BYPASSED_ON_ERROR = "bypassed_on_error"
    # The turns are recorded; the pass after the running one sees them.
    BYPASSED_AGING_IN_FLIGHT = "bypassed_aging_in_flight"
    # The turns classified for compaction do not yet add up to a batch worth a cut.
    BYPASSED_BATCH_TOO_SMALL = "bypassed_batch_too_small"
    # A cut was planned and compacted, but would not have made the context smaller,
    # so it was abandoned.
    BYPASSED_NO_SAVING = "bypassed_no_saving"
    # The session was closed while this pass was running; its result was dropped
    # so a closed session stays closed.
    DISCARDED_SESSION_CLOSED = "discarded_session_closed"
    # The client rewound its history while this pass was running, past the turns it prepared;
    # they are no longer the session's oldest, so nothing was staged.
    DISCARDED_HISTORY_CHANGED = "discarded_history_changed"
    # Collection is turned off for this session: it is recorded and shown, never collected.
    BYPASSED_NOT_COLLECTED = "bypassed_not_collected"
    # LLM-GC is yielding to the upstream's rate limit: none of its own model calls until the pause ends.
    BYPASSED_PAUSED = "bypassed_paused"
