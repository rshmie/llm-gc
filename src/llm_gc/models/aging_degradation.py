"""What a collection intended to do and did not manage.

A span made without its summary is kept rather than compacted, and one made
without its facts is archived by its index line alone; without these records the
front would be larger, or thinner, than anything reported says. See
doc/design/visibility/overview.md.
"""

from enum import Enum

from pydantic import BaseModel


class DegradationReason(Enum):
    """Why part of a collection fell short.

    Separate values because the operator's next action differs for each.
    """

    COMPACTION_REFUSED = "compaction_refused"
    """The model answered and the compactor judged the answer unusable. The guard
    working, not an error."""

    COMPACTION_FAILED = "compaction_failed"
    """The provider call failed."""

    ARCHIVE_FAILED = "archive_failed"
    """Extraction raised, so the span has no facts."""

    RECAP_OVER_BUDGET = "recap_over_budget"
    """Every span was folded out of view and the front still did not fit its budget."""


class AgingDegradation(BaseModel):
    """One thing a completed GC pass meant to do and didn't.

    Attributes:
        turn_indices: The turns of the span that fell short. A list, not a
            count, so a consumer can name which turns were affected.
        error_code: `LLMGCError.code` when the cause was one of ours; None
            otherwise, so consumers can branch on a stable string.
        detail: Short cause for logs and the dashboard. Carries the exception
            message only for an `LLMGCError`, whose messages are content-free by
            convention; anything else contributes only its type name, since event
            payloads reach the dashboard over HTTP.
    """

    model_config = {"frozen": True}

    reason: DegradationReason
    turn_indices: list[int]
    error_code: str | None = None
    detail: str
