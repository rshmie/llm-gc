"""What a GC pass intended to do and did not manage.

A pass can complete and still fall short. The sweeper classifies a run as COMPACT,
the compactor refuses the summary it gets back, and the turns stay in the young
generation verbatim. Nothing is lost and the context is correct — but it is larger
than the pass set out to make it, and the counts on `GCResult` describe the
sweeper's *decision*, not the outcome.

Without this record a consumer reading `compacted_count: 5` would draw five
compactions that never happened. That is the project's own premise failing on its
own internals, which is worse than an ordinary bug: the transparency layer would
be asserting something false about what the model is being sent.
"""

from enum import Enum

from pydantic import BaseModel


class DegradationReason(Enum):
    """Why a turn that should have aged did not.

    Separate values rather than one "aging failed", because the operator's next
    action differs: a refusal means the run or the cap was wrong and the next pass
    will try a different run; a provider failure means check the provider; an
    archive failure means check extraction and storage.
    """

    COMPACTION_REFUSED = "compaction_refused"
    """The model answered, and the compactor judged the answer unusable — it was
    truncated, empty, or no shorter than the run it would replace. Not an error:
    the guard working. The run stays verbatim."""

    COMPACTION_FAILED = "compaction_failed"
    """The call to the provider failed — timeout, rate limit, bad response. The
    run stays verbatim."""

    ARCHIVE_FAILED = "archive_failed"
    """Extraction or permanent-generation storage raised. The turn stays verbatim,
    which is the safe direction: a failed archive that still dropped the turn
    would lose it outright."""


class AgingDegradation(BaseModel):
    """One thing a completed GC pass meant to do and didn't.

    Frozen: a record of something that already happened has no correct reason to
    be edited afterwards.
    """

    model_config = {"frozen": True}

    reason: DegradationReason

    turn_indices: list[int]
    """The turns that stayed in the young generation as a result. A list rather
    than a count, because a consumer that wants to show *which* turns were
    affected cannot recover identity from a number — the same reason
    `MESSAGE_PROMOTED_TO_OLD_GEN` carries `source_turn_indices`."""

    error_code: str | None = None
    """`LLMGCError.code` when the cause was one of ours, so a consumer can branch
    on a stable string rather than parse prose. None for anything else."""

    detail: str
    """Short human-readable cause, for logs and the dashboard.

    Carries the exception's message only when the cause was an `LLMGCError`,
    whose messages are content-free by the convention documented in
    `exceptions.py`. For any other exception — including one raised by a
    third-party compactor we did not write — only the type name is recorded.
    Event payloads reach the dashboard over HTTP and carry no raw conversation
    content (CLAUDE.md §11), and an arbitrary exception message is not a
    boundary we control.
    """
