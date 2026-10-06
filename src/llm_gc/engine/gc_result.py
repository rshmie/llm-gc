from typing import Optional
from pydantic import BaseModel, Field

from llm_gc.engine.aging_degradation import AgingDegradation
from llm_gc.engine.gc_status import GCStatus
from llm_gc.models.message import Message

class CutOutcome(BaseModel):
    """Where the turns a cut took ended up, and which turns condensed earlier it moved.

    The same pass re-plans every span, so it can show earlier turns with less
    detail (to fit the budget, or because they went quiet) or with more (because
    the conversation came back to them). Turn indices rather than counts, so a
    reader can name every message a collection moved.
    """

    model_config = {"frozen": True}

    kept: list[int] = Field(default_factory=list)        # new turns, quoted word for word
    compacted: list[int] = Field(default_factory=list)   # new turns, under their span's summary
    archived: list[int] = Field(default_factory=list)    # new turns, shown by their span's opening line and facts
    folded: list[int] = Field(default_factory=list)      # new turns, folded out of view
    lowered: list[int] = Field(default_factory=list)     # turns condensed earlier, now shown with less detail
    raised: list[int] = Field(default_factory=list)      # turns condensed earlier, now shown with more detail

    @property
    def taken(self) -> list[int]:
        """Every turn the cut took, at whichever level."""
        return sorted(self.kept + self.compacted + self.archived + self.folded)


class AbandonedCut(BaseModel):
    """A cut that was planned, then not made because it would not have shrunk the context.

    What it would have done, so a reader can see by how much it fell short and
    why. The outcome is that nothing changed; these numbers are the plan.
    """

    model_config = {"frozen": True}

    turns: list[int]            # the turns it would have taken out of the verbatim tail
    outcome: CutOutcome         # where those turns would have ended up
    tokens_taken: int           # what those turns cost now, word for word
    context_tokens_after: int   # what the whole context would have been


class GCResult(BaseModel):
    """Output of a garbage collection run.

    Contains the optimized message list and metrics showing token savings and processing cost.
    """
    final_messages: list[Message]
    gc_run_id: str
    status: GCStatus
    failure_reason: Optional[str] = None  # human-readable; None if not bypassed
    failure_stage: Optional[str] = None  # "score" | "sweep" | "age" | "sweep_old_gen" | None
    tokens_before: int = Field(ge=0)
    tokens_in_final: int = Field(ge=0)
    kept_count: int = Field(ge=0)
    compacted_count: int = Field(ge=0)
    archived_count: int = Field(ge=0)
    # The cut's turns by the level the plan gave them, in their own tokens.
    kept_tokens: int = Field(default=0, ge=0)
    compacted_tokens: int = Field(default=0, ge=0)
    archived_tokens: int = Field(default=0, ge=0)
    duration_ms: float = Field(ge=0)

    degradations: list[AgingDegradation] = Field(default_factory=list)
    """What this pass intended and did not manage - see `AgingDegradation`.

    Empty on a clean pass, which is the common case and why it defaults rather
    than being required. `kept_count` / `compacted_count` / `archived_count` count
    the cut's turns (the tail as kept) by the level the plan gave them; a consumer
    showing what a pass did to earlier turns as well reads `outcome`.

    `status` deliberately stays COMPLETED when this list is non-empty. Status
    answers "did the pass run"; this answers "did everything it intended happen".
    A fourth status value would make every existing `== COMPLETED` check silently
    stop matching, for information already carried here.
    """

    outcome: Optional[CutOutcome] = None
    """Set only when a cut was committed: where its turns ended up."""

    abandoned_cut: Optional[AbandonedCut] = None
    """Set only on a `BYPASSED_NO_SAVING` pass: the cut it planned and did not make."""

    @property
    def tokens_saved(self) -> int:
        return self.tokens_before - self.tokens_in_final