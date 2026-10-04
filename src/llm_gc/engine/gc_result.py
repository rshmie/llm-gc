from typing import Optional
from pydantic import BaseModel, Field

from llm_gc.engine.aging_degradation import AgingDegradation
from llm_gc.engine.gc_status import GCStatus
from llm_gc.engine.sweep.sweep_result import SweepResult
from llm_gc.models.knowledge_entry import KnowledgeEntry
from llm_gc.models.message import Message

class AbandonedCut(BaseModel):
    """A cut that was planned, then not made because it would not have shrunk the context.

    What it would have done, so a reader can see by how much it fell short and
    why. The outcome is that nothing changed; these numbers are the plan.
    """

    model_config = {"frozen": True}

    turns: list[int]            # the turns it would have taken out of the verbatim tail
    quoted: int                 # of those, how many would have been quoted word for word
    summarised: int
    archived: int
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
    failure_stage: Optional[str] = None  # "score" | "sweep" | "compose" | None
    tokens_before: int = Field(ge=0)
    tokens_in_final: int = Field(ge=0)
    kept_count: int = Field(ge=0)
    compacted_count: int = Field(ge=0)
    archived_count: int = Field(ge=0)
    sweep_result: Optional[SweepResult] = None
    duration_ms: float = Field(ge=0)

    archived_knowledge: list[KnowledgeEntry] = Field(default_factory=list)
    """Facts recovered from the turns this pass archived, supersession applied.

    Filled by the stateless `GarbageCollector`, which stores nothing and so hands
    back everything it produced. Only this pass's archives: a caller that feeds
    `final_messages` back in has already dropped the earlier archived turns, so it
    accumulates these the same way it accumulates the conversation. Empty on the
    session path, where facts are committed to the session instead.
    """

    degradations: list[AgingDegradation] = Field(default_factory=list)
    """What this pass intended and did not manage - see `AgingDegradation`.

    Empty on a clean pass, which is the common case and why it defaults rather
    than being required. Note that `kept_count` / `compacted_count` /
    `archived_count` describe the *sweeper's decision*; this field is the only
    place the difference between decision and outcome is recorded, so a consumer
    rendering those counts has to read this too or it will draw compactions that
    never happened.

    `status` deliberately stays COMPLETED when this list is non-empty. Status
    answers "did the pass run"; this answers "did everything it intended happen".
    A fourth status value would make every existing `== COMPLETED` check silently
    stop matching, for information already carried here.
    """

    abandoned_cut: Optional[AbandonedCut] = None
    """Set only on a `BYPASSED_NO_SAVING` pass: the cut it planned and did not make."""

    @property
    def tokens_saved(self) -> int:
        return self.tokens_before - self.tokens_in_final