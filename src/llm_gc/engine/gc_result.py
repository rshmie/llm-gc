from typing import Optional
from pydantic import BaseModel, Field

from llm_gc.engine.aging_degradation import AgingDegradation
from llm_gc.engine.gc_status import GCStatus
from llm_gc.engine.sweep.sweep_result import SweepResult
from llm_gc.models.message import Message

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

    @property
    def tokens_saved(self) -> int:
        return self.tokens_before - self.tokens_in_final