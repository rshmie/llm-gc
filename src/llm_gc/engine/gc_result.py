from typing import Optional
from pydantic import BaseModel, Field

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

    @property
    def tokens_saved(self) -> int:
        return self.tokens_before - self.tokens_in_final