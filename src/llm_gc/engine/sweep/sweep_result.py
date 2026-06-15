from pydantic import BaseModel

from llm_gc.engine.sweep.sweep_classification import SweepClassification
from llm_gc.engine.sweep.sweep_entry import SweepEntry

class SweepResult(BaseModel):
    """Aggregate output of the sweep phase.

    Contains per-message classification entries plus summary metrics
    (counts, token totals, processing time) for observability and
    downstream consumption by the Compactor.
    """
    sweep_entries: list[SweepEntry]
    classification_counts: dict[SweepClassification, int]
    processing_time_ms: float
    total_keep_tokens: int
    total_compact_tokens: int
    total_archive_tokens: int = 0
