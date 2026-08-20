# SweepClassification is defined in llm_gc.models (shared vocabulary). It is
# re-exported here so existing `from llm_gc.engine.sweep import SweepClassification`
# imports keep working — the definition moved, the import path did not.
from llm_gc.models.sweep_classification import SweepClassification

__all__ = ["SweepClassification"]
