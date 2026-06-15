from llm_gc.engine.compaction.compaction_strategy import CompactionStrategy
from llm_gc.models import Message
from pydantic import BaseModel, Field


class CompactionResult(BaseModel):
    """The output of a compaction operation.
     Wraps the summary 'Message' produced by a 'BaseCompactor' together with the metadata callers need to compute savings
     and surface provenance. Returned instead of a bare 'Message' so the compactor can report what it knows (original token
     count, which method produced the summary) without forcing the caller to recompute it. See ADR-0001 for the provenance contract.

     Attributes:
         summary: The compacted Message that will be inserted into the cleaned context.
             Its `token_count` reflects the summarized length, not the original. Its `content` carries a marker
             identifying it as a compacted summary.
         original_token_count: Sum of token counts of the original messages that fed this compaction. Used by the orchestrator
             to compute `tokens_saved` for `GCResult`.
         compaction_strategy: Provenance marker identifying which compactor mechanism produced this summary (`"noop"`, `"llm"`, ...).
            Surfaced to the dashboard and post-session report so the developer can see *how* a turn was compacted, not just that it was.
    """
    summary: Message
    original_token_count: int = Field(ge=0)
    compaction_strategy: CompactionStrategy

