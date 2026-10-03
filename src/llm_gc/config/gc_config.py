from pydantic import BaseModel, Field
from llm_gc.config.constants import (
    DEFAULT_MODEL, DEFAULT_GC_THRESHOLD,
    DEFAULT_SWEEP_STRATEGY,
    DEFAULT_MIN_COMPACTABLE_TOKENS, DEFAULT_LAST_N_TURNS_TO_KEEP, DEFAULT_ARCHIVE_THRESHOLD, DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_MEMORY_TOKENS, DEFAULT_MAX_MEMORY_FACTS
)

class GCConfig(BaseModel):
    """Central configuration for the GC engine.

    Holds thresholds, strategy selection, and tunable parameters
    that control how aggressively context is handled.
    """
    model_name: str = Field(default=DEFAULT_MODEL)
    # Total input-token capacity of the target model. GC engages when the conversation reaches
    # context_window * gc_threshold tokens.
    context_window: int = Field(default=DEFAULT_CONTEXT_WINDOW, gt=0)
    gc_threshold: float = Field(default=DEFAULT_GC_THRESHOLD, ge=0.0, le = 1.0)
    sweep_strategy: str = Field(default=DEFAULT_SWEEP_STRATEGY)
    keep_threshold: float = Field(default=DEFAULT_GC_THRESHOLD, ge=0.0, le = 1.0)
    archive_threshold: float = Field(default=DEFAULT_ARCHIVE_THRESHOLD, ge=0.0, le = 1.0)
    min_compactable_tokens: int = Field(default=DEFAULT_MIN_COMPACTABLE_TOKENS, ge=0)
    last_n_turns_to_keep: int = Field(default=DEFAULT_LAST_N_TURNS_TO_KEEP, ge=0)
    # Permanent-generation injection. Zero on either disables it, which is how a
    # deployment opts out and how a benchmark measures the difference it makes.
    max_memory_tokens: int = Field(default=DEFAULT_MAX_MEMORY_TOKENS, ge=0)
    max_memory_facts: int = Field(default=DEFAULT_MAX_MEMORY_FACTS, ge=0)