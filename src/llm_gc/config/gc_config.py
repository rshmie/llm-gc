from pydantic import BaseModel, Field
from llm_gc.config.constants import (
    DEFAULT_MODEL, DEFAULT_GC_THRESHOLD, DEFAULT_KEEP_THRESHOLD, DEFAULT_MIN_CUT_TOKENS,
    DEFAULT_SWEEP_STRATEGY,
    DEFAULT_MIN_COMPACTABLE_TOKENS, DEFAULT_LAST_N_TURNS_TO_KEEP, DEFAULT_ARCHIVE_THRESHOLD, DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_MEMORY_RATIO, DEFAULT_MAX_MEMORY_FACTS
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
    keep_threshold: float = Field(default=DEFAULT_KEEP_THRESHOLD, ge=0.0, le = 1.0)
    archive_threshold: float = Field(default=DEFAULT_ARCHIVE_THRESHOLD, ge=0.0, le = 1.0)
    min_compactable_tokens: int = Field(default=DEFAULT_MIN_COMPACTABLE_TOKENS, ge=0)
    last_n_turns_to_keep: int = Field(default=DEFAULT_LAST_N_TURNS_TO_KEEP, ge=0)
    # A session is cut only once its compactable turns add up to this many tokens.
    min_cut_tokens: int = Field(default=DEFAULT_MIN_CUT_TOKENS, ge=0)
    # Permanent-generation injection. Zero on either disables it, which is how a
    # deployment opts out and how a benchmark measures the difference it makes.
    max_memory_ratio: float = Field(default=DEFAULT_MAX_MEMORY_RATIO, ge=0.0, le=1.0)
    max_memory_facts: int = Field(default=DEFAULT_MAX_MEMORY_FACTS, ge=0)

    @property
    def max_memory_tokens(self) -> int:
        """Token ceiling for the injected memory block, derived from the window.

        A property rather than a stored field so there is one source of truth: a
        stored absolute alongside the window would let the two drift, and the
        drift's symptom is a proxy that silently costs more than it saves.
        Consumers read this and never need to know it is derived.
        """
        return int(self.context_window * self.max_memory_ratio)