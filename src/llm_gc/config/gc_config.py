from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Self

from llm_gc.config.constants import (
    DEFAULT_ARCHIVE_THRESHOLD,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_KEEP_THRESHOLD,
    DEFAULT_LAST_N_MESSAGES_TO_KEEP,
    DEFAULT_LEVEL_MARGIN,
    DEFAULT_MAX_RECAP_RATIO,
    DEFAULT_MAX_STALE_RATIO,
    DEFAULT_MIN_CUT_TOKENS,
    DEFAULT_MIN_SAVING_SHARE,
)


class GCConfig(BaseModel):
    """Central configuration for the GC engine.

    Holds thresholds, strategy selection, and tunable parameters
    that control how aggressively context is handled.
    """

    # A misspelt or retired setting fails here, instead of being silently ignored
    # while the caller believes it took effect.
    model_config = ConfigDict(extra="forbid")

    # Total input-token capacity of the target model; the recap's budget is a share of it.
    context_window: int = Field(default=DEFAULT_CONTEXT_WINDOW, gt=0)
    # The score lines between detail levels: at or above keep a span is shown word for
    # word, below archive only by its index line and facts, and summarised between.
    keep_threshold: float = Field(default=DEFAULT_KEEP_THRESHOLD, ge=0.0, le=1.0)
    archive_threshold: float = Field(default=DEFAULT_ARCHIVE_THRESHOLD, ge=0.0, le=1.0)
    # How far past a line a score must be to move a span across it, either way.
    level_margin: float = Field(default=DEFAULT_LEVEL_MARGIN, ge=0.0)
    # How many of the newest messages always stay as written: each user message, reply and
    # tool result counts as one, so one prompt with tool calls is several.
    last_n_messages_to_keep: int = Field(default=DEFAULT_LAST_N_MESSAGES_TO_KEEP, ge=0)
    # A cut is prepared only once the new turns it would condense add up to this many tokens.
    min_cut_tokens: int = Field(default=DEFAULT_MIN_CUT_TOKENS, ge=0)
    # Everything the front shows for condensed turns, as a share of the window.
    max_recap_ratio: float = Field(default=DEFAULT_MAX_RECAP_RATIO, ge=0.0, le=1.0)
    # With the cache warm, how many times what a switch makes the provider cache again it must remove.
    min_saving_share: float = Field(default=DEFAULT_MIN_SAVING_SHARE, ge=0.0)
    # The most stale weight the model may carry before a staged cut is switched in regardless, as a share of the window.
    max_stale_ratio: float = Field(default=DEFAULT_MAX_STALE_RATIO, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _bands_can_move_a_span(self) -> Self:
        if not self.archive_threshold < self.keep_threshold:
            raise ValueError(
                f"keep_threshold ({self.keep_threshold}) must be above archive_threshold ({self.archive_threshold})"
            )
        if not self.level_margin < (self.keep_threshold - self.archive_threshold) / 2:
            raise ValueError(
                f"level_margin ({self.level_margin}) must be under half the gap between the "
                "thresholds, or a score between them could never move a span"
            )
        return self

    @property
    def max_recap_tokens(self) -> int:
        """Token ceiling for the recap at the configured window."""
        return self.recap_budget()

    def recap_budget(self, context_window: int | None = None) -> int:
        """Token ceiling for the recap: its share of `context_window`, or of the configured window.

        `context_window` is counted as the recap is. Derived rather than stored so there is one
        source of truth: a stored absolute alongside the window would let the two drift, and the
        drift's symptom is a proxy that silently costs more than it saves.
        """
        return int((context_window or self.context_window) * self.max_recap_ratio)

    def stale_bound(self, context_window: int | None = None) -> int:
        """The most stale weight the model may carry: its share of `context_window`, or of the configured window.

        `context_window` is counted as the stale weight is.
        """
        return int((context_window or self.context_window) * self.max_stale_ratio)
