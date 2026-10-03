from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from llm_gc.engine import AgingDegradation, GCStatus
from llm_gc.models import KnowledgeEntry


class TransitionType(Enum):
    """Kind of generation-lifecycle transition a GenerationTransition records.

    A transition is an event, not a state - it records that something moved,
    at a specific moment, not what its current status is. PROMOTED and ARCHIVED
    intentionally share vocabulary with SweepClassification and KnowledgeStatus:
    they name the same real-world occurrence from the monitor's historical-record
    angle rather than the classifier's decision-time angle. PROMOTED has no
    analogue anywhere else in the codebase because detecting it means comparing
    a turn's classification across two collect() passes - nothing before this
    monitor has ever needed to look at more than one pass at a time.
    """
    PROMOTED = "promoted"      # a turn's classification moved KEEP -> COMPACT
    ARCHIVED = "archived"      # a turn was removed from the live context after extraction
    SUPERSEDED = "superseded"  # a knowledge entry was contradicted by a newer one


class TokenBudgetSignal(BaseModel):
    """How crowded the live context is right now, relative to the model's budget.

    Every field is read from the most recent GCResult and GCConfig - this
    signal never computes anything a component upstream hasn't already computed.
    """
    current_tokens: int          # gc_result.tokens_in_final
    context_window: int          # gc_config.context_window
    pressure_ratio: float        # current_tokens / context_window
    current_msg_turn_index: int  # gc_result.sweep_result.sweep_entries[-1].turn_index
    # ^ how far into the conversation this is - not how many turns are
    # currently visible, which is a different (and smaller) number once any
    # archiving has happened.


class GCBreakdownSignal(BaseModel):
    """What the most recent GC pass actually did, and whether it succeeded.

    Answers "how much did the last cleanup change" - a delta/action question,
    distinct from TokenBudgetSignal's "how crowded is context right now" and
    from GenerationLifecycleSignal's "where does everything currently live".
    Every field is copied from the last GCResult, never recomputed. status,
    duration, and failure detail are included alongside the counts (not just
    the counts alone) so a bypassed or failed pass is visible on the dashboard,
    rather than silently hiding behind a breakdown that still looks normal.
    """
    keep_count: int
    compact_count: int
    archive_count: int
    keep_tokens: int
    compact_tokens: int
    archive_tokens: int
    tokens_before: int          # gc_result.tokens_before - raw size going into the last pass
    tokens_saved: int           # gc_result.tokens_saved - copied, not recomputed
    gc_run_id: str
    gc_status: GCStatus
    gc_completed_at: float      # the GC_FINISHED event's own timestamp, not this snapshot's build time
    duration_ms: float
    failure_reason: str | None  # None unless gc_status is BYPASSED_ON_ERROR
    failure_stage: str | None   # "score" | "sweep" | "compose" | None

    degradations: list[AgingDegradation] = Field(default_factory=list)
    """What the pass intended and did not manage - copied from GCResult.

    Carried alongside the counts because the counts are the sweeper's *decision*:
    a pass whose compactions were all refused reports a non-zero compact_count and
    files nothing. A dashboard rendering compact_count without this would draw
    compactions that never happened, which is exactly the kind of confident-looking
    falsehood this component exists to prevent. Empty on a clean pass.
    """

    @property
    def is_degraded(self) -> bool:
        """Whether this pass fell short of its own plan.

        A named property rather than `len(signal.degradations) > 0` at each call
        site, for the same reason `LLMResponse.is_complete` is one: this is the
        check most likely to be forgotten, and the one whose absence fails
        silently.
        """
        return bool(self.degradations)


class GenerationTransition(BaseModel):
    """A single, timestamped record that one transition happened.

    Deliberately not a field on Message: a field can only hold one turn's
    current status, overwritten every time it changes, which can't represent
    several things happening to the same turn over its life. For SUPERSEDED
    specifically, a snapshot of current state cannot show this occurred at
    all - superseded entries are excluded from the active view by design, so
    this log is the only place that moment is ever visible. One instance is
    one event: five promotions are five separate GenerationTransition
    objects, never one object with a count.
    """
    transition_type: TransitionType
    turn_index: int    # set for PROMOTED / ARCHIVED / SUPERSEDED
    topic_label: str | None   # set for SUPERSEDED, None for PROMOTED / ARCHIVED
    occurred_at: float


class GenerationLifeCycleSignal(BaseModel):
    """Where every turn currently lives, and how it got there.

    "Generation" names the conceptual young/old/permanent tiers this project's
    generational-memory model is built around - not a claim that this signal
    manages memory itself. Young/old-gen counts, tokens, and turn indices are
    read straight from the last GCResult's sweep_result (KEEP -> young,
    COMPACT -> old); none of it is storage this signal or GenerationalMemory
    independently tracks. permanent_gen_entries is the one part backed by real,
    standalone storage, because an archived turn is removed from the live
    context entirely - without that storage, its knowledge would be gone.
    """
    young_gen_count: int
    old_gen_count: int
    permanent_gen_count: int
    young_gen_tokens: int
    old_gen_tokens: int
    young_gen_turn_indices: list[int]
    old_gen_turn_indices: list[int]
    permanent_gen_entries: list[KnowledgeEntry]
    recent_transitions: list[GenerationTransition]  # bounded to the last N by the monitor


class ContextHealth(BaseModel):
    """An immutable, point-in-time snapshot of context state.

    Rebuilt whole and swapped by reference on every relevant event - never
    mutated field-by-field - so a reader can never observe a torn, half-updated
    state. token_budget, gc_breakdown, and transformation_ratio are None until
    the first GC pass has ever completed: reporting a fabricated zero would be
    a false claim ("measured, and it's low") rather than an honest "not measured
    yet". generation_lifecycle is never None - young/old/permanent membership is
    a standing fact about the world, true (and correctly empty) even before any
    collect() call has ever run.
    """
    model_config = ConfigDict(frozen=True)
    token_budget: TokenBudgetSignal | None
    gc_breakdown: GCBreakdownSignal | None
    generation_lifecycle: GenerationLifeCycleSignal
    transformation_ratio: float | None  # internal baseline signal - never surfaced as a headline number
    snapshot_taken_at: float            # when this snapshot was built, distinct from gc_breakdown.gc_completed_at
