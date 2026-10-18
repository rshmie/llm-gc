"""A read-only view of one session, for the transparency layer.

Everything here is produced by `GcService` from a single snapshot, using the same
code that builds the prompt and plans a cut, so what a dashboard shows cannot
drift from what the engine does.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from llm_gc.engine.generations.front import Front
from llm_gc.engine.switch_policy import CacheState, SwitchDecision
from llm_gc.models import Message
from llm_gc.models.generation_snapshot import DetailLevel, GenerationSnapshot


def batch_worth_cutting(cut: int, compactable_tokens: int, min_cut_tokens: int) -> bool:
    """Whether a batch justifies a cut: something to condense, and at least `min_cut_tokens` of it.

    A batch the plan would keep entirely word for word condenses nothing - moving
    it into the front would cost a cache reset for no saving - so zero is never
    enough, even when `min_cut_tokens` is zero.
    """
    return cut > 0 and compactable_tokens > 0 and compactable_tokens >= min_cut_tokens


@dataclass(frozen=True)
class CutPreview:
    """What the next collection would do for the conversation as it stands.

    Planned the way a collection plans, without model calls and without announcing
    anything; summaries are estimated. A ready batch is therefore necessary, not
    sufficient: the collection still abandons the cut if it would not shrink the
    context, which only real summaries can show.
    """

    turns: tuple[int, ...]  # the verbatim turns the collection would take, in order
    spans: tuple[tuple[int, ...], ...]  # those turns, as the spans the collection would make of them
    levels: Mapping[int, DetailLevel]  # each of those turns: the level its span would be shown at
    scores: Mapping[int, float]  # each of those turns: its span's score
    compactable_tokens: int  # tokens of those turns the plan would not keep word for word
    min_cut_tokens: int
    # What the cut would save at estimated summary sizes: the check a collection makes
    # before paying for any summary. Zero or less, and the collection is skipped.
    estimated_saving: int = 0
    # Every span the old generation already holds, in the snapshot's order: the
    # level the collection would give it, and its score against what is discussed now.
    span_levels: tuple[DetailLevel, ...] = ()
    span_scores: tuple[float, ...] = ()

    @property
    def batch_ready(self) -> bool:
        return batch_worth_cutting(len(self.turns), self.compactable_tokens, self.min_cut_tokens)


@dataclass(frozen=True)
class SwitchPreview:
    """A staged cut, the measures the switch policy weighs, and what it would decide for the next request.

    The same measures `gc_switch` decides on, so a dashboard cannot show a different
    verdict from the one the request path reaches.
    """

    saving_tokens: int  # what its cut removes from the prompt
    rewritten_tokens: int  # what the provider would cache again: the front it renders plus the tail now
    stale_tokens: int  # what its turns the plan condenses add up to
    stale_bound_tokens: int  # the most stale weight the model may carry
    min_saving_share: float  # when warm, how many times `rewritten_tokens` the saving must be
    cache: CacheState  # the state it was decided for
    decision: SwitchDecision
    # Estimates included in the measures above: the thinking a switch stops sending (in `saving_tokens`),
    # and the thinking of the turns its cut condenses (in `stale_tokens`).
    thinking_dropped_tokens: int = 0
    stale_thinking_tokens: int = 0
    # Included in `rewritten_tokens`: what is cached again besides the front and the tail, such as instructions.
    rewrite_overhead_tokens: int = 0


@dataclass(frozen=True)
class SessionInspection:
    """One session at one moment: what it holds, what the model is sent, and what comes next."""

    session_id: str
    snapshot: GenerationSnapshot
    front: Front | None
    tail: tuple[Message, ...]
    next_cut: CutPreview
    aging_in_flight: bool
    last_activity: datetime
    context_window: int  # the window the recap is sized for, in the model's tokens
    recap_budget: int  # in LLM-GC's count: the window's share, converted at `token_rate`
    collection_enabled: bool  # False: recorded and shown, never collected, so no collection is forecast
    staged: SwitchPreview | None = None  # set while a prepared cut waits to be switched in
    token_rate: float | None = None  # the model's tokens per LLM-GC token; None until a reply measures it
    window_in_own_count: int = 0  # the window converted at `token_rate`: what the budgets are shares of
    thinking_tokens: int = 0  # estimated thinking the next request sends on top of `context_tokens`

    @property
    def context(self) -> list[Message]:
        """The conversation the next request forwards, before its new turn."""
        return ([self.front.message] if self.front is not None else []) + list(self.tail)

    @property
    def context_tokens(self) -> int:
        return sum(message.token_count for message in self.context)
