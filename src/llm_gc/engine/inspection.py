"""A read-only view of one session, for the transparency layer.

Everything here is produced by `GcService` from a single snapshot, using the same
code that builds the prompt and plans a cut, so what a dashboard shows cannot
drift from what the engine does.
"""

from dataclasses import dataclass
from datetime import datetime

from llm_gc.engine.generations.front import Front
from llm_gc.engine.sweep import SweepEntry
from llm_gc.models import Message
from llm_gc.models.generation_snapshot import GenerationSnapshot


def batch_worth_cutting(cut: int, compactable_tokens: int, min_cut_tokens: int) -> bool:
    """Whether a batch justifies a cut: something to compact, and at least `min_cut_tokens` of it.

    A batch of only KEEP turns has nothing to compact - quoting them would cost a
    cache reset for no saving - so zero compactable tokens is never enough, even
    when `min_cut_tokens` is zero.
    """
    return cut > 0 and compactable_tokens > 0 and compactable_tokens >= min_cut_tokens


@dataclass(frozen=True)
class CutPreview:
    """What the next aging pass would decide for the conversation as it stands.

    Computed by the pass's own marking step, without model calls and without
    announcing anything on the bus. A ready batch is necessary, not sufficient: the
    pass still abandons the cut if a summary is refused or the cut would not shrink
    the context, and only the compactor's output can show that.
    """

    entries: tuple[SweepEntry, ...]   # every verbatim turn, scored and classified, in order
    cut: int                          # leading verbatim turns a cut may remove; 0 when none is safe
    compactable_tokens: int           # COMPACT and ARCHIVE tokens before the cut point
    min_cut_tokens: int

    @property
    def batch_ready(self) -> bool:
        return batch_worth_cutting(self.cut, self.compactable_tokens, self.min_cut_tokens)


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

    @property
    def context(self) -> list[Message]:
        """The conversation the next request forwards, before its new turn."""
        return ([self.front.message] if self.front is not None else []) + list(self.tail)

    @property
    def context_tokens(self) -> int:
        return sum(message.token_count for message in self.context)
