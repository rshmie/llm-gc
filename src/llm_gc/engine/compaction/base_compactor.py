from abc import ABC, abstractmethod
from collections.abc import Sequence

from llm_gc.engine.compaction.compaction_result import CompactionResult
from llm_gc.engine.compaction.compaction_strategy import CompactionStrategy
from llm_gc.events import EventBus
from llm_gc.models import Message


class BaseCompactor(ABC):
    """Turns a run of COMPACT-classified messages into one shorter summary.

    An ABC rather than a Protocol because `_format_summary_marker` is shared
    behaviour every compactor must use. See
    doc/design/sweeper-and-composition/compaction-mechanism.md.
    """

    COMPACTION_STRATEGY: CompactionStrategy

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus

    @abstractmethod
    async def compact(self, messages: list[Message]) -> CompactionResult:
        """Summarise a contiguous run of messages.

        Args:
            messages: A non-empty, adjacent run in turn-index order, every entry
                COMPACT-classified by the sweeper.

        Returns:
            A `CompactionResult` whose `summary` has role `"assistant"`, content
            opening with `_format_summary_marker`, `token_count` of the summary
            (not the originals), and `turn_index` of the run's *first* message so
            the composer can re-interleave it chronologically.

        Raises:
            CompactionRefused: The summary is unusable and must not replace the
                run. Callers leave the messages untouched.
        """
        ...

    def worth_compacting(self, messages: Sequence[Message]) -> bool:
        """Whether a summary of these turns could be shorter than they are.

        Asked before `compact`, so a run too small to beat this compactor's marker
        is simply left unsummarised, without a call or a refusal to report. True by
        default: a compactor without a floor is asked to try.
        """
        return True

    def _format_summary_marker(self, first_turn: int, last_turn: int) -> str:
        """Build the marker every summary opens with, so the model never sees a
        transformed turn presented as an original."""
        return f"[Compacted (compaction_strategy={self.COMPACTION_STRATEGY.value}) turns {first_turn}-{last_turn}]"
