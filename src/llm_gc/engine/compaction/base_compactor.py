from abc import ABC, abstractmethod

from llm_gc.engine.compaction.compaction_result import CompactionResult
from llm_gc.engine.compaction.compaction_strategy import CompactionStrategy
from llm_gc.events import EventBus
from llm_gc.models import Message


class BaseCompactor(ABC):
    """Abstract contract for components that compact runs of COMPACT-classified messages.

    A compactor takes a run of adjacent messages the sweeper-and-composition has classified as COMPACT and produces a
    single shorter `Message` that preserves the gist. This is an abstract base class: it defines the contract every
    concrete compactor must satisfy, but does not implement compaction itself. Concrete implementations provide the
    actual summarization strategy. See ADR-0001 for design rationale.
    """

    COMPACTION_STRATEGY: CompactionStrategy
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus

    @abstractmethod
    def compact(self, messages: list[Message]) -> CompactionResult:
        """Produce a single summary `Message` from a run of adjacent COMPACT-classified messages.

        Called by `ContextComposer` during `gc.collect` and by `gc.update` when a compact-run needs to be summarized.
        Implementations vary in how they produce the summary (no-op passthrough, LLM call and others), but every implementation
        must satisfy the contract below. See ADR-0001 for the "honest about its own outputs" constraint on what compacted
        output is allowed to claim about itself.

        Args:
            messages: The run of messages to compact. Callers must ensure:
                - the list is non-empty,
                - every message is COMPACT-classified by the sweeper-and-composition,
                - the messages are adjacent in turn-index order
                  (a contiguous run, not a scattered selection).

        Returns:
            A `CompactionResult` whose fields satisfy the following
            invariants for every concrete implementation:

            - `summary.role`: `"assistant"` — the role the model accepts for transformed-but-not-original content.
            - `summary.content`: opens with a structurally recognizable marker that (a) identifies the message as a compacted
              summary, (b) names the original turn range covered, and (c) carries a provenance marker for the method used
              (e.g. `method=noop`, `method=llm`). Per ADR-0001 the model must never see a transformed turn pretending to be original.
            - `summary.token_count`: the token count of the *summarized* content (what the model will see), not the originals. This
              lets the orchestrator compute `tokens_saved` for `GCResult` without re-tokenizing.
            - `summary.turn_index`: the turn index of the *first* original message in the run. Required so the composer can re-interleave
              the summary with KEEP and ARCHIVE turns in chronological order; using the last (or any other) original index would
              scramble the conversation order.
            - `original_token_count`: sum of `token_count` across the input messages, used by the orchestrator to compute
              `tokens_saved`.
            - `method`: short string identifying which compactor produced the summary (e.g. `"noop"`, `"llm"`). Surfaced to the
              dashboard and post-session report so the developer can see *how* a turn was compacted, not just that it was.
        """

        ...

    def _format_summary_marker(self, first_turn: int, last_turn: int) -> str:
        """Build the structural marker every compacted summary opens with.

          Returns a string of the form `[Compacted (compaction_strategy=<name>) turns N-M]`.
          Concrete compactors call this to satisfy the `summary.content` contract
          in `BaseCompactor.compact` (recognizable as compacted, names turn range, carries provenance).
        """
        return  f"[Compacted (compaction_strategy={self.COMPACTION_STRATEGY.value}) turns {first_turn}-{last_turn}]"