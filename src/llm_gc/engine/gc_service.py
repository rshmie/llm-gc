import logging

from llm_gc.engine.garbage_collector import GarbageCollector
from llm_gc.engine.gc_result import GCResult
from llm_gc.engine.generations.generational_memory import GenerationalMemory
from llm_gc.engine.sweep import Sweeper
from llm_gc.models import Message, SweepClassification
from llm_gc.scoring import RelevanceScorer
from llm_gc.session import SessionManager

logger = logging.getLogger(__name__)


class GcService:
    """Session-aware facade over the GC engine.

    Presents the two operations a caller (the proxy) uses: a synchronous
    hot-path `gc_collect` that assembles a cleaned context fast, and an
    asynchronous cold-path `gc_update` that does the expensive, stateful work
    between turns. It owns no pipeline logic itself - it holds the stateless
    pipeline (`GarbageCollector`), the session manager, generational memory, and
    the scorer/sweeper, and wires them together.
    """

    def __init__(self, gc_collector: GarbageCollector, session_manager: SessionManager,
                 generational_memory: GenerationalMemory, relevance_scorer: RelevanceScorer,
                 sweeper: Sweeper) -> None:
        self.gc_collector = gc_collector
        self.session_manager = session_manager
        self.generational_memory = generational_memory
        self.relevance_scorer = relevance_scorer
        self.sweeper = sweeper

    def gc_collect(self, messages: list[Message]) -> GCResult:
        """Hot path: run the stateless cleaning pipeline on a message list."""
        return self.gc_collector.collect(messages)

    async def gc_update(self, session_id: str, new_turn: Message) -> None:
        """Cold path, fire-and-forget: fold a new turn into session state and age
        any turns that have cooled from KEEP to COMPACT into the old generation.

        Runs after the LLM has responded, so it can afford a full mark pass. The
        whole read-modify-write is held under the session lock so a concurrent update
        for the same session cannot interleave and lose work.
        """
        async with self.session_manager.acquire(session_id) as state:
            state.messages.append(new_turn)

            # MARK: score, then sweep the whole conversation (same mark collect runs).
            scores = [self.relevance_scorer.score(message, state.messages) for message in state.messages]
            sweep_result = self.sweeper.sweep(state.messages, scores)

            # DIFF + ACT: a turn that was KEEP last pass and is COMPACT this pass
            # has cooled - promote it. Each promotion is a single-turn run, which
            # is trivially a contiguous run for the compactor; grouping adjacent
            # cooled turns into one summary is a later optimisation.
            for entry in sweep_result.sweep_entries:
                prior = state.last_classifications.get(entry.turn_index)
                if prior == SweepClassification.KEEP and entry.classification == SweepClassification.COMPACT:
                    self.generational_memory.promote_to_old_gen([entry.message])

            # RECORD the new baseline UNCONDITIONALLY: every turn's current
            # classification becomes next pass's "prior". A turn seen for the first
            # time is recorded here with no promotion, so it can only look like a
            # promotion once it has actually changed on a later pass.
            for entry in sweep_result.sweep_entries:
                state.last_classifications[entry.turn_index] = entry.classification
