import logging

from llm_gc.engine.generations.generational_memory import GenerationalMemory
from llm_gc.engine.sweep import Sweeper
from llm_gc.models import Message, SweepClassification
from llm_gc.scoring import RelevanceScorer
from llm_gc.session import SessionManager

logger = logging.getLogger(__name__)


class GcService:
    """Session-aware engine facade: the two operations the proxy calls.

    - `gc_collect` (hot path): assembles a session's cleaned context by stitching
      the old-generation summaries together with the young-generation turns that
      `gc_update` has been maintaining. Fast, because it only reads and
      concatenates - no scoring happens on this path.
    - `gc_update` (cold path, fire-and-forget): after a response, folds the new
      turn into session state and ages turns out of the young generation -
      COMPACT turns become old-gen summaries, ARCHIVE turns have their facts
      pulled into permanent gen.

    It owns no scoring or sweeping logic itself; it holds the scorer, sweeper,
    generational memory, and session manager, and wires them together under the
    per-session lock.
    """

    def __init__(self, session_manager: SessionManager, generational_memory: GenerationalMemory,
                 relevance_scorer: RelevanceScorer, sweeper: Sweeper) -> None:
        self.session_manager = session_manager
        self.generational_memory = generational_memory
        self.relevance_scorer = relevance_scorer
        self.sweeper = sweeper

    async def gc_collect(self, session_id: str) -> list[Message]:
        """Hot path: assemble the cleaned context for a session.

        Stitches the old-generation summaries together with the young-generation
        turns, ordered by turn index so the conversation still reads
        chronologically (an old-gen summary carries the turn index of the first
        turn in its run, so it slots back into the right place). No scoring here -
        `gc_update` already did that work; collect only reads and concatenates.
        Held under the session lock so it never reads state that an in-flight
        update is halfway through rewriting.
        """
        async with self.session_manager.acquire(session_id) as state:
            assembled = self.generational_memory.get_old_gen() + state.messages
            assembled.sort(key=lambda message: message.turn_index)
            return assembled

    async def gc_update(self, session_id: str, new_turn: Message) -> None:
        """Cold path, fire-and-forget: fold a new turn into session state and age
        turns out of the young generation based on this pass's classification.

        Runs after the LLM has responded, so it can afford a full mark pass. The
        whole read-modify-write is held under the session lock so a concurrent
        update for the same session cannot interleave and lose work.
        """
        async with self.session_manager.acquire(session_id) as state:
            state.messages.append(new_turn)

            # MARK: score, then sweep the whole young generation.
            scores = [self.relevance_scorer.score(message, state.messages) for message in state.messages]
            sweep_result = self.sweeper.sweep(state.messages, scores)

            # ACT: each turn's current classification is its instruction. COMPACT ->
            # summarise into old gen; ARCHIVE -> extract its facts into permanent
            # gen. Both then leave the young generation; KEEP stays. State-based, not
            # a diff against last pass: an acted-on turn is *removed*, so it can never
            # be re-processed - nothing to remember (that is the monitor's job; it
            # observes without moving, gc_update moves).
            #
            # Each turn is aged independently, and a turn joins `departed` only AFTER
            # its produce-action succeeds. So a failure (e.g. a Phase-7 LLMCompactor
            # API error) loses nothing - the turn stays in young for next pass - and
            # never leaves a summary in old gen whose original is also still live
            # (which would be re-promoted into a duplicate). One failing turn must not
            # abort the whole update, so we log loudly and continue: the same
            # loud-but-contained posture as the EventBus. `except Exception`, not
            # bare, so KeyboardInterrupt / SystemExit still propagate.
            departed_turn_indices: set[int] = set()
            for entry in sweep_result.sweep_entries:
                try:
                    if entry.classification == SweepClassification.COMPACT:
                        self.generational_memory.promote_to_old_gen([entry.message])
                        departed_turn_indices.add(entry.turn_index)
                    elif entry.classification == SweepClassification.ARCHIVE:
                        self.generational_memory.archive_message(entry.message)
                        departed_turn_indices.add(entry.turn_index)
                except Exception:
                    logger.exception(
                        "Failed to age turn out of young generation; leaving it for retry",
                        extra={"session_id": session_id, "turn_index": entry.turn_index,
                               "classification": entry.classification.value},
                    )

            # The departed turns leave the young generation - their summaries (old
            # gen) or facts (permanent gen) now stand in for them, so the assembler
            # must not also carry the originals. Rebuild keeping survivors rather than
            # deleting in place - mutating a list while iterating it skips elements.
            if departed_turn_indices:
                state.messages = [m for m in state.messages if m.turn_index not in departed_turn_indices]

            logger.debug("gc_update completed", extra={"session_id": session_id,
                                                       "turns_aged_out": len(departed_turn_indices),
                                                       "young_remaining": len(state.messages)})
