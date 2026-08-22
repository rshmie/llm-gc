import logging
import uuid
from time import perf_counter

from llm_gc.config.gc_config import GCConfig
from llm_gc.engine.gc_result import GCResult
from llm_gc.engine.gc_status import GCStatus
from llm_gc.engine.generations.generational_memory import GenerationalMemory
from llm_gc.engine.sweep import Sweeper, SweepResult
from llm_gc.events import Event, EventBus, EventType
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
      pulled into permanent gen. This is the pass that does the real GC work, so
      this is the pass that reports it on the event bus.

    It owns no scoring or sweeping logic itself; it holds the scorer, sweeper,
    generational memory, and session manager, and wires them together under the
    per-session lock.
    """

    def __init__(self, session_manager: SessionManager, generational_memory: GenerationalMemory,
                 relevance_scorer: RelevanceScorer, sweeper: Sweeper, event_bus: EventBus,
                 gc_config: GCConfig) -> None:
        self.session_manager = session_manager
        self.generational_memory = generational_memory
        self.relevance_scorer = relevance_scorer
        self.sweeper = sweeper
        self.event_bus = event_bus
        self.gc_config = gc_config

    def _assemble(self, young_messages: list[Message]) -> list[Message]:
        """Stitch old-generation summaries together with young turns, in turn order.

        The single definition of "the context this session would send". Both
        `gc_collect` (which returns it) and `gc_update` (which reports it as
        `final_messages`) call this, so the number the dashboard shows and the
        payload the proxy forwards can never drift apart.

        An old-gen summary carries the turn index of the first turn in its run,
        so sorting by `turn_index` slots it back into its original position and
        the conversation still reads chronologically.
        """
        assembled = self.generational_memory.get_old_gen() + young_messages
        assembled.sort(key=lambda message: message.turn_index)
        return assembled

    async def gc_collect(self, session_id: str) -> list[Message]:
        """Hot path: assemble the cleaned context for a session.

        No scoring here - `gc_update` already did that work; collect only reads
        and concatenates. Held under the session lock so it never reads state
        that an in-flight update is halfway through rewriting.

        Emits nothing: this path changes no state, and the health monitor
        observes state changes. A read that reports itself as a "GC run" would
        overwrite the last real run's breakdown with a no-op.
        """
        async with self.session_manager.acquire(session_id) as state:
            return self._assemble(state.messages)

    async def gc_update(self, session_id: str, new_turn: Message) -> None:
        """Cold path, fire-and-forget: fold a new turn into session state and,
        if the context is under pressure, age turns out of the young generation
        based on this pass's classification.

        Runs after the LLM has responded, so it can afford a full mark pass. The
        whole read-modify-write is held under the session lock so a concurrent
        update for the same session cannot interleave and lose work.

        This is the pass that does the real GC work, so this is the pass that
        emits GC_FINISHED. `gc_collect` only reads. There is exactly one emit and
        it happens *after* the lock is released: `EventBus.emit` runs every
        subscriber synchronously, and holding a session's lock through arbitrary
        subscriber code would let a slow dashboard handler push other requests
        for that session past `session_lock_timeout_ms`.
        """
        start = perf_counter()
        gc_run_id = str(uuid.uuid4())

        async with self.session_manager.acquire(session_id) as state:
            # The new turn is always recorded - that is bookkeeping, not aging.
            state.messages.append(new_turn)

            # Measure the prompt as it stands *before* this pass ages anything,
            # so tokens_saved reports what this pass actually removed. Measured
            # on the assembled context, not on young alone: old-gen summaries are
            # real tokens in the real prompt, and a budget meter that omitted them
            # would under-report pressure in the reassuring direction.
            tokens_before = sum(message.token_count for message in self._assemble(state.messages))
            threshold_tokens = int(self.gc_config.context_window * self.gc_config.gc_threshold)

            if tokens_before < threshold_tokens:
                # Aging is triggered by pressure, not by arrival. Below the
                # threshold there is no reason to spend a compaction on a turn
                # nobody needed compacted: the turn stays verbatim, which is
                # strictly better context and - once the compactor is a real LLM -
                # strictly cheaper. Same guard GarbageCollector.collect applies,
                # measured on the same assembled context the caller will send.
                untouched = self._assemble(state.messages)
                gc_result = GCResult(
                    final_messages=untouched, gc_run_id=gc_run_id,
                    status=GCStatus.BYPASSED_BELOW_THRESHOLD,
                    tokens_before=tokens_before,
                    tokens_in_final=sum(message.token_count for message in untouched),
                    kept_count=len(untouched), compacted_count=0, archived_count=0,
                    sweep_result=None, duration_ms=(perf_counter() - start) * 1000,
                )
                logger.debug("gc_update bypassed below threshold",
                             extra={"session_id": session_id, "gc_run_id": gc_run_id,
                                    "tokens": tokens_before, "threshold_tokens": threshold_tokens})
            else:
                current_stage = "score"
                try:
                    # MARK: score, then sweep the whole young generation.
                    scores = [self.relevance_scorer.score(message, state.messages)
                              for message in state.messages]
                    current_stage = "sweep"
                    sweep_result = self.sweeper.sweep(state.messages, scores)
                    current_stage = "age"
                    departed_turn_indices = self._age_out_turns(session_id, sweep_result)

                    # The departed turns leave the young generation - their
                    # summaries (old gen) or facts (permanent gen) now stand in for
                    # them, so the assembler must not also carry the originals.
                    # Rebuild keeping survivors rather than deleting in place -
                    # mutating a list while iterating it skips elements.
                    if departed_turn_indices:
                        state.messages = [m for m in state.messages
                                          if m.turn_index not in departed_turn_indices]

                    final_messages = self._assemble(state.messages)
                    counts = sweep_result.classification_counts
                    gc_result = GCResult(
                        final_messages=final_messages, gc_run_id=gc_run_id, status=GCStatus.COMPLETED,
                        tokens_before=tokens_before,
                        tokens_in_final=sum(message.token_count for message in final_messages),
                        # The sweep's *decision* profile, not the aging outcome. If
                        # one turn's promotion failed it stayed young, so it still
                        # shows in final_messages and tokens_in_final - the savings
                        # stay honest; only the count reflects intent, not result.
                        kept_count=counts[SweepClassification.KEEP],
                        compacted_count=counts[SweepClassification.COMPACT],
                        archived_count=counts[SweepClassification.ARCHIVE],
                        sweep_result=sweep_result, duration_ms=(perf_counter() - start) * 1000,
                    )
                    logger.debug("gc_update completed",
                                 extra={"session_id": session_id, "gc_run_id": gc_run_id,
                                        "turns_aged_out": len(departed_turn_indices),
                                        "young_remaining": len(state.messages)})
                except Exception as e:
                    # First do no harm: the mark stage failed before anything was
                    # aged, so the session is exactly as it was plus the new turn.
                    # Report the failure rather than let it die as an unretrieved
                    # task exception - gc_update is fire-and-forget, so an exception
                    # escaping here is invisible to logs and dashboard alike.
                    logger.exception("gc_update failed",
                                     extra={"session_id": session_id, "gc_run_id": gc_run_id,
                                            "stage": current_stage})
                    unchanged = self._assemble(state.messages)
                    gc_result = GCResult(
                        final_messages=unchanged, gc_run_id=gc_run_id, status=GCStatus.BYPASSED_ON_ERROR,
                        failure_reason=str(e), failure_stage=current_stage,
                        tokens_before=tokens_before,
                        tokens_in_final=sum(message.token_count for message in unchanged),
                        kept_count=len(unchanged), compacted_count=0, archived_count=0,
                        sweep_result=None, duration_ms=(perf_counter() - start) * 1000,
                    )

        self.event_bus.emit(Event(event_type=EventType.GC_FINISHED,
                                  data={"gc_run_id": gc_run_id, "session_id": session_id,
                                        "gc_result": gc_result}))

    def _age_out_turns(self, session_id: str, sweep_result: SweepResult) -> set[int]:
        """ACT: each turn's current classification is its instruction. COMPACT ->
        summarise into old gen; ARCHIVE -> extract its facts into permanent gen.
        Returns the turn indices that successfully left the young generation.

        State-based, not a diff against the last pass: an acted-on turn is
        *removed*, so it can never be re-processed - there is nothing to
        remember. (That is the monitor's job; it observes without moving,
        gc_update moves.)

        Each turn ages independently, and joins the departed set only AFTER its
        produce-action succeeds. So a failure (e.g. a Phase-7 LLMCompactor API
        error) loses nothing - the turn stays young for the next pass - and never
        leaves a summary in old gen whose original is also still live, which
        would be re-promoted into a duplicate. One failing turn must not abort
        the whole update, so we log loudly and continue: the same loud-but-
        contained posture as the EventBus. `except Exception`, not bare, so
        KeyboardInterrupt / SystemExit still propagate.
        """
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
        return departed_turn_indices
