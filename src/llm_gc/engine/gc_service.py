import logging
import uuid
from time import perf_counter

from llm_gc.config.gc_config import GCConfig
from llm_gc.engine.aging_degradation import AgingDegradation, DegradationReason
from llm_gc.engine.gc_result import GCResult
from llm_gc.engine.gc_status import GCStatus
from llm_gc.engine.generations.fact_retriever import FactRetriever, LexicalFactRetriever
from llm_gc.engine.generations.generational_memory import GenerationalMemory
from llm_gc.engine.generations.memory_block import format_memory_block
from llm_gc.engine.sweep import Sweeper, SweepResult
from llm_gc.events import Event, EventBus, EventType
from llm_gc.exceptions import CompactionRefused, LLMGCError
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
                 gc_config: GCConfig, fact_retriever: FactRetriever | None = None) -> None:
        self.session_manager = session_manager
        self.generational_memory = generational_memory
        self.relevance_scorer = relevance_scorer
        self.sweeper = sweeper
        self.event_bus = event_bus
        self.gc_config = gc_config
        # Defaulted rather than required, so existing wiring keeps working and a
        # deployment with its own vector store supplies one instead. Ranking
        # quality is tunable; that it happens at all is not.
        self.fact_retriever = fact_retriever if fact_retriever is not None else LexicalFactRetriever()

    def _assemble(self, young_messages: list[Message], query: str | None = None) -> list[Message]:
        """Stitch all three generations together, in turn order.

        The single definition of "the context this session would send". Both
        `gc_collect` (which returns it) and `gc_update` (which reports it as
        `final_messages`) call this, so the number the dashboard shows and the
        payload the proxy forwards can never drift apart.

        Three generations, in increasing order of compression:

        - **permanent** - facts recovered from turns that were archived, as one
          announced memory block. Retrieved rather than dumped: permanent
          generation accumulates for the life of the session, and injecting all of
          it would recreate the unbounded growth that archiving exists to stop.
        - **old** - summaries. Each carries the turn index of the first turn in its
          run, so sorting slots it back into position.
        - **young** - verbatim turns.

        Sorting by `turn_index` keeps the conversation chronological; the memory
        block uses index -1 so it precedes every real turn without colliding with
        turn 0, which may still be present.

        Args:
            young_messages: The verbatim turns this session still holds.
            query: Text to rank stored facts against, normally the turn about to be
                sent. None falls back to the most recent facts - a caller with no
                query has not asked for no memory.
        """
        assembled = self.generational_memory.get_old_gen() + young_messages
        assembled.sort(key=lambda message: message.turn_index)

        memory_block = self._build_memory_block(query)
        if memory_block is not None:
            # Prepended, not sorted in: the block is not a turn, and giving it a
            # real turn index would make it compete for position with the turns it
            # summarises.
            return [memory_block] + assembled
        return assembled

    def _build_memory_block(self, query: str | None) -> Message | None:
        """Retrieve the relevant stored facts and render them, or None.

        Both budgets are checked here rather than inside the retriever, because
        "how many to consider" and "how many tokens they may occupy" are the
        caller's policy and the retriever's job is only ranking. Zero on either
        knob disables injection entirely, which is how a deployment opts out and
        how a benchmark measures what injection is worth.
        """
        if self.gc_config.max_memory_facts <= 0 or self.gc_config.max_memory_tokens <= 0:
            return None

        # Active entries only - supersession has already been applied, so a fact
        # that was contradicted later never reaches the prompt. That is the whole
        # point of tracking supersession: without this filter the model would read
        # "the database is postgres" and "the database is mysql" as equal claims.
        entries = self.generational_memory.get_permanent_gen()
        if not entries:
            return None

        relevant = self.fact_retriever.retrieve(entries, query, self.gc_config.max_memory_facts)
        return format_memory_block(relevant, self.gc_config.max_memory_tokens)

    async def gc_collect(self, session_id: str, query: str | None = None) -> list[Message]:
        """Hot path: assemble the cleaned context for a session.

        No scoring here - `gc_update` already did that work; collect only reads
        and concatenates. Held under the session lock so it never reads state
        that an in-flight update is halfway through rewriting.

        Emits nothing: this path changes no state, and the health monitor
        observes state changes. A read that reports itself as a "GC run" would
        overwrite the last real run's breakdown with a no-op.
        """
        async with self.session_manager.acquire(session_id) as state:
            # The query is the incoming turn, which only the caller has: collect
            # runs *before* the turn is recorded. Optional so existing callers keep
            # working, and so a caller that genuinely has no query (a dashboard
            # reading the context) still gets memory, ranked by recency.
            return self._assemble(state.messages, query=query)

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
                    departed_turn_indices, degradations = await self._age_out_turns(session_id, sweep_result)

                    # The departed turns leave the young generation - their
                    # summaries (old gen) or facts (permanent gen) now stand in for
                    # them, so the assembler must not also carry the originals.
                    # Rebuild keeping survivors rather than deleting in place -
                    # mutating a list while iterating it skips elements.
                    if departed_turn_indices:
                        state.messages = [m for m in state.messages
                                          if m.turn_index not in departed_turn_indices]

                    # The old generation's own aging pass. Inside the pressure
                    # gate with everything else: evicting summaries nobody needed
                    # evicted is what the gate exists to prevent.
                    current_stage = "sweep_old_gen"
                    evicted, old_gen_degradations = self._sweep_old_gen(session_id, state.messages)
                    degradations.extend(old_gen_degradations)

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
                        degradations=degradations,
                    )
                    logger.debug("gc_update completed",
                                 extra={"session_id": session_id, "gc_run_id": gc_run_id,
                                        "turns_aged_out": len(departed_turn_indices),
                                        "old_gen_summaries_evicted": evicted,
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

    async def _age_out_turns(
        self, session_id: str, sweep_result: SweepResult
    ) -> tuple[set[int], list[AgingDegradation]]:
        """ACT: each turn's current classification is its instruction. COMPACT ->
        summarise into old gen; ARCHIVE -> extract its facts into permanent gen.

        Returns the turn indices that successfully left the young generation, and a
        record of everything this pass intended and did not manage. The second half
        is not optional bookkeeping: the GCResult counts describe the *sweeper's
        decision*, so a pass whose compactions were all refused would otherwise
        report five compactions and file none, and no consumer could tell.

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
        degradations: list[AgingDegradation] = []
        # Adjacent COMPACT turns are summarised together, as one run. Promoting
        # them one at a time is what the session path used to do, and with a real
        # LLMCompactor it cannot work: every summary carries a ~15-token provenance
        # marker, so a single 20-token turn can never be replaced by something
        # shorter than itself and the compactor refuses every promotion. Grouping
        # also produces better summaries - a run of turns has a thread to follow,
        # where one turn in isolation has only itself. Same grouping rule as
        # ContextComposer.compose: a run breaks at the first non-COMPACT turn, so
        # a summary never spans turns that were separated by a kept one.
        compact_run: list[Message] = []

        for entry in sweep_result.sweep_entries:
            if compact_run and entry.classification != SweepClassification.COMPACT:
                departed, degradation = await self._promote_run(session_id, compact_run)
                departed_turn_indices |= departed
                if degradation is not None:
                    degradations.append(degradation)
                compact_run = []

            if entry.classification == SweepClassification.COMPACT:
                compact_run.append(entry.message)
            elif entry.classification == SweepClassification.ARCHIVE:
                try:
                    self.generational_memory.archive_message(entry.message)
                    departed_turn_indices.add(entry.turn_index)
                except Exception as err:
                    logger.exception(
                        "Failed to archive turn out of young generation; leaving it for retry",
                        extra={"session_id": session_id, "turn_index": entry.turn_index,
                               "classification": entry.classification.value},
                    )
                    degradations.append(
                        self._record_degradation(
                            session_id, DegradationReason.ARCHIVE_FAILED, [entry.turn_index], err
                        )
                    )

        if compact_run:
            departed, degradation = await self._promote_run(session_id, compact_run)
            departed_turn_indices |= departed
            if degradation is not None:
                degradations.append(degradation)

        return departed_turn_indices, degradations

    def _sweep_old_gen(self, session_id: str, young_messages: list[Message]) -> tuple[int, list[AgingDegradation]]:
        """Score the old generation and archive the summaries that have gone cold.

        Without this, a turn's journey ends at old gen: `_old_gen` is append-only,
        so the assembled context grows by one summary per aged run forever. Old gen
        becomes the leak that young gen was fixed to avoid.

        **Two bands here, not three.** A summary is kept or archived. The
        sweep strategy's middle band - re-compact into a longer-horizon summary -
        is deliberately not used: summarising a summary compounds loss, because each
        pass through a model drops detail and the second pass drops detail about
        detail we can no longer inspect. It also costs a model call per eviction,
        where extraction costs none. A benchmark showing re-compaction retains more
        than extraction would reopen it. Since the decision is binary, this
        compares the score to `archive_threshold` directly rather than reusing
        `ThresholdSweepStrategy`, whose three-way split has no middle band to mean
        anything here.

        **Scored against the assembled context, not against old gen alone.**
        Recency is positional, and a summary's age is its distance from the newest
        turn in the real conversation. Scoring summaries among themselves would
        make the oldest surviving summary always look recent, so the coldest
        summary would be permanently safe - exactly the bug that would make this
        pass look like it worked while bounding nothing.

        No separate token budget. Summaries carry the first turn index of their
        run, so decay already applies and the bound already exists; a budget would
        be a second mechanism able to disagree with the score. The thing to watch
        is whether decay bounds old gen *fast enough*, which is a measurement, not
        an argument.

        Returns the number of summaries evicted, and a record of any that could not
        be.
        """
        old_gen = self.generational_memory.get_old_gen()
        if not old_gen:
            return 0, []

        # Same list the caller will send, so positional recency means what it says.
        assembled = sorted(old_gen + young_messages, key=lambda message: message.turn_index)

        evicted = 0
        degradations: list[AgingDegradation] = []
        for summary in old_gen:
            score = self.relevance_scorer.score(summary, assembled).combined_score
            if score >= self.gc_config.archive_threshold:
                continue
            try:
                self.generational_memory.evict_from_old_gen(summary)
                evicted += 1
            except Exception as err:
                logger.exception(
                    "Failed to archive an old-gen summary; leaving it in old gen",
                    extra={"session_id": session_id, "turn_index": summary.turn_index},
                )
                degradations.append(
                    self._record_degradation(
                        session_id, DegradationReason.ARCHIVE_FAILED, [summary.turn_index], err
                    )
                )

        # Iterating `old_gen` (a copy) while `evict_from_old_gen` mutates the real
        # list is safe precisely because `get_old_gen` returns a copy. Iterating the
        # live list and removing from it would skip elements.
        return evicted, degradations

    async def _promote_run(
        self, session_id: str, compact_run: list[Message]
    ) -> tuple[set[int], AgingDegradation | None]:
        """Summarise one run into old gen; return what left and what degraded.

        All-or-nothing per run, not per turn: the summary covers the whole run, so
        filing it while leaving one of its turns young would send both the summary
        and the original. On failure the entire run stays young and verbatim, which
        costs one wasted call and loses nothing.

        A `CompactionRefused` is reported separately from a provider failure even
        though both leave the run verbatim, because the next action differs. A
        refusal is the guard working - the summary was truncated, empty, or no
        shorter than its input - and the usual remedy is a *different run* rather
        than the same call again, which the next pass produces for free as more
        turns cool and the run grows. A provider failure means check the provider.
        """
        turn_indices = [message.turn_index for message in compact_run]

        try:
            await self.generational_memory.promote_to_old_gen(compact_run)
        except CompactionRefused as err:
            logger.warning(
                "Compaction refused; the run stays verbatim for the next pass",
                extra={"session_id": session_id, "turn_indices": turn_indices, "detail": str(err)},
            )
            return set(), self._record_degradation(
                session_id, DegradationReason.COMPACTION_REFUSED, turn_indices, err
            )
        except Exception as err:
            logger.exception(
                "Failed to promote a run out of young generation; leaving it for retry",
                extra={"session_id": session_id, "turn_indices": turn_indices},
            )
            return set(), self._record_degradation(
                session_id, DegradationReason.COMPACTION_FAILED, turn_indices, err
            )
        return set(turn_indices), None

    def _record_degradation(
        self, session_id: str, reason: DegradationReason, turn_indices: list[int], err: Exception
    ) -> AgingDegradation:
        """Build the record and announce it on the bus.

        Emitted here rather than only folded into GC_FINISHED so a subscriber gets
        the detail as it happens, and so degradation is never something a consumer
        has to infer from counts that do not add up.

        The message is taken from the exception only when it is one of ours:
        `LLMGCError` messages are content-free by the convention documented in
        `exceptions.py`, while an arbitrary exception - including from a
        third-party compactor - is not a boundary we control, and event payloads
        reach the dashboard over HTTP.
        """
        if isinstance(err, LLMGCError):
            degradation = AgingDegradation(
                reason=reason, turn_indices=turn_indices, error_code=err.code, detail=str(err)
            )
        else:
            degradation = AgingDegradation(
                reason=reason, turn_indices=turn_indices, error_code=None, detail=type(err).__name__
            )

        self.event_bus.emit(
            Event(
                event_type=EventType.AGING_DEGRADED,
                data={"session_id": session_id, "degradation": degradation},
            )
        )
        return degradation
