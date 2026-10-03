import logging
import uuid
from dataclasses import dataclass, field
from time import perf_counter

from llm_gc.config.gc_config import GCConfig
from llm_gc.engine.aging_degradation import AgingDegradation, DegradationReason
from llm_gc.engine.gc_result import GCResult
from llm_gc.engine.gc_status import GCStatus
from llm_gc.engine.generations.fact_retriever import FactRetriever, LexicalFactRetriever
from llm_gc.engine.generations.generational_memory import GenerationalMemory
from llm_gc.engine.generations.memory_block import format_memory_block
from llm_gc.engine.compaction import CompactionResult
from llm_gc.engine.sweep import Sweeper, SweepResult
from llm_gc.events import Event, EventBus, EventType
from llm_gc.exceptions import CompactionRefused, LLMGCError
from llm_gc.models import KnowledgeEntry, Message, SweepClassification
from llm_gc.models.generation_snapshot import AgingDelta, AppliedAging, GenerationSnapshot, apply_aging
from llm_gc.scoring import RelevanceScorer
from llm_gc.session import SessionManager, SessionState

logger = logging.getLogger(__name__)


@dataclass
class _AgingPlan:
    """One aging pass's decisions, accumulated while the pass runs.

    Local to the pass and mutable. Nothing in it reaches session state until
    `delta()` is applied at commit; the records beyond the delta exist so events
    can be announced after the commit, describing changes that actually happened.
    """

    departed_turns: set[int] = field(default_factory=set)
    new_summaries: list[Message] = field(default_factory=list)
    evicted_summaries: list[Message] = field(default_factory=list)
    new_knowledge: list[KnowledgeEntry] = field(default_factory=list)
    promotions: list[tuple[list[int], CompactionResult]] = field(default_factory=list)
    archives: list[tuple[Message, list[KnowledgeEntry], str]] = field(default_factory=list)
    degradations: list[AgingDegradation] = field(default_factory=list)

    def delta(self) -> AgingDelta:
        return AgingDelta(
            departed_turns=frozenset(self.departed_turns),
            new_summaries=tuple(self.new_summaries),
            evicted_summaries=tuple(self.evicted_summaries),
            new_knowledge=tuple(self.new_knowledge),
        )


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

    It owns no scoring or sweeping logic and no conversation data: each session's
    generations live in its own `SessionState.snapshot`. Reads never wait for
    writes; see doc/design/session/overview.md for the concurrency model.
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

    def _assemble(self, snapshot: GenerationSnapshot, query: str | None = None) -> list[Message]:
        """Stitch all three generations of one snapshot together, in turn order.

        The single definition of "the context this session would send". Both
        `gc_collect` (which returns it) and `gc_update` (which reports it as
        `final_messages`) call this, so the number the dashboard shows and the
        payload the proxy forwards can never drift apart.

        - **permanent** - facts recovered from archived turns, as one announced
          memory block. Retrieved rather than dumped: permanent generation
          accumulates for the life of the session, and injecting all of it would
          recreate the unbounded growth archiving exists to stop.
        - **old** - summaries, each carrying the turn index of the first turn in
          its run, so sorting slots it back into position.
        - **young** - verbatim turns.

        Args:
            snapshot: The session state to assemble, read once by the caller.
            query: Text to rank stored facts against, normally the turn about to be
                sent. None falls back to the most recent facts - a caller with no
                query has not asked for no memory.
        """
        assembled = sorted(snapshot.old + snapshot.young, key=lambda message: message.turn_index)

        memory_block = self._build_memory_block(snapshot, query)
        if memory_block is not None:
            # Prepended, not sorted in: the block is not a turn, and giving it a
            # real turn index would make it compete for position with the turns it
            # summarises.
            return [memory_block] + assembled
        return assembled

    def _build_memory_block(self, snapshot: GenerationSnapshot, query: str | None) -> Message | None:
        """Retrieve the relevant stored facts and render them, or None.

        Zero on either budget disables injection entirely, which is how a
        deployment opts out and how a benchmark measures what injection is worth.
        Active entries only: a fact contradicted later never reaches the prompt.
        """
        if self.gc_config.max_memory_facts <= 0 or self.gc_config.max_memory_tokens <= 0:
            return None

        entries = snapshot.active_knowledge
        if not entries:
            return None

        relevant = self.fact_retriever.retrieve(entries, query, self.gc_config.max_memory_facts)
        return format_memory_block(relevant, self.gc_config.max_memory_tokens)

    async def gc_collect(self, session_id: str, query: str | None = None) -> list[Message]:
        """Hot path: assemble the session's context from its current snapshot.

        Never waits. The snapshot is immutable, so an aging pass that is mid-way
        through a model call cannot be observed half-done; at worst this returns
        the context from before that pass, which is complete and correct, only
        less compressed.

        Emits nothing: this path changes no state.
        """
        state = self.session_manager.get_or_create(session_id)
        return self._assemble(state.snapshot, query=query)

    async def gc_update(self, session_id: str, new_turn: Message) -> None:
        """Cold path: record a turn, then age the session if it is under pressure.

        Recording is immediate, so the next `gc_collect` sees the turn even while
        an earlier aging pass is still running. Aging runs at most once per session
        at a time: a second trigger is reported and skipped rather than queued, and
        pressure that persists triggers it again on the next turn.

        Emits GC_FINISHED exactly once, after any commit.
        """
        start = perf_counter()
        gc_run_id = str(uuid.uuid4())

        state = self.session_manager.get_or_create(session_id)
        state.snapshot = state.snapshot.with_turn(new_turn)
        base = state.snapshot

        # Measured on the assembled context, not young alone: old-gen summaries
        # are real tokens in the real prompt, and a meter that omitted them would
        # under-report pressure in the reassuring direction.
        tokens_before = sum(message.token_count for message in self._assemble(base))
        threshold_tokens = int(self.gc_config.context_window * self.gc_config.gc_threshold)

        if tokens_before < threshold_tokens:
            # Aging is triggered by pressure, not by arrival: below the threshold
            # the turn stays verbatim, which is better context and cheaper.
            gc_result = self._unchanged_result(base, gc_run_id, GCStatus.BYPASSED_BELOW_THRESHOLD,
                                               tokens_before, start)
        elif state.aging_in_flight:
            logger.debug("aging already in flight; skipping this trigger",
                         extra={"session_id": session_id, "gc_run_id": gc_run_id})
            gc_result = self._unchanged_result(base, gc_run_id, GCStatus.BYPASSED_AGING_IN_FLIGHT,
                                               tokens_before, start)
        else:
            state.aging_in_flight = True
            try:
                gc_result = await self._age(state, base, gc_run_id, tokens_before, start)
            finally:
                state.aging_in_flight = False

        self.event_bus.emit(Event(event_type=EventType.GC_FINISHED,
                                  data={"gc_run_id": gc_run_id, "session_id": session_id,
                                        "gc_result": gc_result}))

    async def _age(
        self, state: SessionState, base: GenerationSnapshot, gc_run_id: str, tokens_before: int, start: float
    ) -> GCResult:
        """Plan against `base`, then commit onto whatever the snapshot is now.

        Nothing reaches session state until the commit, which is synchronous: no
        `await` separates reading the current snapshot from replacing it. A failure
        while planning therefore leaves the session exactly as it was.
        """
        session_id = state.session_id
        current_stage = "score"
        plan = _AgingPlan()
        try:
            young = list(base.young)
            scores = [self.relevance_scorer.score(message, young) for message in young]
            current_stage = "sweep"
            sweep_result = self.sweeper.sweep(young, scores)
            current_stage = "age"
            await self._age_out_turns(session_id, sweep_result, plan)
            current_stage = "sweep_old_gen"
            self._sweep_old_gen(session_id, base, plan)
        except Exception as err:
            logger.exception("gc_update failed",
                             extra={"session_id": session_id, "gc_run_id": gc_run_id, "stage": current_stage})
            return self._unchanged_result(base, gc_run_id, GCStatus.BYPASSED_ON_ERROR, tokens_before, start,
                                          failure_reason=str(err), failure_stage=current_stage)

        if not self.session_manager.is_live(state):
            logger.info("session closed during aging; result discarded",
                        extra={"session_id": session_id, "gc_run_id": gc_run_id})
            return self._unchanged_result(base, gc_run_id, GCStatus.DISCARDED_SESSION_CLOSED, tokens_before, start)

        applied = apply_aging(state.snapshot, plan.delta())
        state.snapshot = applied.snapshot
        self._announce(session_id, plan, applied)

        final_messages = self._assemble(state.snapshot)
        counts = sweep_result.classification_counts
        logger.debug("gc_update completed",
                     extra={"session_id": session_id, "gc_run_id": gc_run_id,
                            "turns_aged_out": len(plan.departed_turns),
                            "old_gen_summaries_evicted": len(plan.evicted_summaries),
                            "young_remaining": len(state.snapshot.young)})
        return GCResult(
            final_messages=final_messages, gc_run_id=gc_run_id, status=GCStatus.COMPLETED,
            tokens_before=tokens_before,
            tokens_in_final=sum(message.token_count for message in final_messages),
            # The sweep's *decision* profile, not the outcome; `degradations`
            # records the difference.
            kept_count=counts[SweepClassification.KEEP],
            compacted_count=counts[SweepClassification.COMPACT],
            archived_count=counts[SweepClassification.ARCHIVE],
            sweep_result=sweep_result, duration_ms=(perf_counter() - start) * 1000,
            degradations=plan.degradations,
        )

    def _unchanged_result(
        self, snapshot: GenerationSnapshot, gc_run_id: str, status: GCStatus, tokens_before: int, start: float,
        failure_reason: str | None = None, failure_stage: str | None = None,
    ) -> GCResult:
        messages = self._assemble(snapshot)
        return GCResult(
            final_messages=messages, gc_run_id=gc_run_id, status=status,
            failure_reason=failure_reason, failure_stage=failure_stage,
            tokens_before=tokens_before, tokens_in_final=sum(message.token_count for message in messages),
            kept_count=len(messages), compacted_count=0, archived_count=0,
            sweep_result=None, duration_ms=(perf_counter() - start) * 1000,
        )

    async def _age_out_turns(self, session_id: str, sweep_result: SweepResult, plan: _AgingPlan) -> None:
        """ACT: COMPACT runs become summaries, ARCHIVE turns become facts.

        Each run or turn joins the departed set only after its producer succeeded,
        so a failure loses nothing: the turn stays young for the next pass, and no
        summary is ever filed while its original is still live.

        Adjacent COMPACT turns are summarised together, as one run. One turn at a
        time cannot work with a real compactor: every summary carries a provenance
        marker, so a short turn can never be replaced by something shorter and
        every promotion would be refused. A run breaks at the first non-COMPACT
        turn, so a summary never spans turns separated by a kept one.
        """
        compact_run: list[Message] = []

        for entry in sweep_result.sweep_entries:
            if compact_run and entry.classification != SweepClassification.COMPACT:
                await self._promote_run(session_id, compact_run, plan)
                compact_run = []

            if entry.classification == SweepClassification.COMPACT:
                compact_run.append(entry.message)
            elif entry.classification == SweepClassification.ARCHIVE:
                if self._archive(session_id, entry.message, "young", plan):
                    plan.departed_turns.add(entry.turn_index)

        if compact_run:
            await self._promote_run(session_id, compact_run, plan)

    def _sweep_old_gen(self, session_id: str, base: GenerationSnapshot, plan: _AgingPlan) -> None:
        """Archive the old-gen summaries that have gone cold.

        Without this, old gen grows by one summary per aged run forever. Two bands,
        not three: summarising a summary compounds loss, so a cold summary is
        archived rather than re-compacted.

        Candidates include summaries this pass just produced, and scoring runs
        against the context as it will stand after this pass - recency is
        positional, and scoring summaries among themselves would make the oldest
        survivor always look recent and never leave.
        """
        candidates = list(base.old) + plan.new_summaries
        if not candidates:
            return

        young_after = [message for message in base.young if message.turn_index not in plan.departed_turns]
        assembled = sorted(candidates + young_after, key=lambda message: message.turn_index)

        for summary in candidates:
            score = self.relevance_scorer.score(summary, assembled).combined_score
            if score >= self.gc_config.archive_threshold:
                continue
            if not self._archive(session_id, summary, "old", plan):
                continue
            if summary in plan.new_summaries:
                # Produced and archived in one pass: it never enters old gen.
                plan.new_summaries.remove(summary)
            else:
                plan.evicted_summaries.append(summary)

    def _archive(self, session_id: str, message: Message, source_generation: str, plan: _AgingPlan) -> bool:
        """Extract a turn's or summary's facts into the plan; False if extraction failed."""
        try:
            entries = self.generational_memory.extract_knowledge(message)
        except Exception as err:
            logger.exception("Failed to archive; leaving it in place for the next pass",
                             extra={"session_id": session_id, "turn_index": message.turn_index,
                                    "source_generation": source_generation})
            plan.degradations.append(
                self._record_degradation(session_id, DegradationReason.ARCHIVE_FAILED, [message.turn_index], err)
            )
            return False
        plan.new_knowledge.extend(entries)
        plan.archives.append((message, entries, source_generation))
        return True

    async def _promote_run(self, session_id: str, compact_run: list[Message], plan: _AgingPlan) -> None:
        """Summarise one run into the plan, all or nothing.

        A `CompactionRefused` is reported separately from a provider failure even
        though both leave the run verbatim, because the next action differs: a
        refusal is the guard working, and the next pass produces a different run
        for free; a provider failure means check the provider.
        """
        turn_indices = [message.turn_index for message in compact_run]
        try:
            result = await self.generational_memory.summarise(compact_run)
        except CompactionRefused as err:
            logger.warning("Compaction refused; the run stays verbatim for the next pass",
                           extra={"session_id": session_id, "turn_indices": turn_indices, "detail": str(err)})
            plan.degradations.append(
                self._record_degradation(session_id, DegradationReason.COMPACTION_REFUSED, turn_indices, err))
            return
        except Exception as err:
            logger.exception("Failed to promote a run out of young generation; leaving it for retry",
                             extra={"session_id": session_id, "turn_indices": turn_indices})
            plan.degradations.append(
                self._record_degradation(session_id, DegradationReason.COMPACTION_FAILED, turn_indices, err))
            return
        plan.departed_turns.update(turn_indices)
        plan.new_summaries.append(result.summary)
        plan.promotions.append((turn_indices, result))

    def _announce(self, session_id: str, plan: _AgingPlan, applied: AppliedAging) -> None:
        """Emit the events for a committed pass, after the commit.

        Emitted only now so an event always describes a change that happened: a
        pass discarded because its session closed announces nothing.
        """
        for turn_indices, result in plan.promotions:
            self.event_bus.emit(Event(event_type=EventType.MESSAGE_PROMOTED_TO_OLD_GEN, data={
                "session_id": session_id,
                "source_turn_indices": turn_indices,
                "messages_compacted": len(turn_indices),
                "tokens_before": result.original_token_count,
                "tokens_after": result.summary.token_count,
            }))

        superseded_by = {id(s.new_entry): s.superseded for s in applied.supersessions}
        for message, entries, source_generation in plan.archives:
            for entry in entries:
                self.event_bus.emit(Event(event_type=EventType.KNOWLEDGE_ENTRY_ADDED,
                                          data={"session_id": session_id, "knowledge_entry": entry}))
                if id(entry) in superseded_by:
                    self.event_bus.emit(Event(event_type=EventType.KNOWLEDGE_ENTRY_SUPERSEDED, data={
                        "session_id": session_id,
                        "new_knowledge_entry": entry,
                        "superseded_knowledge_entries": list(superseded_by[id(entry)]),
                    }))
            self.event_bus.emit(Event(event_type=EventType.MESSAGE_ARCHIVED, data={
                "session_id": session_id,
                "message": message,
                "extracted_knowledge_entries": entries,
                "source_generation": source_generation,
            }))

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
