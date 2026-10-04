import copy
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Protocol

from llm_gc.config.gc_config import GCConfig
from llm_gc.engine.aging_degradation import AgingDegradation, DegradationReason
from llm_gc.engine.gc_result import AbandonedCut, GCResult
from llm_gc.engine.gc_status import GCStatus
from llm_gc.engine.generations.fact_retriever import FactRetriever, LexicalFactRetriever
from llm_gc.engine.generations.generational_memory import GenerationalMemory
from llm_gc.engine.generations.front import Front, quote_turn, render_front
from llm_gc.engine.compaction import CompactionResult
from llm_gc.engine.cut_point import find_cut_point, is_safe_cut
from llm_gc.engine.inspection import CutPreview, SessionInspection, batch_worth_cutting
from llm_gc.engine.sweep import SweepEntry, Sweeper, SweepResult
from llm_gc.events import Event, EventBus, EventType
from llm_gc.exceptions import CompactionRefused, LLMGCError
from llm_gc.models import KnowledgeEntry, Message, SweepClassification
from llm_gc.models.generation_snapshot import (
    AgingDelta,
    AppliedAging,
    EntryKind,
    FrontEntry,
    GenerationSnapshot,
    apply_aging,
)
from llm_gc.scoring import RelevanceScorer, RelevanceScorerResult
from llm_gc.session import SessionManager, SessionState
from llm_gc.utils import count_tokens

logger = logging.getLogger(__name__)


@dataclass
class _AgingPlan:
    """One aging pass's decisions, accumulated while the pass runs.

    Local to the pass and mutable. Nothing in it reaches session state until
    `delta()` is applied at commit; the records beyond the delta exist so events
    can be announced after the commit, describing changes that actually happened.
    """

    departed_turns: set[int] = field(default_factory=set)
    new_entries: list[FrontEntry] = field(default_factory=list)
    evicted_entries: list[FrontEntry] = field(default_factory=list)
    new_knowledge: list[KnowledgeEntry] = field(default_factory=list)
    promotions: list[tuple[list[int], CompactionResult]] = field(default_factory=list)
    archives: list[tuple[Message, list[KnowledgeEntry], str]] = field(default_factory=list)
    degradations: list[AgingDegradation] = field(default_factory=list)

    def delta(self) -> AgingDelta:
        return AgingDelta(
            departed_turns=frozenset(self.departed_turns),
            new_entries=tuple(self.new_entries),
            evicted_entries=tuple(self.evicted_entries),
            new_knowledge=tuple(self.new_knowledge),
        )


@dataclass(frozen=True)
class _Marked:
    """The young generation swept and measured against the latest safe cut point."""

    young: list[Message]
    sweep_result: SweepResult
    cut: int
    batch: list[SweepEntry]

    @property
    def compactable_tokens(self) -> int:
        return _compactable_tokens(self.batch)


class _Announcing(Protocol):
    event_bus: EventBus | None


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

    def _assemble(self, snapshot: GenerationSnapshot) -> list[Message]:
        """The context this session sends: the front, then the verbatim tail.

        The single definition of "the context this session would send". Both
        `gc_collect` (which returns it) and `gc_age` (which reports it as
        `final_messages`) call this, so the number the dashboard shows and the
        payload the proxy forwards can never drift apart.

        Every change to the prompt happens at the front, and only when a cut
        commits; the tail is the conversation exactly as recorded. A prompt that
        changed anywhere else would reset the provider's cache and invalidate the
        model's thinking blocks on every request.
        """
        front = self._render_front(snapshot)
        return ([front.message] if front is not None else []) + _in_order(snapshot.young)

    def _render_front(self, snapshot: GenerationSnapshot) -> Front | None:
        """Render the front from the generations a cut produced.

        Facts are chosen when the cut is made, not for each question: a block that
        changed with every question would change the prompt on every request.
        Zero on either memory budget leaves facts out, which is how a deployment
        opts out and how a benchmark measures what they are worth.
        """
        facts: list[KnowledgeEntry] = []
        if self.gc_config.max_memory_facts > 0 and self.gc_config.max_memory_tokens > 0:
            facts = self.fact_retriever.retrieve(snapshot.active_knowledge, None, self.gc_config.max_memory_facts)
        return render_front(snapshot.old, facts, self.gc_config.max_memory_tokens)

    async def gc_collect(self, session_id: str) -> list[Message]:
        """Hot path: the session's context, from its current snapshot.

        Never waits. The snapshot is immutable, so an aging pass mid-way through a
        model call cannot be observed half-done; at worst this returns the context
        from before that pass, which is complete and correct.

        Emits nothing: this path changes no state.
        """
        state = self.session_manager.get_or_create(session_id)
        return self._assemble(state.snapshot)

    def inspect(self, session_id: str) -> SessionInspection | None:
        """A read-only view of one session, or None if there is no such session.

        Uses the same rendering as `gc_collect` and the same marking step as
        `gc_age`, so the view cannot drift from what the engine does. No model
        calls, no events, and unlike the request-path methods it never creates a
        session or resets its idle clock: looking must not change what is looked at.
        """
        state = self.session_manager.store.get(session_id)
        if state is None:
            return None
        snapshot = state.snapshot
        young = _in_order(snapshot.young)
        marked = self._sweep_and_measure(young, self._score(young, _quiet(self.relevance_scorer)),
                                         _quiet(self.sweeper))
        return SessionInspection(
            session_id=session_id,
            snapshot=snapshot,
            front=self._render_front(snapshot),
            tail=tuple(young),
            next_cut=CutPreview(entries=tuple(marked.sweep_result.sweep_entries), cut=marked.cut,
                                compactable_tokens=marked.compactable_tokens,
                                min_cut_tokens=self.gc_config.min_cut_tokens),
            aging_in_flight=state.aging_in_flight,
            last_activity=state.last_updated_at,
        )

    def mark_front_sent(self, session_id: str, history_turns: int) -> int:
        """Note that the current front is being sent after `history_turns` turns of history.

        Returns the first turn whose thinking blocks are still valid. A thinking
        block is bound to the prompt it was produced against, so every turn sent
        with the first request that carries a new front predates it - including the
        reply to the request during which the cut was planned, which the commit
        cannot see. Replies from then on are produced against this front and keep
        their thinking until the next cut. Immediate.
        """
        state = self.session_manager.store.get(session_id)
        if state is None:
            return history_turns
        valid_from = state.snapshot.thinking_valid_from
        if valid_from is None:
            valid_from = history_turns
            state.snapshot = replace(state.snapshot, thinking_valid_from=valid_from)
        return valid_from

    def gc_record(self, session_id: str, turns: Sequence[Message], lineage: Sequence[str]) -> SessionState:
        """Record turns the conversation has gained, with their chain hashes. Immediate.

        Used by callers that identify a session by its content: each turn arrives
        with the hash that places it in the conversation, so the session can later
        be found by - and checked against - the client's own history.
        """
        state = self.session_manager.get_or_create(session_id)
        state.snapshot = state.snapshot.with_turns(turns, lineage)
        self.session_manager.retip(state)
        return state

    async def gc_update(self, session_id: str, new_turn: Message) -> None:
        """Record one turn, then age the session if it is under pressure."""
        state = self.session_manager.get_or_create(session_id)
        state.snapshot = state.snapshot.with_turn(new_turn)
        await self.gc_age(session_id)

    async def gc_age(self, session_id: str, *, ignore_batch_minimum: bool = False) -> GCResult:
        """Cold path: cut the session if enough of it is ready to be compacted.

        Classification decides what is compacted and how; the batch size decides
        when. Recording is separate and immediate, so the next `gc_collect` sees new
        turns even while a pass is running. At most one pass runs per session: a
        second trigger is reported and skipped rather than queued.

        `ignore_batch_minimum` is the manual lever: the same pass with
        `min_cut_tokens` taken as zero. Every other rule still holds - the newest
        turns stay verbatim, no tool round is split, something must be compacted,
        and the cut must shrink the context.

        Emits GC_FINISHED exactly once, after any commit, and returns the result.
        """
        start = perf_counter()
        gc_run_id = str(uuid.uuid4())
        state = self.session_manager.get_or_create(session_id)
        base = state.snapshot

        if state.aging_in_flight:
            logger.debug("aging already in flight; skipping this trigger",
                         extra={"session_id": session_id, "gc_run_id": gc_run_id})
            gc_result = self._unchanged_result(base, gc_run_id, GCStatus.BYPASSED_AGING_IN_FLIGHT, start)
        else:
            state.aging_in_flight = True
            try:
                min_cut_tokens = 0 if ignore_batch_minimum else self.gc_config.min_cut_tokens
                gc_result = await self._age(state, base, gc_run_id, start, min_cut_tokens)
            finally:
                state.aging_in_flight = False

        self.event_bus.emit(Event(event_type=EventType.GC_FINISHED,
                                  data={"gc_run_id": gc_run_id, "session_id": session_id,
                                        "gc_result": gc_result}))
        return gc_result

    async def _age(self, state: SessionState, base: GenerationSnapshot, gc_run_id: str, start: float,
                   min_cut_tokens: int) -> GCResult:
        """Plan a cut against `base`, then commit it onto whatever the snapshot is now.

        Nothing reaches session state until the commit, which is synchronous: no
        `await` separates reading the current snapshot from replacing it. A failure
        while planning therefore leaves the session exactly as it was.
        """
        session_id = state.session_id
        current_stage = "score"
        plan = _AgingPlan()
        try:
            young = _in_order(base.young)
            scores = self._score(young, self.relevance_scorer)
            current_stage = "sweep"
            marked = self._sweep_and_measure(young, scores, self.sweeper)
            sweep_result = marked.sweep_result
            if not batch_worth_cutting(marked.cut, marked.compactable_tokens, min_cut_tokens):
                return self._unchanged_result(base, gc_run_id, GCStatus.BYPASSED_BATCH_TOO_SMALL, start)

            current_stage = "age"
            intended = marked.batch
            plan, cut = await self._plan_batch(session_id, young, intended)
            batch = intended[:cut]
            if not batch_worth_cutting(cut, _compactable_tokens(batch), min_cut_tokens):
                # A failure shortened the cut below what is worth a cache reset. The
                # pass still ran to its decision, so it reports what it intended; the
                # degradations say what stopped it.
                logger.info("cut abandoned after a failure shortened it",
                            extra={"session_id": session_id, "gc_run_id": gc_run_id, "turns": cut})
                return self._completed_result(base, base, gc_run_id, sweep_result, intended,
                                              len(young) - len(intended), plan.degradations, start)
            current_stage = "sweep_old_gen"
            self._sweep_old_gen(session_id, base, plan)
            # Measured on the context the cut would produce, not summed: the front
            # carries facts and quote markers, so a cut can make the prompt larger
            # than the turns it removes - a cache reset that costs tokens too.
            projected = self._context_tokens(apply_aging(base, plan.delta()).snapshot)
            if projected >= self._context_tokens(base):
                logger.info("cut abandoned: it would not shrink the context",
                            extra={"session_id": session_id, "gc_run_id": gc_run_id})
                return self._unchanged_result(base, gc_run_id, GCStatus.BYPASSED_NO_SAVING, start,
                                              degradations=plan.degradations,
                                              abandoned_cut=_abandoned_cut(batch, projected))
        except Exception as err:
            logger.exception("gc_age failed",
                             extra={"session_id": session_id, "gc_run_id": gc_run_id, "stage": current_stage})
            return self._unchanged_result(base, gc_run_id, GCStatus.BYPASSED_ON_ERROR, start,
                                          failure_reason=str(err), failure_stage=current_stage)

        if not self.session_manager.is_live(state):
            logger.info("session closed during aging; result discarded",
                        extra={"session_id": session_id, "gc_run_id": gc_run_id})
            return self._unchanged_result(base, gc_run_id, GCStatus.DISCARDED_SESSION_CLOSED, start)

        applied = apply_aging(state.snapshot, plan.delta())
        state.snapshot = replace(applied.snapshot, thinking_valid_from=None, cut_count=applied.snapshot.cut_count + 1)
        self._announce(session_id, plan, applied)

        logger.debug("cut committed", extra={"session_id": session_id, "gc_run_id": gc_run_id,
                                             "turns_cut": len(batch), "young_remaining": len(state.snapshot.young)})
        return self._completed_result(base, state.snapshot, gc_run_id, sweep_result, batch, len(young) - cut,
                                      plan.degradations, start)

    def _completed_result(
        self, base: GenerationSnapshot, final: GenerationSnapshot, gc_run_id: str, sweep_result: SweepResult,
        batch: Sequence[SweepEntry], tail_length: int, degradations: list[AgingDegradation], start: float,
    ) -> GCResult:
        """A pass that ran to its decision, whether or not every intention was met.

        The counts are the cut's decision profile - turns kept (verbatim in the tail
        or quoted in the front), summarised, or reduced to facts - and `degradations`
        is the only place the difference between decision and outcome is recorded.
        """
        final_messages = self._assemble(final)
        counts = {c: sum(1 for entry in batch if entry.classification == c) for c in SweepClassification}
        return GCResult(
            final_messages=final_messages, gc_run_id=gc_run_id, status=GCStatus.COMPLETED,
            tokens_before=self._context_tokens(base),
            tokens_in_final=sum(message.token_count for message in final_messages),
            kept_count=counts[SweepClassification.KEEP] + tail_length,
            compacted_count=counts[SweepClassification.COMPACT],
            archived_count=counts[SweepClassification.ARCHIVE],
            sweep_result=sweep_result, duration_ms=(perf_counter() - start) * 1000,
            degradations=degradations,
        )

    def _context_tokens(self, snapshot: GenerationSnapshot) -> int:
        return sum(message.token_count for message in self._assemble(snapshot))

    @staticmethod
    def _score(young: list[Message], scorer: RelevanceScorer) -> list[RelevanceScorerResult]:
        return [scorer.score(message, young) for message in young]

    def _sweep_and_measure(self, young: list[Message], scores: list[RelevanceScorerResult],
                           sweeper: Sweeper) -> _Marked:
        """Classify the verbatim turns and find the batch the latest safe cut would take.

        The one definition of a cut's batch, shared by a real pass and a preview.
        """
        sweep_result = sweeper.sweep(young, scores)
        cut = find_cut_point(young, self.gc_config.last_n_turns_to_keep)
        return _Marked(young=young, sweep_result=sweep_result, cut=cut, batch=sweep_result.sweep_entries[:cut])

    def _unchanged_result(
        self, snapshot: GenerationSnapshot, gc_run_id: str, status: GCStatus, start: float,
        failure_reason: str | None = None, failure_stage: str | None = None,
        degradations: list[AgingDegradation] | None = None, abandoned_cut: AbandonedCut | None = None,
    ) -> GCResult:
        messages = self._assemble(snapshot)
        tokens = sum(message.token_count for message in messages)
        return GCResult(
            final_messages=messages, gc_run_id=gc_run_id, status=status,
            failure_reason=failure_reason, failure_stage=failure_stage,
            tokens_before=tokens, tokens_in_final=tokens,
            kept_count=len(messages), compacted_count=0, archived_count=0,
            sweep_result=None, duration_ms=(perf_counter() - start) * 1000,
            degradations=degradations or [], abandoned_cut=abandoned_cut,
        )

    async def _plan_batch(
        self, session_id: str, young: Sequence[Message], batch: Sequence[SweepEntry]
    ) -> tuple[_AgingPlan, int]:
        """Represent the batch in the front, and return how much of it the cut covers.

        COMPACT runs become summaries, ARCHIVE turns become facts, and KEEP turns are
        quoted word for word: a turn scored too important to summarise is never lost
        to a cut. Adjacent COMPACT turns are summarised together - a summary carries
        a provenance marker, so one short turn could never be replaced by something
        shorter - and a run breaks at the first non-COMPACT turn.

        On the first summary or extraction that fails, the cut retreats to just
        before it: those turns stay in the tail, so the next pass meets them in a
        longer run, which is the retry that can actually succeed. The cut only ever
        lands on a segment boundary that leaves no tool result without its call.
        """
        segments: list[tuple[int, _AgingPlan]] = []
        degradations: list[AgingDegradation] = []
        position = 0
        while position < len(batch):
            entry = batch[position]
            end = position + 1
            segment = _AgingPlan()
            if entry.classification == SweepClassification.COMPACT:
                while end < len(batch) and batch[end].classification == SweepClassification.COMPACT:
                    end += 1
                ok = await self._promote_run(session_id, [e.message for e in batch[position:end]], segment)
            elif entry.classification == SweepClassification.ARCHIVE:
                ok = self._archive(session_id, entry.message, "young", segment)
            else:
                segment.new_entries.append(quote_turn(entry.message))
                ok = True
            degradations.extend(segment.degradations)
            if not ok:
                break
            segment.departed_turns.update(e.turn_index for e in batch[position:end])
            segments.append((end, segment))
            position = end

        boundaries = [0] + [end for end, _ in segments]
        cut = max(boundary for boundary in boundaries if is_safe_cut(young, boundary))
        plan = _AgingPlan(degradations=degradations)
        for end, segment in segments:
            if end > cut:
                break
            plan.departed_turns |= segment.departed_turns
            plan.new_entries.extend(segment.new_entries)
            plan.new_knowledge.extend(segment.new_knowledge)
            plan.archives.extend(segment.archives)
            plan.promotions.extend(segment.promotions)
        return plan, cut

    def _sweep_old_gen(self, session_id: str, base: GenerationSnapshot, plan: _AgingPlan) -> None:
        """Archive the front's entries that have gone cold.

        Without this, the front grows by one summary per cut forever. Two bands, not
        three: summarising a summary compounds loss, so a cold entry is archived into
        facts rather than re-summarised.

        Candidates include entries this cut just produced, and scoring runs against
        the context as it will stand after it - recency is positional, and scoring
        entries among themselves would make the oldest survivor always look recent.
        """
        candidates = list(base.old) + plan.new_entries
        if not candidates:
            return

        tail_after = [message for message in base.young if message.turn_index not in plan.departed_turns]
        assembled = _in_order([entry.message for entry in candidates] + tail_after)

        for entry in candidates:
            score = self.relevance_scorer.score(entry.message, assembled).combined_score
            if score >= self.gc_config.archive_threshold:
                continue
            stated = entry.message.model_copy(update={"content": entry.text, "token_count": count_tokens(entry.text)})
            if not self._archive(session_id, stated, "old", plan, source_turns=entry.source_turns):
                continue
            if entry in plan.new_entries:
                # Produced and archived in one cut: it never enters old gen.
                plan.new_entries.remove(entry)
            else:
                plan.evicted_entries.append(entry)

    def _archive(self, session_id: str, message: Message, source_generation: str, plan: _AgingPlan,
                 source_turns: tuple[int, ...] = ()) -> bool:
        """Extract a turn's or entry's facts into the plan; False if extraction failed.

        `source_turns` carries an old-generation entry's coverage onto its facts, so
        turns summarised together stay traceable after the summary is archived.
        """
        try:
            entries = self.generational_memory.extract_knowledge(message)
        except Exception as err:
            logger.exception("Failed to archive; it stays where it is for a later cut",
                             extra={"session_id": session_id, "turn_index": message.turn_index,
                                    "source_generation": source_generation})
            plan.degradations.append(
                self._record_degradation(session_id, DegradationReason.ARCHIVE_FAILED, [message.turn_index], err)
            )
            return False
        if source_turns:
            entries = [replace(entry, source_turns=source_turns) for entry in entries]
        plan.new_knowledge.extend(entries)
        plan.archives.append((message, entries, source_generation))
        return True

    async def _promote_run(self, session_id: str, compact_run: list[Message], plan: _AgingPlan) -> bool:
        """Summarise one run into the plan; False, with a degradation, if that failed.

        A `CompactionRefused` is reported separately from a provider failure even
        though both leave the run verbatim, because the next action differs: a
        refusal is the guard working and a longer run will usually succeed, while a
        provider failure means check the provider.
        """
        turn_indices = [message.turn_index for message in compact_run]
        try:
            result = await self.generational_memory.summarise(compact_run)
        except CompactionRefused as err:
            logger.warning("Compaction refused; the run stays verbatim for a later cut",
                           extra={"session_id": session_id, "turn_indices": turn_indices, "detail": str(err)})
            plan.degradations.append(
                self._record_degradation(session_id, DegradationReason.COMPACTION_REFUSED, turn_indices, err))
            return False
        except Exception as err:
            logger.exception("Failed to summarise a run; it stays verbatim for a later cut",
                             extra={"session_id": session_id, "turn_indices": turn_indices})
            plan.degradations.append(
                self._record_degradation(session_id, DegradationReason.COMPACTION_FAILED, turn_indices, err))
            return False
        text = result.summary_text if result.summary_text is not None else result.summary.content
        plan.new_entries.append(FrontEntry(message=result.summary, kind=EntryKind.SUMMARY,
                                           source_turns=tuple(turn_indices), text=text))
        plan.promotions.append((turn_indices, result))
        return True

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


def _in_order(messages: Sequence[Message]) -> list[Message]:
    return sorted(messages, key=lambda message: message.turn_index)


def _abandoned_cut(batch: Sequence[SweepEntry], context_tokens_after: int) -> AbandonedCut:
    counts = {c: sum(1 for entry in batch if entry.classification == c) for c in SweepClassification}
    return AbandonedCut(
        turns=[entry.turn_index for entry in batch], quoted=counts[SweepClassification.KEEP],
        summarised=counts[SweepClassification.COMPACT], archived=counts[SweepClassification.ARCHIVE],
        tokens_taken=sum(entry.message.token_count for entry in batch), context_tokens_after=context_tokens_after,
    )


def _compactable_tokens(batch: Sequence[SweepEntry]) -> int:
    """Tokens a cut of `batch` would compact: everything not quoted verbatim."""
    return sum(entry.message.token_count for entry in batch if entry.classification != SweepClassification.KEEP)


def _quiet[T: _Announcing](component: T) -> T:
    """A shallow copy of a scorer or sweeper that announces nothing on the bus.

    A dry run reported as a real sweep would have every subscriber record work
    that never happened. Shallow, so the copy shares the original's scorers and
    strategy and differs only in having no bus.
    """
    twin = copy.copy(component)
    twin.event_bus = None
    return twin
