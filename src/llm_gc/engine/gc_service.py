import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from time import perf_counter

from llm_gc.config.constants import DEFAULT_COMPACTION_TARGET_RATIO
from llm_gc.config.gc_config import GCConfig
from llm_gc.engine.aging_degradation import AgingDegradation, DegradationReason
from llm_gc.engine.cut_point import find_cut_point, tool_result_ids
from llm_gc.engine.gc_result import AbandonedCut, CutOutcome, GCResult
from llm_gc.engine.gc_status import GCStatus
from llm_gc.engine.generations.condensing import Condensed
from llm_gc.engine.generations.front import (
    FRONT_OVERHEAD_TOKENS,
    Front,
    active_facts_by_turn,
    facts_of,
    folded_line,
    level_costs,
    render_front,
)
from llm_gc.engine.generations.generational_memory import GenerationalMemory
from llm_gc.engine.generations.planning import Bands, LevelOptions, Plan, plan_levels
from llm_gc.engine.inspection import CutPreview, SessionInspection, batch_worth_cutting
from llm_gc.events import Event, EventBus, EventType
from llm_gc.exceptions import CompactionRefused, LLMGCError
from llm_gc.models import KnowledgeEntry, Message
from llm_gc.models.generation_snapshot import (
    AgingDelta,
    AppliedAging,
    DetailLevel,
    GenerationSnapshot,
    Span,
    apply_aging,
    file_knowledge,
)
from llm_gc.scoring import SpanScorer
from llm_gc.session import SessionManager, SessionState
from llm_gc.utils import count_tokens

logger = logging.getLogger(__name__)

_RANK = {DetailLevel.OUT_OF_VIEW: 0, DetailLevel.ARCHIVED: 1, DetailLevel.COMPACTED: 2, DetailLevel.KEPT: 3}
_CONDENSED = (DetailLevel.ARCHIVED, DetailLevel.OUT_OF_VIEW)


@dataclass(frozen=True)
class _Planned:
    """One plan over every span: the old generation's as they stand, then the cut's new ones.

    New spans carry a provisional level until the plan gives them theirs.
    """

    spans: tuple[Span, ...]
    new_from: int
    scores: tuple[float, ...]
    plan: Plan
    new_facts: tuple[KnowledgeEntry, ...]

    @property
    def placed(self) -> list[tuple[Span, DetailLevel]]:
        """The cut's new spans, each with the level the plan gave it."""
        return list(zip(self.spans[self.new_from:], self.plan.levels[self.new_from:], strict=True))

    @property
    def replanned(self) -> list[tuple[Span, DetailLevel]]:
        """The old generation's spans, each with the level the plan gives it now."""
        return list(zip(self.spans[: self.new_from], self.plan.levels[: self.new_from], strict=True))

    @property
    def cut_turns(self) -> list[Message]:
        return [turn for span in self.spans[self.new_from:] for turn in span.turns]

    @property
    def compactable_tokens(self) -> int:
        """Tokens of the cut's turns the plan would not keep word for word."""
        return sum(turn.token_count for span, level in self.placed if level is not DetailLevel.KEPT
                   for turn in span.turns)

    def tokens_at(self, *levels: DetailLevel) -> int:
        return sum(turn.token_count for span, level in self.placed if level in levels for turn in span.turns)

    def turns_at(self, *levels: DetailLevel) -> int:
        return sum(len(span.turns) for span, level in self.placed if level in levels)

    def delta(self) -> AgingDelta:
        return AgingDelta(
            departed_turns=frozenset(turn.turn_index for turn in self.cut_turns),
            new_spans=tuple(replace(span, level=level) for span, level in self.placed),
            level_changes=tuple((span, level) for span, level in self.replanned if level is not span.level),
            new_knowledge=self.new_facts,
        )

    def outcome(self) -> CutOutcome:
        lowered = sum(len(span.turns) for span, level in self.replanned
                      if level in _CONDENSED and _RANK[level] < _RANK[span.level])
        raised = sum(len(span.turns) for span, level in self.replanned if _RANK[level] > _RANK[span.level])
        return CutOutcome(quoted=self.turns_at(DetailLevel.KEPT), summarised=self.turns_at(DetailLevel.COMPACTED),
                          archived=self.turns_at(*_CONDENSED), front_archived=lowered, front_raised=raised)

    def preview(self, min_cut_tokens: int) -> CutPreview:
        new = self.spans[self.new_from:]
        scores = self.scores[self.new_from:]
        return CutPreview(
            turns=tuple(turn.turn_index for turn in self.cut_turns),
            levels={turn.turn_index: level for span, level in self.placed for turn in span.turns},
            scores={turn.turn_index: score for span, score in zip(new, scores, strict=True) for turn in span.turns},
            compactable_tokens=self.compactable_tokens,
            min_cut_tokens=min_cut_tokens,
        )


class GcService:
    """Session-aware engine facade: the operations the proxy calls.

    - `gc_collect` (hot path): the session's context - the front, then the turns
      after it word for word. Fast, because it only renders what the last
      collection decided.
    - `gc_age` (cold path, after a response): a collection. The turns older than
      the tail are condensed into spans, every span is scored against what is being
      discussed now, and each is given the detail level its score earns within the
      recap's budget. This is the pass that reports on the event bus.

    It owns no conversation data: each session's generations live in its own
    `SessionState.snapshot`. Reads never wait for writes; see
    doc/design/session/overview.md for the concurrency model.
    """

    def __init__(self, session_manager: SessionManager, generational_memory: GenerationalMemory,
                 span_scorer: SpanScorer, event_bus: EventBus, gc_config: GCConfig) -> None:
        self.session_manager = session_manager
        self.generational_memory = generational_memory
        self.span_scorer = span_scorer
        self.event_bus = event_bus
        self.gc_config = gc_config
        self._bands = Bands(keep=gc_config.keep_threshold, archive=gc_config.archive_threshold,
                            margin=gc_config.level_margin)

    def _assemble(self, snapshot: GenerationSnapshot) -> list[Message]:
        """The context this session sends: the front, then the verbatim tail.

        The single definition of "the context this session would send". Both
        `gc_collect` (which returns it) and `gc_age` (which reports it as
        `final_messages`) call this, so the number the dashboard shows and the
        payload the proxy forwards can never drift apart.

        Every change to the prompt happens at the front, and only when a collection
        commits; the tail is the conversation exactly as recorded. A prompt that
        changed anywhere else would reset the provider's cache and invalidate the
        model's thinking blocks on every request.
        """
        front = self._render_front(snapshot)
        return ([front.message] if front is not None else []) + _in_order(snapshot.young)

    @staticmethod
    def _render_front(snapshot: GenerationSnapshot) -> Front | None:
        return render_front(snapshot.old, snapshot.knowledge)

    async def gc_collect(self, session_id: str) -> list[Message]:
        """Hot path: the session's context, from its current snapshot.

        Never waits. The snapshot is immutable, so a collection mid-way through a
        model call cannot be observed half-done; at worst this returns the context
        from before it, which is complete and correct.

        Emits nothing: this path changes no state.
        """
        state = self.session_manager.get_or_create(session_id)
        return self._assemble(state.snapshot)

    def inspect(self, session_id: str) -> SessionInspection | None:
        """A read-only view of one session, or None if there is no such session.

        Uses the same rendering as `gc_collect` and the same planning as `gc_age`,
        so the view cannot drift from what the engine does. No model calls, and
        unlike the request-path methods it never creates a session or resets its
        idle clock: looking must not change what is looked at.
        """
        state = self.session_manager.store.get(session_id)
        if state is None:
            return None
        snapshot = state.snapshot
        young = _in_order(snapshot.young)
        cut = find_cut_point(young, self.gc_config.last_n_turns_to_keep)
        planned = self._plan(snapshot, young, cut, self.generational_memory.draft(young[:cut]),
                             estimate_summaries=True)
        return SessionInspection(
            session_id=session_id,
            snapshot=snapshot,
            front=self._render_front(snapshot),
            tail=tuple(young),
            next_cut=planned.preview(self.gc_config.min_cut_tokens),
            aging_in_flight=state.aging_in_flight,
            last_activity=state.last_updated_at,
        )

    def mark_front_sent(self, session_id: str, history_turns: int) -> int:
        """Note that the current front is being sent after `history_turns` turns of history.

        Returns the first turn whose thinking blocks are still valid. A thinking
        block is bound to the prompt it was produced against, so every turn sent
        with the first request that carries a new front predates it - including the
        reply to the request during which the collection was planned, which the
        commit cannot see. Replies from then on are produced against this front and
        keep their thinking until the next collection. Immediate.
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
        """Record one turn, then run a pass over the session: the library's one-call loop."""
        state = self.session_manager.get_or_create(session_id)
        state.snapshot = state.snapshot.with_turn(new_turn)
        await self.gc_age(session_id)

    async def gc_age(self, session_id: str, *, ignore_batch_minimum: bool = False) -> GCResult:
        """Cold path: collect the session if enough of it would be condensed.

        The plan decides how each span is shown; the batch size decides when.
        Recording is separate and immediate, so the next `gc_collect` sees new
        turns even while a pass is running. At most one pass runs per session: a
        second trigger is reported and skipped rather than queued.

        `ignore_batch_minimum` is the manual lever: the same pass with
        `min_cut_tokens` taken as zero. Every other rule still holds - the newest
        turns stay verbatim, no tool round is split, something must be condensed,
        and the collection must shrink the context.

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
        """Plan a collection against `base`, then commit it onto whatever the snapshot is now.

        Plans twice: first with estimated summaries, to decide whether the batch is
        worth paying for summaries at all, then with the real ones. Nothing reaches
        session state until the commit, which is synchronous: no `await` separates
        reading the current snapshot from replacing it. A failure while planning
        therefore leaves the session exactly as it was.
        """
        session_id = state.session_id
        current_stage = "plan"
        degradations: list[AgingDegradation] = []
        try:
            young = _in_order(base.young)
            cut = find_cut_point(young, self.gc_config.last_n_turns_to_keep)
            drafts = self.generational_memory.draft(young[:cut])
            if not batch_worth_cutting(cut, self._plan(base, young, cut, drafts, estimate_summaries=True)
                                       .compactable_tokens, min_cut_tokens):
                return self._unchanged_result(base, gc_run_id, GCStatus.BYPASSED_BATCH_TOO_SMALL, start)

            current_stage = "condense"
            condensed = await self.generational_memory.summarise(drafts)
            degradations.extend(self._report_failures(session_id, condensed))

            current_stage = "plan"
            planned = self._plan(base, young, cut, condensed, estimate_summaries=False)
            if planned.plan.over_budget:
                degradations.append(self._record_degradation(
                    session_id, DegradationReason.RECAP_OVER_BUDGET, [turn.turn_index for turn in young[:cut]],
                    LLMGCError(f"the folded front needs {planned.plan.tokens} tokens of a "
                               f"{self.gc_config.max_recap_tokens}-token recap budget")))
            delta = planned.delta()
            # The cut must pay for itself: its new spans, measured on the context
            # they would produce, against the turns they replace - a quote's marker
            # can cost more than the turn did. Re-planned older spans are left out:
            # restoring one the conversation returned to is spending the score
            # decided and the budget bounds, not a saving that failed.
            projected = self._context_tokens(apply_aging(base, replace(delta, level_changes=())).snapshot)
            if projected >= self._context_tokens(base):
                logger.info("collection abandoned: it would not shrink the context",
                            extra={"session_id": session_id, "gc_run_id": gc_run_id})
                return self._unchanged_result(base, gc_run_id, GCStatus.BYPASSED_NO_SAVING, start,
                                              degradations=degradations,
                                              abandoned_cut=AbandonedCut(
                                                  turns=[turn.turn_index for turn in planned.cut_turns],
                                                  outcome=planned.outcome(),
                                                  tokens_taken=sum(t.token_count for t in planned.cut_turns),
                                                  context_tokens_after=projected))
        except Exception as err:
            logger.exception("gc_age failed",
                             extra={"session_id": session_id, "gc_run_id": gc_run_id, "stage": current_stage})
            return self._unchanged_result(base, gc_run_id, GCStatus.BYPASSED_ON_ERROR, start,
                                          failure_reason=str(err), failure_stage=current_stage)

        if not self.session_manager.is_live(state):
            logger.info("session closed during aging; result discarded",
                        extra={"session_id": session_id, "gc_run_id": gc_run_id})
            return self._unchanged_result(base, gc_run_id, GCStatus.DISCARDED_SESSION_CLOSED, start)

        applied = apply_aging(state.snapshot, delta)
        state.snapshot = replace(applied.snapshot, thinking_valid_from=None, cut_count=applied.snapshot.cut_count + 1)
        self._announce(session_id, planned, applied)

        logger.debug("collection committed", extra={"session_id": session_id, "gc_run_id": gc_run_id,
                                                     "turns_cut": cut, "young_remaining": len(state.snapshot.young)})
        return self._completed_result(base, state.snapshot, gc_run_id, planned, len(young) - cut, degradations,
                                      start)

    def _plan(self, base: GenerationSnapshot, young: Sequence[Message], cut: int, condensed: Sequence[Condensed],
              *, estimate_summaries: bool) -> _Planned:
        """Score every span against what is being discussed now, and give each its level.

        "Now" is what the user said in the tail. With `estimate_summaries`, a new
        span that could be summarised is costed at the compactor's target ratio,
        because its summary has not been paid for yet.
        """
        new = [Span(turns=draft.turns, level=DetailLevel.KEPT, summary=draft.summary,
                    summary_text=draft.summary_text, index_line=draft.index_line) for draft in condensed]
        spans = list(base.old) + new
        new_facts = tuple(fact for draft in condensed for fact in draft.facts)
        knowledge, _ = file_knowledge(base.knowledge, new_facts)
        facts_by_turn = active_facts_by_turn(knowledge)

        now = [turn for turn in young[cut:] if turn.role == "user" and not tool_result_ids(turn)]
        newest_turn = base.recorded_turn_count - 1
        scores = [score.combined for score in self.span_scorer.score([span.turns for span in spans], now, newest_turn)]

        options = []
        for index, (span, score) in enumerate(zip(spans, scores, strict=True)):
            kept, compacted, archived = level_costs(span, facts_of(span, facts_by_turn))
            is_new = index >= len(base.old)
            if compacted is None and is_new and estimate_summaries and self.generational_memory.worth_summarising(
                    span.turns):
                compacted = int(kept * DEFAULT_COMPACTION_TARGET_RATIO)
            options.append(LevelOptions(score=score, current=None if is_new else span.level, kept_tokens=kept,
                                        compacted_tokens=compacted, archived_tokens=archived))

        fold = count_tokens(folded_line(spans[0].first_turn, spans[-1].turns[-1].turn_index, len(spans))) if spans else 0
        plan = plan_levels(options, self._bands, self.gc_config.max_recap_tokens - FRONT_OVERHEAD_TOKENS, fold)
        return _Planned(spans=tuple(spans), new_from=len(base.old), scores=tuple(scores), plan=plan,
                        new_facts=new_facts)

    def _report_failures(self, session_id: str, condensed: Sequence[Condensed]) -> list[AgingDegradation]:
        """A degradation for each span made without its summary, or without its facts, because something failed.

        Neither keeps the span from being made: without a summary it can still be
        kept or archived, and without facts it is archived by its index line alone.
        """
        degradations = []
        for draft in condensed:
            turns = [turn.turn_index for turn in draft.turns]
            if draft.summary_error is not None:
                reason = (DegradationReason.COMPACTION_REFUSED if isinstance(draft.summary_error, CompactionRefused)
                          else DegradationReason.COMPACTION_FAILED)
                logger.warning("a span was made without its summary",
                               extra={"session_id": session_id, "turn_indices": turns, "reason": reason.value})
                degradations.append(self._record_degradation(session_id, reason, turns, draft.summary_error))
            if draft.facts_error is not None:
                logger.warning("a span was made without its facts",
                               extra={"session_id": session_id, "turn_indices": turns})
                degradations.append(self._record_degradation(session_id, DegradationReason.ARCHIVE_FAILED, turns,
                                                             draft.facts_error))
        return degradations

    def _completed_result(self, base: GenerationSnapshot, final: GenerationSnapshot, gc_run_id: str,
                          planned: _Planned, tail_length: int, degradations: list[AgingDegradation],
                          start: float) -> GCResult:
        final_messages = self._assemble(final)
        return GCResult(
            final_messages=final_messages, gc_run_id=gc_run_id, status=GCStatus.COMPLETED,
            tokens_before=self._context_tokens(base),
            tokens_in_final=sum(message.token_count for message in final_messages),
            kept_count=planned.turns_at(DetailLevel.KEPT) + tail_length,
            compacted_count=planned.turns_at(DetailLevel.COMPACTED),
            archived_count=planned.turns_at(*_CONDENSED),
            kept_tokens=planned.tokens_at(DetailLevel.KEPT),
            compacted_tokens=planned.tokens_at(DetailLevel.COMPACTED),
            archived_tokens=planned.tokens_at(*_CONDENSED),
            duration_ms=(perf_counter() - start) * 1000,
            degradations=degradations, outcome=planned.outcome(),
        )

    def _context_tokens(self, snapshot: GenerationSnapshot) -> int:
        return sum(message.token_count for message in self._assemble(snapshot))

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
            duration_ms=(perf_counter() - start) * 1000,
            degradations=degradations or [], abandoned_cut=abandoned_cut,
        )

    def _announce(self, session_id: str, planned: _Planned, applied: AppliedAging) -> None:
        """Emit the events for a committed collection, after the commit.

        Emitted only now so an event always describes a change that happened: a
        pass discarded because its session closed announces nothing.
        """
        for span, level in planned.placed:
            turns = [turn.turn_index for turn in span.turns]
            if level is DetailLevel.COMPACTED and span.summary is not None:
                self.event_bus.emit(Event(event_type=EventType.MESSAGE_PROMOTED_TO_OLD_GEN, data={
                    "session_id": session_id,
                    "source_turn_indices": turns,
                    "messages_compacted": len(turns),
                    "tokens_before": sum(turn.token_count for turn in span.turns),
                    "tokens_after": span.summary.token_count,
                }))
            elif level in _CONDENSED:
                self._announce_archived(session_id, span, [f for f in planned.new_facts if f.message_turn in turns],
                                        "young")
        for span, level in planned.replanned:
            if level in _CONDENSED and _RANK[level] < _RANK[span.level]:
                self._announce_archived(session_id, span, [], "old")

        superseded_by = {id(s.new_entry): s.superseded for s in applied.supersessions}
        for entry in planned.new_facts:
            self.event_bus.emit(Event(event_type=EventType.KNOWLEDGE_ENTRY_ADDED,
                                      data={"session_id": session_id, "knowledge_entry": entry}))
            if id(entry) in superseded_by:
                self.event_bus.emit(Event(event_type=EventType.KNOWLEDGE_ENTRY_SUPERSEDED, data={
                    "session_id": session_id,
                    "new_knowledge_entry": entry,
                    "superseded_knowledge_entries": list(superseded_by[id(entry)]),
                }))

    def _announce_archived(self, session_id: str, span: Span, facts: list[KnowledgeEntry], source: str) -> None:
        self.event_bus.emit(Event(event_type=EventType.MESSAGE_ARCHIVED, data={
            "session_id": session_id,
            "message": span.turns[0],
            "extracted_knowledge_entries": facts,
            "source_generation": source,
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
