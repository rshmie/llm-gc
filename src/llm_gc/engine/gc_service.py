import logging
import math
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from time import perf_counter

from llm_gc.config.constants import DEFAULT_COMPACTION_TARGET_RATIO
from llm_gc.config.gc_config import GCConfig
from llm_gc.models.aging_degradation import AgingDegradation, DegradationReason
from llm_gc.engine.cut_point import find_cut_point, is_safe_cut
from llm_gc.engine.gc_result import AbandonedCut, CutOutcome, GCResult
from llm_gc.engine.gc_status import GCStatus
from llm_gc.engine.generations.condensing import Condensed
from llm_gc.engine.generations.front import (
    FRONT_OVERHEAD_TOKENS,
    Front,
    KnowledgeIndex,
    folded_line,
    level_costs,
    render_front,
)
from llm_gc.engine.generations.generational_memory import GenerationalMemory
from llm_gc.engine.generations.planning import Bands, LevelOptions, Plan, plan_levels
from llm_gc.engine.inspection import CutPreview, SessionInspection, SwitchPreview, batch_worth_cutting
from llm_gc.events import Event, EventBus, EventType
from llm_gc.engine.pause import MAX_PAUSE, ModelCallPause
from llm_gc.engine.switch_policy import CacheState, SwitchReason, decide_switch
from llm_gc.exceptions import CompactionRefused, LLMAuthError, LLMGCError, LLMRateLimitError, LLMResponseError
from llm_gc.llm.metering import measure_model_usage
from llm_gc.models import KnowledgeEntry, Message
from llm_gc.models.knowledge_entry import KnowledgeStatus
from llm_gc.models.message import said_by_user
from llm_gc.models.generation_snapshot import (
    AgingDelta,
    AppliedAging,
    DetailLevel,
    GenerationSnapshot,
    Span,
    StagedCut,
    apply_aging,
    file_knowledge,
)
from llm_gc.scoring import SpanScorer
from llm_gc.session import SessionManager, SessionState
from llm_gc.utils import count_tokens, count_tokens_cached

logger = logging.getLogger(__name__)

_RANK = {DetailLevel.OUT_OF_VIEW: 0, DetailLevel.ARCHIVED: 1, DetailLevel.COMPACTED: 2, DetailLevel.KEPT: 3}
_CONDENSED = (DetailLevel.ARCHIVED, DetailLevel.OUT_OF_VIEW)


@dataclass(frozen=True)
class _Planned:
    """One plan over every span: the old generation's as they stand, then the cut's new ones.

    New spans carry a provisional level until the plan gives them theirs. Of the new ones,
    those before `drafts_from` are already condensed - a staged cut's - and those from it
    on are drafts whose turns a pass would still pay to condense.
    """

    spans: tuple[Span, ...]
    new_from: int
    drafts_from: int
    scores: tuple[float, ...]
    options: tuple[LevelOptions, ...]
    plan: Plan
    new_facts: tuple[KnowledgeEntry, ...]

    @property
    def placed(self) -> list[tuple[Span, DetailLevel]]:
        """The cut's new spans, each with the level the plan gave it."""
        return list(zip(self.spans[self.new_from :], self.plan.levels[self.new_from :], strict=True))

    @property
    def replanned(self) -> list[tuple[Span, DetailLevel]]:
        """The old generation's spans, each with the level the plan gives it now."""
        return list(zip(self.spans[: self.new_from], self.plan.levels[: self.new_from], strict=True))

    @property
    def condensed_spans(self) -> list[Span]:
        """The cut's new spans that were condensed before this plan: a staged cut's."""
        return list(self.spans[self.new_from :][: self.drafts_from])

    @property
    def cut_turns(self) -> list[Message]:
        return [turn for span in self.spans[self.new_from :] for turn in span.turns]

    @property
    def compactable_tokens(self) -> int:
        """Tokens of the drafted turns the plan would not keep word for word: what a pass would pay to condense."""
        return _not_kept_tokens(self.placed[self.drafts_from :])

    @property
    def stale_tokens(self) -> int:
        """Tokens of all the cut's turns the plan would not keep word for word: the stale weight."""
        return _not_kept_tokens(self.placed)

    def cut_saving(self, front_exists: bool) -> int:
        """What the cut's new spans save against the turns they replace, at the costs planned.

        Estimated when the summaries are: a collection checks it before paying for
        them. A session's first cut also pays for the front's own header.
        """
        new = zip(
            self.spans[self.new_from :], self.options[self.new_from :], self.plan.levels[self.new_from :], strict=True
        )
        saved = sum(sum(turn.token_count for turn in span.turns) - option.cost(level) for span, option, level in new)
        return saved - (0 if front_exists else FRONT_OVERHEAD_TOKENS)

    def delta(self) -> AgingDelta:
        return AgingDelta(
            departed_turns=frozenset(turn.turn_index for turn in self.cut_turns),
            new_spans=tuple(replace(span, level=level) for span, level in self.placed),
            level_changes=tuple((span, level) for span, level in self.replanned if level is not span.level),
            new_knowledge=self.new_facts,
        )

    def outcome(self) -> CutOutcome:
        # From the levels, not from `delta()`: an estimated plan can place a draft at a level
        # it has no material for yet, which is fine for a forecast and not for a span.
        return _outcome(self.placed, self.replanned)

    def preview(self, min_cut_tokens: int, front_exists: bool) -> CutPreview:
        new = self.spans[self.new_from :]
        scores = self.scores[self.new_from :]
        return CutPreview(
            turns=tuple(turn.turn_index for turn in self.cut_turns),
            levels={turn.turn_index: level for span, level in self.placed for turn in span.turns},
            scores={turn.turn_index: score for span, score in zip(new, scores, strict=True) for turn in span.turns},
            compactable_tokens=self.compactable_tokens,
            min_cut_tokens=min_cut_tokens,
            estimated_saving=self.cut_saving(front_exists) if new else 0,
            spans=tuple(span.turn_indices for span in new),
            span_levels=self.plan.levels[: self.new_from],
            span_scores=self.scores[: self.new_from],
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

    def __init__(
        self,
        session_manager: SessionManager,
        generational_memory: GenerationalMemory,
        span_scorer: SpanScorer,
        event_bus: EventBus,
        gc_config: GCConfig,
    ) -> None:
        self.session_manager = session_manager
        self.generational_memory = generational_memory
        self.span_scorer = span_scorer
        self.event_bus = event_bus
        self.gc_config = gc_config
        # LLM-GC's own model calls yield to the upstream's rate limit; see `pause_model_calls`.
        self.model_calls = ModelCallPause()
        self._bands = Bands(
            keep=gc_config.keep_threshold, archive=gc_config.archive_threshold, margin=gc_config.level_margin
        )

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

    def gc_switch(
        self, session_id: str, cache: CacheState, *, requested: bool = False, tool_round_open: bool = False
    ) -> GCResult | None:
        """Request path: switch the session's staged cut in, if the switch policy says now. Immediate.

        `cache` is the provider cache the request being built would meet; `requested` is
        the developer asking for a collection now; `tool_round_open` says the request
        answers the previous reply's tool calls. A request the open round holds back is
        remembered, and made by the first request outside one. Returns the collection when
        one is switched in, None otherwise. The commit is one synchronous step onto the
        current snapshot with no model call, so the request being built is the first to
        carry the new front. Emits the collection's events, then GC_FINISHED, after the commit.
        """
        start = perf_counter()
        state = self.session_manager.store.get(session_id)
        if state is None:
            return None
        if (staged := state.snapshot.staged) is None:
            state.switch_requested = False  # the cut it asked for is gone
            return None
        requested = requested or state.switch_requested
        preview = self._switch_preview(state, staged, cache, requested, tool_round_open)
        held_by_round = preview is not None and preview.decision.reason is SwitchReason.TOOL_ROUND_OPEN
        state.switch_requested = requested and held_by_round
        if preview is None or not preview.decision.switch:
            return None
        before = state.snapshot
        applied = apply_aging(before, staged.delta)
        state.snapshot = replace(
            applied.snapshot, staged=None, thinking_valid_from=None, cut_count=applied.snapshot.cut_count + 1
        )
        self._announce(session_id, staged.delta, applied)
        result = self._switched_result(before, state.snapshot, staged, preview.decision.reason, start)
        reason = preview.decision.reason.value
        logger.info(
            "applied a collection in %s (%s): about %s fewer tokens on each request, %s of them earlier thinking",
            session_id,
            reason.replace("_", " "),
            f"{preview.saving_tokens:,}",
            f"{preview.thinking_dropped_tokens:,}",
            extra={
                "session_id": session_id,
                "reason": reason,
                "saving_tokens": preview.saving_tokens,
                "thinking_dropped_tokens": preview.thinking_dropped_tokens,
                "young_remaining": len(state.snapshot.young),
            },
        )
        self.event_bus.emit(
            Event(
                event_type=EventType.GC_FINISHED,
                data={"gc_run_id": result.gc_run_id, "session_id": session_id, "report": result.report()},
            )
        )
        return result

    def inspect(
        self, session_id: str, cache: CacheState = CacheState.COLD, *, tool_round_open: bool = False
    ) -> SessionInspection | None:
        """A read-only view of one session, or None if there is no such session.

        Uses the same rendering as `gc_collect`, the same planning as `gc_age` and
        the same measures as `gc_switch`, so the view cannot drift from what the
        engine does. `cache` is the state the next request would meet, and
        `tool_round_open` whether it will answer tool calls, for the verdict on a staged cut. No model calls, and unlike the request-path methods it never
        creates a session or resets its idle clock: looking must not change what is
        looked at.
        """
        state = self.session_manager.store.get(session_id)
        if state is None:
            return None
        snapshot = state.snapshot
        young = _in_order(snapshot.young)
        cut = find_cut_point(young, self.gc_config.last_n_messages_to_keep)
        own_window = self._window_in_own_count(state)
        recap_budget = self.gc_config.recap_budget(own_window)
        planned, _ = self._estimate(snapshot, young, cut, recap_budget)
        preview = planned.preview(self.gc_config.min_cut_tokens, front_exists=bool(snapshot.old))
        if not state.collection_enabled:
            # Nothing will be collected, so nothing is forecast to change.
            preview = replace(
                preview,
                turns=(),
                spans=(),
                levels={},
                compactable_tokens=0,
                estimated_saving=0,
                span_levels=tuple(span.level for span in snapshot.old),
            )
        return SessionInspection(
            session_id=session_id,
            snapshot=snapshot,
            front=self._render_front(snapshot),
            tail=tuple(young),
            next_cut=preview,
            aging_in_flight=state.aging_in_flight,
            last_activity=state.last_updated_at,
            context_window=state.context_window or self.gc_config.context_window,
            recap_budget=recap_budget,
            collection_enabled=state.collection_enabled,
            staged=(
                self._switch_preview(state, snapshot.staged, cache, state.switch_requested, tool_round_open)
                if snapshot.staged is not None
                else None
            ),
            token_rate=state.token_rate,
            window_in_own_count=own_window,
            thinking_tokens=snapshot.sent_thinking_tokens(),
            attachment_tokens=snapshot.sent_attachment_tokens(),
        )

    def use_context_window(self, session_id: str, context_window: int) -> None:
        """Size this session's recap for the window of the model its requests name now, in that model's tokens.

        Read at the next collection. A session that never hears of one uses the configured window.
        """
        state = self.session_manager.store.get(session_id)
        if state is not None:
            state.context_window = context_window

    def use_token_rate(self, session_id: str, token_rate: float) -> None:
        """Note how many of the model's tokens one of LLM-GC's is, as measured on this session's newest reply.

        The window is in the model's tokens; the recap and stale weight are in LLM-GC's own. The
        window is converted at this rate before the recap budget and the stale bound are taken from
        it, so each is the share of the window its setting names. Until a reply is measured, the
        window is used as it is.
        """
        if not (math.isfinite(token_rate) and token_rate > 0):
            raise ValueError(f"a token rate is a positive number, not {token_rate}")
        state = self.session_manager.store.get(session_id)
        if state is not None:
            state.token_rate = token_rate

    def use_rewrite_overhead(self, session_id: str, tokens: int) -> None:
        """Note what a switch would also make the provider cache again, besides the front and the tail's text.

        Measured by the caller on its newest request: content sent after the prompt's last cached
        point that a collection never condenses, such as a client's instructions. Added to the
        cost a switch must repay.
        """
        if tokens < 0:
            raise ValueError(f"a rewrite overhead cannot be negative, got {tokens}")
        state = self.session_manager.store.get(session_id)
        if state is not None:
            state.rewrite_overhead_tokens = tokens

    def pause_model_calls(self, retry_after_s: float | None, reason: str, *, lasting: bool = False) -> None:
        """Make no model call of LLM-GC's own until the upstream's pressure has passed.

        For the upstream's Retry-After when it gave one, a minute otherwise; a later call
        extends the pause. `lasting` marks a pause because LLM-GC's own calls are failing.
        Collections skipped meanwhile are reported as paused, with the reason. The client's
        own requests are never held back.
        """
        if self.model_calls.pause(retry_after_s, reason, lasting=lasting):
            pause = self.model_calls.current()
            logger.warning(
                "LLM-GC pauses its own model calls until %s: %s",
                f"{pause.until.astimezone():%H:%M:%S}" if pause is not None else "?",
                reason,
                extra={"retry_after_s": retry_after_s},
            )

    def set_collection_enabled(self, session_id: str, enabled: bool) -> None:
        """Turn collection on or off for one session. Off, it is still recorded and inspected, never collected."""
        state = self.session_manager.store.get(session_id)
        if state is not None:
            state.collection_enabled = enabled

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

    async def gc_update(self, session_id: str, new_turn: Message, *, cache: CacheState = CacheState.COLD) -> None:
        """Record one turn, prepare a cut, and switch it in if the policy says so: the library's one-call loop.

        `cache` is the provider cache the caller's next request would meet. Cold by default:
        a caller that says nothing of one has none to protect, so a cut takes effect as soon
        as it saves something.

        `new_turn.turn_index` must be the session's next turn: 0 in a new session, then one
        more each call. Cuts and facts are filed by turn, so any other index is refused with
        a ValueError, and nothing is recorded. A turn whose `token_count` is 0 is counted
        here, the way the proxy counts its turns; a count the caller gives is kept.
        """
        state = self.session_manager.store.get(session_id)
        expected = state.snapshot.recorded_turn_count if state is not None else 0
        if new_turn.turn_index != expected:
            raise ValueError(f"turn {expected} is next in this session, got turn_index {new_turn.turn_index}")
        if new_turn.token_count == 0 and new_turn.content:
            new_turn = new_turn.model_copy(update={"token_count": count_tokens(new_turn.content)})
        state = self.session_manager.get_or_create(session_id)
        state.snapshot = state.snapshot.with_turn(new_turn)
        await self.gc_age(session_id)
        self.gc_switch(session_id, cache)

    async def gc_age(self, session_id: str, *, ignore_batch_minimum: bool = False) -> GCResult:
        """Cold path: prepare a cut and stage it, if enough of the session would be condensed.

        The plan decides how each span is shown; the batch size decides whether a pass
        is worth its model calls. A staged cut changes nothing the model is sent until
        `gc_switch` switches it in; a later pass extends it rather than starting again.
        Recording is separate and immediate, so the next `gc_collect` sees new turns even
        while a pass is running. At most one pass runs per session: a second trigger is
        reported and skipped rather than queued.

        `ignore_batch_minimum` is the manual lever: the same pass with
        `min_cut_tokens` taken as zero. Every other rule still holds - the newest
        turns stay verbatim, no tool round is split, something must be condensed,
        and the cut must shrink the context.

        Emits GC_FINISHED exactly once, after any staging, and returns the result.
        """
        start = perf_counter()
        gc_run_id = str(uuid.uuid4())
        state = self.session_manager.get_or_create(session_id)
        base = state.snapshot

        if not state.collection_enabled:
            gc_result = self._unchanged_result(base, gc_run_id, GCStatus.BYPASSED_NOT_COLLECTED, start)
        elif (pause := self.model_calls.current()) is not None:
            gc_result = self._unchanged_result(
                base,
                gc_run_id,
                GCStatus.BYPASSED_PAUSED,
                start,
                failure_reason=f"paused until {pause.until.astimezone():%H:%M:%S}: {pause.reason}",
            )
        elif state.aging_in_flight:
            logger.debug(
                "aging already in flight; skipping this trigger",
                extra={"session_id": session_id, "gc_run_id": gc_run_id},
            )
            gc_result = self._unchanged_result(base, gc_run_id, GCStatus.BYPASSED_AGING_IN_FLIGHT, start)
        else:
            state.aging_in_flight = True
            try:
                min_cut_tokens = 0 if ignore_batch_minimum else self.gc_config.min_cut_tokens
                # Every path counts what it spent, including a cut abandoned after its summaries were bought.
                with measure_model_usage() as spent:
                    gc_result = await self._age(state, base, gc_run_id, start, min_cut_tokens)
                gc_result = gc_result.model_copy(
                    update={
                        "summary_calls": spent.calls,
                        "summary_input_tokens": spent.input_tokens,
                        "summary_output_tokens": spent.output_tokens,
                    }
                )
            finally:
                state.aging_in_flight = False

        self.event_bus.emit(
            Event(
                event_type=EventType.GC_FINISHED,
                data={"gc_run_id": gc_run_id, "session_id": session_id, "report": gc_result.report()},
            )
        )
        return gc_result

    async def _age(
        self, state: SessionState, base: GenerationSnapshot, gc_run_id: str, start: float, min_cut_tokens: int
    ) -> GCResult:
        """Prepare a cut against `base`, then stage it onto whatever the snapshot is now.

        Plans first with the staged spans and estimated summaries for the turns past them,
        to decide whether those turns are worth paying to condense; then condenses them and
        plans again, against the snapshot as it is once the model calls are back. A switch,
        a rewind or turns recorded in the meantime are taken in there: spans whose turns are
        no longer the oldest young turns are left out, so nothing is staged against a stale
        base and nothing condensed is paid for twice. Nothing reaches session state until
        the staging, which is synchronous: no `await` separates reading the current snapshot
        from replacing it. A failure therefore leaves the session exactly as it was.
        """
        session_id = state.session_id
        keep_last = self.gc_config.last_n_messages_to_keep
        current_stage = "plan"
        degradations: list[AgingDegradation] = []
        try:
            young = _in_order(base.young)
            cut = find_cut_point(young, keep_last)
            # Sized for the window of the model this conversation's requests name now.
            recap_budget = self.gc_config.recap_budget(self._window_in_own_count(state))
            estimated, drafts = self._estimate(base, young, cut, recap_budget)
            if not batch_worth_cutting(cut, estimated.compactable_tokens, min_cut_tokens):
                return self._unchanged_result(base, gc_run_id, GCStatus.BYPASSED_BATCH_TOO_SMALL, start)
            if (saving := estimated.cut_saving(front_exists=bool(base.old))) <= 0:
                # Decided before any summary is bought: a cut that cannot pay for itself
                # would buy summaries only to discard them, on every pass.
                return self._unchanged_result(
                    base,
                    gc_run_id,
                    GCStatus.BYPASSED_NO_SAVING,
                    start,
                    abandoned_cut=self._abandoned(base, estimated, saving),
                )

            current_stage = "condense"
            filed, _ = file_knowledge(base.knowledge, estimated.new_facts)
            on_record = [entry for entry in filed if entry.knowledge_status is KnowledgeStatus.ACTIVE]
            condensed = await self.generational_memory.complete(drafts, on_record)
            degradations.extend(self._report_failures(session_id, condensed))
            # Facts an earlier pass failed to read are read again, in their own call, so one exchange's
            # content cannot fail the new spans' facts. Spans this staged cut has read again already are not.
            carried = base.staged.delta.facts_read if base.staged is not None else ()
            unread = [span for span in base.old if span.facts_unread and span not in carried]
            reread, reread_error = await self.generational_memory.read_facts([span.turns for span in unread], on_record)
            if reread_error is not None:
                logger.info(
                    "facts of earlier exchanges could not be read again; they stay marked",
                    extra={"session_id": session_id, "gc_run_id": gc_run_id, "error_type": type(reread_error).__name__},
                )

            # No await from here to the staging: plan against the snapshot as it is now.
            current_stage = "plan"
            current = state.snapshot
            now_young = _in_order(current.young)
            material = _still_oldest(now_young, [*estimated.condensed_spans, *map(_span_of, condensed)])
            cut_now = sum(len(span.turns) for span in material)
            if not cut_now or len(now_young) - cut_now < keep_last or not is_safe_cut(now_young, cut_now):
                logger.info(
                    "the prepared turns are no longer the session's oldest; not staged",
                    extra={"session_id": session_id, "gc_run_id": gc_run_id},
                )
                return self._unchanged_result(
                    current, gc_run_id, GCStatus.DISCARDED_HISTORY_CHANGED, start, degradations=degradations
                )
            held = {turn.turn_index for span in material for turn in span.turns}
            facts = [
                fact
                for fact in (*estimated.new_facts, *(f for draft in condensed for f in draft.facts))
                if fact.message_turn in held
            ]
            # Spans whose facts were read again, now or by the pass that staged this cut, as the snapshot still
            # holds them; planned as readable, so they can be archived again.
            read_again = tuple(
                span for span in (*carried, *(unread if reread_error is None else ())) if span in current.old
            )
            read_turns = {turn.turn_index for span in read_again for turn in span.turns}
            staged_facts = current.staged.delta.new_knowledge if current.staged is not None else ()
            late_facts = [fact for fact in (*staged_facts, *reread) if fact.message_turn in read_turns]
            readable = (
                replace(
                    current,
                    old=tuple(
                        replace(span, facts_unread=False) if span in read_again else span for span in current.old
                    ),
                )
                if read_again
                else current
            )
            planned = self._plan(
                readable,
                now_young,
                cut_now,
                material,
                [*facts, *late_facts],
                drafts_from=len(material),
                recap_budget=recap_budget,
            )
            if planned.plan.over_budget:
                degradations.append(
                    self._record_degradation(
                        session_id,
                        DegradationReason.RECAP_OVER_BUDGET,
                        sorted(held),
                        LLMGCError(
                            f"the folded front needs {planned.plan.tokens} tokens of a "
                            f"{recap_budget}-token recap budget"
                        ),
                    )
                )
            delta = replace(planned.delta(), facts_read=read_again)
            # The cut must pay for itself: its new spans, measured on the context
            # they would produce, against the turns they replace - a quote's marker
            # can cost more than the turn did. Re-planned older spans are left out:
            # restoring one the conversation returned to is spending the score
            # decided and the budget bounds, not a saving that failed.
            before = self._context_tokens(current)
            saving = before - self._context_tokens(apply_aging(current, replace(delta, level_changes=())).snapshot)
            if saving <= 0:
                logger.info(
                    "cut not staged: it would not shrink the context",
                    extra={"session_id": session_id, "gc_run_id": gc_run_id},
                )
                return self._unchanged_result(
                    current,
                    gc_run_id,
                    GCStatus.BYPASSED_NO_SAVING,
                    start,
                    degradations=degradations,
                    abandoned_cut=self._abandoned(current, planned, saving),
                )
            front = self._render_front(apply_aging(current, delta).snapshot)
            # What an earlier preparation of the spans still held fell short of stays with them.
            earlier = current.staged.degradations if current.staged is not None else ()
            staged = StagedCut(
                delta=delta,
                saving_tokens=saving,
                stale_tokens=planned.stale_tokens,
                front_tokens=front.message.token_count if front is not None else 0,
                degradations=(*(d for d in earlier if held & set(d.turn_indices)), *degradations),
            )
        except Exception as err:
            logger.exception(
                "gc_age failed", extra={"session_id": session_id, "gc_run_id": gc_run_id, "stage": current_stage}
            )
            # The result reaches the bus and the dashboard, so as in `_record_degradation` only our own
            # errors' messages, which carry no content, are shown; the log above keeps the rest.
            reason = str(err) if isinstance(err, LLMGCError) else type(err).__name__
            return self._unchanged_result(
                base, gc_run_id, GCStatus.BYPASSED_ON_ERROR, start, failure_reason=reason, failure_stage=current_stage
            )

        if not self.session_manager.is_live(state):
            logger.info(
                "session closed during aging; result discarded",
                extra={"session_id": session_id, "gc_run_id": gc_run_id},
            )
            return self._unchanged_result(base, gc_run_id, GCStatus.DISCARDED_SESSION_CLOSED, start)

        state.snapshot = replace(current, staged=staged, version=current.version + 1)
        logger.info(
            "prepared a collection in %s: %d older turns condensed, %s tokens of text shorter; not applied yet",
            session_id,
            cut_now,
            f"{saving:,}",
            extra={"session_id": session_id, "gc_run_id": gc_run_id, "turns_staged": cut_now, "saving_tokens": saving},
        )
        return self._unchanged_result(state.snapshot, gc_run_id, GCStatus.STAGED, start, degradations=degradations)

    def _estimate(
        self, base: GenerationSnapshot, young: Sequence[Message], cut: int, recap_budget: int
    ) -> tuple[_Planned, list[Condensed]]:
        """The cut a pass would prepare now, at estimated costs, and the drafts it would pay to condense.

        A staged cut's spans come first, as they are; the turns past them up to `cut` are
        drafted. The same estimate is the pass's decision and the dashboard's preview.
        """
        staged = base.staged.delta if base.staged is not None else AgingDelta()
        held = len(staged.departed_turns)
        drafts = self.generational_memory.draft(young[held:cut])
        planned = self._plan(
            base,
            young,
            max(cut, held),
            [*staged.new_spans, *map(_span_of, drafts)],
            staged.new_knowledge,
            drafts_from=len(staged.new_spans),
            recap_budget=recap_budget,
        )
        return planned, drafts

    def _plan(
        self,
        base: GenerationSnapshot,
        young: Sequence[Message],
        cut: int,
        new: Sequence[Span],
        new_facts: Sequence[KnowledgeEntry],
        *,
        drafts_from: int,
        recap_budget: int,
    ) -> _Planned:
        """Score every span against what is being discussed now, and give each its level.

        "Now" is what the user said in the tail. `new` are the cut's spans; those from
        `drafts_from` on are drafts, so one that could be summarised is costed as its
        archived line plus the compactor's target share of its words, because its summary
        has not been paid for yet.
        """
        spans = list(base.old) + [replace(span, level=DetailLevel.KEPT) for span in new]
        knowledge, _ = file_knowledge(base.knowledge, tuple(new_facts))
        annotated = KnowledgeIndex(knowledge)

        now = [turn for turn in young[cut:] if said_by_user(turn)]
        newest_turn = base.recorded_turn_count - 1
        scores = [score.combined for score in self.span_scorer.score([span.turns for span in spans], now, newest_turn)]

        options = []
        for index, (span, score) in enumerate(zip(spans, scores, strict=True)):
            kept, compacted, archived = level_costs(span, annotated.for_span(span))
            position = index - len(base.old)
            is_new = position >= 0
            if compacted is None and position >= drafts_from and self.generational_memory.worth_summarising(span.turns):
                # A summarised span shows its archived line too, so its summary costs on top of that.
                compacted = archived + int(kept * DEFAULT_COMPACTION_TARGET_RATIO)
            options.append(
                LevelOptions(
                    score=score,
                    current=None if is_new else span.level,
                    kept_tokens=kept,
                    compacted_tokens=compacted,
                    archived_tokens=archived,
                    archivable=not span.facts_unread,
                )
            )

        fold = (
            count_tokens_cached(folded_line(spans[0].first_turn, spans[-1].turns[-1].turn_index, len(spans)))
            if spans
            else 0
        )
        plan = plan_levels(options, self._bands, recap_budget - FRONT_OVERHEAD_TOKENS, fold)
        return _Planned(
            spans=tuple(spans),
            new_from=len(base.old),
            drafts_from=drafts_from,
            scores=tuple(scores),
            options=tuple(options),
            plan=plan,
            new_facts=tuple(new_facts),
        )

    def _abandoned(self, base: GenerationSnapshot, planned: _Planned, saving: int) -> AbandonedCut:
        """The cut a pass planned and did not make, and what it would have left the context at."""
        return AbandonedCut(
            turns=[turn.turn_index for turn in planned.cut_turns],
            outcome=planned.outcome(),
            tokens_taken=sum(turn.token_count for turn in planned.cut_turns),
            context_tokens_after=self._context_tokens(base) - saving,
        )

    def _report_failures(self, session_id: str, condensed: Sequence[Condensed]) -> list[AgingDegradation]:
        """A degradation for each summary that failed, and one for facts that could not be read.

        Neither keeps a span from being made: without a summary it can still be
        kept or archived, and without facts it is shown by its opening line. Facts
        are read once for every span, so one failure names all their turns.
        """
        degradations = []
        for draft in condensed:
            if draft.summary_error is not None:
                turns = [turn.turn_index for turn in draft.turns]
                reason = (
                    DegradationReason.COMPACTION_REFUSED
                    if isinstance(draft.summary_error, CompactionRefused)
                    else DegradationReason.COMPACTION_FAILED
                )
                logger.warning(
                    "a span was made without its summary",
                    extra={"session_id": session_id, "turn_indices": turns, "reason": reason.value},
                )
                degradations.append(self._record_degradation(session_id, reason, turns, draft.summary_error))
        # One of LLM-GC's own calls was rate-limited: it shares the upstream's limit with the client, so it yields.
        for error in [draft.summary_error for draft in condensed] + [draft.facts_error for draft in condensed]:
            if isinstance(error, LLMRateLimitError):
                self.pause_model_calls(error.retry_after_s, "the upstream rate-limited one of LLM-GC's own calls")
                break
        # Every call failed in a way retrying cannot fix: calling again each pass would fail the same way.
        if (lasting := _failure_needing_a_person(condensed)) is not None:
            self.pause_model_calls(
                MAX_PAUSE.total_seconds(), f"every one of its model calls failed: {lasting}", lasting=True
            )
        unread = [(draft, draft.facts_error) for draft in condensed if draft.facts_error is not None]
        if unread:
            turns = [turn.turn_index for draft, _ in unread for turn in draft.turns]
            logger.warning(
                "spans were made without their facts", extra={"session_id": session_id, "turn_indices": turns}
            )
            degradations.append(
                self._record_degradation(session_id, DegradationReason.ARCHIVE_FAILED, turns, unread[0][1])
            )
        return degradations

    def _switch_preview(
        self,
        state: SessionState,
        staged: StagedCut,
        cache: CacheState,
        requested: bool = False,
        tool_round_open: bool = False,
    ) -> SwitchPreview | None:
        """The measures the switch policy weighs for the session's staged cut now, and its verdict.

        None when it may not be switched in at all: collection is off for the session, or
        a rewind has left fewer turns after the cut than stay verbatim. The one place the
        measures are taken, for the switch and for the dashboard alike.
        """
        if not state.collection_enabled:
            return None
        snapshot = state.snapshot
        tail = [turn for turn in snapshot.young if turn.turn_index not in staged.delta.departed_turns]
        if len(tail) < self.gc_config.last_n_messages_to_keep:
            return None
        departed = [turn for turn in snapshot.young if turn.turn_index in staged.delta.departed_turns]
        # After a switch no turn sends its thinking, so only text and attachments are cached again, with
        # whatever the caller sends after the prompt's last cached point and never condenses.
        overhead = state.rewrite_overhead_tokens
        tail_attachments = snapshot.sent_attachment_tokens(tail)
        rewritten = staged.front_tokens + sum(turn.token_count for turn in tail) + tail_attachments + overhead
        # The front carries a taken turn's attachments only as placeholders, so a switch stops sending them all.
        dropped_attachments = snapshot.sent_attachment_tokens(departed)
        # Measured now, not at staging: every reply since then has added thinking the switch would drop.
        dropped = snapshot.sent_thinking_tokens()
        stale_thinking = snapshot.sent_thinking_tokens(
            turn for span in staged.delta.new_spans if span.level is not DetailLevel.KEPT for turn in span.turns
        )
        saving = staged.saving_tokens + dropped + dropped_attachments
        stale = staged.stale_tokens + stale_thinking + dropped_attachments
        bound = self.gc_config.stale_bound(self._window_in_own_count(state))
        share = self.gc_config.min_saving_share
        decision = decide_switch(
            saving_tokens=saving,
            rewritten_tokens=rewritten,
            stale_tokens=stale,
            stale_bound_tokens=bound,
            min_saving_share=share,
            cache=cache,
            requested=requested,
            tool_round_open=tool_round_open,
        )
        return SwitchPreview(
            saving_tokens=saving,
            rewritten_tokens=rewritten,
            stale_tokens=stale,
            stale_bound_tokens=bound,
            min_saving_share=share,
            cache=cache,
            decision=decision,
            thinking_dropped_tokens=dropped,
            stale_thinking_tokens=stale_thinking,
            rewrite_overhead_tokens=overhead,
            attachments_dropped_tokens=dropped_attachments,
            rewritten_attachment_tokens=tail_attachments,
        )

    def _window_in_own_count(self, state: SessionState) -> int:
        """The window of the model this session uses, counted as LLM-GC counts: what its recap budget and
        stale bound are shares of, since what they are set against is counted that way."""
        window = state.context_window or self.gc_config.context_window
        return int(window / state.token_rate) if state.token_rate is not None else window

    def _switched_result(
        self,
        before: GenerationSnapshot,
        after: GenerationSnapshot,
        staged: StagedCut,
        reason: SwitchReason,
        start: float,
    ) -> GCResult:
        """The collection a switch made, with what preparing its cut fell short of."""
        delta = staged.delta

        def turns_at(*levels: DetailLevel) -> int:
            return sum(len(span.turns) for span in delta.new_spans if span.level in levels)

        def tokens_at(*levels: DetailLevel) -> int:
            return sum(turn.token_count for span in delta.new_spans if span.level in levels for turn in span.turns)

        final_messages = self._assemble(after)
        return GCResult(
            final_messages=final_messages,
            gc_run_id=str(uuid.uuid4()),
            status=GCStatus.COMPLETED,
            tokens_before=self._context_tokens(before),
            tokens_in_final=sum(message.token_count for message in final_messages),
            kept_count=turns_at(DetailLevel.KEPT) + len(after.young),
            compacted_count=turns_at(DetailLevel.COMPACTED),
            archived_count=turns_at(*_CONDENSED),
            kept_tokens=tokens_at(DetailLevel.KEPT),
            compacted_tokens=tokens_at(DetailLevel.COMPACTED),
            archived_tokens=tokens_at(*_CONDENSED),
            duration_ms=(perf_counter() - start) * 1000,
            degradations=list(staged.degradations),
            outcome=_outcome([(span, span.level) for span in delta.new_spans], delta.level_changes),
            switch_reason=reason,
            thinking_dropped_tokens=before.sent_thinking_tokens(),
            attachments_dropped_tokens=before.sent_attachment_tokens(
                turn for turn in before.young if turn.turn_index in delta.departed_turns
            ),
        )

    def _context_tokens(self, snapshot: GenerationSnapshot) -> int:
        return sum(message.token_count for message in self._assemble(snapshot))

    def _unchanged_result(
        self,
        snapshot: GenerationSnapshot,
        gc_run_id: str,
        status: GCStatus,
        start: float,
        failure_reason: str | None = None,
        failure_stage: str | None = None,
        degradations: list[AgingDegradation] | None = None,
        abandoned_cut: AbandonedCut | None = None,
    ) -> GCResult:
        messages = self._assemble(snapshot)
        tokens = sum(message.token_count for message in messages)
        return GCResult(
            final_messages=messages,
            gc_run_id=gc_run_id,
            status=status,
            failure_reason=failure_reason,
            failure_stage=failure_stage,
            tokens_before=tokens,
            tokens_in_final=tokens,
            kept_count=len(messages),
            compacted_count=0,
            archived_count=0,
            duration_ms=(perf_counter() - start) * 1000,
            degradations=degradations or [],
            abandoned_cut=abandoned_cut,
        )

    def _announce(self, session_id: str, delta: AgingDelta, applied: AppliedAging) -> None:
        """Emit the events for a collection switched in, after the commit.

        Emitted only now so an event always describes a change that happened: a cut
        that is staged and never switched in announces nothing, and a fact that was not
        filed is not announced. Payloads name turns and counts, never what was said: a
        subscriber may log or export what it hears, and the words stay in the session.
        """
        for span in delta.new_spans:
            turns = [turn.turn_index for turn in span.turns]
            if span.level is DetailLevel.COMPACTED and span.summary is not None:
                self.event_bus.emit(
                    Event(
                        event_type=EventType.MESSAGE_PROMOTED_TO_OLD_GEN,
                        data={
                            "session_id": session_id,
                            "source_turn_indices": turns,
                            "messages_compacted": len(turns),
                            "tokens_before": sum(turn.token_count for turn in span.turns),
                            "tokens_after": span.summary.token_count,
                        },
                    )
                )
            elif span.level in _CONDENSED:
                filed = sum(1 for entry in applied.filed if entry.message_turn in turns)
                self._announce_archived(session_id, span, filed, "young")
        for span, level in delta.level_changes:
            if level in _CONDENSED and _RANK[level] < _RANK[span.level]:
                self._announce_archived(session_id, span, 0, "old")

        for entry in applied.filed:
            self.event_bus.emit(
                Event(
                    event_type=EventType.KNOWLEDGE_ENTRY_ADDED,
                    data={
                        "session_id": session_id,
                        "message_turn": entry.message_turn,
                        "knowledge_type": entry.knowledge_type.value,
                        "knowledge_status": entry.knowledge_status.value,
                        "speaker": entry.speaker,
                    },
                )
            )
        for supersession in applied.supersessions:
            self.event_bus.emit(
                Event(
                    event_type=EventType.KNOWLEDGE_ENTRY_SUPERSEDED,
                    data={
                        "session_id": session_id,
                        "message_turn": supersession.new_entry.message_turn,
                        "superseded_turns": [entry.message_turn for entry in supersession.superseded],
                    },
                )
            )

    def _announce_archived(self, session_id: str, span: Span, facts_filed: int, source: str) -> None:
        self.event_bus.emit(
            Event(
                event_type=EventType.MESSAGE_ARCHIVED,
                data={
                    "session_id": session_id,
                    "source_turn_indices": [turn.turn_index for turn in span.turns],
                    "knowledge_entries_filed": facts_filed,
                    "source_generation": source,
                },
            )
        )

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


def _failure_needing_a_person(condensed: Sequence[Condensed]) -> LLMGCError | None:
    """The error every model call of a pass failed with, when only a person can fix it; None otherwise.

    That is a refused key, or a request refused for a reason other than its rate or its size,
    such as an account out of credit. A pass in which any call worked, or any failure was
    one that retrying can fix, returns None.
    """
    errors = [error for draft in condensed for error in (draft.summary_error, draft.facts_error) if error is not None]
    worked = any(draft.summary is not None or draft.facts_error is None for draft in condensed)
    if worked or not errors or not all(map(_needs_a_person, errors)):
        return None
    first = errors[0]
    return first if isinstance(first, LLMGCError) else None


def _needs_a_person(error: Exception) -> bool:
    if isinstance(error, LLMAuthError):
        return True
    status = error.status_code if isinstance(error, LLMResponseError) else None
    return isinstance(error, LLMResponseError) and not error.retryable and status is not None and 400 <= status < 500


def _in_order(messages: Sequence[Message]) -> list[Message]:
    return sorted(messages, key=lambda message: message.turn_index)


def _span_of(condensed: Condensed) -> Span:
    """A condensed span's material as a span, at a provisional level until a plan gives it one."""
    return Span(
        turns=condensed.turns,
        level=DetailLevel.KEPT,
        summary=condensed.summary,
        summary_text=condensed.summary_text,
        index_line=condensed.index_line,
        facts_unread=condensed.facts_error is not None,
    )


def _still_oldest(young: Sequence[Message], spans: Sequence[Span]) -> list[Span]:
    """The spans that still hold the oldest young turns, unchanged and in order.

    Spans whose turns have already left the young generation - switched in while a pass
    ran - are passed over. Matching stops at the first span whose turns no longer match,
    which only a rewind causes.
    """
    if not young:
        return []
    matched: list[Span] = []
    position = 0
    for span in spans:
        if span.turns[-1].turn_index < young[0].turn_index:
            continue
        if tuple(young[position : position + len(span.turns)]) != span.turns:
            break
        matched.append(span)
        position += len(span.turns)
    return matched


def _not_kept_tokens(placed: Sequence[tuple[Span, DetailLevel]]) -> int:
    return sum(turn.token_count for span, level in placed if level is not DetailLevel.KEPT for turn in span.turns)


def _outcome(placed: Sequence[tuple[Span, DetailLevel]], replanned: Sequence[tuple[Span, DetailLevel]]) -> CutOutcome:
    """Where a cut's turns end up, by turn: its new spans at their levels, and older spans re-planned.

    Each older span is paired with its level after the cut; its own `level` is the one before.
    """

    def placed_at(level: DetailLevel) -> list[int]:
        return [turn for span, at in placed if at is level for turn in span.turn_indices]

    def moved(direction: Callable[[int, int], bool]) -> list[int]:
        return [
            turn
            for span, level in replanned
            if direction(_RANK[level], _RANK[span.level])
            for turn in span.turn_indices
        ]

    return CutOutcome(
        kept=placed_at(DetailLevel.KEPT),
        compacted=placed_at(DetailLevel.COMPACTED),
        archived=placed_at(DetailLevel.ARCHIVED),
        folded=placed_at(DetailLevel.OUT_OF_VIEW),
        lowered=moved(lambda now, was: now < was),
        raised=moved(lambda now, was: now > was),
    )
