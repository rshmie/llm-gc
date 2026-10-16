"""A pass stages a cut and the request path switches it in: what each step may change, and when."""

import asyncio

from llm_gc.config import GCConfig
from llm_gc.engine import GCStatus
from llm_gc.engine.compaction import BaseCompactor, CompactionResult
from llm_gc.engine.compaction.compaction_strategy import CompactionStrategy
from llm_gc.engine.gc_service import GcService
from llm_gc.engine.generations import GenerationalMemory
from llm_gc.engine.switch_policy import CacheState, SwitchReason
from llm_gc.events import EventBus, EventType
from llm_gc.extraction import PatternFactReader
from llm_gc.models import Message
from llm_gc.session import SessionManager
from tests.engine.support import COMPACTED_SCORE, ScriptedSpanScorer, make_service, turn

SESSION = "s"
# A warm cache with these never switches on its own: no saving repays a thousandfold, and the bound is the window.
PATIENT = {"min_saving_share": 1000.0, "max_stale_ratio": 1.0}


def _service(**config):
    service, _, provider = make_service(GCConfig(**{"min_cut_tokens": 0, "last_n_turns_to_keep": 1, **config}),
                                        default_score=COMPACTED_SCORE)
    return service, provider


def _record(service, *indices):
    state = service.session_manager.get_or_create(SESSION)
    for index in indices:
        state.snapshot = state.snapshot.with_turn(turn(index))
    return state


def _prepare(service):
    return asyncio.run(service.gc_age(SESSION))


def _sent(service) -> list[int]:
    return [message.turn_index for message in asyncio.run(service.gc_collect(SESSION))]


def _listen(service, *event_types):
    heard = []
    for event_type in event_types:
        service.event_bus.subscribe(event_type, heard.append)
    return heard


class TestAPassChangesNothingTheModelIsSent:
    def test_a_prepared_cut_is_staged_and_the_prompt_is_unchanged(self):
        service, _ = _service()
        state = _record(service, 0, 1, 2, 3)
        before = _sent(service)

        result = _prepare(service)

        assert result.status is GCStatus.STAGED
        assert _sent(service) == before == [0, 1, 2, 3]
        assert state.snapshot.staged is not None and state.snapshot.cut_count == 0

    def test_staging_announces_no_collection(self):
        service, _ = _service()
        _record(service, 0, 1, 2, 3)
        heard = _listen(service, EventType.MESSAGE_PROMOTED_TO_OLD_GEN, EventType.MESSAGE_ARCHIVED,
                        EventType.KNOWLEDGE_ENTRY_ADDED)

        _prepare(service)

        assert heard == [], "events describe changes to what the model is sent, and nothing changed"


class TestTheSwitch:
    def test_a_cold_cache_switches_the_staged_cut_in(self):
        service, _ = _service()
        state = _record(service, 0, 1, 2, 3)
        _prepare(service)

        result = service.gc_switch(SESSION, CacheState.COLD)

        assert result is not None and result.status is GCStatus.COMPLETED
        assert result.switch_reason is SwitchReason.CACHE_COLD
        assert result.tokens_in_final < result.tokens_before
        assert state.snapshot.staged is None and state.snapshot.cut_count == 1
        assert _sent(service)[0] == -1, "the request being built is the first to carry the front"
        assert state.snapshot.thinking_valid_from is None, "set by the first request that sends the front"

    def test_a_warm_cache_waits_while_the_cut_would_not_repay_its_rewrite(self):
        service, _ = _service(**PATIENT)
        state = _record(service, 0, 1, 2, 3)
        _prepare(service)
        heard = _listen(service, EventType.GC_FINISHED)

        assert service.gc_switch(SESSION, CacheState.WARM) is None
        assert state.snapshot.staged is not None and _sent(service) == [0, 1, 2, 3]
        assert heard == []

    def test_stale_weight_at_the_bound_switches_on_a_warm_cache(self):
        service, _ = _service(min_saving_share=1000.0, max_stale_ratio=0.0001)
        _record(service, 0, 1, 2, 3)
        _prepare(service)

        result = service.gc_switch(SESSION, CacheState.WARM)

        assert result is not None and result.switch_reason is SwitchReason.STALE_WEIGHT_AT_BOUND

    def test_the_developer_asking_switches_a_cut_that_would_wait(self):
        service, _ = _service(**PATIENT)
        _record(service, 0, 1, 2, 3)
        _prepare(service)

        result = service.gc_switch(SESSION, CacheState.WARM, requested=True)

        assert result is not None and result.switch_reason is SwitchReason.REQUESTED

    def test_with_nothing_staged_nothing_is_switched_or_announced(self):
        service, _ = _service()
        _record(service, 0, 1, 2, 3)
        heard = _listen(service, EventType.GC_FINISHED)

        assert service.gc_switch(SESSION, CacheState.COLD) is None
        assert heard == [] and _sent(service) == [0, 1, 2, 3]

    def test_a_session_no_longer_collected_is_not_switched(self):
        service, _ = _service()
        state = _record(service, 0, 1, 2, 3)
        _prepare(service)
        service.set_collection_enabled(SESSION, False)

        assert service.gc_switch(SESSION, CacheState.COLD) is None
        assert state.snapshot.cut_count == 0


class TestALaterPassExtendsTheStagedCut:
    def test_turns_staged_earlier_are_not_paid_for_again(self):
        service, provider = _service(**PATIENT)
        state = _record(service, 0, 1, 2, 3)
        _prepare(service)
        first = state.snapshot.staged
        paid_first = len(provider.calls)

        _record(service, 4, 5, 6, 7)
        assert _prepare(service).status is GCStatus.STAGED

        extended = state.snapshot.staged
        assert first is not None and extended is not None
        assert first.delta.departed_turns < extended.delta.departed_turns
        added = [span for span in extended.delta.new_spans if span.first_turn not in first.delta.departed_turns]
        assert paid_first == len(first.delta.new_spans)
        assert len(provider.calls) - paid_first == len(added), "one summary for each new span, none for staged ones"


class _GatedCompactor(BaseCompactor):
    """Summarises only while `gate` is open; `started` fires once a call is waiting on it."""

    COMPACTION_STRATEGY = CompactionStrategy.NOOP

    def __init__(self, event_bus: EventBus) -> None:
        super().__init__(event_bus=event_bus)
        self.started = asyncio.Event()
        self.gate = asyncio.Event()

    async def compact(self, messages):
        self.started.set()
        await self.gate.wait()
        summary = Message(role="assistant", content=f"[Compacted] from turn {messages[0].turn_index}",
                          token_count=1, turn_index=messages[0].turn_index)
        return CompactionResult(summary=summary, original_token_count=sum(m.token_count for m in messages),
                                compaction_strategy=self.COMPACTION_STRATEGY)


class TestASwitchWhileAPassIsRunning:
    def test_the_pass_stages_only_what_is_left_and_no_turn_is_lost_or_doubled(self):
        """A switch lands between a pass's model calls: the pass must re-base, not stage onto a stale snapshot."""
        bus = EventBus()
        compactor = _GatedCompactor(bus)
        service = GcService(
            session_manager=SessionManager(),
            generational_memory=GenerationalMemory(fact_reader=PatternFactReader(), compactor=compactor),
            span_scorer=ScriptedSpanScorer(default=COMPACTED_SCORE),  # type: ignore[arg-type]
            event_bus=bus,
            gc_config=GCConfig(context_window=1000, min_cut_tokens=0, last_n_turns_to_keep=0,
                               archive_threshold=0.0, **PATIENT),
        )

        async def scenario():
            compactor.gate.set()
            _record(service, 0, 1, 2, 3)
            await service.gc_age(SESSION)
            first = service.session_manager.store.get(SESSION).snapshot.staged

            _record(service, 4, 5, 6, 7)
            compactor.gate.clear()
            compactor.started.clear()
            running = asyncio.create_task(service.gc_age(SESSION))
            await asyncio.wait_for(compactor.started.wait(), timeout=1)
            switched = service.gc_switch(SESSION, CacheState.COLD)
            compactor.gate.set()
            return first, switched, await asyncio.wait_for(running, timeout=1)

        first, switched, prepared = asyncio.run(scenario())
        snapshot = service.session_manager.store.get(SESSION).snapshot

        assert switched is not None and set(switched.outcome.compacted) == first.delta.departed_turns
        assert prepared.status is GCStatus.STAGED
        assert snapshot.staged is not None and snapshot.staged.delta.departed_turns == {4, 5, 6, 7}

        assert service.gc_switch(SESSION, CacheState.COLD) is not None
        held = [turn for span in service.session_manager.store.get(SESSION).snapshot.old for turn in span.turn_indices]
        assert sorted(held) == list(range(8)), "every turn in exactly one span"


class TestTheInspectionShowsTheSwitchsVerdict:
    def test_the_verdict_shown_is_the_one_the_switch_reaches(self):
        service, _ = _service(**PATIENT)
        _record(service, 0, 1, 2, 3)
        _prepare(service)

        warm = service.inspect(SESSION, cache=CacheState.WARM).staged
        cold = service.inspect(SESSION, cache=CacheState.COLD).staged

        assert warm is not None and (warm.decision.switch, warm.decision.reason) == (
            False, SwitchReason.WAITING_FOR_PAYBACK)
        assert cold is not None and (cold.decision.switch, cold.decision.reason) == (True, SwitchReason.CACHE_COLD)
        assert service.gc_switch(SESSION, CacheState.WARM) is None
        assert service.gc_switch(SESSION, CacheState.COLD).switch_reason is cold.decision.reason
        assert service.inspect(SESSION).staged is None, "nothing waits once it is switched in"

    def test_the_stale_bound_is_its_share_of_the_window_counted_as_the_stale_weight_is(self):
        """Stale weight is LLM-GC's count; at 2 of the model's tokens per LLM-GC token, the bound halves."""
        service, _ = _service(context_window=10_000, max_stale_ratio=0.1, min_saving_share=1000.0)
        _record(service, 0, 1, 2, 3)
        _prepare(service)
        assert service.inspect(SESSION, cache=CacheState.WARM).staged.stale_bound_tokens == 1_000

        service.use_token_rate(SESSION, 2.0)

        assert service.inspect(SESSION, cache=CacheState.WARM).staged.stale_bound_tokens == 500

    def test_it_shows_the_measures_the_policy_weighs(self):
        service, _ = _service(**PATIENT)
        state = _record(service, 0, 1, 2, 3)
        _prepare(service)
        staged = state.snapshot.staged

        preview = service.inspect(SESSION, cache=CacheState.WARM).staged

        tail = [turn for turn in state.snapshot.young if turn.turn_index not in staged.delta.departed_turns]
        assert (preview.saving_tokens, preview.stale_tokens) == (staged.saving_tokens, staged.stale_tokens)
        assert preview.rewritten_tokens == staged.front_tokens + sum(turn.token_count for turn in tail)
        assert preview.stale_bound_tokens == service.gc_config.stale_bound(state.context_window)
        assert preview.min_saving_share == 1000.0

