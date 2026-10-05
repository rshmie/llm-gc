import asyncio

from hypothesis import given, settings
from hypothesis import strategies as st

from llm_gc.config import GCConfig
from llm_gc.engine import DegradationReason
from llm_gc.engine.compaction import BaseCompactor, NoOpCompactor
from llm_gc.engine.compaction.compaction_strategy import CompactionStrategy
from llm_gc.engine.gc_result import CutOutcome
from llm_gc.engine.gc_service import GcService
from llm_gc.engine.gc_status import GCStatus
from llm_gc.engine.generations.front import span_message
from llm_gc.engine.generations.generational_memory import GenerationalMemory
from llm_gc.engine.sweep import SweepClassification, SweepEntry, SweepResult
from llm_gc.events import Event, EventBus, EventType
from llm_gc.extraction import KnowledgeExtractor
from llm_gc.models import Message
from llm_gc.scoring.relevance_scorer_result import RelevanceScorerResult
from llm_gc.session import SessionManager
from llm_gc.utils import count_tokens


class _StubScorer:
    """Always scores 1.0, meaning "no opinion, keep it".

    The stub sweeper ignores the score when classifying young turns, but the
    old-generation pass reads it directly, to choose which entries make room when
    the old generation is over its budget.
    """

    def score(self, message, conversation):
        return RelevanceScorerResult(message=message, combined_score=1.0, scorer_results=[])


class _ScoreByTurn:
    """Scores each message by a caller-supplied {turn_index: score} map, defaulting to 1.0.

    Only the front sweep reads scores (the stub sweeper ignores them), so this
    decides exactly which front entries an over-budget pass archives first.
    """

    def __init__(self, scores: dict[int, float] | None = None) -> None:
        self.scores = scores or {}

    def score(self, message, conversation):
        return RelevanceScorerResult(message=message, combined_score=self.scores.get(message.turn_index, 1.0),
                                     scorer_results=[])


class _StubSweeper:
    """Classifies turns by a caller-supplied {turn_index: classification} map,
    defaulting to KEEP. Lets a test place a specific turn in COMPACT or ARCHIVE
    without depending on the real scorers."""

    def __init__(self) -> None:
        self.classifications: dict[int, SweepClassification] = {}

    def sweep(self, conversation, relevance_scores):
        entries = [
            SweepEntry(
                message=m,
                classification=self.classifications.get(m.turn_index, SweepClassification.KEEP),
                relevance_score=0.5,
                turn_index=m.turn_index,
            )
            for m in conversation
        ]
        counts = {c: sum(1 for e in entries if e.classification == c) for c in SweepClassification}
        return SweepResult(sweep_entries=entries, classification_counts=counts, processing_time_ms=0.0,
                           total_keep_tokens=0, total_compact_tokens=0, total_archive_tokens=0)


class _RaisingScorer:
    """A scorer that always fails - drives gc_update's MARK-stage failure path,
    where nothing has been aged yet and the whole pass must be reported bypassed."""

    def score(self, message, conversation):
        raise RuntimeError("scoring failed")


class _RaisingCompactor(BaseCompactor):
    """A compactor that always fails - stands in for a Phase-7 LLMCompactor whose
    API call errors, so a test can drive gc_update's failure path."""

    COMPACTION_STRATEGY = CompactionStrategy.NOOP

    async def compact(self, messages):
        raise RuntimeError("compaction failed")


def _old_gen(service: GcService, session_id: str = "s1") -> list[Message]:
    """What the front carries for the session's old generation: its kept and compacted spans."""
    state = service.session_manager.store.get(session_id)
    if state is None:
        return []
    return [message for span in state.snapshot.old if (message := span_message(span)) is not None]


def _facts(service: GcService, session_id: str = "s1") -> list:
    state = service.session_manager.store.get(session_id)
    return state.snapshot.active_knowledge if state is not None else []


def _msg(turn_index: int) -> Message:
    # Sized like a real turn: a cut has to remove more than the front's own
    # headers cost, or it would grow the context and be refused.
    return Message(role="user", content=f"turn {turn_index}", token_count=100, turn_index=turn_index)


# Any compactable turn is worth a cut, and only the newest turn is kept verbatim.
# The tests below are about *what* a cut does, not *when*; the trigger has its own
# class.
_ALWAYS_AGE = GCConfig(context_window=1000, min_cut_tokens=0, last_n_turns_to_keep=1)

# Every quote of a `_msg` turn is 15 tokens, so on a 1000-token window each 0.015
# of `max_old_gen_ratio` is room for one more.
_QUOTE_TOKENS = 15


def _old_gen_budget(ratio: float) -> GCConfig:
    """`_ALWAYS_AGE` with an old-generation budget, built anew so it is validated."""
    return GCConfig(context_window=1000, min_cut_tokens=0, last_n_turns_to_keep=1, max_old_gen_ratio=ratio)


def _make_service(compactor: BaseCompactor | None = None, scorer=None,
                  gc_config: GCConfig | None = None) -> tuple[GcService, _StubSweeper]:
    bus = EventBus()
    generational_memory = GenerationalMemory(
        knowledge_extractor=KnowledgeExtractor(event_bus=bus),
        compactor=compactor if compactor is not None else NoOpCompactor(event_bus=bus),
    )
    sweeper = _StubSweeper()
    service = GcService(session_manager=SessionManager(), generational_memory=generational_memory,
                        relevance_scorer=scorer if scorer is not None else _StubScorer(),
                        sweeper=sweeper, event_bus=bus,
                        gc_config=gc_config if gc_config is not None else _ALWAYS_AGE)
    return service, sweeper


def _capture_gc_finished(service: GcService) -> list[Event]:
    """Collect every GC_FINISHED the service emits, in order."""
    captured: list[Event] = []
    service.event_bus.subscribe(EventType.GC_FINISHED, captured.append)
    return captured


class TestGcUpdateActsOnClassification:
    def test_compact_turn_is_summarised_and_leaves_young(self):
        service, sweeper = _make_service()

        async def scenario():
            await service.gc_update("s1", _msg(1))              # turn 1 KEEP (default)
            sweeper.classifications = {1: SweepClassification.COMPACT}
            await service.gc_update("s1", _msg(2))              # turn 1 cools to COMPACT
            state = service.session_manager.store.get("s1")
            return [m.turn_index for m in list(state.snapshot.young)], len(_old_gen(service))

        young, old_gen_count = asyncio.run(scenario())
        assert young == [2]          # turn 1 left the desk; turn 2 stays
        assert old_gen_count == 1    # turn 1's summary is filed in old gen

    def test_archive_turn_is_extracted_and_leaves_young(self):
        service, sweeper = _make_service()

        async def scenario():
            await service.gc_update("s1", _msg(1))
            sweeper.classifications = {1: SweepClassification.ARCHIVE}
            await service.gc_update("s1", _msg(2))
            state = service.session_manager.store.get("s1")
            return [m.turn_index for m in list(state.snapshot.young)], len(_facts(service))

        young, permanent_count = asyncio.run(scenario())
        assert young == [2]              # turn 1 left the desk
        assert permanent_count >= 1      # its facts (or a raw fallback) went to permanent gen

    def test_keep_turns_all_stay_young(self):
        service, _ = _make_service()

        async def scenario():
            await service.gc_update("s1", _msg(1))
            await service.gc_update("s1", _msg(2))
            state = service.session_manager.store.get("s1")
            return [m.turn_index for m in list(state.snapshot.young)]

        # Nothing classified COMPACT/ARCHIVE, so nothing departs.
        assert asyncio.run(scenario()) == [1, 2]

    def test_old_gen_stays_empty_while_everything_is_kept(self):
        service, _ = _make_service()

        async def scenario():
            await service.gc_update("s1", _msg(1))
            return _old_gen(service)

        assert asyncio.run(scenario()) == []

    def test_a_failed_summary_leaves_the_turn_in_the_tail_and_does_not_propagate(self):
        """The cut retreats to just before the failure: the turn stays verbatim, and
        the next pass meets it in a longer run."""
        service, sweeper = _make_service(compactor=_RaisingCompactor(event_bus=EventBus()))
        captured = _capture_gc_finished(service)

        async def scenario():
            await service.gc_update("s1", _msg(1))              # turn 1 KEEP
            sweeper.classifications = {1: SweepClassification.COMPACT}
            await service.gc_update("s1", _msg(2))              # turn 1 COMPACT -> summary raises
            state = service.session_manager.store.get("s1")
            return [m.turn_index for m in list(state.snapshot.young)], _old_gen(service)

        young, old_gen = asyncio.run(scenario())
        assert young == [1, 2]
        assert old_gen == []
        assert captured[-1].data["gc_result"].degradations, "the failure is reported, not hidden"

    def test_a_kept_turn_inside_a_cut_is_quoted_word_for_word(self):
        service, sweeper = _make_service()

        async def scenario():
            for index in (1, 2, 3):
                await service.gc_update("s1", _msg(index))
            sweeper.classifications = {2: SweepClassification.COMPACT}   # turn 1 stays KEEP
            await service.gc_update("s1", _msg(4))
            return _old_gen(service)

        old_gen = asyncio.run(scenario())
        quoted = [m for m in old_gen if "[Quoted exactly]" in m.content]
        assert [m.turn_index for m in quoted] == [1, 3]


class TestGcCollectAssembly:
    def test_the_front_comes_first_then_the_verbatim_tail(self):
        """Every change to the prompt lives in the front; the tail is untouched."""
        service, sweeper = _make_service()

        async def scenario():
            await service.gc_update("s1", _msg(1))
            sweeper.classifications = {1: SweepClassification.COMPACT}
            await service.gc_update("s1", _msg(2))  # turn 1 -> summary in the front
            return await service.gc_collect("s1")

        assembled = asyncio.run(scenario())
        assert [m.turn_index for m in assembled] == [-1, 2]
        assert assembled[0].role == "user", "the provider requires the conversation to open with a user turn"
        assert "Compacted" in assembled[0].content
        assert assembled[1].content == "turn 2"

    def test_collect_on_empty_session_returns_empty(self):
        service, _ = _make_service()
        assert asyncio.run(service.gc_collect("s1")) == []


class TestGcUpdateReportsOnTheBus:
    """gc_update is the pass that does the real GC work, so it is the pass the
    health monitor has to be able to see. Before this, the session path emitted
    no GC_FINISHED at all and the dashboard stayed blank on a real conversation."""

    def test_update_emits_gc_finished_carrying_the_session_id(self):
        service, _ = _make_service()
        captured = _capture_gc_finished(service)

        asyncio.run(service.gc_update("s1", _msg(1)))

        assert len(captured) == 1
        assert captured[0].data["session_id"] == "s1"
        assert captured[0].data["gc_result"].status == GCStatus.BYPASSED_BATCH_TOO_SMALL

    def test_collect_emits_nothing(self):
        service, _ = _make_service()
        captured = _capture_gc_finished(service)

        asyncio.run(service.gc_collect("s1"))

        # collect only reads. A read reporting itself as a "GC run" would
        # overwrite the last real run's breakdown with a no-op.
        assert captured == []

    def test_final_messages_equal_what_collect_would_return(self):
        """The invariant that keeps the dashboard honest: what update *reports*
        as the resulting context is exactly what collect *sends*."""
        service, sweeper = _make_service()
        captured = _capture_gc_finished(service)

        async def scenario():
            await service.gc_update("s1", _msg(1))
            sweeper.classifications = {1: SweepClassification.COMPACT}
            await service.gc_update("s1", _msg(2))   # turn 1 -> old-gen summary
            return await service.gc_collect("s1")

        assembled = asyncio.run(scenario())
        reported = captured[-1].data["gc_result"].final_messages
        assert [m.turn_index for m in reported] == [m.turn_index for m in assembled]
        assert [m.content for m in reported] == [m.content for m in assembled]

    def test_tokens_in_final_counts_the_front_not_just_young(self):
        """A budget meter that omitted the front would under-report the real prompt
        size - in the reassuring direction, which is the one kind of wrong this
        project refuses to ship."""
        service, sweeper = _make_service()
        captured = _capture_gc_finished(service)

        async def scenario():
            await service.gc_update("s1", _msg(1))
            sweeper.classifications = {1: SweepClassification.COMPACT}
            await service.gc_update("s1", _msg(2))
            state = service.session_manager.store.get("s1")
            return list(state.snapshot.young)

        young = asyncio.run(scenario())
        gc_result = captured[-1].data["gc_result"]
        young_tokens = sum(m.token_count for m in young)
        front = asyncio.run(service.gc_collect("s1"))[0]
        assert front.turn_index == -1 and front.token_count > 0      # there is something to omit
        assert gc_result.tokens_in_final == young_tokens + front.token_count

    def test_mark_failure_is_reported_bypassed_and_never_escapes(self):
        """gc_update is fire-and-forget: an exception that escapes it dies as an
        unretrieved task exception, invisible to logs and dashboard alike."""
        service, _ = _make_service(scorer=_RaisingScorer())
        captured = _capture_gc_finished(service)

        asyncio.run(service.gc_update("s1", _msg(1)))                # must not raise

        gc_result = captured[0].data["gc_result"]
        assert gc_result.status == GCStatus.BYPASSED_ON_ERROR
        assert gc_result.failure_stage == "score"
        assert "scoring failed" in gc_result.failure_reason
        # First do no harm: the new turn is still there, nothing was aged.
        assert [m.turn_index for m in gc_result.final_messages] == [1]
        assert _old_gen(service) == []


class TestCutsAreBatched:
    """Classification decides what a cut compacts; the batch size decides when.
    Every cut resets the provider's prompt cache once, so cutting for a handful of
    tokens costs more than it saves."""

    def test_a_batch_below_the_minimum_is_not_cut(self):
        service, sweeper = _make_service(gc_config=GCConfig(min_cut_tokens=250, last_n_turns_to_keep=1))
        captured = _capture_gc_finished(service)

        async def scenario():
            await service.gc_update("s1", _msg(1))
            sweeper.classifications = {1: SweepClassification.COMPACT}
            await service.gc_update("s1", _msg(2))     # 100 compactable tokens: under 250
            return [m.turn_index for m in service.session_manager.store.get("s1").snapshot.young]

        assert asyncio.run(scenario()) == [1, 2]
        assert captured[-1].data["gc_result"].status == GCStatus.BYPASSED_BATCH_TOO_SMALL

    def test_a_batch_that_reaches_the_minimum_is_cut(self):
        service, sweeper = _make_service(gc_config=GCConfig(min_cut_tokens=250, last_n_turns_to_keep=1))
        captured = _capture_gc_finished(service)
        sweeper.classifications = {1: SweepClassification.COMPACT, 2: SweepClassification.COMPACT,
                                   3: SweepClassification.COMPACT}

        async def scenario():
            for index in (1, 2, 3, 4):                 # turns 1-3: 300 compactable tokens
                await service.gc_update("s1", _msg(index))
            return [m.turn_index for m in service.session_manager.store.get("s1").snapshot.young]

        assert asyncio.run(scenario()) == [4]
        assert captured[-1].data["gc_result"].status == GCStatus.COMPLETED

    def test_a_batch_of_only_kept_turns_is_never_cut(self):
        """Quoting turns verbatim compacts nothing; cutting for it would reset the
        cache for no saving at all."""
        service, _ = _make_service()
        captured = _capture_gc_finished(service)

        async def scenario():
            for index in (1, 2, 3):
                await service.gc_update("s1", _msg(index))

        asyncio.run(scenario())
        assert _old_gen(service) == []
        assert all(e.data["gc_result"].status == GCStatus.BYPASSED_BATCH_TOO_SMALL for e in captured)

    def test_a_bypassed_update_still_records_the_new_turn(self):
        """Bypassing aging must not bypass bookkeeping - the turn is part of the
        conversation whether or not there is pressure to act on it."""
        service, _ = _make_service(gc_config=GCConfig(min_cut_tokens=1000))
        captured = _capture_gc_finished(service)

        asyncio.run(service.gc_update("s1", _msg(1)))

        assert [m.turn_index for m in service.session_manager.store.get("s1").snapshot.young] == [1]
        assert [m.turn_index for m in captured[0].data["gc_result"].final_messages] == [1]


class TestTheManualCut:
    """The dashboard's "Run GC now": the same pass with only the batch minimum waived."""

    _HIGH_MINIMUM = GCConfig(context_window=1000, min_cut_tokens=10_000, last_n_turns_to_keep=1)

    def test_it_cuts_a_batch_below_the_minimum(self):
        service, sweeper = _make_service(gc_config=self._HIGH_MINIMUM)

        async def scenario():
            for index in (1, 2, 3):
                await service.gc_update("s1", _msg(index))
            sweeper.classifications = {1: SweepClassification.COMPACT, 2: SweepClassification.COMPACT}
            automatic = await service.gc_age("s1")
            manual = await service.gc_age("s1", ignore_batch_minimum=True)
            return automatic, manual

        automatic, manual = asyncio.run(scenario())
        assert automatic.status is GCStatus.BYPASSED_BATCH_TOO_SMALL
        assert manual.status is GCStatus.COMPLETED
        assert manual.tokens_saved > 0

    def test_every_other_rule_still_holds(self):
        """The newest turns stay verbatim and a batch of only kept turns is not cut:
        quoting them would reset the provider's cache and compact nothing."""
        service, sweeper = _make_service(gc_config=self._HIGH_MINIMUM)

        async def scenario():
            for index in (1, 2, 3):
                await service.gc_update("s1", _msg(index))
            all_kept = await service.gc_age("s1", ignore_batch_minimum=True)
            sweeper.classifications = {i: SweepClassification.COMPACT for i in (1, 2, 3)}
            await service.gc_age("s1", ignore_batch_minimum=True)
            return all_kept

        all_kept = asyncio.run(scenario())
        assert all_kept.status is GCStatus.BYPASSED_BATCH_TOO_SMALL
        assert [m.turn_index for m in service.session_manager.store.get("s1").snapshot.young] == [3]

    def test_a_cut_that_would_not_shrink_the_context_says_so(self):
        """It used to be reported as "batch below the minimum" - false whenever the
        batch was large enough, and visibly false once a person presses the button."""
        service, sweeper = _make_service(gc_config=self._HIGH_MINIMUM)

        def honest(turn_index: int) -> Message:
            text = f"turn {turn_index}"
            return Message(role="user", content=text, token_count=count_tokens(text), turn_index=turn_index)

        async def scenario():
            for index in (1, 2, 3):
                await service.gc_update("s1", honest(index))
            sweeper.classifications = {1: SweepClassification.COMPACT, 2: SweepClassification.COMPACT}
            return await service.gc_age("s1", ignore_batch_minimum=True)

        # The no-op compactor wraps turns in a marker, so its "summary" is longer than they were.
        result = asyncio.run(scenario())
        assert result.status is GCStatus.BYPASSED_NO_SAVING
        assert len(service.session_manager.store.get("s1").snapshot.young) == 3

        # What it would have done, so a reader can see by how much it fell short.
        abandoned = result.abandoned_cut
        assert abandoned is not None
        assert abandoned.turns == [1, 2]
        assert abandoned.outcome == CutOutcome(quoted=0, summarised=2, archived=0)
        assert abandoned.context_tokens_after >= result.tokens_before
        assert result.tokens_in_final == result.tokens_before, "and nothing changed"


class TestTheCutOutcome:
    """Where a cut's turns ended up, which the sweeper's decision counts cannot say.

    The same pass sweeps the front after the cut, so it can archive an entry the cut
    has just made. A dashboard that showed the decision reported "1 summarised, 6
    reduced to facts" for a clean-up that had reduced 12 messages to facts.
    """

    def test_a_quote_archived_in_the_pass_that_made_it_is_reported_as_archived(self):
        service, sweeper = _make_service(gc_config=_old_gen_budget(0.0))

        async def scenario():
            for index in (1, 2, 3):
                await service.gc_update("s1", _msg(index))
            sweeper.classifications = {2: SweepClassification.ARCHIVE}   # 1 is quoted, 3 is the tail
            return await service.gc_age("s1")

        result = asyncio.run(scenario())
        assert result.status is GCStatus.COMPLETED
        assert result.kept_count == 2, "the decision: quote turn 1, keep turn 3"
        assert result.outcome == CutOutcome(quoted=0, summarised=0, archived=2)
        assert _old_gen(service) == [], "the quote went straight to facts"

    def test_an_earlier_entry_pushed_out_is_reported_apart_from_the_cut(self):
        service, sweeper = _make_service(gc_config=_old_gen_budget(0.02))   # room for one quote

        async def scenario():
            for index in (1, 2, 3):
                await service.gc_update("s1", _msg(index))
            sweeper.classifications = {2: SweepClassification.ARCHIVE}
            first = await service.gc_age("s1")
            for index in (4, 5):
                await service.gc_update("s1", _msg(index))
            # Turn 3 is quoted beside turn 1's quote, with room for only one.
            sweeper.classifications = {4: SweepClassification.ARCHIVE}
            return first, await service.gc_age("s1")

        first, second = asyncio.run(scenario())
        assert first.outcome == CutOutcome(quoted=1, summarised=0, archived=1)
        assert second.outcome == CutOutcome(quoted=1, summarised=0, archived=1, front_archived=1)

    def test_a_pass_that_cuts_nothing_has_no_outcome(self):
        service, _ = _make_service()

        async def scenario():
            for index in (1, 2):
                await service.gc_update("s1", _msg(index))
            return await service.gc_age("s1")

        result = asyncio.run(scenario())
        assert result.status is GCStatus.BYPASSED_BATCH_TOO_SMALL
        assert result.outcome is None


class TestTheOldGenBudget:
    """The budget decides when an old-generation entry is archived; the score decides which.

    Archived, an entry reaches the model only as facts and lines cut to a fixed
    length, so archiving one the front had room for loses detail and saves nothing
    that was needed. The decision-survival check lost seven answers that way, each
    in the tail of a summary archived only because its score had decayed.
    """

    @staticmethod
    def _cut_into_quotes(service: GcService, sweeper: _StubSweeper, quotes: int):
        """Record turns 1..`quotes` plus two more, then cut once: the first `quotes`
        turns are quoted, the next is archived so the cut compacts something, and the
        last stays in the tail."""
        async def scenario():
            for index in range(1, quotes + 3):
                await service.gc_update("s1", _msg(index))
            sweeper.classifications = {quotes + 1: SweepClassification.ARCHIVE}
            return await service.gc_age("s1")

        return asyncio.run(scenario())

    def test_a_cold_entry_stays_while_the_front_has_room(self):
        """The fix itself. Scored as cold as possible, and kept, because nothing needs the room."""
        service, sweeper = _make_service(scorer=_ScoreByTurn({1: 0.0, 2: 0.0}), gc_config=_old_gen_budget(0.03))

        self._cut_into_quotes(service, sweeper, quotes=2)

        assert [m.turn_index for m in _old_gen(service)] == [1, 2]

    def test_over_budget_the_coldest_entry_goes_not_the_oldest(self):
        service, sweeper = _make_service(scorer=_ScoreByTurn({1: 0.9, 2: 0.1, 3: 0.5}),
                                         gc_config=_old_gen_budget(0.03))   # room for two quotes

        self._cut_into_quotes(service, sweeper, quotes=3)

        assert [m.turn_index for m in _old_gen(service)] == [1, 3]

    def test_it_archives_only_until_the_old_generation_fits(self):
        """Three quotes against room for exactly two: one goes, and a front exactly at
        its budget is within it. With equal scores, the older entry goes."""
        service, sweeper = _make_service(gc_config=_old_gen_budget(0.03))

        self._cut_into_quotes(service, sweeper, quotes=3)

        assert [m.turn_index for m in _old_gen(service)] == [2, 3]
        assert sum(m.token_count for m in _old_gen(service)) == 2 * _QUOTE_TOKENS == service.gc_config.max_old_gen_tokens

    def test_an_entry_that_fails_to_archive_stays_and_the_next_coldest_makes_room(self, monkeypatch):
        """Archive first, remove second: a failed archive keeps the entry, and the
        budget is still met by the next-coldest one rather than left exceeded."""
        service, sweeper = _make_service(scorer=_ScoreByTurn({1: 0.0, 2: 0.5}),
                                         gc_config=_old_gen_budget(0.015))   # room for one quote
        extract = service.generational_memory.extract_knowledge

        def failing_on_turn_one(message):
            if message.turn_index == 1:
                raise RuntimeError("extraction unavailable")
            return extract(message)

        monkeypatch.setattr(service.generational_memory, "extract_knowledge", failing_on_turn_one)

        result = self._cut_into_quotes(service, sweeper, quotes=2)

        assert [m.turn_index for m in _old_gen(service)] == [1]
        assert [(d.reason, d.turn_indices) for d in result.degradations] == [(DegradationReason.ARCHIVE_FAILED, [1])]


class TestNothingRecordedIsDestroyed:
    """A turn leaves the young generation only into a span holding it as recorded,
    and a span is never removed - whatever the passes decide, and whatever fails."""

    @settings(max_examples=60, deadline=None)
    @given(classifications=st.lists(st.sampled_from(list(SweepClassification)), min_size=1, max_size=24),
           ratio=st.sampled_from([0.0, 0.015, 0.2]),
           unreadable=st.sets(st.integers(0, 23), max_size=4))
    def test_every_turn_is_held_exactly_once(self, classifications, ratio, unreadable):
        service, sweeper = _make_service(gc_config=_old_gen_budget(ratio))
        extract = service.generational_memory.extract_knowledge

        def extract_or_fail(message):
            if message.turn_index in unreadable:
                raise RuntimeError("extraction unavailable")
            return extract(message)

        service.generational_memory.extract_knowledge = extract_or_fail
        finished = _capture_gc_finished(service)

        async def scenario():
            for index, classification in enumerate(classifications):
                sweeper.classifications[index] = classification
                await service.gc_update("s1", _msg(index))

        asyncio.run(scenario())
        snapshot = service.session_manager.store.get("s1").snapshot

        held = list(snapshot.young) + [turn for span in snapshot.old for turn in span.turns]
        assert sorted(turn.turn_index for turn in held) == list(range(len(classifications)))
        assert all(turn == _msg(turn.turn_index) for turn in held)
        # A plan that would lose a turn is refused at commit and the pass reports an
        # error: safe, but a collection that never happens. No pass may end that way.
        assert not [event for event in finished if event.data["gc_result"].status is GCStatus.BYPASSED_ON_ERROR]
