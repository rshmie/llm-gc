"""A collection: condense what aged out of the tail, score every span, give each its level."""

import asyncio

from hypothesis import given, settings
from hypothesis import strategies as st

from llm_gc.config import GCConfig
from llm_gc.engine import DegradationReason
from llm_gc.engine.compaction import BaseCompactor
from llm_gc.engine.compaction.compaction_strategy import CompactionStrategy
from llm_gc.engine.gc_result import CutOutcome
from llm_gc.engine.gc_status import GCStatus
from llm_gc.events import EventType
from llm_gc.models.generation_snapshot import DetailLevel
from llm_gc.utils import count_tokens
from tests.engine.support import ARCHIVED_SCORE, COMPACTED_SCORE, KEPT_SCORE, SUMMARY, make_service, turn

KEPT, COMPACTED, ARCHIVED = DetailLevel.KEPT, DetailLevel.COMPACTED, DetailLevel.ARCHIVED


class _FailingCompactor(BaseCompactor):
    """Stands in for a summariser whose provider is down."""

    COMPACTION_STRATEGY = CompactionStrategy.LLM

    async def compact(self, messages):
        raise RuntimeError("provider unavailable")


class _RaisingScorer:
    def score(self, spans, now, newest_turn):
        raise RuntimeError("scoring failed")


def _record(service, message) -> None:
    state = service.session_manager.get_or_create("s")
    state.snapshot = state.snapshot.with_turn(message)


def _converse(service, turns: int, start: int = 0) -> list:
    """Record turns one by one, a pass after each; every GCResult, in order."""
    results = []

    async def scenario():
        for index in range(start, start + turns):
            _record(service, turn(index))
            results.append(await service.gc_age("s"))

    asyncio.run(scenario())
    return results


def _snapshot(service):
    return service.session_manager.store.get("s").snapshot


def _levels(service) -> dict[int, DetailLevel]:
    return {span.first_turn: span.level for span in _snapshot(service).old}


def _young(service) -> list[int]:
    return [message.turn_index for message in _snapshot(service).young]


class TestEachSpanTakesTheLevelItsScoreEarns:
    def test_turns_older_than_the_tail_become_spans_at_their_levels(self):
        service, _, _ = make_service(scores={0: KEPT_SCORE, 1: COMPACTED_SCORE, 2: ARCHIVED_SCORE})

        _converse(service, 4)

        assert _levels(service) == {0: KEPT, 1: COMPACTED, 2: ARCHIVED}
        assert _young(service) == [3]

    def test_the_front_shows_each_span_at_its_level_in_order_then_the_tail(self):
        service, _, _ = make_service(scores={0: KEPT_SCORE, 1: COMPACTED_SCORE, 2: ARCHIVED_SCORE})
        _converse(service, 4)

        prompt = asyncio.run(service.gc_collect("s"))
        lines = prompt[0].content.split("\n")

        assert [message.turn_index for message in prompt] == [-1, 3]
        assert prompt[0].role == "user", "the provider requires the conversation to open with a user turn"
        assert lines[-4].startswith("[Quoted exactly] (turn 0, user) Turn 0 is about storage.")
        assert lines[-3].startswith("[Summarised] (turn 1) Turn 1 is about storage."), "its opening line, then"
        assert lines[-2] == SUMMARY.text, "its summary"
        assert lines[-1].startswith("[Archived] (turn 2) Turn 2 is about storage.")

    def test_an_archived_span_carries_the_facts_read_from_its_turns(self):
        service, _, _ = make_service(scores={0: ARCHIVED_SCORE})

        async def scenario():
            for index, opening in enumerate(["The database is postgres.", None]):
                _record(service, turn(index, opening=opening))
                await service.gc_age("s")

        asyncio.run(scenario())
        front = asyncio.run(service.gc_collect("s"))[0].content
        assert "[Archived] (turn 0) The database is postgres. · database: postgres" in front

    def test_a_span_whose_archived_line_costs_more_than_its_words_stays_kept(self):
        """A level is used only if it is cheaper than the richer one: a short statement
        plus the fact read from it is longer than the statement quoted."""
        service, _, _ = make_service(scores={0: ARCHIVED_SCORE, 1: ARCHIVED_SCORE})
        stated = "The database is postgres."

        async def scenario():
            for index in range(3):
                message = turn(index)
                if index == 0:
                    message = message.model_copy(update={"content": stated, "token_count": count_tokens(stated)})
                _record(service, message)
                await service.gc_age("s")

        asyncio.run(scenario())
        assert _levels(service) == {0: KEPT, 1: ARCHIVED}


class TestNowIsWhatTheUserIsDiscussing:
    def test_spans_are_scored_against_the_users_turns_in_the_tail(self):
        """Not the assistant's replies, which restate and expand and would make every
        span look relevant; not tool results, which are output rather than a subject."""
        service, scorer, _ = make_service(GCConfig(min_cut_tokens=0, last_n_turns_to_keep=3),
                                          default_score=COMPACTED_SCORE)
        heard: list[list[int]] = []
        score = scorer.score

        def recording(spans, now, newest_turn):
            heard.append([message.turn_index for message in now])
            return score(spans, now, newest_turn)

        scorer.score = recording  # type: ignore[method-assign]

        async def scenario():
            for index in range(5):
                message = turn(index, role="user" if index % 2 == 0 else "assistant")
                if index == 4:
                    message = message.model_copy(update={"raw_content": [
                        {"type": "tool_result", "tool_use_id": "t", "content": "ok"}]})
                _record(service, message)
                await service.gc_age("s")

        asyncio.run(scenario())
        assert heard[-1] == [2], "the tail is turns 2-4: one user turn, a reply, and a tool result"


class TestTheBatchDecidesWhen:
    """The plan decides how each span is shown; the batch size decides when. Every
    collection resets the provider's cache once, so collecting for a few tokens costs
    more than it saves."""

    def test_below_the_minimum_nothing_is_collected_or_paid_for(self):
        """The plan runs on estimated summaries first: no model call until the batch is worth one."""
        service, _, provider = make_service(GCConfig(min_cut_tokens=10_000, last_n_turns_to_keep=1),
                                            default_score=COMPACTED_SCORE)

        results = _converse(service, 3)

        assert {result.status for result in results} == {GCStatus.BYPASSED_BATCH_TOO_SMALL}
        assert _young(service) == [0, 1, 2]
        assert provider.calls == []

    def test_a_batch_the_plan_would_keep_whole_is_never_collected(self):
        """Moving kept turns into the front saves nothing and resets the cache."""
        service, _, provider = make_service(default_score=KEPT_SCORE)

        results = _converse(service, 4)

        assert {result.status for result in results} == {GCStatus.BYPASSED_BATCH_TOO_SMALL}
        assert _snapshot(service).old == ()
        assert provider.calls == []

    def test_reaching_the_minimum_collects_the_whole_batch(self):
        tokens = turn(0).token_count
        service, _, _ = make_service(GCConfig(min_cut_tokens=2 * tokens, last_n_turns_to_keep=1),
                                     default_score=COMPACTED_SCORE)

        results = _converse(service, 3)

        assert [result.status for result in results] == [GCStatus.BYPASSED_BATCH_TOO_SMALL] * 2 + [GCStatus.COMPLETED]
        assert _levels(service) == {0: COMPACTED, 1: COMPACTED}

    def test_the_manual_lever_waives_only_the_minimum(self):
        service, _, _ = make_service(GCConfig(min_cut_tokens=10_000, last_n_turns_to_keep=1),
                                     default_score=COMPACTED_SCORE)
        _converse(service, 3)

        manual = asyncio.run(service.gc_age("s", ignore_batch_minimum=True))

        assert manual.status is GCStatus.COMPLETED and manual.tokens_saved > 0
        assert _young(service) == [2], "the newest turns stay word for word"

    def test_a_cut_that_cannot_pay_for_itself_buys_no_summaries(self):
        """Decided on the estimate: a session's first cut also pays for the front's
        header, and one short span's summary saves less than that. Bought and then
        discarded, the summaries would be paid for again on every pass."""
        service, _, provider = make_service(default_score=COMPACTED_SCORE)
        results = []

        async def scenario():
            for index in range(2):
                _record(service, turn(index, length=2))
                results.append(await service.gc_age("s"))

        asyncio.run(scenario())
        result = results[-1]
        assert result.status is GCStatus.BYPASSED_NO_SAVING and provider.calls == []
        assert result.abandoned_cut is not None and result.abandoned_cut.turns == [0]
        assert result.abandoned_cut.context_tokens_after >= result.tokens_before, "and it says by how much"
        assert _young(service) == [0, 1]

    def test_a_cut_whose_summaries_fail_is_abandoned_after_the_fact(self):
        """The estimate said the cut would pay; without its summary the span is kept,
        quoted, and larger than it was - so the cut does not pay, and says so."""
        service, _, _ = make_service(compactor=_FailingCompactor(event_bus=None), default_score=COMPACTED_SCORE)
        results = []

        async def scenario():
            for index in range(2):
                _record(service, turn(index, length=8))
                results.append(await service.gc_age("s"))

        asyncio.run(scenario())
        result = results[-1]
        assert result.status is GCStatus.BYPASSED_NO_SAVING
        assert result.abandoned_cut is not None
        assert result.abandoned_cut.outcome == CutOutcome(kept=[0])
        assert DegradationReason.COMPACTION_FAILED in {d.reason for d in result.degradations}
        assert _young(service) == [0, 1]


class TestLevelsAreReplannedAtEveryCollection:
    def test_a_span_the_conversation_returns_to_is_shown_in_full_again(self):
        """Restored from its own turns, not from anything derived from them."""
        service, scorer, _ = make_service(default_score=ARCHIVED_SCORE)
        _converse(service, 3)
        assert _levels(service)[0] is ARCHIVED

        scorer.scores[0] = KEPT_SCORE
        results = _converse(service, 1, start=3)

        assert _levels(service)[0] is KEPT
        assert results[-1].outcome.raised == [0]
        assert f"[Quoted exactly] (turn 0, user) {turn(0).content}" in asyncio.run(service.gc_collect("s"))[0].content

    def test_a_score_near_a_line_does_not_move_a_span(self):
        service, scorer, _ = make_service(default_score=COMPACTED_SCORE)
        _converse(service, 3)

        scorer.scores[0] = 0.28   # under the archive line, inside its margin
        _converse(service, 2, start=3)

        assert _levels(service)[0] is COMPACTED

    def test_over_the_budget_the_coldest_span_loses_detail_first(self):
        """Room for two quotes, a summary and an archived line - not three quotes."""
        config = GCConfig(context_window=1000, max_recap_ratio=0.378, min_cut_tokens=0, last_n_turns_to_keep=1)
        service, _, _ = make_service(config, scores={0: 0.8, 1: 0.95, 2: 0.95, 3: ARCHIVED_SCORE})

        _converse(service, 5)

        assert _levels(service) == {0: COMPACTED, 1: KEPT, 2: KEPT, 3: ARCHIVED}


class TestFailuresAreReportedAndLoseNothing:
    def test_a_failed_summary_keeps_the_span_whole_and_says_so(self):
        service, _, _ = make_service(compactor=_FailingCompactor(event_bus=None),
                                     scores={0: COMPACTED_SCORE, 1: ARCHIVED_SCORE})

        result = _converse(service, 3)[-1]

        assert _levels(service) == {0: KEPT, 1: ARCHIVED}, "kept rather than lose detail to a failed summary"
        assert (DegradationReason.COMPACTION_FAILED, [0]) in [(d.reason, d.turn_indices) for d in result.degradations]

    def test_a_failed_extraction_still_makes_the_span(self):
        class _FailingReader:
            async def read(self, spans, on_record):
                raise RuntimeError("extraction unavailable")

        service, _, _ = make_service(fact_reader=_FailingReader(), default_score=ARCHIVED_SCORE)  # type: ignore[arg-type]

        result = _converse(service, 2)[-1]

        assert _levels(service) == {0: ARCHIVED}
        assert [d.reason for d in result.degradations] == [DegradationReason.ARCHIVE_FAILED]
        assert "[Archived] (turn 0) Turn 0 is about storage." in asyncio.run(service.gc_collect("s"))[0].content
        # Kept on the span, so the dashboard can say why it has no facts after the warning has scrolled away.
        assert [span.facts_unread for span in service.session_manager.store.get("s").snapshot.old] == [True]

    def test_a_failing_scorer_changes_nothing_and_never_escapes(self):
        """gc_age runs fire-and-forget: an exception that escaped it would vanish unseen."""
        service, _, _ = make_service()
        service.span_scorer = _RaisingScorer()  # type: ignore[assignment]

        result = _converse(service, 2)[-1]

        assert result.status is GCStatus.BYPASSED_ON_ERROR
        assert result.failure_stage == "plan" and "scoring failed" in result.failure_reason
        assert _young(service) == [0, 1] and _snapshot(service).old == ()


class TestTheBusSeesWhatTheCollectionDid:
    def test_every_pass_emits_one_gc_finished_and_collect_emits_none(self):
        service, _, _ = make_service(default_score=COMPACTED_SCORE)
        finished = []
        service.event_bus.subscribe(EventType.GC_FINISHED, finished.append)

        _converse(service, 3)
        asyncio.run(service.gc_collect("s"))

        assert len(finished) == 3 and all(event.data["session_id"] == "s" for event in finished)

    def test_what_a_pass_reports_is_exactly_what_collect_sends(self):
        service, _, _ = make_service(scores={0: COMPACTED_SCORE, 1: ARCHIVED_SCORE})
        result = _converse(service, 3)[-1]

        sent = asyncio.run(service.gc_collect("s"))
        assert [(m.turn_index, m.content) for m in result.final_messages] == [(m.turn_index, m.content) for m in sent]
        assert result.tokens_in_final == sum(m.token_count for m in sent), "the front's tokens are counted too"

    def test_new_spans_are_announced_by_level(self):
        service, _, _ = make_service(scores={0: COMPACTED_SCORE, 1: ARCHIVED_SCORE})
        promoted, archived = [], []
        service.event_bus.subscribe(EventType.MESSAGE_PROMOTED_TO_OLD_GEN, promoted.append)
        service.event_bus.subscribe(EventType.MESSAGE_ARCHIVED, archived.append)

        _converse(service, 3)

        assert [event.data["source_turn_indices"] for event in promoted] == [[0]]
        assert [(event.data["source_turn_indices"], event.data["source_generation"]) for event in archived] == [
            ([1], "young")]


class TestNothingRecordedIsDestroyed:
    """A turn leaves the young generation only into a span holding it as recorded,
    and a span is never removed - whatever the plan decides."""

    @settings(max_examples=40, deadline=None)
    @given(scores=st.lists(st.sampled_from([KEPT_SCORE, COMPACTED_SCORE, ARCHIVED_SCORE, 0.31, 0.69]),
                           min_size=1, max_size=14),
           ratio=st.sampled_from([0.02, 0.1, 0.3]))
    def test_every_turn_is_held_exactly_once(self, scores, ratio):
        config = GCConfig(context_window=2000, max_recap_ratio=ratio, min_cut_tokens=0, last_n_turns_to_keep=1)
        service, _, _ = make_service(config, scores=dict(enumerate(scores)))
        finished = []
        service.event_bus.subscribe(EventType.GC_FINISHED, finished.append)

        _converse(service, len(scores))

        snapshot = _snapshot(service)
        held = list(snapshot.young) + [message for span in snapshot.old for message in span.turns]
        assert sorted(message.turn_index for message in held) == list(range(len(scores)))
        assert all(message == turn(message.turn_index) for message in held)
        assert not [e for e in finished if e.data["gc_result"].status is GCStatus.BYPASSED_ON_ERROR]


class TestTheRecapIsSizedForTheModelsWindow:
    def test_a_session_told_its_window_sizes_its_recap_for_it(self):
        service, _, _ = make_service(GCConfig(context_window=200_000, max_recap_ratio=0.3))
        _converse(service, 1)

        assert service.inspect("s").recap_budget == 60_000
        service.use_context_window("s", 1_000_000)
        inspection = service.inspect("s")
        assert (inspection.context_window, inspection.recap_budget) == (1_000_000, 300_000)

    def test_the_window_moves_only_the_budget(self):
        """The same conversation as the budget test above: on its small window the coldest quote is
        summarised to fit; on a model with a larger window it stays quoted. Savings still come from scores."""
        config = GCConfig(context_window=1000, max_recap_ratio=0.378, min_cut_tokens=0, last_n_turns_to_keep=1)
        scores = {0: 0.8, 1: 0.95, 2: 0.95, 3: ARCHIVED_SCORE}
        small, _, _ = make_service(config, scores=scores)
        large, _, _ = make_service(config, scores=scores)
        large.session_manager.get_or_create("s")
        large.use_context_window("s", 100_000)

        _converse(small, 5)
        _converse(large, 5)

        assert _levels(small) == {0: COMPACTED, 1: KEPT, 2: KEPT, 3: ARCHIVED}
        assert _levels(large) == {0: KEPT, 1: KEPT, 2: KEPT, 3: ARCHIVED}

    def test_a_rewound_session_keeps_its_window(self):
        service, _, _ = make_service()
        _converse(service, 2)
        service.use_context_window("s", 1_000_000)
        state = service.session_manager.store.get("s")

        fresh = service.session_manager.replace(state, state.snapshot)

        assert fresh.context_window == 1_000_000


class TestASessionCanBeRecordedWithoutBeingCollected:
    def test_with_collection_off_a_pass_changes_nothing_and_buys_nothing(self):
        service, _, provider = make_service(default_score=COMPACTED_SCORE)
        service.session_manager.get_or_create("s")
        service.set_collection_enabled("s", False)

        results = _converse(service, 4)

        assert {result.status for result in results} == {GCStatus.BYPASSED_NOT_COLLECTED}
        assert _snapshot(service).old == () and provider.calls == []
        assert _young(service) == [0, 1, 2, 3]

    def test_with_collection_off_no_collection_is_forecast(self):
        """The page draws its forecasts from the preview: one that will not run must forecast nothing."""
        service, _, _ = make_service(default_score=COMPACTED_SCORE)
        service.session_manager.get_or_create("s")
        service.set_collection_enabled("s", False)
        _converse(service, 4)

        inspection = service.inspect("s")
        assert not inspection.collection_enabled
        assert inspection.next_cut.turns == () and not inspection.next_cut.batch_ready
        assert inspection.next_cut.levels == {}

    def test_turned_back_on_the_same_session_is_collected(self):
        service, _, _ = make_service(default_score=COMPACTED_SCORE)
        service.session_manager.get_or_create("s")
        service.set_collection_enabled("s", False)
        _converse(service, 3)
        service.set_collection_enabled("s", True)

        [result] = _converse(service, 1, start=3)

        assert result.status is GCStatus.COMPLETED

    def test_a_rewound_session_keeps_its_choice(self):
        service, _, _ = make_service()
        _converse(service, 2)
        service.set_collection_enabled("s", False)
        state = service.session_manager.store.get("s")

        assert service.session_manager.replace(state, state.snapshot).collection_enabled is False


def test_a_relayed_report_in_the_tail_is_not_what_the_user_is_discussing_now():
    """Scored against it, every exchange near a subagent's findings would look like the subject."""
    service, scorer, _ = make_service(GCConfig(min_cut_tokens=0, last_n_turns_to_keep=3), default_score=COMPACTED_SCORE)
    heard: list[list[int]] = []
    score = scorer.score

    def recording(spans, now, newest_turn):
        heard.append([message.turn_index for message in now])
        return score(spans, now, newest_turn)

    scorer.score = recording  # type: ignore[method-assign]

    async def scenario():
        for index in range(5):
            message = turn(index, role="user" if index % 2 == 0 else "assistant")
            if index == 4:
                message = message.model_copy(update={"relayed": True})
            _record(service, message)
            await service.gc_age("s")

    asyncio.run(scenario())
    assert heard[-1] == [2], "the tail is turns 2-4: the user's turn, a reply, and a relayed report"
