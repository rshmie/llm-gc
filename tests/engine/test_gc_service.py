"""A collection: condense what aged out of the tail, score every span, give each its level."""

import asyncio
from dataclasses import replace
import math
from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from llm_gc.config import GCConfig
from llm_gc.engine import DegradationReason
from llm_gc.engine.compaction import BaseCompactor, LLMCompactor
from llm_gc.engine.compaction.compaction_strategy import CompactionStrategy
from llm_gc.engine.gc_result import CutOutcome
from llm_gc.engine.gc_status import GCStatus
from llm_gc.engine.pause import ModelCallPause
from llm_gc.events import EventBus, EventType
from llm_gc.exceptions import ConfigError, LLMAuthError, LLMRateLimitError, LLMResponseError
from llm_gc.llm import FakeProvider
from llm_gc.engine.switch_policy import CacheState
from llm_gc.extraction import PatternFactReader
from llm_gc.models.generation_snapshot import DetailLevel
from llm_gc.models.knowledge_entry import KnowledgeStatus
from llm_gc.utils import count_tokens
from tests.engine.support import ARCHIVED_SCORE, COMPACTED_SCORE, KEPT_SCORE, SUMMARY, collect, make_service, turn

KEPT, COMPACTED, ARCHIVED = DetailLevel.KEPT, DetailLevel.COMPACTED, DetailLevel.ARCHIVED


class _FailingCompactor(BaseCompactor):
    """Stands in for a summariser whose provider is down."""

    COMPACTION_STRATEGY = CompactionStrategy.LLM

    async def compact(self, messages):
        raise RuntimeError("provider unavailable")


class _RaisingScorer:
    def __init__(self, error: Exception | None = None):
        self.error = error if error is not None else RuntimeError("scoring failed")

    def score(self, spans, now, newest_turn):
        raise self.error


def _record(service, message) -> None:
    state = service.session_manager.get_or_create("s")
    state.snapshot = state.snapshot.with_turn(message)


def _converse(service, turns: int, start: int = 0) -> list:
    """Record turns one by one, a pass after each; every GCResult, in order."""
    results = []

    async def scenario():
        for index in range(start, start + turns):
            _record(service, turn(index))
            results.append(await collect(service, "s"))

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
                await collect(service, "s")

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
                await collect(service, "s")

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
                await collect(service, "s")

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

        manual = asyncio.run(collect(service, "s", ignore_batch_minimum=True))

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
                results.append(await collect(service, "s"))

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
                results.append(await collect(service, "s"))

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


class _FailingReader:
    async def read(self, spans, on_record):
        raise RuntimeError("extraction unavailable")


class TestFailuresAreReportedAndLoseNothing:
    def test_a_failed_summary_keeps_the_span_whole_and_says_so(self):
        service, _, _ = make_service(compactor=_FailingCompactor(event_bus=None),
                                     scores={0: COMPACTED_SCORE, 1: ARCHIVED_SCORE})

        result = _converse(service, 3)[-1]

        assert _levels(service) == {0: KEPT, 1: ARCHIVED}, "kept rather than lose detail to a failed summary"
        assert (DegradationReason.COMPACTION_FAILED, [0]) in [(d.reason, d.turn_indices) for d in result.degradations]

    def test_a_failed_extraction_still_makes_the_span_and_never_archives_it(self):
        """Live run: with its facts unread, an archived exchange was only its opening line. Like a failed
        summary, a failed fact read now costs no detail: the span is shown at the next richer level."""
        service, _, _ = make_service(fact_reader=_FailingReader(), default_score=ARCHIVED_SCORE)  # type: ignore[arg-type]

        result = _converse(service, 2)[-1]

        assert _levels(service) == {0: COMPACTED}, "its summary was written, so it is summarised"
        assert [d.reason for d in result.degradations] == [DegradationReason.ARCHIVE_FAILED]
        assert "[Archived]" not in asyncio.run(service.gc_collect("s"))[0].content
        # Kept on the span, so the dashboard can say why it has no facts after the warning has scrolled away.
        assert [span.facts_unread for span in service.session_manager.store.get("s").snapshot.old] == [True]

    def test_a_span_archived_without_its_facts_is_shown_in_more_detail_at_the_next_collection(self):
        """A conversation archived before this rule held keeps those spans; the next plan raises them."""
        service, _, _ = make_service(fact_reader=_FailingReader(), default_score=ARCHIVED_SCORE)  # type: ignore[arg-type]
        _converse(service, 2)
        state = service.session_manager.store.get("s")
        state.snapshot = replace(state.snapshot, old=tuple(replace(span, level=ARCHIVED) for span in state.snapshot.old))

        _converse(service, 2, start=2)

        assert _levels(service)[0] is COMPACTED

    def test_with_neither_its_summary_nor_its_facts_a_span_is_kept_whole(self):
        """Both calls failed, as when the summary account ran out of credit."""
        service, _, _ = make_service(compactor=_FailingCompactor(event_bus=None), fact_reader=_FailingReader(),  # type: ignore[arg-type]
                                     default_score=ARCHIVED_SCORE)

        results = _converse(service, 3)

        assert _levels(service) == {}, "nothing is reduced to its opening line"
        assert results[-1].status is GCStatus.BYPASSED_NO_SAVING, "quoting everything saves nothing, so none is made"

    def test_a_failing_scorer_changes_nothing_and_never_escapes(self):
        """gc_age runs fire-and-forget: an exception that escaped it would vanish unseen."""
        service, _, _ = make_service()
        service.span_scorer = _RaisingScorer()  # type: ignore[assignment]

        result = _converse(service, 2)[-1]

        assert result.status is GCStatus.BYPASSED_ON_ERROR
        assert result.failure_stage == "plan" and result.failure_reason == "RuntimeError"
        assert _young(service) == [0, 1] and _snapshot(service).old == ()

    @pytest.mark.parametrize("error, shown", [
        (RuntimeError("could not score: the user wrote 'my card is 4242'"), "RuntimeError"),
        (ConfigError("span weights must be positive"), "span weights must be positive"),
    ])
    def test_a_failure_shows_a_message_only_when_the_error_is_llm_gcs_own(self, error, shown):
        """The result reaches the bus and the dashboard, and another library's message may quote the conversation."""
        service, _, _ = make_service()
        service.span_scorer = _RaisingScorer(error)  # type: ignore[assignment]

        result = _converse(service, 2)[-1]

        assert result.status is GCStatus.BYPASSED_ON_ERROR and result.failure_reason == shown


class TestTheBusSeesWhatTheCollectionDid:
    def test_every_pass_and_every_switch_emits_one_gc_finished_and_collect_emits_none(self):
        service, _, _ = make_service(default_score=COMPACTED_SCORE)
        finished = []
        service.event_bus.subscribe(EventType.GC_FINISHED, finished.append)

        _converse(service, 3)
        emitted = len(finished)
        asyncio.run(service.gc_collect("s"))

        switches = [e for e in finished if e.data["gc_result"].status is GCStatus.COMPLETED]
        assert len(finished) == emitted, "rendering the prompt changes nothing and announces nothing"
        assert len(finished) - len(switches) == 3, "one per pass"
        assert len(switches) == service.session_manager.store.get("s").snapshot.cut_count > 0, "one per switch"
        assert all(event.data["session_id"] == "s" for event in finished)

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

    def test_a_rewound_session_keeps_its_window_and_its_rate(self):
        service, _, _ = make_service()
        _converse(service, 2)
        service.use_context_window("s", 1_000_000)
        service.use_token_rate("s", 1.48)
        state = service.session_manager.store.get("s")

        fresh = service.session_manager.replace(state, state.snapshot)

        assert (fresh.context_window, fresh.token_rate) == (1_000_000, 1.48)


class TestTheWindowIsSetAgainstLLMGCsOwnCount:
    """The window is in the model's tokens, the recap in LLM-GC's; a rate measured on a reply joins them."""

    def test_the_recap_budget_is_its_share_of_the_window_converted_at_the_rate(self):
        """At 1.48 of the model's tokens per LLM-GC token, a 200,000 window is 135,135 of LLM-GC's, and 30% of
        that is 40,540. Multiplied by the rate instead, the budget would grow to 88,800."""
        service, _, _ = make_service(GCConfig(context_window=200_000, max_recap_ratio=0.3))
        _converse(service, 1)

        service.use_token_rate("s", 1.48)
        inspection = service.inspect("s")

        assert (inspection.window_in_own_count, inspection.recap_budget) == (135_135, 40_540)
        assert (inspection.context_window, inspection.token_rate) == (200_000, 1.48), "the window stays the model's"

    def test_until_a_reply_is_measured_the_window_is_used_as_it_is(self):
        service, _, _ = make_service(GCConfig(context_window=200_000, max_recap_ratio=0.3))
        _converse(service, 1)

        inspection = service.inspect("s")

        assert (inspection.token_rate, inspection.window_in_own_count, inspection.recap_budget) == (None, 200_000, 60_000)

    @pytest.mark.parametrize("rate", [0.0, -1.48, math.nan, math.inf])
    def test_a_rate_that_measures_nothing_is_refused(self, rate):
        service, _, _ = make_service()
        _converse(service, 1)

        with pytest.raises(ValueError, match="positive"):
            service.use_token_rate("s", rate)
        assert service.inspect("s").token_rate is None


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
            await collect(service, "s")

    asyncio.run(scenario())
    assert heard[-1] == [2], "the tail is turns 2-4: the user's turn, a reply, and a relayed report"


class TestLlmGcYieldsToTheUpstreamsRateLimit:
    def test_while_paused_no_collection_starts_and_no_model_is_called(self):
        service, _, provider = make_service(default_score=COMPACTED_SCORE)
        service.pause_model_calls(60, "the upstream rate-limited a request from the client")

        results = _converse(service, 4)

        assert {result.status for result in results} == {GCStatus.BYPASSED_PAUSED}
        assert "rate-limited a request from the client" in results[-1].failure_reason
        assert provider.calls == [] and _snapshot(service).old == ()

    def test_when_the_pause_ends_collection_resumes(self):
        clock = [datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)]
        service, _, _ = make_service(default_score=COMPACTED_SCORE)
        service.model_calls = ModelCallPause(clock=lambda: clock[0])
        service.pause_model_calls(60, "overloaded")
        _converse(service, 3)
        clock[0] += timedelta(seconds=61)

        [result] = _converse(service, 1, start=3)

        assert result.status is GCStatus.COMPLETED

    def test_llm_gcs_own_call_being_rate_limited_pauses_it_too(self):
        """Its calls share the limit with the client's: a 429 on one of them is the same pressure."""
        limited = FakeProvider(responses=[LLMRateLimitError("slow down", retry_after_s=45)] * 20)
        service, _, _ = make_service(compactor=LLMCompactor(event_bus=EventBus(), provider=limited),
                                     default_score=COMPACTED_SCORE)

        _converse(service, 3)

        pause = service.model_calls.current()
        assert pause is not None and "LLM-GC's own calls" in pause.reason


# What the provider answered when the account LLM-GC summarises with ran out of credit.
NO_CREDIT = LLMResponseError("anthropic returned 400: Your credit balance is too low to access the Anthropic API.",
                             provider="anthropic", status_code=400, retryable=False)


class _Refusing:
    """A summariser and a fact reader whose every call fails with the given error."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    async def read(self, spans, on_record):
        raise self.error


class _RefusingCompactor(_FailingCompactor):
    def __init__(self, error: Exception) -> None:
        super().__init__(event_bus=None)
        self.error = error

    async def compact(self, messages):
        raise self.error


class TestLlmGcPausesWhenItsOwnCallsCannotSucceed:
    """Live run: with the summary account out of credit, every pass called again, failed again, and the page
    showed ordinary shortfalls rather than saying something needed fixing."""

    def _service(self, summary_error: Exception | None, facts_error: Exception | None):
        service, _, _ = make_service(compactor=_RefusingCompactor(summary_error) if summary_error else None,
                                     fact_reader=_Refusing(facts_error) if facts_error else None,  # type: ignore[arg-type]
                                     default_score=COMPACTED_SCORE)
        return service

    @pytest.mark.parametrize("error", [NO_CREDIT, LLMAuthError("anthropic rejected our credentials (401)")])
    def test_every_call_failing_for_a_lasting_reason_pauses_collection_and_says_why(self, error):
        service = self._service(error, error)

        results = _converse(service, 4)

        pause = service.model_calls.current()
        assert pause is not None and pause.lasting and str(error) in pause.reason
        assert results[-1].status is GCStatus.BYPASSED_PAUSED, "no more calls that would fail the same way"

    def test_a_pass_in_which_a_call_worked_does_not_pause(self):
        service = self._service(NO_CREDIT, None)

        _converse(service, 4)

        assert service.model_calls.current() is None

    @pytest.mark.parametrize("error", [
        LLMResponseError("anthropic returned 503: overloaded", status_code=503, retryable=True),
        RuntimeError("a third-party summariser broke"),
    ])
    def test_a_failure_retrying_could_fix_does_not_pause(self, error):
        service = self._service(error, error)

        _converse(service, 4)

        assert service.model_calls.current() is None


class _RecoveringReader:
    """Reads facts by patterns, failing for the turns in `failing`; records the turns each call read."""

    speakers = PatternFactReader.speakers

    def __init__(self, failing: set[int]) -> None:
        self.failing = failing
        self.calls: list[list[int]] = []
        self._patterns = PatternFactReader()

    async def read(self, spans, on_record):
        turns = [message.turn_index for span in spans for message in span]
        self.calls.append(turns)
        if self.failing & set(turns):
            raise RuntimeError("fact reader unavailable")
        return await self._patterns.read(spans, on_record)


def _said_turns(service, *openings: tuple[int, str]) -> None:
    for index, opening in openings:
        _record(service, turn(index, opening=opening))


class TestFactsThatFailedAreReadAgain:
    """Turn 0 says Postgres while the reader is down; turn 2 changes it to MySQL after it recovers. Read late
    and filed by reading order, Postgres would replace MySQL."""

    def _after_the_outage(self, **config):
        reader = _RecoveringReader(failing={0})
        service, _, _ = make_service(GCConfig(**{"min_cut_tokens": 0, "last_n_turns_to_keep": 1, **config}),
                                     fact_reader=reader, default_score=ARCHIVED_SCORE)  # type: ignore[arg-type]
        _said_turns(service, (0, "The database is Postgres."), (1, "Turn 1 is about storage."))
        asyncio.run(collect(service, "s"))
        assert [span.facts_unread for span in _snapshot(service).old] == [True]
        reader.failing = set()
        _said_turns(service, (2, "Change of plan: the database is MySQL."), (3, "Turn 3 is about storage."))
        return service, reader

    def test_the_next_collection_reads_them_again_and_files_them_in_their_place(self):
        service, reader = self._after_the_outage()

        asyncio.run(collect(service, "s"))

        assert [0] in reader.calls[1:], "read again, in a call of its own"
        knowledge = _snapshot(service).knowledge
        assert [(e.message_turn, e.content, e.knowledge_status) for e in knowledge if e.topic_label == "database"] == [
            (0, "Postgres", KnowledgeStatus.SUPERSEDED), (2, "MySQL", KnowledgeStatus.ACTIVE)]
        assert not any(span.facts_unread for span in _snapshot(service).old)
        assert _levels(service)[0] is ARCHIVED, "with its facts read, it can be archived again"

    def test_a_failed_reread_leaves_it_marked_and_costs_the_new_facts_nothing(self):
        service, reader = self._after_the_outage()
        reader.failing = {0}

        asyncio.run(collect(service, "s"))

        assert [span.facts_unread for span in _snapshot(service).old][0] is True
        assert [e.content for e in _snapshot(service).knowledge if e.topic_label == "database"] == ["MySQL"]

    def test_a_staged_cut_keeps_what_it_read_again_until_it_is_switched_in(self):
        """A warm cache can hold a prepared cut over several passes; the read is made once and survives each."""
        service, reader = self._after_the_outage(min_saving_share=1000.0, max_stale_ratio=1.0)
        asyncio.run(service.gc_age("s"))
        assert service.gc_switch("s", CacheState.WARM) is None, "prepared, waiting"
        reads_of_turn_0 = sum(0 in call for call in reader.calls)
        _said_turns(service, (4, "Turn 4 is about storage."), (5, "Turn 5 is about storage."))

        asyncio.run(service.gc_age("s"))
        switched = service.gc_switch("s", CacheState.COLD)

        assert switched is not None
        assert sum(0 in call for call in reader.calls) == reads_of_turn_0, "not read a second time"
        assert not _snapshot(service).old[0].facts_unread
        assert [e.content for e in _snapshot(service).knowledge if e.knowledge_status is KnowledgeStatus.ACTIVE
                and e.topic_label == "database"] == ["MySQL"]
