"""The knowledge-state projection: where every turn is now, and what the model sees of it."""

import asyncio
from dataclasses import replace

from llm_gc.config import GCConfig
from llm_gc.engine import GCResult, GCStatus, build_gc_service
from llm_gc.engine.gc_service import GcService
from llm_gc.extraction import PatternFactReader
from llm_gc.llm import FakeProvider, LLMResponse, StopReason
from llm_gc.models import KnowledgeEntry, Message
from llm_gc.models.generation_snapshot import DetailLevel, GenerationSnapshot, Span, file_knowledge
from llm_gc.models.knowledge_entry import KnowledgeStatus, KnowledgeType
from llm_gc.monitoring.knowledge_state import KnowledgeState, TurnFidelity, project_knowledge_state
from llm_gc.utils import count_tokens

from tests.engine.support import collect

SESSION = "s"

SUMMARY = LLMResponse(text="Earlier turns covered the storage decision and the retention window in brief.",
                      stop_reason=StopReason.COMPLETE, input_tokens=0, output_tokens=14,
                      model="scripted", provider="offline")


def _service(**overrides) -> GcService:
    config = GCConfig(**{"context_window": 700, "keep_threshold": 0.55, "archive_threshold": 0.25,
                         "last_n_turns_to_keep": 3, "min_cut_tokens": 60, **overrides})
    return build_gc_service(config, summary_provider=FakeProvider(responses=[SUMMARY] * 400),
                            fact_reader=PatternFactReader())


# Eight subjects with their own words, in turn: a span goes quiet while the
# conversation is elsewhere and becomes relevant again when its subject returns.
_SUBJECTS = ("database postgres replicas failover", "cache redis eviction expiry", "queue kafka partitions consumers",
             "auth tokens sessions oauth", "billing invoices stripe refunds", "search elasticsearch ranking synonyms",
             "deploy kubernetes rollout canary", "metrics prometheus alerts dashboards")


def _turn(index: int) -> Message:
    words = _SUBJECTS[index % 8].split()
    text = f"The {words[0]} is version {index}. " + (f"We weighed {', '.join(words[1:])} and the {words[1]} "
                                                       f"trade-offs at some length this morning. ") * 4
    return Message(role="user" if index % 2 == 0 else "assistant", content=text,
                   token_count=count_tokens(text), turn_index=index)


def _converse(service: GcService, turns: int) -> KnowledgeState:
    async def scenario() -> None:
        for index in range(turns):
            service.gc_record(SESSION, [_turn(index)], [f"h{index}"])
            await collect(service, SESSION)

    asyncio.run(scenario())
    return project_knowledge_state(service.inspect(SESSION))


def _seed(service: GcService, snapshot: GenerationSnapshot) -> KnowledgeState:
    service.session_manager.get_or_create(SESSION).snapshot = snapshot
    return project_knowledge_state(service.inspect(SESSION))


def _fact(turn: int, topic: str, content: str, kind: KnowledgeType = KnowledgeType.DECISION) -> KnowledgeEntry:
    return KnowledgeEntry(message_turn=turn, content=content, topic_label=topic, knowledge_type=kind)


def _span(level: DetailLevel, *said: tuple[int, str, str], opening: str = "") -> Span:
    turns = tuple(Message(role=role, content=text, token_count=count_tokens(text), turn_index=turn)
                  for turn, role, text in said)
    return Span(turns=turns, level=level, index_line=opening)


class TestWhereEachTurnIs:
    def test_before_any_cut_every_turn_is_verbatim_and_there_is_no_front(self):
        state = _converse(_service(min_cut_tokens=100_000), 6)

        assert [t.fidelity for t in state.turns] == [TurnFidelity.VERBATIM] * 6
        assert state.front_text is None and state.spans == []
        assert state.cuts == 0

    def test_the_page_is_told_the_window_in_both_counts_and_the_rate_between_them(self):
        """The window meter divides the provider's count by the model's window; the bound is set in LLM-GC's."""
        service = _service(context_window=200_000)
        _converse(service, 2)
        service.use_token_rate(SESSION, 2.0)

        state = project_knowledge_state(service.inspect(SESSION))

        assert (state.context_window, state.token_rate, state.window_in_own_count) == (200_000, 2.0, 100_000)

    def test_after_cuts_every_turn_is_in_the_tail_or_shown_at_its_spans_level(self):
        state = _converse(_service(), 40)

        assert [t.turn_index for t in state.turns] == list(range(40)) and state.recorded_turns == 40
        assert state.cuts > 0
        for turn in state.turns:
            if turn.fidelity is TurnFidelity.VERBATIM:
                assert turn.span is None
            else:
                span = state.spans[turn.span]
                assert turn.turn_index in span.turns
                assert turn.fidelity.value == span.level.value
        assert state.turns[-1].fidelity is TurnFidelity.VERBATIM, "the newest turns are always verbatim"

    def test_a_condensed_turn_keeps_its_own_words(self):
        """A span holds its turns' originals, so the dashboard shows what was said
        without a copy of its own."""
        state = _converse(_service(), 40)

        condensed = [t for t in state.turns if t.fidelity is not TurnFidelity.VERBATIM]
        assert condensed
        assert all(t.text == _turn(t.turn_index).content for t in condensed)


    def test_tool_turns_are_marked_so_tool_results_are_not_shown_as_the_users_words(self):
        call = [{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}}]
        result = [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "README.md"}]
        turns = [Message(role="user", content="list the files", token_count=3, turn_index=0),
                 Message(role="assistant", content="[tool call: Bash]", token_count=5, turn_index=1, raw_content=call),
                 Message(role="user", content="[tool result] README.md", token_count=6, turn_index=2,
                         raw_content=result)]
        service = _service(min_cut_tokens=100_000)
        service.gc_record(SESSION, turns, ["h0", "h1", "h2"])

        state = project_knowledge_state(service.inspect(SESSION))

        assert [t.tool_calls for t in state.turns] == [False, True, False]
        assert [t.tool_results for t in state.turns] == [False, False, True]

class TestThePassOutcome:
    def test_every_cut_reports_where_its_turns_actually_went(self):
        """Read off the knowledge state before and after each pass, a cut's outcome
        must name exactly the turns that moved - including turns condensed earlier
        that the same pass lowered or raised, by any number of levels."""
        service = _service(context_window=1_500)
        rank = {TurnFidelity.OUT_OF_VIEW: 0, TurnFidelity.ARCHIVED: 1, TurnFidelity.COMPACTED: 2, TurnFidelity.KEPT: 3}

        async def scenario() -> list[tuple[KnowledgeState, GCResult, KnowledgeState]]:
            passes = []
            for index in range(80):
                service.gc_record(SESSION, [_turn(index)], [f"h{index}"])
                before = project_knowledge_state(service.inspect(SESSION))
                result = await collect(service, SESSION)
                passes.append((before, result, project_knowledge_state(service.inspect(SESSION))))
            return passes

        cuts = [(before, result, after) for before, result, after in asyncio.run(scenario())
                if result.status is GCStatus.COMPLETED]
        assert cuts
        for before, result, after in cuts:
            was = {t.turn_index: t.fidelity for t in before.turns}
            now = {t.turn_index: t.fidelity for t in after.turns}
            taken = [turn for turn, fidelity in was.items() if fidelity is TurnFidelity.VERBATIM
                     and now[turn] is not TurnFidelity.VERBATIM]
            outcome = result.outcome
            assert outcome is not None
            assert outcome.taken == taken
            assert outcome.kept == [t for t in taken if now[t] is TurnFidelity.KEPT]
            assert outcome.compacted == [t for t in taken if now[t] is TurnFidelity.COMPACTED]
            assert outcome.archived == [t for t in taken if now[t] is TurnFidelity.ARCHIVED]
            assert outcome.folded == [t for t in taken if now[t] is TurnFidelity.OUT_OF_VIEW]
            earlier = [t for t in was if was[t] is not TurnFidelity.VERBATIM]
            assert sorted(outcome.lowered) == [t for t in earlier if rank[now[t]] < rank[was[t]]]
            assert sorted(outcome.raised) == [t for t in earlier if rank[now[t]] > rank[was[t]]]
        assert any(result.outcome.lowered for _, result, _ in cuts), "re-planning was exercised"


class TestWhatTheModelSees:
    def test_a_reply_without_facts_of_its_own_is_archived_not_out_of_view(self):
        """The model reads an archived span's line for every turn in it, so a reply
        that yielded no fact is no less visible than the question it answered."""
        span = _span(DetailLevel.ARCHIVED, (0, "user", "Let's use Stripe for payments, with a 4% fee."),
                     (1, "assistant", "Sounds good."), opening="Let's use Stripe for payments, with a 4% fee.")
        state = _seed(_service(), GenerationSnapshot(old=(span,), knowledge=(_fact(0, "fee", "4%"),)))

        assert [t.fidelity for t in state.turns] == [TurnFidelity.ARCHIVED] * 2
        assert state.spans[0].text == "[Archived] (turns 0–1) Let's use Stripe for payments, with a 4% fee. · fee: 4%"
        assert state.spans[0].text in (state.front_text or ""), "exactly what the model receives"

    def test_a_span_whose_facts_could_not_be_read_says_so(self):
        """An empty facts panel read as "nothing settled" when the read had failed."""
        unread = replace(_span(DetailLevel.ARCHIVED, (0, "user", "Let's use Stripe."), (1, "assistant", "Done.")),
                         facts_unread=True)
        state = _seed(_service(), GenerationSnapshot(old=(unread,)))

        assert [span.facts_unread for span in state.spans] == [True]

    def test_a_folded_span_shares_its_runs_line_and_its_facts_are_not_sent(self):
        """The signal that matters most: kept, and still invisible to the model,
        because the front had no room even for its span's line."""
        old = (_span(DetailLevel.OUT_OF_VIEW, (0, "user", "the region is frankfurt")),
               _span(DetailLevel.ARCHIVED, (1, "user", "the database is postgres"), opening="the database"),
               _span(DetailLevel.KEPT, (2, "user", "the cache is redis")))
        facts = (_fact(0, "region", "frankfurt"), _fact(1, "database", "postgres"), _fact(2, "cache", "redis"))
        state = _seed(_service(), GenerationSnapshot(old=old, knowledge=facts))

        folded, archived, kept = state.spans
        assert folded.text == "[Out of view] (turn 0) 1 earlier exchange, folded to stay within the budget."
        assert folded.tokens is None and archived.tokens is not None
        assert kept.text == "[Quoted exactly] (turn 2, user) the cache is redis"
        assert state.turns[0].fidelity is TurnFidelity.OUT_OF_VIEW
        assert [(f.content, f.sent, f.span_level) for f in state.facts] == [
            ("frankfurt", False, DetailLevel.OUT_OF_VIEW),
            ("postgres", True, DetailLevel.ARCHIVED),
            ("redis", False, DetailLevel.KEPT)]

    def test_a_contradicted_fact_names_what_replaced_it(self):
        knowledge, _ = file_knowledge((), (_fact(1, "database", "postgres"), _fact(6, "database", "mysql")))
        state = _seed(_service(), GenerationSnapshot(young=(_turn(7),), knowledge=knowledge))

        postgres = next(f for f in state.facts if f.content == "postgres")
        assert postgres.status is KnowledgeStatus.SUPERSEDED
        assert (postgres.superseded_by_turn, postgres.superseded_by) == (6, "mysql")
        assert postgres.superseded_by_speaker is None, "the facts in this test name no speaker"
        assert postgres.sent is False, "a superseded fact is never put in front of the model as current"

    def test_a_compacted_span_reports_its_summary_without_the_marker(self):
        state = _converse(_service(context_window=100_000), 40)

        compacted = [span for span in state.spans if span.level is DetailLevel.COMPACTED]
        assert compacted
        assert all(span.summary == SUMMARY.text and span.summary in span.text for span in compacted)


class TestTheNextCollection:
    def test_only_turns_the_cut_would_take_say_what_they_become(self):
        service = _service(min_cut_tokens=100_000)   # never cut, so the preview has a full batch to show
        state = _converse(service, 12)

        in_cut = set(state.next_cut.turns)
        assert in_cut, "with three turns kept, the first nine are cuttable"
        assert not state.next_cut.batch_ready
        assert [turn for span in state.next_cut.spans for turn in span] == state.next_cut.turns
        for turn in state.turns:
            assert (turn.becomes is not None) == (turn.turn_index in in_cut)
            assert (turn.score is not None) == (turn.turn_index in in_cut)

    def test_a_span_the_next_collection_would_move_says_so(self):
        """Levels are re-planned at every collection, so a span already shown can
        change too: a long kept span nobody has mentioned since goes quiet."""
        words = "We compared the replicas, the failover drill and the backup schedule in detail. " * 12
        old = (_span(DetailLevel.KEPT, (0, "user", "Which database?"), (1, "assistant", words),
                     opening="Which database?"),)
        young = tuple(Message(role="user" if i % 2 == 0 else "assistant", content="unrelated chatter",
                              token_count=3, turn_index=i) for i in range(2, 120))
        state = _seed(_service(context_window=100_000), GenerationSnapshot(old=old, young=young))

        [span] = state.spans
        assert span.level is DetailLevel.KEPT and span.becomes is DetailLevel.ARCHIVED
        assert [t.becomes for t in state.turns[:2]] == [TurnFidelity.ARCHIVED] * 2
        assert span.score < 0.3


class TestWhatTheDashboardSaysIsWhatTheModelReceives:
    """The audit behind every figure on the page, over a long conversation: kept,
    compacted, archived and folded spans, facts replaced along the way."""

    @staticmethod
    def _states() -> list[KnowledgeState]:
        service = _service(context_window=2_500)

        async def scenario() -> list[KnowledgeState]:
            states = []
            for index in range(120):
                service.gc_record(SESSION, [_turn(index)], [f"h{index}"])
                await collect(service, SESSION)
                if index % 7 == 6:
                    states.append(project_knowledge_state(service.inspect(SESSION)))
            return states

        return asyncio.run(scenario())

    def test_every_claim_matches_the_prompt(self):
        states = self._states()
        assert any(span.level is DetailLevel.OUT_OF_VIEW for span in states[-1].spans), "folding was exercised"
        assert any(fact.status is KnowledgeStatus.SUPERSEDED for fact in states[-1].facts), "and replacement"
        for state in states:
            front = state.front_text or ""
            verbatim = [turn for turn in state.turns if turn.fidelity is TurnFidelity.VERBATIM]
            assert state.context_tokens == state.front_tokens + sum(turn.tokens for turn in verbatim), \
                "the size shown is the recap plus the recent messages"
            assert [turn.turn_index for turn in state.turns] == list(range(state.recorded_turns)), \
                "every recorded message is shown, once"
            for span in state.spans:
                assert span.text in front, "an exchange's text is exactly what the recap carries"
                for index in span.turns:
                    assert state.turns[index].fidelity.value == span.level.value
                    assert (state.turns[index].becomes is not None) == (span.becomes is not None)
            level_of = {index: span.level for span in state.spans for index in span.turns}
            for fact in state.facts:
                line = next(span.text for span in state.spans if fact.turn in span.turns)
                stated = f"{fact.topic}: {fact.content}"
                if fact.status is KnowledgeStatus.SUPERSEDED:
                    assert not fact.sent
                    if level_of[fact.turn] is not DetailLevel.OUT_OF_VIEW:
                        assert f"not {fact.content}." in line, "a replaced value is marked on its exchange"
                else:
                    assert fact.sent == (level_of[fact.turn] in (DetailLevel.COMPACTED, DetailLevel.ARCHIVED))
                    assert (stated in line) == fact.sent, "a fact marked sent is on its exchange's line"
            taken = set(state.next_cut.turns)
            for turn in verbatim:
                assert (turn.becomes is not None) == (turn.turn_index in taken)
