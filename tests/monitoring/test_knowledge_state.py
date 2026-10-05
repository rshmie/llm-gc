"""The knowledge-state projection: where every turn is now, and what the model sees of it."""

import asyncio

from llm_gc.config import GCConfig
from llm_gc.engine import GCResult, GCStatus, build_gc_service
from llm_gc.engine.gc_service import GcService
from llm_gc.llm import FakeProvider, LLMResponse, StopReason
from llm_gc.models import KnowledgeEntry, Message, SweepClassification
from llm_gc.models.generation_snapshot import DetailLevel, GenerationSnapshot, Span, file_knowledge
from llm_gc.models.knowledge_entry import KnowledgeStatus, KnowledgeType
from llm_gc.monitoring.knowledge_state import KnowledgeState, TurnFidelity, project_knowledge_state
from llm_gc.utils import count_tokens

SESSION = "s"

SUMMARY = LLMResponse(text="Earlier turns covered the storage decision and the retention window in brief.",
                      stop_reason=StopReason.COMPLETE, input_tokens=0, output_tokens=14,
                      model="scripted", provider="offline")


def _service(**overrides) -> GcService:
    config = GCConfig(**{"context_window": 700, "keep_threshold": 0.55, "archive_threshold": 0.25,
                         "last_n_turns_to_keep": 3, "min_cut_tokens": 60, **overrides})
    return build_gc_service(config, summary_provider=FakeProvider(responses=[SUMMARY] * 400))


# Eight subjects with their own words, in turn: a span goes quiet while the
# conversation is elsewhere and becomes relevant again when its subject returns.
_SUBJECTS = ("database postgres replicas failover", "cache redis eviction expiry", "queue kafka partitions consumers",
             "auth tokens sessions oauth", "billing invoices stripe refunds", "search elasticsearch ranking synonyms",
             "deploy kubernetes rollout canary", "metrics prometheus alerts dashboards")


def _turn(index: int) -> Message:
    words = _SUBJECTS[index % 8].split()
    text = (f"The {words[0]} is version {index}. We weighed {', '.join(words[1:])} and the {words[1]} "
            f"trade-offs at some length this morning.")
    return Message(role="user" if index % 2 == 0 else "assistant", content=text,
                   token_count=count_tokens(text), turn_index=index)


def _converse(service: GcService, turns: int) -> KnowledgeState:
    async def scenario() -> None:
        for index in range(turns):
            service.gc_record(SESSION, [_turn(index)], [f"h{index}"])
            await service.gc_age(SESSION)

    asyncio.run(scenario())
    return project_knowledge_state(service.inspect(SESSION))


def _seed(service: GcService, snapshot: GenerationSnapshot) -> KnowledgeState:
    service.session_manager.get_or_create(SESSION).snapshot = snapshot
    return project_knowledge_state(service.inspect(SESSION))


def _fact(turn: int, topic: str, content: str, kind: KnowledgeType = KnowledgeType.DECISION) -> KnowledgeEntry:
    return KnowledgeEntry(message_turn=turn, content=content, topic_label=topic, knowledge_type=kind)


class TestWhereEachTurnIs:
    def test_before_any_cut_every_turn_is_verbatim_and_there_is_no_front(self):
        state = _converse(_service(min_cut_tokens=100_000), 6)

        assert [t.fidelity for t in state.turns] == [TurnFidelity.VERBATIM] * 6
        assert state.front_text is None
        assert state.cuts == 0

    def test_after_cuts_each_turn_is_reported_where_it_actually_is(self):
        state = _converse(_service(), 40)
        tail = {t.turn_index for t in state.turns if t.fidelity is TurnFidelity.VERBATIM}

        assert len(state.turns) == state.recorded_turns == 40
        assert state.cuts > 0
        for turn in state.turns:
            if turn.fidelity in (TurnFidelity.QUOTED, TurnFidelity.SUMMARISED):
                entry = state.front_entries[turn.front_entry]
                assert turn.turn_index in entry.source_turns
                assert (entry.kind is DetailLevel.KEPT) == (turn.fidelity is TurnFidelity.QUOTED)
        assert max(tail) == 39, "the newest turns are always verbatim"

    def test_archived_turns_keep_the_facts_read_from_them(self):
        """Facts are read from every turn when its span is made, so a turn shown only
        by its span's line still has its own facts - and the model receives them."""
        state = _converse(_service(), 80)
        archived = [t for t in state.turns if t.fidelity in (TurnFidelity.FACTS_ONLY, TurnFidelity.SUPERSEDED)]

        assert archived
        assert all(t.facts_stored > 0 for t in archived)
        assert any(t.facts_sent > 0 for t in archived)


class TestThePassOutcome:
    def test_every_cut_reports_where_its_turns_actually_went(self):
        """Read off the knowledge state before and after each pass, a cut's outcome
        must match it - including turns condensed earlier that the same pass moved.
        A roomy window, so nothing folds and every move is the score's."""
        service = _service(context_window=100_000)
        condensed = {TurnFidelity.QUOTED, TurnFidelity.SUMMARISED}
        as_facts = {TurnFidelity.FACTS_ONLY, TurnFidelity.SUPERSEDED}

        async def scenario() -> list[tuple[KnowledgeState, GCResult, KnowledgeState]]:
            passes = []
            for index in range(80):
                service.gc_record(SESSION, [_turn(index)], [f"h{index}"])
                before = project_knowledge_state(service.inspect(SESSION))
                result = await service.gc_age(SESSION)
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
            assert result.outcome is not None
            assert result.outcome.quoted == sum(now[t] is TurnFidelity.QUOTED for t in taken)
            assert result.outcome.summarised == sum(now[t] is TurnFidelity.SUMMARISED for t in taken)
            assert result.outcome.archived == sum(now[t] in as_facts for t in taken)
            assert result.outcome.front_archived == sum(was[t] in condensed and now[t] in as_facts for t in was)
            assert result.outcome.front_raised == sum(was[t] in as_facts and now[t] in condensed for t in was)
        assert any(result.outcome.front_archived for _, result, _ in cuts), "re-planning was exercised"


class TestWhatTheModelSees:
    def test_a_folded_spans_facts_are_stored_but_not_sent(self):
        """The signal that matters most: kept, and still invisible to the model,
        because the front had no room even for its span's line."""
        def span(first: int, level: DetailLevel) -> Span:
            return Span(turns=(Message(role="user", content=f"said {first}", token_count=1, turn_index=first),),
                        level=level, index_line=f"said {first}")

        facts = (_fact(0, "region", "frankfurt"), _fact(1, "database", "postgres"))
        old = (span(0, DetailLevel.OUT_OF_VIEW), span(1, DetailLevel.ARCHIVED), span(2, DetailLevel.KEPT))
        state = _seed(_service(), GenerationSnapshot(old=old, knowledge=facts))

        assert [(f.content, f.sent) for f in state.facts] == [("frankfurt", False), ("postgres", True)]
        assert [(t.facts_stored, t.facts_sent) for t in state.turns[:2]] == [(1, 0), (1, 1)]
        assert state.front_entries[0].text == "[Quoted exactly] (turn 2, user) said 2", "exactly what the model receives"
        assert state.front_entries[0].stated_text == "said 2", "what a person reads, without the marker"

    def test_a_contradicted_fact_names_what_replaced_it(self):
        knowledge, _ = file_knowledge((), (_fact(1, "database", "postgres"), _fact(6, "database", "mysql")))
        state = _seed(_service(), GenerationSnapshot(young=(_turn(7),), knowledge=knowledge))

        postgres = next(f for f in state.facts if f.content == "postgres")
        assert postgres.status is KnowledgeStatus.SUPERSEDED
        assert postgres.superseded_by_turn == 6
        assert postgres.sent is False, "a superseded fact is never put in front of the model"
        assert state.turns[1].fidelity is TurnFidelity.SUPERSEDED
        assert state.turns[6].fidelity is TurnFidelity.FACTS_ONLY

    def test_an_archived_spans_facts_are_sent_whole(self):
        long_content = "postgres 16 on rds, " * 20
        archived = Span(turns=(Message(role="user", content="which database?", token_count=3, turn_index=1),),
                        level=DetailLevel.ARCHIVED, index_line="which database?")
        state = _seed(_service(), GenerationSnapshot(old=(archived,), young=(_turn(2),),
                                                     knowledge=(_fact(1, "database", long_content),)))

        [fact] = state.facts
        assert fact.sent and not fact.shortened and fact.sent_content == long_content
        assert long_content in (state.front_text or ""), "exactly the words the front carries"

    def test_a_raw_excerpt_supersedes_nothing(self):
        knowledge, _ = file_knowledge((), (_fact(1, "database", "postgres"),
                                           _fact(2, "__raw__", "chatter", KnowledgeType.RAW)))
        state = _seed(_service(), GenerationSnapshot(young=(_turn(3),), knowledge=knowledge))

        assert all(f.status is KnowledgeStatus.ACTIVE for f in state.facts)


class TestTheNextCut:
    def test_only_turns_the_cut_would_take_say_what_they_become(self):
        service = _service(min_cut_tokens=100_000)   # never cut, so the preview has a full batch to show
        state = _converse(service, 12)

        in_cut = set(state.next_cut.turns)
        assert in_cut, "with three turns kept, the first nine are cuttable"
        assert not state.next_cut.batch_ready
        expected = {SweepClassification.KEEP: TurnFidelity.QUOTED,
                    SweepClassification.COMPACT: TurnFidelity.SUMMARISED,
                    SweepClassification.ARCHIVE: TurnFidelity.FACTS_ONLY}
        for turn in state.turns:
            if turn.turn_index in in_cut:
                assert turn.becomes is expected[turn.classification]
            else:
                assert turn.becomes is None
        assert all(t.score is None for t in state.turns[-3:]), "the newest turns are not part of the next collection"
