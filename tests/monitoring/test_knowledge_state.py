"""The knowledge-state projection: where every turn is now, and what the model sees of it."""

import asyncio

from llm_gc.config import GCConfig
from llm_gc.engine import GCResult, GCStatus
from llm_gc.engine.compaction import LLMCompactor
from llm_gc.engine.gc_service import GcService
from llm_gc.engine.generations import GenerationalMemory
from llm_gc.engine.sweep import Sweeper, ThresholdSweepStrategy
from llm_gc.events import EventBus
from llm_gc.extraction import KnowledgeExtractor
from llm_gc.llm import FakeProvider, LLMResponse, StopReason
from llm_gc.models import KnowledgeEntry, Message, SweepClassification
from llm_gc.models.generation_snapshot import EntryKind, FrontEntry, GenerationSnapshot, file_knowledge
from llm_gc.models.knowledge_entry import KnowledgeStatus, KnowledgeType
from llm_gc.monitoring.knowledge_state import KnowledgeState, TurnFidelity, project_knowledge_state
from llm_gc.scoring import RecencyScorer, RelevanceScorer, RelevanceType
from llm_gc.session import SessionManager
from llm_gc.utils import count_tokens

SESSION = "s"

SUMMARY = LLMResponse(text="Earlier turns covered the storage decision and the retention window in brief.",
                      stop_reason=StopReason.COMPLETE, input_tokens=0, output_tokens=14,
                      model="scripted", provider="offline")


def _service(**overrides) -> GcService:
    bus = EventBus()
    config = GCConfig(**{"context_window": 700, "keep_threshold": 0.55, "archive_threshold": 0.25,
                         "min_compactable_tokens": 10, "last_n_turns_to_keep": 3, "min_cut_tokens": 60,
                         **overrides})
    return GcService(
        session_manager=SessionManager(),
        generational_memory=GenerationalMemory(
            knowledge_extractor=KnowledgeExtractor(event_bus=bus),
            compactor=LLMCompactor(event_bus=bus, provider=FakeProvider(responses=[SUMMARY] * 400))),
        relevance_scorer=RelevanceScorer(scorers=[RecencyScorer(decay_rate=0.15)],
                                         weights={RelevanceType.RECENCY: 1.0}, event_bus=bus),
        sweeper=Sweeper(sweeper_strategy=ThresholdSweepStrategy(gc_config=config), gc_config=config, event_bus=bus),
        event_bus=bus, gc_config=config)


def _turn(index: int) -> Message:
    text = (f"the topic{index % 8} is version {index} and we walked through the implications "
            f"for throughput and retention at some length this morning.")
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
                assert (entry.kind is EntryKind.QUOTE) == (turn.fidelity is TurnFidelity.QUOTED)
        assert max(tail) == 39, "the newest turns are always verbatim"

    def test_turns_of_an_archived_summary_keep_their_facts(self):
        """A summary of turns 3-6 that goes cold is archived into facts labelled
        turn 3. Without its coverage, turns 4-6 would read as reduced to nothing."""
        state = _converse(_service(), 80)
        archived = [t for t in state.turns if t.fidelity in (TurnFidelity.FACTS_ONLY, TurnFidelity.SUPERSEDED)]

        assert archived
        assert all(t.facts_stored > 0 for t in archived)


class TestThePassOutcome:
    def test_every_cut_reports_where_its_turns_actually_went(self):
        """Read off the knowledge state before and after each pass, a cut's outcome
        must match it - including turns the same pass moved from the front to facts."""
        service = _service()
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
        assert any(result.outcome.front_archived for _, result, _ in cuts), "the front sweep was exercised"


class TestWhatTheModelSees:
    def test_a_fact_left_out_of_the_front_is_stored_but_not_sent(self):
        """The signal that matters most: archived, kept, and still invisible to the
        model, because the front had no room for it."""
        facts = tuple(_fact(i, f"subject{i}", f"value{i}") for i in range(10))
        quote = FrontEntry(message=Message(role="user", content="[Quoted exactly] hi", token_count=5,
                                           turn_index=10), kind=EntryKind.QUOTE, source_turns=(10,), text="hi")
        state = _seed(_service(context_window=2000, max_memory_ratio=0.03),
                      GenerationSnapshot(old=(quote,), knowledge=facts))

        sent = [f for f in state.facts if f.sent]
        assert 0 < len(sent) < len(facts)
        assert state.front_entries[0].text == "[Quoted exactly] hi", "exactly what the model receives"
        assert state.front_entries[0].stated_text == "hi", "what a person reads, without the marker"
        for turn in state.turns[:10]:
            assert turn.fidelity is TurnFidelity.FACTS_ONLY
            assert turn.facts_stored == 1
            assert turn.facts_sent == (1 if any(f.turn == turn.turn_index for f in sent) else 0)

    def test_a_contradicted_fact_names_what_replaced_it(self):
        knowledge, _ = file_knowledge((), (_fact(1, "database", "postgres"), _fact(6, "database", "mysql")))
        state = _seed(_service(), GenerationSnapshot(young=(_turn(7),), knowledge=knowledge))

        postgres = next(f for f in state.facts if f.content == "postgres")
        assert postgres.status is KnowledgeStatus.SUPERSEDED
        assert postgres.superseded_by_turn == 6
        assert postgres.sent is False, "a superseded fact is never put in front of the model"
        assert state.turns[1].fidelity is TurnFidelity.SUPERSEDED
        assert state.turns[6].fidelity is TurnFidelity.FACTS_ONLY

    def test_a_long_excerpt_reports_the_cut_the_model_gets(self):
        """The dashboard once called a 2,136-token excerpt "the message itself" while
        the model received its first 180 characters."""
        long_text = "word " * 400
        knowledge = (_fact(1, "__raw__", long_text, KnowledgeType.RAW), _fact(2, "__raw__", "short", KnowledgeType.RAW))
        state = _seed(_service(context_window=100_000), GenerationSnapshot(young=(_turn(3),), knowledge=knowledge))

        cut, whole = state.facts
        assert cut.sent and whole.sent
        assert cut.shortened and cut.sent_content.endswith("...") and len(cut.sent_content) < len(long_text)
        assert cut.sent_content in (state.front_text or ""), "exactly the words the front carries"
        assert not whole.shortened and whole.sent_content == "short"

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
        assert all(t.forced_keep for t in state.turns[-3:]), "the newest turns are kept by rule, not by score"
