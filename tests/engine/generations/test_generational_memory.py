import asyncio

from llm_gc.engine.compaction import NoOpCompactor
from llm_gc.engine.generations.generational_memory import GenerationalMemory
from llm_gc.events import EventBus
from llm_gc.extraction import KnowledgeExtractor
from llm_gc.models import Message
from llm_gc.models.knowledge_entry import KnowledgeType


def _make_message(content: str, turn_index: int = 1) -> Message:
    return Message(role="user", content=content, token_count=10, turn_index=turn_index)


def _make_generational_memory() -> GenerationalMemory:
    bus = EventBus()
    return GenerationalMemory(knowledge_extractor=KnowledgeExtractor(event_bus=bus),
                              compactor=NoOpCompactor(event_bus=bus))


class TestVerbatimFallback:
    """Every archived turn leaves a trace: when extraction finds nothing, the
    content is kept verbatim as a RAW entry rather than silently dropped."""

    def test_a_turn_with_nothing_extractable_yields_one_verbatim_raw_entry(self):
        entries = _make_generational_memory().extract_knowledge(_make_message("Thanks for the help"))
        assert len(entries) == 1
        assert entries[0].knowledge_type == KnowledgeType.RAW
        assert entries[0].content == "Thanks for the help"
        assert entries[0].topic_label == "__raw__"
        assert entries[0].message_turn == 1

    def test_a_turn_with_extractable_knowledge_yields_no_raw_entry(self):
        entries = _make_generational_memory().extract_knowledge(_make_message("The database is PostgreSQL"))
        assert entries
        assert all(e.knowledge_type != KnowledgeType.RAW for e in entries)
        assert any("PostgreSQL" in e.content for e in entries)


    def test_a_partly_read_turn_keeps_the_sentences_no_fact_came_from(self):
        """Archived as only its facts, this turn kept the database and lost the
        latency requirement - the loss the verbatim fallback exists to prevent."""
        text = "OK, the database is postgres. Also the API must answer p99 under 200 ms; that's a hard requirement."

        entries = _make_generational_memory().extract_knowledge(_make_message(text))

        assert [(e.knowledge_type, e.content) for e in entries] == [
            (KnowledgeType.FACT, "postgres"),
            (KnowledgeType.RAW, "Also the API must answer p99 under 200 ms; that's a hard requirement.")]

    def test_a_question_beside_a_fact_is_kept_and_a_courtesy_is_not(self):
        entries = _make_generational_memory().extract_knowledge(
            _make_message("Thanks! The cache is redis. Should we shard it?"))
        assert [e.content for e in entries] == ["redis", "Should we shard it?"]


class TestItStoresNothing:
    """Shared by every conversation it serves, so anything it kept would be
    shared too - which is how one user's facts once reached another's prompt."""

    def test_producing_leaves_no_state_behind(self):
        memory = _make_generational_memory()
        before = dict(vars(memory))
        memory.extract_knowledge(_make_message("The database is PostgreSQL"))
        asyncio.run(memory.summarise([_make_message("turn one", 1), _make_message("turn two", 2)]))
        assert vars(memory) == before
