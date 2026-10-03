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


class TestItStoresNothing:
    """Shared by every conversation it serves, so anything it kept would be
    shared too - which is how one user's facts once reached another's prompt."""

    def test_producing_leaves_no_state_behind(self):
        memory = _make_generational_memory()
        before = dict(vars(memory))
        memory.extract_knowledge(_make_message("The database is PostgreSQL"))
        asyncio.run(memory.summarise([_make_message("turn one", 1), _make_message("turn two", 2)]))
        assert vars(memory) == before
