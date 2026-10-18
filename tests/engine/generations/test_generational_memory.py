import asyncio

from llm_gc.engine.compaction import NoOpCompactor
from llm_gc.engine.generations.generational_memory import GenerationalMemory
from llm_gc.events import EventBus
from llm_gc.extraction import PatternFactReader
from llm_gc.models import Message


def _make_message(content: str, turn_index: int = 1) -> Message:
    return Message(role="user", content=content, token_count=10, turn_index=turn_index)


def _make_generational_memory() -> GenerationalMemory:
    bus = EventBus()
    return GenerationalMemory(fact_reader=PatternFactReader(), compactor=NoOpCompactor(event_bus=bus))


class TestItStoresNothing:
    """Shared by every conversation it serves, so anything it kept would be
    shared too - which is how one user's facts once reached another's prompt."""

    def test_producing_leaves_no_state_behind(self):
        memory = _make_generational_memory()
        before = dict(vars(memory))
        asyncio.run(memory.condense([_make_message("The database is PostgreSQL", 1), _make_message("turn two", 2)]))
        assert vars(memory) == before
