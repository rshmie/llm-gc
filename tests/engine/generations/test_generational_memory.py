from llm_gc.engine.generations.generational_memory import GenerationalMemory
from llm_gc.engine.generations.permanent_generation import PermanentGeneration
from llm_gc.events import EventBus, EventType
from llm_gc.extraction import KnowledgeExtractor
from llm_gc.models import Message
from llm_gc.models.knowledge_entry import KnowledgeType


def _make_message(content: str, turn_index: int = 1) -> Message:
    return Message(role="user", content=content, token_count=10, turn_index=turn_index)


def _make_generational_memory(event_bus: EventBus | None = None) -> GenerationalMemory:
    bus = event_bus if event_bus is not None else EventBus()
    extractor = KnowledgeExtractor(event_bus=bus)
    permanent_gen = PermanentGeneration(event_bus=bus)
    return GenerationalMemory(
        event_bus=bus,
        knowledge_extractor=extractor,
        permanent_generation=permanent_gen,
    )


class TestArchiveMessage:
    def test_archive_fresh_message_never_previously_tracked(self):
        """Regression: the composer archives messages straight from this pass's
        classification result. archive_message must not assume the message was
        ever handed to generational memory before now.
        """
        gen_mem = _make_generational_memory()
        msg = _make_message("Ordinary chatter", turn_index=1)

        gen_mem.archive_message(msg)  # must not raise

        assert len(gen_mem.get_permanent_gen()) == 1

    def test_extracts_knowledge_and_stores_in_permanent_gen(self):
        gen_mem = _make_generational_memory()
        msg = _make_message("The database is PostgreSQL", turn_index=1)

        gen_mem.archive_message(msg)

        permanent = gen_mem.get_permanent_gen()
        assert len(permanent) >= 1
        assert any("PostgreSQL" in e.content for e in permanent)

    def test_emits_archived_event(self):
        bus = EventBus()
        archived_events = []
        bus.subscribe(EventType.MESSAGE_ARCHIVED, lambda e: archived_events.append(e))
        gen_mem = _make_generational_memory(event_bus=bus)
        msg = _make_message("The database is PostgreSQL", turn_index=1)

        gen_mem.archive_message(msg)

        assert len(archived_events) == 1
        assert archived_events[0].data["message"] == msg
        assert "extracted_knowledge_entries" in archived_events[0].data


class TestGetters:
    def test_get_permanent_gen_returns_active_entries(self):
        gen_mem = _make_generational_memory()
        gen_mem.archive_message(_make_message("The database is PostgreSQL", turn_index=1))

        permanent = gen_mem.get_permanent_gen()
        assert len(permanent) >= 1


class TestArchiveVerbatimFallback:
    """The guarantee: every archived turn leaves exactly one trace.

    When conservative extraction finds no knowledge in a turn, archival must not
    drop it silently. The turn's content is preserved verbatim as a RAW
    KnowledgeEntry, so removing it from active context never loses information.
    """

    def test_archive_without_extractable_knowledge_keeps_verbatim_raw_entry(self):
        gen_mem = _make_generational_memory()
        # Matches none of the extractor's fact/decision/preference patterns.
        msg = _make_message("Thanks for the help", turn_index=1)

        gen_mem.archive_message(msg)

        permanent = gen_mem.get_permanent_gen()
        assert len(permanent) == 1
        raw_entry = permanent[0]
        assert raw_entry.knowledge_type == KnowledgeType.RAW
        assert raw_entry.content == "Thanks for the help"  # verbatim, not interpreted
        assert raw_entry.topic_label == "__raw__"
        assert raw_entry.message_turn == 1

    def test_archive_with_extractable_knowledge_adds_no_raw_entry(self):
        gen_mem = _make_generational_memory()
        msg = _make_message("The database is PostgreSQL", turn_index=1)

        gen_mem.archive_message(msg)

        permanent = gen_mem.get_permanent_gen()
        assert len(permanent) >= 1
        assert all(e.knowledge_type != KnowledgeType.RAW for e in permanent)
        assert any("PostgreSQL" in e.content for e in permanent)

    def test_two_unextractable_archives_both_survive_as_active(self):
        gen_mem = _make_generational_memory()

        msg1 = _make_message("Thanks for the help", turn_index=1)
        msg2 = _make_message("Sounds good to me", turn_index=2)

        gen_mem.archive_message(msg1)
        gen_mem.archive_message(msg2)

        permanent = gen_mem.get_permanent_gen()
        assert len(permanent) == 2  # neither raw entry superseded the other
        assert all(e.knowledge_type == KnowledgeType.RAW for e in permanent)
        assert {e.content for e in permanent} == {"Thanks for the help", "Sounds good to me"}
