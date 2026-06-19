from llm_gc.engine.generations.generational_memory import GenerationalMemory
from llm_gc.engine.generations.permanent_generation import PermanentGeneration
from llm_gc.events import EventBus, EventType
from llm_gc.extraction import KnowledgeExtractor
from llm_gc.models import Message
from llm_gc.models.knowledge_entry import KnowledgeType


def _make_message(content: str, turn_index: int = 1) -> Message:
    return Message(role="user", content=content, token_count=10, turn_index=turn_index)


def _make_generational_memory(event_bus: EventBus | None = None, max_young_gen_size: int = 3) -> GenerationalMemory:
    bus = event_bus if event_bus is not None else EventBus()
    extractor = KnowledgeExtractor(event_bus=bus)
    permanent_gen = PermanentGeneration(event_bus=bus)
    return GenerationalMemory(
        event_bus=bus,
        knowledge_extractor=extractor,
        permanent_generation=permanent_gen,
        max_young_gen_size=max_young_gen_size,
    )


class TestAddMessageTurn:
    def test_adds_message_to_young_gen(self):
        gen_mem = _make_generational_memory()
        msg = _make_message("Hello world", turn_index=1)

        gen_mem.add_message_turn(msg)

        young = gen_mem.get_young_gen()
        assert len(young) == 1
        assert young[0].content == "Hello world"

    def test_young_gen_holds_up_to_max_size(self):
        gen_mem = _make_generational_memory(max_young_gen_size=3)

        for i in range(3):
            gen_mem.add_message_turn(_make_message(f"Turn {i}", turn_index=i))

        assert len(gen_mem.get_young_gen()) == 3
        assert len(gen_mem.get_old_gen()) == 0

    def test_emits_message_added_event(self):
        bus = EventBus()
        received_events = []
        bus.subscribe(EventType.MESSAGE_ADDED_TO_YOUNG_GEN, lambda e: received_events.append(e))
        gen_mem = _make_generational_memory(event_bus=bus)

        msg = _make_message("Hello", turn_index=1)
        gen_mem.add_message_turn(msg)

        assert len(received_events) == 1
        assert received_events[0].data["message"] == msg
        assert received_events[0].data["current_young_gen_size"] == 1


class TestPromotion:
    def test_promotes_oldest_to_old_gen_when_young_full(self):
        gen_mem = _make_generational_memory(max_young_gen_size=2)

        msg1 = _make_message("First", turn_index=1)
        msg2 = _make_message("Second", turn_index=2)
        msg3 = _make_message("Third", turn_index=3)

        gen_mem.add_message_turn(msg1)
        gen_mem.add_message_turn(msg2)
        gen_mem.add_message_turn(msg3)

        young = gen_mem.get_young_gen()
        old = gen_mem.get_old_gen()
        assert len(young) == 2
        assert young[0].content == "Second"
        assert young[1].content == "Third"
        assert len(old) == 1
        assert old[0].content == "First"

    def test_multiple_promotions(self):
        gen_mem = _make_generational_memory(max_young_gen_size=2)

        for i in range(5):
            gen_mem.add_message_turn(_make_message(f"Turn {i}", turn_index=i))

        young = gen_mem.get_young_gen()
        old = gen_mem.get_old_gen()
        assert len(young) == 2
        assert young[0].content == "Turn 3"
        assert young[1].content == "Turn 4"
        assert len(old) == 3
        assert old[0].content == "Turn 0"
        assert old[1].content == "Turn 1"
        assert old[2].content == "Turn 2"

    def test_emits_promotion_event(self):
        bus = EventBus()
        promoted_events = []
        bus.subscribe(EventType.MESSAGE_PROMOTED_TO_OLD_GEN, lambda e: promoted_events.append(e))
        gen_mem = _make_generational_memory(event_bus=bus, max_young_gen_size=2)

        gen_mem.add_message_turn(_make_message("First", turn_index=1))
        gen_mem.add_message_turn(_make_message("Second", turn_index=2))
        assert len(promoted_events) == 0

        gen_mem.add_message_turn(_make_message("Third", turn_index=3))
        assert len(promoted_events) == 1
        assert promoted_events[0].data["promoted_message"].content == "First"

    def test_promotion_does_not_prevent_add_event(self):
        bus = EventBus()
        added_events = []
        bus.subscribe(EventType.MESSAGE_ADDED_TO_YOUNG_GEN, lambda e: added_events.append(e))
        gen_mem = _make_generational_memory(event_bus=bus, max_young_gen_size=2)

        gen_mem.add_message_turn(_make_message("First", turn_index=1))
        gen_mem.add_message_turn(_make_message("Second", turn_index=2))
        gen_mem.add_message_turn(_make_message("Third", turn_index=3))

        assert len(added_events) == 3
        assert added_events[2].data["message"].content == "Third"


class TestArchiveMessage:
    def test_removes_message_from_old_gen(self):
        gen_mem = _make_generational_memory(max_young_gen_size=2)

        msg1 = _make_message("First", turn_index=1)
        msg2 = _make_message("Second", turn_index=2)
        msg3 = _make_message("Third", turn_index=3)

        gen_mem.add_message_turn(msg1)
        gen_mem.add_message_turn(msg2)
        gen_mem.add_message_turn(msg3)

        gen_mem.archive_message(msg1)

        assert len(gen_mem.get_old_gen()) == 0

    def test_extracts_knowledge_and_stores_in_permanent_gen(self):
        gen_mem = _make_generational_memory(max_young_gen_size=2)

        msg1 = _make_message("The database is PostgreSQL", turn_index=1)
        msg2 = _make_message("Second turn", turn_index=2)
        msg3 = _make_message("Third turn", turn_index=3)

        gen_mem.add_message_turn(msg1)
        gen_mem.add_message_turn(msg2)
        gen_mem.add_message_turn(msg3)

        gen_mem.archive_message(msg1)

        permanent = gen_mem.get_permanent_gen()
        assert len(permanent) >= 1
        assert any("PostgreSQL" in e.content for e in permanent)

    def test_emits_archived_event(self):
        bus = EventBus()
        archived_events = []
        bus.subscribe(EventType.MESSAGE_ARCHIVED, lambda e: archived_events.append(e))
        gen_mem = _make_generational_memory(event_bus=bus, max_young_gen_size=2)

        msg1 = _make_message("The database is PostgreSQL", turn_index=1)
        msg2 = _make_message("Second", turn_index=2)
        msg3 = _make_message("Third", turn_index=3)

        gen_mem.add_message_turn(msg1)
        gen_mem.add_message_turn(msg2)
        gen_mem.add_message_turn(msg3)

        gen_mem.archive_message(msg1)

        assert len(archived_events) == 1
        assert archived_events[0].data["message"] == msg1
        assert "extracted_knowledge_entries" in archived_events[0].data

    def test_archive_message_not_in_old_gen_raises_error(self):
        gen_mem = _make_generational_memory(max_young_gen_size=3)
        msg = _make_message("Not in old gen", turn_index=1)
        gen_mem.add_message_turn(msg)

        try:
            gen_mem.archive_message(msg)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestGetters:
    def test_get_young_gen_returns_copy(self):
        gen_mem = _make_generational_memory()
        gen_mem.add_message_turn(_make_message("Hello", turn_index=1))

        young = gen_mem.get_young_gen()
        young.append(_make_message("Intruder", turn_index=99))

        assert len(gen_mem.get_young_gen()) == 1

    def test_get_old_gen_returns_copy(self):
        gen_mem = _make_generational_memory(max_young_gen_size=1)
        gen_mem.add_message_turn(_make_message("First", turn_index=1))
        gen_mem.add_message_turn(_make_message("Second", turn_index=2))

        old = gen_mem.get_old_gen()
        old.append(_make_message("Intruder", turn_index=99))

        assert len(gen_mem.get_old_gen()) == 1

    def test_get_permanent_gen_returns_active_entries(self):
        gen_mem = _make_generational_memory(max_young_gen_size=1)

        gen_mem.add_message_turn(_make_message("The database is PostgreSQL", turn_index=1))
        gen_mem.add_message_turn(_make_message("Second turn", turn_index=2))

        msg_to_archive = gen_mem.get_old_gen()[0]
        gen_mem.archive_message(msg_to_archive)

        permanent = gen_mem.get_permanent_gen()
        assert len(permanent) >= 1


class TestArchiveVerbatimFallback:
    """The guarantee: every archived turn leaves exactly one trace.

    When conservative extraction finds no knowledge in a turn, archival must not
    drop it silently. The turn's content is preserved verbatim as a RAW
    KnowledgeEntry, so removing it from active context never loses information.
    """

    def test_archive_without_extractable_knowledge_keeps_verbatim_raw_entry(self):
        gen_mem = _make_generational_memory(max_young_gen_size=2)

        # None of these match the extractor's fact/decision/preference patterns.
        msg1 = _make_message("Thanks for the help", turn_index=1)
        msg2 = _make_message("Sounds good to me", turn_index=2)
        msg3 = _make_message("Talk soon", turn_index=3)

        gen_mem.add_message_turn(msg1)
        gen_mem.add_message_turn(msg2)
        gen_mem.add_message_turn(msg3)  # promotes msg1 into old gen

        gen_mem.archive_message(msg1)

        permanent = gen_mem.get_permanent_gen()
        assert len(permanent) == 1
        raw_entry = permanent[0]
        assert raw_entry.knowledge_type == KnowledgeType.RAW
        assert raw_entry.content == "Thanks for the help"  # verbatim, not interpreted
        assert raw_entry.topic_label == "__raw__"
        assert raw_entry.message_turn == 1

    def test_archive_with_extractable_knowledge_adds_no_raw_entry(self):
        gen_mem = _make_generational_memory(max_young_gen_size=2)

        msg1 = _make_message("The database is PostgreSQL", turn_index=1)
        msg2 = _make_message("Sounds good to me", turn_index=2)
        msg3 = _make_message("Talk soon", turn_index=3)

        gen_mem.add_message_turn(msg1)
        gen_mem.add_message_turn(msg2)
        gen_mem.add_message_turn(msg3)  # promotes msg1 into old gen

        gen_mem.archive_message(msg1)

        permanent = gen_mem.get_permanent_gen()
        assert len(permanent) >= 1
        assert all(e.knowledge_type != KnowledgeType.RAW for e in permanent)
        assert any("PostgreSQL" in e.content for e in permanent)

    def test_two_unextractable_archives_both_survive_as_active(self):
        # max young size 1 so each new message promotes the previous into old gen.
        gen_mem = _make_generational_memory(max_young_gen_size=1)

        msg1 = _make_message("Thanks for the help", turn_index=1)
        msg2 = _make_message("Sounds good to me", turn_index=2)
        msg3 = _make_message("Talk soon", turn_index=3)

        gen_mem.add_message_turn(msg1)
        gen_mem.add_message_turn(msg2)  # promotes msg1 into old gen
        gen_mem.add_message_turn(msg3)  # promotes msg2 into old gen

        gen_mem.archive_message(msg1)
        gen_mem.archive_message(msg2)

        permanent = gen_mem.get_permanent_gen()
        assert len(permanent) == 2  # neither raw entry superseded the other
        assert all(e.knowledge_type == KnowledgeType.RAW for e in permanent)
        assert {e.content for e in permanent} == {"Thanks for the help", "Sounds good to me"}