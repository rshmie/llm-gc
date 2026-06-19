from llm_gc.events import EventBus, EventType
from llm_gc.models import KnowledgeEntry
from llm_gc.models.knowledge_entry import KnowledgeType, KnowledgeStatus
from llm_gc.engine.generations.permanent_generation import PermanentGeneration

def _make_entry(content: str, topic_label: str, message_turn: int = 1,
                knowledge_type: KnowledgeType = KnowledgeType.FACT) -> KnowledgeEntry:
    return KnowledgeEntry(
        message_turn=message_turn,
        content=content,
        topic_label=topic_label,
        knowledge_type=knowledge_type,
    )


def _make_permanent_gen(event_bus: EventBus | None = None) -> PermanentGeneration:
    bus = event_bus if event_bus is not None else EventBus()
    return PermanentGeneration(event_bus=bus)


class TestAddKnowledgeEntry:
    def test_add_single_entry(self):
        perm_gen = _make_permanent_gen()
        entry = _make_entry("Database is MySQL", topic_label="database", message_turn=8)

        perm_gen.add_knowledge_entry(entry)

        entries = perm_gen.get_by_topic("database")
        assert len(entries) == 1
        assert entries[0].content == "Database is MySQL"
        assert entries[0].knowledge_status == KnowledgeStatus.ACTIVE

    def test_add_multiple_entries_different_topics(self):
        perm_gen = _make_permanent_gen()
        db_entry = _make_entry("Database is MySQL", topic_label="database")
        auth_entry = _make_entry("Auth uses JWT", topic_label="auth")

        perm_gen.add_knowledge_entry(db_entry)
        perm_gen.add_knowledge_entry(auth_entry)

        assert len(perm_gen.get_by_topic("database")) == 1
        assert len(perm_gen.get_by_topic("auth")) == 1

    def test_add_emits_knowledge_entry_added_event(self):
        bus = EventBus()
        received_events = []
        bus.subscribe(EventType.KNOWLEDGE_ENTRY_ADDED, lambda e: received_events.append(e))
        perm_gen = _make_permanent_gen(event_bus=bus)

        entry = _make_entry("Database is MySQL", topic_label="database")
        perm_gen.add_knowledge_entry(entry)

        assert len(received_events) == 1
        assert received_events[0].data["knowledge_entry"] == entry


class TestContradictionDetection:
    def test_new_entry_supersedes_existing_active_entry_same_topic(self):
        perm_gen = _make_permanent_gen()
        old_entry = _make_entry("Database is PostgreSQL", topic_label="database", message_turn=5)
        new_entry = _make_entry("Database is MySQL", topic_label="database", message_turn=8)

        perm_gen.add_knowledge_entry(old_entry)
        perm_gen.add_knowledge_entry(new_entry)

        entries = perm_gen.get_by_topic("database")
        assert len(entries) == 2
        assert entries[0].knowledge_status == KnowledgeStatus.SUPERSEDED
        assert entries[0].content == "Database is PostgreSQL"
        assert entries[1].knowledge_status == KnowledgeStatus.ACTIVE
        assert entries[1].content == "Database is MySQL"

    def test_supersession_does_not_affect_other_topics(self):
        perm_gen = _make_permanent_gen()
        db_entry = _make_entry("Database is PostgreSQL", topic_label="database", message_turn=5)
        auth_entry = _make_entry("Auth uses JWT", topic_label="auth", message_turn=6)
        new_db_entry = _make_entry("Database is MySQL", topic_label="database", message_turn=8)

        perm_gen.add_knowledge_entry(db_entry)
        perm_gen.add_knowledge_entry(auth_entry)
        perm_gen.add_knowledge_entry(new_db_entry)

        auth_entries = perm_gen.get_by_topic("auth")
        assert len(auth_entries) == 1
        assert auth_entries[0].knowledge_status == KnowledgeStatus.ACTIVE

    def test_already_superseded_entry_is_not_superseded_again(self):
        perm_gen = _make_permanent_gen()
        entry_v1 = _make_entry("Database is PostgreSQL", topic_label="database", message_turn=5)
        entry_v2 = _make_entry("Database is MySQL", topic_label="database", message_turn=8)
        entry_v3 = _make_entry("Database is SQLite", topic_label="database", message_turn=15)

        perm_gen.add_knowledge_entry(entry_v1)
        perm_gen.add_knowledge_entry(entry_v2)
        perm_gen.add_knowledge_entry(entry_v3)

        entries = perm_gen.get_by_topic("database")
        assert len(entries) == 3
        assert entries[0].knowledge_status == KnowledgeStatus.SUPERSEDED
        assert entries[1].knowledge_status == KnowledgeStatus.SUPERSEDED
        assert entries[2].knowledge_status == KnowledgeStatus.ACTIVE

    def test_contradiction_emits_superseded_event(self):
        bus = EventBus()
        superseded_events = []
        bus.subscribe(EventType.KNOWLEDGE_ENTRY_SUPERSEDED, lambda e: superseded_events.append(e))
        perm_gen = _make_permanent_gen(event_bus=bus)

        old_entry = _make_entry("Database is PostgreSQL", topic_label="database", message_turn=5)
        new_entry = _make_entry("Database is MySQL", topic_label="database", message_turn=8)

        perm_gen.add_knowledge_entry(old_entry)
        perm_gen.add_knowledge_entry(new_entry)

        assert len(superseded_events) == 1
        assert superseded_events[0].data["new_knowledge_entry"] == new_entry
        assert old_entry in superseded_events[0].data["superseded_knowledge_entries"]

    def test_no_contradiction_does_not_emit_superseded_event(self):
        bus = EventBus()
        superseded_events = []
        bus.subscribe(EventType.KNOWLEDGE_ENTRY_SUPERSEDED, lambda e: superseded_events.append(e))
        perm_gen = _make_permanent_gen(event_bus=bus)

        perm_gen.add_knowledge_entry(_make_entry("Database is MySQL", topic_label="database"))
        perm_gen.add_knowledge_entry(_make_entry("Auth uses JWT", topic_label="auth"))

        assert len(superseded_events) == 0

    def test_raw_entries_under_same_label_do_not_supersede_each_other(self):
        # Verbatim fallbacks all share the "__raw__" sentinel label. They are not
        # assertions and must never supersede one another, or the no-information-loss
        # guarantee would leak (all but the latest would be dropped from active view).
        perm_gen = _make_permanent_gen()
        raw1 = _make_entry("Thanks for the help", topic_label="__raw__",
                           message_turn=1, knowledge_type=KnowledgeType.RAW)
        raw2 = _make_entry("Sounds good to me", topic_label="__raw__",
                           message_turn=2, knowledge_type=KnowledgeType.RAW)

        perm_gen.add_knowledge_entry(raw1)
        perm_gen.add_knowledge_entry(raw2)

        active = perm_gen.get_all_active_entries()
        assert len(active) == 2
        assert all(e.knowledge_status == KnowledgeStatus.ACTIVE for e in active)


class TestGetByTopic:
    def test_returns_empty_list_for_unknown_topic(self):
        perm_gen = _make_permanent_gen()
        assert perm_gen.get_by_topic("nonexistent") == []

    def test_returns_all_entries_including_superseded(self):
        perm_gen = _make_permanent_gen()
        perm_gen.add_knowledge_entry(_make_entry("Database is PostgreSQL", topic_label="database", message_turn=5))
        perm_gen.add_knowledge_entry(_make_entry("Database is MySQL", topic_label="database", message_turn=8))

        entries = perm_gen.get_by_topic("database")
        assert len(entries) == 2


class TestGetAllActiveEntries:
    def test_returns_only_active_entries_across_topics(self):
        perm_gen = _make_permanent_gen()
        perm_gen.add_knowledge_entry(_make_entry("Database is PostgreSQL", topic_label="database", message_turn=5))
        perm_gen.add_knowledge_entry(_make_entry("Database is MySQL", topic_label="database", message_turn=8))
        perm_gen.add_knowledge_entry(_make_entry("Auth uses JWT", topic_label="auth", message_turn=6))

        active = perm_gen.get_all_active_entries()
        assert len(active) == 2
        contents = [e.content for e in active]
        assert "Database is MySQL" in contents
        assert "Auth uses JWT" in contents
        assert "Database is PostgreSQL" not in contents

    def test_returns_empty_list_when_no_entries(self):
        perm_gen = _make_permanent_gen()
        assert perm_gen.get_all_active_entries() == []