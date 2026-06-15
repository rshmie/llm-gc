from llm_gc.events import EventBus, EventType
from llm_gc.models import Message
from llm_gc.models.knowledge_entry import KnowledgeType
from llm_gc.extraction import KnowledgeExtractor


def _make_message(content: str) -> Message:
    return Message(role="user", content=content, token_count=10, turn_index=1)


def _make_extractor(event_bus: EventBus | None = None) -> KnowledgeExtractor:
    bus = event_bus if event_bus is not None else EventBus()
    return KnowledgeExtractor(event_bus=bus)


class TestFactExtraction:
    def test_extracts_fact_with_is_pattern(self):
        extractor = _make_extractor()
        message = _make_message("The database is PostgreSQL")

        entries = extractor.extract_knowledge(message, message_turn=5)

        fact_entries = [e for e in entries if e.knowledge_type == KnowledgeType.FACT]
        assert len(fact_entries) >= 1
        assert any(e.content == "PostgreSQL" and "database" in e.topic_label for e in fact_entries)

    def test_extracts_fact_with_using_for_pattern(self):
        extractor = _make_extractor()
        message = _make_message("We're using Redis for caching")

        entries = extractor.extract_knowledge(message, message_turn=3)

        fact_entries = [e for e in entries if e.knowledge_type == KnowledgeType.FACT]
        assert len(fact_entries) >= 1
        assert any(e.content == "Redis" and "caching" in e.topic_label for e in fact_entries)

    def test_extracts_fact_with_runs_on_pattern(self):
        extractor = _make_extractor()
        message = _make_message("The API runs on port 8080")

        entries = extractor.extract_knowledge(message, message_turn=2)

        fact_entries = [e for e in entries if e.knowledge_type == KnowledgeType.FACT]
        assert len(fact_entries) >= 1
        assert any("8080" in e.content for e in fact_entries)

    def test_fact_entry_has_correct_message_turn(self):
        extractor = _make_extractor()
        message = _make_message("The database is PostgreSQL")

        entries = extractor.extract_knowledge(message, message_turn=7)

        assert all(e.message_turn == 7 for e in entries)


class TestDecisionExtraction:
    def test_extracts_decision_with_decided_to_pattern(self):
        extractor = _make_extractor()
        message = _make_message("We decided to use JWT for authentication")

        entries = extractor.extract_knowledge(message, message_turn=4)

        decision_entries = [e for e in entries if e.knowledge_type == KnowledgeType.DECISION]
        assert len(decision_entries) >= 1
        assert any("JWT" in e.content for e in decision_entries)

    def test_extracts_decision_with_lets_go_with_pattern(self):
        extractor = _make_extractor()
        message = _make_message("Let's go with Redis for caching")

        entries = extractor.extract_knowledge(message, message_turn=6)

        decision_entries = [e for e in entries if e.knowledge_type == KnowledgeType.DECISION]
        assert len(decision_entries) >= 1
        assert any("Redis" in e.content for e in decision_entries)

    def test_extracts_decision_with_chose_over_pattern(self):
        extractor = _make_extractor()
        message = _make_message("We chose microservices over monolith for architecture")

        entries = extractor.extract_knowledge(message, message_turn=9)

        decision_entries = [e for e in entries if e.knowledge_type == KnowledgeType.DECISION]
        assert len(decision_entries) >= 1
        assert any("microservices" in e.content for e in decision_entries)

    def test_decision_without_topic_falls_back_to_general(self):
        extractor = _make_extractor()
        message = _make_message("We decided to use microservices")

        entries = extractor.extract_knowledge(message, message_turn=4)

        decision_entries = [e for e in entries if e.knowledge_type == KnowledgeType.DECISION]
        assert len(decision_entries) >= 1
        assert any(e.topic_label == "general" for e in decision_entries)


class TestPreferenceExtraction:
    def test_extracts_preference_with_prefer_pattern(self):
        extractor = _make_extractor()
        message = _make_message("I prefer tabs for indentation")

        entries = extractor.extract_knowledge(message, message_turn=2)

        pref_entries = [e for e in entries if e.knowledge_type == KnowledgeType.PREFERENCE]
        assert len(pref_entries) >= 1
        assert any("tabs" in e.content for e in pref_entries)

    def test_extracts_preference_with_rather_use_pattern(self):
        extractor = _make_extractor()
        message = _make_message("I'd rather use dataclasses for models")

        entries = extractor.extract_knowledge(message, message_turn=3)

        pref_entries = [e for e in entries if e.knowledge_type == KnowledgeType.PREFERENCE]
        assert len(pref_entries) >= 1
        assert any("dataclasses" in e.content for e in pref_entries)

    def test_preference_without_topic_falls_back_to_general(self):
        extractor = _make_extractor()
        message = _make_message("We prefer simplicity")

        entries = extractor.extract_knowledge(message, message_turn=1)

        pref_entries = [e for e in entries if e.knowledge_type == KnowledgeType.PREFERENCE]
        assert len(pref_entries) >= 1
        assert any(e.topic_label == "general" for e in pref_entries)


class TestMultipleEntries:
    def test_extracts_multiple_types_from_one_message(self):
        extractor = _make_extractor()
        message = _make_message("The database is PostgreSQL. We decided to use JWT for auth")

        entries = extractor.extract_knowledge(message, message_turn=5)

        types_found = {e.knowledge_type for e in entries}
        assert KnowledgeType.FACT in types_found
        assert KnowledgeType.DECISION in types_found


class TestNoMatch:
    def test_returns_empty_list_for_unrecognized_text(self):
        extractor = _make_extractor()
        message = _make_message("Thanks for the help yesterday")

        entries = extractor.extract_knowledge(message, message_turn=1)

        assert entries == []

    def test_returns_empty_list_for_empty_content(self):
        extractor = _make_extractor()
        message = _make_message("")

        entries = extractor.extract_knowledge(message, message_turn=1)

        assert entries == []


class TestEventEmission:
    def test_emits_event_when_entries_extracted(self):
        bus = EventBus()
        received_events = []
        bus.subscribe(EventType.KNOWLEDGE_ENTRIES_EXTRACTED, lambda e: received_events.append(e))
        extractor = _make_extractor(event_bus=bus)

        message = _make_message("The database is PostgreSQL")
        extractor.extract_knowledge(message, message_turn=5)

        assert len(received_events) == 1
        assert "entries" in received_events[0].data
        assert received_events[0].data["message_turn"] == 5

    def test_does_not_emit_event_when_no_entries(self):
        bus = EventBus()
        received_events = []
        bus.subscribe(EventType.KNOWLEDGE_ENTRIES_EXTRACTED, lambda e: received_events.append(e))
        extractor = _make_extractor(event_bus=bus)

        message = _make_message("Thanks for the help yesterday")
        extractor.extract_knowledge(message, message_turn=1)

        assert len(received_events) == 0
