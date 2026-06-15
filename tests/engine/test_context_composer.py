from unittest.mock import MagicMock

from llm_gc.engine.compaction import NoOpCompactor
from llm_gc.engine.context_composer import ContextComposer
from llm_gc.engine.sweep import SweepResult, SweepClassification, SweepEntry
from llm_gc.events import EventBus, EventType
from llm_gc.models import Message
from llm_gc.utils import count_tokens


def _make_message(content: str, token_count: int = 10, role: str = "user", turn_index: int = 0) -> Message:
    return Message(role=role, content=content, token_count=token_count, turn_index=turn_index)


def _make_sweep_entry(message: Message, classification: SweepClassification) -> SweepEntry:
    return SweepEntry(message=message, classification=classification, relevance_score=0.5, turn_index=message.turn_index)


def _make_sweep_result(entries: list[SweepEntry]) -> SweepResult:
    counts = {SweepClassification.KEEP: 0, SweepClassification.COMPACT: 0, SweepClassification.ARCHIVE: 0}
    total_keep, total_compact, total_archive = 0, 0, 0
    for entry in entries:
        counts[entry.classification] += 1
        if entry.classification == SweepClassification.KEEP:
            total_keep += entry.message.token_count
        elif entry.classification == SweepClassification.COMPACT:
            total_compact += entry.message.token_count
        elif entry.classification == SweepClassification.ARCHIVE:
            total_archive += entry.message.token_count
    return SweepResult(
        sweep_entries=entries,
        classification_counts=counts,
        processing_time_ms=1.0,
        total_keep_tokens=total_keep,
        total_compact_tokens=total_compact,
        total_archive_tokens=total_archive,
    )


def _make_composer(event_bus: EventBus | None = None, generational_memory=None) -> ContextComposer:
    bus = event_bus if event_bus is not None else EventBus()
    gen_mem = generational_memory if generational_memory is not None else MagicMock()
    return ContextComposer(event_bus=bus, generational_memory=gen_mem, compactor=NoOpCompactor(event_bus=bus))


def _expected_noop_summary_content(messages: list[Message]) -> str:
    """Build the exact content NoOpCompactor produces for a given run.

    Mirrors NoOpCompactor.compact: marker followed by the verbatim
    space-joined content of the originals.
    """
    first_turn = messages[0].turn_index
    last_turn = messages[-1].turn_index
    marker = f"[Compacted (compaction_strategy=noop) turns {first_turn}-{last_turn}]"
    verbatim = " ".join(m.content for m in messages)
    return f"{marker} {verbatim}"


class TestComposeAllKeep:
    def test_all_keep_messages_pass_through_unchanged(self):
        m1 = _make_message("Hello", turn_index=0)
        m2 = _make_message("How are you?", turn_index=1, role="assistant")
        m3 = _make_message("Fine thanks", turn_index=2)
        entries = [
            _make_sweep_entry(m1, SweepClassification.KEEP),
            _make_sweep_entry(m2, SweepClassification.KEEP),
            _make_sweep_entry(m3, SweepClassification.KEEP),
        ]
        composer = _make_composer()

        result = composer.compose(_make_sweep_result(entries))

        assert result == [m1, m2, m3]

    def test_all_keep_preserves_original_message_attributes(self):
        msg = _make_message("Important content", token_count=42, role="system", turn_index=5)
        entries = [_make_sweep_entry(msg, SweepClassification.KEEP)]
        composer = _make_composer()

        result = composer.compose(_make_sweep_result(entries))

        assert result[0].role == "system"
        assert result[0].token_count == 42
        assert result[0].turn_index == 5


class TestComposeAllCompact:
    def test_all_compact_produces_single_synthetic_message(self):
        m1 = _make_message("First message", token_count=10, turn_index=0)
        m2 = _make_message("Second message", token_count=15, turn_index=1, role="assistant")
        m3 = _make_message("Third message", token_count=20, turn_index=2)
        entries = [
            _make_sweep_entry(m1, SweepClassification.COMPACT),
            _make_sweep_entry(m2, SweepClassification.COMPACT),
            _make_sweep_entry(m3, SweepClassification.COMPACT),
        ]
        composer = _make_composer()

        result = composer.compose(_make_sweep_result(entries))

        assert len(result) == 1

    def test_synthetic_message_has_assistant_role(self):
        m1 = _make_message("Hello", turn_index=0)
        m2 = _make_message("World", turn_index=1)
        entries = [
            _make_sweep_entry(m1, SweepClassification.COMPACT),
            _make_sweep_entry(m2, SweepClassification.COMPACT),
        ]
        composer = _make_composer()

        result = composer.compose(_make_sweep_result(entries))

        assert result[0].role == "assistant"

    def test_synthetic_message_contains_summary_format(self):
        m1 = _make_message("Let's use PostgreSQL", token_count=10, turn_index=3)
        m2 = _make_message("Great choice for our needs", token_count=12, turn_index=4)
        entries = [
            _make_sweep_entry(m1, SweepClassification.COMPACT),
            _make_sweep_entry(m2, SweepClassification.COMPACT),
        ]
        composer = _make_composer()

        result = composer.compose(_make_sweep_result(entries))

        assert result[0].content == _expected_noop_summary_content([m1, m2])

    def test_synthetic_message_token_count_matches_compactor_output(self):
        m1 = _make_message("First", token_count=10, turn_index=0)
        m2 = _make_message("Second", token_count=25, turn_index=1)
        m3 = _make_message("Third", token_count=15, turn_index=2)
        entries = [
            _make_sweep_entry(m1, SweepClassification.COMPACT),
            _make_sweep_entry(m2, SweepClassification.COMPACT),
            _make_sweep_entry(m3, SweepClassification.COMPACT),
        ]
        composer = _make_composer()

        result = composer.compose(_make_sweep_result(entries))

        # NoOpCompactor recomputes token count over the synthesized full
        # content (marker + verbatim), not as a sum of originals.
        expected_content = _expected_noop_summary_content([m1, m2, m3])
        assert result[0].token_count == count_tokens(expected_content)

    def test_synthetic_message_turn_index_is_first_in_run(self):
        m1 = _make_message("First", turn_index=5)
        m2 = _make_message("Second", turn_index=6)
        entries = [
            _make_sweep_entry(m1, SweepClassification.COMPACT),
            _make_sweep_entry(m2, SweepClassification.COMPACT),
        ]
        composer = _make_composer()

        result = composer.compose(_make_sweep_result(entries))

        assert result[0].turn_index == 5


class TestComposeSingleCompact:
    def test_single_compact_still_produces_synthetic_message(self):
        msg = _make_message("Only one compacted", token_count=20, turn_index=3)
        entries = [_make_sweep_entry(msg, SweepClassification.COMPACT)]
        composer = _make_composer()

        result = composer.compose(_make_sweep_result(entries))

        expected_content = _expected_noop_summary_content([msg])
        assert len(result) == 1
        assert result[0].content == expected_content
        assert result[0].token_count == count_tokens(expected_content)


class TestComposeMixedKeepAndCompact:
    def test_compact_run_between_keeps_preserves_order(self):
        m1 = _make_message("Keep first", token_count=10, turn_index=0)
        m2 = _make_message("Compact this", token_count=10, turn_index=1)
        m3 = _make_message("And this", token_count=10, turn_index=2)
        m4 = _make_message("Keep last", token_count=10, turn_index=3)
        entries = [
            _make_sweep_entry(m1, SweepClassification.KEEP),
            _make_sweep_entry(m2, SweepClassification.COMPACT),
            _make_sweep_entry(m3, SweepClassification.COMPACT),
            _make_sweep_entry(m4, SweepClassification.KEEP),
        ]
        composer = _make_composer()

        result = composer.compose(_make_sweep_result(entries))

        assert len(result) == 3
        assert result[0] == m1
        assert result[1].content == _expected_noop_summary_content([m2, m3])
        assert result[2] == m4

    def test_multiple_compact_runs_separated_by_keep(self):
        m1 = _make_message("Compact A1", token_count=10, turn_index=0)
        m2 = _make_message("Compact A2", token_count=10, turn_index=1)
        m3 = _make_message("Keep middle", token_count=10, turn_index=2)
        m4 = _make_message("Compact B1", token_count=10, turn_index=3)
        m5 = _make_message("Compact B2", token_count=10, turn_index=4)
        entries = [
            _make_sweep_entry(m1, SweepClassification.COMPACT),
            _make_sweep_entry(m2, SweepClassification.COMPACT),
            _make_sweep_entry(m3, SweepClassification.KEEP),
            _make_sweep_entry(m4, SweepClassification.COMPACT),
            _make_sweep_entry(m5, SweepClassification.COMPACT),
        ]
        composer = _make_composer()

        result = composer.compose(_make_sweep_result(entries))

        assert len(result) == 3
        assert result[0].content == _expected_noop_summary_content([m1, m2])
        assert result[1] == m3
        assert result[2].content == _expected_noop_summary_content([m4, m5])

    def test_compact_run_at_end_is_flushed(self):
        m1 = _make_message("Keep this", turn_index=0)
        m2 = _make_message("Compact trailing", turn_index=1)
        m3 = _make_message("Also trailing", turn_index=2)
        entries = [
            _make_sweep_entry(m1, SweepClassification.KEEP),
            _make_sweep_entry(m2, SweepClassification.COMPACT),
            _make_sweep_entry(m3, SweepClassification.COMPACT),
        ]
        composer = _make_composer()

        result = composer.compose(_make_sweep_result(entries))

        assert len(result) == 2
        assert result[0] == m1
        assert result[1].content == _expected_noop_summary_content([m2, m3])


class TestComposeArchive:
    def test_archive_messages_not_in_output(self):
        m1 = _make_message("Keep this", turn_index=0)
        m2 = _make_message("Archive this", turn_index=1)
        m3 = _make_message("Keep this too", turn_index=2)
        entries = [
            _make_sweep_entry(m1, SweepClassification.KEEP),
            _make_sweep_entry(m2, SweepClassification.ARCHIVE),
            _make_sweep_entry(m3, SweepClassification.KEEP),
        ]
        composer = _make_composer()

        result = composer.compose(_make_sweep_result(entries))

        assert len(result) == 2
        assert result[0] == m1
        assert result[1] == m3

    def test_archive_calls_generational_memory(self):
        mock_gen_mem = MagicMock()
        m1 = _make_message("Archive me", turn_index=0)
        entries = [_make_sweep_entry(m1, SweepClassification.ARCHIVE)]
        composer = _make_composer(generational_memory=mock_gen_mem)

        composer.compose(_make_sweep_result(entries))

        mock_gen_mem.archive_message.assert_called_once_with(m1)

    def test_multiple_archives_each_call_generational_memory(self):
        mock_gen_mem = MagicMock()
        m1 = _make_message("Archive 1", turn_index=0)
        m2 = _make_message("Archive 2", turn_index=1)
        m3 = _make_message("Archive 3", turn_index=2)
        entries = [
            _make_sweep_entry(m1, SweepClassification.ARCHIVE),
            _make_sweep_entry(m2, SweepClassification.ARCHIVE),
            _make_sweep_entry(m3, SweepClassification.ARCHIVE),
        ]
        composer = _make_composer(generational_memory=mock_gen_mem)

        composer.compose(_make_sweep_result(entries))

        assert mock_gen_mem.archive_message.call_count == 3

    def test_archive_between_compact_runs_flushes_correctly(self):
        m1 = _make_message("Compact 1", token_count=10, turn_index=0)
        m2 = _make_message("Compact 2", token_count=10, turn_index=1)
        m3 = _make_message("Archive this", token_count=10, turn_index=2)
        m4 = _make_message("Compact 3", token_count=10, turn_index=3)
        m5 = _make_message("Compact 4", token_count=10, turn_index=4)
        entries = [
            _make_sweep_entry(m1, SweepClassification.COMPACT),
            _make_sweep_entry(m2, SweepClassification.COMPACT),
            _make_sweep_entry(m3, SweepClassification.ARCHIVE),
            _make_sweep_entry(m4, SweepClassification.COMPACT),
            _make_sweep_entry(m5, SweepClassification.COMPACT),
        ]
        composer = _make_composer()

        result = composer.compose(_make_sweep_result(entries))

        assert len(result) == 2
        assert result[0].content == _expected_noop_summary_content([m1, m2])
        assert result[1].content == _expected_noop_summary_content([m4, m5])


class TestComposeEventEmission:
    def test_emits_context_composed_event(self):
        received_events = []
        bus = EventBus()
        bus.subscribe(EventType.CONTEXT_COMPOSED, lambda e: received_events.append(e))
        m1 = _make_message("Keep", token_count=10, turn_index=0)
        entries = [_make_sweep_entry(m1, SweepClassification.KEEP)]
        composer = _make_composer(event_bus=bus)

        composer.compose(_make_sweep_result(entries))

        assert len(received_events) == 1

    def test_event_data_has_correct_kept_count(self):
        received_events = []
        bus = EventBus()
        bus.subscribe(EventType.CONTEXT_COMPOSED, lambda e: received_events.append(e))
        entries = [
            _make_sweep_entry(_make_message("K1", turn_index=0), SweepClassification.KEEP),
            _make_sweep_entry(_make_message("K2", turn_index=1), SweepClassification.KEEP),
            _make_sweep_entry(_make_message("C1", turn_index=2), SweepClassification.COMPACT),
        ]
        composer = _make_composer(event_bus=bus)

        composer.compose(_make_sweep_result(entries))

        assert received_events[0].data["messages_kept"] == 2

    def test_event_data_has_correct_compacted_count(self):
        received_events = []
        bus = EventBus()
        bus.subscribe(EventType.CONTEXT_COMPOSED, lambda e: received_events.append(e))
        entries = [
            _make_sweep_entry(_make_message("C1", turn_index=0), SweepClassification.COMPACT),
            _make_sweep_entry(_make_message("C2", turn_index=1), SweepClassification.COMPACT),
            _make_sweep_entry(_make_message("C3", turn_index=2), SweepClassification.COMPACT),
        ]
        composer = _make_composer(event_bus=bus)

        composer.compose(_make_sweep_result(entries))

        assert received_events[0].data["messages_compacted"] == 3

    def test_event_data_has_correct_archived_count(self):
        received_events = []
        bus = EventBus()
        bus.subscribe(EventType.CONTEXT_COMPOSED, lambda e: received_events.append(e))
        entries = [
            _make_sweep_entry(_make_message("A1", turn_index=0), SweepClassification.ARCHIVE),
            _make_sweep_entry(_make_message("A2", turn_index=1), SweepClassification.ARCHIVE),
        ]
        composer = _make_composer(event_bus=bus)

        composer.compose(_make_sweep_result(entries))

        assert received_events[0].data["messages_archived"] == 2

    def test_event_data_has_correct_compact_runs_count(self):
        received_events = []
        bus = EventBus()
        bus.subscribe(EventType.CONTEXT_COMPOSED, lambda e: received_events.append(e))
        entries = [
            _make_sweep_entry(_make_message("C1", turn_index=0), SweepClassification.COMPACT),
            _make_sweep_entry(_make_message("C2", turn_index=1), SweepClassification.COMPACT),
            _make_sweep_entry(_make_message("K1", turn_index=2), SweepClassification.KEEP),
            _make_sweep_entry(_make_message("C3", turn_index=3), SweepClassification.COMPACT),
        ]
        composer = _make_composer(event_bus=bus)

        composer.compose(_make_sweep_result(entries))

        assert received_events[0].data["compact_runs_created"] == 2

    def test_event_data_has_correct_token_counts(self):
        received_events = []
        bus = EventBus()
        bus.subscribe(EventType.CONTEXT_COMPOSED, lambda e: received_events.append(e))
        keep_msg = _make_message("Keep", token_count=20, turn_index=0)
        compact_msg = _make_message("Compact", token_count=30, turn_index=1)
        archive_msg = _make_message("Archive", token_count=50, turn_index=2)
        entries = [
            _make_sweep_entry(keep_msg, SweepClassification.KEEP),
            _make_sweep_entry(compact_msg, SweepClassification.COMPACT),
            _make_sweep_entry(archive_msg, SweepClassification.ARCHIVE),
        ]
        composer = _make_composer(event_bus=bus)

        composer.compose(_make_sweep_result(entries))

        # tokens_before is the sum of every classification's tokens fed into the sweeper-and-composition.
        assert received_events[0].data["tokens_before"] == 100
        # tokens_after is the sum of token counts on the *final* messages emitted by the
        # composer: KEEP messages keep their original token_count, and the COMPACT run
        # produces a NoOp summary whose token_count is recomputed over the synthesized
        # content (marker + verbatim).
        expected_summary_content = _expected_noop_summary_content([compact_msg])
        expected_tokens_after = keep_msg.token_count + count_tokens(expected_summary_content)
        assert received_events[0].data["tokens_after"] == expected_tokens_after
