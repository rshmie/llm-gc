from llm_gc.engine.generations import format_memory_block
from llm_gc.engine.generations.memory_block import MEMORY_BLOCK_HEADER, MEMORY_FACT_MAX_CHARS
from llm_gc.models import KnowledgeEntry
from llm_gc.models.knowledge_entry import KnowledgeType
from llm_gc.utils import count_tokens


def _entry(topic: str, content: str, turn: int = 1, kind: KnowledgeType = KnowledgeType.FACT) -> KnowledgeEntry:
    return KnowledgeEntry(message_turn=turn, content=content, topic_label=topic, knowledge_type=kind)


GENEROUS = 10_000


class TestWhatTheBlockSays:
    def test_it_announces_itself(self):
        """Announced, not smuggled in. Presenting distilled facts as if they were
        original conversation is the dishonesty this project is built against - the
        compacted-summary marker exists for the identical reason."""
        block = format_memory_block([_entry("database", "postgres")], GENEROUS)

        assert block is not None
        assert block.content.startswith(MEMORY_BLOCK_HEADER)

    def test_each_fact_carries_its_origin_turn(self):
        """So a reader - human or model - can tell how far back the claim comes
        from, and so the dashboard and the prompt agree about provenance."""
        block = format_memory_block([_entry("database", "postgres", turn=7)], GENEROUS)

        assert block is not None
        assert "(turn 7)" in block.content

    def test_a_typed_fact_is_labelled_with_its_topic(self):
        block = format_memory_block([_entry("database", "postgres", turn=1)], GENEROUS)

        assert block is not None
        assert "database: postgres" in block.content

    def test_a_raw_entry_is_labelled_an_excerpt_not_a_fact(self):
        """Extraction found nothing in that turn. Dressing it up as a parsed fact
        would overstate what is known about it."""
        block = format_memory_block(
            [_entry("__raw__", "We walked through the numbers", kind=KnowledgeType.RAW)], GENEROUS
        )

        assert block is not None
        assert "excerpt:" in block.content
        assert "__raw__" not in block.content


class TestPositionAndShape:
    def test_it_sorts_before_every_real_turn(self):
        """Index -1, not 0: turn 0 is a real turn that may still be present, and the
        block must not compete with it for position."""
        block = format_memory_block([_entry("database", "postgres", turn=5)], GENEROUS)

        assert block is not None
        assert block.turn_index == -1

    def test_its_token_count_matches_its_content(self):
        """The assembler sums token_count to report context size; a wrong count here
        would under-report the prompt the model actually receives."""
        block = format_memory_block([_entry("database", "postgres")], GENEROUS)

        assert block is not None
        assert block.token_count == count_tokens(block.content)

    def test_facts_are_shown_in_conversation_order(self):
        """Two orderings doing different jobs: relevance decides *what* survives the
        budget, chronology makes what survives readable."""
        ranked_newest_first = [
            _entry("queue", "rabbitmq", turn=30),
            _entry("database", "postgres", turn=2),
            _entry("cache", "redis", turn=11),
        ]

        block = format_memory_block(ranked_newest_first, GENEROUS)

        assert block is not None
        assert block.content.index("turn 2") < block.content.index("turn 11") < block.content.index("turn 30")


class TestTheTokenBudget:
    def test_the_whole_block_stays_within_budget(self):
        entries = [_entry(f"topic{i}", f"value number {i} with some trailing prose", turn=i) for i in range(30)]

        block = format_memory_block(entries, 120)

        assert block is not None
        assert block.token_count <= 120

    def test_the_least_relevant_entries_are_the_ones_dropped(self):
        """The budget is spent from the front of the ranked list, so what gets cut
        is what the retriever ranked last - not whatever happened to be long."""
        most_relevant = _entry("database", "postgres", turn=1)
        padding = [_entry(f"topic{i}", "x" * 120, turn=i + 2) for i in range(10)]

        block = format_memory_block([most_relevant] + padding, 80)

        assert block is not None
        assert "database: postgres" in block.content

    def test_it_stops_at_the_first_entry_that_does_not_fit(self):
        """Skipping an oversized entry and taking the next would silently reorder by
        size, discarding the relevance ordering."""
        first = _entry("database", "postgres", turn=1)
        too_big = _entry("verbose", "word " * 100, turn=2)
        would_fit = _entry("cache", "redis", turn=3)

        block = format_memory_block([first, too_big, would_fit], 60)

        assert block is not None
        assert "cache: redis" not in block.content

    def test_no_room_for_the_header_yields_nothing(self):
        """Emitting a header alone would spend tokens announcing memory and show
        none."""
        assert format_memory_block([_entry("database", "postgres")], 3) is None

    def test_a_zero_budget_yields_nothing(self):
        assert format_memory_block([_entry("database", "postgres")], 0) is None

    def test_no_entries_yields_nothing(self):
        """None rather than an empty message, so the caller appends nothing."""
        assert format_memory_block([], GENEROUS) is None


class TestPerFactTruncation:
    def test_a_long_fact_is_clipped(self):
        """One long entry - typically a RAW whole-turn excerpt - must not consume
        the whole block on its own."""
        block = format_memory_block([_entry("notes", "y" * 600, turn=1)], GENEROUS)

        assert block is not None
        assert len(block.content) < 600

    def test_truncation_is_visible(self):
        """A clipped fact reads as clipped. Silent truncation would hand the model a
        sentence that stops mid-thought with nothing saying why."""
        block = format_memory_block([_entry("notes", "y" * 600, turn=1)], GENEROUS)

        assert block is not None
        assert "..." in block.content

    def test_a_short_fact_is_untouched(self):
        content = "postgres, partitioned by day"
        assert len(content) < MEMORY_FACT_MAX_CHARS
        block = format_memory_block([_entry("database", content, turn=1)], GENEROUS)

        assert block is not None
        assert content in block.content
