"""The front: what the model reads for each span, at each level."""

import pytest

from llm_gc.engine.generations.front import (
    ARCHIVED_PREFIX,
    FOLDED_PREFIX,
    FRONT_HEADER,
    QUOTE_PREFIX,
    SUMMARISED_PREFIX,
    Annotations,
    KnowledgeIndex,
    level_costs,
    render_front,
    span_text,
)
from llm_gc.models import KnowledgeEntry, Message
from llm_gc.models.generation_snapshot import DetailLevel, Span, file_knowledge
from llm_gc.models.knowledge_entry import KnowledgeType

KEPT, COMPACTED, ARCHIVED, OUT = DetailLevel.KEPT, DetailLevel.COMPACTED, DetailLevel.ARCHIVED, DetailLevel.OUT_OF_VIEW


def _span(first: int, level: DetailLevel, opening: str = "We will use postgres.") -> Span:
    turns = (Message(role="user", content=f"{opening} It handles the ingest load.", token_count=12, turn_index=first),
             Message(role="assistant", content="Agreed; postgres with two replicas.", token_count=8,
                     turn_index=first + 1))
    summary = Message(role="assistant", content="[marker] Postgres was chosen for ingest.", token_count=9,
                      turn_index=first)
    return Span(turns=turns, level=level, summary=summary, summary_text="Postgres was chosen for ingest.",
                index_line=opening)


def _fact(turn: int, content: str, topic: str = "database") -> KnowledgeEntry:
    return KnowledgeEntry(message_turn=turn, content=content, topic_label=topic, knowledge_type=KnowledgeType.DECISION)


class TestLevelsAreNested:
    """Each level shows everything the level below it shows, so showing a span with
    more detail never takes anything away."""

    annotations = Annotations(facts=(_fact(0, "postgres"),))

    def test_a_summarised_span_shows_its_archived_line_then_its_summary(self):
        archived = span_text(_span(0, ARCHIVED), self.annotations)
        summarised = span_text(_span(0, COMPACTED), self.annotations)

        assert archived == f"{ARCHIVED_PREFIX} (turns 0–1) We will use postgres. · database: postgres"
        assert summarised == (f"{SUMMARISED_PREFIX} (turns 0–1) We will use postgres. · database: postgres\n"
                              "Postgres was chosen for ingest.")

    def test_a_kept_span_shows_every_word_of_its_turns(self):
        span = _span(0, KEPT)
        kept = span_text(span, self.annotations)

        assert kept.startswith(f"{QUOTE_PREFIX} (turn 0, user) ")
        assert all(turn.content in kept for turn in span.turns)

    def test_a_quoted_tool_result_is_named_as_the_tools(self):
        result = Message(role="user", content="[tool result] 42 passed", token_count=6, turn_index=4,
                         raw_content=[{"type": "tool_result", "tool_use_id": "t1", "content": "42 passed"}])
        kept = span_text(Span(turns=(result,), level=KEPT), Annotations())

        assert kept == f"{QUOTE_PREFIX} (turn 4, tool results) [tool result] 42 passed"

    def test_each_richer_level_costs_more_so_the_planner_sees_what_it_buys(self):
        kept, compacted, archived = level_costs(_span(0, ARCHIVED), self.annotations)
        assert compacted is not None and archived < compacted

    def test_a_folded_span_has_no_line_of_its_own(self):
        with pytest.raises(ValueError):
            span_text(_span(0, OUT), self.annotations)


class TestAReplacedDecisionIsNamedWhereverItIsShown:
    """An archived span's opening is the user's own sentence, and it can state the
    decision the conversation later replaced. The model must not read it as current."""

    @staticmethod
    def _knowledge():
        knowledge, _ = file_knowledge((), (_fact(0, "postgres"), _fact(6, "mysql")))
        return knowledge

    @pytest.mark.parametrize("level", [KEPT, COMPACTED, ARCHIVED])
    def test_every_level_shown_carries_the_note(self, level):
        text = span_text(_span(0, level), KnowledgeIndex(self._knowledge()).for_span(_span(0, level)))
        assert text.endswith("↳ Replaced later (turn 6): database: mysql, not postgres.")

    def test_the_span_that_replaced_it_carries_no_note(self):
        replacing = _span(6, ARCHIVED, opening="Switch to mysql.")
        assert "Replaced later" not in span_text(replacing, KnowledgeIndex(self._knowledge()).for_span(replacing))

    def test_a_superseded_fact_is_never_shown_as_one_of_the_spans_facts(self):
        annotations = KnowledgeIndex(self._knowledge()).for_span(_span(0, ARCHIVED))
        assert annotations.facts == () and [old.content for old, _ in annotations.replaced] == ["postgres"]

    def test_the_note_is_paid_for_in_the_spans_cost(self):
        bare = level_costs(_span(0, ARCHIVED), Annotations())
        noted = level_costs(_span(0, ARCHIVED), KnowledgeIndex(self._knowledge()).for_span(_span(0, ARCHIVED)))
        assert all(with_note > without for with_note, without in zip(noted, bare))  # type: ignore[operator]


class TestTheModelIsToldWhatItHas:
    def test_the_header_names_every_level_and_asks_for_honesty_about_gaps(self):
        assert all(word in FRONT_HEADER for word in ("quoted exactly", "summarised", "archived", "out of view"))
        assert "say so rather than guess" in FRONT_HEADER

    def test_folded_spans_share_one_line_in_order_with_the_rest(self):
        front = render_front([_span(0, OUT), _span(2, OUT), _span(4, ARCHIVED)], ())
        assert front is not None
        lines = front.message.content.split("\n")
        assert lines[-2] == f"{FOLDED_PREFIX} (turns 0–3) 2 earlier exchanges, folded to stay within the budget."
        assert lines[-1].startswith(f"{ARCHIVED_PREFIX} (turns 4–5)")

    def test_the_facts_reported_sent_are_the_ones_on_summarised_and_archived_lines(self):
        """A kept span's facts are in its own words; listing them as sent would double count."""
        knowledge = (_fact(0, "postgres"), _fact(2, "redis", "cache"), _fact(4, "kafka", "queue"))
        front = render_front([_span(0, KEPT), _span(2, COMPACTED), _span(4, ARCHIVED)], knowledge)
        assert front is not None
        assert [fact.content for fact in front.facts] == ["redis", "kafka"]
