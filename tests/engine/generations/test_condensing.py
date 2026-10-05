"""Condensing: which turns become one span, and the material each span is made with."""

import asyncio

from hypothesis import given
from hypothesis import strategies as st

from llm_gc.engine.compaction import LLMCompactor
from llm_gc.engine.cut_point import tool_result_ids, tool_use_ids
from llm_gc.engine.generations import GenerationalMemory, OpeningWordsIndexWriter
from llm_gc.engine.generations.condensing import INDEX_LINE_MAX_CHARS, group_exchanges
from llm_gc.events import EventBus
from llm_gc.extraction import KnowledgeExtractor
from llm_gc.llm import FakeProvider, LLMResponse, StopReason
from llm_gc.exceptions import LLMTimeoutError
from llm_gc.models import Message
from llm_gc.models.knowledge_entry import KnowledgeType
from llm_gc.utils import count_tokens


def _user(index: int, text: str = "a question about the design", tokens: int = 20) -> Message:
    return Message(role="user", content=text, token_count=tokens, turn_index=index)


def _assistant(index: int, text: str = "an answer about the design", tokens: int = 200, calls: tuple[str, ...] = ()
               ) -> Message:
    raw = [{"type": "tool_use", "id": call, "name": "lookup", "input": {}} for call in calls] or None
    return Message(role="assistant", content=text, token_count=tokens, turn_index=index, raw_content=raw)


def _results(index: int, calls: tuple[str, ...]) -> Message:
    raw = [{"type": "tool_result", "tool_use_id": call, "content": "ok"} for call in calls]
    return Message(role="user", content="ok", token_count=5, turn_index=index, raw_content=raw)


def _indices(groups) -> list[list[int]]:
    return [[turn.turn_index for turn in group] for group in groups]


class TestGroupingIntoExchanges:
    def test_each_exchange_is_a_span(self):
        turns = [_user(0), _assistant(1), _user(2), _assistant(3)]
        assert _indices(group_exchanges(turns)) == [[0, 1], [2, 3]]

    def test_tool_results_stay_in_the_exchange_that_called_for_them(self):
        """A tool result arrives as a user turn, but it continues the exchange; it does not open one."""
        turns = [_user(0), _assistant(1, calls=("a",)), _results(2, ("a",)), _assistant(3)]
        assert _indices(group_exchanges(turns)) == [[0, 1, 2, 3]]

    def test_a_large_exchange_is_divided_between_tool_rounds(self):
        turns = [_user(0), _assistant(1, tokens=600, calls=("a",)), _results(2, ("a",)),
                 _assistant(3, tokens=600, calls=("b",)), _results(4, ("b",)), _assistant(5, tokens=600)]
        assert _indices(group_exchanges(turns, max_span_tokens=1000)) == [[0, 1, 2], [3, 4], [5]]

    def test_a_question_is_never_parted_from_its_first_answer(self):
        """Even an answer far over the limit: a span of the question alone would name nothing."""
        turns = [_user(0), _assistant(1, tokens=50_000), _user(2), _assistant(3)]
        assert _indices(group_exchanges(turns, max_span_tokens=1000)) == [[0, 1], [2, 3]]

    def test_a_batch_that_opens_mid_exchange_starts_a_span_there(self):
        """The previous collection took the question; this one takes the answer."""
        turns = [_assistant(4), _user(5), _assistant(6)]
        assert _indices(group_exchanges(turns)) == [[4], [5, 6]]

    @given(exchanges=st.lists(st.tuples(st.integers(0, 3), st.lists(st.integers(1, 3000), min_size=4, max_size=4)),
                              min_size=1, max_size=8),
           limit=st.integers(100, 5000))
    def test_spans_partition_the_turns_and_never_part_a_call_from_its_result(self, exchanges, limit):
        turns: list[Message] = []
        for number, (rounds, sizes) in enumerate(exchanges):
            turns.append(_user(len(turns), tokens=sizes[0]))
            for round_number in range(rounds):
                call = (f"{number}-{round_number}",)
                turns.append(_assistant(len(turns), tokens=sizes[1], calls=call))
                turns.append(_results(len(turns), call))
            turns.append(_assistant(len(turns), tokens=sizes[2]))

        groups = group_exchanges(turns, max_span_tokens=limit)

        assert [turn for group in groups for turn in group] == turns
        for group in groups:
            called = {call for turn in group for call in tool_use_ids(turn)}
            answered = {call for turn in group for call in tool_result_ids(turn)}
            assert answered <= called, "a tool result is in a different span from its call"


class TestTheIndexLine:
    writer = OpeningWordsIndexWriter()

    def test_it_is_the_first_sentence_the_user_wrote(self):
        turns = [_user(0, "How long do we hold seats?  Also, what about refunds?"), _assistant(1, "Seven minutes.")]
        assert self.writer.write(turns) == "How long do we hold seats?"

    def test_a_tool_result_is_not_what_was_asked(self):
        turns = [_results(0, ("a",)), _assistant(1, "The lookup shows the pool is saturated. Raise it.")]
        assert self.writer.write(turns) == "The lookup shows the pool is saturated."

    def test_a_long_sentence_is_cut_at_a_word_on_one_line(self):
        text = "We need to decide\nhow the ingest service " + "handles very large batches " * 10
        line = self.writer.write([_user(0, text)])
        assert len(line) <= INDEX_LINE_MAX_CHARS
        assert line.endswith("…") and "\n" not in line
        assert line.startswith("We need to decide how the ingest service")


_SUMMARY = LLMResponse(text="They settled on mysql for the database after a long discussion.",
                       stop_reason=StopReason.COMPLETE, input_tokens=0, output_tokens=14,
                       model="scripted", provider="offline")


def _memory(responses) -> tuple[GenerationalMemory, FakeProvider]:
    bus = EventBus()
    provider = FakeProvider(responses=responses)
    return GenerationalMemory(knowledge_extractor=KnowledgeExtractor(),
                              compactor=LLMCompactor(event_bus=bus, provider=provider)), provider


def _said(index: int, role: str, text: str) -> Message:
    return Message(role=role, content=text, token_count=count_tokens(text), turn_index=index)


_LONG_ANSWER = ("The database is postgres. " + "We compared replication, failover and the cost of each option. " * 12)


class TestCondensing:
    def test_facts_are_read_from_the_original_turns_never_the_summary(self):
        """The summary says mysql; the turns said postgres. A fact read second-hand
        would put words in the conversation's mouth."""
        memory, _ = _memory([_SUMMARY])
        [span] = asyncio.run(memory.condense([_said(0, "user", "Which database?"),
                                              _said(1, "assistant", _LONG_ANSWER)]))

        assert span.summary is not None
        assert [(fact.topic_label, fact.content) for fact in span.facts] == [("database", "postgres")]
        assert span.facts[0].message_turn == 1, "a fact names the turn it was read from"

    def test_nothing_is_kept_as_raw_text(self):
        """The span keeps the turns themselves, so text extraction cannot read is not copied anywhere."""
        memory, _ = _memory([_SUMMARY])
        [span] = asyncio.run(memory.condense([_said(0, "user", "Thanks, that helps a lot."),
                                              _said(1, "assistant", "Glad to hear it. " * 40)]))

        assert span.facts == ()
        assert not [fact for fact in span.facts if fact.knowledge_type is KnowledgeType.RAW]

    def test_a_span_too_small_to_shorten_is_not_sent_to_the_model(self):
        memory, provider = _memory([])
        [span] = asyncio.run(memory.condense([_said(0, "user", "ok")]))

        assert provider.calls == []
        assert span.summary is None and span.summary_error is None

    def test_a_failed_summary_is_recorded_and_the_span_is_still_made(self):
        memory, _ = _memory([LLMTimeoutError("the provider did not answer in time")])
        [span] = asyncio.run(memory.condense([_said(0, "user", "Which database?"),
                                              _said(1, "assistant", _LONG_ANSWER)]))

        assert span.summary is None
        assert isinstance(span.summary_error, LLMTimeoutError)
        assert span.facts, "the facts do not depend on the summary"
        assert span.index_line == "Which database?"

    def test_each_span_gets_its_own_summary(self):
        memory, provider = _memory([_SUMMARY, _SUMMARY])
        spans = asyncio.run(memory.condense([_said(0, "user", "Which database?"), _said(1, "assistant", _LONG_ANSWER),
                                             _said(2, "user", "And the cache?"), _said(3, "assistant", _LONG_ANSWER)]))

        assert [[turn.turn_index for turn in span.turns] for span in spans] == [[0, 1], [2, 3]]
        assert len(provider.calls) == 2
        assert all(span.summary is not None for span in spans)
