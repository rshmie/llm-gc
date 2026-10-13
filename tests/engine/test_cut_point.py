"""Where a conversation may be cut: never into the recent turns, never between a question and
its answer, never inside a tool round."""

from hypothesis import given
from hypothesis import strategies as st

from llm_gc.engine.cut_point import _span_starts, find_cut_point
from llm_gc.models.message import tool_result_ids, tool_use_ids
from llm_gc.models import Message


def _text(index: int, role: str) -> Message:
    return Message(role=role, content=f"turn {index}", turn_index=index)


def _call(index: int, *ids: str) -> Message:
    return Message(role="assistant", content="", turn_index=index,
                   raw_content=[{"type": "tool_use", "id": i, "name": "lookup", "input": {}} for i in ids])


def _result(index: int, *ids: str) -> Message:
    return Message(role="user", content="", turn_index=index,
                   raw_content=[{"type": "tool_result", "tool_use_id": i, "content": "ok"} for i in ids])


def _chat(n: int) -> list[Message]:
    return [_text(i, "user" if i % 2 == 0 else "assistant") for i in range(n)]


class TestRecentTurnsStayVerbatim:
    def test_the_latest_cut_keeps_exactly_the_last_n_when_an_exchange_opens_there(self):
        assert find_cut_point(_chat(10), keep_last=4) == 6

    def test_otherwise_it_keeps_one_more_rather_than_cut_a_question_from_its_answer(self):
        """With five kept, the cut would fall between question 4 and its answer."""
        assert find_cut_point(_chat(10), keep_last=5) == 4

    def test_a_conversation_no_longer_than_keep_last_is_not_cut(self):
        assert find_cut_point(_chat(4), keep_last=4) == 0
        assert find_cut_point([], keep_last=2) == 0


class TestToolRoundsAreNeverSplit:
    def test_the_cut_moves_back_rather_than_orphan_a_tool_result(self):
        """A cut at 3 would send turn 3's tool result without the call in turn 2."""
        turns = [_text(0, "user"), _text(1, "assistant"), _call(2, "t1"), _result(3, "t1"),
                 _text(4, "assistant"), _text(5, "user")]
        assert find_cut_point(turns, keep_last=3) == 2

    def test_parallel_calls_answered_in_one_turn_stay_together(self):
        turns = [_text(0, "user"), _call(1, "a", "b"), _result(2, "a", "b"), _text(3, "assistant"), _text(4, "user")]
        assert find_cut_point(turns, keep_last=2) == 3

    def test_a_question_stays_with_the_tool_call_that_answers_it(self):
        """Cutting at 1 would separate the question from its first answer, a tool call."""
        turns = [_text(0, "user"), _call(1, "a"), _result(2, "a"), _text(3, "assistant")]
        assert find_cut_point(turns, keep_last=2) == 0


class TestALongAgenticExchangeIsStillCollected:
    def test_it_is_cut_between_tool_rounds(self):
        """One question, then rounds of tool calls: with no later question to cut
        before, the cut falls between two rounds, so the exchange's start can still be
        condensed while the agent works on."""
        turns = [_text(0, "user"), _call(1, "a"), _result(2, "a"), _call(3, "b"), _result(4, "b"),
                 _call(5, "c"), _result(6, "c"), _text(7, "assistant")]
        assert find_cut_point(turns, keep_last=2) == 5

    def test_no_safe_cut_means_no_cut(self):
        """Every candidate cut would orphan the result in the last turn."""
        turns = [_call(0, "t1"), _text(1, "user"), _text(2, "assistant"), _result(3, "t1")]
        assert find_cut_point(turns, keep_last=1) == 0


def _conversations():
    """Text turns interleaved with tool rounds, some with parallel calls."""
    def build(shape: list[tuple[str, int]]) -> list[Message]:
        turns: list[Message] = []
        call_id = 0
        for kind, width in shape:
            if kind == "text":
                turns.append(_text(len(turns), "user" if len(turns) % 2 == 0 else "assistant"))
            else:
                ids = [f"t{call_id + k}" for k in range(width)]
                call_id += width
                turns.append(_call(len(turns), *ids))
                turns.append(_result(len(turns), *ids))
        return turns
    shape = st.lists(st.tuples(st.sampled_from(["text", "tool"]), st.integers(1, 3)), max_size=25)
    return shape.map(build)


def _splits_a_question_from_its_answer(turns: list[Message], cut: int) -> bool:
    """The turn after the cut is the first answer to the question before it."""
    if not 0 < cut < len(turns):
        return False
    before, after = turns[cut - 1], turns[cut]
    return before.role == "user" and not tool_result_ids(before) and after.role == "assistant"


def _safe(turns: list[Message], cut: int, keep_last: int) -> bool:
    cut_calls = {i for turn in turns[:cut] for i in tool_use_ids(turn)}
    tail_results = {i for turn in turns[cut:] for i in tool_result_ids(turn)}
    return len(turns) - cut >= keep_last and not (cut_calls & tail_results)


@given(turns=_conversations(), keep_last=st.integers(0, 6))
def test_the_cut_is_the_latest_safe_one_where_a_span_may_begin(turns, keep_last):
    cut = find_cut_point(turns, keep_last)
    if cut:
        assert _safe(turns, cut, keep_last)
        assert _span_starts(turns)[cut]
        assert not _splits_a_question_from_its_answer(turns, cut)
    later = range(cut + 1, len(turns) - keep_last + 1)
    assert not any(_safe(turns, c, keep_last) and _span_starts(turns)[c] for c in later), "a later cut was missed"


def test_a_relayed_report_is_not_where_an_exchange_opens():
    """It joins the exchange that started the subagent, so no cut falls just before it."""
    report = Message(role="user", content="<agent-message>done</agent-message>", turn_index=2, relayed=True)
    turns = [_text(0, "user"), _text(1, "assistant"), report, _text(3, "assistant"), _text(4, "user")]
    starts = _span_starts(turns)
    assert starts[2] is False and starts[4] is True
