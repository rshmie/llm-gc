"""Where a conversation can be cut, so that everything before it becomes the front.

A cut replaces the turns before it with one summarised front and keeps every turn
after it verbatim. Three rules decide where it may fall: the most recent turns stay
verbatim; a question is never cut from its answer, so the cut falls only where a
span may begin; and a tool round is never split - a `tool_result` sent without the
`tool_use` it answers is rejected by the provider, which would fail the user's
request because of LLM-GC.
"""

from collections.abc import Sequence

from llm_gc.models import Message
from llm_gc.models.message import said_by_user, tool_result_ids, tool_use_ids


def find_cut_point(turns: Sequence[Message], keep_last: int) -> int:
    """The number of leading turns that may be cut, or 0 if no cut is safe.

    Takes the latest position that keeps at least `keep_last` turns verbatim, where
    a span may begin, and that leaves no tool result in the kept tail answering a
    tool call in the cut part.
    """
    # One entry per cut position, 0 through len(turns): cutting after the last turn
    # is a position too, when keep_last is 0.
    tool_calls_before: list[set[str]] = [set()]
    for turn in turns:
        tool_calls_before.append(tool_calls_before[-1] | set(tool_use_ids(turn)))
    span_starts = _span_starts(turns)

    for cut in range(len(turns) - keep_last, 0, -1):
        cut_calls = tool_calls_before[cut]
        if span_starts[cut] and not any(set(tool_result_ids(turn)) & cut_calls for turn in turns[cut:]):
            return cut
    return 0


def _span_starts(turns: Sequence[Message]) -> list[bool]:
    """For each cut position, whether a span may begin there.

    Where an exchange opens - a turn the user wrote (`said_by_user`) - and before
    an assistant turn in an exchange the assistant has already answered: between the
    tool rounds of a long agentic exchange, where `group_exchanges` divides one too.
    Never between a question and its first answer. After the last turn, always.
    """
    starts: list[bool] = []
    answered = False
    for turn in turns:
        opens = said_by_user(turn)
        starts.append(opens or (turn.role == "assistant" and answered))
        answered = False if opens else answered or turn.role == "assistant"
    return starts + [True]


def is_safe_cut(turns: Sequence[Message], cut: int) -> bool:
    """Whether cutting the first `cut` turns leaves every kept tool result its call."""
    cut_calls = {i for turn in turns[:cut] for i in tool_use_ids(turn)}
    return not any(set(tool_result_ids(turn)) & cut_calls for turn in turns[cut:])

