"""Where a conversation can be cut, so that everything before it becomes the front.

A cut replaces the turns before it with one summarised front and keeps every turn
after it verbatim. Two rules decide where it may fall: the most recent turns stay
verbatim, and a tool round is never split - a `tool_result` sent without the
`tool_use` it answers is rejected by the provider, which would fail the user's
request because of LLM-GC.
"""

from collections.abc import Iterable, Sequence

from llm_gc.models import Message


def find_cut_point(turns: Sequence[Message], keep_last: int) -> int:
    """The number of leading turns that may be cut, or 0 if no cut is safe.

    Takes the latest position that keeps at least `keep_last` turns verbatim and
    leaves no tool result in the kept tail answering a tool call in the cut part.
    """
    # One entry per cut position, 0 through len(turns): cutting after the last turn
    # is a position too, when keep_last is 0.
    tool_calls_before: list[set[str]] = [set()]
    for turn in turns:
        tool_calls_before.append(tool_calls_before[-1] | set(tool_use_ids(turn)))

    for cut in range(len(turns) - keep_last, 0, -1):
        cut_calls = tool_calls_before[cut]
        if not any(set(tool_result_ids(turn)) & cut_calls for turn in turns[cut:]):
            return cut
    return 0


def is_safe_cut(turns: Sequence[Message], cut: int) -> bool:
    """Whether cutting the first `cut` turns leaves every kept tool result its call."""
    cut_calls = {i for turn in turns[:cut] for i in tool_use_ids(turn)}
    return not any(set(tool_result_ids(turn)) & cut_calls for turn in turns[cut:])


def tool_use_ids(message: Message) -> list[str]:
    return [block["id"] for block in _blocks(message) if block.get("type") == "tool_use" and "id" in block]


def tool_result_ids(message: Message) -> list[str]:
    return [block["tool_use_id"] for block in _blocks(message)
            if block.get("type") == "tool_result" and "tool_use_id" in block]


def _blocks(message: Message) -> Iterable[dict]:
    if isinstance(message.raw_content, list):
        return (block for block in message.raw_content if isinstance(block, dict))
    return ()
