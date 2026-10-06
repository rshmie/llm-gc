"""Reading a request's conversation and writing it back: the client's instructions are never condensed."""

import pytest

from llm_gc.engine.generations.front import Annotations, span_text
from llm_gc.exceptions import ConfigError
from llm_gc.models import Message
from llm_gc.models.generation_snapshot import DetailLevel, Span
from llm_gc.proxy.anthropic_translation import (
    Instruction,
    carry_instructions,
    extract_instructions,
    extract_messages,
    rewrite_messages,
)
from llm_gc.utils import count_tokens

# The shape Claude Code sends: its environment, then its tools' instructions, as system
# messages after the user turns they accompany.
CLAUDE_CODE = {"model": "m", "max_tokens": 10, "messages": [
    {"role": "user", "content": "build a CSV converter"},
    {"role": "system", "content": "# Environment: working directory /tmp/project"},
    {"role": "assistant", "content": "Done."},
    {"role": "user", "content": "add a --pretty flag"},
    {"role": "system", "content": [{"type": "text", "text": "# MCP Server Instructions"}]},
]}


def _turn(index: int, role: str = "user") -> Message:
    return Message(role=role, content=f"turn {index}", token_count=2, turn_index=index)


class TestReading:
    def test_instructions_are_not_turns_and_do_not_shift_turn_indices(self):
        turns = extract_messages(CLAUDE_CODE)

        assert [(t.turn_index, t.role, t.content) for t in turns] == [
            (0, "user", "build a CSV converter"), (1, "assistant", "Done."), (2, "user", "add a --pretty flag")]

    def test_each_instruction_is_kept_whole_with_its_place(self):
        instructions = extract_instructions(CLAUDE_CODE)

        assert [(i.after_turns, i.raw) for i in instructions] == [
            (1, CLAUDE_CODE["messages"][1]), (3, CLAUDE_CODE["messages"][4])]

    def test_a_request_with_no_conversation_is_not_one_we_can_process(self):
        with pytest.raises(ConfigError):
            extract_messages({"messages": [{"role": "system", "content": "only instructions"}]})


# One tool round as Claude Code sends it: the assistant explains and calls a tool,
# and the client answers with what the tool returned.
TOOL_ROUND = {"model": "m", "max_tokens": 10, "messages": [
    {"role": "user", "content": "run the tests"},
    {"role": "assistant", "content": [
        {"type": "thinking", "thinking": "The suite is in tests/.", "signature": "sig"},
        {"type": "text", "text": "Running them."},
        {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "pytest -q"}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "42 passed in 1.2s"}]},
]}


class TestToolTurnsAreRead:
    """A tool call and its result are part of what a turn says: everything that sizes,
    scores, summarises or quotes a turn reads its text, so a block left out of it is
    lost from every collected turn."""

    def test_a_tool_call_reads_as_its_name_and_input_and_thinking_is_left_out(self):
        call = extract_messages(TOOL_ROUND)[1]

        assert call.content == 'Running them.\n[tool call: Bash] {"command": "pytest -q"}'
        assert call.token_count == count_tokens(call.content)

    def test_a_tool_result_reads_as_its_output(self):
        assert extract_messages(TOOL_ROUND)[2].content == "[tool result] 42 passed in 1.2s"

    def test_a_failed_tool_says_so_and_an_attachment_in_its_output_is_a_placeholder(self):
        result = {"type": "tool_result", "tool_use_id": "toolu_1", "is_error": True,
                  "content": [{"type": "text", "text": "No such file"}, {"type": "image", "source": {}}]}
        payload = {"messages": [{"role": "user", "content": [result]}]}

        assert extract_messages(payload)[0].content == "[tool result, error] No such file\n[image]"

    def test_a_tool_result_with_no_output_still_reads_as_one(self):
        payload = {"messages": [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1"}]}]}

        assert extract_messages(payload)[0].content == "[tool result] "

    def test_a_kept_span_quotes_what_the_tool_returned(self):
        """The subtle failure: a span kept word for word whose quote leaves out the
        tool's output, while the dashboard says nothing was lost."""
        span = Span(turns=tuple(extract_messages(TOOL_ROUND)), level=DetailLevel.KEPT)

        quoted = span_text(span, Annotations())

        assert '[tool call: Bash] {"command": "pytest -q"}' in quoted
        assert "[tool result] 42 passed in 1.2s" in quoted


REMINDER = "<system-reminder>\nContents of CLAUDE.md: use uv, never pip.\n</system-reminder>"


class TestInstructionsInsideATurn:
    """A client's instructions inside a message are not the conversation: the engine
    must not summarise them, and the model must still receive them word for word."""

    def test_they_are_lifted_out_of_the_users_words_in_order(self):
        later = "<system-reminder>Today is Tuesday.</system-reminder>"
        payload = {"messages": [{"role": "user", "content": [
            {"type": "text", "text": REMINDER}, {"type": "text", "text": later}, {"type": "text", "text": "Help me design it."}]}]}

        turn = extract_messages(payload)[0]

        assert turn.content == "Help me design it."
        assert turn.instructions == (REMINDER, later)
        assert turn.token_count == count_tokens("Help me design it.")

    def test_they_are_lifted_out_of_a_tool_result(self):
        result = {"type": "tool_result", "tool_use_id": "t1", "content": f"line one\n{REMINDER}"}

        turn = extract_messages({"messages": [{"role": "user", "content": [result]}]})[0]

        assert turn.content == "[tool result] line one" and turn.instructions == (REMINDER,)

    def test_an_assistant_reply_quoting_one_is_conversation(self):
        payload = {"messages": [{"role": "user", "content": "what does it say?"},
                                {"role": "assistant", "content": f"It says: {REMINDER}"}]}

        reply = extract_messages(payload)[1]

        assert REMINDER in reply.content and reply.instructions == ()

    def test_they_open_the_front_once_each_in_order(self):
        """Claude Code repeats some instructions after every file it reads; a copy
        per turn would add nothing the first one did not say."""
        later = "<system-reminder>Today is Tuesday.</system-reminder>"
        stood_in = [Message(role="user", content="a", turn_index=0, instructions=(REMINDER,)),
                    Message(role="user", content="b", turn_index=2, instructions=(later, REMINDER))]
        front = Message(role="user", content="[front]", token_count=1, turn_index=-1)

        carried = carry_instructions(front, stood_in)

        assert carried.raw_content == [{"type": "text", "text": REMINDER}, {"type": "text", "text": later},
                                       {"type": "text", "text": "[front]"}]
        assert carry_instructions(front, [_turn(0)]) is front, "nothing to carry, nothing changed"


class TestWritingBack:
    def test_every_instruction_returns_once_in_place_after_a_cut(self):
        """Turns 0-1 are collected into the front; the instruction that followed turn 0
        goes right after the front, the one that followed turn 3 right after turn 3."""
        early = Instruction(after_turns=1, raw={"role": "system", "content": "environment"})
        late = Instruction(after_turns=4, raw={"role": "system", "content": "tools"})
        front = Message(role="user", content="[front]", token_count=1, turn_index=-1)

        sent = rewrite_messages({"messages": []}, [front, _turn(2), _turn(3, "assistant"), _turn(4)],
                                instructions=[early, late])["messages"]

        assert [m["content"] for m in sent] == ["[front]", "environment", "turn 2", "turn 3", "tools", "turn 4"]
        assert [m["role"] for m in sent if m["content"] in ("environment", "tools")] == ["system", "system"]

    def test_without_a_front_every_instruction_keeps_its_original_place(self):
        instructions = extract_instructions(CLAUDE_CODE)

        sent = rewrite_messages(CLAUDE_CODE, extract_messages(CLAUDE_CODE), instructions=instructions)["messages"]

        assert sent == CLAUDE_CODE["messages"]
