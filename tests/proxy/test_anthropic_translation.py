"""Reading a request's conversation and writing it back: the client's instructions are never condensed."""

import pytest

from llm_gc.exceptions import ConfigError
from llm_gc.models import Message
from llm_gc.proxy.anthropic_translation import Instruction, extract_instructions, extract_messages, rewrite_messages

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
