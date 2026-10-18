"""Reading a request's conversation and writing it back: the client's instructions are never condensed."""

import json
import uuid

import pytest

from llm_gc.engine.generations.front import Annotations, span_text
from llm_gc.exceptions import ConfigError
from llm_gc.models import Message
from llm_gc.models.generation_snapshot import DetailLevel, Span
from llm_gc.models.message import parts_by_writer, person_text
from llm_gc.proxy.anthropic_translation import (
    DEFAULT_CACHE_LIFETIME_S,
    Instruction,
    cache_lifetime_s,
    carry_instructions,
    extract_instructions,
    extract_messages,
    fixed_prompt_tokens,
    instruction_tokens,
    reply_text,
    rewrite_messages,
    rewrite_overhead_tokens,
    uncached_prefix_tokens,
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


def _events(*events: dict) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


STREAMED_TOOL_ROUND = (
    {"type": "message_start", "message": {"id": "m", "role": "assistant", "content": []}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "The suite is in tests/."}},
    {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Running "}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "them."}},
    {"type": "content_block_start", "index": 2,
     "content_block": {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}}},
    {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": '{"command": '}},
    {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": '"pytest -q"}'}},
    {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
)


class TestTheNewestReplyIsRead:
    """Shown on the dashboard before the next request records it, so it must read as
    the recorded turn will: a reply shown one way and recorded another is two answers."""

    def test_a_streamed_reply_reads_as_its_recorded_turn_will(self):
        streamed = reply_text(_events(*STREAMED_TOOL_ROUND, {"type": "message_stop"}), streamed=True)
        assert streamed == extract_messages(TOOL_ROUND)[1].content

    def test_a_stream_that_stopped_before_its_end_is_not_a_reply(self):
        assert reply_text(_events(*STREAMED_TOOL_ROUND), streamed=True) is None

    def test_an_answer_reads_as_its_recorded_turn_will(self):
        body = {"type": "message", "role": "assistant", "stop_reason": "tool_use",
                "content": TOOL_ROUND["messages"][1]["content"]}
        assert reply_text(json.dumps(body).encode(), streamed=False) == extract_messages(TOOL_ROUND)[1].content

    def test_an_error_or_garbage_is_not_a_reply(self):
        error = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
        assert reply_text(json.dumps(error).encode(), streamed=False) is None
        assert reply_text(b"not json", streamed=False) is None
        assert reply_text(b"data: {broken\n\n", streamed=True) is None


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

    @pytest.mark.parametrize("said", [
        "Check how `<system-reminder>` blocks are lifted.",
        "Wrap it in <system-reminder>x</system-reminder> tags, like Claude Code does.",
        "Is the tag\n<system-reminder> on its own line\nread as the start of one?",
    ])
    def test_a_person_naming_the_element_in_their_words_keeps_every_word(self, said):
        """Live run, 2026-10-17: a mention read as an element ran on to the end of the real one after it,
        filing the rest of the person's words as the client's instructions."""
        payload = {"messages": [{"role": "user", "content": [
            {"type": "text", "text": said}, {"type": "text", "text": REMINDER}]}]}

        turn = extract_messages(payload)[0]

        assert turn.content == said, "every word is conversation"
        assert turn.instructions == (REMINDER,), "the client's own element is still lifted"

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


class TestARelayedReportIsMarked:
    """Claude Code relays a background subagent's report as a user turn wrapped in <agent-message>."""

    @staticmethod
    def _turn(content) -> Message:
        return extract_messages({"messages": [{"role": "user", "content": "go"}, {"role": "assistant", "content": "ok"},
                                              {"role": "user", "content": content}]})[-1]

    def test_a_turn_that_opens_with_the_element_is_relayed(self):
        assert self._turn('<agent-message agent="a1">Found 42 test files.</agent-message>').relayed

    def test_it_is_still_relayed_after_the_instructions_before_it_are_lifted(self):
        """Run 2 of the trace: a reminder block, then the report, in one user turn."""
        turn = self._turn([{"type": "text", "text": "<system-reminder>Today is Tuesday.</system-reminder>"},
                           {"type": "text", "text": "<agent-message>README: three lines.</agent-message>"}])
        assert turn.relayed and turn.instructions and turn.content.startswith("<agent-message")

    def test_it_is_relayed_after_the_line_claude_code_writes_before_it(self):
        """The shape seen on a live run, the element's `from` naming the subagent."""
        assert self._turn('Another Claude session sent a message:\n<agent-message from="a1">\n'
                          '[Subagent report] Found 42 test files.\n</agent-message>').relayed

    @pytest.mark.parametrize("content", ["Please read <agent-message> tags carefully.", "What did the agent find?",
                                         'Look at this: <agent-message from="a1">done</agent-message>'])
    def test_a_message_the_user_wrote_is_not(self, content):
        assert not self._turn(content).relayed

    def test_an_assistant_turn_quoting_the_element_is_not(self):
        turns = extract_messages({"messages": [{"role": "user", "content": "go"},
                                               {"role": "assistant", "content": "<agent-message>x</agent-message>"}]})
        assert not turns[-1].relayed

    def test_a_quoted_report_is_named_as_one_in_the_recap(self):
        report = Message(role="user", content="<agent-message>Found 42 test files.</agent-message>", turn_index=4,
                         relayed=True)
        text = span_text(Span(turns=(report,), level=DetailLevel.KEPT), Annotations())
        assert "(turn 4, subagent report)" in text and "(turn 4, user)" not in text


class TestTextClaudeCodeWroteIsMarked:
    """Claude Code merges its own text into the person's turn: a command's expansion, a local command's output."""

    COMMAND = ("<command-message>audit is running…</command-message>\n<command-name>/audit</command-name>\n"
               "<command-args>verify</command-args>")
    EXPANSION = "Base directory for this skill: /skills/audit\n\nKeep at most 5 rows."

    @staticmethod
    def _turn(content) -> Message:
        return extract_messages({"messages": [{"role": "user", "content": "go"}, {"role": "assistant", "content": "ok"},
                                              {"role": "user", "content": content}]})[-1]

    def test_a_skill_commands_expansion_is_marked_after_the_instructions_are_lifted(self):
        """The wire shape: the command, its expansion and a reminder, as separate blocks of one turn."""
        turn = self._turn([{"type": "text", "text": self.COMMAND}, {"type": "text", "text": self.EXPANSION},
                           {"type": "text", "text": "<system-reminder>Today is Tuesday.</system-reminder>"}])

        assert turn.instructions and person_text(turn) == self.COMMAND
        assert [part.text for part in parts_by_writer(turn) if part.by_client] == [self.EXPANSION]

    def test_the_words_typed_after_a_local_command_stay_the_persons(self):
        turn = self._turn([{"type": "text", "text": "<local-command-caveat>Caveat: local.</local-command-caveat>"},
                           {"type": "text", "text": "<command-name>/model</command-name>"},
                           {"type": "text", "text": "<local-command-stdout>Set model to Sonnet</local-command-stdout>"},
                           {"type": "text", "text": "Now use Postgres."}])

        assert person_text(turn) == "<command-name>/model</command-name>\n\nNow use Postgres."

    def test_tool_output_quoting_the_elements_is_not_marked(self):
        """Reading a transcript file brings Claude Code's elements back as tool output, which is no one's words."""
        turn = self._turn([{"type": "tool_result", "tool_use_id": "t1",
                            "content": f"{self.COMMAND}\n{self.EXPANSION}"}])
        assert turn.client_written == ()

    def test_a_relayed_report_and_an_assistant_turn_are_not_marked(self):
        assert self._turn(f'<agent-message from="a1">\n{self.COMMAND}\n{self.EXPANSION}\n</agent-message>'
                          ).client_written == ()
        turns = extract_messages({"messages": [{"role": "user", "content": "go"},
                                               {"role": "assistant", "content": f"{self.COMMAND}\n{self.EXPANSION}"}]})
        assert turns[-1].client_written == ()


def _marked(ttl: str | None = None) -> dict:
    return {"type": "ephemeral", **({"ttl": ttl} if ttl is not None else {})}


class TestTheCacheLifetimeAsked:
    """The lifetime of the entry covering the conversation: what a collection makes the provider write again."""

    def test_a_marker_naming_no_lifetime_gets_the_providers_default(self):
        payload = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi",
                                                              "cache_control": _marked()}]}]}
        assert cache_lifetime_s(payload) == DEFAULT_CACHE_LIFETIME_S == 300.0

    def test_an_hour_is_read_as_an_hour(self):
        payload = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi",
                                                              "cache_control": _marked("1h")}]}]}
        assert cache_lifetime_s(payload) == 3600.0

    def test_the_last_marker_in_prompt_order_decides(self):
        """An hour on the system prompt, five minutes on the conversation: the conversation's entry is the one at stake."""
        payload = {"system": [{"type": "text", "text": "be careful", "cache_control": _marked("1h")}],
                   "messages": [{"role": "user", "content": [{"type": "text", "text": "hi",
                                                              "cache_control": _marked("5m")}]}]}
        assert cache_lifetime_s(payload) == 300.0

    def test_a_top_level_marker_comes_last_because_the_provider_applies_it_to_the_last_block(self):
        payload = {"cache_control": _marked("1h"),
                   "messages": [{"role": "user", "content": [{"type": "text", "text": "hi",
                                                              "cache_control": _marked("5m")}]}]}
        assert cache_lifetime_s(payload) == 3600.0

    @pytest.mark.parametrize("payload", [
        {"messages": [{"role": "user", "content": "no markers at all"}]},
        {"messages": "not a list", "system": "a string system prompt", "tools": None},
        {"cache_control": _marked("forever")},
        {"cache_control": _marked(3600)},
    ])
    def test_no_marker_or_an_unreadable_one_gets_the_default(self, payload):
        assert cache_lifetime_s(payload) == DEFAULT_CACHE_LIFETIME_S


MARK = {"type": "ephemeral", "ttl": "1h"}
STATIC = "You are a careful coding assistant. Follow the repository's conventions. " * 30
DYNAMIC = "Working directory: /repo. Git status: clean. Today's date: 2026-10-18. " * 20
TOOL = {"name": "Read", "description": "Reads a file from the local filesystem. " * 10,
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}


class TestWhatACollectionCachesAgain:
    """Live run: a collection's re-cache was estimated at 4,101 tokens, and the provider wrote about 17,500
    of LLM-GC's: the client's instructions and the end of its system prompt were left out."""

    def test_the_system_prompt_after_the_last_cache_marker_is_cached_again(self):
        payload = {"tools": [TOOL], "system": [{"type": "text", "text": STATIC, "cache_control": MARK},
                                               {"type": "text", "text": DYNAMIC}]}
        assert uncached_prefix_tokens(payload) == count_tokens(DYNAMIC)

    def test_nothing_after_a_marker_on_the_last_block(self):
        payload = {"tools": [TOOL], "system": [{"type": "text", "text": STATIC},
                                               {"type": "text", "text": DYNAMIC, "cache_control": MARK}]}
        assert uncached_prefix_tokens(payload) == 0

    def test_a_marker_on_a_tool_leaves_the_whole_system_prompt_after_it(self):
        payload = {"tools": [{**TOOL, "cache_control": MARK}], "system": STATIC}
        assert uncached_prefix_tokens(payload) == count_tokens(STATIC)

    def test_with_no_marker_everything_is_cached_again(self):
        payload = {"tools": [TOOL], "system": [{"type": "text", "text": STATIC}, {"type": "text", "text": DYNAMIC}]}
        assert uncached_prefix_tokens(payload) == fixed_prompt_tokens(payload)

    def test_each_instruction_counts_once_with_the_instruction_messages(self):
        payload = {**CLAUDE_CODE, "messages": [
            {"role": "user", "content": f"{REMINDER}\n\nbuild a CSV converter"},
            {"role": "assistant", "content": "Done."},
            {"role": "user", "content": f"{REMINDER}\n\nadd a --pretty flag"},
            {"role": "system", "content": "# Environment: working directory /tmp/project"}]}

        overhead = rewrite_overhead_tokens(payload, extract_messages(payload), extract_instructions(payload))

        assert overhead == count_tokens(REMINDER) + count_tokens("# Environment: working directory /tmp/project")


def _numbered_request(marker: str, turns: int) -> dict:
    """A conversation of `turns` plain turns, each saying something no other text does."""
    return {"model": "m", "max_tokens": 10, "messages": [
        {"role": "user" if number % 2 == 0 else "assistant", "content": f"{marker} turn {number}"}
        for number in range(turns)]}


class TestCountsAreRemembered:
    """A client sends its whole conversation on every request. Tokenising all of it each time
    costs more with every turn, on the event loop, and the proxy's latency would show it.

    The count cache is shared by the whole run, so each test builds its texts from a marker of
    its own that no other test has counted."""

    def test_a_conversation_that_grew_is_tokenised_only_where_it_is_new(self, tokenised):
        marker = uuid.uuid4().hex
        first = extract_messages(_numbered_request(marker, 6))
        assert len(tokenised) == 6
        tokenised.clear()

        grown = extract_messages(_numbered_request(marker, 8))

        assert tokenised == [f"{marker} turn 6", f"{marker} turn 7"]
        assert [turn.token_count for turn in grown[:6]] == [turn.token_count for turn in first]

    def test_text_the_client_sends_a_second_time_in_another_conversation_is_not_tokenised_again(self, tokenised):
        """A count depends on the text alone, not on which conversation sent it."""
        marker = uuid.uuid4().hex
        extract_messages(_numbered_request(marker, 4))
        tokenised.clear()

        extract_messages(_numbered_request(marker, 4))

        assert tokenised == []

    def test_instructions_and_the_system_prompt_are_tokenised_once_however_many_figures_use_them(self, tokenised):
        """Every figure of a request counts the same reminders and system prompt: the sizes of
        the request as sent and as forwarded, what a collection makes the provider cache again,
        the dashboard. Fixing only the turns would leave these tokenised on every request."""
        marker = uuid.uuid4().hex
        reminder = f"<system-reminder>\n{marker} the date is today\n</system-reminder>"
        payload = {"model": "m", "max_tokens": 10, "system": f"{marker} you are a coding assistant",
                   "messages": [{"role": "user", "content": f"{marker} question\n{reminder}"},
                                {"role": "system", "content": f"{marker} environment notes"}]}
        messages, instructions = extract_messages(payload), extract_instructions(payload)

        for _ in range(3):
            instruction_tokens(messages)
            fixed_prompt_tokens(payload)
            rewrite_overhead_tokens(payload, messages, instructions)
            sum(instruction.token_count for instruction in instructions)

        assert len(tokenised) == len(set(tokenised))
        assert tokenised.count(reminder) == 1
