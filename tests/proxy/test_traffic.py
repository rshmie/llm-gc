"""The traffic ledger: every request counted once, where it belongs, and nothing counted that did not happen."""

import json

import pytest
from datetime import datetime, timedelta, timezone

from llm_gc.llm import ModelUsage
from llm_gc.models import ClientOrigin
from llm_gc.proxy.anthropic_translation import (REFUSAL_REASON_MAX_CHARS, ReplyOutline, StreamOutlineReader, ToolCall,
                                                fixed_prompt_tokens, read_reply, refusal_reason)
from llm_gc.proxy.traffic import ProviderUsage, TrafficLedger, token_rate
from llm_gc.utils import count_tokens

MAIN = ClientOrigin(session_id="cc-1")
AGENT_A = ClientOrigin(session_id="cc-1", agent_id="agent-a")
AGENT_B = ClientOrigin(session_id="cc-1", agent_id="agent-b")


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


def _ledger(**limits) -> TrafficLedger:
    return TrafficLedger(clock=_Clock(), **limits)


def _request(ledger, conversation_id, origin=MAIN, opening="h0", sent=100, forwarded=100):
    return ledger.record_request(conversation_id=conversation_id, origin=origin, model="claude-opus-5-5",
                                 opening=opening, tokens_from_client=sent, tokens_forwarded=forwarded)


def _session(ledger, client_session="cc-1") -> dict:
    return next(s for s in ledger.as_dict()["client_sessions"] if s["client_session"] == client_session)


def _conversation(ledger, conversation_id) -> dict:
    sessions = ledger.as_dict()["client_sessions"]
    return next(c for s in sessions for c in s["conversations"] if c["conversation_id"] == conversation_id)


class TestEachRequestIsCountedWhereItBelongs:
    def test_a_conversations_first_request_is_claimed_on_its_second(self):
        """The first request has no conversation yet; leaving it out would understate every conversation by its
        opening request, which carries the whole system prompt and tools."""
        ledger = _ledger()
        _request(ledger, None, sent=50, forwarded=50)
        _request(ledger, "s_1", sent=80, forwarded=60)

        traffic = _conversation(ledger, "s_1")["traffic"]
        assert (traffic["requests"], traffic["tokens_from_client"], traffic["tokens_forwarded"]) == (2, 130, 110)
        assert _session(ledger)["outside_conversations"]["requests"] == 0

    def test_two_subagents_that_open_the_same_way_are_never_mixed(self):
        ledger = _ledger()
        _request(ledger, None, origin=AGENT_A, opening="same task", sent=10)
        _request(ledger, None, origin=AGENT_B, opening="same task", sent=20)
        _request(ledger, "s_a", origin=AGENT_A, opening="same task", sent=30)
        _request(ledger, "s_b", origin=AGENT_B, opening="same task", sent=40)

        assert _conversation(ledger, "s_a")["traffic"]["tokens_from_client"] == 40
        assert _conversation(ledger, "s_b")["traffic"]["tokens_from_client"] == 60

    def test_a_side_request_no_conversation_claims_stays_in_its_client_session(self):
        ledger = _ledger()
        _request(ledger, None, opening="suggest a title", sent=7)
        _request(ledger, None, opening="h0", sent=50)
        _request(ledger, "s_1", opening="h0", sent=80)

        session = _session(ledger)
        assert session["traffic"]["requests"] == 3
        assert session["outside_conversations"]["requests"] == 1
        assert session["outside_conversations"]["tokens_from_client"] == 7
        assert ledger.total.requests == 3

    def test_a_conversation_without_a_client_session_is_listed_on_its_own(self):
        ledger = _ledger()
        _request(ledger, "s_1", origin=None)

        assert ledger.as_dict()["client_sessions"] == []
        assert [c["conversation_id"] for c in ledger.as_dict()["other_conversations"]] == ["s_1"]


class TestWhatTheProviderReported:
    def test_a_replys_usage_is_added_to_its_conversation_session_and_the_total(self):
        ledger = _ledger()
        counted = _request(ledger, "s_1")
        ledger.record_usage(counted, ProviderUsage(replies=1, input_tokens=5, cache_read_input_tokens=900,
                                                   cache_creation_input_tokens=40, output_tokens=12))

        assert ledger.total.provider.cache_read_input_tokens == 900
        assert _session(ledger)["traffic"]["provider"]["cache_read_input_tokens"] == 900
        assert _conversation(ledger, "s_1")["traffic"]["provider"]["cache_read_input_tokens"] == 900

    def test_the_rate_is_the_providers_whole_prompt_per_token_llm_gc_forwarded(self):
        usage = ProviderUsage(replies=1, input_tokens=40, cache_read_input_tokens=900, cache_creation_input_tokens=540,
                              output_tokens=7)
        assert token_rate(1_000, usage) == 1.48, "every prompt token counts, cached or not; output does not"

    @pytest.mark.parametrize(("forwarded", "usage"), [(0, ProviderUsage(input_tokens=10)),
                                                      (1_000, ProviderUsage(output_tokens=10))])
    def test_a_request_with_either_count_at_zero_measures_no_rate(self, forwarded, usage):
        assert token_rate(forwarded, usage) is None

    def test_a_late_reply_to_a_first_request_still_reaches_the_conversation_that_claimed_it(self):
        ledger = _ledger()
        first = _request(ledger, None)
        _request(ledger, "s_1")
        ledger.record_usage(first, ProviderUsage(replies=1, output_tokens=33))

        assert _conversation(ledger, "s_1")["traffic"]["provider"]["output_tokens"] == 33
        assert ledger.total.provider.output_tokens == 33


class TestCollections:
    def test_a_collection_after_the_last_request_is_not_yet_sent_until_another_request_comes(self):
        """A conversation that never sends again - a finished subagent - must not read as having saved anything."""
        ledger = _ledger()
        _request(ledger, "s_1", sent=100, forwarded=100)
        ledger.record_collection("s_1", completed=True, spent=ModelUsage(calls=2, input_tokens=300, output_tokens=40))

        conversation = _conversation(ledger, "s_1")
        assert (conversation["collections"], conversation["collections_not_yet_sent"]) == (1, 1)
        assert conversation["traffic"]["tokens_saved"] == 0
        assert conversation["traffic"]["summaries"] == {"calls": 2, "input_tokens": 300, "output_tokens": 40}

        _request(ledger, "s_1", sent=120, forwarded=70)
        conversation = _conversation(ledger, "s_1")
        assert conversation["collections_not_yet_sent"] == 0
        assert conversation["traffic"]["tokens_saved"] == 50

    def test_a_collection_that_made_no_cut_still_counts_what_it_spent(self):
        ledger = _ledger()
        _request(ledger, "s_1")
        ledger.record_collection("s_1", completed=False, spent=ModelUsage(calls=1, input_tokens=10, output_tokens=2))

        assert _conversation(ledger, "s_1")["collections"] == 0
        assert ledger.total.summaries.calls == 1
        assert _session(ledger)["traffic"]["summaries"]["calls"] == 1


class TestBounds:
    def test_a_conversation_dropped_above_the_limit_stays_counted_in_the_totals(self):
        ledger = _ledger(max_conversations=1)
        _request(ledger, "s_1", sent=10)
        _request(ledger, "s_2", sent=20)

        session = _session(ledger)
        assert [c["conversation_id"] for c in session["conversations"]] == ["s_2"]
        assert session["traffic"]["tokens_from_client"] == 30
        assert session["outside_conversations"]["tokens_from_client"] == 10


class TestTheWholeRequestIsCounted:
    def test_the_system_prompt_and_tools_count_in_either_form(self):
        tools = [{"name": "Read", "description": "Reads a file from disk.", "input_schema": {"type": "object"}}]
        as_string = fixed_prompt_tokens({"system": "You are Claude Code.", "tools": tools})
        as_blocks = fixed_prompt_tokens({"system": [{"type": "text", "text": "You are Claude Code.",
                                                     "cache_control": {"type": "ephemeral"}}], "tools": tools})

        assert as_string == as_blocks == count_tokens("You are Claude Code.") + count_tokens(json.dumps(tools))

    def test_a_request_with_neither_counts_nothing_extra(self):
        assert fixed_prompt_tokens({"messages": []}) == 0
        assert fixed_prompt_tokens({"system": 42, "tools": "not a list"}) == 0


class TestReadingTheReplysOutline:
    def test_an_unstreamed_replys_usage_stop_reason_and_tool_calls(self):
        raw = json.dumps({"type": "message", "stop_reason": "tool_use",
                          "content": [{"type": "text", "text": "Checking."},
                                      {"type": "tool_use", "id": "toolu_1", "name": "Agent", "input": {"prompt": "x"}}],
                          "usage": {"input_tokens": 3, "cache_read_input_tokens": 2000,
                                    "cache_creation_input_tokens": None, "output_tokens": 50}})
        assert read_reply(200, raw.encode())[0] == ReplyOutline(
            status=200, finished=True, stop_reason="tool_use",
            usage=ProviderUsage(replies=1, input_tokens=3, cache_read_input_tokens=2000, output_tokens=50),
            tool_calls=(ToolCall(name="Agent", call_id="toolu_1"),))

    def test_an_error_body_or_garbage_outlines_only_its_status_and_error_type(self):
        assert read_reply(529, b'{"type": "error", "error": {}}')[0] == ReplyOutline(status=529)
        assert read_reply(200, b"not json")[0] == ReplyOutline(status=200)
        refused = b'{"type": "error", "error": {"type": "invalid_request_error", "message": "messages.31: bad"}}'
        assert read_reply(400, refused)[0] == ReplyOutline(status=400, error_type="invalid_request_error")

    def test_a_refusals_reason_is_its_type_and_message_cut_short(self):
        message = "messages.31.content.0: Invalid `signature` in `thinking` block. " * 20
        raw = json.dumps({"type": "error", "error": {"type": "invalid_request_error", "message": message}}).encode()

        reason = refusal_reason(raw)

        assert reason.startswith("invalid_request_error: messages.31.content.0: Invalid `signature`")
        assert len(reason) == REFUSAL_REASON_MAX_CHARS
        assert refusal_reason(b"<html>bad gateway</html>") == "no readable error"
        assert refusal_reason(b'{"error": {"type": "overloaded_error"}}') == "overloaded_error"

    def test_a_stream_split_anywhere_reads_the_same(self):
        """A chunk boundary can fall inside a line; a reader that parsed per chunk would lose events."""
        events = [
            {"type": "message_start", "message": {"usage": {"input_tokens": 4, "cache_read_input_tokens": 1500,
                                                            "cache_creation_input_tokens": 60, "output_tokens": 1}}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta",
                                                                  "text": 'a "message_delta" in the text'}},
            {"type": "content_block_start", "index": 1,
             "content_block": {"type": "tool_use", "id": "toolu_9", "name": "Bash", "input": {}}},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 87}},
            {"type": "message_stop"},
        ]
        stream = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()
        expected = ReplyOutline(status=200, finished=True, stop_reason="tool_use",
                                usage=ProviderUsage(replies=1, input_tokens=4, cache_read_input_tokens=1500,
                                                    cache_creation_input_tokens=60, output_tokens=87),
                                tool_calls=(ToolCall(name="Bash", call_id="toolu_9"),))
        for size in (1, 7, 64, len(stream)):
            reader = StreamOutlineReader()
            for start in range(0, len(stream), size):
                reader.feed(stream[start:start + size])
            assert reader.outline() == expected, size

    def test_a_stream_cut_short_is_unfinished_and_reports_the_prompt_it_was_billed_for(self):
        reader = StreamOutlineReader()
        start = {"type": "message_start", "message": {"usage": {"input_tokens": 900, "output_tokens": 1}}}
        reader.feed(f"data: {json.dumps(start)}\n\n".encode())

        assert reader.outline() == ReplyOutline(status=200, finished=False,
                                                usage=ProviderUsage(replies=1, input_tokens=900, output_tokens=1))

    def test_a_tool_name_that_is_not_a_plain_short_string_is_not_kept(self):
        raw = json.dumps({"type": "message", "stop_reason": "tool_use", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "x" * 500, "input": {}},
            {"type": "tool_use", "id": "toolu_2", "name": "Read\u0007", "input": {}}]})
        assert read_reply(200, raw.encode())[0].tool_calls == ()


class TestReadingASpawnedPrompt:
    """The prompt a tool call gives a subagent, read for linking only and never kept in the outline."""

    PROMPT = "Count the test files under tests/ and report the total."

    def _stream(self, tool_input: dict) -> bytes:
        pieces = json.dumps(tool_input)
        events = [
            {"type": "message_start", "message": {"usage": {"input_tokens": 4, "output_tokens": 1}}},
            {"type": "content_block_start", "index": 0,
             "content_block": {"type": "tool_use", "id": "toolu_a", "name": "Agent", "input": {}}},
            *({"type": "content_block_delta", "index": 0,
               "delta": {"type": "input_json_delta", "partial_json": pieces[at:at + 9]}} for at in range(0, len(pieces), 9)),
            {"type": "content_block_stop", "index": 0},
            {"type": "message_stop"},
        ]
        return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()

    def _read(self, body: bytes, size: int = 13, read_prompts: bool = True) -> StreamOutlineReader:
        reader = StreamOutlineReader(read_prompts=read_prompts)
        for start in range(0, len(body), size):
            reader.feed(body[start:start + size])
        return reader

    def test_a_streamed_input_split_anywhere_gives_its_prompt(self):
        body = self._stream({"description": "count", "prompt": self.PROMPT, "subagent_type": "general"})
        for size in (1, 13, len(body)):
            reader = self._read(body, size)
            assert reader.spawn_prompts() == {"toolu_a": self.PROMPT}, size
            assert reader.outline().tool_calls == (ToolCall(name="Agent", call_id="toolu_a"),)

    def test_nothing_is_read_unless_asked(self):
        assert self._read(self._stream({"prompt": self.PROMPT}), read_prompts=False).spawn_prompts() == {}

    @pytest.mark.parametrize("tool_input", [
        {"command": "pytest -q tests/ --maxfail=1"},         # no prompt
        {"prompt": "short"},                                  # a word, not a task
        {"prompt": "x" * 40_000},                             # larger than any prompt: let go while it streams
    ])
    def test_an_input_that_is_not_a_subagents_task_gives_none(self, tool_input):
        assert self._read(self._stream(tool_input)).spawn_prompts() == {}

    def test_an_unstreamed_reply_gives_its_prompts_apart_from_its_outline(self):
        raw = json.dumps({"type": "message", "stop_reason": "tool_use", "content": [
            {"type": "tool_use", "id": "toolu_a", "name": "Task", "input": {"prompt": self.PROMPT}},
            {"type": "tool_use", "id": "toolu_b", "name": "Read", "input": {"file_path": "/repo/README.md"}}]})

        outline, prompts = read_reply(200, raw.encode(), read_prompts=True)

        assert prompts == {"toolu_a": self.PROMPT}
        assert self.PROMPT not in repr(outline)


class TestTheNewestRequestCountedBothWays:
    def test_each_reply_replaces_the_newest_pair_where_its_request_was_counted(self):
        ledger = _ledger()
        targets = _request(ledger, "c1", forwarded=1_000)
        TrafficLedger.record_usage(targets, ProviderUsage(replies=1, input_tokens=100, cache_read_input_tokens=1_300,
                                                          cache_creation_input_tokens=80), tokens_forwarded=1_000)
        targets = _request(ledger, "c1", forwarded=1_200)
        TrafficLedger.record_usage(targets, ProviderUsage(replies=1, input_tokens=20, cache_read_input_tokens=1_700,
                                                          cache_creation_input_tokens=60), tokens_forwarded=1_200)

        last = _conversation(ledger, "c1")["traffic"]["last_reply"]
        assert last["tokens_forwarded"] == 1_200
        assert last["provider_prompt_tokens"] == 20 + 1_700 + 60, "read in full, from the cache, and written to it"
        assert ledger.total.provider.replies == 2, "the totals still add every reply up"

    def test_without_llm_gcs_count_no_pair_is_kept(self):
        ledger = _ledger()
        TrafficLedger.record_usage(_request(ledger, "c1"), ProviderUsage(replies=1, input_tokens=5))
        assert _conversation(ledger, "c1")["traffic"]["last_reply"] is None


class TestCollectionsWaitForARequestThatReachesTheModel:
    def test_a_request_sent_as_written_leaves_them_waiting(self):
        ledger = _ledger()
        _request(ledger, "s_1")
        ledger.record_collection("s_1", completed=True, spent=ModelUsage())

        ledger.record_request(conversation_id="s_1", origin=MAIN, model="m", opening="h0", tokens_from_client=100,
                              tokens_forwarded=100, carries_front=False)

        assert len(ledger.collections_waiting("s_1")) == 1

    def test_a_refused_request_gives_back_what_it_was_to_carry(self):
        ledger = _ledger()
        _request(ledger, "s_1")
        ledger.record_collection("s_1", completed=True, spent=ModelUsage(), switch_reason="cache_cold")
        _request(ledger, "s_1")
        assert ledger.collections_waiting("s_1") == (), "the request carrying it is on its way"

        ledger.restore_waiting("s_1", ("cache_cold",))

        assert ledger.collections_waiting("s_1") == ("cache_cold",)

    def test_what_a_refused_request_gives_back_stays_ahead_of_a_collection_made_meanwhile(self):
        ledger = _ledger()
        _request(ledger, "s_1")
        ledger.record_collection("s_1", completed=True, spent=ModelUsage(), switch_reason="cache_cold")
        carried = ledger.collections_waiting("s_1")
        _request(ledger, "s_1")
        ledger.record_collection("s_1", completed=True, spent=ModelUsage(), switch_reason="requested")

        ledger.restore_waiting("s_1", carried)

        assert ledger.collections_waiting("s_1") == ("cache_cold", "requested"), "oldest first"
        assert _conversation(ledger, "s_1")["collections_not_yet_sent"] == 2

    def test_what_is_waiting_is_a_copy_the_next_request_cannot_empty(self):
        """The proxy reads what a request carries before the ledger counts it, which clears the waiting list."""
        ledger = _ledger()
        _request(ledger, "s_1")
        ledger.record_collection("s_1", completed=True, spent=ModelUsage(), switch_reason="repays_rewrite")

        carried = ledger.collections_waiting("s_1")
        _request(ledger, "s_1")

        assert carried == ("repays_rewrite",)
