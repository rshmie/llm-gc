"""The traffic ledger: every request counted once, where it belongs, and nothing counted that did not happen."""

import json
from datetime import datetime, timedelta, timezone

from llm_gc.llm import ModelUsage
from llm_gc.models import ClientOrigin
from llm_gc.proxy.anthropic_translation import StreamUsageReader, fixed_prompt_tokens, reply_usage
from llm_gc.proxy.traffic import ProviderUsage, TrafficLedger
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


class TestReadingUsage:
    def test_an_unstreamed_replys_usage(self):
        raw = json.dumps({"type": "message", "usage": {"input_tokens": 3, "cache_read_input_tokens": 2000,
                                                       "cache_creation_input_tokens": None, "output_tokens": 50}})
        assert reply_usage(raw.encode()) == ProviderUsage(replies=1, input_tokens=3, cache_read_input_tokens=2000,
                                                          output_tokens=50)

    def test_an_error_body_or_garbage_reports_none(self):
        assert reply_usage(b'{"type": "error", "error": {}}') is None
        assert reply_usage(b"not json") is None

    def test_a_stream_split_anywhere_reads_the_same(self):
        """A chunk boundary can fall inside a line; a reader that parsed per chunk would lose the usage event."""
        events = [
            {"type": "message_start", "message": {"usage": {"input_tokens": 4, "cache_read_input_tokens": 1500,
                                                            "cache_creation_input_tokens": 60, "output_tokens": 1}}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta",
                                                                  "text": 'a "message_delta" in the text'}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 87}},
            {"type": "message_stop"},
        ]
        stream = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()
        expected = ProviderUsage(replies=1, input_tokens=4, cache_read_input_tokens=1500,
                                 cache_creation_input_tokens=60, output_tokens=87)
        for size in (1, 7, 64, len(stream)):
            reader = StreamUsageReader()
            for start in range(0, len(stream), size):
                reader.feed(stream[start:start + size])
            assert reader.usage() == expected, size

    def test_a_stream_cut_short_reports_the_prompt_it_was_billed_for(self):
        reader = StreamUsageReader()
        start = {"type": "message_start", "message": {"usage": {"input_tokens": 900, "output_tokens": 1}}}
        reader.feed(f"data: {json.dumps(start)}\n\n".encode())

        assert reader.usage() == ProviderUsage(replies=1, input_tokens=900, output_tokens=1)
