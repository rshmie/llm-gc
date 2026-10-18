"""The request log: one record per request, settled by its conversation's later requests.

Hashes here are written as the turns they cover - "abc" is the chain hash through turns
a, b and c - so a test reads as the conversation it describes.
"""

from datetime import datetime, timedelta, timezone

from llm_gc.models import ClientOrigin, Message
from llm_gc.proxy.anthropic_translation import ReplyOutline, ToolCall
from llm_gc.proxy.request_log import RequestLog, newest_kind
from tests.models.support import turn_with_typed_words

MAIN = ClientOrigin(session_id="cc-1")
AGENT = ClientOrigin(session_id="cc-1", agent_id="agent-a")


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


def _chain(turns: str) -> list[str]:
    return [turns[:end] for end in range(1, len(turns) + 1)]


def _begin(log: RequestLog, conversation: str | None, turns: str, origin: ClientOrigin | None = MAIN):
    return log.begin(
        conversation_id=conversation,
        origin=origin,
        hashes=_chain(turns),
        newest="typed",
        model="m",
        tokens_from_client=10,
        tokens_forwarded=10,
        collected=True,
    )


class TestALaterRequestSettlesAnEarlierOne:
    def test_a_first_request_is_claimed_and_kept_when_its_conversation_continues_it(self):
        log = RequestLog(clock=_Clock())
        first = _begin(log, None, "a")
        second = _begin(log, "s_1", "abc")

        assert (first.conversation_id, first.kept) == ("s_1", True)
        assert second.kept is None, "nothing has continued the newest request yet"

    def test_a_side_request_is_replaced_when_the_conversation_carries_on_without_it(self):
        """Run 2 of the trace: a request at position 6 the conversation never continued; a report took its place."""
        log = RequestLog(clock=_Clock())
        main = _begin(log, "s_1", "abcde")
        side = _begin(log, "s_1", "abcdeX")
        report = _begin(log, "s_1", "abcdeRfG")

        assert (main.kept, side.kept, report.kept) == (True, False, None)

    def test_a_retry_and_the_request_it_repeats_are_both_kept(self):
        """Run 1: a request failed during a login and was sent again; the trace shows the repeat by its hash."""
        log = RequestLog(clock=_Clock())
        failed = _begin(log, "s_1", "abc")
        log.finish(failed, ReplyOutline(status=401))
        retried = _begin(log, "s_1", "abc")
        _begin(log, "s_1", "abcde")

        assert failed.kept and retried.kept and failed.newest_hash == retried.newest_hash
        assert failed.outline == ReplyOutline(status=401)

    def test_two_conversations_that_open_alike_claim_only_their_own_first_request(self):
        log = RequestLog(clock=_Clock())
        agents_first = _begin(log, None, "t", origin=AGENT)
        _begin(log, "s_main", "tuv", origin=MAIN)
        assert agents_first.conversation_id is None

        _begin(log, "s_agent", "tuv", origin=AGENT)
        assert (agents_first.conversation_id, agents_first.kept) == ("s_agent", True)

    def test_a_request_no_conversation_carries_stays_unsettled(self):
        """A title or quota request opens alone and is never continued: neither kept nor known to be replaced."""
        log = RequestLog(clock=_Clock())
        title = _begin(log, None, "q")
        _begin(log, "s_1", "abc")

        assert (title.conversation_id, title.kept) == (None, None)

    def test_a_request_beyond_what_the_history_reaches_waits(self):
        log = RequestLog(clock=_Clock())
        ahead = _begin(log, "s_1", "abcd")
        _begin(log, "s_1", "ab")

        assert ahead.kept is None


class TestARecordCloses:
    def test_with_its_replys_outline_after_it_was_forwarded(self):
        log = RequestLog(clock=_Clock())
        record = _begin(log, None, "a")
        outline = ReplyOutline(
            status=200, finished=True, stop_reason="tool_use", tool_calls=(ToolCall(name="Agent", call_id="toolu_1"),)
        )

        log.finish(record, outline)

        assert record.outline == outline and record.ended_at > record.forwarded_at


class TestBounds:
    def test_the_oldest_records_are_dropped_above_the_limit(self):
        log = RequestLog(max_records=2, clock=_Clock())
        for turns in ("a", "ab", "abc"):
            _begin(log, "s_1", turns)
        assert [record.position for record in log.records()] == [2, 3]

    def test_the_oldest_unsettled_record_stops_being_watched_and_stays_unsettled(self):
        log = RequestLog(max_pending=1, clock=_Clock())
        first = _begin(log, None, "a")
        _begin(log, None, "q")
        _begin(log, "s_1", "abc")

        assert first.kept is None


def test_a_requests_newest_message_is_named_by_who_wrote_it():
    typed = Message(role="user", content="which runner?", turn_index=0)
    results = Message(
        role="user",
        content="ok",
        turn_index=2,
        raw_content=[{"type": "tool_result", "tool_use_id": "t", "content": "ok"}],
    )
    report = Message(role="user", content="<agent-message>done</agent-message>", turn_index=4, relayed=True)
    reply = Message(role="assistant", content="Here", turn_index=5)

    assert [newest_kind(m) for m in (typed, results, report, reply)] == [
        "typed",
        "tool_results",
        "relayed",
        "assistant",
    ]


def test_words_typed_after_interrupting_a_tool_round_are_typed():
    """The person's turn in the session trace starts there: what follows answers their new words."""
    assert newest_kind(turn_with_typed_words(6, "t", "Stop, use MySQL.")) == "typed"


def test_a_newest_message_the_client_wrote_all_of_is_not_typed():
    """A background task finishing sends a request the person did not type."""
    notice = Message(
        role="user", content="<task-notification>done</task-notification>", turn_index=6, client_written=((0, 43),)
    )
    assert newest_kind(notice) == "client"


class TestAnOpenToolRound:
    """Whether a conversation's next request will answer tool calls, from its own records."""

    def test_open_while_unanswered_and_after_a_reply_that_called_tools(self):
        log = RequestLog()
        record = _begin(log, "s_1", "ab")
        assert log.tool_round_open("s_1") is True, "unsure while the reply is on its way"

        log.finish(record, ReplyOutline(status=200, finished=True, stop_reason="tool_use"))
        assert log.tool_round_open("s_1") is True

        log.finish(_begin(log, "s_1", "abcd"), ReplyOutline(status=200, finished=True, stop_reason="end_turn"))
        assert log.tool_round_open("s_1") is False

    def test_a_refused_request_is_passed_over(self):
        log = RequestLog()
        log.finish(_begin(log, "s_1", "ab"), ReplyOutline(status=200, finished=True, stop_reason="tool_use"))
        log.finish(_begin(log, "s_1", "abcd"), ReplyOutline(status=429))

        assert log.tool_round_open("s_1") is True, "the refused request answered nothing"

    def test_a_conversation_with_no_requests_has_no_round_open(self):
        assert RequestLog().tool_round_open("s_9") is False


class TestARecordNamesItsRequest:
    def test_a_record_takes_the_client_requests_id(self):
        record = RequestLog(clock=_Clock()).begin(
            conversation_id=None,
            origin=MAIN,
            hashes=_chain("a"),
            newest="typed",
            model="m",
            tokens_from_client=10,
            tokens_forwarded=10,
            collected=True,
            request_id="3f2a9c01b7de",
        )

        assert record.request_id == "3f2a9c01b7de"

    def test_a_record_opened_outside_a_request_gets_an_id_of_its_own(self):
        log = RequestLog(clock=_Clock())

        assert _begin(log, None, "a").request_id != _begin(log, None, "b").request_id

    def test_two_records_that_agree_in_every_field_are_still_two(self):
        """A record is one forward, settled as later requests arrive: who it is never depends on its fields."""
        log = RequestLog(clock=lambda: datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc))
        first = log.begin(
            conversation_id="s_1",
            origin=MAIN,
            hashes=_chain("ab"),
            newest="typed",
            model="m",
            tokens_from_client=10,
            tokens_forwarded=10,
            collected=True,
            request_id="3f2a9c01b7de",
        )
        second = log.begin(
            conversation_id="s_1",
            origin=MAIN,
            hashes=_chain("ab"),
            newest="typed",
            model="m",
            tokens_from_client=10,
            tokens_forwarded=10,
            collected=True,
            request_id="3f2a9c01b7de",
        )

        assert first != second and len({first, second}) == 2
