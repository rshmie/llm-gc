"""A client session's trace, built from the request log.

The main scenario follows a live Claude Code run in its order: a quota check and a title
request before anything is said, a first message whose reply starts two subagents, a side
request the conversation never continues, a subagent's report, and the person's next
message while a subagent is still at work. Chain hashes are written as the turns they
cover, so "abc" is the hash through turns a, b and c.
"""

from datetime import datetime, timedelta, timezone

from llm_gc.llm import ModelUsage
from llm_gc.models import ClientOrigin
from llm_gc.proxy.anthropic_translation import ReplyOutline, ToolCall
from llm_gc.proxy.request_log import RequestLog, SpawnRef
from llm_gc.proxy.traffic import ProviderUsage
from llm_gc.proxy.trace import build_trace

MAIN = ClientOrigin(session_id="cc-1")


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


class _Session:
    def __init__(self) -> None:
        self.log = RequestLog(clock=_Clock())

    def send(
        self,
        conversation,
        turns,
        newest="typed",
        origin=MAIN,
        calls=(),
        status=200,
        reasons=(),
        sent=100,
        forwarded=100,
        request_id=None,
        rewritten=None,
        cut_short=False,
    ):
        record = self.log.begin(
            conversation_id=conversation,
            origin=origin,
            hashes=[turns[:end] for end in range(1, len(turns) + 1)],
            newest=newest,
            model="m",
            tokens_from_client=sent,
            tokens_forwarded=forwarded,
            collected=origin.agent_id is None,
            switch_reasons=reasons,
            rewritten_tokens=rewritten,
            request_id=request_id,
        )
        self.log.finish(
            record,
            ReplyOutline(
                status=status,
                finished=status == 200 and not cut_short,
                stop_reason="tool_use" if calls else "end_turn",
                usage=ProviderUsage(
                    replies=1, cache_read_input_tokens=sent, cache_creation_input_tokens=sent, output_tokens=10
                ),
                tool_calls=tuple(ToolCall(name="Agent", call_id=call) for call in calls),
            ),
        )
        return record

    def trace(self, client_session="cc-1"):
        return build_trace(self.log.records(), client_session)


def _agent(name):
    return ClientOrigin(session_id="cc-1", agent_id=name)


def _kinds(rows):
    return [row["kind"] for row in rows]


def test_a_live_runs_requests_become_turns_and_the_subagents_each_turn_started():
    s = _Session()
    s.send(None, "q")  # a quota check
    s.send(None, "z")  # a title request
    first = s.send(None, "a", calls=("toolu_0", "toolu_1"))  # "use two subagents"
    count = s.send(None, "P", origin=_agent("a1"))
    count.spawned_by = SpawnRef(first.request_id, "toolu_0", "Agent")
    readme = s.send(None, "S", origin=_agent("a2"))
    readme.spawned_by = SpawnRef(first.request_id, "toolu_1", "Agent")
    s.send("s_main", "abc", newest="tool_results")  # both started
    s.send("s_a1", "PQR", newest="tool_results", origin=_agent("a1"))
    s.send("s_main", "abcdX")  # a side request
    s.send("s_main", "abcdR", newest="relayed")  # subagent 1 reports
    s.send("s_main", "abcdRfG")  # the person's next message
    s.send("s_a2", "STU", newest="tool_results", origin=_agent("a2"))  # subagent 2 still at work

    trace = s.trace()

    assert _kinds(trace["before"]) == ["unclaimed", "unclaimed"]
    first_turn, second_turn = trace["turns"]
    assert _kinds(first_turn["requests"]) == ["turn", "tool_round", "replaced", "report"]
    assert first_turn["requests"][0]["spawned"] == ["a1", "a2"]
    assert [
        (node["label"], node["spawned_by"]["call_id"], len(node["requests"])) for node in first_turn["subagents"]
    ] == [("Subagent 1", "toolu_0", 2), ("Subagent 2", "toolu_1", 2)]
    assert _kinds(first_turn["subagents"][0]["requests"]) == ["task", "tool_round"]
    assert first_turn["totals"]["requests"] == 8, "the turn costs its own requests and its subagents'"
    assert _kinds(second_turn["requests"]) == ["turn"] and second_turn["subagents"] == []


def test_a_tool_round_the_person_interrupted_is_shown_in_its_turn_not_as_a_side_request():
    """Live run: the person pressed Esc during a reply to tool results and typed. Claude Code sent
    the results again in the same place, with its mark and their words, which replaced the request."""
    s = _Session()
    s.send("s_1", "a", calls=("toolu_0",))
    s.send("s_1", "abc", newest="tool_results", cut_short=True)  # the reply stopped by Esc
    s.send("s_1", "abC")  # the same results, Claude Code's mark and the person's words
    s.send("s_1", "abCde")  # carried on, which settles the stopped request as replaced

    first_turn, second_turn, _ = s.trace()["turns"]
    assert _kinds(first_turn["requests"]) == ["turn", "interrupted"]
    assert _kinds(second_turn["requests"]) == ["turn"]


def test_a_replaced_request_is_interrupted_only_if_it_answered_tool_results_and_was_cut_short():
    """A side request is never the conversation's own tool round, and a reply that reached its end
    was not interrupted, whatever replaced it later."""
    s = _Session()
    s.send("s_1", "a")
    s.send("s_1", "abX", cut_short=True)  # a suggestion the client stopped waiting for
    s.send("s_1", "abY", newest="tool_results")  # a tool round later rewritten, its reply whole
    s.send("s_1", "abZ")
    s.send("s_1", "abZde")

    assert _kinds(s.trace()["turns"][0]["requests"]) == ["turn", "replaced", "replaced"]


def test_a_retry_is_shown_as_one_and_does_not_start_a_turn():
    s = _Session()
    s.send("s_1", "abc", status=401)
    s.send("s_1", "abc")
    s.send("s_1", "abcde")

    first_turn, second_turn = s.trace()["turns"]
    assert _kinds(first_turn["requests"]) == ["turn", "retry"]
    assert first_turn["requests"][0]["status"] == 401
    assert _kinds(second_turn["requests"]) == ["turn"]


def test_a_new_main_conversation_after_compact_is_marked_and_the_old_one_is_earlier():
    s = _Session()
    s.send("s_old", "abc")
    s.send("s_old", "abcde")
    s.send("s_old", "abcdeSU")  # /compact asks for a summary
    s.send("s_new", "xyz")

    turns = s.trace()["turns"]
    assert [(t["part"], t["new_conversation"]) for t in turns] == [
        ("earlier_main", False),
        ("earlier_main", False),
        ("main", True),
    ]
    assert _kinds(turns[1]["requests"]) == ["turn", "closing"], "the summary request was not typed by the person"


def test_a_typed_message_whose_place_another_request_took_is_not_a_turn_until_settled():
    s = _Session()
    s.send("s_1", "abc")
    s.send("s_1", "abcdX")
    s.send("s_1", "abcdR", newest="relayed")

    [turn] = s.trace()["turns"]
    assert _kinds(turn["requests"]) == ["turn", "pending", "report"]


def test_a_request_the_client_sent_on_its_own_is_a_notice_within_the_turn():
    """A background task finishing makes Claude Code send a request: the person typed nothing."""
    s = _Session()
    s.send("s_1", "abc")
    s.send("s_1", "abcdN", newest="client")

    [turn] = s.trace()["turns"]
    assert _kinds(turn["requests"]) == ["turn", "notice"]


def test_a_subagent_started_by_a_subagent_stays_in_the_turn_that_started_the_first():
    s = _Session()
    first = s.send("s_main", "abc", calls=("toolu_0",))
    parent = s.send(None, "P", origin=_agent("a1"), calls=("toolu_9",))
    parent.spawned_by = SpawnRef(first.request_id, "toolu_0", "Agent")
    s.send("s_main", "abcdef")  # the person types again
    child = s.send(None, "C", origin=_agent("a2"))
    child.spawned_by = SpawnRef(parent.request_id, "toolu_9", "Agent")

    first_turn, second_turn = s.trace()["turns"]
    assert [node["agent"] for node in first_turn["subagents"]] == ["a1", "a2"]
    assert first_turn["subagents"][0]["requests"][0]["spawned"] == ["a2"]
    assert second_turn["subagents"] == []


def test_a_subagent_no_call_is_known_to_have_started_falls_in_the_turn_of_its_first_request():
    s = _Session()
    s.send("s_main", "abc")
    s.send("s_main", "abcde")
    s.send(None, "P", origin=_agent("a1"))

    turns = s.trace()["turns"]
    assert [len(t["subagents"]) for t in turns] == [0, 1]
    assert turns[1]["subagents"][0]["spawned_by"] is None


def test_a_tools_own_model_call_inside_a_subagent_is_not_taken_for_its_task():
    """WebFetch reads a page with a model call of its own, which carries the subagent's id but no conversation."""
    s = _Session()
    first = s.send("s_main", "abc", calls=("toolu_0",))
    opening = s.send(None, "P", origin=_agent("a1"))
    opening.spawned_by = SpawnRef(first.request_id, "toolu_0", "Agent")
    s.send(None, "W", origin=_agent("a1"))  # WebFetch reads a page

    [turn] = s.trace()["turns"]
    assert _kinds(turn["subagents"][0]["requests"]) == ["task", "unclaimed"], "the opening is known by its call"

    s.send("s_a1", "PQR", newest="tool_results", origin=_agent("a1"))
    [turn] = s.trace()["turns"]
    assert _kinds(turn["subagents"][0]["requests"]) == ["task", "unclaimed", "tool_round"]


def test_a_request_that_carried_a_collection_says_so_and_totals_add_the_providers_count():
    s = _Session()
    s.send("s_1", "abc", sent=1000, forwarded=1000)
    s.send("s_1", "abcde", newest="tool_results", reasons=("repays_rewrite",), sent=1200, forwarded=400, rewritten=350)

    [turn] = s.trace()["turns"]
    assert [row["carries_collections"] for row in turn["requests"]] == [0, 1]
    assert [row["switch_reasons"] for row in turn["requests"]] == [[], ["repays_rewrite"]], "and why it was switched in"
    assert [row["rewritten_tokens"] for row in turn["requests"]] == [None, 350], "and what it expected cached again"
    assert (turn["totals"]["tokens_saved"], turn["totals"]["cache_read_input_tokens"]) == (800, 2200)


def test_only_the_named_client_sessions_requests_are_traced():
    s = _Session()
    s.send("s_1", "abc")
    s.send("s_2", "xyz", origin=ClientOrigin("cc-2"))

    assert len(s.trace()["turns"]) == 1
    assert s.trace("cc-unknown") is None


def test_the_cache_writes_of_requests_that_carried_a_collection_are_totalled_as_the_provider_counted_them():
    """A collection changes the recap, so the prompt after it is cached again: the cost a token count hides."""
    s = _Session()
    s.send("s_1", "abc", sent=1000)
    s.send("s_1", "abcde", newest="tool_results", reasons=("repays_rewrite",), sent=1200, forwarded=400)
    s.send("s_1", "abcdefg", newest="tool_results", sent=1300, forwarded=500)

    [turn] = s.trace()["turns"]
    # The test session reports a cache write of each request's size, so the collection's is 1,200.
    assert turn["totals"]["collection_cache_write_tokens"] == 1200


def test_llm_gcs_own_calls_for_the_session_are_carried_with_its_trace():
    s = _Session()
    s.send("s_1", "abc")

    trace = build_trace(s.log.records(), "cc-1", summaries=ModelUsage(calls=2, input_tokens=10_000, output_tokens=900))
    assert trace["summaries"] == {"calls": 2, "input_tokens": 10_000, "output_tokens": 900}


class TestARequestSentTwice:
    """A request whose rewrite the provider refused is sent again as written: two records, one client request."""

    def test_both_attempts_show_under_its_one_id_and_keep_their_own_kinds(self):
        s = _Session()
        s.send("s_1", "abc")
        s.send("s_1", "abcde", status=400, request_id="3f2a9c01b7de")
        s.send("s_1", "abcde", request_id="3f2a9c01b7de")

        refused, resent = s.trace()["turns"][-1]["requests"]
        assert (refused["request_id"], resent["request_id"]) == ("3f2a9c01b7de", "3f2a9c01b7de")
        assert (refused["kind"], resent["kind"]) == ("turn", "retry")

    def test_a_subagent_shows_under_the_attempt_whose_reply_started_it(self):
        """Not under the refused attempt too: it had no reply, so it started nothing."""
        s = _Session()
        s.send("s_1", "abc")
        s.send("s_1", "abcde", status=400, request_id="3f2a9c01b7de")
        s.send("s_1", "abcde", calls=("toolu_0",), request_id="3f2a9c01b7de")
        task = s.send(None, "P", origin=_agent("a1"))
        task.spawned_by = SpawnRef("3f2a9c01b7de", "toolu_0", "Agent")

        refused, resent = s.trace()["turns"][-1]["requests"]
        assert (refused["spawned"], resent["spawned"]) == ([], ["a1"])
