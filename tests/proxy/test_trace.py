"""A client session's trace, built from the request log.

The main scenario follows a live Claude Code run in its order: a quota check and a title
request before anything is said, a first message whose reply starts two subagents, a side
request the conversation never continues, a subagent's report, and the person's next
message while a subagent is still at work. Chain hashes are written as the turns they
cover, so "abc" is the hash through turns a, b and c.
"""

from datetime import datetime, timedelta, timezone

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

    def send(self, conversation, turns, newest="typed", origin=MAIN, calls=(), status=200, carries=0, sent=100,
             forwarded=100):
        record = self.log.begin(conversation_id=conversation, origin=origin,
                                hashes=[turns[:end] for end in range(1, len(turns) + 1)], newest=newest, model="m",
                                tokens_from_client=sent, tokens_forwarded=forwarded, collected=origin.agent_id is None,
                                carries_collections=carries)
        self.log.finish(record, ReplyOutline(
            status=status, finished=status == 200, stop_reason="tool_use" if calls else "end_turn",
            usage=ProviderUsage(replies=1, cache_read_input_tokens=sent, output_tokens=10),
            tool_calls=tuple(ToolCall(name="Agent", call_id=call) for call in calls)))
        return record

    def trace(self, client_session="cc-1"):
        return build_trace(self.log.records(), client_session)


def _agent(name):
    return ClientOrigin(session_id="cc-1", agent_id=name)


def _kinds(rows):
    return [row["kind"] for row in rows]


def test_a_live_runs_requests_become_turns_and_the_subagents_each_turn_started():
    s = _Session()
    s.send(None, "q")                                                          # a quota check
    s.send(None, "z")                                                          # a title request
    first = s.send(None, "a", calls=("toolu_0", "toolu_1"))                    # "use two subagents"
    count = s.send(None, "P", origin=_agent("a1"))
    count.spawned_by = SpawnRef(first.request_id, "toolu_0", "Agent")
    readme = s.send(None, "S", origin=_agent("a2"))
    readme.spawned_by = SpawnRef(first.request_id, "toolu_1", "Agent")
    s.send("s_main", "abc", newest="tool_results")                             # both started
    s.send("s_a1", "PQR", newest="tool_results", origin=_agent("a1"))
    s.send("s_main", "abcdX")                                                   # a side request
    s.send("s_main", "abcdR", newest="relayed")                                 # subagent 1 reports
    s.send("s_main", "abcdRfG")                                                 # the person's next message
    s.send("s_a2", "STU", newest="tool_results", origin=_agent("a2"))           # subagent 2 still at work

    trace = s.trace()

    assert _kinds(trace["before"]) == ["unclaimed", "unclaimed"]
    first_turn, second_turn = trace["turns"]
    assert _kinds(first_turn["requests"]) == ["turn", "tool_round", "replaced", "report"]
    assert first_turn["requests"][0]["spawned"] == ["a1", "a2"]
    assert [(node["label"], node["spawned_by"]["call_id"], len(node["requests"])) for node in first_turn["subagents"]] == [
        ("Subagent 1", "toolu_0", 2), ("Subagent 2", "toolu_1", 2)]
    assert _kinds(first_turn["subagents"][0]["requests"]) == ["task", "tool_round"]
    assert first_turn["totals"]["requests"] == 8, "the turn costs its own requests and its subagents'"
    assert _kinds(second_turn["requests"]) == ["turn"] and second_turn["subagents"] == []


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
    s.send("s_new", "xyz")

    turns = s.trace()["turns"]
    assert [(t["part"], t["new_conversation"]) for t in turns] == [
        ("earlier_main", False), ("earlier_main", False), ("main", True)]


def test_a_typed_message_whose_place_another_request_took_is_not_a_turn_until_settled():
    s = _Session()
    s.send("s_1", "abc")
    s.send("s_1", "abcdX")
    s.send("s_1", "abcdR", newest="relayed")

    [turn] = s.trace()["turns"]
    assert _kinds(turn["requests"]) == ["turn", "pending", "report"]


def test_a_subagent_started_by_a_subagent_stays_in_the_turn_that_started_the_first():
    s = _Session()
    first = s.send("s_main", "abc", calls=("toolu_0",))
    parent = s.send(None, "P", origin=_agent("a1"), calls=("toolu_9",))
    parent.spawned_by = SpawnRef(first.request_id, "toolu_0", "Agent")
    s.send("s_main", "abcdef")                                                  # the person types again
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


def test_a_request_that_carried_a_collection_says_so_and_totals_add_the_providers_count():
    s = _Session()
    s.send("s_1", "abc", sent=1000, forwarded=1000)
    s.send("s_1", "abcde", newest="tool_results", carries=1, sent=1200, forwarded=400)

    [turn] = s.trace()["turns"]
    assert [row["carries_collections"] for row in turn["requests"]] == [0, 1]
    assert (turn["totals"]["tokens_saved"], turn["totals"]["cache_read_input_tokens"]) == (800, 2200)


def test_only_the_named_client_sessions_requests_are_traced():
    s = _Session()
    s.send("s_1", "abc")
    s.send("s_2", "xyz", origin=ClientOrigin("cc-2"))

    assert len(s.trace()["turns"]) == 1
    assert s.trace("cc-unknown") is None
