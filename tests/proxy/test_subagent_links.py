"""Linking a subagent to the tool call that started it, whichever reaches the proxy first."""

from datetime import datetime, timedelta, timezone

from llm_gc.models import ClientOrigin
from llm_gc.proxy.anthropic_translation import ReplyOutline, ToolCall
from llm_gc.proxy.request_log import RequestLog, SpawnRef
from llm_gc.proxy.subagent_links import SpawnLinker

COUNT = "Count the test files under tests/ and report the total."
README = "Summarise README.md in three lines for the user."


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now


class _Session:
    """A request log and a linker on one clock, for one Claude Code session."""

    def __init__(self) -> None:
        self.clock = _Clock()
        self.log = RequestLog(clock=self.clock)
        self.linker = SpawnLinker(clock=self.clock)

    def reply_calls(self, *prompts: str, session: str = "cc-1"):
        record = self.log.begin(
            conversation_id=None,
            origin=ClientOrigin(session),
            hashes=["m"],
            newest="typed",
            model="m",
            tokens_from_client=1,
            tokens_forwarded=1,
            collected=True,
        )
        calls = tuple(ToolCall(name="Agent", call_id=f"toolu_{i}") for i in range(len(prompts)))
        self.log.finish(record, ReplyOutline(status=200, finished=True, stop_reason="tool_use", tool_calls=calls))
        self.linker.offer_calls(session, record, {call.call_id: prompt for call, prompt in zip(calls, prompts)})
        return record

    def subagent_opens(self, prompt: str, agent: str, session: str = "cc-1"):
        record = self.log.begin(
            conversation_id=None,
            origin=ClientOrigin(session, agent),
            hashes=[agent],
            newest="typed",
            model="m",
            tokens_from_client=1,
            tokens_forwarded=1,
            collected=False,
        )
        # The client adds its own context around the prompt it was given.
        self.linker.offer_opening(session, record, f"<system-reminder>context</system-reminder>\n{prompt}\nYour task.")
        return record


def test_a_subagent_is_linked_to_the_call_whose_prompt_it_opens_with():
    s = _Session()
    spawner = s.reply_calls(COUNT, README)
    count, readme = s.subagent_opens(COUNT, "a1"), s.subagent_opens(README, "a2")

    assert count.spawned_by == SpawnRef(spawner.request_id, "toolu_0", "Agent")
    assert readme.spawned_by == SpawnRef(spawner.request_id, "toolu_1", "Agent")


def test_a_subagent_that_reaches_the_proxy_before_its_spawning_reply_ends_is_linked_too():
    """The reply's end and the subagent's first request land together, in no fixed order."""
    s = _Session()
    count = s.subagent_opens(COUNT, "a1")
    spawner = s.reply_calls(COUNT)

    assert count.spawned_by == SpawnRef(spawner.request_id, "toolu_0", "Agent")


def test_two_calls_with_the_same_prompt_start_their_subagents_in_order():
    s = _Session()
    s.reply_calls(COUNT, COUNT)
    first, second = s.subagent_opens(COUNT, "a1"), s.subagent_opens(COUNT, "a2")

    assert (first.spawned_by.call_id, second.spawned_by.call_id) == ("toolu_0", "toolu_1")


def test_the_longest_prompt_held_in_the_opening_wins():
    """A short prompt can sit inside a longer one's subagent by chance."""
    s = _Session()
    s.reply_calls("Count the test files under tests/", COUNT)
    subagent = s.subagent_opens(COUNT, "a1")

    assert subagent.spawned_by.call_id == "toolu_1"


def test_a_call_in_another_client_session_never_links():
    s = _Session()
    s.reply_calls(COUNT, session="cc-other")
    subagent = s.subagent_opens(COUNT, "a1")

    assert subagent.spawned_by is None


def test_a_subagent_whose_prompt_no_call_gave_stays_unlinked():
    s = _Session()
    s.reply_calls(README)

    assert s.subagent_opens(COUNT, "a1").spawned_by is None


def test_a_side_that_waits_past_the_window_is_let_go():
    s = _Session()
    s.reply_calls(COUNT)
    s.clock.now += timedelta(minutes=11)

    assert s.subagent_opens(COUNT, "a1").spawned_by is None
