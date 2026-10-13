"""LLM-GC's own model calls yield while the upstream is rate-limiting or overloaded."""

from datetime import datetime, timedelta, timezone

from llm_gc.engine.pause import DEFAULT_PAUSE, MAX_PAUSE, ModelCallPause


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now


def test_a_pause_lasts_as_long_as_the_upstream_asked():
    clock = _Clock()
    pause = ModelCallPause(clock=clock)

    assert pause.pause(30, "rate limited") is True
    assert pause.current().until == clock.now + timedelta(seconds=30)


def test_without_a_retry_after_a_pause_lasts_a_minute():
    clock = _Clock()
    pause = ModelCallPause(clock=clock)
    pause.pause(None, "overloaded")

    assert pause.current().until == clock.now + DEFAULT_PAUSE


def test_a_later_refusal_extends_a_pause_and_a_shorter_one_never_shortens_it():
    clock = _Clock()
    pause = ModelCallPause(clock=clock)
    pause.pause(120, "first")

    assert pause.pause(10, "second") is False, "a pause was already running"
    assert pause.current().until == clock.now + timedelta(seconds=120)
    pause.pause(300, "third")
    assert pause.current().until == clock.now + timedelta(seconds=300)


def test_no_pause_outlasts_the_cap():
    clock = _Clock()
    pause = ModelCallPause(clock=clock)
    pause.pause(86_400, "a day")

    assert pause.current().until == clock.now + MAX_PAUSE


def test_a_pause_ends_on_its_own():
    clock = _Clock()
    pause = ModelCallPause(clock=clock)
    pause.pause(30, "rate limited")
    clock.now += timedelta(seconds=31)

    assert pause.current() is None
