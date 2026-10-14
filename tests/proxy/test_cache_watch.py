"""Whether a conversation's cache is warm: cold only when the provider's own replies and the clock say so."""

from llm_gc.engine.switch_policy import CacheState
from llm_gc.proxy.cache_watch import EXPIRY_MARGIN_S, CacheWatch
from llm_gc.proxy.traffic import ProviderUsage

FIVE_MINUTES = 300.0
CACHED = ProviderUsage(replies=1, input_tokens=5, cache_read_input_tokens=60_000, cache_creation_input_tokens=1_700)
WRITTEN = ProviderUsage(replies=1, input_tokens=5, cache_creation_input_tokens=60_000)
UNCACHED = ProviderUsage(replies=1, input_tokens=60_000)


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def _watch() -> tuple[CacheWatch, _Clock]:
    clock = _Clock()
    return CacheWatch(clock=clock), clock


def _answered(watch, clock, usage=CACHED, conversation="c", lifetime=FIVE_MINUTES, took_s=2.0) -> float:
    """A request starts now and its reply arrives `took_s` later. Returns when it started."""
    started = watch.now()
    clock.now += took_s
    watch.observe(conversation, started_at=started, lifetime_s=lifetime, usage=usage)
    return started


class TestWarm:
    def test_a_request_inside_the_lifetime_reads_the_cache(self):
        watch, clock = _watch()
        _answered(watch, clock)
        clock.now += 60
        assert watch.state("c") is CacheState.WARM

    def test_the_first_request_writing_the_cache_counts_as_using_it(self):
        watch, clock = _watch()
        _answered(watch, clock, usage=WRITTEN)
        assert watch.state("c") is CacheState.WARM

    def test_a_conversation_with_no_answer_yet_counts_as_warm(self):
        """Nothing says cold, and a wrong cold costs a rewrite."""
        watch, _ = _watch()
        assert watch.state("never-seen") is CacheState.WARM

    def test_a_longer_lifetime_keeps_it_warm_through_a_long_pause(self):
        watch, clock = _watch()
        _answered(watch, clock, lifetime=3600.0)
        clock.now += 20 * 60
        assert watch.state("c") is CacheState.WARM


class TestCold:
    def test_idle_past_the_lifetime_is_cold(self):
        watch, clock = _watch()
        started = _answered(watch, clock)
        clock.now = started + FIVE_MINUTES + EXPIRY_MARGIN_S + 1
        assert watch.state("c") is CacheState.COLD

    def test_just_past_the_lifetime_is_still_warm_inside_the_margin(self):
        watch, clock = _watch()
        started = _answered(watch, clock)
        clock.now = started + FIVE_MINUTES + 1
        assert watch.state("c") is CacheState.WARM

    def test_the_lifetime_counts_from_the_request_start_not_the_reply_end(self):
        """A reply streamed for four minutes does not buy the entry four more."""
        watch, clock = _watch()
        started = _answered(watch, clock, took_s=240.0)
        clock.now = started + FIVE_MINUTES + EXPIRY_MARGIN_S + 1
        assert watch.state("c") is CacheState.COLD

    def test_a_conversation_whose_replies_never_used_the_cache_is_always_cold(self):
        """Every token is billed in full, so a smaller prompt is strictly cheaper."""
        watch, clock = _watch()
        _answered(watch, clock, usage=UNCACHED)
        assert watch.state("c") is CacheState.COLD


class TestAcrossRequests:
    def test_each_answered_request_restarts_the_lifetime(self):
        watch, clock = _watch()
        for _ in range(5):
            _answered(watch, clock)
            clock.now += 200
        assert watch.state("c") is CacheState.WARM

    def test_one_reply_too_short_to_cache_does_not_make_a_caching_conversation_cold(self):
        watch, clock = _watch()
        _answered(watch, clock)
        _answered(watch, clock, usage=UNCACHED)
        assert watch.state("c") is CacheState.WARM

    def test_a_late_reply_to_an_older_request_does_not_move_the_clock_back(self):
        watch, clock = _watch()
        older = watch.now()
        clock.now += 250
        newer = _answered(watch, clock)
        watch.observe("c", started_at=older, lifetime_s=FIVE_MINUTES, usage=CACHED)
        clock.now = newer + FIVE_MINUTES - 1
        assert watch.state("c") is CacheState.WARM

    def test_conversations_are_watched_apart(self):
        watch, clock = _watch()
        _answered(watch, clock, conversation="main")
        _answered(watch, clock, conversation="agent", usage=UNCACHED)
        assert (watch.state("main"), watch.state("agent")) == (CacheState.WARM, CacheState.COLD)


class TestBounds:
    def test_above_the_limit_the_least_recently_answered_is_forgotten_and_reads_as_unknown(self):
        clock = _Clock()
        watch = CacheWatch(max_conversations=2, clock=clock)
        for conversation in ("a", "b", "c"):
            watch.observe(conversation, started_at=clock.now, lifetime_s=FIVE_MINUTES, usage=UNCACHED)
        assert watch.state("a") is CacheState.WARM, "forgotten: unknown, so warm"
        assert watch.state("c") is CacheState.COLD
