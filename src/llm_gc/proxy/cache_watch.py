"""Whether a conversation's prompt cache is warm, read from what crossed the proxy. Application code.

The engine's switch policy needs one fact it cannot see: would this conversation's next
request read the provider's cache, or miss it anyway? The proxy sees each request start and
each reply's usage, which is enough to tell. Vendor-neutral: the wire format's translator
supplies the lifetime a request asked for and the usage it reported.

See doc/design/generational-memory/when-to-collect.md.
"""

import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from llm_gc.engine.switch_policy import CacheReading, CacheState
from llm_gc.proxy.traffic import MAX_CONVERSATIONS, ProviderUsage

# How long past its lifetime an entry must be before it counts as gone. The proxy times a
# request from when it forwards it, the provider from when it receives it; the margin absorbs
# that jitter on the side of warm, because a wrong "cold" costs a rewrite and a wrong "warm"
# only a missed free switch.
EXPIRY_MARGIN_S = 5.0


@dataclass(slots=True)
class _Watched:
    last_used_at: float     # when the newest request the provider answered began, on the watch's clock
    lifetime_s: float       # the lifetime that request asked for
    uses_cache: bool        # any reply so far read from the cache or wrote to it


class CacheWatch:
    """Per conversation: when its cache was last used, and whether it is used at all.

    Single-threaded: called from the proxy's event loop only. Bounded: above
    `max_conversations` the least recently answered is forgotten, and reads as unknown.
    """

    def __init__(self, *, max_conversations: int = MAX_CONVERSATIONS,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._watched: OrderedDict[str, _Watched] = OrderedDict()
        self._max_conversations = max_conversations
        self._clock = clock

    def now(self) -> float:
        """The watch's clock, so a request's start is stamped on the same clock its gap is measured on."""
        return self._clock()

    def observe(self, conversation_id: str, *, started_at: float, lifetime_s: float, usage: ProviderUsage) -> None:
        """Record a reply the provider answered: when its request began, and whether it used the cache.

        The lifetime counts from the start of the request that last read or wrote the
        entry, not from the end of its reply. A reply that lands after a newer one moves
        nothing back.
        """
        used = usage.cache_read_input_tokens > 0 or usage.cache_creation_input_tokens > 0
        watched = self._watched.get(conversation_id)
        if watched is None:
            self._watched[conversation_id] = _Watched(started_at, lifetime_s, used)
            while len(self._watched) > self._max_conversations:
                self._watched.popitem(last=False)
            return
        self._watched.move_to_end(conversation_id)
        # Once a conversation has used the cache, a reply too short to be cached does not mean it stopped.
        watched.uses_cache = watched.uses_cache or used
        if started_at >= watched.last_used_at:
            watched.last_used_at, watched.lifetime_s = started_at, lifetime_s

    def state(self, conversation_id: str) -> CacheState:
        """Warm or cold for the request being built now. Unknown counts as warm."""
        return self.reading(conversation_id).state

    def reading(self, conversation_id: str) -> CacheReading:
        """The state for the request being built now, with what it was read from.

        Cold when the conversation's replies show it does not use the cache, or when it has
        been idle longer than its entry lives. With no answered request yet, nothing says
        either, and warm is the side whose error costs only an opportunity.
        """
        watched = self._watched.get(conversation_id)
        if watched is None:
            return CacheReading(CacheState.WARM)
        idle_s = self._clock() - watched.last_used_at
        cold = not watched.uses_cache or idle_s > watched.lifetime_s + EXPIRY_MARGIN_S
        return CacheReading(CacheState.COLD if cold else CacheState.WARM, uses_cache=watched.uses_cache,
                            idle_s=idle_s, lifetime_s=watched.lifetime_s)
