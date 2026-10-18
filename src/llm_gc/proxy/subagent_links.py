"""Which tool call started each subagent, matched by the prompt the call gave it. Application code.

A subagent's first message holds the prompt the spawning call gave it, inside the context
the client adds around it. Either side can reach the proxy first - the reply that made the
call, or the subagent's first request - so each waits for the other, briefly. The prompt
and the opening are conversation content: held in memory only until matched, never logged,
and only while something will show the link.
"""

from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from llm_gc.proxy.request_log import RequestRecord, SpawnRef

# How long a call or an opening waits for its other side. A subagent starts within a second
# of the reply that calls it; a side that waits longer has none.
LINK_WINDOW = timedelta(minutes=10)
# Calls and openings held per client session, and client sessions held. Above these, the oldest go.
MAX_HELD_PER_SESSION = 64
MAX_SESSIONS_HELD = 256


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class _Call:
    ref: SpawnRef
    prompt: str
    at: datetime


@dataclass(slots=True)
class _Opening:
    record: RequestRecord
    text: str
    at: datetime


@dataclass(slots=True)
class _Held:
    calls: list[_Call]
    openings: list[_Opening]


class SpawnLinker:
    """Pairs a subagent's first request with the tool call whose prompt it opens with.

    Single-threaded: called from the proxy's event loop only.
    """

    def __init__(
        self,
        *,
        window: timedelta = LINK_WINDOW,
        max_held: int = MAX_HELD_PER_SESSION,
        max_sessions: int = MAX_SESSIONS_HELD,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        self._held: OrderedDict[str, _Held] = OrderedDict()
        self._window, self._max_held, self._max_sessions = window, max_held, max_sessions
        self._clock = clock

    def offer_calls(self, client_session: str, record: RequestRecord, prompts: Mapping[str, str]) -> None:
        """The tool calls a reply made that gave a prompt, by call id, in the order the reply made them."""
        if record.outline is None or not prompts:
            return
        now = self._clock()
        held = self._session(client_session)
        held.calls += [
            _Call(SpawnRef(record.request_id, call.call_id, call.name), prompts[call.call_id], now)
            for call in record.outline.tool_calls
            if call.call_id in prompts
        ]
        self._match(held, now)

    def offer_opening(self, client_session: str, record: RequestRecord, text: str) -> None:
        """A subagent conversation's first request, and the text of its first message."""
        now = self._clock()
        held = self._session(client_session)
        held.openings.append(_Opening(record, text, now))
        self._match(held, now)

    def _session(self, client_session: str) -> _Held:
        held = self._held.get(client_session)
        if held is None:
            held = self._held[client_session] = _Held([], [])
            while len(self._held) > self._max_sessions:
                self._held.popitem(last=False)
        else:
            self._held.move_to_end(client_session)
        return held

    def _match(self, held: _Held, now: datetime) -> None:
        """Pair each opening with a call whose prompt it holds: the longest such prompt, then the earliest call.

        The longest first, because a short prompt can sit inside a longer one's subagent
        by chance; the earliest first, so two calls with the same prompt start their
        subagents in the order they were made.
        """
        fresh = now - self._window
        held.calls = [call for call in held.calls if call.at >= fresh][-self._max_held :]
        held.openings = [opening for opening in held.openings if opening.at >= fresh][-self._max_held :]
        pairs = sorted(
            ((call, opening) for call in held.calls for opening in held.openings if call.prompt in opening.text),
            key=lambda pair: (-len(pair[0].prompt), pair[0].at, pair[1].at),
        )
        used_calls: set[int] = set()
        used_openings: set[int] = set()
        for call, opening in pairs:
            if id(call) in used_calls or id(opening) in used_openings:
                continue
            opening.record.spawned_by = call.ref
            used_calls.add(id(call))
            used_openings.add(id(opening))
        held.calls = [call for call in held.calls if id(call) not in used_calls]
        held.openings = [opening for opening in held.openings if id(opening) not in used_openings]
