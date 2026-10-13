"""What crossed the proxy, measured: per conversation, per client session, and in all.

Application code. Counts, ids and model names only, never conversation content. Every
figure here happened - a request sent, a reply's reported usage, a collection's own
model calls. What the next request would carry is the engine's forecast, not this.
"""

from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, TypeVar

from llm_gc.llm.metering import ModelUsage
from llm_gc.models import ClientOrigin
from llm_gc.proxy.model_windows import ModelWindow

# Conversations and client sessions whose counts are kept. Above these, the least recently
# active is dropped; its requests stay counted in the totals above it.
MAX_CONVERSATIONS = 1024
MAX_CLIENT_SESSIONS = 256
# A conversation's first request belongs to no conversation yet: it is recognised on the
# second, when the history it opened comes back. Held until then, up to this many.
MAX_UNCLAIMED_OPENINGS = 256

_K = TypeVar("_K")
_V = TypeVar("_V")


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class ProviderUsage:
    """The `usage` the provider reported on its replies: what it bills, by its own tokenizer."""

    replies: int = 0
    input_tokens: int = 0                  # read at the full price
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    output_tokens: int = 0

    @classmethod
    def from_reported(cls, usage: Mapping[str, Any]) -> "ProviderUsage":
        """One reply's usage; a field the provider left out or sent as null counts as zero."""
        def tokens(key: str) -> int:
            value = usage.get(key)
            return value if isinstance(value, int) and value >= 0 else 0

        return cls(replies=1, input_tokens=tokens("input_tokens"),
                   cache_read_input_tokens=tokens("cache_read_input_tokens"),
                   cache_creation_input_tokens=tokens("cache_creation_input_tokens"),
                   output_tokens=tokens("output_tokens"))

    def add(self, other: "ProviderUsage") -> None:
        self.replies += other.replies
        self.input_tokens += other.input_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        self.output_tokens += other.output_tokens


@dataclass(slots=True)
class Traffic:
    """Requests through the proxy and what they carried, for one conversation, one client session, or all."""

    requests: int = 0
    tokens_from_client: int = 0    # each request whole as the client sent it: system prompt, tools and messages
    tokens_forwarded: int = 0      # the same requests as LLM-GC forwarded them
    provider: ProviderUsage = field(default_factory=ProviderUsage)
    summaries: ModelUsage = field(default_factory=ModelUsage)   # LLM-GC's own model calls

    @property
    def tokens_saved(self) -> int:
        return self.tokens_from_client - self.tokens_forwarded

    def add(self, other: "Traffic") -> None:
        self.requests += other.requests
        self.tokens_from_client += other.tokens_from_client
        self.tokens_forwarded += other.tokens_forwarded
        self.provider.add(other.provider)
        self.summaries.add(other.summaries)

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "tokens_saved": self.tokens_saved}


@dataclass(slots=True)
class ConversationTraffic:
    """One conversation's measured traffic."""

    conversation_id: str
    origin: ClientOrigin | None
    model: str | None                  # as its newest request named it
    context_window: ModelWindow | None  # that model's window, as its recap is sized for it
    collected: bool                    # False: recorded and shown, never collected
    first_request_at: datetime
    last_request_at: datetime
    traffic: Traffic = field(default_factory=Traffic)
    collections: int = 0
    # Collections made since its newest request. Their saving reaches the model only with the next one.
    collections_not_yet_sent: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"conversation_id": self.conversation_id,
                "agent": self.origin.agent_id if self.origin is not None else None,
                "model": self.model,
                "context_window": asdict(self.context_window) if self.context_window is not None else None,
                "collected": self.collected,
                "first_request_at": self.first_request_at.isoformat(),
                "last_request_at": self.last_request_at.isoformat(), "collections": self.collections,
                "collections_not_yet_sent": self.collections_not_yet_sent, "traffic": self.traffic.as_dict()}


@dataclass(slots=True)
class ClientSessionTraffic:
    """Every request that named one client session, whether or not a conversation claimed it."""

    client_session: str
    first_request_at: datetime
    last_request_at: datetime
    traffic: Traffic = field(default_factory=Traffic)


@dataclass(slots=True)
class _Unclaimed:
    first_request_at: datetime
    traffic: Traffic


# Where one request's numbers are added: the totals it belongs to, most general first.
RequestTargets = tuple[Traffic, ...]


class TrafficLedger:
    """Counts each request where it belongs. Single-threaded: called from the proxy's event loop only."""

    def __init__(self, *, max_conversations: int = MAX_CONVERSATIONS, max_client_sessions: int = MAX_CLIENT_SESSIONS,
                 max_unclaimed: int = MAX_UNCLAIMED_OPENINGS, clock: Callable[[], datetime] = _now) -> None:
        self.total = Traffic()
        self._conversations: OrderedDict[str, ConversationTraffic] = OrderedDict()
        self._client_sessions: OrderedDict[str, ClientSessionTraffic] = OrderedDict()
        self._unclaimed: OrderedDict[tuple[str, ClientOrigin | None], _Unclaimed] = OrderedDict()
        self._limits = (max_conversations, max_client_sessions, max_unclaimed)
        self._clock = clock

    def record_request(self, *, conversation_id: str | None, origin: ClientOrigin | None, model: str | None,
                       opening: str, tokens_from_client: int, tokens_forwarded: int,
                       context_window: ModelWindow | None = None, collected: bool = True) -> RequestTargets:
        """Count one forwarded request. Returns where its reply's usage is to be added.

        `opening` identifies the conversation's first turn. A request with no conversation
        yet is held under it, with its origin, until a conversation opening the same way
        appears and claims it.
        """
        now = self._clock()
        targets = [self.total]
        if origin is not None:
            targets.append(self._client_session(origin.session_id, now).traffic)
        if conversation_id is not None:
            conversation = self._conversation(conversation_id, origin, model, opening, now)
            conversation.context_window = context_window or conversation.context_window
            conversation.collected = collected
            # This request carries every collection made so far.
            conversation.collections_not_yet_sent = 0
            targets.append(conversation.traffic)
        else:
            unclaimed = self._unclaimed.get((opening, origin))
            if unclaimed is None:
                unclaimed = self._unclaimed[(opening, origin)] = _Unclaimed(now, Traffic())
                _bound(self._unclaimed, self._limits[2])
            targets.append(unclaimed.traffic)

        request = Traffic(requests=1, tokens_from_client=tokens_from_client, tokens_forwarded=tokens_forwarded)
        for target in targets:
            target.add(request)
        return tuple(targets)

    @staticmethod
    def record_usage(targets: RequestTargets, usage: ProviderUsage) -> None:
        """Add what the provider reported for one reply to everything its request was counted in."""
        for target in targets:
            target.provider.add(usage)

    def session_summaries(self, client_session: str) -> ModelUsage | None:
        """LLM-GC's own model calls for one client session's conversations, or None for a session not counted."""
        client = self._client_sessions.get(client_session)
        return client.traffic.summaries if client is not None else None

    def collections_waiting(self, conversation_id: str) -> int:
        """Collections made since the conversation's newest request: the next request carries them."""
        conversation = self._conversations.get(conversation_id)
        return conversation.collections_not_yet_sent if conversation is not None else 0

    def record_collection(self, conversation_id: str, *, completed: bool, spent: ModelUsage) -> None:
        """Count a collection's own model calls, and the collection itself when one was made."""
        self.total.summaries.add(spent)
        conversation = self._conversations.get(conversation_id)
        if conversation is None:
            return
        conversation.traffic.summaries.add(spent)
        if conversation.origin is not None and (client := self._client_sessions.get(conversation.origin.session_id)):
            client.traffic.summaries.add(spent)
        if completed:
            conversation.collections += 1
            conversation.collections_not_yet_sent += 1

    def as_dict(self) -> dict[str, Any]:
        """Every count, grouped by client session, the most recently active first."""
        conversations = sorted(self._conversations.values(), key=lambda c: c.last_request_at, reverse=True)
        sessions = []
        for client in sorted(self._client_sessions.values(), key=lambda c: c.last_request_at, reverse=True):
            members = [c for c in conversations if c.origin is not None and c.origin.session_id == client.client_session]
            listed = Traffic()
            for member in members:
                listed.add(member.traffic)
            sessions.append({"client_session": client.client_session,
                             "first_request_at": client.first_request_at.isoformat(),
                             "last_request_at": client.last_request_at.isoformat(),
                             "traffic": client.traffic.as_dict(),
                             # First requests no conversation claimed (side requests), and any dropped above the limit.
                             "outside_conversations": _difference(client.traffic, listed).as_dict(),
                             "conversations": [{**member.as_dict(), "part": part} for member, part in _parts(members)]})
        # A conversation whose client session was dropped above the limit is listed here rather than lost.
        return {"client_sessions": sessions,
                "other_conversations": [c.as_dict() for c in conversations
                                        if c.origin is None or c.origin.session_id not in self._client_sessions]}

    def _conversation(self, conversation_id: str, origin: ClientOrigin | None, model: str | None, opening: str,
                      now: datetime) -> ConversationTraffic:
        conversation = self._conversations.get(conversation_id)
        if conversation is None:
            claimed = self._unclaimed.pop((opening, origin), None)
            conversation = ConversationTraffic(
                conversation_id=conversation_id, origin=origin, model=model, context_window=None, collected=True,
                first_request_at=claimed.first_request_at if claimed is not None else now, last_request_at=now,
                # The same object, so a reply to the first request still on its way lands here too.
                traffic=claimed.traffic if claimed is not None else Traffic())
            self._conversations[conversation_id] = conversation
            _bound(self._conversations, self._limits[0])
            return conversation
        self._conversations.move_to_end(conversation_id)
        conversation.last_request_at = now
        conversation.model = model or conversation.model
        conversation.origin = conversation.origin or origin
        return conversation

    def _client_session(self, client_session: str, now: datetime) -> ClientSessionTraffic:
        client = self._client_sessions.get(client_session)
        if client is None:
            client = self._client_sessions[client_session] = ClientSessionTraffic(client_session, now, now)
            _bound(self._client_sessions, self._limits[1])
        else:
            self._client_sessions.move_to_end(client_session)
            client.last_request_at = now
        return client


def _parts(members: list[ConversationTraffic]) -> list[tuple[ConversationTraffic, str]]:
    """Each conversation's part in its client session, in the order a reader takes them in.

    The newest main conversation is the one in use; an older one was replaced - after
    /compact, or an edited message - and nothing continues it. Then the subagents, in
    the order they started, and the replaced main conversations, newest first.
    """
    by_start = sorted(members, key=lambda c: c.first_request_at)
    mains = [c for c in by_start if c.origin is None or c.origin.agent_id is None]
    agents = [c for c in by_start if c.origin is not None and c.origin.agent_id is not None]
    return ([(c, "main") for c in mains[-1:]] + [(c, "subagent") for c in agents]
            + [(c, "earlier_main") for c in reversed(mains[:-1])])


def _bound(entries: "OrderedDict[_K, _V]", limit: int) -> None:
    while len(entries) > limit:
        entries.popitem(last=False)


def _difference(whole: Traffic, part: Traffic) -> Traffic:
    def minus(a: _V, b: _V) -> _V:
        return type(a)(**{f.name: getattr(a, f.name) - getattr(b, f.name) for f in fields(a)})  # type: ignore[arg-type]

    return Traffic(requests=whole.requests - part.requests,
                   tokens_from_client=whole.tokens_from_client - part.tokens_from_client,
                   tokens_forwarded=whole.tokens_forwarded - part.tokens_forwarded,
                   provider=minus(whole.provider, part.provider), summaries=minus(whole.summaries, part.summaries))
