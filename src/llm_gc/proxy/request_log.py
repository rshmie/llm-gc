"""Every request the proxy forwarded, one record each, for the session trace. Application code.

Structure only, never content: when it went and came back, which conversation it
belongs to, what kind its newest message is, the tokens it carried and the reply's
outline. Its newest message is identified by the conversation's chain hash through
it, so a later request of the same conversation settles whether it was kept - the
conversation carried on from it - or replaced: a side request, a regenerated reply,
an abandoned branch.
"""

from collections import OrderedDict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from llm_gc.models import ClientOrigin, Message
from llm_gc.models.message import said_by_user, tool_result_ids
from llm_gc.proxy.anthropic_translation import ReplyOutline

# Records kept in all; above this the oldest is dropped. Its counts stay in the ledger's totals.
MAX_RECORDS = 10_000
# Unsettled records watched per client session and agent: a conversation's newest request,
# a first request not yet claimed, and side requests nothing will continue.
MAX_PENDING_PER_ORIGIN = 64
MAX_ORIGINS_WATCHED = 1024

NewestKind = Literal["typed", "tool_results", "relayed", "assistant"]
_OriginKey = tuple[str | None, str | None]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def newest_kind(message: Message) -> NewestKind:
    """What a request's newest message is: words the person typed, tool output, another agent's report, or neither."""
    if said_by_user(message):
        return "typed"
    if message.relayed:
        return "relayed"
    return "tool_results" if tool_result_ids(message) else "assistant"


@dataclass(slots=True)
class RequestRecord:
    """One forwarded request: what it carried, and what came back."""

    request_id: str
    origin: ClientOrigin | None
    conversation_id: str | None    # None until recognised: a first request, or a side request never continued
    position: int                  # messages it carried; its newest is the last of them
    newest: NewestKind
    newest_hash: str               # the conversation's chain hash through its newest message
    model: str | None
    forwarded_at: datetime
    tokens_from_client: int
    tokens_forwarded: int
    collected: bool
    # True once a later request of its conversation carries its newest message, False once one
    # replaces it; None until then.
    kept: bool | None = None
    ended_at: datetime | None = None
    outline: ReplyOutline | None = None


class RequestLog:
    """Opens a record per request, settles earlier ones from later requests, closes each on its reply.

    Single-threaded: called from the proxy's event loop only.
    """

    def __init__(self, *, max_records: int = MAX_RECORDS, max_pending: int = MAX_PENDING_PER_ORIGIN,
                 max_origins: int = MAX_ORIGINS_WATCHED, clock: Callable[[], datetime] = _now) -> None:
        self._records: deque[RequestRecord] = deque(maxlen=max_records)
        self._pending: OrderedDict[_OriginKey, list[RequestRecord]] = OrderedDict()
        self._max_pending, self._max_origins = max_pending, max_origins
        self._clock = clock

    def begin(self, *, conversation_id: str | None, origin: ClientOrigin | None, hashes: Sequence[str],
              newest: NewestKind, model: str | None, tokens_from_client: int, tokens_forwarded: int,
              collected: bool) -> RequestRecord:
        """Open a record for a request about to be forwarded, and settle the earlier records it answers.

        `hashes` are the chain hashes of every message the request carries, in order.
        """
        if conversation_id is not None:
            self._settle(conversation_id, origin, hashes[:-1])
        record = RequestRecord(request_id=uuid4().hex[:12], origin=origin, conversation_id=conversation_id,
                               position=len(hashes), newest=newest, newest_hash=hashes[-1], model=model,
                               forwarded_at=self._clock(), tokens_from_client=tokens_from_client,
                               tokens_forwarded=tokens_forwarded, collected=collected)
        self._records.append(record)
        self._watch(origin, record)
        return record

    def finish(self, record: RequestRecord, outline: ReplyOutline) -> None:
        record.ended_at = self._clock()
        record.outline = outline

    def records(self) -> list[RequestRecord]:
        """Every record kept, in the order the requests were forwarded."""
        return list(self._records)

    def _settle(self, conversation_id: str, origin: ClientOrigin | None, history: Sequence[str]) -> None:
        """Settle what this request's history answers: each earlier request whose newest message it reaches.

        Kept when the history says the same through that message - which also claims a
        first request for this conversation - and replaced when it says otherwise. A
        record of no conversation that the history does not carry may be another
        conversation's first request, so it is left unsettled.
        """
        pending = self._pending.get(_key(origin))
        if pending is None:
            return
        waiting = []
        for record in pending:
            mine = record.conversation_id == conversation_id
            if not (mine or record.conversation_id is None) or record.position > len(history):
                waiting.append(record)
            elif history[record.position - 1] == record.newest_hash:
                record.kept, record.conversation_id = True, conversation_id
            elif mine:
                record.kept = False
            else:
                waiting.append(record)
        pending[:] = waiting

    def _watch(self, origin: ClientOrigin | None, record: RequestRecord) -> None:
        key = _key(origin)
        pending = self._pending.get(key)
        if pending is None:
            pending = self._pending[key] = []
            while len(self._pending) > self._max_origins:
                self._pending.popitem(last=False)
        else:
            self._pending.move_to_end(key)
        pending.append(record)
        # The oldest unsettled records stop being watched: side requests nothing will continue, mostly.
        del pending[:-self._max_pending]


def _key(origin: ClientOrigin | None) -> _OriginKey:
    return (origin.session_id, origin.agent_id) if origin is not None else (None, None)
