"""Each conversation as its client last sent it, kept for the dashboard.

The engine's spans hold the words of every turn the session has recorded. This
copy covers what the session does not have yet: the newest message, recorded only
with its reply, a conversation's opening words for the session picker, and the
instructions the client puts inside each message now - a client can rewrite them
after the turn was recorded, and they are sent as it sends them - and the newest
reply, which the proxy reads as it reaches the client and the session records
only when the next request carries it. The
client sends the whole conversation on every request, so the proxy hands each
request's history to this recorder.

Held only while the dashboard runs, in memory, as text, bounded by session count,
and dropped when its session closes.
"""

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

from llm_gc.models import Message
from llm_gc.utils import count_tokens

# Sessions whose transcript is kept. Above this, the least recently active is
# dropped; a transcript is the whole conversation, so the bound is deliberately low.
DEFAULT_TRANSCRIPT_MAX_SESSIONS = 50


@dataclass(frozen=True, slots=True)
class TranscriptTurn:
    turn_index: int
    role: str
    text: str     # the turn's text; empty for a turn that is only tool calls or attachments
    tokens: int
    instruction_tokens: int = 0   # the client's instructions inside the turn as it sends them now


class TranscriptRecorder:
    """The latest client history per session. Feed it from the proxy; read it from the dashboard."""

    def __init__(self, max_sessions: int = DEFAULT_TRANSCRIPT_MAX_SESSIONS) -> None:
        self._max_sessions = max_sessions
        self._transcripts: OrderedDict[str, tuple[TranscriptTurn, ...]] = OrderedDict()
        self._replies: dict[str, TranscriptTurn] = {}

    def record(self, session_id: str, messages: Sequence[Message]) -> None:
        """Replace the session's transcript with what the client just sent.

        Replaced rather than appended, so an edited or regenerated conversation
        shows what the client holds now. Text only: keeping `raw_content` would hold
        every attached image in memory for the life of the session.
        """
        self._transcripts[session_id] = tuple(
            TranscriptTurn(turn_index=message.turn_index, role=message.role, text=message.content,
                           tokens=message.token_count,
                           instruction_tokens=sum(count_tokens(text) for text in message.instructions))
            for message in messages
        )
        # The reply it showed is in this history now, or the client moved on without it.
        self._replies.pop(session_id, None)
        self._transcripts.move_to_end(session_id)
        if len(self._transcripts) > self._max_sessions:
            self.forget(next(iter(self._transcripts)))

    def record_reply(self, session_id: str, answering: Message, text: str) -> None:
        """Keep a finished reply to show, while it still answers the newest message.

        A client can have several requests in flight; a slower reply to an earlier
        one is dropped rather than shown under a message it does not answer.
        """
        turns = self._transcripts.get(session_id)
        newest = turns[-1] if turns else None
        if newest is None or newest.turn_index != answering.turn_index or newest.text != answering.content:
            return
        self._replies[session_id] = TranscriptTurn(turn_index=answering.turn_index + 1, role="assistant", text=text,
                                                   tokens=count_tokens(text))

    def get(self, session_id: str) -> tuple[TranscriptTurn, ...] | None:
        return self._transcripts.get(session_id)

    def reply(self, session_id: str) -> TranscriptTurn | None:
        """The newest reply, until the next request records it."""
        return self._replies.get(session_id)

    def forget(self, session_id: str) -> None:
        self._transcripts.pop(session_id, None)
        self._replies.pop(session_id, None)
