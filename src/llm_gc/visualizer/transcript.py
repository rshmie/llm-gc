"""Each conversation as its client last sent it, kept for the dashboard.

The dashboard shows every turn's words beside what the model receives instead.
The client sends the whole conversation on every request, so the proxy hands each
request's history to this recorder. The engine's spans hold the same words for
every condensed turn; this copy goes once the dashboard reads them from the spans.

Held only while the dashboard runs, in memory, as text, bounded by session count,
and dropped when its session closes.
"""

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

from llm_gc.models import Message

# Sessions whose transcript is kept. Above this, the least recently active is
# dropped; a transcript is the whole conversation, so the bound is deliberately low.
DEFAULT_TRANSCRIPT_MAX_SESSIONS = 50


@dataclass(frozen=True, slots=True)
class TranscriptTurn:
    turn_index: int
    role: str
    text: str     # the turn's text; empty for a turn that is only tool calls or attachments
    tokens: int


class TranscriptRecorder:
    """The latest client history per session. Feed it from the proxy; read it from the dashboard."""

    def __init__(self, max_sessions: int = DEFAULT_TRANSCRIPT_MAX_SESSIONS) -> None:
        self._max_sessions = max_sessions
        self._transcripts: OrderedDict[str, tuple[TranscriptTurn, ...]] = OrderedDict()

    def record(self, session_id: str, messages: Sequence[Message]) -> None:
        """Replace the session's transcript with what the client just sent.

        Replaced rather than appended, so an edited or regenerated conversation
        shows what the client holds now. Text only: keeping `raw_content` would hold
        every attached image in memory for the life of the session.
        """
        self._transcripts[session_id] = tuple(
            TranscriptTurn(turn_index=message.turn_index, role=message.role, text=message.content,
                           tokens=message.token_count)
            for message in messages
        )
        self._transcripts.move_to_end(session_id)
        if len(self._transcripts) > self._max_sessions:
            self._transcripts.popitem(last=False)

    def get(self, session_id: str) -> tuple[TranscriptTurn, ...] | None:
        return self._transcripts.get(session_id)

    def forget(self, session_id: str) -> None:
        self._transcripts.pop(session_id, None)
