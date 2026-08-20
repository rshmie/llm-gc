from dataclasses import dataclass, field
from datetime import datetime, timezone

from llm_gc.models import Message, SweepClassification


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class SessionState:
    """The remembered state for one conversation, carried between turns.

    A session is a single continuous conversation. Everything the engine needs
    to carry from one turn to the next lives here: the running list of messages
    and the timestamps the idle-timeout sweeper reads to decide when a session
    has gone quiet.
    """

    session_id: str
    messages: list[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=_now)
    last_updated_at: datetime = field(default_factory=_now)
    last_classifications: dict[int, SweepClassification] = field(default_factory=dict)

    def touch(self) -> None:
        """Mark the session as just-used, resetting its idle clock.

        Called on every collect/update so the sweeper measures idleness from
        the last real activity, not from when the session was created.
        """
        self.last_updated_at = _now()
