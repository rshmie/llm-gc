from dataclasses import dataclass, field
from datetime import datetime, timezone

from llm_gc.models.generation_snapshot import GenerationSnapshot


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class SessionState:
    """Everything one conversation carries from turn to turn.

    `snapshot` holds all three generations and is replaced, never mutated, so a
    reader that took it keeps a consistent view with no lock. This object is the
    single owner of a session's conversation data: removing it from the store
    removes all of it.

    `aging_in_flight` is process-local coordination, not conversation data. It
    sits outside the snapshot and is not something a store would persist.
    """

    session_id: str
    snapshot: GenerationSnapshot = field(default_factory=GenerationSnapshot)
    created_at: datetime = field(default_factory=_now)
    last_updated_at: datetime = field(default_factory=_now)
    aging_in_flight: bool = False

    def touch(self) -> None:
        """Reset the idle clock; the sweeper measures idleness from here."""
        self.last_updated_at = _now()
