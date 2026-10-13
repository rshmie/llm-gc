"""Pausing LLM-GC's own model calls while the upstream pushes back.

LLM-GC's summaries and fact reads sometimes go to the same upstream as the client's own requests,
often under the same credential, so they share its rate limit. When the upstream answers
"too many requests" or "overloaded", LLM-GC stops adding to it: no collection starts until
the pause ends. The client's requests are never held back; only LLM-GC's extra calls wait.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# How long to pause when the upstream gives no Retry-After, and the most a pause may last.
DEFAULT_PAUSE = timedelta(seconds=60)
MAX_PAUSE = timedelta(minutes=10)


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class Pause:
    until: datetime
    reason: str


class ModelCallPause:
    """Whether LLM-GC may call a model now. Single-threaded: used from the proxy's event loop only."""

    def __init__(self, *, clock: Callable[[], datetime] = _now) -> None:
        self._clock = clock
        self._pause: Pause | None = None

    def pause(self, retry_after_s: float | None, reason: str) -> bool:
        """Pause for the upstream's Retry-After, or the default; extend a pause already running.

        Returns True when a pause begins, False when one was already running.
        """
        now = self._clock()
        wait = timedelta(seconds=retry_after_s) if retry_after_s is not None and retry_after_s > 0 else DEFAULT_PAUSE
        until = now + min(wait, MAX_PAUSE)
        began = self.current() is None
        if self._pause is None or until > self._pause.until:
            self._pause = Pause(until=until, reason=reason)
        return began

    def current(self) -> Pause | None:
        """The pause in force, or None."""
        if self._pause is not None and self._clock() >= self._pause.until:
            self._pause = None
        return self._pause
