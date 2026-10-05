"""Progress for the check's long-running scripts: what is happening now, and how long is left.

A person at a terminal sees one line redrawn every second while an item runs, so the clock
moves even when nothing else does; a log file gets only the line each finished item leaves.
"""

import sys
import threading
import time
from typing import TextIO


class Progress:
    """A live line for the item in progress, then one line per item with the time left."""

    def __init__(self, total: int, out: TextIO = sys.stdout) -> None:
        self.total, self.kept, self.out = total, 0, out
        self.seconds: list[float] = []
        self.label, self.status, self.started = "", "", 0.0
        self.live = out.isatty()   # a log file gets the finished lines only
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def begin(self, label: str, status: str) -> None:
        self.label, self.status, self.started = label, status, time.monotonic()
        self._stop.clear()
        if self.live:
            # Redrawn every second, so the clock moves while the model thinks and nothing streams.
            threading.Thread(target=self._tick, daemon=True).start()

    def update(self, status: str) -> None:
        self.status = status

    def end(self, kept: bool, note: str) -> None:
        self._stop.set()
        self.seconds.append(time.monotonic() - self.started)
        self.kept += kept
        left = (self.total - self.kept) * sum(self.seconds) / len(self.seconds)
        mark = f"[{self.kept}/{self.total}]" if kept else "retry"
        remaining = "all done" if self.kept == self.total else f"about {_duration(left)} left"
        self._write(f"  {mark} {self.label}: {note} in {self.seconds[-1]:.0f}s · {remaining}\n")

    def _tick(self) -> None:
        while not self._stop.wait(1.0):
            self._write(f"  … {self.label}: {self.status} ({time.monotonic() - self.started:.0f}s)")

    def _write(self, text: str) -> None:
        with self._lock:
            self.out.write(f"\r\033[K{text}" if self.live else text if text.endswith("\n") else "")
            self.out.flush()


def _duration(seconds: float) -> str:
    return f"{seconds / 60:.0f} min" if seconds >= 90 else f"{seconds:.0f}s"


def plural(count: int, word: str) -> str:
    return f"{count:,} {word}" + ("" if count == 1 else "s")
