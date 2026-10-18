"""Remembering what a function of a text returned, so the same text is not worked through twice."""

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Callable

# A text is remembered by a hash of it. 128 bits make two texts sharing one a practical
# impossibility, so a remembered value is never another text's.
_DIGEST_BYTES = 16


class TextCache[T]:
    """Remembers `compute(text)` by text, so a text seen before is not computed again.

    For work that depends on nothing but its text, such as a token count or a regex score.
    A remembered value can then never be out of date, and there is nothing to invalidate.
    A client sends its whole conversation on every request, and LLM-GC plans every span of
    it again at each pass, so almost every text it reads was read the time before.

    Texts are kept as hashes, never as themselves: a tool result can run to hundreds of
    kilobytes, and a cache that held them would hold the conversation a second time. The
    least recently used value is dropped first. The table is guarded by a lock, so any
    thread may use it; the lock is not held while `compute` runs. A value is handed out
    as remembered, so `T` should be immutable, or never changed by whoever receives it.
    """

    def __init__(self, compute: Callable[[str], T], capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be at least 1, got {capacity}")
        self._values: OrderedDict[bytes, T] = OrderedDict()
        self._compute = compute
        self._capacity = capacity
        self._lock = threading.Lock()

    def get(self, text: str) -> T:
        """`compute(text)`, computing it only if it is not remembered.

        Raises:
            Exception: Whatever `compute` raises. A failure is not remembered.
        """
        # `surrogatepass`: text parsed from JSON can hold a lone surrogate, which a plain
        # UTF-8 encode refuses. Failing here would be a new way for a request to fail.
        key = hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=_DIGEST_BYTES).digest()
        with self._lock:
            if key in self._values:         # not `if found:`; a remembered value can be 0
                self._values.move_to_end(key)
                return self._values[key]
        value = self._compute(text)
        with self._lock:
            self._values[key] = value
            self._values.move_to_end(key)   # another thread may have computed it meanwhile
            while len(self._values) > self._capacity:
                self._values.popitem(last=False)
        return value

    def __len__(self) -> int:
        return len(self._values)
