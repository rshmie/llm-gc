import hashlib
import threading
from collections import OrderedDict
from collections.abc import Callable

import tiktoken

# An approximation for every provider's models: tiktoken knows OpenAI's encodings only,
# so a model name would select the same one anyway.
_ENCODING = tiktoken.get_encoding("cl100k_base")

# Counts remembered by the shared cache. One entry is about 150 bytes whatever the size of its
# text, so this is about 8 MB. It has to hold every distinct text of the conversations in use
# at once (a long one is a few thousand turns); a text dropped is only counted again.
TOKEN_COUNTS_REMEMBERED = 50_000

# A text is remembered by a hash of it. 128 bits make two texts sharing one a practical
# impossibility, so a remembered count is never another text's.
_DIGEST_BYTES = 16


def count_tokens(text: str) -> int:
    """The number of tokens in `text`, by the cl100k_base encoding.

    Text that spells a special token, such as `<|endoftext|>`, is counted as the ordinary text it
    is. tiktoken raises on it by default, to stop a user's text passing for a control token to an
    OpenAI model. Here nothing is sent to that model, and a conversation about tokenizers is
    legitimate input that must not make the whole request unreadable.
    """
    return len(_ENCODING.encode(text, disallowed_special=()))


class TokenCountCache:
    """Remembers token counts by text, so a text seen before is not tokenized again.

    A client sends its whole conversation on every request, so almost all of a request's
    text gets counted by the one before it. Tokenizing is the costliest step in reading a
    request and grows with the conversation. A count depends on nothing but its text, so a
    remembered count can never be out of date and there is nothing to invalidate.

    Texts are kept as hashes, never as themselves: a tool result can run to hundreds of
    kilobytes, and a cache that held them would hold the conversation a second time. The
    least recently used count is dropped first. The table is guarded by a lock, so any
    thread may use it; the lock is not held while a text is tokenized.
    """

    def __init__(self, capacity: int = TOKEN_COUNTS_REMEMBERED,
                 counter: Callable[[str], int] = count_tokens) -> None:
        """`counter` is what counts a text not remembered; tests pass one that records its calls."""
        if capacity < 1:
            raise ValueError(f"capacity must be at least 1, got {capacity}")
        self._counts: OrderedDict[bytes, int] = OrderedDict()
        self._capacity = capacity
        self._counter = counter
        self._lock = threading.Lock()

    def count(self, text: str) -> int:
        """The number of tokens in `text`, tokenizing it only if it is not remembered."""
        # `surrogatepass`: text parsed from JSON can hold a lone surrogate, which a plain
        # UTF-8 encode refuses. Failing here would be a new way for a request to fail.
        key = hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=_DIGEST_BYTES).digest()
        with self._lock:
            remembered = self._counts.get(key)
            if remembered is not None:      # not `if remembered:`; an empty text's count is 0
                self._counts.move_to_end(key)
                return remembered
        count = self._counter(text)
        with self._lock:
            self._counts[key] = count
            self._counts.move_to_end(key)   # another thread may have counted it meanwhile
            while len(self._counts) > self._capacity:
                self._counts.popitem(last=False)
        return count

    def __len__(self) -> int:
        return len(self._counts)


_SHARED = TokenCountCache()


def count_tokens_cached(text: str) -> int:
    """`count_tokens`, remembering the answer: the call for text a conversation sends again and again.

    One cache serves the whole process. That is safe because a count does not depend on who asks.
    """
    return _SHARED.count(text)
