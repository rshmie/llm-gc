import tiktoken

from llm_gc.utils.text_cache import TextCache

# An approximation for every provider's models: tiktoken knows OpenAI's encodings only,
# so a model name would select the same one anyway.
_ENCODING = tiktoken.get_encoding("cl100k_base")

# Counts remembered by the shared cache. One entry is about 150 bytes whatever the size of its
# text, so this is about 8 MB. It has to hold every distinct text of the conversations in use
# at once: each turn, and each span as the planner prices it at each level (a long conversation
# is a few thousand turns and about half as many spans). A text dropped is only counted again.
TOKEN_COUNTS_REMEMBERED = 50_000


def count_tokens(text: str) -> int:
    """The number of tokens in `text`, by the cl100k_base encoding.

    Text that spells a special token, such as `<|endoftext|>`, is counted as the ordinary text it
    is. tiktoken raises on it by default, to stop a user's text passing for a control token to an
    OpenAI model. Here nothing is sent to that model, and a conversation about tokenizers is
    legitimate input that must not make the whole request unreadable.
    """
    return len(_ENCODING.encode(text, disallowed_special=()))


_SHARED = TextCache(count_tokens, TOKEN_COUNTS_REMEMBERED)


def count_tokens_cached(text: str) -> int:
    """`count_tokens`, remembering the answer: the call for text a conversation sends again and again.

    A client sends its whole conversation on every request, so almost all of a request's text was
    counted by the one before it, and tokenizing is the costliest step in reading a request. One
    cache serves the whole process. That is safe because a count does not depend on who asks.
    """
    return _SHARED.get(text)
