import tiktoken

# An approximation for every provider's models: tiktoken knows OpenAI's encodings only,
# so a model name would select the same one anyway.
_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """The number of tokens in `text`, by the cl100k_base encoding."""
    return len(_ENCODING.encode(text))