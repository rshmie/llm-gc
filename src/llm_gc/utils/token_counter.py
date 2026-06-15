import tiktoken
from llm_gc.config.constants import DEFAULT_MODEL

def count_tokens(text: str, model: str = DEFAULT_MODEL) -> int:
    """Count the number of tokens in a text string using tiktoken.

    Falls back to cl100k_base encoding if the model is not recognized.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return len(tokens)