import tiktoken

from llm_gc.utils.token_counter import count_tokens, count_tokens_cached

def test_token_counter():
    text = "Hi, there!"
    assert(count_tokens(text) > 0)

def test_tokens_are_counted_by_the_cl100k_base_encoding():
    text = "Hi, there!"
    assert count_tokens(text) == len(tiktoken.get_encoding("cl100k_base").encode(text))

def test_empty_string_returns_zero_token():
    text = ""
    token_count = count_tokens(text)
    assert(token_count == 0)

def test_longer_text_returns_more_tokens_than_shorter_text():
    shorter_text = "Hi, there!"
    longer_text = "Hi, there! How are you?"
    assert(count_tokens(longer_text) > count_tokens(shorter_text))

def test_text_naming_a_special_token_is_counted_as_ordinary_text():
    """A conversation about tokenizers contains `<|endoftext|>`. tiktoken refuses that text unless told
    otherwise, and no provider reads it as a special token, so it is counted like any other text."""
    text = "the end marker is <|endoftext|>"
    assert count_tokens(text) == len(tiktoken.get_encoding("cl100k_base").encode(text, disallowed_special=()))
    assert count_tokens("<|endoftext|>") > 1, "one token would mean it was read as the special token"


def test_the_shared_cache_gives_the_tokenizers_own_answer():
    assert count_tokens_cached("Hi, there! How are you?") == count_tokens("Hi, there! How are you?")
