import tiktoken
from llm_gc.utils.token_counter import count_tokens

def test_token_counter():
    text = "Hi, there!"
    assert(count_tokens(text) > 0)

def test_token_counter_with_anthropic_fall_back():
    text = "Hi, there!"
    claude_token_count = count_tokens(text, "claude-sonnet-4-6")
    open_ai_token_count = len(tiktoken.get_encoding("cl100k_base").encode(text))
    assert(claude_token_count == open_ai_token_count)

def test_empty_string_returns_zero_token():
    text = ""
    token_count = count_tokens(text)
    assert(token_count == 0)

def test_longer_text_returns_more_tokens_than_shorter_text():
    shorter_text = "Hi, there!"
    longer_text = "Hi, there! How are you?"
    assert(count_tokens(longer_text) > count_tokens(shorter_text))
