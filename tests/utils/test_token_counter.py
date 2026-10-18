import sys
import threading

import pytest
import tiktoken
from hypothesis import given
from hypothesis import strategies as st

from llm_gc.utils.token_counter import TokenCountCache, count_tokens, count_tokens_cached

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


class _Counter:
    """Stands in for the tokenizer, and remembers what it was asked to count."""

    def __init__(self):
        self.asked = []

    def __call__(self, text):
        self.asked.append(text)
        return len(text)


class TestTokenCountCache:
    def test_a_text_is_counted_once_however_often_it_is_asked_for(self):
        counter = _Counter()
        cache = TokenCountCache(counter=counter)

        assert [cache.count("hello") for _ in range(3)] == [5, 5, 5]
        assert counter.asked == ["hello"]

    def test_an_empty_text_is_remembered_too(self):
        """Its count is 0. Checking for a remembered count with `if found:` would treat it as missing."""
        counter = _Counter()
        cache = TokenCountCache(counter=counter)

        assert [cache.count(""), cache.count("")] == [0, 0]
        assert counter.asked == [""]

    def test_texts_of_the_same_length_do_not_share_a_count(self):
        cache = TokenCountCache(counter=int)

        assert (cache.count("11"), cache.count("12"), cache.count("11")) == (11, 12, 11)

    def test_the_least_recently_used_text_is_dropped_first(self):
        """A text asked for again is recent. Dropping in order of arrival would drop a
        conversation's oldest turns first, and every request asks for those."""
        counter = _Counter()
        cache = TokenCountCache(capacity=2, counter=counter)
        for text in ("a", "b", "a", "c"):      # "b" is now the least recent
            cache.count(text)
        counter.asked.clear()

        cache.count("a")
        cache.count("c")
        assert counter.asked == []
        cache.count("b")
        assert counter.asked == ["b"]
        assert len(cache) == 2

    def test_a_failed_count_is_not_remembered(self):
        """The tokenizer refuses some text, such as a special token written out. A refusal
        must not turn into a stored count."""
        def refuse(text):
            raise ValueError("refused")

        cache = TokenCountCache(counter=refuse)
        for _ in range(2):
            with pytest.raises(ValueError):
                cache.count("<|endoftext|>")

        assert len(cache) == 0

    def test_a_lone_surrogate_is_counted_not_refused(self):
        """Text parsed from JSON can hold one. Hashing it must not be a new way for a request to fail."""
        text = "a\ud800b"

        assert TokenCountCache().count(text) == count_tokens(text)

    def test_the_capacity_must_be_at_least_one(self):
        with pytest.raises(ValueError):
            TokenCountCache(capacity=0)

    def test_the_shared_cache_gives_the_tokenizers_own_answer(self):
        assert count_tokens_cached("Hi, there! How are you?") == count_tokens("Hi, there! How are you?")

    @given(texts=st.lists(st.text(max_size=30), max_size=40), capacity=st.integers(min_value=1, max_value=5))
    def test_remembering_never_changes_a_count_and_never_outgrows_the_capacity(self, texts, capacity):
        cache = TokenCountCache(capacity=capacity)

        assert [cache.count(text) for text in texts] == [count_tokens(text) for text in texts]
        assert len(cache) <= capacity

    def test_threads_sharing_one_cache_all_get_the_right_count(self):
        """A table touched by several threads while it evicts could raise from the thread that
        looks up a text another has just dropped. A small capacity makes that constant, and a
        short switch interval makes the threads take turns inside a lookup, which they seldom
        do at the default."""
        cache = TokenCountCache(capacity=4)
        texts = [f"text number {number}" for number in range(40)]
        expected = {text: count_tokens(text) for text in texts}
        failures = []

        def work():
            try:
                for _ in range(200):
                    for text in texts:
                        assert cache.count(text) == expected[text]
            except Exception as err:
                failures.append(err)

        threads = [threading.Thread(target=work) for _ in range(6)]
        interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        finally:
            sys.setswitchinterval(interval)

        assert failures == []
