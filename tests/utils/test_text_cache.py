"""The text cache: a value computed from a text is remembered by a hash of the text, least recently used dropped first."""

import sys
import threading

import pytest
from hypothesis import given
from hypothesis import strategies as st

from llm_gc.utils import TextCache, count_tokens


class _Counter:
    """Stands in for the tokenizer, and remembers what it was asked to count."""

    def __init__(self):
        self.asked = []

    def __call__(self, text):
        self.asked.append(text)
        return len(text)


class TestTextCache:
    def test_a_text_is_counted_once_however_often_it_is_asked_for(self):
        counter = _Counter()
        cache = TextCache(counter, capacity=10)

        assert [cache.get("hello") for _ in range(3)] == [5, 5, 5]
        assert counter.asked == ["hello"]

    def test_an_empty_text_is_remembered_too(self):
        """Its count is 0. Checking for a remembered count with `if found:` would treat it as missing."""
        counter = _Counter()
        cache = TextCache(counter, capacity=10)

        assert [cache.get(""), cache.get("")] == [0, 0]
        assert counter.asked == [""]

    def test_a_value_of_none_is_remembered_too(self):
        """The table is checked for the text, not for a value, so any value can be remembered."""
        asked = []
        cache = TextCache(asked.append, capacity=10)

        cache.get("x")
        cache.get("x")
        assert asked == ["x"]

    def test_texts_of_the_same_length_do_not_share_a_count(self):
        cache = TextCache(int, capacity=10)

        assert (cache.get("11"), cache.get("12"), cache.get("11")) == (11, 12, 11)

    def test_the_least_recently_used_text_is_dropped_first(self):
        """A text asked for again is recent. Dropping in order of arrival would drop a
        conversation's oldest turns first, and every request asks for those."""
        counter = _Counter()
        cache = TextCache(counter, capacity=2)
        for text in ("a", "b", "a", "c"):      # "b" is now the least recent
            cache.get(text)
        counter.asked.clear()

        cache.get("a")
        cache.get("c")
        assert counter.asked == []
        cache.get("b")
        assert counter.asked == ["b"]
        assert len(cache) == 2

    def test_a_failed_count_is_not_remembered(self):
        """The tokenizer refuses some text, such as a special token written out. A refusal
        must not turn into a stored count."""
        def refuse(text):
            raise ValueError("refused")

        cache = TextCache(refuse, capacity=10)
        for _ in range(2):
            with pytest.raises(ValueError):
                cache.get("<|endoftext|>")

        assert len(cache) == 0

    def test_a_lone_surrogate_is_counted_not_refused(self):
        """Text parsed from JSON can hold one. Hashing it must not be a new way for a request to fail."""
        text = "a\ud800b"

        assert TextCache(count_tokens, capacity=10).get(text) == count_tokens(text)

    def test_the_capacity_must_be_at_least_one(self):
        with pytest.raises(ValueError):
            TextCache(count_tokens, capacity=0)

    @given(texts=st.lists(st.text(max_size=30), max_size=40), capacity=st.integers(min_value=1, max_value=5))
    def test_remembering_never_changes_a_count_and_never_outgrows_the_capacity(self, texts, capacity):
        cache = TextCache(count_tokens, capacity=capacity)

        assert [cache.get(text) for text in texts] == [count_tokens(text) for text in texts]
        assert len(cache) <= capacity

    def test_threads_sharing_one_cache_all_get_the_right_count(self):
        """A table touched by several threads while it evicts could raise from the thread that
        looks up a text another has just dropped. A small capacity makes that constant, and a
        short switch interval makes the threads take turns inside a lookup, which they seldom
        do at the default."""
        cache = TextCache(count_tokens, capacity=4)
        texts = [f"text number {number}" for number in range(40)]
        expected = {text: count_tokens(text) for text in texts}
        failures = []

        def work():
            try:
                for _ in range(200):
                    for text in texts:
                        assert cache.get(text) == expected[text]
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
