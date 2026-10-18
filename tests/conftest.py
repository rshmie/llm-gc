"""Fixtures shared by every test module."""

from collections.abc import Callable
from typing import Any

import pytest

from llm_gc.utils import token_counter


@pytest.fixture
def tokenised(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The texts the tokenizer is asked to count, in order, from the moment a test asks for this.

    The real tokenizer still answers, so counts are unchanged. It is for tests of what must not
    be counted again. The count cache is shared by the whole test run, so such a test builds
    its texts from a value of its own that no other test uses.
    """
    texts: list[str] = []
    real: Callable[..., Any] = token_counter._ENCODING.encode

    def record(text: str, *args: Any, **kwargs: Any) -> Any:
        texts.append(text)
        return real(text, *args, **kwargs)

    monkeypatch.setattr(token_counter._ENCODING, "encode", record)
    return texts
