"""The README's library example, run exactly as written, so it cannot drift from the API.

The one change is the model: `AnthropicClient` is replaced with a `FakeProvider`, so the test makes no
network call.
"""

import asyncio
import re
from pathlib import Path

import llm_gc
from llm_gc import Message
from llm_gc.engine.generations.front import FRONT_HEADER
from llm_gc.llm import FakeProvider

README = Path(__file__).parents[1] / "README.md"
LIBRARY_HEADING = "### As a library"
# Enough older turns to pass the default `min_cut_tokens`, so the example makes a collection.
TURNS = 40


def read_library_example() -> str:
    """Return the first Python block under the README's library heading."""
    section = README.read_text(encoding="utf-8").split(LIBRARY_HEADING, 1)[1]
    match = re.search(r"```python\n(.*?)```", section, re.DOTALL)
    assert match, f"the README has no Python example under {LIBRARY_HEADING!r}"
    return match.group(1)


def test_the_readme_library_example_condenses_a_long_conversation(monkeypatch):
    monkeypatch.setattr(llm_gc, "AnthropicClient", lambda **_: FakeProvider())
    example: dict[str, object] = {}
    exec(read_library_example(), example)
    on_turn = example["on_turn"]

    async def converse() -> list[Message]:
        sent: list[Message] = []
        for index in range(TURNS):
            role = "user" if index % 2 == 0 else "assistant"
            text = f"Turn {index}: we compared the options for the event store in detail. " * 20
            sent = await on_turn("session", text, role, index)  # type: ignore[operator]
        return sent

    sent = asyncio.run(converse())

    assert sent[0].content.startswith(FRONT_HEADER), "the older turns arrive as one recap"
    assert len(sent) < TURNS
    assert sent[-1].turn_index == TURNS - 1, "the newest turn is sent as written"
