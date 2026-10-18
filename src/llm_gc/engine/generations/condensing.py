"""Grouping the turns a collection removes into spans, and naming each one.

A span's material - its summary, facts and index line - is made once, when its
turns are condensed, so choosing a span's level later never waits on a model.
See doc/design/generational-memory/overview.md.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from llm_gc.models import KnowledgeEntry, Message
from llm_gc.models.message import person_text, said_by_user, tool_result_ids

# An exchange larger than this is split between tool rounds, so one long agentic
# exchange becomes several summaries rather than a single call over all of it.
DEFAULT_MAX_SPAN_TOKENS = 8_000

# An index line names a span; it does not summarise it.
INDEX_LINE_MAX_CHARS = 100

_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")
_ELLIPSIS = "…"


@dataclass(frozen=True)
class Condensed:
    """One span's material, made when its turns were condensed. Its level is decided elsewhere.

    `summary_error` is why a summary was attempted and not made. No summary and no
    error means none was attempted: the turns were too few to be shortened.
    `facts_error` is why extraction failed; the span is made either way.
    """

    turns: tuple[Message, ...]
    index_line: str
    facts: tuple[KnowledgeEntry, ...]
    summary: Message | None = None
    summary_text: str | None = None
    summary_error: Exception | None = None
    facts_error: Exception | None = None


class IndexWriter(Protocol):
    """Names what a span is about, in one short line."""

    def write(self, turns: Sequence[Message]) -> str: ...


class OpeningWordsIndexWriter:
    """A span's first sentence, from the user when they spoke: what was asked, in their words.

    Terse, and never wrong, because it invents nothing. A model-written title is
    the better line; this is what an `IndexWriter` must at least match.
    """

    def write(self, turns: Sequence[Message]) -> str:
        """See `IndexWriter.write`."""
        asked = next((turn for turn in turns if _says_something(turn) and said_by_user(turn)), None)
        if asked is not None:
            return _first_sentence(person_text(asked))
        opener = next((turn for turn in turns if _says_something(turn)), None)
        return _first_sentence(opener.content) if opener is not None else ""


def group_exchanges(turns: Sequence[Message], max_span_tokens: int = DEFAULT_MAX_SPAN_TOKENS
                    ) -> list[tuple[Message, ...]]:
    """Split consecutive turns into spans: one exchange each, divided again when an exchange is too large.

    An exchange opens at a turn the user wrote (`said_by_user`) and runs to the next
    one: tool results and relayed reports join the exchange they answer. A large exchange is divided only before an assistant turn that follows
    another assistant turn in the span: a tool call stays with its result, and a
    question with its first answer.
    """
    groups: list[list[Message]] = []
    size = 0
    for turn in turns:
        opens_exchange = said_by_user(turn)
        overflows = (bool(groups) and turn.role == "assistant" and size + turn.token_count > max_span_tokens
                     and any(earlier.role == "assistant" for earlier in groups[-1]))
        if not groups or opens_exchange or overflows:
            groups.append([])
            size = 0
        groups[-1].append(turn)
        size += turn.token_count
    return [tuple(group) for group in groups]


def _says_something(turn: Message) -> bool:
    return bool(turn.content.strip()) and not tool_result_ids(turn)


def _first_sentence(text: str) -> str:
    """The text's first sentence on one line, cut at a word boundary if it runs long."""
    line = " ".join(text.split())
    end = _SENTENCE_END.search(line)
    sentence = line[: end.end()] if end else line
    if len(sentence) <= INDEX_LINE_MAX_CHARS:
        return sentence
    return sentence[: INDEX_LINE_MAX_CHARS - len(_ELLIPSIS)].rsplit(" ", 1)[0] + _ELLIPSIS
