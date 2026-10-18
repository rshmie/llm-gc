"""The seam a collection reads facts through, and the reader for a deployment without a model.

Facts are read once per collection, over every span it makes, given the facts
already on record: a change of plan in one span has to be filed under the topic it
changes, wherever that topic was first stated. See
doc/design/generational-memory/overview.md.
"""

from collections.abc import Sequence
from dataclasses import replace
from typing import Literal, Protocol

from llm_gc.extraction.knowledge_extractor import KnowledgeExtractor
from llm_gc.models import KnowledgeEntry, Message
from llm_gc.models.message import person_text, said_by_user


class FactReader(Protocol):
    """Reads what a collection's spans settled, as facts filed against the turns that state them."""

    @property
    def speakers(self) -> frozenset[Literal["user", "assistant"]]:
        """Whose turns it reads facts from.

        A side it does not read has no facts on record, which the dashboard says
        rather than showing that side as having settled nothing.
        """
        ...

    async def read(
        self, spans: Sequence[Sequence[Message]], on_record: Sequence[KnowledgeEntry]
    ) -> list[KnowledgeEntry]:
        """The facts these spans state, in conversation order.

        Args:
            spans: The spans a collection is making, each its turns in order.
            on_record: The conversation's active facts. A fact that changes one of
                them carries its topic, so filing it marks the old value replaced.

        Raises:
            Exception: Any failure to read. The collection makes its spans without
                facts and reports it; the turns themselves are kept either way.
        """
        ...


class PatternFactReader:
    """Facts matched by patterns, from the user's own words: the reader when there is no model.

    Only the user's turns. On the decision-survival check, what the patterns read
    from a user turn was a real statement - a choice, a name, a date - while what
    they read from the assistant's explanations was mostly sentence fragments, and
    a fragment filed twice under the same words read as a decision that changed.
    Tool results are output, not statements, and are skipped with them.
    """

    speakers: frozenset[Literal["user", "assistant"]] = frozenset({"user"})

    def __init__(self, extractor: KnowledgeExtractor | None = None) -> None:
        self.extractor = extractor if extractor is not None else KnowledgeExtractor()

    async def read(
        self, spans: Sequence[Sequence[Message]], on_record: Sequence[KnowledgeEntry]
    ) -> list[KnowledgeEntry]:
        """See `FactReader.read`. Patterns do not consult the facts on record."""
        return [
            replace(entry, speaker="user")
            for span in spans
            for turn in span
            if said_by_user(turn)
            for entry in self.extractor.extract_knowledge(_person_words(turn), turn.turn_index)
        ]


def _person_words(turn: Message) -> Message:
    """The turn with only the words the person wrote, so a pattern never matches text the client or a tool wrote."""
    if not turn.client_written and not turn.tool_output:
        return turn
    return turn.model_copy(update={"content": person_text(turn), "client_written": (), "tool_output": ()})
