from collections.abc import Sequence

from llm_gc.engine.compaction import BaseCompactor, CompactionResult
from llm_gc.engine.generations.condensing import (
    DEFAULT_MAX_SPAN_TOKENS,
    Condensed,
    IndexWriter,
    OpeningWordsIndexWriter,
    group_exchanges,
)
from llm_gc.extraction import KnowledgeExtractor
from llm_gc.models import KnowledgeEntry, Message
from llm_gc.models.knowledge_entry import KnowledgeType


class GenerationalMemory:
    """Produces what the generations hold, and stores none of it.

    Summarising a run of turns and extracting the facts from a turn are shared
    behaviour; the summaries and facts they produce belong to one conversation and
    live in that session's snapshot (or, on the stateless path, in the returned
    result). Holding them here would make every conversation served by this
    instance share one memory. See doc/design/generational-memory/overview.md.
    """

    def __init__(self, knowledge_extractor: KnowledgeExtractor, compactor: BaseCompactor,
                 index_writer: IndexWriter | None = None, max_span_tokens: int = DEFAULT_MAX_SPAN_TOKENS) -> None:
        self.knowledge_extractor = knowledge_extractor
        self.compactor = compactor
        self.index_writer = index_writer if index_writer is not None else OpeningWordsIndexWriter()
        self.max_span_tokens = max_span_tokens

    async def condense(self, turns: Sequence[Message]) -> list[Condensed]:
        """Group the turns a collection removes into spans, and make each span's material.

        Facts are read from the original turns, never from a summary, and nothing
        is kept as raw text: the span keeps the turns themselves. A summary or
        extraction that fails is recorded on its span instead of raised, because
        the span can still be shown word for word or by its index line.
        """
        return [await self._condense_span(group) for group in group_exchanges(turns, self.max_span_tokens)]

    async def _condense_span(self, turns: tuple[Message, ...]) -> Condensed:
        facts, facts_error = self._read_facts(turns)
        index_line = self.index_writer.write(turns)
        if not self.compactor.worth_compacting(turns):
            return Condensed(turns=turns, index_line=index_line, facts=facts, facts_error=facts_error)
        try:
            result = await self.compactor.compact(list(turns))
        except Exception as err:
            # Any compactor's failure, a third party's included: the span is still made.
            return Condensed(turns=turns, index_line=index_line, facts=facts, facts_error=facts_error,
                             summary_error=err)
        text = result.summary_text if result.summary_text is not None else result.summary.content
        return Condensed(turns=turns, index_line=index_line, facts=facts, facts_error=facts_error,
                         summary=result.summary, summary_text=text)

    def _read_facts(self, turns: tuple[Message, ...]) -> tuple[tuple[KnowledgeEntry, ...], Exception | None]:
        try:
            return tuple(entry for turn in turns
                         for entry in self.knowledge_extractor.extract_knowledge(turn, turn.turn_index)), None
        except Exception as err:
            return (), err

    def extract_knowledge(self, message: Message) -> list[KnowledgeEntry]:
        """The facts a turn or summary yields, and whatever it says that they do not.

        What extraction cannot read is kept verbatim as a RAW entry - the whole turn
        when nothing was read, otherwise the sentences no fact came from - so
        archiving never loses information.
        """
        reading = self.knowledge_extractor.read(message=message, message_turn=message.turn_index)
        if not reading.entries:
            return [_verbatim(message, message.content)]
        if reading.unread:
            return [*reading.entries, _verbatim(message, reading.unread)]
        return reading.entries

    async def summarise(self, compact_run: list[Message]) -> CompactionResult:
        """Compact a run of turns into one summary."""
        return await self.compactor.compact(compact_run)


def _verbatim(message: Message, text: str) -> KnowledgeEntry:
    return KnowledgeEntry(message_turn=message.turn_index, content=text, topic_label="__raw__",
                          knowledge_type=KnowledgeType.RAW)
