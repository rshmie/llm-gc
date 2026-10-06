from collections.abc import Sequence
from dataclasses import replace

from llm_gc.engine.compaction import BaseCompactor
from llm_gc.engine.generations.condensing import (
    DEFAULT_MAX_SPAN_TOKENS,
    Condensed,
    IndexWriter,
    OpeningWordsIndexWriter,
    group_exchanges,
)
from llm_gc.extraction import KnowledgeExtractor
from llm_gc.models import KnowledgeEntry, Message


class GenerationalMemory:
    """Produces what the generations hold, and stores none of it.

    Condensing turns into spans - grouping, summarising, extracting facts, naming
    each span - is shared behaviour; the spans and facts it produces belong to one
    conversation and live in that session's snapshot. Holding them here would make
    every conversation served by this instance share one memory. See
    doc/design/generational-memory/overview.md.
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
        return await self.summarise(self.draft(turns))

    def draft(self, turns: Sequence[Message]) -> list[Condensed]:
        """The spans these turns would make, with their facts and index lines but no summaries.

        Cheap - no model call - so a collection can plan before it pays for
        summaries, and a preview can show what the next collection would do.
        """
        drafts = []
        for group in group_exchanges(turns, self.max_span_tokens):
            facts, facts_error = self._read_facts(group)
            drafts.append(Condensed(turns=group, index_line=self.index_writer.write(group), facts=facts,
                                    facts_error=facts_error))
        return drafts

    def worth_summarising(self, turns: Sequence[Message]) -> bool:
        """Whether a summary of these turns could be shorter than they are."""
        return self.compactor.worth_compacting(turns)

    async def summarise(self, drafts: Sequence[Condensed]) -> list[Condensed]:
        """Each draft with its summary, or with why it has none."""
        return [await self._summarise_one(draft) for draft in drafts]

    async def _summarise_one(self, draft: Condensed) -> Condensed:
        if not self.compactor.worth_compacting(draft.turns):
            return draft
        try:
            result = await self.compactor.compact(list(draft.turns))
        except Exception as err:
            # Any compactor's failure, a third party's included: the span is still made.
            return replace(draft, summary_error=err)
        text = result.summary_text if result.summary_text is not None else result.summary.content
        return replace(draft, summary=result.summary, summary_text=text)

    def _read_facts(self, turns: tuple[Message, ...]) -> tuple[tuple[KnowledgeEntry, ...], Exception | None]:
        try:
            return tuple(entry for turn in turns
                         for entry in self.knowledge_extractor.extract_knowledge(turn, turn.turn_index)), None
        except Exception as err:
            return (), err
