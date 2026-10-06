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
from llm_gc.extraction import FactReader
from llm_gc.models import KnowledgeEntry, Message


class GenerationalMemory:
    """Produces what the generations hold, and stores none of it.

    Condensing turns into spans - grouping, summarising, reading facts, naming
    each span - is shared behaviour; the spans and facts it produces belong to one
    conversation and live in that session's snapshot. Holding them here would make
    every conversation served by this instance share one memory. See
    doc/design/generational-memory/overview.md.
    """

    def __init__(self, fact_reader: FactReader, compactor: BaseCompactor, index_writer: IndexWriter | None = None,
                 max_span_tokens: int = DEFAULT_MAX_SPAN_TOKENS) -> None:
        self.fact_reader = fact_reader
        self.compactor = compactor
        self.index_writer = index_writer if index_writer is not None else OpeningWordsIndexWriter()
        self.max_span_tokens = max_span_tokens

    async def condense(self, turns: Sequence[Message], on_record: Sequence[KnowledgeEntry] = ()) -> list[Condensed]:
        """Group turns into spans and make each span's material: `complete` over `draft`."""
        return await self.complete(self.draft(turns), on_record)

    def draft(self, turns: Sequence[Message]) -> list[Condensed]:
        """The spans these turns would make, with their index lines but no summaries or facts.

        Cheap - no model call - so a collection can plan before it pays for
        anything, and a preview can show what the next collection would do.
        """
        return [Condensed(turns=group, index_line=self.index_writer.write(group), facts=())
                for group in group_exchanges(turns, self.max_span_tokens)]

    def worth_summarising(self, turns: Sequence[Message]) -> bool:
        """Whether a summary of these turns could be shorter than they are."""
        return self.compactor.worth_compacting(turns)

    async def complete(self, drafts: Sequence[Condensed], on_record: Sequence[KnowledgeEntry]) -> list[Condensed]:
        """Each draft with its facts and its summary, or with why it has none.

        Facts are read once for all the drafts, given the facts on record, then
        each summary is written. In turn rather than at once: a collection runs
        after the response, so nobody waits on it, and in turn the calls arrive in
        one known order. Facts are read from the original turns, never from a
        summary. A failure is recorded on the spans it affects instead of raised,
        because a span can still be shown without its summary or its facts.
        """
        read, error = await self._read_facts(drafts, on_record)
        return [replace(draft, facts=tuple(fact for fact in read if fact.message_turn in _turns_of(draft)),
                        facts_error=error) for draft in await self._summarise_all(drafts)]

    async def _read_facts(self, drafts: Sequence[Condensed], on_record: Sequence[KnowledgeEntry]
                          ) -> tuple[list[KnowledgeEntry], Exception | None]:
        if not drafts:
            return [], None
        try:
            return await self.fact_reader.read([draft.turns for draft in drafts], on_record), None
        except Exception as err:
            # Any reader's failure, a third party's included: the spans are still made.
            return [], err

    async def _summarise_all(self, drafts: Sequence[Condensed]) -> list[Condensed]:
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


def _turns_of(draft: Condensed) -> set[int]:
    return {turn.turn_index for turn in draft.turns}
