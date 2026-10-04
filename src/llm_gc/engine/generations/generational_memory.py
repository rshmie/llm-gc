from llm_gc.engine.compaction import BaseCompactor, CompactionResult
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

    def __init__(self, knowledge_extractor: KnowledgeExtractor, compactor: BaseCompactor) -> None:
        self.knowledge_extractor = knowledge_extractor
        self.compactor = compactor

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
