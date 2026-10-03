from llm_gc.engine.compaction import BaseCompactor, CompactionResult
from llm_gc.engine.compaction.compaction_strategy import CompactionStrategy
from llm_gc.events import EventType, Event
from llm_gc.models import Message
from llm_gc.utils import count_tokens


class NoOpCompactor(BaseCompactor):
    """A compactor that performs no modifications to the message list.

    This is useful for testing and as a baseline to compare against more
    sophisticated compaction strategies.
    """
    COMPACTION_STRATEGY: CompactionStrategy = CompactionStrategy.NOOP

    async def compact(self, messages: list[Message]) -> CompactionResult:
        """Return the input conversation unmodified."""
        first_turn = messages[0].turn_index
        last_turn = messages[-1].turn_index
        original_token_count = sum(msg.token_count for msg in messages)
        marker = self._format_summary_marker(first_turn, last_turn)
        verbatim_content = " ".join(msg.content for msg in messages)
        full_content = f"{marker} {verbatim_content}"
        summary_message = Message(role="assistant", content=full_content, token_count=count_tokens(full_content), turn_index=first_turn)
        self.event_bus.emit(Event(event_type=EventType.MESSAGE_COMPACTED,
                                  data={
                                      "compaction_strategy": self.COMPACTION_STRATEGY.value,
                                      "turn_range_start": first_turn,
                                      "turn_range_end": last_turn,
                                      "original_token_count": original_token_count,
                                      "token_count_after_compaction": summary_message.token_count,
                                      "messages_in_run": len(messages)
                                  }))
        return CompactionResult(summary= summary_message, original_token_count=original_token_count, compaction_strategy=self.COMPACTION_STRATEGY)