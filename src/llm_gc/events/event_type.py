from enum import Enum

class EventType(Enum):
   """ Enumeration of all event types emitted by GC components.

    Each component emits specific event types that subscribers can
    selectively listen to via the EventBus.
   """
   GC_FINISHED = "gc_finished"
   TOKEN_COUNT = "token_count"
   MESSAGE_RELEVANCE_SCORED = "message_relevance_scored"
   SWEEP_COMPLETED = "sweep_completed"
   KNOWLEDGE_ENTRY_ADDED = "knowledge_entry_added"
   KNOWLEDGE_ENTRY_SUPERSEDED = "knowledge_entry_superseded"
   KNOWLEDGE_ENTRIES_EXTRACTED = "knowledge_entries_extracted"
   MESSAGE_ADDED_TO_YOUNG_GEN = "message_added_to_young_gen"
   MESSAGE_PROMOTED_TO_OLD_GEN = "message_promoted_to_old_gen"
   MESSAGE_ARCHIVED = "message_archived"
   CONTEXT_COMPOSED = "context_composed"
   # Fires once per compactor invocation, regardless of whether real compaction occurred. Subscribers should inspect `method` and the
   # token-count delta in the payload to determine the outcome.
   MESSAGE_COMPACTED = "message_compacted"
