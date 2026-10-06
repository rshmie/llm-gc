from enum import Enum

class EventType(Enum):
   """ Enumeration of all event types emitted by GC components.

    Each component emits specific event types that subscribers can
    selectively listen to via the EventBus.
   """
   GC_FINISHED = "gc_finished"
   KNOWLEDGE_ENTRY_ADDED = "knowledge_entry_added"
   KNOWLEDGE_ENTRY_SUPERSEDED = "knowledge_entry_superseded"
   MESSAGE_PROMOTED_TO_OLD_GEN = "message_promoted_to_old_gen"
   MESSAGE_ARCHIVED = "message_archived"
   # Fires once per compactor invocation, regardless of whether real compaction occurred. Subscribers should inspect `method` and the
   # token-count delta in the payload to determine the outcome.
   MESSAGE_COMPACTED = "message_compacted"
   # Fires once per thing a collection intended and did not manage - a refused or
   # failed summary, a failed extraction, a front that could not fit its budget. The
   # span is still made, so nothing is lost; the front is larger, or thinner, than the
   # plan wanted. Emitted separately from GC_FINISHED
   AGING_DEGRADED = "aging_degraded"
