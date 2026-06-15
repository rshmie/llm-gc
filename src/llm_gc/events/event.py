import time
from pydantic import BaseModel, Field
from llm_gc.events.event_type import EventType

class Event(BaseModel):
    """A structured event emitted by GC components for observability.

    Carries an event type, arbitrary data payload, and timestamp.
    Consumed by the EventBus subscribers (visualizer, logger, benchmarks etc.)
    """
    event_type: EventType
    data: dict
    timestamp: float = Field(default_factory=time.time)