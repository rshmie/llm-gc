from typing import Callable

from llm_gc.events.event import Event
from llm_gc.events.event_type import EventType

class EventBus:
    """Publish/subscribe event bus for component instrumentation.

    Components emit events without knowing who listens. Subscribers can register callbacks for specific event types.
    """
    def __init__(self):
        self.subscribers = {}

    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        self.subscribers.setdefault(event_type, []).append(callback)

    def emit(self, event: Event) -> None:
        for callback in self.subscribers.get(event.event_type, []):
            callback(event)
