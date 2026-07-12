import logging
from typing import Callable

from llm_gc.events.event import Event
from llm_gc.events.event_type import EventType

logger = logging.getLogger(__name__)
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
           try:
               callback(event)
           except Exception as e:
               logger.exception("Exception raised when emitting event", extra={"event_type": event.event_type,
                                                                               "exception_message": str(e)})

