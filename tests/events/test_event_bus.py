from llm_gc.events import Event, EventBus, EventType

def test_subscribe_and_emit():
    bus = EventBus()
    received = []
    def test_callback(event: Event):
        received.append(event)
    bus.subscribe(EventType.TOKEN_COUNT, test_callback)

    bus.emit(Event(event_type=EventType.TOKEN_COUNT, data={"count": 5}))
    assert len(received) == 1
    assert received[0].data["count"] == 5

def test_emit_with_no_subscribers_do_not_raise_error():
    bus = EventBus()
    bus.emit(Event(event_type=EventType.TOKEN_COUNT, data={"count": 10}))

def test_multiple_callbacks():
    bus = EventBus()
    received1 = []
    received2 = []
    def callback1(event: Event):
        received1.append(event)
    def callback2(event: Event):
        received2.append(event)

    bus.subscribe(EventType.GC_FINISHED, callback1)
    bus.subscribe(EventType.GC_FINISHED, callback2)

    bus.emit(Event(event_type=EventType.GC_FINISHED, data={"gc_run_id": 1}))
    assert len(received1) == 1
    assert len(received2) == 1
    assert received1[0].data["gc_run_id"] == 1
    assert received2[0].data["gc_run_id"] == 1


def test_wrong_event_type():
    bus = EventBus()
    received = []
    def test_callback(event: Event):
        received.append(event)
    bus.subscribe(EventType.TOKEN_COUNT, test_callback)

    bus.emit(Event(event_type=EventType.GC_FINISHED, data={"gc_run_id": 2}))
    assert len(received) == 0


def test_emit_does_not_raise_when_subscriber_throws():
    bus = EventBus()
    def bad_subscriber(event: Event):
        raise ValueError("boom")
    bus.subscribe(EventType.TOKEN_COUNT, bad_subscriber)

    bus.emit(Event(event_type=EventType.TOKEN_COUNT, data={"count": 1}))  # must not raise


def test_emit_calls_remaining_subscribers_after_one_throws():
    bus = EventBus()
    received = []
    def bad_subscriber(event: Event):
        raise ValueError("boom")
    def good_subscriber(event: Event):
        received.append(event)

    bus.subscribe(EventType.TOKEN_COUNT, bad_subscriber)
    bus.subscribe(EventType.TOKEN_COUNT, good_subscriber)

    bus.emit(Event(event_type=EventType.TOKEN_COUNT, data={"count": 1}))

    assert len(received) == 1
