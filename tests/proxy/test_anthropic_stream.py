"""The SSE observer, fed the way TCP actually delivers bytes: in arbitrary pieces."""

import json

from llm_gc.proxy.anthropic_stream import MAX_PENDING_EVENT_BYTES, StreamedReply


def _event(payload: dict) -> bytes:
    return f"event: {payload['type']}\ndata: {json.dumps(payload)}\n\n".encode()


def _text(text: str) -> bytes:
    return _event({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}})


START = _event({"type": "message_start", "message": {"id": "m", "role": "assistant", "content": []}})
STOP = _event({"type": "message_stop"})


def _stream(*texts: str, stop: bool = True) -> bytes:
    return START + b"".join(_text(t) for t in texts) + (STOP if stop else b"")


def _fed(data: bytes, chunk_size: int | None = None) -> StreamedReply:
    reply = StreamedReply()
    size = chunk_size or len(data) or 1
    for start in range(0, len(data), size):
        reply.feed(data[start:start + size])
    return reply


class TestReassembly:
    def test_text_deltas_are_joined_in_order(self):
        reply = _fed(_stream("Postgres, ", "partitioned by day."))
        assert reply.text == "Postgres, partitioned by day."
        assert reply.completed

    def test_chunk_boundaries_do_not_matter(self):
        """Fed one byte at a time, an event is never whole until its last byte."""
        reply = _fed(_stream("Postgres, ", "partitioned by day."), chunk_size=1)
        assert reply.text == "Postgres, partitioned by day."
        assert reply.completed

    def test_a_multibyte_character_split_across_chunks_survives(self):
        """Decoding each chunk separately would raise on the split emoji."""
        reply = _fed(_stream("ship it 🚀 today"), chunk_size=1)
        assert reply.text == "ship it 🚀 today"

    def test_crlf_line_endings_are_accepted_even_when_split(self):
        reply = _fed(_stream("hello").replace(b"\n", b"\r\n"), chunk_size=1)
        assert reply.text == "hello"
        assert reply.completed

    def test_multiline_data_fields_are_joined(self):
        """The SSE spec allows one JSON document across several `data:` lines."""
        reply = _fed(START + b'data: {"type": "content_block_delta",\ndata: '
                     b'"delta": {"type": "text_delta", "text": "hi"}}\n\n' + STOP)
        assert reply.text == "hi"


class TestWhatIsNotTheReply:
    def test_tool_input_and_thinking_deltas_are_ignored(self):
        """The non-streaming path records text blocks only; the two must agree."""
        noise = (
            _event({"type": "content_block_delta", "delta": {"type": "input_json_delta", "partial_json": "{\"q\""}})
            + _event({"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "hmm"}})
            + _event({"type": "ping"})
        )
        reply = _fed(START + noise + _text("answer") + STOP)
        assert reply.text == "answer"


class TestHowAStreamCanFail:
    def test_a_stream_without_message_stop_is_not_complete(self):
        reply = _fed(_stream("half an ans", stop=False))
        assert not reply.completed
        assert reply.text == "half an ans"

    def test_a_provider_error_event_is_reported(self):
        error = _event({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})
        reply = _fed(START + _text("partial") + error)
        assert reply.errored
        assert not reply.completed

    def test_a_malformed_event_is_counted_and_skipped(self):
        reply = _fed(START + b"data: {not json\n\n" + _text("still here") + STOP)
        assert reply.malformed_events == 1
        assert reply.text == "still here"
        assert reply.completed

    def test_a_body_with_no_event_boundary_is_abandoned_not_buffered(self):
        reply = StreamedReply()
        reply.feed(b"x" * (MAX_PENDING_EVENT_BYTES + 1))
        assert reply.abandoned
        reply.feed(STOP)
        assert not reply.completed, "an abandoned parse must not be revived by later bytes"
