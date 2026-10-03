"""Observing an Anthropic SSE stream as it passes through the proxy.

The proxy relays streamed bytes to the client untouched; this parser watches a
copy of them so the assistant's reply can be recorded afterwards. It never
produces output for the client, so a stream it cannot understand costs one
recorded reply, never the response.
"""

import json
import logging

logger = logging.getLogger(__name__)

# An SSE event is terminated by a blank line. A body that never produces one is
# not an event stream, and buffering it unboundedly would let one bad upstream
# response hold arbitrary memory.
MAX_PENDING_EVENT_BYTES = 1024 * 1024


class StreamedReply:
    """Reassembles the assistant's text from SSE chunks fed in arrival order.

    Reports facts, not policy: whether the stream finished, whether the provider
    signalled an error, and the text seen. The caller decides what to record.
    """

    def __init__(self) -> None:
        self._pending = b""
        self._text_parts: list[str] = []
        self.completed = False
        self.errored = False
        self.abandoned = False
        self.malformed_events = 0

    @property
    def text(self) -> str:
        return "".join(self._text_parts)

    def feed(self, chunk: bytes) -> None:
        """Consume one chunk. Chunk boundaries need not align with events."""
        if self.abandoned:
            return

        # Normalised on the whole buffer, so a CRLF split across two chunks is
        # still caught once the second half arrives.
        self._pending = (self._pending + chunk).replace(b"\r\n", b"\n")

        while b"\n\n" in self._pending:
            raw_event, _, self._pending = self._pending.partition(b"\n\n")
            self._consume_event(raw_event)

        if len(self._pending) > MAX_PENDING_EVENT_BYTES:
            logger.warning("abandoned parsing a stream with no event boundary",
                           extra={"pending_bytes": len(self._pending)})
            self.abandoned = True
            self._pending = b""

    def _consume_event(self, raw_event: bytes) -> None:
        # Per the SSE spec, multiple `data:` lines in one event join with newlines.
        # The `event:` line is ignored: Anthropic repeats the type inside the JSON.
        data_lines = [
            line[len(b"data:"):].lstrip(b" ")
            for line in raw_event.split(b"\n")
            if line.startswith(b"data:")
        ]
        if not data_lines:
            return

        try:
            event = json.loads(b"\n".join(data_lines))
        except ValueError:
            self.malformed_events += 1
            return
        if not isinstance(event, dict):
            self.malformed_events += 1
            return

        event_type = event.get("type")
        if event_type == "content_block_delta":
            delta = event.get("delta")
            # Only text deltas, matching what the non-streaming path records:
            # tool-input JSON and thinking are not the reply the user reads.
            if isinstance(delta, dict) and delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                self._text_parts.append(delta["text"])
        elif event_type == "message_stop":
            self.completed = True
        elif event_type == "error":
            self.errored = True
