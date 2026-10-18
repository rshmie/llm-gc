"""The application's log lines: each names the request it was written in, as text or as JSON."""

import asyncio
import json
import logging
import sys
from collections.abc import Callable
from enum import Enum

from llm_gc.cli.logs import JsonFormatter, RequestIdFilter, TextFormatter
from llm_gc.proxy.request_id import REQUEST_ID_CHARS, RequestIdMiddleware


def _record(message: str = "claude -> 200", *, extra: dict | None = None,
            exc_info: bool = False) -> logging.LogRecord:
    record = logging.makeLogRecord({"name": "llm_gc.proxy.server", "levelno": logging.INFO, "levelname": "INFO",
                                    "msg": message, **(extra or {})})
    if exc_info:
        try:
            raise ValueError("boom")
        except ValueError:
            record.exc_info = sys.exc_info()
    return record


def _inside_a_request(action: Callable[[], None]) -> None:
    """Run `action` as the proxy's routes run: inside the middleware that gives the request its ID."""
    async def app(scope, receive, send) -> None:
        action()

    asyncio.run(RequestIdMiddleware(app)({"type": "http"}, None, None))


def _stamped(record: logging.LogRecord, *, in_a_request: bool) -> logging.LogRecord:
    if in_a_request:
        _inside_a_request(lambda: RequestIdFilter().filter(record))
    else:
        RequestIdFilter().filter(record)
    return record


class TestTheFilter:
    def test_a_record_written_in_a_request_gets_its_id(self):
        record = _stamped(_record(), in_a_request=True)

        assert isinstance(record.request_id, str) and len(record.request_id) == REQUEST_ID_CHARS

    def test_a_record_written_outside_a_request_gets_none(self):
        assert _stamped(_record(), in_a_request=False).request_id is None


class TestTheTextFormat:
    def test_a_line_in_a_request_names_it_in_brackets_after_the_logger(self):
        record = _stamped(_record(), in_a_request=True)

        line = TextFormatter().format(record)

        assert f"INFO llm_gc.proxy.server [{record.request_id}]: claude -> 200" in line

    def test_a_line_outside_a_request_has_no_brackets(self):
        line = TextFormatter().format(_stamped(_record(), in_a_request=False))

        assert "INFO llm_gc.proxy.server: claude -> 200" in line


class TestTheJsonFormat:
    def test_a_line_carries_the_message_the_request_id_and_every_extra_field(self):
        record = _stamped(_record(extra={"status": 200, "tokens_forwarded": 812}), in_a_request=True)

        entry = json.loads(JsonFormatter().format(record))

        assert entry["message"] == "claude -> 200"
        assert entry["level"] == "INFO" and entry["logger"] == "llm_gc.proxy.server"
        assert entry["request_id"] == record.request_id
        assert (entry["status"], entry["tokens_forwarded"]) == (200, 812)
        assert entry["time"].endswith("+00:00"), "UTC, so lines from different machines sort together"

    def test_the_records_own_attributes_are_not_repeated_as_fields(self):
        entry = json.loads(JsonFormatter().format(_stamped(_record(), in_a_request=False)))

        assert entry["request_id"] is None
        assert not {"msg", "args", "levelno", "pathname"} & entry.keys()

    def test_an_exception_is_written_as_text_in_the_same_line(self):
        line = JsonFormatter().format(_stamped(_record(exc_info=True), in_a_request=False))

        assert "\n" not in line
        assert "ValueError: boom" in json.loads(line)["exception"]

    def test_a_value_json_cannot_hold_is_written_as_text_rather_than_losing_the_line(self):
        class Status(Enum):
            COMPLETED = "completed"

        entry = json.loads(JsonFormatter().format(_stamped(_record(extra={"outcome": Status.COMPLETED}),
                                                           in_a_request=False)))

        assert entry["outcome"] == "Status.COMPLETED"

    def test_an_extra_field_cannot_overwrite_the_lines_own(self):
        entry = json.loads(JsonFormatter().format(_stamped(_record(extra={"level": "forged"}), in_a_request=False)))

        assert entry["level"] == "INFO"
