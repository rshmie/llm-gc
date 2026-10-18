"""How the application writes its log: a line per record, as text for a person or JSON for a program.

Application code. The library only creates log records; the handler, the level and the
format are the application's choice, made here once at startup. Either format names the
request a line was written in, so a line can be matched to its row on the Requests tab.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Literal

from llm_gc.web.request_id import current_request_id

LogFormat = Literal["text", "json"]

# Libraries whose INFO lines describe their own plumbing, once per request.
_QUIET_LOGGERS = ("httpx2",)

_TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s%(request_tag)s: %(message)s"
_TEXT_TIME_FORMAT = "%H:%M:%S"

# The attributes every record has. Any other attribute came from `extra`, which the JSON format
# writes out as fields; the text format shows only the message.
_RECORD_ATTRIBUTES = frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime", "request_id",
                                                                   "request_tag"}


def configure_logging(level: str, log_format: LogFormat) -> None:
    """Write every log record to stderr in `log_format`, each with the ID of the request it was written in.

    Like `logging.basicConfig`, it changes nothing when the root logger already has a handler,
    so a test runner that captures logs keeps doing so.
    """
    handler = logging.StreamHandler()
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(JsonFormatter() if log_format == "json" else TextFormatter())
    logging.basicConfig(level=level, handlers=[handler])
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


class RequestIdFilter(logging.Filter):
    """Adds `request_id` to each record: the ID of the request it was written in, or None outside one.

    Attached to the handler rather than to a logger, because a logger's filters never see the
    records its child loggers pass up, and nearly every record comes from a child logger
    such as `llm_gc.engine.gc_service`.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = current_request_id()
        return True


class TextFormatter(logging.Formatter):
    """A line for a person: time, level, logger, the request's ID in brackets when there is one, the message."""

    def __init__(self) -> None:
        super().__init__(_TEXT_FORMAT, datefmt=_TEXT_TIME_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        request_id = getattr(record, "request_id", None)
        record.request_tag = f" [{request_id}]" if request_id else ""
        return super().format(record)


class JsonFormatter(logging.Formatter):
    """One JSON object per line: time, level, logger, message, request ID, and every `extra` field."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None),
        }
        # The `extra` fields follow the line's own, and never replace one.
        entry.update((key, value) for key, value in vars(record).items()
                     if key not in _RECORD_ATTRIBUTES and key not in entry)
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        # default=str: an `extra` value JSON has no type for, such as an enum, is written as its text
        # rather than costing the whole line.
        return json.dumps(entry, default=str)
