
from __future__ import annotations
import json
import logging
import sys
from typing import Any

_STRUCTURED_FIELDS = (
    "query",
    "company",
    "mode",
    "tools_invoked",
    "latency_ms",
    "status_code",
    "path",
    "method",
    "round",
    "attempt",
    "retries",
    "tool",
    "args",
    "ticker",
    "company_slug",
    "error",
    "request_id",
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for field in _STRUCTURED_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Idempotent: safe to call multiple times (e.g. once from api/main.py,
    once from a test fixture) without duplicating handlers."""
    root = logging.getLogger("sovereign_fa")
    root.setLevel(level)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.propagate = False
