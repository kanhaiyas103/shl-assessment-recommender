"""Structured, secret-safe application logging."""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any, Final

_STANDARD_LOG_RECORD_FIELDS: Final[frozenset[str]] = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
)


class JsonFormatter(logging.Formatter):
    """Render one JSON object per log record for hosted environments."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize stable fields and explicitly supplied structured context."""
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_LOG_RECORD_FIELDS and key not in {"message", "asctime"}
        }
        if extras:
            payload["context"] = extras
        if record.exc_info is not None:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str) -> None:
    """Configure the process root logger idempotently."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    logging.captureWarnings(True)
    logging.getLogger("uvicorn.access").propagate = True
