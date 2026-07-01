"""Tests for structured logging."""

import json
import logging
import sys

from shl_agent.utils.logging import JsonFormatter


def raise_failure() -> None:
    raise RuntimeError("failure")


def test_json_formatter_emits_structured_context() -> None:
    record = logging.LogRecord(
        name="shl_agent.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="request completed",
        args=(),
        exc_info=None,
    )
    record.request_id = "request-123"

    payload = json.loads(JsonFormatter().format(record))

    assert payload["message"] == "request completed"
    assert payload["context"]["request_id"] == "request-123"


def test_json_formatter_includes_exception() -> None:
    try:
        raise_failure()
    except RuntimeError:
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="shl_agent.test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="request failed",
        args=(),
        exc_info=exc_info,
    )

    payload = json.loads(JsonFormatter().format(record))

    assert "RuntimeError: failure" in payload["exception"]
