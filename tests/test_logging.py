"""Tests for _logging.py: formatter, decorator, context manager."""

from __future__ import annotations

import json
import logging

import pytest

from mcp_wandb import _logging
from mcp_wandb.settings import Settings, set_settings


@pytest.fixture(autouse=True)
def _reset_logging() -> None:
    _logging._configured = False
    yield
    _logging._configured = False


def _capture(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    caplog.set_level(logging.DEBUG, logger="mcp_wandb")
    return caplog.records


def test_instrumented_logs_success(caplog: pytest.LogCaptureFixture) -> None:
    records = _capture(caplog)

    @_logging.instrumented("widget")
    def widget(x: int) -> int:
        return x + 1

    assert widget(2) == 3
    tool_records = [r for r in records if r.name == "mcp_wandb.tool"]
    assert len(tool_records) == 1
    rec = tool_records[0]
    assert rec.message == "tool.call"
    assert rec.tool == "widget"
    assert rec.success is True
    assert isinstance(rec.latency_ms, int)


def test_instrumented_logs_failure_then_reraises(caplog: pytest.LogCaptureFixture) -> None:
    records = _capture(caplog)

    @_logging.instrumented("boom")
    def boom() -> None:
        raise ValueError("nope")

    with pytest.raises(ValueError):
        boom()
    rec = next(r for r in records if r.name == "mcp_wandb.tool")
    assert rec.success is False
    assert rec.error_class == "ValueError"


def test_instrumented_preserves_signature() -> None:
    import inspect

    @_logging.instrumented("typed")
    def typed(a: int, b: str = "x") -> bool:
        return True

    sig = inspect.signature(typed)
    assert list(sig.parameters) == ["a", "b"]
    assert sig.parameters["b"].default == "x"


def test_api_call_emits_one_summary(caplog: pytest.LogCaptureFixture) -> None:
    records = _capture(caplog)
    with _logging.api_call("runs", "demo/x") as counters:
        counters["retries"] = 2
        counters["rate_limited_ms"] = 150
    api_records = [r for r in records if r.name == "mcp_wandb.api"]
    assert len(api_records) == 1
    rec = api_records[0]
    assert rec.method == "runs"
    assert rec.path == "demo/x"
    assert rec.retries == 2
    assert rec.rate_limited_ms == 150


def test_text_formatter_renders_extras() -> None:
    record = logging.LogRecord(
        name="mcp_wandb.tool",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg="tool.call",
        args=(),
        exc_info=None,
    )
    record.tool = "list_projects"
    record.latency_ms = 47
    record.success = True
    out = _logging._TextFormatter().format(record)
    assert "tool=list_projects" in out
    assert "latency_ms=47" in out
    assert "success=true" in out


def test_json_formatter_is_valid_json() -> None:
    record = logging.LogRecord(
        name="mcp_wandb.tool",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg="tool.call",
        args=(),
        exc_info=None,
    )
    record.tool = "list_projects"
    record.latency_ms = 47
    out = _logging._JsonFormatter().format(record)
    parsed = json.loads(out)
    assert parsed["msg"] == "tool.call"
    assert parsed["tool"] == "list_projects"
    assert parsed["latency_ms"] == 47


def test_configure_logging_is_idempotent() -> None:
    set_settings(Settings())
    _logging.configure_logging()
    _logging.configure_logging()  # second call must not stack handlers
    root = logging.getLogger("mcp_wandb")
    assert len(root.handlers) == 1
