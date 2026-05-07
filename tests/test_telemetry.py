"""Tests for the OpenTelemetry hooks."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from mcp_wandb import _logging, _telemetry
from mcp_wandb.settings import Settings, set_settings


@pytest.fixture(autouse=True)
def _reset_telemetry() -> None:
    _telemetry.reset_telemetry()
    yield
    _telemetry.reset_telemetry()


# ---------------------------------------------------------------------------
# Disabled-by-default path
# ---------------------------------------------------------------------------


def test_init_telemetry_noop_when_disabled() -> None:
    set_settings(Settings(telemetry_enabled=False))
    _telemetry.init_telemetry()
    assert _telemetry._state.tracer is None


def test_span_yields_none_when_no_tracer() -> None:
    _telemetry.reset_telemetry()
    with _telemetry.span("foo") as s:
        assert s is None


def test_set_attribute_silent_when_none() -> None:
    # Should never raise.
    _telemetry.set_attribute(None, "k", "v")


def test_record_exception_silent_when_none() -> None:
    _telemetry.record_exception(None, RuntimeError("boom"))


# ---------------------------------------------------------------------------
# Soft-optional dependency
# ---------------------------------------------------------------------------


def test_init_telemetry_handles_missing_opentelemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    """When enabled but the package is missing, init should silently no-op."""
    set_settings(Settings(telemetry_enabled=True))

    import builtins

    original_import = builtins.__import__

    def _import_blocking_opentelemetry(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("opentelemetry"):
            raise ImportError(f"simulated missing module: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import_blocking_opentelemetry)
    _telemetry.init_telemetry()
    # No tracer; calls remain safe.
    assert _telemetry._state.tracer is None
    with _telemetry.span("anything") as s:
        assert s is None


# ---------------------------------------------------------------------------
# Mocked-tracer path
# ---------------------------------------------------------------------------


def _install_fake_tracer() -> tuple[MagicMock, list[tuple[str, dict[str, Any]]]]:
    """Patch a fake tracer into the global state and return capture refs."""
    captured: list[tuple[str, dict[str, Any]]] = []

    class _FakeSpan:
        def __init__(self, name: str) -> None:
            self._name = name
            self._attrs: dict[str, Any] = {}

        def set_attribute(self, key: str, value: Any) -> None:
            self._attrs[key] = value

        def record_exception(self, exc: BaseException) -> None:
            self._attrs["__exception__"] = type(exc).__name__

        def __enter__(self) -> _FakeSpan:
            return self

        def __exit__(self, *args: Any) -> None:
            captured.append((self._name, dict(self._attrs)))

    fake_tracer = MagicMock()
    fake_tracer.start_as_current_span.side_effect = lambda name: _FakeSpan(name)
    _telemetry._state.tracer = fake_tracer
    _telemetry._state.initialized = True
    return fake_tracer, captured


def test_span_emits_with_attributes() -> None:
    _, captured = _install_fake_tracer()
    with _telemetry.span("test.span", tool="x", latency_ms=42) as s:
        assert s is not None
    assert len(captured) == 1
    name, attrs = captured[0]
    assert name == "test.span"
    assert attrs["tool"] == "x"
    assert attrs["latency_ms"] == 42


def test_span_skips_none_attribute_values() -> None:
    _, captured = _install_fake_tracer()
    with _telemetry.span("test.span", tool="x", maybe=None):
        pass
    _name, attrs = captured[0]
    assert "maybe" not in attrs
    assert attrs["tool"] == "x"


# ---------------------------------------------------------------------------
# Integration: @instrumented decorator emits a span
# ---------------------------------------------------------------------------


def test_instrumented_decorator_emits_tool_span() -> None:
    _, captured = _install_fake_tracer()

    @_logging.instrumented("my_tool")
    def my_tool(x: int) -> int:
        return x + 1

    assert my_tool(5) == 6
    assert len(captured) == 1
    name, attrs = captured[0]
    assert name == "mcp_wandb.tool.my_tool"
    assert attrs["tool"] == "my_tool"
    assert attrs["success"] is True
    assert isinstance(attrs["latency_ms"], int)


def test_instrumented_records_exception_on_span() -> None:
    _, captured = _install_fake_tracer()

    @_logging.instrumented("boom")
    def boom() -> None:
        raise ValueError("nope")

    with pytest.raises(ValueError):
        boom()
    _name, attrs = captured[0]
    assert attrs["__exception__"] == "ValueError"
    assert attrs["success"] is False


# ---------------------------------------------------------------------------
# Integration: api_call context manager emits a span
# ---------------------------------------------------------------------------


def test_api_call_emits_span_with_counters() -> None:
    _, captured = _install_fake_tracer()
    with _logging.api_call("runs", "demo/x") as counters:
        counters["retries"] = 2
        counters["rate_limited_ms"] = 150
    name, attrs = captured[0]
    assert name == "mcp_wandb.api.runs"
    assert attrs["method"] == "runs"
    assert attrs["path"] == "demo/x"
    assert attrs["retries"] == 2
    assert attrs["rate_limited_ms"] == 150
    assert attrs["success"] is True


def test_api_call_records_exception() -> None:
    _, captured = _install_fake_tracer()
    with pytest.raises(RuntimeError), _logging.api_call("run", "demo/x/run-1"):
        raise RuntimeError("fail")
    _name, attrs = captured[0]
    assert attrs["__exception__"] == "RuntimeError"
    assert attrs["success"] is False
