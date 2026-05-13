"""OpenTelemetry hooks: opt-in observability for hosted deployments.

`@instrumented` (tool entrypoints) and `api_call` (W&B API operations)
already capture every datum a span needs: name, latency, success/error,
retry count, rate-limit wait. This module just sets up an OpenTelemetry
tracer when the operator wants it, and exposes a ``span(name, **attrs)``
context manager the existing hooks call.

Soft-optional dependency: the ``opentelemetry-*`` packages are not
required at import time. If telemetry is disabled (default) or the
packages aren't installed, every entry point in this module is a no-op.

Configuration:

* ``Settings.telemetry_enabled`` (default ``False``) or
  ``MCP_WANDB_OTEL_ENABLED=1`` enables it.
* The OTLP exporter, if installed, picks up standard environment
  variables: ``OTEL_EXPORTER_OTLP_ENDPOINT``,
  ``OTEL_EXPORTER_OTLP_HEADERS``, ``OTEL_SERVICE_NAME``.
* If the OTLP exporter is not installed but the SDK is, falls back to
  the SDK's ``ConsoleSpanExporter`` so spans still reach somewhere
  visible (helpful for operators iterating on local config).
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

from . import __version__
from .settings import get_settings

logger = logging.getLogger("mcp_wandb.telemetry")


@dataclass
class _TelemetryState:
    tracer: Any | None = None
    initialized: bool = False


_state: _TelemetryState = _TelemetryState()
_init_lock: threading.Lock = threading.Lock()


def init_telemetry() -> None:
    """Idempotent; call once at server startup.

    If telemetry is disabled or the ``opentelemetry-*`` packages aren't
    installed, this leaves ``_state.tracer = None`` and every later
    ``span(...)`` call is a no-op. We never raise.
    """
    with _init_lock:
        if _state.initialized:
            return
        _state.initialized = True
        if not get_settings().telemetry_enabled:
            return
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import (
                BatchSpanProcessor,
                ConsoleSpanExporter,
            )
        except ImportError:
            logger.info(
                "telemetry enabled but opentelemetry-sdk not installed; "
                "spans will be a no-op. Install with `pip install mcp-wandb[telemetry]`."
            )
            return

        exporter: Any
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter()
        except ImportError:
            logger.info(
                "OTLP gRPC exporter not installed; falling back to "
                "ConsoleSpanExporter. Install opentelemetry-exporter-otlp "
                "to ship spans to a collector."
            )
            exporter = ConsoleSpanExporter()

        resource = Resource.create(
            {
                "service.name": "mcp-wandb",
                "service.version": __version__,
            }
        )
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _state.tracer = trace.get_tracer("mcp_wandb", __version__)


def reset_telemetry() -> None:
    """Test hook: forget any tracer setup so init_telemetry() runs fresh."""
    with _init_lock:
        _state.tracer = None
        _state.initialized = False


@contextlib.contextmanager
def span(name: str, **attrs: Any) -> Generator[Any, None, None]:
    """Context manager that yields a tracer span (or ``None`` when off)."""
    tracer = _state.tracer
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(name) as current:
        for k, v in attrs.items():
            if v is None:
                continue
            current.set_attribute(k, v)
        yield current


def set_attribute(current: Any, key: str, value: Any) -> None:
    """Defensive `span.set_attribute`; silently no-ops on None/exception."""
    if current is None or value is None:
        return
    # We never want telemetry to break the call path.
    with contextlib.suppress(Exception):
        current.set_attribute(key, value)


def record_exception(current: Any, exc: BaseException) -> None:
    """Defensive `span.record_exception`; silently no-ops on None/exception."""
    if current is None:
        return
    try:
        current.record_exception(exc)
        current.set_attribute("error.type", type(exc).__name__)
    except Exception:
        pass
