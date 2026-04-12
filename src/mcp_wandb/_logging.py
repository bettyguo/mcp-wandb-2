"""Structured logging for `mcp-wandb`.

* In stdio mode (``log_format='text'``) emits ``timestamp LEVEL name | message [k=v ...]``.
* In hosted mode (``log_format='json'``) emits one JSON object per line.
* Every tool call produces exactly one summary line via ``tool_call_logger``.
* Every W&B API call produces one summary line via ``api_call_logger``.

Keep this module dependency-free so tests can run with just stdlib + pytest.
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import sys
import time
from collections.abc import Callable, Generator
from typing import Any, ParamSpec, TypeVar

from .settings import get_settings

_P = ParamSpec("_P")
_R = TypeVar("_R")


_TOOL_LOGGER_NAME = "mcp_wandb.tool"
_API_LOGGER_NAME = "mcp_wandb.api"

_configured = False


class _TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<5} {record.name} | {record.getMessage()}"
        extras = _extra_pairs(record)
        if extras:
            base += " " + " ".join(f"{k}={_fmt(v)}" for k, v in extras)
        return base


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": int(record.created * 1000),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        payload.update(dict(_extra_pairs(record)))
        return json.dumps(payload, default=str)


def _extra_pairs(record: logging.LogRecord) -> list[tuple[str, Any]]:
    skip = {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "message", "module",
        "msecs", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName", "taskName",
    }
    return [(k, v) for k, v in record.__dict__.items() if k not in skip and not k.startswith("_")]


def _fmt(v: Any) -> str:
    if isinstance(v, str) and " " not in v and "=" not in v:
        return v
    return json.dumps(v, default=str)


def configure_logging(force: bool = False) -> None:
    """Idempotent; call once at server startup."""
    global _configured
    if _configured and not force:
        return
    settings = get_settings()
    root = logging.getLogger("mcp_wandb")
    root.setLevel(settings.log_level.upper())
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(stream=sys.stderr)
    if settings.log_format == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(_TextFormatter())
    root.addHandler(handler)
    root.propagate = False
    _configured = True


def get_logger(suffix: str) -> logging.Logger:
    return logging.getLogger(f"mcp_wandb.{suffix}")


# ---------------------------------------------------------------------------
# Tool-call instrumentation
# ---------------------------------------------------------------------------


def instrumented(
    tool_name: str,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Wrap a tool function with one structured log line per invocation.

    Uses ``ParamSpec`` + ``TypeVar`` so the wrapper preserves the wrapped
    function's exact signature, which matters for both ``inspect.signature``
    (FastMCP introspects it for JSON-schema generation) and ``mypy --strict``.
    """

    log = logging.getLogger(_TOOL_LOGGER_NAME)

    def decorator(fn: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(fn)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            from . import _telemetry

            start = time.monotonic()
            success = True
            error_class: str | None = None
            with _telemetry.span(f"mcp_wandb.tool.{tool_name}", tool=tool_name) as span_obj:
                try:
                    return fn(*args, **kwargs)
                except BaseException as exc:
                    success = False
                    error_class = type(exc).__name__
                    _telemetry.record_exception(span_obj, exc)
                    raise
                finally:
                    latency_ms = int((time.monotonic() - start) * 1000)
                    _telemetry.set_attribute(span_obj, "latency_ms", latency_ms)
                    _telemetry.set_attribute(span_obj, "success", success)
                    extra: dict[str, Any] = {
                        "tool": tool_name,
                        "latency_ms": latency_ms,
                        "success": success,
                    }
                    if error_class:
                        extra["error_class"] = error_class
                    log.info("tool.call", extra=extra)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# W&B API-call instrumentation (used by client.py)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def api_call(method: str, path: str | None = None) -> Generator[dict[str, Any], None, None]:
    """Context manager that emits one summary line for a W&B API operation.

    ``yield``s a mutable counters dict so the caller can record ``retries`` and
    ``rate_limited_ms`` inside the with-block.
    """
    from . import _telemetry

    log = logging.getLogger(_API_LOGGER_NAME)
    counters: dict[str, Any] = {"retries": 0, "rate_limited_ms": 0}
    start = time.monotonic()
    success = True
    error_class: str | None = None
    with _telemetry.span(
        f"mcp_wandb.api.{method}",
        method=method,
        path=path,
    ) as span_obj:
        try:
            yield counters
        except BaseException as exc:
            success = False
            error_class = type(exc).__name__
            _telemetry.record_exception(span_obj, exc)
            raise
        finally:
            latency_ms = int((time.monotonic() - start) * 1000)
            _telemetry.set_attribute(span_obj, "latency_ms", latency_ms)
            _telemetry.set_attribute(span_obj, "success", success)
            _telemetry.set_attribute(span_obj, "retries", counters.get("retries", 0))
            _telemetry.set_attribute(
                span_obj, "rate_limited_ms", counters.get("rate_limited_ms", 0)
            )

            # Feed the rolling-window back-pressure aggregator. Lazy import
            # so test suites that don't touch the W&B API path don't pay the
            # import cost.
            from . import _metrics

            _metrics.record_api_call(
                rate_limited_ms=int(counters.get("rate_limited_ms", 0)),
                retries=int(counters.get("retries", 0)),
            )

            extra: dict[str, Any] = {
                "method": method,
                "latency_ms": latency_ms,
                "success": success,
                "retries": counters.get("retries", 0),
                "rate_limited_ms": counters.get("rate_limited_ms", 0),
            }
            if path:
                extra["path"] = path
            if error_class:
                extra["error_class"] = error_class
            log.info("wandb.api", extra=extra)
