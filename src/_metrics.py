"""Rolling-window aggregator for W&B-side back-pressure signals.

``api_call(method, path)`` in ``_logging.py`` already captures the two
numbers we care about:

  * ``rate_limited_ms``: total time the token bucket made the call wait
  * ``retries``: count of tenacity retries (so 1 means we retried once)

We plumb those into a process-global rolling window so operators can ask
"are we being throttled right now?" via the ``mcp-wandb://cache/stats``
resource.

Window defaults to 600 s. Records with zero signal are not stored, so the
window only fills when there's something worth seeing.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _ApiCallRecord:
    timestamp: float
    rate_limited_ms: int
    retries: int


class WandbApiMetrics:
    """Process-global, thread-safe rolling-window back-pressure aggregator."""

    def __init__(self, window_seconds: float = 600.0) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive.")
        self._window: float = float(window_seconds)
        self._records: deque[_ApiCallRecord] = deque()
        self._lock: threading.Lock = threading.Lock()

    @property
    def window_seconds(self) -> float:
        return self._window

    def record(self, rate_limited_ms: int = 0, retries: int = 0) -> None:
        """Record a single ``api_call`` outcome.

        Calls with no back-pressure (both counters zero) are silently
        dropped so the window only stores meaningful records.
        """
        if rate_limited_ms <= 0 and retries <= 0:
            return
        now = time.monotonic()
        with self._lock:
            self._records.append(
                _ApiCallRecord(
                    timestamp=now,
                    rate_limited_ms=int(rate_limited_ms),
                    retries=int(retries),
                )
            )
            self._evict_locked(now)

    def stats(self) -> dict[str, Any]:
        """Return a JSON-ready snapshot of the current window."""
        now = time.monotonic()
        with self._lock:
            self._evict_locked(now)
            n = len(self._records)
            total_rate_limited_ms = sum(r.rate_limited_ms for r in self._records)
            total_retries = sum(r.retries for r in self._records)
            max_rate_limited_ms = (
                max(r.rate_limited_ms for r in self._records) if self._records else 0
            )
        return {
            "window_seconds": self._window,
            "calls_with_pressure": n,
            "total_rate_limited_ms": int(total_rate_limited_ms),
            "total_retries": int(total_retries),
            "max_rate_limited_ms": int(max_rate_limited_ms),
        }

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def _evict_locked(self, now: float) -> None:
        cutoff = now - self._window
        while self._records and self._records[0].timestamp < cutoff:
            self._records.popleft()


_global_metrics: WandbApiMetrics | None = None
_metrics_lock: threading.Lock = threading.Lock()


def get_metrics() -> WandbApiMetrics:
    """Return the process-global metrics aggregator (lazy-creating it)."""
    global _global_metrics
    with _metrics_lock:
        if _global_metrics is None:
            _global_metrics = WandbApiMetrics()
        return _global_metrics


def reset_metrics() -> None:
    """Test hook: drop the aggregator so the next ``get_metrics()`` rebuilds."""
    global _global_metrics
    with _metrics_lock:
        _global_metrics = None


def record_api_call(rate_limited_ms: int = 0, retries: int = 0) -> None:
    """Module-level shim used by ``_logging.api_call`` to record a single call."""
    get_metrics().record(rate_limited_ms=rate_limited_ms, retries=retries)


def wandb_api_metrics_payload() -> dict[str, Any]:
    """JSON-ready snapshot for the ``mcp-wandb://cache/stats`` resource.

    Adds the configured rate-limit settings alongside the rolling window
    so the agent can narrate "we're being throttled at X requests/min, and
    over the last 10 minutes we've waited a total of Y ms for the bucket."
    """
    from .settings import get_settings

    stats = get_metrics().stats()
    settings = get_settings()
    stats["rate_limit_per_min"] = settings.rate_limit_per_min
    stats["rate_limit_burst"] = settings.rate_limit_burst
    return stats
