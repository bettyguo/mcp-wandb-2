"""Tests for the W&B back-pressure rolling-window aggregator."""

from __future__ import annotations

import threading
import time

import pytest

from mcp_wandb import _logging
from mcp_wandb._cache import cache_health_payload
from mcp_wandb._metrics import (
    WandbApiMetrics,
    get_metrics,
    record_api_call,
    reset_metrics,
    wandb_api_metrics_payload,
)
from mcp_wandb.settings import Settings, set_settings


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_metrics()
    yield
    reset_metrics()


# ---------------------------------------------------------------------------
# WandbApiMetrics unit tests
# ---------------------------------------------------------------------------


def test_empty_aggregator_returns_zeros() -> None:
    m = WandbApiMetrics(window_seconds=60.0)
    s = m.stats()
    assert s["window_seconds"] == 60.0
    assert s["calls_with_pressure"] == 0
    assert s["total_rate_limited_ms"] == 0
    assert s["total_retries"] == 0
    assert s["max_rate_limited_ms"] == 0


def test_record_with_zero_signal_is_dropped() -> None:
    m = WandbApiMetrics(window_seconds=60.0)
    m.record(rate_limited_ms=0, retries=0)
    assert m.stats()["calls_with_pressure"] == 0


def test_record_with_rate_limit_only() -> None:
    m = WandbApiMetrics(window_seconds=60.0)
    m.record(rate_limited_ms=120)
    s = m.stats()
    assert s["calls_with_pressure"] == 1
    assert s["total_rate_limited_ms"] == 120
    assert s["max_rate_limited_ms"] == 120
    assert s["total_retries"] == 0


def test_record_with_retries_only() -> None:
    m = WandbApiMetrics(window_seconds=60.0)
    m.record(retries=2)
    s = m.stats()
    assert s["calls_with_pressure"] == 1
    assert s["total_retries"] == 2
    assert s["total_rate_limited_ms"] == 0


def test_aggregator_sums_across_records() -> None:
    m = WandbApiMetrics(window_seconds=60.0)
    m.record(rate_limited_ms=100, retries=1)
    m.record(rate_limited_ms=50, retries=0)
    m.record(rate_limited_ms=200, retries=2)
    s = m.stats()
    assert s["calls_with_pressure"] == 3
    assert s["total_rate_limited_ms"] == 350
    assert s["total_retries"] == 3
    assert s["max_rate_limited_ms"] == 200


def test_window_evicts_old_records() -> None:
    m = WandbApiMetrics(window_seconds=0.1)
    m.record(rate_limited_ms=999, retries=0)
    assert m.stats()["calls_with_pressure"] == 1
    time.sleep(0.15)
    s = m.stats()
    assert s["calls_with_pressure"] == 0
    assert s["total_rate_limited_ms"] == 0


def test_clear_resets_window() -> None:
    m = WandbApiMetrics(window_seconds=60.0)
    m.record(rate_limited_ms=100, retries=2)
    m.clear()
    s = m.stats()
    assert s["calls_with_pressure"] == 0
    assert s["total_rate_limited_ms"] == 0
    assert s["total_retries"] == 0


def test_init_validates_window() -> None:
    with pytest.raises(ValueError):
        WandbApiMetrics(window_seconds=0)
    with pytest.raises(ValueError):
        WandbApiMetrics(window_seconds=-5)


def test_thread_safety_concurrent_records() -> None:
    m = WandbApiMetrics(window_seconds=60.0)
    n_threads = 8
    per_thread = 100

    def _hammer() -> None:
        for _ in range(per_thread):
            m.record(rate_limited_ms=10, retries=1)

    threads = [threading.Thread(target=_hammer) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    s = m.stats()
    assert s["calls_with_pressure"] == n_threads * per_thread
    assert s["total_rate_limited_ms"] == n_threads * per_thread * 10
    assert s["total_retries"] == n_threads * per_thread


# ---------------------------------------------------------------------------
# Module-level singleton + record_api_call helper
# ---------------------------------------------------------------------------


def test_get_metrics_returns_same_instance() -> None:
    a = get_metrics()
    b = get_metrics()
    assert a is b


def test_reset_metrics_drops_singleton() -> None:
    a = get_metrics()
    a.record(rate_limited_ms=100)
    reset_metrics()
    b = get_metrics()
    assert b is not a
    assert b.stats()["calls_with_pressure"] == 0


def test_record_api_call_writes_to_singleton() -> None:
    record_api_call(rate_limited_ms=50, retries=1)
    record_api_call(rate_limited_ms=0, retries=0)  # dropped
    record_api_call(rate_limited_ms=100, retries=0)
    s = get_metrics().stats()
    assert s["calls_with_pressure"] == 2
    assert s["total_rate_limited_ms"] == 150


# ---------------------------------------------------------------------------
# wandb_api_metrics_payload: settings exposed
# ---------------------------------------------------------------------------


def test_payload_includes_rate_limit_settings() -> None:
    set_settings(Settings(rate_limit_per_min=42, rate_limit_burst=99))
    record_api_call(rate_limited_ms=10, retries=0)
    p = wandb_api_metrics_payload()
    assert p["rate_limit_per_min"] == 42
    assert p["rate_limit_burst"] == 99
    assert p["calls_with_pressure"] == 1


# ---------------------------------------------------------------------------
# Integration with api_call() context manager
# ---------------------------------------------------------------------------


def test_api_call_records_back_pressure_signals() -> None:
    set_settings(Settings(rate_limit_per_min=60))
    with _logging.api_call("runs", "demo/x") as counters:
        counters["rate_limited_ms"] = 250
        counters["retries"] = 1

    s = get_metrics().stats()
    assert s["calls_with_pressure"] == 1
    assert s["total_rate_limited_ms"] == 250
    assert s["total_retries"] == 1


def test_api_call_with_no_pressure_does_not_pollute_window() -> None:
    with _logging.api_call("runs", "demo/x"):
        pass  # no counters set
    assert get_metrics().stats()["calls_with_pressure"] == 0


def test_api_call_records_on_exception_path_too() -> None:
    """If the call raises, we still want the back-pressure recorded."""
    with pytest.raises(RuntimeError), _logging.api_call("runs", "demo/x") as counters:
        counters["rate_limited_ms"] = 30
        raise RuntimeError("boom")
    s = get_metrics().stats()
    assert s["calls_with_pressure"] == 1
    assert s["total_rate_limited_ms"] == 30


# ---------------------------------------------------------------------------
# Integration with cache_health_payload
# ---------------------------------------------------------------------------


def test_cache_health_payload_includes_wandb_api_block() -> None:
    set_settings(Settings(cache_enabled=False, cache_dir=None, rate_limit_per_min=60))
    record_api_call(rate_limited_ms=500, retries=3)
    p = cache_health_payload()
    assert "wandb_api" in p
    assert p["wandb_api"]["total_rate_limited_ms"] == 500
    assert p["wandb_api"]["total_retries"] == 3
    assert p["wandb_api"]["rate_limit_per_min"] == 60


def test_cache_health_payload_wandb_api_zeroed_when_no_traffic() -> None:
    """Even with no API calls, the wandb_api block is present (just zeroed)."""
    set_settings(Settings(cache_enabled=False, cache_dir=None))
    p = cache_health_payload()
    assert "wandb_api" in p
    assert p["wandb_api"]["calls_with_pressure"] == 0
    assert p["wandb_api"]["total_rate_limited_ms"] == 0
