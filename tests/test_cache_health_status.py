"""Tests for the cache-health top-line status field."""

from __future__ import annotations

import pytest

from mcp_wandb._cache import (
    _STATUS_BUSY_MAX_RATE_LIMITED_MS,
    _STATUS_DEGRADED_TOTAL_RATE_LIMITED_MS,
    _STATUS_DEGRADED_TOTAL_RETRIES,
    _status_from_wandb_api,
    cache_health_payload,
)
from mcp_wandb._metrics import record_api_call, reset_metrics
from mcp_wandb.settings import Settings, set_settings


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_metrics()
    yield
    reset_metrics()


# ---------------------------------------------------------------------------
# _status_from_wandb_api: invariants
# ---------------------------------------------------------------------------


def test_status_ok_when_no_traffic() -> None:
    assert _status_from_wandb_api({}) == "ok"
    assert (
        _status_from_wandb_api(
            {
                "max_rate_limited_ms": 0,
                "total_rate_limited_ms": 0,
                "total_retries": 0,
            }
        )
        == "ok"
    )


def test_status_busy_when_max_wait_exceeds_threshold() -> None:
    p = {
        "max_rate_limited_ms": _STATUS_BUSY_MAX_RATE_LIMITED_MS + 1,
        "total_rate_limited_ms": 0,
        "total_retries": 0,
    }
    assert _status_from_wandb_api(p) == "busy"


def test_status_degraded_when_total_wait_exceeds_threshold() -> None:
    p = {
        "max_rate_limited_ms": 100,
        "total_rate_limited_ms": _STATUS_DEGRADED_TOTAL_RATE_LIMITED_MS + 1,
        "total_retries": 0,
    }
    assert _status_from_wandb_api(p) == "degraded"


def test_status_degraded_when_total_retries_exceeds_threshold() -> None:
    p = {
        "max_rate_limited_ms": 100,
        "total_rate_limited_ms": 0,
        "total_retries": _STATUS_DEGRADED_TOTAL_RETRIES + 1,
    }
    assert _status_from_wandb_api(p) == "degraded"


def test_status_busy_wins_over_degraded() -> None:
    """If both conditions match, ``busy`` takes precedence."""
    p = {
        "max_rate_limited_ms": _STATUS_BUSY_MAX_RATE_LIMITED_MS + 1,
        "total_rate_limited_ms": _STATUS_DEGRADED_TOTAL_RATE_LIMITED_MS + 1,
        "total_retries": _STATUS_DEGRADED_TOTAL_RETRIES + 1,
    }
    assert _status_from_wandb_api(p) == "busy"


def test_status_threshold_is_strict_greater_than() -> None:
    """Exactly at the threshold is ``ok`` / ``degraded`` boundary, not yet busy."""
    p_at_busy = {
        "max_rate_limited_ms": _STATUS_BUSY_MAX_RATE_LIMITED_MS,  # exactly equal
        "total_rate_limited_ms": 0,
        "total_retries": 0,
    }
    assert _status_from_wandb_api(p_at_busy) == "ok"

    p_at_degraded = {
        "max_rate_limited_ms": 0,
        "total_rate_limited_ms": _STATUS_DEGRADED_TOTAL_RATE_LIMITED_MS,  # exactly equal
        "total_retries": 0,
    }
    assert _status_from_wandb_api(p_at_degraded) == "ok"


# ---------------------------------------------------------------------------
# cache_health_payload: status field appears + reflects api_call recording
# ---------------------------------------------------------------------------


def test_payload_status_ok_with_no_api_traffic() -> None:
    set_settings(Settings(cache_enabled=False, cache_dir=None))
    p = cache_health_payload()
    assert p["status"] == "ok"


def test_payload_status_busy_after_a_long_single_wait() -> None:
    set_settings(Settings(cache_enabled=False, cache_dir=None))
    record_api_call(rate_limited_ms=_STATUS_BUSY_MAX_RATE_LIMITED_MS + 100, retries=0)
    p = cache_health_payload()
    assert p["status"] == "busy"


def test_payload_status_degraded_after_sustained_pressure() -> None:
    set_settings(Settings(cache_enabled=False, cache_dir=None))
    # Many smaller waits accumulate past the degraded total threshold but
    # never breach the busy single-wait threshold.
    n = (_STATUS_DEGRADED_TOTAL_RATE_LIMITED_MS // 1000) + 2  # ≈ 62 calls @ 1 s
    for _ in range(n):
        record_api_call(rate_limited_ms=1000, retries=0)
    p = cache_health_payload()
    assert p["status"] == "degraded"


def test_payload_status_appears_first_in_key_order() -> None:
    """``status`` ships before the per-layer blocks so the payload reads top-down."""
    p = cache_health_payload()
    keys = list(p.keys())
    assert keys[0] == "status"
    assert "memory" in keys
    assert "disk" in keys
    assert "wandb_api" in keys
