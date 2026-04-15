"""WandbClient retry / rate-limit policy tests (no real network)."""

from __future__ import annotations

import pytest

from mcp_wandb import client as client_mod
from mcp_wandb.settings import Settings, set_settings


def test_is_retryable_classifies_429() -> None:
    assert client_mod.WandbClient._is_retryable(Exception("429 Too Many Requests")) is True
    assert client_mod.WandbClient._is_retryable(Exception("Rate limit exceeded")) is True


def test_is_retryable_passes_through_404() -> None:
    assert client_mod.WandbClient._is_retryable(Exception("404 not found")) is False


def test_rate_limiter_resets_with_new_settings() -> None:
    client_mod.reset_rate_limiter()
    set_settings(Settings(rate_limit_per_min=120, rate_limit_burst=200))
    bucket = client_mod._get_bucket()
    assert bucket.capacity == 200
    assert bucket.rate_per_sec == pytest.approx(120.0 / 60.0)
