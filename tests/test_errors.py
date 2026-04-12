"""Tests for the error catalog."""

from __future__ import annotations

import pytest

from mcp_wandb._errors import (
    WandbAuthError,
    WandbNotFoundError,
    WandbPermissionError,
    WandbRateLimitError,
    WandbTransientError,
    is_retryable,
    map_wandb_exception,
)


@pytest.mark.parametrize(
    ("msg", "expected_cls", "expected_code"),
    [
        ("401 Unauthorized", WandbAuthError, "auth.bad_key"),
        ("Invalid api key supplied", WandbAuthError, "auth.bad_key"),
        ("403 forbidden", WandbPermissionError, "permission.denied"),
        ("permission denied for entity", WandbPermissionError, "permission.denied"),
        ("404 Not Found", WandbNotFoundError, "not_found.resource"),
        ("no such project foo/bar", WandbNotFoundError, "not_found.resource"),
        ("429 Too Many Requests", WandbRateLimitError, "quota.rate_limit"),
        ("Rate limit exceeded", WandbRateLimitError, "quota.rate_limit"),
        ("503 service unavailable", WandbTransientError, "transient.try_again"),
        ("Connection refused", WandbTransientError, "transient.try_again"),
    ],
)
def test_map_wandb_exception_classifies(msg: str, expected_cls: type, expected_code: str) -> None:
    out = map_wandb_exception(Exception(msg))
    assert isinstance(out, expected_cls)
    assert out.error_code == expected_code
    assert "original:" in str(out)


def test_map_wandb_exception_preserves_cause() -> None:
    original = ValueError("401 Unauthorized")
    mapped = map_wandb_exception(original)
    assert mapped.__cause__ is original


def test_map_wandb_exception_unknown_falls_through() -> None:
    out = map_wandb_exception(Exception("some unmatched error string"))
    assert out.error_code == "wandb.unknown"


@pytest.mark.parametrize(
    ("msg", "expected"),
    [
        ("429 Too Many Requests", True),
        ("503 service unavailable", True),
        ("timeout reading", True),
        ("404 Not Found", False),
        ("401 Unauthorized", False),
    ],
)
def test_is_retryable(msg: str, expected: bool) -> None:
    assert is_retryable(Exception(msg)) is expected
