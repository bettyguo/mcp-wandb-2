"""Tests for the small helpers in mcp_wandb._util."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mcp_wandb._util import (
    coerce_metric_value,
    config_hash,
    flatten_config,
    parse_project,
    parse_run_id,
    parse_since,
)


def test_parse_since_relative() -> None:
    now = datetime.now(UTC)
    parsed = parse_since("7d")
    assert parsed is not None
    delta = now - parsed
    assert timedelta(days=6, hours=23) < delta < timedelta(days=7, hours=1)


def test_parse_since_absolute_iso() -> None:
    parsed = parse_since("2026-01-15T10:00:00+00:00")
    assert parsed == datetime(2026, 1, 15, 10, 0, 0, tzinfo=UTC)


def test_parse_since_none() -> None:
    assert parse_since(None) is None


@pytest.mark.parametrize(
    ("inp", "expected"),
    [
        ("24h", timedelta(hours=24)),
        ("30m", timedelta(minutes=30)),
        ("2w", timedelta(weeks=2)),
    ],
)
def test_parse_since_units(inp: str, expected: timedelta) -> None:
    now = datetime.now(UTC)
    parsed = parse_since(inp)
    assert parsed is not None
    delta = now - parsed
    assert abs((delta - expected).total_seconds()) < 5


def test_parse_project_with_entity() -> None:
    assert parse_project("alice/cifar10") == ("alice", "cifar10")


def test_parse_project_without_entity_requires_default() -> None:
    with pytest.raises(ValueError):
        parse_project("cifar10")


def test_parse_project_with_default() -> None:
    assert parse_project("cifar10", default_entity="alice") == ("alice", "cifar10")


def test_parse_run_id_full() -> None:
    assert parse_run_id("alice/cifar10/abc123") == "alice/cifar10/abc123"


def test_parse_run_id_short_with_default() -> None:
    assert parse_run_id("abc123", default_project="alice/cifar10") == "alice/cifar10/abc123"


def test_parse_run_id_short_without_default_raises() -> None:
    with pytest.raises(ValueError):
        parse_run_id("abc123")


def test_config_hash_is_stable() -> None:
    a = {"lr": 0.001, "bs": 32, "nested": {"x": 1}}
    b = {"nested": {"x": 1}, "bs": 32, "lr": 0.001}
    assert config_hash(a) == config_hash(b)


def test_config_hash_distinguishes() -> None:
    a = {"lr": 0.001}
    b = {"lr": 0.01}
    assert config_hash(a) != config_hash(b)


def test_config_hash_unwraps_value_envelope() -> None:
    plain = {"lr": 0.001}
    wrapped = {"lr": {"value": 0.001}}
    assert config_hash(plain) == config_hash(wrapped)


def test_flatten_config_nested() -> None:
    cfg = {"model": {"layers": 3, "dim": 128}, "lr": 0.001}
    flat = flatten_config(cfg)
    assert flat == {"model.layers": 3, "model.dim": 128, "lr": 0.001}


def test_flatten_config_drops_private_keys() -> None:
    cfg = {"_step": 1, "lr": 0.001}
    flat = flatten_config(cfg)
    assert "_step" not in flat
    assert flat["lr"] == 0.001


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.5, 0.5),
        (1, 1.0),
        (True, 1.0),
        (None, None),
        ({"value": 3.14}, 3.14),
        ({"mean": 0.42, "min": 0.0}, 0.42),
        ("0.7", 0.7),
        ("not a number", None),
    ],
)
def test_coerce_metric_value(value: object, expected: float | None) -> None:
    assert coerce_metric_value(value) == expected
