"""Tests for the disk-persistence cache."""

from __future__ import annotations

import json
import time
from datetime import UTC
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mcp_wandb._cache import (
    DiskCache,
    SnapshotMethodUnsupported,
    SnapshotProxy,
    get_cache,
    get_disk_cache,
    materialize,
    reset_cache,
    reset_disk_cache,
)
from mcp_wandb.settings import Settings, set_settings
from tests.conftest import FakeRun


@pytest.fixture(autouse=True)
def _reset_cache_state() -> None:
    reset_cache()
    reset_disk_cache()
    yield
    reset_cache()
    reset_disk_cache()


# ---------------------------------------------------------------------------
# materialize()
# ---------------------------------------------------------------------------


def test_materialize_captures_read_side_fields() -> None:
    run = FakeRun(
        id="run001",
        name="trial-001",
        entity="demo",
        project="proj",
        state="finished",
        tags=["baseline", "manual"],
        config={"lr": 0.001, "bs": 32},
        summary_metrics={"val_acc": 0.92},
        notes="hand-curated",
        url="https://wandb.ai/demo/proj/runs/run001",
    )
    snap = materialize(run)
    assert snap["_format_version"] == 1
    assert snap["id"] == "run001"
    assert snap["entity"] == "demo"
    assert snap["project"] == "proj"
    assert snap["tags"] == ["baseline", "manual"]
    assert snap["config"] == {"lr": 0.001, "bs": 32}
    assert snap["summary_metrics"] == {"val_acc": 0.92}
    assert snap["url"].endswith("run001")
    # JSON-serializable end-to-end.
    json.dumps(snap)


def test_materialize_serializes_datetime_as_iso() -> None:
    from datetime import datetime

    run = FakeRun(
        id="run002",
        name="trial-002",
        entity="demo",
        project="proj",
        created_at=datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC),
    )
    snap = materialize(run)
    assert snap["created_at"] == "2026-01-15T12:00:00+00:00"


def test_materialize_handles_sweep_attr() -> None:
    sweep_mock = MagicMock()
    sweep_mock.id = "sweep-abc"
    run = FakeRun(id="r", name="r", entity="e", project="p")
    object.__setattr__(run, "sweep", sweep_mock)
    snap = materialize(run)
    assert snap["sweep_id"] == "sweep-abc"


# ---------------------------------------------------------------------------
# SnapshotProxy
# ---------------------------------------------------------------------------


def test_snapshot_proxy_read_attributes() -> None:
    snap = {
        "_format_version": 1,
        "id": "r1",
        "name": "trial",
        "entity": "demo",
        "project": "proj",
        "state": "finished",
        "tags": ["baseline"],
        "config": {"lr": 0.01},
        "summary_metrics": {"val_acc": 0.9},
        "url": "https://w.ai/demo/proj/runs/r1",
    }
    proxy = SnapshotProxy(snap)
    assert proxy.id == "r1"
    assert proxy.tags == ["baseline"]
    assert proxy.config == {"lr": 0.01}
    # ``summary`` alias for summary_metrics (matches wandb.Run shape).
    assert proxy.summary == {"val_acc": 0.9}


def test_snapshot_proxy_history_raises_unsupported() -> None:
    proxy = SnapshotProxy({"_format_version": 1, "id": "r"})
    with pytest.raises(SnapshotMethodUnsupported):
        proxy.history(keys=["loss"])
    with pytest.raises(SnapshotMethodUnsupported):
        proxy.update()
    with pytest.raises(SnapshotMethodUnsupported):
        proxy.delete()


def test_snapshot_proxy_setattr_blocked() -> None:
    proxy = SnapshotProxy({"_format_version": 1, "id": "r", "tags": []})
    with pytest.raises(SnapshotMethodUnsupported):
        proxy.tags = ["new"]


def test_snapshot_proxy_attribute_missing_raises() -> None:
    proxy = SnapshotProxy({"_format_version": 1, "id": "r"})
    with pytest.raises(AttributeError):
        _ = proxy.nonexistent_attr


# ---------------------------------------------------------------------------
# DiskCache
# ---------------------------------------------------------------------------


def test_disk_cache_round_trip(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path, ttl_seconds=60.0)
    cache.set("alice/proj/run1", {"_format_version": 1, "id": "run1"})
    out = cache.get("alice/proj/run1")
    assert out is not None
    assert out["id"] == "run1"


def test_disk_cache_miss_returns_none(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path, ttl_seconds=60.0)
    assert cache.get("nope") is None


def test_disk_cache_ttl_expires(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path, ttl_seconds=0.05)
    cache.set("k", {"_format_version": 1, "id": "x"})
    assert cache.get("k") is not None
    time.sleep(0.1)
    assert cache.get("k") is None  # expired, file should also be removed
    assert not list(tmp_path.rglob("*.json"))


def test_disk_cache_invalidate(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path, ttl_seconds=60.0)
    cache.set("k", {"_format_version": 1, "id": "x"})
    cache.invalidate("k")
    assert cache.get("k") is None
    cache.invalidate("nonexistent")  # idempotent


def test_disk_cache_clear(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path, ttl_seconds=60.0)
    for i in range(5):
        cache.set(f"k{i}", {"_format_version": 1, "id": f"id-{i}"})
    cache.clear()
    for i in range(5):
        assert cache.get(f"k{i}") is None


def test_disk_cache_rejects_corrupt_payload(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path, ttl_seconds=60.0)
    # Write a non-JSON file by hand at the expected location.
    import hashlib

    digest = hashlib.sha256(b"bad-key").hexdigest()
    path = tmp_path / digest[:2] / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not-json!!!")
    assert cache.get("bad-key") is None  # corrupt → miss, file dropped
    assert not path.exists()


def test_disk_cache_rejects_wrong_format_version(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path, ttl_seconds=60.0)
    cache.set("k", {"_format_version": 99, "id": "x"})
    assert cache.get("k") is None  # format v99 not supported


def test_disk_cache_rejects_non_dict_payload(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path, ttl_seconds=60.0)
    import hashlib

    digest = hashlib.sha256(b"k").hexdigest()
    path = tmp_path / digest[:2] / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[]")  # JSON but not a dict
    assert cache.get("k") is None


# ---------------------------------------------------------------------------
# Settings integration: get_disk_cache()
# ---------------------------------------------------------------------------


def test_get_disk_cache_none_when_unconfigured() -> None:
    set_settings(Settings(cache_dir=None))
    assert get_disk_cache() is None


def test_get_disk_cache_active_when_dir_set(tmp_path: Path) -> None:
    set_settings(Settings(cache_dir=str(tmp_path)))
    cache = get_disk_cache()
    assert cache is not None
    assert cache.root == tmp_path


def test_get_disk_cache_singleton(tmp_path: Path) -> None:
    set_settings(Settings(cache_dir=str(tmp_path)))
    assert get_disk_cache() is get_disk_cache()


def test_get_disk_cache_rebuilds_on_settings_change(tmp_path: Path) -> None:
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    set_settings(Settings(cache_dir=str(a_dir)))
    first = get_disk_cache()
    set_settings(Settings(cache_dir=str(b_dir)))
    second = get_disk_cache()
    assert first is not second
    assert second is not None
    assert second.root == b_dir


# ---------------------------------------------------------------------------
# WandbClient integration
# ---------------------------------------------------------------------------


def test_client_run_returns_snapshot_proxy_on_disk_hit(tmp_path: Path) -> None:
    """First call fetches live + materializes; second call (after dropping
    the in-memory cache) returns a SnapshotProxy."""
    from mcp_wandb.client import WandbClient

    set_settings(Settings(cache_enabled=True, cache_dir=str(tmp_path)))

    client = WandbClient.__new__(WandbClient)
    client._api = MagicMock()
    client._wandb = MagicMock()
    live_run = FakeRun(id="r1", name="trial", entity="demo", project="proj", tags=["x"])
    client._api.run = MagicMock(return_value=live_run)
    client._with_policy = lambda method, path, fn: fn()  # type: ignore[method-assign]

    # First call: live + writes to memory + disk.
    a = client.run("demo/proj/r1")
    assert a is live_run
    assert client._api.run.call_count == 1
    assert get_disk_cache().get("demo/proj/r1") is not None  # type: ignore[union-attr]

    # Drop in-memory cache only; disk persists.
    reset_cache()
    b = client.run("demo/proj/r1")
    assert isinstance(b, SnapshotProxy)
    assert b.id == "r1"
    assert b.tags == ["x"]
    # Underlying api wasn't called again.
    assert client._api.run.call_count == 1


def test_run_live_bypasses_disk_snapshot(tmp_path: Path) -> None:
    from mcp_wandb.client import WandbClient

    set_settings(Settings(cache_enabled=True, cache_dir=str(tmp_path)))
    client = WandbClient.__new__(WandbClient)
    client._api = MagicMock()
    client._wandb = MagicMock()
    live_run = FakeRun(id="r1", name="trial", entity="demo", project="proj")
    client._api.run = MagicMock(return_value=live_run)
    client._with_policy = lambda method, path, fn: fn()  # type: ignore[method-assign]

    # Prime disk with a snapshot.
    client.run("demo/proj/r1")
    reset_cache()

    # run_live should bypass disk and fetch live again.
    result = client.run_live("demo/proj/r1")
    assert result is live_run  # not a SnapshotProxy
    assert client._api.run.call_count == 2  # fetched live both times


def test_run_live_drops_snapshot_proxy_from_memory_cache(tmp_path: Path) -> None:
    """If memory has a SnapshotProxy from a prior client.run, run_live must skip it."""
    from mcp_wandb.client import WandbClient

    set_settings(Settings(cache_enabled=True, cache_dir=str(tmp_path)))
    client = WandbClient.__new__(WandbClient)
    client._api = MagicMock()
    client._wandb = MagicMock()
    live_run = FakeRun(id="r1", name="trial", entity="demo", project="proj")
    client._api.run = MagicMock(return_value=live_run)
    client._with_policy = lambda method, path, fn: fn()  # type: ignore[method-assign]

    # Prime memory cache with a SnapshotProxy explicitly.
    mem = get_cache()
    assert mem is not None
    mem.set("demo/proj/r1", SnapshotProxy({"_format_version": 1, "id": "r1"}))

    result = client.run_live("demo/proj/r1")
    assert result is live_run
    assert client._api.run.call_count == 1
