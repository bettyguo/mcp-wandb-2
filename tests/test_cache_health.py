"""Tests for the cache-health MCP resource surface.

We test the underlying ``cache_health_payload`` helper directly; FastMCP's
resource registration is exercised via ``build_app`` smoke in the existing
CLI tests, and the real resources/read flow runs against the live MCP
client in the demo-smoke CI job.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mcp_wandb._cache import (
    DiskCache,
    cache_health_payload,
    get_cache,
    reset_cache,
    reset_disk_cache,
)
from mcp_wandb.settings import Settings, set_settings


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_cache()
    reset_disk_cache()
    yield
    reset_cache()
    reset_disk_cache()


# ---------------------------------------------------------------------------
# cache_health_payload, both layers individually
# ---------------------------------------------------------------------------


def test_payload_when_nothing_enabled() -> None:
    set_settings(Settings(cache_enabled=False, cache_dir=None))
    from mcp_wandb._metrics import reset_metrics

    reset_metrics()
    p = cache_health_payload()
    assert p["memory"] == {"enabled": False}
    assert p["disk"] == {"enabled": False}
    # The wandb_api block; with no traffic, all counters are zero.
    assert p["wandb_api"]["calls_with_pressure"] == 0
    assert p["wandb_api"]["total_rate_limited_ms"] == 0


def test_payload_memory_only(tmp_path: Path) -> None:
    set_settings(Settings(cache_enabled=True, cache_dir=None))
    cache = get_cache()
    assert cache is not None
    cache.set("a", "value-1")
    cache.get("a")  # 1 hit
    cache.get("missing")  # 1 miss

    p = cache_health_payload()
    assert p["disk"] == {"enabled": False}
    mem = p["memory"]
    assert mem["enabled"] is True
    assert mem["size"] == 1
    assert mem["hits"] == 1
    assert mem["misses"] == 1
    assert mem["evictions"] == 0
    assert mem["hit_rate"] == pytest.approx(0.5)
    assert "max_entries" in mem
    assert "ttl_seconds" in mem


def test_payload_disk_only(tmp_path: Path) -> None:
    set_settings(Settings(cache_enabled=False, cache_dir=str(tmp_path)))

    p = cache_health_payload()
    assert p["memory"] == {"enabled": False}
    disk = p["disk"]
    assert disk["enabled"] is True
    assert disk["root"] == str(tmp_path)
    assert disk["entry_count"] == 0
    assert disk["size_bytes"] == 0


def test_payload_both_layers_populated(tmp_path: Path) -> None:
    set_settings(Settings(cache_enabled=True, cache_dir=str(tmp_path)))

    # Touch each layer.
    mem = get_cache()
    assert mem is not None
    mem.set("a", "value")
    mem.get("a")

    from mcp_wandb._cache import get_disk_cache

    disk = get_disk_cache()
    assert disk is not None
    disk.set("a", {"_format_version": 1, "id": "a"})
    disk.set("b", {"_format_version": 1, "id": "b"})

    p = cache_health_payload()
    assert p["memory"]["enabled"] is True
    assert p["memory"]["size"] == 1
    assert p["memory"]["hits"] == 1
    assert p["disk"]["enabled"] is True
    assert p["disk"]["entry_count"] == 2
    assert p["disk"]["size_bytes"] > 0


def test_payload_hit_rate_none_with_no_traffic() -> None:
    set_settings(Settings(cache_enabled=True))
    p = cache_health_payload()
    assert p["memory"]["hit_rate"] is None  # no hits or misses yet


# ---------------------------------------------------------------------------
# DiskCache.stats()
# ---------------------------------------------------------------------------


def test_disk_stats_empty(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path, ttl_seconds=60.0)
    s = cache.stats()
    assert s.entry_count == 0
    assert s.size_bytes == 0
    assert s.root == str(tmp_path)
    assert s.ttl_seconds == 60.0


def test_disk_stats_after_set(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path, ttl_seconds=60.0)
    cache.set("a", {"_format_version": 1, "id": "a", "data": "x" * 100})
    cache.set("b", {"_format_version": 1, "id": "b", "data": "y" * 100})

    s = cache.stats()
    assert s.entry_count == 2
    assert s.size_bytes > 200  # roughly each entry plus json overhead


def test_disk_stats_after_clear(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path, ttl_seconds=60.0)
    cache.set("a", {"_format_version": 1, "id": "a"})
    cache.clear()
    s = cache.stats()
    assert s.entry_count == 0
    assert s.size_bytes == 0


def test_disk_stats_after_invalidate(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path, ttl_seconds=60.0)
    cache.set("a", {"_format_version": 1, "id": "a"})
    cache.set("b", {"_format_version": 1, "id": "b"})
    cache.invalidate("a")
    s = cache.stats()
    assert s.entry_count == 1


# ---------------------------------------------------------------------------
# Resource registration smoke
# ---------------------------------------------------------------------------


def test_register_resources_no_op_on_old_fastmcp() -> None:
    """If the FastMCP instance has no `.resource()` method, registration
    should silently skip (we don't want this to break the server)."""
    # Stub fastmcp so we can import mcp_wandb.server in the minimal venv.
    import sys
    import types

    if "fastmcp" not in sys.modules:
        mod = types.ModuleType("fastmcp")

        class _StubFastMCP:
            def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
                pass

            def tool(self, fn=None, **kwargs):  # type: ignore[no-untyped-def]
                return fn if fn is not None else (lambda f: f)

        mod.FastMCP = _StubFastMCP  # type: ignore[attr-defined]
        sys.modules["fastmcp"] = mod

    from mcp_wandb.server import _register_resources

    class _OldFastMCP:
        # No `.resource` attribute on purpose.
        pass

    # Must not raise.
    _register_resources(_OldFastMCP())  # type: ignore[arg-type]
