"""Tests for the run-metadata cache."""

from __future__ import annotations

import threading
import time

import pytest

from mcp_wandb._cache import RunCache, get_cache, reset_cache
from mcp_wandb.settings import Settings, set_settings


@pytest.fixture(autouse=True)
def _reset_cache_state() -> None:
    reset_cache()
    yield
    reset_cache()


# ---------------------------------------------------------------------------
# RunCache unit tests
# ---------------------------------------------------------------------------


def test_get_miss_returns_none() -> None:
    cache = RunCache(ttl_seconds=60, max_entries=10)
    assert cache.get("nope") is None
    assert cache.stats().misses == 1
    assert cache.stats().hits == 0


def test_set_then_get_hit() -> None:
    cache = RunCache(ttl_seconds=60, max_entries=10)
    cache.set("a", object())
    cache.get("a")
    s = cache.stats()
    assert s.hits == 1
    assert s.misses == 0
    assert s.size == 1


def test_ttl_expiration_evicts_on_read() -> None:
    cache = RunCache(ttl_seconds=0.05, max_entries=10)
    cache.set("a", "value")
    assert cache.get("a") == "value"
    time.sleep(0.10)
    assert cache.get("a") is None
    s = cache.stats()
    assert s.evictions >= 1
    assert s.size == 0


def test_lru_evicts_oldest_at_capacity() -> None:
    cache = RunCache(ttl_seconds=60, max_entries=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)  # evicts "a"
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_lru_touch_on_hit() -> None:
    cache = RunCache(ttl_seconds=60, max_entries=2)
    cache.set("a", 1)
    cache.set("b", 2)
    # Hit "a"; it should now be most-recently-used.
    assert cache.get("a") == 1
    cache.set("c", 3)  # should evict "b", not "a"
    assert cache.get("a") == 1
    assert cache.get("b") is None
    assert cache.get("c") == 3


def test_invalidate_removes_entry() -> None:
    cache = RunCache(ttl_seconds=60, max_entries=10)
    cache.set("a", 1)
    cache.invalidate("a")
    assert cache.get("a") is None
    cache.invalidate("a")  # no-op, no raise


def test_clear_drops_everything_and_zeroes_counters() -> None:
    cache = RunCache(ttl_seconds=60, max_entries=10)
    cache.set("a", 1)
    cache.get("a")
    cache.get("b")  # miss
    cache.clear()
    s = cache.stats()
    assert s.size == 0
    assert s.hits == 0
    assert s.misses == 0


def test_thread_safety_no_double_count_under_contention() -> None:
    cache = RunCache(ttl_seconds=60, max_entries=10000)
    n_threads = 8
    per_thread = 200

    def _hammer(thread_id: int) -> None:
        for i in range(per_thread):
            key = f"t{thread_id}-{i}"
            cache.set(key, i)
            cache.get(key)
            cache.get("not-present")

    threads = [threading.Thread(target=_hammer, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    s = cache.stats()
    # Each thread does per_thread hits + per_thread misses.
    assert s.hits == n_threads * per_thread
    assert s.misses == n_threads * per_thread


def test_init_validates_args() -> None:
    with pytest.raises(ValueError):
        RunCache(ttl_seconds=0, max_entries=10)
    with pytest.raises(ValueError):
        RunCache(ttl_seconds=60, max_entries=0)


# ---------------------------------------------------------------------------
# Settings integration via get_cache()
# ---------------------------------------------------------------------------


def test_get_cache_returns_none_when_disabled() -> None:
    set_settings(Settings(cache_enabled=False))
    assert get_cache() is None


def test_get_cache_returns_cache_when_enabled() -> None:
    set_settings(Settings(cache_enabled=True, cache_ttl_seconds=42.0, cache_max_entries=7))
    c = get_cache()
    assert c is not None
    assert c.ttl_seconds == 42.0
    assert c.max_entries == 7


def test_get_cache_returns_same_instance_across_calls() -> None:
    set_settings(Settings(cache_enabled=True))
    assert get_cache() is get_cache()


def test_get_cache_rebuilds_when_settings_change() -> None:
    set_settings(Settings(cache_enabled=True, cache_ttl_seconds=10.0, cache_max_entries=100))
    a = get_cache()
    assert a is not None
    set_settings(Settings(cache_enabled=True, cache_ttl_seconds=20.0, cache_max_entries=100))
    b = get_cache()
    assert b is not None
    assert a is not b
    assert b.ttl_seconds == 20.0


# ---------------------------------------------------------------------------
# Integration with WandbClient (via FakeWandbClient).
# We can't drive the real WandbClient (no wandb in test deps), but the
# behavior we care about lives in WandbClient.run(), which is one short
# branch we exercise by importing it and patching its _with_policy method.
# ---------------------------------------------------------------------------


def test_client_run_consults_cache_before_fetching() -> None:
    """When the cache is enabled, the second client.run(path) is a hit."""
    from unittest.mock import MagicMock

    from mcp_wandb.client import WandbClient

    set_settings(Settings(cache_enabled=True, cache_ttl_seconds=60.0, cache_max_entries=10))
    reset_cache()

    # Construct a WandbClient without the real wandb.
    client = WandbClient.__new__(WandbClient)
    client._api = MagicMock()
    client._wandb = MagicMock()
    sentinel = object()
    client._api.run = MagicMock(return_value=sentinel)
    # Skip rate-limit + retry machinery.
    client._with_policy = lambda method, path, fn: fn()  # type: ignore[method-assign]

    a = client.run("entity/project/run-1")
    b = client.run("entity/project/run-1")
    c = client.run("entity/project/run-1")

    assert a is sentinel
    assert b is sentinel
    assert c is sentinel
    # Only the first call should have hit the underlying API.
    assert client._api.run.call_count == 1


def test_client_run_skips_cache_when_disabled() -> None:
    from unittest.mock import MagicMock

    from mcp_wandb.client import WandbClient

    set_settings(Settings(cache_enabled=False))
    reset_cache()

    client = WandbClient.__new__(WandbClient)
    client._api = MagicMock()
    client._wandb = MagicMock()
    client._api.run = MagicMock(return_value=object())
    client._with_policy = lambda method, path, fn: fn()  # type: ignore[method-assign]

    client.run("entity/project/run-1")
    client.run("entity/project/run-1")
    # Every call should hit the underlying API.
    assert client._api.run.call_count == 2
