"""Tests for the concurrent run-fetching helper."""

from __future__ import annotations

import threading
import time
from typing import Any

from mcp_wandb.client import batch_runs


class _SlowClient:
    """Each ``run()`` call sleeps for ``delay`` seconds; counts concurrent invocations."""

    def __init__(self, delay: float = 0.05) -> None:
        self.delay = delay
        self.calls: list[str] = []
        self._active = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()

    def run(self, path: str) -> dict[str, Any]:
        with self._lock:
            self._active += 1
            self.max_concurrent = max(self.max_concurrent, self._active)
            self.calls.append(path)
        time.sleep(self.delay)
        with self._lock:
            self._active -= 1
        return {"path": path}


def test_batch_runs_returns_all_results() -> None:
    client = _SlowClient(delay=0.01)
    paths = [f"e/p/{i}" for i in range(5)]
    results = batch_runs(client, paths, max_workers=5)
    assert len(results) == 5
    assert all(r is not None for r in results)
    # Order preserved.
    for i, r in enumerate(results):
        assert r["path"] == paths[i]


def test_batch_runs_parallelizes() -> None:
    client = _SlowClient(delay=0.1)
    paths = [f"e/p/{i}" for i in range(10)]
    start = time.monotonic()
    batch_runs(client, paths, max_workers=10)
    elapsed = time.monotonic() - start
    # Serial would be 10 * 0.1 = 1.0s; with 10 workers expect ~0.1s + overhead.
    assert elapsed < 0.5, f"batch_runs not parallelizing; elapsed {elapsed:.2f}s"
    assert client.max_concurrent > 1


def test_batch_runs_empty_input() -> None:
    client = _SlowClient()
    assert batch_runs(client, []) == []


def test_batch_runs_partial_failure_returns_none() -> None:
    class _Failing:
        def run(self, path: str) -> Any:
            if path.endswith("/bad"):
                raise RuntimeError("nope")
            return {"path": path}

    paths = ["e/p/good", "e/p/bad", "e/p/other"]
    results = batch_runs(_Failing(), paths, max_workers=3)
    assert results[0] == {"path": "e/p/good"}
    assert results[1] is None
    assert results[2] == {"path": "e/p/other"}


def test_batch_runs_workers_capped_at_input_size() -> None:
    """Asking for 100 workers on 3 inputs shouldn't spin up 100 threads."""
    client = _SlowClient(delay=0.01)
    paths = ["e/p/a", "e/p/b", "e/p/c"]
    batch_runs(client, paths, max_workers=100)
    assert client.max_concurrent <= 3
