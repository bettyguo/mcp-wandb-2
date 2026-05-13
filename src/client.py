"""Thin wrapper over ``wandb.Api()`` with retry, rate-limiting, and pagination.

Every tool calls into this module. Direct ``import wandb`` outside of here is
discouraged so retry and rate-limit policy stay enforced in one place. We
don't want to hammer the W&B API.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ._cache import get_cache
from ._errors import (
    WandbApiError,  # re-exported for back-compat
    is_retryable,
    map_wandb_exception,
)
from ._logging import api_call
from .auth import Credentials
from .settings import get_settings

logger = logging.getLogger(__name__)


__all__ = ["WandbApiError", "WandbClient", "batch_runs", "reset_rate_limiter"]


class _RunFetcher(Protocol):
    """Minimal protocol both WandbClient and the test FakeWandbClient satisfy."""

    def run(self, path: str) -> Any: ...


def batch_runs(
    client: _RunFetcher,
    paths: list[str],
    max_workers: int = 10,
) -> list[Any | None]:
    """Fetch many runs concurrently, returning a same-length list with None for failures.

    Threadpool-based, not asyncio: ``wandb.Api()`` is sync. The rate-limit
    bucket on ``WandbClient`` serializes the underlying HTTP calls so this is
    mostly a latency hider; the real benefit is on chains like
    ``compare_runs([20 ids])`` where the round-trips dominate wall-clock.

    Failed fetches are logged once and reported as ``None`` in the output;
    callers decide whether to drop or surface them.
    """
    if not paths:
        return []
    n = len(paths)
    results: list[Any | None] = [None] * n
    workers = max(1, min(max_workers, n))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(client.run, p): i for i, p in enumerate(paths)}
        for fut in concurrent.futures.as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:
                logger.warning("batch_runs: failed to fetch %s: %s", paths[i], exc)
    return results


@dataclass
class _Bucket:
    """A trivial token bucket. Refills continuously to ``rate_per_sec``."""

    capacity: float
    rate_per_sec: float
    tokens: float
    last_refill: float
    lock: threading.Lock

    def take(self, n: float = 1.0) -> None:
        # Compute the sleep inside the lock so contending callers serialize:
        # each one reserves its n tokens (going into debt) and sleeps for
        # exactly the time needed for the bucket to refill back to 0. Without
        # this, two callers could both observe tokens >= n after a sleep and
        # double-spend.
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.last_refill = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_sec)
            if self.tokens >= n:
                self.tokens -= n
                return
            deficit = n - self.tokens
            self.tokens -= n  # reserve; balance goes negative
            sleep_for = deficit / self.rate_per_sec
        time.sleep(sleep_for)


def _make_bucket() -> _Bucket:
    s = get_settings()
    return _Bucket(
        capacity=float(s.rate_limit_burst),
        rate_per_sec=s.rate_limit_per_min / 60.0,
        tokens=float(s.rate_limit_burst),
        last_refill=time.monotonic(),
        lock=threading.Lock(),
    )


_bucket: _Bucket | None = None
_bucket_lock = threading.Lock()


def _get_bucket() -> _Bucket:
    global _bucket
    with _bucket_lock:
        if _bucket is None:
            _bucket = _make_bucket()
        return _bucket


def reset_rate_limiter() -> None:
    """Test hook: drop and recreate the bucket from current settings."""
    global _bucket
    with _bucket_lock:
        _bucket = None


class WandbClient:
    """Wrapper around ``wandb.Api()`` with policy enforcement.

    Construct one per request; ``wandb.Api`` is cheap to instantiate and
    holding a long-lived reference complicates credential isolation.
    """

    def __init__(self, creds: Credentials) -> None:
        import wandb  # imported lazily so test suites without wandb-installed work

        api_kwargs: dict[str, Any] = {"api_key": creds.api_key}
        if creds.base_url:
            api_kwargs["overrides"] = {"base_url": creds.base_url}
        self._api = wandb.Api(**api_kwargs)
        self._wandb = wandb

    @property
    def api(self) -> Any:
        return self._api

    @property
    def wandb_module(self) -> Any:
        return self._wandb

    def projects(self, entity: str | None = None) -> Iterator[Any]:
        result: Iterator[Any] = self._with_policy(
            "projects", None, lambda: self._api.projects(entity=entity)
        )
        return result

    def project(self, name: str, entity: str | None = None) -> Any:
        return self._with_policy(
            "project",
            f"{entity}/{name}" if entity else name,
            lambda: self._api.project(name=name, entity=entity),
        )

    def runs(
        self,
        path: str,
        filters: dict[str, Any] | None = None,
        order: str = "-created_at",
        per_page: int | None = None,
    ) -> Any:
        per_page = per_page or get_settings().default_per_page
        return self._with_policy(
            "runs",
            path,
            lambda: self._api.runs(
                path=path,
                filters=filters or {},
                order=order,
                per_page=per_page,
            ),
        )

    def run(self, path: str) -> Any:
        """Return a run, hitting in-memory then disk caches before live fetch.

        On disk-cache hit, returns a ``SnapshotProxy`` (read-only). Tools that
        need ``run.history()`` / ``run.update()`` / ``run.delete()`` must
        call ``run_live(path)`` instead.
        """
        from ._cache import SnapshotProxy, get_disk_cache, materialize

        mem = get_cache()
        if mem is not None:
            cached = mem.get(path)
            if cached is not None:
                return cached
        disk = get_disk_cache()
        if disk is not None:
            snap = disk.get(path)
            if snap is not None:
                proxy = SnapshotProxy(snap)
                if mem is not None:
                    mem.set(path, proxy)
                return proxy
        result = self._with_policy("run", path, lambda: self._api.run(path))
        if mem is not None:
            mem.set(path, result)
        if disk is not None:
            # Disk persistence is best-effort; never break the call path.
            with contextlib.suppress(Exception):
                disk.set(path, materialize(result))
        return result

    def run_live(self, path: str) -> Any:
        """Bypass the disk snapshot cache; always return a live ``wandb.Run``.

        Use from tools that need methods (``history``, ``update``, ``delete``)
        or that mutate the run.
        """
        from ._cache import SnapshotProxy, get_disk_cache, materialize

        mem = get_cache()
        if mem is not None:
            cached = mem.get(path)
            # If the in-memory entry is a SnapshotProxy (came from disk),
            # discard it; caller asked for live.
            if cached is not None and not isinstance(cached, SnapshotProxy):
                return cached
        result = self._with_policy("run", path, lambda: self._api.run(path))
        if mem is not None:
            mem.set(path, result)
        disk = get_disk_cache()
        if disk is not None:
            with contextlib.suppress(Exception):
                disk.set(path, materialize(result))
        return result

    def sweep(self, path: str) -> Any:
        return self._with_policy("sweep", path, lambda: self._api.sweep(path))

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        return is_retryable(exc)

    def _with_policy(self, method: str, path: str | None, fn: Any) -> Any:
        with api_call(method=method, path=path) as counters:
            _get_bucket().take()

            attempts = {"n": 0}

            @retry(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1.0, max=10.0),
                retry=retry_if_exception_type(Exception),
                reraise=True,
            )
            def _attempt() -> Any:
                attempts["n"] += 1
                try:
                    return fn()
                except Exception as exc:
                    if not is_retryable(exc):
                        raise map_wandb_exception(exc) from exc
                    raise

            try:
                return _attempt()
            except WandbApiError:
                counters["retries"] = max(0, attempts["n"] - 1)
                raise
            except Exception as exc:
                counters["retries"] = max(0, attempts["n"] - 1)
                raise map_wandb_exception(exc) from exc
            finally:
                counters["retries"] = max(0, attempts["n"] - 1)
