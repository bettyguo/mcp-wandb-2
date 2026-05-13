"""Process-global cache for ``client.run(path)`` lookups.

Two layers:

1. In-memory (TTL + LRU) holds live ``wandb.Run`` instances; lives only as
   long as the server process.
2. Disk (TTL only; opt-in via ``Settings.cache_dir``) stores JSON snapshot
   files and survives restarts. On hit we return a ``SnapshotProxy``
   instead of a live run. Tools that need ``run.history()``, ``update()``,
   or ``delete()`` go through ``WandbClient.run_live(path)``.

The cache is process-global because each tool call constructs a fresh
``WandbClient`` (see ``server.py._client``); an instance-attached cache
would die immediately.

Externally-mutated runs (W&B web UI, another script) won't invalidate the
cache automatically. Mutations made through our own ``add_tag`` and
``delete_run`` invalidate both layers for the touched path.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .settings import get_settings


@dataclass
class CacheStats:
    size: int
    hits: int
    misses: int
    evictions: int
    max_entries: int
    ttl_seconds: float


class RunCache:
    """Bounded TTL+LRU cache, thread-safe."""

    def __init__(self, ttl_seconds: float, max_entries: int) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive.")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive.")
        self._ttl: float = float(ttl_seconds)
        self._max: int = int(max_entries)
        # OrderedDict gives us O(1) move-to-end + popitem(last=False) for LRU.
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock: threading.Lock = threading.Lock()
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

    @property
    def ttl_seconds(self) -> float:
        return self._ttl

    @property
    def max_entries(self) -> int:
        return self._max

    def get(self, key: str) -> Any | None:
        """Return the cached value or ``None`` (miss / expired)."""
        now = time.monotonic()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self._misses += 1
                return None
            ts, value = entry
            if now - ts > self._ttl:
                # Expired; drop and report miss.
                del self._data[key]
                self._misses += 1
                self._evictions += 1
                return None
            # LRU touch on hit.
            self._data.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        """Insert or refresh a cache entry; evict LRU if over capacity."""
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = (time.monotonic(), value)
            while len(self._data) > self._max:
                self._data.popitem(last=False)
                self._evictions += 1

    def invalidate(self, key: str) -> None:
        """Drop the entry for ``key`` if present (no-op otherwise)."""
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        """Drop every entry and zero the counters."""
        with self._lock:
            self._data.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                size=len(self._data),
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                max_entries=self._max,
                ttl_seconds=self._ttl,
            )


_global_cache: RunCache | None = None
_global_lock: threading.Lock = threading.Lock()


def get_cache() -> RunCache | None:
    """Return the active cache (or ``None`` when caching is disabled).

    Honors ``Settings.cache_enabled``; re-creates the cache instance if the
    settings ttl / max-entries have changed between calls (rare, but lets
    tests adjust without a server restart).
    """
    settings = get_settings()
    if not settings.cache_enabled:
        return None
    global _global_cache
    with _global_lock:
        if (
            _global_cache is None
            or _global_cache.ttl_seconds != settings.cache_ttl_seconds
            or _global_cache.max_entries != settings.cache_max_entries
        ):
            _global_cache = RunCache(
                ttl_seconds=settings.cache_ttl_seconds,
                max_entries=settings.cache_max_entries,
            )
        return _global_cache


def reset_cache() -> None:
    """Test hook: drop the cache so the next ``get_cache()`` rebuilds."""
    global _global_cache
    with _global_lock:
        _global_cache = None


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------


def materialize(run: Any) -> dict[str, Any]:
    """Extract a JSON-serializable snapshot dict from a wandb.Run-like object.

    Captures every attribute the read-only tools consume (id, name, state,
    tags, config, summary_metrics, etc.). ``run.history()`` /
    ``run.update()`` / ``run.delete()`` are method calls and are *not*
    captured; tools that need them must use ``WandbClient.run_live(path)``.
    """
    sm = getattr(run, "summary_metrics", None) or getattr(run, "summary", {}) or {}
    if not isinstance(sm, dict):
        sm = dict(sm)
    cfg = getattr(run, "config", {}) or {}
    if not isinstance(cfg, dict):
        cfg = dict(cfg)
    sys_metrics = getattr(run, "systemMetrics", {}) or {}
    if not isinstance(sys_metrics, dict):
        sys_metrics = dict(sys_metrics)
    sweep = getattr(run, "sweep", None)
    sweep_id = None
    if sweep is not None:
        sweep_id = getattr(sweep, "id", None) or getattr(sweep, "name", None)
    created_at = getattr(run, "created_at", None)
    if created_at is not None and not isinstance(created_at, str):
        # datetime → ISO 8601
        try:
            created_at = created_at.isoformat()
        except Exception:
            created_at = str(created_at)
    return {
        "_format_version": 1,
        "id": str(getattr(run, "id", "")),
        "name": str(getattr(run, "name", "")),
        "entity": str(getattr(run, "entity", "") or ""),
        "project": str(getattr(run, "project", "") or ""),
        "state": str(getattr(run, "state", "unknown") or "unknown"),
        "tags": list(getattr(run, "tags", []) or []),
        "config": dict(cfg),
        "summary_metrics": {k: v for k, v in sm.items()},
        "systemMetrics": dict(sys_metrics),
        "notes": getattr(run, "notes", None),
        "sweep_id": sweep_id,
        "created_at": created_at,
        "runtime": _safe_float(getattr(run, "runtime", None)),
        "url": getattr(run, "url", None),
    }


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class SnapshotProxy:
    """Read-only stand-in for a wandb.Run, hydrated from a snapshot dict.

    Quacks like the wandb.Run object for read access. ``getattr(proxy, "id")``,
    ``proxy.summary``, ``proxy.config`` all work. Mutating attributes and
    method calls that hit the network (``history``, ``update``, ``delete``)
    raise ``SnapshotMethodUnsupported``; the caller should route those
    through ``WandbClient.run_live(path)``.
    """

    __slots__ = ("_snap",)

    def __init__(self, snap: dict[str, Any]) -> None:
        object.__setattr__(self, "_snap", snap)

    def __getattr__(self, name: str) -> Any:
        # Called only when the attribute is NOT found on the instance/class.
        snap = object.__getattribute__(self, "_snap")
        if name == "summary":  # wandb's read-side alias
            return snap.get("summary_metrics", {})
        if name == "sweep":
            sid = snap.get("sweep_id")
            return _SweepRef(sid) if sid else None
        if name in snap:
            return snap[name]
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_snap":
            object.__setattr__(self, name, value)
            return
        raise SnapshotMethodUnsupported(
            f"SnapshotProxy is read-only (attempted to set {name!r}). "
            "Use WandbClient.run_live(path) for mutations."
        )

    def history(self, *args: Any, **kwargs: Any) -> Any:
        raise SnapshotMethodUnsupported(
            "SnapshotProxy.history() is not supported (snapshots don't carry "
            "history data). Use WandbClient.run_live(path).history(...)."
        )

    def update(self, *args: Any, **kwargs: Any) -> Any:
        raise SnapshotMethodUnsupported(
            "SnapshotProxy.update() is not supported. Use WandbClient.run_live(path)."
        )

    def delete(self, *args: Any, **kwargs: Any) -> Any:
        raise SnapshotMethodUnsupported(
            "SnapshotProxy.delete() is not supported. Use WandbClient.run_live(path)."
        )


class SnapshotMethodUnsupported(RuntimeError):
    """Raised when SnapshotProxy is asked for a method-call attribute."""


@dataclass
class _SweepRef:
    """Tiny stand-in matching the read-side ``run.sweep.id`` shape."""

    id: str | None


class DiskCache:
    """JSON-file-backed snapshot store with TTL eviction.

    Keys are SHA-256-hashed so filesystem-illegal characters in paths
    (slashes, colons) never break us. Snapshots are written atomically
    (write-to-temp, rename) so a concurrent reader never sees a half-flushed
    file.
    """

    def __init__(self, root: Path, ttl_seconds: float) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive.")
        self._root: Path = Path(root)
        self._ttl: float = float(ttl_seconds)
        self._lock: threading.Lock = threading.Lock()
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def ttl_seconds(self) -> float:
        return self._ttl

    def _path_for(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        # Shard into 256 buckets so a single directory doesn't blow up.
        return self._root / digest[:2] / f"{digest}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path_for(key)
        if not path.exists():
            return None
        try:
            stat = path.stat()
            if time.time() - stat.st_mtime > self._ttl:
                # Expired; drop and report miss.
                self._unlink_quiet(path)
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._unlink_quiet(path)
            return None
        if not isinstance(data, dict) or data.get("_format_version") != 1:
            self._unlink_quiet(path)
            return None
        return data

    def set(self, key: str, value: dict[str, Any]) -> None:
        path = self._path_for(key)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic write: tempfile in the same dir, then rename.
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(value, default=str), encoding="utf-8")
            tmp.replace(path)

    def invalidate(self, key: str) -> None:
        self._unlink_quiet(self._path_for(key))

    def clear(self) -> None:
        with self._lock:
            if self._root.exists():
                shutil.rmtree(self._root, ignore_errors=True)
            self._root.mkdir(parents=True, exist_ok=True)

    def stats(self) -> DiskCacheStats:
        """Walk the cache directory and report entry count + on-disk size.

        Best-effort: ignores files we can't stat (e.g. concurrent removal).
        """
        entry_count = 0
        size_bytes = 0
        if self._root.exists():
            for p in self._root.rglob("*.json"):
                try:
                    size_bytes += p.stat().st_size
                    entry_count += 1
                except OSError:
                    continue
        return DiskCacheStats(
            root=str(self._root),
            ttl_seconds=self._ttl,
            entry_count=entry_count,
            size_bytes=size_bytes,
        )

    @staticmethod
    def _unlink_quiet(path: Path) -> None:
        with contextlib.suppress(OSError):
            path.unlink()


@dataclass
class DiskCacheStats:
    root: str
    ttl_seconds: float
    entry_count: int
    size_bytes: int


_global_disk_cache: DiskCache | None = None
_disk_lock: threading.Lock = threading.Lock()


def get_disk_cache() -> DiskCache | None:
    """Return the active disk cache (or ``None`` when not configured)."""
    settings = get_settings()
    cache_dir = settings.cache_dir
    if not cache_dir:
        return None
    global _global_disk_cache
    with _disk_lock:
        if _global_disk_cache is None or str(_global_disk_cache.root) != str(cache_dir) or _global_disk_cache.ttl_seconds != settings.cache_ttl_seconds:
            _global_disk_cache = DiskCache(
                root=Path(cache_dir),
                ttl_seconds=settings.cache_ttl_seconds,
            )
        return _global_disk_cache


def reset_disk_cache() -> None:
    """Test hook: drop the disk cache singleton."""
    global _global_disk_cache
    with _disk_lock:
        _global_disk_cache = None


def cache_health_payload() -> dict[str, Any]:
    """Build a JSON-ready snapshot of both cache layers and the W&B
    back-pressure window. Used by the ``mcp-wandb://cache/stats`` resource.

    The top-level ``status`` is one of:

    * ``busy`` if any single recent W&B call waited more than 5 s for the
      rate-limit bucket (short-burst pressure);
    * ``degraded`` if cumulative back-pressure over the window exceeded
      60 s of wait or more than 10 retries (sustained pressure);
    * ``ok`` otherwise.

    Precedence is ``busy > degraded > ok``. Thresholds live in module
    constants below; consider lifting them into ``Settings`` if you need
    per-deployment tuning.
    """
    from ._metrics import wandb_api_metrics_payload

    mem = get_cache()
    disk = get_disk_cache()

    if mem is None:
        memory_payload: dict[str, Any] = {"enabled": False}
    else:
        s = mem.stats()
        memory_payload = {
            "enabled": True,
            "size": s.size,
            "hits": s.hits,
            "misses": s.misses,
            "evictions": s.evictions,
            "hit_rate": (s.hits / (s.hits + s.misses)) if (s.hits + s.misses) else None,
            "max_entries": s.max_entries,
            "ttl_seconds": s.ttl_seconds,
        }

    if disk is None:
        disk_payload: dict[str, Any] = {"enabled": False}
    else:
        d = disk.stats()
        disk_payload = {
            "enabled": True,
            "root": d.root,
            "ttl_seconds": d.ttl_seconds,
            "entry_count": d.entry_count,
            "size_bytes": d.size_bytes,
        }

    wandb_api_payload = wandb_api_metrics_payload()
    return {
        "status": _status_from_wandb_api(wandb_api_payload),
        "memory": memory_payload,
        "disk": disk_payload,
        "wandb_api": wandb_api_payload,
    }


# Status thresholds. Kept module-level for easy tuning.
_STATUS_BUSY_MAX_RATE_LIMITED_MS = 5_000
_STATUS_DEGRADED_TOTAL_RATE_LIMITED_MS = 60_000
_STATUS_DEGRADED_TOTAL_RETRIES = 10


def _status_from_wandb_api(wandb_api: dict[str, Any]) -> str:
    """Top-line health string for the ``mcp-wandb://cache/stats`` resource.

    Order of precedence: ``busy`` > ``degraded`` > ``ok``. The first
    matching threshold wins so the agent sees the most-urgent label.
    """
    if wandb_api.get("max_rate_limited_ms", 0) > _STATUS_BUSY_MAX_RATE_LIMITED_MS:
        return "busy"
    if (
        wandb_api.get("total_rate_limited_ms", 0) > _STATUS_DEGRADED_TOTAL_RATE_LIMITED_MS
        or wandb_api.get("total_retries", 0) > _STATUS_DEGRADED_TOTAL_RETRIES
    ):
        return "degraded"
    return "ok"
