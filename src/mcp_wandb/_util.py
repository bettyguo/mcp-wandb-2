"""Small shared helpers: date parsing, run-path parsing, dict diffing."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from dateutil import parser as dateparser

_RELATIVE_RE = re.compile(r"^\s*(\d+)\s*([smhdw])\s*$", re.IGNORECASE)
_UNIT_TO_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def parse_since(value: str | None) -> datetime | None:
    """Parse a relative or absolute ``since`` value into a tz-aware datetime.

    Accepts ``"7d"``, ``"24h"``, ``"30m"``, ISO-8601 strings, or any string the
    ``dateutil`` parser understands. Returns None for None input.
    """
    if value is None:
        return None
    m = _RELATIVE_RE.match(value)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()
        seconds = n * _UNIT_TO_SECONDS[unit]
        return datetime.now(UTC) - timedelta(seconds=seconds)
    parsed = dateparser.parse(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def parse_project(project: str, default_entity: str | None = None) -> tuple[str, str]:
    """Split ``"entity/project"`` or ``"project"`` into (entity, project)."""
    if "/" in project:
        entity, name = project.split("/", 1)
        return entity, name
    if default_entity is None:
        raise ValueError(
            f"project='{project}' is missing an entity prefix. Pass "
            "'entity/project' or set default_entity first."
        )
    return default_entity, project


def parse_run_id(run_id: str, default_project: str | None = None) -> str:
    """Normalize a run id to the ``entity/project/run_id`` form W&B's Api expects."""
    parts = run_id.split("/")
    if len(parts) == 3:
        return run_id
    if len(parts) == 1 and default_project:
        return f"{default_project}/{run_id}"
    raise ValueError(
        f"run_id='{run_id}' must be 'entity/project/run_id' (or 'run_id' with "
        "default_project)."
    )


def parse_sweep_id(sweep_id: str) -> str:
    parts = sweep_id.split("/")
    if len(parts) != 3 and "sweeps" not in sweep_id:
        raise ValueError(
            f"sweep_id='{sweep_id}' must be 'entity/project/sweep_id'."
        )
    return sweep_id


def config_hash(config: dict[str, Any]) -> str:
    """Stable short hash of a config dict for quick-equality checks."""
    canonical = json.dumps(_clean_config(config), sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _clean_config(config: dict[str, Any]) -> dict[str, Any]:
    """W&B configs sometimes wrap values in ``{"value": ...}`` envelopes."""
    out: dict[str, Any] = {}
    for k, v in config.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict) and set(v.keys()) == {"value"}:
            out[k] = v["value"]
        else:
            out[k] = v
    return out


def flatten_config(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts to dotted keys so RF can see every leaf as a feature."""
    cleaned = _clean_config(config)
    out: dict[str, Any] = {}
    for k, v in cleaned.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_config(v, prefix=key))
        else:
            out[key] = v
    return out


def coerce_metric_value(value: Any) -> float | None:
    """Best-effort cast of a W&B summary-metric value to float."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for k in ("value", "mean", "last"):
            if k in value:
                return coerce_metric_value(value[k])
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
