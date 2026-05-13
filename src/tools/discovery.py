"""Discovery tools: thin wrappers over ``wandb.Api()`` with sugar.

These overlap the official server's surface but add (a) relative-date support
on ``list_runs.since``, (b) a stable Pydantic response shape, and (c)
self-contained operation if the user has only installed ``mcp-wandb``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .._util import config_hash, parse_project, parse_run_id, parse_since
from ..client import WandbClient
from ..models import ListRunsResponse, ProjectSummary, Run, RunSummary
from ..settings import get_settings


def list_projects(
    client: WandbClient,
    entity: str | None = None,
    limit: int = 100,
) -> list[ProjectSummary]:
    """Lists W&B projects under the given entity (user or team).

    Use when the user asks "what projects do I have" or needs to discover a
    project name before any other tool. Returns up to 100 projects sorted by
    most-recently-active. Pass ``entity`` to scope to a teammate or team;
    if omitted, the authenticated user's default entity is used.
    """
    out: list[ProjectSummary] = []
    for proj in client.projects(entity=entity):
        out.append(
            ProjectSummary(
                name=getattr(proj, "name", ""),
                entity=getattr(proj, "entity", "") or (entity or ""),
                created_at=_dt(getattr(proj, "created_at", None)),
                last_active_at=_dt(getattr(proj, "lastActive", None)),
                run_count=getattr(proj, "run_count", None),
                url=getattr(proj, "url", None),
            )
        )
        if len(out) >= limit:
            break
    return out


def list_runs(
    client: WandbClient,
    project: str,
    filters: dict[str, Any] | None = None,
    limit: int = 50,
    since: str | None = None,
    order_by: str = "-created_at",
    tag: str | None = None,
) -> ListRunsResponse:
    """Lists W&B runs in a project.

    Use when the user wants to see a slice of recent experiments. Supports
    relative-date filtering via ``since`` (e.g., ``since='7d'``, ``'24h'``,
    ``'2026-05-06'``). Pass ``tag='baseline'`` as a shortcut for
    ``filters={'tags': {'$in': ['baseline']}}`` (the explicit ``filters['tags']``
    wins if both are provided). Combine with ``filters`` for richer queries
    (e.g., ``filters={'config.lr': {'$lt': 0.01}}``).

    Returns summary fields only; call ``get_run`` for full detail. Cap is
    ``limit=200``; ask twice with cursoring through ``query_runs`` for more.
    """
    s = get_settings()
    limit = max(1, min(limit, s.max_per_page))

    full_filters: dict[str, Any] = dict(filters or {})
    since_dt = parse_since(since)
    if since_dt is not None:
        full_filters.setdefault("createdAt", {})
        if isinstance(full_filters["createdAt"], dict):
            full_filters["createdAt"]["$gte"] = since_dt.isoformat()
    if tag is not None and "tags" not in full_filters:
        full_filters["tags"] = {"$in": [tag]}

    entity, name = parse_project(project)
    path = f"{entity}/{name}"
    runs_iter = client.runs(path=path, filters=full_filters, order=order_by, per_page=min(limit, 200))

    out: list[RunSummary] = []
    for run in runs_iter:
        out.append(_run_to_summary(run, entity=entity, project_name=name))
        if len(out) >= limit:
            break
    return ListRunsResponse(runs=out, next_cursor=None)


def get_run(client: WandbClient, run_id: str) -> Run:
    """Returns the full configuration, summary metrics, system info, and metadata for one run.

    Use after ``list_runs`` when the agent needs detail to answer a follow-up
    question (e.g., "what was the optimizer in run X"). ``run_id`` may be
    either the full ``entity/project/run_id`` form or just the run id if the
    caller has set a default project elsewhere.
    """
    canonical = parse_run_id(run_id)
    run = client.run(canonical)
    return _run_to_detail(run)


def _run_to_summary(run: Any, entity: str, project_name: str) -> RunSummary:
    config = getattr(run, "config", {}) or {}
    summary = getattr(run, "summary_metrics", None) or getattr(run, "summary", {}) or {}
    if not isinstance(summary, dict):
        summary = dict(summary)
    return RunSummary(
        id=str(getattr(run, "id", "")),
        name=str(getattr(run, "name", "")),
        entity=entity,
        project=project_name,
        state=str(getattr(run, "state", "unknown")),
        tags=list(getattr(run, "tags", []) or []),
        summary_metrics=_strip_private(summary),
        config_hash=config_hash(dict(config)),
        sweep_id=_sweep_id_of(run),
        created_at=_dt(getattr(run, "created_at", None)),
        runtime_s=_runtime_seconds(run),
        url=getattr(run, "url", None),
    )


def _run_to_detail(run: Any) -> Run:
    config = dict(getattr(run, "config", {}) or {})
    summary = getattr(run, "summary_metrics", None) or getattr(run, "summary", {}) or {}
    if not isinstance(summary, dict):
        summary = dict(summary)
    entity = getattr(run, "entity", "")
    project = getattr(run, "project", "")
    return Run(
        id=str(getattr(run, "id", "")),
        name=str(getattr(run, "name", "")),
        entity=entity,
        project=project,
        state=str(getattr(run, "state", "unknown")),
        tags=list(getattr(run, "tags", []) or []),
        config=_strip_private(config),
        summary_metrics=_strip_private(summary),
        system_metrics=_strip_private(getattr(run, "systemMetrics", {}) or {}),
        notes=getattr(run, "notes", None),
        sweep_id=_sweep_id_of(run),
        created_at=_dt(getattr(run, "created_at", None)),
        runtime_s=_runtime_seconds(run),
        url=getattr(run, "url", None),
    )


def _strip_private(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if not str(k).startswith("_")}


def _sweep_id_of(run: Any) -> str | None:
    sweep = getattr(run, "sweep", None)
    if sweep is None:
        return None
    return getattr(sweep, "id", None) or getattr(sweep, "name", None)


def _runtime_seconds(run: Any) -> float | None:
    candidates = (
        getattr(run, "runtime", None),
        (getattr(run, "summary", {}) or {}).get("_runtime"),
        (getattr(run, "summary_metrics", {}) or {}).get("_runtime"),
    )
    for c in candidates:
        if c is None:
            continue
        try:
            return float(c)
        except (TypeError, ValueError):
            continue
    return None


def _dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        from dateutil import parser

        return parser.parse(str(value))
    except (ValueError, TypeError):
        return None
