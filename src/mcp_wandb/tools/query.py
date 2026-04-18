"""Power-user query tool: pass-through to W&B's mongo-style filter DSL."""

from __future__ import annotations

from typing import Any

from .. import _cursor
from .._util import parse_project, parse_since
from ..client import WandbClient
from ..models import ListRunsResponse
from ..settings import get_settings
from .discovery import _run_to_summary


def query_runs(
    client: WandbClient,
    project: str,
    mongo_query: dict[str, Any],
    limit: int = 50,
    cursor: str | None = None,
    since: str | None = None,
) -> ListRunsResponse:
    """Power-user filter using W&B's MongoDB-style query DSL.

    Same syntax as the W&B web UI's filter box. Use when an agent needs exact
    filter semantics (existence checks, regex, range operators). For simple
    cases, ``list_runs(filters=…)`` is friendlier.

    Pass ``since`` as a relative-date shortcut (e.g. ``since='7d'``,
    ``'24h'``, ``'2026-05-06'``). It is merged into ``mongo_query`` as a
    ``createdAt >= …`` clause. **Important:** if the caller already provides
    a ``createdAt`` clause in ``mongo_query``, the explicit clause wins -
    we never silently overwrite it. The cursor is bound to the *merged*
    query, so changing ``since`` between calls invalidates the cursor (same
    safety property the cursor offers for any other filter change).

    Returns paginated results. The cursor is opaque, versioned, and bound to
    the exact filter that produced it; passing a cursor with a different
    filter is rejected (this is the desired behavior so an agent rotating
    filters never gets confusingly partial pages). Examples::

        {"config.lr": {"$gte": 0.001, "$lt": 0.1}}
        {"tags": {"$in": ["baseline"]}}
        {"state": "finished", "summary_metrics.val_acc": {"$gt": 0.9}}
    """
    s = get_settings()
    limit = max(1, min(limit, s.max_per_page))

    effective_query: dict[str, Any] = dict(mongo_query)
    since_dt = parse_since(since)
    if since_dt is not None and "createdAt" not in effective_query:
        effective_query["createdAt"] = {"$gte": since_dt.isoformat()}

    offset = _cursor.decode(cursor, effective_query) if cursor else 0
    entity, name = parse_project(project)

    # Fetch offset + limit + 1; the last item, if present, signals more pages.
    fetch_n = min(offset + limit + 1, 500)
    runs_iter = client.runs(
        path=f"{entity}/{name}",
        filters=effective_query,
        per_page=fetch_n,
    )

    collected: list[Any] = []
    for run in runs_iter:
        collected.append(run)
        if len(collected) >= offset + limit + 1:
            break

    page = collected[offset : offset + limit]
    has_more = len(collected) > offset + limit
    summaries = [_run_to_summary(r, entity=entity, project_name=name) for r in page]
    next_cursor = _cursor.encode(offset + limit, effective_query) if has_more else None
    return ListRunsResponse(runs=summaries, next_cursor=next_cursor)
