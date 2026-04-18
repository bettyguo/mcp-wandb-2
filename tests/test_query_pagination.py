"""Cursor-pagination integration tests against query_runs."""

from __future__ import annotations

import pytest

from mcp_wandb import _cursor
from mcp_wandb.tools.query import query_runs


def test_pagination_traverses_all_runs(fake_client) -> None:  # type: ignore[no-untyped-def]
    """Cursoring through a 30-run sweep with limit=10 yields 30 unique ids."""
    seen: list[str] = []
    cursor: str | None = None
    q = {"state": "finished"}
    while True:
        page = query_runs(fake_client, project="demo/cifar10-sweep", mongo_query=q, limit=10, cursor=cursor)
        seen.extend(r.id for r in page.runs)
        if not page.next_cursor:
            break
        cursor = page.next_cursor
        if len(seen) > 100:
            pytest.fail("cursor never terminated")
    assert len(seen) == 30
    assert len(set(seen)) == 30


def test_pagination_rejects_cursor_with_different_query(fake_client) -> None:  # type: ignore[no-untyped-def]
    q1 = {"state": "finished"}
    page = query_runs(fake_client, project="demo/cifar10-sweep", mongo_query=q1, limit=5)
    assert page.next_cursor

    with pytest.raises(_cursor.CursorError):
        query_runs(
            fake_client,
            project="demo/cifar10-sweep",
            mongo_query={"state": "running"},
            limit=5,
            cursor=page.next_cursor,
        )


def test_pagination_returns_no_cursor_on_last_page(fake_client) -> None:  # type: ignore[no-untyped-def]
    # 30 runs total, limit=50 => one page, no cursor.
    page = query_runs(
        fake_client,
        project="demo/cifar10-sweep",
        mongo_query={"state": "finished"},
        limit=50,
    )
    assert len(page.runs) == 30
    assert page.next_cursor is None
