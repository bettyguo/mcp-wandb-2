"""Tests for ergonomic shortcuts.

  - list_runs(tag=...) shortcut for the common 'show me runs tagged X' case.
  - query_runs(since=...) sugar for relative-date filtering on the power tool.
  - tools list --json for programmatic consumers.
"""

from __future__ import annotations

import json as _json
import sys
import types
from datetime import UTC, datetime, timedelta

import pytest
from typer.testing import CliRunner

from mcp_wandb.tools.discovery import list_runs
from mcp_wandb.tools.query import query_runs

# ---------------------------------------------------------------------------
# list_runs(tag=...)
# ---------------------------------------------------------------------------


def test_list_runs_tag_shortcut_filters_to_baselines(fake_client) -> None:  # type: ignore[no-untyped-def]
    response = list_runs(
        fake_client,
        project="demo/cifar10-sweep",
        tag="baseline",
        limit=100,
    )
    assert len(response.runs) >= 1
    # Every returned run must carry the requested tag.
    for r in response.runs:
        assert "baseline" in r.tags


def test_list_runs_tag_with_explicit_filter_wins(fake_client) -> None:  # type: ignore[no-untyped-def]
    """If the caller passes a filters['tags'] AND a tag shortcut, the explicit clause wins."""
    response = list_runs(
        fake_client,
        project="demo/cifar10-sweep",
        tag="baseline",  # shortcut would mean tags $in [baseline]
        filters={"tags": {"$in": ["nonexistent-tag-xyz"]}},  # explicit clause
        limit=100,
    )
    # Explicit filter says "tag nonexistent-tag-xyz"; nothing matches.
    assert response.runs == []


def test_list_runs_tag_none_is_no_op(fake_client) -> None:  # type: ignore[no-untyped-def]
    # tag=None behaves identically to omitting it.
    a = list_runs(fake_client, project="demo/cifar10-sweep", limit=100)
    b = list_runs(fake_client, project="demo/cifar10-sweep", tag=None, limit=100)
    assert [r.id for r in a.runs] == [r.id for r in b.runs]


# ---------------------------------------------------------------------------
# query_runs(since=...)
# ---------------------------------------------------------------------------


def test_query_runs_since_sugar_narrows_results(fake_client) -> None:  # type: ignore[no-untyped-def]
    """With a 1h `since`, we should see far fewer runs than without."""
    response_wide = query_runs(
        fake_client,
        project="demo/cifar10-sweep",
        mongo_query={"state": "finished"},
        limit=100,
    )
    response_narrow = query_runs(
        fake_client,
        project="demo/cifar10-sweep",
        mongo_query={"state": "finished"},
        since="1h",
        limit=100,
    )
    # The synthetic fixture spans 3 days; 1h slice should be empty or very small.
    assert len(response_narrow.runs) < len(response_wide.runs)


def test_query_runs_since_does_not_override_explicit_createdAt(fake_client) -> None:  # type: ignore[no-untyped-def]
    """If the caller already gave a createdAt clause, since shouldn't overwrite it."""
    threshold = (datetime.now(UTC) - timedelta(days=10)).isoformat()
    response = query_runs(
        fake_client,
        project="demo/cifar10-sweep",
        mongo_query={"state": "finished", "createdAt": {"$gte": threshold}},
        since="1h",  # would conflict; explicit wins
        limit=100,
    )
    # Explicit clause was 10 days back; the wide range should pull all runs.
    assert len(response.runs) >= 20


def test_query_runs_since_cursor_invalidates_when_since_changes(fake_client) -> None:  # type: ignore[no-untyped-def]
    """Changing `since` between calls should reject the cursor (filter-binding rule)."""
    from mcp_wandb import _cursor

    page1 = query_runs(
        fake_client,
        project="demo/cifar10-sweep",
        mongo_query={"state": "finished"},
        since="30d",
        limit=5,
    )
    assert page1.next_cursor is not None
    with pytest.raises(_cursor.CursorError):
        query_runs(
            fake_client,
            project="demo/cifar10-sweep",
            mongo_query={"state": "finished"},
            since="7d",  # different since => different merged filter
            limit=5,
            cursor=page1.next_cursor,
        )


# ---------------------------------------------------------------------------
# tools list --json
# ---------------------------------------------------------------------------


def _stub_fastmcp() -> None:
    if "fastmcp" in sys.modules:
        return
    mod = types.ModuleType("fastmcp")

    class _StubFastMCP:
        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            pass

        def tool(self, fn=None, **kwargs):  # type: ignore[no-untyped-def]
            return fn if fn is not None else (lambda f: f)

        def run(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            pass

        def http_app(self) -> None:
            raise AttributeError("stub")

    mod.FastMCP = _StubFastMCP  # type: ignore[attr-defined]
    sys.modules["fastmcp"] = mod


def test_tools_list_json_emits_valid_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_fastmcp()
    import mcp_wandb.cli as cli_mod
    import mcp_wandb.server as server_mod
    from mcp_wandb.cli import app

    class _Fake:
        def __init__(self, name: str, description: str) -> None:
            self.name = name
            self.description = description

    class _FakeManager:
        def list_tools(self) -> list[_Fake]:
            return [
                _Fake("tool_a", "Does A. Follow-up sentence."),
                _Fake("tool_b", "Does B. Follow-up."),
            ]

    class _FakeApp:
        _tool_manager = _FakeManager()

    monkeypatch.setattr(server_mod, "build_app", lambda: _FakeApp())

    runner = CliRunner()
    result = runner.invoke(app, ["tools", "list", "--json"])
    assert result.exit_code == 0
    payload = _json.loads(result.stdout)
    assert payload["count"] == 2
    assert payload["ceiling"] == cli_mod.TOOL_CEILING
    assert payload["tools"] == [
        {"name": "tool_a", "description": "Does A. Follow-up sentence."},
        {"name": "tool_b", "description": "Does B. Follow-up."},
    ]


def test_tools_list_json_with_strict_fails_on_count_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_fastmcp()
    import mcp_wandb.server as server_mod
    from mcp_wandb.cli import app

    class _Fake:
        name = "x"
        description = "y"

    class _FakeManager:
        def list_tools(self) -> list[_Fake]:
            return [_Fake() for _ in range(14)]

    class _FakeApp:
        _tool_manager = _FakeManager()

    monkeypatch.setattr(server_mod, "build_app", lambda: _FakeApp())

    runner = CliRunner()
    result = runner.invoke(app, ["tools", "list", "--json", "--strict"])
    # JSON payload still emits first (so the consumer sees what's wrong).
    # Then strict gate exits non-zero.
    assert result.exit_code != 0
    # The JSON payload should be present in stdout (we don't try to parse it
    # mid-stream, since the FAIL line on stderr can confuse naive bracket matching).
    assert '"count": 14' in result.output
    assert '"ceiling":' in result.output
