"""Tests for ergonomic improvements: nan_keys, state_counts, CLI gate."""

from __future__ import annotations

import math
import sys
import types

import pytest
from typer.testing import CliRunner

from mcp_wandb.cli import app
from mcp_wandb.tools.analysis import compare_runs, summarize_sweep


def _stub_fastmcp() -> None:
    """Inject a minimal fake ``fastmcp`` module so ``mcp_wandb.server`` imports.

    Tests in the minimal venv don't install fastmcp; the CLI tests only need
    ``build_app`` to be callable, and monkeypatch replaces it anyway.
    """
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


def test_compare_runs_populates_nan_keys(fake_client) -> None:  # type: ignore[no-untyped-def]
    runs = fake_client.api.runs_data["demo/cifar10-sweep"][:2]
    runs[0].summary_metrics["weird"] = math.nan
    runs[1].summary_metrics["weird"] = 0.5
    ids = [f"demo/cifar10-sweep/{r.id}" for r in runs]
    diff = compare_runs(fake_client, run_ids=ids)
    weird = next(e for e in diff.metric_diff if e.metric == "weird")
    assert runs[0].id in weird.nan_keys
    assert weird.values[runs[0].id] is None
    assert weird.values[runs[1].id] == 0.5


def test_compare_runs_no_nan_keys_when_all_clean(fake_client) -> None:  # type: ignore[no-untyped-def]
    runs = fake_client.api.runs_data["demo/cifar10-sweep"][:2]
    ids = [f"demo/cifar10-sweep/{r.id}" for r in runs]
    diff = compare_runs(fake_client, run_ids=ids)
    val_acc = next(e for e in diff.metric_diff if e.metric == "val_acc")
    assert val_acc.nan_keys == []


def test_summarize_sweep_populates_state_counts(fake_client) -> None:  # type: ignore[no-untyped-def]
    # Mark one run as "preempted" (a non-standard state).
    runs = fake_client.api.runs_data["demo/cifar10-sweep"]
    runs[0].state = "preempted"
    runs[1].state = "preempted"

    summary = summarize_sweep(
        fake_client, sweep_id="demo/cifar10-sweep/sweeps/abc123"
    )
    assert summary.state_counts["preempted"] == 2
    assert summary.state_counts["finished"] == len(runs) - 2
    # Fixed fields stay back-compat-aligned with state_counts.
    assert summary.finished == summary.state_counts["finished"]


def test_cli_tools_list_prints_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tools list command should announce the registered count and ceiling."""
    _stub_fastmcp()
    import mcp_wandb.cli as cli_mod
    import mcp_wandb.server as server_mod

    class _Fake:
        name = "fake_tool"
        description = "A fake tool. With a follow-on sentence."

    class _FakeManager:
        def list_tools(self) -> list[_Fake]:
            return [_Fake() for _ in range(cli_mod.TOOL_CEILING)]

    class _FakeApp:
        _tool_manager = _FakeManager()

    monkeypatch.setattr(server_mod, "build_app", lambda: _FakeApp())

    runner = CliRunner()
    result = runner.invoke(app, ["tools", "list"])
    assert result.exit_code == 0
    assert f"{cli_mod.TOOL_CEILING} tools registered" in result.stdout
    assert f"ceiling: {cli_mod.TOOL_CEILING}" in result.stdout


def test_cli_tools_list_strict_fails_when_count_drifts(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_fastmcp()
    import mcp_wandb.server as server_mod

    class _Fake:
        name = "fake"
        description = "x"

    class _FakeManager:
        def list_tools(self) -> list[_Fake]:
            return [_Fake() for _ in range(14)]

    class _FakeApp:
        _tool_manager = _FakeManager()

    monkeypatch.setattr(server_mod, "build_app", lambda: _FakeApp())
    runner = CliRunner()
    result = runner.invoke(app, ["tools", "list", "--strict"])
    assert result.exit_code != 0
    assert "FAIL" in result.output or "fail" in result.output
