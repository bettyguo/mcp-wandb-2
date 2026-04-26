"""Edge cases for chart history fetching."""

from __future__ import annotations

import base64
from typing import Any

import pytest

from mcp_wandb.tools import charts


class _EmptyHistoryRun:
    id = "empty"
    name = "empty-run"
    entity = "demo"
    project = "cifar10-sweep"

    def history(self, keys: list[str] | None = None, samples: int = 500) -> list[dict[str, Any]]:
        return []


class _NoneHistoryRun(_EmptyHistoryRun):
    id = "none-history"

    def history(self, keys: list[str] | None = None, samples: int = 500) -> Any:
        return None


class _FakeEmptyClient:
    def __init__(self, run: Any) -> None:
        self._run = run

    def run(self, path: str) -> Any:
        return self._run

    def run_live(self, path: str) -> Any:
        return self._run


def test_plot_metrics_handles_empty_history(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        charts, "render_png_b64", lambda fig: (base64.b64encode(b"PNG").decode("ascii"), 3)
    )
    client = _FakeEmptyClient(_EmptyHistoryRun())
    resp = charts.plot_metrics(
        client,  # type: ignore[arg-type]
        run_ids=["demo/cifar10-sweep/empty"],
        metric="val_acc",
    )
    assert resp.runs_plotted == 1
    assert resp.points_sampled == 0


def test_plot_metrics_handles_none_history(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        charts, "render_png_b64", lambda fig: (base64.b64encode(b"PNG").decode("ascii"), 3)
    )
    client = _FakeEmptyClient(_NoneHistoryRun())
    resp = charts.plot_metrics(
        client,  # type: ignore[arg-type]
        run_ids=["demo/cifar10-sweep/none-history"],
        metric="val_acc",
    )
    assert resp.runs_plotted == 1
