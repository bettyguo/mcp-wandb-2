"""End-to-end demo path smoke test.

Runs the full 5-tool sequence against the fake W&B API. The same test runs
against the *real* W&B API in the weekly ``demo-smoke.yml`` workflow (marked
``@pytest.mark.live``).
"""

from __future__ import annotations

import base64

import pytest

from mcp_wandb.tools import charts
from mcp_wandb.tools.analysis import (
    compare_runs,
    find_baseline_runs,
    hyperparam_importance,
)
from mcp_wandb.tools.discovery import list_runs


def test_demo_path_end_to_end(fake_client, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    # 1) Discover recent runs in the sweep.
    recent = list_runs(fake_client, project="demo/cifar10-sweep", limit=20, since="7d")
    assert len(recent.runs) == 20

    # 2) Identify the baseline.
    baselines = find_baseline_runs(fake_client, project="demo/cifar10-sweep", tag="baseline")
    assert len(baselines) >= 1
    baseline_id = f"demo/cifar10-sweep/{baselines[0].id}"

    # 3) Structured diff of the 20 runs against the baseline.
    diff = compare_runs(
        fake_client,
        run_ids=[f"demo/cifar10-sweep/{r.id}" for r in recent.runs] + [baseline_id],
    )
    assert diff.n_runs == 21
    distinct_keys = {e.key for e in diff.config_diff if e.distinct}
    assert "lr" in distinct_keys

    # 4) Headline: hyperparam importance.
    imp = hyperparam_importance(
        fake_client,
        run_ids=[f"demo/cifar10-sweep/{r.id}" for r in recent.runs],
        target_metric="val_acc",
    )
    assert imp.ranking[0].param == "lr"
    assert imp.model_r2 > 0.3

    # 5) Visualize best 5 + baseline. Stub kaleido so the test doesn't need it.
    monkeypatch.setattr(
        charts,
        "render_png_b64",
        lambda fig: (base64.b64encode(b"PNG").decode("ascii"), 3),
    )

    best5_ids = [f"demo/cifar10-sweep/{r.id}" for r in recent.runs[:5]]
    chart = charts.plot_metrics(
        fake_client,
        run_ids=[*best5_ids, baseline_id],
        metric="val_acc",
        smoothing=0.3,
    )
    assert chart.runs_plotted == 6
    assert chart.png_b64
