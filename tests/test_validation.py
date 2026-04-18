"""Argument-validation tests for every tool."""

from __future__ import annotations

import pytest

from mcp_wandb.settings import Settings, set_settings
from mcp_wandb.tools import analysis, charts


def test_compare_runs_rejects_zero(fake_client) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        analysis.compare_runs(fake_client, run_ids=[])


def test_compare_runs_rejects_too_many(fake_client) -> None:  # type: ignore[no-untyped-def]
    ids = [f"demo/cifar10-sweep/run{i:03d}" for i in range(51)]
    with pytest.raises(ValueError):
        analysis.compare_runs(fake_client, run_ids=ids)


def test_hyperparam_importance_clamps_top_k(fake_client) -> None:  # type: ignore[no-untyped-def]
    runs = fake_client.api.runs_data["demo/cifar10-sweep"]
    ids = [f"demo/cifar10-sweep/{r.id}" for r in runs]
    result = analysis.hyperparam_importance(
        fake_client, run_ids=ids, target_metric="val_acc", top_k=9999
    )
    assert len(result.ranking) <= 100


def test_hyperparam_importance_rejects_empty_metric(fake_client) -> None:  # type: ignore[no-untyped-def]
    runs = fake_client.api.runs_data["demo/cifar10-sweep"]
    ids = [f"demo/cifar10-sweep/{r.id}" for r in runs]
    with pytest.raises(ValueError):
        analysis.hyperparam_importance(fake_client, run_ids=ids, target_metric="")


def test_plot_metrics_rejects_empty_metric(fake_client, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    runs = fake_client.api.runs_data["demo/cifar10-sweep"][:3]
    ids = [f"demo/cifar10-sweep/{r.id}" for r in runs]
    with pytest.raises(ValueError):
        charts.plot_metrics(fake_client, run_ids=ids, metric="")


def test_launch_sweep_rejects_negative_n_runs(fake_client) -> None:  # type: ignore[no-untyped-def]
    from mcp_wandb.tools.actions import launch_sweep

    set_settings(Settings(enable_actions=True))
    with pytest.raises(ValueError):
        launch_sweep(
            fake_client,
            project="demo/cifar10-sweep",
            sweep_config={"method": "grid", "metric": {"name": "val_acc", "goal": "maximize"}, "parameters": {}},
            n_runs=-5,
            confirm=True,
        )


def test_compare_runs_unicode_equality(fake_client) -> None:  # type: ignore[no-untyped-def]
    """Two runs with the same Unicode config value should NOT be marked distinct."""
    runs = fake_client.api.runs_data["demo/cifar10-sweep"][:2]
    runs[0].config["task"] = "优化"
    runs[1].config["task"] = "优化"
    ids = [f"demo/cifar10-sweep/{r.id}" for r in runs]
    diff = analysis.compare_runs(fake_client, run_ids=ids)
    task_entries = [e for e in diff.config_diff if e.key == "task"]
    assert task_entries, "task key missing from diff"
    assert task_entries[0].distinct is False


def test_summarize_sweep_populates_best_value(fake_client) -> None:  # type: ignore[no-untyped-def]
    """Each SweepParamRange should carry the best run's value for that param."""
    summary = analysis.summarize_sweep(
        fake_client, sweep_id="demo/cifar10-sweep/sweeps/abc123"
    )
    assert summary.best is not None
    best_param_values = {p.param: p.best_value for p in summary.param_ranges}
    # lr / batch_size / optimizer should all have non-None best_value now.
    assert best_param_values["lr"] is not None
    assert best_param_values["batch_size"] is not None
    assert best_param_values["optimizer"] is not None
