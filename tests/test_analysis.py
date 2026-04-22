"""Tests for the analysis tools."""

from __future__ import annotations

import math

import pytest

from mcp_wandb.tools.analysis import (
    compare_runs,
    find_baseline_runs,
    find_best_run,
    hyperparam_importance,
    summarize_sweep,
)


def test_find_best_run_returns_top_val_acc(fake_client) -> None:  # type: ignore[no-untyped-def]
    best = find_best_run(fake_client, project="demo/cifar10-sweep", metric="val_acc", mode="max")
    assert best is not None
    assert "val_acc" in best.summary_metrics
    # The best run's val_acc should be the highest in the sweep.
    all_vals = [r.summary_metrics["val_acc"] for r in fake_client.api.runs_data["demo/cifar10-sweep"]]
    assert math.isclose(best.summary_metrics["val_acc"], max(all_vals), rel_tol=1e-9)


def test_find_best_run_min_mode(fake_client) -> None:  # type: ignore[no-untyped-def]
    best = find_best_run(fake_client, project="demo/cifar10-sweep", metric="val_loss", mode="min")
    assert best is not None
    all_vals = [r.summary_metrics["val_loss"] for r in fake_client.api.runs_data["demo/cifar10-sweep"]]
    assert math.isclose(best.summary_metrics["val_loss"], min(all_vals), rel_tol=1e-9)


def test_find_baseline_runs_uses_tag_filter(fake_client) -> None:  # type: ignore[no-untyped-def]
    baselines = find_baseline_runs(fake_client, project="demo/cifar10-sweep", tag="baseline")
    assert len(baselines) >= 1
    for b in baselines:
        assert "baseline" in b.tags


def test_compare_runs_marks_distinct_keys(fake_client) -> None:  # type: ignore[no-untyped-def]
    runs = fake_client.api.runs_data["demo/cifar10-sweep"][:3]
    ids = [f"demo/cifar10-sweep/{r.id}" for r in runs]
    diff = compare_runs(fake_client, run_ids=ids)
    assert diff.n_runs == 3
    # lr differs across runs (sampled from continuous range).
    lr_entries = [e for e in diff.config_diff if e.key == "lr"]
    assert lr_entries and lr_entries[0].distinct is True
    # All three should have val_acc populated in metric_diff.
    val_acc_entries = [e for e in diff.metric_diff if e.metric == "val_acc"]
    assert val_acc_entries
    assert all(v is not None for v in val_acc_entries[0].values.values())


def test_compare_runs_rejects_singleton(fake_client) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        compare_runs(fake_client, run_ids=["demo/cifar10-sweep/run000"])


def test_hyperparam_importance_ranks_lr_first(fake_client) -> None:  # type: ignore[no-untyped-def]
    runs = fake_client.api.runs_data["demo/cifar10-sweep"]
    ids = [f"demo/cifar10-sweep/{r.id}" for r in runs]
    result = hyperparam_importance(fake_client, run_ids=ids, target_metric="val_acc", method="rf")
    assert result.n_runs == len(runs)
    assert result.method == "rf"
    assert result.target_metric == "val_acc"
    top_param = result.ranking[0].param
    # The fixture is constructed so lr dominates; sanity-check the ranking.
    assert top_param == "lr", f"expected lr top, got {top_param}: {[(e.param, e.importance) for e in result.ranking]}"
    assert result.model_r2 > 0.3
    assert "R²" in result.notes or "r2" in result.notes.lower() or "R^2" in result.notes


def test_hyperparam_importance_handles_too_few_runs(fake_client) -> None:  # type: ignore[no-untyped-def]
    runs = fake_client.api.runs_data["demo/cifar10-sweep"][:3]
    ids = [f"demo/cifar10-sweep/{r.id}" for r in runs]
    with pytest.raises(ValueError):
        hyperparam_importance(fake_client, run_ids=ids, target_metric="val_acc")


def test_summarize_sweep_reports_counts(fake_client) -> None:  # type: ignore[no-untyped-def]
    summary = summarize_sweep(fake_client, sweep_id="demo/cifar10-sweep/sweeps/abc123")
    assert summary.n_runs == 30
    assert summary.finished == 30
    assert summary.best is not None
    assert summary.worst is not None
    assert summary.median is not None
    assert summary.target_metric == "val_acc"
    assert summary.importance is not None
    assert summary.importance.ranking[0].param == "lr"
    assert {p.param for p in summary.param_ranges} >= {"lr", "batch_size", "optimizer"}
