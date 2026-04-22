"""Tests for the recommend_next_sweep tool."""

from __future__ import annotations

import pytest

from mcp_wandb.tools.analysis import recommend_next_sweep


def test_returns_a_valid_recommendation(fake_client) -> None:  # type: ignore[no-untyped-def]
    rec = recommend_next_sweep(
        fake_client,
        sweep_id="demo/cifar10-sweep/sweeps/abc123",
    )
    assert rec.based_on_sweep_id == "demo/cifar10-sweep/sweeps/abc123"
    assert rec.n_trials_recommended == 30
    assert rec.narrow_factor == 0.3
    assert rec.recommended_config["method"] == "bayes"
    assert "parameters" in rec.recommended_config
    assert rec.confidence in {"high", "moderate", "low"}


def test_narrows_log_uniform_lr_around_best(fake_client) -> None:  # type: ignore[no-untyped-def]
    rec = recommend_next_sweep(
        fake_client,
        sweep_id="demo/cifar10-sweep/sweeps/abc123",
    )
    lr_spec = rec.recommended_config["parameters"].get("lr", {})
    # The original was log_uniform_values [1e-5, 1e-1]; the narrowed range
    # must be strictly smaller.
    assert lr_spec.get("distribution") == "log_uniform_values"
    assert lr_spec["min"] > 1e-5
    assert lr_spec["max"] < 1e-1


def test_diff_records_decision_per_param(fake_client) -> None:  # type: ignore[no-untyped-def]
    rec = recommend_next_sweep(
        fake_client,
        sweep_id="demo/cifar10-sweep/sweeps/abc123",
    )
    # The synthetic sweep has lr, batch_size, optimizer; each should appear in the diff.
    assert {"lr", "batch_size", "optimizer"} <= set(rec.diff_from_original.keys())
    for v in rec.diff_from_original.values():
        assert any(verb in v for verb in ("narrowed", "kept", "dropped"))


def test_rationale_mentions_top_driver(fake_client) -> None:  # type: ignore[no-untyped-def]
    rec = recommend_next_sweep(
        fake_client,
        sweep_id="demo/cifar10-sweep/sweeps/abc123",
    )
    # The synthetic sweep is constructed so lr dominates; the rationale should
    # call it out.
    assert "lr" in rec.rationale or "Top driver" in rec.rationale
    assert "Confidence" in rec.rationale


def test_confidence_reflects_r2(fake_client) -> None:  # type: ignore[no-untyped-def]
    rec = recommend_next_sweep(
        fake_client,
        sweep_id="demo/cifar10-sweep/sweeps/abc123",
    )
    if rec.importance_r2 >= 0.6:
        assert rec.confidence == "high"
    elif rec.importance_r2 >= 0.3:
        assert rec.confidence == "moderate"
    else:
        assert rec.confidence == "low"


def test_raises_when_no_importance_available(fake_client) -> None:  # type: ignore[no-untyped-def]
    # Wipe every run's summary so importance can't be computed (fewer than 4
    # runs have the target metric).
    runs = fake_client.api.runs_data["demo/cifar10-sweep"]
    for r in runs[:-2]:  # leave 2 with val_acc, under the 4 threshold
        r.summary_metrics.pop("val_acc", None)
    with pytest.raises(ValueError, match="no importance ranking"):
        recommend_next_sweep(
            fake_client,
            sweep_id="demo/cifar10-sweep/sweeps/abc123",
        )


def test_drop_threshold_zero_keeps_all_params(fake_client) -> None:  # type: ignore[no-untyped-def]
    """With drop_threshold=0, no param should be dropped (all narrowed or kept)."""
    rec = recommend_next_sweep(
        fake_client,
        sweep_id="demo/cifar10-sweep/sweeps/abc123",
        drop_threshold=0.0,
    )
    for v in rec.diff_from_original.values():
        assert "dropped" not in v
