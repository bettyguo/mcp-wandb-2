"""Unit tests for the importance backends without the W&B layer."""

from __future__ import annotations

import math
import random

from mcp_wandb.importance import compute_importance


def _synthetic_rows(n: int = 100) -> list[dict[str, object]]:
    rng = random.Random(0)
    rows = []
    for _ in range(n):
        lr = 10 ** rng.uniform(-5, -1)
        bs = rng.choice([32, 64, 128, 256])
        signal = -((math.log10(lr) - math.log10(3e-4)) ** 2) * 0.05 + 0.01 * math.log2(bs / 64)
        metric = 0.9 + signal + rng.gauss(0, 0.005)
        rows.append({"config_flat": {"lr": lr, "batch_size": bs}, "metric": metric})
    return rows


def test_compute_importance_rf_signal_strong() -> None:
    rows = _synthetic_rows(100)
    result = compute_importance(rows, target_metric="val_acc", method="rf")
    assert result.method == "rf"
    assert result.n_runs == 100
    assert result.n_features == 2
    assert result.ranking[0].param in {"lr", "batch_size"}
    assert result.model_r2 > 0.3


def test_compute_importance_too_few_rows_returns_empty() -> None:
    rows = _synthetic_rows(3)
    result = compute_importance(rows, target_metric="val_acc", method="rf")
    assert result.ranking == []
    assert "Not enough runs" in result.notes


def test_compute_importance_low_signal_warns_in_notes() -> None:
    rng = random.Random(7)
    rows = [
        {"config_flat": {"lr": rng.random()}, "metric": rng.gauss(0.5, 0.01)}
        for _ in range(30)
    ]
    result = compute_importance(rows, target_metric="val_acc", method="rf")
    assert result.model_r2 < 0.6
    if result.model_r2 < 0.3:
        assert "LOW" in result.notes or "unreliable" in result.notes.lower()


def test_compute_importance_drops_constant_feature() -> None:
    rng = random.Random(1)
    rows = []
    for _ in range(40):
        lr = rng.uniform(0.001, 0.1)
        rows.append({"config_flat": {"lr": lr, "constant": 42}, "metric": 0.5 + lr})
    result = compute_importance(rows, target_metric="val_acc", method="rf")
    params = [e.param for e in result.ranking]
    assert "constant" not in params
    assert "lr" in params
