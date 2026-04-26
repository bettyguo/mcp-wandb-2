"""Tests for the figure-construction helpers (no kaleido)."""

from __future__ import annotations

from mcp_wandb.plotting import _ema, comparison_figure, metrics_figure, subsample


def _series(name: str, n: int = 10, base: float = 0.0) -> dict[str, object]:
    return {
        "id": name,
        "name": name,
        "xs": list(range(n)),
        "ys": [base + i * 0.1 for i in range(n)],
        "is_baseline": False,
    }


def test_metrics_figure_traces_match_series_count() -> None:
    series = [_series("a"), _series("b"), _series("c")]
    fig = metrics_figure(series, metric="val_acc", smoothing=0.0)
    assert len(fig.data) == 3
    assert fig.layout.title.text == "val_acc"


def test_metrics_figure_smoothing_changes_values() -> None:
    series = [_series("a", n=5)]
    raw_y = list(series[0]["ys"])  # type: ignore[arg-type]
    fig = metrics_figure(series, metric="loss", smoothing=0.5)
    smoothed = list(fig.data[0].y)
    assert smoothed != raw_y
    # EMA with alpha=0.5 should still start at the same point.
    assert smoothed[0] == raw_y[0]


def test_comparison_figure_drops_baseline_trace() -> None:
    series = [_series("a"), _series("b"), _series("baseline")]
    fig = comparison_figure(series, baseline_id="baseline", metric="val_acc", smoothing=0.0)
    names = {t.name for t in fig.data}
    assert "baseline" not in names
    assert {"a", "b"} <= names


def test_ema_handles_empty() -> None:
    assert _ema([], 0.5) == []


def test_subsample_within_limit_unchanged() -> None:
    xs = list(range(10))
    ys = list(range(10))
    out_xs, out_ys = subsample(xs, ys, max_points=20)
    assert out_xs == xs
    assert out_ys == ys


def test_subsample_above_limit_thins() -> None:
    xs = list(range(100))
    ys = list(range(100))
    out_xs, _out_ys = subsample(xs, ys, max_points=10)
    assert len(out_xs) <= 12
    assert out_xs[0] == 0
