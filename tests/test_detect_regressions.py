"""Tests for the detect_regressions tool."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import pytest

from mcp_wandb.tools.analysis import (
    _classify,
    _two_tailed_normal_p,
    detect_regressions,
)
from tests.conftest import FakeProject, FakeRun, FakeWandbApi, FakeWandbClient


def _build_api(
    metric: str,
    baseline_values: list[float],
    candidate_values: list[float],
    baseline_tag: str = "baseline",
    entity: str = "demo",
    project: str = "regress-test",
) -> FakeWandbClient:
    """Build a fake API populated with two cohorts: tagged baselines + candidates.

    Candidates are tagged with `created_at` in the recent past so the `since`
    filter on detect_regressions returns them.
    """
    api = FakeWandbApi()
    api.projects_data.append(FakeProject(name=project, entity=entity))
    path = f"{entity}/{project}"
    runs: list[FakeRun] = []
    now = datetime.now(UTC)

    for i, v in enumerate(baseline_values):
        r = FakeRun(
            id=f"base{i:03d}",
            name=f"baseline-{i:03d}",
            entity=entity,
            project=project,
            state="finished",
            tags=[baseline_tag],
            summary_metrics={metric: v},
            created_at=now - timedelta(days=30, minutes=i),
            url=f"https://wandb.ai/{path}/runs/base{i:03d}",
        )
        runs.append(r)
        api.run_lookup[f"{path}/{r.id}"] = r

    for i, v in enumerate(candidate_values):
        r = FakeRun(
            id=f"cand{i:03d}",
            name=f"candidate-{i:03d}",
            entity=entity,
            project=project,
            state="finished",
            tags=[],
            summary_metrics={metric: v},
            created_at=now - timedelta(hours=1 + i),
            url=f"https://wandb.ai/{path}/runs/cand{i:03d}",
        )
        runs.append(r)
        api.run_lookup[f"{path}/{r.id}"] = r

    api.runs_data[path] = runs
    return FakeWandbClient(api)


def test_flags_clear_regression_in_max_mode() -> None:
    rng = random.Random(0)
    baselines = [0.90 + rng.gauss(0, 0.01) for _ in range(20)]
    # All candidates well below baseline mean → all regressions for max-mode.
    candidates = [0.70, 0.72, 0.75, 0.68]
    client = _build_api("val_acc", baselines, candidates)
    report = detect_regressions(
        client,
        project="demo/regress-test",
        metric="val_acc",
        mode="max",
        since="24h",
        significance_level=0.05,
    )
    assert report.n_baseline == 20
    assert report.n_compared == 4
    assert len(report.regressions) == 4
    assert all(f.direction == "regression" for f in report.regressions)
    assert all(f.z_score < 0 for f in report.regressions)
    # Most-significant first
    assert report.regressions[0].p_value <= report.regressions[-1].p_value


def test_flags_clear_improvement_in_max_mode() -> None:
    rng = random.Random(1)
    baselines = [0.70 + rng.gauss(0, 0.01) for _ in range(20)]
    candidates = [0.95, 0.94, 0.96]  # well above baseline → improvements
    client = _build_api("val_acc", baselines, candidates)
    report = detect_regressions(
        client,
        project="demo/regress-test",
        metric="val_acc",
        mode="max",
        since="24h",
    )
    assert len(report.improvements) == 3
    assert all(f.direction == "improvement" for f in report.improvements)


def test_min_mode_inverts_direction() -> None:
    """For loss-like metrics, higher = regression."""
    rng = random.Random(2)
    baselines = [0.10 + rng.gauss(0, 0.005) for _ in range(20)]
    # Candidates much higher than baseline → regressions (worse loss).
    candidates = [0.30, 0.32, 0.28]
    client = _build_api("val_loss", baselines, candidates)
    report = detect_regressions(
        client,
        project="demo/regress-test",
        metric="val_loss",
        mode="min",
        since="24h",
    )
    assert len(report.regressions) == 3
    assert all(f.direction == "regression" for f in report.regressions)
    # z > 0 (above baseline mean) is unfavorable for min-mode.
    assert all(f.z_score > 0 for f in report.regressions)


def test_no_change_when_candidate_matches_baseline() -> None:
    rng = random.Random(3)
    baselines = [0.85 + rng.gauss(0, 0.01) for _ in range(20)]
    candidates = [0.85, 0.851, 0.849]  # right at the baseline mean
    client = _build_api("val_acc", baselines, candidates)
    report = detect_regressions(
        client,
        project="demo/regress-test",
        metric="val_acc",
        mode="max",
        since="24h",
        significance_level=0.05,
    )
    # None should be flagged at p < 0.05
    assert report.regressions == []
    assert report.improvements == []
    assert report.n_compared == 3


def test_rejects_insufficient_baselines() -> None:
    client = _build_api("val_acc", baseline_values=[0.9], candidate_values=[0.5])
    with pytest.raises(ValueError, match="≥2 baseline runs"):
        detect_regressions(
            client,
            project="demo/regress-test",
            metric="val_acc",
        )


def test_rejects_bad_significance_level() -> None:
    client = _build_api("val_acc", baseline_values=[0.9, 0.91], candidate_values=[0.5])
    with pytest.raises(ValueError):
        detect_regressions(
            client,
            project="demo/regress-test",
            metric="val_acc",
            significance_level=0.0,
        )
    with pytest.raises(ValueError):
        detect_regressions(
            client,
            project="demo/regress-test",
            metric="val_acc",
            significance_level=1.5,
        )


def test_notes_warn_on_small_baseline() -> None:
    client = _build_api(
        "val_acc",
        baseline_values=[0.90, 0.91, 0.89],
        candidate_values=[0.85],
    )
    report = detect_regressions(
        client,
        project="demo/regress-test",
        metric="val_acc",
        since="24h",
    )
    assert "Small-baseline warning" in report.notes


def test_notes_warn_on_zero_std_baseline() -> None:
    client = _build_api(
        "val_acc",
        baseline_values=[0.9, 0.9, 0.9, 0.9],  # all identical
        candidate_values=[0.5],
    )
    report = detect_regressions(
        client,
        project="demo/regress-test",
        metric="val_acc",
        since="24h",
    )
    assert "std=0" in report.notes
    assert report.baseline_std > 0  # epsilon fallback


def test_two_tailed_normal_p_at_known_values() -> None:
    # P(|Z| > 1.96) ≈ 0.05, the canonical 5%-significance z.
    assert _two_tailed_normal_p(1.96) == pytest.approx(0.05, abs=1e-3)
    # P(|Z| > 0) = 1.0
    assert _two_tailed_normal_p(0.0) == pytest.approx(1.0, abs=1e-9)
    # P(|Z| > 5) ≈ tiny.
    assert _two_tailed_normal_p(5.0) < 1e-5


def test_classify_no_change_above_threshold() -> None:
    # p above α → no-change regardless of sign.
    assert _classify(z=-2.0, p=0.10, significance_level=0.05, mode="max") == "no-change"
    assert _classify(z=2.0, p=0.10, significance_level=0.05, mode="max") == "no-change"


def test_classify_directions() -> None:
    assert _classify(z=-3.0, p=0.001, significance_level=0.05, mode="max") == "regression"
    assert _classify(z=3.0, p=0.001, significance_level=0.05, mode="max") == "improvement"
    assert _classify(z=-3.0, p=0.001, significance_level=0.05, mode="min") == "improvement"
    assert _classify(z=3.0, p=0.001, significance_level=0.05, mode="min") == "regression"


def test_baseline_runs_excluded_from_candidates() -> None:
    """A run tagged as baseline must never appear in the regressions/improvements list."""
    rng = random.Random(4)
    baselines = [0.90 + rng.gauss(0, 0.01) for _ in range(20)]
    candidates = [0.70, 0.72]
    client = _build_api("val_acc", baselines, candidates)
    report = detect_regressions(
        client,
        project="demo/regress-test",
        metric="val_acc",
        mode="max",
        since="60d",  # wide window, would otherwise match baselines too
    )
    flagged_ids = {f.run_id for f in report.regressions + report.improvements}
    assert all(rid.startswith("cand") for rid in flagged_ids)
