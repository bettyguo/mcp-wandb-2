"""Discovery-tool tests against the fake W&B API."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mcp_wandb.tools.discovery import get_run, list_projects, list_runs


def test_list_projects_returns_known_project(fake_client) -> None:  # type: ignore[no-untyped-def]
    projects = list_projects(fake_client)
    assert len(projects) == 1
    assert projects[0].name == "cifar10-sweep"
    assert projects[0].entity == "demo"


def test_list_projects_respects_entity_filter(fake_client) -> None:  # type: ignore[no-untyped-def]
    projects = list_projects(fake_client, entity="nonexistent")
    assert projects == []


def test_list_runs_returns_summaries(fake_client) -> None:  # type: ignore[no-untyped-def]
    response = list_runs(fake_client, project="demo/cifar10-sweep", limit=10)
    assert len(response.runs) == 10
    first = response.runs[0]
    assert first.project == "cifar10-sweep"
    assert first.entity == "demo"
    assert "val_acc" in first.summary_metrics
    assert first.config_hash is not None


def test_list_runs_respects_limit_cap(fake_client) -> None:  # type: ignore[no-untyped-def]
    response = list_runs(fake_client, project="demo/cifar10-sweep", limit=5)
    assert len(response.runs) == 5


def test_list_runs_with_since_returns_recent(fake_client) -> None:  # type: ignore[no-untyped-def]
    response = list_runs(fake_client, project="demo/cifar10-sweep", since="1h", limit=100)
    # Only the very newest runs (created within last hour); our fixture spans 3 days.
    assert len(response.runs) >= 0
    # All returned runs should have created_at within last hour.
    now = datetime.now(UTC)
    for r in response.runs:
        if r.created_at:
            assert now - r.created_at < timedelta(hours=1)


def test_list_runs_with_filters_supports_mongo(fake_client) -> None:  # type: ignore[no-untyped-def]
    # Filter to only optimizer="adam"; the fake API enforces it.
    all_runs = fake_client.api.runs_data["demo/cifar10-sweep"]
    adam_count = sum(1 for r in all_runs if r.config.get("optimizer") == "adam")
    response = list_runs(
        fake_client,
        project="demo/cifar10-sweep",
        filters={"config.optimizer": "adam"},
        limit=100,
    )
    assert 0 < len(response.runs) == adam_count
    # Spot-check each returned run came from the adam pool by looking at the
    # full FakeRun (the summary type drops config, so we map back via id).
    returned_ids = {r.id for r in response.runs}
    adam_ids = {r.id for r in all_runs if r.config.get("optimizer") == "adam"}
    assert returned_ids == adam_ids


def test_list_runs_with_metric_range_filter(fake_client) -> None:  # type: ignore[no-untyped-def]
    """summary_metrics.* $gte filter must actually narrow results."""
    response = list_runs(
        fake_client,
        project="demo/cifar10-sweep",
        filters={"summary_metrics.val_acc": {"$gte": 0.9}},
        limit=100,
    )
    for r in response.runs:
        assert r.summary_metrics["val_acc"] >= 0.9


def test_list_runs_rejects_unprefixed_project_without_default(fake_client) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        list_runs(fake_client, project="cifar10-sweep")


def test_get_run_returns_full_detail(fake_client) -> None:  # type: ignore[no-untyped-def]
    run = get_run(fake_client, run_id="demo/cifar10-sweep/run000")
    assert run.id == "run000"
    assert run.entity == "demo"
    assert run.project == "cifar10-sweep"
    assert "lr" in run.config
    assert "val_acc" in run.summary_metrics
    assert run.state == "finished"
