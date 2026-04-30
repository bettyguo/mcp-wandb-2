"""Tests for the gated action tools."""

from __future__ import annotations

import pytest

from mcp_wandb.settings import Settings, set_settings
from mcp_wandb.tools.actions import (
    ActionsDisabledError,
    ConfirmationRequiredError,
    add_tag,
    delete_run,
    launch_run,
    launch_sweep,
)


def test_launch_run_rejects_when_actions_disabled(fake_client) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ActionsDisabledError):
        launch_run(
            fake_client,
            project="demo/cifar10-sweep",
            job="demo/jobs/cifar:latest",
            config={"lr": 1e-3},
            confirm=True,
        )


def test_launch_run_requires_confirm(fake_client) -> None:  # type: ignore[no-untyped-def]
    set_settings(Settings(enable_actions=True))
    with pytest.raises(ConfirmationRequiredError):
        launch_run(
            fake_client,
            project="demo/cifar10-sweep",
            job="demo/jobs/cifar:latest",
            config={"lr": 1e-3},
            confirm=False,
        )


def test_launch_run_happy_path(fake_client) -> None:  # type: ignore[no-untyped-def]
    set_settings(Settings(enable_actions=True))
    resp = launch_run(
        fake_client,
        project="demo/cifar10-sweep",
        job="demo/jobs/cifar:latest",
        config={"lr": 1e-3},
        confirm=True,
    )
    assert resp.run_id == "launched-run-id"
    assert fake_client.wandb_module.launch_calls, "launch SDK was not called"


def test_launch_sweep_calls_wandb_sweep(fake_client) -> None:  # type: ignore[no-untyped-def]
    set_settings(Settings(enable_actions=True))
    resp = launch_sweep(
        fake_client,
        project="demo/cifar10-sweep",
        sweep_config={"method": "grid", "metric": {"name": "val_acc", "goal": "maximize"}, "parameters": {}},
        n_runs=4,
        confirm=True,
    )
    assert resp.sweep_id == "fake-sweep-id"
    assert resp.n_runs_queued == 4
    assert fake_client.wandb_module.sweep_calls, "wandb.sweep was not called"


def test_add_tag_mutates_run(fake_client) -> None:  # type: ignore[no-untyped-def]
    set_settings(Settings(enable_actions=True))
    resp = add_tag(fake_client, run_id="demo/cifar10-sweep/run000", tag="winner")
    assert "winner" in resp.tags_after
    # Underlying fake run was marked updated.
    run = fake_client.api.run("demo/cifar10-sweep/run000")
    assert run._updated is True


def test_add_tag_idempotent(fake_client) -> None:  # type: ignore[no-untyped-def]
    set_settings(Settings(enable_actions=True))
    add_tag(fake_client, run_id="demo/cifar10-sweep/run000", tag="winner")
    resp = add_tag(fake_client, run_id="demo/cifar10-sweep/run000", tag="winner")
    assert resp.tags_after.count("winner") == 1


def test_delete_run_requires_confirm(fake_client) -> None:  # type: ignore[no-untyped-def]
    set_settings(Settings(enable_actions=True))
    with pytest.raises(ConfirmationRequiredError):
        delete_run(fake_client, run_id="demo/cifar10-sweep/run000", confirm=False)


def test_delete_run_happy_path(fake_client) -> None:  # type: ignore[no-untyped-def]
    set_settings(Settings(enable_actions=True))
    resp = delete_run(fake_client, run_id="demo/cifar10-sweep/run000", confirm=True)
    assert resp.run_id == "run000"
    run = fake_client.api.run("demo/cifar10-sweep/run000")
    assert run._deleted is True
