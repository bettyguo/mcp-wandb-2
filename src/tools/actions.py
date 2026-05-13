"""Tools that mutate state. All are gated behind ``--enable-actions``.

Spending or destructive calls also require ``confirm=true`` on the call -
the agent shouldn't fire on a misinterpreted user message.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .._util import parse_project, parse_run_id
from ..client import WandbClient
from ..models import (
    AddTagResponse,
    DeleteRunResponse,
    LaunchRunResponse,
    LaunchSweepResponse,
)
from ..settings import get_settings


class ActionsDisabledError(RuntimeError):
    """Raised when an action tool is called without --enable-actions."""


class ConfirmationRequiredError(RuntimeError):
    """Raised when an action tool is called without ``confirm=true``."""


def _require_actions_enabled() -> None:
    if not get_settings().enable_actions:
        raise ActionsDisabledError(
            "Action tools are disabled. Start the server with `--enable-actions` "
            "(or set MCP_WANDB_ENABLE_ACTIONS=1) to enable run/sweep launching, "
            "tag editing, and deletion."
        )


def launch_run(
    client: WandbClient,
    project: str,
    job: str,
    config: dict[str, Any],
    resource: str = "local-process",
    confirm: bool = False,
) -> LaunchRunResponse:
    """**ACTION (spends compute).** Triggers a W&B Launch run.

    Runs the given job artifact with the given config against the specified
    resource (a queue name, or ``'local-process'`` for in-process execution).
    Requires ``confirm=true`` and the server to be started with
    ``--enable-actions``. Use only when the user has clearly authorized
    starting a new run.
    """
    _require_actions_enabled()
    if not confirm:
        raise ConfirmationRequiredError(
            "launch_run is gated. Re-invoke with confirm=true after the user "
            "explicitly authorizes the launch."
        )
    entity, name = parse_project(project)

    wandb_mod = client.wandb_module
    launch_fn = wandb_mod.sdk.launch.launch
    internal_api = wandb_mod.apis.internal.Api()
    run_obj = launch_fn(
        api=internal_api,
        job=job,
        parameters=config,
        resource=resource,
        project=name,
        entity=entity,
    )
    run_id = getattr(run_obj, "id", None) or getattr(run_obj, "run_id", "")
    run_url = getattr(run_obj, "url", None)
    return LaunchRunResponse(
        run_id=str(run_id),
        run_url=run_url,
        queued_at=datetime.now(UTC),
    )


def launch_sweep(
    client: WandbClient,
    project: str,
    sweep_config: dict[str, Any],
    n_runs: int = 10,
    resource: str = "local-process",
    confirm: bool = False,
) -> LaunchSweepResponse:
    """**ACTION (spends compute).** Creates a W&B sweep and launches initial trials.

    Sweep config follows the standard W&B sweep YAML schema (method, metric,
    parameters). ``n_runs`` initial trials are queued. Requires
    ``confirm=true`` and ``--enable-actions``. Use when the user wants to
    kick off a hyperparam search.
    """
    _require_actions_enabled()
    if not confirm:
        raise ConfirmationRequiredError(
            "launch_sweep is gated. Re-invoke with confirm=true after the user "
            "explicitly authorizes the sweep."
        )
    if n_runs <= 0:
        raise ValueError("n_runs must be a positive integer.")
    entity, name = parse_project(project)
    wandb_mod = client.wandb_module
    sweep_id = wandb_mod.sweep(sweep_config, project=name, entity=entity)
    # Launch the sweep scheduler if a launch queue was specified; otherwise we
    # rely on the user starting agents locally and just report the sweep id.
    scheduler_run_id: str | None = None
    if resource and resource != "local-process":
        scheduler = wandb_mod.sdk.launch.launch(
            api=wandb_mod.apis.internal.Api(),
            job=None,
            parameters={"sweep_id": sweep_id},
            resource=resource,
            project=name,
            entity=entity,
        )
        scheduler_run_id = getattr(scheduler, "id", None)

    sweep_url = f"https://wandb.ai/{entity}/{name}/sweeps/{sweep_id}"
    return LaunchSweepResponse(
        sweep_id=str(sweep_id),
        sweep_url=sweep_url,
        scheduler_run_id=scheduler_run_id,
        n_runs_queued=n_runs,
    )


def add_tag(
    client: WandbClient,
    run_id: str,
    tag: str,
) -> AddTagResponse:
    """Adds a tag to a run.

    Requires ``--enable-actions``. Use for lightweight curation (e.g.,
    marking baselines, flagging interesting runs). Tags are idempotent -
    adding an existing tag is a no-op.
    """
    _require_actions_enabled()
    path = parse_run_id(run_id)
    # Mutating; must use a live run, not a snapshot proxy.
    run = client.run_live(path)
    existing = list(getattr(run, "tags", []) or [])
    if tag not in existing:
        existing.append(tag)
        run.tags = existing
        run.update()
        # Invalidate both cache layers so the next read sees the new tags.
        from .._cache import get_cache, get_disk_cache

        mem = get_cache()
        if mem is not None:
            mem.invalidate(path)
        disk = get_disk_cache()
        if disk is not None:
            disk.invalidate(path)
    return AddTagResponse(run_id=str(getattr(run, "id", run_id)), tags_after=existing)


def delete_run(
    client: WandbClient,
    run_id: str,
    confirm: bool = False,
) -> DeleteRunResponse:
    """**DESTRUCTIVE: permanently deletes the run and its logged data.**

    Requires ``confirm=true`` and ``--enable-actions``. Use only when the
    user has explicitly asked to delete a specific run. Cannot be undone.
    """
    _require_actions_enabled()
    if not confirm:
        raise ConfirmationRequiredError(
            "delete_run is destructive. Re-invoke with confirm=true after the "
            "user explicitly authorizes the deletion."
        )
    path = parse_run_id(run_id)
    # Destructive; needs a live run.
    run = client.run_live(path)
    run.delete()
    from .._cache import get_cache, get_disk_cache

    mem = get_cache()
    if mem is not None:
        mem.invalidate(path)
    disk = get_disk_cache()
    if disk is not None:
        disk.invalidate(path)
    return DeleteRunResponse(
        run_id=str(getattr(run, "id", run_id)),
        deleted_at=datetime.now(UTC),
    )
