"""Shared test fixtures.

We never hit the real W&B API in unit tests; all interaction goes through
``FakeWandbClient`` which mirrors the surface ``client.WandbClient`` exposes.
The single ``test_live_demo`` test is skipped without ``WANDB_API_KEY`` and
runs only in the weekly CI smoke workflow.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from mcp_wandb.settings import Settings, set_settings

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeProject:
    name: str
    entity: str
    created_at: datetime | None = None
    lastActive: datetime | None = None
    run_count: int = 0
    url: str = ""


@dataclass
class FakeRun:
    id: str
    name: str
    entity: str
    project: str
    state: str = "finished"
    tags: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    summary_metrics: dict[str, Any] = field(default_factory=dict)
    systemMetrics: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None
    sweep: Any | None = None
    created_at: datetime | None = None
    runtime: float | None = None
    url: str = ""
    _history: list[dict[str, Any]] = field(default_factory=list)
    _deleted: bool = False
    _updated: bool = False

    @property
    def summary(self) -> dict[str, Any]:
        return self.summary_metrics

    def history(self, keys: list[str] | None = None, samples: int = 500) -> list[dict[str, Any]]:
        if not keys:
            return self._history
        out = []
        for row in self._history:
            entry = {"_step": row.get("_step")}
            for k in keys:
                if k in row:
                    entry[k] = row[k]
            out.append(entry)
        return out

    def update(self) -> None:
        self._updated = True

    def delete(self) -> None:
        self._deleted = True


@dataclass
class FakeSweep:
    id: str
    entity: str
    project: str
    config: dict[str, Any]
    runs: list[FakeRun] = field(default_factory=list)
    url: str = ""


class FakeWandbApi:
    """Mimics the wandb.Api surface used by WandbClient."""

    def __init__(self) -> None:
        self.projects_data: list[FakeProject] = []
        self.runs_data: dict[str, list[FakeRun]] = {}
        self.run_lookup: dict[str, FakeRun] = {}
        self.sweeps_data: dict[str, FakeSweep] = {}

    def projects(self, entity: str | None = None) -> list[FakeProject]:
        if entity is None:
            return list(self.projects_data)
        return [p for p in self.projects_data if p.entity == entity]

    def runs(
        self,
        path: str,
        filters: dict[str, Any] | None = None,
        order: str = "-created_at",
        per_page: int = 50,
    ) -> list[FakeRun]:
        candidates = list(self.runs_data.get(path, []))
        if filters:
            candidates = [r for r in candidates if _passes_filter(r, filters)]
        # Sort by order key.
        reverse = order.startswith("-")
        key = order.lstrip("+-")
        if key == "created_at":
            candidates.sort(key=lambda r: r.created_at or datetime.min.replace(tzinfo=UTC), reverse=reverse)
        elif key.startswith("summary_metrics."):
            metric = key.split(".", 1)[1]
            candidates.sort(
                key=lambda r: r.summary_metrics.get(metric, float("-inf") if reverse else float("inf")),
                reverse=reverse,
            )
        return candidates[:per_page]

    def run(self, path: str) -> FakeRun:
        if path not in self.run_lookup:
            raise KeyError(f"FakeRun not found: {path}")
        return self.run_lookup[path]

    def sweep(self, path: str) -> FakeSweep:
        if path not in self.sweeps_data:
            raise KeyError(f"FakeSweep not found: {path}")
        return self.sweeps_data[path]


def _passes_filter(run: FakeRun, mongo_filter: dict[str, Any]) -> bool:
    for key, value in mongo_filter.items():
        if key == "tags":
            if isinstance(value, dict) and "$in" in value:
                if not any(t in run.tags for t in value["$in"]):
                    return False
            elif value not in run.tags:
                return False
        elif key == "createdAt":
            if isinstance(value, dict) and "$gte" in value and run.created_at:
                threshold = datetime.fromisoformat(value["$gte"])
                if run.created_at < threshold:
                    return False
        elif key == "state":
            if run.state != value:
                return False
        elif key.startswith("config."):
            param = key.split(".", 1)[1]
            actual = run.config.get(param)
            if isinstance(value, dict):
                for op, threshold in value.items():
                    if op == "$lt" and not (actual is not None and actual < threshold):
                        return False
                    if op == "$lte" and not (actual is not None and actual <= threshold):
                        return False
                    if op == "$gt" and not (actual is not None and actual > threshold):
                        return False
                    if op == "$gte" and not (actual is not None and actual >= threshold):
                        return False
                    if op == "$eq" and actual != threshold:
                        return False
            elif actual != value:
                return False
        elif key.startswith("summary_metrics."):
            metric = key.split(".", 1)[1]
            actual = run.summary_metrics.get(metric)
            if isinstance(value, dict):
                for op, threshold in value.items():
                    if op == "$lt" and not (actual is not None and actual < threshold):
                        return False
                    if op == "$lte" and not (actual is not None and actual <= threshold):
                        return False
                    if op == "$gt" and not (actual is not None and actual > threshold):
                        return False
                    if op == "$gte" and not (actual is not None and actual >= threshold):
                        return False
                    if op == "$eq" and actual != threshold:
                        return False
            elif actual != value:
                return False
    return True


class FakeWandbModule:
    """Stand-in for the top-level wandb module used by client.WandbClient.wandb_module."""

    def __init__(self, api: FakeWandbApi) -> None:
        self._api = api
        self.sweep_calls: list[tuple[dict[str, Any], str | None, str | None]] = []
        self.launch_calls: list[dict[str, Any]] = []

    def Api(self, **kwargs: Any) -> FakeWandbApi:
        return self._api

    def sweep(self, config: dict[str, Any], project: str | None = None, entity: str | None = None) -> str:
        self.sweep_calls.append((config, project, entity))
        return "fake-sweep-id"

    @property
    def sdk(self) -> Any:
        outer = self

        class _Launch:
            @staticmethod
            def launch(**kwargs: Any) -> Any:
                outer.launch_calls.append(kwargs)

                class _Run:
                    id = "launched-run-id"
                    url = "https://wandb.ai/fake/run/launched-run-id"

                return _Run()

        class _Sdk:
            launch = _Launch()

        return _Sdk()

    @property
    def apis(self) -> Any:
        class _Internal:
            class Api:
                def __init__(self) -> None:
                    pass

        class _Apis:
            internal = _Internal()

        return _Apis()


class FakeWandbClient:
    """Stand-in for WandbClient that bypasses rate-limiting and retries."""

    def __init__(self, fake_api: FakeWandbApi, fake_module: FakeWandbModule | None = None) -> None:
        self._api = fake_api
        self._wandb = fake_module or FakeWandbModule(fake_api)

    @property
    def api(self) -> FakeWandbApi:
        return self._api

    @property
    def wandb_module(self) -> FakeWandbModule:
        return self._wandb

    def projects(self, entity: str | None = None) -> list[FakeProject]:
        return self._api.projects(entity=entity)

    def project(self, name: str, entity: str | None = None) -> FakeProject:
        for p in self._api.projects_data:
            if p.name == name and (entity is None or p.entity == entity):
                return p
        raise KeyError(name)

    def runs(self, path: str, filters: dict[str, Any] | None = None, order: str = "-created_at", per_page: int | None = None) -> list[FakeRun]:
        return self._api.runs(path=path, filters=filters, order=order, per_page=per_page or 50)

    def run(self, path: str) -> FakeRun:
        return self._api.run(path)

    def run_live(self, path: str) -> FakeRun:
        # The fake API doesn't model a disk cache; mirror the live path.
        return self._api.run(path)

    def sweep(self, path: str) -> FakeSweep:
        return self._api.sweep(path)


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------


def _make_history(metric: str, n: int = 50, base: float = 0.5, jitter: float = 0.01) -> list[dict[str, Any]]:
    rng = random.Random(hash(metric) & 0xFFFF)
    history = []
    for step in range(n):
        progress = step / max(1, n - 1)
        value = base + (1 - base) * progress + rng.gauss(0, jitter)
        history.append({"_step": step, metric: value, "epoch": step // 5, "wall_time": 1700000000 + step * 60})
    return history


def make_sweep_with_runs(
    entity: str = "demo",
    project: str = "cifar10-sweep",
    sweep_id: str = "abc123",
    n_runs: int = 30,
    target_metric: str = "val_acc",
) -> tuple[FakeWandbApi, FakeSweep]:
    """Build a realistic sweep with varying hyperparameters.

    Crafted so that ``lr`` is the dominant predictor of ``val_acc`` and
    ``batch_size`` is moderate, so the RF should rank lr first.
    """
    api = FakeWandbApi()
    api.projects_data.append(FakeProject(name=project, entity=entity, run_count=n_runs))
    rng = random.Random(42)
    runs: list[FakeRun] = []
    path = f"{entity}/{project}"
    sweep = FakeSweep(
        id=sweep_id,
        entity=entity,
        project=project,
        config={
            "method": "bayes",
            "metric": {"name": target_metric, "goal": "maximize"},
            "parameters": {
                "lr": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-1},
                "batch_size": {"values": [32, 64, 128, 256]},
                "optimizer": {"values": ["adam", "sgd"]},
            },
        },
        url=f"https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}",
    )
    for i in range(n_runs):
        lr = 10 ** rng.uniform(-5, -1)
        batch_size = rng.choice([32, 64, 128, 256])
        optimizer = rng.choice(["adam", "sgd"])
        # Truth: val_acc is best when lr ~ 3e-4, batch_size moderate impact, optimizer small.
        lr_signal = -((math.log10(lr) - math.log10(3e-4)) ** 2) * 0.06
        bs_signal = 0.005 * (math.log2(batch_size) - 6)
        opt_signal = 0.01 if optimizer == "adam" else 0.0
        noise = rng.gauss(0, 0.01)
        val_acc = 0.9 + lr_signal + bs_signal + opt_signal + noise
        val_loss = 1.5 - val_acc - rng.uniform(0, 0.05)

        run_id = f"run{i:03d}"
        full_id = f"{path}/{run_id}"
        run = FakeRun(
            id=run_id,
            name=f"trial-{i:03d}",
            entity=entity,
            project=project,
            state="finished",
            tags=["sweep"] + (["baseline"] if i == 0 else []),
            config={"lr": lr, "batch_size": batch_size, "optimizer": optimizer},
            summary_metrics={target_metric: val_acc, "val_loss": val_loss, "_runtime": 320 + i},
            sweep=sweep,
            created_at=datetime.now(UTC) - timedelta(days=3, minutes=i * 5),
            url=f"https://wandb.ai/{path}/runs/{run_id}",
            _history=[*_make_history(target_metric, n=50, base=val_acc - 0.1)],
        )
        # Replace history with one that converges to val_acc.
        run._history = _make_history(target_metric, n=50, base=val_acc - 0.05, jitter=0.005)
        for row in run._history:
            row["val_loss"] = max(0.01, 1.0 - row[target_metric] + rng.gauss(0, 0.01))
        runs.append(run)
        api.run_lookup[full_id] = run

    api.runs_data[path] = runs
    sweep.runs = runs
    api.sweeps_data[f"{path}/sweeps/{sweep_id}"] = sweep
    return api, sweep


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_settings() -> None:
    """Each test starts with default Settings (no actions, normal rate limits)."""
    set_settings(Settings())


@pytest.fixture
def sweep_api() -> tuple[FakeWandbApi, FakeSweep]:
    return make_sweep_with_runs()


@pytest.fixture
def fake_client(sweep_api: tuple[FakeWandbApi, FakeSweep]) -> FakeWandbClient:
    api, _ = sweep_api
    return FakeWandbClient(api)
