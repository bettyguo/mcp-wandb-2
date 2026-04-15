"""Pydantic response models shared across tools.

Every tool returns a typed model so the FastMCP-generated JSON schema is
stable, agents see consistent shapes, and tests can validate without
hand-rolling assertions.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class ProjectSummary(_Base):
    name: str
    entity: str
    created_at: datetime | None = None
    last_active_at: datetime | None = None
    run_count: int | None = None
    url: str | None = None


class RunSummary(_Base):
    """The lightweight view of a run returned by list/query tools."""

    id: str
    name: str
    entity: str
    project: str
    state: str
    tags: list[str] = Field(default_factory=list)
    summary_metrics: dict[str, Any] = Field(default_factory=dict)
    config_hash: str | None = None
    sweep_id: str | None = None
    created_at: datetime | None = None
    runtime_s: float | None = None
    url: str | None = None


class Run(_Base):
    """Full detail for a single run."""

    id: str
    name: str
    entity: str
    project: str
    state: str
    tags: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    summary_metrics: dict[str, Any] = Field(default_factory=dict)
    system_metrics: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None
    sweep_id: str | None = None
    created_at: datetime | None = None
    runtime_s: float | None = None
    url: str | None = None


class ListRunsResponse(_Base):
    runs: list[RunSummary]
    next_cursor: str | None = None


class ConfigDiffEntry(_Base):
    key: str
    values: dict[str, Any]
    distinct: bool


class MetricDiffEntry(_Base):
    metric: str
    values: dict[str, float | int | None]
    nan_keys: list[str] = Field(default_factory=list)


class CompareRunsResponse(_Base):
    config_diff: list[ConfigDiffEntry]
    metric_diff: list[MetricDiffEntry]
    agreement_score: float = Field(ge=0.0, le=1.0)
    n_runs: int


class ImportanceEntry(_Base):
    param: str
    importance: float
    direction: Literal["positive", "negative", "none"]
    correlation: float


class HyperparamImportanceResponse(_Base):
    ranking: list[ImportanceEntry]
    method: Literal["rf", "shap"]
    target_metric: str
    n_runs: int
    n_features: int
    model_r2: float
    notes: str


class SweepParamRange(_Base):
    param: str
    min: float | None = None
    max: float | None = None
    best_value: Any | None = None
    distribution: str | None = None


class SummarizeSweepResponse(_Base):
    sweep_id: str
    sweep_config: dict[str, Any]
    target_metric: str
    n_runs: int
    finished: int
    running: int
    failed: int
    crashed: int
    state_counts: dict[str, int] = Field(default_factory=dict)
    """Counts keyed by run state; superset of finished/running/failed/crashed.
    Captures non-standard states (e.g., ``preempted``) the fixed fields don't."""
    best: RunSummary | None
    worst: RunSummary | None
    median: RunSummary | None
    param_ranges: list[SweepParamRange]
    importance: HyperparamImportanceResponse | None
    url: str | None = None


class ChartResponse(_Base):
    png_b64: str
    caption: str
    runs_plotted: int
    points_sampled: int
    bytes_: int = Field(alias="bytes")
    quality: Literal["full", "subsampled", "reduced-resolution", "exceeded"] = "full"
    """Fallback stage reached during rendering:

    * ``full``: first render landed under the byte budget.
    * ``subsampled``: needed a second render with halved ``max_points``.
    * ``reduced-resolution``: needed a third render at 800x450 dims.
    * ``exceeded``: even the smallest version is over budget; a warning
      was logged and the caption carries a note about reduced quality.
    """


class LaunchRunResponse(_Base):
    run_id: str
    run_url: str | None = None
    queued_at: datetime


class LaunchSweepResponse(_Base):
    sweep_id: str
    sweep_url: str | None = None
    scheduler_run_id: str | None = None
    n_runs_queued: int


class AddTagResponse(_Base):
    run_id: str
    tags_after: list[str]


class DeleteRunResponse(_Base):
    run_id: str
    deleted_at: datetime


class SweepRecommendation(_Base):
    """Closing-the-loop suggestion produced after a sweep finishes.

    Read-only by design: returns a sweep config the user (not the agent
    alone) composes with ``launch_sweep`` to actually run.
    """

    rationale: str
    based_on_sweep_id: str
    recommended_config: dict[str, Any]
    diff_from_original: dict[str, str]
    n_trials_recommended: int
    narrow_factor: float
    drop_threshold: float
    importance_r2: float
    confidence: Literal["high", "moderate", "low"]


class RegressionFlag(_Base):
    run_id: str
    run_name: str
    metric_value: float
    z_score: float
    p_value: float = Field(ge=0.0, le=1.0)
    direction: Literal["regression", "improvement", "no-change"]
    created_at: datetime | None = None
    url: str | None = None


class RegressionReport(_Base):
    """Output of ``detect_regressions``.

    All flagging is anchored to a baseline cohort (runs tagged with
    ``baseline_tag``); the report includes the baseline statistics so the
    agent can narrate the methodology and the user can audit.
    """

    project: str
    metric: str
    mode: Literal["min", "max"]
    baseline_tag: str
    baseline_run_ids: list[str]
    baseline_mean: float
    baseline_std: float
    n_baseline: int
    n_candidates: int
    n_compared: int
    significance_level: float
    regressions: list[RegressionFlag]
    improvements: list[RegressionFlag]
    notes: str
