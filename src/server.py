"""FastMCP server entry. Wires every tool to a typed JSON schema.

Supports two transports:

* **stdio** (default), for Claude Desktop, Cursor, Claude Code local config.
* **streamable-http**, for hosted deployments. Bearer-token auth using a
  W&B API key (OAuth 2.1 fully-fledged provider tracked for v1.1; until W&B
  publishes their OAuth endpoints we proxy the same bearer mechanism the
  official server uses for its hosted endpoint).
"""

from __future__ import annotations

import contextvars
import logging
from typing import Any, Literal

from fastmcp import FastMCP

from ._logging import configure_logging, instrumented
from .auth import Credentials, resolve
from .client import WandbClient
from .models import (
    AddTagResponse,
    ChartResponse,
    CompareRunsResponse,
    DeleteRunResponse,
    HyperparamImportanceResponse,
    LaunchRunResponse,
    LaunchSweepResponse,
    ListRunsResponse,
    ProjectSummary,
    RegressionReport,
    Run,
    RunSummary,
    SummarizeSweepResponse,
    SweepRecommendation,
)
from .settings import get_settings
from .tools import actions as actions_mod
from .tools import analysis as analysis_mod
from .tools import charts as charts_mod
from .tools import discovery as discovery_mod
from .tools import query as query_mod

logger = logging.getLogger(__name__)


_bearer_token_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "bearer_token", default=None
)


def set_bearer_token(token: str | None) -> None:
    _bearer_token_var.set(token)


def _client() -> WandbClient:
    bearer = _bearer_token_var.get()
    creds: Credentials = resolve(bearer_token=bearer, base_url=get_settings().wandb_base_url)
    return WandbClient(creds)


def build_app() -> FastMCP:
    """Construct the FastMCP application and register every tool.

    Kept as a function so tests can build isolated instances and the CLI
    can construct the app after settings are finalized.
    """
    mcp = FastMCP(
        name="mcp-wandb",
        instructions=(
            "Analytical MCP companion for Weights & Biases. Use the tools here "
            "for hyperparam importance, sweep summaries, run-delta analysis, and "
            "Plotly chart rendering. For raw W&B queries, Weave traces, artifact "
            "registries, or W&B report creation, prefer the official "
            "wandb/wandb-mcp-server (designed to coexist with this one)."
        ),
    )

    # Discovery

    @mcp.tool
    @instrumented("list_projects")
    def list_projects(entity: str | None = None, limit: int = 100) -> list[ProjectSummary]:
        """Lists W&B projects under the given entity (user or team). Use when the user asks "what projects do I have" or needs to discover a project name before any other tool. Returns up to 100 projects sorted by most-recently-active. Pass entity to scope to a teammate or team."""
        return discovery_mod.list_projects(_client(), entity=entity, limit=limit)

    @mcp.tool
    @instrumented("list_runs")
    def list_runs(
        project: str,
        filters: dict[str, Any] | None = None,
        limit: int = 50,
        since: str | None = None,
        order_by: str = "-created_at",
        tag: str | None = None,
    ) -> ListRunsResponse:
        """Lists W&B runs in a project. Use when the user wants to see a slice of recent experiments. Supports relative-date filtering via since (e.g., since='7d', '24h', '2026-05-06') and a tag shortcut (tag='baseline' expands to filters={'tags': {'$in': ['baseline']}}, but an explicit filters['tags'] wins). Combine with filters for richer queries (e.g., filters={'config.lr': {'$lt': 0.01}}). Returns summary fields only; call get_run for full detail."""
        return discovery_mod.list_runs(
            _client(),
            project=project,
            filters=filters,
            limit=limit,
            since=since,
            order_by=order_by,
            tag=tag,
        )

    @mcp.tool
    @instrumented("get_run")
    def get_run(run_id: str) -> Run:
        """Returns the full configuration, summary metrics, system info, and metadata for one run. Use after list_runs when the agent needs detail to answer a follow-up question. run_id is 'entity/project/run_id'."""
        return discovery_mod.get_run(_client(), run_id=run_id)

    # Query

    @mcp.tool
    @instrumented("query_runs")
    def query_runs(
        project: str,
        mongo_query: dict[str, Any],
        limit: int = 50,
        cursor: str | None = None,
        since: str | None = None,
    ) -> ListRunsResponse:
        """Power-user filter using W&B's MongoDB-style query DSL. Same syntax as the W&B web UI's filter box. Use when an agent needs exact filter semantics (existence checks, regex, range operators). For simple cases, list_runs(filters=…) is friendlier. Pass since as a relative-date shortcut (e.g., since='7d'); it is merged as a createdAt $gte clause unless mongo_query already specifies createdAt. Cursor returned for pagination."""
        return query_mod.query_runs(
            _client(),
            project=project,
            mongo_query=mongo_query,
            limit=limit,
            cursor=cursor,
            since=since,
        )

    # Analysis

    @mcp.tool
    @instrumented("find_best_run")
    def find_best_run(
        project: str,
        metric: str,
        mode: Literal["min", "max"] = "max",
        filters: dict[str, Any] | None = None,
    ) -> RunSummary | None:
        """Finds the single best run in a project by a target metric. Use when the user asks 'what was my best run for val_acc'. Returns the run plus its full config so the agent can reason about why it won. Pass filters to scope to a tag, sweep, or time range. mode='min' for loss-like metrics."""
        return analysis_mod.find_best_run(
            _client(), project=project, metric=metric, mode=mode, filters=filters
        )

    @mcp.tool
    @instrumented("find_baseline_runs")
    def find_baseline_runs(project: str, tag: str = "baseline", limit: int = 5) -> list[RunSummary]:
        """Finds the runs tagged as baselines in a project. Default tag is 'baseline'; pass a different tag if your team uses another convention. Use as the second step in compare-against-baseline workflows, then chain into compare_runs or plot_comparison."""
        return analysis_mod.find_baseline_runs(_client(), project=project, tag=tag, limit=limit)

    @mcp.tool
    @instrumented("compare_runs")
    def compare_runs(run_ids: list[str], metrics: list[str] | None = None) -> CompareRunsResponse:
        """Compares two or more runs and returns a structured config + metric diff. Use when the user wants to know 'what's different between these runs'. Output is designed for an LLM to narrate, not for human eyeballs: config keys that match are reported with distinct=False so the model can call them out as agreement points. For visual comparison, follow up with plot_comparison. Accepts 2-50 run_ids in 'entity/project/run_id' form."""
        return analysis_mod.compare_runs(_client(), run_ids=run_ids, metrics=metrics)

    @mcp.tool
    @instrumented("hyperparam_importance")
    def hyperparam_importance(
        run_ids: list[str],
        target_metric: str,
        method: Literal["rf", "shap"] = "rf",
        top_k: int = 10,
    ) -> HyperparamImportanceResponse:
        """Ranks hyperparameters by influence on a target metric over a set of runs (typically a sweep). Use when the user asks 'what hyperparam mattered'. Default method is RandomForest feature_importances_ (fast); pass method='shap' for SHAP values (slow but more principled, recommended for ≤100 runs). model_r2 is reported so the agent can warn if the importance ranking is unreliable. Methodology disclosure is in the response notes; narrate it to the user."""
        return analysis_mod.hyperparam_importance(
            _client(), run_ids=run_ids, target_metric=target_metric, method=method, top_k=top_k
        )

    @mcp.tool
    @instrumented("summarize_sweep")
    def summarize_sweep(sweep_id: str, target_metric: str | None = None) -> SummarizeSweepResponse:
        """Returns a full narrative of a W&B sweep: best/worst/median runs, parameter ranges explored, finished/failed counts, and an importance ranking. Use when the user asks 'how did my sweep go'. Designed as a one-call analyst summary. If target_metric is omitted it is read from the sweep config. sweep_id must be 'entity/project/sweep_id'."""
        return analysis_mod.summarize_sweep(
            _client(), sweep_id=sweep_id, target_metric=target_metric
        )

    @mcp.tool
    @instrumented("recommend_next_sweep")
    def recommend_next_sweep(
        sweep_id: str,
        target_metric: str | None = None,
        n_trials: int = 30,
        narrow_factor: float = 0.3,
        drop_threshold: float = 0.05,
    ) -> SweepRecommendation:
        """Recommends a refined sweep config based on a just-finished sweep's results. Use after summarize_sweep when the user asks 'what should I try next' or 'help me design the next sweep'. The recommendation is READ-ONLY: it does NOT launch anything; the user composes the returned recommended_config with launch_sweep(confirm=true). Heuristic: narrow high-importance numeric params to ±narrow_factor around the best observed value (log-scale aware), drop low-importance params (fix at best), keep categoricals. Response includes a confidence flag (high/moderate/low) derived from the importance model's R². Warn the user before they invest compute when confidence is low."""
        return analysis_mod.recommend_next_sweep(
            _client(),
            sweep_id=sweep_id,
            target_metric=target_metric,
            n_trials=n_trials,
            narrow_factor=narrow_factor,
            drop_threshold=drop_threshold,
        )

    @mcp.tool
    @instrumented("detect_regressions")
    def detect_regressions(
        project: str,
        metric: str,
        baseline_tag: str = "baseline",
        mode: Literal["min", "max"] = "max",
        since: str = "7d",
        significance_level: float = 0.05,
        limit: int = 100,
    ) -> RegressionReport:
        """Flags recent runs whose metric differs significantly from a tagged baseline cohort. Use when the user asks 'did anything regress recently' or 'is my last week of runs worse than baseline'. Methodology: computes mean+std over the baseline runs, then for each candidate run computes a z-score and a two-tailed normal-approximation p-value (stdlib math, no SciPy). Flags p < significance_level as a regression (direction unfavorable for mode) or improvement (favorable). READ-ONLY: never deletes or tags anything. Methodology caveats (small baseline n, multiple-comparison expected false positives, degenerate-zero-std fallback) are in the response notes; narrate them to the user. mode='max' for accuracy-like metrics, mode='min' for loss-like."""
        return analysis_mod.detect_regressions(
            _client(),
            project=project,
            metric=metric,
            baseline_tag=baseline_tag,
            mode=mode,
            since=since,
            significance_level=significance_level,
            limit=limit,
        )

    # Charting

    @mcp.tool
    @instrumented("plot_metrics")
    def plot_metrics(
        run_ids: list[str],
        metric: str,
        smoothing: float = 0.0,
        x_axis: Literal["step", "epoch", "wall_time"] = "step",
        max_points: int = 1000,
    ) -> ChartResponse:
        """Renders a line chart of a metric across multiple runs. Returns a base64 PNG (max ~250 KB) that Claude renders inline. Use after the user has identified a set of runs worth visualizing. Pass smoothing=0.6 for noisy metrics. For diff vs. a baseline, prefer plot_comparison. Accepts 1-20 run_ids."""
        return charts_mod.plot_metrics(
            _client(),
            run_ids=run_ids,
            metric=metric,
            smoothing=smoothing,
            x_axis=x_axis,
            max_points=max_points,
        )

    @mcp.tool
    @instrumented("plot_comparison")
    def plot_comparison(
        run_ids: list[str],
        metric: str,
        baseline_id: str,
        smoothing: float = 0.0,
        max_points: int = 1000,
    ) -> ChartResponse:
        """Renders a delta chart: each run's metric trajectory minus the baseline's. Use when the user wants to see 'how much better/worse than baseline'. Baseline is drawn at y=0; runs above the line are better (for max-mode metrics). baseline_id must be one of the run_ids."""
        return charts_mod.plot_comparison(
            _client(),
            run_ids=run_ids,
            metric=metric,
            baseline_id=baseline_id,
            smoothing=smoothing,
            max_points=max_points,
        )

    # Actions (gated)

    @mcp.tool
    @instrumented("launch_run")
    def launch_run(
        project: str,
        job: str,
        config: dict[str, Any],
        resource: str = "local-process",
        confirm: bool = False,
    ) -> LaunchRunResponse:
        """ACTION (spends compute). Triggers a W&B Launch run with the given config against the given job artifact. Requires confirm=true and the server to be started with --enable-actions. Use only when the user has clearly authorized starting a new run. Defaults to a local-process resource; pass a queue name for cluster execution."""
        return actions_mod.launch_run(
            _client(),
            project=project,
            job=job,
            config=config,
            resource=resource,
            confirm=confirm,
        )

    @mcp.tool
    @instrumented("launch_sweep")
    def launch_sweep(
        project: str,
        sweep_config: dict[str, Any],
        n_runs: int = 10,
        resource: str = "local-process",
        confirm: bool = False,
    ) -> LaunchSweepResponse:
        """ACTION (spends compute). Creates a W&B sweep with the given config and launches n_runs initial trials. Requires confirm=true and --enable-actions. Use when the user wants to kick off a hyperparam search. Sweep config follows the standard W&B sweep YAML schema (method, metric, parameters)."""
        return actions_mod.launch_sweep(
            _client(),
            project=project,
            sweep_config=sweep_config,
            n_runs=n_runs,
            resource=resource,
            confirm=confirm,
        )

    @mcp.tool
    @instrumented("add_tag")
    def add_tag(run_id: str, tag: str) -> AddTagResponse:
        """Adds a tag to a run. Requires --enable-actions. Use for lightweight curation (e.g., marking baselines, flagging interesting runs). Tags are idempotent: adding an existing tag is a no-op."""
        return actions_mod.add_tag(_client(), run_id=run_id, tag=tag)

    @mcp.tool
    @instrumented("delete_run")
    def delete_run(run_id: str, confirm: bool = False) -> DeleteRunResponse:
        """DESTRUCTIVE: permanently deletes the run and its logged data. Requires confirm=true and --enable-actions. Use only when the user has explicitly asked to delete a specific run. Cannot be undone."""
        return actions_mod.delete_run(_client(), run_id=run_id, confirm=confirm)

    # Resources (read-only; useful for "show me server state" without
    # consuming a tool slot).

    _register_resources(mcp)

    return mcp


def _register_resources(mcp: FastMCP) -> None:
    """Attach MCP resources to the app.

    Pulled into a helper so the ``build_app`` body stays focused on tool
    registration. Resources are read-only and fetched via ``resources/read``
    rather than called as functions, which makes them a good home for
    server-health surfaces like the cache stats below.
    """
    import json as _json

    from ._cache import cache_health_payload

    try:
        @mcp.resource("mcp-wandb://cache/stats", mime_type="application/json")
        def _cache_stats_resource() -> str:
            """Server health snapshot.

            Returns JSON with three blocks: ``memory`` cache stats
            (size/hits/misses/evictions/hit_rate), ``disk`` cache stats
            (entry_count/size_bytes), and ``wandb_api`` back-pressure
            aggregates (rate_limited_ms + retries over the rolling window).
            A top-level ``status`` field ("ok"/"degraded"/"busy") gives the
            agent a glance summary.
            """
            return _json.dumps(cache_health_payload(), indent=2, default=str)

    except AttributeError:
        # Older fastmcp may not expose @resource on the FastMCP instance.
        # Skip the registration silently; the helper ``cache_health_payload()``
        # is still importable for operators who want to surface the stats
        # some other way.
        return


def serve_stdio() -> None:
    """Run the server over stdio (Claude Desktop / Cursor / Claude Code local)."""
    configure_logging()
    from ._telemetry import init_telemetry

    init_telemetry()
    app = build_app()
    app.run()


def serve_http(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the server over Streamable HTTP with Bearer-token auth.

    The ``Authorization: Bearer <wandb-api-key>`` header on each request is
    captured into a contextvar (with a per-request reset Token) so
    concurrent asyncio tasks don't see each other's credentials. OAuth 2.1
    will land once W&B publishes their endpoints.
    """
    configure_logging()
    from ._telemetry import init_telemetry

    init_telemetry()
    app = build_app()
    try:
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.requests import Request as _Request

        class _BearerCapture(BaseHTTPMiddleware):  # type: ignore[misc]
            async def dispatch(self, request: _Request, call_next: Any) -> Any:
                auth = request.headers.get("authorization", "")
                token = auth[7:].strip() if auth.lower().startswith("bearer ") else None
                reset_token = _bearer_token_var.set(token)
                try:
                    return await call_next(request)
                finally:
                    _bearer_token_var.reset(reset_token)

        http_app = app.http_app()  # may AttributeError on older fastmcp
        http_app.add_middleware(_BearerCapture)
        import uvicorn

        uvicorn.run(http_app, host=host, port=port)
    except (ImportError, AttributeError):
        # Older FastMCP versions don't expose http_app(); fall back to the
        # built-in transport runner. Bearer auth is not captured in that path;
        # users must rely on env-based WANDB_API_KEY in that fallback.
        app.run(transport="streamable-http", host=host, port=port)
