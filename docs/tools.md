# Tools reference

The 17 registered tools, by category. Tool descriptions follow a consistent
shape: lead with the verb, state the typical use case, note any sharp edges,
and explicitly mark anything destructive.

## Discovery

### `list_projects(entity?, limit=100) -> ProjectSummary[]`

Lists W&B projects under an entity. Returns up to 100, sorted by
most-recently-active.

### `list_runs(project, filters?, limit=50, since?, order_by="-created_at", tag?) -> ListRunsResponse`

Lists W&B runs in a project. Supports relative-date filtering (`since="7d"`,
`"24h"`, `"2026-05-06"`) and a `tag` shortcut for the common "show me runs
tagged X" case. Returns summary fields only; call `get_run` for full detail.

### `get_run(run_id) -> Run`

Returns the full configuration, summary metrics, system info, and metadata
for one run.

## Query

### `query_runs(project, mongo_query, limit=50, cursor?, since?) -> ListRunsResponse`

Power-user filter using W&B's MongoDB-style query DSL, same syntax as the
W&B web UI's filter box. Use when you need exact filter semantics (existence
checks, regex, range operators). For simple cases, `list_runs(filters=…)`
is friendlier.

## Analysis

### `find_best_run(project, metric, mode="max", filters?) -> RunSummary?`

Returns the single best run by a target metric. `mode='min'` for loss-like
metrics.

### `find_baseline_runs(project, tag="baseline", limit=5) -> RunSummary[]`

Convenience: finds runs with the baseline tag. Chains into `compare_runs`
or `plot_comparison`.

### `compare_runs(run_ids, metrics?) -> CompareRunsResponse`

Returns a structured config + metric diff across 2–50 runs.

### `hyperparam_importance(run_ids, target_metric, method="rf", top_k=10) -> HyperparamImportanceResponse`

Ranks hyperparameters by influence on a target metric. RandomForest
`feature_importances_` by default (fast); pass `method='shap'` for SHAP
values (slower, more principled, recommended for ≤100 runs). Returns
`model_r2` (out-of-bag) so the agent can warn when the signal is weak.

### `summarize_sweep(sweep_id, target_metric?) -> SummarizeSweepResponse`

One-call narrative of a W&B sweep: best/worst/median runs, parameter ranges,
state counts, and importance ranking.

### `recommend_next_sweep(sweep_id, target_metric?, n_trials=30, narrow_factor=0.3, drop_threshold=0.05) -> SweepRecommendation`

Given a finished sweep, returns a refined sweep config: high-importance
params narrowed around the best observed value (log-scale aware), low-
importance params fixed at their best, categoricals kept. Read-only.
Includes a `confidence` flag derived from the importance model R².

### `detect_regressions(project, metric, baseline_tag="baseline", mode="max", since="7d", significance_level=0.05, limit=100) -> RegressionReport`

Pulls baseline runs, computes mean+std of the chosen metric, then z-tests
recent runs against that baseline. Flags `regression` / `improvement` /
`no-change` with a two-tailed normal-approximation p-value. Methodology
caveats live in `notes`.

## Charting

### `plot_metrics(run_ids, metric, smoothing=0.0, x_axis="step", max_points=1000) -> ChartResponse`

Line chart across 1–20 runs. Inline base64 PNG capped at ~250 KB. Larger
charts fall back through subsampling, dimension shrinking, and finally
return-with-warning; the `quality` field on the response says which stage
was reached.

### `plot_comparison(run_ids, metric, baseline_id, smoothing=0.0, max_points=1000) -> ChartResponse`

Δ-vs-baseline chart. Baseline is drawn at y=0; runs above the line are
better for max-mode metrics.

## Actions (gated)

All four require `--enable-actions` (or `MCP_WANDB_ENABLE_ACTIONS=1`).

### `launch_run(project, job, config, resource="local-process", confirm=False) -> LaunchRunResponse`

Triggers a W&B Launch run. Requires `confirm=true`. Spends compute.

### `launch_sweep(project, sweep_config, n_runs=10, resource="local-process", confirm=False) -> LaunchSweepResponse`

Creates a W&B sweep and queues `n_runs` initial trials. Requires
`confirm=true`. Spends compute.

### `add_tag(run_id, tag) -> AddTagResponse`

Adds a tag to a run. Idempotent.

### `delete_run(run_id, confirm=False) -> DeleteRunResponse`

Destructive: permanently deletes the run. Requires `confirm=true`. Cannot
be undone.

## Resources

In addition to the tools above, the server exposes one MCP resource:

### `mcp-wandb://cache/stats`

JSON snapshot of both cache layers plus a rolling W&B back-pressure
aggregate. Useful when the agent wants to ask "how is the server doing?"
without firing a tool call. The top-level `"status"` field is `"ok"`,
`"degraded"`, or `"busy"`.
