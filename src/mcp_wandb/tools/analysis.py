"""Analysis tools.

These are the tools the official W&B MCP server doesn't provide. Each
returns a structured response that an LLM can narrate, with methodology
notes embedded where the result depends on a heuristic (RF importance,
z-test p-values).
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Literal

from .._util import (
    coerce_metric_value,
    flatten_config,
    parse_project,
    parse_run_id,
    parse_since,
    parse_sweep_id,
)
from ..client import WandbClient, batch_runs
from ..importance import compute_importance
from ..models import (
    CompareRunsResponse,
    ConfigDiffEntry,
    HyperparamImportanceResponse,
    MetricDiffEntry,
    RegressionFlag,
    RegressionReport,
    RunSummary,
    SummarizeSweepResponse,
    SweepParamRange,
    SweepRecommendation,
)
from .discovery import _run_to_summary


def find_best_run(
    client: WandbClient,
    project: str,
    metric: str,
    mode: Literal["min", "max"] = "max",
    filters: dict[str, Any] | None = None,
) -> RunSummary | None:
    """Finds the single best run in a project by a target metric.

    Use when the user asks "what was my best run for `val_acc`". Returns the
    run plus its full config so the agent can reason about why it won. Pass
    ``filters`` to scope to a tag, sweep, or time range (mongo syntax).
    Use ``mode='min'`` for loss-like metrics.
    """
    summary_key = metric if metric.startswith("summary_metrics.") else f"summary_metrics.{metric}"
    metric_short = metric.split(".")[-1]
    # W&B's order param: leading "-" means descending; no prefix is ascending.
    order = f"-{summary_key}" if mode == "max" else summary_key

    entity, name = parse_project(project)
    runs_iter = client.runs(
        path=f"{entity}/{name}",
        filters=filters or {},
        order=order,
        per_page=10,
    )
    best: RunSummary | None = None
    best_val: float | None = None
    for run in runs_iter:
        summary = getattr(run, "summary_metrics", None) or getattr(run, "summary", {}) or {}
        val = coerce_metric_value(summary.get(metric_short))
        if val is None:
            continue
        if best is None or (mode == "max" and val > (best_val or float("-inf"))) or (
            mode == "min" and val < (best_val or float("inf"))
        ):
            best = _run_to_summary(run, entity=entity, project_name=name)
            best_val = val
            break  # the first one in the sorted iterator is the best by definition
    return best


def find_baseline_runs(
    client: WandbClient,
    project: str,
    tag: str = "baseline",
    limit: int = 5,
) -> list[RunSummary]:
    """Finds the runs tagged as baselines in a project.

    Default tag is ``'baseline'``; pass a different tag if your team uses
    another convention. Use as the second step in compare-against-baseline
    workflows; chain into ``compare_runs`` or ``plot_comparison``.
    """
    entity, name = parse_project(project)
    runs_iter = client.runs(
        path=f"{entity}/{name}",
        filters={"tags": {"$in": [tag]}},
        per_page=min(limit, 50),
    )
    out = []
    for run in runs_iter:
        out.append(_run_to_summary(run, entity=entity, project_name=name))
        if len(out) >= limit:
            break
    return out


def compare_runs(
    client: WandbClient,
    run_ids: list[str],
    metrics: list[str] | None = None,
) -> CompareRunsResponse:
    """Compares two or more runs and returns a structured config + metric diff.

    Use when the user wants to know "what's different between these runs".
    Output is designed for an LLM to narrate, not for human eyeballs: config
    keys that match are reported with ``distinct=False`` so the model can call
    them out as agreement points. For visual comparison, follow up with
    ``plot_comparison``.

    ``run_ids`` accepts 2–50 ids in ``entity/project/run_id`` form.
    """
    if not (2 <= len(run_ids) <= 50):
        raise ValueError("compare_runs requires 2..50 run_ids.")

    fetched = batch_runs(client, [parse_run_id(rid) for rid in run_ids])
    runs = [r for r in fetched if r is not None]
    if len(runs) < 2:
        raise ValueError(
            f"compare_runs needs at least 2 fetchable runs; got {len(runs)} "
            f"of {len(run_ids)} (rest failed to fetch; check the run ids)."
        )

    configs: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for run in runs:
        rid = str(getattr(run, "id", ""))
        configs[rid] = flatten_config(dict(getattr(run, "config", {}) or {}))
        sm = getattr(run, "summary_metrics", None) or getattr(run, "summary", {}) or {}
        if not isinstance(sm, dict):
            sm = dict(sm)
        summaries[rid] = {k: v for k, v in sm.items() if not str(k).startswith("_")}

    all_config_keys = sorted({k for c in configs.values() for k in c})
    config_diff: list[ConfigDiffEntry] = []
    agree_count = 0
    for key in all_config_keys:
        values = {rid: c.get(key) for rid, c in configs.items()}
        is_distinct = _has_distinct(list(values.values()))
        if not is_distinct:
            agree_count += 1
        config_diff.append(ConfigDiffEntry(key=key, values=values, distinct=is_distinct))

    if metrics is None:
        all_metric_keys = sorted({k for s in summaries.values() for k in s})
        # Default to numeric-only metrics; drop the noisy non-numerics.
        all_metric_keys = [
            k for k in all_metric_keys
            if all(coerce_metric_value(s.get(k)) is not None or s.get(k) is None for s in summaries.values())
        ]
    else:
        all_metric_keys = metrics

    metric_diff: list[MetricDiffEntry] = []
    for m in all_metric_keys:
        metric_values: dict[str, float | int | None] = {}
        nan_keys: list[str] = []
        for rid, s in summaries.items():
            raw = s.get(m)
            if isinstance(raw, float) and math.isnan(raw):
                nan_keys.append(rid)
                metric_values[rid] = None
            else:
                metric_values[rid] = coerce_metric_value(raw)
        metric_diff.append(MetricDiffEntry(metric=m, values=metric_values, nan_keys=nan_keys))

    agreement = agree_count / len(all_config_keys) if all_config_keys else 1.0

    return CompareRunsResponse(
        config_diff=config_diff,
        metric_diff=metric_diff,
        agreement_score=agreement,
        n_runs=len(runs),
    )


def hyperparam_importance(
    client: WandbClient,
    run_ids: list[str],
    target_metric: str,
    method: Literal["rf", "shap"] = "rf",
    top_k: int = 10,
) -> HyperparamImportanceResponse:
    """Ranks hyperparameters by influence on a target metric over a set of runs.

    Use when the user asks "what hyperparam mattered". Default method is
    RandomForest ``feature_importances_`` (fast). Pass ``method='shap'`` for
    SHAP values: slower but more principled, recommended for ≤100 runs.

    Returns a ranking plus ``model_r2`` so the agent can warn when the
    importance signal is weak (R² < 0.3). Methodology is disclosed in the
    response ``notes`` field; narrate it to the user.
    """
    if not (4 <= len(run_ids) <= 1000):
        raise ValueError("hyperparam_importance requires 4..1000 run_ids.")
    if not target_metric or not target_metric.split(".")[-1]:
        raise ValueError("target_metric must be a non-empty key.")
    top_k = max(1, min(top_k, 100))

    metric_short = target_metric.split(".")[-1]
    paths = [parse_run_id(rid) for rid in run_ids]
    fetched = batch_runs(client, paths)

    rows: list[dict[str, Any]] = []
    for run in fetched:
        if run is None:
            continue
        sm = getattr(run, "summary_metrics", None) or getattr(run, "summary", {}) or {}
        if not isinstance(sm, dict):
            sm = dict(sm)
        val = coerce_metric_value(sm.get(metric_short))
        if val is None:
            continue
        rows.append(
            {
                "config_flat": flatten_config(dict(getattr(run, "config", {}) or {})),
                "metric": val,
            }
        )

    return compute_importance(rows, target_metric=target_metric, method=method, top_k=top_k)


def summarize_sweep(
    client: WandbClient,
    sweep_id: str,
    target_metric: str | None = None,
) -> SummarizeSweepResponse:
    """Returns a full narrative of a W&B sweep.

    Use when the user asks "how did my sweep go". Bundles best/worst/median
    runs, parameter ranges explored, finished/failed/crashed counts, and an
    importance ranking (using ``method='rf'``) into one call.

    If ``target_metric`` is omitted, it is read from the sweep config's
    ``metric.name`` field. ``sweep_id`` must be in
    ``entity/project/sweep_id`` form.
    """
    sweep = client.sweep(parse_sweep_id(sweep_id))
    sweep_config = dict(getattr(sweep, "config", {}) or {})

    if target_metric is None:
        metric_cfg = sweep_config.get("metric", {})
        if isinstance(metric_cfg, dict):
            target_metric = metric_cfg.get("name")
    if not target_metric:
        raise ValueError(
            "target_metric not provided and not present in sweep config; "
            "pass target_metric explicitly."
        )

    mode_cfg = (sweep_config.get("metric", {}) or {}).get("goal", "maximize")
    mode: Literal["min", "max"] = "min" if str(mode_cfg).lower() in {"minimize", "min"} else "max"

    runs = list(getattr(sweep, "runs", []) or [])

    state_counts: dict[str, int] = {}
    for r in runs:
        s = str(getattr(r, "state", "unknown") or "unknown")
        state_counts[s] = state_counts.get(s, 0) + 1

    finished = state_counts.get("finished", 0)
    running = state_counts.get("running", 0)
    failed = state_counts.get("failed", 0)
    crashed = state_counts.get("crashed", 0)

    metric_short = target_metric.split(".")[-1]
    scored: list[tuple[float, Any]] = []
    rows: list[dict[str, Any]] = []
    for r in runs:
        sm = getattr(r, "summary_metrics", None) or getattr(r, "summary", {}) or {}
        if not isinstance(sm, dict):
            sm = dict(sm)
        val = coerce_metric_value(sm.get(metric_short))
        if val is None:
            continue
        scored.append((val, r))
        rows.append(
            {
                "config_flat": flatten_config(dict(getattr(r, "config", {}) or {})),
                "metric": val,
            }
        )

    entity = getattr(sweep, "entity", "")
    project = getattr(sweep, "project", "")

    best_summary = worst_summary = median_summary = None
    best_config_flat: dict[str, Any] = {}
    if scored:
        scored.sort(key=lambda t: t[0], reverse=(mode == "max"))
        best_run_obj = scored[0][1]
        best_summary = _run_to_summary(best_run_obj, entity=entity, project_name=project)
        best_config_flat = flatten_config(dict(getattr(best_run_obj, "config", {}) or {}))
        worst_summary = _run_to_summary(scored[-1][1], entity=entity, project_name=project)
        sorted_by_val = sorted(scored, key=lambda t: t[0])
        median_run = sorted_by_val[len(sorted_by_val) // 2][1]
        median_summary = _run_to_summary(median_run, entity=entity, project_name=project)

    param_ranges = _param_ranges_from_sweep(sweep_config, rows, best_config_flat)

    importance: HyperparamImportanceResponse | None = None
    if len(rows) >= 4:
        importance = compute_importance(rows, target_metric=target_metric, method="rf", top_k=10)

    return SummarizeSweepResponse(
        sweep_id=getattr(sweep, "id", sweep_id),
        sweep_config=sweep_config,
        target_metric=target_metric,
        n_runs=len(runs),
        finished=finished,
        running=running,
        failed=failed,
        crashed=crashed,
        state_counts=state_counts,
        best=best_summary,
        worst=worst_summary,
        median=median_summary,
        param_ranges=param_ranges,
        importance=importance,
        url=getattr(sweep, "url", None),
    )


def _param_ranges_from_sweep(
    sweep_config: dict[str, Any],
    rows: list[dict[str, Any]],
    best_config_flat: dict[str, Any],
) -> list[SweepParamRange]:
    parameters = sweep_config.get("parameters", {})
    if not isinstance(parameters, dict):
        return []
    out: list[SweepParamRange] = []
    for param, spec in parameters.items():
        observed = [r["config_flat"].get(param) for r in rows if param in r["config_flat"]]
        observed_numeric = [v for v in observed if isinstance(v, (int, float)) and not isinstance(v, bool)]
        min_v: float | None = None
        max_v: float | None = None
        if observed_numeric:
            min_v = float(min(observed_numeric))
            max_v = float(max(observed_numeric))
        distribution = (
            spec.get("distribution") if isinstance(spec, dict) else None
        ) or ("categorical" if isinstance(spec, dict) and "values" in spec else None)
        out.append(
            SweepParamRange(
                param=param,
                min=min_v,
                max=max_v,
                best_value=best_config_flat.get(param),
                distribution=distribution,
            )
        )
    return out


def recommend_next_sweep(
    client: WandbClient,
    sweep_id: str,
    target_metric: str | None = None,
    n_trials: int = 30,
    narrow_factor: float = 0.3,
    drop_threshold: float = 0.05,
) -> SweepRecommendation:
    """Recommends a refined sweep config based on the just-finished sweep's results.

    Use after ``summarize_sweep`` when the user asks "what should I try next"
    or "help me design the next sweep". The recommendation is **read-only** -
    it does NOT launch anything. The user composes the returned
    ``recommended_config`` with ``launch_sweep(confirm=true)`` to actually run.

    Heuristic:
        * For each param with RF importance ≥ ``drop_threshold`` and a numeric
          best value: narrow the range to ±``narrow_factor`` around the best
          (in log-space if the original distribution was log_uniform_values).
        * For each param with importance < ``drop_threshold``: fix at the best
          observed value (drop from the search).
        * For categorical params: keep the original ``values`` list.

    The response includes a ``confidence`` flag derived from the importance
    model's R² (≥0.6 = high, ≥0.3 = moderate, else low) so the agent can
    warn before the user invests compute on the recommendation.
    """
    summary = summarize_sweep(client, sweep_id=sweep_id, target_metric=target_metric)
    if summary.importance is None or not summary.importance.ranking:
        raise ValueError(
            f"Cannot recommend a follow-up sweep: '{sweep_id}' has no importance "
            "ranking (likely fewer than 4 runs reached the target metric)."
        )

    original_params = summary.sweep_config.get("parameters", {}) or {}
    new_params: dict[str, Any] = {}
    diff: dict[str, str] = {}

    importance_lookup = {e.param: e.importance for e in summary.importance.ranking}
    best_values = {p.param: p.best_value for p in summary.param_ranges}

    for param, spec in original_params.items():
        if not isinstance(spec, dict):
            new_params[param] = spec
            diff[param] = "kept (unrecognized spec shape)"
            continue

        importance = importance_lookup.get(param, 0.0)
        best_val = best_values.get(param)
        is_categorical = "values" in spec
        distribution = spec.get("distribution")

        if importance < drop_threshold:
            if best_val is not None:
                new_params[param] = {"value": best_val}
                diff[param] = f"dropped (importance={importance:.3f}, fixed at {best_val})"
            else:
                new_params[param] = spec
                diff[param] = f"kept (importance={importance:.3f}, but no best_value to fix at)"
            continue

        if is_categorical:
            new_params[param] = spec
            diff[param] = f"kept categorical (importance={importance:.3f})"
            continue

        if isinstance(best_val, bool) or not isinstance(best_val, (int, float)):
            new_params[param] = spec
            diff[param] = f"kept (importance={importance:.3f}; non-numeric best_value)"
            continue

        if distribution and "log" in distribution:
            new_params[param] = _narrow_log(spec, distribution, float(best_val), narrow_factor)
            diff[param] = f"narrowed log-uniform around {best_val:g} (importance={importance:.3f})"
        else:
            new_params[param] = _narrow_linear(spec, distribution, float(best_val), narrow_factor)
            diff[param] = f"narrowed ±{narrow_factor * 100:.0f}% around {best_val:g} (importance={importance:.3f})"

    recommended_config = {
        "method": "bayes",
        "metric": summary.sweep_config.get(
            "metric", {"name": summary.target_metric, "goal": "maximize"}
        ),
        "parameters": new_params,
    }

    r2 = summary.importance.model_r2
    confidence: Literal["high", "moderate", "low"]
    if r2 >= 0.6:
        confidence = "high"
    elif r2 >= 0.3:
        confidence = "moderate"
    else:
        confidence = "low"

    rationale = _build_rationale(summary, diff, n_trials, confidence)

    return SweepRecommendation(
        rationale=rationale,
        based_on_sweep_id=sweep_id,
        recommended_config=recommended_config,
        diff_from_original=diff,
        n_trials_recommended=n_trials,
        narrow_factor=narrow_factor,
        drop_threshold=drop_threshold,
        importance_r2=r2,
        confidence=confidence,
    )


def _narrow_log(
    spec: dict[str, Any], distribution: str, best_val: float, narrow_factor: float
) -> dict[str, Any]:
    base = abs(best_val) if best_val != 0 else 1e-3
    log_best = math.log10(base)
    new_min = 10 ** (log_best - narrow_factor)
    new_max = 10 ** (log_best + narrow_factor)
    orig_min = spec.get("min")
    orig_max = spec.get("max")
    if isinstance(orig_min, (int, float)):
        new_min = max(new_min, float(orig_min))
    if isinstance(orig_max, (int, float)):
        new_max = min(new_max, float(orig_max))
    return {"distribution": distribution, "min": new_min, "max": new_max}


def _narrow_linear(
    spec: dict[str, Any], distribution: str | None, best_val: float, narrow_factor: float
) -> dict[str, Any]:
    delta = max(abs(best_val) * narrow_factor, 1e-9)
    new_min = best_val - delta
    new_max = best_val + delta
    orig_min = spec.get("min")
    orig_max = spec.get("max")
    if isinstance(orig_min, (int, float)):
        new_min = max(new_min, float(orig_min))
    if isinstance(orig_max, (int, float)):
        new_max = min(new_max, float(orig_max))
    return {"distribution": distribution or "uniform", "min": new_min, "max": new_max}


def _build_rationale(
    summary: SummarizeSweepResponse,
    diff: dict[str, str],
    n_trials: int,
    confidence: str,
) -> str:
    bits = [
        f"Refined sweep proposed from {summary.n_runs} runs of '{summary.sweep_id}'.",
    ]
    if summary.importance and summary.importance.ranking:
        top = summary.importance.ranking[0]
        bits.append(
            f"Top driver was '{top.param}' (importance={top.importance:.3f}, direction={top.direction})."
        )
    n_narrowed = sum(1 for v in diff.values() if "narrowed" in v)
    n_dropped = sum(1 for v in diff.values() if "dropped" in v)
    n_kept = sum(1 for v in diff.values() if "kept" in v)
    bits.append(
        f"Narrowed {n_narrowed}, dropped {n_dropped}, kept {n_kept}; "
        f"propose {n_trials} bayes-method trials."
    )
    if summary.importance is not None:
        bits.append(
            f"Confidence: {confidence} (RF model R²={summary.importance.model_r2:.2f})."
        )
    else:
        bits.append(f"Confidence: {confidence}.")
    if confidence == "low":
        bits.append(
            "Treat the narrowing/dropping decisions as exploratory. "
            "the importance signal is weak."
        )
    return " ".join(bits)


def detect_regressions(
    client: WandbClient,
    project: str,
    metric: str,
    baseline_tag: str = "baseline",
    mode: Literal["min", "max"] = "max",
    since: str = "7d",
    significance_level: float = 0.05,
    limit: int = 100,
) -> RegressionReport:
    """Flags recent runs whose ``metric`` differs significantly from a baseline cohort.

    Use when the user asks "did anything regress recently" or "have my last
    week of runs gotten worse than baseline". The tool is **read-only** -
    it never deletes or tags anything; the user decides what to do with
    the flagged runs.

    Methodology (disclosed in ``notes``):

    1. Pull baseline runs (tagged ``baseline_tag``) and compute the mean +
       sample-std of ``metric`` across them. Requires ≥2 baselines.
    2. Pull candidate runs (``since``, capped at ``limit``).
    3. For each candidate, compute ``z = (value - baseline_mean) / baseline_std``
       and a two-tailed normal-approximation p-value
       ``p = 2 * (1 - Phi(|z|))`` using stdlib ``math.erf`` (no SciPy dep).
    4. Flag ``p < significance_level``. Classify as **regression** if the
       direction is unfavorable for ``mode`` (lower for max-mode, higher
       for min-mode), **improvement** otherwise.

    Caveats the LLM should narrate:

    * The normal approximation is a heuristic; with very few baselines
      (n < 5) the test is unreliable, and the notes field will warn.
    * With many candidates and uncorrected p-values, expect false-positive
      rate ≈ ``significance_level``; consider Bonferroni externally.
    * Runs that never logged ``metric`` are silently skipped (counted in
      ``n_compared`` vs. ``n_candidates``).
    """
    if not (0.0 < significance_level < 1.0):
        raise ValueError("significance_level must be in (0, 1).")
    metric_short = metric.split(".")[-1]
    if not metric_short:
        raise ValueError("metric must be a non-empty key.")

    baselines = find_baseline_runs(client, project=project, tag=baseline_tag, limit=50)
    baseline_values: list[float] = []
    baseline_ids: list[str] = []
    for b in baselines:
        v = coerce_metric_value(b.summary_metrics.get(metric_short))
        if v is None:
            continue
        baseline_values.append(v)
        baseline_ids.append(b.id)

    if len(baseline_values) < 2:
        raise ValueError(
            f"detect_regressions needs ≥2 baseline runs with '{metric_short}' "
            f"logged; got {len(baseline_values)} (tag={baseline_tag!r}). "
            "Either log the metric on more baselines or pass a different "
            "baseline_tag."
        )

    baseline_mean = statistics.fmean(baseline_values)
    baseline_std = statistics.stdev(baseline_values)
    if baseline_std == 0.0:
        # All baselines identical; fall back to a tiny epsilon so we can
        # still compute z-scores. Surface this in notes.
        baseline_std = max(abs(baseline_mean) * 1e-6, 1e-12)
        baseline_std_was_zero = True
    else:
        baseline_std_was_zero = False

    candidates = list_runs_impl_for_regressions(
        client, project=project, since=since, limit=limit, exclude_ids=set(baseline_ids)
    )

    regressions: list[RegressionFlag] = []
    improvements: list[RegressionFlag] = []
    n_compared = 0
    for cand in candidates:
        v = coerce_metric_value(cand.summary_metrics.get(metric_short))
        if v is None:
            continue
        n_compared += 1
        z = (v - baseline_mean) / baseline_std
        p = _two_tailed_normal_p(z)
        direction = _classify(z, p, significance_level, mode)
        flag = RegressionFlag(
            run_id=cand.id,
            run_name=cand.name,
            metric_value=v,
            z_score=z,
            p_value=p,
            direction=direction,
            created_at=cand.created_at,
            url=cand.url,
        )
        if direction == "regression":
            regressions.append(flag)
        elif direction == "improvement":
            improvements.append(flag)

    # Most-significant first (smallest p).
    regressions.sort(key=lambda f: f.p_value)
    improvements.sort(key=lambda f: f.p_value)

    return RegressionReport(
        project=project,
        metric=metric_short,
        mode=mode,
        baseline_tag=baseline_tag,
        baseline_run_ids=baseline_ids,
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        n_baseline=len(baseline_values),
        n_candidates=len(candidates),
        n_compared=n_compared,
        significance_level=significance_level,
        regressions=regressions,
        improvements=improvements,
        notes=_regression_notes(
            n_baseline=len(baseline_values),
            n_compared=n_compared,
            n_regressions=len(regressions),
            n_improvements=len(improvements),
            significance_level=significance_level,
            mode=mode,
            baseline_std_was_zero=baseline_std_was_zero,
        ),
    )


def list_runs_impl_for_regressions(
    client: WandbClient,
    project: str,
    since: str,
    limit: int,
    exclude_ids: set[str],
) -> list[RunSummary]:
    """Local helper that fetches recent runs and filters out the baseline cohort."""
    entity, name = parse_project(project)
    full_filters: dict[str, Any] = {}
    since_dt = parse_since(since)
    if since_dt is not None:
        full_filters["createdAt"] = {"$gte": since_dt.isoformat()}
    runs_iter = client.runs(
        path=f"{entity}/{name}",
        filters=full_filters,
        per_page=min(limit, 200),
    )
    out: list[RunSummary] = []
    for run in runs_iter:
        rid = str(getattr(run, "id", ""))
        if rid in exclude_ids:
            continue
        out.append(_run_to_summary(run, entity=entity, project_name=name))
        if len(out) >= limit:
            break
    return out


def _two_tailed_normal_p(z: float) -> float:
    """Two-tailed p under the standard-normal approximation, stdlib-only."""
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


def _classify(
    z: float,
    p: float,
    significance_level: float,
    mode: Literal["min", "max"],
) -> Literal["regression", "improvement", "no-change"]:
    if p >= significance_level:
        return "no-change"
    if mode == "max":
        return "regression" if z < 0 else "improvement"
    return "regression" if z > 0 else "improvement"


def _regression_notes(
    *,
    n_baseline: int,
    n_compared: int,
    n_regressions: int,
    n_improvements: int,
    significance_level: float,
    mode: Literal["min", "max"],
    baseline_std_was_zero: bool,
) -> str:
    bits = [
        f"Compared {n_compared} candidate runs against {n_baseline} baselines "
        f"(mode='{mode}', α={significance_level}).",
        f"Flagged {n_regressions} regressions and {n_improvements} improvements.",
    ]
    if n_baseline < 5:
        bits.append(
            f"Small-baseline warning: only {n_baseline} baselines; the normal "
            "approximation is unreliable, so treat individual p-values as rough."
        )
    if baseline_std_was_zero:
        bits.append(
            "All baselines were identical (std=0); used a tiny ε to permit "
            "z-score computation, but the test is degenerate. Investigate "
            "whether the baseline cohort is truly the right comparison."
        )
    if n_compared >= 20:
        expected_fp = n_compared * significance_level
        bits.append(
            f"Multiple-comparisons caveat: at α={significance_level} and "
            f"{n_compared} candidates, expect ~{expected_fp:.1f} false "
            "positives. Consider Bonferroni-correcting (α / n_compared)."
        )
    return " ".join(bits)


def _has_distinct(values: list[Any]) -> bool:
    """True if not all values are equal (treats NaN as equal to NaN)."""
    if len(values) <= 1:
        return False
    first = values[0]
    for v in values[1:]:
        if v != first:
            # Special-case NaN: nan != nan in IEEE, but we treat them as equal.
            try:
                import math

                if isinstance(v, float) and isinstance(first, float) and math.isnan(v) and math.isnan(first):
                    continue
            except (TypeError, ValueError):
                pass
            return True
    return False


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.median(values)
