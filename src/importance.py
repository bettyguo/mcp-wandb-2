"""Hyperparameter-importance backends.

Two methods:

* ``method='rf'`` (default) uses sklearn's
  ``RandomForestRegressor.feature_importances_``. Fast (under 2 s on 500
  runs) and well-understood.
* ``method='shap'`` runs TreeSHAP over the same RF. Slower but more
  principled. Requires the ``mcp-wandb[shap]`` extra.

We report ``model_r2`` (out-of-bag) so the caller can warn when the
ranking is unreliable on low-signal sweeps.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from .models import HyperparamImportanceResponse, ImportanceEntry


def _featurize(
    rows: list[dict[str, Any]],
    target_metric: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Convert list of ``{config_flat: dict, metric: float}`` rows to an (X, y, names) tuple.

    Categorical features are label-encoded; missing values get the median (numeric)
    or "missing" (categorical). Features with zero variance across rows are dropped
    so RF doesn't waste splits on them.
    """
    feature_keys: set[str] = set()
    for r in rows:
        feature_keys.update(r["config_flat"].keys())
    feature_list = sorted(feature_keys)

    y_raw: list[float] = []
    raw_columns: dict[str, list[Any]] = {k: [] for k in feature_list}
    for r in rows:
        y_raw.append(float(r["metric"]))
        for k in feature_list:
            raw_columns[k].append(r["config_flat"].get(k))

    y = np.asarray(y_raw, dtype=float)
    keep_names: list[str] = []
    keep_columns: list[np.ndarray] = []

    for k in feature_list:
        col = raw_columns[k]
        if all(isinstance(v, (int, float, bool)) or v is None for v in col):
            arr = np.array(
                [float(v) if v is not None and not isinstance(v, bool) else (1.0 if v else 0.0) if isinstance(v, bool) else np.nan for v in col],
                dtype=float,
            )
            if np.all(np.isnan(arr)):
                continue
            median = np.nanmedian(arr)
            arr = np.where(np.isnan(arr), median, arr)
            if arr.std() == 0:
                continue
            keep_names.append(k)
            keep_columns.append(arr)
        else:
            sentinel = "<missing>"
            str_col = [str(v) if v is not None else sentinel for v in col]
            uniques = sorted(set(str_col))
            if len(uniques) < 2:
                continue
            lookup = {u: i for i, u in enumerate(uniques)}
            arr = np.array([lookup[v] for v in str_col], dtype=float)
            keep_names.append(k)
            keep_columns.append(arr)

    if not keep_columns:
        return np.empty((len(rows), 0)), y, []
    X = np.column_stack(keep_columns)
    return X, y, keep_names


def compute_importance(
    rows: list[dict[str, Any]],
    target_metric: str,
    method: Literal["rf", "shap"] = "rf",
    top_k: int = 10,
) -> HyperparamImportanceResponse:
    """Train an RF on (config → metric) and return ranked importances.

    ``rows`` is a list of ``{"config_flat": <dict>, "metric": <float>}``.
    The caller has already filtered out runs missing the target metric.
    """
    if len(rows) < 4:
        return HyperparamImportanceResponse(
            ranking=[],
            method=method,
            target_metric=target_metric,
            n_runs=len(rows),
            n_features=0,
            model_r2=0.0,
            notes="Not enough runs (need ≥4) to compute importance.",
        )

    X, y, names = _featurize(rows, target_metric)
    if X.shape[1] == 0:
        return HyperparamImportanceResponse(
            ranking=[],
            method=method,
            target_metric=target_metric,
            n_runs=len(rows),
            n_features=0,
            model_r2=0.0,
            notes="No varying numeric/categorical hyperparameters found across the given runs.",
        )

    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        oob_score=True,
        bootstrap=True,
    )
    rf.fit(X, y)
    r2 = float(getattr(rf, "oob_score_", rf.score(X, y)))

    if method == "shap":
        importances = _shap_importance(rf, X)
        if importances is None:
            method = "rf"
            importances = rf.feature_importances_
    else:
        importances = rf.feature_importances_

    correlations = np.array(
        [
            float(np.corrcoef(X[:, i], y)[0, 1]) if X[:, i].std() > 0 else 0.0
            for i in range(X.shape[1])
        ]
    )

    pairs = sorted(
        zip(names, importances, correlations, strict=False),
        key=lambda t: -float(t[1]),
    )[:top_k]

    ranking = [
        ImportanceEntry(
            param=name,
            importance=float(imp),
            direction=("positive" if corr > 0.05 else "negative" if corr < -0.05 else "none"),
            correlation=float(corr),
        )
        for name, imp, corr in pairs
    ]

    notes = _quality_notes(r2, len(rows), X.shape[1], method)

    return HyperparamImportanceResponse(
        ranking=ranking,
        method=method,
        target_metric=target_metric,
        n_runs=len(rows),
        n_features=X.shape[1],
        model_r2=r2,
        notes=notes,
    )


def _shap_importance(model: Any, X: np.ndarray) -> np.ndarray | None:
    try:
        import shap
    except ImportError:
        return None
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    importances: np.ndarray = np.abs(shap_values).mean(axis=0)
    total = importances.sum()
    if total > 0:
        importances = importances / total
    return importances


def _quality_notes(r2: float, n_runs: int, n_features: int, method: str) -> str:
    bits = [f"Method: {method.upper()} over {n_runs} runs × {n_features} features."]
    if r2 < 0.3:
        bits.append(
            f"Model R²={r2:.2f} is LOW; importance ranking may be unreliable. "
            "The target metric likely depends on factors not captured in config."
        )
    elif r2 < 0.6:
        bits.append(f"Model R²={r2:.2f} is moderate; directionally trustworthy.")
    else:
        bits.append(f"Model R²={r2:.2f}: strong fit, importance ranking is trustworthy.")
    if n_runs < 20:
        bits.append("Small-sample warning: <20 runs; treat absolute importances cautiously.")
    return " ".join(bits)
