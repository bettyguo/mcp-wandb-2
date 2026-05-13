"""Chart tools. Server-side Plotly to base64 PNG so charts render inline
in the chat client without round-tripping to the W&B web UI.

The byte-cap fallback in ``_render_with_byte_cap`` honors the
``chart_max_bytes`` setting through four stages (full, subsampled,
reduced-resolution, exceeded) and tags the response ``quality`` field
accordingly. Anything other than ``full`` appends a note to the caption.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Literal

from .._util import parse_run_id
from ..client import WandbClient
from ..models import ChartResponse
from ..plotting import comparison_figure, metrics_figure, render_png_b64, subsample
from ..settings import get_settings

logger = logging.getLogger("mcp_wandb.charts")

_REDUCED_WIDTH = 800
_REDUCED_HEIGHT = 450

Quality = Literal["full", "subsampled", "reduced-resolution", "exceeded"]


def plot_metrics(
    client: WandbClient,
    run_ids: list[str],
    metric: str,
    smoothing: float = 0.0,
    x_axis: str = "step",
    max_points: int = 1000,
) -> ChartResponse:
    """Renders a line chart of a metric across multiple runs.

    Returns a base64 PNG (capped at ~250 KB) that Claude renders inline. Use
    after the user has identified a set of runs worth visualizing. Pass
    ``smoothing=0.6`` for noisy metrics; ``x_axis='step'|'epoch'|'wall_time'``
    to change the abscissa. For diff vs. a baseline, prefer ``plot_comparison``.

    Accepts 1..20 run ids; subsamples to ``max_points`` per run for speed.
    """
    if not (1 <= len(run_ids) <= 20):
        raise ValueError("plot_metrics requires 1..20 run_ids.")
    if not metric or not metric.split(".")[-1]:
        raise ValueError("metric must be a non-empty key (e.g., 'val_acc').")
    series, total_points = _fetch_series(client, run_ids, metric, x_axis, max_points)

    def _build(srs: list[dict[str, Any]]) -> Any:
        return metrics_figure(srs, metric=metric, smoothing=smoothing, x_axis=x_axis)

    png_b64, byte_size, quality = _render_with_byte_cap(
        series=series, builder=_build, max_points=max_points
    )
    caption = f"{metric} across {len(series)} run(s)" + (
        f", smoothing={smoothing}" if smoothing else ""
    )
    caption = _append_quality_note(caption, quality)
    return ChartResponse(
        png_b64=png_b64,
        caption=caption,
        runs_plotted=len(series),
        points_sampled=total_points,
        bytes=byte_size,
        quality=quality,
    )


def plot_comparison(
    client: WandbClient,
    run_ids: list[str],
    metric: str,
    baseline_id: str,
    smoothing: float = 0.0,
    max_points: int = 1000,
) -> ChartResponse:
    """Renders a delta chart: each run's metric trajectory minus the baseline's.

    Use when the user wants to see "how much better/worse than baseline".
    Baseline is drawn at y=0; runs above the line are better (for max-mode
    metrics). ``baseline_id`` must be present in ``run_ids``.
    """
    if baseline_id not in run_ids:
        raise ValueError("baseline_id must be one of the run_ids.")
    if not (2 <= len(run_ids) <= 20):
        raise ValueError("plot_comparison requires 2..20 run_ids.")
    if not metric or not metric.split(".")[-1]:
        raise ValueError("metric must be a non-empty key (e.g., 'val_acc').")
    series, total_points = _fetch_series(client, run_ids, metric, x_axis="step", max_points=max_points)
    canonical_baseline = parse_run_id(baseline_id).split("/")[-1]
    for s in series:
        if s["id"] == canonical_baseline:
            s["is_baseline"] = True

    def _build(srs: list[dict[str, Any]]) -> Any:
        return comparison_figure(
            srs, baseline_id=canonical_baseline, metric=metric, smoothing=smoothing
        )

    png_b64, byte_size, quality = _render_with_byte_cap(
        series=series, builder=_build, max_points=max_points
    )
    plotted_non_baseline = len(series) - 1
    caption = _append_quality_note(
        f"Δ{metric} vs baseline across {plotted_non_baseline} run(s)", quality
    )
    return ChartResponse(
        png_b64=png_b64,
        caption=caption,
        runs_plotted=plotted_non_baseline,
        points_sampled=total_points,
        bytes=byte_size,
        quality=quality,
    )


def _render_with_byte_cap(
    series: list[dict[str, Any]],
    builder: Callable[[list[dict[str, Any]]], Any],
    max_points: int,
) -> tuple[str, int, Quality]:
    """Four-stage byte-cap fallback ladder.

    Returns ``(png_b64, byte_size, quality)``. ``quality`` indicates how
    far down the ladder we walked:

      1. ``full``: first render fits.
      2. ``subsampled``: second render at ``max_points // 2`` fits.
      3. ``reduced-resolution``: third render at 800×450 dims fits.
      4. ``exceeded``: even the smallest render is over budget; logged
         with ``chart_max_bytes_exceeded`` so the operator can re-tune
         ``chart_max_bytes`` or shrink the chart manually.
    """
    cap = get_settings().chart_max_bytes

    # Stage 1: full quality.
    fig = builder(series)
    png_b64, byte_size = render_png_b64(fig)
    if byte_size <= cap:
        return png_b64, byte_size, "full"

    # Stage 2: halve max_points; re-render at full dimensions.
    halved = max(1, max_points // 2)
    for s in series:
        s["xs"], s["ys"] = subsample(s["xs"], s["ys"], halved)
    fig = builder(series)
    png_b64, byte_size = render_png_b64(fig)
    if byte_size <= cap:
        return png_b64, byte_size, "subsampled"

    # Stage 3: keep the subsampled data, shrink dimensions to 800×450.
    png_b64, byte_size = render_png_b64(fig, width=_REDUCED_WIDTH, height=_REDUCED_HEIGHT)
    if byte_size <= cap:
        return png_b64, byte_size, "reduced-resolution"

    # Stage 4: still over. Log loudly and return the smallest version
    # we have so the caller still gets a chart (with the truth in the
    # caption note and the ``quality`` field).
    logger.warning(
        "chart_max_bytes_exceeded: even after subsampling + dimension "
        "shrink the chart is %d bytes (cap=%d). Consider raising "
        "chart_max_bytes or reducing run_ids / max_points.",
        byte_size,
        cap,
    )
    return png_b64, byte_size, "exceeded"


def _append_quality_note(caption: str, quality: Quality) -> str:
    if quality == "full":
        return caption
    cap_kb = get_settings().chart_max_bytes // 1024
    note_map: dict[Quality, str] = {
        "subsampled": f"(rendered at reduced point density to stay under {cap_kb} KB)",
        "reduced-resolution": f"(rendered at reduced resolution to stay under {cap_kb} KB)",
        "exceeded": (
            f"(rendered at reduced quality; chart still exceeded {cap_kb} KB. "
            "consider raising chart_max_bytes or shrinking the chart)"
        ),
    }
    return f"{caption} {note_map[quality]}"


def _fetch_series(
    client: WandbClient,
    run_ids: list[str],
    metric: str,
    x_axis: str,
    max_points: int,
) -> tuple[list[dict[str, Any]], int]:
    metric_short = metric.split(".")[-1]
    keys = [metric_short]
    if x_axis != "step":
        keys.append(x_axis)
    series = []
    total = 0
    for rid in run_ids:
        # Charts need run.history(); bypass the snapshot disk cache.
        run = client.run_live(parse_run_id(rid))
        history = _fetch_history(run, keys, samples=max_points)
        xs, ys = _history_to_xy(history, metric_short, x_axis)
        xs, ys = subsample(xs, ys, max_points)
        series.append(
            {
                "id": str(getattr(run, "id", "")),
                "name": str(getattr(run, "name", "") or getattr(run, "id", "")),
                "xs": xs,
                "ys": ys,
                "is_baseline": False,
            }
        )
        total += len(xs)
    return series, total


def _fetch_history(run: Any, keys: list[str], samples: int) -> list[dict[str, Any]]:
    history = run.history(keys=keys, samples=samples)
    if history is None:
        return []
    # pandas DataFrame path (real wandb default)
    if hasattr(history, "to_dict") and hasattr(history, "empty"):
        if history.empty:
            return []
        records: list[dict[str, Any]] = history.to_dict(orient="records")
        return records
    if hasattr(history, "to_dict"):
        records = history.to_dict(orient="records")
        return records
    if isinstance(history, list):
        return history
    try:
        return list(history)
    except TypeError:
        return []


def _history_to_xy(
    history: list[dict[str, Any]],
    metric: str,
    x_axis: str,
) -> tuple[list[Any], list[Any]]:
    xs: list[Any] = []
    ys: list[Any] = []
    for i, row in enumerate(history):
        y = row.get(metric)
        if y is None:
            continue
        x = row.get("_step", i) if x_axis == "step" else row.get(x_axis, i)
        xs.append(x)
        ys.append(y)
    return xs, ys
