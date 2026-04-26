"""Plotly figure construction + headless PNG export.

Kept separate from ``tools.charts`` so the figure builders are unit-testable
without spinning up a kaleido subprocess.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import plotly.graph_objects as go

from .settings import get_settings

_COLOR_BASELINE = "#999999"
_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def metrics_figure(
    series: list[dict[str, Any]],
    metric: str,
    smoothing: float = 0.0,
    x_axis: str = "step",
) -> go.Figure:
    """Build a multi-line metric plot.

    ``series`` is a list of dicts ``{"name": str, "id": str, "xs": list, "ys": list,
    "is_baseline": bool}``. Smoothing applies a simple EMA.
    """
    fig = go.Figure()
    color_idx = 0
    for s in series:
        ys = _ema(s["ys"], smoothing) if smoothing > 0 else s["ys"]
        is_baseline = s.get("is_baseline", False)
        color = _COLOR_BASELINE if is_baseline else _PALETTE[color_idx % len(_PALETTE)]
        if not is_baseline:
            color_idx += 1
        fig.add_trace(
            go.Scatter(
                x=s["xs"],
                y=ys,
                mode="lines",
                name=s["name"],
                line={"color": color, "width": 3 if is_baseline else 2, "dash": "dash" if is_baseline else "solid"},
                hovertemplate=f"<b>{s['name']}</b><br>{x_axis}=%{{x}}<br>{metric}=%{{y:.4f}}<extra></extra>",
            )
        )
    fig.update_layout(
        title=f"{metric}",
        xaxis_title=x_axis,
        yaxis_title=metric,
        template="plotly_white",
        legend={"orientation": "h", "y": -0.18},
        margin={"l": 60, "r": 30, "t": 60, "b": 80},
        font={"family": "Inter, system-ui, sans-serif", "size": 13},
    )
    return fig


def comparison_figure(
    series: list[dict[str, Any]],
    baseline_id: str,
    metric: str,
    smoothing: float = 0.0,
) -> go.Figure:
    """Render each run's trajectory as ``run - baseline``.

    Baseline is aligned to y=0; bars above are better for max-mode metrics.
    The series with ``id == baseline_id`` is dropped (it would be all zero).
    """
    baseline_series = next((s for s in series if s["id"] == baseline_id), None)
    if baseline_series is None:
        raise ValueError(f"baseline_id={baseline_id!r} not present in series.")
    baseline_lookup = dict(zip(baseline_series["xs"], baseline_series["ys"], strict=False))

    fig = go.Figure()
    fig.add_hline(y=0, line_color=_COLOR_BASELINE, line_dash="dash", line_width=2, annotation_text="baseline")
    color_idx = 0
    for s in series:
        if s["id"] == baseline_id:
            continue
        delta = [
            (y - baseline_lookup.get(x)) if baseline_lookup.get(x) is not None else None
            for x, y in zip(s["xs"], s["ys"], strict=False)
        ]
        delta_clean = [d for d in delta if d is not None]
        xs_clean = [x for x, d in zip(s["xs"], delta, strict=False) if d is not None]
        if smoothing > 0:
            delta_clean = _ema(delta_clean, smoothing)
        color = _PALETTE[color_idx % len(_PALETTE)]
        color_idx += 1
        fig.add_trace(
            go.Scatter(
                x=xs_clean,
                y=delta_clean,
                mode="lines",
                name=s["name"],
                line={"color": color, "width": 2},
                hovertemplate=f"<b>{s['name']}</b><br>Δ{metric}=%{{y:.4f}}<extra></extra>",
            )
        )
    fig.update_layout(
        title=f"Δ{metric} vs baseline ({baseline_series['name']})",
        xaxis_title="step",
        yaxis_title=f"Δ{metric}",
        template="plotly_white",
        legend={"orientation": "h", "y": -0.18},
        margin={"l": 60, "r": 30, "t": 60, "b": 80},
        font={"family": "Inter, system-ui, sans-serif", "size": 13},
    )
    return fig


def render_png_b64(
    fig: go.Figure,
    width: int | None = None,
    height: int | None = None,
) -> tuple[str, int]:
    """Render a Plotly figure to a base64 PNG, returning (b64, byte_size).

    ``width`` and ``height`` override the defaults from ``Settings``.
    Used by the chart-byte-cap fallback to shrink the rendering when even
    subsampled data overshoots ``chart_max_bytes``.
    """
    s = get_settings()
    w = width if width is not None else s.chart_width
    h = height if height is not None else s.chart_height
    buf = io.BytesIO(fig.to_image(format="png", width=w, height=h, scale=2))
    png_bytes = buf.getvalue()
    return base64.b64encode(png_bytes).decode("ascii"), len(png_bytes)


def _ema(values: list[float], alpha: float) -> list[float]:
    """Exponential moving average; alpha is the weight on the new sample."""
    if not values:
        return []
    out = [float(values[0])]
    for v in values[1:]:
        out.append(alpha * float(v) + (1.0 - alpha) * out[-1])
    return out


def subsample(xs: list[Any], ys: list[Any], max_points: int) -> tuple[list[Any], list[Any]]:
    """Even-stride downsample so charts stay under the byte budget."""
    n = len(xs)
    if n <= max_points:
        return xs, ys
    stride = max(1, n // max_points)
    return xs[::stride], ys[::stride]
