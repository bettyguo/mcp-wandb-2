"""Tests for the chart-byte-cap fallback ladder."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import pytest

from mcp_wandb.settings import Settings, set_settings
from mcp_wandb.tools import charts


@pytest.fixture
def fake_render(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Patch render_png_b64 with a controllable byte-size sequence.

    Yields a function ``set_sizes(*sizes)`` so each test picks the byte
    sizes returned by successive render calls. Also captures the
    width/height passed on each call so we can assert the dimension shrink
    on the third stage.
    """
    captured_dims: list[tuple[int | None, int | None]] = []
    sizes_iter: Iterator[int] = iter([])

    def _stub(fig: Any, width: int | None = None, height: int | None = None) -> tuple[str, int]:
        captured_dims.append((width, height))
        try:
            n = next(sizes_iter)
        except StopIteration:
            n = 0
        return "PNG", n

    monkeypatch.setattr(charts, "render_png_b64", _stub)

    def _set_sizes(*sizes: int) -> None:
        nonlocal sizes_iter
        sizes_iter = iter(sizes)

    return _set_sizes, captured_dims


# ---------------------------------------------------------------------------
# Helper paths exercised directly (no client wiring needed)
# ---------------------------------------------------------------------------


def test_render_helper_full_quality_first_render_fits(fake_render) -> None:  # type: ignore[no-untyped-def]
    set_sizes, dims = fake_render
    set_sizes(50)  # well under default 250 KB
    set_settings(Settings(chart_max_bytes=100))

    series = [{"id": "a", "name": "a", "xs": list(range(100)), "ys": list(range(100)), "is_baseline": False}]
    _png, size, quality = charts._render_with_byte_cap(
        series=series, builder=lambda s: object(), max_points=1000
    )
    assert quality == "full"
    assert size == 50
    assert len(dims) == 1  # only one render call


def test_render_helper_subsamples_when_over_cap(fake_render) -> None:  # type: ignore[no-untyped-def]
    set_sizes, dims = fake_render
    set_sizes(200, 80)  # first over cap, second under
    set_settings(Settings(chart_max_bytes=100))

    series = [{"id": "a", "name": "a", "xs": list(range(100)), "ys": list(range(100)), "is_baseline": False}]
    original_len = len(series[0]["xs"])
    _png, size, quality = charts._render_with_byte_cap(
        series=series, builder=lambda s: object(), max_points=10
    )
    assert quality == "subsampled"
    assert size == 80
    # The second render was at full dims (None / None means defaults).
    assert dims[-1] == (None, None)
    # series was downsampled in place.
    assert len(series[0]["xs"]) < original_len


def test_render_helper_falls_back_to_reduced_resolution(fake_render) -> None:  # type: ignore[no-untyped-def]
    set_sizes, dims = fake_render
    set_sizes(200, 150, 60)  # over, over, under
    set_settings(Settings(chart_max_bytes=100))

    series = [{"id": "a", "name": "a", "xs": list(range(200)), "ys": list(range(200)), "is_baseline": False}]
    _png, size, quality = charts._render_with_byte_cap(
        series=series, builder=lambda s: object(), max_points=20
    )
    assert quality == "reduced-resolution"
    assert size == 60
    # Third render at 800×450.
    assert dims[-1] == (charts._REDUCED_WIDTH, charts._REDUCED_HEIGHT)


def test_render_helper_exceeded_logs_warning_and_returns_smallest(
    fake_render, caplog: pytest.LogCaptureFixture
) -> None:  # type: ignore[no-untyped-def]
    set_sizes, _dims = fake_render
    set_sizes(500, 400, 300)  # never gets under cap
    set_settings(Settings(chart_max_bytes=100))

    series = [{"id": "a", "name": "a", "xs": list(range(50)), "ys": list(range(50)), "is_baseline": False}]
    with caplog.at_level(logging.WARNING, logger="mcp_wandb.charts"):
        _png, size, quality = charts._render_with_byte_cap(
            series=series, builder=lambda s: object(), max_points=10
        )
    assert quality == "exceeded"
    assert size == 300  # still the last render
    assert any("chart_max_bytes_exceeded" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Caption-note helper
# ---------------------------------------------------------------------------


def test_caption_note_skipped_for_full() -> None:
    set_settings(Settings(chart_max_bytes=250_000))
    assert charts._append_quality_note("base", "full") == "base"


def test_caption_note_for_subsampled() -> None:
    set_settings(Settings(chart_max_bytes=100 * 1024))
    note = charts._append_quality_note("base", "subsampled")
    assert "reduced point density" in note
    assert "100 KB" in note


def test_caption_note_for_reduced_resolution() -> None:
    set_settings(Settings(chart_max_bytes=100 * 1024))
    note = charts._append_quality_note("base", "reduced-resolution")
    assert "reduced resolution" in note


def test_caption_note_for_exceeded() -> None:
    set_settings(Settings(chart_max_bytes=50 * 1024))
    note = charts._append_quality_note("base", "exceeded")
    assert "chart still exceeded" in note
    assert "50 KB" in note


# ---------------------------------------------------------------------------
# Integration with plot_metrics: quality field propagates
# ---------------------------------------------------------------------------


def test_plot_metrics_full_quality_response(fake_client, fake_render) -> None:  # type: ignore[no-untyped-def]
    set_sizes, _ = fake_render
    set_sizes(50)  # comfortably under cap
    set_settings(Settings(chart_max_bytes=250_000))

    runs = fake_client.api.runs_data["demo/cifar10-sweep"][:3]
    ids = [f"demo/cifar10-sweep/{r.id}" for r in runs]
    resp = charts.plot_metrics(fake_client, run_ids=ids, metric="val_acc")
    assert resp.quality == "full"
    assert "reduced" not in resp.caption


def test_plot_metrics_subsampled_response_carries_note(fake_client, fake_render) -> None:  # type: ignore[no-untyped-def]
    set_sizes, _ = fake_render
    set_sizes(500, 100)  # first over, second under
    set_settings(Settings(chart_max_bytes=200))

    runs = fake_client.api.runs_data["demo/cifar10-sweep"][:3]
    ids = [f"demo/cifar10-sweep/{r.id}" for r in runs]
    resp = charts.plot_metrics(fake_client, run_ids=ids, metric="val_acc")
    assert resp.quality == "subsampled"
    assert "reduced point density" in resp.caption


def test_plot_comparison_quality_propagates(fake_client, fake_render) -> None:  # type: ignore[no-untyped-def]
    set_sizes, _ = fake_render
    set_sizes(500, 400, 50)  # subsample over, reduced-res under
    set_settings(Settings(chart_max_bytes=200))

    runs = fake_client.api.runs_data["demo/cifar10-sweep"][:3]
    ids = [f"demo/cifar10-sweep/{r.id}" for r in runs]
    resp = charts.plot_comparison(
        fake_client, run_ids=ids, metric="val_acc", baseline_id=ids[0]
    )
    assert resp.quality == "reduced-resolution"
    assert "reduced resolution" in resp.caption
