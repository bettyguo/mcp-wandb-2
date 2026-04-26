"""Chart-tool tests. We skip the kaleido roundtrip and only exercise
the figure shaping; PNG generation is covered by the live smoke test."""

from __future__ import annotations

import base64

import pytest

from mcp_wandb.tools import charts


def _set_no_render(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace render_png_b64 with a deterministic stub so tests don't need kaleido.

    Accepts ``width`` / ``height`` kwargs that the reduced-resolution
    fallback passes; without them the stub would TypeError if a test ever
    reached stage 3 of the byte-cap ladder.
    """

    def _stub(_fig: object, width: int | None = None, height: int | None = None) -> tuple[str, int]:
        return base64.b64encode(b"PNGSTUB").decode("ascii"), 7

    monkeypatch.setattr(charts, "render_png_b64", _stub)


def test_plot_metrics_returns_chart_response(fake_client, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    _set_no_render(monkeypatch)
    runs = fake_client.api.runs_data["demo/cifar10-sweep"][:5]
    ids = [f"demo/cifar10-sweep/{r.id}" for r in runs]
    resp = charts.plot_metrics(fake_client, run_ids=ids, metric="val_acc", smoothing=0.3)
    assert resp.runs_plotted == 5
    assert resp.bytes_ == 7
    assert "val_acc" in resp.caption


def test_plot_metrics_rejects_too_many(fake_client, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    _set_no_render(monkeypatch)
    runs = fake_client.api.runs_data["demo/cifar10-sweep"]
    ids = [f"demo/cifar10-sweep/{r.id}" for r in runs[:21]]
    with pytest.raises(ValueError):
        charts.plot_metrics(fake_client, run_ids=ids, metric="val_acc")


def test_plot_comparison_requires_baseline_in_ids(fake_client, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    _set_no_render(monkeypatch)
    runs = fake_client.api.runs_data["demo/cifar10-sweep"][:3]
    ids = [f"demo/cifar10-sweep/{r.id}" for r in runs]
    with pytest.raises(ValueError):
        charts.plot_comparison(
            fake_client, run_ids=ids, metric="val_acc", baseline_id="demo/cifar10-sweep/runOTHER"
        )


def test_plot_comparison_happy_path(fake_client, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    _set_no_render(monkeypatch)
    runs = fake_client.api.runs_data["demo/cifar10-sweep"][:3]
    ids = [f"demo/cifar10-sweep/{r.id}" for r in runs]
    baseline = ids[0]
    resp = charts.plot_comparison(fake_client, run_ids=ids, metric="val_acc", baseline_id=baseline)
    # runs_plotted reports non-baseline series (the baseline becomes y=0).
    assert resp.runs_plotted == 2
    assert "vs baseline across 2 run(s)" in resp.caption
