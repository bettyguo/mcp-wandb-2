"""Performance benchmark: times the analytical tools on synthetic sweeps.

Usage:
    python scripts/bench.py
    python scripts/bench.py --sizes 30,100,500 --out bench.json

Output is JSON, one object per (tool, sweep_size) pair, with median /
p95 / p99 latencies over 5 trials. Designed to be diffed across PRs so a
regression > 20% is visible.
"""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Make the src layout importable when invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcp_wandb.tools.analysis import (
    compare_runs,
    hyperparam_importance,
    summarize_sweep,
)
from mcp_wandb.tools.charts import plot_metrics

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from conftest import FakeWandbClient, make_sweep_with_runs


def _no_render(_fig: object) -> tuple[str, int]:
    return base64.b64encode(b"PNG").decode("ascii"), 3


def _time(fn, *args, **kwargs) -> float:  # type: ignore[no-untyped-def]
    start = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - start


def bench_one(size: int, trials: int) -> dict[str, dict[str, float]]:
    api, _sweep = make_sweep_with_runs(n_runs=size)
    client = FakeWandbClient(api)
    ids = [f"demo/cifar10-sweep/{r.id}" for r in api.runs_data["demo/cifar10-sweep"]]
    sweep_id = "demo/cifar10-sweep/sweeps/abc123"

    # Monkey-patch the PNG renderer to skip kaleido (out of scope here).
    from mcp_wandb.tools import charts

    charts.render_png_b64 = _no_render  # type: ignore[assignment]

    benchmarks = {
        "summarize_sweep": lambda: summarize_sweep(client, sweep_id=sweep_id),
        "hyperparam_importance": lambda: hyperparam_importance(client, run_ids=ids, target_metric="val_acc"),
        "compare_runs": lambda: compare_runs(client, run_ids=ids[: min(20, size)]),
        "plot_metrics": lambda: plot_metrics(client, run_ids=ids[: min(10, size)], metric="val_acc"),
    }

    out: dict[str, dict[str, float]] = {}
    for name, fn in benchmarks.items():
        samples = [_time(fn) for _ in range(trials)]
        samples.sort()
        out[name] = {
            "median_s": statistics.median(samples),
            "p95_s": samples[max(0, int(len(samples) * 0.95) - 1)],
            "p99_s": samples[max(0, int(len(samples) * 0.99) - 1)],
            "trials": float(trials),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="30,100,500", help="Comma-separated sweep sizes.")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None, help="Write to file as well as stdout.")
    parser.add_argument(
        "--concurrent",
        action="store_true",
        help="Add a serial-vs-concurrent comparison for compare_runs by toggling "
        "batch_runs max_workers; useful for catching parallelism regressions.",
    )
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    summary: dict[str, Any] = {
        "version": "v2" if args.concurrent else "v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "results": {str(size): bench_one(size, args.trials) for size in sizes},
    }
    if args.concurrent:
        summary["concurrency_comparison"] = _bench_concurrency_comparison(args.trials)
    payload = json.dumps(summary, indent=2)
    print(payload)
    if args.out:
        args.out.write_text(payload, encoding="utf-8")


def _bench_concurrency_comparison(trials: int) -> dict[str, dict[str, float]]:
    """Compare ``batch_runs`` at workers=1 vs workers=10 on a 50-run compare.

    On the FakeWandbClient the network is effectively free, so this only
    reveals threadpool overhead. Real value materializes against the live
    W&B API in the demo-smoke CI workflow.
    """
    from mcp_wandb import client as client_mod

    api, _ = make_sweep_with_runs(n_runs=50)
    client = FakeWandbClient(api)
    paths = [f"demo/cifar10-sweep/{r.id}" for r in api.runs_data["demo/cifar10-sweep"]]

    def serial() -> None:
        client_mod.batch_runs(client, paths, max_workers=1)

    def concurrent_() -> None:
        client_mod.batch_runs(client, paths, max_workers=10)

    out: dict[str, dict[str, float]] = {}
    for name, fn in (("serial_workers_1", serial), ("concurrent_workers_10", concurrent_)):
        samples = [_time(fn) for _ in range(trials)]
        samples.sort()
        out[name] = {
            "median_s": statistics.median(samples),
            "min_s": samples[0],
            "max_s": samples[-1],
            "trials": float(trials),
        }
    return out


if __name__ == "__main__":
    main()
