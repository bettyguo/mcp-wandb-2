# Contributing

Thanks for your interest. PRs welcome.

## Setup

```bash
git clone https://github.com/<your-username>/mcp-wandb
cd mcp-wandb
uv sync --extra dev --extra shap
uv run pytest -q
```

If you don't have `uv`: `pip install uv` or see [astral.sh/uv](https://astral.sh/uv).

## Style

* Python 3.11+.
* `ruff check .` and `ruff format .`.
* `mypy --strict` on `src/mcp_wandb`. Tests don't have to be strict.
* Type hints on public functions.
* For the tools in `src/mcp_wandb/tools/`, the docstring doubles as the
  LLM-facing description. Lead with the verb, state the typical use case,
  call out sharp edges, and explicitly mark anything destructive.

## What to look for

* One concern per PR. Bug fix and refactor go in two PRs.
* New code paths need tests. The unit suite mocks the W&B API; the
  `test_demo_path.py` smoke test runs against a live project weekly in CI.
* If your idea overlaps the official [`wandb/wandb-mcp-server`](https://github.com/wandb/wandb-mcp-server),
  please file the equivalent issue there too. We're complementary, not
  competing.

## Layout

```
src/mcp_wandb/
  tools/        # tool implementations (one module per category)
  importance.py # RF / SHAP backends
  plotting.py   # Plotly figure construction
  client.py     # the only direct wandb.Api() consumer
  server.py     # FastMCP wiring
  _cache.py     # in-memory + disk caches
  _logging.py   # structured logger + @instrumented decorator
  _metrics.py   # rolling-window back-pressure aggregator
  _telemetry.py # opt-in OpenTelemetry spans
tests/          # mocked W&B fakes; network-free
examples/       # tool-sequence walkthroughs
docs/           # mkdocs-material site
```

## Release

1. Bump `__version__` in `src/mcp_wandb/__init__.py` and `pyproject.toml`.
2. Tag `v<version>` on `main`.
3. The release workflow publishes to PyPI via trusted publisher.

## Code of conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
