# mcp-wandb

An analytical [MCP](https://modelcontextprotocol.io) server for [Weights & Biases](https://wandb.ai). Built to install alongside the official [`wandb/wandb-mcp-server`](https://github.com/wandb/wandb-mcp-server), not in place of it.

The official server exposes the basic primitives: query runs, fetch history, list artifacts, generate reports. `mcp-wandb` adds the analysis on top:

* `hyperparam_importance`: RandomForest / SHAP feature importance with R² disclosure
* `summarize_sweep`: best/worst/median runs plus importance ranking in one call
* `compare_runs`: structured config + metric diff
* `find_best_run` / `find_baseline_runs`: common-case shortcuts
* `plot_metrics` / `plot_comparison`: Plotly to inline base64 PNG
* `recommend_next_sweep`: refined sweep config based on the last run
* `detect_regressions`: z-test recent runs against a baseline tag
* `launch_run` / `launch_sweep` / `add_tag` / `delete_run`: gated behind `--enable-actions`

Plus a thin layer over `list_projects` / `list_runs` / `get_run` / `query_runs` with relative-date sugar (`since="7d"`) so the package is usable standalone if you don't want a second MCP server in your config.

## Install

```bash
pipx install mcp-wandb
```

## Configure your client

### Claude Desktop / Claude Code

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "wandb": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/wandb/wandb-mcp-server", "wandb_mcp_server"],
      "env": { "WANDB_API_KEY": "<your-api-key>" }
    },
    "wandb-analyst": {
      "command": "mcp-wandb",
      "args": ["stdio"],
      "env": { "WANDB_API_KEY": "<your-api-key>" }
    }
  }
}
```

For the action tools (`launch_run`, `launch_sweep`, `add_tag`, `delete_run`), add `"--enable-actions"` after `"stdio"`. They're off by default.

### Cursor

`~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "wandb-analyst": {
      "command": "mcp-wandb",
      "args": ["stdio"],
      "env": { "WANDB_API_KEY": "<your-api-key>" }
    }
  }
}
```

## Example prompts

```
What projects do I have under entity "alice"?
Last week I ran a sweep on cifar10-sweep. Which hyperparam mattered most?
Compare my best 5 runs to baseline-001 and chart val_acc with smoothing 0.3.
Summarize sweep alice/cifar10-sweep/sweeps/abc123.
Find my best run on cifar10-sweep by val_loss in min mode.
Run a refined sweep with lr log-uniform [1e-4, 1e-3], 30 trials on gpu-spot.
```

See [`examples/`](examples/) for the full tool sequences.

## Transports

Local stdio is the default. There's also a Streamable HTTP transport for self-hosted setups:

```bash
mcp-wandb http --host 0.0.0.0 --port 8000
```

Bearer-token auth on HTTP; pass the W&B API key in the `Authorization` header. Full OAuth 2.1 support is planned for when W&B publishes their endpoints.

## Caching and observability

Both layers are opt-in:

```bash
# In-memory TTL+LRU; survives across requests inside one process
mcp-wandb stdio --cache

# Plus JSON snapshot persistence under <dir>; survives restarts
mcp-wandb stdio --cache --cache-dir ~/.cache/mcp-wandb

# OpenTelemetry spans on every tool call and W&B API operation
pip install 'mcp-wandb[telemetry]'
mcp-wandb stdio --telemetry  # uses standard OTEL_EXPORTER_OTLP_* env vars
```

When caching or telemetry is on, the `mcp-wandb://cache/stats` resource gives you a JSON snapshot of cache hits/misses, disk-snapshot count, and rolling W&B back-pressure (`rate_limited_ms`, retries), with a top-line `"ok" | "degraded" | "busy"` status.

## Safety

* Action tools are off by default. `--enable-actions` to turn them on; `launch_run` / `launch_sweep` / `delete_run` also require `confirm=true` on the call.
* API keys are never logged or written to disk by the server itself; `mcp-wandb auth store` is the explicit opt-in for keyring storage.
* Built-in rate limiter (60 req/min steady, 100 burst) so we don't hammer the W&B API.

## Local development

```bash
git clone <your-fork>
cd mcp-wandb
uv sync --extra dev --extra shap
uv run pytest
uv run ruff check .
uv run mypy src
```

Issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0. See [LICENSE](LICENSE).
