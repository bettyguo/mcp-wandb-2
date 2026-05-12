# Quickstart

## 1. Get a W&B API key

Copy a key from [wandb.ai/authorize](https://wandb.ai/authorize).

## 2. Install

```bash
pipx install mcp-wandb
```

For SHAP-based hyperparam importance:

```bash
pipx install "mcp-wandb[shap]"
```

## 3. Configure your client

See the snippets in the [README](https://github.com/your-org/mcp-wandb#configure-your-client) for Claude Desktop, Cursor, and Claude Code.

## 4. Try a prompt

```
Last week I ran a sweep on <your-project>. Which hyperparam mattered most?
Compare my best 5 runs against my baseline.
```

The agent will chain `list_runs`, `find_baseline_runs`, `compare_runs`,
`hyperparam_importance`, and `plot_metrics`. Expect ~20 s wall time on a
30-run sweep.

## 5. Enable actions

The launch / delete / tag tools are gated:

```bash
mcp-wandb stdio --enable-actions
```

Or add `--enable-actions` to `args` in your MCP client config. Calls that
mutate state still require `confirm=true`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No W&B credentials found.` | Run `wandb login`, set `WANDB_API_KEY`, or `mcp-wandb auth store`. |
| `Actions are disabled.` | Add `--enable-actions`. |
| `Confirmation required.` | Retry with `confirm=true`. |
| Chart > 250 KB | Reduce `max_points` or run count. |
| Rate-limited by W&B | The client retries with backoff; lower `MCP_WANDB_RATE_LIMIT` if persistent. |
