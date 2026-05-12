# Hosted deployment

Runs as a single-process container.

## Dockerfile

```dockerfile
FROM python:3.11-slim
RUN pip install --no-cache-dir 'mcp-wandb[shap]'
EXPOSE 8000
CMD ["mcp-wandb", "http", "--host", "0.0.0.0", "--port", "8000"]
```

## Environment variables

| Var | Purpose |
|---|---|
| `MCP_WANDB_ENABLE_ACTIONS` | Set to `1` to allow `launch_*`, `add_tag`, `delete_run` tools |
| `MCP_WANDB_RATE_LIMIT` | Steady-state requests per minute (default 60) |
| `MCP_WANDB_RATE_BURST` | Burst capacity (default 100) |
| `MCP_WANDB_CACHE` | `1` to enable the in-memory run cache |
| `MCP_WANDB_CACHE_DIR` | Path for the on-disk snapshot cache |
| `MCP_WANDB_OTEL_ENABLED` | `1` to emit OpenTelemetry spans |
| `WANDB_BASE_URL` | W&B host for Dedicated Cloud / On-Prem |
| `MCP_WANDB_LOG_LEVEL` | `DEBUG` / `INFO` / `WARN` / `ERROR` |
| `MCP_WANDB_LOG_FORMAT` | `text` (default) or `json` |

## Scaling

The process is stateless beyond the two opt-in caches. Scale horizontally,
but note the rate-limit budget is per-process: N pods means `N × 60 req/min`
against W&B. If that's close to the throttle, coordinate via an external
limiter (Redis token bucket, etc.).

We don't host a public demo. Use the official server's hosted endpoint
([mcp.withwandb.com](https://mcp.withwandb.com)) for the read primitives and
self-host `mcp-wandb` for the analytical surface.
