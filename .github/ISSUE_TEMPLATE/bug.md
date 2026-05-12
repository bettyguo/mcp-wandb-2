---
name: Bug report
about: Something doesn't behave the way the docs / tool description say it does.
labels: bug
---

## What happened

<!-- One sentence is enough. Include the tool name. -->

## Reproduction

<!-- The smallest snippet (CLI invocation, MCP prompt, or Python call) that
triggers the bug. -->

```python
# e.g.
from mcp_wandb.tools.analysis import find_best_run
find_best_run(client, project="entity/proj", metric="val_loss", mode="min")
```

## Expected vs. actual

| | Expected | Actual |
|---|---|---|
| Result | | |

## Environment

- `mcp-wandb` version: <!-- run `mcp-wandb version` -->
- Python version:
- Platform (macOS / Linux / Windows):
- `wandb` SDK version:
- Client (Claude Desktop / Cursor / Claude Code / stdio direct):

## Logs

<!-- Paste any `mcp_wandb.tool` or `mcp_wandb.api` log lines around the failure. -->

```text
```
