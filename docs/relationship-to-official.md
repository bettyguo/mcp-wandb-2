# Relationship to `wandb/wandb-mcp-server`

`mcp-wandb` complements the official W&B MCP server. Not a fork, not a
re-implementation. Both can be installed side by side and the agent picks
the right tool per call.

## What goes where

| Capability | Official | `mcp-wandb` |
|---|---|---|
| Raw run / project query | `query_wandb_tool` | thin wrapper for standalone use |
| Run history time-series | `get_run_history_tool` | via `plot_metrics` |
| Weave LLM traces | yes | out of scope |
| Artifact / registry | 5 tools | defer to official |
| W&B docs / Support Bot | `search_wandb_docs_tool` | defer to official |
| Create W&B report | `create_wandb_report_tool` | defer to official |
| Log analysis back to W&B | `log_analysis_to_wandb` | defer to official |
| Hyperparam importance | n/a | `hyperparam_importance` (RF / SHAP) |
| Sweep narrative summary | n/a | `summarize_sweep` |
| Structured run diff | n/a | `compare_runs` |
| Best/baseline shortcuts | n/a | `find_best_run`, `find_baseline_runs` |
| Inline PNG line charts | n/a | `plot_metrics`, `plot_comparison` |
| Sweep recommendation | n/a | `recommend_next_sweep` |
| Regression detection | n/a | `detect_regressions` |
| W&B Launch trigger | n/a | `launch_run`, `launch_sweep` (gated) |
| Run mutation | n/a | `add_tag`, `delete_run` (gated) |

## Tool selection

Tools are picked by description. Each tool here leads with the verb plus the
typical question (e.g. "Renders a line chart…", "Compares two or more runs…")
so the LLM routes correctly. If the user asks for a W&B report, the official's
`create_wandb_report_tool` wins; if they ask for an inline chart, ours does.

## Why not upstream?

Patching ~17 new tools into the official server is a multi-quarter review;
shipping a separate package gets the analysis in front of users faster. If
W&B wants to absorb the analytical surface later, that's the happy path.
