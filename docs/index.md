# mcp-wandb

An [MCP](https://modelcontextprotocol.io) server for Weights & Biases that adds
analytical tools on top of the [official `wandb/wandb-mcp-server`](https://github.com/wandb/wandb-mcp-server).
Both can be installed side by side.

The official server exposes the W&B primitives: query runs, fetch run history,
list artifacts, generate reports. `mcp-wandb` adds analysis:

| Question | Tool |
|---|---|
| "Which hyperparam drove val_acc the most?" | [`hyperparam_importance`](tools.md) |
| "How did my sweep go?" | [`summarize_sweep`](tools.md) |
| "What's different between these runs?" | [`compare_runs`](tools.md) |
| "Which run was best?" | [`find_best_run`](tools.md) |
| "Plot these against my baseline." | [`plot_metrics`](tools.md) / [`plot_comparison`](tools.md) |
| "Help me design the next sweep." | [`recommend_next_sweep`](tools.md) |
| "Did anything regress recently?" | [`detect_regressions`](tools.md) |
| "Kick a sweep off." | [`launch_sweep`](tools.md) (gated) |

Plus thin wrappers for `list_projects`, `list_runs` (with `since="7d"` sugar),
`get_run`, and `query_runs` so the package can be used standalone if you don't
want two MCP servers in your config.

## Start here

* [Quickstart](quickstart.md)
* [Tools reference](tools.md)
* [Examples](examples/01_explore.md)
* [Relationship to the official server](relationship-to-official.md)
* [Auth](auth.md) and [hosted deployment](hosted.md)

## Safety

* Action tools (`launch_run`, `launch_sweep`, `add_tag`, `delete_run`) are off
  by default. `--enable-actions` turns them on; `launch_run`, `launch_sweep`,
  and `delete_run` also require `confirm=true` on the call.
* API keys are never logged or persisted by the server. `mcp-wandb auth store`
  is the explicit opt-in for keyring storage.
* Built-in rate limiter (60 req/min steady, 100 burst).

## License

Apache 2.0.
