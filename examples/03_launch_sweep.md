# Launch a follow-up sweep

Continuation of the previous example. The agent has identified
`lr ∈ [1e-4, 1e-3]` as the high-signal region. Now kick off the refined
sweep.

## Setup

The server must be started with actions enabled:

```bash
mcp-wandb stdio --enable-actions
```

or

```bash
export MCP_WANDB_ENABLE_ACTIONS=1
```

## Prompt

```
Kick off a refined sweep: bayes method, val_acc the goal, lr log-uniform
[1e-4, 1e-3], batch_size from [64, 128, 256], optimizer adam only.
30 trials on my "gpu-spot" queue.
```

## Tool call

```python
launch_sweep(
    project="alice/cifar10-sweep",
    sweep_config={
        "method": "bayes",
        "metric": {"name": "val_acc", "goal": "maximize"},
        "parameters": {
            "lr": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-3},
            "batch_size": {"values": [64, 128, 256]},
            "optimizer": {"values": ["adam"]},
        },
    },
    n_runs=30,
    resource="gpu-spot",
    confirm=True,
)
```

## Expected response

> Sweep `refined-002` created and 30 trials queued on `gpu-spot`. Watch it at
> https://wandb.ai/alice/cifar10-sweep/sweeps/refined-002. Once the first
> ~10 trials finish, ask for `summarize_sweep('alice/cifar10-sweep/sweeps/refined-002')`
> for a fresh importance ranking.

## Safety notes

* Action tools are off by default. Without `--enable-actions` the call is
  rejected with a clear error.
* `confirm=true` is required regardless of `--enable-actions` on every
  launch / delete call: a second "yes really" layer so the agent can't
  fire on a misinterpreted user message.
