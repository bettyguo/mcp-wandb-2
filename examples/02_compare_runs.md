# Compare a sweep against a baseline

The end-to-end demo. Five tools chained, one prompt.

## Prompt

```
Last week I ran a sweep on cifar10-sweep. Compare my last 20 runs against
my baseline, tell me which hyperparam mattered most, and chart the best 5
vs. the baseline.
```

## Tool sequence

```python
# Slice the sweep to last week.
list_runs(project="alice/cifar10-sweep", since="7d", limit=20)

# Find the baseline.
find_baseline_runs(project="alice/cifar10-sweep", tag="baseline")
# returns 1 run; agent picks its id as baseline_id

# Structured diff: what did the sweep vary?
compare_runs(run_ids=[...20 sweep ids, baseline_id])

# Importance.
hyperparam_importance(
    run_ids=[...20 sweep ids],
    target_metric="val_acc",
    method="rf",
)

# Chart.
plot_metrics(
    run_ids=[<best 5 by val_acc>, baseline_id],
    metric="val_acc",
    smoothing=0.3,
)
```

## Expected response

> Over the last week your sweep produced 20 runs vs. the baseline
> (`baseline-001`, val_acc=0.881). Most config keys agree across the sweep;
> the search varied just `lr` (1e-5 to 0.1), `batch_size`
> (32, 64, 128, 256), and `optimizer` (adam/sgd).
>
> Learning rate was the dominant factor (RF importance 0.74). Optimizer
> mattered about a third as much (0.21); batch size a tenth (0.07). OOB
> R² = 0.78 so the ranking is reliable.
>
> Best 5 runs all clustered at lr in [1e-4, 3e-4]; adam beat sgd by ~1.2%
> on average. Suggested follow-up: narrow `lr` to [1e-4, 1e-3] log-uniform,
> 30 trials.
>
> [chart]

## Destructive follow-up

If the user follows with "delete the worst 5 runs from this sweep", the
agent calls `delete_run` per id with `confirm=true` *after* the operator
confirms in chat. The server must have been started with `--enable-actions`;
otherwise the call is rejected with an actionable error.
