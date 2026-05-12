# Explore a project

A first-touch walkthrough. Goal: figure out what's in a W&B project without
opening the web UI.

## Prompt

```
What projects do I have under entity "alice"?
Pick the most recently active and show me its last 10 runs.
For the run with the highest val_acc, give me its full config.
```

## Tool sequence

```python
list_projects(entity="alice")
# returns list; agent picks "cifar10-sweep" (highest last_active_at)

list_runs(project="alice/cifar10-sweep", limit=10, order_by="-created_at")
# 10 most recent runs as RunSummary

find_best_run(project="alice/cifar10-sweep", metric="val_acc", mode="max")
# returns the top run with its config inline
```

## Expected response

> Your most active project is `cifar10-sweep` (last activity 18 minutes ago).
> The latest 10 runs are from a bayesian sweep that started yesterday at 6 PM.
> The best so far is `trial-019` with val_acc=0.943, learning rate 2.7e-4,
> batch size 128, optimizer adam. The next 3 are within 0.5%; the worst is
> 7% behind.
