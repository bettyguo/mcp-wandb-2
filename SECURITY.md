# Security policy

## Supported versions

Latest minor on the 0.x track.

## Reporting a vulnerability

Don't open a public GitHub issue. Email `security@<your-domain>` (placeholder
until forked). We aim to acknowledge within 3 business days and ship a fix
within 14 days. No paid bug bounty.

## In scope

* Credential leakage (W&B API keys, bearer tokens) via logs, error messages,
  or tool responses.
* OAuth flow bypass (once OAuth support lands).
* SSRF / command injection via the `launch_run` / `launch_sweep` config
  parameters.
* Path traversal in the optional disk cache.
* Dependency vulnerabilities surfaced by `pip-audit` against a release.

## Out of scope

* Issues in the official `wandb` SDK or `wandb/wandb-mcp-server`; report
  those upstream.
* Issues that require the operator to have intentionally enabled
  `MCP_WANDB_ENABLE_ACTIONS=1` and provided `confirm=true`. That's the
  documented opt-in behavior, not a bug.
