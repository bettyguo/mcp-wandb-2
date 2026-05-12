# Auth

## Local stdio (default)

API key resolved in this order:

1. `WANDB_API_KEY` environment variable
2. `~/.netrc` entry for the W&B host (created by `wandb login`)
3. OS keyring under service `mcp-wandb` (after `mcp-wandb auth store`)

The key is never logged or persisted by `mcp-wandb` unless you ran `auth store`.

## Local HTTP

Pass the key in `Authorization: Bearer …` (same shape as the W&B hosted
server). A contextvar-scoped middleware reads it per-request so the
`WandbClient` instance never crosses request boundaries.

```http
POST /mcp HTTP/1.1
Authorization: Bearer <wandb-api-key>
Accept: application/json, text/event-stream
```

## Hosted HTTP

Bearer auth same as local HTTP. OAuth 2.1 (PKCE, refresh tokens, encrypted
storage) will land once W&B publishes their OAuth endpoints.

## Dedicated Cloud / On-Prem

Set `WANDB_BASE_URL=https://wandb.your-company.com` or pass `--wandb-base-url`.
The base URL flows into the `wandb.Api()` overrides and the netrc host lookup
so credentials resolve correctly for non-SaaS deployments.
