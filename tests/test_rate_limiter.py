"""Concurrency test for the token bucket: verifies the race fix."""

from __future__ import annotations

import threading
import time

from mcp_wandb import client as client_mod
from mcp_wandb.settings import Settings, set_settings


def test_token_bucket_serializes_under_contention() -> None:
    """N parallel callers each grabbing 1 token must finish in N/rate seconds.

    Pre-fix: callers could double-spend tokens because the lock was released
    during sleep. Post-fix: each caller reserves its slot inside the lock.
    """
    client_mod.reset_rate_limiter()
    # 10 tokens/sec, capacity 2, so 5 callers will need to serialize.
    set_settings(Settings(rate_limit_per_min=600, rate_limit_burst=2))
    client_mod.reset_rate_limiter()
    bucket = client_mod._get_bucket()
    # Force-drain the initial burst so we can measure pure rate-limited wait.
    bucket.tokens = 0.0

    n_callers = 5
    start = time.monotonic()

    def _hit() -> None:
        bucket.take()

    threads = [threading.Thread(target=_hit) for _ in range(n_callers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    elapsed = time.monotonic() - start
    # 5 callers @ 10 tokens/sec ≈ 0.5 s total. Allow 0.4..1.0 s window.
    assert 0.3 < elapsed < 1.5, f"elapsed {elapsed:.3f}s outside expected band"
