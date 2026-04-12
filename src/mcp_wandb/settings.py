"""Runtime configuration for the mcp-wandb server.

Resolved once at startup. Lives in a single module so tests can swap a fresh
``Settings`` into the global slot without monkey-patching individual tools.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    """All runtime knobs for the server.

    ``enable_actions`` is the cardinal gate. When False, every tool in
    ``tools.actions`` raises before touching the W&B API.
    """

    enable_actions: bool = False
    rate_limit_per_min: int = 60
    rate_limit_burst: int = 100
    default_per_page: int = 50
    max_per_page: int = 200
    chart_max_bytes: int = 250_000
    chart_width: int = 1200
    chart_height: int = 675
    wandb_base_url: str | None = None
    user_agent: str = "mcp-wandb/0.1.0 (+https://github.com/your-org/mcp-wandb)"
    log_level: str = "INFO"
    log_format: str = "text"  # "text" for stdio; "json" for hosted HTTP
    # In-memory run-metadata cache. Off by default.
    cache_enabled: bool = False
    cache_ttl_seconds: float = 600.0
    cache_max_entries: int = 500
    # Optional disk-persistence layer. When set to a directory, client.run(path)
    # falls through to a JSON snapshot store after the in-memory miss. Tools
    # needing live methods (history / update / delete) bypass via run_live().
    cache_dir: str | None = None
    # OpenTelemetry spans. Standard OTEL_EXPORTER_OTLP_* env vars configure
    # the exporter when enabled.
    telemetry_enabled: bool = False
    allowed_oauth_audiences: tuple[str, ...] = field(default_factory=tuple)


_settings: Settings = Settings()


def get_settings() -> Settings:
    """Return the active Settings instance."""
    return _settings


def set_settings(new: Settings) -> None:
    """Replace the active Settings (called once at CLI bootstrap; tests use ``override``)."""
    global _settings
    _settings = new


def settings_from_env() -> Settings:
    """Build a Settings instance from environment variables.

    Recognized variables:
        MCP_WANDB_ENABLE_ACTIONS    "1" / "true" to enable action tools
        MCP_WANDB_RATE_LIMIT        steady-state requests per minute
        MCP_WANDB_RATE_BURST        max burst requests
        WANDB_BASE_URL              W&B host (for Dedicated Cloud / On-Prem)
        MCP_WANDB_LOG_LEVEL         DEBUG / INFO / WARN / ERROR
    """

    def _bool(name: str, default: bool) -> bool:
        v = os.environ.get(name)
        if v is None:
            return default
        return v.strip().lower() in {"1", "true", "yes", "on"}

    def _int(name: str, default: int) -> int:
        v = os.environ.get(name)
        return int(v) if v else default

    def _float(name: str, default: float) -> float:
        v = os.environ.get(name)
        return float(v) if v else default

    return Settings(
        enable_actions=_bool("MCP_WANDB_ENABLE_ACTIONS", False),
        rate_limit_per_min=_int("MCP_WANDB_RATE_LIMIT", 60),
        rate_limit_burst=_int("MCP_WANDB_RATE_BURST", 100),
        wandb_base_url=os.environ.get("WANDB_BASE_URL"),
        log_level=os.environ.get("MCP_WANDB_LOG_LEVEL", "INFO"),
        log_format=os.environ.get("MCP_WANDB_LOG_FORMAT", "text"),
        cache_enabled=_bool("MCP_WANDB_CACHE", False),
        cache_ttl_seconds=_float("MCP_WANDB_CACHE_TTL", 600.0),
        cache_max_entries=_int("MCP_WANDB_CACHE_MAX_ENTRIES", 500),
        cache_dir=os.environ.get("MCP_WANDB_CACHE_DIR"),
        telemetry_enabled=_bool("MCP_WANDB_OTEL_ENABLED", False),
    )
