"""CLI entry point.

Usage:
    mcp-wandb stdio
    mcp-wandb http --host 127.0.0.1 --port 8000 [--enable-actions]
    mcp-wandb auth store
    mcp-wandb tools list
"""

from __future__ import annotations

import getpass
import logging
import sys
from typing import Any

import typer

from . import __version__
from .auth import store_in_keyring
from .settings import Settings, set_settings, settings_from_env

app = typer.Typer(
    name="mcp-wandb",
    help="Analytical MCP companion for Weights & Biases.",
    no_args_is_help=True,
    add_completion=False,
)
auth_app = typer.Typer(help="Manage W&B credentials stored by mcp-wandb.")
tools_app = typer.Typer(help="Inspect the registered tool surface.")
app.add_typer(auth_app, name="auth")
app.add_typer(tools_app, name="tools")


def _init_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


@app.callback()
def _root(
    enable_actions: bool = typer.Option(
        False,
        "--enable-actions",
        envvar="MCP_WANDB_ENABLE_ACTIONS",
        help="Enable action tools (launch_run/launch_sweep/add_tag/delete_run). OFF by default.",
    ),
    log_level: str = typer.Option(
        "INFO", "--log-level", envvar="MCP_WANDB_LOG_LEVEL", help="Logging level."
    ),
    wandb_base_url: str | None = typer.Option(
        None, "--wandb-base-url", envvar="WANDB_BASE_URL", help="W&B base URL (Dedicated Cloud / On-Prem)."
    ),
    cache: bool = typer.Option(
        False,
        "--cache",
        envvar="MCP_WANDB_CACHE",
        help="Enable the in-memory run-metadata cache (TTL+LRU). OFF by default.",
    ),
    cache_ttl: float = typer.Option(
        600.0,
        "--cache-ttl",
        envvar="MCP_WANDB_CACHE_TTL",
        help="Cache entry lifetime in seconds (only used when --cache is set).",
    ),
    cache_max_entries: int = typer.Option(
        500,
        "--cache-max-entries",
        envvar="MCP_WANDB_CACHE_MAX_ENTRIES",
        help="Cache capacity; LRU-evicts beyond this (only used when --cache is set).",
    ),
    cache_dir: str | None = typer.Option(
        None,
        "--cache-dir",
        envvar="MCP_WANDB_CACHE_DIR",
        help="Persist run snapshots as JSON under this directory; survives "
        "restarts. Off by default. Tools needing run.history/update/delete "
        "still bypass the snapshot cache automatically.",
    ),
    telemetry: bool = typer.Option(
        False,
        "--telemetry",
        envvar="MCP_WANDB_OTEL_ENABLED",
        help="Enable OpenTelemetry spans (requires `mcp-wandb[telemetry]`). OFF by default.",
    ),
) -> None:
    base = settings_from_env()
    set_settings(
        Settings(
            enable_actions=enable_actions or base.enable_actions,
            rate_limit_per_min=base.rate_limit_per_min,
            rate_limit_burst=base.rate_limit_burst,
            default_per_page=base.default_per_page,
            max_per_page=base.max_per_page,
            chart_max_bytes=base.chart_max_bytes,
            chart_width=base.chart_width,
            chart_height=base.chart_height,
            wandb_base_url=wandb_base_url or base.wandb_base_url,
            user_agent=base.user_agent,
            log_level=log_level,
            cache_enabled=cache or base.cache_enabled,
            cache_ttl_seconds=cache_ttl if cache_ttl != 600.0 else base.cache_ttl_seconds,
            cache_max_entries=cache_max_entries if cache_max_entries != 500 else base.cache_max_entries,
            cache_dir=cache_dir or base.cache_dir,
            telemetry_enabled=telemetry or base.telemetry_enabled,
            allowed_oauth_audiences=base.allowed_oauth_audiences,
        )
    )
    _init_logging(log_level)


@app.command()
def stdio() -> None:
    """Run the server over stdio. Use this in Claude Desktop / Cursor / Claude Code configs."""
    from .server import serve_stdio

    serve_stdio()


@app.command()
def http(
    host: str = typer.Option("127.0.0.1", help="Host interface to bind."),
    port: int = typer.Option(8000, help="Port to listen on."),
) -> None:
    """Run the server over Streamable HTTP (for hosted / remote-MCP setups)."""
    from .server import serve_http

    serve_http(host=host, port=port)


@app.command()
def version() -> None:
    """Print the installed mcp-wandb version."""
    typer.echo(__version__)


@auth_app.command("store")
def auth_store() -> None:
    """Store a W&B API key in the OS keyring under service 'mcp-wandb'."""
    key = getpass.getpass("Paste your W&B API key (input hidden): ").strip()
    if not key:
        typer.echo("No key provided.", err=True)
        raise typer.Exit(code=1)
    try:
        store_in_keyring(key)
    except Exception as exc:
        typer.echo(f"Failed to store key in keyring: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo("Stored.")


TOOL_CEILING = 17
"""Maximum tool count we expect to be registered. Tools added beyond this
should usually replace an existing one rather than expand the surface;
LLM tool-selection accuracy degrades as the menu grows."""


@tools_app.command("list")
def tools_list(
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit non-zero if the registered tool count != TOOL_CEILING. "
        "Useful in CI to guard against drift past the ceiling.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit a JSON object {count, ceiling, tools: [{name, description}]} "
        "for programmatic consumers.",
    ),
) -> None:
    """Print the name + first sentence of each registered tool's description.

    Prefixes the list with the tool count so contributors can verify the
    tool ceiling at a glance. Use ``--strict`` in CI to fail on drift, or
    ``--json`` to consume the output programmatically.
    """
    import json as _json

    from .server import build_app

    mcp = build_app()
    tools = getattr(mcp, "_tool_manager", None)
    registered: list[Any] = []
    if tools is not None and hasattr(tools, "list_tools"):
        try:
            registered = list(tools.list_tools())
        except TypeError:
            registered = []
    if not registered:
        registered = list(getattr(mcp, "tools", []))

    count = len(registered)
    entries: list[dict[str, str]] = []
    for tool in registered:
        name = getattr(tool, "name", str(tool))
        desc_full = getattr(tool, "description", "") or ""
        entries.append({"name": str(name), "description": str(desc_full)})

    if json_output:
        typer.echo(
            _json.dumps(
                {"count": count, "ceiling": TOOL_CEILING, "tools": entries},
                indent=2,
                sort_keys=False,
            )
        )
    else:
        typer.echo(f"# {count} tools registered (ceiling: {TOOL_CEILING})")
        for entry in entries:
            first_sentence = entry["description"].split(". ")[0]
            typer.echo(f"  {entry['name']}\t{first_sentence}")

    if strict and count != TOOL_CEILING:
        typer.echo(
            f"FAIL: tool count {count} != ceiling {TOOL_CEILING}. "
            "Either restore the missing tool or open an ADR to raise the ceiling.",
            err=True,
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
