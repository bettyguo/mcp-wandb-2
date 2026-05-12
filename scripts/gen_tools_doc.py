"""Regenerate docs/tools.md from the registered FastMCP tool surface.

Run via:
    uv run python scripts/gen_tools_doc.py > docs/tools.md
"""

from __future__ import annotations

from textwrap import dedent

from mcp_wandb.server import build_app


def main() -> None:
    app = build_app()
    print("# Tools reference\n")
    print("Auto-generated. Do not edit by hand.\n")
    manager = getattr(app, "_tool_manager", None)
    tools = []
    if manager is not None and hasattr(manager, "list_tools"):
        try:
            tools = list(manager.list_tools())
        except TypeError:
            tools = []
    if not tools:
        tools = list(getattr(app, "tools", []))
    for tool in tools:
        name = getattr(tool, "name", "?")
        desc = getattr(tool, "description", "")
        print(f"## `{name}`\n")
        print(dedent(str(desc)).strip())
        print()


if __name__ == "__main__":
    main()
