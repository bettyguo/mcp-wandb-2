"""Credential resolution for the W&B Public API.

Priority order:
    1. Explicit ``Authorization: Bearer <key>`` request header (HTTP transport).
    2. ``WANDB_API_KEY`` environment variable.
    3. ``~/.netrc`` entry for the W&B host.
    4. OS keyring under service ``mcp-wandb`` (Windows / macOS only).

API keys are never logged, never written to disk by this module, and never
echoed back through tool responses.
"""

from __future__ import annotations

import logging
import netrc
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

KEYRING_SERVICE = "mcp-wandb"
DEFAULT_WANDB_HOST = "api.wandb.ai"


class AuthError(RuntimeError):
    """Raised when no credentials can be resolved."""


@dataclass(frozen=True)
class Credentials:
    api_key: str
    source: str
    base_url: str | None = None


def resolve(bearer_token: str | None = None, base_url: str | None = None) -> Credentials:
    """Resolve W&B credentials, returning an opaque Credentials object.

    ``bearer_token`` short-circuits the resolution and is preferred for
    HTTP-transport requests where the agent passes a per-request key.
    """
    if bearer_token:
        return Credentials(api_key=bearer_token, source="bearer", base_url=base_url)

    env_key = os.environ.get("WANDB_API_KEY")
    if env_key:
        return Credentials(api_key=env_key, source="env", base_url=base_url)

    netrc_key = _read_netrc(base_url)
    if netrc_key:
        return Credentials(api_key=netrc_key, source="netrc", base_url=base_url)

    keyring_key = _read_keyring()
    if keyring_key:
        return Credentials(api_key=keyring_key, source="keyring", base_url=base_url)

    raise AuthError(
        "No W&B credentials found. Set WANDB_API_KEY, log in with `wandb login` "
        "(writes ~/.netrc), or store a key with `mcp-wandb auth store`."
    )


def _read_netrc(base_url: str | None) -> str | None:
    netrc_path = _netrc_path()
    if not netrc_path.exists():
        return None
    try:
        host = _host_from_url(base_url) or DEFAULT_WANDB_HOST
        auth = netrc.netrc(str(netrc_path)).authenticators(host)
    except (netrc.NetrcParseError, OSError) as exc:
        logger.debug("netrc parse failed: %s", exc)
        return None
    if not auth:
        return None
    _login, _account, password = auth
    return password if password else None


def _read_keyring() -> str | None:
    try:
        import keyring
    except ImportError:
        return None
    try:
        result = keyring.get_password(KEYRING_SERVICE, "default")
    except Exception as exc:
        logger.debug("keyring read failed: %s", exc)
        return None
    return str(result) if result is not None else None


def store_in_keyring(api_key: str) -> None:
    """Used by ``mcp-wandb auth store``. Not called from request paths."""
    import keyring

    keyring.set_password(KEYRING_SERVICE, "default", api_key)


def _netrc_path() -> Path:
    home = Path.home()
    if os.name == "nt":
        win = home / "_netrc"
        if win.exists():
            return win
    return home / ".netrc"


def _host_from_url(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    return parsed.hostname
