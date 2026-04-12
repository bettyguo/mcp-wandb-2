"""Credential-resolution tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mcp_wandb import auth


def test_resolve_prefers_bearer() -> None:
    creds = auth.resolve(bearer_token="bearer-key")
    assert creds.api_key == "bearer-key"
    assert creds.source == "bearer"


def test_resolve_uses_env_when_no_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "env-key")
    creds = auth.resolve()
    assert creds.api_key == "env-key"
    assert creds.source == "env"


def test_resolve_raises_when_nothing_available(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(auth, "_netrc_path", lambda: tmp_path / "missing-netrc")
    monkeypatch.setattr(auth, "_read_keyring", lambda: None)
    with pytest.raises(auth.AuthError):
        auth.resolve()


def test_resolve_reads_netrc(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(auth, "_read_keyring", lambda: None)
    netrc_path = tmp_path / ".netrc"
    netrc_path.write_text("machine api.wandb.ai login user password netrc-key\n")
    # netrc requires permissions on POSIX; skip strict check.
    if os.name != "nt":
        netrc_path.chmod(0o600)
    monkeypatch.setattr(auth, "_netrc_path", lambda: netrc_path)
    creds = auth.resolve()
    assert creds.api_key == "netrc-key"
    assert creds.source == "netrc"
