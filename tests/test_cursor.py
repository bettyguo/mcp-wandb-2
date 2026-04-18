"""Tests for the opaque cursor encoder."""

from __future__ import annotations

import pytest

from mcp_wandb import _cursor


def test_encode_decode_round_trip() -> None:
    q = {"config.lr": {"$lt": 0.1}}
    cursor = _cursor.encode(offset=50, query=q)
    assert _cursor.decode(cursor, q) == 50


def test_encode_is_url_safe() -> None:
    cursor = _cursor.encode(offset=10, query={"x": 1})
    # No `=` padding; only URL-safe base64 alphabet.
    assert "=" not in cursor
    assert all(c.isalnum() or c in "-_" for c in cursor)


def test_decode_rejects_different_query() -> None:
    cursor = _cursor.encode(offset=10, query={"x": 1})
    with pytest.raises(_cursor.CursorError):
        _cursor.decode(cursor, query={"x": 2})


def test_decode_rejects_empty_cursor() -> None:
    with pytest.raises(_cursor.CursorError):
        _cursor.decode("", query={"x": 1})


def test_decode_rejects_garbage_cursor() -> None:
    with pytest.raises(_cursor.CursorError):
        _cursor.decode("@@@notbase64@@@", query={"x": 1})


def test_decode_rejects_unsupported_version() -> None:
    import base64
    import json

    payload = base64.urlsafe_b64encode(json.dumps({"v": 99, "offset": 0, "q": "x"}).encode()).decode().rstrip("=")
    with pytest.raises(_cursor.CursorError):
        _cursor.decode(payload, query={"x": 1})


def test_hash_query_is_stable() -> None:
    q1 = {"a": 1, "b": 2}
    q2 = {"b": 2, "a": 1}
    assert _cursor.hash_query(q1) == _cursor.hash_query(q2)
