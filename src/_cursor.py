"""Versioned opaque cursors for paginated tool responses.

v1 format::

    base64url(json({"v": 1, "offset": int, "q": <16-hex-query-hash>}))

* ``offset``: number of items the client has already received.
* ``q``: short hash of the filter that produced this cursor. We refuse to
  resume against a different filter so an agent rotating filters never gets
  silently-incorrect pages.

The format is *opaque*; tools must never expose the encoding to the agent.
The version byte gives us headroom to switch to a keyset cursor in v2
without breaking existing in-flight conversations.
"""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any


class CursorError(ValueError):
    """Raised when a cursor is malformed, expired, or filter-mismatched."""


def hash_query(query: dict[str, Any]) -> str:
    canonical = json.dumps(query, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def encode(offset: int, query: dict[str, Any]) -> str:
    payload = {"v": 1, "offset": int(offset), "q": hash_query(query)}
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode(cursor: str, query: dict[str, Any]) -> int:
    if not cursor:
        raise CursorError("Empty cursor.")
    padded = cursor + "=" * ((4 - len(cursor) % 4) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
        payload = json.loads(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        raise CursorError(f"Malformed cursor: {exc}") from exc
    if payload.get("v") != 1:
        raise CursorError(
            f"Unsupported cursor version {payload.get('v')!r}; "
            "discard the cursor and re-query without one."
        )
    if payload.get("q") != hash_query(query):
        raise CursorError(
            "Cursor does not match the supplied mongo_query. Cursors are "
            "scoped to the filter that produced them; if the filter changed, "
            "drop the cursor and start a fresh query."
        )
    offset = payload.get("offset")
    if not isinstance(offset, int) or offset < 0:
        raise CursorError("Cursor offset is missing or invalid.")
    return offset
