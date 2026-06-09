"""Encode and decode the opaque pagination cursor for the session index.

The session list pages by ``(updated_at, thread_id)`` sort key. That key is
serialized into a URL-safe ``idx:<base64-json>`` token so a "load more" link can
resume exactly where the previous page stopped, tolerating malformed or
out-of-range cursors by decoding to ``None``. The codec is dependency-free; the
view layer builds the sort keys and feeds them through here.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class _IndexCursor:
    updated_at: float
    thread_id: str
    exact_updated_at: bool = False

    @property
    def sort_key(self) -> tuple[float, str]:
        return (self.updated_at, self.thread_id)


def _is_index_cursor(cursor: str) -> bool:
    return cursor.startswith("idx:")


def _index_cursor_for_sort_key(
    sort_key: tuple[float, str], *, exact_updated_at: bool = False
) -> str:
    payload = {
        "updated_at": sort_key[0],
        "id": sort_key[1],
    }
    if exact_updated_at:
        payload["updated_at_precision"] = "exact"
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode()
    ).decode()
    return f"idx:{encoded}"


def _index_cursor_sort_key(cursor: str) -> tuple[float, str] | None:
    parsed = _index_cursor(cursor)
    return parsed.sort_key if parsed is not None else None


def _index_cursor(cursor: str) -> _IndexCursor | None:
    if not _is_index_cursor(cursor):
        return None
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor[4:].encode()).decode())
    except (ValueError, binascii.Error, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    updated_at = payload.get("updated_at")
    thread_id = payload.get("id")
    if not isinstance(updated_at, int | float) or not isinstance(thread_id, str):
        return None
    updated_at_float = _index_cursor_updated_at(updated_at)
    if updated_at_float is None:
        return None
    return _IndexCursor(
        updated_at=updated_at_float,
        thread_id=thread_id,
        exact_updated_at=payload.get("updated_at_precision") == "exact",
    )


def _index_cursor_updated_at(updated_at: int | float) -> float | None:
    updated_at_float = float(updated_at)
    if not math.isfinite(updated_at_float):
        return None
    try:
        datetime.fromtimestamp(updated_at_float, UTC)
    except (OverflowError, OSError, ValueError):
        return None
    return updated_at_float
