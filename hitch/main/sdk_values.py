"""Small, dependency-free coercion helpers for reading Codex SDK values.

Codex thread items arrive either as pydantic models (with attributes and a
``model_dump``) or as plain JSON dicts from a rollout file, so the view and
rendering layers need a handful of tolerant accessors that work on both shapes.
These live here, apart from the view handlers, because several modules read SDK
values and none of them should have to import the others to do so.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


def string_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return raw.strip() if isinstance(raw, str) else ""


def value_for(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def plain_sdk_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: plain_sdk_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [plain_sdk_value(child) for child in value]
    if isinstance(value, tuple):
        return [plain_sdk_value(child) for child in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return sdk_model_dump_value(value)


def sdk_model_dump_value(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if not callable(model_dump):
        return value
    try:
        dumped = model_dump(by_alias=True)
    except TypeError:
        dumped = model_dump()
    return plain_sdk_value(dumped) if dumped is not value else value


def sequence_value(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def int_value(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def string_from_any(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.isdecimal():
        parsed = int(value)
        return parsed if parsed > 0 else None
    return None


def datetime_value(value: Any) -> datetime | None:
    return value if isinstance(value, datetime) else None


def updated_at_seconds(updated_at: Any) -> float | None:
    if isinstance(updated_at, bool):
        return None
    if isinstance(updated_at, int | float):
        return float(updated_at)
    if isinstance(updated_at, datetime):
        return updated_at.timestamp()
    return None


def latest_updated_at(*values: Any) -> Any:
    latest: Any = None
    latest_seconds: float | None = None
    for value in values:
        seconds = updated_at_seconds(value)
        if seconds is None:
            continue
        if latest_seconds is None or seconds > latest_seconds:
            latest = value
            latest_seconds = seconds
    if isinstance(latest, datetime):
        return int(latest.timestamp())
    return latest if latest is not None else 0
