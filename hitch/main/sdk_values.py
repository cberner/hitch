"""Small, dependency-free coercion helpers for reading Codex SDK values.

Codex thread items arrive either as pydantic models (with attributes and a
``model_dump``) or as plain JSON dicts from a rollout file, so the view and
rendering layers need a handful of tolerant accessors that work on both shapes.
These live here, apart from the view handlers, because several modules read SDK
values and none of them should have to import the others to do so.
"""

from __future__ import annotations

from typing import Any


def _string_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return raw.strip() if isinstance(raw, str) else ""


def _value_for(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _plain_sdk_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain_sdk_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_plain_sdk_value(child) for child in value]
    if isinstance(value, tuple):
        return [_plain_sdk_value(child) for child in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return _sdk_model_dump_value(value)


def _sdk_model_dump_value(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if not callable(model_dump):
        return value
    try:
        dumped = model_dump(by_alias=True)
    except TypeError:
        dumped = model_dump()
    return _plain_sdk_value(dumped) if dumped is not value else value


def _sequence_value(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0
