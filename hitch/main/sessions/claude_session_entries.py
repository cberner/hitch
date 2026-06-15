"""Reconstruct session-detail entries from a Claude worker events file.

Codex sessions render their initial (server-side) transcript from a Codex
rollout file via :mod:`hitch.main.rollout`. Claude sessions have no rollout --
their transcript lives in the worker's JSONL events file in the Codex
*notification* schema (``item/completed`` etc.). This module maps those events
onto the same entry dicts the session template consumes (``kind`` of ``user`` /
``agent`` / ``plan`` and ``rollout._tool_call`` shapes), so the shared session
view can render a completed Claude session without resuming a Codex thread.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from hitch.main.runtime.rollout import _tool_call
from hitch.main.runtime.sdk_values import is_nonbool_int

logger = logging.getLogger(__name__)

_ITEM_COMPLETED = "item/completed"

_TOOL_LABELS = {
    "reasoning": "Reasoning",
    "commandExecution": "Command",
    "fileChange": "File change",
    "mcpToolCall": "MCP tool call",
    "webSearch": "Web search",
    "dynamicToolCall": "Tool call",
}


def session_entries(events_path: str | Path) -> list[dict[str, Any]]:
    """Return ordered session-detail entries parsed from ``events_path``."""
    entries: list[dict[str, Any]] = []
    try:
        with Path(events_path).open("r", encoding="utf-8") as fh:
            for raw in fh:
                entry = _entry_from_event_line(raw)
                if entry is not None:
                    entries.append(entry)
    except FileNotFoundError:
        return entries
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("failed to read Claude events %s: %s", events_path, exc)
    return entries


def _entry_from_event_line(raw: str) -> dict[str, Any] | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        event = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(event, dict) or event.get("method") != _ITEM_COMPLETED:
        return None
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return None
    item = payload.get("item")
    if not isinstance(item, dict):
        return None
    timestamp = _timestamp_seconds(event.get("recordedAt"))
    return _entry_from_item(item, timestamp)


def _entry_from_item(item: dict[str, Any], timestamp: int | None) -> dict[str, Any] | None:
    item_type = item.get("type")
    text = item.get("text") if isinstance(item.get("text"), str) else ""
    if item_type == "userMessage":
        return {"kind": "user", "text": text, "timestamp": timestamp} if text else None
    if item_type == "agentMessage":
        if not text:
            return None
        phase = item.get("phase")
        return {
            "kind": "agent",
            "text": text,
            "timestamp": timestamp,
            "phase": phase if isinstance(phase, str) else None,
        }
    if item_type == "plan":
        return {"kind": "plan", "text": text, "timestamp": timestamp} if text else None
    if item_type in _TOOL_LABELS:
        return _tool_call(
            item_type,
            _TOOL_LABELS[item_type],
            _tool_detail(item_type, item),
            _tool_status(item.get("status")),
            timestamp,
        )
    return None


def _tool_detail(item_type: str, item: dict[str, Any]) -> str:
    if item_type == "commandExecution":
        return _str(item.get("command"))
    if item_type == "fileChange":
        changes = item.get("changes")
        if isinstance(changes, list):
            paths = [c.get("path") for c in changes if isinstance(c, dict)]
            return ", ".join(p for p in paths if isinstance(p, str) and p)
        return ""
    if item_type == "mcpToolCall":
        return f"{_str(item.get('server'))} / {_str(item.get('tool'))}"
    if item_type == "webSearch":
        return _str(item.get("query"))
    if item_type == "reasoning":
        return _str(item.get("text")).split("\n", 1)[0]
    if item_type == "dynamicToolCall":
        return _str(item.get("tool"))
    return ""


def _tool_status(status: Any) -> str | None:
    # ``completed`` is the unremarkable success state; surface only the others.
    if isinstance(status, str) and status and status != "completed":
        return status
    return None


def _timestamp_seconds(recorded_at: Any) -> int | None:
    # Worker ``recordedAt`` is microseconds since the epoch (time_ns // 1000).
    if is_nonbool_int(recorded_at):
        return recorded_at // 1_000_000
    return None


def _str(value: Any) -> str:
    return value if isinstance(value, str) else ""
