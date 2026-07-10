"""Best-effort failure details for workers that died before reporting.

Reads the tail of a dead worker's events JSONL (and its stderr log) to
build the human-readable error stored on the CodexInstance row: the last
rejected auto-approval, error notification, failed command, or visible
activity wins, in that order.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

from hitch.main.models import CodexInstance
from hitch.main.runtime import codex_pool

# Forwarded SDK notifications live under these namespaces. Hitch bridge frames
# such as approval/resolved do not prove worker/Codex progress after a retry.
_CODEX_STREAM_PROGRESS_PREFIXES = ("item/", "turn/", "thread/")


def _dead_worker_error(instance: CodexInstance) -> str:
    if instance.error:
        return instance.error
    detail = _dead_worker_last_event_detail(instance.events_path)
    log_detail = _dead_worker_log_detail(instance.pk)
    if detail:
        message = f"worker process exited before reporting completion; last event: {detail}"
        if log_detail:
            message = f"{message}; worker log: {log_detail}"
        return message
    if log_detail:
        return (
            "worker process exited before reporting completion; "
            f"worker log: {log_detail}"
        )
    return "worker process exited before reporting completion"

def _dead_worker_last_event_detail(events_path: str) -> str:
    if not events_path:
        return ""
    try:
        with Path(events_path).open(encoding="utf-8") as events_file:
            recent_lines = deque(events_file, maxlen=200)
    except OSError:
        return ""

    recent_events: list[tuple[tuple[int, int, int, int], dict[str, Any]]] = []
    for file_index, line in enumerate(recent_lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        recent_events.append((_event_order_key(event, file_index), event))

    auto_approval_started_detail = ""
    completed_auto_approval_keys: set[str] = set()
    failed_command_detail = ""
    fallback_detail = ""
    retryable_error_detail = ""
    newer_progress_seen = False
    for _order, event in sorted(recent_events, reverse=True):
        method = event.get("method")
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        if method == "item/autoApprovalReview/completed":
            completed_auto_approval_keys.add(_auto_approval_event_key(payload))
            detail = _auto_approval_event_detail(payload)
            if detail:
                return _with_retryable_error_context(detail, retryable_error_detail)
        if (
            method == "item/autoApprovalReview/started"
            and _auto_approval_event_key(payload) not in completed_auto_approval_keys
        ):
            detail = _auto_approval_event_detail(payload)
            if detail and not auto_approval_started_detail:
                auto_approval_started_detail = detail
        if method == "error":
            detail = _error_event_detail(payload)
            if detail:
                if _error_event_will_retry(payload):
                    if not retryable_error_detail and not newer_progress_seen:
                        retryable_error_detail = detail
                    continue
                return _with_retryable_error_context(detail, retryable_error_detail)
        if method == "item/completed" and not failed_command_detail:
            failed_command_detail = _failed_command_event_detail(payload)
        if not fallback_detail:
            fallback_detail = _last_visible_event_detail(method, payload)
        if _event_is_progress_after_retryable_error(method):
            newer_progress_seen = True
    return _with_retryable_error_context(
        failed_command_detail or auto_approval_started_detail or fallback_detail,
        retryable_error_detail,
    )

def _auto_approval_event_detail(payload: dict[str, Any]) -> str:
    review = payload.get("review")
    if not isinstance(review, dict):
        return ""
    status = str(review.get("status") or "").strip()
    if not status or status == "approved":
        return ""
    action = payload.get("action")
    command = ""
    if isinstance(action, dict):
        command = _compact_error_detail(str(action.get("command") or ""), limit=160)
    rationale = _compact_error_detail(str(review.get("rationale") or ""), limit=240)
    status_text = _approval_status_text(status)
    if command and rationale:
        return f"auto-approval review {status_text} for `{command}`: {rationale}"
    if command:
        return f"auto-approval review {status_text} for `{command}`"
    if rationale:
        return f"auto-approval review {status_text}: {rationale}"
    return f"auto-approval review {status_text}"

def _auto_approval_event_key(payload: dict[str, Any]) -> str:
    action = payload.get("action")
    if not isinstance(action, dict):
        return ""
    try:
        return json.dumps(action, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(action)

def _error_event_detail(payload: dict[str, Any]) -> str:
    error = payload.get("error")
    if not isinstance(error, dict):
        return ""
    message = _compact_error_detail(str(error.get("message") or ""), limit=160)
    details = _compact_error_detail(
        str(error.get("additionalDetails") or ""), limit=240
    )
    if message and details:
        return f"{message}: {details}"
    return message or details

def _error_event_will_retry(payload: dict[str, Any]) -> bool:
    return payload.get("willRetry") is True

def _with_retryable_error_context(detail: str, retryable_error_detail: str) -> str:
    if not retryable_error_detail:
        return detail
    if not detail:
        return f"retryable error: {retryable_error_detail}"
    return f"{detail}; last retryable error: {retryable_error_detail}"

def _event_is_progress_after_retryable_error(method: object) -> bool:
    """Only SDK stream events prove Codex recovered from an older retry."""
    if not isinstance(method, str):
        return False
    return method.startswith(_CODEX_STREAM_PROGRESS_PREFIXES)

def _event_order_key(event: dict[str, Any], file_index: int) -> tuple[int, int, int, int]:
    """Prefer SDK arrival order; fall back to append order for older logs."""
    recorded_at = event.get("recordedAt")
    if type(recorded_at) is not int:
        return (0, file_index, 0, file_index)
    event_seq = event.get("eventSeq")
    if type(event_seq) is not int:
        event_seq = 0
    return (1, recorded_at, event_seq, file_index)

def _failed_command_event_detail(payload: dict[str, Any]) -> str:
    item = payload.get("item")
    if not isinstance(item, dict):
        return ""
    if item.get("type") != "commandExecution" or item.get("status") != "failed":
        return ""
    command = _compact_error_detail(str(item.get("command") or ""), limit=200)
    if command:
        return f"command failed: `{command}`"
    return "command failed"

def _last_visible_event_detail(method: object, payload: dict[str, Any]) -> str:
    if method not in ("item/started", "item/completed"):
        return ""
    item = payload.get("item")
    if not isinstance(item, dict):
        return ""
    item_type = item.get("type")
    if item_type == "commandExecution":
        command = _compact_error_detail(str(item.get("command") or ""), limit=200)
        if not command:
            return ""
        status = str(item.get("status") or "").strip()
        if method == "item/started" or status == "inProgress":
            return f"command started: `{command}`"
        if status == "completed":
            return f"command completed: `{command}`"
        return f"command {status}: `{command}`" if status else f"command: `{command}`"
    if item_type == "agentMessage":
        text = _compact_error_detail(str(item.get("text") or ""), limit=240)
        if text:
            return f"agent last said: {text}"
    return ""

def _dead_worker_log_detail(instance_id: int) -> str:
    if not codex_pool.worker_log_io_enabled():
        return ""
    try:
        with codex_pool.worker_log_path(instance_id).open(encoding="utf-8", errors="replace") as log_file:
            recent_lines = deque(log_file, maxlen=20)
    except OSError:
        return ""
    tail = [line.strip() for line in recent_lines if line.strip()]
    if not tail:
        return ""
    return _compact_error_detail(" | ".join(tail[-3:]), limit=500)

def _approval_status_text(status: str) -> str:
    if status == "inProgress":
        return "in progress"
    if status == "timedOut":
        return "timed out"
    return status

def _compact_error_detail(value: str, *, limit: int) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."
