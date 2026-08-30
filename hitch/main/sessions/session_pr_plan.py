"""Derive a thread/session's registered PR, plan-mode, and auto-flag state."""

from __future__ import annotations

from typing import Any, NamedTuple

from hitch.main.models import CodexInstance, SessionMetadata, SessionPullRequest
from hitch.main.runtime import codex_pool, rollout
from hitch.main.runtime.rollout_state import _rollout_path_for
from hitch.main.runtime.sdk_values import string_value
from hitch.main.workflows import pr_tracking

_ROLLOUT_COLLABORATION_MODE_NOT_PROVIDED = object()


class _ThreadPlanModeState(NamedTuple):
    active: bool
    awaiting_approval: bool


def _thread_plan_mode_state(
    session_id: str,
    thread: Any,
    entries: list[dict[str, Any]],
    *,
    active_instance: CodexInstance | None = None,
    latest_collaboration_mode: str
    | None
    | object = _ROLLOUT_COLLABORATION_MODE_NOT_PROVIDED,
) -> _ThreadPlanModeState:
    """Return the Plan Mode state Codex recorded for this thread."""
    awaiting_approval = _entries_await_plan_approval(entries)
    latest_mode = (
        _latest_rollout_collaboration_mode(thread)
        if latest_collaboration_mode is _ROLLOUT_COLLABORATION_MODE_NOT_PROVIDED
        else latest_collaboration_mode
    )
    stored_plan_mode = (
        _latest_user_instance_ended_in_plan_mode(session_id)
        if latest_mode is None
        else False
    )
    active = (
        awaiting_approval
        or latest_mode == "plan"
        or stored_plan_mode
        or (active_instance is not None and active_instance.plan_mode)
    )
    return _ThreadPlanModeState(active=active, awaiting_approval=awaiting_approval)


def _latest_user_instance_ended_in_plan_mode(session_id: str) -> bool:
    latest = codex_pool.latest_for_thread(session_id)
    return bool(
        latest is not None
        and latest.purpose == CodexInstance.PURPOSE_USER
        and latest.workflow_id is None
        and latest.status == CodexInstance.STATUS_COMPLETED
        and latest.plan_mode
    )


def _latest_rollout_collaboration_mode(thread: Any) -> str | None:
    rollout_path = _rollout_path_for(thread)
    if rollout_path is None:
        return None
    return rollout.latest_collaboration_mode(rollout_path)


def _fix_pr_url_for_thread(session_id: str) -> str | None:
    registered_pr = pr_tracking.record_for_thread(session_id)
    return _registered_pr_url(registered_pr)


def _registered_pr_url(record: SessionPullRequest | None) -> str | None:
    handoff = pr_tracking.pr_handoff_for_record(record)
    url = string_value(handoff.get("url"))
    if url:
        return url
    repository = string_value(handoff.get("repository_full_name"))
    number = handoff.get("pr_number")
    repository_parts = repository.split("/")
    if (
        len(repository_parts) == 2
        and all(
            part and not any(char.isspace() for char in part)
            for part in repository_parts
        )
        and isinstance(number, int)
        and not isinstance(number, bool)
        and number > 0
    ):
        return f"https://github.com/{repository}/pull/{number}"
    return None


def _entries_await_plan_approval(entries: list[dict[str, Any]]) -> bool:
    return rollout.entries_await_plan_approval(entries)


def _pending_plan_entry(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    return rollout.pending_plan_entry(entries)


def _mark_pending_plan_actions(
    entries: list[dict[str, Any]], *, enabled: bool = True
) -> None:
    _clear_plan_actions(entries)
    if not enabled:
        return
    pending_plan = _pending_plan_entry(entries)
    if pending_plan is not None:
        pending_plan["show_plan_actions"] = True


def _clear_plan_actions(entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        if entry.get("kind") == "plan":
            entry["show_plan_actions"] = False
        elif entry.get("kind") == "intermediate":
            _clear_plan_actions(entry.get("items", []))


def _count_user_entries(entries: list[dict[str, Any]]) -> int:
    count = 0
    for entry in entries:
        if entry.get("kind") == "user":
            count += 1
        elif entry.get("kind") == "intermediate":
            count += _count_user_entries(entry.get("items", []))
    return count


def _auto_pr_enabled_for_session(session_id: str) -> bool:
    return SessionMetadata.objects.filter(
        thread_id=session_id, auto_pr_enabled=True
    ).exists()


def _auto_qa_enabled_for_session(session_id: str) -> bool:
    return SessionMetadata.objects.filter(
        thread_id=session_id, auto_qa_enabled=True
    ).exists()
