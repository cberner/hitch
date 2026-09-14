"""Session-list stage derivation and cached-stage persistence."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import partial
from typing import Any

from django.db.models import Q

from hitch.main.models import (
    ApprovalRequest,
    CodexInstance,
    UserInputRequest,
)
from hitch.main.runtime.rollout_state import (
    _rollout_file_state_from_value,
)
from hitch.main.runtime.sdk_values import is_nonbool_int, string_value
from hitch.main.sequences import unique_nonempty
from hitch.main.sessions import session_stage, stage_resolution
from hitch.main.workflows import pr_stage, pr_tracking


def _attach_session_stage_context(sessions: list[dict[str, Any]]) -> None:
    thread_ids = [
        session["id"] for session in sessions if isinstance(session.get("id"), str)
    ]
    registered_prs_by_thread_id = pr_tracking.records_by_thread_id(thread_ids)
    active_instances_by_thread_id = _active_instances_by_thread_id(thread_ids)
    waiting_thread_ids = _thread_ids_awaiting_input(thread_ids)
    for session in sessions:
        session_id = session.get("id")
        if not isinstance(session_id, str):
            continue
        rollout_state = _rollout_file_state_from_value(session.get("codex_path"))
        stored_pr = registered_prs_by_thread_id.get(session_id)
        active_instance = active_instances_by_thread_id.get(session_id)
        pr = stage_resolution.pr_display_state(stored_pr, active_instance)
        resolution = stage_resolution.resolve_stage(
            entries=(), history_loader=partial(
                stage_resolution.read_stage_history,
                rollout_state.path if rollout_state is not None else None,
                has_activity=bool(session.get("has_activity")),
            ),
            history_complete=True, pr=pr, active_instance=active_instance,
            awaiting_user_input=session_id in waiting_thread_ids,
            cache=stage_resolution.StageCache(
                string_value(session.get("stage_cache_key")), session.get("stage_cache_mtime_ns", 0),
            ),
            source_mtime_ns=rollout_state.mtime_ns if rollout_state is not None else None,
        )
        stage = resolution.stage
        session["stage"] = _session_list_stage_context(
            stage, pr_snapshot=pr.snapshot,
            executing=stage == session_stage.IMPLEMENTATION and active_instance is not None,
        )
        if resolution.should_persist:
            pr_stage._update_cached_stage_best_effort(
                session_id, stage, rollout_state.mtime_ns if rollout_state is not None else 0,
            )


def _session_list_stage_context(
    stage: session_stage.SessionStage,
    *,
    pr_snapshot: Mapping[str, Any] | None = None,
    executing: bool = False,
) -> dict[str, Any]:
    context: dict[str, Any] = dict(stage.as_context())
    if stage.key == session_stage.IMPLEMENTATION.key:
        if executing:
            context["executing"] = True
        else:
            context["tone"] = "idle"
    if stage.key != session_stage.PR.key:
        return context
    pr_number = _pr_number_from_snapshot(pr_snapshot)
    if pr_number is not None:
        context["label"] = f"{stage.label} #{pr_number}"
    return context


def _pr_number_from_snapshot(snapshot: Mapping[str, Any] | None) -> int | None:
    if not snapshot:
        return None
    number = snapshot.get("pr_number")
    if is_nonbool_int(number) and number > 0:
        return number
    return None


def _thread_ids_awaiting_input(thread_ids: Iterable[str]) -> set[str]:
    ids = unique_nonempty(thread_ids)
    if not ids:
        return set()
    active_statuses = CodexInstance.ACTIVE_STATUSES
    direct_input_thread_ids = UserInputRequest.objects.filter(
        Q(params__isBlocking__isnull=True) | ~Q(params__isBlocking=False),
        response__isnull=True,
        instance__thread_id__in=ids,
        instance__status__in=active_statuses,
    ).values_list("instance__thread_id", flat=True)
    direct_approval_thread_ids = ApprovalRequest.objects.filter(
        decision=ApprovalRequest.DECISION_PENDING,
        instance__thread_id__in=ids,
        instance__status__in=active_statuses,
    ).values_list("instance__thread_id", flat=True)
    waiting_thread_ids: set[str] = set()
    for thread_ids_result in (
        direct_input_thread_ids,
        direct_approval_thread_ids,
    ):
        for thread_id in thread_ids_result:
            if isinstance(thread_id, str) and thread_id:
                waiting_thread_ids.add(thread_id)
    return waiting_thread_ids


def _active_instances_by_thread_id(
    thread_ids: Iterable[str],
) -> dict[str, CodexInstance]:
    ids = unique_nonempty(thread_ids)
    if not ids:
        return {}
    active_instances = (
        CodexInstance.objects.filter(
            thread_id__in=ids,
            status__in=CodexInstance.ACTIVE_STATUSES,
        )
        .order_by("thread_id", "-started_at", "-pk")
    )
    by_thread_id: dict[str, CodexInstance] = {}
    for instance in active_instances:
        by_thread_id.setdefault(instance.thread_id, instance)
    return by_thread_id
