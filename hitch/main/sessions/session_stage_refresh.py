"""Session-list PR-stage refresh scheduling and per-row stage derivation.

This module owns the session-list page's background PR-stage refresh scheduling
(the off-request ``gh`` refresh and its in-flight de-duplication), the
awaiting-input and active-instance lookups the list render needs, and the
per-row stage-cache derivation that serves the last-known stage immediately.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from django.conf import settings as django_settings
from django.db import close_old_connections

from hitch.main.models import (
    ApprovalRequest,
    CodexInstance,
    SessionMetadata,
    SystemWorkflow,
    UserInputRequest,
)
from hitch.main.runtime import codex_events, rollout
from hitch.main.runtime.rollout_state import (
    _rollout_file_state_from_value,
    _RolloutFileState,
)
from hitch.main.runtime.sdk_values import (
    datetime_value,
    is_nonbool_int,
    string_value,
)
from hitch.main.sequences import unique_nonempty
from hitch.main.sessions import session_stage
from hitch.main.sessions.session_pr_plan import (
    _pr_observation_result_for_rollout_path,
    _pr_snapshot_identity,
    _workflow_activity_ownership_by_id,
    _workflow_after_main_lifecycle,
)
from hitch.main.workflows import pr_qa, pr_stage

logger = logging.getLogger(__name__)

_SESSION_LIST_PR_STAGE_REFRESH_LIMIT = 1
# Sessions whose gh-backed PR stage is being refreshed in a background thread,
# so concurrent renders in this process do not spawn duplicate workers. The
# per-PR global floor lives in the DB via ``rate_limit``; this set only avoids
# redundant threads within one process.
_PR_STAGE_REFRESH_INFLIGHT_LOCK = threading.Lock()
_PR_STAGE_REFRESH_INFLIGHT: set[str] = set()


def _schedule_pr_stage_refresh(session_id: str) -> None:
    """Refresh a session's gh-backed PR stage off the request path.

    The render serves the last-known stage immediately and flags it as
    refreshing; this performs the actual ``gh`` call and persists the result so a
    later render (nudged by the refreshing flag) shows it. Runs synchronously
    under TESTING for deterministic tests. De-duplicated per session within the
    process by an in-flight set, and per PR across the whole app by the
    ``rate_limit`` claim inside the system-agent refreshers.
    """
    if getattr(django_settings, "TESTING", False):
        _refresh_session_pr_stage(session_id)
        return
    with _PR_STAGE_REFRESH_INFLIGHT_LOCK:
        if session_id in _PR_STAGE_REFRESH_INFLIGHT:
            return
        _PR_STAGE_REFRESH_INFLIGHT.add(session_id)
    try:
        threading.Thread(
            target=_pr_stage_refresh_worker,
            args=(session_id,),
            name="pr-stage-refresh",
            daemon=True,
        ).start()
    except Exception:
        with _PR_STAGE_REFRESH_INFLIGHT_LOCK:
            _PR_STAGE_REFRESH_INFLIGHT.discard(session_id)
        logger.exception("failed to start PR stage refresh thread")


def _pr_stage_refresh_worker(session_id: str) -> None:
    close_old_connections()
    try:
        _refresh_session_pr_stage(session_id)
    except Exception:
        logger.exception("background PR stage refresh failed for %s", session_id)
    finally:
        close_old_connections()
        with _PR_STAGE_REFRESH_INFLIGHT_LOCK:
            _PR_STAGE_REFRESH_INFLIGHT.discard(session_id)


def _refresh_session_pr_stage(session_id: str) -> None:
    """Perform the gh-backed PR stage refresh for one session and persist it.

    Mirrors the refresh the list/detail render used to do inline: the workflow
    handoff path persists onto the ``SystemWorkflow``, the log-snapshot path
    re-derives and updates the cached ``derived_stage``. The gh call is gated by
    the per-PR global ``rate_limit`` claim inside the refreshers, so this is a
    cheap no-op when the same PR was refreshed elsewhere recently.
    """
    metadata = SessionMetadata.objects.filter(thread_id=session_id).first()
    rollout_state = _rollout_file_state_from_value(
        metadata.codex_path if metadata is not None else None
    )
    rollout_path = rollout_state.path if rollout_state is not None else None
    pr_observation = _pr_observation_result_for_rollout_path(rollout_path)
    main_updated_at = metadata.codex_updated_at if metadata is not None else None
    latest_pr_workflow = pr_stage._latest_pr_workflow_for_thread(session_id)
    activity_ownership = _workflow_activity_ownership_by_id(
        [(latest_pr_workflow, main_updated_at)]
    )
    stage_pr_workflow = _workflow_after_main_lifecycle(
        latest_pr_workflow,
        pr_observation,
        main_updated_at=main_updated_at,
        newer_main_activity_owned=bool(
            latest_pr_workflow is not None
            and latest_pr_workflow.pk is not None
            and activity_ownership.get(latest_pr_workflow.pk)
        ),
    )
    if (
        stage_pr_workflow is not None
        and pr_qa.pr_monitor_backoff_stage_refresh_due(stage_pr_workflow)
    ):
        pr_qa.refresh_due_pr_monitor_backoffs(
            limit=1, workflow_id=stage_pr_workflow.pk
        )
        return
    if stage_pr_workflow is not None:
        pr_qa.refreshed_pr_handoff_for_stage(stage_pr_workflow)
        return
    snapshot = pr_observation.snapshot
    if metadata is None or snapshot is None or rollout_state is None:
        return
    if not pr_qa.pr_snapshot_stage_refresh_due(
        cwd=metadata.cwd,
        snapshot=snapshot,
        attempted_at=metadata.derived_stage_pr_refresh_attempted_at,
    ):
        return
    pr_stage._mark_cached_pr_stage_refresh_attempt(session_id)
    refreshed = pr_qa.refreshed_pr_snapshot_for_stage(
        cwd=metadata.cwd, snapshot=snapshot
    )
    stage = session_stage.derive_stage(pr_snapshot=refreshed)
    pr_stage._update_cached_stage_best_effort(session_id, stage, rollout_state.mtime_ns)


def _attach_session_stage_context(sessions: list[dict[str, Any]]) -> None:
    thread_ids = [
        session["id"] for session in sessions if isinstance(session.get("id"), str)
    ]
    workflows_by_thread_id = pr_stage._latest_stage_workflows_by_thread_id(thread_ids)
    activity_ownership = _workflow_activity_ownership_by_id(
        (
            workflows_by_thread_id.get(session["id"]),
            session.get("stage_main_updated_at"),
        )
        for session in sessions
        if isinstance(session.get("id"), str)
    )
    active_instances_by_thread_id = _active_instances_by_thread_id(thread_ids)
    waiting_thread_ids = _thread_ids_awaiting_input(thread_ids)
    pr_stage_refreshes_remaining = _SESSION_LIST_PR_STAGE_REFRESH_LIMIT
    for session in sessions:
        session_id = session.get("id")
        if not isinstance(session_id, str):
            continue
        rollout_state = _rollout_file_state_from_value(session.get("codex_path"))
        workflow = workflows_by_thread_id.get(session_id)
        active_instance = active_instances_by_thread_id.get(session_id)
        awaiting_user_input = session_id in waiting_thread_ids
        cached_stage = _cached_stage_for_session_row(session, rollout_state)
        if (
            active_instance is None
            and workflow is None
            and not awaiting_user_input
            and cached_stage is not None
        ):
            assert rollout_state is not None
            stage, pr_snapshot, pr_stage_refreshes_remaining, refreshing = (
                _stage_from_cached_session_row(
                    session_id,
                    session,
                    rollout_state=rollout_state,
                    cached_stage=cached_stage,
                    pr_stage_refreshes_remaining=pr_stage_refreshes_remaining,
                )
            )
            session["stage"] = _session_list_stage_context(
                stage, pr_snapshot=pr_snapshot, refreshing=refreshing
            )
            continue
        rollout_path = rollout_state.path if rollout_state is not None else None
        entries, pr_observation = _session_stage_data_for_rollout_path(rollout_path)
        if not entries and session.get("has_activity"):
            entries = [{"kind": "user"}]
        stage_workflow = _workflow_after_main_lifecycle(
            workflow,
            pr_observation,
            main_updated_at=session.get("stage_main_updated_at"),
            newer_main_activity_owned=bool(
                workflow is not None
                and workflow.pk is not None
                and activity_ownership.get(workflow.pk)
            ),
        )
        if (
            active_instance is None
            and stage_workflow is None
            and not awaiting_user_input
            and cached_stage is not None
        ):
            assert rollout_state is not None
            stage, pr_snapshot, pr_stage_refreshes_remaining, refreshing = (
                _stage_from_cached_session_row(
                    session_id,
                    session,
                    rollout_state=rollout_state,
                    cached_stage=cached_stage,
                    pr_stage_refreshes_remaining=pr_stage_refreshes_remaining,
                )
            )
            session["stage"] = _session_list_stage_context(
                stage, pr_snapshot=pr_snapshot, refreshing=refreshing
            )
            continue
        log_pr_snapshot = pr_observation.snapshot
        # Serve the last-known PR stage now; when a gh refresh is due, flag the
        # badge as refreshing and do the actual refresh off-request so the page
        # is not blocked on a ``gh`` call (the result lands on a later render).
        workflow_pr_snapshot = pr_qa.pr_handoff_for_workflow(stage_workflow)
        # Only the PR stage gets the refreshing badge: an active worker or a
        # waiting-for-input row shows its own stage, and flagging that refreshing
        # would schedule a needless worker and reload.
        pr_stage_displayed = active_instance is None and not awaiting_user_input
        refresh_due = pr_stage_displayed and (
            pr_qa.pr_handoff_stage_refresh_due(stage_workflow)
            or pr_qa.pr_monitor_backoff_stage_refresh_due(stage_workflow)
        )
        if (
            pr_stage_displayed
            and stage_workflow is None
            and log_pr_snapshot is not None
            and pr_qa.pr_snapshot_stage_refresh_due(
                cwd=string_value(session.get("cwd")),
                snapshot=log_pr_snapshot,
                attempted_at=datetime_value(
                    session.get("stage_pr_refresh_attempted_at")
                ),
            )
        ):
            refresh_due = True
        # Flag the badge refreshing only when a refresh actually runs this
        # render. A row whose refresh is due but falls outside the per-render
        # budget must not keep data-refreshing set, or _stage_refresh_script
        # reloads every 7s for a result that never lands; the reload still fires
        # while a scheduled refresh is pending, so budget-deferred rows are
        # picked up on a later render. ``refresh_due`` (independent of the
        # budget) still gates the cache write below.
        badge_refreshing = False
        if refresh_due and pr_stage_refreshes_remaining > 0:
            _schedule_pr_stage_refresh(session_id)
            pr_stage_refreshes_remaining -= 1
            badge_refreshing = True
        stage = session_stage.derive_stage(
            entries=entries,
            active_instance=active_instance,
            workflow=stage_workflow,
            awaiting_user_input=awaiting_user_input,
            pr_snapshot=log_pr_snapshot,
            workflow_pr_snapshot=workflow_pr_snapshot,
        )
        stage_executing = stage.key == session_stage.IMPLEMENTATION.key and (
            active_instance is not None
            or (
                stage_workflow is not None
                and stage_workflow.is_active
            )
        )
        session["stage"] = _session_list_stage_context(
            stage,
            pr_snapshot=_session_list_pr_snapshot_for_stage(
                stage_workflow=stage_workflow,
                log_pr_snapshot=log_pr_snapshot,
                workflow_pr_snapshot=workflow_pr_snapshot,
            ),
            refreshing=badge_refreshing,
            executing=stage_executing,
        )
        # The stage cache is keyed only on the rollout file's mtime, so it may
        # only hold stages that are a pure function of the rollout. A stage that
        # an active worker or a PR/QA workflow forced (e.g. Implementation while
        # a turn runs) is transient state the mtime key cannot track: once the
        # worker/workflow goes away without rewriting the rollout, the cached
        # row would still satisfy the read guard and resurrect the stale active
        # badge. Persist only when no such owner contributed to the stage.
        # Skip whenever a refresh is due -- even if the budget deferred it this
        # render -- because the snapshot is known-stale: caching its derived
        # stage (possibly a stale terminal PR stage) under the rollout mtime
        # would let the cached fast path serve it without ever rechecking.
        if (
            active_instance is None
            and stage_workflow is None
            and not awaiting_user_input
            and not refresh_due
        ):
            pr_stage._update_cached_stage_best_effort(
                session_id,
                stage,
                rollout_state.mtime_ns if rollout_state is not None else 0,
            )


def _stage_from_cached_session_row(
    session_id: str,
    session: Mapping[str, Any],
    *,
    rollout_state: _RolloutFileState,
    cached_stage: session_stage.SessionStage,
    pr_stage_refreshes_remaining: int,
) -> tuple[session_stage.SessionStage, Mapping[str, Any] | None, int, bool]:
    pr_snapshot = None
    stage = cached_stage
    refreshing = False
    if cached_stage.key == session_stage.PR.key:
        pr_snapshot = _pr_snapshot_for_rollout_path(rollout_state.path)
        if pr_snapshot is not None and pr_qa.pr_snapshot_stage_refresh_due(
            cwd=string_value(session.get("cwd")),
            snapshot=pr_snapshot,
            attempted_at=datetime_value(session.get("stage_pr_refresh_attempted_at")),
        ):
            # Serve the cached stage now and refresh off-request; the result is
            # persisted to the stage cache for a later render to read back.
            refreshing = True
            if pr_stage_refreshes_remaining > 0:
                _schedule_pr_stage_refresh(session_id)
                pr_stage_refreshes_remaining -= 1
            else:
                # Budget spent on earlier PR rows: no refresh scheduled, so don't
                # flag this badge refreshing or _stage_refresh_script reloads
                # every 7s for a result that never lands (mirrors the
                # rollout-derived path in _attach_session_stage_context).
                refreshing = False
    return stage, pr_snapshot, pr_stage_refreshes_remaining, refreshing


def _session_list_pr_snapshot_for_stage(
    *,
    stage_workflow: SystemWorkflow | None,
    log_pr_snapshot: Mapping[str, Any] | None,
    workflow_pr_snapshot: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if (
        stage_workflow is not None
        and _pr_snapshot_identity(workflow_pr_snapshot) is None
    ):
        return workflow_pr_snapshot
    return session_stage.merge_pr_snapshots(
        log_pr_snapshot=log_pr_snapshot,
        workflow_pr_snapshot=workflow_pr_snapshot,
    )


def _session_list_stage_context(
    stage: session_stage.SessionStage,
    *,
    pr_snapshot: Mapping[str, Any] | None = None,
    refreshing: bool = False,
    executing: bool = False,
) -> dict[str, Any]:
    context: dict[str, Any] = dict(stage.as_context())
    if stage.key == session_stage.IMPLEMENTATION.key:
        if executing:
            context["executing"] = True
        else:
            context["tone"] = "idle"
    if refreshing:
        context["refreshing"] = True
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
    identity = _pr_snapshot_identity(snapshot)
    return identity[1] if identity is not None else None


def _thread_ids_awaiting_input(thread_ids: Iterable[str]) -> set[str]:
    ids = unique_nonempty(thread_ids)
    if not ids:
        return set()
    active_statuses = CodexInstance.ACTIVE_STATUSES
    direct_input_thread_ids = UserInputRequest.objects.filter(
        response__isnull=True,
        instance__thread_id__in=ids,
        instance__status__in=active_statuses,
    ).values_list("instance__thread_id", flat=True)
    direct_approval_thread_ids = ApprovalRequest.objects.filter(
        decision=ApprovalRequest.DECISION_PENDING,
        instance__thread_id__in=ids,
        instance__status__in=active_statuses,
    ).values_list("instance__thread_id", flat=True)
    workflow_input_thread_ids = UserInputRequest.objects.filter(
        response__isnull=True,
        instance__system_agent_runs__workflow__main_thread_id__in=ids,
        instance__system_agent_runs__workflow__status=SystemWorkflow.STATUS_RUNNING,
    ).values_list(
        "instance__system_agent_runs__workflow__main_thread_id", flat=True
    )
    workflow_approval_thread_ids = ApprovalRequest.objects.filter(
        decision=ApprovalRequest.DECISION_PENDING,
        instance__system_agent_runs__workflow__main_thread_id__in=ids,
        instance__system_agent_runs__workflow__status=SystemWorkflow.STATUS_RUNNING,
    ).values_list(
        "instance__system_agent_runs__workflow__main_thread_id", flat=True
    )
    waiting_thread_ids: set[str] = set()
    for thread_ids_result in (
        direct_input_thread_ids,
        direct_approval_thread_ids,
        workflow_input_thread_ids,
        workflow_approval_thread_ids,
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


def _cached_stage_for_session_row(
    session: Mapping[str, Any],
    rollout_state: _RolloutFileState | None,
) -> session_stage.SessionStage | None:
    if rollout_state is None:
        return None
    cached = session_stage.stage_for_key(string_value(session.get("stage_cache_key")))
    if cached is None:
        return None
    return (
        cached
        if session.get("stage_cache_mtime_ns") == rollout_state.mtime_ns
        else None
    )


def _session_stage_data_for_rollout_path(
    rollout_path: Path | None,
) -> tuple[list[dict[str, Any]], codex_events.PrObservationResult]:
    empty_pr_observation = codex_events.PrObservationResult(snapshot=None)
    if rollout_path is None:
        return [], empty_pr_observation
    try:
        stage_data = rollout.session_stage_data(rollout_path)
    except Exception:
        logger.exception("failed to parse rollout %s for session stage", rollout_path)
        return [], empty_pr_observation
    if stage_data is None:
        return [], empty_pr_observation
    return list(stage_data.entries), stage_data.pr_observation


def _pr_snapshot_for_rollout_path(rollout_path: Path | None) -> dict[str, Any] | None:
    return _pr_observation_result_for_rollout_path(rollout_path).snapshot
