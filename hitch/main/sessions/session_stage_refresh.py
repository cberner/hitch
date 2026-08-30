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
    SystemWorkflow,
    UserInputRequest,
)
from hitch.main.runtime import rollout
from hitch.main.runtime.rollout_state import (
    _rollout_file_state_from_value,
    _RolloutFileState,
)
from hitch.main.runtime.sdk_values import is_nonbool_int, string_value
from hitch.main.sequences import unique_nonempty
from hitch.main.sessions import agent_tasks, session_stage
from hitch.main.workflows import pr_stage, pr_tracking

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
    """Refresh one session's durable PR state when it is still current."""
    stored_pr = pr_tracking.stored_record_for_thread(session_id)
    if stored_pr is not None and pr_tracking.record_is_current(stored_pr):
        pr_tracking.refreshed_pr_handoff_for_stage(stored_pr)


def _attach_session_stage_context(sessions: list[dict[str, Any]]) -> None:
    thread_ids = [
        session["id"] for session in sessions if isinstance(session.get("id"), str)
    ]
    registered_prs_by_thread_id = pr_tracking.records_by_thread_id(thread_ids)
    active_instances_by_thread_id = _active_instances_by_thread_id(thread_ids)
    waiting_thread_ids = _thread_ids_awaiting_input(thread_ids)
    pr_stage_refreshes_remaining = _SESSION_LIST_PR_STAGE_REFRESH_LIMIT
    for session in sessions:
        session_id = session.get("id")
        if not isinstance(session_id, str):
            continue
        rollout_state = _rollout_file_state_from_value(session.get("codex_path"))
        stored_pr = registered_prs_by_thread_id.get(session_id)
        registered_pr = (
            stored_pr if pr_tracking.record_is_current(stored_pr) else None
        )
        active_instance = active_instances_by_thread_id.get(session_id)
        publishing_before_registration = bool(
            active_instance is not None
            and active_instance.agent_kind == agent_tasks.PR_PUBLISH_AGENT_KIND
            and not pr_tracking.watch_registered_by_instance(
                registered_pr, active_instance.pk
            )
        )
        awaiting_user_input = session_id in waiting_thread_ids
        cached_stage = _cached_stage_for_session_row(session, rollout_state)
        if (
            active_instance is None
            and stored_pr is None
            and not awaiting_user_input
            and cached_stage is not None
            and cached_stage.key
            not in {
                session_stage.PR.key,
                session_stage.DONE_MERGED.key,
                session_stage.DONE_CLOSED.key,
            }
        ):
            session["stage"] = _session_list_stage_context(cached_stage)
            continue
        rollout_path = rollout_state.path if rollout_state is not None else None
        entries = _session_stage_data_for_rollout_path(rollout_path)
        if not entries and session.get("has_activity"):
            entries = [{"kind": "user"}]
        # Serve the last-known PR stage now; when a gh refresh is due, flag the
        # badge as refreshing and do the actual refresh off-request so the page
        # is not blocked on a ``gh`` call (the result lands on a later render).
        pr_snapshot = (
            {}
            if publishing_before_registration
            else pr_tracking.pr_handoff_for_record(registered_pr)
        )
        # Only the PR stage gets the refreshing badge: an active worker or a
        # waiting-for-input row shows its own stage, and flagging that refreshing
        # would schedule a needless worker and reload.
        pr_stage_displayed = active_instance is None and not awaiting_user_input
        refresh_due = (
            pr_stage_displayed
            and pr_tracking.pr_handoff_stage_refresh_due(registered_pr)
        )
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
            awaiting_user_input=awaiting_user_input,
            pr_snapshot=pr_snapshot,
        )
        stage_executing = stage.key == session_stage.IMPLEMENTATION.key and (
            active_instance is not None
        )
        session["stage"] = _session_list_stage_context(
            stage,
            pr_snapshot=pr_snapshot,
            refreshing=badge_refreshing,
            executing=stage_executing,
        )
        # The stage cache is keyed only on the rollout file's mtime, so it may
        # only hold stages that are a pure function of the rollout. A stage that
        # an active worker forced (e.g. Implementation while
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
            and not awaiting_user_input
            and not refresh_due
        ):
            pr_stage._update_cached_stage_best_effort(
                session_id,
                stage,
                rollout_state.mtime_ns if rollout_state is not None else 0,
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
    return None


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
) -> list[dict[str, Any]]:
    if rollout_path is None:
        return []
    try:
        stage_data = rollout.session_stage_data(rollout_path)
    except Exception:
        logger.exception("failed to parse rollout %s for session stage", rollout_path)
        return []
    if stage_data is None:
        return []
    return list(stage_data.entries)
