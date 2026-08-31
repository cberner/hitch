"""Lifecycle routing for Hitch-owned autonomous-goal turns."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from datetime import timedelta
from pathlib import Path
from typing import Any

from django.db import models
from django.utils import timezone

from hitch.main.models import (
    CodexInstance,
    ProposedSession,
    SessionMetadata,
    SystemAgentRun,
    SystemWorkflow,
)
from hitch.main.sessions import session_index
from hitch.main.workflows import pr_tracking

logger = logging.getLogger(__name__)

AUTONOMOUS_GOAL_AGENT_KIND: str = SystemWorkflow.KIND_AUTONOMOUS_GOAL_RUN
AUTONOMOUS_GOAL_REVIEWER_AGENT_KIND = "autonomous_goal_reviewer"
AUTONOMOUS_GOAL_DISPLAY_AUTHOR = "Autonomous goal agent"
AUTONOMOUS_GOAL_REVIEWER_DISPLAY_AUTHOR = "Autonomous goal reviewer"
AUTONOMOUS_GOAL_AGENT_PROMPT_TITLE = session_index.AUTONOMOUS_GOAL_AGENT_PROMPT_TITLE
AUTONOMOUS_GOAL_JUDGE_PROMPT_TITLE = session_index.AUTONOMOUS_GOAL_JUDGE_PROMPT_TITLE

AUTONOMOUS_GOAL_DELETED_ERROR = "Autonomous goal deleted by user"
AUTONOMOUS_GOAL_PROPOSAL_ACCEPTED_ERROR = "Autonomous goal proposal accepted by user"
AUTONOMOUS_GOAL_PROPOSAL_REJECTED_ERROR = "Autonomous goal proposal rejected by user"
AUTONOMOUS_GOAL_PROPOSAL_DISMISSED_ERROR = "Autonomous goal proposal dismissed by user"

SYSTEM_AGENT_APPROVAL_MODE = "auto_review"
AUTONOMOUS_GOAL_IMPLEMENTATION_SANDBOX_POLICY = "workspaceWrite"

STEP_BLOCKED = "blocked"
STEP_AUTONOMOUS_GOAL_RUNNING = "autonomous_goal_running"
STEP_AUTONOMOUS_GOAL_PROPOSED = "autonomous_goal_proposed"
STEP_AUTONOMOUS_GOAL_SKIPPED = "autonomous_goal_skipped"

_ROUTE_CLAIM_TIMEOUT = timedelta(minutes=10)


def legacy_promoted_system_thread_ids() -> set[str]:
    """Return candidate threads promoted before proposals began starting fresh."""
    return set(
        ProposedSession.objects.filter(
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            candidate_session__isnull=False,
            accepted_session=models.F("candidate_session"),
        ).values_list("candidate_session__thread_id", flat=True)
    )


def hidden_thread_ids() -> set[str]:
    hidden_ids = set(SystemAgentRun.objects.exclude(thread_id="").values_list("thread_id", flat=True).distinct())
    hidden_ids.update(
        CodexInstance.objects.filter(purpose=CodexInstance.PURPOSE_SYSTEM_AGENT)
        .exclude(thread_id="")
        .values_list("thread_id", flat=True)
        .distinct()
    )
    hidden_ids.update(
        SessionMetadata.objects.filter(is_hidden_system_session=True)
        .exclude(thread_id="")
        .values_list("thread_id", flat=True)
        .distinct()
    )
    return hidden_ids - legacy_promoted_system_thread_ids()


def hidden_thread_ids_from_threads(threads: Iterable[Any]) -> set[str]:
    hidden_ids = {
        thread_id
        for thread in threads
        if isinstance(thread_id := getattr(thread, "id", None), str) and hitch_system_agent_thread(thread)
    }
    return hidden_ids - legacy_promoted_system_thread_ids()


def hitch_system_agent_thread(thread: Any) -> bool:
    return session_index.hidden_system_session_from_metadata(
        name=_thread_metadata_value(getattr(thread, "name", None)).strip(),
        preview=_thread_metadata_value(getattr(thread, "preview", None)).strip(),
        thread_source=_thread_metadata_value(getattr(thread, "thread_source", None)),
    )


def _thread_metadata_value(value: Any) -> str:
    root = getattr(value, "root", value)
    raw = getattr(root, "value", root)
    return raw if isinstance(raw, str) else ""


def reconcile_terminal_workflow_instances(*, main_thread_id: str | None = None, workflow_id: int | None = None) -> int:
    """Route terminal AG workers and recover a stale run with no worker."""
    workflows = SystemWorkflow.objects.filter(
        kind=AUTONOMOUS_GOAL_AGENT_KIND,
        status=SystemWorkflow.STATUS_RUNNING,
    )
    if main_thread_id is not None:
        workflows = workflows.filter(main_thread_id=main_thread_id)
    if workflow_id is not None:
        workflows = workflows.filter(pk=workflow_id)
    workflow_rows = list(workflows.order_by("created_at", "id"))
    from hitch.main.workflows import autonomous_goals

    reconciled = autonomous_goals.reconcile_pending_run_cleanups(
        main_thread_id=main_thread_id,
        workflow_id=workflow_id,
    )
    if not workflow_rows:
        return reconciled
    instances = (
        CodexInstance.objects.filter(
            workflow_id__in=[workflow.pk for workflow in workflow_rows],
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            status__in=(CodexInstance.STATUS_COMPLETED, CodexInstance.STATUS_FAILED),
        )
        .filter(_unclaimed_route_filter())
        .exclude(
            system_agent_runs__status__in=(
                SystemAgentRun.STATUS_COMPLETED,
                SystemAgentRun.STATUS_FAILED,
            )
        )
        .order_by("started_at", "id")
    )
    for instance in instances:
        try:
            if on_codex_instance_finished(instance):
                reconciled += 1
        except Exception:
            logger.exception("failed to reconcile terminal AG instance %s", instance.pk)
    reconciled += autonomous_goals.recover_orphaned_workflows(workflow_rows)
    return reconciled


def _unclaimed_route_filter() -> models.Q:
    stale_before = timezone.now() - _ROUTE_CLAIM_TIMEOUT
    return models.Q(workflow_routing_started_at__isnull=True) | models.Q(workflow_routing_started_at__lt=stale_before)


def workflow_has_inflight_instance(workflow_id: int) -> bool:
    """Return whether a worker is active or its terminal route is still owned."""
    fresh_claim_after = timezone.now() - _ROUTE_CLAIM_TIMEOUT
    return (
        CodexInstance.objects.filter(
            workflow_id=workflow_id,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
        )
        .filter(
            models.Q(status__in=CodexInstance.ACTIVE_STATUSES)
            | models.Q(
                status__in=(CodexInstance.STATUS_COMPLETED, CodexInstance.STATUS_FAILED),
                system_agent_runs__status__in=(
                    SystemAgentRun.STATUS_STARTING,
                    SystemAgentRun.STATUS_RUNNING,
                ),
            )
            | models.Q(
                status__in=(CodexInstance.STATUS_COMPLETED, CodexInstance.STATUS_FAILED),
                workflow_routing_started_at__gte=fresh_claim_after,
            )
        )
        .exists()
    )


def on_codex_instance_finished(instance: CodexInstance) -> bool:
    """Route one terminal worker to its AG ledger, idempotently."""
    if instance.purpose != CodexInstance.PURPOSE_SYSTEM_AGENT:
        pr_tracking.supersede_pr_after_turn(instance)
        return False
    if instance.workflow_id is None:
        return False
    if not _claim_instance_for_routing(instance):
        return True
    try:
        run = _system_agent_run_for_instance(instance)
        if run is None:
            return False
        from hitch.main.workflows import autonomous_goals

        if run.status in (SystemAgentRun.STATUS_COMPLETED, SystemAgentRun.STATUS_FAILED):
            autonomous_goals.cleanup_terminal_run(run)
            return True
        autonomous_goals.on_agent_finished(instance, run, run.workflow)
        return True
    except Exception:
        _clear_instance_route_claim(instance)
        raise


def _claim_instance_for_routing(instance: CodexInstance) -> bool:
    now = timezone.now()
    claimed = (
        CodexInstance.objects.filter(
            pk=instance.pk,
            workflow_id__isnull=False,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            status__in=(CodexInstance.STATUS_COMPLETED, CodexInstance.STATUS_FAILED),
        )
        .filter(_unclaimed_route_filter())
        .update(workflow_routing_started_at=now)
    )
    if claimed:
        instance.workflow_routing_started_at = now
    return bool(claimed)


def _clear_instance_route_claim(instance: CodexInstance) -> None:
    claimed_at = instance.workflow_routing_started_at
    if claimed_at is None:
        return
    if CodexInstance.objects.filter(
        pk=instance.pk,
        workflow_routing_started_at=claimed_at,
    ).update(workflow_routing_started_at=None):
        instance.workflow_routing_started_at = None


def _system_agent_run_for_instance(instance: CodexInstance) -> SystemAgentRun | None:
    run = SystemAgentRun.objects.select_related("workflow").filter(instance=instance).first()
    if run is not None:
        return run
    if instance.workflow_id is None or not instance.agent_kind:
        return None
    workflow = SystemWorkflow.objects.filter(pk=instance.workflow_id).first()
    if workflow is None:
        return None
    run, _created = SystemAgentRun.objects.get_or_create(
        instance=instance,
        defaults={
            "workflow": workflow,
            "agent_kind": instance.agent_kind,
            "thread_id": instance.thread_id,
            "status": SystemAgentRun.STATUS_RUNNING,
            "input": {"cwd": instance.cwd},
        },
    )
    return run


def final_agent_text(events_path: str) -> str:
    path = Path(events_path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    latest = ""
    deltas: dict[str, str] = {}
    for raw in lines:
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        method = event.get("method")
        payload = event.get("payload") or {}
        if method == "item/agentMessage/delta":
            item_id = payload.get("itemId")
            delta = payload.get("delta")
            if isinstance(item_id, str) and isinstance(delta, str):
                deltas[item_id] = deltas.get(item_id, "") + delta
                latest = deltas[item_id]
        elif method == "item/completed":
            item = payload.get("item") or {}
            if (
                item.get("type") == "agentMessage"
                and item.get("phase") != "commentary"
                and isinstance(item.get("text"), str)
            ):
                latest = item["text"]
    return latest
