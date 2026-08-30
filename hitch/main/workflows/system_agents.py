"""Reusable orchestration for Hitch-owned background Codex agents."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from django.db import models
from django.db.models import QuerySet
from django.utils import timezone

from hitch.main.models import (
    CodexInstance,
    ProposedSession,
    SessionMetadata,
    SystemAgentRun,
    SystemWorkflow,
)
from hitch.main.sessions import session_index
from hitch.main.workflows import engine, pr_tracking
from hitch.main.workflows.workflow_state import (
    _session_metadata_from_state,
    _state_string,
)

logger = logging.getLogger(__name__)

AUTONOMOUS_GOAL_AGENT_KIND: str = SystemWorkflow.KIND_AUTONOMOUS_GOAL_RUN
LEGACY_AUTONOMOUS_GOAL_HISTORY_AGENT_KIND = "autonomous_goal_history_summary"
AUTONOMOUS_GOAL_JUDGE_AGENT_KIND = "autonomous_goal_judge"
AUTONOMOUS_GOAL_DISPLAY_AUTHOR = "Autonomous goal agent"
AUTONOMOUS_GOAL_JUDGE_DISPLAY_AUTHOR = "Autonomous goal judge"
AUTONOMOUS_GOAL_DELETED_ERROR = "Autonomous goal deleted by user"
AUTONOMOUS_GOAL_PROPOSAL_ACCEPTED_ERROR = "Autonomous goal proposal accepted by user"
AUTONOMOUS_GOAL_PROPOSAL_REJECTED_ERROR = "Autonomous goal proposal rejected by user"
AUTONOMOUS_GOAL_PROPOSAL_DISMISSED_ERROR = "Autonomous goal proposal dismissed by user"



AUTONOMOUS_GOAL_AGENT_PROMPT_TITLE = session_index.AUTONOMOUS_GOAL_AGENT_PROMPT_TITLE
AUTONOMOUS_GOAL_JUDGE_PROMPT_TITLE = session_index.AUTONOMOUS_GOAL_JUDGE_PROMPT_TITLE
SYSTEM_AGENT_APPROVAL_MODE = "auto_review"
AUTONOMOUS_GOAL_IMPLEMENTATION_SANDBOX_POLICY = "workspaceWrite"
STEP_BLOCKED = "blocked"
STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING = "autonomous_goal_candidate_running"
LEGACY_STEP_AUTONOMOUS_GOAL_HISTORY = "autonomous_goal_history_summarizing"
STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING = "autonomous_goal_judge_running"
STEP_AUTONOMOUS_GOAL_PROPOSED = "autonomous_goal_proposed"
STEP_AUTONOMOUS_GOAL_SKIPPED = "autonomous_goal_skipped"
_WORKFLOW_ROUTE_CLAIM_TIMEOUT = timedelta(minutes=10)
# A system workflow can commit a transient step before spawning its worker.
# Reconciliation retries that step after this window if no worker appeared.
_WORKFLOW_SPAWN_STALE_TIMEOUT = timedelta(minutes=15)
def _sync_workflow_instance(target: SystemWorkflow, source: SystemWorkflow) -> None:
    target.status = source.status
    target.step = source.step
    target.state = source.state


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
    hidden_ids = set(
        SystemAgentRun.objects.exclude(thread_id="")
        .values_list("thread_id", flat=True)
        .distinct()
    )
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
    """Route terminal workflow-owned workers that missed their finish callback."""
    workflows = list(
        _running_workflows_for_reconciliation(
            main_thread_id=main_thread_id,
            workflow_id=workflow_id,
        )
    )
    reconciled = 0
    if workflows:
        reconciled += _reconcile_terminal_system_agent_instances(workflows)
        reconciled += _drive_orphaned_workflow_spawns(workflows)
    return reconciled


def _drive_orphaned_workflow_spawns(workflows: list[SystemWorkflow]) -> int:
    """Re-drive (or block) every workflow stranded by a dead spawn handler.

    One sweep over the handlers' registered recovery specs replaces the former
    per-step reconcilers: for each stale RUNNING workflow whose step has a spec
    and whose expected worker is missing, claim the step and run its recovery.
    ``needs_recovery`` is re-checked after the claim (a worker may have appeared
    since the batch was loaded) so recovery never races a live spawn; like the
    former reconcilers, the check stays outside the claim's write lock.
    """
    now = timezone.now()
    reconciled = 0
    for workflow in workflows:
        spec = engine.spawn_recovery_spec(workflow.kind, workflow.step)
        if spec is None:
            continue
        stale_before = now - spec.stale_timeout
        if workflow.updated_at > stale_before:
            continue
        if not spec.needs_recovery(workflow):
            continue
        locked = _claim_stale_workflow_step(workflow, step=workflow.step, stale_before=stale_before)
        if locked is None or not spec.needs_recovery(locked):
            continue
        spec.recover(locked)
        reconciled += 1
    return reconciled


def _claim_stale_workflow_step(workflow: SystemWorkflow, *, step: str, stale_before: datetime) -> SystemWorkflow | None:
    """Lock and claim a stale RUNNING workflow still at ``step``.

    Returns the locked row (with ``updated_at`` bumped so concurrent reconcilers
    back off for a fresh stale window) or ``None`` if it is not eligible.
    """

    def _touch(locked: SystemWorkflow) -> SystemWorkflow:
        locked.save(update_fields=["updated_at"])
        return locked

    return engine.claim_workflow_transition(
        workflow,
        _touch,
        expect_step=step,
        guard=lambda locked: locked.updated_at <= stale_before,
    )


def _running_workflows_for_reconciliation(
    *, main_thread_id: str | None, workflow_id: int | None
) -> QuerySet[SystemWorkflow]:
    workflows = SystemWorkflow.objects.filter(status=SystemWorkflow.STATUS_RUNNING)
    if main_thread_id is not None:
        workflows = workflows.filter(main_thread_id=main_thread_id)
    if workflow_id is not None:
        workflows = workflows.filter(pk=workflow_id)
    return workflows.order_by("created_at", "id")


def _reconcile_terminal_system_agent_instances(workflows: list[SystemWorkflow]) -> int:
    filters: models.Q = models.Q(pk__in=[])
    has_instance_filter = False
    for workflow in workflows:
        agent_kinds = _expected_system_agent_kinds_for_step(workflow)
        if not agent_kinds:
            continue
        filters |= models.Q(workflow_id=workflow.pk, agent_kind__in=agent_kinds)
        has_instance_filter = True
    if not has_instance_filter:
        return 0
    instances = (
        CodexInstance.objects.filter(
            filters,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            status__in=(
                CodexInstance.STATUS_COMPLETED,
                CodexInstance.STATUS_FAILED,
            ),
        )
        .filter(_unclaimed_workflow_instance_filter())
        .exclude(agent_kind="")
        .exclude(
            system_agent_runs__status__in=(
                SystemAgentRun.STATUS_COMPLETED,
                SystemAgentRun.STATUS_FAILED,
            )
        )
        .order_by("started_at", "id")
    )
    reconciled = 0
    routed_instance_ids: set[int] = set()
    for instance in instances:
        if instance.pk in routed_instance_ids:
            continue
        routed_instance_ids.add(instance.pk)
        if _route_terminal_workflow_instance(instance):
            reconciled += 1
    return reconciled


def _expected_system_agent_kinds_for_step(workflow: SystemWorkflow) -> tuple[str, ...]:
    if workflow.kind == AUTONOMOUS_GOAL_AGENT_KIND:
        if workflow.step == LEGACY_STEP_AUTONOMOUS_GOAL_HISTORY:
            return (LEGACY_AUTONOMOUS_GOAL_HISTORY_AGENT_KIND,)
        if workflow.step == STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING:
            return (AUTONOMOUS_GOAL_AGENT_KIND,)
        if workflow.step == STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING:
            return (AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,)
    return ()


def _unclaimed_workflow_instance_filter() -> models.Q:
    stale_before = timezone.now() - _WORKFLOW_ROUTE_CLAIM_TIMEOUT
    return models.Q(workflow_routing_started_at__isnull=True) | models.Q(workflow_routing_started_at__lt=stale_before)


def _route_terminal_workflow_instance(instance: CodexInstance) -> bool:
    try:
        return on_codex_instance_finished(instance)
    except Exception:
        logger.exception(
            "failed to reconcile terminal workflow instance %s",
            instance.pk,
        )
        return False


def on_codex_instance_finished(instance: CodexInstance) -> bool:
    """Route a terminal worker to its owning system workflow, if any."""
    route_claimed = False
    if _workflow_owned_instance_requires_route_claim(instance):
        if not _claim_workflow_instance_for_routing(instance):
            return True
        route_claimed = True
    try:
        return _route_finished_codex_instance(instance)
    except Exception:
        if route_claimed:
            _clear_workflow_instance_routing_claim(instance)
        raise


def _route_finished_codex_instance(instance: CodexInstance) -> bool:
    if instance.purpose == CodexInstance.PURPOSE_SYSTEM_AGENT:
        return _handle_system_agent_finished(instance)
    if instance.purpose == CodexInstance.PURPOSE_SYSTEM_FEEDBACK:
        _dispatch_workflow_event(instance, "on_feedback_finished")
        return True
    pr_tracking.supersede_pr_after_turn(instance)
    return False


def _dispatch_workflow_event(instance: CodexInstance, event: str) -> None:
    workflow = _workflow_for_instance(instance)
    if workflow is None:
        return
    handler = engine.primary_handler(workflow.kind)
    if handler is None:
        return
    getattr(handler, event)(instance, workflow)


def _workflow_owned_instance_requires_route_claim(instance: CodexInstance) -> bool:
    return instance.workflow_id is not None and instance.purpose in (
        CodexInstance.PURPOSE_SYSTEM_AGENT,
        CodexInstance.PURPOSE_SYSTEM_FEEDBACK,
    )


def _claim_workflow_instance_for_routing(instance: CodexInstance) -> bool:
    now = timezone.now()
    claimed = (
        CodexInstance.objects.filter(
            pk=instance.pk,
            workflow_id__isnull=False,
            purpose__in=(
                CodexInstance.PURPOSE_SYSTEM_AGENT,
                CodexInstance.PURPOSE_SYSTEM_FEEDBACK,
            ),
            status__in=(
                CodexInstance.STATUS_COMPLETED,
                CodexInstance.STATUS_FAILED,
            ),
        )
        .filter(_unclaimed_workflow_instance_filter())
        .update(workflow_routing_started_at=now)
    )
    if claimed:
        instance.workflow_routing_started_at = now
        return True
    return False


def _clear_workflow_instance_routing_claim(instance: CodexInstance) -> None:
    claimed_at = instance.workflow_routing_started_at
    if claimed_at is None:
        return
    cleared = CodexInstance.objects.filter(
        pk=instance.pk,
        workflow_routing_started_at=claimed_at,
    ).update(workflow_routing_started_at=None)
    if cleared:
        instance.workflow_routing_started_at = None


def _handle_system_agent_finished(instance: CodexInstance) -> bool:
    run = _system_agent_run_for_instance(instance)
    if run is None:
        return False
    workflow = run.workflow
    _route_system_agent_finished(instance, run, workflow)
    return True


def _route_system_agent_finished(instance: CodexInstance, run: SystemAgentRun, workflow: SystemWorkflow) -> None:
    if run.status in (SystemAgentRun.STATUS_COMPLETED, SystemAgentRun.STATUS_FAILED):
        autonomous_goals._cleanup_cancelled_autonomous_goal_terminal_run(instance, run, workflow)
        return
    handler = engine.handler_for(workflow, run=run, instance=instance)
    if handler is None:
        _fail_unsupported_system_agent_run(run, workflow)
        return
    handler.on_agent_finished(instance, run, workflow)


def _fail_unsupported_system_agent_run(run: SystemAgentRun, workflow: SystemWorkflow) -> None:
    error = f"system workflow kind {workflow.kind!r} is no longer supported"
    if workflow.is_active:
        _fail_run_and_block_workflow(run, error, surface_to_thread=False)
        return
    run.status = SystemAgentRun.STATUS_FAILED
    run.error = error
    run.save(update_fields=["status", "error", "updated_at"])


def _resolved_stack_proposal_candidate_cleanup_cwd(
    proposal: ProposedSession,
) -> str:
    if proposal.outcome_status not in {
        ProposedSession.OUTCOME_DISMISSED,
        ProposedSession.OUTCOME_REJECTED,
    }:
        return ""
    if proposal.accepted_session_id is not None:
        return ""
    candidate = proposal.candidate_session
    return candidate.cwd if candidate is not None and candidate.cwd else ""


def _state_without_current_candidate_result(state: Mapping[str, Any]) -> dict[str, Any]:
    next_state = dict(state)
    for key in ("candidate", "judgment", "judge_session_id"):
        next_state.pop(key, None)
    return next_state


def _candidate_session_cwd_from_state(workflow: SystemWorkflow, key: str) -> str:
    metadata = _session_metadata_from_state(workflow, key)
    return metadata.cwd if metadata is not None else ""


def _mark_system_agent_runs_failed(runs: list[SystemAgentRun], error: str) -> None:
    for run in runs:
        run.status = SystemAgentRun.STATUS_FAILED
        run.error = error
        run.save(update_fields=["status", "error", "updated_at"])


def _final_agent_text(events_path: str) -> str:
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


def _fail_run_and_block_workflow(
    run: SystemAgentRun,
    error: str,
    raw_output: str = "",
    *,
    surface_to_thread: bool = True,
) -> None:
    _fail_run(
        run,
        error,
        raw_output=raw_output,
        block_workflow=True,
        surface_to_thread=surface_to_thread,
    )


def _fail_run(
    run: SystemAgentRun,
    error: str,
    *,
    raw_output: str = "",
    block_workflow: bool,
    surface_to_thread: bool = True,
) -> None:
    run.status = SystemAgentRun.STATUS_FAILED
    run.error = error
    run.raw_output = raw_output
    run.save(update_fields=["status", "error", "raw_output", "updated_at"])
    if not block_workflow:
        return
    workflow = run.workflow
    _block_workflow(workflow, error, surface_to_thread=surface_to_thread)


def _complete_workflow(
    workflow: SystemWorkflow,
    step: str,
    *,
    status: str = SystemWorkflow.STATUS_COMPLETED,
) -> None:
    """Move a workflow to a terminal status/step and persist it.

    Single home for the terminal transition write so every completion path
    persists the same columns (status, step, state, updated_at); callers
    that also change ``workflow.state`` assign it before calling.
    """
    _validate_workflow_step(workflow, step)
    workflow.status = status
    workflow.step = step
    workflow.save(update_fields=["status", "step", "state", "updated_at"])


def _validate_workflow_step(workflow: SystemWorkflow, step: str) -> None:
    """Refuse to persist a step the workflow's kind does not declare.

    Catches a transition wired to the wrong workflow object (or a typo'd
    step constant) at write time instead of leaving the row in a state no
    reconciler recognizes.
    """
    legal = engine.legal_steps(workflow.kind)
    if legal is not None and step not in legal:
        raise ValueError(f"illegal step {step!r} for workflow kind {workflow.kind!r}")


def _block_workflow(
    workflow: SystemWorkflow,
    error: str,
    *,
    surface_to_thread: bool = True,
    defer_surface_until_workers_stop: bool = False,
    only_if: Callable[[SystemWorkflow], bool] | None = None,
) -> bool:
    # ``only_if`` runs against the locked row so ownership checks and the block
    # remain one atomic decision. Blocking is legal from any status.
    def _block(locked: SystemWorkflow) -> bool:
        _persist_workflow_block(locked, error)
        return True

    blocked = engine.claim_workflow_transition(workflow, _block, guard=only_if, require_active=False)
    return bool(blocked)


def _persist_workflow_block(workflow: SystemWorkflow, error: str) -> None:
    """Persist a blocked transition without launching follow-up work."""
    workflow.status = SystemWorkflow.STATUS_BLOCKED
    workflow.step = STEP_BLOCKED
    workflow.state = {**workflow.state, "error": error}
    workflow.save(update_fields=["status", "step", "state", "updated_at"])


def _workflow_web_search_mode(workflow: SystemWorkflow) -> str | None:
    return _state_string(workflow, "web_search_mode") or None


def _workflow_for_instance(instance: CodexInstance) -> SystemWorkflow | None:
    if instance.workflow_id is None:
        return None
    try:
        return SystemWorkflow.objects.get(pk=instance.workflow_id)
    except SystemWorkflow.DoesNotExist:
        return None


def _system_agent_run_for_instance(instance: CodexInstance) -> SystemAgentRun | None:
    try:
        return SystemAgentRun.objects.select_related("workflow").get(instance=instance)
    except SystemAgentRun.DoesNotExist:
        pass
    if instance.workflow_id is None or not instance.agent_kind:
        return None
    try:
        workflow = SystemWorkflow.objects.get(pk=instance.workflow_id)
    except SystemWorkflow.DoesNotExist:
        return None
    run, _created = SystemAgentRun.objects.get_or_create(
        instance=instance,
        defaults={
            "workflow": workflow,
            "agent_kind": instance.agent_kind,
            "thread_id": instance.thread_id,
            "status": SystemAgentRun.STATUS_RUNNING,
            "input": _recovered_system_agent_run_input(instance),
        },
    )
    return run


def _recovered_system_agent_run_input(instance: CodexInstance) -> dict[str, Any]:
    return {"cwd": instance.cwd}


# Imported last: the kind modules register their WorkflowHandler with the
# engine and reach back into this module for the shared spawn/transition
# helpers, so they need its namespace to be fully initialized.
from hitch.main.workflows import autonomous_goals  # noqa: E402
