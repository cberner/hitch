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

from hitch.main.goals.autonomous_goal_proposal_stack import (
    _proposal_outcome_metadata,
)
from hitch.main.models import (
    CodexInstance,
    ProposedSession,
    SessionMetadata,
    SystemAgentRun,
    SystemWorkflow,
    UserInputRequest,
)
from hitch.main.runtime import codex_pool, rollout
from hitch.main.runtime.sdk_values import is_nonbool_int
from hitch.main.sessions import agent_tasks, session_index
from hitch.main.sessions import lifecycle as session_lifecycle
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



class _AutoReviewDeferredError(RuntimeError):
    pass

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
_WORKFLOW_TURN_DEATH_RETRY_STATE_KEY = "workflow_turn_death_retries"
_WORKFLOW_TURN_DEATH_RETRY_LIMIT = 1
_WORKER_EXITED_BEFORE_COMPLETION_ERROR = (
    "worker process exited before reporting completion"
)
_LEGACY_SERVER_OVERLOADED_ERROR = (
    "Selected model is at capacity. Please try a different model."
)
def _sync_workflow_instance(target: SystemWorkflow, source: SystemWorkflow) -> None:
    target.status = source.status
    target.step = source.step
    target.state = source.state


def accepted_visible_system_thread_ids() -> set[str]:
    return set(
        ProposedSession.objects.filter(
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            candidate_session__isnull=False,
            accepted_session=models.F("candidate_session"),
        ).values_list("candidate_session__thread_id", flat=True)
    )


def hidden_thread_ids(*, accepted_visible_thread_ids: set[str] | None = None) -> set[str]:
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
    if accepted_visible_thread_ids is None:
        accepted_visible_thread_ids = accepted_visible_system_thread_ids()
    return hidden_ids - accepted_visible_thread_ids


def hidden_thread_ids_from_threads(
    threads: Iterable[Any], *, accepted_visible_thread_ids: set[str] | None = None
) -> set[str]:
    hidden_ids = {
        thread_id
        for thread in threads
        if isinstance(thread_id := getattr(thread, "id", None), str) and hitch_system_agent_thread(thread)
    }
    if accepted_visible_thread_ids is None:
        accepted_visible_thread_ids = accepted_visible_system_thread_ids()
    return hidden_ids - accepted_visible_thread_ids


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
    _maybe_start_auto_review_task(instance)
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


def _maybe_start_auto_review_task(
    instance: CodexInstance, *, lifecycle_lock_held: bool = False
) -> None:
    if (
        instance.purpose != CodexInstance.PURPOSE_USER
        or instance.workflow_id is not None
        or not (instance.auto_pr_enabled or instance.auto_qa_enabled)
        or instance.plan_mode
        or instance.status != CodexInstance.STATUS_COMPLETED
    ):
        return
    if _completed_turn_has_pending_proposed_plan(instance):
        return
    automation = "auto_pr" if instance.auto_pr_enabled else "auto_qa"
    trigger_field = "auto_pr_triggered_at" if automation == "auto_pr" else "auto_qa_triggered_at"
    claimed = CodexInstance.objects.filter(
        pk=instance.pk,
        **{f"{trigger_field}__isnull": True},
    ).update(**{trigger_field: timezone.now()})
    if not claimed:
        return
    try:
        if automation == "auto_pr" and _auto_pr_watch_unavailable(instance):
            raise agent_tasks.PrWatchUnavailableError
        task = agent_tasks.review_task(
            prepare_pull_request=automation == "auto_pr",
            pr_title=(
                _accepted_auto_pr_proposal_title(instance.thread_id)
                if automation == "auto_pr"
                else ""
            ),
        )
        task_kwargs: dict[str, Any] = {
            "thread_id": instance.thread_id,
            "cwd": instance.cwd,
            "prompt": task.prompt,
            "sandbox_policy": instance.sandbox_policy or None,
            "approval_mode": instance.approval_mode or SYSTEM_AGENT_APPROVAL_MODE,
            "model": instance.model or None,
            "stored_model": instance.model or None,
            "reasoning_effort": instance.reasoning_effort or None,
            "stored_reasoning_effort": instance.reasoning_effort or None,
            "developer_instructions": instance.developer_instructions or None,
            "enable_memories": instance.enable_memories,
            "web_search_mode": instance.web_search_mode or None,
            "user_message_index": (
                instance.user_message_index + 1
                if instance.user_message_index is not None
                else None
            ),
            "agent_kind": task.agent_kind,
        }

        def spawn_task() -> CodexInstance:
            archived = SessionMetadata.objects.filter(
                thread_id=instance.thread_id,
                codex_archived=True,
            ).exists()
            active_turn = CodexInstance.objects.filter(
                thread_id=instance.thread_id,
                status__in=CodexInstance.ACTIVE_STATUSES,
            ).exists()
            if archived or active_turn:
                raise _AutoReviewDeferredError(
                    f"session {instance.thread_id} is archived or already running work"
                )
            return codex_pool.spawn_turn(**task_kwargs)

        if lifecycle_lock_held:
            task_instance = spawn_task()
        else:
            with session_lifecycle.hold(instance.thread_id):
                task_instance = spawn_task()
        _record_auto_review_task_for_proposals(
            instance,
            task_instance,
            automation=automation,
        )
    except _AutoReviewDeferredError:
        CodexInstance.objects.filter(pk=instance.pk).update(**{trigger_field: None})
        return
    except agent_tasks.PrWatchUnavailableError:
        CodexInstance.objects.filter(pk=instance.pk).update(**{trigger_field: None})
        return
    except Exception:
        CodexInstance.objects.filter(pk=instance.pk).update(**{trigger_field: None})
        raise


def retry_deferred_auto_review_for_thread(thread_id: str, *, lifecycle_lock_held: bool = False) -> None:
    """Retry the latest auto-review deferred while its session was archived."""
    instance = (
        CodexInstance.objects.filter(
            thread_id=thread_id,
            purpose=CodexInstance.PURPOSE_USER,
            workflow_id__isnull=True,
            status=CodexInstance.STATUS_COMPLETED,
            plan_mode=False,
        )
        .filter(
            models.Q(auto_pr_enabled=True, auto_pr_triggered_at__isnull=True)
            | models.Q(
                auto_pr_enabled=False,
                auto_qa_enabled=True,
                auto_qa_triggered_at__isnull=True,
            )
        )
        .order_by("-started_at", "-pk")
        .first()
    )
    if instance is not None:
        _maybe_start_auto_review_task(instance, lifecycle_lock_held=lifecycle_lock_held)


def _accepted_auto_pr_proposal_title(thread_id: str) -> str:
    proposal = (
        ProposedSession.objects.filter(
            accepted_session__thread_id=thread_id,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            outcome_metadata__auto_pr_enabled=True,
        )
        .order_by("-updated_at", "-pk")
        .first()
    )
    if proposal is None:
        return ""
    return " ".join(proposal.title.split())


def auto_review_intentionally_skipped(instance: CodexInstance) -> bool:
    """Whether auto-PR/QA would decline for this completed turn by design.

    ``_maybe_start_auto_review_task`` returns without claiming a trigger when
    the turn ends with a pending proposed plan, so a null trigger timestamp is
    expected rather than a dropped follow-up. The orphan reaper uses this so it
    does not rewrite such an intentionally-skipped successful turn as failed.
    """
    return _completed_turn_has_pending_proposed_plan(instance) or _auto_pr_watch_unavailable(instance)


def _auto_pr_watch_unavailable(instance: CodexInstance) -> bool:
    if not instance.auto_pr_enabled:
        return False
    from hitch.main.sessions.session_resume import thread_has_dynamic_tool

    return not thread_has_dynamic_tool(
        instance.thread_id,
        namespace="hitch",
        name="watch_pr",
    )


def auto_review_waits_for_unarchive(instance: CodexInstance) -> bool:
    """Whether a completed turn's unclaimed auto-review is durably deferred."""
    return SessionMetadata.objects.filter(
        thread_id=instance.thread_id,
        codex_archived=True,
    ).exists()


def _completed_turn_has_pending_proposed_plan(instance: CodexInstance) -> bool:
    rollout_pending = _thread_rollout_has_pending_plan(instance.thread_id)
    if rollout_pending is not None:
        return rollout_pending
    final_text = _final_agent_text(instance.events_path)
    plan_text = rollout.proposed_plan_text_from_agent_text(final_text) if final_text else None
    return plan_text is not None and rollout.looks_like_plan_text(plan_text)


def _thread_rollout_has_pending_plan(thread_id: str) -> bool | None:
    metadata = SessionMetadata.objects.filter(thread_id=thread_id).first()
    if metadata is None or not metadata.codex_path:
        return None
    entries = list(rollout.iter_entries(Path(metadata.codex_path)))
    if not entries:
        return None
    return rollout.entries_await_plan_approval(entries)


def _record_auto_review_task_for_proposals(
    instance: CodexInstance, task_instance: CodexInstance, *, automation: str
) -> None:
    metadata = SessionMetadata.objects.filter(thread_id=instance.thread_id).first()
    if metadata is None:
        return
    if automation == "auto_qa":
        base_updates: dict[str, object] = {
            "auto_qa_status": "started",
            "auto_qa_instance_id": task_instance.pk,
        }
    else:
        base_updates = {
            "auto_pr_status": "started",
            "auto_pr_instance_id": task_instance.pk,
        }
    for proposal in ProposedSession.objects.filter(accepted_session=metadata):
        proposal.outcome_metadata = _proposal_outcome_metadata(
            proposal,
            base_updates,
        )
        proposal.save(update_fields=["outcome_metadata", "updated_at"])


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


def _claim_workflow_turn_retry(
    workflow: SystemWorkflow,
    instance: CodexInstance,
    retry_kind: str,
) -> bool:
    """Claim one bounded retry for a transient autonomous-workflow failure."""
    if (
        not workflow.is_active
        or not retry_kind
        or _instance_interrupt_requested(instance)
        or not _is_retryable_workflow_turn_error(instance)
    ):
        return False
    retries = _workflow_turn_death_retries(workflow.state)
    retry_count = retries.get(retry_kind, 0)
    if retry_count >= _WORKFLOW_TURN_DEATH_RETRY_LIMIT:
        return False
    workflow.state = {
        **workflow.state,
        _WORKFLOW_TURN_DEATH_RETRY_STATE_KEY: {
            **retries,
            retry_kind: retry_count + 1,
        },
    }
    workflow.save(update_fields=["state", "updated_at"])
    return True


def _instance_interrupt_requested(instance: CodexInstance) -> bool:
    """Return the latest Stop state even when the routed worker object is stale."""
    if instance.interrupt_requested_at is not None:
        return True
    return CodexInstance.objects.filter(
        pk=instance.pk,
        interrupt_requested_at__isnull=False,
    ).exists()


def _workflow_turn_death_retries(state: Mapping[str, Any]) -> dict[str, int]:
    raw = state.get(_WORKFLOW_TURN_DEATH_RETRY_STATE_KEY)
    if not isinstance(raw, dict):
        return {}
    return {
        str(key): value
        for key, value in raw.items()
        if is_nonbool_int(value) and value > 0
    }


def _state_without_workflow_turn_death_retry(
    state: Mapping[str, Any], retry_kind: str
) -> dict[str, Any]:
    retries = _workflow_turn_death_retries(state)
    if retry_kind not in retries:
        return dict(state)
    retries.pop(retry_kind, None)
    updated = dict(state)
    if retries:
        updated[_WORKFLOW_TURN_DEATH_RETRY_STATE_KEY] = retries
    else:
        updated.pop(_WORKFLOW_TURN_DEATH_RETRY_STATE_KEY, None)
    return updated


def _is_retryable_workflow_turn_error(instance: CodexInstance) -> bool:
    error_info: object = instance.codex_error_info
    if error_info is not None:
        return error_info == CodexInstance.CODEX_ERROR_SERVER_OVERLOADED
    normalized = instance.error.strip()
    return normalized.startswith(
        _WORKER_EXITED_BEFORE_COMPLETION_ERROR
    ) or normalized == _LEGACY_SERVER_OVERLOADED_ERROR


def _run_gh_observation_fallback(run: SystemAgentRun) -> dict[str, Any]:
    run_input = run.input if isinstance(run.input, dict) else {}
    gh_observation = run_input.get("gh_observation")
    return gh_observation if isinstance(gh_observation, dict) else {}


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


def _advance_workflow_step(workflow: SystemWorkflow, step: str, *, bump_iteration: bool = False) -> None:
    """Advance a running workflow to its next transient step and persist it.

    Counterpart of _complete_workflow for non-terminal transitions; callers
    that also change ``workflow.state`` assign it before calling.
    """
    _validate_workflow_step(workflow, step)
    update_fields = ["step", "state", "updated_at"]
    if bump_iteration:
        workflow.iteration += 1
        update_fields.insert(0, "iteration")
    workflow.step = step
    workflow.save(update_fields=update_fields)


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


def _question_for_user_input(question: dict[str, Any]) -> dict[str, Any]:
    required = question.get("required") is True
    safe_default = question.get("safe_default")
    has_safe_default = (
        question.get("allow_safe_default") is True and isinstance(safe_default, str) and bool(safe_default.strip())
    )
    return {
        "id": question["id"],
        "header": question.get("header") or question["id"],
        "question": question.get("question") or question["id"],
        "required": required,
        "requires_explicit_choice": required and not has_safe_default,
        "options": question.get("options") or [],
    }


def _answers_from_input_request(input_request: UserInputRequest) -> dict[str, Any]:
    response = input_request.response if isinstance(input_request.response, dict) else {}
    answers = response.get("answers")
    return answers if isinstance(answers, dict) else {}


def _answer_is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list | tuple | dict):
        return bool(value)
    return True


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
