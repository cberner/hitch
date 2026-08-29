"""Reusable orchestration for Hitch-owned background Codex agents."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from django.db import models, transaction
from django.db.models import QuerySet
from django.utils import timezone

from hitch.main.git_support import resolved_path
from hitch.main.goals.autonomous_goal_proposal_stack import (
    _proposal_outcome_metadata,
)
from hitch.main.models import (
    CodexInstance,
    Project,
    ProposedSession,
    SessionMetadata,
    SystemAgentRun,
    SystemWorkflow,
    UserInputRequest,
)
from hitch.main.repos import (
    AutoPullError,
    AutoPullResult,
    pull_default_branch_from_origin,
    repo_root,
    same_repo_or_worktree,
)
from hitch.main.runtime import codex_pool, rollout
from hitch.main.runtime.sdk_values import is_nonbool_int
from hitch.main.sessions import lifecycle as session_lifecycle
from hitch.main.sessions import session_index
from hitch.main.sessions.pr_prompts import (
    PR_SLASH_DISPLAY_PROMPT as PR_SLASH_DISPLAY_PROMPT,
)
from hitch.main.sessions.pr_prompts import PR_SLASH_PROMPT as PR_SLASH_PROMPT
from hitch.main.workflows import engine, pr_watch
from hitch.main.workflows.workflow_state import (
    _session_metadata_from_state,
    _state_bool,
    _state_int,
    _state_string,
)

logger = logging.getLogger(__name__)

AUTONOMOUS_GOAL_AGENT_KIND: str = SystemWorkflow.KIND_AUTONOMOUS_GOAL_RUN
AUTONOMOUS_GOAL_HISTORY_SUMMARY_AGENT_KIND = "autonomous_goal_history_summary"
AUTONOMOUS_GOAL_JUDGE_AGENT_KIND = "autonomous_goal_judge"
REVIEW_WORKFLOW_DISPLAY_AUTHOR = "Review workflow"
PR_WORKFLOW_DISPLAY_AUTHOR = "PR workflow"
AUTONOMOUS_GOAL_DISPLAY_AUTHOR = "Autonomous goal agent"
AUTONOMOUS_GOAL_HISTORY_SUMMARY_DISPLAY_AUTHOR = "Autonomous goal history"
AUTONOMOUS_GOAL_JUDGE_DISPLAY_AUTHOR = "Autonomous goal judge"
AUTONOMOUS_GOAL_DELETED_ERROR = "Autonomous goal deleted by user"
AUTONOMOUS_GOAL_PROPOSAL_ACCEPTED_ERROR = "Autonomous goal proposal accepted by user"
AUTONOMOUS_GOAL_PROPOSAL_REJECTED_ERROR = "Autonomous goal proposal rejected by user"
AUTONOMOUS_GOAL_PROPOSAL_DISMISSED_ERROR = "Autonomous goal proposal dismissed by user"

WorkflowStartBlockedByArchiveError = session_lifecycle.WorkflowStartBlockedError

AUTONOMOUS_GOAL_AGENT_PROMPT_TITLE = session_index.AUTONOMOUS_GOAL_AGENT_PROMPT_TITLE
AUTONOMOUS_GOAL_JUDGE_PROMPT_TITLE = session_index.AUTONOMOUS_GOAL_JUDGE_PROMPT_TITLE
QA_SLASH_DISPLAY_PROMPT = (
    "Ask the coding agent to inspect the changes and optionally use a reviewer subagent"
)
SYSTEM_AGENT_APPROVAL_MODE = "auto_review"
AUTONOMOUS_GOAL_IMPLEMENTATION_SANDBOX_POLICY = "workspaceWrite"
STEP_USER_STEERING_RUNNING = "user_steering_running"
STEP_BLOCKED = "blocked"
STEP_REVIEW_COMPLETED = "review_completed"
STEP_PR_PROMPT_SPAWNED = "pr_prompt_spawned"
STEP_PR_PROMPT_RUNNING = "pr_prompt_running"
STEP_PR_WATCH_RUNNING = pr_watch.STEP_PR_WATCH_RUNNING
STEP_PR_WATCH_COMPLETED = pr_watch.STEP_PR_WATCH_COMPLETED
STEP_PR_READY = "pr_ready"
STEP_PR_CLOSED = "pr_closed"
STEP_PR_NO_CHANGES = "pr_no_changes"
STEP_ARCHIVED = "archived"
STEP_LOCAL_BRANCH_MERGED = "local_branch_merged"
STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING = "autonomous_goal_candidate_running"
STEP_AUTONOMOUS_GOAL_HISTORY_SUMMARIZING = "autonomous_goal_history_summarizing"
STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING = "autonomous_goal_judge_running"
STEP_AUTONOMOUS_GOAL_PROPOSED = "autonomous_goal_proposed"
STEP_AUTONOMOUS_GOAL_SKIPPED = "autonomous_goal_skipped"
PR_QA_STEERABLE_STEPS = frozenset(
    {
        STEP_USER_STEERING_RUNNING,
        STEP_PR_PROMPT_RUNNING,
        STEP_PR_WATCH_RUNNING,
    }
)
# The scheduler ticks once a minute, but the account rate-limit query that
# backs the quota pause is a remote round-trip to the Codex backend. Cache its
# verdict so the network call fires at most once per this interval regardless
# of tick cadence.


_WORKFLOW_FAILURE_OWNER_STATE_KEY = "failure_owner"
_DEFERRED_FAILURE_SURFACE_STATE_KEY = "deferred_failure_surface"
_WORKFLOW_STEERING_REVISION_STATE_KEY = "workflow_steering_revision"
_WORKFLOW_STOP_REQUESTED_STATE_KEY = "workflow_stop_requested"
_PR_PUBLICATION_INSTANCE_STATE_KEY = "pr_publication_instance"
_ARCHIVED_FROM_BLOCKED_STATE_KEY = "archived_from_blocked"
# How long a blocked PR-QA workflow lingers before it is auto-archived off the
# inbox Blocked stage. Shared by the maintenance scheduler (which applies the
# archive) and the health dashboard (which previews the same cutoff), so both
# agree on what "stale" means.
STALE_BLOCKED_AGE = timedelta(days=7)
_WORKFLOW_FAILURE_OWNER_REVIEW = "review"
_WORKFLOW_FAILURE_OWNER_PR = "pr"
REVIEW_GUIDANCE_STATE_KEY = "review_guidance"
_WORKFLOW_ROUTE_CLAIM_TIMEOUT = timedelta(minutes=10)
# A PR workflow commits its next transient step and then spawns the worker in the
# just-finished worker process (or the web request). If that process dies before
# the worker's CodexInstance row is created -- e.g. the orphan-worker reaper
# SIGKILLs its scope during a SQLite-lock storm -- the workflow zombies in that
# step with no worker and nothing to route, because there is no instance for the
# terminal-instance/turn reconcilers to find. Reconciliation recovers the
# workflow once the row is older than this window.
# The window sits well above ``_WORKFLOW_ROUTE_CLAIM_TIMEOUT`` so a slow-but-live
# spawn is never raced into a double review or a spurious failure.
_WORKFLOW_SPAWN_STALE_TIMEOUT = timedelta(minutes=15)
_ZOMBIE_TURN_STEP_MESSAGES = {
    STEP_USER_STEERING_RUNNING: "coding turn",
}
_PR_HANDOFF_STATE_KEY = "pr_handoff"
_PR_GATES_STATE_KEY = "pr_gates"
_WORKFLOW_TURN_OWNER_INDEX_STATE_KEY = "workflow_turn_owner_index"
_WORKFLOW_TURN_OWNER_STEP_STATE_KEY = "workflow_turn_owner_step"
_WORKFLOW_TURN_DEATH_RETRY_STATE_KEY = "workflow_turn_death_retries"
_WORKFLOW_TURN_DEATH_RETRY_LIMIT = 1
_WORKER_EXITED_BEFORE_COMPLETION_ERROR = (
    "worker process exited before reporting completion"
)
_LEGACY_SERVER_OVERLOADED_ERROR = (
    "Selected model is at capacity. Please try a different model."
)
AUTO_MERGE_REVIEWED_DIFF_STATE_KEY = "auto_merge_reviewed_diff"
AUTO_MERGE_REVIEWED_TARGET_SHA_STATE_KEY = "auto_merge_reviewed_target_sha"
AUTO_MERGE_SESSION_BASE_SHA_STATE_KEY = "auto_merge_session_base_sha"
AUTO_MERGE_REVIEWED_SOURCE_TREE_STATE_KEY = "auto_merge_reviewed_source_tree"
AUTO_PULL_RESULT_STATE_KEY = "auto_pull_result"
_PR_STAGE_REFRESH_TIMEOUT_SECONDS = 5
_REMOVED_PR_WORKFLOW_STEPS = frozenset({"pr_monitoring", "pr_feedback_running"})
_REMOVED_PR_WORKFLOW_ERROR = (
    "The framework-driven PR monitor was removed. Start a new session to use "
    "hitch.watch_pr."
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


def active_workflow_for_thread(main_thread_id: str, *, reconcile: bool = True) -> SystemWorkflow | None:
    if reconcile:
        reconcile_terminal_workflow_instances(main_thread_id=main_thread_id)
    return active_workflow_snapshot_for_thread(main_thread_id)


def active_workflow_snapshot_for_thread(
    main_thread_id: str,
) -> SystemWorkflow | None:
    """Read the active workflow without routing terminal workers."""
    return (
        SystemWorkflow.objects.filter(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id=main_thread_id,
            status=SystemWorkflow.STATUS_RUNNING,
        )
        .order_by("-created_at")
        .first()
    )


def workflow_accepts_steering(workflow: SystemWorkflow | None) -> bool:
    publication_owner = workflow.state.get(_PR_PUBLICATION_INSTANCE_STATE_KEY) if workflow is not None else None
    return (
        workflow is not None
        and workflow.kind == SystemWorkflow.KIND_PR_QA
        and workflow.is_active
        and workflow.step in PR_QA_STEERABLE_STEPS
        and workflow.state.get(_WORKFLOW_STOP_REQUESTED_STATE_KEY) is not True
        and not (isinstance(publication_owner, int) and not isinstance(publication_owner, bool))
    )


def reconcile_terminal_workflow_instances(*, main_thread_id: str | None = None, workflow_id: int | None = None) -> int:
    """Route terminal workflow-owned workers that missed their finish callback."""
    workflows = list(
        _running_workflows_for_reconciliation(
            main_thread_id=main_thread_id,
            workflow_id=workflow_id,
        )
    )
    reconciled = _retire_removed_pr_workflows(workflows)
    workflows = [
        workflow
        for workflow in workflows
        if workflow.step not in _REMOVED_PR_WORKFLOW_STEPS
    ]
    if workflows:
        reconciled += _reconcile_terminal_system_agent_instances(workflows)
        reconciled += _reconcile_terminal_workflow_turns(workflows)
        reconciled += _drive_orphaned_workflow_spawns(workflows)
    deferred_blocks = SystemWorkflow.objects.filter(
        status=SystemWorkflow.STATUS_BLOCKED,
        state__deferred_failure_surface=True,
    )
    if main_thread_id is not None:
        deferred_blocks = deferred_blocks.filter(main_thread_id=main_thread_id)
    if workflow_id is not None:
        deferred_blocks = deferred_blocks.filter(pk=workflow_id)
    for deferred_id in deferred_blocks.values_list("pk", flat=True):
        if _finish_deferred_workflow_block_if_settled(deferred_id):
            reconciled += 1
    return reconciled


def _retire_removed_pr_workflows(
    workflows: Iterable[SystemWorkflow],
) -> int:
    retired = 0
    for workflow in workflows:
        if (
            workflow.kind != SystemWorkflow.KIND_PR_QA
            or workflow.step not in _REMOVED_PR_WORKFLOW_STEPS
        ):
            continue
        if stop_active_workflow(
            workflow.main_thread_id,
            expected_workflow_id=workflow.pk,
        ):
            retired += 1
            continue
        if CodexInstance.objects.filter(
            workflow_id=workflow.pk,
            status__in=CodexInstance.ACTIVE_STATUSES,
        ).exists():
            continue
        if _block_workflow(workflow, _REMOVED_PR_WORKFLOW_ERROR):
            retired += 1
    return retired


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


def _respawn_or_block(
    workflow: SystemWorkflow,
    spawn: Callable[[SystemWorkflow], object],
    failure_message: str,
) -> None:
    """Re-drive a recoverable spawn, blocking the workflow if it raises."""
    try:
        spawn(workflow)
    except Exception as exc:
        _block_workflow(workflow, failure_message.format(exc=exc))


def _block_zombie_workflow_turn(workflow: SystemWorkflow) -> None:
    """Block a turn whose prompt is gone and so cannot be re-driven."""
    expected_step = workflow.step
    label = _ZOMBIE_TURN_STEP_MESSAGES[expected_step]
    _block_workflow(
        workflow,
        f"{label} never started: its spawn handler died before the worker launched. Restart the workflow to continue.",
        only_if=lambda locked: (locked.is_active and locked.step == expected_step),
    )


def _workflow_turn_settling(workflow: SystemWorkflow) -> bool:
    """True while a worker is live or a finished turn is still being routed.

    A starting/running instance is a live (or reaper-bound) worker. A terminal
    turn whose routing claim is still fresh is being handed off to its finish
    handler right now; the terminal-turn reconciler (or the original finisher)
    will advance the step. In either case the workflow is not yet a zombie.
    """
    instances = CodexInstance.objects.filter(workflow_id=workflow.pk)
    if instances.filter(status__in=CodexInstance.ACTIVE_STATUSES).exists():
        return True
    fresh_claim = timezone.now() - _WORKFLOW_ROUTE_CLAIM_TIMEOUT
    owned_indices = _workflow_turn_owned_indices(workflow)
    if not owned_indices:
        return False
    return instances.filter(
        purpose__in=(
            CodexInstance.PURPOSE_USER,
            CodexInstance.PURPOSE_SYSTEM_FEEDBACK,
        ),
        user_message_index__in=owned_indices,
        workflow_routing_started_at__gte=fresh_claim,
    ).exists()


def _workflow_turn_owned_indices(workflow: SystemWorkflow) -> tuple[int, ...]:
    """Return the reserved turn index, with legacy cursor compatibility."""
    owner_step = _state_string(workflow, _WORKFLOW_TURN_OWNER_STEP_STATE_KEY)
    owner_index = workflow.state.get(_WORKFLOW_TURN_OWNER_INDEX_STATE_KEY)
    if (
        owner_step == workflow.step
        and isinstance(owner_index, int)
        and not isinstance(owner_index, bool)
        and owner_index >= 0
    ):
        return (owner_index,)
    current_index = _state_int(workflow, "next_user_message_index") - 1
    return tuple(index for index in (current_index, current_index + 1) if index >= 0)


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
        if workflow.step == STEP_AUTONOMOUS_GOAL_HISTORY_SUMMARIZING:
            return (AUTONOMOUS_GOAL_HISTORY_SUMMARY_AGENT_KIND,)
        if workflow.step == STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING:
            return (AUTONOMOUS_GOAL_AGENT_KIND,)
        if workflow.step == STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING:
            return (AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,)
    return ()


def _reconcile_terminal_workflow_turns(workflows: list[SystemWorkflow]) -> int:
    filters: models.Q = models.Q(pk__in=[])
    has_turn_filter = False
    for workflow in workflows:
        if workflow.kind != SystemWorkflow.KIND_PR_QA:
            continue
        turn_indices = _workflow_turn_owned_indices(workflow)
        if not turn_indices:
            continue
        if workflow.step in (
            STEP_USER_STEERING_RUNNING,
            STEP_PR_PROMPT_RUNNING,
            STEP_PR_WATCH_RUNNING,
        ):
            filters |= models.Q(
                workflow_id=workflow.pk,
                purpose=CodexInstance.PURPOSE_USER,
                user_message_index__in=turn_indices,
            )
            has_turn_filter = True
    if not has_turn_filter:
        return 0
    instances = (
        CodexInstance.objects.filter(
            filters,
            status__in=(CodexInstance.STATUS_COMPLETED, CodexInstance.STATUS_FAILED),
        )
        .filter(_unclaimed_workflow_instance_filter())
        .order_by("started_at", "id")
    )
    reconciled = 0
    for instance in instances:
        if _route_terminal_workflow_instance(instance):
            reconciled += 1
    return reconciled


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


def stop_active_workflow(main_thread_id: str, *, expected_workflow_id: int | None = None) -> bool:
    deferred_workflow_id: int | None = None
    with session_lifecycle.hold(main_thread_id):
        workflow = active_workflow_snapshot_for_thread(main_thread_id)
        if workflow is None or (expected_workflow_id is not None and workflow.pk != expected_workflow_id):
            return False
        runs = list(
            workflow.agent_runs.filter(status=SystemAgentRun.STATUS_RUNNING)
            .select_related("instance")
            .order_by("-created_at")
        )
        run_instance_ids = [run.instance_id for run in runs]
        turns = list(
            CodexInstance.objects.filter(
                workflow_id=workflow.pk,
                status__in=CodexInstance.ACTIVE_STATUSES,
            )
            .exclude(pk__in=run_instance_ids)
            .order_by("-started_at", "-pk")
        )
        if not runs and not turns:
            if workflow.kind == SystemWorkflow.KIND_PR_QA and (
                workflow.step
                in (
                    STEP_USER_STEERING_RUNNING,
                    STEP_PR_PROMPT_RUNNING,
                    STEP_PR_WATCH_RUNNING,
                )
                or workflow.steering_messages.exists()
            ):
                _block_workflow(workflow, _workflow_stopped_error(workflow))
                return True
            return False
        live_runs = [run for run in runs if run.instance.is_active]
        # The claim distinguishes a user Stop from steering interrupts.
        # Persist it before signaling so a fast terminal callback cannot
        # resume the workflow while another worker remains live.
        workflow.state = {
            **workflow.state,
            _WORKFLOW_STOP_REQUESTED_STATE_KEY: True,
        }
        workflow.save(update_fields=["state", "updated_at"])
        workers = [*(run.instance for run in live_runs), *turns]
        if any(
            worker.pid <= 0 or (worker.systemd_scope_unit and worker.status == CodexInstance.STATUS_STARTING)
            for worker in workers
        ):
            # A launch has not published a safe signal target yet. Keep
            # the workflow active so the next Stop can retry it.
            return False
        error = _workflow_stopped_error(workflow)
        # Retire signalable runs before their asynchronous finish callbacks
        # can advance the workflow. If any worker cannot be signaled, put
        # only that run back and keep the workflow active so Stop remains
        # available for another attempt.
        with transaction.atomic():
            _mark_system_agent_runs_failed(runs, error)
            interrupted_runs = _interrupt_system_agent_runs(live_runs)
            interrupted_run_ids = {run.pk for run in interrupted_runs}
            interrupted_turns = [
                turn
                for turn in turns
                if codex_pool.interrupt_instance(
                    turn.pk, expected_thread_id=turn.thread_id
                )
                is not None
            ]
            interrupted_turn_ids = {turn.pk for turn in interrupted_turns}
            unresolved_run_ids = set(
                CodexInstance.objects.filter(
                    pk__in=[
                        run.instance_id
                        for run in live_runs
                        if run.pk not in interrupted_run_ids
                    ],
                    status__in=CodexInstance.ACTIVE_STATUSES,
                ).values_list("pk", flat=True)
            )
            unresolved_runs = [
                run for run in live_runs if run.instance_id in unresolved_run_ids
            ]
            unresolved_turns_exist = CodexInstance.objects.filter(
                pk__in=[
                    turn.pk
                    for turn in turns
                    if turn.pk not in interrupted_turn_ids
                ],
                workflow_id=workflow.pk,
                status__in=CodexInstance.ACTIVE_STATUSES,
            ).exists()
            if unresolved_runs or unresolved_turns_exist:
                for run in unresolved_runs:
                    run.status = SystemAgentRun.STATUS_RUNNING
                    run.error = ""
                    run.save(update_fields=["status", "error", "updated_at"])
                return False
            _block_workflow(
                workflow,
                error,
                defer_surface_until_workers_stop=True,
            )
        deferred_workflow_id = workflow.pk
    if deferred_workflow_id is not None:
        _finish_deferred_workflow_block_if_settled(deferred_workflow_id)
    return True


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
    if _settle_requested_pr_qa_stop(instance):
        return True
    if instance.purpose == CodexInstance.PURPOSE_SYSTEM_AGENT:
        handled = _handle_system_agent_finished(instance)
        _finish_deferred_workflow_block_if_settled(instance.workflow_id)
        return handled
    if instance.purpose == CodexInstance.PURPOSE_SYSTEM_FEEDBACK:
        _dispatch_workflow_event(instance, "on_feedback_finished")
        _finish_deferred_workflow_block_if_settled(instance.workflow_id)
        return True
    if instance.purpose == CodexInstance.PURPOSE_USER and instance.workflow_id is not None:
        _dispatch_workflow_event(instance, "on_user_turn_finished")
        _finish_deferred_workflow_block_if_settled(instance.workflow_id)
        return True
    _maybe_start_auto_review_workflow(instance)
    return False


def _settle_requested_pr_qa_stop(instance: CodexInstance) -> bool:
    """Retire callbacks after a partial Stop; the last worker blocks."""
    if instance.workflow_id is None:
        return False
    try:
        workflow = SystemWorkflow.objects.get(pk=instance.workflow_id)
    except SystemWorkflow.DoesNotExist:
        return False
    if workflow.kind != SystemWorkflow.KIND_PR_QA or workflow.state.get(_WORKFLOW_STOP_REQUESTED_STATE_KEY) is not True:
        return False
    settled_workflow_id: int | None = None
    with session_lifecycle.hold(workflow.main_thread_id):
        workflow.refresh_from_db()
        if not workflow.is_active or workflow.state.get(_WORKFLOW_STOP_REQUESTED_STATE_KEY) is not True:
            return True
        if CodexInstance.objects.filter(
            workflow_id=workflow.pk,
            status__in=CodexInstance.ACTIVE_STATUSES,
        ).exists():
            return True
        error = _workflow_stopped_error(workflow)
        with transaction.atomic():
            remaining_runs = list(workflow.agent_runs.filter(status=SystemAgentRun.STATUS_RUNNING))
            _mark_system_agent_runs_failed(remaining_runs, error)
            _block_workflow(
                workflow,
                error,
                defer_surface_until_workers_stop=True,
            )
        settled_workflow_id = workflow.pk
    _finish_deferred_workflow_block_if_settled(settled_workflow_id)
    return True


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
        CodexInstance.PURPOSE_USER,
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
                CodexInstance.PURPOSE_USER,
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


def _maybe_start_auto_review_workflow(instance: CodexInstance, *, lifecycle_lock_held: bool = False) -> None:
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
        from hitch.main.sessions.session_resume import thread_has_dynamic_tool

        workflow_kwargs: dict[str, Any] = {
            "main_thread_id": instance.thread_id,
            "cwd": instance.cwd,
            "sandbox_policy": instance.sandbox_policy or None,
            "approval_mode": instance.approval_mode or SYSTEM_AGENT_APPROVAL_MODE,
            "model": instance.model or None,
            "reasoning_effort": instance.reasoning_effort or None,
            "developer_instructions": instance.developer_instructions or None,
            "enable_memories": instance.enable_memories,
            "web_search_mode": instance.web_search_mode or None,
            "initial_user_message_index": (instance.user_message_index or 0) + 1,
            "pr_watch_tool_available": thread_has_dynamic_tool(
                instance.thread_id,
                namespace="hitch",
                name="watch_pr",
            ),
        }
        if automation == "auto_pr":
            pr_title = _accepted_auto_pr_proposal_title(instance.thread_id)
            if pr_title:
                workflow_kwargs["pr_title"] = pr_title
        if automation == "auto_qa":
            workflow_kwargs["open_pr_on_lgtm"] = False
        auto_merge_branch = instance.auto_merge_branch.strip() if instance.auto_merge_to_local_branch else ""
        if auto_merge_branch:
            workflow_kwargs["open_pr_on_lgtm"] = False
            workflow_kwargs["auto_merge_branch"] = auto_merge_branch
        if lifecycle_lock_held:
            workflow_kwargs["lifecycle_lock_held"] = True
        workflow = pr_qa.start_pr_qa_workflow(**workflow_kwargs)
        if isinstance(workflow, SystemWorkflow):
            _record_auto_review_workflow_for_proposals(instance, workflow, automation=automation)
    except WorkflowStartBlockedByArchiveError:
        CodexInstance.objects.filter(pk=instance.pk).update(**{trigger_field: None})
        return
    except pr_qa.PrWatchUnavailableError:
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
        _maybe_start_auto_review_workflow(instance, lifecycle_lock_held=lifecycle_lock_held)


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

    ``_maybe_start_auto_review_workflow`` returns without claiming a trigger when
    the turn ends with a pending proposed plan, so a null trigger timestamp is
    expected rather than a dropped follow-up. The orphan reaper uses this so it
    does not rewrite such an intentionally-skipped successful turn as failed.
    """
    return _completed_turn_has_pending_proposed_plan(instance) or _auto_pr_watch_unavailable(instance)


def _auto_pr_watch_unavailable(instance: CodexInstance) -> bool:
    if (
        not instance.auto_pr_enabled
        or (
            instance.auto_merge_to_local_branch
            and bool(instance.auto_merge_branch.strip())
        )
    ):
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


def _record_auto_review_workflow_for_proposals(
    instance: CodexInstance, workflow: SystemWorkflow, *, automation: str
) -> None:
    metadata = SessionMetadata.objects.filter(thread_id=instance.thread_id).first()
    if metadata is None:
        return
    if automation == "auto_qa":
        base_updates: dict[str, object] = {
            "auto_qa_status": "started",
            "auto_qa_workflow_id": workflow.pk,
        }
    else:
        base_updates = {
            "auto_pr_status": "started",
            "auto_pr_workflow_id": workflow.pk,
        }
    for proposal in ProposedSession.objects.filter(accepted_session=metadata):
        updates = dict(base_updates)
        auto_merge_branch = _state_string(workflow, "auto_merge_branch")
        if auto_merge_branch:
            updates["auto_merge_branch"] = auto_merge_branch
            if workflow.status == SystemWorkflow.STATUS_BLOCKED:
                updates["auto_merge_status"] = "failed"
                updates["auto_merge_error"] = _state_string(workflow, "error")
            else:
                updates["auto_merge_status"] = "qa_started"
        proposal.outcome_metadata = _proposal_outcome_metadata(
            proposal,
            updates,
        )
        proposal.save(update_fields=["outcome_metadata", "updated_at"])


def _record_auto_merge_result_for_proposals(workflow: SystemWorkflow, updates: dict[str, object]) -> None:
    metadata = SessionMetadata.objects.filter(thread_id=workflow.main_thread_id).first()
    if metadata is None:
        return
    for proposal in ProposedSession.objects.filter(accepted_session=metadata):
        proposal.outcome_metadata = _proposal_outcome_metadata(proposal, updates)
        proposal.save(update_fields=["outcome_metadata", "updated_at"])


def _maybe_auto_pull_default_repo_after_pr_merge(
    workflow: SystemWorkflow,
) -> None:
    if workflow.step != STEP_PR_CLOSED:
        return
    project = _auto_pull_project_for_workflow(workflow)
    if project is None or not project.auto_pull_enabled:
        return
    skip_reason = _auto_pull_skip_reason(workflow, project)
    if skip_reason:
        _record_auto_pull_result(
            workflow,
            {
                "status": "skipped",
                "reason": skip_reason,
            },
        )
        return
    _record_auto_pull_result(workflow, {"status": "running"})
    try:
        result = pull_default_branch_from_origin(project.repo_path)
    except AutoPullError as exc:
        logger.warning(
            "auto-pull failed for project %s after workflow %s: %s",
            project.pk,
            workflow.pk,
            exc,
        )
        _record_auto_pull_result(
            workflow,
            {
                "status": "failed",
                "error": str(exc),
            },
        )
        return
    except Exception as exc:
        logger.exception(
            "unexpected auto-pull failure for project %s after workflow %s",
            project.pk,
            workflow.pk,
        )
        _record_auto_pull_result(
            workflow,
            {
                "status": "failed",
                "error": str(exc),
            },
        )
        return
    _record_auto_pull_result(workflow, _auto_pull_result_dict(result))


def _auto_pull_project_for_workflow(workflow: SystemWorkflow) -> Project | None:
    metadata = SessionMetadata.objects.select_related("project").filter(thread_id=workflow.main_thread_id).first()
    if metadata is None:
        return None
    return metadata.project


def _auto_pull_skip_reason(workflow: SystemWorkflow, project: Project) -> str:
    cwd = workflow.cwd.strip()
    if not cwd:
        return "workflow checkout is unavailable"
    if _same_checkout(cwd, project.repo_path):
        return "default checkout is the active session checkout"
    if not same_repo_or_worktree(cwd, project.repo_path, project.git_common_dir):
        return "project repository does not match workflow checkout"
    return ""


def _same_checkout(cwd: str, repo_path: str) -> bool:
    cwd_root = repo_root(cwd)
    cwd_path = cwd_root if cwd_root is not None else Path(cwd).expanduser()
    return resolved_path(cwd_path) == resolved_path(Path(repo_path).expanduser())


def _auto_pull_result_dict(result: AutoPullResult) -> dict[str, object]:
    return {
        "status": "pulled" if result.changed else "up_to_date",
        "branch": result.branch,
        "before_sha": result.before_sha,
        "after_sha": result.after_sha,
        "changed": result.changed,
    }


def _record_auto_pull_result(workflow: SystemWorkflow, result: dict[str, object]) -> None:
    try:
        with transaction.atomic():
            locked = SystemWorkflow.objects.select_for_update().get(pk=workflow.pk)
            locked.state = {
                **locked.state,
                AUTO_PULL_RESULT_STATE_KEY: result,
            }
            locked.save(update_fields=["state"])
            workflow.state = locked.state
    except Exception:
        logger.exception("failed to record auto-pull result for workflow %s", workflow.pk)


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


def _handle_workflow_user_turn_finished(instance: CodexInstance) -> None:
    workflow = _workflow_for_instance(instance)
    if workflow is None or workflow.kind != SystemWorkflow.KIND_PR_QA:
        return
    if not workflow.is_active:
        return
    if workflow.step == STEP_USER_STEERING_RUNNING:
        pr_qa._handle_user_steering_finished(instance, workflow)
        return
    if workflow.step == STEP_PR_PROMPT_RUNNING:
        pr_qa._handle_pr_prompt_finished(instance, workflow)
        return
    if workflow.step == STEP_PR_WATCH_RUNNING:
        pr_qa._handle_pr_watch_finished(instance, workflow)


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
    for key in ("candidate", "judgment", "judge_session_id", "history_files"):
        next_state.pop(key, None)
    return next_state


def _candidate_session_cwd_from_state(workflow: SystemWorkflow, key: str) -> str:
    metadata = _session_metadata_from_state(workflow, key)
    return metadata.cwd if metadata is not None else ""


def _spawn_workflow_failure_turn(workflow: SystemWorkflow, error: str) -> CodexInstance:
    headline, display_author = _workflow_failure_turn_context(workflow, error)
    workflow_label = (
        "review workflow"
        if _workflow_failure_owner(workflow, error)
        == _WORKFLOW_FAILURE_OWNER_REVIEW
        else "PR workflow"
    )
    return _spawn_workflow_turn(
        workflow,
        prompt=(
            f"{headline}\n\n"
            f"Status: {error}\n\n"
            f"Tell the user the {workflow_label} needs attention before continuing."
        ),
        purpose=CodexInstance.PURPOSE_SYSTEM_FEEDBACK,
        display_author=display_author,
    )


def _workflow_failure_turn_context(
    workflow: SystemWorkflow, error: str
) -> tuple[str, str]:
    owner = _workflow_failure_owner(workflow, error)
    if owner == _WORKFLOW_FAILURE_OWNER_REVIEW:
        return (
            "Hitch review workflow could not complete.",
            REVIEW_WORKFLOW_DISPLAY_AUTHOR,
        )
    return "Hitch PR workflow could not complete.", PR_WORKFLOW_DISPLAY_AUTHOR


def _workflow_failure_owner(workflow: SystemWorkflow, error: str) -> str:
    stored_owner = workflow.state.get(_WORKFLOW_FAILURE_OWNER_STATE_KEY)
    if stored_owner in {
        _WORKFLOW_FAILURE_OWNER_REVIEW,
        _WORKFLOW_FAILURE_OWNER_PR,
    }:
        return str(stored_owner)
    if is_review_guidance_only_workflow(workflow):
        return _WORKFLOW_FAILURE_OWNER_REVIEW
    return _WORKFLOW_FAILURE_OWNER_PR


def is_review_guidance_only_workflow(workflow: SystemWorkflow) -> bool:
    return (
        workflow.state.get("open_pr_on_lgtm", True) is not True
        and workflow.state.get(REVIEW_GUIDANCE_STATE_KEY) is True
    )


def _workflow_stopped_error(workflow: SystemWorkflow) -> str:
    if workflow.step in _REMOVED_PR_WORKFLOW_STEPS:
        return _REMOVED_PR_WORKFLOW_ERROR
    if is_review_guidance_only_workflow(workflow):
        return "Review workflow stopped by user"
    if workflow.state.get(REVIEW_GUIDANCE_STATE_KEY) is True:
        return "PR workflow stopped by user"
    return "PR workflow stopped by user"


def _spawn_workflow_turn(
    workflow: SystemWorkflow,
    *,
    prompt: str,
    purpose: str = CodexInstance.PURPOSE_USER,
    display_author: str = "",
    agent_kind: str = "",
    user_message_index: int | None = None,
    additional_developer_instructions: str = "",
) -> CodexInstance:
    next_user_message_index = _state_int(workflow, "next_user_message_index")
    if user_message_index is None:
        user_message_index = next_user_message_index
    workflow.state = {
        **workflow.state,
        _WORKFLOW_TURN_OWNER_INDEX_STATE_KEY: user_message_index,
        _WORKFLOW_TURN_OWNER_STEP_STATE_KEY: workflow.step,
    }
    workflow.save(update_fields=["state", "updated_at"])
    developer_instructions = "\n\n".join(
        instruction
        for instruction in (
            _state_string(workflow, "developer_instructions"),
            additional_developer_instructions,
        )
        if instruction
    )
    instance = codex_pool.spawn_turn(
        thread_id=workflow.main_thread_id,
        cwd=workflow.cwd,
        prompt=prompt,
        model=_state_string(workflow, "model") or None,
        reasoning_effort=_state_string(workflow, "reasoning_effort") or None,
        developer_instructions=developer_instructions or None,
        sandbox_policy=_state_string(workflow, "sandbox_policy") or None,
        approval_mode=_state_string(workflow, "approval_mode") or None,
        enable_memories=_state_bool(workflow, "enable_memories"),
        web_search_mode=_workflow_web_search_mode(workflow),
        purpose=purpose,
        workflow_id=workflow.pk,
        agent_kind=agent_kind,
        display_author=display_author,
        user_message_index=user_message_index,
    )
    workflow.state = {
        **workflow.state,
        "next_user_message_index": max(next_user_message_index, user_message_index + 1),
    }
    workflow.save(update_fields=["state", "updated_at"])
    return instance


def _interrupt_system_agent_runs(runs: list[SystemAgentRun]) -> list[SystemAgentRun]:
    interrupted_runs: list[SystemAgentRun] = []
    for run in runs:
        interrupted = codex_pool.interrupt_instance(run.instance_id, expected_thread_id=run.thread_id)
        if interrupted is not None:
            interrupted_runs.append(run)
    return interrupted_runs


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


def archive_stale_blocked_workflows(*, older_than: datetime, apply: bool) -> list[int]:
    """Archive blocked PR-QA workflows last updated before ``older_than``.

    Historical failures keep surfacing as a Blocked stage in the session inbox
    long after their root cause was fixed. Move stale blocked rows to a terminal
    completed state with the ``archived`` step (which maps to no inbox stage) so
    they stop being flagged, recording a sentinel in ``state`` for auditing.

    Only ``KIND_PR_QA`` workflows drive the inbox Blocked stage, so other kinds
    (e.g. autonomous goal runs, whose UI still reports their blocked state) are
    left untouched.

    With ``apply=False`` nothing is written; the matching workflow ids are still
    returned so callers can preview the cleanup. Returns the affected ids in pk
    order.
    """
    workflows = SystemWorkflow.objects.filter(
        kind=SystemWorkflow.KIND_PR_QA,
        status=SystemWorkflow.STATUS_BLOCKED,
        updated_at__lt=older_than,
    ).order_by("pk")
    archived_ids: list[int] = []
    for workflow in workflows:
        archived_ids.append(workflow.pk)
        if not apply:
            continue
        # Use update() rather than save() so ``updated_at`` (auto_now) is left
        # as-is: the session list orders threads by -updated_at, so bumping it to
        # now would let this archived row shadow a newer/running workflow on the
        # same thread instead of merely dropping the stale Blocked badge.
        SystemWorkflow.objects.filter(pk=workflow.pk).update(
            status=SystemWorkflow.STATUS_COMPLETED,
            step=STEP_ARCHIVED,
            state={**workflow.state, _ARCHIVED_FROM_BLOCKED_STATE_KEY: True},
        )
    return archived_ids


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
    # ``only_if`` runs against the locked row so a caller whose claim on the
    # workflow may have been superseded (a stale QA verdict racing a user
    # steering claim) makes the ownership check and the block one atomic
    # decision; returns whether the workflow was blocked. Blocking is legal
    # from any status, so the claim does not require an active row.
    def _block(locked: SystemWorkflow) -> bool:
        if defer_surface_until_workers_stop:
            locked.state = {
                **locked.state,
                _DEFERRED_FAILURE_SURFACE_STATE_KEY: True,
            }
        _persist_workflow_block(locked, error)
        return True

    blocked = engine.claim_workflow_transition(workflow, _block, guard=only_if, require_active=False)
    if not blocked:
        return False
    if defer_surface_until_workers_stop:
        return True
    if workflow.state.get(_DEFERRED_FAILURE_SURFACE_STATE_KEY) is True:
        _finish_deferred_workflow_block_if_settled(workflow.pk)
        return True
    _finish_workflow_block(workflow, error, surface_to_thread=surface_to_thread)
    return True


def _finish_deferred_workflow_block_if_settled(
    workflow_id: int | None,
) -> bool:
    """Surface a stopped workflow only after all of its workers have exited."""
    if workflow_id is None:
        return False

    def _claim(locked: SystemWorkflow) -> tuple[bool, str]:
        if (
            locked.status != SystemWorkflow.STATUS_BLOCKED
            or locked.state.get(_DEFERRED_FAILURE_SURFACE_STATE_KEY) is not True
            or CodexInstance.objects.filter(
                workflow_id=locked.pk,
                status__in=CodexInstance.ACTIVE_STATUSES,
            ).exists()
        ):
            return False, ""
        newer_session_work = (
            CodexInstance.objects.filter(
                thread_id=locked.main_thread_id,
            )
            .exclude(workflow_id=locked.pk)
            .filter(models.Q(status__in=CodexInstance.ACTIVE_STATUSES) | models.Q(started_at__gt=locked.updated_at))
            .exists()
            or SystemWorkflow.objects.filter(
                main_thread_id=locked.main_thread_id,
            )
            .exclude(pk=locked.pk)
            .filter(models.Q(status=SystemWorkflow.STATUS_RUNNING) | models.Q(created_at__gt=locked.updated_at))
            .exists()
        )
        error = _state_string(locked, "error")
        locked.state = dict(locked.state)
        locked.state.pop(_DEFERRED_FAILURE_SURFACE_STATE_KEY, None)
        locked.save(update_fields=(["state"] if newer_session_work else ["state", "updated_at"]))
        return True, "" if newer_session_work else error

    workflow = SystemWorkflow.objects.filter(pk=workflow_id).first()
    if workflow is None:
        return False
    with session_lifecycle.hold(workflow.main_thread_id):
        result = engine.claim_workflow_transition(
            workflow,
            _claim,
            require_active=False,
        )
        if result is None:
            return False
        settled, error = result
        if not settled:
            return False
        if error:
            _finish_workflow_block(workflow, error)
        return True


def _persist_workflow_block(workflow: SystemWorkflow, error: str) -> None:
    """Persist a blocked transition without launching follow-up work."""
    failure_owner = _workflow_failure_owner(workflow, error)
    workflow.status = SystemWorkflow.STATUS_BLOCKED
    workflow.step = STEP_BLOCKED
    workflow.state = {
        **workflow.state,
        "error": error,
        _WORKFLOW_FAILURE_OWNER_STATE_KEY: failure_owner,
    }
    workflow.state.pop(_WORKFLOW_STOP_REQUESTED_STATE_KEY, None)
    workflow.save(update_fields=["status", "step", "state", "updated_at"])
    workflow.steering_messages.all().delete()


def _finish_workflow_block(
    workflow: SystemWorkflow,
    error: str,
    *,
    surface_to_thread: bool = True,
) -> None:
    """Run side effects for an already committed blocked transition."""
    if surface_to_thread:
        _surface_workflow_failure(workflow, error)


def _surface_workflow_failure(workflow: SystemWorkflow, error: str) -> None:
    # Make the check-then-set atomic per workflow so concurrent failure routes
    # cannot double-post the failure message or double-increment the user
    # message index.
    def _mark_surfaced(locked: SystemWorkflow) -> bool:
        failure_owner = _workflow_failure_owner(locked, error)
        locked.state = {
            **locked.state,
            "failure_surfaced": True,
            _WORKFLOW_FAILURE_OWNER_STATE_KEY: failure_owner,
        }
        locked.save(update_fields=["state", "updated_at"])
        return True

    claimed = engine.claim_workflow_transition(
        workflow,
        _mark_surfaced,
        guard=lambda locked: locked.state.get("failure_surfaced") is not True,
        require_active=False,
    )
    if not claimed:
        return
    try:
        _spawn_workflow_failure_turn(workflow, error)
    except Exception:
        logger.exception("failed to surface system workflow failure for workflow %s", workflow.pk)


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
from hitch.main.workflows import autonomous_goals, pr_qa  # noqa: E402
