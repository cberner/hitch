"""Reusable orchestration for Hitch-owned background Codex agents."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, override

from django.db import models, transaction
from django.db.models import QuerySet
from django.utils import timezone

from hitch.main import demo
from hitch.main.diffs import build_worktree_diff_text
from hitch.main.git_support import resolved_path
from hitch.main.goals.autonomous_goal_proposal_stack import (
    _proposal_outcome_metadata,
)
from hitch.main.local_merges import (
    build_auto_merge_review_patch,
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
from hitch.main.runtime.sdk_values import (
    is_nonbool_int,
    string_from_any,
)
from hitch.main.sessions import session_index
from hitch.main.workflows import engine
from hitch.main.workflows.agent_io import (
    SPEC_SYNTHESIZER_AGENT_KIND,
)
from hitch.main.workflows.gh_cli import (
    _GH_PR_MONITOR_TIMEOUT_SECONDS,
    _GH_REVIEW_THREAD_PAGE_LIMIT,
    _GH_STATUS_CHECK_PAGE_LIMIT,
)
from hitch.main.workflows.pr_handoff import (
    _PR_HANDOFF_BOOLEAN_FIELDS,
    _PR_HANDOFF_FIELDS,
    _PR_HANDOFF_INTEGER_FIELDS,
    _PR_HANDOFF_LIST_FIELDS,
    _PR_SAFE_LIST_ITEM_FIELDS,
)
from hitch.main.workflows.spec_critic_prompts import (
    _SPEC_CRITIC_ANALYSIS_AGENT_KINDS,
)
from hitch.main.workflows.workflow_state import (
    _session_metadata_from_state,
    _state_bool,
    _state_int,
    _state_string,
)

logger = logging.getLogger(__name__)

PR_QA_AGENT_KIND = "pr_qa"
PR_FOLLOWUP_MONITOR_AGENT_KIND = "pr_followup_monitor"
AUTONOMOUS_GOAL_AGENT_KIND = SystemWorkflow.KIND_AUTONOMOUS_GOAL_RUN
AUTONOMOUS_GOAL_HISTORY_SUMMARY_AGENT_KIND = "autonomous_goal_history_summary"
AUTONOMOUS_GOAL_JUDGE_AGENT_KIND = "autonomous_goal_judge"
SPEC_CRITIC_WORKFLOW_KIND = "spec_critic"
QA_DISPLAY_AUTHOR = "QA agent"
PR_WORKFLOW_DISPLAY_AUTHOR = "PR workflow"
PR_MONITOR_DISPLAY_AUTHOR = "PR monitor"
AUTONOMOUS_GOAL_DISPLAY_AUTHOR = "Autonomous goal agent"
AUTONOMOUS_GOAL_HISTORY_SUMMARY_DISPLAY_AUTHOR = "Autonomous goal history"
AUTONOMOUS_GOAL_JUDGE_DISPLAY_AUTHOR = "Autonomous goal judge"
AUTONOMOUS_GOAL_DELETED_ERROR = "Autonomous goal deleted by user"
AUTONOMOUS_GOAL_PROPOSAL_ACCEPTED_ERROR = "Autonomous goal proposal accepted by user"
AUTONOMOUS_GOAL_PROPOSAL_REJECTED_ERROR = "Autonomous goal proposal rejected by user"
AUTONOMOUS_GOAL_PROPOSAL_DISMISSED_ERROR = "Autonomous goal proposal dismissed by user"
AUTONOMOUS_GOAL_AGENT_PROMPT_TITLE = session_index.AUTONOMOUS_GOAL_AGENT_PROMPT_TITLE
AUTONOMOUS_GOAL_JUDGE_PROMPT_TITLE = session_index.AUTONOMOUS_GOAL_JUDGE_PROMPT_TITLE
SPEC_CRITIC_DISPLAY_AUTHOR = "Spec Critic"
PR_SLASH_DISPLAY_PROMPT = (
    "Rebase on the default branch, clean it up, and then open a PR"
)
QA_SLASH_DISPLAY_PROMPT = (
    "Run the QA agent on the current diff and fix anything it finds"
)
PR_SLASH_PROMPT = (
    "Rebase on the default branch, polish it, get it ready, "
    "and commit the final changes. "
    "Do not push the branch or open a PR; Hitch will push and open it "
    "after this turn completes."
)
SYSTEM_AGENT_APPROVAL_MODE = "auto_review"
# Auto-review workflows (auto-QA and auto-PR) start without an explicit
# user action and spawn hidden QA subagents pinned to ``auto_review``.
# Bypassing the user's approval-control intent that way is only acceptable
# when the source turn itself runs under a non-interactive mode; with
# ``prompt_user`` or ``deny_all`` the user has asked to gate every action,
# and the follow-up PR-prompt turn would also stall (prompt_user) or have
# every action denied (deny_all), so refuse to start either workflow.
AUTO_REVIEW_BLOCKED_APPROVAL_MODES = frozenset({"deny_all", "prompt_user"})
AUTONOMOUS_GOAL_IMPLEMENTATION_SANDBOX_POLICY = "workspaceWrite"
QA_WORKFLOW_MAX_ITERATIONS = 10
PR_QA_WORKFLOW_MAX_ITERATIONS = QA_WORKFLOW_MAX_ITERATIONS + 3
STEP_QA_RUNNING = "qa_running"
STEP_FEEDBACK_RUNNING = "feedback_running"
STEP_USER_STEERING_RUNNING = "user_steering_running"
STEP_BLOCKED = "blocked"
STEP_MAX_ITERATIONS_REACHED = "max_iterations_reached"
STEP_QA_APPROVED = "qa_approved"
STEP_PR_PROMPT_SPAWNED = "pr_prompt_spawned"
STEP_PR_PROMPT_RUNNING = "pr_prompt_running"
STEP_PR_MONITORING = "pr_monitoring"
STEP_PR_FEEDBACK_RUNNING = "pr_feedback_running"
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
STEP_SPEC_CRITIC_CLASSIFYING = "spec_critic_classifying"
STEP_SPEC_CRITIC_ANALYZING = "spec_critic_analyzing"
STEP_SPEC_CRITIC_CLARIFYING = "spec_critic_clarifying"
STEP_SPEC_CRITIC_SYNTHESIZING = "spec_critic_synthesizing"
STEP_SPEC_CRITIC_IMPLEMENTATION_SPAWNED = "spec_critic_implementation_spawned"
SPEC_CRITIC_CLARIFICATION_METHOD = "hitch/spec_critic/clarification"


# The scheduler ticks once a minute, but the account rate-limit query that
# backs the quota pause is a remote round-trip to the Codex backend. Cache its
# verdict so the network call fires at most once per this interval regardless
# of tick cadence.


_WORKFLOW_FAILURE_OWNER_STATE_KEY = "failure_owner"
_ARCHIVED_FROM_BLOCKED_STATE_KEY = "archived_from_blocked"
# How long a blocked PR-QA workflow lingers before it is auto-archived off the
# inbox Blocked stage. Shared by the maintenance scheduler (which applies the
# archive) and the health dashboard (which previews the same cutoff), so both
# agree on what "stale" means.
STALE_BLOCKED_AGE = timedelta(days=7)
_WORKFLOW_FAILURE_OWNER_QA = "qa"
_WORKFLOW_FAILURE_OWNER_PR = "pr"
_WORKFLOW_ROUTE_CLAIM_TIMEOUT = timedelta(minutes=10)
# A Spec Critic classification runs in an in-process daemon thread, so it is
# lost if the web process restarts mid-flight. Reconciliation re-arms a
# CLASSIFYING workflow whose timestamp is older than this; the window is well
# above a normal classification (a few seconds) to avoid racing a live thread.


# A PR-QA workflow commits its next transient step (qa_running, feedback_running,
# pr_prompt_running, ...) and *then* spawns the worker for it inside the
# just-finished worker process (or the web request). If that process dies before
# the worker's CodexInstance row is created -- e.g. the orphan-worker reaper
# SIGKILLs its scope during a SQLite-lock storm -- the workflow zombies in that
# step with no worker and nothing to route, because there is no instance for the
# terminal-instance/turn reconcilers to find. Reconciliation recovers the
# workflow once the row is older than this window (re-driving the QA review,
# whose prompt is reconstructable, or surfacing a clear failure for a lost turn).
# The window sits well above ``_WORKFLOW_ROUTE_CLAIM_TIMEOUT`` so a slow-but-live
# spawn is never raced into a double review or a spurious failure.
_WORKFLOW_SPAWN_STALE_TIMEOUT = timedelta(minutes=15)
# Transient PR-QA steps whose worker is a visible coding/feedback turn spawned
# right after the step is committed, and whose prompt is *not* reconstructable
# (the QA feedback or the user's text is gone). A lost spawn here cannot be
# re-driven, so the workflow is blocked with a surfaced explanation instead.
# (pr_prompt_running is recovered separately by re-driving _spawn_pr_prompt,
# whose prompt is reconstructable from state.)
_ZOMBIE_TURN_STEP_MESSAGES = {
    STEP_FEEDBACK_RUNNING: "QA feedback turn",
    STEP_PR_FEEDBACK_RUNNING: "PR follow-up turn",
    STEP_USER_STEERING_RUNNING: "coding turn",
}
_PR_HANDOFF_STATE_KEY = "pr_handoff"
_PR_MONITOR_STATE_KEY = "last_pr_monitor"
_PR_MONITOR_BACKOFF_STATE_KEY = "pr_monitor_backoff"
_PR_MONITOR_FEEDBACK_OBSERVATION_KEY = "monitor_feedback_observation"
_PR_MONITOR_REINTERPRETATION_REQUIRED_KEY = "monitor_reinterpretation_required"
_PR_GATES_STATE_KEY = "pr_gates"
_PR_PENDING_CHECKS_STATE_KEY = "pr_pending_checks"
_WORKFLOW_TURN_DEATH_RETRY_STATE_KEY = "workflow_turn_death_retries"
_WORKFLOW_TURN_DEATH_RETRY_LIMIT = 1
_WORKER_EXITED_BEFORE_COMPLETION_ERROR = (
    "worker process exited before reporting completion"
)
_LEGACY_SERVER_OVERLOADED_ERROR = (
    "Selected model is at capacity. Please try a different model."
)


_SECONDS_PER_MINUTE = 60
QA_APPROVAL_INSERT_INDEX_STATE_KEY = "qa_approval_insert_index"
AUTO_MERGE_REVIEWED_DIFF_STATE_KEY = "auto_merge_reviewed_diff"
AUTO_MERGE_REVIEWED_TARGET_SHA_STATE_KEY = "auto_merge_reviewed_target_sha"
AUTO_MERGE_SESSION_BASE_SHA_STATE_KEY = "auto_merge_session_base_sha"
AUTO_MERGE_REVIEWED_SOURCE_TREE_STATE_KEY = "auto_merge_reviewed_source_tree"
AUTO_PULL_RESULT_STATE_KEY = "auto_pull_result"
_PR_STAGE_REFRESH_TIMEOUT_SECONDS = 5
_PR_MONITOR_PENDING_POLL_MIN_SECONDS = 5 * _SECONDS_PER_MINUTE
_PR_MONITOR_PENDING_POLL_MAX_SECONDS = 30 * _SECONDS_PER_MINUTE
_GH_PR_VIEW_FIELDS = (
    "url",
    "number",
    "state",
    "isDraft",
    "title",
    "baseRefName",
    "headRefName",
    "headRefOid",
    "mergeable",
    "mergeCommit",
    "createdAt",
    "updatedAt",
    "closedAt",
    "mergedAt",
)
_GH_PR_MONITOR_FIELDS = (
    *_GH_PR_VIEW_FIELDS,
    "comments",
    "latestReviews",
    "reactionGroups",
    "reviewDecision",
    "reviews",
)
_GH_MONITOR_TEXT_MAX_CHARS = 6000
# The claim lease must cover the worst-case poll so a crash between claiming a
# workflow and advancing/rescheduling its backoff doesn't re-expose it sooner
# than a poll could still be running -- but no longer, or a crashed poll leaves
# the workflow stuck for the full lease. The poll is now bounded by the monitor
# timeout (gh pr view + the two paginated page caps), so derive it from that.
_PR_MONITOR_BACKOFF_CLAIM_SECONDS = (
    _GH_PR_MONITOR_TIMEOUT_SECONDS
    * (1 + _GH_REVIEW_THREAD_PAGE_LIMIT + _GH_STATUS_CHECK_PAGE_LIMIT)
    + _SECONDS_PER_MINUTE
)
_PR_MONITOR_RETRY_LIMIT_REASONS = frozenset({"missing_cwd", "gh_error"})


def _nullable_schema(schema_type: str) -> dict[str, Any]:
    return {"type": [schema_type, "null"]}


def _pr_handoff_output_property_schema(field: str) -> dict[str, Any]:
    if field in _PR_HANDOFF_BOOLEAN_FIELDS:
        return _nullable_schema("boolean")
    if field in _PR_HANDOFF_INTEGER_FIELDS:
        return _nullable_schema("integer")
    if field == "ci_status":
        return {
            "type": ["string", "null"],
            "enum": ["success", "pending", "failure", None],
        }
    if field in _PR_HANDOFF_LIST_FIELDS:
        return {
            "type": ["array", "null"],
            "items": {
                "anyOf": [
                    {"type": "string"},
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": list(_PR_SAFE_LIST_ITEM_FIELDS),
                        "properties": {
                            key: {"type": ["string", "integer", "null"]}
                            for key in _PR_SAFE_LIST_ITEM_FIELDS
                        },
                    },
                ]
            },
        }
    return _nullable_schema("string")


_PR_HANDOFF_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": list(_PR_HANDOFF_FIELDS),
    "properties": {
        field: _pr_handoff_output_property_schema(field)
        for field in _PR_HANDOFF_FIELDS
    },
}

_QA_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["feedback", "lgtm"],
    "properties": {
        "feedback": {"type": "string"},
        "lgtm": {"type": "boolean"},
    },
}


_SPEC_REQUIREMENTS_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "requirements", "assumptions", "repo_signals"],
    "properties": {
        "summary": {"type": "string"},
        "requirements": {"type": "array", "items": {"type": "string"}},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "repo_signals": {"type": "array", "items": {"type": "string"}},
    },
}

_SPEC_RISK_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "ambiguities", "risks", "questions"],
    "properties": {
        "summary": {"type": "string"},
        "ambiguities": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "id",
                    "header",
                    "question",
                    "required",
                    "allow_safe_default",
                    "safe_default",
                    "options",
                ],
                "properties": {
                    "id": {"type": "string"},
                    "header": {"type": "string"},
                    "question": {"type": "string"},
                    "required": {"type": "boolean"},
                    "allow_safe_default": {"type": "boolean"},
                    "safe_default": {"type": ["string", "null"]},
                    "options": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["label", "description"],
                            "properties": {
                                "label": {"type": "string"},
                                "description": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
    },
}

_SPEC_TEST_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "summary",
        "acceptance_criteria",
        "test_strategy",
        "manual_checks",
    ],
    "properties": {
        "summary": {"type": "string"},
        "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
        "test_strategy": {"type": "array", "items": {"type": "string"}},
        "manual_checks": {"type": "array", "items": {"type": "string"}},
    },
}

_SPEC_SYNTHESIS_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["brief"],
    "properties": {
        "brief": {"type": "string"},
    },
}

_PR_MONITOR_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "summary", "feedback", "pr", "blockers"],
    "properties": {
        "status": {
            "type": "string",
            "enum": ["blocked", "terminal"],
        },
        "summary": {"type": "string"},
        "feedback": {"type": "string"},
        "pr": _PR_HANDOFF_OUTPUT_SCHEMA,
        "blockers": {"type": "array", "items": {"type": "string"}},
    },
}

_LEGACY_QA_PANEL_LANE_AGENT_KINDS = (
    "pr_qa_correctness",
    "pr_qa_tests",
    "pr_qa_ux_manual",
    "pr_qa_security",
    "pr_qa_maintainability",
)
_LEGACY_QA_PANEL_AGENT_KINDS = (
    *_LEGACY_QA_PANEL_LANE_AGENT_KINDS,
    "pr_qa_panel_synthesizer",
)
_LEGACY_QA_PANEL_CANCELLED_ERROR = (
    "legacy QA panel run cancelled because the QA panel feature was removed"
)
_QA_VERDICT_AGENT_KINDS = (PR_QA_AGENT_KIND,)
_QA_INTERRUPTIBLE_AGENT_KINDS = (
    *_QA_VERDICT_AGENT_KINDS,
    *_LEGACY_QA_PANEL_AGENT_KINDS,
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


def hidden_thread_ids(
    *, accepted_visible_thread_ids: set[str] | None = None
) -> set[str]:
    hidden_ids = set(
        SystemAgentRun.objects.exclude(thread_id="")
        .exclude(agent_kind="demo")
        .values_list("thread_id", flat=True)
        .distinct()
    )
    hidden_ids.update(
        CodexInstance.objects.filter(purpose=CodexInstance.PURPOSE_SYSTEM_AGENT)
        .exclude(thread_id="")
        .exclude(agent_kind=demo.DEMO_AGENT_KIND)
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
        if isinstance(thread_id := getattr(thread, "id", None), str)
        and hitch_system_agent_thread(thread)
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


def active_workflow_for_thread(main_thread_id: str) -> SystemWorkflow | None:
    reconcile_terminal_workflow_instances(main_thread_id=main_thread_id)
    return (
        SystemWorkflow.objects.filter(
            kind__in=(SystemWorkflow.KIND_PR_QA, SPEC_CRITIC_WORKFLOW_KIND),
            main_thread_id=main_thread_id,
            status=SystemWorkflow.STATUS_RUNNING,
        )
        .order_by("-created_at")
        .first()
    )


def reconcile_terminal_workflow_instances(
    *, main_thread_id: str | None = None, workflow_id: int | None = None
) -> int:
    """Route terminal workflow-owned workers that missed their finish callback."""
    workflows = list(
        _running_workflows_for_reconciliation(
            main_thread_id=main_thread_id,
            workflow_id=workflow_id,
        )
    )
    if not workflows:
        return 0
    reconciled = _reconcile_terminal_system_agent_instances(workflows)
    reconciled += _reconcile_terminal_workflow_turns(workflows)
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
        locked = _claim_stale_workflow_step(
            workflow, step=workflow.step, stale_before=stale_before
        )
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
    label = _ZOMBIE_TURN_STEP_MESSAGES[workflow.step]
    _block_workflow(
        workflow,
        f"{label} never started: its spawn handler died before the worker "
        "launched. Restart the workflow to continue.",
    )


def _workflow_turn_settling(workflow: SystemWorkflow) -> bool:
    """True while a worker is live or a finished turn is still being routed.

    A starting/running instance is a live (or reaper-bound) worker. A terminal
    turn whose routing claim is still fresh is being handed off to its finish
    handler right now; the terminal-turn reconciler (or the original finisher)
    will advance the step. In either case the workflow is not yet a zombie.
    """
    instances = CodexInstance.objects.filter(workflow_id=workflow.pk)
    if instances.filter(
        status__in=CodexInstance.ACTIVE_STATUSES
    ).exists():
        return True
    fresh_claim = timezone.now() - _WORKFLOW_ROUTE_CLAIM_TIMEOUT
    return instances.filter(
        purpose__in=(
            CodexInstance.PURPOSE_USER,
            CodexInstance.PURPOSE_SYSTEM_FEEDBACK,
        ),
        workflow_routing_started_at__gte=fresh_claim,
    ).exists()


def _claim_stale_workflow_step(
    workflow: SystemWorkflow, *, step: str, stale_before: datetime
) -> SystemWorkflow | None:
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
        .exclude(agent_kind=demo.DEMO_AGENT_KIND)
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
    if workflow.kind == SystemWorkflow.KIND_PR_QA:
        if workflow.step == STEP_QA_RUNNING:
            return _QA_INTERRUPTIBLE_AGENT_KINDS
        if workflow.step == STEP_PR_MONITORING:
            return (PR_FOLLOWUP_MONITOR_AGENT_KIND,)
        return ()
    if workflow.kind == SPEC_CRITIC_WORKFLOW_KIND:
        if workflow.step == STEP_SPEC_CRITIC_ANALYZING:
            return _SPEC_CRITIC_ANALYSIS_AGENT_KINDS
        if workflow.step == STEP_SPEC_CRITIC_SYNTHESIZING:
            return (SPEC_SYNTHESIZER_AGENT_KIND,)
        return ()
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
        current_user_message_index = _state_int(workflow, "next_user_message_index") - 1
        # ``_spawn_workflow_turn`` creates the turn *before* it saves the
        # incremented ``next_user_message_index``. If the spawner dies in that
        # gap, the durable turn carries ``next_user_message_index`` (one ahead of
        # the value used here), so match both indices to route it rather than
        # strand it. No turn is assigned the higher index in healthy states.
        # When the stored index is still 0 (normal for /pr-from-proposal
        # workflows) only the higher arm can match -- skipping the workflow
        # outright would strand exactly the dead-spawn turn this dual match
        # exists for.
        turn_indices = tuple(
            index
            for index in (current_user_message_index, current_user_message_index + 1)
            if index >= 0
        )
        if not turn_indices:
            continue
        if workflow.step in (STEP_FEEDBACK_RUNNING, STEP_PR_FEEDBACK_RUNNING):
            filters |= models.Q(
                workflow_id=workflow.pk,
                purpose=CodexInstance.PURPOSE_SYSTEM_FEEDBACK,
                user_message_index__in=turn_indices,
            )
            has_turn_filter = True
        elif workflow.step in (STEP_USER_STEERING_RUNNING, STEP_PR_PROMPT_RUNNING):
            filters |= models.Q(
                workflow_id=workflow.pk,
                purpose=CodexInstance.PURPOSE_USER,
                user_message_index__in=turn_indices,
            )
            has_turn_filter = True
    if not has_turn_filter:
        return 0
    instances = CodexInstance.objects.filter(
        filters,
        status__in=(CodexInstance.STATUS_COMPLETED, CodexInstance.STATUS_FAILED),
    ).filter(_unclaimed_workflow_instance_filter()).order_by("started_at", "id")
    reconciled = 0
    for instance in instances:
        if _route_terminal_workflow_instance(instance):
            reconciled += 1
    return reconciled


def _unclaimed_workflow_instance_filter() -> models.Q:
    stale_before = timezone.now() - _WORKFLOW_ROUTE_CLAIM_TIMEOUT
    return models.Q(workflow_routing_started_at__isnull=True) | models.Q(
        workflow_routing_started_at__lt=stale_before
    )


def _route_terminal_workflow_instance(instance: CodexInstance) -> bool:
    try:
        return on_codex_instance_finished(instance)
    except Exception:
        logger.exception(
            "failed to reconcile terminal workflow instance %s",
            instance.pk,
        )
        return False


def stop_active_workflow(main_thread_id: str) -> bool:
    workflow = active_workflow_for_thread(main_thread_id)
    if workflow is None:
        return False
    runs = list(
        workflow.agent_runs.filter(status=SystemAgentRun.STATUS_RUNNING)
        .select_related("instance")
        .order_by("-created_at")
    )
    if not runs:
        if pr_qa._workflow_waits_on_pr_monitor_backoff(workflow):
            workflow.state = dict(workflow.state)
            workflow.state.pop(_PR_MONITOR_BACKOFF_STATE_KEY, None)
            workflow.save(update_fields=["state", "updated_at"])
            _block_workflow(workflow, "QA workflow stopped by user")
            return True
        if workflow.kind == SPEC_CRITIC_WORKFLOW_KIND:
            error = "Spec Critic workflow stopped by user"
            spec_critic._cancel_pending_spec_critic_input_requests(workflow, error)
            spec_critic._block_spec_critic_workflow(workflow, error)
            return True
        return False
    interrupted_runs = _interrupt_system_agent_runs(runs)
    if not interrupted_runs:
        return False
    if workflow.kind == SPEC_CRITIC_WORKFLOW_KIND:
        error = "Spec Critic workflow stopped by user"
        spec_critic._cancel_pending_spec_critic_input_requests(workflow, error)
        _mark_system_agent_runs_failed(interrupted_runs, error)
        spec_critic._block_spec_critic_workflow(workflow, error)
    else:
        _mark_system_agent_runs_failed(interrupted_runs, "QA workflow stopped by user")
        _block_workflow(workflow, "QA workflow stopped by user")
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
    if instance.purpose == CodexInstance.PURPOSE_SYSTEM_AGENT:
        return _handle_system_agent_finished(instance)
    if instance.purpose == CodexInstance.PURPOSE_SYSTEM_FEEDBACK:
        _dispatch_workflow_event(instance, "on_feedback_finished")
        return True
    if (
        instance.purpose == CodexInstance.PURPOSE_USER
        and instance.workflow_id is not None
    ):
        _dispatch_workflow_event(instance, "on_user_turn_finished")
        return True
    _maybe_start_auto_review_workflow(instance)
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


def _maybe_start_auto_review_workflow(instance: CodexInstance) -> None:
    if (
        instance.purpose != CodexInstance.PURPOSE_USER
        or instance.workflow_id is not None
        or not (instance.auto_pr_enabled or instance.auto_qa_enabled)
        or instance.plan_mode
        or instance.status != CodexInstance.STATUS_COMPLETED
    ):
        return
    # The post-QA work-agent and PR-prompt turns reuse the instance's
    # approval_mode (see _spawn_workflow_turn), so prompt_user/deny_all would
    # stall the workflow or have every action auto-denied regardless of which
    # automation triggered it.
    if _auto_review_requires_visible_approval(instance):
        return
    if _completed_turn_has_pending_proposed_plan(instance):
        return
    automation = "auto_pr" if instance.auto_pr_enabled else "auto_qa"
    trigger_field = (
        "auto_pr_triggered_at"
        if automation == "auto_pr"
        else "auto_qa_triggered_at"
    )
    claimed = CodexInstance.objects.filter(
        pk=instance.pk,
        **{f"{trigger_field}__isnull": True},
    ).update(**{trigger_field: timezone.now()})
    if not claimed:
        return
    try:
        workflow_kwargs: dict[str, Any] = {
            "main_thread_id": instance.thread_id,
            "cwd": instance.cwd,
            "sandbox_policy": instance.sandbox_policy or None,
            "approval_mode": instance.approval_mode or SYSTEM_AGENT_APPROVAL_MODE,
            "model": instance.model or None,
            "reasoning_effort": instance.reasoning_effort or None,
            "base_instructions": instance.base_instructions or None,
            "developer_instructions": instance.developer_instructions or None,
            "enable_memories": instance.enable_memories,
            "web_search_mode": instance.web_search_mode or None,
            "initial_user_message_index": (instance.user_message_index or 0) + 1,
        }
        if automation == "auto_pr":
            pr_title = _accepted_auto_pr_proposal_title(instance.thread_id)
            if pr_title:
                workflow_kwargs["pr_title"] = pr_title
        if automation == "auto_qa":
            workflow_kwargs["open_pr_on_lgtm"] = False
        auto_merge_branch = (
            instance.auto_merge_branch.strip()
            if instance.auto_merge_to_local_branch
            else ""
        )
        if auto_merge_branch:
            workflow_kwargs["open_pr_on_lgtm"] = False
            workflow_kwargs["auto_merge_branch"] = auto_merge_branch
        workflow = pr_qa.start_pr_qa_workflow(**workflow_kwargs)
        if isinstance(workflow, SystemWorkflow):
            _record_auto_review_workflow_for_proposals(
                instance, workflow, automation=automation
            )
    except Exception:
        CodexInstance.objects.filter(pk=instance.pk).update(**{trigger_field: None})
        raise


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
    the turn needs visible approval or ends with a pending proposed plan, so the
    null ``auto_pr_triggered_at`` / ``auto_qa_triggered_at`` are expected rather
    than a dropped follow-up. The orphan reaper uses this so it does not rewrite
    such an intentionally-skipped (but successful) turn as failed.
    """
    return _auto_review_requires_visible_approval(
        instance
    ) or _completed_turn_has_pending_proposed_plan(instance)


def _auto_review_requires_visible_approval(instance: CodexInstance) -> bool:
    return (
        instance.approval_mode or SYSTEM_AGENT_APPROVAL_MODE
    ) in AUTO_REVIEW_BLOCKED_APPROVAL_MODES


def _completed_turn_has_pending_proposed_plan(instance: CodexInstance) -> bool:
    rollout_pending = _thread_rollout_has_pending_plan(instance.thread_id)
    if rollout_pending is not None:
        return rollout_pending
    final_text = _final_agent_text(instance.events_path)
    plan_text = (
        rollout.proposed_plan_text_from_agent_text(final_text) if final_text else None
    )
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


def _record_auto_merge_result_for_proposals(
    workflow: SystemWorkflow, updates: dict[str, object]
) -> None:
    metadata = SessionMetadata.objects.filter(thread_id=workflow.main_thread_id).first()
    if metadata is None:
        return
    for proposal in ProposedSession.objects.filter(accepted_session=metadata):
        proposal.outcome_metadata = _proposal_outcome_metadata(proposal, updates)
        proposal.save(update_fields=["outcome_metadata", "updated_at"])


def _maybe_auto_pull_default_repo_after_pr_monitor_merge(
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
    metadata = (
        SessionMetadata.objects.select_related("project")
        .filter(thread_id=workflow.main_thread_id)
        .first()
    )
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
    return resolved_path(cwd_path) == resolved_path(
        Path(repo_path).expanduser()
    )


def _auto_pull_result_dict(result: AutoPullResult) -> dict[str, object]:
    return {
        "status": "pulled" if result.changed else "up_to_date",
        "branch": result.branch,
        "before_sha": result.before_sha,
        "after_sha": result.after_sha,
        "changed": result.changed,
    }


def _record_auto_pull_result(
    workflow: SystemWorkflow, result: dict[str, object]
) -> None:
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
        logger.exception(
            "failed to record auto-pull result for workflow %s", workflow.pk
        )


def _handle_system_agent_finished(instance: CodexInstance) -> bool:
    run = _system_agent_run_for_instance(instance)
    if run is None:
        return False
    workflow = run.workflow
    _route_system_agent_finished(instance, run, workflow)
    return True


def _route_system_agent_finished(
    instance: CodexInstance, run: SystemAgentRun, workflow: SystemWorkflow
) -> None:
    if run.status in (SystemAgentRun.STATUS_COMPLETED, SystemAgentRun.STATUS_FAILED):
        autonomous_goals._cleanup_cancelled_autonomous_goal_terminal_run(instance, run, workflow)
        return
    handler = engine.handler_for(workflow, run=run, instance=instance)
    if handler is None:
        _fail_unsupported_system_agent_run(run, workflow)
        return
    handler.on_agent_finished(instance, run, workflow)


@engine.register
class _DemoHandler(engine.WorkflowHandler):
    kind = demo.DEMO_WORKFLOW_KIND
    # The demo subsystem owns its workflow lifecycle end to end.
    steps = None

    @override
    def matches_run(self, run: SystemAgentRun, instance: CodexInstance) -> bool:
        return (
            run.agent_kind == demo.DEMO_AGENT_KIND
            and instance.agent_kind == demo.DEMO_AGENT_KIND
        )

    @override
    def on_agent_finished(
        self,
        instance: CodexInstance,
        run: SystemAgentRun,
        workflow: SystemWorkflow,
    ) -> None:
        _handle_demo_agent_finished(instance, run, workflow)


# Shared by the PR-QA review phases and the followup monitor, which run in
# the same KIND_PR_QA workflow row.


def _handle_demo_agent_finished(
    instance: CodexInstance, run: SystemAgentRun, workflow: SystemWorkflow
) -> None:
    try:
        demo.on_codex_instance_finished(instance)
    except Exception as exc:
        logger.exception(
            "failed to route completed worker %s to demo workflow", instance.pk
        )
        error = f"demo workflow router failed: {exc}"
        if run.status not in (SystemAgentRun.STATUS_COMPLETED, SystemAgentRun.STATUS_FAILED):
            run.status = SystemAgentRun.STATUS_FAILED
            run.error = error
            run.save(update_fields=["status", "error", "updated_at"])
        if workflow.is_active:
            workflow.status = SystemWorkflow.STATUS_FAILED
            workflow.save(update_fields=["status", "updated_at"])


def _claim_workflow_turn_retry(
    workflow: SystemWorkflow, instance: CodexInstance, retry_kind: str
) -> bool:
    """Record one more bounded retry for a transient workflow-turn failure.

    Single source of the retry rule shared by every workflow turn: the
    workflow must still be active, the failure must be recoverable without a
    user Stop request, and the per-step retry budget
    (``_WORKFLOW_TURN_DEATH_RETRY_LIMIT``) must not be exhausted. Bumps and
    persists the per-kind count when it returns True. The persisted state key
    retains its original ``death`` name for compatibility.
    """
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
    retries: dict[str, int] = {}
    for key, value in raw.items():
        if is_nonbool_int(value) and value > 0:
            retries[str(key)] = value
    return retries


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


def _is_worker_exited_before_completion_error(error: str) -> bool:
    return error.strip().startswith(_WORKER_EXITED_BEFORE_COMPLETION_ERROR)


def _is_retryable_workflow_turn_error(instance: CodexInstance) -> bool:
    if instance.codex_error_info == CodexInstance.CODEX_ERROR_SERVER_OVERLOADED:
        return True
    normalized = instance.error.strip()
    return (
        _is_worker_exited_before_completion_error(normalized)
        or normalized == _LEGACY_SERVER_OVERLOADED_ERROR
    )


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


def _copy_gh_string(
    source: dict[str, Any], target: dict[str, Any], source_key: str, target_key: str
) -> None:
    value = string_from_any(source.get(source_key))
    if value:
        target[target_key] = value


def _gh_mergeable_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in {"mergeable", "clean", "has_hooks", "unstable"}:
        return True
    if normalized in {"conflicting", "dirty", "blocked"}:
        return False
    return None


def _run_gh_observation_fallback(run: SystemAgentRun) -> dict[str, Any]:
    run_input = run.input if isinstance(run.input, dict) else {}
    gh_observation = run_input.get("gh_observation")
    return gh_observation if isinstance(gh_observation, dict) else {}


_PR_MONITOR_MAX_ITERATIONS_FEEDBACK = (
    "PR follow-up monitor reached the maximum feedback loop count "
    "without reaching a clean PR state."
)


def _fail_unsupported_system_agent_run(
    run: SystemAgentRun, workflow: SystemWorkflow
) -> None:
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


def _state_without_current_candidate_result(
    state: Mapping[str, Any]
) -> dict[str, Any]:
    next_state = dict(state)
    for key in ("candidate", "judgment", "judge_session_id", "history_files"):
        next_state.pop(key, None)
    return next_state


def _candidate_session_cwd_from_state(workflow: SystemWorkflow, key: str) -> str:
    metadata = _session_metadata_from_state(workflow, key)
    return metadata.cwd if metadata is not None else ""


# Hidden QA subagents do not surface approval prompts in the main workflow UI.
# Keep their approval mode fixed; workflow approval state is for visible turns.


def _review_diff_text_for_workflow(workflow: SystemWorkflow) -> str:
    auto_merge_branch = _state_string(workflow, "auto_merge_branch")
    if not auto_merge_branch:
        return build_worktree_diff_text(workflow.cwd)
    review_patch = build_auto_merge_review_patch(workflow.cwd, auto_merge_branch)
    workflow.state = {
        **workflow.state,
        AUTO_MERGE_REVIEWED_DIFF_STATE_KEY: review_patch.patch,
        AUTO_MERGE_REVIEWED_TARGET_SHA_STATE_KEY: review_patch.target_sha,
        AUTO_MERGE_SESSION_BASE_SHA_STATE_KEY: review_patch.base_sha,
        AUTO_MERGE_REVIEWED_SOURCE_TREE_STATE_KEY: review_patch.source_tree_sha,
    }
    workflow.save(update_fields=["state", "updated_at"])
    # The lossless patch (surrogateescape-decoded bytes) is what gets applied
    # from state; the copy embedded in the QA prompt must be valid UTF-8 for
    # the app-server's JSON parser. Render any non-UTF-8 bytes as visible
    # ``\xNN`` escapes so the QA agent reviews the exact byte values that
    # will be merged, instead of a lossy substitution hiding them.
    raw_patch = review_patch.patch.encode("utf-8", errors="surrogateescape")
    return raw_patch.decode("utf-8", errors="backslashreplace")


def _spawn_workflow_failure_turn(
    workflow: SystemWorkflow, error: str
) -> CodexInstance:
    headline, display_author = _workflow_failure_turn_context(workflow, error)
    return _spawn_workflow_turn(
        workflow,
        prompt=(
            f"{headline}\n\n"
            f"Status: {error}\n\n"
            "Tell the user the PR workflow needs attention before continuing."
        ),
        purpose=CodexInstance.PURPOSE_SYSTEM_FEEDBACK,
        display_author=display_author,
    )


def _workflow_failure_turn_context(
    workflow: SystemWorkflow, error: str
) -> tuple[str, str]:
    if _workflow_failure_owner(workflow, error) == _WORKFLOW_FAILURE_OWNER_QA:
        return "Hitch QA agent could not complete the PR workflow.", QA_DISPLAY_AUTHOR
    return "Hitch PR workflow could not complete.", PR_WORKFLOW_DISPLAY_AUTHOR


def _workflow_failure_owner(workflow: SystemWorkflow, error: str) -> str:
    stored_owner = workflow.state.get(_WORKFLOW_FAILURE_OWNER_STATE_KEY)
    if stored_owner in {_WORKFLOW_FAILURE_OWNER_QA, _WORKFLOW_FAILURE_OWNER_PR}:
        return str(stored_owner)
    step_owner = _workflow_failure_owner_for_step(workflow.step)
    if step_owner:
        return step_owner
    if _is_qa_workflow_failure(error):
        return _WORKFLOW_FAILURE_OWNER_QA
    return _WORKFLOW_FAILURE_OWNER_PR


def _workflow_failure_owner_for_step(step: str) -> str:
    if step in {STEP_QA_RUNNING, STEP_FEEDBACK_RUNNING}:
        return _WORKFLOW_FAILURE_OWNER_QA
    if step in {
        STEP_USER_STEERING_RUNNING,
        STEP_PR_PROMPT_SPAWNED,
        STEP_PR_PROMPT_RUNNING,
        STEP_PR_MONITORING,
        STEP_PR_FEEDBACK_RUNNING,
    }:
        return _WORKFLOW_FAILURE_OWNER_PR
    return ""


def _is_qa_workflow_failure(error: str) -> bool:
    return error.startswith(
        (
            "QA agent reached",
            "QA feedback worker failed",
            "QA output ",
            "QA worker ",
            "failed to restart QA agent",
            "failed to start QA agent",
            "failed to start QA feedback turn",
            "legacy QA panel run cancelled",
            "unsupported PR QA agent kind",
        )
    )


def _spawn_workflow_turn(
    workflow: SystemWorkflow,
    *,
    prompt: str,
    purpose: str = CodexInstance.PURPOSE_USER,
    display_author: str = "",
    agent_kind: str = "",
) -> CodexInstance:
    user_message_index = _state_int(workflow, "next_user_message_index")
    instance = codex_pool.spawn_turn(
        thread_id=workflow.main_thread_id,
        cwd=workflow.cwd,
        prompt=prompt,
        model=_state_string(workflow, "model") or None,
        reasoning_effort=_state_string(workflow, "reasoning_effort") or None,
        base_instructions=_state_string(workflow, "base_instructions") or None,
        developer_instructions=_state_string(workflow, "developer_instructions") or None,
        sandbox_policy=_state_string(workflow, "sandbox_policy") or None,
        approval_mode=_state_string(workflow, "approval_mode") or None,
        enable_memories=_state_bool(workflow, "enable_memories"),
        web_search_mode=_workflow_web_search_mode(workflow),
        purpose=purpose,
        workflow_id=workflow.pk,
        agent_kind=(
            agent_kind or PR_QA_AGENT_KIND
            if purpose != CodexInstance.PURPOSE_USER
            else ""
        ),
        display_author=display_author,
        user_message_index=user_message_index,
    )
    workflow.state = {
        **workflow.state,
        "next_user_message_index": user_message_index + 1,
    }
    workflow.save(update_fields=["state", "updated_at"])
    return instance


def _interrupt_system_agent_runs(runs: list[SystemAgentRun]) -> list[SystemAgentRun]:
    interrupted_runs: list[SystemAgentRun] = []
    for run in runs:
        interrupted = codex_pool.interrupt_instance(
            run.instance_id, expected_thread_id=run.thread_id
        )
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


def archive_stale_blocked_workflows(
    *, older_than: datetime, apply: bool
) -> list[int]:
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


def _advance_workflow_step(
    workflow: SystemWorkflow, step: str, *, bump_iteration: bool = False
) -> None:
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
        raise ValueError(
            f"illegal step {step!r} for workflow kind {workflow.kind!r}"
        )


def _block_workflow(
    workflow: SystemWorkflow,
    error: str,
    *,
    surface_to_thread: bool = True,
    only_if: Callable[[SystemWorkflow], bool] | None = None,
) -> bool:
    # ``only_if`` runs against the locked row so a caller whose claim on the
    # workflow may have been superseded (a stale QA verdict racing a user
    # steering claim) makes the ownership check and the block one atomic
    # decision; returns whether the workflow was blocked. Blocking is legal
    # from any status, so the claim does not require an active row.
    def _block(locked: SystemWorkflow) -> bool:
        failure_owner = _workflow_failure_owner(locked, error)
        locked.status = SystemWorkflow.STATUS_BLOCKED
        locked.step = STEP_BLOCKED
        locked.state = {
            **locked.state,
            "error": error,
            _WORKFLOW_FAILURE_OWNER_STATE_KEY: failure_owner,
        }
        locked.save(update_fields=["status", "step", "state", "updated_at"])
        return True

    blocked = engine.claim_workflow_transition(
        workflow, _block, guard=only_if, require_active=False
    )
    if not blocked:
        return False
    pr_qa._interrupt_orphaned_qa_review_runs(workflow, error)
    if surface_to_thread:
        _surface_workflow_failure(workflow, error)
    return True


def _surface_workflow_failure(workflow: SystemWorkflow, error: str) -> None:
    # Make the check-then-set atomic per workflow so concurrent failure routes
    # cannot double-post the failure message or double-increment the user
    # message index. Mirrors _surface_spec_critic_failure.
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
        logger.exception(
            "failed to surface system workflow failure for workflow %s", workflow.pk
        )


def _question_for_user_input(question: dict[str, Any]) -> dict[str, Any]:
    required = question.get("required") is True
    safe_default = question.get("safe_default")
    has_safe_default = (
        question.get("allow_safe_default") is True
        and isinstance(safe_default, str)
        and bool(safe_default.strip())
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
            "input": _recovered_system_agent_run_input(instance, workflow),
        },
    )
    return run


def _recovered_system_agent_run_input(
    instance: CodexInstance, workflow: SystemWorkflow
) -> dict[str, Any]:
    run_input: dict[str, Any] = {"cwd": instance.cwd}
    if workflow.kind != SystemWorkflow.KIND_PR_QA:
        return run_input
    if instance.agent_kind not in _QA_INTERRUPTIBLE_AGENT_KINDS:
        return run_input
    revision = instance.user_message_index
    run_input["qa_review_revision"] = (
        revision if revision is not None else 0
    )
    return run_input


# Imported last: the kind modules register their WorkflowHandler with the
# engine and reach back into this module for the shared spawn/transition
# helpers, so they need its namespace to be fully initialized.
from hitch.main.workflows import autonomous_goals, pr_qa, spec_critic  # noqa: E402
