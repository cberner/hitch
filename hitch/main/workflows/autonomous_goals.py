"""The autonomous-goal workflow: candidate, judge, and stacked proposals.

State machine: candidate_running -> judge_running -> proposed/skipped, with
budgeted retries, dead-worker retries, stacked-diff continuation (a chain
of proposals forked from the previous candidate's worktree), and the
auto-proposal scheduler entry points (quota pause, per-goal start claims).

Shared spawn/transition/blocking helpers stay in ``system_agents`` and are
reached through the module object so test patches on that namespace keep
intercepting.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast, override

from django.db import IntegrityError, transaction
from django.utils import timezone
from django.utils.dateparse import parse_datetime
from openai_codex import AppServerError, Codex
from openai_codex.generated.v2_all import GetAccountRateLimitsResponse, ThreadSource

from hitch.main.goals.autonomous_goal_prompts import (
    _AUTONOMOUS_GOAL_FAILED_ATTEMPTS_STATE_KEY,
    _AUTONOMOUS_GOAL_LAST_FAILURE_STATE_KEY,
    _AUTONOMOUS_GOAL_NO_PROGRESS_RETRIES_STATE_KEY,
    _AUTONOMOUS_GOAL_PROPOSAL_BUDGET_STATE_KEY,
    _AUTONOMOUS_GOAL_PROPOSAL_BUDGET_USED_STATE_KEY,
    _AUTONOMOUS_GOAL_PROPOSAL_HISTORY_SUMMARY_STATE_KEY,
    _AUTONOMOUS_GOAL_SESSION_CWD_STATE_KEY,
    _AUTONOMOUS_GOAL_STACKED_DEPTH_STATE_KEY,
    _AUTONOMOUS_GOAL_STACKED_ITERATION_STATE_KEY,
    _autonomous_goal_candidate_allows_code_changes,
    _autonomous_goal_candidate_prompt,
    _autonomous_goal_candidate_retry_prompt,
    _autonomous_goal_failed_attempts,
    _autonomous_goal_judge_prompt,
    _autonomous_goal_no_progress_budget_retries,
    _autonomous_goal_proposal_budget_metadata,
    _autonomous_goal_proposal_budget_tokens_used,
    _autonomous_goal_proposal_summary,
    _autonomous_goal_proposed_session_prompt,
    _autonomous_goal_recent_proposal_run_references,
    _autonomous_goal_session_cwd,
    _autonomous_goal_stack_iteration,
    _autonomous_goal_workflow_proposal_budget,
    _autonomous_goal_workflow_stacked_diff_depth,
    _store_autonomous_goal_memory,
)
from hitch.main.goals.autonomous_goal_proposal_stack import (
    _AUTONOMOUS_GOAL_STACKED_CONTINUATION_STOP_REASON_METADATA_KEY,
    _autonomous_goal_in_flight_automation_exists,
    _autonomous_goal_pending_proposal_blocks_start,
    _autonomous_goal_proposal_stack_continuation_metadata,
    _autonomous_goal_stack_continuation_proposal,
    _autonomous_goal_unresolved_failure_notice_exists,
    _claim_autonomous_goal_stack_continuation_proposal,
    _proposal_metadata_non_negative_int,
    _proposal_outcome_metadata,
)
from hitch.main.models import (
    AutonomousGoal,
    CodexInstance,
    Project,
    ProposedSession,
    SessionMetadata,
    SystemAgentRun,
    SystemWorkflow,
)
from hitch.main.repos import commit_hash_for_ref, default_branch_commit_hash
from hitch.main.runtime import app_server_pool, codex_events, codex_pool, rollout
from hitch.main.runtime.rollout_state import _rollout_path_from_value
from hitch.main.runtime.sdk_values import truncate_for_prompt
from hitch.main.sessions import session_index
from hitch.main.workflows import engine, system_agents
from hitch.main.workflows.agent_io import (
    _AUTONOMOUS_GOAL_TITLE_MAX_LEN,
    _autonomous_goal_history_sections,
    _parse_autonomous_goal_candidate_output,
    _parse_autonomous_goal_history_summary_output,
    _parse_autonomous_goal_judge_output,
    _split_autonomous_goal_history,
    _string_list,
    _write_autonomous_goal_history_files,
)
from hitch.main.workflows.spec_critic_prompts import (
    _below_threshold_notice_summary,
    _below_threshold_notice_title,
    _candidate_notice_title,
)
from hitch.main.workflows.workflow_state import (
    _confidence_meets_threshold,
    _session_metadata_from_state,
    _state_bool,
    _state_dict,
    _state_int,
    _state_string,
)
from hitch.main.worktrees import (
    ManagedWorktree,
    WorktreeCleanupError,
    WorktreeCreationError,
    cleanup_managed_worktree_path,
    cleanup_worktree,
    create_worktree_for_session,
    snapshot_worktree_to_commit,
)

logger = logging.getLogger(__name__)

_AUTO_PROPOSAL_UNKNOWN_DEFAULT_BRANCH_SHA = "__unknown__"

_AUTO_PROPOSAL_QUOTA_CACHE_TTL = timedelta(minutes=5)

_quota_cache_lock = threading.Lock()

_quota_cache_paused = False

_quota_cache_checked_at: datetime | None = None

_AUTONOMOUS_GOAL_USE_WORKTREES_STATE_KEY = "use_worktrees"

_AUTONOMOUS_GOAL_STACKED_FORK_CWD_STATE_KEY = "stacked_diff_fork_from_cwd"

_AUTONOMOUS_GOAL_STACKED_CONTINUATION_STOP_ERROR_METADATA_KEY = (
    "stacked_diff_continuation_error"
)

_AUTONOMOUS_GOAL_STACKED_HIDDEN_OUTCOME_NOTES = (
    "Hidden while stacked diff workflow continues."
)

_AUTONOMOUS_GOAL_STACKED_PROPOSAL_STOP_REASONS = {
    ProposedSession.OUTCOME_ACCEPTED: "proposal_accepted",
    ProposedSession.OUTCOME_REJECTED: "proposal_rejected",
    ProposedSession.OUTCOME_DISMISSED: "proposal_dismissed",
}

_AUTONOMOUS_GOAL_PROPOSAL_RESOLUTION_ERRORS = {
    ProposedSession.OUTCOME_ACCEPTED: system_agents.AUTONOMOUS_GOAL_PROPOSAL_ACCEPTED_ERROR,
    ProposedSession.OUTCOME_REJECTED: system_agents.AUTONOMOUS_GOAL_PROPOSAL_REJECTED_ERROR,
    ProposedSession.OUTCOME_DISMISSED: system_agents.AUTONOMOUS_GOAL_PROPOSAL_DISMISSED_ERROR,
}

_AUTONOMOUS_GOAL_PROPOSAL_RESOLUTION_ERROR_VALUES = frozenset(
    _AUTONOMOUS_GOAL_PROPOSAL_RESOLUTION_ERRORS.values()
)

_AUTONOMOUS_GOAL_PROPOSAL_BUDGET_TOKEN_TOTALS_STATE_KEY = (
    "proposal_budget_token_totals"
)

_AUTONOMOUS_GOAL_NO_PROGRESS_RETRY_LIMIT = 1

_AUTONOMOUS_GOAL_CANDIDATE_RETRY_KIND = "autonomous_goal_candidate"

_AUTONOMOUS_GOAL_HISTORY_SUMMARY_RETRY_KIND = "autonomous_goal_history_summary"

_AUTONOMOUS_GOAL_JUDGE_RETRY_KIND = "autonomous_goal_judge"

_AUTONOMOUS_GOAL_SPAWN_JUDGE_ACTION = "spawn_judge"

_AUTONOMOUS_GOAL_SPAWN_NEXT_CANDIDATE_ACTION = "spawn_next_candidate"

_AUTONOMOUS_GOAL_RETRY_CANDIDATE_ACTION = "retry_candidate"

_AUTONOMOUS_GOAL_RETRY_CANDIDATE_CONTINUATION_ACTION = "retry_candidate_continuation"

_AUTONOMOUS_GOAL_RETRY_JUDGE_ACTION = "retry_judge"

_AUTO_PROPOSAL_QUOTA_THRESHOLD_FRACTION = 0.5

_AUTONOMOUS_GOAL_HISTORY_SUMMARY_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "brief",
        "recent_stack",
        "accepted_lessons",
        "avoid_or_reconsider",
        "promising_next_directions",
        "important_files",
    ],
    "properties": {
        "brief": {"type": "string"},
        "recent_stack": {"type": "array", "items": {"type": "string"}},
        "accepted_lessons": {"type": "array", "items": {"type": "string"}},
        "avoid_or_reconsider": {"type": "array", "items": {"type": "string"}},
        "promising_next_directions": {"type": "array", "items": {"type": "string"}},
        "important_files": {"type": "array", "items": {"type": "string"}},
    },
}

_AUTONOMOUS_GOAL_CANDIDATE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "proposal",
        "message",
        "next_steps_summary",
        "memory_relevant_files",
    ],
    "properties": {
        "proposal": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "required": [
                "title",
                "summary",
                "impact",
                "implemented_changes",
                "implementation_direction",
                "verification",
                "rough_edges",
                "suggested_continuation",
                "relevant_files",
            ],
            "properties": {
                "title": {"type": "string"},
                "summary": {"type": "string"},
                "impact": {"type": "string"},
                "implemented_changes": {"type": "string"},
                "implementation_direction": {"type": "string"},
                "verification": {"type": "string"},
                "rough_edges": {"type": "string"},
                "suggested_continuation": {"type": "string"},
                "relevant_files": {"type": "array", "items": {"type": "string"}},
            },
        },
        "message": {"type": "string"},
        "next_steps_summary": {"type": "string"},
        "memory_relevant_files": {"type": "array", "items": {"type": "string"}},
    },
}

_AUTONOMOUS_GOAL_JUDGE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["confidence", "summary", "rationale"],
    "properties": {
        "confidence": {
            "type": "string",
            "enum": [
                AutonomousGoal.CONFIDENCE_MEDIUM,
                AutonomousGoal.CONFIDENCE_HIGH,
                AutonomousGoal.CONFIDENCE_VERY_HIGH,
            ],
        },
        "summary": {"type": "string"},
        "rationale": {"type": "string"},
    },
}


class _AutonomousGoalHistorySummaryUnpreservedError(RuntimeError):
    pass


def maybe_start_auto_proposal_workflows(*, project: Project | None = None) -> int:
    goals = AutonomousGoal.objects.select_related("project").filter(
        auto_proposal_enabled=True,
        deleted_at__isnull=True,
    )
    if project is not None:
        goals = goals.filter(project=project)
    if goals.exists() and _auto_proposals_paused_by_usage_quota_throttled():
        return 0

    started = 0
    for autonomous_goal_id in goals.order_by("created_at", "id").values_list(
        "id", flat=True
    ):
        # The id list is a snapshot, so a goal (or its project) deleted mid-tick
        # makes the select_for_update().get() raise. Isolate each goal so one bad
        # row can't abort the rest of the batch -- the scheduler loop swallows the
        # tick error, but the run_auto_proposals command does not.
        try:
            if _maybe_start_auto_proposal_workflow(autonomous_goal_id):
                started += 1
        except (AutonomousGoal.DoesNotExist, Project.DoesNotExist):
            continue
        except Exception:
            system_agents.logger.exception(
                "failed to start auto-proposal workflow for goal %s",
                autonomous_goal_id,
            )
    return started

def _reset_auto_proposal_quota_cache() -> None:
    """Clear the throttled quota verdict. Used by tests to isolate the
    module-level cache between cases."""
    global _quota_cache_paused, _quota_cache_checked_at
    with _quota_cache_lock:
        _quota_cache_paused = False
        _quota_cache_checked_at = None

def _auto_proposals_paused_by_usage_quota_throttled() -> bool:
    """Return the quota pause verdict, refreshing the remote check at most once
    per ``_AUTO_PROPOSAL_QUOTA_CACHE_TTL`` so the minute-cadence scheduler does
    not poll the Codex rate-limit endpoint every tick."""
    global _quota_cache_paused, _quota_cache_checked_at
    with _quota_cache_lock:
        now = timezone.now()
        if (
            _quota_cache_checked_at is not None
            and now - _quota_cache_checked_at < _AUTO_PROPOSAL_QUOTA_CACHE_TTL
        ):
            return _quota_cache_paused
        _quota_cache_paused = _auto_proposals_paused_by_usage_quota()
        _quota_cache_checked_at = now
        return _quota_cache_paused

def _auto_proposals_paused_by_usage_quota() -> bool:
    try:
        with app_server_pool.borrow_codex(Codex) as codex:
            response = codex._client.request(
                "account/rateLimits/read",
                None,
                response_model=GetAccountRateLimitsResponse,
            )
    except AppServerError:
        return False
    except Exception:
        system_agents.logger.exception(
            "failed to fetch account rate limits for auto-proposal quota pause"
        )
        return False

    now = timezone.now()
    for window in (response.rate_limits.primary, response.rate_limits.secondary):
        if window is not None and _rate_limit_window_below_auto_proposal_quota(
            window, now=now
        ):
            return True
    return False

def _rate_limit_window_below_auto_proposal_quota(
    window: Any, *, now: datetime
) -> bool:
    used_percent = getattr(window, "used_percent", None)
    resets_at = getattr(window, "resets_at", None)
    duration_mins = getattr(window, "window_duration_mins", None)
    if used_percent is None or resets_at is None or duration_mins is None:
        return False

    try:
        used = float(used_percent)
        reset_timestamp = float(resets_at)
        duration_seconds = float(duration_mins) * system_agents._SECONDS_PER_MINUTE
    except (TypeError, ValueError):
        return False
    if duration_seconds <= 0:
        return False

    if timezone.is_naive(now):
        now = now.replace(tzinfo=UTC)
    remaining_percent = 100 - max(0.0, min(100.0, used))
    reset_at = datetime.fromtimestamp(reset_timestamp, tz=UTC)
    seconds_until_reset = max(
        0.0, min((reset_at - now).total_seconds(), duration_seconds)
    )
    expected_remaining_percent = (seconds_until_reset / duration_seconds) * 100
    pause_threshold = (
        expected_remaining_percent * _AUTO_PROPOSAL_QUOTA_THRESHOLD_FRACTION
    )
    return remaining_percent < pause_threshold

def _maybe_start_auto_proposal_workflow(autonomous_goal_id: int) -> bool:
    autonomous_goal = (
        AutonomousGoal.objects.select_related("project").get(pk=autonomous_goal_id)
    )
    if not autonomous_goal.auto_proposal_enabled:
        return False
    start_snapshot = _autonomous_goal_auto_proposal_start_snapshot(autonomous_goal)
    default_branch_sha = _autonomous_goal_auto_proposal_start_sha(
        autonomous_goal,
        start_snapshot=start_snapshot,
    )
    if default_branch_sha is None:
        return False

    with transaction.atomic():
        autonomous_goal = (
            AutonomousGoal.objects.select_related("project")
            .select_for_update()
            .get(pk=autonomous_goal_id, deleted_at__isnull=True)
        )
        Project.objects.select_for_update().get(pk=autonomous_goal.project_id)
        if not autonomous_goal.auto_proposal_enabled:
            return False
        if not _autonomous_goal_auto_proposal_snapshot_matches(
            autonomous_goal, start_snapshot
        ):
            return False
        if not _autonomous_goal_auto_proposal_db_allows_start(
            autonomous_goal, default_branch_sha
        ):
            return False
        stack_continuation_proposal = (
            _autonomous_goal_stack_continuation_proposal(autonomous_goal)
        )
        if stack_continuation_proposal is not None:
            stack_continuation_proposal = (
                _claim_autonomous_goal_stack_continuation_proposal(
                    stack_continuation_proposal
                )
            )
            if stack_continuation_proposal is None:
                return False
        workflow, created = _create_autonomous_goal_workflow_record(
            autonomous_goal=autonomous_goal,
            auto_proposal=True,
            default_branch_sha=default_branch_sha,
            use_worktrees=True,
            stack_continuation_proposal=stack_continuation_proposal,
        )
    if created:
        _spawn_autonomous_goal_history_summary_or_candidate(workflow, autonomous_goal)
    return workflow.is_active

def _autonomous_goal_auto_proposal_db_allows_start(
    autonomous_goal: AutonomousGoal, current_sha: str
) -> bool:
    stack_continuation_proposal = _autonomous_goal_stack_continuation_proposal(
        autonomous_goal
    )
    if _autonomous_goal_pending_proposal_blocks_start(autonomous_goal):
        return False
    if _autonomous_goal_unresolved_failure_notice_exists(autonomous_goal):
        return False
    if _autonomous_goal_in_flight_automation_exists(autonomous_goal):
        return False
    if _project_running_auto_proposal_workflow_exists(autonomous_goal):
        return False
    if _autonomous_goal_running_workflow_exists(autonomous_goal):
        return False
    if stack_continuation_proposal is not None:
        return True
    last_no_proposal_sha = autonomous_goal.auto_proposal_last_no_proposal_sha.strip()
    return not last_no_proposal_sha or last_no_proposal_sha != current_sha

@dataclass(frozen=True)
class _AutonomousGoalAutoProposalStartSnapshot:
    project_id: int
    repo_path: str
    autonomy: str
    auto_qa_enabled: bool
    stacked_diff_depth: int
    proposal_budget: int | None
    auto_merge_to_local_branch: bool
    auto_merge_branch: str
    base_ref: str

@dataclass(frozen=True)
class _AutonomousGoalPostCommitAction:
    kind: str = ""
    candidate: dict[str, Any] | None = None
    cleanup_candidate_cwds: tuple[str, ...] = ()

def _autonomous_goal_auto_proposal_start_snapshot(
    autonomous_goal: AutonomousGoal,
) -> _AutonomousGoalAutoProposalStartSnapshot:
    return _AutonomousGoalAutoProposalStartSnapshot(
        project_id=autonomous_goal.project_id,
        repo_path=autonomous_goal.project.repo_path,
        autonomy=autonomous_goal.autonomy,
        auto_qa_enabled=autonomous_goal.auto_qa_enabled,
        stacked_diff_depth=autonomous_goal.effective_stacked_diff_depth,
        proposal_budget=autonomous_goal.proposal_budget,
        auto_merge_to_local_branch=autonomous_goal.auto_merge_to_local_branch,
        auto_merge_branch=autonomous_goal.auto_merge_branch,
        base_ref=_autonomous_goal_auto_merge_base_ref(autonomous_goal),
    )

def _autonomous_goal_auto_proposal_snapshot_matches(
    autonomous_goal: AutonomousGoal,
    start_snapshot: _AutonomousGoalAutoProposalStartSnapshot,
) -> bool:
    return _autonomous_goal_auto_proposal_start_snapshot(autonomous_goal) == start_snapshot

def _autonomous_goal_auto_proposal_start_sha(
    autonomous_goal: AutonomousGoal,
    *,
    start_snapshot: _AutonomousGoalAutoProposalStartSnapshot,
) -> str | None:
    stack_continuation_proposal = _autonomous_goal_stack_continuation_proposal(
        autonomous_goal
    )
    if _autonomous_goal_pending_proposal_blocks_start(autonomous_goal):
        return None
    if _autonomous_goal_unresolved_failure_notice_exists(autonomous_goal):
        return None
    if _autonomous_goal_in_flight_automation_exists(autonomous_goal):
        return None
    if _project_running_auto_proposal_workflow_exists(autonomous_goal):
        return None
    if _autonomous_goal_running_workflow_exists(autonomous_goal):
        return None

    current_sha = _autonomous_goal_auto_proposal_base_sha_for_snapshot(start_snapshot)
    if stack_continuation_proposal is not None:
        return current_sha or _AUTO_PROPOSAL_UNKNOWN_DEFAULT_BRANCH_SHA
    if not current_sha:
        return None
    last_no_proposal_sha = autonomous_goal.auto_proposal_last_no_proposal_sha.strip()
    if not last_no_proposal_sha:
        return current_sha
    if current_sha == last_no_proposal_sha:
        return None
    return current_sha

def _autonomous_goal_auto_proposal_base_sha(
    autonomous_goal: AutonomousGoal,
) -> str | None:
    return _autonomous_goal_auto_proposal_base_sha_for_snapshot(
        _autonomous_goal_auto_proposal_start_snapshot(autonomous_goal)
    )

def _autonomous_goal_auto_proposal_base_sha_for_snapshot(
    start_snapshot: _AutonomousGoalAutoProposalStartSnapshot,
) -> str | None:
    if start_snapshot.base_ref:
        return commit_hash_for_ref(start_snapshot.repo_path, start_snapshot.base_ref)
    return default_branch_commit_hash(start_snapshot.repo_path)

def _autonomous_goal_auto_merge_base_ref(
    autonomous_goal: AutonomousGoal,
) -> str:
    auto_merge_branch = _autonomous_goal_auto_merge_branch_for_implementation(
        autonomous_goal
    )
    return f"refs/heads/{auto_merge_branch}" if auto_merge_branch else ""

def _project_running_auto_proposal_workflow_exists(
    autonomous_goal: AutonomousGoal,
) -> bool:
    return SystemWorkflow.objects.filter(
        kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        cwd=autonomous_goal.project.repo_path,
        status=SystemWorkflow.STATUS_RUNNING,
        state__auto_proposal=True,
    ).exists()

def _autonomous_goal_running_workflow_exists(autonomous_goal: AutonomousGoal) -> bool:
    return SystemWorkflow.objects.filter(
        kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        main_thread_id=_autonomous_goal_main_thread_id(autonomous_goal.pk),
        status=SystemWorkflow.STATUS_RUNNING,
    ).exists()

def start_autonomous_goal_workflow(
    *,
    autonomous_goal: AutonomousGoal,
    auto_proposal: bool = False,
    default_branch_sha: str | None = None,
    use_worktrees: bool = False,
) -> SystemWorkflow:
    with transaction.atomic():
        autonomous_goal = (
            AutonomousGoal.objects.select_related("project")
            .select_for_update()
            .filter(pk=autonomous_goal.pk, deleted_at__isnull=True)
            .get()
        )
        workflow, created = _create_autonomous_goal_workflow_record(
            autonomous_goal=autonomous_goal,
            auto_proposal=auto_proposal,
            default_branch_sha=default_branch_sha,
            use_worktrees=use_worktrees,
            stack_continuation_proposal=None,
        )
    if created:
        _spawn_autonomous_goal_history_summary_or_candidate(workflow, autonomous_goal)
    return workflow

def _create_autonomous_goal_workflow_record(
    *,
    autonomous_goal: AutonomousGoal,
    auto_proposal: bool,
    default_branch_sha: str | None,
    use_worktrees: bool,
    stack_continuation_proposal: ProposedSession | None,
) -> tuple[SystemWorkflow, bool]:
    main_thread_id = _autonomous_goal_main_thread_id(autonomous_goal.pk)
    state: dict[str, Any] = {
        "autonomous_goal_id": autonomous_goal.pk,
        "auto_proposal": auto_proposal,
        _AUTONOMOUS_GOAL_USE_WORKTREES_STATE_KEY: use_worktrees,
        _AUTONOMOUS_GOAL_STACKED_DEPTH_STATE_KEY: (
            autonomous_goal.effective_stacked_diff_depth
        ),
        _AUTONOMOUS_GOAL_STACKED_ITERATION_STATE_KEY: 1,
        "autonomous_goal_updated_at": autonomous_goal.updated_at.isoformat(),
        "web_search_mode": autonomous_goal.web_search_mode,
    }
    if autonomous_goal.proposal_budget is not None:
        state[_AUTONOMOUS_GOAL_PROPOSAL_BUDGET_STATE_KEY] = (
            autonomous_goal.proposal_budget
        )
    if auto_proposal:
        default_branch_sha = default_branch_sha or (
            _autonomous_goal_auto_proposal_base_sha(autonomous_goal)
            or _AUTO_PROPOSAL_UNKNOWN_DEFAULT_BRANCH_SHA
        )
        state["default_branch_sha"] = default_branch_sha
    if stack_continuation_proposal is not None:
        stack_metadata = _autonomous_goal_proposal_stack_continuation_metadata(
            stack_continuation_proposal, autonomous_goal
        )
        if stack_metadata is None:
            raise ValueError("stack continuation proposal missing stack metadata")
        _seed_stack_continuation_proposal_budget_state(
            state, stack_continuation_proposal
        )
        state[_AUTONOMOUS_GOAL_STACKED_DEPTH_STATE_KEY] = stack_metadata.depth
        state[_AUTONOMOUS_GOAL_STACKED_ITERATION_STATE_KEY] = (
            stack_metadata.iteration + 1
        )
        state[_AUTONOMOUS_GOAL_STACKED_FORK_CWD_STATE_KEY] = (
            stack_continuation_proposal.candidate_session.cwd
            if stack_continuation_proposal.candidate_session is not None
            else ""
        )
        state["proposal_id"] = stack_continuation_proposal.pk
    try:
        with transaction.atomic():
            workflow = SystemWorkflow.objects.create(
                kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
                main_thread_id=main_thread_id,
                cwd=autonomous_goal.project.repo_path,
                status=SystemWorkflow.STATUS_RUNNING,
                step=_initial_autonomous_goal_workflow_step(autonomous_goal),
                state=state,
            )
    except IntegrityError:
        existing_workflow = SystemWorkflow.objects.filter(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=main_thread_id,
            status=SystemWorkflow.STATUS_RUNNING,
        ).first()
        if existing_workflow is None:
            raise
        return existing_workflow, False

    return workflow, True

def _initial_autonomous_goal_workflow_step(autonomous_goal: AutonomousGoal) -> str:
    if _autonomous_goal_resolved_proposal_history_exists(autonomous_goal):
        return system_agents.STEP_AUTONOMOUS_GOAL_HISTORY_SUMMARIZING
    return system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING


def _autonomous_goal_resolved_proposal_history_exists(
    autonomous_goal: AutonomousGoal,
) -> bool:
    return (
        autonomous_goal.proposed_sessions.filter(
            inbox_kind=ProposedSession.INBOX_KIND_PROPOSAL
        )
        .exclude(outcome_status=ProposedSession.OUTCOME_UNSET)
        .exists()
    )

def _seed_stack_continuation_proposal_budget_state(
    state: dict[str, Any], proposal: ProposedSession
) -> None:
    if _AUTONOMOUS_GOAL_PROPOSAL_BUDGET_STATE_KEY not in state:
        return
    metadata = _proposal_outcome_metadata(proposal, {})
    for key in (
        _AUTONOMOUS_GOAL_PROPOSAL_BUDGET_USED_STATE_KEY,
        _AUTONOMOUS_GOAL_FAILED_ATTEMPTS_STATE_KEY,
    ):
        value = _proposal_metadata_non_negative_int(metadata, key)
        if value is not None:
            state[key] = value


def _spawn_autonomous_goal_history_summary_or_candidate(
    workflow: SystemWorkflow, autonomous_goal: AutonomousGoal
) -> None:
    if workflow.step == system_agents.STEP_AUTONOMOUS_GOAL_HISTORY_SUMMARIZING:
        _spawn_autonomous_goal_history_summary_or_fallback(workflow, autonomous_goal)
        return
    _spawn_autonomous_goal_candidate_or_block(workflow, autonomous_goal)


def _spawn_autonomous_goal_history_summary_or_fallback(
    workflow: SystemWorkflow, autonomous_goal: AutonomousGoal
) -> None:
    original_workflow = workflow
    workflow, locked_goal, should_spawn = _claim_active_autonomous_goal_workflow(
        workflow_id=workflow.pk,
        autonomous_goal_id=autonomous_goal.pk,
    )
    system_agents._sync_workflow_instance(original_workflow, workflow)
    if not should_spawn or locked_goal is None:
        return
    if not _autonomous_goal_resolved_proposal_history_exists(locked_goal):
        workflow.state = _state_without_proposal_history_summary(workflow.state)
        system_agents._advance_workflow_step(
            workflow, system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING
        )
        system_agents._sync_workflow_instance(original_workflow, workflow)
        _spawn_autonomous_goal_candidate_or_block(workflow, locked_goal)
        return
    try:
        run = _spawn_autonomous_goal_history_summary_run(workflow, locked_goal)
    except _AutonomousGoalHistorySummaryUnpreservedError as exc:
        workflow = _block_autonomous_goal_spawn_failure_if_active(
            workflow_id=workflow.pk,
            autonomous_goal_id=locked_goal.pk,
            error=str(exc),
        )
        system_agents._sync_workflow_instance(original_workflow, workflow)
        return
    except Exception as exc:
        # The spawn helper either never created a summarizer or interrupted it
        # before re-raising, so this fallback cannot race an orphan summary.
        workflow = _record_autonomous_goal_history_summary_fallback_if_active(
            workflow_id=workflow.pk,
            autonomous_goal_id=locked_goal.pk,
            error=f"failed to start autonomous goal history summarizer: {exc!r}",
        )
        system_agents._sync_workflow_instance(original_workflow, workflow)
        if workflow is not None and workflow.is_active:
            _spawn_candidate_after_history_summary_fallback(workflow, locked_goal)
        return
    workflow = _interrupt_spawned_autonomous_goal_run_if_inactive(run)
    system_agents._sync_workflow_instance(original_workflow, workflow)


def _spawn_candidate_after_history_summary_fallback(
    workflow: SystemWorkflow, autonomous_goal: AutonomousGoal
) -> None:
    _spawn_autonomous_goal_candidate_or_block(workflow, autonomous_goal)


def _spawn_autonomous_goal_candidate_or_block(
    workflow: SystemWorkflow, autonomous_goal: AutonomousGoal
) -> None:
    original_workflow = workflow
    workflow, locked_goal, should_spawn = _claim_active_autonomous_goal_workflow(
        workflow_id=workflow.pk,
        autonomous_goal_id=autonomous_goal.pk,
    )
    system_agents._sync_workflow_instance(original_workflow, workflow)
    if not should_spawn or locked_goal is None:
        return
    try:
        run = _spawn_autonomous_goal_candidate_run(workflow, locked_goal)
    except Exception as exc:
        workflow = _block_autonomous_goal_spawn_failure_if_active(
            workflow_id=workflow.pk,
            autonomous_goal_id=locked_goal.pk,
            error=f"failed to start autonomous goal agent: {exc!r}",
        )
        system_agents._sync_workflow_instance(original_workflow, workflow)
        return
    workflow = _interrupt_spawned_autonomous_goal_run_if_inactive(run)
    system_agents._sync_workflow_instance(original_workflow, workflow)

def _spawn_autonomous_goal_candidate_retry_or_block(
    workflow: SystemWorkflow, autonomous_goal: AutonomousGoal
) -> None:
    workflow, locked_goal, should_spawn = _claim_active_autonomous_goal_workflow(
        workflow_id=workflow.pk,
        autonomous_goal_id=autonomous_goal.pk,
    )
    if not should_spawn or locked_goal is None:
        return
    try:
        run = _spawn_autonomous_goal_candidate_retry_run(workflow, locked_goal)
    except Exception as exc:
        _block_autonomous_goal_spawn_failure_if_active(
            workflow_id=workflow.pk,
            autonomous_goal_id=locked_goal.pk,
            error=f"failed to retry autonomous goal agent: {exc!r}",
        )
        return
    _interrupt_spawned_autonomous_goal_run_if_inactive(run)

def _claim_active_autonomous_goal_workflow(
    *, workflow_id: int, autonomous_goal_id: int
) -> tuple[SystemWorkflow, AutonomousGoal | None, bool]:
    with transaction.atomic():
        autonomous_goal = (
            AutonomousGoal.objects.select_related("project")
            .select_for_update()
            .filter(pk=autonomous_goal_id, deleted_at__isnull=True)
            .first()
        )
        workflow = SystemWorkflow.objects.select_for_update().get(pk=workflow_id)
        if not workflow.is_active:
            return workflow, autonomous_goal, False
        if autonomous_goal is None:
            system_agents._block_workflow(
                workflow,
                "autonomous goal no longer exists",
                surface_to_thread=False,
            )
            return workflow, None, False
        return workflow, autonomous_goal, True

def _block_autonomous_goal_spawn_failure_if_active(
    *, workflow_id: int, autonomous_goal_id: int, error: str
) -> SystemWorkflow:
    with transaction.atomic():
        autonomous_goal = (
            AutonomousGoal.objects.select_related("project")
            .select_for_update()
            .filter(pk=autonomous_goal_id, deleted_at__isnull=True)
            .first()
        )
        workflow = SystemWorkflow.objects.select_for_update().get(pk=workflow_id)
        if not workflow.is_active:
            return workflow
        if autonomous_goal is None:
            system_agents._block_workflow(
                workflow,
                "autonomous goal no longer exists",
                surface_to_thread=False,
            )
            return workflow
        if _complete_autonomous_goal_with_current_stack_proposal(
            workflow,
            error=error,
        ):
            return workflow
        _block_autonomous_goal_workflow(workflow, autonomous_goal, error)
        return workflow


def _record_autonomous_goal_history_summary_fallback_if_active(
    *, workflow_id: int, autonomous_goal_id: int, error: str
) -> SystemWorkflow:
    with transaction.atomic():
        autonomous_goal = (
            AutonomousGoal.objects.select_related("project")
            .select_for_update()
            .filter(pk=autonomous_goal_id, deleted_at__isnull=True)
            .first()
        )
        workflow = SystemWorkflow.objects.select_for_update().get(pk=workflow_id)
        if not workflow.is_active:
            return workflow
        if autonomous_goal is None:
            system_agents._block_workflow(
                workflow,
                "autonomous goal no longer exists",
                surface_to_thread=False,
            )
            return workflow
        workflow.state = _state_without_proposal_history_summary(workflow.state)
        workflow.state["proposal_history_summary_error"] = truncate_for_prompt(
            error, 800
        )
        system_agents._advance_workflow_step(
            workflow, system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING
        )
        return workflow

def _interrupt_spawned_autonomous_goal_run_if_inactive(
    run: SystemAgentRun,
) -> SystemWorkflow:
    workflow, _autonomous_goal, should_continue = _claim_active_autonomous_goal_workflow(
        workflow_id=run.workflow_id,
        autonomous_goal_id=int(run.input.get("autonomous_goal_id") or 0),
    )
    if should_continue:
        return workflow
    error = _state_string(workflow, "error") or "autonomous goal no longer exists"
    interrupted_runs, terminal_instance_returned = _interrupt_autonomous_goal_runs([run])
    if not interrupted_runs:
        return workflow
    system_agents._mark_system_agent_runs_failed(interrupted_runs, error)
    if terminal_instance_returned:
        _cleanup_autonomous_goal_workflow_worktree(workflow)
    return workflow

def stop_running_autonomous_goal_workflow(autonomous_goal_id: int, error: str) -> bool:
    """Stop a goal-owned workflow before the goal becomes unreachable.

    Returns ``False`` only when a running agent exists but could not be
    interrupted.
    """
    main_thread_id = _autonomous_goal_main_thread_id(autonomous_goal_id)
    system_agents.reconcile_terminal_workflow_instances(main_thread_id=main_thread_id)
    workflow = (
        SystemWorkflow.objects.filter(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=main_thread_id,
            status=SystemWorkflow.STATUS_RUNNING,
        )
        .order_by("-created_at")
        .first()
    )
    if workflow is None:
        return True
    runs = list(
        workflow.agent_runs.filter(status=SystemAgentRun.STATUS_RUNNING)
        .select_related("instance")
        .order_by("-created_at")
    )
    terminal_instance_returned = False
    if runs:
        interrupted_runs, terminal_instance_returned = _interrupt_autonomous_goal_runs(
            runs
        )
        if not interrupted_runs:
            return False
        system_agents._mark_system_agent_runs_failed(interrupted_runs, error)
    system_agents._block_workflow(workflow, error, surface_to_thread=False)
    if runs and terminal_instance_returned:
        _cleanup_autonomous_goal_workflow_worktree(workflow)
    return True

def stop_running_autonomous_goal_stack_after_proposal_resolution(
    autonomous_goal_id: int,
    proposal_id: int,
    outcome_status: str,
) -> bool:
    """Stop background stack work after the user resolves the current proposal."""
    error = _autonomous_goal_proposal_resolution_error(outcome_status)
    if not error:
        return True
    main_thread_id = _autonomous_goal_main_thread_id(autonomous_goal_id)
    system_agents.reconcile_terminal_workflow_instances(main_thread_id=main_thread_id)
    workflow = (
        SystemWorkflow.objects.filter(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=main_thread_id,
            status=SystemWorkflow.STATUS_RUNNING,
        )
        .order_by("-created_at")
        .first()
    )
    if workflow is None:
        return True
    terminal_instance_returned = False
    runs: list[SystemAgentRun] = []
    cleanup_cwd = ""
    with transaction.atomic():
        # The proposal id and running-run set are one lifecycle boundary: a stale
        # inbox decision must not complete a workflow that has advanced stacks.
        locked = SystemWorkflow.objects.select_for_update().get(pk=workflow.pk)
        if not locked.is_active:
            return True
        if _state_int(locked, "proposal_id") != proposal_id:
            return True
        runs = list(
            locked.agent_runs.select_for_update()
            .filter(status=SystemAgentRun.STATUS_RUNNING)
            .select_related("instance")
            .order_by("-created_at")
        )
        if runs:
            interrupted_runs, terminal_instance_returned = (
                _interrupt_autonomous_goal_runs(runs)
            )
            if not interrupted_runs:
                return False
            system_agents._mark_system_agent_runs_failed(interrupted_runs, error)
            if len(interrupted_runs) != len(runs):
                return False
        _complete_autonomous_goal_workflow_after_proposal_resolution(
            locked,
            outcome_status=outcome_status,
        )
        cleanup_cwd = _autonomous_goal_stack_resolution_continuation_cleanup_cwd(
            locked,
            proposal_id,
        )
        workflow = locked
    if cleanup_cwd and (not runs or terminal_instance_returned):
        _cleanup_autonomous_goal_candidate_cwd(cleanup_cwd)
    return True

def _autonomous_goal_proposal_resolution_error(outcome_status: str) -> str:
    return _AUTONOMOUS_GOAL_PROPOSAL_RESOLUTION_ERRORS.get(outcome_status, "")

def _autonomous_goal_stack_proposal_stop_reason(outcome_status: str) -> str:
    return _AUTONOMOUS_GOAL_STACKED_PROPOSAL_STOP_REASONS.get(outcome_status, "")

def _autonomous_goal_stack_resolution_continuation_cleanup_cwd(
    workflow: SystemWorkflow, proposal_id: int
) -> str:
    session_cwd = _autonomous_goal_session_cwd(workflow)
    if session_cwd == workflow.cwd:
        return ""
    # Between stack turns, session_cwd still belongs to the resolved proposal.
    # Only this helper owns cleanup for a distinct continuation worktree.
    protected_cwds = {
        _state_string(workflow, _AUTONOMOUS_GOAL_STACKED_FORK_CWD_STATE_KEY),
        _autonomous_goal_stack_proposal_candidate_cwd(workflow, proposal_id),
    }
    if session_cwd in protected_cwds:
        return ""
    return session_cwd

def _autonomous_goal_stack_proposal_candidate_cwd(
    workflow: SystemWorkflow, proposal_id: int
) -> str:
    proposal_query = ProposedSession.objects.select_related("candidate_session").filter(
        pk=proposal_id
    )
    autonomous_goal_id = _state_int(workflow, "autonomous_goal_id")
    if autonomous_goal_id:
        proposal_query = proposal_query.filter(autonomous_goal_id=autonomous_goal_id)
    else:
        proposal_query = proposal_query.filter(source_workflow=workflow)
    proposal = proposal_query.first()
    if proposal is None or proposal.candidate_session is None:
        return ""
    return proposal.candidate_session.cwd or ""

@engine.register
class _AutonomousGoalHandler(engine.WorkflowHandler):
    kind = system_agents.AUTONOMOUS_GOAL_AGENT_KIND
    steps = frozenset(
        {
            system_agents.STEP_AUTONOMOUS_GOAL_HISTORY_SUMMARIZING,
            system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
            system_agents.STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING,
            system_agents.STEP_AUTONOMOUS_GOAL_PROPOSED,
            system_agents.STEP_AUTONOMOUS_GOAL_SKIPPED,
        }
    )
    # Top-level SystemWorkflow.state keys this machine reads and writes (the
    # engine-shared turn-config/failure keys live in engine.SHARED_STATE_KEYS).
    state_keys = frozenset(
        {
            "auto_merge_branch",
            "auto_merge_to_local_branch",
            "auto_pr_enabled",
            "auto_proposal",
            "auto_qa_enabled",
            "autonomous_goal_id",
            "autonomous_goal_updated_at",
            "autonomy",
            "candidate",
            "candidate_session_id",
            "default_branch_sha",
            "history_files",
            "judge_session_id",
            "judgment",
            "proposal_history_files",
            "proposal_history_summary",
            "proposal_history_summary_error",
            "proposal_history_summary_session_id",
            "proposal_budget",
            "proposal_budget_failed_attempts",
            "proposal_budget_last_failure",
            "proposal_budget_no_progress_retries",
            "proposal_budget_token_totals",
            "proposal_budget_tokens_used",
            "proposal_id",
            "session_cwd",
            "stacked_diff_continuation_error",
            "stacked_diff_depth",
            "stacked_diff_fork_from_cwd",
            "stacked_diff_iteration",
            "stacked_diff_stopped_reason",
            "use_worktrees",
        }
    )

    @override
    def on_agent_finished(
        self,
        instance: CodexInstance,
        run: SystemAgentRun,
        workflow: SystemWorkflow,
    ) -> None:
        _handle_autonomous_goal_agent_finished(instance, run, workflow)

def _cleanup_cancelled_autonomous_goal_terminal_run(
    instance: CodexInstance, run: SystemAgentRun, workflow: SystemWorkflow
) -> None:
    if workflow.kind != system_agents.AUTONOMOUS_GOAL_AGENT_KIND:
        return
    if run.status != SystemAgentRun.STATUS_FAILED:
        return
    if (
        run.error != system_agents.AUTONOMOUS_GOAL_DELETED_ERROR
        and run.error not in _AUTONOMOUS_GOAL_PROPOSAL_RESOLUTION_ERROR_VALUES
    ):
        return
    if (
        run.error in _AUTONOMOUS_GOAL_PROPOSAL_RESOLUTION_ERROR_VALUES
        and workflow.agent_runs.filter(status=SystemAgentRun.STATUS_RUNNING).exists()
    ):
        return
    if instance.status not in (
        CodexInstance.STATUS_COMPLETED,
        CodexInstance.STATUS_FAILED,
    ):
        return
    _cleanup_autonomous_goal_workflow_worktree(workflow)

def _complete_autonomous_goal_workflow_after_proposal_resolution(
    workflow: SystemWorkflow, *, outcome_status: str
) -> None:
    reason = _autonomous_goal_stack_proposal_stop_reason(outcome_status)
    if not reason:
        return
    workflow.state = {
        **workflow.state,
        "stacked_diff_stopped_reason": reason,
    }
    system_agents._complete_workflow(workflow, system_agents.STEP_AUTONOMOUS_GOAL_PROPOSED)

def _handle_autonomous_goal_agent_finished(
    instance: CodexInstance, run: SystemAgentRun, workflow: SystemWorkflow
) -> None:
    autonomous_goal_id = _state_int(workflow, "autonomous_goal_id")
    post_commit_action: _AutonomousGoalPostCommitAction | None = None
    autonomous_goal: AutonomousGoal | None = None
    # Read and parse the agent's JSONL events file before taking the write
    # lock: doing it inside the IMMEDIATE/select_for_update transaction below
    # would hold SQLite's single global writer for the whole (unbounded) file
    # read+parse. Mirrors the QA/spec-critic finish handlers, which all read
    # ``_final_agent_text`` before their locked section.
    raw_output = system_agents._final_agent_text(instance.events_path)
    tokens_used = _autonomous_goal_instance_tokens_used(instance)
    with transaction.atomic():
        autonomous_goal = (
            AutonomousGoal.objects.select_related("project")
            .select_for_update()
            .filter(
                pk=autonomous_goal_id,
                deleted_at__isnull=True,
            )
            .first()
        )
        run = SystemAgentRun.objects.select_for_update().get(pk=run.pk)
        workflow = SystemWorkflow.objects.select_for_update().get(pk=workflow.pk)
        run.workflow = workflow
        if run.status in (
            SystemAgentRun.STATUS_COMPLETED,
            SystemAgentRun.STATUS_FAILED,
        ):
            return
        if not workflow.is_active:
            run.status = SystemAgentRun.STATUS_FAILED
            run.error = (
                _state_string(workflow, "error")
                or "autonomous goal workflow is no longer active"
            )
            run.raw_output = raw_output
            run.save(update_fields=["status", "error", "raw_output", "updated_at"])
            return
        if autonomous_goal is None:
            system_agents._fail_run_and_block_workflow(
                run,
                "autonomous goal no longer exists",
                surface_to_thread=False,
            )
            return
        token_delta = _record_autonomous_goal_proposal_budget_tokens(
            workflow, instance, tokens_used
        )
        post_commit_action = _handle_autonomous_goal_agent_finished_locked(
            instance,
            run,
            workflow,
            autonomous_goal,
            raw_output,
            tokens_used,
            token_delta,
        )
    if post_commit_action is None:
        return
    for cwd in post_commit_action.cleanup_candidate_cwds:
        _cleanup_autonomous_goal_candidate_cwd(cwd)
    if autonomous_goal is None:
        return
    if post_commit_action.kind == _AUTONOMOUS_GOAL_SPAWN_JUDGE_ACTION:
        if post_commit_action.candidate is not None:
            _spawn_autonomous_goal_judge_or_block(
                workflow, autonomous_goal, post_commit_action.candidate
            )
        return
    if post_commit_action.kind == _AUTONOMOUS_GOAL_RETRY_CANDIDATE_ACTION:
        _spawn_autonomous_goal_candidate_or_block(workflow, autonomous_goal)
        return
    if post_commit_action.kind == _AUTONOMOUS_GOAL_RETRY_CANDIDATE_CONTINUATION_ACTION:
        _spawn_autonomous_goal_candidate_retry_or_block(workflow, autonomous_goal)
        return
    if (
        post_commit_action.kind == _AUTONOMOUS_GOAL_RETRY_JUDGE_ACTION
        and post_commit_action.candidate is not None
    ):
        _spawn_autonomous_goal_judge_or_block(
            workflow, autonomous_goal, post_commit_action.candidate
        )
        return
    if post_commit_action.kind == _AUTONOMOUS_GOAL_SPAWN_NEXT_CANDIDATE_ACTION:
        _spawn_autonomous_goal_history_summary_or_candidate(workflow, autonomous_goal)
        return

def _handle_autonomous_goal_agent_finished_locked(
    instance: CodexInstance,
    run: SystemAgentRun,
    workflow: SystemWorkflow,
    autonomous_goal: AutonomousGoal,
    raw_output: str,
    tokens_used: int | None,
    token_delta: int,
) -> _AutonomousGoalPostCommitAction | None:
    if not workflow.is_active:
        return None
    (
        proposal_outcome,
        resolved_proposal_cleanup_cwd,
    ) = _autonomous_goal_current_stack_proposal_resolution(workflow)
    if proposal_outcome:
        run.status = SystemAgentRun.STATUS_FAILED
        run.error = _autonomous_goal_proposal_resolution_error(proposal_outcome)
        run.save(update_fields=["status", "error", "updated_at"])
        _complete_autonomous_goal_workflow_after_proposal_resolution(
            workflow,
            outcome_status=proposal_outcome,
        )
        cleanup_cwd = system_agents._candidate_session_cwd_from_state(
            workflow, "candidate_session_id"
        )
        resolution_cleanup_cwds = tuple(
            dict.fromkeys(
                cwd
                for cwd in (cleanup_cwd, resolved_proposal_cleanup_cwd)
                if cwd
            )
        )
        return _AutonomousGoalPostCommitAction(
            cleanup_candidate_cwds=resolution_cleanup_cwds
        )
    if instance.status != CodexInstance.STATUS_COMPLETED:
        if workflow.step == system_agents.STEP_AUTONOMOUS_GOAL_HISTORY_SUMMARIZING:
            return _fallback_from_failed_autonomous_goal_history_summary(
                run,
                workflow,
                autonomous_goal,
                error=f"autonomous goal history summarizer failed: {instance.error}",
                raw_output=raw_output,
            )
        error = f"autonomous goal worker failed: {instance.error}"
        if system_agents._is_worker_exited_before_completion_error(instance.error):
            retry_action = _retry_dead_autonomous_goal_worker(instance, run, workflow)
            if retry_action is not None:
                return retry_action
            retry_action = _retry_budgeted_failed_autonomous_goal_candidate(
                run,
                workflow,
                error=error,
                raw_output=raw_output,
                tokens_used=tokens_used,
                token_delta=token_delta,
            )
            if retry_action is not None:
                return retry_action
            return _fail_autonomous_goal_run_and_block_workflow(
                run,
                autonomous_goal,
                error,
            )
        retry_action = _retry_budgeted_failed_autonomous_goal_candidate(
            run,
            workflow,
            error=error,
            raw_output=raw_output,
            tokens_used=tokens_used,
            token_delta=token_delta,
        )
        if retry_action is not None:
            return retry_action
        return _fail_autonomous_goal_run_and_block_workflow(
            run,
            autonomous_goal,
            f"autonomous goal worker failed: {instance.error}",
        )

    if workflow.step == system_agents.STEP_AUTONOMOUS_GOAL_HISTORY_SUMMARIZING:
        summary = _parse_autonomous_goal_history_summary_output(raw_output)
        if summary is None:
            return _fallback_from_failed_autonomous_goal_history_summary(
                run,
                workflow,
                autonomous_goal,
                error="autonomous goal history summarizer output was not valid JSON",
                raw_output=raw_output,
            )
        run.status = SystemAgentRun.STATUS_COMPLETED
        run.output = summary
        run.raw_output = raw_output
        run.save(update_fields=["status", "output", "raw_output", "updated_at"])
        workflow.state = {
            **system_agents._state_without_workflow_turn_death_retry(
                workflow.state, _AUTONOMOUS_GOAL_HISTORY_SUMMARY_RETRY_KIND
            ),
            _AUTONOMOUS_GOAL_PROPOSAL_HISTORY_SUMMARY_STATE_KEY: summary,
        }
        workflow.state.pop("proposal_history_summary_error", None)
        if _autonomous_goal_proposal_budget_exhausted(workflow):
            return _stop_autonomous_goal_after_history_summary_budget(
                workflow, autonomous_goal
            )
        return _advance_to_candidate_after_history_summary(workflow)

    if workflow.step == system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING:
        candidate_output = _parse_autonomous_goal_candidate_output(raw_output)
        if candidate_output is None:
            retry_action = _retry_budgeted_failed_autonomous_goal_candidate(
                run,
                workflow,
                error="autonomous goal candidate output was not valid JSON",
                raw_output=raw_output,
                tokens_used=tokens_used,
                token_delta=token_delta,
            )
            if retry_action is not None:
                return retry_action
            return _fail_autonomous_goal_run_and_block_workflow(
                run,
                autonomous_goal,
                "autonomous goal candidate output was not valid JSON",
                raw_output,
            )
        run.status = SystemAgentRun.STATUS_COMPLETED
        run.output = candidate_output
        run.raw_output = raw_output
        run.save(update_fields=["status", "output", "raw_output", "updated_at"])
        _store_autonomous_goal_memory(autonomous_goal, workflow, candidate_output)
        state = system_agents._state_without_workflow_turn_death_retry(
            workflow.state, _AUTONOMOUS_GOAL_CANDIDATE_RETRY_KIND
        )
        if candidate_output["proposal"] is None:
            previous_proposal = _autonomous_goal_current_stack_proposal(workflow)
            message = str(candidate_output["message"])
            workflow.state = state
            retry_action = _retry_budgeted_unaccepted_autonomous_goal_candidate(
                workflow,
                reason="candidate_no_proposal",
                message=message,
                tokens_used=tokens_used,
                token_delta=token_delta,
            )
            if retry_action is not None:
                workflow.state = system_agents._state_without_current_candidate_result(
                    workflow.state
                )
                system_agents._advance_workflow_step(workflow, system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING)
                return retry_action
            if previous_proposal is not None and _publish_current_stack_proposal(
                previous_proposal,
                workflow=workflow,
                continuation_stopped_reason="candidate_no_proposal",
            ):
                cleanup_cwd = system_agents._candidate_session_cwd_from_state(
                    workflow, "candidate_session_id"
                )
                workflow.state = {
                    **state,
                    "candidate": candidate_output,
                    "stacked_diff_stopped_reason": "candidate_no_proposal",
                }
                system_agents._complete_workflow(workflow, system_agents.STEP_AUTONOMOUS_GOAL_PROPOSED)
                return _AutonomousGoalPostCommitAction(
                    cleanup_candidate_cwds=((cleanup_cwd,) if cleanup_cwd else ())
                )
            ProposedSession.objects.create(
                project=autonomous_goal.project,
                autonomous_goal=autonomous_goal,
                source_workflow=workflow,
                title=f"No proposal from {autonomous_goal.title}"[
                    :_AUTONOMOUS_GOAL_TITLE_MAX_LEN
                ],
                inbox_kind=ProposedSession.INBOX_KIND_NOTICE,
                summary=message,
                candidate_session=_session_metadata_from_state(
                    workflow, "candidate_session_id"
                ),
                outcome_metadata=_autonomous_goal_proposal_budget_metadata(workflow),
            )
            _record_autonomous_goal_no_proposal(autonomous_goal, workflow)
            workflow.state = {**state, "candidate": candidate_output}
            system_agents._complete_workflow(workflow, system_agents.STEP_AUTONOMOUS_GOAL_SKIPPED)
            return None
        candidate = cast(dict[str, Any], candidate_output["proposal"])
        workflow.state = {**state, "candidate": candidate}
        system_agents._advance_workflow_step(workflow, system_agents.STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING)
        return _AutonomousGoalPostCommitAction(
            _AUTONOMOUS_GOAL_SPAWN_JUDGE_ACTION,
            candidate,
        )

    if workflow.step != system_agents.STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING:
        return None
    judgment = _parse_autonomous_goal_judge_output(raw_output)
    if judgment is None:
        return _fail_autonomous_goal_run_and_block_workflow(
            run,
            autonomous_goal,
            "autonomous goal judge output was not valid JSON",
            raw_output,
        )
    run.status = SystemAgentRun.STATUS_COMPLETED
    run.output = judgment
    run.raw_output = raw_output
    run.save(update_fields=["status", "output", "raw_output", "updated_at"])

    state = system_agents._state_without_workflow_turn_death_retry(
        workflow.state, _AUTONOMOUS_GOAL_JUDGE_RETRY_KIND
    )
    candidate = workflow.state.get("candidate")
    if not isinstance(candidate, dict):
        candidate = {}
    cleanup_cwds: tuple[str, ...] = ()
    if _confidence_meets_threshold(
        judgment["confidence"], autonomous_goal.confidence_threshold
    ):
        should_continue_stack = _autonomous_goal_should_continue_stack(
            workflow, autonomous_goal
        )
        previous_proposal = _autonomous_goal_current_stack_proposal(workflow)
        proposal = _create_autonomous_goal_proposal(
            workflow,
            autonomous_goal,
            candidate,
            judgment,
        )
        state = _state_after_autonomous_goal_proposal_progress(state)
        cleanup_cwds = _dismiss_replaced_autonomous_goal_proposal(
            previous_proposal, replacement=proposal
        )
        _record_autonomous_goal_proposal_created(autonomous_goal)
        workflow.state = {
            **state,
            "judgment": judgment,
            "proposal_id": proposal.pk,
            "autonomy": autonomous_goal.autonomy,
        }
        if should_continue_stack:
            workflow.state = _autonomous_goal_next_stack_candidate_state(
                workflow, proposal
            )
            workflow.state = _state_without_proposal_history_summary(workflow.state)
            system_agents._advance_workflow_step(
                workflow, system_agents.STEP_AUTONOMOUS_GOAL_HISTORY_SUMMARIZING
            )
            return _AutonomousGoalPostCommitAction(
                _AUTONOMOUS_GOAL_SPAWN_NEXT_CANDIDATE_ACTION,
                cleanup_candidate_cwds=cleanup_cwds,
            )
        workflow.step = system_agents.STEP_AUTONOMOUS_GOAL_PROPOSED
    else:
        previous_proposal = _autonomous_goal_current_stack_proposal(workflow)
        workflow.state = state
        retry_action = _retry_budgeted_unaccepted_autonomous_goal_candidate(
            workflow,
            reason="judge_confidence_below_threshold",
            candidate=candidate,
            judgment=judgment,
            tokens_used=tokens_used,
            token_delta=token_delta,
        )
        if retry_action is not None:
            workflow.state = system_agents._state_without_current_candidate_result(workflow.state)
            system_agents._advance_workflow_step(workflow, system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING)
            return retry_action
        if previous_proposal is not None and _publish_current_stack_proposal(
            previous_proposal,
            workflow=workflow,
            continuation_stopped_reason="judge_confidence_below_threshold",
        ):
            cleanup_cwd = system_agents._candidate_session_cwd_from_state(
                workflow, "candidate_session_id"
            )
            workflow.state = {
                **state,
                "judgment": judgment,
                "stacked_diff_stopped_reason": "judge_confidence_below_threshold",
            }
            system_agents._complete_workflow(workflow, system_agents.STEP_AUTONOMOUS_GOAL_PROPOSED)
            return _AutonomousGoalPostCommitAction(
                cleanup_candidate_cwds=((cleanup_cwd,) if cleanup_cwd else ())
            )
        _create_autonomous_goal_skipped_notice(
            workflow,
            autonomous_goal,
            title=_below_threshold_notice_title(candidate, autonomous_goal),
            summary=_below_threshold_notice_summary(
                candidate, judgment, autonomous_goal.confidence_threshold
            ),
            metadata={
                "automation_status": "skipped",
                "skip_reason": "judge_confidence_below_threshold",
                "judge_confidence": judgment["confidence"],
                "confidence_threshold": autonomous_goal.confidence_threshold,
                "candidate_title": _candidate_notice_title(candidate),
                "judge_rationale": judgment["rationale"],
            },
        )
        _record_autonomous_goal_no_proposal(autonomous_goal, workflow)
        workflow.step = system_agents.STEP_AUTONOMOUS_GOAL_SKIPPED
        workflow.state = state
    workflow.status = SystemWorkflow.STATUS_COMPLETED
    workflow.state = {**workflow.state, "judgment": judgment}
    workflow.save(update_fields=["status", "step", "state", "updated_at"])
    return _AutonomousGoalPostCommitAction(cleanup_candidate_cwds=cleanup_cwds)

def _create_autonomous_goal_proposal(
    workflow: SystemWorkflow,
    autonomous_goal: AutonomousGoal,
    candidate: dict[str, Any],
    judgment: dict[str, str],
    *,
    publish: bool = True,
) -> ProposedSession:
    auto_pr_enabled = autonomous_goal.autonomy == AutonomousGoal.AUTONOMY_DRAFT_PR
    auto_qa_enabled = autonomous_goal.auto_qa_enabled and not auto_pr_enabled
    auto_merge_branch = _autonomous_goal_auto_merge_branch_for_implementation(
        autonomous_goal
    )
    auto_merge_to_local_branch = bool(auto_qa_enabled and auto_merge_branch)
    hidden_until_complete = not publish
    return ProposedSession.objects.create(
        project=autonomous_goal.project,
        autonomous_goal=autonomous_goal,
        source_workflow=workflow,
        title=str(candidate.get("title", autonomous_goal.title))[
            :_AUTONOMOUS_GOAL_TITLE_MAX_LEN
        ],
        summary=_autonomous_goal_proposal_summary(candidate, judgment),
        prompt=_autonomous_goal_proposed_session_prompt(
            autonomous_goal, candidate, judgment
        ),
        confidence=judgment["confidence"],
        relevant_files=_string_list(candidate.get("relevant_files")),
        candidate_session=_session_metadata_from_state(
            workflow, "candidate_session_id"
        ),
        judge_session=_session_metadata_from_state(workflow, "judge_session_id"),
        outcome_status=(
            ProposedSession.OUTCOME_UNSET
            if publish
            else ProposedSession.OUTCOME_DISMISSED
        ),
        outcome_notes=(
            "" if publish else _AUTONOMOUS_GOAL_STACKED_HIDDEN_OUTCOME_NOTES
        ),
        outcome_metadata={
            "autonomous_goal_autonomy": autonomous_goal.autonomy,
            "automation_status": "proposed",
            "auto_pr_enabled": auto_pr_enabled,
            "auto_qa_enabled": auto_qa_enabled,
            "auto_merge_to_local_branch": auto_merge_to_local_branch,
            "auto_merge_branch": auto_merge_branch,
            "stacked_diff_depth": _autonomous_goal_workflow_stacked_diff_depth(
                workflow, autonomous_goal
            ),
            "stacked_diff_iteration": _autonomous_goal_stack_iteration(workflow),
            "stacked_diff_hidden_until_complete": hidden_until_complete,
            "implemented_changes": str(
                candidate.get("implemented_changes", "")
            ).strip(),
            "verification": str(candidate.get("verification", "")).strip(),
            "rough_edges": str(candidate.get("rough_edges", "")).strip(),
            **_autonomous_goal_proposal_budget_metadata(workflow),
        },
    )

def _autonomous_goal_current_stack_proposal(
    workflow: SystemWorkflow,
) -> ProposedSession | None:
    proposal_id = _state_int(workflow, "proposal_id")
    if not proposal_id:
        return None
    proposal_query = ProposedSession.objects.select_related("candidate_session").filter(
        pk=proposal_id
    )
    autonomous_goal_id = _state_int(workflow, "autonomous_goal_id")
    if autonomous_goal_id:
        proposal_query = proposal_query.filter(autonomous_goal_id=autonomous_goal_id)
    else:
        proposal_query = proposal_query.filter(source_workflow=workflow)
    proposal = proposal_query.first()
    if proposal is None:
        return None
    if proposal.outcome_status == ProposedSession.OUTCOME_UNSET:
        return proposal
    if _autonomous_goal_proposal_hidden_until_complete(proposal):
        return proposal
    return None

def _autonomous_goal_current_stack_proposal_resolution(
    workflow: SystemWorkflow,
) -> tuple[str, str]:
    proposal_id = _state_int(workflow, "proposal_id")
    if not proposal_id:
        return "", ""
    proposal_query = ProposedSession.objects.select_related("candidate_session").filter(
        pk=proposal_id
    )
    autonomous_goal_id = _state_int(workflow, "autonomous_goal_id")
    if autonomous_goal_id:
        proposal_query = proposal_query.filter(autonomous_goal_id=autonomous_goal_id)
    else:
        proposal_query = proposal_query.filter(source_workflow=workflow)
    proposal = proposal_query.first()
    if proposal is None:
        return "", ""
    if _autonomous_goal_proposal_hidden_until_complete(proposal):
        return "", ""
    if _autonomous_goal_proposal_resolution_error(proposal.outcome_status):
        return (
            proposal.outcome_status,
            system_agents._resolved_stack_proposal_candidate_cleanup_cwd(proposal),
        )
    return "", ""

def _autonomous_goal_instance_tokens_used(instance: CodexInstance) -> int | None:
    """Cumulative thread token usage for an autonomous-goal worker.

    Codex only emits ``thread/goal/updated`` (and its ``tokensUsed`` counter)
    when the model itself sets a thread goal, which the hidden candidate and
    judge sessions normally never do -- reading only goal events left budget
    tracking at ``None`` forever, so proposals displayed zero tokens and every
    budgeted retry counted against the no-progress cap instead of the budget.
    Prefer the rollout file's per-turn TokenCount totals, which Codex always
    persists, keeping goal events as a fallback when no rollout is readable.
    """
    rollout_tokens = _rollout_total_tokens_for_thread(instance.thread_id)
    goal_tokens = codex_events.latest_goal_tokens_for_instance(instance)
    if rollout_tokens is None:
        return goal_tokens
    if goal_tokens is None:
        return rollout_tokens
    return max(rollout_tokens, goal_tokens)

def _rollout_total_tokens_for_thread(thread_id: str) -> int | None:
    if not thread_id:
        return None
    codex_path = (
        SessionMetadata.objects.filter(thread_id=thread_id)
        .exclude(codex_path="")
        .values_list("codex_path", flat=True)
        .first()
    )
    rollout_path = _rollout_path_from_value(codex_path)
    if rollout_path is None:
        return None
    usage = rollout.latest_token_usage(rollout_path)
    if usage is None:
        return None
    return usage["total_tokens"]

def _record_autonomous_goal_proposal_budget_tokens(
    workflow: SystemWorkflow, instance: CodexInstance, tokens_used: int | None
) -> int:
    if _autonomous_goal_workflow_proposal_budget(workflow) <= 0:
        return 0
    if tokens_used is None or tokens_used < 0:
        return 0
    token_totals = _state_dict(
        workflow, _AUTONOMOUS_GOAL_PROPOSAL_BUDGET_TOKEN_TOTALS_STATE_KEY
    )
    previous_value = token_totals.get(instance.thread_id)
    previous_tokens = (
        previous_value
        if isinstance(previous_value, int)
        and not isinstance(previous_value, bool)
        and previous_value >= 0
        else 0
    )
    thread_tokens = max(previous_tokens, tokens_used)
    token_delta = thread_tokens - previous_tokens
    next_state = {
        **workflow.state,
        _AUTONOMOUS_GOAL_PROPOSAL_BUDGET_USED_STATE_KEY: (
            _autonomous_goal_proposal_budget_tokens_used(workflow) + token_delta
        ),
        _AUTONOMOUS_GOAL_PROPOSAL_BUDGET_TOKEN_TOTALS_STATE_KEY: {
            **token_totals,
            instance.thread_id: thread_tokens,
        },
    }
    if token_delta > 0:
        next_state.pop(_AUTONOMOUS_GOAL_NO_PROGRESS_RETRIES_STATE_KEY, None)
    workflow.state = next_state
    return token_delta

def _retry_budgeted_failed_autonomous_goal_candidate(
    run: SystemAgentRun,
    workflow: SystemWorkflow,
    *,
    error: str,
    raw_output: str,
    tokens_used: int | None,
    token_delta: int,
) -> _AutonomousGoalPostCommitAction | None:
    if workflow.step != system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING:
        return None
    if not _autonomous_goal_proposal_budget_allows_retry(
        workflow, tokens_used=tokens_used, token_delta=token_delta
    ):
        return None
    run.status = SystemAgentRun.STATUS_FAILED
    run.error = error
    run.raw_output = raw_output
    run.save(update_fields=["status", "error", "raw_output", "updated_at"])
    _record_autonomous_goal_failed_attempt(
        workflow,
        reason="candidate_failed",
        error=error,
        raw_output=raw_output,
        tokens_used=tokens_used,
        token_delta=token_delta,
    )
    workflow.save(update_fields=["state", "updated_at"])
    return _AutonomousGoalPostCommitAction(
        _AUTONOMOUS_GOAL_RETRY_CANDIDATE_CONTINUATION_ACTION
    )

def _retry_budgeted_unaccepted_autonomous_goal_candidate(
    workflow: SystemWorkflow,
    *,
    reason: str,
    tokens_used: int | None,
    token_delta: int,
    candidate: dict[str, Any] | None = None,
    judgment: dict[str, Any] | None = None,
    message: str = "",
) -> _AutonomousGoalPostCommitAction | None:
    if _session_metadata_from_state(workflow, "candidate_session_id") is None:
        return None
    if not _autonomous_goal_proposal_budget_allows_retry(
        workflow, tokens_used=tokens_used, token_delta=token_delta
    ):
        return None
    _record_autonomous_goal_failed_attempt(
        workflow,
        reason=reason,
        message=message,
        candidate=candidate,
        judgment=judgment,
        tokens_used=tokens_used,
        token_delta=token_delta,
    )
    return _AutonomousGoalPostCommitAction(
        _AUTONOMOUS_GOAL_RETRY_CANDIDATE_CONTINUATION_ACTION
    )

def _advance_to_candidate_after_history_summary(
    workflow: SystemWorkflow,
) -> _AutonomousGoalPostCommitAction:
    system_agents._advance_workflow_step(
        workflow, system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING
    )
    return _AutonomousGoalPostCommitAction(_AUTONOMOUS_GOAL_RETRY_CANDIDATE_ACTION)

def _autonomous_goal_proposal_budget_allows_retry(
    workflow: SystemWorkflow, *, tokens_used: int | None, token_delta: int
) -> bool:
    budget = _autonomous_goal_workflow_proposal_budget(workflow)
    if budget <= 0:
        return False
    if _autonomous_goal_proposal_budget_tokens_used(workflow) >= budget:
        return False
    if _autonomous_goal_budget_token_progressed(
        tokens_used=tokens_used, token_delta=token_delta
    ):
        return True
    return (
        _autonomous_goal_no_progress_budget_retries(workflow)
        < _AUTONOMOUS_GOAL_NO_PROGRESS_RETRY_LIMIT
    )

def _autonomous_goal_proposal_budget_exhausted(workflow: SystemWorkflow) -> bool:
    budget = _autonomous_goal_workflow_proposal_budget(workflow)
    return budget > 0 and _autonomous_goal_proposal_budget_tokens_used(workflow) >= budget

def _stop_autonomous_goal_after_history_summary_budget(
    workflow: SystemWorkflow,
    autonomous_goal: AutonomousGoal,
) -> _AutonomousGoalPostCommitAction | None:
    error = "autonomous goal proposal budget was exhausted by the history summarizer"
    if _complete_autonomous_goal_with_current_stack_proposal(workflow, error=error):
        cleanup_cwd = system_agents._candidate_session_cwd_from_state(
            workflow, "candidate_session_id"
        )
        return _AutonomousGoalPostCommitAction(
            cleanup_candidate_cwds=((cleanup_cwd,) if cleanup_cwd else ())
        )
    _block_autonomous_goal_workflow(workflow, autonomous_goal, error)
    return None

def _autonomous_goal_budget_token_progressed(
    *, tokens_used: int | None, token_delta: int
) -> bool:
    return tokens_used is not None and tokens_used > 0 and token_delta > 0

def _record_autonomous_goal_failed_attempt(
    workflow: SystemWorkflow,
    *,
    reason: str,
    error: str = "",
    raw_output: str = "",
    message: str = "",
    candidate: dict[str, Any] | None = None,
    judgment: dict[str, Any] | None = None,
    tokens_used: int | None = None,
    token_delta: int = 0,
) -> None:
    failure: dict[str, object] = {
        "reason": reason,
        "proposal_budget": _autonomous_goal_workflow_proposal_budget(workflow),
        "proposal_budget_tokens_used": _autonomous_goal_proposal_budget_tokens_used(
            workflow
        ),
    }
    if tokens_used is not None:
        failure["tokens_used"] = tokens_used
    if error:
        failure["error"] = truncate_for_prompt(error, 800)
    if message:
        failure["message"] = truncate_for_prompt(message, 1200)
    if raw_output:
        failure["raw_output"] = truncate_for_prompt(raw_output, 2000)
    if candidate is not None:
        failure["candidate"] = _autonomous_goal_failed_candidate_context(candidate)
    if judgment is not None:
        failure["judgment"] = _autonomous_goal_failed_judgment_context(judgment)
    next_state = {
        **workflow.state,
        _AUTONOMOUS_GOAL_FAILED_ATTEMPTS_STATE_KEY: (
            _autonomous_goal_failed_attempts(workflow) + 1
        ),
        _AUTONOMOUS_GOAL_LAST_FAILURE_STATE_KEY: failure,
    }
    if _autonomous_goal_workflow_proposal_budget(workflow) > 0:
        if _autonomous_goal_budget_token_progressed(
            tokens_used=tokens_used, token_delta=token_delta
        ):
            next_state.pop(_AUTONOMOUS_GOAL_NO_PROGRESS_RETRIES_STATE_KEY, None)
        else:
            next_state[_AUTONOMOUS_GOAL_NO_PROGRESS_RETRIES_STATE_KEY] = (
                _autonomous_goal_no_progress_budget_retries(workflow) + 1
            )
    workflow.state = next_state

def _autonomous_goal_failed_candidate_context(
    candidate: dict[str, Any]
) -> dict[str, object]:
    context: dict[str, object] = {}
    for key in (
        "title",
        "summary",
        "impact",
        "implemented_changes",
        "implementation_direction",
        "verification",
        "rough_edges",
    ):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            context[key] = truncate_for_prompt(value, 800)
    files = _string_list(candidate.get("relevant_files"))
    if files:
        context["relevant_files"] = files[:20]
    return context

def _autonomous_goal_failed_judgment_context(
    judgment: dict[str, Any]
) -> dict[str, object]:
    context: dict[str, object] = {}
    for key in ("confidence", "summary", "rationale"):
        value = judgment.get(key)
        if isinstance(value, str) and value.strip():
            context[key] = truncate_for_prompt(value, 1200)
    return context

def _state_after_autonomous_goal_proposal_progress(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    next_state = dict(state)
    next_state.pop(_AUTONOMOUS_GOAL_NO_PROGRESS_RETRIES_STATE_KEY, None)
    next_state.pop(_AUTONOMOUS_GOAL_LAST_FAILURE_STATE_KEY, None)
    return next_state

def _retry_dead_autonomous_goal_worker(
    instance: CodexInstance,
    run: SystemAgentRun,
    workflow: SystemWorkflow,
) -> _AutonomousGoalPostCommitAction | None:
    retry_kind = _autonomous_goal_worker_retry_kind(workflow)
    candidate: dict[str, Any] | None = None
    if retry_kind == _AUTONOMOUS_GOAL_CANDIDATE_RETRY_KIND:
        action_kind = _AUTONOMOUS_GOAL_RETRY_CANDIDATE_ACTION
    elif retry_kind:
        raw_candidate = workflow.state.get("candidate")
        if not isinstance(raw_candidate, dict):
            return None
        candidate = raw_candidate
        action_kind = _AUTONOMOUS_GOAL_RETRY_JUDGE_ACTION
    else:
        return None
    if not system_agents._claim_workflow_turn_death_retry(workflow, instance, retry_kind):
        return None

    run.status = SystemAgentRun.STATUS_FAILED
    run.error = f"autonomous goal worker failed: {instance.error}"
    run.save(update_fields=["status", "error", "updated_at"])
    return _AutonomousGoalPostCommitAction(action_kind, candidate)

def _autonomous_goal_worker_retry_kind(workflow: SystemWorkflow) -> str:
    if workflow.step == system_agents.STEP_AUTONOMOUS_GOAL_HISTORY_SUMMARIZING:
        return _AUTONOMOUS_GOAL_HISTORY_SUMMARY_RETRY_KIND
    if workflow.step == system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING:
        return _AUTONOMOUS_GOAL_CANDIDATE_RETRY_KIND
    if workflow.step == system_agents.STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING:
        return _AUTONOMOUS_GOAL_JUDGE_RETRY_KIND
    return ""

def _dismiss_replaced_autonomous_goal_proposal(
    previous: ProposedSession | None, *, replacement: ProposedSession
) -> tuple[str, ...]:
    if previous is None or previous.pk == replacement.pk:
        return ()
    cleanup_cwd = previous.candidate_session.cwd if previous.candidate_session else ""
    if (
        previous.outcome_status != ProposedSession.OUTCOME_UNSET
        and not _autonomous_goal_proposal_hidden_until_complete(previous)
    ):
        return ()
    outcome_metadata = {
        **_proposal_outcome_metadata(previous, {}),
        "stacked_diff_hidden_until_complete": False,
        "stacked_diff_replaced_by": replacement.pk,
    }
    applied = ProposedSession.objects.filter(
        pk=previous.pk,
        outcome_status=previous.outcome_status,
    ).update(
        outcome_status=ProposedSession.OUTCOME_DISMISSED,
        outcome_notes=f"Replaced by stacked diff proposal #{replacement.pk}.",
        outcome_metadata=outcome_metadata,
        updated_at=timezone.now(),
    )
    return (cleanup_cwd,) if applied and cleanup_cwd else ()


def _fallback_from_failed_autonomous_goal_history_summary(
    run: SystemAgentRun,
    workflow: SystemWorkflow,
    autonomous_goal: AutonomousGoal,
    *,
    error: str,
    raw_output: str,
) -> _AutonomousGoalPostCommitAction | None:
    run.status = SystemAgentRun.STATUS_FAILED
    run.error = error
    run.raw_output = raw_output
    run.save(update_fields=["status", "error", "raw_output", "updated_at"])
    workflow.state = _state_without_proposal_history_summary(workflow.state)
    workflow.state["proposal_history_summary_error"] = truncate_for_prompt(error, 800)
    if _autonomous_goal_proposal_budget_exhausted(workflow):
        return _stop_autonomous_goal_after_history_summary_budget(
            workflow, autonomous_goal
        )
    system_agents._advance_workflow_step(
        workflow, system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING
    )
    return _AutonomousGoalPostCommitAction(_AUTONOMOUS_GOAL_RETRY_CANDIDATE_ACTION)


def _state_without_proposal_history_summary(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    next_state = dict(state)
    next_state.pop(_AUTONOMOUS_GOAL_PROPOSAL_HISTORY_SUMMARY_STATE_KEY, None)
    next_state.pop("proposal_history_summary_error", None)
    return next_state

def _publish_current_stack_proposal(
    proposal: ProposedSession,
    *,
    workflow: SystemWorkflow | None = None,
    continuation_stopped_reason: str = "",
    continuation_stopped_error: str = "",
) -> bool:
    budget_metadata = (
        _autonomous_goal_proposal_budget_metadata(workflow)
        if workflow is not None
        else {}
    )
    stop_metadata = _autonomous_goal_stack_continuation_stop_metadata(
        reason=continuation_stopped_reason,
        error=continuation_stopped_error,
    )
    if proposal.outcome_status == ProposedSession.OUTCOME_UNSET:
        if not budget_metadata and not stop_metadata:
            return ProposedSession.objects.filter(
                pk=proposal.pk,
                outcome_status=ProposedSession.OUTCOME_UNSET,
            ).exists()
        outcome_metadata = {
            **_proposal_outcome_metadata(proposal, {}),
            **budget_metadata,
            **stop_metadata,
        }
        return bool(
            ProposedSession.objects.filter(
                pk=proposal.pk,
                outcome_status=ProposedSession.OUTCOME_UNSET,
            ).update(
                outcome_metadata=outcome_metadata,
                updated_at=timezone.now(),
            )
        )
    if not _autonomous_goal_proposal_hidden_until_complete(proposal):
        return False
    outcome_metadata = {
        **_proposal_outcome_metadata(proposal, {}),
        "stacked_diff_hidden_until_complete": False,
        **budget_metadata,
        **stop_metadata,
    }
    return bool(
        ProposedSession.objects.filter(
            pk=proposal.pk,
            outcome_status=ProposedSession.OUTCOME_DISMISSED,
            outcome_metadata__stacked_diff_hidden_until_complete=True,
        ).update(
            outcome_status=ProposedSession.OUTCOME_UNSET,
            outcome_notes="",
            outcome_metadata=outcome_metadata,
            updated_at=timezone.now(),
        )
    )

def _autonomous_goal_stack_continuation_stop_metadata(
    *, reason: str, error: str = ""
) -> dict[str, object]:
    if not reason:
        return {}
    metadata: dict[str, object] = {
        _AUTONOMOUS_GOAL_STACKED_CONTINUATION_STOP_REASON_METADATA_KEY: reason
    }
    if error:
        metadata[_AUTONOMOUS_GOAL_STACKED_CONTINUATION_STOP_ERROR_METADATA_KEY] = error
    return metadata

def _complete_autonomous_goal_with_current_stack_proposal(
    workflow: SystemWorkflow, *, error: str
) -> bool:
    proposal = _autonomous_goal_current_stack_proposal(workflow)
    if proposal is None or not _publish_current_stack_proposal(
        proposal,
        workflow=workflow,
        continuation_stopped_reason="stacked_diff_continuation_failed",
        continuation_stopped_error=error,
    ):
        return False
    workflow.state = {
        **workflow.state,
        "stacked_diff_stopped_reason": "stacked_diff_continuation_failed",
        "stacked_diff_continuation_error": error,
    }
    system_agents._complete_workflow(workflow, system_agents.STEP_AUTONOMOUS_GOAL_PROPOSED)
    return True

def _autonomous_goal_proposal_hidden_until_complete(
    proposal: ProposedSession,
) -> bool:
    return (
        proposal.outcome_status == ProposedSession.OUTCOME_DISMISSED
        and _proposal_outcome_metadata(proposal, {}).get(
            "stacked_diff_hidden_until_complete"
        )
        is True
    )

def _autonomous_goal_should_continue_stack(
    workflow: SystemWorkflow, autonomous_goal: AutonomousGoal
) -> bool:
    if not _autonomous_goal_candidate_allows_code_changes(workflow):
        return False
    budget = _autonomous_goal_workflow_proposal_budget(workflow)
    if budget > 0 and _autonomous_goal_proposal_budget_tokens_used(workflow) >= budget:
        return False
    return _autonomous_goal_stack_iteration(
        workflow
    ) < _autonomous_goal_workflow_stacked_diff_depth(workflow, autonomous_goal)

def _autonomous_goal_next_stack_candidate_state(
    workflow: SystemWorkflow, proposal: ProposedSession
) -> dict[str, Any]:
    fork_cwd = (
        proposal.candidate_session.cwd
        if proposal.candidate_session and proposal.candidate_session.cwd
        else _autonomous_goal_session_cwd(workflow)
    )
    state = dict(workflow.state)
    state[_AUTONOMOUS_GOAL_STACKED_ITERATION_STATE_KEY] = (
        _autonomous_goal_stack_iteration(workflow) + 1
    )
    state[_AUTONOMOUS_GOAL_STACKED_FORK_CWD_STATE_KEY] = fork_cwd
    state["proposal_id"] = proposal.pk
    for key in (
        "candidate",
        "candidate_session_id",
        "judge_session_id",
        "judgment",
        "history_files",
    ):
        state.pop(key, None)
    return state


def _autonomous_goal_stack_snapshot_message(workflow: SystemWorkflow) -> str:
    proposal = _autonomous_goal_current_stack_proposal(workflow)
    if proposal is None:
        return ""
    return " ".join(proposal.title.split())

def _cleanup_autonomous_goal_candidate_cwd(cwd: str) -> None:
    if not cwd:
        return
    try:
        cleanup_managed_worktree_path(cwd)
    except WorktreeCleanupError:
        system_agents.logger.exception("failed to clean up autonomous goal candidate worktree %s", cwd)

def _prepare_autonomous_goal_candidate_cwd(
    workflow: SystemWorkflow, autonomous_goal: AutonomousGoal
) -> tuple[str, ManagedWorktree | None]:
    fork_cwd = _state_string(workflow, _AUTONOMOUS_GOAL_STACKED_FORK_CWD_STATE_KEY)
    if fork_cwd:
        snapshot_message = _autonomous_goal_stack_snapshot_message(workflow)
        if snapshot_message:
            base_ref = snapshot_worktree_to_commit(fork_cwd, message=snapshot_message)
        else:
            base_ref = snapshot_worktree_to_commit(fork_cwd)
        managed_worktree = create_worktree_for_session(
            autonomous_goal.project.repo_path,
            base_ref=base_ref,
        )
        session_cwd = str(managed_worktree.path)
        workflow.state = {
            **workflow.state,
            _AUTONOMOUS_GOAL_SESSION_CWD_STATE_KEY: session_cwd,
            _AUTONOMOUS_GOAL_STACKED_FORK_CWD_STATE_KEY: "",
        }
        try:
            workflow.save(update_fields=["state", "updated_at"])
        except Exception:
            _cleanup_new_autonomous_goal_worktree(managed_worktree)
            raise
        return session_cwd, managed_worktree

    session_cwd = _autonomous_goal_session_cwd(workflow)
    if session_cwd != workflow.cwd:
        return session_cwd, None
    if not _state_bool(workflow, _AUTONOMOUS_GOAL_USE_WORKTREES_STATE_KEY):
        return workflow.cwd, None

    auto_merge_ref = _autonomous_goal_auto_merge_worktree_base_ref(
        workflow, autonomous_goal
    )
    if auto_merge_ref:
        managed_worktree = create_worktree_for_session(
            autonomous_goal.project.repo_path,
            base_ref=auto_merge_ref,
            disable_hooks=True,
        )
    else:
        base_ref = _autonomous_goal_default_worktree_base_ref(
            workflow, autonomous_goal
        )
        if not base_ref:
            raise WorktreeCreationError("project default branch is unavailable")
        managed_worktree = create_worktree_for_session(
            autonomous_goal.project.repo_path,
            base_ref=base_ref,
        )
    session_cwd = str(managed_worktree.path)
    workflow.state = {
        **workflow.state,
        _AUTONOMOUS_GOAL_SESSION_CWD_STATE_KEY: session_cwd,
    }
    try:
        workflow.save(update_fields=["state", "updated_at"])
    except Exception:
        _cleanup_new_autonomous_goal_worktree(managed_worktree)
        raise
    return session_cwd, managed_worktree

def _autonomous_goal_default_worktree_base_ref(
    workflow: SystemWorkflow, autonomous_goal: AutonomousGoal
) -> str:
    start_sha = _autonomous_goal_recorded_base_sha(workflow)
    if start_sha:
        return start_sha
    return default_branch_commit_hash(autonomous_goal.project.repo_path) or ""

def _autonomous_goal_auto_merge_worktree_base_ref(
    workflow: SystemWorkflow, autonomous_goal: AutonomousGoal
) -> str:
    auto_merge_ref = _autonomous_goal_auto_merge_base_ref(autonomous_goal)
    if not auto_merge_ref:
        return ""
    return _autonomous_goal_recorded_base_sha(workflow) or auto_merge_ref

def _autonomous_goal_recorded_base_sha(workflow: SystemWorkflow) -> str:
    start_sha = _state_string(workflow, "default_branch_sha")
    if start_sha and start_sha != _AUTO_PROPOSAL_UNKNOWN_DEFAULT_BRANCH_SHA:
        return start_sha
    return ""

def _cleanup_new_autonomous_goal_worktree(worktree: ManagedWorktree | None) -> None:
    if worktree is None:
        return
    try:
        cleanup_worktree(worktree)
    except WorktreeCleanupError:
        system_agents.logger.exception(
            "failed to clean up autonomous goal candidate worktree %s",
            worktree.path,
        )

def _cleanup_autonomous_goal_workflow_worktree(workflow: SystemWorkflow) -> None:
    session_cwd = _autonomous_goal_session_cwd(workflow)
    if session_cwd == workflow.cwd:
        return
    try:
        cleanup_managed_worktree_path(session_cwd)
    except WorktreeCleanupError:
        system_agents.logger.exception(
            "failed to clean up autonomous goal workflow worktree %s",
            session_cwd,
        )

def _spawn_autonomous_goal_history_summary_run(
    workflow: SystemWorkflow, autonomous_goal: AutonomousGoal
) -> SystemAgentRun:
    session_cwd = _autonomous_goal_session_cwd(workflow)
    prompt, history_files = _autonomous_goal_history_summary_prompt(
        workflow, autonomous_goal
    )
    if history_files:
        workflow.state = {**workflow.state, "proposal_history_files": history_files}
        workflow.save(update_fields=["state", "updated_at"])
    instance: CodexInstance | None = None
    run: SystemAgentRun | None = None
    try:
        instance = codex_pool.spawn_new_session(
            cwd=session_cwd,
            prompt=prompt,
            approval_mode=system_agents.SYSTEM_AGENT_APPROVAL_MODE,
            sandbox_policy="readOnly",
            web_search_mode=system_agents._workflow_web_search_mode(workflow),
            thread_source=ThreadSource.subagent,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_HISTORY_SUMMARY_AGENT_KIND,
            display_author=system_agents.AUTONOMOUS_GOAL_HISTORY_SUMMARY_DISPLAY_AUTHOR,
            output_schema=_AUTONOMOUS_GOAL_HISTORY_SUMMARY_OUTPUT_SCHEMA,
            model=_autonomous_goal_history_summary_model(),
            reasoning_effort="low",
        )
        run = _get_or_create_autonomous_goal_history_summary_run(
            instance=instance,
            workflow=workflow,
            autonomous_goal=autonomous_goal,
            session_cwd=session_cwd,
            history_files=history_files,
        )
        metadata = session_index.upsert_local_session(
            thread_id=instance.thread_id,
            cwd=session_cwd,
            project=autonomous_goal.project,
            preview=prompt,
            auto_pr_enabled=False,
            auto_qa_enabled=False,
            auto_merge_to_local_branch=False,
            auto_merge_branch="",
            codex_path=codex_pool.thread_path_for_instance(instance),
            is_hidden_system_session=True,
        )
        workflow.state = {
            **workflow.state,
            "proposal_history_summary_session_id": metadata.pk,
        }
        workflow.save(update_fields=["state", "updated_at"])
    except Exception as exc:
        if instance is not None:
            cancelled = _cancel_partially_spawned_autonomous_goal_history_summary(
                instance=instance,
                run=run,
                error=f"failed to start autonomous goal history summarizer: {exc!r}",
            )
            if not cancelled:
                if run is None:
                    run = _preserve_partially_spawned_autonomous_goal_history_summary_run(
                        instance=instance,
                        workflow=workflow,
                        autonomous_goal=autonomous_goal,
                        session_cwd=session_cwd,
                        history_files=history_files,
                    )
                    if run is None:
                        raise _AutonomousGoalHistorySummaryUnpreservedError(
                            "failed to start autonomous goal history "
                            "summarizer and could not preserve a run for "
                            "the live summarizer"
                        ) from exc
                    assert run is not None
                return run
        raise
    assert run is not None
    return run


def _get_or_create_autonomous_goal_history_summary_run(
    *,
    instance: CodexInstance,
    workflow: SystemWorkflow,
    autonomous_goal: AutonomousGoal,
    session_cwd: str,
    history_files: list[str],
) -> SystemAgentRun:
    run, _created = SystemAgentRun.objects.get_or_create(
        instance=instance,
        defaults={
            "workflow": workflow,
            "agent_kind": system_agents.AUTONOMOUS_GOAL_HISTORY_SUMMARY_AGENT_KIND,
            "thread_id": instance.thread_id,
            "status": SystemAgentRun.STATUS_RUNNING,
            "input": {
                "cwd": session_cwd,
                "autonomous_goal_id": autonomous_goal.pk,
                "history_files": history_files,
            },
        },
    )
    return run


def _preserve_partially_spawned_autonomous_goal_history_summary_run(
    *,
    instance: CodexInstance,
    workflow: SystemWorkflow,
    autonomous_goal: AutonomousGoal,
    session_cwd: str,
    history_files: list[str],
) -> SystemAgentRun | None:
    try:
        return _get_or_create_autonomous_goal_history_summary_run(
            instance=instance,
            workflow=workflow,
            autonomous_goal=autonomous_goal,
            session_cwd=session_cwd,
            history_files=history_files,
        )
    except Exception:
        system_agents.logger.exception(
            "failed to preserve autonomous goal history summarizer run"
        )
        return None


def _cancel_partially_spawned_autonomous_goal_history_summary(
    *,
    instance: CodexInstance,
    run: SystemAgentRun | None,
    error: str,
) -> bool:
    interrupted = codex_pool.interrupt_instance(
        instance.pk, expected_thread_id=instance.thread_id
    )
    if interrupted is None:
        return False
    if run is not None:
        run.status = SystemAgentRun.STATUS_FAILED
        run.error = error
        run.save(update_fields=["status", "error", "updated_at"])
        return True
    return _detach_partially_spawned_autonomous_goal_history_summary(instance)


def _detach_partially_spawned_autonomous_goal_history_summary(
    instance: CodexInstance,
) -> bool:
    CodexInstance.objects.filter(pk=instance.pk).update(
        workflow_id=None,
        agent_kind="",
    )
    return True


def _autonomous_goal_history_summary_model() -> str | None:
    from django.conf import settings

    value = getattr(settings, "AUTONOMOUS_GOAL_HISTORY_SUMMARY_MODEL", "")
    return value.strip() or None if isinstance(value, str) else None


def _autonomous_goal_history_summary_prompt(
    workflow: SystemWorkflow, autonomous_goal: AutonomousGoal
) -> tuple[str, list[str]]:
    history_sections = _autonomous_goal_history_sections(autonomous_goal)
    inline_history, overflow_history = _split_autonomous_goal_history(history_sections)
    history_files = _write_autonomous_goal_history_files(workflow, overflow_history)
    history_file_text = (
        "\n".join(f"- {path}" for path in history_files) if history_files else "(none)"
    )
    run_references = _autonomous_goal_recent_proposal_run_references(autonomous_goal)
    prompt = (
        "You are Hitch's autonomous-goal proposal history summarizer.\n\n"
        "Summarize resolved proposal history so the next autonomous-goal "
        "candidate can make better planning decisions. Preserve concrete "
        "lessons: what has already worked, what was superseded by later stacked "
        "proposals, what was rejected or should be avoided, important files, "
        "benchmark deltas, and promising next directions. Treat proposals "
        "dismissed with notes like 'Replaced by stacked diff proposal #...' as "
        "superseded lineage, not failed ideas.\n\n"
        f"Autonomous goal title: {autonomous_goal.title}\n\n"
        "Autonomous goal objective:\n"
        f"{autonomous_goal.goal}\n\n"
        "Recent proposal run references that must remain available to the next "
        "candidate:\n"
        f"{run_references}\n\n"
        "Resolved proposal history included inline:\n"
        f"{inline_history or '(none)'}\n\n"
        "Additional history files:\n"
        f"{history_file_text}\n\n"
        "Return only JSON matching this shape: "
        '{"brief": string, "recent_stack": [string], '
        '"accepted_lessons": [string], "avoid_or_reconsider": [string], '
        '"promising_next_directions": [string], "important_files": [string]}. '
        "Keep the brief compact but specific. Include proposal IDs in list items "
        "when useful. Do not invent session IDs, file paths, benchmark numbers, "
        "or outcomes."
    )
    return prompt, history_files


def _spawn_autonomous_goal_candidate_run(
    workflow: SystemWorkflow, autonomous_goal: AutonomousGoal
) -> SystemAgentRun:
    session_cwd, managed_worktree = _prepare_autonomous_goal_candidate_cwd(
        workflow, autonomous_goal
    )
    try:
        (
            prompt,
            memory_context,
            proposal_history_context,
        ) = _autonomous_goal_candidate_prompt(workflow, autonomous_goal)
        instance = codex_pool.spawn_new_session(
            cwd=session_cwd,
            prompt=prompt,
            approval_mode=system_agents.SYSTEM_AGENT_APPROVAL_MODE,
            sandbox_policy=(
                system_agents.AUTONOMOUS_GOAL_IMPLEMENTATION_SANDBOX_POLICY
                if _autonomous_goal_candidate_allows_code_changes(workflow)
                # A no-code (proposal-only) candidate runs in the user's real repo
                # cwd with no worktree, so it must not write. An empty sandbox
                # defaults to workspace-write at the app-server, which would let a
                # misbehaving or prompt-injected run mutate the real repo despite
                # the "do not make code changes" prompt -- pin it read-only.
                else "readOnly"
            ),
            web_search_mode=system_agents._workflow_web_search_mode(workflow),
            thread_source=ThreadSource.subagent,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            display_author=system_agents.AUTONOMOUS_GOAL_DISPLAY_AUTHOR,
            output_schema=_AUTONOMOUS_GOAL_CANDIDATE_OUTPUT_SCHEMA,
        )
    except Exception:
        _cleanup_new_autonomous_goal_worktree(managed_worktree)
        raise
    metadata = session_index.upsert_local_session(
        thread_id=instance.thread_id,
        cwd=session_cwd,
        project=autonomous_goal.project,
        preview=prompt,
        auto_pr_enabled=False,
        auto_qa_enabled=False,
        auto_merge_to_local_branch=False,
        auto_merge_branch="",
        codex_path=codex_pool.thread_path_for_instance(instance),
        is_hidden_system_session=True,
    )
    workflow.state = {**workflow.state, "candidate_session_id": metadata.pk}
    workflow.save(update_fields=["state", "updated_at"])
    run, _created = SystemAgentRun.objects.get_or_create(
        instance=instance,
        defaults={
            "workflow": workflow,
            "agent_kind": system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            "thread_id": instance.thread_id,
            "status": SystemAgentRun.STATUS_RUNNING,
            "input": {
                "cwd": session_cwd,
                "autonomous_goal_id": autonomous_goal.pk,
                "memory_count": memory_context.count,
                "memory_compacted": memory_context.compacted,
                "proposal_history_count": proposal_history_context.count,
                "proposal_history_compacted": proposal_history_context.compacted,
            },
        },
    )
    return run

def _spawn_autonomous_goal_candidate_retry_run(
    workflow: SystemWorkflow, autonomous_goal: AutonomousGoal
) -> SystemAgentRun:
    candidate_session = _session_metadata_from_state(workflow, "candidate_session_id")
    if candidate_session is None:
        raise RuntimeError("candidate session is unavailable")
    session_cwd = candidate_session.cwd or _autonomous_goal_session_cwd(workflow)
    prompt = _autonomous_goal_candidate_retry_prompt(workflow, autonomous_goal)
    instance = codex_pool.spawn_turn(
        thread_id=candidate_session.thread_id,
        cwd=session_cwd,
        prompt=prompt,
        approval_mode=system_agents.SYSTEM_AGENT_APPROVAL_MODE,
        sandbox_policy=(
            system_agents.AUTONOMOUS_GOAL_IMPLEMENTATION_SANDBOX_POLICY
            if _autonomous_goal_candidate_allows_code_changes(workflow)
            # No-code candidate retry: same as the initial spawn, the run is in the
            # real repo cwd and must not write -- pin it read-only rather than
            # letting the empty sandbox default to workspace-write.
            else "readOnly"
        ),
        web_search_mode=system_agents._workflow_web_search_mode(workflow),
        purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
        workflow_id=workflow.pk,
        agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        display_author=system_agents.AUTONOMOUS_GOAL_DISPLAY_AUTHOR,
        output_schema=_AUTONOMOUS_GOAL_CANDIDATE_OUTPUT_SCHEMA,
    )
    run, _created = SystemAgentRun.objects.get_or_create(
        instance=instance,
        defaults={
            "workflow": workflow,
            "agent_kind": system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            "thread_id": instance.thread_id,
            "status": SystemAgentRun.STATUS_RUNNING,
            "input": {
                "cwd": session_cwd,
                "autonomous_goal_id": autonomous_goal.pk,
                "proposal_budget": _autonomous_goal_workflow_proposal_budget(
                    workflow
                ),
                "proposal_budget_tokens_used": (
                    _autonomous_goal_proposal_budget_tokens_used(workflow)
                ),
                "retry_attempt": _autonomous_goal_failed_attempts(workflow),
            },
        },
    )
    return run

def _spawn_autonomous_goal_judge_or_block(
    workflow: SystemWorkflow,
    autonomous_goal: AutonomousGoal,
    candidate: dict[str, Any],
) -> None:
    workflow, locked_goal, should_spawn = _claim_active_autonomous_goal_workflow(
        workflow_id=workflow.pk,
        autonomous_goal_id=autonomous_goal.pk,
    )
    if not should_spawn or locked_goal is None:
        return
    try:
        run = _spawn_autonomous_goal_judge_run(workflow, locked_goal, candidate)
    except Exception as exc:
        _block_autonomous_goal_spawn_failure_if_active(
            workflow_id=workflow.pk,
            autonomous_goal_id=locked_goal.pk,
            error=f"failed to start autonomous goal judge: {exc!r}",
        )
        return
    _interrupt_spawned_autonomous_goal_run_if_inactive(run)

def _spawn_autonomous_goal_judge_run(
    workflow: SystemWorkflow, autonomous_goal: AutonomousGoal, candidate: dict[str, Any]
) -> SystemAgentRun:
    session_cwd = _autonomous_goal_session_cwd(workflow)
    prompt, history_files = _autonomous_goal_judge_prompt(
        workflow, autonomous_goal, candidate
    )
    if history_files:
        workflow.state = {**workflow.state, "history_files": history_files}
        workflow.save(update_fields=["state", "updated_at"])
    instance = codex_pool.spawn_new_session(
        cwd=session_cwd,
        prompt=prompt,
        approval_mode=system_agents.SYSTEM_AGENT_APPROVAL_MODE,
        # The judge only evaluates the candidate, so it never writes -- pin it
        # read-only. This matters for no-code goals where ``session_cwd`` is the
        # user's real repo: an empty sandbox defaults to workspace-write at the
        # app-server, which would let the evaluation step mutate the repo the
        # no-code candidate was deliberately kept out of.
        sandbox_policy="readOnly",
        web_search_mode=system_agents._workflow_web_search_mode(workflow),
        thread_source=ThreadSource.subagent,
        purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
        workflow_id=workflow.pk,
        agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
        display_author=system_agents.AUTONOMOUS_GOAL_JUDGE_DISPLAY_AUTHOR,
        output_schema=_AUTONOMOUS_GOAL_JUDGE_OUTPUT_SCHEMA,
    )
    metadata = session_index.upsert_local_session(
        thread_id=instance.thread_id,
        cwd=session_cwd,
        project=autonomous_goal.project,
        preview=prompt,
        auto_pr_enabled=False,
        auto_qa_enabled=False,
        auto_merge_to_local_branch=False,
        auto_merge_branch="",
        codex_path=codex_pool.thread_path_for_instance(instance),
        is_hidden_system_session=True,
    )
    workflow.state = {**workflow.state, "judge_session_id": metadata.pk}
    workflow.save(update_fields=["state", "updated_at"])
    run, _created = SystemAgentRun.objects.get_or_create(
        instance=instance,
        defaults={
            "workflow": workflow,
            "agent_kind": system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
            "thread_id": instance.thread_id,
            "status": SystemAgentRun.STATUS_RUNNING,
            "input": {
                "cwd": session_cwd,
                "autonomous_goal_id": autonomous_goal.pk,
                "candidate": candidate,
                "history_files": history_files,
            },
        },
    )
    return run

def _autonomous_goal_auto_merge_branch_for_implementation(
    autonomous_goal: AutonomousGoal,
) -> str:
    if autonomous_goal.autonomy == AutonomousGoal.AUTONOMY_DRAFT_PR:
        return ""
    if not autonomous_goal.auto_qa_enabled:
        return ""
    if not autonomous_goal.auto_merge_to_local_branch:
        return ""
    return autonomous_goal.auto_merge_branch.strip()

def _record_autonomous_goal_no_proposal(
    autonomous_goal: AutonomousGoal, workflow: SystemWorkflow
) -> None:
    if workflow.state.get("auto_proposal") is not True:
        return
    sha = _state_string(workflow, "default_branch_sha")
    if not sha:
        return
    filters: dict[str, Any] = {"pk": autonomous_goal.pk}
    snapshot = _state_string(workflow, "autonomous_goal_updated_at")
    if snapshot:
        snapshot_datetime = parse_datetime(snapshot)
        if snapshot_datetime is None:
            return
        filters["updated_at"] = snapshot_datetime
    AutonomousGoal.objects.filter(**filters).update(
        auto_proposal_last_no_proposal_sha=sha,
        updated_at=timezone.now(),
    )

def _record_autonomous_goal_proposal_created(autonomous_goal: AutonomousGoal) -> None:
    if not autonomous_goal.auto_proposal_last_no_proposal_sha:
        return
    AutonomousGoal.objects.filter(pk=autonomous_goal.pk).update(
        auto_proposal_last_no_proposal_sha="",
        updated_at=timezone.now(),
    )

def _interrupt_autonomous_goal_runs(
    runs: list[SystemAgentRun],
) -> tuple[list[SystemAgentRun], bool]:
    interrupted_runs: list[SystemAgentRun] = []
    terminal_instance_returned = False
    for run in runs:
        interrupted = codex_pool.interrupt_instance(
            run.instance_id, expected_thread_id=run.thread_id
        )
        if interrupted is None:
            continue
        interrupted_runs.append(run)
        if interrupted.status in (
            CodexInstance.STATUS_COMPLETED,
            CodexInstance.STATUS_FAILED,
        ):
            terminal_instance_returned = True
    return interrupted_runs, terminal_instance_returned

def _fail_autonomous_goal_run_and_block_workflow(
    run: SystemAgentRun,
    autonomous_goal: AutonomousGoal,
    error: str,
    raw_output: str = "",
) -> _AutonomousGoalPostCommitAction | None:
    run.status = SystemAgentRun.STATUS_FAILED
    run.error = error
    run.raw_output = raw_output
    run.save(update_fields=["status", "error", "raw_output", "updated_at"])
    workflow = run.workflow
    if _complete_autonomous_goal_with_current_stack_proposal(workflow, error=error):
        cleanup_cwd = system_agents._candidate_session_cwd_from_state(
            workflow, "candidate_session_id"
        )
        return _AutonomousGoalPostCommitAction(
            cleanup_candidate_cwds=((cleanup_cwd,) if cleanup_cwd else ())
        )
    _block_autonomous_goal_workflow(run.workflow, autonomous_goal, error)
    return None

def _block_autonomous_goal_workflow(
    workflow: SystemWorkflow, autonomous_goal: AutonomousGoal, error: str
) -> None:
    # The finish handler may have recorded budget tokens on this locked instance.
    # Persist them before _block_workflow re-reads the row.
    workflow.save(update_fields=["state", "updated_at"])
    _create_autonomous_goal_failure_notice(workflow, autonomous_goal, error)
    system_agents._block_workflow(workflow, error, surface_to_thread=False)

def _create_autonomous_goal_failure_notice(
    workflow: SystemWorkflow, autonomous_goal: AutonomousGoal, error: str
) -> None:
    if ProposedSession.objects.filter(
        source_workflow=workflow,
        outcome_status=ProposedSession.OUTCOME_UNSET,
    ).exists():
        return
    ProposedSession.objects.create(
        project=autonomous_goal.project,
        autonomous_goal=autonomous_goal,
        source_workflow=workflow,
        title=f"Autonomous goal failed: {autonomous_goal.title}"[
            :_AUTONOMOUS_GOAL_TITLE_MAX_LEN
        ],
        inbox_kind=ProposedSession.INBOX_KIND_NOTICE,
        summary=f"Hitch could not finish this autonomous goal run: {error}",
        candidate_session=_session_metadata_from_state(workflow, "candidate_session_id"),
        judge_session=_session_metadata_from_state(workflow, "judge_session_id"),
        outcome_metadata={
            "autonomous_goal_autonomy": autonomous_goal.autonomy,
            "automation_status": "failed",
            "automation_error": error,
            **_autonomous_goal_proposal_budget_metadata(workflow),
        },
    )

def _create_autonomous_goal_skipped_notice(
    workflow: SystemWorkflow,
    autonomous_goal: AutonomousGoal,
    *,
    title: str,
    summary: str,
    metadata: dict[str, object] | None = None,
) -> None:
    if ProposedSession.objects.filter(
        source_workflow=workflow,
        inbox_kind=ProposedSession.INBOX_KIND_NOTICE,
        outcome_status=ProposedSession.OUTCOME_UNSET,
    ).exists():
        return
    ProposedSession.objects.create(
        project=autonomous_goal.project,
        autonomous_goal=autonomous_goal,
        source_workflow=workflow,
        title=title,
        inbox_kind=ProposedSession.INBOX_KIND_NOTICE,
        summary=summary,
        candidate_session=_session_metadata_from_state(workflow, "candidate_session_id"),
        judge_session=_session_metadata_from_state(workflow, "judge_session_id"),
        outcome_metadata={
            **(metadata or {}),
            **_autonomous_goal_proposal_budget_metadata(workflow),
        },
    )

def _autonomous_goal_main_thread_id(autonomous_goal_id: int) -> str:
    return f"autonomous-goal:{autonomous_goal_id}"
