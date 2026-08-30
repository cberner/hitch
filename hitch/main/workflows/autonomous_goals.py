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

import json
import logging
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast, override

from django.db import IntegrityError, OperationalError, transaction
from django.utils import timezone
from django.utils.dateparse import parse_datetime
from openai_codex import Codex, CodexError
from openai_codex.generated.v2_all import GetAccountRateLimitsResponse, ThreadSource

from hitch.main.goals.autonomous_goal_prompts import (
    _AUTONOMOUS_GOAL_FAILED_ATTEMPTS_STATE_KEY,
    _AUTONOMOUS_GOAL_LAST_FAILURE_STATE_KEY,
    _AUTONOMOUS_GOAL_NO_PROGRESS_RETRIES_STATE_KEY,
    _AUTONOMOUS_GOAL_PROPOSAL_BUDGET_STATE_KEY,
    _AUTONOMOUS_GOAL_PROPOSAL_BUDGET_USED_STATE_KEY,
    _AUTONOMOUS_GOAL_SESSION_CWD_STATE_KEY,
    _AUTONOMOUS_GOAL_STACKED_DEPTH_STATE_KEY,
    _AUTONOMOUS_GOAL_STACKED_ITERATION_STATE_KEY,
    _AUTONOMOUS_GOAL_TITLE_MAX_LEN,
    _autonomous_goal_candidate_allows_code_changes,
    _autonomous_goal_failed_attempts,
    _autonomous_goal_no_progress_budget_retries,
    _autonomous_goal_proposal_budget_metadata,
    _autonomous_goal_proposal_budget_tokens_used,
    _autonomous_goal_proposal_summary,
    _autonomous_goal_proposed_session_prompt,
    _autonomous_goal_session_cwd,
    _autonomous_goal_stack_iteration,
    _autonomous_goal_workflow_proposal_budget,
    _autonomous_goal_workflow_stacked_diff_depth,
    _string_list,
)
from hitch.main.goals.autonomous_goal_proposal_stack import (
    _AUTONOMOUS_GOAL_STACKED_CONTINUATION_STOP_REASON_METADATA_KEY,
    AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_METADATA_KEY,
    AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_REF_METADATA_KEY,
    AUTONOMOUS_GOAL_TOOL_PROTOCOL_METADATA_KEY,
    _autonomous_goal_accepted_session_blocks_start,
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
    RefreshThrottle,
    SessionMetadata,
    SystemAgentRun,
    SystemWorkflow,
)
from hitch.main.repos import default_branch_commit_hash
from hitch.main.runtime import app_server_pool, codex_events, codex_pool, db, rollout
from hitch.main.runtime.rollout_state import _rollout_path_from_value
from hitch.main.runtime.sdk_values import truncate_for_prompt
from hitch.main.sequences import unique_nonempty
from hitch.main.sessions import session_index
from hitch.main.workflows import engine, system_agents
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
    release_snapshot_commit_ref,
    snapshot_worktree_to_commit,
)

logger = logging.getLogger(__name__)

_AUTO_PROPOSAL_UNKNOWN_DEFAULT_BRANCH_SHA = "__unknown__"

_AUTO_PROPOSAL_QUOTA_CACHE_TTL = timedelta(minutes=5)

_quota_cache_lock = threading.Lock()

AutoProposalQuotaStatus = Literal["available", "low", "unavailable"]

_quota_cache_status: AutoProposalQuotaStatus = "available"

_quota_cache_checked_at: datetime | None = None

_AUTONOMOUS_GOAL_USE_WORKTREES_STATE_KEY = "use_worktrees"

_AUTONOMOUS_GOAL_STACKED_FORK_CWD_STATE_KEY = "stacked_diff_fork_from_cwd"
_AUTONOMOUS_GOAL_STACKED_BASE_REF_STATE_KEY = "stacked_diff_base_ref"

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

_AUTONOMOUS_GOAL_SPAWN_NEXT_CANDIDATE_ACTION = "spawn_next_candidate"

_AUTONOMOUS_GOAL_RETRY_CANDIDATE_ACTION = "retry_candidate"

_AUTONOMOUS_GOAL_RETRY_CANDIDATE_CONTINUATION_ACTION = "retry_candidate_continuation"

_AUTONOMOUS_GOAL_PROTOCOL_RECOVERY_ACTION = "protocol_recovery"
_AUTONOMOUS_GOAL_JUDGE_PROTOCOL_RECOVERY_ACTION = "judge_protocol_recovery"

_AUTONOMOUS_GOAL_TOOL_PROTOCOL_STATE_KEY = "tool_protocol"
_AUTONOMOUS_GOAL_JUDGMENT_ATTEMPTS_STATE_KEY = "judgment_attempts"
_AUTONOMOUS_GOAL_JUDGMENT_REQUEST_STATE_KEY = "judgment_request_id"
_AUTONOMOUS_GOAL_JUDGMENT_VERDICT_STATE_KEY = "judgment_verdict_id"
_AUTONOMOUS_GOAL_JUDGMENT_RESULT_STATE_KEY = "judgment_result_id"
_AUTONOMOUS_GOAL_CANDIDATE_TERMINAL_STATE_KEY = "candidate_terminal"
_AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_STATE_KEY = "approved_snapshot_sha"
_AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_REF_STATE_KEY = "approved_snapshot_ref"
_AUTONOMOUS_GOAL_JUDGE_SNAPSHOT_CWD_STATE_KEY = "judge_snapshot_cwd"
_AUTONOMOUS_GOAL_PROTOCOL_RECOVERIES_STATE_KEY = "protocol_recoveries"
_AUTONOMOUS_GOAL_JUDGE_PROTOCOL_RECOVERIES_STATE_KEY = (
    "judge_protocol_recoveries"
)
_AUTONOMOUS_GOAL_MAX_JUDGMENTS = 2
_AUTONOMOUS_GOAL_MAX_PROTOCOL_RECOVERIES = 3
_AUTONOMOUS_GOAL_MAX_JUDGE_PROTOCOL_RECOVERIES = 1

_AUTO_PROPOSAL_QUOTA_THRESHOLD_FRACTION = 0.5

_AUTO_PROPOSAL_QUEUE_LOCK_KEY = "autonomous_goal:auto_proposal_queue"


def candidate_goal_data(context: Any) -> dict[str, object]:
    workflow, autonomous_goal = _candidate_tool_scope(context)
    judgment = workflow.state.get("judgment")
    feedback = ""
    if isinstance(judgment, dict) and judgment.get("verdict") == "deny":
        raw_feedback = judgment.get("feedback")
        feedback = raw_feedback.strip() if isinstance(raw_feedback, str) else ""
    attempts = _state_int(workflow, _AUTONOMOUS_GOAL_JUDGMENT_ATTEMPTS_STATE_KEY)
    return {
        "title": autonomous_goal.title,
        "goal": autonomous_goal.goal,
        "ambition": autonomous_goal.ambition,
        "autonomy": autonomous_goal.autonomy,
        "confidence_threshold": autonomous_goal.confidence_threshold,
        "stack_iteration": _autonomous_goal_stack_iteration(workflow),
        "stack_depth": _autonomous_goal_workflow_stacked_diff_depth(
            workflow, autonomous_goal
        ),
        "proposal_budget": _autonomous_goal_workflow_proposal_budget(workflow),
        "proposal_budget_tokens_used": (
            _autonomous_goal_proposal_budget_tokens_used(workflow)
        ),
        "judgment_attempts_used": attempts,
        "judgment_attempts_remaining": max(
            _AUTONOMOUS_GOAL_MAX_JUDGMENTS - attempts, 0
        ),
        "last_judge_feedback": feedback,
    }


def candidate_goal_sessions(context: Any) -> list[dict[str, object]]:
    workflow, autonomous_goal = _candidate_tool_scope(context)
    proposals = list(
        ProposedSession.objects.filter(autonomous_goal=autonomous_goal)
        .select_related("candidate_session", "accepted_session")
        .order_by("created_at", "id")
    )
    session_rows: dict[int, dict[str, object]] = {}

    for proposal in proposals:
        if proposal.candidate_session is not None:
            _add_goal_session_row(
                session_rows,
                proposal.candidate_session,
                kind="candidate",
                outcome=proposal.outcome_status or "pending",
                proposal_id=proposal.pk,
            )
        if proposal.accepted_session is not None:
            _add_goal_session_row(
                session_rows,
                proposal.accepted_session,
                kind="accepted_work",
                outcome="accepted",
                proposal_id=proposal.pk,
            )

    workflow_ids = list(
        SystemWorkflow.objects.filter(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            state__autonomous_goal_id=autonomous_goal.pk,
        ).values_list("id", flat=True)
    )
    candidate_thread_ids = list(
        SystemAgentRun.objects.filter(
            workflow_id__in=workflow_ids,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        .exclude(thread_id=context.thread_id)
        .values_list("thread_id", flat=True)
        .distinct()
    )
    for metadata in SessionMetadata.objects.filter(
        thread_id__in=candidate_thread_ids
    ).order_by("created_at", "id"):
        _add_goal_session_row(
            session_rows,
            metadata,
            kind="candidate",
            outcome="completed",
            proposal_id=None,
        )

    current_candidate_id = _state_int(workflow, "candidate_session_id")
    rows = [
        row for session_id, row in session_rows.items() if session_id != current_candidate_id
    ]
    rows.sort(key=lambda row: (str(row["created_at"]), str(row["session_id"])))
    return rows


def _add_goal_session_row(
    rows: dict[int, dict[str, object]],
    metadata: SessionMetadata,
    *,
    kind: str,
    outcome: str,
    proposal_id: int | None,
) -> None:
    path = metadata.codex_path.strip()
    resolved_path = str(Path(path).expanduser()) if path else ""
    row: dict[str, object] = {
        "session_id": metadata.thread_id,
        "title": (
            metadata.codex_name
            or metadata.codex_display_title
            or (
                metadata.codex_preview.splitlines()[0][:200]
                if metadata.codex_preview
                else metadata.thread_id
            )
        ),
        "kind": kind,
        "outcome": outcome,
        "proposal_id": proposal_id,
        "created_at": metadata.created_at.isoformat(),
        "session_file": resolved_path,
        "session_file_available": bool(
            resolved_path and Path(resolved_path).is_file()
        ),
    }
    previous = rows.get(metadata.pk)
    if previous is None or kind == "accepted_work":
        rows[metadata.pk] = row


def candidate_request_judgment(
    arguments: dict[str, Any], context: Any
) -> dict[str, object]:
    candidate = _candidate_from_tool_arguments(arguments)
    workflow, autonomous_goal = _candidate_tool_scope(
        context, require_candidate_step=True
    )
    attempts = _state_int(workflow, _AUTONOMOUS_GOAL_JUDGMENT_ATTEMPTS_STATE_KEY)
    if attempts >= _AUTONOMOUS_GOAL_MAX_JUDGMENTS:
        raise ValueError("the candidate has already used both judgment attempts")
    if _state_string(workflow, _AUTONOMOUS_GOAL_CANDIDATE_TERMINAL_STATE_KEY):
        raise ValueError("the candidate has already finished")

    request_id = uuid.uuid4().hex
    snapshot_sha = ""
    snapshot_ref = ""
    try:
        if _autonomous_goal_candidate_allows_code_changes(workflow):
            snapshot_ref = (
                f"refs/hitch/autonomous-goals/{workflow.pk}/{request_id}"
            )
            snapshot_sha = snapshot_worktree_to_commit(
                context.cwd,
                message=f"Snapshot AG judgment attempt {attempts + 1}",
                retain_ref=snapshot_ref,
            )
        with transaction.atomic():
            workflow = SystemWorkflow.objects.select_for_update().get(pk=workflow.pk)
            _validate_candidate_tool_workflow(
                workflow, context, require_candidate_step=True
            )
            attempts = _state_int(
                workflow, _AUTONOMOUS_GOAL_JUDGMENT_ATTEMPTS_STATE_KEY
            )
            if attempts >= _AUTONOMOUS_GOAL_MAX_JUDGMENTS:
                raise ValueError(
                    "the candidate has already used both judgment attempts"
                )
            state = dict(workflow.state)
            state.update(
                {
                    "candidate": candidate,
                    _AUTONOMOUS_GOAL_JUDGMENT_ATTEMPTS_STATE_KEY: attempts + 1,
                    _AUTONOMOUS_GOAL_JUDGMENT_REQUEST_STATE_KEY: request_id,
                    _AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_STATE_KEY: snapshot_sha,
                    _AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_REF_STATE_KEY: snapshot_ref,
                }
            )
            state.pop(_AUTONOMOUS_GOAL_JUDGMENT_RESULT_STATE_KEY, None)
            state.pop(_AUTONOMOUS_GOAL_JUDGMENT_VERDICT_STATE_KEY, None)
            state.pop(_AUTONOMOUS_GOAL_CANDIDATE_TERMINAL_STATE_KEY, None)
            state.pop(_AUTONOMOUS_GOAL_JUDGE_PROTOCOL_RECOVERIES_STATE_KEY, None)
            workflow.state = state
            workflow.step = system_agents.STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING
            workflow.save(update_fields=["step", "state", "updated_at"])
    except Exception:
        _release_autonomous_goal_snapshot_ref(
            autonomous_goal.project.repo_path, snapshot_ref
        )
        raise

    try:
        judge_run = _spawn_autonomous_goal_judge_run(
            workflow, autonomous_goal, candidate
        )
        _interrupt_spawned_autonomous_goal_run_if_inactive(judge_run)
    except Exception as exc:
        _restore_failed_judge_spawn(
            workflow_id=workflow.pk,
            request_id=request_id,
            attempts_before=attempts,
        )
        _release_autonomous_goal_snapshot_ref(
            autonomous_goal.project.repo_path, snapshot_ref
        )
        raise ValueError(f"failed to start autonomous goal judge: {exc!r}") from exc

    while True:
        if context.cancel_requested():
            raise ValueError("candidate stopped while waiting for the judge")
        current = SystemWorkflow.objects.only("status", "step", "state").get(
            pk=workflow.pk
        )
        if _state_string(
            current, _AUTONOMOUS_GOAL_JUDGMENT_RESULT_STATE_KEY
        ) == request_id:
            judgment = current.state.get("judgment")
            if not isinstance(judgment, dict):
                raise ValueError("judge completed without a valid verdict")
            verdict = str(judgment.get("verdict") or "")
            return {
                "verdict": verdict,
                "confidence": str(judgment.get("confidence") or ""),
                "feedback": str(judgment.get("feedback") or ""),
                "judgment_attempts_used": _state_int(
                    current, _AUTONOMOUS_GOAL_JUDGMENT_ATTEMPTS_STATE_KEY
                ),
                "judgment_attempts_remaining": max(
                    _AUTONOMOUS_GOAL_MAX_JUDGMENTS
                    - _state_int(
                        current, _AUTONOMOUS_GOAL_JUDGMENT_ATTEMPTS_STATE_KEY
                    ),
                    0,
                ),
            }
        if not current.is_active:
            raise ValueError(
                _state_string(current, "error")
                or "autonomous goal ended while waiting for the judge"
            )
        time.sleep(0.25)


def _restore_failed_judge_spawn(
    *, workflow_id: int, request_id: str, attempts_before: int
) -> None:
    with transaction.atomic():
        workflow = SystemWorkflow.objects.select_for_update().get(pk=workflow_id)
        if (
            not workflow.is_active
            or workflow.step
            != system_agents.STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING
            or
            _state_string(workflow, _AUTONOMOUS_GOAL_JUDGMENT_REQUEST_STATE_KEY)
            != request_id
        ):
            return
        state = dict(workflow.state)
        state[_AUTONOMOUS_GOAL_JUDGMENT_ATTEMPTS_STATE_KEY] = attempts_before
        for key in (
            _AUTONOMOUS_GOAL_JUDGMENT_REQUEST_STATE_KEY,
            _AUTONOMOUS_GOAL_JUDGMENT_VERDICT_STATE_KEY,
            _AUTONOMOUS_GOAL_JUDGMENT_RESULT_STATE_KEY,
            _AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_STATE_KEY,
            _AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_REF_STATE_KEY,
            _AUTONOMOUS_GOAL_JUDGE_SNAPSHOT_CWD_STATE_KEY,
            "judge_session_id",
        ):
            state.pop(key, None)
        workflow.state = state
        workflow.step = system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING
        workflow.save(update_fields=["step", "state", "updated_at"])


def candidate_decline_proposal(
    arguments: dict[str, Any], context: Any
) -> dict[str, object]:
    reason = _required_tool_string(arguments, "reason")
    unexpected = set(arguments) - {"reason"}
    if unexpected:
        raise ValueError(f"unexpected no_proposal fields: {', '.join(sorted(unexpected))}")
    workflow, _autonomous_goal = _candidate_tool_scope(
        context, require_candidate_step=True
    )
    with transaction.atomic():
        workflow = SystemWorkflow.objects.select_for_update().get(pk=workflow.pk)
        _validate_candidate_tool_workflow(
            workflow, context, require_candidate_step=True
        )
        state = dict(workflow.state)
        if isinstance(
            state.get(_AUTONOMOUS_GOAL_CANDIDATE_TERMINAL_STATE_KEY), str
        ):
            raise ValueError("the candidate has already finished")
        state[_AUTONOMOUS_GOAL_CANDIDATE_TERMINAL_STATE_KEY] = "no_proposal"
        state["candidate"] = {
            "proposal": None,
            "message": reason,
        }
        workflow.state = state
        workflow.save(update_fields=["state", "updated_at"])
    return {"status": "no_proposal", "reason": reason}


def judge_record_verdict(
    arguments: dict[str, Any], context: Any, *, approved: bool
) -> dict[str, object]:
    confidence = _required_tool_string(arguments, "confidence")
    if confidence not in {value for value, _label in AutonomousGoal.CONFIDENCE_CHOICES}:
        raise ValueError("confidence must be medium, high, or very_high")
    feedback = _optional_tool_string(arguments, "feedback")
    unexpected = set(arguments) - {"confidence", "feedback"}
    if unexpected:
        raise ValueError(f"unexpected verdict fields: {', '.join(sorted(unexpected))}")
    workflow, autonomous_goal = _judge_tool_scope(context)
    if approved and not _confidence_meets_threshold(
        confidence, autonomous_goal.confidence_threshold
    ):
        raise ValueError(
            "approval confidence is below this goal's confidence threshold"
        )
    with transaction.atomic():
        workflow = SystemWorkflow.objects.select_for_update().get(pk=workflow.pk)
        _validate_judge_tool_workflow(workflow, context)
        request_id = _state_string(
            workflow, _AUTONOMOUS_GOAL_JUDGMENT_REQUEST_STATE_KEY
        )
        if not request_id:
            raise ValueError("no judgment request is active")
        if _state_string(
            workflow, _AUTONOMOUS_GOAL_JUDGMENT_VERDICT_STATE_KEY
        ):
            raise ValueError("this judge has already recorded a verdict")
        verdict = "approve" if approved else "deny"
        judgment = {
            "verdict": verdict,
            "confidence": confidence,
            "feedback": feedback,
        }
        state = {
            **workflow.state,
            "judgment": judgment,
            _AUTONOMOUS_GOAL_JUDGMENT_VERDICT_STATE_KEY: request_id,
        }
        if approved:
            state[_AUTONOMOUS_GOAL_CANDIDATE_TERMINAL_STATE_KEY] = "approved"
        workflow.state = state
        workflow.save(update_fields=["state", "updated_at"])
    return {"verdict": verdict, "confidence": confidence, "feedback": feedback}


def _candidate_tool_scope(
    context: Any, *, require_candidate_step: bool = False
) -> tuple[SystemWorkflow, AutonomousGoal]:
    workflow = _tool_workflow(context)
    _validate_candidate_tool_workflow(
        workflow, context, require_candidate_step=require_candidate_step
    )
    return workflow, _tool_autonomous_goal(workflow)


def _judge_tool_scope(context: Any) -> tuple[SystemWorkflow, AutonomousGoal]:
    workflow = _tool_workflow(context)
    _validate_judge_tool_workflow(workflow, context)
    return workflow, _tool_autonomous_goal(workflow)


def _tool_workflow(context: Any) -> SystemWorkflow:
    workflow_id = getattr(context, "workflow_id", None)
    if not isinstance(workflow_id, int) or isinstance(workflow_id, bool):
        raise ValueError("this tool is not attached to an autonomous goal workflow")
    workflow = SystemWorkflow.objects.filter(pk=workflow_id).first()
    if workflow is None:
        raise ValueError("autonomous goal workflow no longer exists")
    return workflow


def _tool_autonomous_goal(workflow: SystemWorkflow) -> AutonomousGoal:
    autonomous_goal = (
        AutonomousGoal.objects.select_related("project")
        .filter(
            pk=_state_int(workflow, "autonomous_goal_id"),
            deleted_at__isnull=True,
        )
        .first()
    )
    if autonomous_goal is None:
        raise ValueError("autonomous goal no longer exists")
    return autonomous_goal


def _validate_candidate_tool_workflow(
    workflow: SystemWorkflow,
    context: Any,
    *,
    require_candidate_step: bool,
) -> None:
    if getattr(context, "purpose", "") != CodexInstance.PURPOSE_SYSTEM_AGENT:
        raise ValueError("candidate tools require a hidden system session")
    if getattr(context, "agent_kind", "") != system_agents.AUTONOMOUS_GOAL_AGENT_KIND:
        raise ValueError("candidate tool called from the wrong agent role")
    if workflow.kind != system_agents.AUTONOMOUS_GOAL_AGENT_KIND:
        raise ValueError("candidate tool called for the wrong workflow kind")
    if workflow.state.get(_AUTONOMOUS_GOAL_TOOL_PROTOCOL_STATE_KEY) is not True:
        raise ValueError("this workflow predates the autonomous goal tool protocol")
    if not workflow.is_active:
        raise ValueError("autonomous goal workflow is no longer active")
    metadata = _session_metadata_from_state(workflow, "candidate_session_id")
    if metadata is None or metadata.thread_id != getattr(context, "thread_id", ""):
        raise ValueError("candidate tool called from a different session")
    if require_candidate_step and workflow.step != system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING:
        raise ValueError("candidate is not currently allowed to finish or request judgment")


def _validate_judge_tool_workflow(workflow: SystemWorkflow, context: Any) -> None:
    if getattr(context, "purpose", "") != CodexInstance.PURPOSE_SYSTEM_AGENT:
        raise ValueError("judge tools require a hidden system session")
    if getattr(context, "agent_kind", "") != system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND:
        raise ValueError("judge tool called from the wrong agent role")
    if workflow.kind != system_agents.AUTONOMOUS_GOAL_AGENT_KIND:
        raise ValueError("judge tool called for the wrong workflow kind")
    if workflow.state.get(_AUTONOMOUS_GOAL_TOOL_PROTOCOL_STATE_KEY) is not True:
        raise ValueError("this workflow predates the autonomous goal tool protocol")
    if not workflow.is_active or workflow.step != system_agents.STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING:
        raise ValueError("no judgment request is active")
    metadata = _session_metadata_from_state(workflow, "judge_session_id")
    if metadata is None or metadata.thread_id != getattr(context, "thread_id", ""):
        raise ValueError("judge tool called from a different session")


def _candidate_from_tool_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    required = {
        "title",
        "summary",
        "impact",
        "implemented_changes",
        "implementation_direction",
        "verification",
        "rough_edges",
        "suggested_continuation",
        "relevant_files",
    }
    missing = required - set(arguments)
    unexpected = set(arguments) - required
    if missing:
        raise ValueError(f"missing candidate fields: {', '.join(sorted(missing))}")
    if unexpected:
        raise ValueError(f"unexpected candidate fields: {', '.join(sorted(unexpected))}")
    candidate: dict[str, Any] = {
        key: _required_tool_string(arguments, key)
        for key in required - {"relevant_files"}
    }
    if not candidate["title"]:
        raise ValueError("title is required")
    candidate["relevant_files"] = _tool_string_list(arguments, "relevant_files")
    return candidate


def _required_tool_string(arguments: Mapping[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value.strip()


def _optional_tool_string(arguments: Mapping[str, Any], name: str) -> str:
    value = arguments.get(name, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value.strip()


def _tool_string_list(arguments: Mapping[str, Any], name: str) -> list[str]:
    value = arguments.get(name, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a list of strings")
    return _string_list(value)


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
    global _quota_cache_status, _quota_cache_checked_at
    with _quota_cache_lock:
        _quota_cache_status = "available"
        _quota_cache_checked_at = None

def _auto_proposals_paused_by_usage_quota_throttled() -> bool:
    return _auto_proposal_quota_status_throttled() != "available"

def _auto_proposal_quota_status_throttled() -> AutoProposalQuotaStatus:
    """Return quota status, refreshing the remote check at most once
    per ``_AUTO_PROPOSAL_QUOTA_CACHE_TTL`` so the minute-cadence scheduler does
    not poll the Codex rate-limit endpoint every tick."""
    global _quota_cache_status, _quota_cache_checked_at
    with _quota_cache_lock:
        now = timezone.now()
        if (
            _quota_cache_checked_at is not None
            and now - _quota_cache_checked_at < _AUTO_PROPOSAL_QUOTA_CACHE_TTL
        ):
            return _quota_cache_status
        _quota_cache_status = _auto_proposal_quota_status()
        _quota_cache_checked_at = now
        return _quota_cache_status

def _auto_proposals_paused_by_usage_quota() -> bool:
    return _auto_proposal_quota_status() != "available"

def _auto_proposal_quota_status() -> AutoProposalQuotaStatus:
    try:
        with app_server_pool.borrow_codex(Codex) as codex:
            response = codex._client.request(
                "account/rateLimits/read",
                None,
                response_model=GetAccountRateLimitsResponse,
            )
        return _auto_proposal_quota_status_from_rate_limits(
            response.rate_limits,
            now=timezone.now(),
        )
    except CodexError:
        # Automatic work must fail closed when the account snapshot is
        # unavailable. A manual Run bypasses this scheduler guard, so users can
        # still explicitly override an unverified quota state.
        return "unavailable"
    except Exception:
        system_agents.logger.exception(
            "failed to verify account rate limits for auto-proposal quota pause"
        )
        return "unavailable"


def _auto_proposal_quota_status_from_rate_limits(
    rate_limits: Any, *, now: datetime
) -> AutoProposalQuotaStatus:
    statuses = tuple(
        (
            _rate_limit_window_auto_proposal_quota_status(window, now=now)
            if window is not None
            else None
        )
        for window in (rate_limits.primary, rate_limits.secondary)
    )
    if None in statuses:
        return "unavailable"
    if "low" in statuses:
        return "low"
    return "available"

def _rate_limit_window_below_auto_proposal_quota(
    window: Any, *, now: datetime
) -> bool:
    return _rate_limit_window_auto_proposal_quota_status(window, now=now) == "low"

def _rate_limit_window_auto_proposal_quota_status(
    window: Any, *, now: datetime
) -> Literal["available", "low"] | None:
    used_percent = getattr(window, "used_percent", None)
    resets_at = getattr(window, "resets_at", None)
    duration_mins = getattr(window, "window_duration_mins", None)
    if used_percent is None or resets_at is None or duration_mins is None:
        return None

    try:
        used = float(used_percent)
        reset_timestamp = float(resets_at)
        duration_seconds = float(duration_mins) * 60
    except (TypeError, ValueError):
        return None
    if duration_seconds <= 0:
        return None

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
    return "low" if remaining_percent < pause_threshold else "available"

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

    try:
        with transaction.atomic():
            _lock_auto_proposal_queue()
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
    except OperationalError as exc:
        if not db.is_database_locked_error(exc):
            raise
        logger.warning(
            "skipping auto-proposal workflow start for goal %s because database is locked",
            autonomous_goal_id,
        )
        return False
    if created:
        _spawn_autonomous_goal_candidate_or_block(workflow, autonomous_goal)
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
    if _autonomous_goal_accepted_session_blocks_start(autonomous_goal):
        return False
    if autonomous_goal_queue_busy():
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

@dataclass(frozen=True)
class _AutonomousGoalPostCommitAction:
    kind: str = ""
    candidate: dict[str, Any] | None = None
    cleanup_candidate_cwds: tuple[str, ...] = ()
    release_snapshot_refs: tuple[tuple[str, str], ...] = ()


def _tool_protocol_resource_cleanup_action(
    workflow: SystemWorkflow,
    *,
    repo_path: str,
    cleanup_candidate_cwds: tuple[str, ...] = (),
) -> _AutonomousGoalPostCommitAction:
    if workflow.state.get(_AUTONOMOUS_GOAL_TOOL_PROTOCOL_STATE_KEY) is not True:
        return _AutonomousGoalPostCommitAction(
            cleanup_candidate_cwds=cleanup_candidate_cwds
        )
    judge_cwd = _state_string(
        workflow, _AUTONOMOUS_GOAL_JUDGE_SNAPSHOT_CWD_STATE_KEY
    )
    snapshot_ref = _state_string(
        workflow, _AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_REF_STATE_KEY
    )
    return _AutonomousGoalPostCommitAction(
        cleanup_candidate_cwds=tuple(
            unique_nonempty((*cleanup_candidate_cwds, judge_cwd))
        ),
        release_snapshot_refs=(
            ((repo_path, snapshot_ref),) if repo_path and snapshot_ref else ()
        ),
    )

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
    if _autonomous_goal_accepted_session_blocks_start(autonomous_goal):
        return None
    if autonomous_goal_queue_busy():
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
    return default_branch_commit_hash(start_snapshot.repo_path)

def _lock_autonomous_goal_queue() -> None:
    """Serialize the global autonomous-goal check/create critical section."""
    RefreshThrottle.objects.get_or_create(
        key=_AUTO_PROPOSAL_QUEUE_LOCK_KEY,
        defaults={"attempted_at": timezone.now()},
    )
    RefreshThrottle.objects.select_for_update().get(
        key=_AUTO_PROPOSAL_QUEUE_LOCK_KEY
    )


def _lock_auto_proposal_queue() -> None:
    _lock_autonomous_goal_queue()


def _running_auto_proposal_workflow_exists() -> bool:
    return SystemWorkflow.objects.filter(
        kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        status=SystemWorkflow.STATUS_RUNNING,
        state__auto_proposal=True,
    ).exists()


def autonomous_goal_queue_busy() -> bool:
    return SystemWorkflow.objects.filter(
        kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        status=SystemWorkflow.STATUS_RUNNING,
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
        _spawn_autonomous_goal_candidate_or_block(workflow, autonomous_goal)
    return workflow


def start_autonomous_goal_workflow_if_queue_idle(
    *,
    autonomous_goal: AutonomousGoal,
    use_worktrees: bool = False,
) -> SystemWorkflow | None:
    with transaction.atomic():
        _lock_autonomous_goal_queue()
        autonomous_goal = (
            AutonomousGoal.objects.select_related("project")
            .select_for_update()
            .filter(pk=autonomous_goal.pk, deleted_at__isnull=True)
            .get()
        )
        Project.objects.select_for_update().get(pk=autonomous_goal.project_id)
        if autonomous_goal_queue_busy():
            return None
        workflow, created = _create_autonomous_goal_workflow_record(
            autonomous_goal=autonomous_goal,
            auto_proposal=False,
            default_branch_sha=None,
            use_worktrees=use_worktrees,
            stack_continuation_proposal=None,
        )
    if created:
        _spawn_autonomous_goal_candidate_or_block(workflow, autonomous_goal)
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
        _AUTONOMOUS_GOAL_TOOL_PROTOCOL_STATE_KEY: True,
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
        snapshot_sha = str(
            _proposal_outcome_metadata(stack_continuation_proposal).get(
                AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_METADATA_KEY, ""
            )
            or ""
        )
        if snapshot_sha:
            state[_AUTONOMOUS_GOAL_STACKED_BASE_REF_STATE_KEY] = snapshot_sha
            state[_AUTONOMOUS_GOAL_STACKED_FORK_CWD_STATE_KEY] = ""
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
    return system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING


def _seed_stack_continuation_proposal_budget_state(
    state: dict[str, Any], proposal: ProposedSession
) -> None:
    if _AUTONOMOUS_GOAL_PROPOSAL_BUDGET_STATE_KEY not in state:
        return
    metadata = _proposal_outcome_metadata(proposal)
    for key in (
        _AUTONOMOUS_GOAL_PROPOSAL_BUDGET_USED_STATE_KEY,
        _AUTONOMOUS_GOAL_FAILED_ATTEMPTS_STATE_KEY,
    ):
        value = _proposal_metadata_non_negative_int(metadata, key)
        if value is not None:
            state[key] = value


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


def _spawn_autonomous_goal_candidate_protocol_recovery_or_block(
    workflow: SystemWorkflow, autonomous_goal: AutonomousGoal
) -> None:
    workflow, locked_goal, should_spawn = _claim_active_autonomous_goal_workflow(
        workflow_id=workflow.pk,
        autonomous_goal_id=autonomous_goal.pk,
    )
    if not should_spawn or locked_goal is None:
        return
    try:
        run = _spawn_autonomous_goal_candidate_protocol_recovery_run(
            workflow, locked_goal
        )
    except Exception as exc:
        _block_autonomous_goal_spawn_failure_if_active(
            workflow_id=workflow.pk,
            autonomous_goal_id=locked_goal.pk,
            error=f"failed to resume autonomous goal protocol: {exc!r}",
        )
        return
    _interrupt_spawned_autonomous_goal_run_if_inactive(run)


def _spawn_autonomous_goal_judge_protocol_recovery_or_block(
    workflow: SystemWorkflow, autonomous_goal: AutonomousGoal
) -> None:
    workflow, locked_goal, should_spawn = _claim_active_autonomous_goal_workflow(
        workflow_id=workflow.pk,
        autonomous_goal_id=autonomous_goal.pk,
    )
    if not should_spawn or locked_goal is None:
        return
    expected_request_id = _state_string(
        workflow, _AUTONOMOUS_GOAL_JUDGMENT_REQUEST_STATE_KEY
    )
    try:
        run = _spawn_autonomous_goal_judge_protocol_recovery_run(
            workflow, locked_goal
        )
    except Exception as exc:
        cleanup_cwd = ""
        snapshot_ref = ""
        with transaction.atomic():
            locked = SystemWorkflow.objects.select_for_update().get(pk=workflow.pk)
            request_id = _state_string(
                locked, _AUTONOMOUS_GOAL_JUDGMENT_REQUEST_STATE_KEY
            )
            if (
                not locked.is_active
                or locked.step
                != system_agents.STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING
                or not expected_request_id
                or request_id != expected_request_id
            ):
                return
            error = f"failed to resume judge protocol: {exc!r}"
            state = {
                **locked.state,
                "judgment": {
                    "verdict": "deny",
                    "confidence": AutonomousGoal.CONFIDENCE_MEDIUM,
                    "feedback": error,
                    "summary": "",
                    "rationale": error,
                },
                _AUTONOMOUS_GOAL_JUDGMENT_RESULT_STATE_KEY: request_id,
            }
            cleanup_cwd = str(
                state.pop(_AUTONOMOUS_GOAL_JUDGE_SNAPSHOT_CWD_STATE_KEY, "")
                or ""
            )
            snapshot_ref = str(
                state.pop(
                    _AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_REF_STATE_KEY, ""
                )
                or ""
            )
            state.pop(_AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_STATE_KEY, None)
            locked.state = state
            locked.step = system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING
            locked.save(update_fields=["step", "state", "updated_at"])
        _cleanup_autonomous_goal_candidate_cwd(cleanup_cwd)
        _release_autonomous_goal_snapshot_ref(
            locked_goal.project.repo_path, snapshot_ref
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
    interrupted_runs = _interrupt_autonomous_goal_runs([run], error=error)
    if not interrupted_runs:
        return workflow
    system_agents._mark_system_agent_runs_failed(interrupted_runs, error)
    cleanup_action = _tool_protocol_resource_cleanup_action(
        workflow, repo_path=workflow.cwd
    )
    for repo_path, ref in cleanup_action.release_snapshot_refs:
        _release_autonomous_goal_snapshot_ref(repo_path, ref)
    _cleanup_autonomous_goal_workflow_worktree(workflow)
    for cwd in cleanup_action.cleanup_candidate_cwds:
        _cleanup_autonomous_goal_candidate_cwd(cwd)
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
    cleanup_action = _tool_protocol_resource_cleanup_action(
        workflow, repo_path=workflow.cwd
    )
    runs = list(
        workflow.agent_runs.filter(status=SystemAgentRun.STATUS_RUNNING)
        .select_related("instance")
        .order_by("-created_at")
    )
    if runs:
        interrupted_runs = _interrupt_autonomous_goal_runs(runs, error=error)
        if len(interrupted_runs) != len(runs):
            return False
        system_agents._mark_system_agent_runs_failed(interrupted_runs, error)
    system_agents._block_workflow(workflow, error, surface_to_thread=False)
    for repo_path, ref in cleanup_action.release_snapshot_refs:
        _release_autonomous_goal_snapshot_ref(repo_path, ref)
    _cleanup_autonomous_goal_workflow_worktree(workflow)
    for cwd in cleanup_action.cleanup_candidate_cwds:
        _cleanup_autonomous_goal_candidate_cwd(cwd)
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
    runs: list[SystemAgentRun] = []
    cleanup_cwd = ""
    resource_cleanup_action = _AutonomousGoalPostCommitAction()
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
            interrupted_runs = _interrupt_autonomous_goal_runs(runs, error=error)
            if len(interrupted_runs) != len(runs):
                return False
            system_agents._mark_system_agent_runs_failed(interrupted_runs, error)
        _complete_autonomous_goal_workflow_after_proposal_resolution(
            locked,
            outcome_status=outcome_status,
        )
        cleanup_cwd = _autonomous_goal_stack_resolution_continuation_cleanup_cwd(
            locked,
            proposal_id,
        )
        resource_cleanup_action = _tool_protocol_resource_cleanup_action(
            locked, repo_path=locked.cwd
        )
        workflow = locked
    for repo_path, ref in resource_cleanup_action.release_snapshot_refs:
        _release_autonomous_goal_snapshot_ref(repo_path, ref)
    if cleanup_cwd:
        _cleanup_autonomous_goal_candidate_cwd(cleanup_cwd)
    for cwd in resource_cleanup_action.cleanup_candidate_cwds:
        _cleanup_autonomous_goal_candidate_cwd(cwd)
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

def _autonomous_goal_spawn_needs_recovery(workflow: SystemWorkflow) -> bool:
    """True when an autonomous-goal workflow lost its worker to a dead spawn.

    The step is still owned -- so recovery must defer -- when any of these hold:
    a live (starting/running) instance; a terminal instance whose routing claim
    is still fresh (a finished worker mid-handoff, before its ``SystemAgentRun``
    row is recovered and the step advanced); or an in-flight ``SystemAgentRun``.
    Only a workflow with none of these is genuinely stranded by a spawn handler
    that died before launching the worker.
    """
    instances = CodexInstance.objects.filter(workflow_id=workflow.pk)
    if instances.filter(status__in=CodexInstance.ACTIVE_STATUSES).exists():
        return False
    fresh_claim = timezone.now() - system_agents._WORKFLOW_ROUTE_CLAIM_TIMEOUT
    if instances.filter(workflow_routing_started_at__gte=fresh_claim).exists():
        return False
    return not workflow.agent_runs.filter(
        status=SystemAgentRun.STATUS_RUNNING
    ).exists()

def _recover_stranded_autonomous_goal_workflow(workflow: SystemWorkflow) -> None:
    """Recover an autonomous-goal workflow stranded by a dead spawn handler.

    A tool-protocol judge is read-only and safe to re-drive. Candidate spawns
    create managed worktrees, so re-driving one could duplicate resources or
    use the wrong dispatch. Those runs, and any workflow from before the tool
    protocol, are blocked so the goal can start fresh.
    """
    autonomous_goal_id = _state_int(workflow, "autonomous_goal_id")
    autonomous_goal = (
        AutonomousGoal.objects.select_related("project")
        .filter(pk=autonomous_goal_id, deleted_at__isnull=True)
        .first()
    )
    if autonomous_goal is not None:
        candidate = workflow.state.get("candidate")
        if (
            workflow.state.get(_AUTONOMOUS_GOAL_TOOL_PROTOCOL_STATE_KEY) is True
            and
            workflow.step == system_agents.STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING
            and isinstance(candidate, dict)
        ):
            _spawn_autonomous_goal_judge_or_block(
                workflow, autonomous_goal, candidate
            )
            return
    _block_autonomous_goal_spawn_failure_if_active(
        workflow_id=workflow.pk,
        autonomous_goal_id=autonomous_goal_id,
        error=(
            "autonomous goal run never started: its spawn handler died before "
            "the worker launched"
        ),
    )

@engine.register
class _AutonomousGoalHandler(engine.WorkflowHandler):
    kind = system_agents.AUTONOMOUS_GOAL_AGENT_KIND
    steps = frozenset(
        {
            system_agents.LEGACY_STEP_AUTONOMOUS_GOAL_HISTORY,
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
            "auto_pr_enabled",
            "auto_proposal",
            "auto_qa_enabled",
            "autonomous_goal_id",
            "autonomous_goal_updated_at",
            "autonomy",
            "candidate",
            "candidate_session_id",
            "default_branch_sha",
            "tool_protocol",
            "judgment_attempts",
            "judgment_request_id",
            "judgment_verdict_id",
            "judgment_result_id",
            "candidate_terminal",
            "approved_snapshot_sha",
            "approved_snapshot_ref",
            "judge_snapshot_cwd",
            "protocol_recoveries",
            "judge_protocol_recoveries",
            "judge_session_id",
            "judgment",
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
            "stacked_diff_base_ref",
            "stacked_diff_iteration",
            "stacked_diff_stopped_reason",
            "use_worktrees",
        }
    )

    @override
    def spawn_recovery_specs(self) -> tuple[engine.SpawnRecoverySpec, ...]:
        # Candidate and judge spawns each commit their RUNNING step
        # and then launch the worker; a process death in that gap leaves the
        # workflow RUNNING with no worker, which would otherwise pin the goal
        # (blocking future auto-proposals) and its worktree forever.
        spawn_stale = system_agents._WORKFLOW_SPAWN_STALE_TIMEOUT
        return tuple(
            engine.SpawnRecoverySpec(
                kind=self.kind,
                step=step,
                stale_timeout=spawn_stale,
                needs_recovery=_autonomous_goal_spawn_needs_recovery,
                recover=_recover_stranded_autonomous_goal_workflow,
            )
            for step in (
                system_agents.LEGACY_STEP_AUTONOMOUS_GOAL_HISTORY,
                system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
                system_agents.STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING,
            )
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
    # read+parse. Mirrors the QA finish handlers, which read
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
    for repo_path, ref in post_commit_action.release_snapshot_refs:
        _release_autonomous_goal_snapshot_ref(repo_path, ref)
    if autonomous_goal is None:
        return
    if post_commit_action.kind == _AUTONOMOUS_GOAL_RETRY_CANDIDATE_ACTION:
        _spawn_autonomous_goal_candidate_or_block(workflow, autonomous_goal)
        return
    if post_commit_action.kind == _AUTONOMOUS_GOAL_RETRY_CANDIDATE_CONTINUATION_ACTION:
        _spawn_autonomous_goal_candidate_retry_or_block(workflow, autonomous_goal)
        return
    if post_commit_action.kind == _AUTONOMOUS_GOAL_PROTOCOL_RECOVERY_ACTION:
        _spawn_autonomous_goal_candidate_protocol_recovery_or_block(
            workflow, autonomous_goal
        )
        return
    if post_commit_action.kind == _AUTONOMOUS_GOAL_JUDGE_PROTOCOL_RECOVERY_ACTION:
        _spawn_autonomous_goal_judge_protocol_recovery_or_block(
            workflow, autonomous_goal
        )
        return
    if post_commit_action.kind == _AUTONOMOUS_GOAL_SPAWN_NEXT_CANDIDATE_ACTION:
        _spawn_autonomous_goal_candidate_or_block(workflow, autonomous_goal)
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
            unique_nonempty((cleanup_cwd, resolved_proposal_cleanup_cwd))
        )
        return _tool_protocol_resource_cleanup_action(
            workflow,
            repo_path=autonomous_goal.project.repo_path,
            cleanup_candidate_cwds=resolution_cleanup_cwds,
        )
    if workflow.state.get(_AUTONOMOUS_GOAL_TOOL_PROTOCOL_STATE_KEY) is not True:
        action = _fail_autonomous_goal_run_and_block_workflow(
            run,
            autonomous_goal,
            "autonomous goal run was retired by the tool-protocol upgrade",
            raw_output,
        ) or _AutonomousGoalPostCommitAction()
        cleanup_cwd = system_agents._candidate_session_cwd_from_state(
            workflow, "candidate_session_id"
        )
        return _AutonomousGoalPostCommitAction(
            action.kind,
            action.candidate,
            cleanup_candidate_cwds=tuple(
                unique_nonempty((*action.cleanup_candidate_cwds, cleanup_cwd))
            ),
            release_snapshot_refs=action.release_snapshot_refs,
        )
    return _handle_tool_protocol_agent_finished_locked(
        instance,
        run,
        workflow,
        autonomous_goal,
        raw_output,
        tokens_used,
        token_delta,
    )


def _handle_tool_protocol_agent_finished_locked(
    instance: CodexInstance,
    run: SystemAgentRun,
    workflow: SystemWorkflow,
    autonomous_goal: AutonomousGoal,
    raw_output: str,
    tokens_used: int | None,
    token_delta: int,
) -> _AutonomousGoalPostCommitAction | None:
    if run.agent_kind == system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND:
        return _handle_tool_judge_finished_locked(
            instance, run, workflow, raw_output
        )
    if run.agent_kind != system_agents.AUTONOMOUS_GOAL_AGENT_KIND:
        return _fail_autonomous_goal_run_and_block_workflow(
            run,
            autonomous_goal,
            "unexpected autonomous goal tool-protocol agent role",
            raw_output,
        )
    terminal = _state_string(
        workflow, _AUTONOMOUS_GOAL_CANDIDATE_TERMINAL_STATE_KEY
    )
    if workflow.step == system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING:
        if terminal == "approved":
            return _finish_tool_approved_candidate(
                run, workflow, autonomous_goal, raw_output
            )
        if terminal == "no_proposal":
            return _finish_tool_no_proposal_candidate(
                run,
                workflow,
                autonomous_goal,
                raw_output,
            )
    if instance.status != CodexInstance.STATUS_COMPLETED:
        error = f"autonomous goal worker failed: {instance.error}"
        if workflow.step == system_agents.STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING:
            run.status = SystemAgentRun.STATUS_FAILED
            run.error = error
            run.raw_output = raw_output
            run.save(
                update_fields=["status", "error", "raw_output", "updated_at"]
            )
            if not _interrupt_tool_protocol_sibling_runs(
                run, workflow, error=error
            ):
                return None
            return _fail_autonomous_goal_run_and_block_workflow(
                run, autonomous_goal, error, raw_output
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
            resource_cleanup_action = _tool_protocol_resource_cleanup_action(
                workflow, repo_path=autonomous_goal.project.repo_path
            )
            workflow.state = _state_without_tool_candidate_result(workflow.state)
            workflow.save(update_fields=["state", "updated_at"])
            return _AutonomousGoalPostCommitAction(
                retry_action.kind,
                retry_action.candidate,
                cleanup_candidate_cwds=(
                    resource_cleanup_action.cleanup_candidate_cwds
                ),
                release_snapshot_refs=(
                    resource_cleanup_action.release_snapshot_refs
                ),
            )
        return _fail_autonomous_goal_run_and_block_workflow(
            run, autonomous_goal, error, raw_output
        )

    return _recover_tool_candidate_protocol(
        run, workflow, autonomous_goal, raw_output
    )


def _interrupt_tool_protocol_sibling_runs(
    run: SystemAgentRun, workflow: SystemWorkflow, *, error: str
) -> bool:
    siblings = list(
        workflow.agent_runs.select_for_update()
        .filter(status=SystemAgentRun.STATUS_RUNNING)
        .exclude(pk=run.pk)
        .select_related("instance")
        .order_by("created_at", "id")
    )
    if not siblings:
        return True
    interrupted_runs = _interrupt_autonomous_goal_runs(siblings, error=error)
    if len(interrupted_runs) != len(siblings):
        return False
    system_agents._mark_system_agent_runs_failed(interrupted_runs, error)
    return True


def _handle_tool_judge_finished_locked(
    instance: CodexInstance,
    run: SystemAgentRun,
    workflow: SystemWorkflow,
    raw_output: str,
) -> _AutonomousGoalPostCommitAction | None:
    request_id = _state_string(
        workflow, _AUTONOMOUS_GOAL_JUDGMENT_REQUEST_STATE_KEY
    )
    result_id = _state_string(
        workflow, _AUTONOMOUS_GOAL_JUDGMENT_RESULT_STATE_KEY
    )
    verdict_id = _state_string(
        workflow, _AUTONOMOUS_GOAL_JUDGMENT_VERDICT_STATE_KEY
    )
    if request_id and result_id == request_id:
        run.status = SystemAgentRun.STATUS_COMPLETED
        run.output = workflow.state.get("judgment", {})
        run.raw_output = raw_output
        run.save(update_fields=["status", "output", "raw_output", "updated_at"])
        return None
    if request_id and verdict_id == request_id:
        judgment = workflow.state.get("judgment")
        if not isinstance(judgment, dict):
            return _fail_autonomous_goal_run_and_block_workflow(
                run,
                _tool_autonomous_goal(workflow),
                "autonomous goal judge recorded an invalid verdict",
                raw_output,
            )
        run.status = SystemAgentRun.STATUS_COMPLETED
        run.output = judgment
        run.raw_output = raw_output
        run.save(update_fields=["status", "output", "raw_output", "updated_at"])
        state = {
            **workflow.state,
            _AUTONOMOUS_GOAL_JUDGMENT_RESULT_STATE_KEY: request_id,
        }
        state.pop(_AUTONOMOUS_GOAL_JUDGMENT_VERDICT_STATE_KEY, None)
        judge_cwd = str(
            state.pop(_AUTONOMOUS_GOAL_JUDGE_SNAPSHOT_CWD_STATE_KEY, "") or ""
        )
        release_snapshot_refs: tuple[tuple[str, str], ...] = ()
        if judgment.get("verdict") != "approve":
            snapshot_ref = str(
                state.pop(
                    _AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_REF_STATE_KEY, ""
                )
                or ""
            )
            state.pop(_AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_STATE_KEY, None)
            if snapshot_ref:
                release_snapshot_refs = (
                    (_tool_autonomous_goal(workflow).project.repo_path, snapshot_ref),
                )
        workflow.state = state
        workflow.step = system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING
        workflow.save(update_fields=["step", "state", "updated_at"])
        return _AutonomousGoalPostCommitAction(
            cleanup_candidate_cwds=((judge_cwd,) if judge_cwd else ()),
            release_snapshot_refs=release_snapshot_refs,
        )

    recoveries = _state_int(
        workflow, _AUTONOMOUS_GOAL_JUDGE_PROTOCOL_RECOVERIES_STATE_KEY
    )
    if (
        instance.status == CodexInstance.STATUS_COMPLETED
        and recoveries < _AUTONOMOUS_GOAL_MAX_JUDGE_PROTOCOL_RECOVERIES
    ):
        run.status = SystemAgentRun.STATUS_COMPLETED
        run.raw_output = raw_output
        run.save(update_fields=["status", "raw_output", "updated_at"])
        workflow.state = {
            **workflow.state,
            _AUTONOMOUS_GOAL_JUDGE_PROTOCOL_RECOVERIES_STATE_KEY: recoveries + 1,
        }
        workflow.save(update_fields=["state", "updated_at"])
        return _AutonomousGoalPostCommitAction(
            _AUTONOMOUS_GOAL_JUDGE_PROTOCOL_RECOVERY_ACTION
        )

    error = (
        "autonomous goal judge stopped twice without calling approve or deny"
        if instance.status == CodexInstance.STATUS_COMPLETED
        else f"autonomous goal judge failed: {instance.error}"
    )
    run.status = SystemAgentRun.STATUS_FAILED
    run.error = error
    run.raw_output = raw_output
    run.save(update_fields=["status", "error", "raw_output", "updated_at"])
    if request_id:
        judgment = {
            "verdict": "deny",
            "confidence": AutonomousGoal.CONFIDENCE_MEDIUM,
            "feedback": error,
        }
        state = {
            **workflow.state,
            "judgment": judgment,
            _AUTONOMOUS_GOAL_JUDGMENT_RESULT_STATE_KEY: request_id,
        }
        judge_cwd = str(
            state.pop(_AUTONOMOUS_GOAL_JUDGE_SNAPSHOT_CWD_STATE_KEY, "") or ""
        )
        snapshot_ref = str(
            state.pop(_AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_REF_STATE_KEY, "") or ""
        )
        state.pop(_AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_STATE_KEY, None)
        workflow.state = state
        workflow.step = system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING
        workflow.save(update_fields=["step", "state", "updated_at"])
        return _AutonomousGoalPostCommitAction(
            cleanup_candidate_cwds=((judge_cwd,) if judge_cwd else ()),
            release_snapshot_refs=(
                ((_tool_autonomous_goal(workflow).project.repo_path, snapshot_ref),)
                if snapshot_ref
                else ()
            ),
        )
    return _fail_autonomous_goal_run_and_block_workflow(
        run, _tool_autonomous_goal(workflow), error, raw_output
    )


def _recover_tool_candidate_protocol(
    run: SystemAgentRun,
    workflow: SystemWorkflow,
    autonomous_goal: AutonomousGoal,
    raw_output: str,
) -> _AutonomousGoalPostCommitAction | None:
    recoveries = _state_int(
        workflow, _AUTONOMOUS_GOAL_PROTOCOL_RECOVERIES_STATE_KEY
    )
    if recoveries >= _AUTONOMOUS_GOAL_MAX_PROTOCOL_RECOVERIES:
        return _fail_autonomous_goal_run_and_block_workflow(
            run,
            autonomous_goal,
            "autonomous goal candidate repeatedly stopped without calling judge or no_proposal",
            raw_output,
        )
    run.status = SystemAgentRun.STATUS_COMPLETED
    run.raw_output = raw_output
    run.save(update_fields=["status", "raw_output", "updated_at"])
    workflow.state = {
        **workflow.state,
        _AUTONOMOUS_GOAL_PROTOCOL_RECOVERIES_STATE_KEY: recoveries + 1,
    }
    workflow.save(update_fields=["state", "updated_at"])
    return _AutonomousGoalPostCommitAction(_AUTONOMOUS_GOAL_PROTOCOL_RECOVERY_ACTION)


def _finish_tool_approved_candidate(
    run: SystemAgentRun,
    workflow: SystemWorkflow,
    autonomous_goal: AutonomousGoal,
    raw_output: str,
) -> _AutonomousGoalPostCommitAction:
    candidate = workflow.state.get("candidate")
    judgment = workflow.state.get("judgment")
    if not isinstance(candidate, dict) or not isinstance(judgment, dict):
        return _fail_autonomous_goal_run_and_block_workflow(
            run,
            autonomous_goal,
            "approved autonomous goal candidate data is unavailable",
            raw_output,
        ) or _AutonomousGoalPostCommitAction()
    run.status = SystemAgentRun.STATUS_COMPLETED
    run.output = candidate
    run.raw_output = raw_output
    run.save(update_fields=["status", "output", "raw_output", "updated_at"])
    previous_proposal = _autonomous_goal_current_stack_proposal(workflow)
    proposal = _create_autonomous_goal_proposal(
        workflow,
        autonomous_goal,
        candidate,
        cast(dict[str, str], judgment),
    )
    cleanup_cwds, release_snapshot_refs = (
        _dismiss_replaced_autonomous_goal_proposal(
            previous_proposal, replacement=proposal
        )
    )
    _record_autonomous_goal_proposal_created(autonomous_goal)
    state = _state_after_autonomous_goal_proposal_progress(workflow.state)
    workflow.state = {
        **state,
        "proposal_id": proposal.pk,
        "autonomy": autonomous_goal.autonomy,
    }
    if _autonomous_goal_should_continue_stack(workflow, autonomous_goal):
        workflow.state = _autonomous_goal_next_stack_candidate_state(
            workflow, proposal
        )
        workflow.step = system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING
        workflow.save(update_fields=["step", "state", "updated_at"])
        return _AutonomousGoalPostCommitAction(
            _AUTONOMOUS_GOAL_SPAWN_NEXT_CANDIDATE_ACTION,
            cleanup_candidate_cwds=cleanup_cwds,
            release_snapshot_refs=release_snapshot_refs,
        )
    system_agents._complete_workflow(
        workflow, system_agents.STEP_AUTONOMOUS_GOAL_PROPOSED
    )
    return _AutonomousGoalPostCommitAction(
        cleanup_candidate_cwds=cleanup_cwds,
        release_snapshot_refs=release_snapshot_refs,
    )


def _finish_tool_no_proposal_candidate(
    run: SystemAgentRun,
    workflow: SystemWorkflow,
    autonomous_goal: AutonomousGoal,
    raw_output: str,
) -> _AutonomousGoalPostCommitAction | None:
    candidate_output = workflow.state.get("candidate")
    if not isinstance(candidate_output, dict):
        candidate_output = {
            "proposal": None,
            "message": "No worthwhile proposal was found.",
        }
    message = str(candidate_output.get("message") or "No worthwhile proposal was found.")
    run.status = SystemAgentRun.STATUS_COMPLETED
    run.output = candidate_output
    run.raw_output = raw_output
    run.save(update_fields=["status", "output", "raw_output", "updated_at"])
    cleanup_cwd = system_agents._candidate_session_cwd_from_state(
        workflow, "candidate_session_id"
    )
    previous_proposal = _autonomous_goal_current_stack_proposal(workflow)
    stop_reason = "candidate_no_proposal"
    if previous_proposal is not None and _publish_current_stack_proposal(
        previous_proposal,
        workflow=workflow,
        continuation_stopped_reason=stop_reason,
    ):
        workflow.state = {
            **workflow.state,
            "stacked_diff_stopped_reason": stop_reason,
        }
        system_agents._complete_workflow(
            workflow, system_agents.STEP_AUTONOMOUS_GOAL_PROPOSED
        )
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
        outcome_metadata={
            "automation_status": "skipped",
            "skip_reason": "candidate_no_proposal",
            **_autonomous_goal_proposal_budget_metadata(workflow),
        },
    )
    _record_autonomous_goal_no_proposal(autonomous_goal, workflow)
    system_agents._complete_workflow(
        workflow, system_agents.STEP_AUTONOMOUS_GOAL_SKIPPED
    )
    return _AutonomousGoalPostCommitAction(
        cleanup_candidate_cwds=((cleanup_cwd,) if cleanup_cwd else ())
    )


def _state_without_tool_candidate_result(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    next_state = system_agents._state_without_current_candidate_result(state)
    for key in (
        _AUTONOMOUS_GOAL_CANDIDATE_TERMINAL_STATE_KEY,
        _AUTONOMOUS_GOAL_JUDGMENT_ATTEMPTS_STATE_KEY,
        _AUTONOMOUS_GOAL_JUDGMENT_REQUEST_STATE_KEY,
        _AUTONOMOUS_GOAL_JUDGMENT_VERDICT_STATE_KEY,
        _AUTONOMOUS_GOAL_JUDGMENT_RESULT_STATE_KEY,
        _AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_STATE_KEY,
        _AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_REF_STATE_KEY,
        _AUTONOMOUS_GOAL_JUDGE_SNAPSHOT_CWD_STATE_KEY,
        _AUTONOMOUS_GOAL_PROTOCOL_RECOVERIES_STATE_KEY,
        _AUTONOMOUS_GOAL_JUDGE_PROTOCOL_RECOVERIES_STATE_KEY,
    ):
        next_state.pop(key, None)
    return next_state

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
            AUTONOMOUS_GOAL_TOOL_PROTOCOL_METADATA_KEY: True,
            AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_METADATA_KEY: _state_string(
                workflow, _AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_STATE_KEY
            ),
            AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_REF_METADATA_KEY: _state_string(
                workflow, _AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_REF_STATE_KEY
            ),
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
        error=error,
        raw_output=raw_output,
        tokens_used=tokens_used,
        token_delta=token_delta,
    )
    workflow.save(update_fields=["state", "updated_at"])
    return _AutonomousGoalPostCommitAction(
        _AUTONOMOUS_GOAL_RETRY_CANDIDATE_CONTINUATION_ACTION
    )

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

def _autonomous_goal_budget_token_progressed(
    *, tokens_used: int | None, token_delta: int
) -> bool:
    return tokens_used is not None and tokens_used > 0 and token_delta > 0

def _record_autonomous_goal_failed_attempt(
    workflow: SystemWorkflow,
    *,
    error: str,
    raw_output: str,
    tokens_used: int | None = None,
    token_delta: int = 0,
) -> None:
    failure: dict[str, object] = {
        "reason": "candidate_failed",
        "proposal_budget": _autonomous_goal_workflow_proposal_budget(workflow),
        "proposal_budget_tokens_used": _autonomous_goal_proposal_budget_tokens_used(
            workflow
        ),
    }
    if tokens_used is not None:
        failure["tokens_used"] = tokens_used
    if error:
        failure["error"] = truncate_for_prompt(error, 800)
    if raw_output:
        failure["raw_output"] = truncate_for_prompt(raw_output, 2000)
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

def _state_after_autonomous_goal_proposal_progress(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    next_state = dict(state)
    next_state.pop(_AUTONOMOUS_GOAL_NO_PROGRESS_RETRIES_STATE_KEY, None)
    next_state.pop(_AUTONOMOUS_GOAL_LAST_FAILURE_STATE_KEY, None)
    return next_state

def _dismiss_replaced_autonomous_goal_proposal(
    previous: ProposedSession | None, *, replacement: ProposedSession
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    if previous is None or previous.pk == replacement.pk:
        return (), ()
    cleanup_cwd = previous.candidate_session.cwd if previous.candidate_session else ""
    if (
        previous.outcome_status != ProposedSession.OUTCOME_UNSET
        and not _autonomous_goal_proposal_hidden_until_complete(previous)
    ):
        return (), ()
    outcome_metadata = {
        **_proposal_outcome_metadata(previous),
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
    if not applied:
        return (), ()
    snapshot_ref = str(
        _proposal_outcome_metadata(previous).get(
            AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_REF_METADATA_KEY, ""
        )
        or ""
    )
    project = previous.project
    repo_path = project.repo_path if project is not None else ""
    return (
        ((cleanup_cwd,) if cleanup_cwd else ()),
        (((repo_path, snapshot_ref),) if repo_path and snapshot_ref else ()),
    )


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
            **_proposal_outcome_metadata(proposal),
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
        **_proposal_outcome_metadata(proposal),
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
        and _proposal_outcome_metadata(proposal).get(
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
    snapshot_sha = _state_string(
        workflow, _AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_STATE_KEY
    )
    if snapshot_sha:
        state[_AUTONOMOUS_GOAL_STACKED_BASE_REF_STATE_KEY] = snapshot_sha
        state[_AUTONOMOUS_GOAL_STACKED_FORK_CWD_STATE_KEY] = ""
    state["proposal_id"] = proposal.pk
    for key in (
        "candidate",
        "candidate_session_id",
        "judge_session_id",
        "judgment",
        _AUTONOMOUS_GOAL_CANDIDATE_TERMINAL_STATE_KEY,
        _AUTONOMOUS_GOAL_JUDGMENT_ATTEMPTS_STATE_KEY,
        _AUTONOMOUS_GOAL_JUDGMENT_REQUEST_STATE_KEY,
        _AUTONOMOUS_GOAL_JUDGMENT_VERDICT_STATE_KEY,
        _AUTONOMOUS_GOAL_JUDGMENT_RESULT_STATE_KEY,
        _AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_STATE_KEY,
        _AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_REF_STATE_KEY,
        _AUTONOMOUS_GOAL_JUDGE_SNAPSHOT_CWD_STATE_KEY,
        _AUTONOMOUS_GOAL_PROTOCOL_RECOVERIES_STATE_KEY,
        _AUTONOMOUS_GOAL_JUDGE_PROTOCOL_RECOVERIES_STATE_KEY,
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


def _release_autonomous_goal_snapshot_ref(repo_path: str, ref: str) -> None:
    if not ref:
        return
    try:
        release_snapshot_commit_ref(repo_path, ref)
    except WorktreeCleanupError:
        system_agents.logger.exception(
            "failed to release autonomous goal snapshot ref %s", ref
        )

def _prepare_autonomous_goal_candidate_cwd(
    workflow: SystemWorkflow, autonomous_goal: AutonomousGoal
) -> tuple[str, ManagedWorktree | None]:
    stacked_base_ref = _state_string(
        workflow, _AUTONOMOUS_GOAL_STACKED_BASE_REF_STATE_KEY
    )
    if stacked_base_ref:
        managed_worktree = create_worktree_for_session(
            autonomous_goal.project.repo_path,
            base_ref=stacked_base_ref,
        )
        session_cwd = str(managed_worktree.path)
        workflow.state = {
            **workflow.state,
            _AUTONOMOUS_GOAL_SESSION_CWD_STATE_KEY: session_cwd,
            _AUTONOMOUS_GOAL_STACKED_BASE_REF_STATE_KEY: "",
        }
        try:
            workflow.save(update_fields=["state", "updated_at"])
        except Exception:
            _cleanup_new_autonomous_goal_worktree(managed_worktree)
            raise
        return session_cwd, managed_worktree
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

def _autonomous_goal_tool_candidate_prompt(
    workflow: SystemWorkflow, autonomous_goal: AutonomousGoal
) -> str:
    ambition = autonomous_goal.ambition.replace("_", " ")
    code_guidance = (
        "Do not make code changes; investigate and propose only."
        if not _autonomous_goal_candidate_allows_code_changes(workflow)
        else (
            "Make concrete changes in this hidden checkout, keep the diff "
            "coherent, and run relevant checks when practical. Do not push or "
            "open a pull request."
        )
    )
    return (
        "You are Hitch's autonomous goal candidate agent.\n\n"
        f"Make {ambition} progress toward the goal below. {code_guidance}\n\n"
        f"Repository cwd: {_autonomous_goal_session_cwd(workflow)}\n"
        f"Stack round: {_autonomous_goal_stack_iteration(workflow)} of "
        f"{_autonomous_goal_workflow_stacked_diff_depth(workflow, autonomous_goal)}\n"
        f"Goal title: {autonomous_goal.title}\n\n"
        f"Goal:\n{autonomous_goal.goal}\n\n"
        "Use hitch.get_goal whenever you need the current limits or prior judge "
        "feedback. Use hitch.list_goal_sessions to discover prior sessions; it "
        "returns rollout file paths that you may inspect directly. Hitch does "
        "not summarize history for you.\n\n"
        "When you have a worthwhile concrete result, call hitch.judge with the "
        "complete proposal. You may call it at most twice. If the first call is "
        "denied, address its feedback before the second. If no worthwhile "
        "proposal remains, call hitch.no_proposal. Do not finish your turn "
        "without calling hitch.judge or hitch.no_proposal. Final prose does not "
        "change the workflow state."
    )


def _autonomous_goal_tool_candidate_retry_prompt(
    workflow: SystemWorkflow, autonomous_goal: AutonomousGoal
) -> str:
    return (
        "Continue this autonomous goal in the same hidden candidate session. "
        "Review hitch.get_goal for the current budget and judgment state, and "
        "use hitch.list_goal_sessions if earlier work is relevant. Find a new "
        "or improved candidate, then call hitch.judge; otherwise call "
        "hitch.no_proposal. Final prose alone does not finish the workflow.\n\n"
        f"Goal title: {autonomous_goal.title}\n"
        f"Goal: {autonomous_goal.goal}\n"
        "Prior failure context: "
        f"{json.dumps(_state_dict(workflow, _AUTONOMOUS_GOAL_LAST_FAILURE_STATE_KEY), sort_keys=True)}"
    )


def _autonomous_goal_tool_judge_prompt(
    workflow: SystemWorkflow,
    autonomous_goal: AutonomousGoal,
    candidate: dict[str, Any],
) -> str:
    attempts = _state_int(workflow, _AUTONOMOUS_GOAL_JUDGMENT_ATTEMPTS_STATE_KEY)
    previous = workflow.state.get("judgment")
    previous_feedback = ""
    if isinstance(previous, dict) and previous.get("verdict") == "deny":
        previous_feedback = str(previous.get("feedback") or "").strip()
    snapshot_sha = _state_string(
        workflow, _AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_STATE_KEY
    )
    return (
        "You are Hitch's autonomous goal judge. Evaluate the candidate and its "
        "exact checkout read-only. Call hitch.approve if it makes meaningful, "
        "concrete progress and meets the confidence threshold; otherwise call "
        "hitch.deny. Both tools require confidence and accept optional feedback. "
        "You must call exactly one of them. Final prose is ignored.\n\n"
        f"Goal title: {autonomous_goal.title}\n"
        f"Goal: {autonomous_goal.goal}\n"
        f"Ambition: {autonomous_goal.ambition}\n"
        f"Confidence threshold: {autonomous_goal.confidence_threshold}\n"
        f"Judgment attempt: {attempts} of {_AUTONOMOUS_GOAL_MAX_JUDGMENTS}\n"
        f"Candidate snapshot: {snapshot_sha or '(no-code proposal)'}\n"
        f"Previous judge feedback: {previous_feedback or '(none)'}\n\n"
        "Candidate:\n"
        f"{json.dumps(candidate, indent=2, sort_keys=True)}"
    )


def _spawn_autonomous_goal_candidate_run(
    workflow: SystemWorkflow, autonomous_goal: AutonomousGoal
) -> SystemAgentRun:
    session_cwd, managed_worktree = _prepare_autonomous_goal_candidate_cwd(
        workflow, autonomous_goal
    )
    sandbox_policy = (
        system_agents.AUTONOMOUS_GOAL_IMPLEMENTATION_SANDBOX_POLICY
        if _autonomous_goal_candidate_allows_code_changes(workflow)
        # A no-code candidate runs in the user's real repo, so it must be
        # pinned read-only rather than inheriting workspace-write defaults.
        else "readOnly"
    )
    try:
        prompt = _autonomous_goal_tool_candidate_prompt(workflow, autonomous_goal)
        thread_id, thread_path = codex_pool.create_session_thread_with_path(
            cwd=session_cwd,
            name=prompt,
            web_search_mode=system_agents._workflow_web_search_mode(workflow),
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_source=ThreadSource.subagent,
        )
        metadata = session_index.upsert_local_session(
            thread_id=thread_id,
            cwd=session_cwd,
            project=autonomous_goal.project,
            preview=prompt,
            auto_pr_enabled=False,
            auto_qa_enabled=False,
            codex_path=thread_path,
            is_hidden_system_session=True,
        )
        workflow.state = {
            **workflow.state,
            "candidate_session_id": metadata.pk,
        }
        workflow.save(update_fields=["state", "updated_at"])
        instance = codex_pool.spawn_turn(
            thread_id=thread_id,
            cwd=session_cwd,
            prompt=prompt,
            approval_mode=system_agents.SYSTEM_AGENT_APPROVAL_MODE,
            sandbox_policy=sandbox_policy,
            web_search_mode=system_agents._workflow_web_search_mode(workflow),
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            display_author=system_agents.AUTONOMOUS_GOAL_DISPLAY_AUTHOR,
        )
    except Exception:
        _cleanup_new_autonomous_goal_worktree(managed_worktree)
        raise
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
    prompt = _autonomous_goal_tool_candidate_retry_prompt(workflow, autonomous_goal)
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


def _spawn_autonomous_goal_candidate_protocol_recovery_run(
    workflow: SystemWorkflow, autonomous_goal: AutonomousGoal
) -> SystemAgentRun:
    candidate_session = _session_metadata_from_state(
        workflow, "candidate_session_id"
    )
    if candidate_session is None:
        raise RuntimeError("candidate session is unavailable")
    attempts = _state_int(workflow, _AUTONOMOUS_GOAL_JUDGMENT_ATTEMPTS_STATE_KEY)
    judgment = workflow.state.get("judgment")
    denied_feedback = ""
    if isinstance(judgment, dict) and judgment.get("verdict") == "deny":
        denied_feedback = str(judgment.get("feedback") or "").strip()
    if attempts >= _AUTONOMOUS_GOAL_MAX_JUDGMENTS:
        instruction = (
            "Both judgment attempts have been used. Call hitch.no_proposal now."
        )
    elif denied_feedback:
        instruction = (
            "The first judgment was denied. Address this feedback, then call "
            f"hitch.judge once more, or call hitch.no_proposal: {denied_feedback}"
        )
    else:
        instruction = (
            "Continue the work, then call hitch.judge with a complete proposal "
            "or call hitch.no_proposal."
        )
    prompt = (
        "You stopped without completing the autonomous-goal tool protocol. "
        f"{instruction} Final prose does not complete this workflow."
    )
    instance = codex_pool.spawn_turn(
        thread_id=candidate_session.thread_id,
        cwd=candidate_session.cwd or _autonomous_goal_session_cwd(workflow),
        prompt=prompt,
        approval_mode=system_agents.SYSTEM_AGENT_APPROVAL_MODE,
        sandbox_policy=(
            system_agents.AUTONOMOUS_GOAL_IMPLEMENTATION_SANDBOX_POLICY
            if _autonomous_goal_candidate_allows_code_changes(workflow)
            else "readOnly"
        ),
        web_search_mode=system_agents._workflow_web_search_mode(workflow),
        purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
        workflow_id=workflow.pk,
        agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        display_author=system_agents.AUTONOMOUS_GOAL_DISPLAY_AUTHOR,
    )
    return SystemAgentRun.objects.create(
        instance=instance,
        workflow=workflow,
        agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        thread_id=instance.thread_id,
        status=SystemAgentRun.STATUS_RUNNING,
        input={
            "cwd": instance.cwd,
            "autonomous_goal_id": autonomous_goal.pk,
            "protocol_recovery": _state_int(
                workflow, _AUTONOMOUS_GOAL_PROTOCOL_RECOVERIES_STATE_KEY
            ),
        },
    )


def _spawn_autonomous_goal_judge_protocol_recovery_run(
    workflow: SystemWorkflow, autonomous_goal: AutonomousGoal
) -> SystemAgentRun:
    judge_session = _session_metadata_from_state(workflow, "judge_session_id")
    if judge_session is None:
        raise RuntimeError("judge session is unavailable")
    instance = codex_pool.spawn_turn(
        thread_id=judge_session.thread_id,
        cwd=judge_session.cwd or _autonomous_goal_session_cwd(workflow),
        prompt=(
            "You stopped without recording a judgment. Review the candidate "
            "already in this thread and call exactly one of hitch.approve or "
            "hitch.deny now. Final prose is ignored."
        ),
        approval_mode=system_agents.SYSTEM_AGENT_APPROVAL_MODE,
        sandbox_policy="readOnly",
        web_search_mode=system_agents._workflow_web_search_mode(workflow),
        purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
        workflow_id=workflow.pk,
        agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
        display_author=system_agents.AUTONOMOUS_GOAL_JUDGE_DISPLAY_AUTHOR,
    )
    return SystemAgentRun.objects.create(
        instance=instance,
        workflow=workflow,
        agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
        thread_id=instance.thread_id,
        status=SystemAgentRun.STATUS_RUNNING,
        input={
            "cwd": instance.cwd,
            "autonomous_goal_id": autonomous_goal.pk,
            "protocol_recovery": _state_int(
                workflow, _AUTONOMOUS_GOAL_JUDGE_PROTOCOL_RECOVERIES_STATE_KEY
            ),
        },
    )

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
        blocked_workflow = _block_autonomous_goal_spawn_failure_if_active(
            workflow_id=workflow.pk,
            autonomous_goal_id=locked_goal.pk,
            error=f"failed to start autonomous goal judge: {exc!r}",
        )
        cleanup_action = _tool_protocol_resource_cleanup_action(
            blocked_workflow, repo_path=locked_goal.project.repo_path
        )
        for cwd in cleanup_action.cleanup_candidate_cwds:
            _cleanup_autonomous_goal_candidate_cwd(cwd)
        for repo_path, ref in cleanup_action.release_snapshot_refs:
            _release_autonomous_goal_snapshot_ref(repo_path, ref)
        return
    _interrupt_spawned_autonomous_goal_run_if_inactive(run)

def _spawn_autonomous_goal_judge_run(
    workflow: SystemWorkflow, autonomous_goal: AutonomousGoal, candidate: dict[str, Any]
) -> SystemAgentRun:
    session_cwd = _autonomous_goal_session_cwd(workflow)
    previous_judge_cwd = _state_string(
        workflow, _AUTONOMOUS_GOAL_JUDGE_SNAPSHOT_CWD_STATE_KEY
    )
    if previous_judge_cwd:
        _cleanup_autonomous_goal_candidate_cwd(previous_judge_cwd)
    managed_judge_worktree: ManagedWorktree | None = None
    snapshot_sha = _state_string(
        workflow, _AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_STATE_KEY
    )
    if snapshot_sha:
        managed_judge_worktree = create_worktree_for_session(
            autonomous_goal.project.repo_path,
            base_ref=snapshot_sha,
        )
        session_cwd = str(managed_judge_worktree.path)
    try:
        prompt = _autonomous_goal_tool_judge_prompt(
            workflow, autonomous_goal, candidate
        )
        thread_id, thread_path = codex_pool.create_session_thread_with_path(
            cwd=session_cwd,
            name=prompt,
            web_search_mode=system_agents._workflow_web_search_mode(workflow),
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
            thread_source=ThreadSource.subagent,
        )
        metadata = session_index.upsert_local_session(
            thread_id=thread_id,
            cwd=session_cwd,
            project=autonomous_goal.project,
            preview=prompt,
            auto_pr_enabled=False,
            auto_qa_enabled=False,
            codex_path=thread_path,
            is_hidden_system_session=True,
        )
        workflow.state = {
            **workflow.state,
            "judge_session_id": metadata.pk,
            _AUTONOMOUS_GOAL_JUDGE_SNAPSHOT_CWD_STATE_KEY: (
                session_cwd if managed_judge_worktree is not None else ""
            ),
        }
        workflow.save(update_fields=["state", "updated_at"])
        instance = codex_pool.spawn_turn(
            thread_id=thread_id,
            cwd=session_cwd,
            prompt=prompt,
            approval_mode=system_agents.SYSTEM_AGENT_APPROVAL_MODE,
            sandbox_policy="readOnly",
            web_search_mode=system_agents._workflow_web_search_mode(workflow),
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
            display_author=system_agents.AUTONOMOUS_GOAL_JUDGE_DISPLAY_AUTHOR,
        )
    except Exception:
        _cleanup_new_autonomous_goal_worktree(managed_judge_worktree)
        raise
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
            },
        },
    )
    return run

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
    runs: list[SystemAgentRun], *, error: str
) -> list[SystemAgentRun]:
    interrupted_runs: list[SystemAgentRun] = []
    for run in runs:
        interrupted = codex_pool.interrupt_instance(
            run.instance_id,
            expected_thread_id=run.thread_id,
            force=True,
            error=error,
        )
        if interrupted is None:
            continue
        interrupted_runs.append(run)
    return interrupted_runs

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
    resource_cleanup_action = _tool_protocol_resource_cleanup_action(
        workflow, repo_path=autonomous_goal.project.repo_path
    )
    if _complete_autonomous_goal_with_current_stack_proposal(workflow, error=error):
        cleanup_cwd = system_agents._candidate_session_cwd_from_state(
            workflow, "candidate_session_id"
        )
        return _tool_protocol_resource_cleanup_action(
            workflow,
            repo_path=autonomous_goal.project.repo_path,
            cleanup_candidate_cwds=((cleanup_cwd,) if cleanup_cwd else ()),
        )
    _block_autonomous_goal_workflow(run.workflow, autonomous_goal, error)
    if (
        resource_cleanup_action.cleanup_candidate_cwds
        or resource_cleanup_action.release_snapshot_refs
    ):
        return resource_cleanup_action
    return None

def _block_autonomous_goal_workflow(
    workflow: SystemWorkflow, autonomous_goal: AutonomousGoal, error: str
) -> None:
    # The finish handler may have recorded budget tokens on this locked instance.
    # Persist them before _block_workflow re-reads the row.
    workflow.save(update_fields=["state", "updated_at"])
    _create_autonomous_goal_notice(
        workflow,
        autonomous_goal,
        title=f"Autonomous goal failed: {autonomous_goal.title}"[
            :_AUTONOMOUS_GOAL_TITLE_MAX_LEN
        ],
        summary=f"Hitch could not finish this autonomous goal run: {error}",
        metadata={
            "autonomous_goal_autonomy": autonomous_goal.autonomy,
            "automation_status": "failed",
            "automation_error": error,
        },
        block_on_any_pending_item=True,
    )
    system_agents._block_workflow(workflow, error, surface_to_thread=False)

def _create_autonomous_goal_notice(
    workflow: SystemWorkflow,
    autonomous_goal: AutonomousGoal,
    *,
    title: str,
    summary: str,
    metadata: dict[str, object] | None = None,
    block_on_any_pending_item: bool = False,
) -> None:
    pending = ProposedSession.objects.filter(
        source_workflow=workflow, outcome_status=ProposedSession.OUTCOME_UNSET
    )
    if not block_on_any_pending_item:
        pending = pending.filter(inbox_kind=ProposedSession.INBOX_KIND_NOTICE)
    if pending.exists():
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
