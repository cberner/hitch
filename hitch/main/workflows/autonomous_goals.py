"""Thin durable control plane for autonomous-goal agent runs.

The candidate owns investigation and revision. Hitch only schedules a bounded
candidate turn, runs an isolated reviewer when the candidate calls
``hitch.review``, and durably publishes either the reviewed proposal or a
terminal notice. ``SystemWorkflow`` and ``SystemAgentRun`` remain as the run
ledger for upgrade compatibility; they are not a general workflow engine.
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
from typing import Any, Literal, cast

from django.db import IntegrityError, OperationalError, transaction
from django.utils import timezone
from django.utils.dateparse import parse_datetime
from openai_codex import Codex, CodexError
from openai_codex.generated.v2_all import GetAccountRateLimitsResponse, ThreadSource

from hitch.main.goals.autonomous_goal_prompts import (
    _AUTONOMOUS_GOAL_PROPOSAL_BUDGET_STATE_KEY,
    _AUTONOMOUS_GOAL_PROPOSAL_BUDGET_USED_STATE_KEY,
    _AUTONOMOUS_GOAL_STACKED_DEPTH_STATE_KEY,
    _AUTONOMOUS_GOAL_TITLE_MAX_LEN,
    _autonomous_goal_proposal_summary,
    _autonomous_goal_proposed_session_prompt,
    _string_list,
)
from hitch.main.goals.autonomous_goal_proposal_stack import (
    _AUTONOMOUS_GOAL_STACKED_CONTINUATION_STOP_REASON_METADATA_KEY,
    AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_METADATA_KEY,
    AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_REF_METADATA_KEY,
    AUTONOMOUS_GOAL_TOOL_PROTOCOL_METADATA_KEY,
    _autonomous_goal_accepted_session_blocks_start,
    _autonomous_goal_pending_proposal_blocks_start,
    _autonomous_goal_unresolved_failure_notice_exists,
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
from hitch.main.sequences import unique_nonempty
from hitch.main.sessions import session_index
from hitch.main.workflows import system_agents
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

AutoProposalQuotaStatus = Literal["available", "low", "unavailable"]

_AUTO_PROPOSAL_UNKNOWN_DEFAULT_BRANCH_SHA = "__unknown__"
_AUTO_PROPOSAL_QUOTA_CACHE_TTL = timedelta(minutes=5)
_AUTO_PROPOSAL_QUOTA_THRESHOLD_FRACTION = 0.5
_AUTO_PROPOSAL_QUEUE_LOCK_KEY = "autonomous_goal:auto_proposal_queue"
_ORPHANED_RUN_TIMEOUT = timedelta(minutes=15)
_MAX_REVIEWS = 2

_AUTONOMOUS_GOAL_USE_WORKTREES_STATE_KEY = "use_worktrees"
_AUTONOMOUS_GOAL_RESULT_REASON_STATE_KEY = "result_reason"
_AUTONOMOUS_GOAL_ERROR_STATE_KEY = "error"
_AUTONOMOUS_GOAL_REVIEW_AGENT_KIND = "autonomous_goal_reviewer"
_AUTONOMOUS_GOAL_PROPOSAL_TERMINAL = "propose"
_AUTONOMOUS_GOAL_NO_PROPOSAL_TERMINAL = "no_proposal"
_RUN_PENDING_CLEANUP_OUTPUT_KEY = "pending_cleanup"
_LEGACY_WORKFLOW_UPGRADE_ERROR = "autonomous goal run retired during the tool-driven protocol upgrade"
_LEGACY_RESOURCE_CWD_STATE_KEYS = ("session_cwd", "judge_snapshot_cwd")
_LEGACY_RESOURCE_REF_STATE_KEYS = ("approved_snapshot_ref",)
_quota_cache_lock = threading.Lock()
_quota_cache_status: AutoProposalQuotaStatus = "available"
_quota_cache_checked_at: datetime | None = None


@dataclass(frozen=True)
class _PostFinishAction:
    spawn_candidate: bool = False
    cleanup_cwds: tuple[str, ...] = ()
    release_refs: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class _AutoProposalStartSnapshot:
    updated_at: datetime
    project_id: int
    repo_path: str
    autonomy: str
    auto_qa_enabled: bool
    stack_depth: int
    proposal_budget: int | None


def _state_int(workflow: SystemWorkflow, key: str) -> int:
    value = workflow.state.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _state_string(workflow: SystemWorkflow, key: str) -> str:
    value = workflow.state.get(key)
    return value.strip() if isinstance(value, str) else ""


def _workflow_accepts_new_workers(workflow: SystemWorkflow) -> bool:
    return workflow.is_active and not _state_string(workflow, _AUTONOMOUS_GOAL_ERROR_STATE_KEY)


def _state_bool(workflow: SystemWorkflow, key: str) -> bool:
    return workflow.state.get(key) is True


def _run_output(run: SystemAgentRun) -> dict[str, Any]:
    return dict(run.output) if isinstance(run.output, dict) else {}


def _run_input(run: SystemAgentRun) -> dict[str, Any]:
    return dict(run.input) if isinstance(run.input, dict) else {}


def _positive_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def _workflow_goal(workflow: SystemWorkflow) -> AutonomousGoal | None:
    return (
        AutonomousGoal.objects.select_related("project")
        .filter(
            pk=_state_int(workflow, "autonomous_goal_id"),
            deleted_at__isnull=True,
        )
        .first()
    )


def _candidate_run_for_context(context: Any) -> SystemAgentRun:
    if getattr(context, "purpose", "") != CodexInstance.PURPOSE_SYSTEM_AGENT:
        raise ValueError("autonomous-goal tools require a hidden system session")
    if getattr(context, "agent_kind", "") != system_agents.AUTONOMOUS_GOAL_AGENT_KIND:
        raise ValueError("candidate tool called from the wrong agent role")
    instance_id = getattr(context, "instance_id", 0)
    if not isinstance(instance_id, int) or isinstance(instance_id, bool) or instance_id < 1:
        raise ValueError("candidate tool is not attached to a running turn")
    run = SystemAgentRun.objects.select_related("workflow", "instance").filter(instance_id=instance_id).first()
    if run is None or run.agent_kind != system_agents.AUTONOMOUS_GOAL_AGENT_KIND:
        raise ValueError("candidate run no longer exists")
    workflow = run.workflow
    if (
        workflow.kind != system_agents.AUTONOMOUS_GOAL_AGENT_KIND
        or not _workflow_accepts_new_workers(workflow)
        or run.status not in (SystemAgentRun.STATUS_STARTING, SystemAgentRun.STATUS_RUNNING)
        or run.thread_id != getattr(context, "thread_id", "")
        or workflow.pk != getattr(context, "workflow_id", None)
    ):
        raise ValueError("autonomous goal run is no longer active")
    return run


def _review_run_for_context(context: Any) -> SystemAgentRun:
    if getattr(context, "purpose", "") != CodexInstance.PURPOSE_SYSTEM_AGENT:
        raise ValueError("review tools require a hidden system session")
    if getattr(context, "agent_kind", "") != _AUTONOMOUS_GOAL_REVIEW_AGENT_KIND:
        raise ValueError("review tool called from the wrong agent role")
    instance_id = getattr(context, "instance_id", 0)
    run = SystemAgentRun.objects.select_related("workflow", "instance").filter(instance_id=instance_id).first()
    if (
        run is None
        or run.agent_kind != _AUTONOMOUS_GOAL_REVIEW_AGENT_KIND
        or run.thread_id != getattr(context, "thread_id", "")
        or run.workflow_id != getattr(context, "workflow_id", None)
        or not _workflow_accepts_new_workers(run.workflow)
        or run.status not in (SystemAgentRun.STATUS_STARTING, SystemAgentRun.STATUS_RUNNING)
    ):
        raise ValueError("review request is no longer active")
    return run


def candidate_goal_data(context: Any) -> dict[str, object]:
    candidate_run = _candidate_run_for_context(context)
    workflow = candidate_run.workflow
    autonomous_goal = _workflow_goal(workflow)
    if autonomous_goal is None:
        raise ValueError("autonomous goal no longer exists")
    reviews = list(_candidate_reviews(candidate_run))
    last_feedback = ""
    if reviews:
        feedback = _run_output(reviews[-1]).get("feedback")
        last_feedback = feedback.strip() if isinstance(feedback, str) else ""
    return {
        "title": autonomous_goal.title,
        "goal": autonomous_goal.goal,
        "ambition": autonomous_goal.ambition,
        "autonomy": autonomous_goal.autonomy,
        "confidence_threshold": autonomous_goal.confidence_threshold,
        "stack_iteration": _candidate_iteration(candidate_run),
        "stack_depth": _workflow_stack_depth(workflow, autonomous_goal),
        "proposal_budget": _workflow_budget(workflow),
        "proposal_budget_tokens_used": _workflow_tokens_used(workflow),
        "reviews_used": len(reviews),
        "reviews_remaining": max(_MAX_REVIEWS - len(reviews), 0),
        "last_review_feedback": last_feedback,
    }


def candidate_goal_sessions(context: Any) -> list[dict[str, object]]:
    candidate_run = _candidate_run_for_context(context)
    workflow = candidate_run.workflow
    autonomous_goal = _workflow_goal(workflow)
    if autonomous_goal is None:
        raise ValueError("autonomous goal no longer exists")

    rows: dict[int, dict[str, object]] = {}
    proposals = (
        ProposedSession.objects.filter(autonomous_goal=autonomous_goal)
        .select_related("candidate_session", "accepted_session")
        .order_by("created_at", "id")
    )
    for proposal in proposals:
        if proposal.candidate_session is not None:
            _add_goal_session_row(
                rows,
                proposal.candidate_session,
                kind="candidate",
                outcome=proposal.outcome_status or "pending",
                proposal_id=proposal.pk,
            )
        if proposal.accepted_session is not None:
            _add_goal_session_row(
                rows,
                proposal.accepted_session,
                kind="accepted_work",
                outcome="accepted",
                proposal_id=proposal.pk,
            )

    workflow_ids = SystemWorkflow.objects.filter(
        kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        state__autonomous_goal_id=autonomous_goal.pk,
    ).values_list("id", flat=True)
    thread_ids = (
        SystemAgentRun.objects.filter(
            workflow_id__in=workflow_ids,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        .exclude(thread_id=context.thread_id)
        .values_list("thread_id", flat=True)
        .distinct()
    )
    for metadata in SessionMetadata.objects.filter(thread_id__in=thread_ids).order_by("created_at", "id"):
        _add_goal_session_row(
            rows,
            metadata,
            kind="candidate",
            outcome="completed",
            proposal_id=None,
        )
    result = list(rows.values())
    result.sort(key=lambda row: (str(row["created_at"]), str(row["session_id"])))
    return result


def _add_goal_session_row(
    rows: dict[int, dict[str, object]],
    metadata: SessionMetadata,
    *,
    kind: str,
    outcome: str,
    proposal_id: int | None,
) -> None:
    raw_path = metadata.codex_path.strip()
    path = str(Path(raw_path).expanduser()) if raw_path else ""
    row: dict[str, object] = {
        "session_id": metadata.thread_id,
        "title": (
            metadata.codex_name
            or metadata.codex_display_title
            or (metadata.codex_preview.splitlines()[0][:200] if metadata.codex_preview else metadata.thread_id)
        ),
        "kind": kind,
        "outcome": outcome,
        "proposal_id": proposal_id,
        "created_at": metadata.created_at.isoformat(),
        "session_file": path,
        "session_file_available": bool(path and Path(path).is_file()),
    }
    if metadata.pk not in rows or kind == "accepted_work":
        rows[metadata.pk] = row


def candidate_request_review(arguments: dict[str, Any], context: Any) -> dict[str, object]:
    candidate = _candidate_from_tool_arguments(arguments)
    candidate_run = _candidate_run_for_context(context)
    existing_reviews = list(_candidate_reviews(candidate_run))
    if len(existing_reviews) >= _MAX_REVIEWS:
        raise ValueError("the candidate has already used both reviews")
    if any(_run_output(run).get("verdict") == "approve" for run in existing_reviews):
        raise ValueError("the candidate is already approved; call hitch.propose_session")
    if _run_output(candidate_run).get("terminal"):
        raise ValueError("the candidate has already finished")

    workflow = candidate_run.workflow
    autonomous_goal = _workflow_goal(workflow)
    if autonomous_goal is None:
        raise ValueError("autonomous goal no longer exists")
    review = _spawn_review_run(
        workflow,
        autonomous_goal,
        candidate_run=candidate_run,
        candidate=candidate,
        attempt=len(existing_reviews) + 1,
        candidate_cwd=context.cwd,
    )
    approved = False
    cancellation_requested = False
    try:
        while True:
            cancellation_requested = cancellation_requested or context.cancel_requested()
            review.refresh_from_db()
            review.instance.refresh_from_db()
            if review.status in (
                SystemAgentRun.STATUS_COMPLETED,
                SystemAgentRun.STATUS_FAILED,
            ):
                break
            if review.instance.status in (
                CodexInstance.STATUS_COMPLETED,
                CodexInstance.STATUS_FAILED,
            ):
                system_agents.on_codex_instance_finished(review.instance)
                continue
            if cancellation_requested:
                interrupted = codex_pool.interrupt_instance(
                    review.instance_id,
                    expected_thread_id=review.thread_id,
                    force=True,
                    error="candidate stopped while waiting for review",
                )
                if interrupted is None:
                    time.sleep(0.25)
                continue
            time.sleep(0.25)
        if cancellation_requested:
            raise ValueError("candidate stopped while waiting for review")
        output = _run_output(review)
        approved = review.status == SystemAgentRun.STATUS_COMPLETED and output.get("verdict") == "approve"
        feedback = output.get("feedback")
        if review.status == SystemAgentRun.STATUS_FAILED:
            feedback = review.error or "the isolated reviewer failed"
        attempts = SystemAgentRun.objects.filter(
            workflow=workflow,
            agent_kind=_AUTONOMOUS_GOAL_REVIEW_AGENT_KIND,
            input__candidate_instance_id=candidate_run.instance_id,
        ).count()
        return {
            "verdict": "approve" if approved else "deny",
            "confidence": str(output.get("confidence") or AutonomousGoal.CONFIDENCE_MEDIUM),
            "feedback": str(feedback or ""),
            "reviews_used": attempts,
            "reviews_remaining": max(_MAX_REVIEWS - attempts, 0),
        }
    finally:
        review_input = _run_input(review)
        if review_input.get("managed_review_cwd") is True:
            _cleanup_autonomous_goal_candidate_cwd(str(review_input.get("cwd") or ""))
        if not approved:
            _release_autonomous_goal_snapshot_ref(
                autonomous_goal.project.repo_path,
                str(review_input.get("snapshot_ref") or ""),
            )


def candidate_submit_proposal(arguments: dict[str, Any], context: Any) -> dict[str, object]:
    if arguments:
        raise ValueError("hitch.propose_session does not accept arguments in an autonomous goal")
    candidate_run = _candidate_run_for_context(context)
    reviews = list(_candidate_reviews(candidate_run))
    approved_review = next(
        (
            run
            for run in reversed(reviews)
            if run.status == SystemAgentRun.STATUS_COMPLETED and _run_output(run).get("verdict") == "approve"
        ),
        None,
    )
    if approved_review is None:
        raise ValueError("the current candidate has not been approved by hitch.review")
    review_input = _run_input(approved_review)
    candidate = review_input.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError("the approved candidate data is unavailable")
    with transaction.atomic():
        locked = SystemAgentRun.objects.select_related("workflow").select_for_update().get(pk=candidate_run.pk)
        if not _workflow_accepts_new_workers(locked.workflow) or locked.status not in (
            SystemAgentRun.STATUS_STARTING,
            SystemAgentRun.STATUS_RUNNING,
        ):
            raise ValueError("autonomous goal run is no longer active")
        output = _run_output(locked)
        terminal = output.get("terminal")
        if terminal and terminal != _AUTONOMOUS_GOAL_PROPOSAL_TERMINAL:
            raise ValueError("the candidate has already finished without a proposal")
        output.update(
            {
                "terminal": _AUTONOMOUS_GOAL_PROPOSAL_TERMINAL,
                "candidate": candidate,
                "review": _run_output(approved_review),
                "snapshot_sha": str(review_input.get("snapshot_sha") or ""),
                "snapshot_ref": str(review_input.get("snapshot_ref") or ""),
                "review_thread_id": approved_review.thread_id,
            }
        )
        locked.output = output
        locked.save(update_fields=["output", "updated_at"])
    return {"status": "proposal_ready", "title": str(candidate.get("title") or "")}


def candidate_decline_proposal(arguments: dict[str, Any], context: Any) -> dict[str, object]:
    reason = _required_tool_string(arguments, "reason")
    unexpected = set(arguments) - {"reason"}
    if unexpected:
        raise ValueError(f"unexpected no_proposal fields: {', '.join(sorted(unexpected))}")
    candidate_run = _candidate_run_for_context(context)
    with transaction.atomic():
        locked = SystemAgentRun.objects.select_related("workflow").select_for_update().get(pk=candidate_run.pk)
        if not _workflow_accepts_new_workers(locked.workflow) or locked.status not in (
            SystemAgentRun.STATUS_STARTING,
            SystemAgentRun.STATUS_RUNNING,
        ):
            raise ValueError("autonomous goal run is no longer active")
        output = _run_output(locked)
        if output.get("terminal"):
            raise ValueError("the candidate has already finished")
        output.update(
            {
                "terminal": _AUTONOMOUS_GOAL_NO_PROPOSAL_TERMINAL,
                "reason": reason,
            }
        )
        locked.output = output
        locked.save(update_fields=["output", "updated_at"])
    return {"status": "no_proposal", "reason": reason}


def reviewer_record_verdict(arguments: dict[str, Any], context: Any, *, approved: bool) -> dict[str, object]:
    confidence = _required_tool_string(arguments, "confidence")
    valid_confidences = {value for value, _label in AutonomousGoal.CONFIDENCE_CHOICES}
    if confidence not in valid_confidences:
        raise ValueError("confidence must be medium, high, or very_high")
    feedback = _optional_tool_string(arguments, "feedback")
    unexpected = set(arguments) - {"confidence", "feedback"}
    if unexpected:
        raise ValueError(f"unexpected verdict fields: {', '.join(sorted(unexpected))}")
    review = _review_run_for_context(context)
    autonomous_goal = _workflow_goal(review.workflow)
    if autonomous_goal is None:
        raise ValueError("autonomous goal no longer exists")
    if approved and not _confidence_meets_threshold(confidence, autonomous_goal.confidence_threshold):
        raise ValueError("approval confidence is below this goal's confidence threshold")
    with transaction.atomic():
        locked = SystemAgentRun.objects.select_related("workflow").select_for_update().get(pk=review.pk)
        if not _workflow_accepts_new_workers(locked.workflow) or locked.status not in (
            SystemAgentRun.STATUS_STARTING,
            SystemAgentRun.STATUS_RUNNING,
        ):
            raise ValueError("review request is no longer active")
        if _run_output(locked).get("verdict"):
            raise ValueError("this reviewer has already recorded a verdict")
        output = {
            "verdict": "approve" if approved else "deny",
            "confidence": confidence,
            "feedback": feedback,
        }
        locked.output = output
        locked.save(update_fields=["output", "updated_at"])
    return cast(dict[str, object], output)


def _confidence_meets_threshold(confidence: str, threshold: str) -> bool:
    rank = {
        AutonomousGoal.CONFIDENCE_MEDIUM: 1,
        AutonomousGoal.CONFIDENCE_HIGH: 2,
        AutonomousGoal.CONFIDENCE_VERY_HIGH: 3,
    }
    return rank.get(confidence, 0) >= rank.get(threshold, 0)


def _candidate_reviews(candidate_run: SystemAgentRun) -> Any:
    return SystemAgentRun.objects.filter(
        workflow=candidate_run.workflow,
        agent_kind=_AUTONOMOUS_GOAL_REVIEW_AGENT_KIND,
        input__candidate_instance_id=candidate_run.instance_id,
    ).order_by("created_at", "id")


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
    candidate: dict[str, Any] = {key: _required_tool_string(arguments, key) for key in required - {"relevant_files"}}
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


def _spawn_review_run(
    workflow: SystemWorkflow,
    autonomous_goal: AutonomousGoal,
    *,
    candidate_run: SystemAgentRun,
    candidate: dict[str, Any],
    attempt: int,
    candidate_cwd: str,
) -> SystemAgentRun:
    snapshot_sha = ""
    snapshot_ref = ""
    review_cwd = autonomous_goal.project.repo_path
    managed_review_worktree: ManagedWorktree | None = None
    review_run: SystemAgentRun | None = None
    try:
        if _candidate_allows_code_changes(candidate_run, autonomous_goal):
            snapshot_ref = f"refs/hitch/autonomous-goals/{workflow.pk}/{uuid.uuid4().hex}"
            snapshot_sha = snapshot_worktree_to_commit(
                candidate_cwd,
                message=f"Snapshot AG review {attempt}",
                retain_ref=snapshot_ref,
            )
            managed_review_worktree = create_worktree_for_session(
                autonomous_goal.project.repo_path,
                base_ref=snapshot_sha,
            )
            review_cwd = str(managed_review_worktree.path)
        prompt = _review_prompt(
            autonomous_goal,
            candidate,
            attempt=attempt,
            snapshot_sha=snapshot_sha,
        )
        thread_id, thread_path = codex_pool.create_session_thread_with_path(
            cwd=review_cwd,
            name=prompt,
            web_search_mode=_workflow_web_search_mode(workflow),
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=_AUTONOMOUS_GOAL_REVIEW_AGENT_KIND,
            thread_source=ThreadSource.subagent,
        )
        session_index.upsert_local_session(
            thread_id=thread_id,
            cwd=review_cwd,
            project=autonomous_goal.project,
            preview=prompt,
            codex_path=thread_path,
            is_hidden_system_session=True,
        )

        def bind_review_run(instance: CodexInstance) -> None:
            nonlocal review_run
            locked_workflow = SystemWorkflow.objects.select_for_update().get(pk=workflow.pk)
            if not _workflow_accepts_new_workers(locked_workflow):
                raise ValueError("autonomous goal run is no longer active")
            review_run = SystemAgentRun.objects.create(
                workflow=locked_workflow,
                agent_kind=_AUTONOMOUS_GOAL_REVIEW_AGENT_KIND,
                thread_id=thread_id,
                instance=instance,
                status=SystemAgentRun.STATUS_RUNNING,
                input={
                    "autonomous_goal_id": autonomous_goal.pk,
                    "candidate_instance_id": candidate_run.instance_id,
                    "candidate": candidate,
                    "attempt": attempt,
                    "snapshot_sha": snapshot_sha,
                    "snapshot_ref": snapshot_ref,
                    "cwd": review_cwd,
                    "managed_review_cwd": managed_review_worktree is not None,
                },
            )

        codex_pool.spawn_turn(
            thread_id=thread_id,
            cwd=review_cwd,
            prompt=prompt,
            approval_mode=system_agents.SYSTEM_AGENT_APPROVAL_MODE,
            sandbox_policy="readOnly",
            web_search_mode=_workflow_web_search_mode(workflow),
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=_AUTONOMOUS_GOAL_REVIEW_AGENT_KIND,
            display_author=system_agents.AUTONOMOUS_GOAL_REVIEWER_DISPLAY_AUTHOR,
            before_worker_launch=bind_review_run,
        )
        if review_run is None:
            raise RuntimeError("review worker launched without a durable run binding")
        return review_run
    except Exception as exc:
        if review_run is not None:
            review_run.status = SystemAgentRun.STATUS_FAILED
            review_run.error = f"failed to launch autonomous goal reviewer: {exc!r}"
            review_run.save(update_fields=["status", "error", "updated_at"])
        _cleanup_new_worktree(managed_review_worktree)
        _release_autonomous_goal_snapshot_ref(autonomous_goal.project.repo_path, snapshot_ref)
        raise


def _review_prompt(
    autonomous_goal: AutonomousGoal,
    candidate: dict[str, Any],
    *,
    attempt: int,
    snapshot_sha: str,
) -> str:
    return (
        "You are Hitch's isolated autonomous-goal reviewer. Evaluate the candidate "
        "and the checkout read-only. Call hitch.approve if it makes meaningful, "
        "concrete progress and meets the confidence threshold; otherwise call "
        "hitch.deny with actionable feedback. You must call exactly one tool; "
        "final prose is ignored.\n\n"
        f"Goal title: {autonomous_goal.title}\n"
        f"Goal: {autonomous_goal.goal}\n"
        f"Ambition: {autonomous_goal.ambition}\n"
        f"Confidence threshold: {autonomous_goal.confidence_threshold}\n"
        f"Review: {attempt} of {_MAX_REVIEWS}\n"
        f"Candidate snapshot: {snapshot_sha or '(no-code proposal)'}\n\n"
        f"Candidate:\n{json.dumps(candidate, indent=2, sort_keys=True)}"
    )


def maybe_start_auto_proposal_workflows(*, project: Project | None = None) -> int:
    goals = AutonomousGoal.objects.select_related("project").filter(
        auto_proposal_enabled=True,
        deleted_at__isnull=True,
    )
    if project is not None:
        goals = goals.filter(project=project)
    if goals.exists() and _auto_proposals_paused_by_usage_quota_throttled():
        return 0
    for goal_id in goals.order_by("created_at", "id").values_list("id", flat=True):
        try:
            if _maybe_start_auto_proposal_workflow(goal_id):
                return 1
        except (AutonomousGoal.DoesNotExist, Project.DoesNotExist):
            continue
        except Exception:
            logger.exception("failed to start auto-proposal run for goal %s", goal_id)
    return 0


def _auto_proposals_paused_by_usage_quota_throttled() -> bool:
    return _auto_proposal_quota_status_throttled() != "available"


def _auto_proposal_quota_status_throttled() -> AutoProposalQuotaStatus:
    global _quota_cache_status, _quota_cache_checked_at
    with _quota_cache_lock:
        now = timezone.now()
        if _quota_cache_checked_at is not None and now - _quota_cache_checked_at < _AUTO_PROPOSAL_QUOTA_CACHE_TTL:
            return _quota_cache_status
        _quota_cache_status = _auto_proposal_quota_status()
        _quota_cache_checked_at = now
        return _quota_cache_status


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
        return "unavailable"
    except Exception:
        logger.exception("failed to verify account rate limits for auto-proposal")
        return "unavailable"


def _auto_proposal_quota_status_from_rate_limits(rate_limits: Any, *, now: datetime) -> AutoProposalQuotaStatus:
    statuses = tuple(
        _rate_limit_window_auto_proposal_quota_status(window, now=now) if window is not None else None
        for window in (rate_limits.primary, rate_limits.secondary)
    )
    if None in statuses:
        return "unavailable"
    return "low" if "low" in statuses else "available"


def _rate_limit_window_auto_proposal_quota_status(window: Any, *, now: datetime) -> Literal["available", "low"] | None:
    try:
        used = float(window.used_percent)
        reset_timestamp = float(window.resets_at)
        duration_seconds = float(window.window_duration_mins) * 60
    except (AttributeError, TypeError, ValueError):
        return None
    if duration_seconds <= 0:
        return None
    if timezone.is_naive(now):
        now = now.replace(tzinfo=UTC)
    remaining_percent = 100 - max(0.0, min(100.0, used))
    reset_at = datetime.fromtimestamp(reset_timestamp, tz=UTC)
    seconds_until_reset = max(0.0, min((reset_at - now).total_seconds(), duration_seconds))
    expected_remaining = (seconds_until_reset / duration_seconds) * 100
    threshold = expected_remaining * _AUTO_PROPOSAL_QUOTA_THRESHOLD_FRACTION
    return "low" if remaining_percent < threshold else "available"


def _auto_proposal_start_snapshot(goal: AutonomousGoal) -> _AutoProposalStartSnapshot:
    return _AutoProposalStartSnapshot(
        updated_at=goal.updated_at,
        project_id=goal.project_id,
        repo_path=goal.project.repo_path,
        autonomy=goal.autonomy,
        auto_qa_enabled=goal.auto_qa_enabled,
        stack_depth=goal.effective_stacked_diff_depth,
        proposal_budget=goal.proposal_budget,
    )


def _maybe_start_auto_proposal_workflow(autonomous_goal_id: int) -> bool:
    goal = AutonomousGoal.objects.select_related("project").get(pk=autonomous_goal_id)
    if not goal.auto_proposal_enabled:
        return False
    snapshot = _auto_proposal_start_snapshot(goal)
    branch_sha = default_branch_commit_hash(snapshot.repo_path)
    if not branch_sha:
        return False
    try:
        with transaction.atomic():
            _lock_autonomous_goal_queue()
            goal = (
                AutonomousGoal.objects.select_related("project")
                .select_for_update()
                .get(pk=autonomous_goal_id, deleted_at__isnull=True)
            )
            Project.objects.select_for_update().get(pk=goal.project_id)
            if (
                not goal.auto_proposal_enabled
                or _auto_proposal_start_snapshot(goal) != snapshot
                or not _autonomous_goal_db_allows_start(goal, branch_sha)
            ):
                return False
            workflow, created = _create_autonomous_goal_workflow_record(
                autonomous_goal=goal,
                auto_proposal=True,
                default_branch_sha=branch_sha,
                use_worktrees=True,
            )
    except OperationalError as exc:
        if not db.is_database_locked_error(exc):
            raise
        logger.warning(
            "skipping auto-proposal start for goal %s because database is locked",
            autonomous_goal_id,
        )
        return False
    if created:
        _spawn_autonomous_goal_candidate_or_finish(workflow, goal)
    return workflow.is_active


def _autonomous_goal_db_allows_start(goal: AutonomousGoal, branch_sha: str) -> bool:
    if autonomous_goal_queue_busy():
        return False
    if _autonomous_goal_pending_proposal_blocks_start(goal):
        return False
    if _autonomous_goal_unresolved_failure_notice_exists(goal):
        return False
    if _autonomous_goal_accepted_session_blocks_start(goal):
        return False
    if _autonomous_goal_running_workflow_exists(goal):
        return False
    previous_sha = goal.auto_proposal_last_no_proposal_sha.strip()
    return not previous_sha or previous_sha != branch_sha


def _autonomous_goal_auto_proposal_base_sha(goal: AutonomousGoal) -> str | None:
    return default_branch_commit_hash(goal.project.repo_path)


def _lock_autonomous_goal_queue() -> None:
    RefreshThrottle.objects.get_or_create(
        key=_AUTO_PROPOSAL_QUEUE_LOCK_KEY,
        defaults={"attempted_at": timezone.now()},
    )
    RefreshThrottle.objects.select_for_update().get(key=_AUTO_PROPOSAL_QUEUE_LOCK_KEY)


def autonomous_goal_queue_busy() -> bool:
    return SystemWorkflow.objects.filter(
        kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        status=SystemWorkflow.STATUS_RUNNING,
    ).exists()


def _autonomous_goal_running_workflow_exists(goal: AutonomousGoal) -> bool:
    return SystemWorkflow.objects.filter(
        kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        main_thread_id=_autonomous_goal_main_thread_id(goal.pk),
        status=SystemWorkflow.STATUS_RUNNING,
    ).exists()


def start_autonomous_goal_workflow_if_queue_idle(
    *, autonomous_goal: AutonomousGoal, use_worktrees: bool = False
) -> SystemWorkflow | None:
    with transaction.atomic():
        _lock_autonomous_goal_queue()
        goal = (
            AutonomousGoal.objects.select_related("project")
            .select_for_update()
            .get(pk=autonomous_goal.pk, deleted_at__isnull=True)
        )
        Project.objects.select_for_update().get(pk=goal.project_id)
        if autonomous_goal_queue_busy():
            return None
        workflow, created = _create_autonomous_goal_workflow_record(
            autonomous_goal=goal,
            auto_proposal=False,
            default_branch_sha=None,
            use_worktrees=use_worktrees,
        )
    if created:
        _spawn_autonomous_goal_candidate_or_finish(workflow, goal)
    return workflow


def _create_autonomous_goal_workflow_record(
    *,
    autonomous_goal: AutonomousGoal,
    auto_proposal: bool,
    default_branch_sha: str | None,
    use_worktrees: bool,
) -> tuple[SystemWorkflow, bool]:
    state: dict[str, Any] = {
        "autonomous_goal_id": autonomous_goal.pk,
        "auto_proposal": auto_proposal,
        "autonomous_goal_updated_at": autonomous_goal.updated_at.isoformat(),
        "web_search_mode": autonomous_goal.web_search_mode,
        _AUTONOMOUS_GOAL_USE_WORKTREES_STATE_KEY: use_worktrees,
        _AUTONOMOUS_GOAL_STACKED_DEPTH_STATE_KEY: autonomous_goal.effective_stacked_diff_depth,
    }
    if autonomous_goal.proposal_budget is not None:
        state[_AUTONOMOUS_GOAL_PROPOSAL_BUDGET_STATE_KEY] = autonomous_goal.proposal_budget
    if auto_proposal:
        state["default_branch_sha"] = (
            default_branch_sha
            or _autonomous_goal_auto_proposal_base_sha(autonomous_goal)
            or _AUTO_PROPOSAL_UNKNOWN_DEFAULT_BRANCH_SHA
        )
    try:
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=_autonomous_goal_main_thread_id(autonomous_goal.pk),
            cwd=autonomous_goal.project.repo_path,
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_RUNNING,
            state=state,
        )
    except IntegrityError:
        existing = SystemWorkflow.objects.filter(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=_autonomous_goal_main_thread_id(autonomous_goal.pk),
            status=SystemWorkflow.STATUS_RUNNING,
        ).first()
        if existing is None:
            raise
        return existing, False
    return workflow, True


def _spawn_autonomous_goal_candidate_or_finish(workflow: SystemWorkflow, autonomous_goal: AutonomousGoal) -> None:
    workflow.refresh_from_db()
    if not workflow.is_active:
        return
    try:
        run = _spawn_autonomous_goal_candidate_run(workflow, autonomous_goal)
    except Exception as exc:
        logger.exception("failed to start autonomous goal candidate")
        _finish_spawn_failure(workflow, autonomous_goal, exc)
        return
    workflow.refresh_from_db()
    if workflow.is_active:
        return
    codex_pool.interrupt_instance(
        run.instance_id,
        expected_thread_id=run.thread_id,
        force=True,
        error=_state_string(workflow, _AUTONOMOUS_GOAL_ERROR_STATE_KEY)
        or "autonomous goal stopped before candidate start",
    )


def _spawn_autonomous_goal_candidate_run(workflow: SystemWorkflow, autonomous_goal: AutonomousGoal) -> SystemAgentRun:
    iteration = _next_candidate_iteration(workflow)
    cwd, managed_worktree = _prepare_autonomous_goal_candidate_cwd(workflow, autonomous_goal)
    prompt = _candidate_prompt(workflow, autonomous_goal, iteration=iteration, cwd=cwd)
    sandbox = (
        system_agents.AUTONOMOUS_GOAL_IMPLEMENTATION_SANDBOX_POLICY
        if cwd != autonomous_goal.project.repo_path
        else "readOnly"
    )
    candidate_run: SystemAgentRun | None = None
    try:
        thread_id, thread_path = codex_pool.create_session_thread_with_path(
            cwd=cwd,
            name=prompt,
            web_search_mode=_workflow_web_search_mode(workflow),
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_source=ThreadSource.subagent,
        )
        session_index.upsert_local_session(
            thread_id=thread_id,
            cwd=cwd,
            project=autonomous_goal.project,
            preview=prompt,
            codex_path=thread_path,
            is_hidden_system_session=True,
        )

        def bind_candidate_run(instance: CodexInstance) -> None:
            nonlocal candidate_run
            locked_workflow = SystemWorkflow.objects.select_for_update().get(pk=workflow.pk)
            if not _workflow_accepts_new_workers(locked_workflow):
                raise ValueError("autonomous goal run is no longer active")
            candidate_run = SystemAgentRun.objects.create(
                workflow=locked_workflow,
                agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
                thread_id=thread_id,
                instance=instance,
                status=SystemAgentRun.STATUS_RUNNING,
                input={
                    "autonomous_goal_id": autonomous_goal.pk,
                    "cwd": cwd,
                    "managed_candidate_cwd": managed_worktree is not None,
                    "stack_iteration": iteration,
                },
            )

        codex_pool.spawn_turn(
            thread_id=thread_id,
            cwd=cwd,
            prompt=prompt,
            approval_mode=system_agents.SYSTEM_AGENT_APPROVAL_MODE,
            sandbox_policy=sandbox,
            web_search_mode=_workflow_web_search_mode(workflow),
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            display_author=system_agents.AUTONOMOUS_GOAL_DISPLAY_AUTHOR,
            before_worker_launch=bind_candidate_run,
        )
        if candidate_run is None:
            raise RuntimeError("candidate worker launched without a durable run binding")
        return candidate_run
    except Exception as exc:
        if candidate_run is not None:
            candidate_run.status = SystemAgentRun.STATUS_FAILED
            candidate_run.error = f"failed to launch autonomous goal candidate: {exc!r}"
            candidate_run.save(update_fields=["status", "error", "updated_at"])
        _cleanup_new_worktree(managed_worktree)
        raise


def _prepare_autonomous_goal_candidate_cwd(
    workflow: SystemWorkflow, autonomous_goal: AutonomousGoal
) -> tuple[str, ManagedWorktree | None]:
    if not _state_bool(workflow, _AUTONOMOUS_GOAL_USE_WORKTREES_STATE_KEY):
        return autonomous_goal.project.repo_path, None
    checkpoint = _latest_checkpoint(workflow)
    base_ref = ""
    if checkpoint is not None:
        base_ref = str(_run_output(checkpoint).get("snapshot_sha") or "")
    if not base_ref:
        recorded = _state_string(workflow, "default_branch_sha")
        if recorded and recorded != _AUTO_PROPOSAL_UNKNOWN_DEFAULT_BRANCH_SHA:
            base_ref = recorded
    if not base_ref:
        base_ref = default_branch_commit_hash(autonomous_goal.project.repo_path) or ""
    if not base_ref:
        raise WorktreeCreationError("project default branch is unavailable")
    worktree = create_worktree_for_session(
        autonomous_goal.project.repo_path,
        base_ref=base_ref,
    )
    return str(worktree.path), worktree


def _candidate_prompt(
    workflow: SystemWorkflow,
    autonomous_goal: AutonomousGoal,
    *,
    iteration: int,
    cwd: str,
) -> str:
    previous = _latest_checkpoint(workflow)
    previous_title = ""
    if previous is not None:
        candidate = _run_output(previous).get("candidate")
        if isinstance(candidate, dict):
            previous_title = str(candidate.get("title") or "").strip()
    return (
        "You are Hitch's autonomous goal agent. Own the investigation and decide "
        "what constitutes useful progress. Start with hitch.get_goal and inspect "
        "prior work with hitch.list_goal_sessions when useful. Work directly in "
        "the provided checkout. When you have a concrete candidate, call "
        "hitch.review. Address a denial and review once more if worthwhile. After "
        "approval, call hitch.propose_session with no arguments. If nothing is "
        "worth proposing, call hitch.no_proposal. A final prose response does not "
        "complete the run.\n\n"
        f"Goal title: {autonomous_goal.title}\n"
        f"Goal: {autonomous_goal.goal}\n"
        f"Ambition: {autonomous_goal.ambition.replace('_', ' ')}\n"
        f"Autonomy: {autonomous_goal.autonomy.replace('_', ' ')}\n"
        f"Repository cwd: {cwd}\n"
        f"Stack round: {iteration} of {_workflow_stack_depth(workflow, autonomous_goal)}\n"
        f"Prior approved checkpoint: {previous_title or '(none)'}"
    )


def _workflow_web_search_mode(workflow: SystemWorkflow) -> str | None:
    return _state_string(workflow, "web_search_mode") or None


def _candidate_iteration(run: SystemAgentRun) -> int:
    return max(_positive_int(_run_input(run).get("stack_iteration")), 1)


def _next_candidate_iteration(workflow: SystemWorkflow) -> int:
    checkpoint = _latest_checkpoint(workflow)
    return _candidate_iteration(checkpoint) + 1 if checkpoint is not None else 1


def _candidate_allows_code_changes(run: SystemAgentRun, autonomous_goal: AutonomousGoal) -> bool:
    return str(_run_input(run).get("cwd") or "") != autonomous_goal.project.repo_path


def _workflow_stack_depth(workflow: SystemWorkflow, autonomous_goal: AutonomousGoal) -> int:
    depth = _state_int(workflow, _AUTONOMOUS_GOAL_STACKED_DEPTH_STATE_KEY)
    if depth <= 0:
        depth = autonomous_goal.effective_stacked_diff_depth
    return min(max(depth, 1), AutonomousGoal.STACKED_DIFF_DEPTH_MAX)


def _workflow_budget(workflow: SystemWorkflow) -> int:
    return max(_state_int(workflow, _AUTONOMOUS_GOAL_PROPOSAL_BUDGET_STATE_KEY), 0)


def _workflow_tokens_used(workflow: SystemWorkflow) -> int:
    total = 0
    for output in workflow.agent_runs.values_list("output", flat=True):
        if isinstance(output, dict):
            total += _positive_int(output.get("tokens_used"))
    return total


def _workflow_should_continue(
    workflow: SystemWorkflow,
    autonomous_goal: AutonomousGoal,
    checkpoint: SystemAgentRun,
) -> bool:
    if not _candidate_allows_code_changes(checkpoint, autonomous_goal):
        return False
    if _candidate_iteration(checkpoint) >= _workflow_stack_depth(workflow, autonomous_goal):
        return False
    budget = _workflow_budget(workflow)
    return budget <= 0 or _workflow_tokens_used(workflow) < budget


def _latest_checkpoint(workflow: SystemWorkflow, *, before_run_id: int | None = None) -> SystemAgentRun | None:
    runs = workflow.agent_runs.filter(
        agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        status=SystemAgentRun.STATUS_COMPLETED,
        output__terminal=_AUTONOMOUS_GOAL_PROPOSAL_TERMINAL,
    )
    if before_run_id is not None:
        runs = runs.exclude(pk=before_run_id)
    return runs.order_by("-created_at", "-id").first()


def on_agent_finished(
    instance: CodexInstance,
    run: SystemAgentRun,
    workflow: SystemWorkflow,
) -> None:
    raw_output = system_agents.final_agent_text(instance.events_path)
    tokens_used = _autonomous_goal_instance_tokens_used(instance)
    action = _PostFinishAction()
    autonomous_goal: AutonomousGoal | None = None
    with transaction.atomic():
        run = SystemAgentRun.objects.select_for_update().get(pk=run.pk)
        workflow = SystemWorkflow.objects.select_for_update().get(pk=workflow.pk)
        if run.status in (SystemAgentRun.STATUS_COMPLETED, SystemAgentRun.STATUS_FAILED):
            return
        autonomous_goal = _workflow_goal(workflow)
        output = _run_output(run)
        if tokens_used is not None and tokens_used >= 0:
            output["tokens_used"] = max(_positive_int(output.get("tokens_used")), tokens_used)
        run.output = output
        run.raw_output = raw_output
        if workflow.is_active and workflow.step != system_agents.STEP_AUTONOMOUS_GOAL_RUNNING:
            error = _LEGACY_WORKFLOW_UPGRADE_ERROR
            run.status = SystemAgentRun.STATUS_FAILED
            run.error = error
            run.save(update_fields=["status", "output", "raw_output", "error", "updated_at"])
            runs = list(workflow.agent_runs.select_for_update().all())
            _fail_active_runs_locked(runs, error, exclude_run_id=run.pk)
            if autonomous_goal is not None:
                _create_notice_locked(
                    workflow,
                    autonomous_goal,
                    title=f"Autonomous goal needs retry: {autonomous_goal.title}"[:_AUTONOMOUS_GOAL_TITLE_MAX_LEN],
                    summary=error,
                    automation_status="failed",
                )
                action = _cleanup_action_for_workflow(workflow, autonomous_goal, runs=runs)
            _fail_workflow_locked(workflow, error)
        elif run.agent_kind == _AUTONOMOUS_GOAL_REVIEW_AGENT_KIND:
            verdict = output.get("verdict")
            if verdict in {"approve", "deny"}:
                run.status = SystemAgentRun.STATUS_COMPLETED
            else:
                run.status = SystemAgentRun.STATUS_FAILED
                run.error = (
                    f"autonomous goal reviewer failed: {instance.error}"
                    if instance.status == CodexInstance.STATUS_FAILED
                    else "autonomous goal reviewer stopped without calling approve or deny"
                )
            run.save(update_fields=["status", "output", "raw_output", "error", "updated_at"])
            return
        elif run.agent_kind != system_agents.AUTONOMOUS_GOAL_AGENT_KIND:
            error = f"unsupported autonomous goal agent role: {run.agent_kind}"
            run.status = SystemAgentRun.STATUS_FAILED
            run.error = error
            run.save(update_fields=["status", "output", "raw_output", "error", "updated_at"])
            runs = list(workflow.agent_runs.select_for_update().all())
            _fail_active_runs_locked(runs, error, exclude_run_id=run.pk)
            if autonomous_goal is not None:
                _create_notice_locked(
                    workflow,
                    autonomous_goal,
                    title=f"Autonomous goal failed: {autonomous_goal.title}"[:_AUTONOMOUS_GOAL_TITLE_MAX_LEN],
                    summary=error,
                    automation_status="failed",
                )
                action = _cleanup_action_for_workflow(workflow, autonomous_goal, runs=runs)
            if workflow.is_active:
                _fail_workflow_locked(workflow, error)
        elif autonomous_goal is None:
            run.status = SystemAgentRun.STATUS_FAILED
            run.error = "autonomous goal no longer exists"
            run.save(update_fields=["status", "output", "raw_output", "error", "updated_at"])
            action = _cleanup_action_for_run(run, None)
        elif not _workflow_accepts_new_workers(workflow):
            run.status = SystemAgentRun.STATUS_FAILED
            run.error = _state_string(workflow, _AUTONOMOUS_GOAL_ERROR_STATE_KEY) or "autonomous goal run stopped"
            run.save(update_fields=["status", "output", "raw_output", "error", "updated_at"])
            runs = list(workflow.agent_runs.select_for_update().all())
            latched_stop_finished = workflow.is_active and bool(
                _state_string(workflow, _AUTONOMOUS_GOAL_ERROR_STATE_KEY)
            )
            latched_stop_finished = latched_stop_finished and not any(
                candidate.status in (SystemAgentRun.STATUS_STARTING, SystemAgentRun.STATUS_RUNNING)
                for candidate in runs
            )
            if latched_stop_finished:
                workflow.status = SystemWorkflow.STATUS_BLOCKED
                workflow.step = system_agents.STEP_BLOCKED
                workflow.save(update_fields=["status", "step", "updated_at"])
                action = _cleanup_action_for_workflow(workflow, autonomous_goal, runs=runs)
            else:
                action = _cleanup_action_for_run(run, autonomous_goal)
        else:
            action = _finish_candidate_locked(
                instance,
                run,
                workflow,
                autonomous_goal,
            )
        _record_pending_run_cleanup_locked(run, action)
    _apply_finish_action_for_run(run, action)
    if action.spawn_candidate and autonomous_goal is not None:
        quota_status = (
            _auto_proposal_quota_status()
            if _state_bool(workflow, "auto_proposal")
            else "available"
        )
        if quota_status == "available":
            _spawn_autonomous_goal_candidate_or_finish(workflow, autonomous_goal)
        else:
            _publish_checkpoint_after_quota_guard(
                workflow,
                autonomous_goal,
                run,
                quota_status=quota_status,
            )


def _finish_candidate_locked(
    instance: CodexInstance,
    run: SystemAgentRun,
    workflow: SystemWorkflow,
    autonomous_goal: AutonomousGoal,
) -> _PostFinishAction:
    output = _run_output(run)
    terminal = output.get("terminal")
    run.status = (
        SystemAgentRun.STATUS_COMPLETED
        if terminal
        in {
            _AUTONOMOUS_GOAL_PROPOSAL_TERMINAL,
            _AUTONOMOUS_GOAL_NO_PROPOSAL_TERMINAL,
        }
        else SystemAgentRun.STATUS_FAILED
    )
    if run.status == SystemAgentRun.STATUS_FAILED:
        run.error = (
            f"autonomous goal candidate failed: {instance.error}"
            if instance.status == CodexInstance.STATUS_FAILED
            else "autonomous goal candidate stopped without calling propose_session or no_proposal"
        )
    run.save(update_fields=["status", "output", "raw_output", "error", "updated_at"])

    if terminal == _AUTONOMOUS_GOAL_PROPOSAL_TERMINAL:
        if _workflow_should_continue(workflow, autonomous_goal, run):
            workflow.step = system_agents.STEP_AUTONOMOUS_GOAL_RUNNING
            workflow.save(update_fields=["step", "updated_at"])
            obsolete = _obsolete_checkpoint_cleanup(workflow, keep=run, autonomous_goal=autonomous_goal)
            return _PostFinishAction(
                spawn_candidate=True,
                cleanup_cwds=obsolete.cleanup_cwds,
                release_refs=obsolete.release_refs,
            )
        return _publish_checkpoint_locked(workflow, autonomous_goal, run)

    previous = _latest_checkpoint(workflow, before_run_id=run.pk)
    if previous is not None:
        reason = "candidate_no_proposal" if terminal == _AUTONOMOUS_GOAL_NO_PROPOSAL_TERMINAL else "candidate_failed"
        action = _publish_checkpoint_locked(
            workflow,
            autonomous_goal,
            previous,
            stop_reason=reason,
        )
        current_cleanup = _cleanup_action_for_run(run, autonomous_goal)
        return _merge_actions(action, current_cleanup)

    if terminal == _AUTONOMOUS_GOAL_NO_PROPOSAL_TERMINAL:
        reason = str(output.get("reason") or "No worthwhile proposal was found.")
        _create_notice_locked(
            workflow,
            autonomous_goal,
            title=f"No proposal from {autonomous_goal.title}"[:_AUTONOMOUS_GOAL_TITLE_MAX_LEN],
            summary=reason,
            automation_status="skipped",
        )
        _record_no_proposal(autonomous_goal, workflow)
        _complete_workflow_locked(
            workflow,
            step=system_agents.STEP_AUTONOMOUS_GOAL_SKIPPED,
            reason=reason,
        )
    else:
        _create_notice_locked(
            workflow,
            autonomous_goal,
            title=f"Autonomous goal failed: {autonomous_goal.title}"[:_AUTONOMOUS_GOAL_TITLE_MAX_LEN],
            summary=f"Hitch could not finish this autonomous goal run: {run.error}",
            automation_status="failed",
        )
        _fail_workflow_locked(workflow, run.error)
    return _cleanup_action_for_run(run, autonomous_goal)


def _publish_checkpoint_locked(
    workflow: SystemWorkflow,
    autonomous_goal: AutonomousGoal,
    checkpoint: SystemAgentRun,
    *,
    stop_reason: str = "",
) -> _PostFinishAction:
    output = _run_output(checkpoint)
    candidate = output.get("candidate")
    review = output.get("review")
    if not isinstance(candidate, dict) or not isinstance(review, dict):
        error = "approved autonomous-goal candidate data is unavailable"
        _create_notice_locked(
            workflow,
            autonomous_goal,
            title=f"Autonomous goal failed: {autonomous_goal.title}"[:_AUTONOMOUS_GOAL_TITLE_MAX_LEN],
            summary=error,
            automation_status="failed",
        )
        _fail_workflow_locked(workflow, error)
        return _cleanup_action_for_run(checkpoint, autonomous_goal)

    candidate_session = SessionMetadata.objects.filter(thread_id=checkpoint.thread_id).first()
    review_thread_id = str(output.get("review_thread_id") or "")
    reviewer_session = SessionMetadata.objects.filter(thread_id=review_thread_id).first()
    metadata: dict[str, object] = {
        "autonomous_goal_autonomy": autonomous_goal.autonomy,
        "auto_pr_enabled": autonomous_goal.autonomy == AutonomousGoal.AUTONOMY_DRAFT_PR,
        "auto_qa_enabled": autonomous_goal.auto_qa_enabled,
        "stacked_diff_depth": _workflow_stack_depth(workflow, autonomous_goal),
        "stacked_diff_iteration": _candidate_iteration(checkpoint),
        "proposal_budget_tokens_used": _workflow_tokens_used(workflow),
        AUTONOMOUS_GOAL_TOOL_PROTOCOL_METADATA_KEY: True,
    }
    budget = _workflow_budget(workflow)
    if budget > 0:
        metadata["proposal_budget"] = budget
    snapshot_sha = str(output.get("snapshot_sha") or "")
    snapshot_ref = str(output.get("snapshot_ref") or "")
    if snapshot_sha:
        metadata[AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_METADATA_KEY] = snapshot_sha
    if snapshot_ref:
        metadata[AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_REF_METADATA_KEY] = snapshot_ref
    if stop_reason:
        metadata[_AUTONOMOUS_GOAL_STACKED_CONTINUATION_STOP_REASON_METADATA_KEY] = stop_reason
    ProposedSession.objects.create(
        project=autonomous_goal.project,
        autonomous_goal=autonomous_goal,
        source_workflow=workflow,
        title=str(candidate.get("title") or autonomous_goal.title)[:_AUTONOMOUS_GOAL_TITLE_MAX_LEN],
        summary=_autonomous_goal_proposal_summary(candidate, cast(dict[str, str], review)),
        prompt=_autonomous_goal_proposed_session_prompt(autonomous_goal, candidate, cast(dict[str, str], review)),
        confidence=str(review.get("confidence") or AutonomousGoal.CONFIDENCE_MEDIUM),
        relevant_files=_string_list(candidate.get("relevant_files")),
        candidate_session=candidate_session,
        judge_session=reviewer_session,
        outcome_metadata=metadata,
    )
    if autonomous_goal.auto_proposal_last_no_proposal_sha:
        AutonomousGoal.objects.filter(pk=autonomous_goal.pk).update(
            auto_proposal_last_no_proposal_sha="",
            updated_at=timezone.now(),
        )
    _complete_workflow_locked(
        workflow,
        step=system_agents.STEP_AUTONOMOUS_GOAL_PROPOSED,
        reason=stop_reason or "proposal_created",
    )
    obsolete = _obsolete_checkpoint_cleanup(
        workflow,
        keep=checkpoint,
        autonomous_goal=autonomous_goal,
    )
    return _merge_actions(obsolete, _candidate_worktree_cleanup(checkpoint))


def _publish_checkpoint_after_quota_guard(
    workflow: SystemWorkflow,
    autonomous_goal: AutonomousGoal,
    checkpoint: SystemAgentRun,
    *,
    quota_status: AutoProposalQuotaStatus,
) -> None:
    action = _PostFinishAction()
    with transaction.atomic():
        locked = SystemWorkflow.objects.select_for_update().get(pk=workflow.pk)
        checkpoint = SystemAgentRun.objects.select_for_update().get(pk=checkpoint.pk)
        if locked.is_active:
            action = _publish_checkpoint_locked(
                locked,
                autonomous_goal,
                checkpoint,
                stop_reason=f"quota_{quota_status}",
            )
            _record_pending_run_cleanup_locked(checkpoint, action)
    _apply_finish_action_for_run(checkpoint, action)


def _create_notice_locked(
    workflow: SystemWorkflow,
    autonomous_goal: AutonomousGoal,
    *,
    title: str,
    summary: str,
    automation_status: str,
) -> None:
    if ProposedSession.objects.filter(
        source_workflow=workflow,
        outcome_status=ProposedSession.OUTCOME_UNSET,
    ).exists():
        return
    latest_candidate = (
        workflow.agent_runs.filter(agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND)
        .order_by("-created_at", "-id")
        .first()
    )
    candidate_session = (
        SessionMetadata.objects.filter(thread_id=latest_candidate.thread_id).first()
        if latest_candidate is not None
        else None
    )
    ProposedSession.objects.create(
        project=autonomous_goal.project,
        autonomous_goal=autonomous_goal,
        source_workflow=workflow,
        title=title,
        inbox_kind=ProposedSession.INBOX_KIND_NOTICE,
        summary=summary,
        candidate_session=candidate_session,
        outcome_metadata={
            "autonomous_goal_autonomy": autonomous_goal.autonomy,
            "automation_status": automation_status,
            "proposal_budget_tokens_used": _workflow_tokens_used(workflow),
        },
    )


def _complete_workflow_locked(workflow: SystemWorkflow, *, step: str, reason: str) -> None:
    workflow.status = SystemWorkflow.STATUS_COMPLETED
    workflow.step = step
    workflow.state = {
        **workflow.state,
        _AUTONOMOUS_GOAL_RESULT_REASON_STATE_KEY: reason,
        _AUTONOMOUS_GOAL_PROPOSAL_BUDGET_USED_STATE_KEY: _workflow_tokens_used(workflow),
    }
    workflow.save(update_fields=["status", "step", "state", "updated_at"])


def _fail_workflow_locked(workflow: SystemWorkflow, error: str) -> None:
    workflow.status = SystemWorkflow.STATUS_FAILED
    workflow.step = system_agents.STEP_BLOCKED
    workflow.state = {
        **workflow.state,
        _AUTONOMOUS_GOAL_ERROR_STATE_KEY: error,
        _AUTONOMOUS_GOAL_PROPOSAL_BUDGET_USED_STATE_KEY: _workflow_tokens_used(workflow),
    }
    workflow.save(update_fields=["status", "step", "state", "updated_at"])


def _record_no_proposal(goal: AutonomousGoal, workflow: SystemWorkflow) -> None:
    if workflow.state.get("auto_proposal") is not True:
        return
    sha = _state_string(workflow, "default_branch_sha")
    if not sha or sha == _AUTO_PROPOSAL_UNKNOWN_DEFAULT_BRANCH_SHA:
        return
    filters: dict[str, object] = {"pk": goal.pk}
    updated_at = parse_datetime(_state_string(workflow, "autonomous_goal_updated_at"))
    if updated_at is not None:
        filters["updated_at"] = updated_at
    AutonomousGoal.objects.filter(**filters).update(
        auto_proposal_last_no_proposal_sha=sha,
        updated_at=timezone.now(),
    )


def _obsolete_checkpoint_cleanup(
    workflow: SystemWorkflow,
    *,
    keep: SystemAgentRun,
    autonomous_goal: AutonomousGoal,
) -> _PostFinishAction:
    cwds: list[str] = []
    refs: list[tuple[str, str]] = []
    for run in workflow.agent_runs.filter(
        agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        status=SystemAgentRun.STATUS_COMPLETED,
        output__terminal=_AUTONOMOUS_GOAL_PROPOSAL_TERMINAL,
    ).exclude(pk=keep.pk):
        cwd = str(_run_input(run).get("cwd") or "")
        if _run_input(run).get("managed_candidate_cwd") is True and cwd:
            cwds.append(cwd)
        ref = str(_run_output(run).get("snapshot_ref") or "")
        if ref:
            refs.append((autonomous_goal.project.repo_path, ref))
    return _PostFinishAction(
        cleanup_cwds=tuple(unique_nonempty(cwds)),
        release_refs=tuple(dict.fromkeys(refs)),
    )


def _cleanup_action_for_run(
    run: SystemAgentRun,
    autonomous_goal: AutonomousGoal | None,
) -> _PostFinishAction:
    run_input = _run_input(run)
    output = _run_output(run)
    repo_path = autonomous_goal.project.repo_path if autonomous_goal is not None else run.workflow.cwd
    cwds: list[str] = []
    refs: list[tuple[str, str]] = []
    cwd = str(run_input.get("cwd") or run.workflow.state.get("session_cwd") or "")
    if (
        run.agent_kind == system_agents.AUTONOMOUS_GOAL_AGENT_KIND
        and (run_input.get("managed_candidate_cwd") is True or (cwd and cwd != run.workflow.cwd))
        and cwd
    ):
        cwds.append(cwd)
    if run.agent_kind == _AUTONOMOUS_GOAL_REVIEW_AGENT_KIND:
        if run_input.get("managed_review_cwd") is True and cwd:
            cwds.append(cwd)
        ref = str(run_input.get("snapshot_ref") or "")
    else:
        ref = str(output.get("snapshot_ref") or "")
        for review in _candidate_reviews(run):
            review_ref = str(_run_input(review).get("snapshot_ref") or "")
            if review_ref:
                refs.append((repo_path, review_ref))
    if ref:
        refs.append((repo_path, ref))
    return _PostFinishAction(
        cleanup_cwds=tuple(unique_nonempty(cwds)),
        release_refs=tuple(dict.fromkeys(refs)),
    )


def _cleanup_action_for_workflow(
    workflow: SystemWorkflow,
    autonomous_goal: AutonomousGoal,
    *,
    runs: list[SystemAgentRun] | None = None,
) -> _PostFinishAction:
    workflow_runs = runs if runs is not None else list(workflow.agent_runs.all())
    actions = [_cleanup_action_for_run(run, autonomous_goal) for run in workflow_runs]
    legacy_cwds = [
        cwd
        for key in _LEGACY_RESOURCE_CWD_STATE_KEYS
        if (cwd := _state_string(workflow, key)) and cwd != workflow.cwd
    ]
    legacy_cwds.extend(
        cwd
        for run in workflow_runs
        if (cwd := str(_run_input(run).get("cwd") or "")) and cwd != workflow.cwd
    )
    legacy_refs = [
        (autonomous_goal.project.repo_path, ref)
        for key in _LEGACY_RESOURCE_REF_STATE_KEYS
        if (ref := _state_string(workflow, key))
    ]
    protected_cwds = set(
        ProposedSession.objects.filter(
            source_workflow=workflow,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            accepted_session__isnull=False,
        )
        .exclude(accepted_session__cwd="")
        .values_list("accepted_session__cwd", flat=True)
    )
    action = _merge_actions(
        *actions,
        _PostFinishAction(
            cleanup_cwds=tuple(unique_nonempty(legacy_cwds)),
            release_refs=tuple(dict.fromkeys(legacy_refs)),
        ),
    )
    return _PostFinishAction(
        spawn_candidate=action.spawn_candidate,
        cleanup_cwds=tuple(cwd for cwd in action.cleanup_cwds if cwd not in protected_cwds),
        release_refs=action.release_refs,
    )


def _fail_active_runs_locked(
    runs: list[SystemAgentRun],
    error: str,
    *,
    exclude_run_id: int | None = None,
) -> None:
    for run in runs:
        if run.pk == exclude_run_id or run.status not in (
            SystemAgentRun.STATUS_STARTING,
            SystemAgentRun.STATUS_RUNNING,
        ):
            continue
        run.status = SystemAgentRun.STATUS_FAILED
        run.error = error
        run.save(update_fields=["status", "error", "updated_at"])


def _candidate_worktree_cleanup(run: SystemAgentRun) -> _PostFinishAction:
    run_input = _run_input(run)
    cwd = str(run_input.get("cwd") or "")
    if run_input.get("managed_candidate_cwd") is not True or not cwd:
        return _PostFinishAction()
    return _PostFinishAction(cleanup_cwds=(cwd,))


def _merge_actions(*actions: _PostFinishAction) -> _PostFinishAction:
    return _PostFinishAction(
        spawn_candidate=any(action.spawn_candidate for action in actions),
        cleanup_cwds=tuple(unique_nonempty(cwd for action in actions for cwd in action.cleanup_cwds)),
        release_refs=tuple(dict.fromkeys(ref for action in actions for ref in action.release_refs)),
    )


def _record_pending_run_cleanup_locked(run: SystemAgentRun, action: _PostFinishAction) -> None:
    if not action.cleanup_cwds and not action.release_refs:
        return
    output = _run_output(run)
    output[_RUN_PENDING_CLEANUP_OUTPUT_KEY] = {
        "cwds": list(action.cleanup_cwds),
        "refs": [list(ref) for ref in action.release_refs],
    }
    run.output = output
    run.save(update_fields=["output", "updated_at"])


def _pending_run_cleanup(run: SystemAgentRun) -> _PostFinishAction:
    value = _run_output(run).get(_RUN_PENDING_CLEANUP_OUTPUT_KEY)
    if not isinstance(value, dict):
        return _PostFinishAction()
    raw_cwds = value.get("cwds")
    raw_refs = value.get("refs")
    cwds = tuple(item for item in raw_cwds if isinstance(item, str) and item) if isinstance(raw_cwds, list) else ()
    refs = (
        tuple(
            (item[0], item[1])
            for item in raw_refs
            if isinstance(item, list)
            and len(item) == 2
            and isinstance(item[0], str)
            and isinstance(item[1], str)
            and item[0]
            and item[1]
        )
        if isinstance(raw_refs, list)
        else ()
    )
    return _PostFinishAction(
        cleanup_cwds=tuple(unique_nonempty(cwds)),
        release_refs=tuple(dict.fromkeys(refs)),
    )


def _clear_pending_run_cleanup(run_id: int) -> None:
    with transaction.atomic():
        run = SystemAgentRun.objects.select_for_update().get(pk=run_id)
        output = _run_output(run)
        if output.pop(_RUN_PENDING_CLEANUP_OUTPUT_KEY, None) is None:
            return
        run.output = output
        run.save(update_fields=["output", "updated_at"])


def _apply_finish_action(action: _PostFinishAction) -> None:
    for cwd in action.cleanup_cwds:
        _cleanup_autonomous_goal_candidate_cwd(cwd)
    for repo_path, ref in action.release_refs:
        _release_autonomous_goal_snapshot_ref(repo_path, ref)


def _apply_finish_action_for_run(run: SystemAgentRun, action: _PostFinishAction) -> None:
    _apply_finish_action(action)
    if action.cleanup_cwds or action.release_refs:
        _clear_pending_run_cleanup(run.pk)


def _finish_spawn_failure(workflow: SystemWorkflow, autonomous_goal: AutonomousGoal, exc: Exception) -> None:
    action = _PostFinishAction()
    with transaction.atomic():
        workflow = SystemWorkflow.objects.select_for_update().get(pk=workflow.pk)
        if not workflow.is_active:
            return
        checkpoint = _latest_checkpoint(workflow)
        if checkpoint is not None:
            action = _publish_checkpoint_locked(
                workflow,
                autonomous_goal,
                checkpoint,
                stop_reason="continuation_failed",
            )
        else:
            error = f"failed to start autonomous goal agent: {exc!r}"
            _create_notice_locked(
                workflow,
                autonomous_goal,
                title=f"Autonomous goal failed: {autonomous_goal.title}"[:_AUTONOMOUS_GOAL_TITLE_MAX_LEN],
                summary=error,
                automation_status="failed",
            )
            _fail_workflow_locked(workflow, error)
    _apply_finish_action(action)


def stop_running_autonomous_goal_workflow(
    autonomous_goal_id: int,
    error: str,
    *,
    workflow_id: int | None = None,
) -> bool:
    workflow_query = SystemWorkflow.objects.filter(
        kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        main_thread_id=_autonomous_goal_main_thread_id(autonomous_goal_id),
        status=SystemWorkflow.STATUS_RUNNING,
    )
    if workflow_id is not None:
        workflow_query = workflow_query.filter(pk=workflow_id)
    workflows = list(workflow_query.order_by("created_at", "id"))
    for workflow in workflows:
        with transaction.atomic():
            locked = SystemWorkflow.objects.select_for_update().get(pk=workflow.pk)
            if not locked.is_active:
                continue
            locked.state = {**locked.state, _AUTONOMOUS_GOAL_ERROR_STATE_KEY: error}
            locked.save(update_fields=["state", "updated_at"])

        active_runs = list(
            workflow.agent_runs.filter(
                status__in=(SystemAgentRun.STATUS_STARTING, SystemAgentRun.STATUS_RUNNING)
            )
            .select_related("instance")
            .order_by("created_at", "id")
        )
        for run in active_runs:
            if (
                run.instance.is_active
                and codex_pool.interrupt_instance(
                    run.instance_id,
                    expected_thread_id=run.thread_id,
                    force=True,
                    error=error,
                )
                is None
            ):
                return False
        goal = _workflow_goal(workflow)
        action = _PostFinishAction()
        with transaction.atomic():
            locked = SystemWorkflow.objects.select_for_update().get(pk=workflow.pk)
            if not locked.is_active:
                continue
            runs = list(locked.agent_runs.select_for_update().all())
            _fail_active_runs_locked(runs, error)
            locked.status = SystemWorkflow.STATUS_BLOCKED
            locked.step = system_agents.STEP_BLOCKED
            locked.save(update_fields=["status", "step", "state", "updated_at"])
            if goal is not None:
                action = _cleanup_action_for_workflow(locked, goal, runs=runs)
        _apply_finish_action(action)
    return True


def stop_running_autonomous_goal_stack_after_proposal_resolution(
    autonomous_goal_id: int,
    proposal_id: int,
    outcome_status: str,
) -> bool:
    error_by_outcome = {
        ProposedSession.OUTCOME_ACCEPTED: system_agents.AUTONOMOUS_GOAL_PROPOSAL_ACCEPTED_ERROR,
        ProposedSession.OUTCOME_REJECTED: system_agents.AUTONOMOUS_GOAL_PROPOSAL_REJECTED_ERROR,
        ProposedSession.OUTCOME_DISMISSED: system_agents.AUTONOMOUS_GOAL_PROPOSAL_DISMISSED_ERROR,
    }
    error = error_by_outcome.get(outcome_status)
    if error is None:
        return True
    proposal = ProposedSession.objects.filter(
        pk=proposal_id,
        autonomous_goal_id=autonomous_goal_id,
    ).first()
    if proposal is None:
        return True
    workflow_ids: set[int] = set()
    if proposal.source_workflow_id is not None:
        workflow_ids.add(proposal.source_workflow_id)
    workflow_ids.update(
        SystemWorkflow.objects.filter(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=_autonomous_goal_main_thread_id(autonomous_goal_id),
            status=SystemWorkflow.STATUS_RUNNING,
            state__proposal_id=proposal_id,
        ).values_list("pk", flat=True)
    )
    return all(
        stop_running_autonomous_goal_workflow(
            autonomous_goal_id,
            error,
            workflow_id=workflow_id,
        )
        for workflow_id in workflow_ids
    )


def recover_orphaned_workflows(workflows: list[SystemWorkflow]) -> int:
    recovered = 0
    stale_before = timezone.now() - _ORPHANED_RUN_TIMEOUT
    for candidate in workflows:
        workflow = SystemWorkflow.objects.filter(pk=candidate.pk).first()
        if workflow is None or not workflow.is_active:
            continue
        if system_agents.workflow_has_inflight_instance(workflow.pk):
            continue
        if workflow.step != system_agents.STEP_AUTONOMOUS_GOAL_RUNNING:
            if _retire_legacy_workflow(workflow):
                recovered += 1
            continue
        if workflow.updated_at > stale_before:
            continue
        workflow = _claim_orphaned_workflow(workflow.pk, stale_before=stale_before)
        if workflow is None:
            continue
        if system_agents.workflow_has_inflight_instance(workflow.pk):
            continue
        goal = _workflow_goal(workflow)
        if goal is None:
            with transaction.atomic():
                locked = SystemWorkflow.objects.select_for_update().get(pk=workflow.pk)
                if locked.is_active:
                    _fail_workflow_locked(locked, "autonomous goal no longer exists")
                    recovered += 1
            continue
        checkpoint = _latest_checkpoint(workflow)
        if checkpoint is not None and _workflow_should_continue(workflow, goal, checkpoint):
            _spawn_autonomous_goal_candidate_or_finish(workflow, goal)
        elif checkpoint is not None:
            with transaction.atomic():
                locked = SystemWorkflow.objects.select_for_update().get(pk=workflow.pk)
                if locked.is_active:
                    action = _publish_checkpoint_locked(locked, goal, checkpoint)
                else:
                    action = _PostFinishAction()
            _apply_finish_action(action)
        elif not workflow.agent_runs.exists():
            _spawn_autonomous_goal_candidate_or_finish(workflow, goal)
        else:
            _finish_spawn_failure(
                workflow,
                goal,
                RuntimeError("autonomous goal run lost its candidate worker"),
            )
        recovered += 1
    return recovered


def reconcile_pending_run_cleanups(
    *,
    main_thread_id: str | None = None,
    workflow_id: int | None = None,
) -> int:
    runs = SystemAgentRun.objects.filter(
        status__in=(SystemAgentRun.STATUS_COMPLETED, SystemAgentRun.STATUS_FAILED),
        output__has_key=_RUN_PENDING_CLEANUP_OUTPUT_KEY,
    ).select_related("workflow")
    if main_thread_id is not None:
        runs = runs.filter(workflow__main_thread_id=main_thread_id)
    if workflow_id is not None:
        runs = runs.filter(workflow_id=workflow_id)
    reconciled = 0
    for run in runs.order_by("created_at", "id"):
        try:
            if cleanup_terminal_run(run):
                reconciled += 1
        except Exception:
            logger.exception("failed to recover cleanup for autonomous goal run %s", run.pk)
    return reconciled


def _claim_orphaned_workflow(workflow_id: int, *, stale_before: datetime) -> SystemWorkflow | None:
    claimed = SystemWorkflow.objects.filter(
        pk=workflow_id,
        kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        status=SystemWorkflow.STATUS_RUNNING,
        step=system_agents.STEP_AUTONOMOUS_GOAL_RUNNING,
        updated_at__lte=stale_before,
    ).update(updated_at=timezone.now())
    if not claimed:
        return None
    return SystemWorkflow.objects.filter(pk=workflow_id).first()


def _retire_legacy_workflow(workflow: SystemWorkflow) -> bool:
    action = _PostFinishAction()
    with transaction.atomic():
        locked = SystemWorkflow.objects.select_for_update().get(pk=workflow.pk)
        if not locked.is_active or locked.step == system_agents.STEP_AUTONOMOUS_GOAL_RUNNING:
            return False
        goal = _workflow_goal(locked)
        runs = list(locked.agent_runs.select_for_update().all())
        _fail_active_runs_locked(runs, _LEGACY_WORKFLOW_UPGRADE_ERROR)
        if goal is not None:
            _create_notice_locked(
                locked,
                goal,
                title=f"Autonomous goal needs retry: {goal.title}"[:_AUTONOMOUS_GOAL_TITLE_MAX_LEN],
                summary=_LEGACY_WORKFLOW_UPGRADE_ERROR,
                automation_status="failed",
            )
            action = _cleanup_action_for_workflow(locked, goal, runs=runs)
        _fail_workflow_locked(locked, _LEGACY_WORKFLOW_UPGRADE_ERROR)
    _apply_finish_action(action)
    return True


def cleanup_terminal_run(run: SystemAgentRun) -> bool:
    if run.status not in (SystemAgentRun.STATUS_COMPLETED, SystemAgentRun.STATUS_FAILED):
        return False
    if _RUN_PENDING_CLEANUP_OUTPUT_KEY in _run_output(run):
        _apply_finish_action_for_run(run, _pending_run_cleanup(run))
        return True
    workflow = run.workflow
    goal = _workflow_goal(workflow)
    if goal is None or workflow.is_active:
        return False
    if run.agent_kind == _AUTONOMOUS_GOAL_REVIEW_AGENT_KIND:
        _apply_finish_action(_cleanup_action_for_run(run, goal))
        return True
    return False


def _autonomous_goal_instance_tokens_used(instance: CodexInstance) -> int | None:
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
    path = _rollout_path_from_value(codex_path)
    if path is None:
        return None
    usage = rollout.latest_token_usage(path)
    return usage["total_tokens"] if usage is not None else None


def _cleanup_autonomous_goal_candidate_cwd(cwd: str) -> None:
    if not cwd:
        return
    try:
        cleanup_managed_worktree_path(cwd)
    except WorktreeCleanupError:
        logger.exception("failed to clean up autonomous goal worktree %s", cwd)


def _release_autonomous_goal_snapshot_ref(repo_path: str, ref: str) -> None:
    if not repo_path or not ref:
        return
    try:
        release_snapshot_commit_ref(repo_path, ref)
    except WorktreeCleanupError:
        logger.exception("failed to release autonomous goal snapshot ref %s", ref)


def _cleanup_new_worktree(worktree: ManagedWorktree | None) -> None:
    if worktree is None:
        return
    try:
        cleanup_worktree(worktree)
    except WorktreeCleanupError:
        logger.exception("failed to clean up new autonomous goal worktree %s", worktree.path)


def _autonomous_goal_main_thread_id(autonomous_goal_id: int) -> str:
    return f"autonomous-goal:{autonomous_goal_id}"
