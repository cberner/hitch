"""Run-display helpers for autonomous goals and proposed sessions.

These functions compute the per-goal run badges, token counts, log URLs, and
proposed-session display state rendered by the autonomous-goals and new-session
pages. They are pure read helpers extracted from ``views`` and must not import
from ``views`` to keep the module dependency graph acyclic.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal, NamedTuple

from django.http import Http404, HttpRequest
from django.urls import reverse
from django.utils import timezone

from hitch.main import codex_events, system_agents, token_usage
from hitch.main.goals import autonomous_goal_prompts, autonomous_goal_proposal_stack
from hitch.main.models import (
    AutonomousGoal,
    CodexInstance,
    Project,
    ProposedSession,
    SystemAgentRun,
    SystemWorkflow,
)
from hitch.main.session_settings import (
    _active_project_from_request,
    _project_for_proposed_session,
)
from hitch.main.settings_cookies import _MAX_BIGAUTOFIELD

AutonomousGoalRunState = Literal[
    "blocked",
    "failed",
    "manual",
    "maxed",
    "queued",
    "quota",
    "ready",
    "review",
    "running",
    "skipped",
    "waiting",
]


class AutonomousGoalRunBadge(NamedTuple):
    state: AutonomousGoalRunState
    label: str
    title: str
    detail: str


def _attach_autonomous_goal_run_state(goals: list[AutonomousGoal]) -> None:
    goal_ids = [goal.pk for goal in goals]
    if not goal_ids:
        return
    pending_proposal_state = autonomous_goal_proposal_stack._autonomous_goal_pending_proposal_state(
        goals
    )
    pending_proposal_goal_ids = pending_proposal_state.blocking_goal_ids
    unresolved_failure_notice_goal_ids = _autonomous_goal_failure_notice_ids(goal_ids)
    project_running_auto_proposal_ids = _autonomous_goal_running_auto_proposal_project_ids(
        goals
    )
    project_in_flight_automation_ids = _autonomous_goal_in_flight_automation_project_ids(
        goals
    )
    no_change_goal_ids = _autonomous_goal_no_change_ids(
        goals,
        continuable_stack_goal_ids=pending_proposal_state.continuable_stack_goal_ids,
    )
    auto_proposals_paused_by_quota = (
        any(goal.auto_proposal_enabled for goal in goals)
        and system_agents._auto_proposals_paused_by_usage_quota_throttled()
    )
    workflows = (
        SystemWorkflow.objects.filter(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id__in=[
                system_agents._autonomous_goal_main_thread_id(goal_id)
                for goal_id in goal_ids
            ],
        )
        .order_by("main_thread_id", "-created_at")
    )
    workflows_by_thread: dict[str, SystemWorkflow] = {}
    for workflow in workflows:
        workflows_by_thread.setdefault(workflow.main_thread_id, workflow)
    latest_workflows = list(workflows_by_thread.values())
    log_urls_by_workflow_id = _autonomous_goal_log_urls(latest_workflows)
    running_tokens_by_workflow_id = _autonomous_goal_running_token_counts(
        latest_workflows
    )
    for goal in goals:
        latest_workflow = workflows_by_thread.get(
            system_agents._autonomous_goal_main_thread_id(goal.pk)
        )
        goal.run_running = (  # type: ignore[attr-defined]
            latest_workflow is not None
            and latest_workflow.is_active
        )
        goal.run_tokens_used_display = _autonomous_goal_run_tokens_used_display(  # type: ignore[attr-defined]
            latest_workflow,
            running_tokens_by_workflow_id,
        )
        goal.run_log_url = (  # type: ignore[attr-defined]
            log_urls_by_workflow_id.get(latest_workflow.pk) or ""
            if latest_workflow is not None
            else ""
        )
        run_badge = _autonomous_goal_run_badge(
            goal,
            latest_workflow,
            pending_proposal_goal_ids=pending_proposal_goal_ids,
            continuable_stack_goal_ids=(
                pending_proposal_state.continuable_stack_goal_ids
            ),
            unresolved_failure_notice_goal_ids=unresolved_failure_notice_goal_ids,
            project_running_auto_proposal_ids=project_running_auto_proposal_ids,
            project_in_flight_automation_ids=project_in_flight_automation_ids,
            no_change_goal_ids=no_change_goal_ids,
            auto_proposals_paused_by_quota=auto_proposals_paused_by_quota,
        )
        goal.run_status_state = run_badge.state  # type: ignore[attr-defined]
        goal.run_status_label = run_badge.label  # type: ignore[attr-defined]
        goal.run_status_title = run_badge.title  # type: ignore[attr-defined]
        goal.run_status_detail = run_badge.detail  # type: ignore[attr-defined]


def _autonomous_goal_failure_notice_ids(goal_ids: list[int]) -> set[int]:
    return {
        goal_id
        for goal_id in ProposedSession.objects.filter(
            autonomous_goal_id__in=goal_ids,
            inbox_kind=ProposedSession.INBOX_KIND_NOTICE,
            outcome_status=ProposedSession.OUTCOME_UNSET,
            outcome_metadata__automation_status="failed",
        ).values_list("autonomous_goal_id", flat=True)
        if isinstance(goal_id, int)
    }


def _autonomous_goal_running_auto_proposal_project_ids(
    goals: list[AutonomousGoal],
) -> set[int]:
    project_ids = {goal.project_id for goal in goals}
    repo_path_by_project_id = dict(
        Project.objects.filter(pk__in=project_ids).values_list("pk", "repo_path")
    )
    running_auto_proposal_cwds = set(
        SystemWorkflow.objects.filter(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            status=SystemWorkflow.STATUS_RUNNING,
            state__auto_proposal=True,
        ).values_list("cwd", flat=True)
    )
    return {
        project_id
        for project_id, repo_path in repo_path_by_project_id.items()
        if repo_path in running_auto_proposal_cwds
    }


def _autonomous_goal_in_flight_automation_project_ids(
    goals: list[AutonomousGoal],
) -> set[int]:
    project_ids = {goal.project_id for goal in goals}
    if not project_ids:
        return set()
    in_flight_project_ids: set[int] = set()
    claim_key = ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY
    claim_lookup = f"outcome_metadata__{claim_key}__isnull"
    now = timezone.now()
    claimed_metadatas = (
        ProposedSession.objects.filter(
            project_id__in=project_ids,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            accepted_session__isnull=True,
            **{claim_lookup: False},
        )
        .filter(autonomous_goal_proposal_stack._autonomous_goal_in_flight_proposal_criteria())
        .values_list("project_id", "outcome_metadata")
    )
    for project_id, metadata in claimed_metadatas:
        if not isinstance(project_id, int):
            continue
        if ProposedSession.accepted_session_start_claim_is_active(metadata, now=now):
            in_flight_project_ids.add(project_id)

    accepted_thread_project_ids: dict[str, int] = {}
    accepted_threads = (
        ProposedSession.objects.filter(
            project_id__in=project_ids,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            accepted_session__isnull=False,
        )
        .filter(autonomous_goal_proposal_stack._autonomous_goal_in_flight_proposal_criteria())
        .exclude(accepted_session__thread_id="")
        .values_list("project_id", "accepted_session__thread_id")
    )
    for project_id, thread_id in accepted_threads:
        if isinstance(project_id, int) and isinstance(thread_id, str):
            accepted_thread_project_ids[thread_id] = project_id
    if not accepted_thread_project_ids:
        return in_flight_project_ids

    active_thread_ids = set(
        CodexInstance.objects.filter(
            thread_id__in=list(accepted_thread_project_ids),
            status__in=CodexInstance.ACTIVE_STATUSES,
        ).values_list("thread_id", flat=True)
    )
    active_thread_ids.update(
        SystemWorkflow.objects.filter(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id__in=list(accepted_thread_project_ids),
            status=SystemWorkflow.STATUS_RUNNING,
        ).values_list("main_thread_id", flat=True)
    )
    for thread_id in active_thread_ids:
        active_project_id = accepted_thread_project_ids.get(thread_id)
        if active_project_id is not None:
            in_flight_project_ids.add(active_project_id)
    return in_flight_project_ids


def _autonomous_goal_no_change_ids(
    goals: list[AutonomousGoal], *, continuable_stack_goal_ids: set[int]
) -> set[int]:
    no_change_goal_ids: set[int] = set()
    for goal in goals:
        last_no_proposal_sha = goal.auto_proposal_last_no_proposal_sha.strip()
        if (
            not goal.auto_proposal_enabled
            or goal.pk in continuable_stack_goal_ids
            or not last_no_proposal_sha
        ):
            continue
        current_sha = system_agents._autonomous_goal_auto_proposal_base_sha(goal)
        if current_sha == last_no_proposal_sha:
            no_change_goal_ids.add(goal.pk)
    return no_change_goal_ids


def _autonomous_goal_run_badge(
    goal: AutonomousGoal,
    workflow: SystemWorkflow | None,
    *,
    pending_proposal_goal_ids: set[int],
    continuable_stack_goal_ids: set[int],
    unresolved_failure_notice_goal_ids: set[int],
    project_running_auto_proposal_ids: set[int],
    project_in_flight_automation_ids: set[int],
    no_change_goal_ids: set[int],
    auto_proposals_paused_by_quota: bool,
) -> AutonomousGoalRunBadge:
    if workflow is not None:
        if workflow.is_active:
            return AutonomousGoalRunBadge(
                state="running",
                label="Running",
                title="Autonomous goal is running",
                detail="This autonomous goal run is still working.",
            )
        if workflow.status == SystemWorkflow.STATUS_BLOCKED:
            return AutonomousGoalRunBadge(
                state="blocked",
                label="Blocked",
                title="Autonomous goal is blocked",
                detail=(
                    _workflow_state_string(workflow, "error")
                    or "This autonomous goal run is blocked. Open the run log for details."
                ),
            )
        if workflow.status == SystemWorkflow.STATUS_FAILED:
            return AutonomousGoalRunBadge(
                state="failed",
                label="Failed",
                title="Autonomous goal run failed",
                detail=(
                    _workflow_state_string(workflow, "error")
                    or "The last autonomous goal run failed. Open the run log for details."
                ),
            )
        if workflow.status == SystemWorkflow.STATUS_MAX_ITERATIONS_REACHED:
            return AutonomousGoalRunBadge(
                state="maxed",
                label="Maxed",
                title="Autonomous goal reached its iteration limit",
                detail="The last autonomous goal run stopped after reaching its iteration limit.",
            )

    if goal.pk in pending_proposal_goal_ids:
        return AutonomousGoalRunBadge(
            state="review",
            label="Review",
            title="Autonomous goal is waiting for review",
            detail="Not running because a proposal from this goal is waiting in the inbox.",
        )
    if goal.pk in unresolved_failure_notice_goal_ids:
        return AutonomousGoalRunBadge(
            state="failed",
            label="Failed",
            title="Autonomous goal is paused after a failure",
            detail=(
                "Not running because a failure notice from this goal is still in the inbox. "
                "Dismiss or resolve that notice to let auto-proposal try again."
            ),
        )
    if (
        goal.auto_proposal_enabled
        and goal.project_id in project_running_auto_proposal_ids
    ):
        return AutonomousGoalRunBadge(
            state="queued",
            label="Queued",
            title="Autonomous goal is queued",
            detail="Not running because another auto-proposal run is active for this project.",
        )
    if (
        goal.auto_proposal_enabled
        and goal.project_id in project_in_flight_automation_ids
    ):
        return AutonomousGoalRunBadge(
            state="queued",
            label="Queued",
            title="Autonomous goal is queued",
            detail=(
                "Not running because accepted autonomous-goal automation "
                "is still active for this project."
            ),
        )
    if goal.pk in no_change_goal_ids:
        return AutonomousGoalRunBadge(
            state="waiting",
            label="No change",
            title="Autonomous goal is waiting for branch changes",
            detail=(
                "Not running because the last auto-proposal found no useful proposal "
                "for the tracked branch. It will try again after that branch changes."
            ),
        )
    if goal.auto_proposal_enabled and auto_proposals_paused_by_quota:
        return AutonomousGoalRunBadge(
            state="quota",
            label="Quota",
            title="Autonomous goal is paused for quota",
            detail=(
                "Not running because remaining Codex quota is below the "
                "auto-proposal safety threshold. It will try again as quota recovers."
            ),
        )
    if goal.auto_proposal_enabled and goal.pk in continuable_stack_goal_ids:
        return AutonomousGoalRunBadge(
            state="ready",
            label="Ready",
            title="Autonomous goal is ready",
            detail="Auto-proposal is enabled. This goal will start when the scheduler runs and quota allows.",
        )
    if workflow is not None and workflow.status == SystemWorkflow.STATUS_COMPLETED:
        completed_badge = _completed_autonomous_goal_run_badge(workflow)
        if completed_badge is not None:
            return completed_badge
    if not goal.auto_proposal_enabled:
        detail = "Auto-proposal is off. Use Run to start this goal manually."
        latest_detail = _autonomous_goal_latest_run_detail(workflow)
        if latest_detail:
            detail = f"{detail}\n\n{latest_detail}"
        return AutonomousGoalRunBadge(
            state="manual",
            label="Manual",
            title="Autonomous goal is manual",
            detail=detail,
        )
    return AutonomousGoalRunBadge(
        state="ready",
        label="Ready",
        title="Autonomous goal is ready",
        detail="Auto-proposal is enabled. This goal will start when the scheduler runs and quota allows.",
    )


def _completed_autonomous_goal_run_badge(
    workflow: SystemWorkflow,
) -> AutonomousGoalRunBadge | None:
    if workflow.step == system_agents.STEP_AUTONOMOUS_GOAL_SKIPPED:
        return AutonomousGoalRunBadge(
            state="skipped",
            label="Skipped",
            title="Autonomous goal last run was skipped",
            detail=_autonomous_goal_latest_run_detail(workflow),
        )
    return None


def _autonomous_goal_running_token_counts(
    workflows: Iterable[SystemWorkflow],
) -> dict[int, int]:
    workflows_by_id = {
        workflow.pk: workflow
        for workflow in workflows
        if workflow.is_active
    }
    if not workflows_by_id:
        return {}
    runs = (
        SystemAgentRun.objects.select_related("instance")
        .filter(
            workflow_id__in=list(workflows_by_id),
            status__in=(
                SystemAgentRun.STATUS_STARTING,
                SystemAgentRun.STATUS_RUNNING,
            ),
        )
        .exclude(thread_id="")
        .order_by("workflow_id", "-created_at", "-pk")
    )
    tokens_by_workflow_id: dict[int, int] = {}
    for run in runs:
        if run.workflow_id in tokens_by_workflow_id:
            continue
        workflow = workflows_by_id.get(run.workflow_id)
        if workflow is not None:
            tokens_by_workflow_id[run.workflow_id] = (
                _autonomous_goal_running_token_count(workflow, run.instance)
            )
    return tokens_by_workflow_id


def _autonomous_goal_running_token_count(
    workflow: SystemWorkflow, instance: CodexInstance
) -> int:
    persisted_tokens = _workflow_state_int(
        workflow, autonomous_goal_prompts._AUTONOMOUS_GOAL_PROPOSAL_BUDGET_USED_STATE_KEY
    )
    current_tokens = codex_events.latest_goal_tokens_for_instance(instance)
    if current_tokens is None:
        return persisted_tokens
    previous_tokens = _autonomous_goal_recorded_thread_tokens(workflow, instance)
    return persisted_tokens + max(current_tokens - previous_tokens, 0)


def _autonomous_goal_recorded_thread_tokens(
    workflow: SystemWorkflow, instance: CodexInstance
) -> int:
    token_totals = workflow.state.get(
        system_agents._AUTONOMOUS_GOAL_PROPOSAL_BUDGET_TOKEN_TOTALS_STATE_KEY
    )
    if not isinstance(token_totals, dict):
        return 0
    value = token_totals.get(instance.thread_id)
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
        else 0
    )


def _autonomous_goal_run_tokens_used_display(
    workflow: SystemWorkflow | None, running_tokens_by_workflow_id: Mapping[int, int]
) -> str:
    if workflow is None or not workflow.is_active:
        return ""
    tokens = running_tokens_by_workflow_id.get(
        workflow.pk,
        _workflow_state_int(
            workflow, autonomous_goal_prompts._AUTONOMOUS_GOAL_PROPOSAL_BUDGET_USED_STATE_KEY
        ),
    )
    return f"{token_usage._format_token_count(tokens)} tokens"


def _autonomous_goal_latest_run_detail(workflow: SystemWorkflow | None) -> str:
    if workflow is None:
        return ""
    if workflow.status == SystemWorkflow.STATUS_COMPLETED:
        if workflow.step == system_agents.STEP_AUTONOMOUS_GOAL_SKIPPED:
            return _autonomous_goal_skipped_detail(workflow)
        if workflow.step == system_agents.STEP_AUTONOMOUS_GOAL_PROPOSED:
            return _autonomous_goal_proposed_detail(workflow)
        return "The last autonomous goal run completed."
    if workflow.status == SystemWorkflow.STATUS_FAILED:
        return (
            _workflow_state_string(workflow, "error")
            or "The last autonomous goal run failed."
        )
    if workflow.status == SystemWorkflow.STATUS_MAX_ITERATIONS_REACHED:
        return "The last autonomous goal run stopped after reaching its iteration limit."
    return ""


def _autonomous_goal_skipped_detail(workflow: SystemWorkflow) -> str:
    candidate = workflow.state.get("candidate")
    if isinstance(candidate, dict):
        message = candidate.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    judgment = workflow.state.get("judgment")
    if isinstance(judgment, dict):
        rationale = judgment.get("rationale")
        if isinstance(rationale, str) and rationale.strip():
            return rationale.strip()
    return "The last autonomous goal run completed without a proposal."


def _autonomous_goal_proposed_detail(workflow: SystemWorkflow) -> str:
    stopped_reason = _workflow_state_string(workflow, "stacked_diff_stopped_reason")
    if stopped_reason == "candidate_no_proposal":
        return (
            "The last autonomous goal run published the current stacked proposal "
            "because the next candidate produced no proposal."
        )
    if stopped_reason == "judge_confidence_below_threshold":
        return (
            "The last autonomous goal run published the current stacked proposal "
            "because the next candidate fell below the confidence threshold."
        )
    if stopped_reason == "stacked_diff_continuation_failed":
        error = _workflow_state_string(workflow, "stacked_diff_continuation_error")
        if error:
            return (
                "The last autonomous goal run published the current stacked proposal "
                f"after the next candidate failed: {error}"
            )
        return (
            "The last autonomous goal run published the current stacked proposal "
            "after the next candidate failed."
        )
    return "The last autonomous goal run created a proposal and stopped."


def _workflow_state_string(workflow: SystemWorkflow, key: str) -> str:
    value = workflow.state.get(key)
    return value.strip() if isinstance(value, str) else ""


def _attach_proposed_session_display_state(
    proposed_sessions: list[ProposedSession],
) -> None:
    for proposed_session in proposed_sessions:
        files = proposed_session.relevant_files
        proposed_session.display_files = (  # type: ignore[attr-defined]
            [item for item in files if isinstance(item, str) and item.strip()]
            if isinstance(files, list)
            else []
        )
        if proposed_session.candidate_session is not None:
            proposed_session.candidate_log_url = reverse(  # type: ignore[attr-defined]
                "system_session",
                kwargs={"session_id": proposed_session.candidate_session.thread_id},
            )
        else:
            proposed_session.candidate_log_url = ""  # type: ignore[attr-defined]
        if proposed_session.judge_session is not None:
            proposed_session.judge_log_url = reverse(  # type: ignore[attr-defined]
                "system_session",
                kwargs={"session_id": proposed_session.judge_session.thread_id},
            )
        else:
            proposed_session.judge_log_url = ""  # type: ignore[attr-defined]
        proposed_session.session_prompt = _proposed_session_prompt(  # type: ignore[attr-defined]
            proposed_session
        )
        project = _project_for_proposed_session(proposed_session)
        proposed_session.accept_project_id = (  # type: ignore[attr-defined]
            project.pk if project is not None else ""
        )
        auto_pr_enabled, auto_qa_enabled = (
            _auto_review_settings_for_proposed_session(proposed_session)
        )
        proposed_session.accept_auto_pr = auto_pr_enabled  # type: ignore[attr-defined]
        proposed_session.accept_auto_qa = auto_qa_enabled  # type: ignore[attr-defined]
        proposed_session.stack_label = _proposed_session_stack_label(  # type: ignore[attr-defined]
            proposed_session
        )


def _proposed_session_stack_label(proposed_session: ProposedSession) -> str:
    metadata = (
        proposed_session.outcome_metadata
        if isinstance(proposed_session.outcome_metadata, dict)
        else {}
    )
    metadata_depth = metadata.get("stacked_diff_depth")
    metadata_iteration = metadata.get("stacked_diff_iteration")
    if (
        isinstance(metadata_depth, int)
        and not isinstance(metadata_depth, bool)
        and isinstance(metadata_iteration, int)
        and not isinstance(metadata_iteration, bool)
        and metadata_depth > AutonomousGoal.STACKED_DIFF_DEPTH_MIN
    ):
        depth = min(metadata_depth, AutonomousGoal.STACKED_DIFF_DEPTH_MAX)
        if metadata_iteration < 1 or metadata_iteration > depth:
            return ""
        iteration = metadata_iteration
        return f"Stack {iteration} of {depth}"
    return ""


def _proposed_session_prompt(proposed_session: ProposedSession) -> str:
    if proposed_session.prompt.strip():
        return proposed_session.prompt.strip()
    parts = [
        "Go ahead and implement this proposed session.",
        "",
        f"Autonomous goal: {proposed_session.autonomous_goal.title}"
        if proposed_session.autonomous_goal is not None
        else "Source: Coding agent proposal",
    ]
    if (
        proposed_session.autonomous_goal is not None
        and proposed_session.autonomous_goal.goal
    ):
        parts.extend(
            ["", f"Autonomous goal objective:\n{proposed_session.autonomous_goal.goal}"]
        )
    parts.extend(["", f"Proposed session: {proposed_session.title}"])
    if proposed_session.summary:
        parts.extend(["", f"Summary:\n{proposed_session.summary}"])
    files = proposed_session.display_files  # type: ignore[attr-defined]
    if files:
        parts.extend(["", "Relevant files:", *[f"- {file}" for file in files]])
    return "\n".join(parts)


def _autonomous_goal_log_urls(workflows: Iterable[SystemWorkflow]) -> dict[int, str]:
    workflow_ids = [workflow.pk for workflow in workflows]
    if not workflow_ids:
        return {}
    runs = (
        SystemAgentRun.objects.filter(workflow_id__in=workflow_ids)
        .exclude(thread_id="")
        .order_by("workflow_id", "-created_at")
    )
    urls: dict[int, str] = {}
    for run in runs:
        urls.setdefault(
            run.workflow_id,
            reverse("autonomous_goal_run_log", kwargs={"workflow_id": run.workflow_id}),
        )
    return urls


def _autonomous_goal_workflow_for_log(
    request: HttpRequest, workflow_id: int
) -> SystemWorkflow:
    if workflow_id < 1 or workflow_id > _MAX_BIGAUTOFIELD:
        raise Http404("autonomous goal run log not found")
    project = _active_project_from_request(request)
    if project is None:
        raise Http404("autonomous goal run log not found")
    workflow = (
        SystemWorkflow.objects.filter(
            pk=workflow_id,
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        .first()
    )
    if workflow is None:
        raise Http404("autonomous goal run log not found")
    autonomous_goal_id = _workflow_state_int(workflow, "autonomous_goal_id")
    autonomous_goal = AutonomousGoal.objects.filter(
        pk=autonomous_goal_id,
        project=project,
    ).first()
    if autonomous_goal is None:
        raise Http404("autonomous goal run log not found")
    return workflow


def _workflow_state_int(workflow: SystemWorkflow, key: str) -> int:
    value = workflow.state.get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _auto_review_settings_for_proposed_session(
    proposed_session: ProposedSession,
) -> tuple[bool, bool]:
    metadata = _proposal_metadata(proposed_session)
    if "auto_pr_enabled" in metadata or "auto_qa_enabled" in metadata:
        auto_pr_enabled = metadata.get("auto_pr_enabled") is True
        auto_qa_enabled = metadata.get("auto_qa_enabled") is True and not auto_pr_enabled
        return auto_pr_enabled, auto_qa_enabled
    autonomous_goal = proposed_session.autonomous_goal
    if autonomous_goal is None:
        return False, False
    auto_pr_enabled = autonomous_goal.autonomy == AutonomousGoal.AUTONOMY_DRAFT_PR
    auto_qa_enabled = autonomous_goal.auto_qa_enabled and not auto_pr_enabled
    return auto_pr_enabled, auto_qa_enabled


def _proposal_metadata(proposed_session: ProposedSession) -> dict[str, object]:
    return (
        proposed_session.outcome_metadata
        if isinstance(proposed_session.outcome_metadata, dict)
        else {}
    )
