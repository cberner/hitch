"""Shared autonomous-goal proposal, budget, and stack helpers."""

from __future__ import annotations

from typing import Any

from hitch.main.models import AutonomousGoal, SystemWorkflow
from hitch.main.workflows.workflow_state import _state_int, _state_string

_AUTONOMOUS_GOAL_TITLE_MAX_LEN = 200
_AUTONOMOUS_GOAL_SESSION_CWD_STATE_KEY = "session_cwd"
_AUTONOMOUS_GOAL_STACKED_DEPTH_STATE_KEY = "stacked_diff_depth"
_AUTONOMOUS_GOAL_STACKED_ITERATION_STATE_KEY = "stacked_diff_iteration"
_AUTONOMOUS_GOAL_PROPOSAL_BUDGET_STATE_KEY = "proposal_budget"
_AUTONOMOUS_GOAL_PROPOSAL_BUDGET_USED_STATE_KEY = "proposal_budget_tokens_used"
_AUTONOMOUS_GOAL_FAILED_ATTEMPTS_STATE_KEY = "proposal_budget_failed_attempts"
_AUTONOMOUS_GOAL_LAST_FAILURE_STATE_KEY = "proposal_budget_last_failure"
_AUTONOMOUS_GOAL_NO_PROGRESS_RETRIES_STATE_KEY = (
    "proposal_budget_no_progress_retries"
)


def _autonomous_goal_proposed_session_prompt(
    autonomous_goal: AutonomousGoal,
    candidate: dict[str, Any],
    judgment: dict[str, str],
) -> str:
    suggested = candidate.get("suggested_continuation")
    if isinstance(suggested, str) and suggested.strip():
        return suggested.strip()
    parts = [
        "Go ahead and implement this proposed session.",
        "",
        f"Autonomous goal: {autonomous_goal.title}",
    ]
    if autonomous_goal.goal:
        parts.extend(["", f"Autonomous goal objective:\n{autonomous_goal.goal}"])
    title = str(candidate.get("title", autonomous_goal.title)).strip()
    if title:
        parts.extend(["", f"Proposed session: {title}"])
    summary = str(candidate.get("summary") or "").strip()
    if summary:
        parts.extend(["", f"Summary:\n{summary}"])
    judge_feedback = judgment.get("feedback", "").strip()
    if judge_feedback:
        parts.extend(["", f"Judge feedback:\n{judge_feedback}"])
    implementation_direction = candidate.get("implementation_direction")
    if isinstance(implementation_direction, str) and implementation_direction.strip():
        parts.extend(
            ["", f"Implementation guidance:\n{implementation_direction.strip()}"]
        )
    for label, key in (
        ("Implemented so far", "implemented_changes"),
        ("Impact", "impact"),
        ("Verification", "verification"),
        ("Known rough edges", "rough_edges"),
    ):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            parts.extend(["", f"{label}:\n{value.strip()}"])
    files = _string_list(candidate.get("relevant_files"))
    if files:
        parts.extend(["", "Relevant files:", *[f"- {file}" for file in files]])
    return "\n".join(parts)


def _autonomous_goal_proposal_summary(
    candidate: dict[str, Any], judgment: dict[str, str]
) -> str:
    parts = []
    for label, value in (
        ("Summary", candidate.get("summary")),
        ("Judge feedback", judgment.get("feedback", "")),
        ("Implemented", candidate.get("implemented_changes")),
        ("Impact", candidate.get("impact")),
        ("Verification", candidate.get("verification")),
        ("Rough edges", candidate.get("rough_edges")),
    ):
        if isinstance(value, str) and value.strip():
            parts.append(f"{label}: {value.strip()}")
    return "\n\n".join(parts)


def _autonomous_goal_workflow_proposal_budget(workflow: SystemWorkflow) -> int:
    return _state_int(workflow, _AUTONOMOUS_GOAL_PROPOSAL_BUDGET_STATE_KEY)


def _autonomous_goal_proposal_budget_tokens_used(workflow: SystemWorkflow) -> int:
    return _state_int(workflow, _AUTONOMOUS_GOAL_PROPOSAL_BUDGET_USED_STATE_KEY)


def _autonomous_goal_failed_attempts(workflow: SystemWorkflow) -> int:
    return _state_int(workflow, _AUTONOMOUS_GOAL_FAILED_ATTEMPTS_STATE_KEY)


def _autonomous_goal_no_progress_budget_retries(workflow: SystemWorkflow) -> int:
    return _state_int(workflow, _AUTONOMOUS_GOAL_NO_PROGRESS_RETRIES_STATE_KEY)


def _autonomous_goal_proposal_budget_metadata(
    workflow: SystemWorkflow,
) -> dict[str, object]:
    budget = _autonomous_goal_workflow_proposal_budget(workflow)
    if budget <= 0:
        return {}
    return {
        "proposal_budget": budget,
        "proposal_budget_tokens_used": _autonomous_goal_proposal_budget_tokens_used(
            workflow
        ),
        "proposal_budget_failed_attempts": _autonomous_goal_failed_attempts(workflow),
    }


def _autonomous_goal_workflow_stacked_diff_depth(
    workflow: SystemWorkflow, autonomous_goal: AutonomousGoal
) -> int:
    depth = _state_int(workflow, _AUTONOMOUS_GOAL_STACKED_DEPTH_STATE_KEY)
    if not depth:
        depth = autonomous_goal.effective_stacked_diff_depth
    return min(
        max(depth, AutonomousGoal.STACKED_DIFF_DEPTH_MIN),
        AutonomousGoal.STACKED_DIFF_DEPTH_MAX,
    )


def _autonomous_goal_stack_iteration(workflow: SystemWorkflow) -> int:
    return max(_state_int(workflow, _AUTONOMOUS_GOAL_STACKED_ITERATION_STATE_KEY), 1)


def _autonomous_goal_session_cwd(workflow: SystemWorkflow) -> str:
    return _state_string(workflow, _AUTONOMOUS_GOAL_SESSION_CWD_STATE_KEY) or workflow.cwd


def _autonomous_goal_candidate_allows_code_changes(workflow: SystemWorkflow) -> bool:
    return _autonomous_goal_session_cwd(workflow) != workflow.cwd


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        item = item.strip()
        if item and item not in normalized:
            normalized.append(item)
    return normalized
