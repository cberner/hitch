"""Shared autonomous-goal proposal, budget, and stack helpers."""

from __future__ import annotations

from typing import Any

from hitch.main.models import AutonomousGoal

_AUTONOMOUS_GOAL_TITLE_MAX_LEN = 200
_AUTONOMOUS_GOAL_STACKED_DEPTH_STATE_KEY = "stacked_diff_depth"
_AUTONOMOUS_GOAL_PROPOSAL_BUDGET_STATE_KEY = "proposal_budget"
_AUTONOMOUS_GOAL_PROPOSAL_BUDGET_USED_STATE_KEY = "proposal_budget_tokens_used"


def _autonomous_goal_proposed_session_prompt(
    autonomous_goal: AutonomousGoal,
    candidate: dict[str, Any],
    review: dict[str, str],
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
    review_feedback = review.get("feedback", "").strip()
    if review_feedback:
        parts.extend(["", f"Reviewer feedback:\n{review_feedback}"])
    implementation_direction = candidate.get("implementation_direction")
    if isinstance(implementation_direction, str) and implementation_direction.strip():
        parts.extend(["", f"Implementation guidance:\n{implementation_direction.strip()}"])
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


def _autonomous_goal_proposal_summary(candidate: dict[str, Any], review: dict[str, str]) -> str:
    parts = []
    for label, value in (
        ("Summary", candidate.get("summary")),
        ("Reviewer feedback", review.get("feedback", "")),
        ("Implemented", candidate.get("implemented_changes")),
        ("Impact", candidate.get("impact")),
        ("Verification", candidate.get("verification")),
        ("Rough edges", candidate.get("rough_edges")),
    ):
        if isinstance(value, str) and value.strip():
            parts.append(f"{label}: {value.strip()}")
    return "\n\n".join(parts)


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
