"""Prompt and proposal-history builders for the autonomous-goal candidate flow.

The autonomous-goal workflow proposes, summarizes, and continues hidden
candidate sessions. This module owns the dependency-free prompt-text builders
over a candidate's accepted/dismissed proposal history, the proposed-session
continuation prompt, and the ambition-guidance copy those share.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from django.utils import timezone

from hitch.main.agent_io import (
    _AUTONOMOUS_GOAL_MEMORY_COMPACT_RECENT_COUNT,
    _format_limited_strings,
    _string_list,
)
from hitch.main.models import AutonomousGoal, ProposedSession
from hitch.main.sdk_values import truncate_for_prompt

_AUTONOMOUS_GOAL_CANDIDATE_HISTORY_CONTEXT_CHARS = 5_000
_AUTONOMOUS_GOAL_CANDIDATE_HISTORY_MAX_ROWS = 50
_AUTONOMOUS_GOAL_CANDIDATE_HISTORY_RECENT_SUMMARY_CHARS = 650
_AUTONOMOUS_GOAL_CANDIDATE_HISTORY_OLDER_SUMMARY_CHARS = 260
_AUTONOMOUS_GOAL_CANDIDATE_HISTORY_PROMPT_CHARS = 260
_AUTONOMOUS_GOAL_CANDIDATE_HISTORY_FILE_LIMIT = 6
_AUTONOMOUS_GOAL_CANDIDATE_HISTORY_FILE_CHARS = 600


@dataclass(frozen=True)
class _AutonomousGoalProposalHistoryPromptContext:
    text: str
    count: int
    compacted: bool


def _autonomous_goal_candidate_proposal_history_context(
    autonomous_goal: AutonomousGoal,
) -> _AutonomousGoalProposalHistoryPromptContext:
    proposal_queryset = (
        autonomous_goal.proposed_sessions.filter(
            inbox_kind=ProposedSession.INBOX_KIND_PROPOSAL
        )
        .exclude(outcome_status=ProposedSession.OUTCOME_UNSET)
        .select_related("candidate_session", "accepted_session")
        .order_by("-updated_at", "-id")
    )
    total_count = proposal_queryset.count()
    if total_count == 0:
        return _AutonomousGoalProposalHistoryPromptContext("(none)", 0, False)

    proposals = list(proposal_queryset[:_AUTONOMOUS_GOAL_CANDIDATE_HISTORY_MAX_ROWS])
    omitted_count = max(total_count - len(proposals), 0)
    parts: list[str] = []
    used_chars = 0
    compacted = omitted_count > 0
    for index, proposal in enumerate(proposals):
        summary_chars = (
            _AUTONOMOUS_GOAL_CANDIDATE_HISTORY_RECENT_SUMMARY_CHARS
            if index < _AUTONOMOUS_GOAL_MEMORY_COMPACT_RECENT_COUNT
            else _AUTONOMOUS_GOAL_CANDIDATE_HISTORY_OLDER_SUMMARY_CHARS
        )
        section = _format_autonomous_goal_candidate_proposal_history(
            proposal, summary_chars=summary_chars
        )
        section_chars = len(section) + (2 if parts else 0)
        if used_chars + section_chars > _AUTONOMOUS_GOAL_CANDIDATE_HISTORY_CONTEXT_CHARS:
            compacted = True
            if not parts:
                truncated_section = truncate_for_prompt(
                    section, _AUTONOMOUS_GOAL_CANDIDATE_HISTORY_CONTEXT_CHARS
                )
                parts.append(truncated_section)
                used_chars = len(truncated_section)
            break
        parts.append(section)
        used_chars += section_chars

    omitted_count += len(proposals) - len(parts)
    if omitted_count > 0:
        marker = f"{omitted_count} older proposal history rows omitted."
        marker_chars = len(marker) + (2 if parts else 0)
        if used_chars + marker_chars <= _AUTONOMOUS_GOAL_CANDIDATE_HISTORY_CONTEXT_CHARS:
            parts.append(marker)
        elif not parts:
            parts.append(
                truncate_for_prompt(
                    marker, _AUTONOMOUS_GOAL_CANDIDATE_HISTORY_CONTEXT_CHARS
                )
            )
    return _AutonomousGoalProposalHistoryPromptContext(
        "\n\n".join(parts), total_count, compacted
    )


def _format_autonomous_goal_candidate_proposal_history(
    proposal: ProposedSession, *, summary_chars: int
) -> str:
    files = _format_limited_strings(
        _string_list(proposal.relevant_files),
        limit=_AUTONOMOUS_GOAL_CANDIDATE_HISTORY_FILE_LIMIT,
        max_chars=min(
            _AUTONOMOUS_GOAL_CANDIDATE_HISTORY_FILE_CHARS,
            max(80, _AUTONOMOUS_GOAL_CANDIDATE_HISTORY_CONTEXT_CHARS // 8),
        ),
    )
    candidate_id = (
        proposal.candidate_session.thread_id if proposal.candidate_session else "(none)"
    )
    accepted_id = (
        proposal.accepted_session.thread_id if proposal.accepted_session else "(none)"
    )
    parts = [
        f"ProposedSession ID: {proposal.pk}",
        f"Updated: {_proposal_updated_date(proposal)}",
        f"Outcome status: {proposal.outcome_status or '(none)'}",
        f"Candidate session ID: {candidate_id}",
        f"Accepted session ID: {accepted_id}",
        f"Title: {proposal.title or '(none)'}",
    ]
    description = _autonomous_goal_candidate_proposal_description(proposal)
    if description:
        parts.append(f"Description: {truncate_for_prompt(description, summary_chars)}")
    continuation = proposal.prompt.strip()
    if continuation:
        parts.append(
            "Continuation prompt: "
            f"{truncate_for_prompt(continuation, _AUTONOMOUS_GOAL_CANDIDATE_HISTORY_PROMPT_CHARS)}"
        )
    if files:
        parts.append(f"Relevant files: {files}")
    if proposal.outcome_notes.strip():
        parts.append(
            f"Outcome notes: {truncate_for_prompt(proposal.outcome_notes, 180)}"
        )
    return "\n".join(parts)


def _autonomous_goal_candidate_proposal_description(
    proposal: ProposedSession,
) -> str:
    if proposal.summary.strip():
        return proposal.summary.strip()
    metadata = proposal.outcome_metadata
    if not isinstance(metadata, dict):
        return ""
    parts: list[str] = []
    for label, key in (
        ("Implemented", "implemented_changes"),
        ("Verification", "verification"),
        ("Rough edges", "rough_edges"),
    ):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(f"{label}: {value.strip()}")
    return "\n\n".join(parts)


def _proposal_updated_date(proposal: ProposedSession) -> str:
    return timezone.localtime(proposal.updated_at).date().isoformat()


def _autonomous_goal_proposed_session_prompt(
    autonomous_goal: AutonomousGoal, candidate: dict[str, Any], judgment: dict[str, str]
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
    summary = judgment.get("summary", "").strip()
    if summary:
        parts.extend(["", f"Summary:\n{summary}"])
    implementation_direction = candidate.get("implementation_direction")
    if isinstance(implementation_direction, str) and implementation_direction.strip():
        parts.extend(
            [
                "",
                f"Implementation guidance:\n{implementation_direction.strip()}",
            ]
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
        ("Summary", judgment.get("summary", "")),
        ("Implemented", candidate.get("implemented_changes")),
        ("Impact", candidate.get("impact")),
        ("Verification", candidate.get("verification")),
        ("Rough edges", candidate.get("rough_edges")),
    ):
        if isinstance(value, str) and value.strip():
            parts.append(f"{label}: {value.strip()}")
    return "\n\n".join(parts)


@dataclass(frozen=True)
class _AutonomousGoalAmbitionGuidance:
    candidate_progress: str
    candidate_instruction: str
    judge_progress: str
    judge_instruction: str


def _autonomous_goal_ambition_guidance(
    autonomous_goal: AutonomousGoal,
) -> _AutonomousGoalAmbitionGuidance:
    ambitions = {value for value, _label in AutonomousGoal.AMBITION_CHOICES}
    ambition = (
        autonomous_goal.ambition
        if autonomous_goal.ambition in ambitions
        else AutonomousGoal.AMBITION_INCREMENTAL
    )
    if ambition == AutonomousGoal.AMBITION_YOLO:
        return _AutonomousGoalAmbitionGuidance(
            candidate_progress="bold, high-leverage progress",
            candidate_instruction=(
                "For YOLO ambition, prefer a substantial session with clear "
                "upside over a cautious cleanup."
            ),
            judge_progress="bold, high-leverage progress",
            judge_instruction=(
                "For YOLO ambition, confidence should reflect whether this "
                "specific session is substantial and high-upside, not merely "
                "a small cleanup."
            ),
        )
    return _AutonomousGoalAmbitionGuidance(
        candidate_progress=f"{ambition} progress",
        candidate_instruction="",
        judge_progress=f"{ambition} progress",
        judge_instruction=(
            "Confidence should reflect whether this specific session is "
            "likely to advance the goal incrementally."
        ),
    )
