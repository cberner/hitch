"""Prompt and proposal-history builders for the autonomous-goal candidate flow.

The autonomous-goal workflow proposes, summarizes, and continues hidden
candidate sessions. This module owns the dependency-free prompt-text builders
over a candidate's accepted/dismissed proposal history, the proposed-session
continuation prompt, the candidate/judge/retry prompts and their workflow-state
context accessors, and the ambition-guidance copy those share.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from django.utils import timezone

from hitch.main.models import AutonomousGoal, AutonomousGoalMemory, ProposedSession, SystemWorkflow
from hitch.main.runtime.sdk_values import truncate_for_prompt
from hitch.main.workflows.agent_io import (
    _AUTONOMOUS_GOAL_MEMORY_COMPACT_RECENT_COUNT,
    _AUTONOMOUS_GOAL_TITLE_MAX_LEN,
    _autonomous_goal_history_sections,
    _autonomous_goal_memory_context,
    _AutonomousGoalMemoryPromptContext,
    _format_limited_strings,
    _merge_string_lists,
    _split_autonomous_goal_history,
    _string_list,
    _write_autonomous_goal_history_files,
)
from hitch.main.workflows.workflow_state import (
    _session_metadata_from_state,
    _state_dict,
    _state_int,
    _state_string,
)

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
_AUTONOMOUS_GOAL_PROPOSAL_HISTORY_SUMMARY_STATE_KEY = "proposal_history_summary"

_AUTONOMOUS_GOAL_CANDIDATE_HISTORY_CONTEXT_CHARS = 5_000
_AUTONOMOUS_GOAL_CANDIDATE_HISTORY_MAX_ROWS = 50
_AUTONOMOUS_GOAL_CANDIDATE_HISTORY_RECENT_SUMMARY_CHARS = 650
_AUTONOMOUS_GOAL_CANDIDATE_HISTORY_OLDER_SUMMARY_CHARS = 260
_AUTONOMOUS_GOAL_CANDIDATE_HISTORY_PROMPT_CHARS = 260
_AUTONOMOUS_GOAL_CANDIDATE_HISTORY_FILE_LIMIT = 6
_AUTONOMOUS_GOAL_CANDIDATE_HISTORY_FILE_CHARS = 600
_AUTONOMOUS_GOAL_RECENT_PROPOSAL_REFERENCE_LIMIT = 5
_AUTONOMOUS_GOAL_PROPOSAL_HISTORY_SUMMARY_CHARS = 5_000


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


def _autonomous_goal_candidate_proposal_history_prompt_text(
    workflow: SystemWorkflow,
    fallback_context: _AutonomousGoalProposalHistoryPromptContext,
) -> str:
    summary = _state_dict(workflow, _AUTONOMOUS_GOAL_PROPOSAL_HISTORY_SUMMARY_STATE_KEY)
    summary_text = _format_autonomous_goal_proposal_history_summary(summary)
    return summary_text if summary_text else fallback_context.text


def _format_autonomous_goal_proposal_history_summary(
    summary: dict[str, Any],
) -> str:
    brief = summary.get("brief")
    if not isinstance(brief, str) or not brief.strip():
        return ""
    parts = [f"Brief: {brief.strip()}"]
    for label, key in (
        ("Recent stack", "recent_stack"),
        ("Accepted lessons", "accepted_lessons"),
        ("Avoid or reconsider", "avoid_or_reconsider"),
        ("Promising next directions", "promising_next_directions"),
        ("Important files", "important_files"),
    ):
        values = _string_list(summary.get(key))
        if values:
            parts.append(f"{label}:\n" + "\n".join(f"- {value}" for value in values))
    return truncate_for_prompt(
        "\n\n".join(parts), _AUTONOMOUS_GOAL_PROPOSAL_HISTORY_SUMMARY_CHARS
    )


def _autonomous_goal_recent_proposal_run_references(
    autonomous_goal: AutonomousGoal,
    *,
    limit: int = _AUTONOMOUS_GOAL_RECENT_PROPOSAL_REFERENCE_LIMIT,
) -> str:
    proposals = list(
        autonomous_goal.proposed_sessions.filter(
            inbox_kind=ProposedSession.INBOX_KIND_PROPOSAL
        )
        .exclude(outcome_status=ProposedSession.OUTCOME_UNSET)
        .select_related("candidate_session", "accepted_session", "judge_session")
        .order_by("-updated_at", "-id")[:limit]
    )
    if not proposals:
        return "(none)"
    return "\n\n".join(_format_recent_proposal_run_reference(p) for p in proposals)


def _format_recent_proposal_run_reference(proposal: ProposedSession) -> str:
    metadata = (
        proposal.outcome_metadata if isinstance(proposal.outcome_metadata, dict) else {}
    )
    stack_round = ""
    iteration = metadata.get("stacked_diff_iteration")
    depth = metadata.get("stacked_diff_depth")
    if isinstance(iteration, int) and isinstance(depth, int):
        stack_round = f", stack round {iteration} of {depth}"
    files = _format_limited_strings(
        _string_list(proposal.relevant_files), limit=8, max_chars=600
    )
    parts = [
        (
            f"Proposal #{proposal.pk}: {proposal.title or '(none)'} "
            f"({proposal.outcome_status or '(none)'}{stack_round}, "
            f"updated {_proposal_updated_date(proposal)})"
        ),
        _format_session_reference("Candidate", proposal.candidate_session),
    ]
    if (
        proposal.accepted_session is not None
        and proposal.accepted_session_id != proposal.candidate_session_id
    ):
        parts.append(_format_session_reference("Accepted", proposal.accepted_session))
    if proposal.judge_session is not None:
        parts.append(_format_session_reference("Judge", proposal.judge_session))
    if files:
        parts.append(f"Relevant files: {files}")
    if proposal.outcome_notes.strip():
        parts.append(
            f"Outcome notes: {truncate_for_prompt(proposal.outcome_notes, 180)}"
        )
    return "\n".join(parts)


def _format_session_reference(label: str, session: Any | None) -> str:
    if session is None:
        return f"{label}: (none)"
    thread_id = session.thread_id or "(none)"
    path = session.codex_path.strip() if session.codex_path else ""
    if path:
        return f"{label}: thread {thread_id}; session file {path}"
    return f"{label}: thread {thread_id}; session file (none)"


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


def _autonomous_goal_candidate_prompt(
    workflow: SystemWorkflow, autonomous_goal: AutonomousGoal
) -> tuple[
    str,
    _AutonomousGoalMemoryPromptContext,
    _AutonomousGoalProposalHistoryPromptContext,
]:
    ambition = _autonomous_goal_ambition_guidance(autonomous_goal)
    memory_context = _autonomous_goal_memory_context(autonomous_goal)
    proposal_history_context = _autonomous_goal_candidate_proposal_history_context(
        autonomous_goal
    )
    proposal_history_text = _autonomous_goal_candidate_proposal_history_prompt_text(
        workflow, proposal_history_context
    )
    proposal_run_references = _autonomous_goal_recent_proposal_run_references(
        autonomous_goal
    )
    session_cwd = _autonomous_goal_session_cwd(workflow)
    stacked_depth = _autonomous_goal_workflow_stacked_diff_depth(
        workflow, autonomous_goal
    )
    stacked_iteration = _autonomous_goal_stack_iteration(workflow)
    if not _autonomous_goal_candidate_allows_code_changes(workflow):
        code_change_guidance = "Do not make code changes. "
    elif stacked_depth > 1:
        code_change_guidance = (
            "Make code changes that turn the proposal into real, reviewable "
            "progress; leave any changes in this session checkout so Hitch can "
            "continue from them. This autonomous goal is configured for stacked "
            f"diff depth {stacked_depth}; this is candidate round "
            f"{stacked_iteration} of {stacked_depth}. "
            "Before returning a proposal, polish the work you changed in this "
            "round: resolve obvious rough edges, keep the diff coherent, and "
            "run relevant checks when practical. Do not push a branch or open a "
            "PR. If this round is accepted and more depth remains, Hitch will "
            "start another hidden candidate round from this proposal branch. "
        )
    else:
        code_change_guidance = (
            "Make code changes that turn the proposal into real, reviewable "
            "progress; leave any changes in this session checkout so the user can "
            "accept and continue from them. Do not run QA loops or polish this "
            "as a finished PR; the continuation session will do that after user "
            "approval. "
        )
    stack_context = (
        f"Stacked diff round: {stacked_iteration} of {stacked_depth}\n"
        if stacked_depth > 1
        else ""
    )
    prompt = (
        "You are Hitch's autonomous goal agent.\n\n"
        "Thoroughly analyze the codebase and find one way to make "
        f"{ambition.candidate_progress} toward the autonomous goal. "
        f"{code_change_guidance}"
        "Focus on a concrete session that a user could accept and continue from. "
        "Use autonomous-goal memory to avoid repeating recently proposed, skipped, "
        "or processed files unless repeating one is clearly the best next step. "
        "Do not stop just because an optional host command is missing; use an "
        "available fallback, such as Python standard-library tooling for SQLite, "
        "or report the limitation in the JSON output.\n\n"
        f"Repository cwd: {session_cwd}\n"
        f"{stack_context}"
        f"Autonomous goal title: {autonomous_goal.title}\n\n"
        "Autonomous goal objective:\n"
        f"{autonomous_goal.goal}\n\n"
        "Autonomous goal memory from previous candidate runs:\n"
        f"{memory_context.text}\n\n"
        "Accepted/dismissed proposal history summary for candidate planning:\n"
        f"{proposal_history_text}\n\n"
        "Recent proposal run references for optional deeper review:\n"
        f"{proposal_run_references}\n\n"
        "Return only JSON matching this shape: "
        '{"proposal": {"title": string, "summary": string, "impact": string, '
        '"implemented_changes": string, "implementation_direction": string, '
        '"verification": string, "rough_edges": string, '
        '"suggested_continuation": string, "relevant_files": [string]} | null, '
        '"message": string, "next_steps_summary": string, '
        '"memory_relevant_files": [string]}. If you find a concrete proposal, '
        'put it in "proposal" and leave "message" empty. If you find nothing '
        'worth proposing, set "proposal" to null and put a concise user-facing '
        'explanation in "message". The title should be concise. The summary '
        "should explain the proposed session. Impact should describe the likely "
        "user-visible or engineering benefit. Implemented changes should "
        "summarize the concrete code changes already made in this hidden "
        "rollout. Implementation direction should be specific enough for the "
        "user to continue the work in this session. Verification should list "
        "checks you attempted, or say not run. Rough edges should call out "
        "known incompleteness. Suggested continuation should be the editable "
        "message to send if the user accepts the proposal. "
        "The next_steps_summary is durable memory for future autonomous-goal runs: "
        "mention what you inspected or selected, specific files or areas involved, "
        "what you proposed or skipped, and what a future run should try next. "
        "memory_relevant_files should list repo-relative files this run selected, "
        "inspected, or intentionally skipped so future runs can avoid accidental "
        "repetition. "
        f"{ambition.candidate_instruction}"
    )
    return prompt, memory_context, proposal_history_context


def _autonomous_goal_candidate_retry_prompt(
    workflow: SystemWorkflow, autonomous_goal: AutonomousGoal
) -> str:
    ambition = _autonomous_goal_ambition_guidance(autonomous_goal)
    candidate_session = _session_metadata_from_state(workflow, "candidate_session_id")
    session_cwd = (
        candidate_session.cwd
        if candidate_session is not None and candidate_session.cwd
        else _autonomous_goal_session_cwd(workflow)
    )
    stacked_depth = _autonomous_goal_workflow_stacked_diff_depth(
        workflow, autonomous_goal
    )
    stack_context = (
        f"Stacked diff round: {_autonomous_goal_stack_iteration(workflow)} "
        f"of {stacked_depth}\n"
        if stacked_depth > 1
        else ""
    )
    code_change_guidance = (
        "Do not make code changes. "
        if not _autonomous_goal_candidate_allows_code_changes(workflow)
        else (
            "Continue from the current checkout. Keep useful changes from the "
            "prior attempt, revise or remove changes that caused the failure, "
            "and leave the result in this hidden candidate checkout. Do not "
            "push a branch or open a PR. "
        )
    )
    return (
        "You are Hitch's autonomous goal agent.\n\n"
        "Continue this autonomous-goal candidate attempt from the current hidden "
        "candidate thread and checkout. The last attempt did not produce an "
        "accepted proposal. Use the failure context below to avoid repeating "
        f"the same failure and find one way to make {ambition.candidate_progress} "
        "toward the autonomous goal. "
        f"{code_change_guidance}"
        "Focus on a concrete session that a user could accept and continue from.\n\n"
        f"Repository cwd: {session_cwd}\n"
        f"{stack_context}"
        f"Autonomous goal title: {autonomous_goal.title}\n"
        f"Proposal budget: {_autonomous_goal_workflow_proposal_budget(workflow)} tokens\n"
        "Proposal budget tokens used so far: "
        f"{_autonomous_goal_proposal_budget_tokens_used(workflow)}\n"
        f"Failed proposal attempts so far: {_autonomous_goal_failed_attempts(workflow)}\n\n"
        "Autonomous goal objective:\n"
        f"{autonomous_goal.goal}\n\n"
        "Last failed attempt context:\n"
        f"{_format_autonomous_goal_last_failure_context(workflow)}\n\n"
        "Return only JSON matching this shape: "
        '{"proposal": {"title": string, "summary": string, "impact": string, '
        '"implemented_changes": string, "implementation_direction": string, '
        '"verification": string, "rough_edges": string, '
        '"suggested_continuation": string, "relevant_files": [string]} | null, '
        '"message": string, "next_steps_summary": string, '
        '"memory_relevant_files": [string]}. If you find a concrete proposal, '
        'put it in "proposal" and leave "message" empty. If you still find '
        'nothing worth proposing, set "proposal" to null and put a concise '
        'user-facing explanation in "message". The next_steps_summary is '
        "durable memory for future autonomous-goal runs: mention what failed, "
        "what you changed or inspected on this retry, and what a future run "
        "should try next. "
        f"{ambition.candidate_instruction}"
    )


def _format_autonomous_goal_last_failure_context(workflow: SystemWorkflow) -> str:
    failure = _state_dict(workflow, _AUTONOMOUS_GOAL_LAST_FAILURE_STATE_KEY)
    if not failure:
        return "(none)"
    return truncate_for_prompt(json.dumps(failure, indent=2, sort_keys=True), 5000)


def _autonomous_goal_judge_prompt(
    workflow: SystemWorkflow, autonomous_goal: AutonomousGoal, candidate: dict[str, Any]
) -> tuple[str, list[str]]:
    history_sections = _autonomous_goal_history_sections(autonomous_goal)
    inline_history, overflow_history = _split_autonomous_goal_history(history_sections)
    history_files = _write_autonomous_goal_history_files(workflow, overflow_history)
    history_file_text = (
        "\n".join(f"- {path}" for path in history_files) if history_files else "(none)"
    )
    ambition = _autonomous_goal_ambition_guidance(autonomous_goal)
    candidate_text = json.dumps(candidate, indent=2, sort_keys=True)
    candidate_session = _session_metadata_from_state(workflow, "candidate_session_id")
    candidate_thread_id = (
        candidate_session.thread_id if candidate_session is not None else "(unknown)"
    )
    session_cwd = _autonomous_goal_session_cwd(workflow)
    stacked_depth = _autonomous_goal_workflow_stacked_diff_depth(
        workflow, autonomous_goal
    )
    stack_context = (
        "Stacked diff round: "
        f"{_autonomous_goal_stack_iteration(workflow)} of {stacked_depth}\n"
        if stacked_depth > 1
        else ""
    )
    return (
        "You are Hitch's autonomous goal confidence judge.\n\n"
        "Judge whether the candidate session is likely to make meaningful "
        f"{ambition.judge_progress} toward the autonomous goal. "
        "Use the autonomous goal's "
        "accepted and rejected proposal history to calibrate your judgment. "
        "Do not reward broad or vague ideas; confidence should reflect whether "
        f"the proposal is concrete and well-scoped. {ambition.judge_instruction}\n\n"
        f"Repository cwd: {session_cwd}\n"
        f"{stack_context}"
        f"Autonomous goal title: {autonomous_goal.title}\n"
        f"Confidence threshold: {autonomous_goal.confidence_threshold}\n\n"
        "Autonomous goal objective:\n"
        f"{autonomous_goal.goal}\n\n"
        "Candidate session JSON:\n"
        f"Candidate session ID: {candidate_thread_id}\n"
        f"{candidate_text}\n\n"
        "Accepted/rejected proposal history included inline:\n"
        f"{inline_history or '(none)'}\n\n"
        "Additional history files:\n"
        f"{history_file_text}\n\n"
        "Return only JSON matching this shape: "
        '{"confidence": "medium" | "high" | "very_high", '
        '"summary": string, "rationale": string}. Summary is shown to the user '
        "in the inbox and should explain the expected impact."
    ), history_files

def _store_autonomous_goal_memory(
    autonomous_goal: AutonomousGoal,
    workflow: SystemWorkflow,
    candidate_output: dict[str, Any],
) -> AutonomousGoalMemory:
    proposal = candidate_output.get("proposal")
    title = ""
    proposal_files: list[str] = []
    if isinstance(proposal, dict):
        title = str(proposal.get("title") or "").strip()
        proposal_files = _string_list(proposal.get("relevant_files"))
    if not title:
        title = f"No proposal from {autonomous_goal.title}"
    summary = str(candidate_output.get("next_steps_summary") or "").strip()
    if not summary:
        summary = str(candidate_output.get("message") or "").strip()
    memory_files = _merge_string_lists(
        _string_list(candidate_output.get("memory_relevant_files")),
        proposal_files,
    )
    memory, _created = AutonomousGoalMemory.objects.update_or_create(
        source_workflow=workflow,
        defaults={
            "autonomous_goal": autonomous_goal,
            "candidate_session": _session_metadata_from_state(
                workflow, "candidate_session_id"
            ),
            "title": title[:_AUTONOMOUS_GOAL_TITLE_MAX_LEN],
            "summary": summary,
            "relevant_files": memory_files,
        },
    )
    return memory
