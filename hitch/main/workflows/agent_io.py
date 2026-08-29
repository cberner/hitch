"""Parsing and prompt-context assembly for Hitch-owned background agents.

This module owns the dependency-free transforms over system-agent Codex run
output: the ``_parse_*`` family that turns raw text/JSON into structured dicts,
and the autonomous-goal memory/history helpers that assemble the prior-run
context blocks injected into candidate prompts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from django.utils import timezone

from hitch.main.models import (
    AutonomousGoal,
    AutonomousGoalMemory,
    ProposedSession,
    SystemWorkflow,
)
from hitch.main.runtime import codex_pool
from hitch.main.runtime.sdk_values import truncate_for_prompt
from hitch.main.workflows.pr_handoff import _compact_pr_handoff

_AUTONOMOUS_GOAL_INLINE_HISTORY_CHARS = 10_000
_AUTONOMOUS_GOAL_MEMORY_CONTEXT_CHARS = 10_000
_AUTONOMOUS_GOAL_MEMORY_MAX_ROWS = 200
_AUTONOMOUS_GOAL_MEMORY_COMPACT_RECENT_COUNT = 8
_AUTONOMOUS_GOAL_MEMORY_FULL_SUMMARY_CHARS = 700
_AUTONOMOUS_GOAL_MEMORY_COMPACT_SUMMARY_CHARS = 180
_AUTONOMOUS_GOAL_MEMORY_FILE_LIMIT = 80
_AUTONOMOUS_GOAL_MEMORY_FILE_SUMMARY_CHARS = 1_200
_AUTONOMOUS_GOAL_MEMORY_LINE_FILE_LIMIT = 4
_AUTONOMOUS_GOAL_MEMORY_FIT_RECENT_SUMMARY_CHARS = 420
_AUTONOMOUS_GOAL_TITLE_MAX_LEN = 200
_CONFIDENCE_RANK = {
    AutonomousGoal.CONFIDENCE_MEDIUM: 1,
    AutonomousGoal.CONFIDENCE_HIGH: 2,
    AutonomousGoal.CONFIDENCE_VERY_HIGH: 3,
}


@dataclass(frozen=True)
class _AutonomousGoalMemoryPromptContext:
    text: str
    count: int
    compacted: bool


def _autonomous_goal_memory_context(
    autonomous_goal: AutonomousGoal,
) -> _AutonomousGoalMemoryPromptContext:
    memory_queryset = autonomous_goal.memories.select_related(
        "candidate_session"
    ).order_by("-created_at", "-id")
    total_count = memory_queryset.count()
    if total_count == 0:
        return _AutonomousGoalMemoryPromptContext("(none)", 0, False)

    memories = list(memory_queryset[:_AUTONOMOUS_GOAL_MEMORY_MAX_ROWS])
    omitted_count = max(total_count - len(memories), 0)
    full_parts: list[str] = []
    used_chars = 0
    full_context_fits = omitted_count == 0
    for memory in memories:
        section = _format_autonomous_goal_memory(
            memory, summary_chars=_AUTONOMOUS_GOAL_MEMORY_FULL_SUMMARY_CHARS
        )
        section_chars = len(section) + (2 if full_parts else 0)
        if used_chars + section_chars > _AUTONOMOUS_GOAL_MEMORY_CONTEXT_CHARS:
            full_context_fits = False
            break
        full_parts.append(section)
        used_chars += section_chars

    if full_context_fits:
        return _AutonomousGoalMemoryPromptContext(
            "\n\n".join(full_parts), total_count, False
        )
    return _AutonomousGoalMemoryPromptContext(
        _compact_autonomous_goal_memories(memories, omitted_count=omitted_count),
        total_count,
        True,
    )


def _format_autonomous_goal_memory(
    memory: AutonomousGoalMemory,
    *,
    summary_chars: int,
    file_limit: int | None = None,
    file_chars: int | None = None,
) -> str:
    candidate_id = (
        memory.candidate_session.thread_id if memory.candidate_session else "(none)"
    )
    files = _string_list(memory.relevant_files)
    file_text = (
        _format_limited_strings(files, limit=file_limit, max_chars=file_chars)
        if file_limit is not None
        else ", ".join(files)
    )
    return (
        f"Memory ID: {memory.pk}\n"
        f"Created: {_memory_created_date(memory)}\n"
        f"Candidate session ID: {candidate_id}\n"
        f"Title: {memory.title or '(none)'}\n"
        f"Relevant files: {file_text if file_text else '(none)'}\n"
        f"Next steps summary: {truncate_for_prompt(memory.summary, summary_chars)}"
    )


def _compact_autonomous_goal_memories(
    memories: list[AutonomousGoalMemory], *, omitted_count: int = 0
) -> str:
    total_count = len(memories) + omitted_count
    recent = memories[:_AUTONOMOUS_GOAL_MEMORY_COMPACT_RECENT_COUNT]
    older = memories[_AUTONOMOUS_GOAL_MEMORY_COMPACT_RECENT_COUNT:]
    files = _format_limited_strings(
        _autonomous_goal_memory_files(memories),
        limit=_AUTONOMOUS_GOAL_MEMORY_FILE_LIMIT,
        max_chars=min(
            _AUTONOMOUS_GOAL_MEMORY_FILE_SUMMARY_CHARS,
            max(80, _AUTONOMOUS_GOAL_MEMORY_CONTEXT_CHARS // 8),
        ),
    )
    sections = [
        (
            f"Compacted from {total_count} prior candidate summaries because "
            "the full autonomous-goal memory would consume too much context."
        ),
        f"Files seen across prior runs: {files or '(none)'}",
        "Recent detailed summaries:",
        "\n\n".join(
            _format_autonomous_goal_memory(
                memory, summary_chars=_AUTONOMOUS_GOAL_MEMORY_COMPACT_SUMMARY_CHARS
            )
            for memory in recent
        ),
    ]
    if older:
        sections.extend(
            [
                "Older compacted summaries:",
                "\n".join(
                    _format_autonomous_goal_memory_line(
                        memory,
                        summary_chars=_AUTONOMOUS_GOAL_MEMORY_COMPACT_SUMMARY_CHARS,
                    )
                    for memory in older
                ),
            ]
        )
    if omitted_count:
        sections.append(
            f"{omitted_count} older memory rows are outside this prompt cap."
        )
    compacted = "\n\n".join(section for section in sections if section)
    if len(compacted) <= _AUTONOMOUS_GOAL_MEMORY_CONTEXT_CHARS:
        return _cap_autonomous_goal_memory_context(compacted)
    return _fit_autonomous_goal_memory_context(
        memories, files, omitted_count=omitted_count
    )


def _fit_autonomous_goal_memory_context(
    memories: list[AutonomousGoalMemory], file_summary: str, *, omitted_count: int = 0
) -> str:
    header = (
        f"Compacted from {len(memories) + omitted_count} prior candidate summaries.\n"
        f"Files seen across prior runs: {file_summary or '(none)'}\n"
        "Recent detailed summaries:"
    )
    parts = [header]
    used_chars = len(header)
    selected_count = 0
    recent = memories[:_AUTONOMOUS_GOAL_MEMORY_COMPACT_RECENT_COUNT]
    older = memories[_AUTONOMOUS_GOAL_MEMORY_COMPACT_RECENT_COUNT:]

    for memory in recent:
        section = _format_autonomous_goal_memory(
            memory,
            summary_chars=_AUTONOMOUS_GOAL_MEMORY_FIT_RECENT_SUMMARY_CHARS,
            file_limit=_AUTONOMOUS_GOAL_MEMORY_LINE_FILE_LIMIT,
            file_chars=180,
        )
        section_chars = len(section) + 2
        if used_chars + section_chars > _AUTONOMOUS_GOAL_MEMORY_CONTEXT_CHARS:
            line = _format_autonomous_goal_memory_line(
                memory,
                summary_chars=_AUTONOMOUS_GOAL_MEMORY_FIT_RECENT_SUMMARY_CHARS,
            )
            line_chars = len(line) + 2
            if used_chars + line_chars > _AUTONOMOUS_GOAL_MEMORY_CONTEXT_CHARS:
                break
            parts.append(line)
            used_chars += line_chars
            selected_count += 1
            continue
        parts.append(section)
        used_chars += section_chars
        selected_count += 1

    older_started = False
    for memory in older:
        line = _format_autonomous_goal_memory_line(
            memory, summary_chars=_AUTONOMOUS_GOAL_MEMORY_COMPACT_SUMMARY_CHARS
        )
        prefix = "\nOlder compacted summaries:\n" if not older_started else "\n"
        line_chars = len(prefix) + len(line)
        if used_chars + line_chars > _AUTONOMOUS_GOAL_MEMORY_CONTEXT_CHARS:
            break
        if not older_started:
            parts.append("Older compacted summaries:")
            used_chars += len("\n\nOlder compacted summaries:")
            older_started = True
        parts.append(line)
        used_chars += len(line) + 1
        selected_count += 1

    omitted = len(memories) - selected_count + omitted_count
    if omitted > 0:
        marker = f"- {omitted} older summaries omitted after compaction."
        marker_chars = len(marker) + 1
        if used_chars + marker_chars <= _AUTONOMOUS_GOAL_MEMORY_CONTEXT_CHARS:
            parts.append(marker)
    return _cap_autonomous_goal_memory_context("\n\n".join(parts))


def _format_autonomous_goal_memory_line(
    memory: AutonomousGoalMemory, *, summary_chars: int
) -> str:
    files = _format_limited_strings(
        _string_list(memory.relevant_files),
        limit=_AUTONOMOUS_GOAL_MEMORY_LINE_FILE_LIMIT,
        max_chars=180,
    )
    return (
        f"- {_memory_created_date(memory)}: {memory.title or '(none)'}; "
        f"files: {files or '(none)'}; "
        f"next: {truncate_for_prompt(memory.summary, summary_chars)}"
    )


def _autonomous_goal_memory_files(memories: list[AutonomousGoalMemory]) -> list[str]:
    files: list[str] = []
    for memory in memories:
        files = _merge_string_lists(files, _string_list(memory.relevant_files))
    return files


def _memory_created_date(memory: AutonomousGoalMemory) -> str:
    return timezone.localtime(memory.created_at).date().isoformat()


def _format_limited_strings(
    values: list[str], *, limit: int, max_chars: int | None = None
) -> str:
    if len(values) <= limit:
        formatted = ", ".join(values)
    else:
        shown = ", ".join(values[:limit])
        formatted = f"{shown}, ... ({len(values) - limit} more)"
    if max_chars is None:
        return formatted
    return truncate_for_prompt(formatted, max_chars)


def _cap_autonomous_goal_memory_context(text: str) -> str:
    if len(text) <= _AUTONOMOUS_GOAL_MEMORY_CONTEXT_CHARS:
        return text
    marker = "\n... (autonomous-goal memory truncated to fit context budget)"
    if len(marker) >= _AUTONOMOUS_GOAL_MEMORY_CONTEXT_CHARS:
        return truncate_for_prompt(text, _AUTONOMOUS_GOAL_MEMORY_CONTEXT_CHARS)
    return (
        text[: _AUTONOMOUS_GOAL_MEMORY_CONTEXT_CHARS - len(marker)].rstrip()
        + marker
    )


def _autonomous_goal_history_sections(autonomous_goal: AutonomousGoal) -> list[str]:
    proposals = (
        autonomous_goal.proposed_sessions.filter(
            inbox_kind=ProposedSession.INBOX_KIND_PROPOSAL
        )
        .exclude(outcome_status=ProposedSession.OUTCOME_UNSET)
        .select_related("candidate_session", "accepted_session")
        .order_by("-updated_at", "-id")[:50]
    )
    return [_format_proposed_session_context(proposal) for proposal in proposals]


def _format_proposed_session_context(proposal: ProposedSession) -> str:
    files = _string_list(proposal.relevant_files)
    candidate_id = (
        proposal.candidate_session.thread_id if proposal.candidate_session else "(none)"
    )
    accepted_id = (
        proposal.accepted_session.thread_id if proposal.accepted_session else "(none)"
    )
    outcome_metadata = (
        json.dumps(proposal.outcome_metadata, sort_keys=True)
        if isinstance(proposal.outcome_metadata, dict)
        and proposal.outcome_metadata
        else "(none)"
    )
    notes_label = (
        "Reject reason"
        if proposal.outcome_status == ProposedSession.OUTCOME_REJECTED
        else "Outcome notes"
    )
    return (
        f"ProposedSession ID: {proposal.pk}\n"
        f"Candidate session ID: {candidate_id}\n"
        f"Accepted session ID: {accepted_id}\n"
        f"Title: {proposal.title}\n"
        f"Confidence: {proposal.confidence}\n"
        f"Summary: {proposal.summary or '(none)'}\n"
        f"Prompt: {proposal.prompt or '(none)'}\n"
        f"Relevant files: {', '.join(files) if files else '(none)'}\n"
        f"Outcome status: {proposal.outcome_status}\n"
        f"Outcome metadata: {outcome_metadata}\n"
        f"{notes_label}: {proposal.outcome_notes or '(none)'}"
    )


def _split_autonomous_goal_history(sections: list[str]) -> tuple[str, list[str]]:
    inline_parts: list[str] = []
    overflow: list[str] = []
    used_chars = 0
    for section in sections:
        section_chars = len(section) + 2
        if used_chars + section_chars <= _AUTONOMOUS_GOAL_INLINE_HISTORY_CHARS:
            inline_parts.append(section)
            used_chars += section_chars
        else:
            overflow.append(section)
    return "\n\n".join(inline_parts), overflow


def _write_autonomous_goal_history_files(
    workflow: SystemWorkflow, sections: list[str]
) -> list[str]:
    if not sections:
        return []
    directory = codex_pool.events_dir() / "autonomous_goal_history" / str(workflow.pk)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "proposal_history.txt"
    path.write_text("\n\n---\n\n".join(sections), encoding="utf-8")
    return [str(path)]


def _parse_json_object(raw_output: str) -> dict[str, Any] | None:
    text = _strip_json_markdown_fence(raw_output)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_qa_output(raw_output: str) -> dict[str, Any] | None:
    parsed = _parse_json_object(raw_output)
    if parsed is None:
        return None
    feedback = parsed.get("feedback")
    lgtm = parsed.get("lgtm")
    if not isinstance(feedback, str) or not isinstance(lgtm, bool):
        return None
    return {"feedback": feedback, "lgtm": lgtm}


def _parse_codex_review_output(
    raw_output: str, review_output: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Translate Codex's structured native review into Hitch's QA verdict."""
    feedback = raw_output.strip()
    if not feedback or review_output is None:
        return None
    findings = review_output.get("findings")
    correctness = review_output.get("overall_correctness")
    if not isinstance(findings, list) or not all(
        isinstance(finding, dict) for finding in findings
    ):
        return None
    if findings:
        return {"feedback": feedback, "lgtm": False}
    if correctness == "patch is correct":
        return {"feedback": feedback, "lgtm": True}
    return None


def _parse_autonomous_goal_candidate_output(raw_output: str) -> dict[str, Any] | None:
    parsed = _parse_json_object(raw_output)
    if parsed is None:
        return None
    if "proposal" in parsed:
        proposal = parsed.get("proposal")
        message = parsed.get("message")
        if proposal is None:
            if not isinstance(message, str) or not message.strip():
                return None
            memory_summary = _candidate_memory_summary(parsed, None, message)
            return {
                "proposal": None,
                "message": message.strip(),
                "next_steps_summary": memory_summary,
                "memory_relevant_files": _string_list(
                    parsed.get("memory_relevant_files")
                ),
            }
        if not isinstance(proposal, dict):
            return None
        normalized = _parse_autonomous_goal_candidate_proposal(proposal)
        if normalized is None:
            return None
        return {
            "proposal": normalized,
            "message": "",
            "next_steps_summary": _candidate_memory_summary(
                parsed, normalized, message if isinstance(message, str) else ""
            ),
            "memory_relevant_files": _merge_string_lists(
                _string_list(parsed.get("memory_relevant_files")),
                _string_list(normalized.get("relevant_files")),
            ),
        }
    normalized = _parse_autonomous_goal_candidate_proposal(parsed)
    if normalized is None:
        return None
    return {
        "proposal": normalized,
        "message": "",
        "next_steps_summary": _candidate_memory_summary(parsed, normalized, ""),
        "memory_relevant_files": _merge_string_lists(
            _string_list(parsed.get("memory_relevant_files")),
            _string_list(normalized.get("relevant_files")),
        ),
    }


def _candidate_memory_summary(
    parsed: dict[str, Any], proposal: dict[str, Any] | None, message: str
) -> str:
    summary = parsed.get("next_steps_summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    if proposal is not None:
        proposal_parts: list[str] = []
        for label, key in (
            ("Summary", "summary"),
            ("Implemented", "implemented_changes"),
            ("Impact", "impact"),
            ("Verification", "verification"),
            ("Rough edges", "rough_edges"),
            ("Suggested continuation", "suggested_continuation"),
        ):
            value = proposal.get(key)
            if isinstance(value, str) and value.strip():
                proposal_parts.append(f"{label}: {value.strip()}")
        if proposal_parts:
            return "\n\n".join(proposal_parts)
    return message.strip()


def _parse_autonomous_goal_candidate_proposal(
    parsed: dict[str, Any],
) -> dict[str, Any] | None:
    title = parsed.get("title")
    summary = parsed.get("summary")
    impact = parsed.get("impact")
    implemented_changes = parsed.get("implemented_changes")
    implementation_direction = parsed.get("implementation_direction")
    verification = parsed.get("verification")
    rough_edges = parsed.get("rough_edges")
    suggested_continuation = parsed.get("suggested_continuation")
    if not isinstance(title, str):
        return None
    if not isinstance(summary, str):
        return None
    if not isinstance(impact, str):
        return None
    if implemented_changes is not None and not isinstance(implemented_changes, str):
        return None
    if not isinstance(implementation_direction, str):
        return None
    if verification is not None and not isinstance(verification, str):
        return None
    if rough_edges is not None and not isinstance(rough_edges, str):
        return None
    if suggested_continuation is not None and not isinstance(
        suggested_continuation, str
    ):
        return None
    title = title.strip()
    if not title:
        return None
    return {
        "title": title,
        "summary": summary.strip(),
        "impact": impact.strip(),
        "implemented_changes": (
            implemented_changes.strip() if isinstance(implemented_changes, str) else ""
        ),
        "implementation_direction": implementation_direction.strip(),
        "verification": verification.strip() if isinstance(verification, str) else "",
        "rough_edges": rough_edges.strip() if isinstance(rough_edges, str) else "",
        "suggested_continuation": (
            suggested_continuation.strip()
            if isinstance(suggested_continuation, str)
            else ""
        ),
        "relevant_files": _string_list(parsed.get("relevant_files")),
    }


def _parse_autonomous_goal_history_summary_output(
    raw_output: str,
) -> dict[str, Any] | None:
    parsed = _parse_json_object(raw_output)
    if parsed is None:
        return None
    brief = parsed.get("brief")
    if not isinstance(brief, str) or not brief.strip():
        return None
    return {
        "brief": brief.strip(),
        "recent_stack": _string_list(parsed.get("recent_stack")),
        "accepted_lessons": _string_list(parsed.get("accepted_lessons")),
        "avoid_or_reconsider": _string_list(parsed.get("avoid_or_reconsider")),
        "promising_next_directions": _string_list(
            parsed.get("promising_next_directions")
        ),
        "important_files": _string_list(parsed.get("important_files")),
    }


def _parse_autonomous_goal_judge_output(raw_output: str) -> dict[str, str] | None:
    parsed = _parse_json_object(raw_output)
    if parsed is None:
        return None
    confidence = parsed.get("confidence")
    summary = parsed.get("summary")
    rationale = parsed.get("rationale")
    if confidence not in _CONFIDENCE_RANK:
        return None
    if not isinstance(summary, str) or not isinstance(rationale, str):
        return None
    return {
        "confidence": confidence,
        "summary": summary.strip(),
        "rationale": rationale.strip(),
    }


def _parse_pr_monitor_output(raw_output: str) -> dict[str, Any] | None:
    parsed = _parse_json_object(raw_output)
    if parsed is None:
        return None
    status = parsed.get("status")
    summary = parsed.get("summary")
    feedback = parsed.get("feedback")
    pr = parsed.get("pr")
    if status == "ready":
        status = "blocked"
    if status not in {"blocked", "terminal"}:
        return None
    if not isinstance(summary, str) or not isinstance(feedback, str):
        return None
    if not isinstance(pr, dict):
        return None
    return {
        "status": status,
        "summary": summary.strip(),
        "feedback": feedback.strip(),
        "pr": _compact_pr_handoff(pr),
        "blockers": _string_list(parsed.get("blockers")),
    }


def _strip_json_markdown_fence(raw_output: str) -> str:
    text = raw_output.strip()
    if text.startswith("```"):
        # Markdown code fences are delimited by ``\n`` only. Split on ``\n``
        # rather than ``str.splitlines``, which would also break on form feed,
        # vertical tab, and the Unicode line/paragraph/NEL separators -- all
        # of which are valid *inside* a JSON string value. ``splitlines`` tears
        # such a payload apart and the ``"\n".join`` below then rewrites the
        # separator as a literal newline, turning an otherwise-valid agent
        # verdict into invalid JSON and blocking the workflow. A trailing ``\r``
        # is dropped so a CRLF-fenced reply parses identically to an LF one.
        lines = [
            line[:-1] if line.endswith("\r") else line for line in text.split("\n")
        ]
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


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


def _merge_string_lists(*values: list[str]) -> list[str]:
    merged: list[str] = []
    for value in values:
        for item in value:
            item = item.strip()
            if item and item not in merged:
                merged.append(item)
    return merged
