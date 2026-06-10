"""QA review prompt builders, design-synthesis gate, and feedback helpers.

The PR-QA workflow runs a review agent and, when feedback recurs, a design
synthesis gate. This module owns the dependency-free pieces of that cluster:
the QA review prompt, the recurring-feedback signal/match heuristics and their
regexes, the synthesis-gate builder and its feedback prompt, and the readers
that pull QA feedback out of a workflow's persisted state and prior runs.
"""

from __future__ import annotations

import re
from typing import Any

from hitch.main.models import SystemAgentRun, SystemWorkflow
from hitch.main.workflows.agent_io import _parse_qa_output
from hitch.main.workflows.workflow_state import _state_int

_QA_VERDICT_AGENT_KINDS = ("pr_qa",)

_QA_DESIGN_SYNTHESIS_STATE_KEY = "qa_design_synthesis_gate"
_QA_REVIEW_REVISION_STATE_KEY = "qa_review_revision"
_QA_DESIGN_SYNTHESIS_MIN_CATEGORY_OVERLAP = 2
_QA_DESIGN_SYNTHESIS_RECENT_RUN_LIMIT = 50
_QA_DESIGN_SYNTHESIS_MATCH_LIMIT = 3
_QA_DESIGN_FEEDBACK_SUMMARY_CHARS = 360
_QA_DESIGN_URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s`<>()\[\]]+", re.IGNORECASE)
_QA_DESIGN_FILE_RE = re.compile(
    r"(?<![\w.:/-])(?:[\w.-]+/)+[\w.-]+\.[A-Za-z0-9]+(?=$|[^\w/.-])"
    r"|(?<![\w.:/-])[\w.-]+\.(?:"
    r"bash|c|cc|cfg|conf|cpp|cs|css|cxx|fish|go|h|hpp|html|ini|java|js|json|"
    r"jsx|kt|lock|md|php|py|pyi|rb|rs|rst|sh|sql|svg|svelte|swift|toml|ts|"
    r"tsx|txt|vue|xml|yaml|yml|zsh"
    r")(?=$|[^\w/.-])",
    re.IGNORECASE,
)
_QA_DESIGN_PATH_RE = re.compile(
    r"(?<![\w.:/-])(?:[\w.-]+/)+[\w.-]+(?:\.[A-Za-z0-9]+)?/?(?=$|[^\w/.-])",
    re.IGNORECASE,
)
_QA_DESIGN_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_QA_DESIGN_KEYWORDS_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    "state_lifecycle": (
        "active",
        "cancelled",
        "cleanup",
        "duplicate",
        "generation",
        "in-flight",
        "pending",
        "race",
        "retry",
        "stale",
        "state",
        "superseded",
        "terminal",
        "overwrite",
    ),
    "authority_boundary": (
        "approval",
        "bypass",
        "permission",
        "sandbox",
        "security",
        "token",
    ),
    "persistence_contract": (
        "database",
        "migration",
        "persist",
        "schema",
        "stored",
        "upgrade",
    ),
    "streaming_visibility": (
        "browser",
        "live",
        "reload",
        "render",
        "status",
        "stream",
        "ui",
    ),
}
_CODEX_REVIEW_GUIDANCE = (
    "Apply the same review standards as Codex /review:\n"
    "- Flag only bugs or risks that meaningfully affect correctness, performance, "
    "security, or maintainability.\n"
    "- Each finding must be discrete, actionable, introduced by this diff, and "
    "something the author would likely fix.\n"
    "- Do not rely on unstated assumptions, speculative downstream breakage, or "
    "intentional behavior changes.\n"
    "- Ignore trivial style unless it obscures meaning or violates documented "
    "standards.\n"
    "- Do not stop at the first issue; keep reviewing until every qualifying "
    "finding is listed.\n"
    "- Prioritize findings as [P0], [P1], [P2], or [P3], using P0 only for "
    "universal release-blocking issues.\n"
    "- For each finding, include the shortest useful file/line reference that "
    "overlaps the diff and a one-paragraph explanation of why the issue matters.\n"
    "- If there are no qualifying findings, say that clearly rather than "
    "inventing nits.\n"
)


def _qa_prompt(cwd: str, diff_text: str) -> str:
    diff = diff_text or "(No current worktree diff was detected.)"
    return (
        "You are Hitch's QA agent for a PR workflow.\n\n"
        "Thoroughly review the current code diff before the PR agent runs its final "
        "cleanup/commit pass.\n\n"
        f"{_CODEX_REVIEW_GUIDANCE}\n"
        "Also do your own manual QA: if there is an interactive interface related "
        "to the diff, manually test it out and include concrete failures or gaps in "
        "your feedback. For browser QA, run `just qa-browser-setup` if Playwright "
        "or Chromium is missing, then use Playwright/Chromium to exercise the "
        "affected UI. If browser setup still fails, include that concrete setup "
        "failure in your feedback.\n\n"
        "Set lgtm to false when there are substantive findings, missing tests, or "
        "manual-QA failures the work agent should fix. Set lgtm to true only when "
        "the diff is ready for the PR agent to continue.\n\n"
        f"Repository cwd: {cwd}\n\n"
        "Current diff:\n"
        "```diff\n"
        f"{diff}\n"
        "```\n\n"
        "Return only JSON matching this shape: "
        '{"feedback": string, "lgtm": boolean}. Put the prioritized review '
        "findings, manual-QA results, or a clear no-findings statement in feedback."
    )


def _qa_review_revision(workflow: SystemWorkflow) -> int:
    return _state_int(workflow, _QA_REVIEW_REVISION_STATE_KEY)


def _maybe_build_qa_design_synthesis_gate(
    workflow: SystemWorkflow, feedback: str, *, current_run_id: int
) -> dict[str, Any] | None:
    if workflow.state.get(_QA_DESIGN_SYNTHESIS_STATE_KEY):
        return None
    current_signal = _qa_design_feedback_signal(feedback)
    if not current_signal["categories"]:
        return None

    matches: list[dict[str, Any]] = []
    recurring_categories: set[str] = set()
    recurring_files: set[str] = set()
    recent_runs = (
        SystemAgentRun.objects.filter(
            workflow__kind=SystemWorkflow.KIND_PR_QA,
            workflow__cwd=workflow.cwd,
            agent_kind__in=_QA_VERDICT_AGENT_KINDS,
            status=SystemAgentRun.STATUS_COMPLETED,
        )
        .exclude(pk=current_run_id)
        .select_related("workflow")
        .order_by("-created_at")[:_QA_DESIGN_SYNTHESIS_RECENT_RUN_LIMIT]
    )
    for prior_run in recent_runs:
        prior_feedback = _qa_feedback_from_run(prior_run)
        if not prior_feedback:
            continue
        prior_signal = _qa_design_feedback_signal(prior_feedback)
        category_overlap = current_signal["categories"] & prior_signal["categories"]
        file_overlap = current_signal["files"] & prior_signal["files"]
        same_workflow = prior_run.workflow_id == workflow.pk
        if not _qa_design_signals_match(
            category_overlap=category_overlap,
            file_overlap=file_overlap,
            same_workflow=same_workflow,
        ):
            continue
        recurring_categories.update(category_overlap)
        recurring_files.update(file_overlap)
        matches.append(
            {
                "workflow_id": prior_run.workflow_id,
                "run_id": prior_run.pk,
                "same_workflow": same_workflow,
                "categories": sorted(category_overlap),
                "files": sorted(file_overlap),
                "feedback": _summarize_qa_feedback(prior_feedback),
            }
        )
        if len(matches) >= _QA_DESIGN_SYNTHESIS_MATCH_LIMIT:
            break

    if not matches:
        return None
    if (
        workflow.iteration < 1
        and len(recurring_categories) < _QA_DESIGN_SYNTHESIS_MIN_CATEGORY_OVERLAP
    ):
        return None
    return {
        "triggered_at_iteration": workflow.iteration + 1,
        "current_categories": sorted(current_signal["categories"]),
        "current_files": sorted(current_signal["files"]),
        "recurring_categories": sorted(recurring_categories),
        "recurring_files": sorted(recurring_files),
        "matches": matches,
    }


def _qa_design_signals_match(
    *,
    category_overlap: set[str],
    file_overlap: set[str],
    same_workflow: bool,
) -> bool:
    if file_overlap and category_overlap:
        return True
    if len(category_overlap) >= _QA_DESIGN_SYNTHESIS_MIN_CATEGORY_OVERLAP:
        return True
    return same_workflow and bool(category_overlap)


def _qa_design_feedback_signal(feedback: str) -> dict[str, set[str]]:
    feedback_without_urls = _QA_DESIGN_URL_RE.sub(" ", feedback)
    feedback_without_paths = _QA_DESIGN_PATH_RE.sub(" ", feedback_without_urls)
    feedback_without_paths = _QA_DESIGN_FILE_RE.sub(" ", feedback_without_paths)
    normalized = feedback_without_paths.lower()
    tokens = set(_QA_DESIGN_TOKEN_RE.findall(normalized))
    categories = {
        category
        for category, keywords in _QA_DESIGN_KEYWORDS_BY_CATEGORY.items()
        if any(keyword in tokens for keyword in keywords)
    }
    files = {
        match.strip("`.,:;()[]")
        for match in _QA_DESIGN_FILE_RE.findall(feedback_without_urls)
    }
    return {"categories": categories, "files": files}


def _qa_feedback_from_run(run: SystemAgentRun) -> str:
    output = run.output
    if isinstance(output, dict):
        feedback = output.get("feedback")
        if output.get("lgtm") is False and isinstance(feedback, str):
            return feedback
    parsed = _parse_qa_output(run.raw_output)
    if parsed is None or parsed["lgtm"] is not False:
        return ""
    feedback = parsed["feedback"]
    return feedback if isinstance(feedback, str) else ""


def _summarize_qa_feedback(feedback: str) -> str:
    summary = " ".join(feedback.split())
    if len(summary) <= _QA_DESIGN_FEEDBACK_SUMMARY_CHARS:
        return summary
    return f"{summary[: _QA_DESIGN_FEEDBACK_SUMMARY_CHARS - 3].rstrip()}..."


def _qa_design_synthesis_feedback_prompt(
    feedback: str, synthesis_gate: dict[str, Any]
) -> str:
    categories = ", ".join(synthesis_gate.get("recurring_categories") or [])
    files = ", ".join(synthesis_gate.get("recurring_files") or [])
    evidence_lines = []
    for match in synthesis_gate.get("matches") or []:
        if not isinstance(match, dict):
            continue
        evidence_lines.append(
            "- "
            f"workflow {match.get('workflow_id')}, run {match.get('run_id')}: "
            f"{match.get('feedback', '')}"
        )
    evidence = "\n".join(evidence_lines) or "- No prior feedback summary available."
    return (
        "QA Design Synthesis Gate\n\n"
        "Hitch QA is seeing recurring design-level feedback, not just isolated "
        "defects. Before applying another tactical fix, pause and simplify the "
        "underlying design.\n\n"
        f"Recurring categories: {categories or 'unspecified'}\n"
        f"Recurring files: {files or 'none detected'}\n\n"
        "Prior related QA feedback:\n"
        f"{evidence}\n\n"
        "Current QA feedback:\n\n"
        f"{feedback}\n\n"
        "First identify the shared invariant, ownership boundary, or lifecycle "
        "rule that keeps breaking. Then implement the smallest coherent design "
        "change that makes that rule explicit and removes the need for another "
        "narrow fixup. Keep the diff focused, preserve existing behavior that is "
        "not implicated by the recurring feedback, and run the relevant tests."
    )
