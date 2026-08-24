"""QA review handoffs, design-synthesis gate, and feedback helpers.

The PR-QA workflow runs a review agent and, when feedback recurs, a design
synthesis gate. This module owns the native Codex review handoff, the
recurring-feedback signal/match heuristics and their regexes, the synthesis-gate
builder and its feedback prompt, and the readers that pull QA feedback out of a
workflow's persisted state and prior runs.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from django.conf import settings

from hitch.main.diffs import validate_reviewer_diff_size
from hitch.main.models import CodexInstance, SystemAgentRun, SystemWorkflow
from hitch.main.workflows.agent_io import _parse_qa_output
from hitch.main.workflows.workflow_state import _state_int

_QA_VERDICT_AGENT_KINDS = ("pr_qa",)

# Keep smaller patches inline for a cheap, exact handoff. Larger patches live
# briefly in private, sandbox-visible Hitch temp storage so QA context does not
# duplicate them and the reviewed repository never retains worktree-only data.
_QA_INLINE_DIFF_MAX_CHARS = 100_000
_QA_HANDOFF_MODE_INLINE = "inline_diff"
_QA_HANDOFF_MODE_INLINE_FALLBACK = "inline_diff_fallback"
_QA_HANDOFF_MODE_CHUNKED_FILES = "chunked_files"
_QA_HANDOFF_CHUNK_MAX_BYTES = 12_000
_QA_HANDOFF_ROOT_NAME = "qa_review_handoffs"
_QA_HANDOFF_PART_SUFFIX = ".diffpart"
_QA_HANDOFF_OWNER_RE = re.compile(
    r"workflow-(?P<workflow_id>\d+)-review-(?P<revision>\d+)-"
    r"iteration-(?P<iteration>\d+)"
)
_QA_HANDOFF_STAGING_PREFIX = ".staging-"

logger = logging.getLogger(__name__)

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

_QA_COMPREHENSIVE_REVIEW_REQUEST = (
    "Review the complete current code changes in one comprehensive pass and "
    "provide prioritized, actionable findings for every concrete defect you can "
    "identify. Inspect "
    "every changed area and relevant surrounding code before returning; do not "
    "stop after the first valid finding or a small sample of issues. When one "
    "defect suggests a broader invariant, check sibling call sites and tests for "
    "the same class of problem."
)
_QA_REMEDIATION_GUIDANCE = (
    "Treat the findings in the current QA feedback as one remediation batch. Fix "
    "every defect in the current feedback, then inspect related changed paths and "
    "tests for sibling instances of each issue class. Prefer one coherent "
    "root-cause correction over serial special cases. Add or update focused "
    "regression coverage, run the relevant checks, and self-review the complete "
    "diff for the same issue classes before finishing this turn."
)


@dataclass(frozen=True)
class QaReviewHandoff:
    prompt: str
    mode: str
    ref: str
    embedded_diff_chars: int
    chunk_count: int = 0
    total_bytes: int = 0


def _qa_review_handoff(
    cwd: str,
    diff_text: str,
    *,
    workflow_id: int,
    review_revision: int,
    workflow_iteration: int,
    target_branch: str = "",
) -> QaReviewHandoff:
    validate_reviewer_diff_size(diff_text)
    if len(diff_text) <= _QA_INLINE_DIFF_MAX_CHARS:
        return QaReviewHandoff(
            prompt=_inline_qa_prompt(cwd, diff_text),
            mode=_QA_HANDOFF_MODE_INLINE,
            ref="",
            embedded_diff_chars=len(diff_text),
        )
    try:
        handoff_dir, chunk_count, total_bytes, digest = _write_qa_handoff_chunks(
            diff_text,
            workflow_id=workflow_id,
            review_revision=review_revision,
            workflow_iteration=workflow_iteration,
        )
    except RuntimeError:
        # Unavailable Hitch storage must not regress an otherwise reviewable
        # checkout. The previous exact inline handoff remains a safe fallback.
        return QaReviewHandoff(
            prompt=_inline_qa_prompt(cwd, diff_text),
            mode=_QA_HANDOFF_MODE_INLINE_FALLBACK,
            ref="",
            embedded_diff_chars=len(diff_text),
        )
    return QaReviewHandoff(
        prompt=_qa_prompt(
            cwd,
            diff_text,
            target_branch=target_branch,
            handoff_dir=handoff_dir,
            chunk_count=chunk_count,
            total_bytes=total_bytes,
            digest=digest,
        ),
        mode=_QA_HANDOFF_MODE_CHUNKED_FILES,
        ref=str(handoff_dir),
        embedded_diff_chars=0,
        chunk_count=chunk_count,
        total_bytes=total_bytes,
    )


def _qa_handoff_root() -> Path:
    deployment = sha256(str(Path(settings.HITCH_HOME_DIR)).encode()).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"{_QA_HANDOFF_ROOT_NAME}-{deployment}"


def _qa_handoff_path(
    workflow_id: int, review_revision: int, workflow_iteration: int
) -> Path:
    return _qa_handoff_root() / (
        f"workflow-{workflow_id}-review-{review_revision}-iteration-{workflow_iteration}"
    )


def _write_qa_handoff_chunks(
    diff_text: str,
    *,
    workflow_id: int,
    review_revision: int,
    workflow_iteration: int,
) -> tuple[Path, int, int, str]:
    root = _qa_handoff_root()
    handoff_dir = _qa_handoff_path(workflow_id, review_revision, workflow_iteration)
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        root.chmod(0o700)
        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=(
                    f"{_QA_HANDOFF_STAGING_PREFIX}workflow-{workflow_id}-"
                    f"review-{review_revision}-iteration-{workflow_iteration}-"
                ),
                dir=root,
            )
        )
        staging_dir.chmod(0o700)
    except OSError as exc:
        raise RuntimeError("could not create oversized reviewer handoff") from exc

    encoded = diff_text.encode("utf-8")
    chunks = _utf8_chunks(encoded, max_bytes=_QA_HANDOFF_CHUNK_MAX_BYTES)
    try:
        parts = []
        for index, chunk in enumerate(chunks, start=1):
            name = f"{index:05d}{_QA_HANDOFF_PART_SUFFIX}"
            part_path = staging_dir / name
            part_path.write_bytes(chunk)
            part_path.chmod(0o600)
            parts.append(
                {
                    "name": name,
                    "bytes": len(chunk),
                    "sha256": sha256(chunk).hexdigest(),
                }
            )
        digest = sha256(encoded).hexdigest()
        manifest = {
            "version": 1,
            "total_bytes": len(encoded),
            "sha256": digest,
            "chunk_max_bytes": _QA_HANDOFF_CHUNK_MAX_BYTES,
            "workflow_id": workflow_id,
            "review_revision": review_revision,
            "workflow_iteration": workflow_iteration,
            "parts": parts,
        }
        manifest_path = staging_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        manifest_path.chmod(0o600)
        shutil.rmtree(handoff_dir, ignore_errors=True)
        staging_dir.rename(handoff_dir)
    except OSError as exc:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise RuntimeError("could not write oversized reviewer handoff") from exc
    return handoff_dir, len(chunks), len(encoded), digest


def _utf8_chunks(data: bytes, *, max_bytes: int) -> list[bytes]:
    chunks: list[bytes] = []
    offset = 0
    while offset < len(data):
        end = min(offset + max_bytes, len(data))
        while (
            end > offset and end < len(data) and data[end] & 0b1100_0000 == 0b1000_0000
        ):
            end -= 1
        if end == offset:
            raise RuntimeError("could not split oversized reviewer handoff")
        chunks.append(data[offset:end])
        offset = end
    return chunks


def _cleanup_qa_handoff(ref: str) -> bool:
    if not ref:
        return False
    candidate = Path(ref).resolve(strict=False)
    if _qa_handoff_owner(ref) is None:
        logger.warning("refusing to clean up QA handoff outside Hitch storage: %s", ref)
        return False
    try:
        shutil.rmtree(candidate)
    except FileNotFoundError:
        pass
    except OSError:
        logger.exception("failed to clean up QA handoff %s", candidate)
        return False
    return True


def reap_stale_qa_handoffs(*, stale_before: datetime) -> int:
    """Remove crash-orphaned handoffs without racing live QA workers."""
    root = _qa_handoff_root()
    try:
        entries = list(root.iterdir())
    except FileNotFoundError:
        return 0
    except OSError:
        logger.exception("failed to list stale QA handoffs")
        return 0

    stale_entries = [
        entry for entry in entries if _qa_handoff_stale(entry, stale_before)
    ]
    owners = {
        owner
        for entry in stale_entries
        if (owner := _qa_handoff_owner(str(entry))) is not None
    }
    active_instances = list(
        CodexInstance.objects.filter(
            workflow_id__in={
                workflow_id for workflow_id, _revision, _iteration in owners
            },
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind="pr_qa",
            status__in=CodexInstance.ACTIVE_STATUSES,
        ).values_list("pk", "workflow_id", "user_message_index")
    )
    active_run_owners_by_instance: dict[int, tuple[int, int, int]] = {}
    for instance_id, run_input in SystemAgentRun.objects.filter(
        instance_id__in={
            instance_id for instance_id, _workflow_id, _revision in active_instances
        }
    ).values_list("instance_id", "input"):
        if not isinstance(run_input, dict):
            continue
        ref = run_input.get("qa_handoff_ref")
        owner = _qa_handoff_owner(ref) if isinstance(ref, str) else None
        if owner is not None:
            active_run_owners_by_instance[instance_id] = owner
    active_owners = set(active_run_owners_by_instance.values())
    active_owners.update(
        owner
        for owner in owners
        if any(
            instance_id not in active_run_owners_by_instance
            and workflow_id == owner[0]
            and (revision if isinstance(revision, int) else 0) == owner[1]
            for instance_id, workflow_id, revision in active_instances
        )
    )
    current_workflows = {
        (workflow.pk, _qa_review_revision(workflow), workflow.iteration): workflow
        for workflow in SystemWorkflow.objects.filter(
            pk__in={workflow_id for workflow_id, _revision, _iteration in owners},
            kind=SystemWorkflow.KIND_PR_QA,
            status=SystemWorkflow.STATUS_RUNNING,
            step="qa_running",
        )
    }
    fresh_owners = {
        owner
        for owner, workflow in current_workflows.items()
        if workflow.updated_at >= stale_before
    }

    removed = 0
    for entry in stale_entries:
        owner = _qa_handoff_owner(str(entry))
        if owner is not None and (owner in active_owners or owner in fresh_owners):
            continue
        try:
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            logger.exception("failed to reap stale QA handoff %s", entry)
            continue
        removed += 1
    return removed


def _qa_handoff_stale(path: Path, stale_before: datetime) -> bool:
    try:
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    except OSError:
        return False
    return modified < stale_before


def _qa_handoff_owner(ref: str) -> tuple[int, int, int] | None:
    candidate = Path(ref).resolve(strict=False)
    if candidate.parent != _qa_handoff_root().resolve(strict=False):
        return None
    match = _QA_HANDOFF_OWNER_RE.fullmatch(candidate.name)
    if match is None:
        return None
    return (
        int(match["workflow_id"]),
        int(match["revision"]),
        int(match["iteration"]),
    )


def _qa_prompt(
    cwd: str,
    diff_text: str,
    *,
    target_branch: str = "",
    handoff_dir: Path | None = None,
    chunk_count: int = 0,
    total_bytes: int = 0,
    digest: str = "",
) -> str:
    diff = diff_text or "(No current worktree diff was detected.)"
    if len(diff) > _QA_INLINE_DIFF_MAX_CHARS:
        if handoff_dir is None or chunk_count < 1 or total_bytes < 1 or not digest:
            raise ValueError("oversized reviewer diff requires a chunked handoff")
        review_scope = (
            f"the exact auto-merge patch for target branch {target_branch!r}"
            if target_branch
            else "the exact validated tracked and untracked worktree patch"
        )
        return (
            f"{_QA_COMPREHENSIVE_REVIEW_REQUEST} Hitch already validated that "
            "the reviewer diff is complete and representable.\n\n"
            f"Repository cwd: {cwd}\n\n"
            f"Review scope: {review_scope}.\n\n"
            f"Validated diff size: {len(diff):,} characters / {total_bytes:,} UTF-8 "
            f"bytes. Hitch stored the exact patch in {chunk_count} ordered files "
            f"under `{handoff_dir}`; the complete SHA-256 is `{digest}`. First read "
            f"`{handoff_dir / 'manifest.json'}`. Then read every numbered "
            f"`*{_QA_HANDOFF_PART_SUFFIX}` file in lexical order, one file per "
            "command. Each part is at most 12,000 bytes, including when a single "
            "diff line spans parts. Never combine parts into one command because "
            "bounded command output can truncate the review. The ordered parts are "
            "the authority for review scope; use the checkout only to cross-reference "
            "surrounding code. Do not substitute a live worktree or branch diff, and "
            "do not limit review to a summary or sample."
        )
    return _inline_qa_prompt(cwd, diff)


def _inline_qa_prompt(cwd: str, diff: str) -> str:
    return (
        f"{_QA_COMPREHENSIVE_REVIEW_REQUEST}\n\n"
        f"Repository cwd: {cwd}\n\n"
        "Proposed diff:\n"
        "```diff\n"
        f"{diff}\n"
        "```"
    )


def _qa_feedback_prompt(feedback: str) -> str:
    return (
        "Feedback from Hitch QA agent:\n\n"
        f"{feedback}\n\n"
        f"{_QA_REMEDIATION_GUIDANCE}"
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
        "Use the prior feedback only as evidence for identifying the shared "
        "invariant; do not treat its findings as current remediation work.\n\n"
        "Current QA feedback:\n\n"
        f"{feedback}\n\n"
        "First identify the shared invariant, ownership boundary, or lifecycle "
        "rule that keeps breaking. Then implement the smallest coherent design "
        "change that makes that rule explicit and removes the need for another "
        "narrow fixup. Keep the diff focused, preserve existing behavior that is "
        "not implicated by the recurring feedback.\n\n"
        f"{_QA_REMEDIATION_GUIDANCE}"
    )
