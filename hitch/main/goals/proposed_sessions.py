"""Shared creation logic for proposed session inbox items."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from django.utils import timezone

from hitch.main.models import AutonomousGoal, Project, ProposedSession, SessionMetadata
from hitch.main.repos import same_repo_or_worktree

_TITLE_MAX_LEN = 200


class ProposedSessionError(ValueError):
    """Raised when a proposed session request cannot be saved."""


@dataclass(frozen=True)
class ProposedSessionInput:
    title: str
    summary: str
    prompt: str
    cwd: str
    relevant_files: list[str]
    confidence: str = AutonomousGoal.CONFIDENCE_MEDIUM
    source_thread_id: str = ""


@dataclass(frozen=True)
class ProposedSessionUpdateInput:
    proposal_id: int
    cwd: str
    title: str | None = None
    summary: str | None = None
    prompt: str | None = None
    relevant_files: list[str] | None = None
    confidence: str | None = None


def create_proposed_session(values: ProposedSessionInput) -> ProposedSession:
    title = _clean_title(values.title)
    summary = _clean_required_text(values.summary, "summary")
    prompt = _clean_required_text(values.prompt, "prompt")
    cwd = _clean_cwd(values.cwd)
    confidence = _clean_confidence(values.confidence)
    project = _project_for_clean_cwd(cwd)
    source_session = None
    source_thread_id = values.source_thread_id.strip()
    if source_thread_id:
        source_session = SessionMetadata.objects.filter(thread_id=source_thread_id).first()
    return ProposedSession.objects.create(
        project=project,
        source_session=source_session,
        title=title,
        summary=summary,
        prompt=prompt,
        confidence=confidence,
        relevant_files=_clean_relevant_files(values.relevant_files),
    )


def update_proposed_session(values: ProposedSessionUpdateInput) -> ProposedSession:
    cwd = _clean_cwd(values.cwd)
    project = _project_for_clean_cwd(cwd)
    proposal = ProposedSession.objects.filter(pk=values.proposal_id).first()
    if proposal is None or proposal.project_id != project.pk:
        raise ProposedSessionError("proposal does not match current Hitch project")
    if proposal.inbox_kind != ProposedSession.INBOX_KIND_PROPOSAL:
        raise ProposedSessionError("proposal item is not editable")
    if proposal.outcome_status != ProposedSession.OUTCOME_UNSET:
        raise ProposedSessionError("proposal has already been resolved")

    update_fields: list[str] = []
    if values.title is not None:
        proposal.title = _clean_title(values.title)
        update_fields.append("title")
    if values.summary is not None:
        proposal.summary = _clean_required_text(values.summary, "summary")
        update_fields.append("summary")
    if values.prompt is not None:
        proposal.prompt = _clean_required_text(values.prompt, "prompt")
        update_fields.append("prompt")
    if values.relevant_files is not None:
        proposal.relevant_files = _clean_relevant_files(values.relevant_files)
        update_fields.append("relevant_files")
    if values.confidence is not None:
        proposal.confidence = _clean_confidence(values.confidence)
        update_fields.append("confidence")
    if not update_fields:
        raise ProposedSessionError("at least one editable field is required")

    updated_at = timezone.now()
    update_values = {field: getattr(proposal, field) for field in update_fields}
    updated = ProposedSession.objects.filter(
        pk=proposal.pk,
        project_id=project.pk,
        inbox_kind=ProposedSession.INBOX_KIND_PROPOSAL,
        outcome_status=ProposedSession.OUTCOME_UNSET,
    ).update(**update_values, updated_at=updated_at)
    if not updated:
        raise ProposedSessionError("proposal has already been resolved")
    proposal.updated_at = updated_at
    return proposal


def _clean_title(value: str) -> str:
    title = value.strip()
    if not title:
        raise ProposedSessionError("title is required")
    if len(title) > _TITLE_MAX_LEN:
        raise ProposedSessionError("title is too long")
    return title


def _clean_required_text(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ProposedSessionError(f"{field_name} is required")
    return cleaned


def _clean_cwd(value: str) -> str:
    cwd = str(Path(value).expanduser()) if value.strip() else ""
    if not cwd:
        raise ProposedSessionError("cwd is required")
    return cwd


def _clean_confidence(value: str) -> str:
    confidence = value.strip() or AutonomousGoal.CONFIDENCE_MEDIUM
    if confidence not in {choice[0] for choice in AutonomousGoal.CONFIDENCE_CHOICES}:
        raise ProposedSessionError("confidence is invalid")
    return confidence


def _project_for_clean_cwd(cwd: str) -> Project:
    project = project_for_cwd(cwd)
    if project is None:
        raise ProposedSessionError("cwd does not match a Hitch project")
    return project


def project_for_cwd(cwd: str) -> Project | None:
    projects = Project.objects.all().order_by("created_at", "id")
    return next(
        (
            project
            for project in projects
            if same_repo_or_worktree(cwd, project.repo_path, project.git_common_dir)
        ),
        None,
    )


def _clean_relevant_files(files: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in files:
        value = item.strip()
        if not value or value in seen:
            continue
        cleaned.append(value)
        seen.add(value)
    return cleaned
