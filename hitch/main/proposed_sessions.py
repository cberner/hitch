"""Shared creation logic for proposed session inbox items."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hitch.main.models import Project, ProposedSession, SessionMetadata, StandingOrder
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
    confidence: str = StandingOrder.CONFIDENCE_MEDIUM
    source_thread_id: str = ""


def create_proposed_session(values: ProposedSessionInput) -> ProposedSession:
    title = values.title.strip()
    summary = values.summary.strip()
    prompt = values.prompt.strip()
    cwd = str(Path(values.cwd).expanduser()) if values.cwd.strip() else ""
    confidence = values.confidence.strip() or StandingOrder.CONFIDENCE_MEDIUM
    if not title:
        raise ProposedSessionError("title is required")
    if len(title) > _TITLE_MAX_LEN:
        raise ProposedSessionError("title is too long")
    if not summary:
        raise ProposedSessionError("summary is required")
    if not prompt:
        raise ProposedSessionError("prompt is required")
    if not cwd:
        raise ProposedSessionError("cwd is required")
    if confidence not in {choice[0] for choice in StandingOrder.CONFIDENCE_CHOICES}:
        raise ProposedSessionError("confidence is invalid")
    project = project_for_cwd(cwd)
    if project is None:
        raise ProposedSessionError("cwd does not match a Hitch project")
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
