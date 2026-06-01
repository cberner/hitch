"""Shared creation logic for proposed session inbox items."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from django.db import transaction
from django.utils import timezone

from hitch.main import session_index
from hitch.main.models import (
    AutonomousGoal,
    CodexInstance,
    Project,
    ProposedSession,
    SessionMetadata,
    SystemWorkflow,
)
from hitch.main.repos import same_repo_or_worktree

_TITLE_MAX_LEN = 200
START_CLAIM_TTL = timedelta(minutes=15)
_SPEC_CRITIC_WORKFLOW_KIND = "spec_critic"


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


def create_proposed_session(values: ProposedSessionInput) -> ProposedSession:
    title = values.title.strip()
    summary = values.summary.strip()
    prompt = values.prompt.strip()
    cwd = str(Path(values.cwd).expanduser()) if values.cwd.strip() else ""
    confidence = values.confidence.strip() or AutonomousGoal.CONFIDENCE_MEDIUM
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
    if confidence not in {choice[0] for choice in AutonomousGoal.CONFIDENCE_CHOICES}:
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


def candidate_start_claim_metadata_updates(
    *,
    claimed_by: str | None,
    candidate_session: SessionMetadata | None,
) -> dict[str, object]:
    return {
        "candidate_start_claimed_by": claimed_by,
        "candidate_start_session_id": (
            candidate_session.pk if candidate_session is not None else None
        ),
        "candidate_start_thread_id": (
            candidate_session.thread_id if candidate_session is not None else None
        ),
    }


def reconcile_stale_candidate_proposal_starts(
    *, autonomous_goal: AutonomousGoal | None = None
) -> int:
    """Recover private start claims that never reached publish or rollback."""
    now = timezone.now()
    cutoff = now - START_CLAIM_TTL
    stale_proposals = ProposedSession.objects.select_related(
        "autonomous_goal",
        "candidate_session",
    ).filter(
        outcome_status=ProposedSession.OUTCOME_STARTING,
        updated_at__lt=cutoff,
    )
    if autonomous_goal is not None:
        stale_proposals = stale_proposals.filter(autonomous_goal=autonomous_goal)

    reconciled = 0
    for proposed_session in list(stale_proposals):
        if _candidate_start_claim_launched_work(proposed_session):
            if _publish_launched_candidate_start_claim(proposed_session, now=now):
                reconciled += 1
            continue
        if _reset_stale_candidate_start_claim(proposed_session, now=now):
            reconciled += 1
    return reconciled


def _candidate_start_claim_launched_work(proposed_session: ProposedSession) -> bool:
    candidate = proposed_session.candidate_session
    if candidate is None or not candidate.thread_id:
        return False
    claim_started_at = proposed_session.updated_at
    if _candidate_thread_has_user_turn_since(candidate.thread_id, claim_started_at):
        return True
    return _candidate_thread_has_workflow_since(candidate.thread_id, claim_started_at)


def _candidate_thread_has_user_turn_since(thread_id: str, claim_started_at: datetime) -> bool:
    return CodexInstance.objects.filter(
        thread_id=thread_id,
        purpose=CodexInstance.PURPOSE_USER,
        workflow_id__isnull=True,
        started_at__gte=claim_started_at,
    ).exists()


def _candidate_thread_has_workflow_since(
    thread_id: str, claim_started_at: datetime
) -> bool:
    return SystemWorkflow.objects.filter(
        main_thread_id=thread_id,
        kind__in=(SystemWorkflow.KIND_PR_QA, _SPEC_CRITIC_WORKFLOW_KIND),
        created_at__gte=claim_started_at,
    ).exists()


def _publish_launched_candidate_start_claim(
    proposed_session: ProposedSession, *, now: datetime
) -> bool:
    candidate = proposed_session.candidate_session
    if candidate is None:
        return False
    auto_pr_enabled, auto_qa_enabled = auto_review_settings_for_proposed_session(
        proposed_session
    )
    auto_merge_to_local_branch, auto_merge_branch = (
        auto_merge_to_local_branch_for_proposal(
            proposed_session,
            auto_qa_enabled=auto_qa_enabled,
        )
    )
    outcome_metadata = proposal_outcome_metadata(
        proposed_session,
        {
            "accepted_by": "user",
            "accepted_session_id": candidate.pk,
            "accepted_thread_id": candidate.thread_id,
            **candidate_start_claim_metadata_updates(
                claimed_by=None,
                candidate_session=None,
            ),
        },
    )
    with transaction.atomic():
        applied = ProposedSession.objects.filter(
            pk=proposed_session.pk,
            outcome_status=ProposedSession.OUTCOME_STARTING,
            updated_at=proposed_session.updated_at,
        ).update(
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            accepted_session=candidate,
            outcome_metadata=outcome_metadata,
            updated_at=now,
        )
        if not applied:
            return False
        SessionMetadata.objects.filter(pk=candidate.pk).update(
            auto_pr_enabled=auto_pr_enabled,
            auto_qa_enabled=auto_qa_enabled,
            auto_merge_to_local_branch=auto_merge_to_local_branch,
            auto_merge_branch=auto_merge_branch,
            is_hidden_system_session=False,
        )
    _apply_proposed_session_title_to_session_metadata(proposed_session, candidate)
    return True


def _reset_stale_candidate_start_claim(
    proposed_session: ProposedSession, *, now: datetime
) -> bool:
    outcome_metadata = proposal_outcome_metadata(
        proposed_session,
        candidate_start_claim_metadata_updates(
            claimed_by=None,
            candidate_session=None,
        ),
    )
    applied = ProposedSession.objects.filter(
        pk=proposed_session.pk,
        outcome_status=ProposedSession.OUTCOME_STARTING,
        updated_at=proposed_session.updated_at,
    ).update(
        outcome_status=ProposedSession.OUTCOME_UNSET,
        accepted_session=None,
        outcome_metadata=outcome_metadata,
        updated_at=now,
    )
    return bool(applied)


def proposal_outcome_metadata(
    proposed_session: ProposedSession, updates: dict[str, object]
) -> dict[str, object]:
    metadata = (
        dict(proposed_session.outcome_metadata)
        if isinstance(proposed_session.outcome_metadata, dict)
        else {}
    )
    for key, value in updates.items():
        if value is None:
            metadata.pop(key, None)
        else:
            metadata[key] = value
    return metadata


def auto_review_settings_for_proposed_session(
    proposed_session: ProposedSession,
) -> tuple[bool, bool]:
    metadata = proposal_metadata(proposed_session)
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


def auto_merge_to_local_branch_for_proposal(
    proposed_session: ProposedSession,
    *,
    auto_qa_enabled: bool,
) -> tuple[bool, str]:
    if not auto_qa_enabled:
        return False, ""
    metadata = proposal_metadata(proposed_session)
    if "auto_merge_to_local_branch" in metadata or "auto_merge_branch" in metadata:
        enabled = metadata.get("auto_merge_to_local_branch") is True
        branch = str(metadata.get("auto_merge_branch") or "").strip()
        if enabled and branch:
            return True, branch
        return False, ""
    if proposed_session.autonomous_goal is None:
        return False, ""
    autonomous_goal = proposed_session.autonomous_goal
    if not autonomous_goal.auto_merge_to_local_branch:
        return False, ""
    branch = autonomous_goal.auto_merge_branch.strip()
    if not branch:
        return False, ""
    return True, branch


def proposal_metadata(proposed_session: ProposedSession) -> dict[str, object]:
    return (
        proposed_session.outcome_metadata
        if isinstance(proposed_session.outcome_metadata, dict)
        else {}
    )


def _apply_proposed_session_title_to_session_metadata(
    proposed_session: ProposedSession,
    session_metadata: SessionMetadata,
) -> None:
    title = proposed_session.title.strip()[:_TITLE_MAX_LEN].rstrip()
    if title:
        session_index.update_cached_name(session_metadata.thread_id, title)


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
