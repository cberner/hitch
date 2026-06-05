"""Best-effort cleanup for Hitch-managed worktrees when ~/.hitch grows too large."""

from __future__ import annotations

import logging
import os
import shutil
import stat
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from django.conf import settings
from django.db import models
from django.utils import timezone

from hitch.main.models import (
    CodexInstance,
    GlobalSettings,
    ProposedSession,
    SessionMetadata,
    SystemAgentRun,
    SystemWorkflow,
)
from hitch.main.worktrees import WorktreeCleanupError, cleanup_managed_worktree_path

logger = logging.getLogger(__name__)

DEFAULT_MAX_ALLOWED_DISK_SPACE_PERCENT = 20.0
ARCHIVED_USER_SESSION_MIN_AGE = timedelta(hours=48)
_PR_DONE_STAGE_KEYS = frozenset({"done_merged", "done_closed"})
_ACTIVE_CODEX_STATUSES = (
    CodexInstance.STATUS_STARTING,
    CodexInstance.STATUS_RUNNING,
)
_ACTIVE_WORKFLOW_STATUSES = (
    SystemWorkflow.STATUS_RUNNING,
    SystemWorkflow.STATUS_BLOCKED,
)
_SYSTEM_CODEX_PURPOSES = (
    CodexInstance.PURPOSE_SYSTEM_AGENT,
    CodexInstance.PURPOSE_SYSTEM_FEEDBACK,
)
_PROPOSAL_SESSION_ID_FIELDS = (
    "candidate_session_id",
    "judge_session_id",
    "source_session_id",
    "accepted_session_id",
)


@dataclass(frozen=True)
class _CleanupCandidate:
    metadata: SessionMetadata
    reason: str


def run_finished_session_disk_cleanup() -> None:
    """Run disk cleanup after a session finishes without affecting that session."""
    if getattr(settings, "TESTING", False):
        return
    try:
        cleaned = cleanup_hitch_disk_usage_if_needed()
    except Exception:
        logger.exception("failed to run Hitch disk cleanup")
        return
    if cleaned:
        logger.info("cleaned up %s Hitch-managed worktree(s)", cleaned)


def cleanup_hitch_disk_usage_if_needed() -> int:
    """Delete eligible managed worktrees until ``~/.hitch`` is under the limit."""
    hitch_home = _hitch_home_dir()
    usage_path = _existing_disk_usage_path(hitch_home)
    try:
        disk_total = shutil.disk_usage(usage_path).total
    except OSError:
        logger.exception("failed to inspect disk usage for %s", usage_path)
        return 0
    if disk_total <= 0:
        return 0
    limit_bytes = int(disk_total * (_max_allowed_percent() / 100.0))
    used_bytes = _directory_size(hitch_home)
    if used_bytes <= limit_bytes:
        return 0

    cleaned = 0
    for candidate in _cleanup_candidates(now=timezone.now()):
        if used_bytes <= limit_bytes:
            break
        try:
            removed = cleanup_managed_worktree_path(candidate.metadata.cwd)
        except WorktreeCleanupError:
            logger.exception(
                "failed to clean up %s worktree for session %s",
                candidate.reason,
                candidate.metadata.thread_id,
            )
            used_bytes = _directory_size(hitch_home)
            continue
        if not removed:
            used_bytes = _directory_size(hitch_home)
            continue
        cleaned += 1
        used_bytes = _directory_size(hitch_home)
    return cleaned


def _cleanup_candidates(*, now: datetime) -> list[_CleanupCandidate]:
    context = _cleanup_context(now=now)
    candidates: list[_CleanupCandidate] = []
    for metadata in _session_metadata_rows():
        if not _metadata_has_managed_worktree(metadata):
            continue
        if not _safe_to_remove_worktree(metadata, context):
            continue
        is_system = _is_system_session(metadata, context)
        is_accepted_visible = metadata.thread_id in context.accepted_visible_thread_ids
        if is_system and not is_accepted_visible:
            candidates.append(_CleanupCandidate(metadata=metadata, reason="system"))
        elif _archived_pr_done_user_session(metadata, context):
            candidates.append(_CleanupCandidate(metadata=metadata, reason="archived_pr"))
        elif _old_archived_user_session(metadata, context, now=now):
            candidates.append(_CleanupCandidate(metadata=metadata, reason="archived_old"))
    return sorted(candidates, key=_candidate_sort_key)


@dataclass(frozen=True)
class _CleanupContext:
    accepted_visible_thread_ids: frozenset[str]
    system_thread_ids: frozenset[str]
    pending_proposal_session_ids: frozenset[int]
    active_thread_ids: frozenset[str]
    active_or_blocked_workflow_thread_ids: frozenset[str]
    protected_worktree_paths: frozenset[str]


def _cleanup_context(*, now: datetime) -> _CleanupContext:
    accepted_visible_thread_ids = _accepted_visible_system_thread_ids()
    pending_proposal_session_ids = _pending_proposal_session_ids()
    active_thread_ids = frozenset(
        CodexInstance.objects.filter(status__in=_ACTIVE_CODEX_STATUSES)
        .exclude(thread_id="")
        .values_list("thread_id", flat=True)
    )
    active_codex_paths = set(
        CodexInstance.objects.filter(status__in=_ACTIVE_CODEX_STATUSES)
        .exclude(cwd="")
        .values_list("cwd", flat=True)
    )
    active_workflows = list(
        SystemWorkflow.objects.filter(status__in=_ACTIVE_WORKFLOW_STATUSES).only(
            "main_thread_id", "cwd", "state"
        )
    )
    active_workflow_paths = set(_active_workflow_paths(active_workflows))
    active_or_blocked_workflow_thread_ids = frozenset(
        workflow.main_thread_id
        for workflow in active_workflows
        if workflow.main_thread_id
    ) | frozenset(
        SystemAgentRun.objects.filter(workflow__status__in=_ACTIVE_WORKFLOW_STATUSES)
        .exclude(thread_id="")
        .values_list("thread_id", flat=True)
    )
    protected_paths = (
        active_codex_paths
        | active_workflow_paths
        | _pending_proposal_worktree_paths(pending_proposal_session_ids)
        | _protected_visible_user_worktree_paths(accepted_visible_thread_ids, now=now)
    )
    return _CleanupContext(
        accepted_visible_thread_ids=frozenset(accepted_visible_thread_ids),
        system_thread_ids=frozenset(_system_thread_ids()),
        pending_proposal_session_ids=frozenset(pending_proposal_session_ids),
        active_thread_ids=active_thread_ids,
        active_or_blocked_workflow_thread_ids=active_or_blocked_workflow_thread_ids,
        protected_worktree_paths=frozenset(
            _normalized_managed_paths(path for path in protected_paths if path)
        ),
    )


def _session_metadata_rows() -> list[SessionMetadata]:
    return list(
        SessionMetadata.objects.exclude(cwd="")
        .only(
            "thread_id",
            "cwd",
            "codex_archived",
            "codex_archived_at",
            "codex_updated_at",
            "is_hidden_system_session",
            "derived_stage",
        )
        .order_by("codex_updated_at", "pk")
    )


def _safe_to_remove_worktree(
    metadata: SessionMetadata, context: _CleanupContext
) -> bool:
    if metadata.pk in context.pending_proposal_session_ids:
        return False
    if metadata.thread_id in context.active_thread_ids:
        return False
    if metadata.thread_id in context.active_or_blocked_workflow_thread_ids:
        return False
    normalized = _normalized_managed_path(metadata.cwd)
    return normalized is not None and normalized not in context.protected_worktree_paths


def _is_system_session(
    metadata: SessionMetadata, context: _CleanupContext
) -> bool:
    return metadata.is_hidden_system_session or metadata.thread_id in context.system_thread_ids


def _archived_pr_done_user_session(
    metadata: SessionMetadata, context: _CleanupContext
) -> bool:
    return (
        metadata.codex_archived
        and _is_user_session(metadata, context)
        and metadata.derived_stage in _PR_DONE_STAGE_KEYS
    )


def _old_archived_user_session(
    metadata: SessionMetadata, context: _CleanupContext, *, now: datetime
) -> bool:
    return (
        metadata.codex_archived
        and _is_user_session(metadata, context)
        and metadata.codex_archived_at is not None
        and metadata.codex_archived_at <= now - ARCHIVED_USER_SESSION_MIN_AGE
    )


def _is_user_session(
    metadata: SessionMetadata, context: _CleanupContext
) -> bool:
    return (
        not _is_system_session(metadata, context)
        or metadata.thread_id in context.accepted_visible_thread_ids
    )


def _candidate_sort_key(candidate: _CleanupCandidate) -> tuple[int, datetime, int]:
    priority = {"system": 0, "archived_pr": 1, "archived_old": 2}[candidate.reason]
    metadata = candidate.metadata
    timestamp = metadata.codex_archived_at or metadata.codex_updated_at or _EARLIEST
    return priority, timestamp, metadata.pk or 0


_EARLIEST = datetime.min.replace(tzinfo=UTC)


def _accepted_visible_system_thread_ids() -> set[str]:
    return set(
        ProposedSession.objects.filter(
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            candidate_session__isnull=False,
            accepted_session=models.F("candidate_session"),
        ).values_list("candidate_session__thread_id", flat=True)
    )


def _pending_proposal_session_ids() -> set[int]:
    pending = ProposedSession.objects.filter(
        outcome_status=ProposedSession.OUTCOME_UNSET
    )
    session_ids: set[int] = set()
    for field in _PROPOSAL_SESSION_ID_FIELDS:
        session_ids.update(
            value
            for value in pending.exclude(**{field: None}).values_list(field, flat=True)
            if isinstance(value, int)
        )
    return session_ids


def _system_thread_ids() -> set[str]:
    thread_ids = set(
        SystemAgentRun.objects.exclude(thread_id="")
        .values_list("thread_id", flat=True)
        .distinct()
    )
    thread_ids.update(
        CodexInstance.objects.filter(purpose__in=_SYSTEM_CODEX_PURPOSES)
        .exclude(thread_id="")
        .values_list("thread_id", flat=True)
        .distinct()
    )
    return thread_ids


def _active_workflow_paths(workflows: list[SystemWorkflow]) -> set[str]:
    paths: set[str] = set()
    for workflow in workflows:
        if workflow.cwd:
            paths.add(workflow.cwd)
        state = workflow.state if isinstance(workflow.state, dict) else {}
        session_cwd = state.get("session_cwd")
        if isinstance(session_cwd, str) and session_cwd:
            paths.add(session_cwd)
    return paths


def _pending_proposal_worktree_paths(session_ids: set[int]) -> set[str]:
    if not session_ids:
        return set()
    return set(
        SessionMetadata.objects.filter(pk__in=session_ids)
        .exclude(cwd="")
        .values_list("cwd", flat=True)
    )


def _protected_visible_user_worktree_paths(
    accepted_visible_thread_ids: set[str],
    *,
    now: datetime,
) -> set[str]:
    paths: set[str] = set()
    rows = (
        SessionMetadata.objects.filter(
            models.Q(is_hidden_system_session=False)
            | models.Q(thread_id__in=accepted_visible_thread_ids)
        )
        .exclude(cwd="")
        .only("cwd", "codex_archived", "codex_archived_at", "derived_stage")
    )
    for metadata in rows:
        if not metadata.codex_archived or (
            metadata.derived_stage not in _PR_DONE_STAGE_KEYS
            and (
                metadata.codex_archived_at is None
                or metadata.codex_archived_at > now - ARCHIVED_USER_SESSION_MIN_AGE
            )
        ):
            paths.add(metadata.cwd)
    return paths


def _metadata_has_managed_worktree(metadata: SessionMetadata) -> bool:
    return _normalized_managed_path(metadata.cwd) is not None


def _normalized_managed_paths(paths: Iterable[str]) -> set[str]:
    normalized: set[str] = set()
    for path in paths:
        normalized_path = _normalized_managed_path(path)
        if normalized_path is not None:
            normalized.add(normalized_path)
    return normalized


def _normalized_managed_path(raw_path: str) -> str | None:
    path = Path(raw_path).expanduser()
    base = Path(settings.HITCH_WORKTREES_DIR).expanduser()
    try:
        resolved_path = path.resolve(strict=False)
        resolved_base = base.resolve(strict=False)
        resolved_path.relative_to(resolved_base)
    except (OSError, ValueError):
        return None
    return str(resolved_path)


def _max_allowed_percent() -> float:
    saved = _saved_max_allowed_percent()
    if saved is not None:
        return saved
    raw = getattr(settings, "HITCH_MAX_ALLOWED_DISK_SPACE_PERCENT", None)
    if raw is None:
        raw = DEFAULT_MAX_ALLOWED_DISK_SPACE_PERCENT
    return _validated_max_allowed_percent(
        raw, setting_name="HITCH_MAX_ALLOWED_DISK_SPACE_PERCENT"
    )


def _saved_max_allowed_percent() -> float | None:
    try:
        saved = (
            GlobalSettings.objects.filter(pk=GlobalSettings.SINGLETON_PK)
            .values_list("disk_usage_max_percent", flat=True)
            .first()
        )
    except Exception:
        logger.exception("failed to load saved Hitch disk usage setting")
        return None
    if saved is None:
        return None
    return _validated_max_allowed_percent(
        saved, setting_name="GlobalSettings.disk_usage_max_percent"
    )


def _validated_max_allowed_percent(
    raw: str | int | float, *, setting_name: str
) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "invalid %s %r; using %s",
            setting_name,
            raw,
            DEFAULT_MAX_ALLOWED_DISK_SPACE_PERCENT,
        )
        return DEFAULT_MAX_ALLOWED_DISK_SPACE_PERCENT
    if value <= 0 or value > 100:
        logger.warning(
            "invalid %s %r; using %s",
            setting_name,
            raw,
            DEFAULT_MAX_ALLOWED_DISK_SPACE_PERCENT,
        )
        return DEFAULT_MAX_ALLOWED_DISK_SPACE_PERCENT
    return value


def _hitch_home_dir() -> Path:
    return Path(getattr(settings, "HITCH_HOME_DIR", Path.home() / ".hitch")).expanduser()


def _existing_disk_usage_path(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _directory_size(path: Path) -> int:
    try:
        stat_result = path.lstat()
    except OSError:
        return 0
    total = _allocated_size(stat_result)
    if not stat.S_ISDIR(stat_result.st_mode) or stat.S_ISLNK(stat_result.st_mode):
        return total
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                total += _directory_size(Path(entry.path))
    except OSError:
        return total
    return total


def _allocated_size(stat_result: os.stat_result) -> int:
    blocks = getattr(stat_result, "st_blocks", 0)
    if isinstance(blocks, int) and blocks > 0:
        return blocks * 512
    return stat_result.st_size
