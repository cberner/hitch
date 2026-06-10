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

from hitch.main import demo, system_agents
from hitch.main.models import (
    CodexInstance,
    GlobalSettings,
    ProposedSession,
    SessionMetadata,
    SystemAgentRun,
    SystemWorkflow,
)
from hitch.main.worktrees import (
    WorktreeCleanupError,
    cleanup_managed_worktree_path,
    discover_managed_worktrees,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_ALLOWED_DISK_SPACE_PERCENT = 20.0
ARCHIVED_USER_SESSION_MIN_AGE = timedelta(hours=1)
_WORKTREE_DIR_TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"
_PR_DONE_STAGE_KEYS = frozenset({"done_merged", "done_closed"})
_PROPOSAL_SESSION_ID_FIELDS = (
    "candidate_session_id",
    "judge_session_id",
    "source_session_id",
    "accepted_session_id",
)
_ACTIVE_WORKFLOW_CWD_STATE_KEYS = ("session_cwd", "stacked_diff_fork_from_cwd")


@dataclass(frozen=True)
class _CleanupCandidate:
    cwd: str
    reason: str
    thread_id: str
    timestamp: datetime
    sequence: int


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
    """Delete enough eligible worktrees to bring ``~/.hitch`` under the limit."""
    hitch_home = _hitch_home_dir()
    usage_path = _existing_disk_usage_path(hitch_home)
    try:
        usage = shutil.disk_usage(usage_path)
    except OSError:
        logger.exception("failed to inspect disk usage for %s", usage_path)
        return 0
    if usage.total <= 0:
        return 0
    limit_bytes = int(usage.total * (_max_allowed_percent() / 100.0))
    # Partition-level prefilter: ~/.hitch can only exceed the configured
    # percentage of total disk if the partition itself does. statvfs is
    # constant-time; the recursive lstat walk below scales with file count.
    if usage.used <= limit_bytes:
        return 0
    used_bytes = _directory_size(hitch_home)
    if used_bytes <= limit_bytes:
        return 0

    cleaned = 0
    candidates = _cleanup_candidates(now=timezone.now())
    usage_by_path = _candidate_worktree_usage_by_path(candidates)
    bytes_to_free = used_bytes - limit_bytes
    successful_bytes = 0
    attempted_paths: set[str] = set()
    for candidate in candidates:
        if successful_bytes >= bytes_to_free:
            # Hardlinked files may remain reachable through other worktrees.
            used_bytes = _directory_size(hitch_home)
            if used_bytes <= limit_bytes:
                break
            bytes_to_free = used_bytes - limit_bytes
            successful_bytes = 0
        normalized_path = _normalized_managed_path(candidate.cwd)
        if normalized_path is None or normalized_path in attempted_paths:
            continue
        usage_bytes = usage_by_path.get(normalized_path, 0)
        if usage_bytes <= 0:
            continue
        attempted_paths.add(normalized_path)
        try:
            removed = cleanup_managed_worktree_path(candidate.cwd)
        except WorktreeCleanupError:
            logger.exception(
                "failed to clean up %s worktree for %s",
                candidate.reason,
                candidate.thread_id or candidate.cwd,
            )
            continue
        if not removed:
            continue
        cleaned += 1
        successful_bytes += usage_bytes
    return cleaned


@dataclass(frozen=True)
class HitchDiskUsage:
    """Snapshot of ``~/.hitch`` disk consumption against its cleanup ceiling."""

    used_bytes: int
    limit_bytes: int
    disk_total_bytes: int

    @property
    def over_limit(self) -> bool:
        return self.used_bytes > self.limit_bytes

    @property
    def percent_of_disk(self) -> float:
        if self.disk_total_bytes <= 0:
            return 0.0
        return self.used_bytes / self.disk_total_bytes * 100.0


def hitch_home_disk_usage() -> HitchDiskUsage | None:
    """Read-only view of the same numbers ``cleanup_hitch_disk_usage_if_needed`` acts on.

    Returns ``None`` when the host disk cannot be inspected so callers can
    render "unavailable" rather than a misleading zero.
    """
    hitch_home = _hitch_home_dir()
    usage_path = _existing_disk_usage_path(hitch_home)
    try:
        disk_total = shutil.disk_usage(usage_path).total
    except OSError:
        logger.exception("failed to inspect disk usage for %s", usage_path)
        return None
    if disk_total <= 0:
        return None
    limit_bytes = int(disk_total * (_max_allowed_percent() / 100.0))
    return HitchDiskUsage(
        used_bytes=_directory_size(hitch_home),
        limit_bytes=limit_bytes,
        disk_total_bytes=disk_total,
    )


def _cleanup_candidates(*, now: datetime) -> list[_CleanupCandidate]:
    context = _cleanup_context(now=now)
    candidates: list[_CleanupCandidate] = []
    metadata_paths: set[str] = set()
    for metadata in _session_metadata_rows():
        normalized_path = _normalized_managed_path(metadata.cwd)
        if normalized_path is None:
            continue
        metadata_paths.add(normalized_path)
        if not _safe_to_remove_worktree(metadata, context):
            continue
        is_system = _is_system_session(metadata, context)
        is_accepted_visible = metadata.thread_id in context.accepted_visible_thread_ids
        if is_system and not is_accepted_visible:
            candidates.append(_metadata_cleanup_candidate(metadata, reason="system"))
        elif _archived_pr_done_user_session(metadata, context):
            candidates.append(_metadata_cleanup_candidate(metadata, reason="archived_pr"))
        elif _old_archived_user_session(metadata, context, now=now):
            candidates.append(_metadata_cleanup_candidate(metadata, reason="archived_old"))
    candidates.extend(
        _orphaned_worktree_candidates(
            context=context,
            metadata_paths=metadata_paths,
            now=now,
        )
    )
    return sorted(candidates, key=_candidate_sort_key)


@dataclass(frozen=True)
class _CleanupContext:
    accepted_visible_thread_ids: frozenset[str]
    hidden_system_thread_ids: frozenset[str]
    protected_proposal_session_ids: frozenset[int]
    active_thread_ids: frozenset[str]
    active_workflow_thread_ids: frozenset[str]
    protected_worktree_paths: frozenset[str]


def _cleanup_context(*, now: datetime) -> _CleanupContext:
    accepted_visible_thread_ids = system_agents.accepted_visible_system_thread_ids()
    hidden_system_thread_ids = _hidden_system_thread_ids()
    protected_proposal_session_ids = _protected_proposal_session_ids()
    active_thread_ids = frozenset(
        CodexInstance.objects.filter(status__in=CodexInstance.ACTIVE_STATUSES)
        .exclude(thread_id="")
        .values_list("thread_id", flat=True)
    )
    active_codex_paths = set(
        CodexInstance.objects.filter(status__in=CodexInstance.ACTIVE_STATUSES)
        .exclude(cwd="")
        .values_list("cwd", flat=True)
    )
    active_workflows = list(
        SystemWorkflow.objects.filter(status__in=SystemWorkflow.ACTIVE_STATUSES).only(
            "main_thread_id", "cwd", "state"
        )
    )
    active_workflow_paths = set(_active_workflow_paths(active_workflows))
    active_workflow_thread_ids = frozenset(
        workflow.main_thread_id
        for workflow in active_workflows
        if workflow.main_thread_id
    ) | frozenset(
        SystemAgentRun.objects.filter(workflow__status__in=SystemWorkflow.ACTIVE_STATUSES)
        .exclude(thread_id="")
        .values_list("thread_id", flat=True)
    )
    protected_paths = (
        active_codex_paths
        | active_workflow_paths
        | _pending_proposal_worktree_paths(protected_proposal_session_ids)
        | _protected_visible_user_worktree_paths(
            accepted_visible_thread_ids,
            hidden_system_thread_ids,
            now=now,
        )
    )
    return _CleanupContext(
        accepted_visible_thread_ids=frozenset(accepted_visible_thread_ids),
        hidden_system_thread_ids=frozenset(hidden_system_thread_ids),
        protected_proposal_session_ids=frozenset(protected_proposal_session_ids),
        active_thread_ids=active_thread_ids,
        active_workflow_thread_ids=active_workflow_thread_ids,
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
    if metadata.pk in context.protected_proposal_session_ids:
        return False
    if metadata.thread_id in context.active_thread_ids:
        return False
    if metadata.thread_id in context.active_workflow_thread_ids:
        return False
    normalized = _normalized_managed_path(metadata.cwd)
    return normalized is not None and normalized not in context.protected_worktree_paths


def _is_system_session(
    metadata: SessionMetadata, context: _CleanupContext
) -> bool:
    return (
        metadata.is_hidden_system_session
        or metadata.thread_id in context.hidden_system_thread_ids
    )


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


def _metadata_cleanup_candidate(
    metadata: SessionMetadata, *, reason: str
) -> _CleanupCandidate:
    return _CleanupCandidate(
        cwd=metadata.cwd,
        reason=reason,
        thread_id=metadata.thread_id,
        timestamp=metadata.codex_archived_at
        or metadata.codex_updated_at
        or _EARLIEST,
        sequence=metadata.pk or 0,
    )


def _orphaned_worktree_candidates(
    *,
    context: _CleanupContext,
    metadata_paths: set[str],
    now: datetime,
) -> list[_CleanupCandidate]:
    candidates: list[_CleanupCandidate] = []
    for path in discover_managed_worktrees():
        normalized_path = _normalized_managed_path(str(path))
        if normalized_path is None:
            continue
        if normalized_path in metadata_paths:
            continue
        if normalized_path in context.protected_worktree_paths:
            continue
        created_at = _managed_worktree_created_at(path)
        if created_at is None:
            continue
        if created_at > now - ARCHIVED_USER_SESSION_MIN_AGE:
            continue
        candidates.append(
            _CleanupCandidate(
                cwd=str(path),
                reason="orphaned",
                thread_id="",
                timestamp=created_at,
                sequence=0,
            )
        )
    return candidates


def _managed_worktree_created_at(path: Path) -> datetime | None:
    timestamp, separator, suffix = path.name.partition("-")
    if (
        separator != "-"
        or len(timestamp) != 14
        or len(suffix) != 8
        or not suffix.isalnum()
    ):
        return None
    try:
        created_at = datetime.strptime(timestamp, _WORKTREE_DIR_TIMESTAMP_FORMAT)
    except ValueError:
        return None
    return created_at.replace(tzinfo=UTC)


def _candidate_worktree_usage_by_path(
    candidates: list[_CleanupCandidate],
) -> dict[str, int]:
    usage_by_path: dict[str, int] = {}
    for candidate in candidates:
        normalized_path = _normalized_managed_path(candidate.cwd)
        if normalized_path is None or normalized_path in usage_by_path:
            continue
        usage_by_path[normalized_path] = _directory_size(Path(normalized_path))
    return usage_by_path


def _candidate_sort_key(
    candidate: _CleanupCandidate,
) -> tuple[int, datetime, int, str]:
    priority = {
        "system": 0,
        "orphaned": 1,
        "archived_pr": 2,
        "archived_old": 3,
    }[candidate.reason]
    return priority, candidate.timestamp, candidate.sequence, candidate.cwd


_EARLIEST = datetime.min.replace(tzinfo=UTC)


def _protected_proposal_session_ids() -> set[int]:
    protected = ProposedSession.objects.filter(
        models.Q(outcome_status=ProposedSession.OUTCOME_UNSET)
        | models.Q(
            outcome_status=ProposedSession.OUTCOME_DISMISSED,
            outcome_metadata__stacked_diff_hidden_until_complete=True,
        )
    )
    session_ids: set[int] = set()
    for field in _PROPOSAL_SESSION_ID_FIELDS:
        session_ids.update(
            value
            for value in protected.exclude(**{field: None}).values_list(
                field, flat=True
            )
            if isinstance(value, int)
        )
    return session_ids


def _hidden_system_thread_ids() -> set[str]:
    thread_ids = set(
        SystemAgentRun.objects.exclude(thread_id="")
        .exclude(agent_kind=demo.DEMO_AGENT_KIND)
        .values_list("thread_id", flat=True)
        .distinct()
    )
    thread_ids.update(
        CodexInstance.objects.filter(purpose=CodexInstance.PURPOSE_SYSTEM_AGENT)
        .exclude(thread_id="")
        .exclude(agent_kind=demo.DEMO_AGENT_KIND)
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
        for key in _ACTIVE_WORKFLOW_CWD_STATE_KEYS:
            cwd = state.get(key)
            if isinstance(cwd, str) and cwd:
                paths.add(cwd)
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
    hidden_system_thread_ids: set[str],
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
        .only(
            "thread_id",
            "cwd",
            "codex_archived",
            "codex_archived_at",
            "is_hidden_system_session",
            "derived_stage",
        )
    )
    for metadata in rows:
        is_system = (
            metadata.is_hidden_system_session
            or metadata.thread_id in hidden_system_thread_ids
        )
        if is_system and metadata.thread_id not in accepted_visible_thread_ids:
            continue
        if not metadata.codex_archived or (
            metadata.derived_stage not in _PR_DONE_STAGE_KEYS
            and (
                metadata.codex_archived_at is None
                or metadata.codex_archived_at > now - ARCHIVED_USER_SESSION_MIN_AGE
            )
        ):
            paths.add(metadata.cwd)
    return paths


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
    # Dedupe by (st_dev, st_ino): managed worktrees share blob data via
    # hardlinks into a common object store, so a naive sum counts each
    # hardlinked file once per worktree it appears in and overcounts real
    # on-disk usage. Directories always recurse; only their allocated size is
    # deduped (directories are not hardlinked across the tree in practice, but
    # treating them uniformly keeps the bookkeeping simple).
    seen: set[tuple[int, int]] = set()
    return _directory_size_inner(path, seen)


def _directory_size_inner(path: Path, seen: set[tuple[int, int]]) -> int:
    try:
        stat_result = path.lstat()
    except OSError:
        return 0
    key = (stat_result.st_dev, stat_result.st_ino)
    if key in seen:
        total = 0
    else:
        seen.add(key)
        total = _allocated_size(stat_result)
    if not stat.S_ISDIR(stat_result.st_mode) or stat.S_ISLNK(stat_result.st_mode):
        return total
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                total += _directory_size_inner(Path(entry.path), seen)
    except OSError:
        return total
    return total


def _allocated_size(stat_result: os.stat_result) -> int:
    blocks = getattr(stat_result, "st_blocks", 0)
    if isinstance(blocks, int) and blocks > 0:
        return blocks * 512
    return stat_result.st_size
