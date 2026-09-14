"""Best-effort cleanup for Hitch-managed data when ~/.hitch grows too large."""

from __future__ import annotations

import logging
import os
import shutil
import stat
import threading
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from django.conf import settings
from django.db import close_old_connections
from django.utils import timezone

from hitch.main import checkouts
from hitch.main.models import (
    CodexInstance,
    GlobalSettings,
    SessionMetadata,
)
from hitch.main.runtime import codex_events
from hitch.main.sessions import checkout_protection
from hitch.main.sessions.checkout_protection import ARCHIVED_USER_SESSION_MIN_AGE
from hitch.main.worktrees import (
    WorktreeCleanupError,
    cleanup_managed_worktree_path,
    discover_managed_worktrees,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_ALLOWED_DISK_SPACE_PERCENT = 20.0
LEGACY_DIFF_EVENT_COMPACTION_MIN_BYTES = 512 * 1024 * 1024
_WORKTREE_DIR_TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"
_DISK_USAGE_SNAPSHOT_TTL = timedelta(minutes=5)
_DISK_USAGE_INVALIDATION_FILE = ".disk-usage-cache-token"


@dataclass(frozen=True)
class _CleanupCandidate:
    cwd: str
    reason: str
    thread_id: str
    timestamp: datetime
    sequence: int
    aliases: tuple[str, ...] = ()


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
    """Prune obsolete events and worktrees to bring ``~/.hitch`` under the limit."""
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
    pruned_event_bytes = _prune_oversized_finished_event_logs()
    if pruned_event_bytes:
        used_bytes = max(0, used_bytes - pruned_event_bytes)
        invalidate_hitch_home_disk_usage()
        if used_bytes <= limit_bytes:
            return 0

    cleaned = 0
    protection_snapshot_at = timezone.now()
    candidates = _cleanup_candidates(now=protection_snapshot_at)
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
        normalized_path = checkouts.managed_key(candidate.cwd)
        if normalized_path is None or normalized_path in attempted_paths:
            continue
        usage_bytes = usage_by_path.get(normalized_path, 0)
        if usage_bytes <= 0:
            continue
        attempted_paths.add(normalized_path)
        try:
            with checkout_protection.removal_lease(
                candidate.cwd, aliases=candidate.aliases, changed_since=protection_snapshot_at,
            ) as acquired:
                if not acquired:
                    continue
                removed = cleanup_managed_worktree_path(candidate.cwd)
        except (WorktreeCleanupError, OSError):
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


def _prune_oversized_finished_event_logs() -> int:
    configured_events_dir = getattr(settings, "CODEX_EVENTS_DIR", None)
    if not configured_events_dir:
        return 0
    events_dir = Path(configured_events_dir).resolve()
    total_freed = 0
    paths = (
        CodexInstance.objects.exclude(status__in=CodexInstance.ACTIVE_STATUSES)
        .exclude(events_path="")
        .values_list("events_path", flat=True)
    )
    for raw_path in paths.iterator():
        try:
            path = Path(raw_path).resolve()
            if not path.is_relative_to(events_dir):
                continue
            if path.stat().st_size < LEGACY_DIFF_EVENT_COMPACTION_MIN_BYTES:
                continue
            total_freed += codex_events.prune_diff_events(path)
        except FileNotFoundError:
            continue
        except (OSError, RuntimeError):
            logger.exception(
                "failed to compact obsolete diff events in %s",
                raw_path,
            )
    if total_freed:
        logger.info(
            "compacted obsolete diff events, freeing %s bytes",
            total_freed,
        )
    return total_freed


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


@dataclass(frozen=True)
class _DiskUsageSnapshot:
    captured_at: datetime
    invalidation_token: str
    usage: HitchDiskUsage


_disk_usage_snapshot_lock = threading.Lock()
_disk_usage_snapshot: _DiskUsageSnapshot | None = None
_disk_usage_refreshing = False
_disk_usage_generation = 0


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


def cached_hitch_home_disk_usage() -> HitchDiskUsage | None:
    """Return the latest snapshot and refresh the expensive tree walk off-request."""
    global _disk_usage_generation, _disk_usage_refreshing, _disk_usage_snapshot
    now = timezone.now()
    invalidation_token = _disk_usage_invalidation_token()
    snapshot: HitchDiskUsage | None
    with _disk_usage_snapshot_lock:
        cached = _disk_usage_snapshot
        if (
            cached is not None
            and cached.invalidation_token == invalidation_token
            and now - cached.captured_at < _DISK_USAGE_SNAPSHOT_TTL
        ):
            snapshot = cached.usage
            needs_refresh = False
        else:
            needs_refresh = True
            snapshot = cached.usage if cached is not None and cached.invalidation_token == invalidation_token else None
            if cached is not None and snapshot is None:
                _disk_usage_snapshot = None
                _disk_usage_generation += 1
        if needs_refresh and not _disk_usage_refreshing:
            _disk_usage_refreshing = True
            generation = _disk_usage_generation
            refresh_thread = threading.Thread(
                target=_refresh_disk_usage_snapshot,
                args=(generation, invalidation_token),
                name="hitch-disk-usage",
                daemon=True,
            )
            try:
                refresh_thread.start()
            except RuntimeError:
                _disk_usage_refreshing = False
                logger.exception("failed to start Hitch disk usage refresh")
    return _disk_usage_with_current_limit(snapshot) if snapshot is not None else None


def _refresh_disk_usage_snapshot(generation: int, invalidation_token: str) -> None:
    global _disk_usage_refreshing, _disk_usage_snapshot
    close_old_connections()
    try:
        snapshot = hitch_home_disk_usage()
    except Exception:
        logger.exception("failed to refresh Hitch disk usage snapshot")
        snapshot = None
    finally:
        close_old_connections()
    current_invalidation_token = _disk_usage_invalidation_token()
    with _disk_usage_snapshot_lock:
        if snapshot is None:
            _disk_usage_snapshot = None
        elif generation == _disk_usage_generation and invalidation_token == current_invalidation_token:
            _disk_usage_snapshot = _DiskUsageSnapshot(
                captured_at=timezone.now(),
                invalidation_token=current_invalidation_token,
                usage=snapshot,
            )
        _disk_usage_refreshing = False


def invalidate_hitch_home_disk_usage() -> None:
    """Invalidate this process and notify other Hitch processes."""
    global _disk_usage_generation, _disk_usage_snapshot
    _publish_disk_usage_invalidation()
    with _disk_usage_snapshot_lock:
        _disk_usage_snapshot = None
        _disk_usage_generation += 1


def _disk_usage_with_current_limit(snapshot: HitchDiskUsage) -> HitchDiskUsage:
    return HitchDiskUsage(
        used_bytes=snapshot.used_bytes,
        limit_bytes=int(snapshot.disk_total_bytes * (_max_allowed_percent() / 100.0)),
        disk_total_bytes=snapshot.disk_total_bytes,
    )


def _disk_usage_invalidation_path() -> Path:
    return _hitch_home_dir() / _DISK_USAGE_INVALIDATION_FILE


def _disk_usage_invalidation_token() -> str:
    try:
        return _disk_usage_invalidation_path().read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except (OSError, UnicodeError):
        logger.exception("failed to read Hitch disk usage invalidation token")
        return ""


def _publish_disk_usage_invalidation() -> None:
    path = _disk_usage_invalidation_path()
    thread_id = threading.get_ident()
    temporary_path = path.with_name(f"{path.name}.{os.getpid()}.{thread_id}.tmp")
    token = f"{time.time_ns()}:{os.getpid()}:{thread_id}"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_text(token, encoding="utf-8")
        os.replace(temporary_path, path)
    except OSError:
        logger.exception("failed to publish Hitch disk usage invalidation")
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            logger.exception("failed to remove disk usage invalidation temporary file")


def _cleanup_candidates(*, now: datetime) -> list[_CleanupCandidate]:
    context = checkout_protection.snapshot(now=now)
    candidates: list[_CleanupCandidate] = []
    metadata_paths: set[str] = set()
    aliases_by_path: dict[str, set[str]] = {}
    for metadata in _session_metadata_rows():
        normalized_path = checkouts.managed_key(metadata.cwd)
        if normalized_path is None:
            continue
        metadata_paths.add(normalized_path)
        aliases_by_path.setdefault(normalized_path, set()).add(metadata.cwd)
        if context.protects(metadata):
            continue
        reason = context.retention.removal_reason(metadata, now=now)
        if reason is not None:
            candidates.append(_metadata_cleanup_candidate(metadata, cwd=normalized_path, reason=reason))
    candidates.extend(
        _orphaned_worktree_candidates(
            context=context,
            metadata_paths=metadata_paths,
            now=now,
        )
    )
    return sorted(
        (replace(candidate, aliases=tuple(aliases_by_path.get(
            checkouts.managed_key(candidate.cwd) or "", set(),
        ))) for candidate in candidates),
        key=_candidate_sort_key,
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


def _metadata_cleanup_candidate(metadata: SessionMetadata, *, cwd: str, reason: str) -> _CleanupCandidate:
    return _CleanupCandidate(
        cwd=cwd,
        reason=reason,
        thread_id=metadata.thread_id,
        timestamp=metadata.codex_archived_at or metadata.codex_updated_at or _EARLIEST,
        sequence=metadata.pk or 0,
    )


def _orphaned_worktree_candidates(
    *,
    context: checkout_protection.ProtectionSnapshot,
    metadata_paths: set[str],
    now: datetime,
) -> list[_CleanupCandidate]:
    candidates: list[_CleanupCandidate] = []
    for path in discover_managed_worktrees():
        normalized_path = checkouts.managed_key(str(path))
        if normalized_path is None:
            continue
        if normalized_path in metadata_paths:
            continue
        if normalized_path in context.protected_worktree_paths:
            continue
        created_at = _managed_worktree_created_at(Path(normalized_path))
        if created_at is None:
            continue
        if created_at > now - ARCHIVED_USER_SESSION_MIN_AGE:
            continue
        candidates.append(
            _CleanupCandidate(
                cwd=normalized_path,
                reason="orphaned",
                thread_id="",
                timestamp=created_at,
                sequence=0,
            )
        )
    return candidates


def _managed_worktree_created_at(path: Path) -> datetime | None:
    timestamp, separator, suffix = path.name.partition("-")
    if separator != "-" or len(timestamp) != 14 or len(suffix) != 8 or not suffix.isalnum():
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
        normalized_path = checkouts.managed_key(candidate.cwd)
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


def _max_allowed_percent() -> float:
    saved = _saved_max_allowed_percent()
    if saved is not None:
        return saved
    raw = getattr(settings, "HITCH_MAX_ALLOWED_DISK_SPACE_PERCENT", None)
    if raw is None:
        raw = DEFAULT_MAX_ALLOWED_DISK_SPACE_PERCENT
    return _validated_max_allowed_percent(raw, setting_name="HITCH_MAX_ALLOWED_DISK_SPACE_PERCENT")


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
    return _validated_max_allowed_percent(saved, setting_name="GlobalSettings.disk_usage_max_percent")


def _validated_max_allowed_percent(raw: str | int | float, *, setting_name: str) -> float:
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
