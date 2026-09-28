"""Shared retention policy and leased protection checks for checkout removal."""

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta

from django.db import models
from django.utils import timezone

from hitch.main import checkouts
from hitch.main.models import CodexInstance, ProposedSession, SessionMetadata, SessionPullRequest

ARCHIVED_USER_SESSION_MIN_AGE = timedelta(hours=1)
_PR_DONE_STAGE_KEYS = frozenset({"done_merged", "done_closed"})
_PROPOSAL_SESSION_ID_FIELDS = ("source_session_id", "accepted_session_id")


@dataclass(frozen=True)
class SessionRetention:
    accepted_thread_ids: frozenset[str]

    def removal_reason(self, metadata: SessionMetadata, *, now: datetime) -> str | None:
        if metadata.codex_thread_source == "subagent" and metadata.thread_id not in self.accepted_thread_ids:
            return "subagent"
        if not metadata.codex_archived:
            return None
        if metadata.derived_stage in _PR_DONE_STAGE_KEYS:
            return "archived_pr"
        if metadata.codex_archived_at is not None and metadata.codex_archived_at <= now - ARCHIVED_USER_SESSION_MIN_AGE:
            return "archived_old"
        return None


@dataclass(frozen=True)
class ProtectionSnapshot:
    retention: SessionRetention
    protected_proposal_session_ids: frozenset[int]
    active_thread_ids: frozenset[str]
    protected_worktree_paths: frozenset[str]

    def protects(self, metadata: SessionMetadata) -> bool:
        if metadata.pk in self.protected_proposal_session_ids or metadata.thread_id in self.active_thread_ids:
            return True
        key = checkouts.managed_key(metadata.cwd)
        return key is None or key in self.protected_worktree_paths


@contextmanager
def removal_lease(cwd: str, *, aliases: Iterable[str], changed_since: datetime) -> Iterator[bool]:
    """Recheck protection under a nonblocking checkout lease before removal."""
    with checkouts.hold(cwd, blocking=False) as acquired:
        yield acquired and not _has_current_protection(cwd, aliases=aliases, changed_since=changed_since)


def _has_current_protection(cwd: str, *, aliases: Iterable[str], changed_since: datetime) -> bool:
    normalized = checkouts.managed_key(cwd)
    if normalized is None:
        return True
    active = list(CodexInstance.objects.filter(
        status__in=CodexInstance.ACTIVE_STATUSES,
    ).values_list("thread_id", "cwd"))
    matching_paths = models.Q(cwd__in=(*aliases, cwd, normalized))
    watched_paths = SessionPullRequest.objects.filter(
        matching_paths | models.Q(updated_at__gte=changed_since),
        models.Q(state__watch_active=True) | models.Q(state__watch_terminal_pending=True),
    ).values_list("cwd", flat=True)
    if any(checkouts.managed_key(path) == normalized for _, path in active if path) or any(
        checkouts.managed_key(path) == normalized for path in watched_paths if path
    ):
        return True
    # Retain known aliases, and resolve only metadata changed since the snapshot
    # to catch newly registered aliases without rescanning the full session index.
    rows = SessionMetadata.objects.filter(
        matching_paths | models.Q(updated_at__gte=changed_since)
    ).only("thread_id", "cwd", "codex_archived", "codex_archived_at", "derived_stage", "codex_thread_source")
    metadata_rows = [row for row in rows if row.cwd and checkouts.managed_key(row.cwd) == normalized]
    active_thread_ids = {thread_id for thread_id, _ in active}
    if any(row.thread_id in active_thread_ids for row in metadata_rows):
        return True
    ids = [row.pk for row in metadata_rows]
    if ProposedSession.objects.filter(outcome_status=ProposedSession.OUTCOME_UNSET).filter(
        models.Q(source_session_id__in=ids) | models.Q(accepted_session_id__in=ids)
    ).exists():
        return True
    retention = SessionRetention(_accepted_thread_ids())
    now = timezone.now()
    return any(retention.removal_reason(row, now=now) is None for row in metadata_rows)


def snapshot(*, now: datetime) -> ProtectionSnapshot:
    retention = SessionRetention(_accepted_thread_ids())
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
    watched_paths = set(
        SessionPullRequest.objects.filter(
            models.Q(state__watch_active=True) | models.Q(state__watch_terminal_pending=True),
        )
        .exclude(cwd="")
        .values_list("cwd", flat=True)
    )
    protected_paths = (
        active_codex_paths
        | watched_paths
        | _pending_proposal_worktree_paths(protected_proposal_session_ids)
        | _protected_visible_user_worktree_paths(
            retention,
            now=now,
        )
    )
    return ProtectionSnapshot(
        retention=retention,
        protected_proposal_session_ids=frozenset(protected_proposal_session_ids),
        active_thread_ids=active_thread_ids,
        protected_worktree_paths=frozenset(_normalized_managed_paths(path for path in protected_paths if path)),
    )


def _accepted_thread_ids() -> frozenset[str]:
    return frozenset(ProposedSession.objects.filter(
        outcome_status=ProposedSession.OUTCOME_ACCEPTED,
        accepted_session__isnull=False,
    ).values_list("accepted_session__thread_id", flat=True))


def _protected_proposal_session_ids() -> set[int]:
    protected = ProposedSession.objects.filter(outcome_status=ProposedSession.OUTCOME_UNSET)
    session_ids: set[int] = set()
    for field in _PROPOSAL_SESSION_ID_FIELDS:
        session_ids.update(
            value
            for value in protected.exclude(**{field: None}).values_list(field, flat=True)
            if isinstance(value, int)
        )
    return session_ids


def _pending_proposal_worktree_paths(session_ids: set[int]) -> set[str]:
    if not session_ids:
        return set()
    return set(SessionMetadata.objects.filter(pk__in=session_ids).exclude(cwd="").values_list("cwd", flat=True))


def _protected_visible_user_worktree_paths(
    retention: SessionRetention,
    *,
    now: datetime,
) -> set[str]:
    paths: set[str] = set()
    rows = (
        SessionMetadata.objects.exclude(cwd="")
        .only(
            "thread_id",
            "cwd",
            "codex_archived",
            "codex_archived_at",
            "derived_stage",
            "codex_thread_source",
        )
    )
    for metadata in rows:
        if retention.removal_reason(metadata, now=now) is None:
            paths.add(metadata.cwd)
    return paths


def _normalized_managed_paths(paths: Iterable[str]) -> set[str]:
    normalized: set[str] = set()
    for path in paths:
        normalized_path = checkouts.managed_key(path)
        if normalized_path is not None:
            normalized.add(normalized_path)
    return normalized
