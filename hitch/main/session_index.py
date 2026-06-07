"""Cached session list metadata for fast index pages."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, NamedTuple

from django.db.models import QuerySet
from django.utils import timezone
from openai_codex import AppServerError, Codex
from openai_codex.generated.v2_all import SortDirection, ThreadSortKey

from hitch.main.models import Project, SessionIndexSyncState, SessionMetadata
from hitch.main.repos import same_repo_or_worktree

logger = logging.getLogger(__name__)

DISPLAY_TITLE_MAX_LEN = 80
THREAD_LIST_FETCH_LIMIT = 100
ARCHIVED_SESSIONS_DIR = "archived_sessions"
# Codex's archived rollouts live at most four levels below the
# ``archived_sessions/`` directory (``archived_sessions/YYYY/MM/DD/rollout-*.jsonl``);
# five gives a small cushion for future structural changes without re-opening
# the false-positive case where a user's CODEX_HOME unrelatedly traverses an
# ``archived_sessions`` parent.
_ARCHIVED_SESSIONS_ANCESTOR_DEPTH = 5
STALE_AFTER = timedelta(seconds=30)
HIDDEN_SYSTEM_THREAD_SOURCE = "subagent"
AUTONOMOUS_GOAL_AGENT_PROMPT_TITLE = "You are Hitch's autonomous goal agent."
AUTONOMOUS_GOAL_JUDGE_PROMPT_TITLE = (
    "You are Hitch's autonomous goal confidence judge."
)
LEGACY_AUTONOMOUS_GOAL_AGENT_PROMPT_TITLE = "You are Hitch's standing order agent."
LEGACY_AUTONOMOUS_GOAL_JUDGE_PROMPT_TITLE = (
    "You are Hitch's standing order confidence judge."
)
_AUTONOMOUS_GOAL_AGENT_PROMPT_TITLES = (
    AUTONOMOUS_GOAL_AGENT_PROMPT_TITLE,
    LEGACY_AUTONOMOUS_GOAL_AGENT_PROMPT_TITLE,
)
_AUTONOMOUS_GOAL_JUDGE_PROMPT_TITLES = (
    AUTONOMOUS_GOAL_JUDGE_PROMPT_TITLE,
    LEGACY_AUTONOMOUS_GOAL_JUDGE_PROMPT_TITLE,
)


class RefreshResult(NamedTuple):
    synced: int
    failed: bool
    active_next_cursor: str = ""
    archived_next_cursor: str = ""


class _SourceRefreshResult(NamedTuple):
    synced: int
    complete: bool
    next_cursor: str
    seen_thread_ids: set[str]


class ActiveWindowResult(NamedTuple):
    synced: int
    next_cursor: str
    complete: bool
    failed: bool


def should_refresh(*, archived: bool) -> bool:
    source = (
        SessionIndexSyncState.SOURCE_ARCHIVED
        if archived
        else SessionIndexSyncState.SOURCE_ACTIVE
    )
    last_synced = (
        SessionIndexSyncState.objects.filter(source=source)
        .values_list("last_synced_at", flat=True)
        .first()
    )
    if last_synced is None:
        return True
    return last_synced < timezone.now() - STALE_AFTER


def is_complete(*, archived: bool) -> bool:
    source = _source_name(archived=archived)
    return bool(
        SessionIndexSyncState.objects.filter(source=source).values_list(
            "is_complete", flat=True
        ).first()
    )


def has_pending_pages(*, archived: bool) -> bool:
    source = _source_name(archived=archived)
    return bool(
        SessionIndexSyncState.objects.filter(source=source).values_list(
            "next_cursor", flat=True
        ).first()
    )


def has_indexed_sessions() -> bool:
    return SessionMetadata.objects.exclude(codex_updated_at__isnull=True).exists()


def refresh_from_codex(
    codex: Codex,
    *,
    projects: list[Project],
    include_active: bool = True,
    include_archived: bool = False,
    use_state_db_only: bool = True,
    max_pages: int | None = 1,
    allow_completion: bool = True,
) -> RefreshResult:
    synced = 0
    failed = False
    active_next_cursor = ""
    archived_next_cursor = ""
    sources = []
    if include_active:
        sources.append(False)
    if include_archived:
        sources.append(True)
    for archived in sources:
        try:
            source_result = _refresh_source(
                codex,
                projects=projects,
                archived=archived,
                use_state_db_only=use_state_db_only,
                max_pages=max_pages,
            )
            synced += source_result.synced
            mark_synced(
                archived=archived,
                complete=source_result.complete and allow_completion,
                next_cursor=source_result.next_cursor,
            )
            if source_result.complete and allow_completion and not use_state_db_only:
                _invalidate_absent_source_rows(
                    archived=archived,
                    seen_thread_ids=source_result.seen_thread_ids,
                )
            if archived:
                archived_next_cursor = source_result.next_cursor
            else:
                active_next_cursor = source_result.next_cursor
        except AppServerError:
            failed = True
            logger.warning("failed to refresh %s session index", "archived" if archived else "active")
    return RefreshResult(
        synced=synced,
        failed=failed,
        active_next_cursor=active_next_cursor,
        archived_next_cursor=archived_next_cursor,
    )


def refresh_active_window(
    codex: Codex,
    *,
    projects: list[Project],
    start_cursor: str = "",
    max_pages: int = 1,
) -> ActiveWindowResult:
    """Refresh one bounded window of the *active* session index from a cursor.

    The background scheduler used to rescan the entire active list on every
    tick (``max_pages=None``). This pages a bounded number of pages from
    ``start_cursor`` instead and returns the cursor to resume from, so the
    scheduler covers the whole list incrementally across ticks without holding
    one app-server busy on a full sweep every minute. An empty returned
    ``next_cursor`` means the list was fully traversed; the caller should resume
    from the front next cycle.

    Deliberately self-contained relative to ``SessionIndexSyncState``: a partial
    background window is not a completed sync and must not advance the
    request-path freshness/pagination cursor. On a *completed* pass it does bump
    the freshness signal (``mark_synced``) so an idle dashboard still skips its
    own refresh, matching the old full-sweep behavior.
    """
    try:
        result = _refresh_source(
            codex,
            projects=projects,
            archived=False,
            use_state_db_only=True,
            max_pages=max_pages,
            start_cursor=start_cursor or None,
        )
    except AppServerError:
        logger.warning("failed to refresh active session index window")
        return ActiveWindowResult(
            synced=0, next_cursor=start_cursor, complete=False, failed=True
        )
    if result.complete:
        mark_synced(archived=False, complete=True)
        return ActiveWindowResult(
            synced=result.synced, next_cursor="", complete=True, failed=False
        )
    return ActiveWindowResult(
        synced=result.synced,
        next_cursor=result.next_cursor,
        complete=False,
        failed=False,
    )


def mark_synced(*, archived: bool, complete: bool, next_cursor: str = "") -> None:
    source = _source_name(archived=archived)
    previous_complete = is_complete(archived=archived)
    SessionIndexSyncState.objects.update_or_create(
        source=source,
        defaults={
            "last_synced_at": timezone.now(),
            "is_complete": previous_complete or complete,
            "next_cursor": "" if complete else next_cursor,
        },
    )


def upsert_thread(thread: Any, *, projects: list[Project]) -> SessionMetadata | None:
    thread_id = getattr(thread, "id", None)
    if not isinstance(thread_id, str) or not thread_id:
        return None
    cwd = _thread_cwd(thread) or ""
    archived = _thread_is_archived(thread)
    existing = SessionMetadata.objects.filter(thread_id=thread_id).first()
    defaults = _codex_defaults(
        thread_id=thread_id,
        cwd=cwd,
        name=getattr(thread, "name", None),
        preview=getattr(thread, "preview", None),
        created_at=_timestamp_to_datetime(getattr(thread, "created_at", None)),
        updated_at=_timestamp_to_datetime(getattr(thread, "updated_at", None)),
        archived=archived,
        path=getattr(thread, "path", None),
        thread_source=_metadata_value(getattr(thread, "thread_source", None)),
        existing=existing,
    )
    if existing is None:
        defaults["project"] = _project_for_cwd(cwd, projects)
        defaults["project_cleared"] = False
    elif existing.project_id is None and not existing.project_cleared:
        defaults["project"] = _project_for_cwd(cwd, projects)
    metadata, _created = SessionMetadata.objects.update_or_create(
        thread_id=thread_id,
        defaults=defaults,
    )
    return metadata


def upsert_local_session(
    *,
    thread_id: str,
    cwd: str,
    projects: list[Project] | None = None,
    project: Project | None = None,
    project_cleared: bool = False,
    name: str = "",
    preview: str = "",
    archived: bool = False,
    auto_pr_enabled: bool | None = None,
    auto_qa_enabled: bool | None = None,
    auto_merge_to_local_branch: bool | None = None,
    auto_merge_branch: str | None = None,
    codex_path: str | None = None,
    is_hidden_system_session: bool = False,
) -> SessionMetadata:
    now = timezone.now()
    existing = SessionMetadata.objects.filter(thread_id=thread_id).first()
    defaults = _codex_defaults(
        thread_id=thread_id,
        cwd=cwd,
        name=name,
        preview=preview,
        created_at=now,
        updated_at=now,
        archived=archived,
        path=codex_path,
        thread_source="",
        existing=existing,
    )
    if codex_path is None:
        defaults.pop("codex_path", None)
    defaults["project"] = None if project_cleared else project
    if defaults["project"] is None and projects is not None and not project_cleared:
        defaults["project"] = _project_for_cwd(cwd, projects)
    defaults["project_cleared"] = project_cleared
    if auto_pr_enabled is not None:
        defaults["auto_pr_enabled"] = auto_pr_enabled
    if auto_qa_enabled is not None:
        defaults["auto_qa_enabled"] = auto_qa_enabled
    if auto_merge_to_local_branch is not None:
        defaults["auto_merge_to_local_branch"] = auto_merge_to_local_branch
    if auto_merge_branch is not None:
        defaults["auto_merge_branch"] = auto_merge_branch
    defaults["is_hidden_system_session"] = is_hidden_system_session
    metadata, _created = SessionMetadata.objects.update_or_create(
        thread_id=thread_id,
        defaults=defaults,
    )
    return metadata


def update_cached_name(thread_id: str, name: str) -> None:
    display_title = display_title_for(thread_id=thread_id, name=name, preview="")
    SessionMetadata.objects.filter(thread_id=thread_id).update(
        codex_name=name,
        codex_display_title=display_title,
        codex_last_synced_at=timezone.now(),
    )


def update_cached_archived(thread_id: str, *, archived: bool) -> None:
    now = timezone.now()
    archived_at = now if archived else None
    SessionMetadata.objects.filter(thread_id=thread_id).update(
        codex_archived=archived,
        codex_archived_at=archived_at,
        codex_updated_at=now,
        codex_last_synced_at=now,
    )


def record_turn_activity(thread_id: str, *, updated_at: datetime | None = None) -> None:
    """Bump a session's recency after a worker turn completes.

    Worker turns run against an isolated Codex ``sqlite_home`` (see
    ``codex_pool``), so their thread-metadata writes never reach the web home's
    state DB that ``thread_list(use_state_db_only=True)`` -- and hence the
    background index refresh -- reads. Writing ``codex_updated_at`` straight to
    the existing row keeps the session list ordered by real activity without
    giving up either the DB-only listing speed or the per-worker isolation.
    No-op when the row is absent (a later refresh creates it from the web home,
    where the thread was registered at creation, and subsequent turns bump it).
    """
    now = timezone.now()
    SessionMetadata.objects.filter(thread_id=thread_id).update(
        codex_updated_at=updated_at or now,
        codex_last_synced_at=now,
    )


def indexed_sessions() -> QuerySet[SessionMetadata]:
    return (
        SessionMetadata.objects.exclude(codex_updated_at__isnull=True)
        .select_related("project")
        .order_by("-codex_updated_at", "-pk")
    )


def display_title_for(*, thread_id: str, name: object, preview: object) -> str:
    candidate = name.strip() if isinstance(name, str) else ""
    if not candidate:
        candidate = (preview if isinstance(preview, str) else "").split("\n", 1)[
            0
        ].strip()
    if not candidate:
        return thread_id
    if len(candidate) > DISPLAY_TITLE_MAX_LEN:
        return candidate[:DISPLAY_TITLE_MAX_LEN].rstrip() + "..."
    return candidate


def hidden_system_session_from_metadata(
    *, name: str, preview: str, thread_source: str
) -> bool:
    if thread_source == HIDDEN_SYSTEM_THREAD_SOURCE:
        return True
    if name in _AUTONOMOUS_GOAL_AGENT_PROMPT_TITLES:
        return (
            preview.startswith(f"{name}\n\n")
            and (
                (
                    "Autonomous goal title:" in preview
                    and "Autonomous goal objective:" in preview
                )
                or (
                    "Standing order title:" in preview
                    and "Standing order goal:" in preview
                )
            )
            and "Return only JSON matching this shape:" in preview
        )
    if name in _AUTONOMOUS_GOAL_JUDGE_PROMPT_TITLES:
        return (
            preview.startswith(f"{name}\n\n")
            and (
                "Autonomous goal title:" in preview
                or "Standing order title:" in preview
            )
            and "Candidate session JSON:" in preview
            and "Return only JSON matching this shape:" in preview
        )
    return False


def updated_at_seconds(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, datetime):
        return value.timestamp()
    return 0.0


def _refresh_source(
    codex: Codex,
    *,
    projects: list[Project],
    archived: bool,
    use_state_db_only: bool,
    max_pages: int | None,
    start_cursor: str | None = None,
) -> _SourceRefreshResult:
    cursor: str | None = start_cursor or None
    # Seed the duplicate-cursor guard with the entry cursor. A windowed refresh
    # (small ``max_pages``, resuming from ``start_cursor`` each tick) otherwise
    # starts every call with an empty set, so a ``thread_list`` response that
    # returns the same cursor it was called with goes undetected and the
    # scheduler refreshes that one page forever instead of progressing.
    seen_cursors: set[str] = {cursor} if cursor else set()
    pages = 0
    synced = 0
    seen_thread_ids: set[str] = set()
    while max_pages is None or pages < max_pages:
        kwargs: dict[str, Any] = {
            "limit": THREAD_LIST_FETCH_LIMIT,
            "sort_key": ThreadSortKey.updated_at,
            "sort_direction": SortDirection.desc,
            "use_state_db_only": use_state_db_only,
        }
        if archived:
            kwargs["archived"] = True
        if cursor:
            kwargs["cursor"] = cursor
        response = codex.thread_list(**kwargs)
        for thread in response.data:
            if (metadata := upsert_thread(thread, projects=projects)) is not None:
                seen_thread_ids.add(metadata.thread_id)
                synced += 1
        pages += 1
        next_cursor = getattr(response, "next_cursor", None)
        if not isinstance(next_cursor, str) or not next_cursor:
            return _SourceRefreshResult(
                synced=synced,
                complete=True,
                next_cursor="",
                seen_thread_ids=seen_thread_ids,
            )
        if next_cursor in seen_cursors:
            # ``thread_list`` handed back a cursor we already paged from (often
            # the very one this window started at). Resuming from it would pin an
            # incremental scheduler on the same page every tick, so reset to the
            # front: report not-complete with an empty cursor and let the next
            # cycle start a clean pass rather than re-pinning the stuck cursor.
            logger.warning(
                "thread list returned duplicate cursor; resetting session index "
                "refresh to the front"
            )
            return _SourceRefreshResult(
                synced=synced,
                complete=False,
                next_cursor="",
                seen_thread_ids=seen_thread_ids,
            )
        seen_cursors.add(next_cursor)
        cursor = next_cursor
    return _SourceRefreshResult(
        synced=synced,
        complete=False,
        next_cursor=cursor or "",
        seen_thread_ids=seen_thread_ids,
    )


def _invalidate_absent_source_rows(*, archived: bool, seen_thread_ids: set[str]) -> None:
    SessionMetadata.objects.exclude(thread_id__in=seen_thread_ids).filter(
        codex_updated_at__isnull=False,
        codex_archived=archived,
    ).update(codex_updated_at=None, codex_last_synced_at=timezone.now())


def _source_name(*, archived: bool) -> str:
    return (
        SessionIndexSyncState.SOURCE_ARCHIVED
        if archived
        else SessionIndexSyncState.SOURCE_ACTIVE
    )


def _codex_defaults(
    *,
    thread_id: str,
    cwd: str,
    name: object,
    preview: object,
    created_at: datetime | None,
    updated_at: datetime | None,
    archived: bool,
    path: object,
    thread_source: str,
    existing: SessionMetadata | None,
) -> dict[str, Any]:
    now = timezone.now()
    name_value = name.strip() if isinstance(name, str) else ""
    preview_value = preview if isinstance(preview, str) else ""
    archived_at = None
    if archived:
        archived_at = (
            existing.codex_archived_at
            if existing is not None and existing.codex_archived
            else None
        )
        if archived_at is None:
            archived_at = now
    # Never regress recency. A worker turn on an isolated sqlite_home bumps the
    # cached row directly (record_turn_activity), but the web home's
    # thread.updated_at stays at pre-turn time, so a DB-only refresh would
    # otherwise overwrite the fresher worker bump with that stale value and the
    # session would fall back down the list right after completing.
    codex_updated_at = updated_at or created_at or now
    if existing is not None and existing.codex_updated_at is not None:
        codex_updated_at = max(codex_updated_at, existing.codex_updated_at)
    is_hidden_system_session = hidden_system_session_from_metadata(
        name=name_value,
        preview=preview_value,
        thread_source=thread_source,
    )
    if existing is not None and existing.is_hidden_system_session:
        is_hidden_system_session = True
    return {
        "cwd": cwd,
        "codex_display_title": display_title_for(
            thread_id=thread_id, name=name_value, preview=preview_value
        ),
        "codex_name": name_value,
        "codex_preview": preview_value,
        "codex_created_at": created_at or updated_at or now,
        "codex_updated_at": codex_updated_at,
        "codex_archived": archived,
        "codex_archived_at": archived_at,
        "codex_path": path if isinstance(path, str) else "",
        "codex_thread_source": thread_source,
        "codex_last_synced_at": now,
        "is_hidden_system_session": is_hidden_system_session,
    }


def _timestamp_to_datetime(value: object) -> datetime | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return datetime.fromtimestamp(value, UTC)
    if isinstance(value, datetime):
        return value
    return None


def _thread_cwd(thread: Any) -> str | None:
    raw = getattr(thread, "cwd", None)
    if isinstance(raw, str):
        return raw or None
    root = getattr(raw, "root", None)
    return root if isinstance(root, str) and root else None


def _thread_is_archived(thread: Any) -> bool:
    # Codex's ``Thread`` exposes archive state as a boolean; trust it over the
    # path heuristic. ``thread_list(archived=True)`` can return threads whose
    # rollout file still lives in the active-storage tree (the archive flag is
    # set before -- or independently of -- moving the rollout file), so
    # falling through to the path check would silently cache them as active
    # and surface them in the wrong session list.
    archived = getattr(thread, "archived", None)
    if isinstance(archived, bool):
        return archived
    path = getattr(thread, "path", None)
    if not isinstance(path, str) or not path:
        return False
    # Walk the rollout file's immediate ancestry rather than scanning the full
    # path. Codex nests archived rollouts at most a handful of date-segmented
    # levels deep under ``archived_sessions/``; matching ``archived_sessions``
    # anywhere in ``Path(path).parts`` would falsely flag every active
    # session whose ``CODEX_HOME`` happens to traverse an unrelated parent
    # directory of that name (e.g. an org-wide
    # ``/data/archived_sessions/<user>/.codex`` layout).
    return any(
        parent.name == ARCHIVED_SESSIONS_DIR
        for parent in list(Path(path).parents)[:_ARCHIVED_SESSIONS_ANCESTOR_DEPTH]
    )


def _metadata_value(value: Any) -> str:
    root = getattr(value, "root", value)
    raw = getattr(root, "value", root)
    return raw if isinstance(raw, str) else ""


def _project_for_cwd(cwd: str, projects: Iterable[Project]) -> Project | None:
    if not cwd:
        return None
    return next(
        (
            project
            for project in projects
            if same_repo_or_worktree(cwd, project.repo_path, project.git_common_dir)
        ),
        None,
    )
