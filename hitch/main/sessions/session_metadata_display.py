"""Index/metadata session-row display and pagination helpers.

Pure code-movement extraction from ``views.py``. These helpers build session
rows from ``SessionMetadata``, compute index-cursor pagination bounds, and sort
visible/system session listings. No behavior changes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from django.db.models import Exists, OuterRef, Q, QuerySet
from django.urls import reverse

from hitch.main import demo
from hitch.main.models import (
    CodexInstance,
    Project,
    SessionMetadata,
    SystemAgentRun,
    SystemWorkflow,
)
from hitch.main.runtime.sdk_values import latest_updated_at, updated_at_seconds
from hitch.main.sessions import session_index
from hitch.main.sessions.session_cursor import _index_cursor_for_sort_key, _IndexCursor
from hitch.main.sessions.system_agent_summary import (
    _system_agent_run_label,
    _system_agent_status,
    _updated_at_sort_key,
)

_SESSION_PAGE_SIZE = 50


def _system_session_metadata_rows(
    *,
    current_project: Project | None,
    show_archived: bool,
    system_thread_ids: set[str],
    accepted_visible_thread_ids: set[str],
) -> QuerySet[SessionMetadata]:
    rows = (
        SessionMetadata.objects.exclude(codex_updated_at__isnull=True)
        .select_related("project")
        .only(
            "thread_id",
            "cwd",
            "codex_display_title",
            "codex_name",
            "codex_updated_at",
            "codex_archived",
            "is_hidden_system_session",
            "project",
            "project__name",
        )
        .order_by("-codex_updated_at", "-thread_id")
    )
    if current_project is not None:
        rows = rows.filter(project=current_project)
    if not show_archived:
        rows = rows.filter(codex_archived=False)
    return rows.exclude(thread_id__in=accepted_visible_thread_ids).filter(
        Q(thread_id__in=system_thread_ids) | Q(is_hidden_system_session=True)
    )


def _legacy_system_metadata_page(
    rows: QuerySet[SessionMetadata], index_cursor: _IndexCursor
) -> tuple[list[SessionMetadata], str, bool]:
    cursor_second_start, cursor_second_end = _index_cursor_second_bounds(index_cursor)
    same_second_rows = rows.filter(
        codex_updated_at__gte=cursor_second_start,
        codex_updated_at__lt=cursor_second_end,
        thread_id__lt=index_cursor.thread_id,
    ).order_by("-thread_id")
    metadata_page = list(same_second_rows[:_SESSION_PAGE_SIZE])
    if len(metadata_page) < _SESSION_PAGE_SIZE:
        earlier_rows = rows.filter(codex_updated_at__lt=cursor_second_start)
        metadata_page.extend(
            earlier_rows[: _SESSION_PAGE_SIZE - len(metadata_page)]
        )
    if not metadata_page or len(metadata_page) < _SESSION_PAGE_SIZE:
        return metadata_page, "", False

    last_metadata = metadata_page[-1]
    if _metadata_in_cursor_second(
        last_metadata,
        start=cursor_second_start,
        end=cursor_second_end,
    ):
        has_more = (
            same_second_rows.filter(thread_id__lt=last_metadata.thread_id).exists()
            or rows.filter(codex_updated_at__lt=cursor_second_start).exists()
        )
        return (
            metadata_page,
            _index_cursor_for_legacy_second(index_cursor, last_metadata)
            if has_more
            else "",
            has_more,
        )

    next_cursor = _index_cursor_for_metadata_row(last_metadata)
    has_more = _metadata_rows_after_index_cursor(rows, next_cursor).exists()
    return (
        metadata_page,
        _index_cursor_for_metadata(last_metadata) if has_more else "",
        has_more,
    )


def _index_cursor_second_bounds(index_cursor: _IndexCursor) -> tuple[datetime, datetime]:
    cursor_second_start = datetime.fromtimestamp(int(index_cursor.updated_at), UTC)
    return cursor_second_start, cursor_second_start + timedelta(seconds=1)


def _metadata_in_cursor_second(
    metadata: SessionMetadata, *, start: datetime, end: datetime
) -> bool:
    updated_at = metadata.codex_updated_at
    return isinstance(updated_at, datetime) and start <= updated_at < end


def _metadata_rows_after_index_cursor(
    rows: QuerySet[SessionMetadata],
    index_cursor: _IndexCursor,
) -> QuerySet[SessionMetadata]:
    if not index_cursor.exact_updated_at:
        cursor_second_start, cursor_second_end = _index_cursor_second_bounds(
            index_cursor
        )
        return rows.filter(
            Q(codex_updated_at__lt=cursor_second_start)
            | Q(
                codex_updated_at__gte=cursor_second_start,
                codex_updated_at__lt=cursor_second_end,
                thread_id__lt=index_cursor.thread_id,
            )
        )
    cursor_updated_at = datetime.fromtimestamp(index_cursor.updated_at, UTC)
    return rows.filter(
        Q(codex_updated_at__lt=cursor_updated_at)
        | Q(codex_updated_at=cursor_updated_at, thread_id__lt=index_cursor.thread_id)
    )


def _filter_visible_session_metadata_rows(
    rows: QuerySet[SessionMetadata],
    *,
    accepted_visible_thread_ids: set[str],
) -> QuerySet[SessionMetadata]:
    system_run_exists = (
        SystemAgentRun.objects.filter(thread_id=OuterRef("thread_id"))
        .exclude(thread_id="")
        .exclude(agent_kind=demo.DEMO_AGENT_KIND)
    )
    system_instance_exists = (
        CodexInstance.objects.filter(
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            thread_id=OuterRef("thread_id"),
        )
        .exclude(thread_id="")
        .exclude(agent_kind=demo.DEMO_AGENT_KIND)
    )
    rows = rows.annotate(
        _has_system_run=Exists(system_run_exists),
        _has_system_instance=Exists(system_instance_exists),
    )
    visible_filter = (
        Q(is_hidden_system_session=False)
        & Q(_has_system_run=False)
        & Q(_has_system_instance=False)
    )
    if accepted_visible_thread_ids:
        visible_filter |= Q(thread_id__in=accepted_visible_thread_ids)
    return rows.filter(visible_filter)


def _sorted_visible_index_rows(
    rows: QuerySet[SessionMetadata],
    *,
    hidden_thread_ids: set[str] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sort_source = list(rows.values("thread_id", "codex_updated_at"))
    main_thread_ids = [str(row["thread_id"]) for row in sort_source]
    qa_updated_at_by_main_thread = _qa_activity_updated_at_by_metadata_thread_ids(
        main_thread_ids,
        hidden_thread_ids,
    )
    return (
        _sort_session_rows(
            [
                {
                    "id": row["thread_id"],
                    "updated_at": latest_updated_at(
                        row["codex_updated_at"],
                        qa_updated_at_by_main_thread.get(row["thread_id"]),
                    ),
                }
                for row in sort_source
            ]
        ),
        qa_updated_at_by_main_thread,
    )


def _ensure_indexed_system_threads(
    system_thread_ids: set[str], *, projects: list[Project]
) -> None:
    missing_thread_ids = set(system_thread_ids) - set(
        SessionMetadata.objects.filter(
            thread_id__in=system_thread_ids,
        )
        .exclude(codex_updated_at__isnull=True)
        .values_list("thread_id", flat=True)
    )
    if not missing_thread_ids:
        return
    instances = (
        CodexInstance.objects.filter(
            thread_id__in=missing_thread_ids,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
        )
        .exclude(thread_id="")
        .order_by("thread_id", "-started_at", "-pk")
    )
    indexed: set[str] = set()
    for instance in instances:
        if instance.thread_id in indexed:
            continue
        indexed.add(instance.thread_id)
        session_index.upsert_local_session(
            thread_id=instance.thread_id,
            cwd=instance.cwd,
            projects=projects,
            name=instance.display_author or instance.agent_kind,
            preview=instance.prompt,
            is_hidden_system_session=instance.agent_kind != demo.DEMO_AGENT_KIND,
        )


def _session_row_for_metadata(
    metadata: SessionMetadata,
    *,
    qa_updated_at_by_main_thread: Mapping[str, Any],
    runs_by_thread_id: dict[str, SystemAgentRun],
    instances_by_thread_id: dict[str, CodexInstance],
    system_only: bool,
) -> dict[str, Any] | None:
    row = {
        "id": metadata.thread_id,
        "cwd": metadata.cwd,
        "updated_at": latest_updated_at(
            metadata.codex_updated_at,
            qa_updated_at_by_main_thread.get(metadata.thread_id),
        ),
        "display_title": metadata.codex_display_title or metadata.thread_id,
        "name_value": metadata.codex_name,
        "is_archived": metadata.codex_archived,
        "project": metadata.project,
    }
    if not system_only:
        row.update(
            {
                "codex_path": metadata.codex_path,
                "has_activity": bool(metadata.codex_preview),
                "stage_main_updated_at": metadata.codex_updated_at,
                "stage_cache_key": metadata.derived_stage,
                "stage_cache_mtime_ns": metadata.derived_stage_source_mtime_ns,
                "stage_pr_refresh_attempted_at": (
                    metadata.derived_stage_pr_refresh_attempted_at
                ),
            }
        )
    if system_only:
        run = runs_by_thread_id.get(metadata.thread_id)
        instance = run.instance if run is not None else instances_by_thread_id.get(metadata.thread_id)
        untracked_hitch_system = metadata.is_hidden_system_session
        if instance is None and not untracked_hitch_system:
            return None
        row.update(
            {
                "detail_url": reverse("system_session", kwargs={"session_id": metadata.thread_id}),
                "system_kind": (
                    _system_agent_run_label(run, instance)
                    if instance is not None
                    else "Hitch system"
                ),
                "system_status": (
                    _system_agent_status(run, instance)
                    if instance is not None
                    else "untracked"
                ),
            }
        )
    return row


def _qa_activity_updated_at_by_metadata_thread_ids(
    main_thread_ids: Iterable[str], hidden_thread_ids: set[str] | None
) -> dict[str, Any]:
    main_thread_ids = [
        thread_id for thread_id in dict.fromkeys(main_thread_ids) if thread_id
    ]
    if not main_thread_ids:
        return {}
    runs = list(
        SystemAgentRun.objects.filter(
            workflow__kind=SystemWorkflow.KIND_PR_QA,
            workflow__main_thread_id__in=main_thread_ids,
        )
        .exclude(thread_id="")
        .select_related("workflow")
    )
    run_thread_ids = {run.thread_id for run in runs if run.thread_id}
    if hidden_thread_ids is not None:
        run_thread_ids &= hidden_thread_ids
    hidden_metadata_by_thread_id = {
        metadata.thread_id: metadata
        for metadata in SessionMetadata.objects.filter(
            thread_id__in=run_thread_ids
        ).only("thread_id", "codex_updated_at", "codex_last_synced_at")
    }
    updated_at_by_main_thread: dict[str, Any] = {}
    for run in runs:
        main_thread_id = run.workflow.main_thread_id
        if not main_thread_id:
            continue
        hidden_metadata = hidden_metadata_by_thread_id.get(run.thread_id)
        local_run_updated_at = latest_updated_at(run.updated_at, run.workflow.updated_at)
        if hidden_metadata is None:
            run_updated_at = local_run_updated_at
        elif (updated_at_seconds(local_run_updated_at) or 0.0) > (
            updated_at_seconds(hidden_metadata.codex_last_synced_at) or 0.0
        ):
            run_updated_at = latest_updated_at(
                hidden_metadata.codex_updated_at,
                local_run_updated_at,
            )
        else:
            run_updated_at = hidden_metadata.codex_updated_at
        updated_at_by_main_thread[main_thread_id] = latest_updated_at(
            updated_at_by_main_thread.get(main_thread_id),
            run_updated_at,
        )
    return updated_at_by_main_thread


def _sort_session_rows(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        sessions,
        key=_session_index_sort_key,
        reverse=True,
    )


def _session_index_sort_key(session: dict[str, Any]) -> tuple[float, str]:
    return (_updated_at_sort_key(session["updated_at"]), str(session["id"]))


def _index_cursor_for_session(session: dict[str, Any]) -> str:
    return _index_cursor_for_sort_key(_session_index_sort_key(session))


def _index_cursor_for_metadata(metadata: SessionMetadata) -> str:
    cursor = _index_cursor_for_metadata_row(metadata)
    return _index_cursor_for_sort_key(cursor.sort_key, exact_updated_at=True)


def _index_cursor_for_metadata_row(metadata: SessionMetadata) -> _IndexCursor:
    return _IndexCursor(
        updated_at=_updated_at_sort_key(metadata.codex_updated_at),
        thread_id=metadata.thread_id,
        exact_updated_at=True,
    )


def _index_cursor_for_legacy_second(
    index_cursor: _IndexCursor, metadata: SessionMetadata
) -> str:
    return _index_cursor_for_sort_key(
        (float(int(index_cursor.updated_at)), metadata.thread_id)
    )


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        return 0
    return max(parsed, 0)
