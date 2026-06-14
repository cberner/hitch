"""Session list pages: index, system sessions, inbox, and usage."""
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, NamedTuple

from django.http import (
    Http404,
    HttpRequest,
    HttpResponse,
)
from django.shortcuts import render
from django.urls import reverse
from django.views.decorators.http import require_http_methods
from openai_codex import AppServerError
from openai_codex.generated.v2_all import (
    SortDirection,
    ThreadSortKey,
)

from hitch.main import demo
from hitch.main.goals.autonomous_goal_run_display import (
    _attach_proposed_session_display_state,
)
from hitch.main.models import (
    CodexInstance,
    Project,
    ProposedSession,
    SessionMetadata,
    SystemAgentRun,
    SystemWorkflow,
)
from hitch.main.runtime import app_server_pool, reconciliation
from hitch.main.runtime.rollout_state import (
    _thread_is_archived,
)
from hitch.main.runtime.sdk_values import (
    string_value,
    updated_at_seconds,
)
from hitch.main.sessions import session_index, system_agent_summary
from hitch.main.sessions.project_visibility import (
    _filter_session_metadata_by_project_visibility,
    _project_visibility_label,
    _project_visibility_shows_project_names,
    _session_list_title,
    _session_project_is_visible,
    _session_project_visibility_context,
)
from hitch.main.sessions.project_visibility import (
    _metadata_by_thread_id as _metadata_by_thread_id,
)
from hitch.main.sessions.session_cursor import (
    _index_cursor,
    _index_cursor_sort_key,
    _is_index_cursor,
)
from hitch.main.sessions.session_entry_display import (
    _display_title,
)
from hitch.main.sessions.session_metadata_display import (
    _ensure_indexed_system_threads,
    _filter_visible_session_metadata_rows,
    _index_cursor_for_metadata,
    _index_cursor_for_metadata_row,
    _index_cursor_for_session,
    _legacy_system_metadata_page,
    _metadata_rows_after_index_cursor,
    _non_negative_int,
    _session_index_sort_key,
    _session_row_for_metadata,
    _sort_session_rows,
    _sorted_visible_index_rows,
    _system_session_metadata_rows,
)
from hitch.main.sessions.session_resume import (
    _session_detail_metadata,
)
from hitch.main.sessions.session_settings import (
    _cached_models_and_settings,
    _selected_project_for_settings,
    _session_project_visibility_for_settings,
)
from hitch.main.sessions.session_stage_refresh import (
    _attach_session_stage_context,
)
from hitch.main.sessions.settings_cookies import (
    SessionProjectVisibility,
    SettingsValues,
    _apply_cookie_updates,
)
from hitch.main.sessions.system_agent_summary import (
    _demo_system_thread_ids,
    _qa_activity_updated_at_by_main_thread_id,
    _session_updated_at,
    _system_agent_instance_for_thread,
    _system_agent_kind,
    _system_agent_run_detail_title,
    _system_agent_run_for_thread,
    _system_agent_run_label,
    _system_agent_status,
    _updated_at_sort_key,
)
from hitch.main.views import common
from hitch.main.workflows import system_agents


class ThreadListPage(NamedTuple):
    threads: list[Any]
    next_cursor: str

class VisibleSessionPage(NamedTuple):
    sessions: list[dict[str, Any]]
    next_cursor: str
    next_offset: int
    needs_materialized_order: bool = False
    materialized_order: bool = False

class QAActivityPageState(NamedTuple):
    updated_at_by_main_thread: dict[str, Any]
    requires_materialized_order: bool

class SessionListPage(NamedTuple):
    sessions: list[dict[str, Any]]
    next_cursor: str
    next_offset: int
    next_done: bool
    include_archived_source: bool
    archived_next_cursor: str
    archived_next_offset: int
    archived_next_done: bool
    materialized_order: bool = False

@dataclass
class SessionPageSource:
    archived: bool
    cursor: str
    offset: int
    page: ThreadListPage | None = None
    next_page_cursor: str = ""
    seen_cursors: set[str] = dataclass_field(default_factory=set)
    metadata_by_thread: dict[str, SessionMetadata] | None = None
    qa_updated_at_by_main_thread: dict[str, Any] | None = None
    candidate: dict[str, Any] | None = None
    candidate_offset: int = 0
    qa_activity_requires_materialized_order: bool = False
    exhausted: bool = False

@dataclass
class _SessionListQuery:
    """Shared inputs for building one session-list page.

    Bundles the viewer/filter context that every page builder needs so it is
    threaded through the pagination variants as one value. The id sets and
    ``project_cache`` are intentionally shared and mutable: builders add
    thread-derived hidden/system ids as they fetch Codex pages, and the cache
    memoizes per-cwd project lookups across rows of the same request.
    """

    projects: list[Project]
    current_project: Project | None
    project_visibility: SessionProjectVisibility | None
    system_only: bool
    accepted_visible_thread_ids: set[str]
    hidden_thread_ids: set[str]
    system_thread_ids: set[str]
    runs_by_thread_id: dict[str, SystemAgentRun]
    instances_by_thread_id: dict[str, CodexInstance]
    project_cache: dict[str, Project | None] = dataclass_field(default_factory=dict)

_SESSION_PAGE_SIZE = 50

def _session_list_page(
    codex: common.Codex,
    request: HttpRequest,
    *,
    current_settings: SettingsValues,
    projects: list[Project],
    current_project: Project | None,
    project_visibility: SessionProjectVisibility | None,
    system_only: bool,
) -> SessionListPage:
    accepted_visible_thread_ids = system_agents.accepted_visible_system_thread_ids()
    hidden_thread_ids = system_agents.hidden_thread_ids(
        accepted_visible_thread_ids=accepted_visible_thread_ids
    )
    system_thread_ids = (
        hidden_thread_ids | _demo_system_thread_ids() if system_only else set()
    )
    query = _SessionListQuery(
        projects=projects,
        current_project=current_project,
        project_visibility=project_visibility,
        system_only=system_only,
        accepted_visible_thread_ids=accepted_visible_thread_ids,
        hidden_thread_ids=hidden_thread_ids,
        system_thread_ids=system_thread_ids,
        runs_by_thread_id=(
            system_agent_summary._system_agent_runs_by_thread_id(system_thread_ids)
            if system_only
            else {}
        ),
        instances_by_thread_id=(
            system_agent_summary._system_agent_instances_by_thread_id(system_thread_ids)
            if system_only
            else {}
        ),
    )
    required_archived = current_settings.show_archived_sessions
    active_complete = session_index.is_complete(archived=False)
    archived_complete = (
        not required_archived or session_index.is_complete(archived=True)
    )
    if (
        not _request_uses_codex_cursor(request)
        and (not active_complete or not archived_complete)
    ):
        session_index.refresh_from_codex(
            codex,
            projects=projects,
            include_active=not active_complete,
            include_archived=required_archived and not archived_complete,
            max_pages=None,
        )
    if (
        _request_uses_codex_cursor(request)
        or not _session_index_sources_complete(include_archived=required_archived)
    ):
        return _session_list_page_from_codex(
            codex, request, query, current_settings=current_settings
        )
    request_uses_index_cursor = _request_uses_index_cursor(request)
    refresh_active = (
        not request_uses_index_cursor
        and (
            session_index.should_refresh(archived=False)
            or session_index.has_pending_pages(archived=False)
        )
    )
    refresh_archived = (
        not request_uses_index_cursor
        and current_settings.show_archived_sessions
        and (
            session_index.should_refresh(archived=True)
            or session_index.has_pending_pages(archived=True)
        )
    )
    if refresh_active or refresh_archived:
        refresh_result = session_index.refresh_from_codex(
            codex,
            projects=projects,
            include_active=refresh_active,
            include_archived=refresh_archived,
            max_pages=1,
        )
        capped_refresh_has_more = bool(
            refresh_result.active_next_cursor
            or (required_archived and refresh_result.archived_next_cursor)
        )
        if capped_refresh_has_more:
            common._schedule_session_index_refresh(
                enable_memories=current_settings.enable_memories,
                include_active=bool(refresh_result.active_next_cursor),
                include_archived=bool(
                    required_archived and refresh_result.archived_next_cursor
                ),
            )
            # Coverage remains the availability contract; a capped freshness
            # probe that sees more pages only hands this response to live
            # cursor pagination when Codex is healthy.
            try:
                return _session_list_page_from_codex(
                    codex, request, query, current_settings=current_settings
                )
            except AppServerError:
                common.logger.warning(
                    "failed to fetch live session page after capped refresh; "
                    "rendering cached sessions"
                )
    return _session_list_page_from_index(
        request, query, show_archived=current_settings.show_archived_sessions
    )

def _request_uses_codex_cursor(request: HttpRequest) -> bool:
    cursor = request.GET.get("cursor", "")
    if cursor and not _is_index_cursor(cursor):
        return True
    return any(
        request.GET.get(param)
        for param in (
            "done",
            "archived_cursor",
            "archived_offset",
            "archived_done",
            "materialized_order",
        )
    )

def _request_uses_index_cursor(request: HttpRequest) -> bool:
    return _is_index_cursor(request.GET.get("cursor", ""))

def _session_index_sources_complete(*, include_archived: bool) -> bool:
    if not session_index.is_complete(archived=False):
        return False
    return not include_archived or session_index.is_complete(archived=True)

def _session_list_page_from_codex(
    codex: common.Codex,
    request: HttpRequest,
    query: _SessionListQuery,
    *,
    current_settings: SettingsValues,
) -> SessionListPage:
    if not query.system_only and request.GET.get("materialized_order") == "1":
        return _materialized_session_list_page_from_codex(
            codex,
            request,
            query,
            include_archived=current_settings.show_archived_sessions,
        )
    if current_settings.show_archived_sessions:
        return _merged_session_list_page_from_codex(codex, request, query)
    active = _visible_session_page_from_codex(
        codex,
        request,
        query,
        archived=False,
        cursor_param="cursor",
    )
    if active.needs_materialized_order:
        return _materialized_session_list_page_from_codex(
            codex,
            request,
            query,
            include_archived=False,
        )
    if active.materialized_order:
        done = not bool(active.next_offset)
        return SessionListPage(
            sessions=active.sessions,
            next_cursor="",
            next_offset=active.next_offset,
            next_done=done,
            include_archived_source=False,
            archived_next_cursor="",
            archived_next_offset=0,
            archived_next_done=True,
            materialized_order=True,
        )
    return SessionListPage(
        sessions=active.sessions,
        next_cursor=active.next_cursor,
        next_offset=active.next_offset,
        next_done=not bool(active.next_cursor or active.next_offset),
        include_archived_source=False,
        archived_next_cursor="",
        archived_next_offset=0,
        archived_next_done=True,
    )

def _session_list_page_from_warm_index(
    request: HttpRequest,
    *,
    current_settings: SettingsValues,
    projects: list[Project],
    current_project: Project | None,
    project_visibility: SessionProjectVisibility | None,
    system_only: bool,
    allow_refresh_needed: bool = False,
) -> SessionListPage | None:
    required_archived = current_settings.show_archived_sessions
    if _request_uses_codex_cursor(request) or not _session_index_sources_complete(
        include_archived=required_archived
    ):
        return None

    accepted_visible_thread_ids = system_agents.accepted_visible_system_thread_ids()
    hidden_thread_ids: set[str] = set()
    system_thread_ids: set[str] = set()
    if system_only:
        hidden_thread_ids = system_agents.hidden_thread_ids(
            accepted_visible_thread_ids=accepted_visible_thread_ids
        )
        system_thread_ids = hidden_thread_ids | _demo_system_thread_ids()
    query = _SessionListQuery(
        projects=projects,
        current_project=current_project,
        project_visibility=project_visibility,
        system_only=system_only,
        accepted_visible_thread_ids=accepted_visible_thread_ids,
        hidden_thread_ids=hidden_thread_ids,
        system_thread_ids=system_thread_ids,
        runs_by_thread_id={},
        instances_by_thread_id={},
    )
    request_uses_index_cursor = _request_uses_index_cursor(request)
    refresh_active = (
        not request_uses_index_cursor
        and (
            session_index.should_refresh(archived=False)
            or session_index.has_pending_pages(archived=False)
        )
    )
    refresh_archived = (
        not request_uses_index_cursor
        and required_archived
        and (
            session_index.should_refresh(archived=True)
            or session_index.has_pending_pages(archived=True)
        )
    )
    if (refresh_active or refresh_archived) and not allow_refresh_needed:
        common._schedule_session_index_refresh(
            enable_memories=current_settings.enable_memories,
            include_active=refresh_active,
            include_archived=refresh_archived,
        )
    if system_only:
        return _system_session_list_page_from_index(
            request,
            current_project=current_project,
            show_archived=current_settings.show_archived_sessions,
            system_thread_ids=system_thread_ids,
            projects=projects,
            accepted_visible_thread_ids=accepted_visible_thread_ids,
        )
    return _session_list_page_from_index(
        request, query, show_archived=current_settings.show_archived_sessions
    )

def _system_session_list_page_from_index(
    request: HttpRequest,
    *,
    current_project: Project | None,
    show_archived: bool,
    system_thread_ids: set[str],
    projects: list[Project],
    accepted_visible_thread_ids: set[str],
) -> SessionListPage:
    _ensure_indexed_system_threads(system_thread_ids, projects=projects)
    indexed_system_thread_ids = set(
        SessionMetadata.objects.filter(is_hidden_system_session=True)
        .exclude(codex_updated_at__isnull=True)
        .exclude(thread_id__in=accepted_visible_thread_ids)
        .values_list("thread_id", flat=True)
    )
    system_thread_ids = system_thread_ids | indexed_system_thread_ids
    rows = _system_session_metadata_rows(
        current_project=current_project,
        show_archived=show_archived,
        system_thread_ids=system_thread_ids,
        accepted_visible_thread_ids=accepted_visible_thread_ids,
    )
    index_cursor = _index_cursor(request.GET.get("cursor", ""))
    next_cursor = ""
    has_more = False
    if index_cursor is not None and not index_cursor.exact_updated_at:
        metadata_page, next_cursor, has_more = _legacy_system_metadata_page(
            rows, index_cursor
        )
        offset = 0
    elif index_cursor is not None:
        rows = _metadata_rows_after_index_cursor(rows, index_cursor)
        offset = 0
        metadata_page = list(rows[:_SESSION_PAGE_SIZE])
    else:
        offset = _non_negative_int(request.GET.get("offset", ""))
        metadata_page = list(rows[offset : offset + _SESSION_PAGE_SIZE])
    page_thread_ids = [metadata.thread_id for metadata in metadata_page]
    runs_by_thread_id = system_agent_summary._system_agent_runs_by_thread_id(
        page_thread_ids
    )
    instances_by_thread_id = (
        system_agent_summary._system_agent_instances_by_thread_id(page_thread_ids)
    )
    page = [
        session
        for metadata in metadata_page
        if (
            session := _session_row_for_metadata(
                metadata,
                qa_updated_at_by_main_thread={},
                runs_by_thread_id=runs_by_thread_id,
                instances_by_thread_id=instances_by_thread_id,
                system_only=True,
            )
        )
        is not None
    ]
    if (
        index_cursor is None or index_cursor.exact_updated_at
    ) and metadata_page and len(metadata_page) == _SESSION_PAGE_SIZE:
        has_more = _metadata_rows_after_index_cursor(
            rows,
            _index_cursor_for_metadata_row(metadata_page[-1]),
        ).exists()
        next_cursor = (
            ""
            if not has_more or not metadata_page
            else _index_cursor_for_metadata(metadata_page[-1])
        )
    next_offset = offset + len(page)
    return SessionListPage(
        sessions=page,
        next_cursor=next_cursor,
        next_offset=0 if next_cursor or not has_more else next_offset,
        next_done=not has_more,
        include_archived_source=False,
        archived_next_cursor="",
        archived_next_offset=0,
        archived_next_done=True,
    )

def _session_list_page_from_index(
    request: HttpRequest,
    query: _SessionListQuery,
    *,
    show_archived: bool,
) -> SessionListPage:
    rows = session_index.indexed_sessions()
    if query.system_only:
        _ensure_indexed_system_threads(query.system_thread_ids, projects=query.projects)
        rows = session_index.indexed_sessions()
        indexed_system_thread_ids = set(
            rows.filter(is_hidden_system_session=True)
            .exclude(thread_id__in=query.accepted_visible_thread_ids)
            .values_list("thread_id", flat=True)
        )
        query.hidden_thread_ids.update(indexed_system_thread_ids)
        query.system_thread_ids.update(indexed_system_thread_ids)
    if query.project_visibility is not None:
        rows = _filter_session_metadata_by_project_visibility(
            rows, query.project_visibility
        )
    elif query.current_project is not None:
        rows = rows.filter(project=query.current_project)
    if not show_archived:
        rows = rows.filter(codex_archived=False)
    if query.system_only:
        rows = rows.filter(thread_id__in=query.system_thread_ids)
    else:
        rows = _filter_visible_session_metadata_rows(
            rows,
            accepted_visible_thread_ids=query.accepted_visible_thread_ids,
        )
        if query.hidden_thread_ids:
            rows = rows.exclude(thread_id__in=query.hidden_thread_ids)
    if query.system_only:
        metadata_rows = list(rows)
        qa_updated_at_by_main_thread: dict[str, Any] = {}
        sessions = [
            session
            for metadata in metadata_rows
            if (
                session := _session_row_for_metadata(
                    metadata,
                    qa_updated_at_by_main_thread=qa_updated_at_by_main_thread,
                    runs_by_thread_id=query.runs_by_thread_id,
                    instances_by_thread_id=query.instances_by_thread_id,
                    system_only=True,
                )
            )
            is not None
        ]
        sessions = _sort_session_rows(sessions)
    else:
        sessions, qa_updated_at_by_main_thread = _sorted_visible_index_rows(
            rows,
            hidden_thread_ids=(
                query.hidden_thread_ids if query.hidden_thread_ids else None
            ),
        )
    index_cursor = _index_cursor_sort_key(request.GET.get("cursor", ""))
    if index_cursor is not None:
        sessions = [
            session
            for session in sessions
            if _session_index_sort_key(session) < index_cursor
        ]
        offset = 0
    else:
        offset = _non_negative_int(request.GET.get("offset", ""))
    page_sort_rows = sessions[offset : offset + _SESSION_PAGE_SIZE]
    if query.system_only:
        page = page_sort_rows
    else:
        page_thread_ids = [str(session["id"]) for session in page_sort_rows]
        metadata_by_thread_id = {
            metadata.thread_id: metadata
            for metadata in rows.filter(thread_id__in=page_thread_ids)
        }
        page = [
            session
            for thread_id in page_thread_ids
            if (metadata := metadata_by_thread_id.get(thread_id)) is not None
            if (
                session := _session_row_for_metadata(
                    metadata,
                    qa_updated_at_by_main_thread=qa_updated_at_by_main_thread,
                    runs_by_thread_id=query.runs_by_thread_id,
                    instances_by_thread_id=query.instances_by_thread_id,
                    system_only=False,
                )
            )
            is not None
        ]
    next_offset = offset + len(page_sort_rows)
    done = next_offset >= len(sessions)
    next_cursor = "" if done or not page else _index_cursor_for_session(page[-1])
    return SessionListPage(
        sessions=page,
        next_cursor=next_cursor,
        next_offset=0 if next_cursor else (next_offset if not done else 0),
        next_done=done,
        include_archived_source=False,
        archived_next_cursor="",
        archived_next_offset=0,
        archived_next_done=True,
    )

def _materialized_session_list_page_from_codex(
    codex: common.Codex,
    request: HttpRequest,
    query: _SessionListQuery,
    *,
    include_archived: bool,
) -> SessionListPage:
    threads = common._all_threads(codex)
    if include_archived:
        threads.extend(common._all_threads(codex, archived=True))
    _add_thread_derived_hidden_ids(query, threads)
    for thread in threads:
        session_index.upsert_thread(thread, projects=query.projects)
    metadata_by_thread = _metadata_by_thread_id(threads)
    qa_updated_at_by_main_thread = _qa_activity_updated_at_by_main_thread_id(
        threads, query.hidden_thread_ids
    )
    sessions = [
        session
        for thread in threads
        if (
            session := _session_row_for_thread(
                thread,
                query,
                metadata_by_thread=metadata_by_thread,
                qa_updated_at_by_main_thread=qa_updated_at_by_main_thread,
            )
        )
        is not None
    ]
    offset = _non_negative_int(request.GET.get("offset", ""))
    page = _sort_session_rows(sessions)[offset : offset + _SESSION_PAGE_SIZE]
    next_offset = offset + len(page)
    done = next_offset >= len(sessions)
    return SessionListPage(
        sessions=page,
        next_cursor="",
        next_offset=next_offset if not done else 0,
        next_done=done,
        include_archived_source=False,
        archived_next_cursor="",
        archived_next_offset=0,
        archived_next_done=True,
        materialized_order=True,
    )

def _merged_session_list_page_from_codex(
    codex: common.Codex,
    request: HttpRequest,
    query: _SessionListQuery,
) -> SessionListPage:
    active = _session_page_source(request, archived=False, cursor_param="cursor")
    archived = _session_page_source(
        request, archived=True, cursor_param="archived_cursor"
    )
    materialized_fallback_allowed = (
        not active.cursor
        and active.offset == 0
        and not archived.cursor
        and archived.offset == 0
    )
    sources = [active, archived]
    sessions: list[dict[str, Any]] = []
    while len(sessions) < _SESSION_PAGE_SIZE:
        candidates = [
            source
            for source in sources
            if _peek_source_session(source, codex, query) is not None
        ]
        if not candidates:
            break
        source = max(
            candidates,
            key=lambda item: (
                _updated_at_sort_key(item.candidate["updated_at"])
                if item.candidate
                else 0
            ),
        )
        session = _pop_source_session(source)
        if session is not None:
            sessions.append(session)
    if materialized_fallback_allowed and active.qa_activity_requires_materialized_order:
        return _materialized_session_list_page_from_codex(
            codex, request, query, include_archived=True
        )
    active_cursor, active_offset, active_done = _source_next_cursor(active)
    archived_cursor, archived_offset, archived_done = _source_next_cursor(archived)
    return SessionListPage(
        sessions=sessions,
        next_cursor=active_cursor,
        next_offset=active_offset,
        next_done=active_done,
        include_archived_source=True,
        archived_next_cursor=archived_cursor,
        archived_next_offset=archived_offset,
        archived_next_done=archived_done,
    )

def _session_page_source(
    request: HttpRequest, *, archived: bool, cursor_param: str
) -> SessionPageSource:
    done = request.GET.get(_cursor_done_param(cursor_param)) == "1"
    return SessionPageSource(
        archived=archived,
        cursor=request.GET.get(cursor_param, ""),
        offset=_non_negative_int(request.GET.get(_cursor_offset_param(cursor_param), "")),
        exhausted=done,
    )

def _peek_source_session(
    source: SessionPageSource,
    codex: common.Codex,
    query: _SessionListQuery,
) -> dict[str, Any] | None:
    if source.candidate is not None or source.exhausted:
        return source.candidate
    while True:
        if source.page is None:
            if source.cursor in source.seen_cursors:
                source.exhausted = True
                return None
            source.seen_cursors.add(source.cursor)
            source.page = _thread_list_page(
                codex, archived=source.archived, cursor=source.cursor
            )
            _add_thread_derived_hidden_ids(query, source.page.threads)
            for thread in source.page.threads:
                session_index.upsert_thread(thread, projects=query.projects)
            source.next_page_cursor = source.page.next_cursor
            source.metadata_by_thread = _metadata_by_thread_id(source.page.threads)
            detect_materialized_order = (
                not query.system_only and not source.archived and source.offset == 0
            )
            qa_activity = _qa_activity_page_state(
                source.page.threads,
                query.hidden_thread_ids,
                detect_materialized_order=detect_materialized_order,
            )
            source.qa_updated_at_by_main_thread = qa_activity.updated_at_by_main_thread
            if detect_materialized_order:
                source.qa_activity_requires_materialized_order = (
                    source.qa_activity_requires_materialized_order
                    or qa_activity.requires_materialized_order
                )
        if source.offset >= len(source.page.threads):
            if not source.next_page_cursor:
                source.exhausted = True
                return None
            source.cursor = source.next_page_cursor
            source.offset = 0
            source.page = None
            source.next_page_cursor = ""
            source.metadata_by_thread = None
            source.qa_updated_at_by_main_thread = None
            continue
        metadata_by_thread = source.metadata_by_thread or {}
        qa_updated_at_by_main_thread = source.qa_updated_at_by_main_thread or {}
        while source.offset < len(source.page.threads):
            index = source.offset
            thread = source.page.threads[index]
            session = _session_row_for_thread(
                thread,
                query,
                metadata_by_thread=metadata_by_thread,
                qa_updated_at_by_main_thread=qa_updated_at_by_main_thread,
            )
            if session is not None:
                source.candidate = session
                source.candidate_offset = index + 1
                return session
            source.offset = index + 1

def _pop_source_session(source: SessionPageSource) -> dict[str, Any] | None:
    session = source.candidate
    if session is not None:
        source.offset = source.candidate_offset
    source.candidate = None
    source.candidate_offset = 0
    return session

def _source_next_cursor(source: SessionPageSource) -> tuple[str, int, bool]:
    if source.exhausted:
        return "", 0, True
    if source.page is None:
        return source.cursor, source.offset, False
    if source.offset < len(source.page.threads):
        return source.cursor, source.offset, False
    if source.next_page_cursor:
        return source.next_page_cursor, 0, False
    return "", 0, True

def _visible_session_page_from_codex(
    codex: common.Codex,
    request: HttpRequest,
    query: _SessionListQuery,
    *,
    archived: bool,
    cursor_param: str,
) -> VisibleSessionPage:
    cursor = request.GET.get(cursor_param, "")
    offset = _non_negative_int(request.GET.get(_cursor_offset_param(cursor_param), ""))
    materialized_fallback_allowed = not cursor and offset == 0
    sessions: list[dict[str, Any]] = []
    seen_cursors: set[str] = set()
    materialized_qa_activity_seen = False
    while len(sessions) < _SESSION_PAGE_SIZE or materialized_qa_activity_seen:
        if cursor in seen_cursors:
            common.logger.warning("thread list returned duplicate cursor; stopping pagination")
            if materialized_qa_activity_seen:
                return _materialized_visible_session_page(sessions)
            return VisibleSessionPage(_sort_session_rows(sessions), "", 0)
        seen_cursors.add(cursor)
        page = _thread_list_page(codex, archived=archived, cursor=cursor)
        _add_thread_derived_hidden_ids(query, page.threads)
        for thread in page.threads:
            session_index.upsert_thread(thread, projects=query.projects)
        can_materialize_qa_activity = (
            materialized_fallback_allowed
            and not query.system_only
            and not archived
            and offset == 0
        )
        if offset >= len(page.threads):
            offset = 0
            next_cursor = page.next_cursor
            if not next_cursor:
                if materialized_qa_activity_seen:
                    return _materialized_visible_session_page(sessions)
                return VisibleSessionPage(_sort_session_rows(sessions), "", 0)
            cursor = next_cursor
            continue
        metadata_by_thread = _metadata_by_thread_id(page.threads)
        qa_activity = _qa_activity_page_state(
            page.threads,
            query.hidden_thread_ids,
            detect_materialized_order=can_materialize_qa_activity,
        )
        qa_updated_at_by_main_thread = qa_activity.updated_at_by_main_thread
        if can_materialize_qa_activity:
            materialized_qa_activity_seen = (
                materialized_qa_activity_seen
                or qa_activity.requires_materialized_order
            )
        for index, thread in enumerate(page.threads[offset:], start=offset):
            session = _session_row_for_thread(
                thread,
                query,
                metadata_by_thread=metadata_by_thread,
                qa_updated_at_by_main_thread=qa_updated_at_by_main_thread,
            )
            if session is not None:
                sessions.append(session)
                if (
                    len(sessions) >= _SESSION_PAGE_SIZE
                    and not materialized_qa_activity_seen
                ):
                    next_offset = index + 1
                    if next_offset < len(page.threads):
                        return VisibleSessionPage(
                            _sort_session_rows(sessions), cursor, next_offset
                        )
                    break
        offset = 0
        next_cursor = page.next_cursor
        if not next_cursor:
            if materialized_qa_activity_seen:
                return _materialized_visible_session_page(sessions)
            return VisibleSessionPage(_sort_session_rows(sessions), "", 0)
        cursor = next_cursor
    return VisibleSessionPage(_sort_session_rows(sessions), cursor, 0)

def _materialized_visible_session_page(
    sessions: list[dict[str, Any]],
) -> VisibleSessionPage:
    sorted_sessions = _sort_session_rows(sessions)
    page = sorted_sessions[:_SESSION_PAGE_SIZE]
    done = len(page) >= len(sorted_sessions)
    return VisibleSessionPage(
        page,
        "",
        0 if done else len(page),
        materialized_order=True,
    )

def _qa_activity_page_state(
    threads: list[Any],
    hidden_thread_ids: set[str],
    *,
    detect_materialized_order: bool,
) -> QAActivityPageState:
    updated_at_by_main_thread = _qa_activity_updated_at_by_main_thread_id(
        threads, hidden_thread_ids
    )
    # Codex cursor pages are ordered by Codex-owned thread activity, but QA
    # runs are Hitch-owned DB activity that can promote their main session
    # ahead of sessions that appear earlier in the fetched Codex page. Once
    # such activity is present, slicing by the Codex cursor before
    # materializing effective order is unsafe.
    requires_materialized_order = detect_materialized_order and (
        bool(updated_at_by_main_thread)
        or _page_has_cross_page_qa_activity(threads, hidden_thread_ids)
    )
    return QAActivityPageState(
        updated_at_by_main_thread=updated_at_by_main_thread,
        requires_materialized_order=requires_materialized_order,
    )

def _page_has_cross_page_qa_activity(
    threads: list[Any], hidden_thread_ids: set[str]
) -> bool:
    thread_ids = {
        thread_id
        for thread in threads
        if isinstance((thread_id := getattr(thread, "id", None)), str)
    }
    hidden_ids = thread_ids & hidden_thread_ids
    if not hidden_ids:
        return False
    return (
        SystemAgentRun.objects.filter(
            workflow__kind=SystemWorkflow.KIND_PR_QA,
            thread_id__in=hidden_ids,
        )
        .exclude(thread_id="")
        .exclude(workflow__main_thread_id="")
        .exclude(workflow__main_thread_id__in=thread_ids)
        .exists()
    )

def _add_thread_derived_hidden_ids(
    query: _SessionListQuery, threads: Iterable[Any]
) -> None:
    thread_hidden_ids = system_agents.hidden_thread_ids_from_threads(
        threads, accepted_visible_thread_ids=query.accepted_visible_thread_ids
    )
    query.hidden_thread_ids.update(thread_hidden_ids)
    if query.system_only:
        query.system_thread_ids.update(thread_hidden_ids)

def _thread_list_page(codex: common.Codex, *, archived: bool, cursor: str) -> ThreadListPage:
    kwargs: dict[str, Any] = {
        "limit": common._THREAD_LIST_FETCH_LIMIT,
        "sort_key": ThreadSortKey.updated_at,
        "sort_direction": SortDirection.desc,
        "use_state_db_only": common._THREAD_LIST_USE_STATE_DB_ONLY,
    }
    if archived:
        kwargs["archived"] = True
    if cursor:
        kwargs["cursor"] = cursor
    response = codex.thread_list(**kwargs)
    next_cursor = getattr(response, "next_cursor", "")
    return ThreadListPage(
        threads=sorted(
            response.data,
            # Normalize through updated_at_seconds: SDK threads may carry
            # datetime, epoch int/float, or no updated_at at all, and sorting a
            # mix (or a datetime against the default 0) would raise TypeError.
            key=lambda thread: updated_at_seconds(getattr(thread, "updated_at", None))
            or 0.0,
            reverse=True,
        ),
        next_cursor=next_cursor if isinstance(next_cursor, str) else "",
    )

def _session_row_for_thread(
    thread: Any,
    query: _SessionListQuery,
    *,
    metadata_by_thread: dict[str, SessionMetadata],
    qa_updated_at_by_main_thread: Mapping[str, Any],
) -> dict[str, Any] | None:
    thread_id = getattr(thread, "id", None)
    if not isinstance(thread_id, str) or not thread_id:
        return None
    if query.system_only:
        if thread_id not in query.system_thread_ids:
            return None
    elif thread_id in query.hidden_thread_ids:
        return None
    session_project = _project_for_thread_cached(
        thread, metadata_by_thread, query.projects, query.project_cache
    )
    if query.project_visibility is not None:
        if not _session_project_is_visible(session_project, query.project_visibility):
            return None
    elif query.current_project is not None and session_project != query.current_project:
        return None
    row = {
        "id": thread_id,
        "cwd": common._thread_cwd(thread) or "",
        "updated_at": _session_updated_at(thread, qa_updated_at_by_main_thread),
        "display_title": _display_title(thread),
        "name_value": getattr(thread, "name", None) or "",
        "is_archived": _thread_is_archived(thread),
        "project": session_project,
    }
    if not query.system_only:
        metadata = metadata_by_thread.get(thread_id)
        codex_path = string_value(getattr(thread, "path", None))
        if not codex_path and metadata is not None:
            codex_path = metadata.codex_path
        row.update(
            {
                "codex_path": codex_path,
                "has_activity": bool(getattr(thread, "preview", None)),
                "stage_main_updated_at": getattr(thread, "updated_at", None),
                "stage_cache_key": metadata.derived_stage if metadata is not None else "",
                "stage_cache_mtime_ns": (
                    metadata.derived_stage_source_mtime_ns
                    if metadata is not None
                    else 0
                ),
                "stage_pr_refresh_attempted_at": (
                    metadata.derived_stage_pr_refresh_attempted_at
                    if metadata is not None
                    else None
                ),
            }
        )
    if query.system_only:
        run = query.runs_by_thread_id.get(thread_id)
        instance = (
            run.instance
            if run is not None
            else query.instances_by_thread_id.get(thread_id)
        )
        row.update(
            {
                "detail_url": reverse(
                    "system_session", kwargs={"session_id": thread_id}
                ),
                "system_kind": (
                    _system_agent_run_label(run, instance)
                    if run is not None or instance is not None
                    else "Hitch system"
                ),
                "system_status": (
                    _system_agent_status(run, instance)
                    if run is not None or instance is not None
                    else "untracked"
                ),
            }
        )
    return row

def _project_for_thread_cached(
    thread: Any,
    metadata_by_thread: dict[str, SessionMetadata],
    projects: list[Project],
    project_cache: dict[str, Project | None],
) -> Project | None:
    thread_id = getattr(thread, "id", "")
    metadata = metadata_by_thread.get(thread_id) if isinstance(thread_id, str) else None
    if metadata is not None and (
        metadata.project_id is not None or metadata.project_cleared
    ):
        return metadata.project
    cwd = common._thread_cwd(thread)
    if not cwd:
        return None
    if cwd not in project_cache:
        project_cache[cwd] = common._project_for_cwd(cwd, projects)
    return project_cache[cwd]

def _next_sessions_url(request: HttpRequest, page: SessionListPage) -> str:
    if (
        page.next_done
        and (not page.include_archived_source or page.archived_next_done)
    ):
        return ""
    params = request.GET.copy()
    if page.materialized_order:
        params["materialized_order"] = "1"
    else:
        params.pop("materialized_order", None)
    _set_cursor_params(
        params, "cursor", page.next_cursor, page.next_offset, page.next_done
    )
    if page.include_archived_source:
        _set_cursor_params(
            params,
            "archived_cursor",
            page.archived_next_cursor,
            page.archived_next_offset,
            page.archived_next_done,
        )
    else:
        _clear_cursor_params(params, "archived_cursor")
    return f"{request.path}?{params.urlencode()}"

def _set_cursor_params(
    params: Any, cursor_param: str, cursor: str, offset: int, done: bool
) -> None:
    offset_param = _cursor_offset_param(cursor_param)
    done_param = _cursor_done_param(cursor_param)
    if done:
        params.pop(cursor_param, None)
        params.pop(offset_param, None)
        params[done_param] = "1"
        return
    params.pop(done_param, None)
    if cursor:
        params[cursor_param] = cursor
        if offset > 0:
            params[offset_param] = str(offset)
        else:
            params.pop(offset_param, None)
        return
    params.pop(cursor_param, None)
    if offset > 0:
        params[offset_param] = str(offset)
    else:
        params.pop(offset_param, None)

def _clear_cursor_params(params: Any, cursor_param: str) -> None:
    params.pop(cursor_param, None)
    params.pop(_cursor_offset_param(cursor_param), None)
    params.pop(_cursor_done_param(cursor_param), None)

def _cursor_offset_param(cursor_param: str) -> str:
    if cursor_param == "cursor":
        return "offset"
    return f"{cursor_param.removesuffix('_cursor')}_offset"

def _cursor_done_param(cursor_param: str) -> str:
    if cursor_param == "cursor":
        return "done"
    return f"{cursor_param.removesuffix('_cursor')}_done"

def _session_list_page_from_codex_or_warm_index(
    request: HttpRequest,
    *,
    current_settings: SettingsValues,
    projects: list[Project],
    current_project: Project | None,
    project_visibility: SessionProjectVisibility | None,
    system_only: bool,
) -> SessionListPage:
    try:
        with app_server_pool.borrow_codex(
            common.Codex, enable_memories=current_settings.enable_memories
        ) as codex:
            return _session_list_page(
                codex,
                request,
                current_settings=current_settings,
                projects=projects,
                current_project=current_project,
                project_visibility=project_visibility,
                system_only=system_only,
            )
    except AppServerError:
        fallback = _session_list_page_from_warm_index(
            request,
            current_settings=current_settings,
            projects=projects,
            current_project=current_project,
            project_visibility=project_visibility,
            system_only=system_only,
            allow_refresh_needed=True,
        )
        if fallback is None:
            raise
        common.logger.warning("failed to open live session list; rendering cached sessions")
        return fallback

def index(request: HttpRequest) -> HttpResponse:
    # Sweep workers whose pid is gone: a Popen that crashed before a worker
    # could record its terminal status (or a row stuck in ``starting``)
    # otherwise stays pending forever, since we don't run a periodic task.
    reconciliation.reconcile_dead_if_due()
    models_data, resolved_settings = _cached_models_and_settings(request)
    current_settings = resolved_settings.values
    cookie_updates = resolved_settings.cookie_updates
    projects = list(Project.objects.all())
    current_project = _selected_project_for_settings(current_settings, projects)
    session_project_visibility = _session_project_visibility_for_settings(
        current_settings, projects
    )
    session_page = _session_list_page_from_warm_index(
        request,
        current_settings=current_settings,
        projects=projects,
        current_project=current_project,
        project_visibility=session_project_visibility,
        system_only=False,
    )
    if session_page is None:
        session_page = _session_list_page_from_codex_or_warm_index(
            request,
            current_settings=current_settings,
            projects=projects,
            current_project=current_project,
            project_visibility=session_project_visibility,
            system_only=False,
        )
    _attach_session_stage_context(session_page.sessions)
    settings_context = common._settings_context(current_settings, models_data)
    response = render(
        request,
        "index.html",
        {
            "sessions": session_page.sessions,
            "next_sessions_url": _next_sessions_url(request, session_page),
            "has_projects": bool(projects),
            "archived_visibility_url": reverse("update_archived_session_visibility"),
            "login_url": reverse("login"),
            "register_url": reverse("register"),
            "current_show_archived_sessions": current_settings.show_archived_sessions,
            "current_project": current_project,
            "session_list_title": _session_list_title(
                session_project_visibility, projects
            ),
            "name_max_len": common._NAME_MAX_LEN,
            "display_title_max_len": session_index.DISPLAY_TITLE_MAX_LEN,
            "show_new_session_controls": True,
            **settings_context,
            **_session_project_visibility_context(
                session_project_visibility, projects
            ),
        },
    )
    _apply_cookie_updates(response, cookie_updates)
    return common._prevent_stale_cache(response)

@require_http_methods(["GET"])
def system_sessions(request: HttpRequest) -> HttpResponse:
    reconciliation.reconcile_dead_if_due()
    models_data, resolved_settings = _cached_models_and_settings(request)
    current_settings = resolved_settings.values
    cookie_updates = resolved_settings.cookie_updates
    projects = list(Project.objects.all())
    current_project = _selected_project_for_settings(current_settings, projects)
    session_page = _session_list_page_from_warm_index(
        request,
        current_settings=current_settings,
        projects=projects,
        current_project=current_project,
        project_visibility=None,
        system_only=True,
    )
    if session_page is None:
        session_page = _session_list_page_from_codex_or_warm_index(
            request,
            current_settings=current_settings,
            projects=projects,
            current_project=current_project,
            project_visibility=None,
            system_only=True,
        )
    settings_context = common._settings_context(current_settings, models_data)
    response = render(
        request,
        "index.html",
        {
            "sessions": session_page.sessions,
            "next_sessions_url": _next_sessions_url(request, session_page),
            "has_projects": bool(projects),
            "archived_visibility_url": reverse("update_archived_session_visibility"),
            "login_url": reverse("login"),
            "register_url": reverse("register"),
            "current_show_archived_sessions": current_settings.show_archived_sessions,
            "name_max_len": common._NAME_MAX_LEN,
            "display_title_max_len": session_index.DISPLAY_TITLE_MAX_LEN,
            "system_session_list": True,
            "show_new_session_controls": False,
            **settings_context,
        },
    )
    _apply_cookie_updates(response, cookie_updates)
    return common._prevent_stale_cache(response)

@require_http_methods(["GET"])
def system_session(request: HttpRequest, session_id: str) -> HttpResponse:
    run_id = _positive_int(request.GET.get("run_id", ""))
    run = _system_agent_run_for_thread(session_id, run_id=run_id)
    instance = run.instance if run is not None else None
    if instance is None and run_id is None:
        instance = _system_agent_instance_for_thread(session_id)
    if instance is None:
        if run_id is not None:
            raise Http404("system session not found")
        metadata = _session_detail_metadata(session_id)
        if not _valid_codex_session_id(session_id) and not (
            metadata is not None and metadata.is_hidden_system_session
        ):
            raise Http404("system session not found")
        return common._render_session_detail(
            request,
            session_id,
            read_only=True,
            require_system_agent_thread=True,
        )
    agent_kind = _system_agent_kind(run, instance)
    hide_demo_agent_entries = agent_kind != demo.DEMO_AGENT_KIND
    return common._render_session_detail(
        request,
        session_id,
        read_only=True,
        display_title=_system_agent_run_detail_title(run, instance),
        system_prompt=instance.prompt,
        hide_demo_agent_entries=hide_demo_agent_entries,
        demo_entries_run_id=(
            run.pk if not hide_demo_agent_entries and run is not None else None
        ),
    )

@require_http_methods(["GET"])
def usage(request: HttpRequest) -> HttpResponse:
    usage_context = common._usage_context(request)
    response = render(request, "usage.html", usage_context.template_context)
    _apply_cookie_updates(response, usage_context.cookie_updates)
    return response

@require_http_methods(["GET"])
def inbox(request: HttpRequest) -> HttpResponse:
    reconciliation.reconcile_dead_if_due()
    models_data, resolved_settings = _cached_models_and_settings(request)
    current_settings = resolved_settings.values
    cookie_updates = resolved_settings.cookie_updates
    projects = list(Project.objects.all())
    current_project = _selected_project_for_settings(current_settings, projects)
    inbox_project_visibility = _session_project_visibility_for_settings(
        current_settings, projects
    )
    proposed_sessions = list(
        common._proposed_session_inbox_queryset(inbox_project_visibility)
        .select_related(
            "project",
            "autonomous_goal",
            "candidate_session",
            "judge_session",
            "source_workflow",
        )
        .order_by("created_at", "id")
    )
    _attach_proposed_session_display_state(proposed_sessions)
    settings_context = common._settings_context(current_settings, models_data)
    response = render(
        request,
        "inbox.html",
        {
            "login_url": reverse("login"),
            "register_url": reverse("register"),
            "current_project": current_project,
            "inbox_project_label": _project_visibility_label(
                inbox_project_visibility, projects
            ),
            "proposed_sessions": proposed_sessions,
            "show_inbox_project_names": _project_visibility_shows_project_names(
                inbox_project_visibility
            ),
            "proposed_session_rejected_status": ProposedSession.OUTCOME_REJECTED,
            "proposed_session_dismissed_status": ProposedSession.OUTCOME_DISMISSED,
            **_session_project_visibility_context(inbox_project_visibility, projects),
            **settings_context,
        },
    )
    _apply_cookie_updates(response, cookie_updates)
    return response

def _positive_int(value: str) -> int | None:
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None

def _valid_codex_session_id(session_id: str) -> bool:
    value = session_id.removeprefix("urn:uuid:")
    try:
        uuid.UUID(value)
    except ValueError:
        return False
    return True
