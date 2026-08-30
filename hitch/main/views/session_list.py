"""Session list pages: index, system sessions, inbox, and usage."""
import uuid
from dataclasses import dataclass
from typing import Any, NamedTuple

from django.http import (
    Http404,
    HttpRequest,
    HttpResponse,
)
from django.shortcuts import render
from django.urls import reverse
from django.views.decorators.http import require_http_methods
from openai_codex import CodexError

from hitch.main.goals.autonomous_goal_run_display import (
    _attach_proposed_session_display_state,
)
from hitch.main.models import (
    CodexInstance,
    Project,
    ProposedSession,
    SessionMetadata,
    SystemAgentRun,
)
from hitch.main.runtime import app_server_pool, reconciliation
from hitch.main.sessions import session_index, system_agent_summary
from hitch.main.sessions.project_visibility import (
    _filter_session_metadata_by_project_visibility,
    _project_visibility_label,
    _project_visibility_shows_project_names,
    _session_list_title,
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
    _system_agent_instance_for_thread,
    _system_agent_run_detail_title,
    _system_agent_run_for_thread,
)
from hitch.main.views import common
from hitch.main.workflows import system_agents


class SessionListPage(NamedTuple):
    sessions: list[dict[str, Any]]
    next_cursor: str
    next_offset: int
    next_done: bool
    include_archived_source: bool
    archived_next_cursor: str
    archived_next_offset: int
    archived_next_done: bool

@dataclass
class _SessionListQuery:
    """Shared inputs for building one session-list page.

    Bundles the viewer/filter context that every page builder needs so it is
    threaded through the pagination variants as one value. The id sets are
    shared and mutable so index builders can add hidden system sessions
    discovered from durable metadata.
    """

    projects: list[Project]
    current_project: Project | None
    project_visibility: SessionProjectVisibility | None
    system_only: bool
    hidden_thread_ids: set[str]
    system_thread_ids: set[str]
    runs_by_thread_id: dict[str, SystemAgentRun]
    instances_by_thread_id: dict[str, CodexInstance]

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
    hidden_thread_ids = system_agents.hidden_thread_ids()
    system_thread_ids = hidden_thread_ids if system_only else set()
    query = _SessionListQuery(
        projects=projects,
        current_project=current_project,
        project_visibility=project_visibility,
        system_only=system_only,
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
    if not active_complete or not archived_complete:
        try:
            session_index.refresh_from_codex(
                codex,
                projects=projects,
                include_active=not active_complete,
                include_archived=required_archived and not archived_complete,
                max_pages=None,
            )
        except CodexError:
            common.logger.warning(
                "failed to initialize the session index; rendering indexed sessions"
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
        try:
            refresh_result = session_index.refresh_from_codex(
                codex,
                projects=projects,
                include_active=refresh_active,
                include_archived=refresh_archived,
                max_pages=1,
            )
        except CodexError:
            common.logger.warning(
                "failed to refresh the session index; rendering indexed sessions"
            )
            refresh_result = None
        if (
            refresh_result is not None
            and (
                refresh_result.active_next_cursor
                or (required_archived and refresh_result.archived_next_cursor)
            )
        ):
            common._schedule_session_index_refresh(
                enable_memories=current_settings.enable_memories,
                include_active=bool(refresh_result.active_next_cursor),
                include_archived=bool(
                    required_archived and refresh_result.archived_next_cursor
                ),
            )
    return _session_list_page_from_index(
        request, query, show_archived=current_settings.show_archived_sessions
    )

def _request_uses_index_cursor(request: HttpRequest) -> bool:
    return _is_index_cursor(request.GET.get("cursor", ""))

def _session_index_sources_complete(*, include_archived: bool) -> bool:
    if not session_index.is_complete(archived=False):
        return False
    return not include_archived or session_index.is_complete(archived=True)

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
    if not _session_index_sources_complete(include_archived=required_archived):
        return None

    hidden_thread_ids: set[str] = set()
    system_thread_ids: set[str] = set()
    if system_only:
        hidden_thread_ids = system_agents.hidden_thread_ids()
        system_thread_ids = hidden_thread_ids
    query = _SessionListQuery(
        projects=projects,
        current_project=current_project,
        project_visibility=project_visibility,
        system_only=system_only,
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
) -> SessionListPage:
    _ensure_indexed_system_threads(system_thread_ids, projects=projects)
    legacy_promoted_ids = system_agents.legacy_promoted_system_thread_ids()
    indexed_system_thread_ids = set(
        SessionMetadata.objects.filter(is_hidden_system_session=True)
        .exclude(codex_updated_at__isnull=True)
        .exclude(thread_id__in=legacy_promoted_ids)
        .values_list("thread_id", flat=True)
    )
    system_thread_ids = system_thread_ids | indexed_system_thread_ids
    rows = _system_session_metadata_rows(
        current_project=current_project,
        show_archived=show_archived,
        system_thread_ids=system_thread_ids,
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
        legacy_promoted_ids = system_agents.legacy_promoted_system_thread_ids()
        indexed_system_thread_ids = set(
            rows.filter(is_hidden_system_session=True)
            .exclude(thread_id__in=legacy_promoted_ids)
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
        rows = _filter_visible_session_metadata_rows(rows)
        if query.hidden_thread_ids:
            rows = rows.exclude(thread_id__in=query.hidden_thread_ids)
    if query.system_only:
        metadata_rows = list(rows)
        sessions = [
            session
            for metadata in metadata_rows
            if (
                session := _session_row_for_metadata(
                    metadata,
                    runs_by_thread_id=query.runs_by_thread_id,
                    instances_by_thread_id=query.instances_by_thread_id,
                    system_only=True,
                )
            )
            is not None
        ]
        sessions = _sort_session_rows(sessions)
    else:
        sessions = _sorted_visible_index_rows(rows)
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

def _next_sessions_url(request: HttpRequest, page: SessionListPage) -> str:
    if (
        page.next_done
        and (not page.include_archived_source or page.archived_next_done)
    ):
        return ""
    params = request.GET.copy()
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
    except CodexError:
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
    return common._render_session_detail(
        request,
        session_id,
        read_only=True,
        display_title=_system_agent_run_detail_title(run, instance),
        system_prompt=instance.prompt,
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
            "source_session",
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
