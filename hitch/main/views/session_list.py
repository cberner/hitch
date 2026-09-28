"""Session list pages: index, inbox, and usage."""
from dataclasses import dataclass
from typing import Any, NamedTuple

from django.db.models import Q
from django.http import (
    HttpRequest,
    HttpResponse,
    JsonResponse,
)
from django.shortcuts import render
from django.template.loader import render_to_string
from django.urls import reverse
from django.views.decorators.http import require_http_methods
from openai_codex import CodexError

from hitch.main.models import (
    Project,
    ProposedSession,
)
from hitch.main.proposals.proposal_display import (
    _attach_proposed_session_display_state,
)
from hitch.main.runtime import app_server_pool, reconciliation
from hitch.main.sessions import session_index
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
    _index_cursor_sort_key,
    _is_index_cursor,
)
from hitch.main.sessions.session_metadata_display import (
    _index_cursor_for_session,
    _non_negative_int,
    _session_index_sort_key,
    _session_row_for_metadata,
    _sorted_visible_index_rows,
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
from hitch.main.views import common


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
    """Project filters for one session-list page."""

    current_project: Project | None
    project_visibility: SessionProjectVisibility | None

_SESSION_PAGE_SIZE = 50

def _session_list_page(
    codex: common.Codex,
    request: HttpRequest,
    *,
    current_settings: SettingsValues,
    projects: list[Project],
    current_project: Project | None,
    project_visibility: SessionProjectVisibility | None,
) -> SessionListPage:
    query = _SessionListQuery(current_project, project_visibility)
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
    allow_refresh_needed: bool = False,
) -> SessionListPage | None:
    required_archived = current_settings.show_archived_sessions
    if not _session_index_sources_complete(include_archived=required_archived):
        return None

    query = _SessionListQuery(current_project, project_visibility)
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
    return _session_list_page_from_index(
        request, query, show_archived=current_settings.show_archived_sessions
    )

def _session_list_page_from_index(
    request: HttpRequest,
    query: _SessionListQuery,
    *,
    show_archived: bool,
) -> SessionListPage:
    rows = session_index.indexed_sessions()
    if query.project_visibility is not None:
        rows = _filter_session_metadata_by_project_visibility(
            rows, query.project_visibility
        )
    elif query.current_project is not None:
        rows = rows.filter(project=query.current_project)
    if not show_archived:
        rows = rows.filter(codex_archived=False)
    # Accepted proposals are user sessions even if their source was a subagent.
    accepted = ProposedSession.objects.filter(
        outcome_status=ProposedSession.OUTCOME_ACCEPTED,
    ).values("accepted_session_id")
    rows = rows.filter(~Q(codex_thread_source="subagent") | Q(pk__in=accepted))
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
    page_thread_ids = [str(session["id"]) for session in page_sort_rows]
    metadata_by_thread_id = {metadata.thread_id: metadata for metadata in rows.filter(thread_id__in=page_thread_ids)}
    page = [
        _session_row_for_metadata(metadata)
        for thread_id in page_thread_ids
        if (metadata := metadata_by_thread_id.get(thread_id)) is not None
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
            )
    except CodexError:
        fallback = _session_list_page_from_warm_index(
            request,
            current_settings=current_settings,
            projects=projects,
            current_project=current_project,
            project_visibility=project_visibility,
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
    )
    if session_page is None:
        session_page = _session_list_page_from_codex_or_warm_index(
            request,
            current_settings=current_settings,
            projects=projects,
            current_project=current_project,
            project_visibility=session_project_visibility,
        )
    _attach_session_stage_context(session_page.sessions)
    settings_context = common._settings_context(current_settings, models_data)
    response = render(
        request,
        "_session_list_content.html" if request.headers.get("X-Hitch-Refresh") == "sessions" else "index.html",
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
def usage(request: HttpRequest) -> HttpResponse:
    usage_context = common._usage_context(request)
    response = render(request, "usage.html", usage_context.template_context)
    _apply_cookie_updates(response, usage_context.cookie_updates)
    return common._prevent_stale_cache(response)


@require_http_methods(["GET"])
def usage_refresh(request: HttpRequest) -> HttpResponse:
    usage_context = common._usage_context(request)
    context = usage_context.template_context
    context["show_project_usage_summary"] = request.GET.get("profile") == "1"
    response = JsonResponse({
        "html": render_to_string("_usage_sections.html", context, request=request),
        "should_poll": context["usage_should_poll"],
        "cursor": context["usage_cursor"],
    })
    _apply_cookie_updates(response, usage_context.cookie_updates)
    return common._prevent_stale_cache(response)

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
            "source_session",
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
