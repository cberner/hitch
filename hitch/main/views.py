import contextlib
import json
import logging
import math
import os
import re
import threading
import uuid
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, NamedTuple
from urllib.parse import urlencode

from django.conf import settings as django_settings
from django.contrib.auth import login as auth_login
from django.contrib.auth import logout as auth_logout
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.core import signing
from django.core.files.uploadedfile import UploadedFile
from django.db import IntegrityError, close_old_connections, transaction
from django.db.models import Exists, OuterRef, Q, QuerySet
from django.http import (
    Http404,
    HttpRequest,
    HttpResponse,
    HttpResponseBadRequest,
    HttpResponseForbidden,
    StreamingHttpResponse,
)
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils import timezone
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from openai_codex import AppServerError, Codex
from openai_codex.errors import InvalidRequestError
from openai_codex.generated.v2_all import (
    ReasoningEffort,
    SortDirection,
    ThreadSortKey,
)

from hitch.main import (
    caches,
    codex_events,
    codex_pool,
    coding_agents,
    demo,
    health,
    pr_stage,
    rollout,
    session_index,
    session_stage,
    streaming,
    system_agents,
    token_usage,
)
from hitch.main import repos as repos_module
from hitch.main import worktrees as worktrees_module
from hitch.main.autonomous_goal_form import (
    _attach_autonomous_goal_display_state,
    _validated_autonomous_goal_values,
)
from hitch.main.diffs import build_worktree_diff
from hitch.main.entry_render import (
    collapse_flat_entries,
    render_entries,
)
from hitch.main.formatting import render_markdown
from hitch.main.input_images import (
    _INPUT_IMAGE_ACCEPT,
    _INPUT_IMAGE_FIELD,
    _INPUT_IMAGE_MAX_BYTES,
    _INPUT_IMAGE_MAX_COUNT,
    _limit_input_image_uploads,
)
from hitch.main.local_merges import local_branch_names
from hitch.main.models import (
    ApprovalRequest,
    ArchivedSessionTokenUsage,
    AutonomousGoal,
    CodexInstance,
    GlobalSettings,
    Project,
    ProposedSession,
    SessionMetadata,
    SystemAgentRun,
    SystemWorkflow,
    UserSettings,
)
from hitch.main.repos import git_common_dir, same_repo_or_worktree
from hitch.main.rollout_state import (
    _rollout_file_state_from_value,
    _rollout_mtime_ns,
    _rollout_path_for,
    _rollout_path_from_value,
    _RolloutFileState,
    _thread_is_archived,
)
from hitch.main.sdk_values import (
    latest_updated_at,
    string_value,
    updated_at_seconds,
)
from hitch.main.session_approval import _parse_instance_id
from hitch.main.session_cursor import (
    _index_cursor,
    _index_cursor_for_sort_key,
    _index_cursor_sort_key,
    _IndexCursor,
    _is_index_cursor,
)
from hitch.main.session_pr_plan import (
    _PR_PROMPT_ALIASES,
    _PR_SLASH_PROMPT,
    _ROLLOUT_COLLABORATION_MODE_NOT_PROVIDED,
    _auto_merge_to_local_branch_for_session,
    _auto_pr_enabled_for_session,
    _auto_qa_enabled_for_session,
    _count_user_entries,
    _current_pr_url_for_thread,
    _entries_await_plan_approval,
    _fix_pr_url_for_thread,
    _mark_pending_plan_actions,
    _pr_observation_result_for_thread,
    _thread_plan_mode_state,
    _workflow_after_main_lifecycle,
)
from hitch.main.session_resume import (
    _entries_include_transcript,
    _metadata_indicates_archived,
    _metadata_resume_for_inactive_session,
    _metadata_rollout_path_indicates_archived,
    _record_session_unarchived,
    _restore_archived_session_for_rejected_turn,
    _session_detail_metadata,
    _thread_resume_archived_error,
    _unarchive_session_for_turn,
)
from hitch.main.session_settings import (
    _BARE_REPO_PROJECT_VALUE,
    _QA_SLASH_PROMPT,
    _active_project_from_request,
    _allowed_session_cwds,
    _authenticated_user,
    _cached_models_and_settings,
    _current_disk_usage_max_percent,
    _effective_approval_mode,
    _effective_approval_mode_for_session,
    _effective_auto_pr_enabled,
    _effective_sandbox_policy_for_cwd,
    _format_disk_usage_max_percent,
    _new_session_form_context,
    _project_for_proposed_session,
    _resolved_settings,
    _save_user_settings,
    _selected_project_for_settings,
    _session_approval_mode_override,
    _session_project_visibility_for_settings,
    _settings_for_user,
    _stored_settings,
    _supported_effort_values,
)
from hitch.main.session_stage_refresh import (
    _attach_session_stage_context,
    _schedule_pr_stage_refresh,
    _thread_ids_awaiting_input,
)
from hitch.main.settings_cookies import (
    _APPROVAL_MODE_OPTIONS,
    _DEFAULT_APPROVAL_MODE,
    _EXTRA_SYSTEM_PROMPT_MAX_LEN,
    _LAST_SELECTED_REPO_COOKIE,
    _LIVE_HANDLER_APPROVAL_MODES,
    _LIVE_PENDING_APPROVAL_DECISIONS_BY_MODE,
    _MODEL_MAX_LEN,
    _SANDBOX_POLICY_OPTIONS,
    _VALID_APPROVAL_MODES,
    _VALID_SANDBOX_POLICIES,
    _VALID_WEB_SEARCH_MODES,
    _WEB_SEARCH_MODE_OPTIONS,
    ResolvedSettings,
    SessionProjectVisibility,
    SettingsValues,
    _apply_cookie_updates,
    _effective_coding_agent,
    _extra_system_prompt_cookie_fits,
    _option_label,
    _settings_cookie_updates,
    _valid_cookie_setting_updates,
    _valid_web_search_mode_or_default,
    _visible_session_project_ids_cookie_fits,
    _web_search_mode_label,
)
from hitch.main.worktrees import (
    WorktreeCleanupError,
    WorktreeCreationError,
    cleanup_managed_worktree_path,
    cleanup_worktree,
    create_worktree_for_session,
)

logger = logging.getLogger(__name__)

_USAGE_SESSION_INDEX_REFRESH_LOCK = threading.Lock()
_USAGE_SESSION_INDEX_REFRESH_IN_FLIGHT = False


class UsageContext(NamedTuple):
    template_context: dict[str, Any]
    cookie_updates: dict[str, str]


class UsageSessionIndexState(NamedTuple):
    active_complete: bool
    archived_complete: bool
    refresh_active: bool
    refresh_archived: bool
    totals_available: bool


AutonomousGoalRunState = Literal[
    "blocked",
    "failed",
    "manual",
    "maxed",
    "queued",
    "quota",
    "ready",
    "review",
    "running",
    "skipped",
    "waiting",
]


class AutonomousGoalRunBadge(NamedTuple):
    state: AutonomousGoalRunState
    label: str
    title: str
    detail: str


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


class _MessageIntent(NamedTuple):
    prompt: str
    plan_mode: bool
    allow_pending_plan_default: bool
    explicit_plan_mode: bool


class _SessionTemplateThread(NamedTuple):
    id: str
    cwd: str
    updated_at: Any


class _NewSessionTarget(NamedTuple):
    cwd: str
    project: Project | None
    project_cleared: bool
    requires_discovered_repo: bool


_DEBUG_CHAT_PROJECT_NAME = "hitch"
_DEBUG_CHAT_PROMPT_TEMPLATE = (
    "Debug and fix the user's issue from session UID {session_id}.\n\n"
    "Hitch server working directory: {server_cwd}\n"
    "Configured Hitch SQLite database path: {database_path}\n"
    "If you need to inspect it, copy the database first and use the copy; "
    "do not modify the main database file. When copying files directly, include "
    "the WAL sidecars {wal_path} and {shm_path} if they exist so recent rows are "
    "included. A SQLite .backup snapshot is also acceptable.\n\n"
    "User issue: "
)

# Upper bound on what we render inline as a session's title. Codex does not
# generate its own thread summaries, so for unnamed threads `Thread.preview`
# (the full first user message) is what we get; that is often paragraphs
# long and would overflow the list rows without a clip.
_DISPLAY_TITLE_MAX_LEN = 80
_SESSION_PAGE_SIZE = 50
_THREAD_LIST_FETCH_LIMIT = 100
_THREAD_LIST_USE_STATE_DB_ONLY = True


# Server-side cap on user-supplied thread names. Matches the `maxlength` we
# set on the edit form so a client without HTML validation cannot push an
# unbounded blob through.
_NAME_MAX_LEN = 200
_PROJECT_NAME_MAX_LEN = 200
_AUTONOMOUS_GOAL_TITLE_MAX_LEN = 200
_VALID_PROJECT_AUTO_PR_MODES = {value for value, _label in Project.AUTO_PR_CHOICES}

# Upper bound for ``CodexInstance.pk`` validation. The project sets
# ``DEFAULT_AUTO_FIELD = BigAutoField``, which is a signed 64-bit
# integer column. A POST'd value larger than this otherwise reaches
# the ORM and surfaces as a backend-specific OverflowError/DataError
# from ``objects.get`` — a 500 for what should be a clean 400.
_MAX_BIGAUTOFIELD = 2**63 - 1
_PLAN_SLASH_COMMAND = "/plan"
_PLAN_APPROVAL_PROMPT = "Implement the plan."
_PLAN_REVISION_PROMPT = "Revise the plan."
_PLAN_ACTION_APPROVE = "approve"
_PLAN_ACTION_REVISE = "revise"
_VALID_PLAN_ACTIONS = frozenset({"", _PLAN_ACTION_APPROVE, _PLAN_ACTION_REVISE})
_PR_SLASH_COMMAND = "/pr"
_FIX_PR_SLASH_COMMAND = "/fix-pr"
_QA_SLASH_COMMAND = "/qa"
_PLAN_MODE_REASONING_EFFORT = ReasoningEffort.medium.value
_DEFAULT_COLLABORATION_MODE = "default"

# Friendly labels for non-message thread item types. Anything not in this map
# falls back to the raw type tag so we never silently drop an item from the UI.
_SESSION_INTERMEDIATE_DEMO_CONTEXT_SALT = "hitch.session-intermediate.demo-context"
_INTERMEDIATE_DETAIL_CACHE_LOCK = threading.Lock()
_INTERMEDIATE_DETAIL_CACHE_MAX_SIZE = 1024
_INTERMEDIATE_DETAIL_CACHE: OrderedDict[
    tuple[str, str, int, bool, int], dict[str, Any]
] = OrderedDict()


def _settings_context(
    current_settings: SettingsValues,
    models_data: list[Any],
) -> dict[str, Any]:
    projects = list(Project.objects.all())
    current_project = _selected_project_for_settings(current_settings, projects)
    project_visibility = _session_project_visibility_for_settings(
        current_settings, projects
    )
    # Each model advertises which reasoning efforts it accepts. The dialog
    # must only offer the efforts the *selected* model supports — otherwise a
    # user can pick an effort the model rejects and ``update_settings`` bounces
    # the save with a raw 400. An empty supported set means the model didn't
    # advertise any constraint, so every effort is allowed (matching
    # ``_validate_settings_against_models``).
    supported_by_model = {m.id: _supported_effort_values(m) for m in models_data}
    current_supported = supported_by_model.get(current_settings.model, set())
    return {
        "settings_url": reverse("update_settings"),
        "new_project_url": reverse("new_project"),
        "edit_project_url": reverse("edit_project"),
        "model_options": [
            {
                "id": m.id,
                "display_name": m.display_name,
                # Space-separated so the template can drop it into a single
                # data attribute the effort-filter script splits on whitespace.
                "supported_efforts": " ".join(sorted(supported_by_model[m.id])),
            }
            for m in models_data
        ],
        "effort_options": [
            {
                "value": effort.value,
                "supported": not current_supported or effort.value in current_supported,
            }
            for effort in ReasoningEffort
        ],
        "sandbox_options": [
            {"id": value, "display_name": label}
            for value, label in _SANDBOX_POLICY_OPTIONS
        ],
        "approval_options": [
            {"id": value, "display_name": label}
            for value, label in _APPROVAL_MODE_OPTIONS
        ],
        "coding_agent_options": [
            {"id": value, "display_name": label}
            for value, label in coding_agents.CODING_AGENT_OPTIONS
        ],
        "web_search_options": [
            {"id": value, "display_name": label}
            for value, label in _WEB_SEARCH_MODE_OPTIONS
        ],
        "current_model": current_settings.model,
        "current_effort": current_settings.reasoning_effort,
        "current_sandbox": current_settings.sandbox_policy,
        "current_approval": current_settings.approval_mode,
        "current_coding_agent": _effective_coding_agent(current_settings),
        "current_extra_system_prompt": current_settings.extra_system_prompt,
        "extra_system_prompt_max_len": _EXTRA_SYSTEM_PROMPT_MAX_LEN,
        "current_use_worktrees": current_settings.use_worktrees,
        "current_auto_pr": current_settings.auto_pr_enabled,
        "current_auto_qa": current_settings.auto_qa_enabled,
        "current_spec_critic": current_settings.spec_critic_enabled,
        "current_web_search": current_settings.web_search_mode,
        "current_enable_memories": current_settings.enable_memories,
        "current_disk_usage_max_percent": _format_disk_usage_max_percent(
            _current_disk_usage_max_percent()
        ),
        "projects": projects,
        "current_project": current_project,
        "current_project_id": current_project.pk if current_project is not None else "",
        "project_name_max_len": _PROJECT_NAME_MAX_LEN,
        "project_auto_pr_options": [
            {"id": value, "display_name": label}
            for value, label in Project.AUTO_PR_CHOICES
        ],
        "project_extra_system_prompt_max_len": _EXTRA_SYSTEM_PROMPT_MAX_LEN,
        "project_auto_pr_follow_global": Project.AUTO_PR_FOLLOW_GLOBAL,
        "project_auto_pr_on": Project.AUTO_PR_ON,
        "project_auto_pr_off": Project.AUTO_PR_OFF,
        "inbox_count": _proposed_session_inbox_count(project_visibility),
    }


def _proposed_session_inbox_queryset(
    project_visibility: SessionProjectVisibility,
) -> QuerySet[ProposedSession]:
    _recover_stale_new_session_proposal_start_claims()
    inbox = ProposedSession.objects.filter(
        outcome_status=ProposedSession.OUTCOME_UNSET,
    )
    return _filter_proposed_sessions_by_project_visibility(inbox, project_visibility)


def _proposed_session_inbox_count(
    project_visibility: SessionProjectVisibility,
) -> int:
    return _proposed_session_inbox_queryset(project_visibility).count()


def _all_threads(codex: Codex, *, archived: bool = False) -> list[Any]:
    """Return every thread from Codex's paginated thread list."""
    threads: list[Any] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()
    while True:
        kwargs: dict[str, Any] = {
            "limit": _THREAD_LIST_FETCH_LIMIT,
            "sort_key": ThreadSortKey.updated_at,
            "sort_direction": SortDirection.desc,
            "use_state_db_only": _THREAD_LIST_USE_STATE_DB_ONLY,
        }
        if archived:
            kwargs["archived"] = True
        if cursor is not None:
            kwargs["cursor"] = cursor
        response = codex.thread_list(**kwargs)
        threads.extend(response.data)
        next_cursor = getattr(response, "next_cursor", None)
        if not isinstance(next_cursor, str) or not next_cursor:
            break
        if next_cursor in seen_cursors:
            logger.warning("thread list returned duplicate cursor; stopping pagination")
            break
        seen_cursors.add(next_cursor)
        cursor = next_cursor
    return threads


def _session_list_page(
    codex: Codex,
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
    runs_by_thread_id = (
        _system_agent_runs_by_thread_id(system_thread_ids) if system_only else {}
    )
    instances_by_thread_id = (
        _system_agent_instances_by_thread_id(system_thread_ids) if system_only else {}
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
            codex,
            request,
            current_settings=current_settings,
            projects=projects,
            current_project=current_project,
            project_visibility=project_visibility,
            hidden_thread_ids=hidden_thread_ids,
            system_thread_ids=system_thread_ids,
            runs_by_thread_id=runs_by_thread_id,
            instances_by_thread_id=instances_by_thread_id,
            system_only=system_only,
            accepted_visible_thread_ids=accepted_visible_thread_ids,
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
            _schedule_session_index_refresh(
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
                    codex,
                    request,
                    current_settings=current_settings,
                    projects=projects,
                    current_project=current_project,
                    project_visibility=project_visibility,
                    hidden_thread_ids=hidden_thread_ids,
                    system_thread_ids=system_thread_ids,
                    runs_by_thread_id=runs_by_thread_id,
                    instances_by_thread_id=instances_by_thread_id,
                    system_only=system_only,
                    accepted_visible_thread_ids=accepted_visible_thread_ids,
                )
            except AppServerError:
                logger.warning(
                    "failed to fetch live session page after capped refresh; "
                    "rendering cached sessions"
                )
    return _session_list_page_from_index(
        request,
        current_project=current_project,
        project_visibility=project_visibility,
        show_archived=current_settings.show_archived_sessions,
        hidden_thread_ids=hidden_thread_ids,
        system_thread_ids=system_thread_ids,
        runs_by_thread_id=runs_by_thread_id,
        instances_by_thread_id=instances_by_thread_id,
        projects=projects,
        system_only=system_only,
        accepted_visible_thread_ids=accepted_visible_thread_ids,
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
    codex: Codex,
    request: HttpRequest,
    *,
    current_settings: SettingsValues,
    projects: list[Project],
    current_project: Project | None,
    project_visibility: SessionProjectVisibility | None,
    hidden_thread_ids: set[str],
    system_thread_ids: set[str],
    runs_by_thread_id: dict[str, SystemAgentRun],
    instances_by_thread_id: dict[str, CodexInstance],
    system_only: bool,
    accepted_visible_thread_ids: set[str],
) -> SessionListPage:
    project_cache: dict[str, Project | None] = {}
    if not system_only and request.GET.get("materialized_order") == "1":
        return _materialized_session_list_page_from_codex(
            codex,
            request,
            projects=projects,
            current_project=current_project,
            project_visibility=project_visibility,
            hidden_thread_ids=hidden_thread_ids,
            system_thread_ids=system_thread_ids,
            runs_by_thread_id=runs_by_thread_id,
            instances_by_thread_id=instances_by_thread_id,
            project_cache=project_cache,
            include_archived=current_settings.show_archived_sessions,
            system_only=system_only,
            accepted_visible_thread_ids=accepted_visible_thread_ids,
        )
    if current_settings.show_archived_sessions:
        return _merged_session_list_page_from_codex(
            codex,
            request,
            projects=projects,
            current_project=current_project,
            project_visibility=project_visibility,
            hidden_thread_ids=hidden_thread_ids,
            system_thread_ids=system_thread_ids,
            runs_by_thread_id=runs_by_thread_id,
            instances_by_thread_id=instances_by_thread_id,
            project_cache=project_cache,
            system_only=system_only,
            accepted_visible_thread_ids=accepted_visible_thread_ids,
        )
    active = _visible_session_page_from_codex(
        codex,
        request,
        projects=projects,
        current_project=current_project,
        project_visibility=project_visibility,
        hidden_thread_ids=hidden_thread_ids,
        system_thread_ids=system_thread_ids,
        runs_by_thread_id=runs_by_thread_id,
        instances_by_thread_id=instances_by_thread_id,
        project_cache=project_cache,
        archived=False,
        cursor_param="cursor",
        system_only=system_only,
        accepted_visible_thread_ids=accepted_visible_thread_ids,
    )
    if active.needs_materialized_order:
        return _materialized_session_list_page_from_codex(
            codex,
            request,
            projects=projects,
            current_project=current_project,
            project_visibility=project_visibility,
            hidden_thread_ids=hidden_thread_ids,
            system_thread_ids=system_thread_ids,
            runs_by_thread_id=runs_by_thread_id,
            instances_by_thread_id=instances_by_thread_id,
            project_cache=project_cache,
            include_archived=False,
            system_only=system_only,
            accepted_visible_thread_ids=accepted_visible_thread_ids,
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
        _schedule_session_index_refresh(
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
        request,
        current_project=current_project,
        project_visibility=project_visibility,
        show_archived=current_settings.show_archived_sessions,
        hidden_thread_ids=hidden_thread_ids,
        system_thread_ids=system_thread_ids,
        runs_by_thread_id={},
        instances_by_thread_id={},
        projects=projects,
        system_only=system_only,
        accepted_visible_thread_ids=accepted_visible_thread_ids,
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
    runs_by_thread_id = _system_agent_runs_by_thread_id(page_thread_ids)
    instances_by_thread_id = _system_agent_instances_by_thread_id(page_thread_ids)
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


def _session_list_page_from_index(
    request: HttpRequest,
    *,
    current_project: Project | None,
    project_visibility: SessionProjectVisibility | None,
    show_archived: bool,
    hidden_thread_ids: set[str],
    system_thread_ids: set[str],
    runs_by_thread_id: dict[str, SystemAgentRun],
    instances_by_thread_id: dict[str, CodexInstance],
    projects: list[Project],
    system_only: bool,
    accepted_visible_thread_ids: set[str],
) -> SessionListPage:
    rows = session_index.indexed_sessions()
    if system_only:
        _ensure_indexed_system_threads(system_thread_ids, projects=projects)
        rows = session_index.indexed_sessions()
    if system_only:
        indexed_system_thread_ids = set(
            rows.filter(is_hidden_system_session=True)
            .exclude(thread_id__in=accepted_visible_thread_ids)
            .values_list("thread_id", flat=True)
        )
        hidden_thread_ids.update(indexed_system_thread_ids)
        system_thread_ids.update(indexed_system_thread_ids)
    if project_visibility is not None:
        rows = _filter_session_metadata_by_project_visibility(rows, project_visibility)
    elif current_project is not None:
        rows = rows.filter(project=current_project)
    if not show_archived:
        rows = rows.filter(codex_archived=False)
    if system_only:
        rows = rows.filter(thread_id__in=system_thread_ids)
    else:
        rows = _filter_visible_session_metadata_rows(
            rows,
            accepted_visible_thread_ids=accepted_visible_thread_ids,
        )
        if hidden_thread_ids:
            rows = rows.exclude(thread_id__in=hidden_thread_ids)
    if system_only:
        metadata_rows = list(rows)
        qa_updated_at_by_main_thread: dict[str, Any] = {}
        sessions = [
            session
            for metadata in metadata_rows
            if (
                session := _session_row_for_metadata(
                    metadata,
                    qa_updated_at_by_main_thread=qa_updated_at_by_main_thread,
                    runs_by_thread_id=runs_by_thread_id,
                    instances_by_thread_id=instances_by_thread_id,
                    system_only=system_only,
                )
            )
            is not None
        ]
        sessions = _sort_session_rows(sessions)
    else:
        sessions, qa_updated_at_by_main_thread = _sorted_visible_index_rows(
            rows,
            hidden_thread_ids=hidden_thread_ids if hidden_thread_ids else None,
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
    if system_only:
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
                    runs_by_thread_id=runs_by_thread_id,
                    instances_by_thread_id=instances_by_thread_id,
                    system_only=system_only,
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


def _qa_activity_updated_at_by_metadata_thread_id(
    metadata_rows: Iterable[SessionMetadata], hidden_thread_ids: set[str] | None = None
) -> dict[str, Any]:
    return _qa_activity_updated_at_by_metadata_thread_ids(
        [metadata.thread_id for metadata in metadata_rows],
        hidden_thread_ids,
    )


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


def _materialized_session_list_page_from_codex(
    codex: Codex,
    request: HttpRequest,
    *,
    projects: list[Project],
    current_project: Project | None,
    project_visibility: SessionProjectVisibility | None,
    hidden_thread_ids: set[str],
    system_thread_ids: set[str],
    runs_by_thread_id: dict[str, SystemAgentRun],
    instances_by_thread_id: dict[str, CodexInstance],
    project_cache: dict[str, Project | None],
    include_archived: bool,
    system_only: bool,
    accepted_visible_thread_ids: set[str],
) -> SessionListPage:
    threads = _all_threads(codex)
    if include_archived:
        threads.extend(_all_threads(codex, archived=True))
    _add_thread_derived_hidden_ids(
        hidden_thread_ids,
        system_thread_ids,
        threads,
        system_only=system_only,
        accepted_visible_thread_ids=accepted_visible_thread_ids,
    )
    for thread in threads:
        session_index.upsert_thread(thread, projects=projects)
    metadata_by_thread = _metadata_by_thread_id(threads)
    qa_updated_at_by_main_thread = _qa_activity_updated_at_by_main_thread_id(
        threads, hidden_thread_ids
    )
    sessions = [
        session
        for thread in threads
        if (
            session := _session_row_for_thread(
                thread,
                projects=projects,
                current_project=current_project,
                project_visibility=project_visibility,
                metadata_by_thread=metadata_by_thread,
                qa_updated_at_by_main_thread=qa_updated_at_by_main_thread,
                hidden_thread_ids=hidden_thread_ids,
                system_thread_ids=system_thread_ids,
                runs_by_thread_id=runs_by_thread_id,
                instances_by_thread_id=instances_by_thread_id,
                project_cache=project_cache,
                system_only=system_only,
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
    codex: Codex,
    request: HttpRequest,
    *,
    projects: list[Project],
    current_project: Project | None,
    project_visibility: SessionProjectVisibility | None,
    hidden_thread_ids: set[str],
    system_thread_ids: set[str],
    runs_by_thread_id: dict[str, SystemAgentRun],
    instances_by_thread_id: dict[str, CodexInstance],
    project_cache: dict[str, Project | None],
    system_only: bool,
    accepted_visible_thread_ids: set[str],
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
            if _peek_source_session(
                source,
                codex,
                projects=projects,
                current_project=current_project,
                project_visibility=project_visibility,
                hidden_thread_ids=hidden_thread_ids,
                system_thread_ids=system_thread_ids,
                runs_by_thread_id=runs_by_thread_id,
                instances_by_thread_id=instances_by_thread_id,
                project_cache=project_cache,
                system_only=system_only,
                accepted_visible_thread_ids=accepted_visible_thread_ids,
            )
            is not None
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
            codex,
            request,
            projects=projects,
            current_project=current_project,
            project_visibility=project_visibility,
            hidden_thread_ids=hidden_thread_ids,
            system_thread_ids=system_thread_ids,
            runs_by_thread_id=runs_by_thread_id,
            instances_by_thread_id=instances_by_thread_id,
            project_cache=project_cache,
            include_archived=True,
            system_only=system_only,
            accepted_visible_thread_ids=accepted_visible_thread_ids,
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
    codex: Codex,
    *,
    projects: list[Project],
    current_project: Project | None,
    project_visibility: SessionProjectVisibility | None,
    hidden_thread_ids: set[str],
    system_thread_ids: set[str],
    runs_by_thread_id: dict[str, SystemAgentRun],
    instances_by_thread_id: dict[str, CodexInstance],
    project_cache: dict[str, Project | None],
    system_only: bool,
    accepted_visible_thread_ids: set[str],
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
            _add_thread_derived_hidden_ids(
                hidden_thread_ids,
                system_thread_ids,
                source.page.threads,
                system_only=system_only,
                accepted_visible_thread_ids=accepted_visible_thread_ids,
            )
            for thread in source.page.threads:
                session_index.upsert_thread(thread, projects=projects)
            source.next_page_cursor = source.page.next_cursor
            source.metadata_by_thread = _metadata_by_thread_id(source.page.threads)
            detect_materialized_order = (
                not system_only and not source.archived and source.offset == 0
            )
            qa_activity = _qa_activity_page_state(
                source.page.threads,
                hidden_thread_ids,
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
                projects=projects,
                current_project=current_project,
                project_visibility=project_visibility,
                metadata_by_thread=metadata_by_thread,
                qa_updated_at_by_main_thread=qa_updated_at_by_main_thread,
                hidden_thread_ids=hidden_thread_ids,
                system_thread_ids=system_thread_ids,
                runs_by_thread_id=runs_by_thread_id,
                instances_by_thread_id=instances_by_thread_id,
                project_cache=project_cache,
                system_only=system_only,
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
    codex: Codex,
    request: HttpRequest,
    *,
    projects: list[Project],
    current_project: Project | None,
    project_visibility: SessionProjectVisibility | None,
    hidden_thread_ids: set[str],
    system_thread_ids: set[str],
    runs_by_thread_id: dict[str, SystemAgentRun],
    instances_by_thread_id: dict[str, CodexInstance],
    project_cache: dict[str, Project | None],
    archived: bool,
    cursor_param: str,
    system_only: bool,
    accepted_visible_thread_ids: set[str],
) -> VisibleSessionPage:
    cursor = request.GET.get(cursor_param, "")
    offset = _non_negative_int(request.GET.get(_cursor_offset_param(cursor_param), ""))
    materialized_fallback_allowed = not cursor and offset == 0
    sessions: list[dict[str, Any]] = []
    seen_cursors: set[str] = set()
    materialized_qa_activity_seen = False
    while len(sessions) < _SESSION_PAGE_SIZE or materialized_qa_activity_seen:
        if cursor in seen_cursors:
            logger.warning("thread list returned duplicate cursor; stopping pagination")
            if materialized_qa_activity_seen:
                return _materialized_visible_session_page(sessions)
            return VisibleSessionPage(_sort_session_rows(sessions), "", 0)
        seen_cursors.add(cursor)
        page = _thread_list_page(codex, archived=archived, cursor=cursor)
        _add_thread_derived_hidden_ids(
            hidden_thread_ids,
            system_thread_ids,
            page.threads,
            system_only=system_only,
            accepted_visible_thread_ids=accepted_visible_thread_ids,
        )
        for thread in page.threads:
            session_index.upsert_thread(thread, projects=projects)
        can_materialize_qa_activity = (
            materialized_fallback_allowed
            and not system_only
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
            hidden_thread_ids,
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
                projects=projects,
                current_project=current_project,
                project_visibility=project_visibility,
                metadata_by_thread=metadata_by_thread,
                qa_updated_at_by_main_thread=qa_updated_at_by_main_thread,
                hidden_thread_ids=hidden_thread_ids,
                system_thread_ids=system_thread_ids,
                runs_by_thread_id=runs_by_thread_id,
                instances_by_thread_id=instances_by_thread_id,
                project_cache=project_cache,
                system_only=system_only,
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
    hidden_thread_ids: set[str],
    system_thread_ids: set[str],
    threads: Iterable[Any],
    *,
    system_only: bool,
    accepted_visible_thread_ids: set[str],
) -> None:
    thread_hidden_ids = system_agents.hidden_thread_ids_from_threads(
        threads, accepted_visible_thread_ids=accepted_visible_thread_ids
    )
    hidden_thread_ids.update(thread_hidden_ids)
    if system_only:
        system_thread_ids.update(thread_hidden_ids)


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


def _thread_list_page(codex: Codex, *, archived: bool, cursor: str) -> ThreadListPage:
    kwargs: dict[str, Any] = {
        "limit": _THREAD_LIST_FETCH_LIMIT,
        "sort_key": ThreadSortKey.updated_at,
        "sort_direction": SortDirection.desc,
        "use_state_db_only": _THREAD_LIST_USE_STATE_DB_ONLY,
    }
    if archived:
        kwargs["archived"] = True
    if cursor:
        kwargs["cursor"] = cursor
    response = codex.thread_list(**kwargs)
    next_cursor = getattr(response, "next_cursor", "")
    return ThreadListPage(
        threads=sorted(
            list(response.data),
            key=lambda thread: getattr(thread, "updated_at", 0),
            reverse=True,
        ),
        next_cursor=next_cursor if isinstance(next_cursor, str) else "",
    )


def _session_row_for_thread(
    thread: Any,
    *,
    projects: list[Project],
    current_project: Project | None,
    project_visibility: SessionProjectVisibility | None,
    metadata_by_thread: dict[str, SessionMetadata],
    qa_updated_at_by_main_thread: Mapping[str, Any],
    hidden_thread_ids: set[str],
    system_thread_ids: set[str],
    runs_by_thread_id: dict[str, SystemAgentRun],
    instances_by_thread_id: dict[str, CodexInstance],
    project_cache: dict[str, Project | None],
    system_only: bool,
) -> dict[str, Any] | None:
    thread_id = getattr(thread, "id", None)
    if not isinstance(thread_id, str) or not thread_id:
        return None
    if system_only:
        if thread_id not in system_thread_ids:
            return None
    elif thread_id in hidden_thread_ids:
        return None
    session_project = _project_for_thread_cached(
        thread, metadata_by_thread, projects, project_cache
    )
    if project_visibility is not None:
        if not _session_project_is_visible(session_project, project_visibility):
            return None
    elif current_project is not None and session_project != current_project:
        return None
    row = {
        "id": thread_id,
        "cwd": _thread_cwd(thread) or "",
        "updated_at": _session_updated_at(thread, qa_updated_at_by_main_thread),
        "display_title": _display_title(thread),
        "name_value": getattr(thread, "name", None) or "",
        "is_archived": _thread_is_archived(thread),
        "project": session_project,
    }
    if not system_only:
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
    if system_only:
        run = runs_by_thread_id.get(thread_id)
        instance = (
            run.instance if run is not None else instances_by_thread_id.get(thread_id)
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
    cwd = _thread_cwd(thread)
    if not cwd:
        return None
    if cwd not in project_cache:
        project_cache[cwd] = _project_for_cwd(cwd, projects)
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


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        return 0
    return max(parsed, 0)


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
        with codex_pool.borrow_codex(
            Codex, enable_memories=current_settings.enable_memories
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
        logger.warning("failed to open live session list; rendering cached sessions")
        return fallback


def _prevent_stale_cache(response: HttpResponse) -> HttpResponse:
    # These pages render live session state (running workers, stage badges, and
    # names/archive bits that may change from another page). no-store keeps them
    # out of the browser's back/forward and heuristic caches so a Back
    # navigation re-renders against current state instead of a frozen snapshot.
    response["Cache-Control"] = "no-store"
    return response


def index(request: HttpRequest) -> HttpResponse:
    # Sweep workers whose pid is gone: a Popen that crashed before a worker
    # could record its terminal status (or a row stuck in ``starting``)
    # otherwise stays pending forever, since we don't run a periodic task.
    codex_pool.reconcile_dead_if_due()
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
    settings_context = _settings_context(current_settings, models_data)
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
            "name_max_len": _NAME_MAX_LEN,
            "show_new_session_controls": True,
            **settings_context,
            **_session_project_visibility_context(
                session_project_visibility, projects
            ),
        },
    )
    _apply_cookie_updates(response, cookie_updates)
    return _prevent_stale_cache(response)


@require_http_methods(["GET"])
def system_sessions(request: HttpRequest) -> HttpResponse:
    codex_pool.reconcile_dead_if_due()
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
    settings_context = _settings_context(current_settings, models_data)
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
            "name_max_len": _NAME_MAX_LEN,
            "system_session_list": True,
            "show_new_session_controls": False,
            **settings_context,
        },
    )
    _apply_cookie_updates(response, cookie_updates)
    return _prevent_stale_cache(response)


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
        return _render_session_detail(
            request,
            session_id,
            read_only=True,
            require_system_agent_thread=True,
        )
    agent_kind = _system_agent_kind(run, instance)
    hide_demo_agent_entries = agent_kind != demo.DEMO_AGENT_KIND
    return _render_session_detail(
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
    usage_context = _usage_context(request)
    response = render(request, "usage.html", usage_context.template_context)
    _apply_cookie_updates(response, usage_context.cookie_updates)
    return response


def _usage_context(request: HttpRequest) -> UsageContext:
    stored_settings = _stored_settings(request)
    models_data = caches._cached_models_data(enable_memories=stored_settings.enable_memories)
    caches._schedule_models_refresh(enable_memories=stored_settings.enable_memories)
    resolved_settings = _resolved_settings(request, models_data)
    current_settings = resolved_settings.values
    cookie_updates = resolved_settings.cookie_updates
    rate_limits = caches._rate_limits_for_usage_context(
        enable_memories=current_settings.enable_memories
    )
    session_index_state = _usage_session_index_state()
    _schedule_usage_session_index_refresh_if_needed(
        enable_memories=current_settings.enable_memories,
        index_state=session_index_state,
    )
    usage_metadata = (
        _metadata_rows_for_usage() if session_index_state.totals_available else []
    )
    lifetime_usage = (
        token_usage._lifetime_token_usage_for_metadata(usage_metadata)
        if session_index_state.totals_available
        else None
    )
    if session_index_state.totals_available:
        token_usage._schedule_usage_token_refresh(usage_metadata)
    settings_context = _settings_context(current_settings, models_data)
    return UsageContext(
        template_context={
            "login_url": reverse("login"),
            "register_url": reverse("register"),
            "rate_limits": rate_limits,
            "lifetime_usage": lifetime_usage,
            **settings_context,
        },
        cookie_updates=cookie_updates,
    )


@require_http_methods(["GET"])
def inbox(request: HttpRequest) -> HttpResponse:
    codex_pool.reconcile_dead_if_due()
    models_data, resolved_settings = _cached_models_and_settings(request)
    current_settings = resolved_settings.values
    cookie_updates = resolved_settings.cookie_updates
    projects = list(Project.objects.all())
    current_project = _selected_project_for_settings(current_settings, projects)
    inbox_project_visibility = _session_project_visibility_for_settings(
        current_settings, projects
    )
    proposed_sessions = list(
        _proposed_session_inbox_queryset(inbox_project_visibility)
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
    settings_context = _settings_context(current_settings, models_data)
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


@require_http_methods(["GET"])
def autonomous_goals(request: HttpRequest) -> HttpResponse:
    codex_pool.reconcile_dead_if_due()
    models_data, resolved_settings = _cached_models_and_settings(request)
    current_settings = resolved_settings.values
    cookie_updates = resolved_settings.cookie_updates
    projects = list(Project.objects.all())
    current_project = _selected_project_for_settings(current_settings, projects)
    goals = (
        list(
            AutonomousGoal.objects.filter(
                project=current_project,
                deleted_at__isnull=True,
            ).select_related("project")
        )
        if current_project is not None
        else []
    )
    local_branch_choices = (
        local_branch_names(current_project.repo_path)
        if current_project is not None
        else []
    )
    _attach_autonomous_goal_run_state(goals)
    _attach_autonomous_goal_display_state(goals)
    settings_context = _settings_context(current_settings, models_data)
    response = render(
        request,
        "autonomous_goals.html",
        {
            "login_url": reverse("login"),
            "register_url": reverse("register"),
            "current_project": current_project,
            "autonomous_goals": goals,
            "autonomous_goal_create_url": reverse("create_autonomous_goal"),
            "autonomous_goal_run_all_url": reverse("run_autonomous_goals"),
            "ambition_choices": AutonomousGoal.AMBITION_CHOICES,
            "default_ambition": AutonomousGoal.AMBITION_INCREMENTAL,
            "autonomy_choices": AutonomousGoal.AUTONOMY_CHOICES,
            "default_autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
            "default_auto_qa": False,
            "auto_qa_supported_autonomies": tuple(AutonomousGoal.AUTO_QA_AUTONOMIES),
            "auto_qa_required_autonomies": tuple(
                AutonomousGoal.AUTO_QA_REQUIRED_AUTONOMIES
            ),
            "default_auto_proposal": False,
            "stacked_diff_supported_autonomies": tuple(
                AutonomousGoal.STACKED_DIFF_AUTONOMIES
            ),
            "default_stacked_diff_depth": AutonomousGoal.STACKED_DIFF_DEPTH_MIN,
            "stacked_diff_depth_min": AutonomousGoal.STACKED_DIFF_DEPTH_MIN,
            "stacked_diff_depth_max": AutonomousGoal.STACKED_DIFF_DEPTH_MAX,
            "default_proposal_budget": "",
            "confidence_choices": AutonomousGoal.CONFIDENCE_CHOICES,
            "default_confidence": AutonomousGoal.CONFIDENCE_HIGH,
            "web_search_mode_choices": _WEB_SEARCH_MODE_OPTIONS,
            "default_web_search_mode": AutonomousGoal.WEB_SEARCH_DEFAULT,
            "local_branch_choices": local_branch_choices,
            "title_max_len": _AUTONOMOUS_GOAL_TITLE_MAX_LEN,
            **settings_context,
        },
    )
    _apply_cookie_updates(response, cookie_updates)
    return response


@require_http_methods(["POST"])
def create_autonomous_goal(request: HttpRequest) -> HttpResponse:
    project = _active_project_from_request(request)
    if project is None:
        return HttpResponseBadRequest("active project is required")
    values, error = _validated_autonomous_goal_values(
        request,
        local_branches=local_branch_names(project.repo_path),
    )
    if error is not None:
        return HttpResponseBadRequest(error)
    assert values is not None
    AutonomousGoal.objects.create(
        project=project,
        title=values.title,
        goal=values.goal,
        ambition=values.ambition,
        autonomy=values.autonomy,
        auto_qa_enabled=values.auto_qa_enabled,
        auto_proposal_enabled=values.auto_proposal_enabled,
        stacked_diff_depth=values.stacked_diff_depth,
        proposal_budget=values.proposal_budget,
        confidence_threshold=values.confidence_threshold,
        web_search_mode=values.web_search_mode,
        auto_merge_to_local_branch=values.auto_merge_to_local_branch,
        auto_merge_branch=values.auto_merge_branch,
    )
    return redirect("autonomous_goals")


@require_http_methods(["POST"])
def edit_autonomous_goal(request: HttpRequest, autonomous_goal_id: int) -> HttpResponse:
    project = _active_project_from_request(request)
    if project is None:
        return HttpResponseBadRequest("active project is required")
    autonomous_goal = AutonomousGoal.objects.filter(
        pk=autonomous_goal_id,
        project=project,
        deleted_at__isnull=True,
    ).first()
    if autonomous_goal is None:
        raise Http404("autonomous goal not found")
    values, error = _validated_autonomous_goal_values(
        request,
        autonomy_default=autonomous_goal.autonomy,
        auto_qa_default=autonomous_goal.auto_qa_enabled,
        web_search_default=autonomous_goal.web_search_mode,
        auto_proposal_default=autonomous_goal.auto_proposal_enabled,
        stacked_diff_depth_default=autonomous_goal.stacked_diff_depth,
        proposal_budget_default=autonomous_goal.proposal_budget,
        local_branches=local_branch_names(project.repo_path),
    )
    if error is not None:
        return HttpResponseBadRequest(error)
    assert values is not None

    updates: list[str] = []
    for field in (
        "title",
        "goal",
        "ambition",
        "autonomy",
        "auto_qa_enabled",
        "auto_proposal_enabled",
        "stacked_diff_depth",
        "proposal_budget",
        "confidence_threshold",
        "web_search_mode",
        "auto_merge_to_local_branch",
        "auto_merge_branch",
    ):
        value = getattr(values, field)
        if getattr(autonomous_goal, field) != value:
            setattr(autonomous_goal, field, value)
            updates.append(field)
    if updates:
        if autonomous_goal.auto_proposal_last_no_proposal_sha:
            autonomous_goal.auto_proposal_last_no_proposal_sha = ""
            updates.append("auto_proposal_last_no_proposal_sha")
        autonomous_goal.save(update_fields=[*updates, "updated_at"])
    return redirect("autonomous_goals")


@require_http_methods(["POST"])
def delete_autonomous_goal(
    request: HttpRequest, autonomous_goal_id: int
) -> HttpResponse:
    project = _active_project_from_request(request)
    if project is None:
        return HttpResponseBadRequest("active project is required")
    stop_error = system_agents.AUTONOMOUS_GOAL_DELETED_ERROR
    with transaction.atomic():
        autonomous_goal = (
            AutonomousGoal.objects.select_for_update()
            .filter(
                pk=autonomous_goal_id,
                project=project,
                deleted_at__isnull=True,
            )
            .first()
        )
        if autonomous_goal is None:
            raise Http404("autonomous goal not found")
        if not system_agents.stop_running_autonomous_goal_workflow(
            autonomous_goal.pk, stop_error
        ):
            return HttpResponseBadRequest("autonomous goal run could not be stopped")
        deleted_at = timezone.now()
        cleanup_proposals = _dismiss_unresolved_autonomous_goal_proposals(
            autonomous_goal,
            reason=stop_error,
            now=deleted_at,
        )
        autonomous_goal.deleted_at = deleted_at
        autonomous_goal.auto_proposal_enabled = False
        autonomous_goal.save(
            update_fields=["deleted_at", "auto_proposal_enabled", "updated_at"]
        )
    for proposal in cleanup_proposals:
        _cleanup_proposed_session_candidate_worktree(proposal)
    return redirect("autonomous_goals")


def _dismiss_unresolved_autonomous_goal_proposals(
    autonomous_goal: AutonomousGoal, *, reason: str, now: datetime
) -> list[ProposedSession]:
    proposals = list(
        ProposedSession.objects.select_for_update()
        .select_related("candidate_session")
        .filter(
            autonomous_goal=autonomous_goal,
        )
        .filter(_autonomous_goal_cleanup_proposal_filter())
    )
    if proposals:
        for proposal in proposals:
            metadata = (
                dict(proposal.outcome_metadata)
                if isinstance(proposal.outcome_metadata, dict)
                else {}
            )
            metadata["stacked_diff_hidden_until_complete"] = False
            proposal.outcome_status = ProposedSession.OUTCOME_DISMISSED
            proposal.outcome_notes = reason
            proposal.outcome_metadata = metadata
            proposal.updated_at = now
        ProposedSession.objects.bulk_update(
            proposals,
            ["outcome_status", "outcome_notes", "outcome_metadata", "updated_at"],
        )
    return proposals


def _autonomous_goal_cleanup_proposal_filter() -> Q:
    return Q(outcome_status=ProposedSession.OUTCOME_UNSET) | Q(
        outcome_status=ProposedSession.OUTCOME_DISMISSED,
        outcome_metadata__stacked_diff_hidden_until_complete=True,
    )


@require_http_methods(["POST"])
def run_autonomous_goal(request: HttpRequest, autonomous_goal_id: int) -> HttpResponse:
    project = _active_project_from_request(request)
    if project is None:
        return HttpResponseBadRequest("active project is required")
    autonomous_goal = AutonomousGoal.objects.filter(
        pk=autonomous_goal_id,
        project=project,
        deleted_at__isnull=True,
    ).first()
    if autonomous_goal is None:
        raise Http404("autonomous goal not found")
    system_agents.start_autonomous_goal_workflow(
        autonomous_goal=autonomous_goal,
        use_worktrees=True,
    )
    return redirect("autonomous_goals")


@require_http_methods(["POST"])
def run_autonomous_goals(request: HttpRequest) -> HttpResponse:
    project = _active_project_from_request(request)
    if project is None:
        return HttpResponseBadRequest("active project is required")
    for autonomous_goal in AutonomousGoal.objects.filter(
        project=project,
        deleted_at__isnull=True,
    ):
        system_agents.start_autonomous_goal_workflow(
            autonomous_goal=autonomous_goal,
            use_worktrees=True,
        )
    return redirect("autonomous_goals")


@require_http_methods(["GET"])
def autonomous_goal_run_log(request: HttpRequest, workflow_id: int) -> HttpResponse:
    workflow = _autonomous_goal_workflow_for_log(request, workflow_id)
    run = workflow.agent_runs.exclude(thread_id="").order_by("-created_at").first()
    if run is None:
        raise Http404("autonomous goal run log not found")
    return _render_session_detail(
        request,
        run.thread_id,
        read_only=True,
        display_title="Autonomous goal run log",
    )


@require_http_methods(["POST"])
def update_proposed_session_outcome(
    request: HttpRequest, proposed_session_id: int
) -> HttpResponse:
    if proposed_session_id < 1 or proposed_session_id > _MAX_BIGAUTOFIELD:
        return HttpResponseBadRequest("proposed session is required")
    current_settings = _stored_settings(request)
    project_visibility = _session_project_visibility_for_settings(
        current_settings, list(Project.objects.all())
    )
    proposed_session_query = _filter_proposed_sessions_by_project_visibility(
        ProposedSession.objects.select_related(
            "project",
            "autonomous_goal__project",
            "candidate_session",
        ).filter(pk=proposed_session_id),
        project_visibility,
    )
    proposed_session = proposed_session_query.first()
    if proposed_session is None:
        return HttpResponseBadRequest("proposed session is required")
    outcome_status = request.POST.get("outcome_status", "")
    # OUTCOME_UNSET is the inbox's pending state, not a decision the endpoint can
    # apply; accepting it as a target would let a request re-open a resolved item.
    valid_statuses = {
        choice[0] for choice in ProposedSession.OUTCOME_CHOICES
    } - {ProposedSession.OUTCOME_UNSET}
    if outcome_status not in valid_statuses:
        return HttpResponseBadRequest("outcome status is invalid")
    outcome_notes = request.POST.get(
        "reason", request.POST.get("outcome_notes", "")
    ).strip()
    if (
        proposed_session.inbox_kind == ProposedSession.INBOX_KIND_PROPOSAL
        and outcome_status == ProposedSession.OUTCOME_REJECTED
        and not outcome_notes
    ):
        return HttpResponseBadRequest("reason is required")
    if (
        proposed_session.inbox_kind == ProposedSession.INBOX_KIND_NOTICE
        and outcome_status != ProposedSession.OUTCOME_DISMISSED
    ):
        return HttpResponseBadRequest("outcome status is invalid")
    update_values: dict[str, Any] = {
        "outcome_status": outcome_status,
        "outcome_notes": outcome_notes,
        # update() bypasses save(), so the auto_now updated_at must be set here.
        "updated_at": timezone.now(),
    }
    outcome_metadata = _proposal_outcome_metadata(
        proposed_session,
        {"resolved_by": "user"},
    )
    if outcome_status == ProposedSession.OUTCOME_ACCEPTED:
        update_values["accepted_session"] = proposed_session.candidate_session
        outcome_metadata = _proposal_outcome_metadata(
            proposed_session,
            {
                **outcome_metadata,
                "accepted_by": "user",
                "accepted_session_id": (
                    proposed_session.candidate_session_id
                    if proposed_session.candidate_session_id is not None
                    else None
                ),
                "accepted_thread_id": (
                    proposed_session.candidate_session.thread_id
                    if proposed_session.candidate_session is not None
                    else ""
                ),
            },
        )
    update_values["outcome_metadata"] = outcome_metadata
    # Inbox decisions are one-way: only an undecided item may be resolved
    # (UNSET -> accepted/rejected/dismissed), matching the OUTCOME_UNSET filter
    # the new-session entry points already apply. Enforce it with a single
    # conditional UPDATE gated on the row still being OUTCOME_UNSET rather than a
    # read-then-write: two near-simultaneous requests (e.g. a stale-tab reject
    # racing an accept) could both read OUTCOME_UNSET before either commits, and
    # the loser would clobber the accepted outcome and re-hide the live session.
    # The atomic WHERE clause serializes the decision -- exactly one request
    # matches a row; the loser updates nothing and bails before any side effects.
    # (A row lock would also work, but only on backends that honor
    # select_for_update; a conditional UPDATE is correct on every backend.)
    applied = ProposedSession.objects.filter(
        pk=proposed_session.pk,
        outcome_status=ProposedSession.OUTCOME_UNSET,
    ).update(**update_values)
    if not applied:
        return HttpResponseBadRequest("proposed session has already been resolved")
    # Mirror the committed values onto the instance for the cleanup side effect.
    for field, value in update_values.items():
        setattr(proposed_session, field, value)
    stack_continuation_stopped = _stop_autonomous_goal_stack_after_proposal_resolution(
        proposed_session
    )
    if (
        outcome_status == ProposedSession.OUTCOME_ACCEPTED
        and proposed_session.candidate_session is not None
    ):
        _rename_codex_thread_from_proposal(
            proposed_session=proposed_session,
            session_metadata=proposed_session.candidate_session,
            settings=_stored_settings(request),
        )
    if outcome_status in {
        ProposedSession.OUTCOME_DISMISSED,
        ProposedSession.OUTCOME_REJECTED,
    } and stack_continuation_stopped:
        _cleanup_proposed_session_candidate_worktree(proposed_session)
    return redirect("inbox")


def _cleanup_proposed_session_candidate_worktree(
    proposed_session: ProposedSession,
) -> None:
    if proposed_session.accepted_session_id is not None:
        return
    candidate = proposed_session.candidate_session
    if candidate is None or not candidate.cwd:
        return
    try:
        cleanup_managed_worktree_path(candidate.cwd)
    except WorktreeCleanupError:
        logger.exception(
            "failed to clean up candidate worktree for proposed session %s",
            proposed_session.pk,
        )


def _stop_autonomous_goal_stack_after_proposal_resolution(
    proposed_session: ProposedSession,
) -> bool:
    if proposed_session.autonomous_goal_id is None:
        return True
    return system_agents.stop_running_autonomous_goal_stack_after_proposal_resolution(
        proposed_session.autonomous_goal_id,
        proposed_session.pk,
        proposed_session.outcome_status,
    )


def _attach_autonomous_goal_run_state(goals: list[AutonomousGoal]) -> None:
    goal_ids = [goal.pk for goal in goals]
    if not goal_ids:
        return
    pending_proposal_state = system_agents._autonomous_goal_pending_proposal_state(
        goals
    )
    pending_proposal_goal_ids = pending_proposal_state.blocking_goal_ids
    unresolved_failure_notice_goal_ids = _autonomous_goal_failure_notice_ids(goal_ids)
    project_running_auto_proposal_ids = _autonomous_goal_running_auto_proposal_project_ids(
        goals
    )
    project_in_flight_automation_ids = _autonomous_goal_in_flight_automation_project_ids(
        goals
    )
    no_change_goal_ids = _autonomous_goal_no_change_ids(
        goals,
        continuable_stack_goal_ids=pending_proposal_state.continuable_stack_goal_ids,
    )
    auto_proposals_paused_by_quota = (
        any(goal.auto_proposal_enabled for goal in goals)
        and system_agents._auto_proposals_paused_by_usage_quota_throttled()
    )
    workflows = (
        SystemWorkflow.objects.filter(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id__in=[
                system_agents._autonomous_goal_main_thread_id(goal_id)
                for goal_id in goal_ids
            ],
        )
        .order_by("main_thread_id", "-created_at")
    )
    workflows_by_thread: dict[str, SystemWorkflow] = {}
    for workflow in workflows:
        workflows_by_thread.setdefault(workflow.main_thread_id, workflow)
    latest_workflows = list(workflows_by_thread.values())
    log_urls_by_workflow_id = _autonomous_goal_log_urls(latest_workflows)
    running_tokens_by_workflow_id = _autonomous_goal_running_token_counts(
        latest_workflows
    )
    for goal in goals:
        latest_workflow = workflows_by_thread.get(
            system_agents._autonomous_goal_main_thread_id(goal.pk)
        )
        goal.run_running = (  # type: ignore[attr-defined]
            latest_workflow is not None
            and latest_workflow.is_active
        )
        goal.run_tokens_used_display = _autonomous_goal_run_tokens_used_display(  # type: ignore[attr-defined]
            latest_workflow,
            running_tokens_by_workflow_id,
        )
        goal.run_log_url = (  # type: ignore[attr-defined]
            log_urls_by_workflow_id.get(latest_workflow.pk) or ""
            if latest_workflow is not None
            else ""
        )
        run_badge = _autonomous_goal_run_badge(
            goal,
            latest_workflow,
            pending_proposal_goal_ids=pending_proposal_goal_ids,
            continuable_stack_goal_ids=(
                pending_proposal_state.continuable_stack_goal_ids
            ),
            unresolved_failure_notice_goal_ids=unresolved_failure_notice_goal_ids,
            project_running_auto_proposal_ids=project_running_auto_proposal_ids,
            project_in_flight_automation_ids=project_in_flight_automation_ids,
            no_change_goal_ids=no_change_goal_ids,
            auto_proposals_paused_by_quota=auto_proposals_paused_by_quota,
        )
        goal.run_status_state = run_badge.state  # type: ignore[attr-defined]
        goal.run_status_label = run_badge.label  # type: ignore[attr-defined]
        goal.run_status_title = run_badge.title  # type: ignore[attr-defined]
        goal.run_status_detail = run_badge.detail  # type: ignore[attr-defined]


def _autonomous_goal_failure_notice_ids(goal_ids: list[int]) -> set[int]:
    return {
        goal_id
        for goal_id in ProposedSession.objects.filter(
            autonomous_goal_id__in=goal_ids,
            inbox_kind=ProposedSession.INBOX_KIND_NOTICE,
            outcome_status=ProposedSession.OUTCOME_UNSET,
            outcome_metadata__automation_status="failed",
        ).values_list("autonomous_goal_id", flat=True)
        if isinstance(goal_id, int)
    }


def _autonomous_goal_running_auto_proposal_project_ids(
    goals: list[AutonomousGoal],
) -> set[int]:
    project_ids = {goal.project_id for goal in goals}
    repo_path_by_project_id = dict(
        Project.objects.filter(pk__in=project_ids).values_list("pk", "repo_path")
    )
    running_auto_proposal_cwds = set(
        SystemWorkflow.objects.filter(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            status=SystemWorkflow.STATUS_RUNNING,
            state__auto_proposal=True,
        ).values_list("cwd", flat=True)
    )
    return {
        project_id
        for project_id, repo_path in repo_path_by_project_id.items()
        if repo_path in running_auto_proposal_cwds
    }


def _autonomous_goal_in_flight_automation_project_ids(
    goals: list[AutonomousGoal],
) -> set[int]:
    project_ids = {goal.project_id for goal in goals}
    if not project_ids:
        return set()
    in_flight_project_ids: set[int] = set()
    claim_key = ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY
    claim_lookup = f"outcome_metadata__{claim_key}__isnull"
    now = timezone.now()
    claimed_metadatas = (
        ProposedSession.objects.filter(
            project_id__in=project_ids,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            accepted_session__isnull=True,
            **{claim_lookup: False},
        )
        .filter(system_agents._autonomous_goal_in_flight_proposal_criteria())
        .values_list("project_id", "outcome_metadata")
    )
    for project_id, metadata in claimed_metadatas:
        if not isinstance(project_id, int):
            continue
        if ProposedSession.accepted_session_start_claim_is_active(metadata, now=now):
            in_flight_project_ids.add(project_id)

    accepted_thread_project_ids: dict[str, int] = {}
    accepted_threads = (
        ProposedSession.objects.filter(
            project_id__in=project_ids,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            accepted_session__isnull=False,
        )
        .filter(system_agents._autonomous_goal_in_flight_proposal_criteria())
        .exclude(accepted_session__thread_id="")
        .values_list("project_id", "accepted_session__thread_id")
    )
    for project_id, thread_id in accepted_threads:
        if isinstance(project_id, int) and isinstance(thread_id, str):
            accepted_thread_project_ids[thread_id] = project_id
    if not accepted_thread_project_ids:
        return in_flight_project_ids

    active_thread_ids = set(
        CodexInstance.objects.filter(
            thread_id__in=list(accepted_thread_project_ids),
            status__in=CodexInstance.ACTIVE_STATUSES,
        ).values_list("thread_id", flat=True)
    )
    active_thread_ids.update(
        SystemWorkflow.objects.filter(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id__in=list(accepted_thread_project_ids),
            status=SystemWorkflow.STATUS_RUNNING,
        ).values_list("main_thread_id", flat=True)
    )
    for thread_id in active_thread_ids:
        active_project_id = accepted_thread_project_ids.get(thread_id)
        if active_project_id is not None:
            in_flight_project_ids.add(active_project_id)
    return in_flight_project_ids


def _autonomous_goal_no_change_ids(
    goals: list[AutonomousGoal], *, continuable_stack_goal_ids: set[int]
) -> set[int]:
    no_change_goal_ids: set[int] = set()
    for goal in goals:
        last_no_proposal_sha = goal.auto_proposal_last_no_proposal_sha.strip()
        if (
            not goal.auto_proposal_enabled
            or goal.pk in continuable_stack_goal_ids
            or not last_no_proposal_sha
        ):
            continue
        current_sha = system_agents._autonomous_goal_auto_proposal_base_sha(goal)
        if current_sha == last_no_proposal_sha:
            no_change_goal_ids.add(goal.pk)
    return no_change_goal_ids


def _autonomous_goal_run_badge(
    goal: AutonomousGoal,
    workflow: SystemWorkflow | None,
    *,
    pending_proposal_goal_ids: set[int],
    continuable_stack_goal_ids: set[int],
    unresolved_failure_notice_goal_ids: set[int],
    project_running_auto_proposal_ids: set[int],
    project_in_flight_automation_ids: set[int],
    no_change_goal_ids: set[int],
    auto_proposals_paused_by_quota: bool,
) -> AutonomousGoalRunBadge:
    if workflow is not None:
        if workflow.is_active:
            return AutonomousGoalRunBadge(
                state="running",
                label="Running",
                title="Autonomous goal is running",
                detail="This autonomous goal run is still working.",
            )
        if workflow.status == SystemWorkflow.STATUS_BLOCKED:
            return AutonomousGoalRunBadge(
                state="blocked",
                label="Blocked",
                title="Autonomous goal is blocked",
                detail=(
                    _workflow_state_string(workflow, "error")
                    or "This autonomous goal run is blocked. Open the run log for details."
                ),
            )
        if workflow.status == SystemWorkflow.STATUS_FAILED:
            return AutonomousGoalRunBadge(
                state="failed",
                label="Failed",
                title="Autonomous goal run failed",
                detail=(
                    _workflow_state_string(workflow, "error")
                    or "The last autonomous goal run failed. Open the run log for details."
                ),
            )
        if workflow.status == SystemWorkflow.STATUS_MAX_ITERATIONS_REACHED:
            return AutonomousGoalRunBadge(
                state="maxed",
                label="Maxed",
                title="Autonomous goal reached its iteration limit",
                detail="The last autonomous goal run stopped after reaching its iteration limit.",
            )

    if goal.pk in pending_proposal_goal_ids:
        return AutonomousGoalRunBadge(
            state="review",
            label="Review",
            title="Autonomous goal is waiting for review",
            detail="Not running because a proposal from this goal is waiting in the inbox.",
        )
    if goal.pk in unresolved_failure_notice_goal_ids:
        return AutonomousGoalRunBadge(
            state="failed",
            label="Failed",
            title="Autonomous goal is paused after a failure",
            detail=(
                "Not running because a failure notice from this goal is still in the inbox. "
                "Dismiss or resolve that notice to let auto-proposal try again."
            ),
        )
    if (
        goal.auto_proposal_enabled
        and goal.project_id in project_running_auto_proposal_ids
    ):
        return AutonomousGoalRunBadge(
            state="queued",
            label="Queued",
            title="Autonomous goal is queued",
            detail="Not running because another auto-proposal run is active for this project.",
        )
    if (
        goal.auto_proposal_enabled
        and goal.project_id in project_in_flight_automation_ids
    ):
        return AutonomousGoalRunBadge(
            state="queued",
            label="Queued",
            title="Autonomous goal is queued",
            detail=(
                "Not running because accepted autonomous-goal automation "
                "is still active for this project."
            ),
        )
    if goal.pk in no_change_goal_ids:
        return AutonomousGoalRunBadge(
            state="waiting",
            label="No change",
            title="Autonomous goal is waiting for branch changes",
            detail=(
                "Not running because the last auto-proposal found no useful proposal "
                "for the tracked branch. It will try again after that branch changes."
            ),
        )
    if goal.auto_proposal_enabled and auto_proposals_paused_by_quota:
        return AutonomousGoalRunBadge(
            state="quota",
            label="Quota",
            title="Autonomous goal is paused for quota",
            detail=(
                "Not running because remaining Codex quota is below the "
                "auto-proposal safety threshold. It will try again as quota recovers."
            ),
        )
    if goal.auto_proposal_enabled and goal.pk in continuable_stack_goal_ids:
        return AutonomousGoalRunBadge(
            state="ready",
            label="Ready",
            title="Autonomous goal is ready",
            detail="Auto-proposal is enabled. This goal will start when the scheduler runs and quota allows.",
        )
    if workflow is not None and workflow.status == SystemWorkflow.STATUS_COMPLETED:
        completed_badge = _completed_autonomous_goal_run_badge(workflow)
        if completed_badge is not None:
            return completed_badge
    if not goal.auto_proposal_enabled:
        detail = "Auto-proposal is off. Use Run to start this goal manually."
        latest_detail = _autonomous_goal_latest_run_detail(workflow)
        if latest_detail:
            detail = f"{detail}\n\n{latest_detail}"
        return AutonomousGoalRunBadge(
            state="manual",
            label="Manual",
            title="Autonomous goal is manual",
            detail=detail,
        )
    return AutonomousGoalRunBadge(
        state="ready",
        label="Ready",
        title="Autonomous goal is ready",
        detail="Auto-proposal is enabled. This goal will start when the scheduler runs and quota allows.",
    )


def _completed_autonomous_goal_run_badge(
    workflow: SystemWorkflow,
) -> AutonomousGoalRunBadge | None:
    if workflow.step == system_agents.STEP_AUTONOMOUS_GOAL_SKIPPED:
        return AutonomousGoalRunBadge(
            state="skipped",
            label="Skipped",
            title="Autonomous goal last run was skipped",
            detail=_autonomous_goal_latest_run_detail(workflow),
        )
    return None


def _autonomous_goal_running_token_counts(
    workflows: Iterable[SystemWorkflow],
) -> dict[int, int]:
    workflows_by_id = {
        workflow.pk: workflow
        for workflow in workflows
        if workflow.is_active
    }
    if not workflows_by_id:
        return {}
    runs = (
        SystemAgentRun.objects.select_related("instance")
        .filter(
            workflow_id__in=list(workflows_by_id),
            status__in=(
                SystemAgentRun.STATUS_STARTING,
                SystemAgentRun.STATUS_RUNNING,
            ),
        )
        .exclude(thread_id="")
        .order_by("workflow_id", "-created_at", "-pk")
    )
    tokens_by_workflow_id: dict[int, int] = {}
    for run in runs:
        if run.workflow_id in tokens_by_workflow_id:
            continue
        workflow = workflows_by_id.get(run.workflow_id)
        if workflow is not None:
            tokens_by_workflow_id[run.workflow_id] = (
                _autonomous_goal_running_token_count(workflow, run.instance)
            )
    return tokens_by_workflow_id


def _autonomous_goal_running_token_count(
    workflow: SystemWorkflow, instance: CodexInstance
) -> int:
    persisted_tokens = _workflow_state_int(
        workflow, system_agents._AUTONOMOUS_GOAL_PROPOSAL_BUDGET_USED_STATE_KEY
    )
    current_tokens = codex_events.latest_goal_tokens_for_instance(instance)
    if current_tokens is None:
        return persisted_tokens
    previous_tokens = _autonomous_goal_recorded_thread_tokens(workflow, instance)
    return persisted_tokens + max(current_tokens - previous_tokens, 0)


def _autonomous_goal_recorded_thread_tokens(
    workflow: SystemWorkflow, instance: CodexInstance
) -> int:
    token_totals = workflow.state.get(
        system_agents._AUTONOMOUS_GOAL_PROPOSAL_BUDGET_TOKEN_TOTALS_STATE_KEY
    )
    if not isinstance(token_totals, dict):
        return 0
    value = token_totals.get(instance.thread_id)
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
        else 0
    )


def _autonomous_goal_run_tokens_used_display(
    workflow: SystemWorkflow | None, running_tokens_by_workflow_id: Mapping[int, int]
) -> str:
    if workflow is None or not workflow.is_active:
        return ""
    tokens = running_tokens_by_workflow_id.get(
        workflow.pk,
        _workflow_state_int(
            workflow, system_agents._AUTONOMOUS_GOAL_PROPOSAL_BUDGET_USED_STATE_KEY
        ),
    )
    return f"{token_usage._format_token_count(tokens)} tokens"


def _autonomous_goal_latest_run_detail(workflow: SystemWorkflow | None) -> str:
    if workflow is None:
        return ""
    if workflow.status == SystemWorkflow.STATUS_COMPLETED:
        if workflow.step == system_agents.STEP_AUTONOMOUS_GOAL_SKIPPED:
            return _autonomous_goal_skipped_detail(workflow)
        if workflow.step == system_agents.STEP_AUTONOMOUS_GOAL_PROPOSED:
            return _autonomous_goal_proposed_detail(workflow)
        return "The last autonomous goal run completed."
    if workflow.status == SystemWorkflow.STATUS_FAILED:
        return (
            _workflow_state_string(workflow, "error")
            or "The last autonomous goal run failed."
        )
    if workflow.status == SystemWorkflow.STATUS_MAX_ITERATIONS_REACHED:
        return "The last autonomous goal run stopped after reaching its iteration limit."
    return ""


def _autonomous_goal_skipped_detail(workflow: SystemWorkflow) -> str:
    candidate = workflow.state.get("candidate")
    if isinstance(candidate, dict):
        message = candidate.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    judgment = workflow.state.get("judgment")
    if isinstance(judgment, dict):
        rationale = judgment.get("rationale")
        if isinstance(rationale, str) and rationale.strip():
            return rationale.strip()
    return "The last autonomous goal run completed without a proposal."


def _autonomous_goal_proposed_detail(workflow: SystemWorkflow) -> str:
    stopped_reason = _workflow_state_string(workflow, "stacked_diff_stopped_reason")
    if stopped_reason == "candidate_no_proposal":
        return (
            "The last autonomous goal run published the current stacked proposal "
            "because the next candidate produced no proposal."
        )
    if stopped_reason == "judge_confidence_below_threshold":
        return (
            "The last autonomous goal run published the current stacked proposal "
            "because the next candidate fell below the confidence threshold."
        )
    if stopped_reason == "stacked_diff_continuation_failed":
        error = _workflow_state_string(workflow, "stacked_diff_continuation_error")
        if error:
            return (
                "The last autonomous goal run published the current stacked proposal "
                f"after the next candidate failed: {error}"
            )
        return (
            "The last autonomous goal run published the current stacked proposal "
            "after the next candidate failed."
        )
    return "The last autonomous goal run created a proposal and stopped."


def _workflow_state_string(workflow: SystemWorkflow, key: str) -> str:
    value = workflow.state.get(key)
    return value.strip() if isinstance(value, str) else ""


def _attach_proposed_session_display_state(
    proposed_sessions: list[ProposedSession],
) -> None:
    for proposed_session in proposed_sessions:
        files = proposed_session.relevant_files
        proposed_session.display_files = (  # type: ignore[attr-defined]
            [item for item in files if isinstance(item, str) and item.strip()]
            if isinstance(files, list)
            else []
        )
        if proposed_session.candidate_session is not None:
            proposed_session.candidate_log_url = reverse(  # type: ignore[attr-defined]
                "system_session",
                kwargs={"session_id": proposed_session.candidate_session.thread_id},
            )
        else:
            proposed_session.candidate_log_url = ""  # type: ignore[attr-defined]
        if proposed_session.judge_session is not None:
            proposed_session.judge_log_url = reverse(  # type: ignore[attr-defined]
                "system_session",
                kwargs={"session_id": proposed_session.judge_session.thread_id},
            )
        else:
            proposed_session.judge_log_url = ""  # type: ignore[attr-defined]
        proposed_session.session_prompt = _proposed_session_prompt(  # type: ignore[attr-defined]
            proposed_session
        )
        project = _project_for_proposed_session(proposed_session)
        proposed_session.accept_project_id = (  # type: ignore[attr-defined]
            project.pk if project is not None else ""
        )
        auto_pr_enabled, auto_qa_enabled = (
            _auto_review_settings_for_proposed_session(proposed_session)
        )
        proposed_session.accept_auto_pr = auto_pr_enabled  # type: ignore[attr-defined]
        proposed_session.accept_auto_qa = auto_qa_enabled  # type: ignore[attr-defined]
        proposed_session.stack_label = _proposed_session_stack_label(  # type: ignore[attr-defined]
            proposed_session
        )


def _proposed_session_stack_label(proposed_session: ProposedSession) -> str:
    metadata = (
        proposed_session.outcome_metadata
        if isinstance(proposed_session.outcome_metadata, dict)
        else {}
    )
    metadata_depth = metadata.get("stacked_diff_depth")
    metadata_iteration = metadata.get("stacked_diff_iteration")
    if (
        isinstance(metadata_depth, int)
        and not isinstance(metadata_depth, bool)
        and isinstance(metadata_iteration, int)
        and not isinstance(metadata_iteration, bool)
        and metadata_depth > AutonomousGoal.STACKED_DIFF_DEPTH_MIN
    ):
        depth = min(metadata_depth, AutonomousGoal.STACKED_DIFF_DEPTH_MAX)
        if metadata_iteration < 1 or metadata_iteration > depth:
            return ""
        iteration = metadata_iteration
        return f"Stack {iteration} of {depth}"
    return ""


def _proposed_session_prompt(proposed_session: ProposedSession) -> str:
    if proposed_session.prompt.strip():
        return proposed_session.prompt.strip()
    parts = [
        "Go ahead and implement this proposed session.",
        "",
        f"Autonomous goal: {proposed_session.autonomous_goal.title}"
        if proposed_session.autonomous_goal is not None
        else "Source: Coding agent proposal",
    ]
    if (
        proposed_session.autonomous_goal is not None
        and proposed_session.autonomous_goal.goal
    ):
        parts.extend(
            ["", f"Autonomous goal objective:\n{proposed_session.autonomous_goal.goal}"]
        )
    parts.extend(["", f"Proposed session: {proposed_session.title}"])
    if proposed_session.summary:
        parts.extend(["", f"Summary:\n{proposed_session.summary}"])
    files = proposed_session.display_files  # type: ignore[attr-defined]
    if files:
        parts.extend(["", "Relevant files:", *[f"- {file}" for file in files]])
    return "\n".join(parts)


def _autonomous_goal_log_urls(workflows: Iterable[SystemWorkflow]) -> dict[int, str]:
    workflow_ids = [workflow.pk for workflow in workflows]
    if not workflow_ids:
        return {}
    runs = (
        SystemAgentRun.objects.filter(workflow_id__in=workflow_ids)
        .exclude(thread_id="")
        .order_by("workflow_id", "-created_at")
    )
    urls: dict[int, str] = {}
    for run in runs:
        urls.setdefault(
            run.workflow_id,
            reverse("autonomous_goal_run_log", kwargs={"workflow_id": run.workflow_id}),
        )
    return urls


def _autonomous_goal_workflow_for_log(
    request: HttpRequest, workflow_id: int
) -> SystemWorkflow:
    if workflow_id < 1 or workflow_id > _MAX_BIGAUTOFIELD:
        raise Http404("autonomous goal run log not found")
    project = _active_project_from_request(request)
    if project is None:
        raise Http404("autonomous goal run log not found")
    workflow = (
        SystemWorkflow.objects.filter(
            pk=workflow_id,
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        .first()
    )
    if workflow is None:
        raise Http404("autonomous goal run log not found")
    autonomous_goal_id = _workflow_state_int(workflow, "autonomous_goal_id")
    autonomous_goal = AutonomousGoal.objects.filter(
        pk=autonomous_goal_id,
        project=project,
    ).first()
    if autonomous_goal is None:
        raise Http404("autonomous goal run log not found")
    return workflow


def _workflow_state_int(workflow: SystemWorkflow, key: str) -> int:
    value = workflow.state.get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _positive_int(value: str) -> int | None:
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def session(request: HttpRequest, session_id: str) -> HttpResponse:
    return _render_session_detail(request, session_id)


def _render_session_detail(
    request: HttpRequest,
    session_id: str,
    *,
    read_only: bool = False,
    display_title: str | None = None,
    system_prompt: str = "",
    hide_demo_agent_entries: bool = True,
    demo_entries_run_id: int | None = None,
    require_system_agent_thread: bool = False,
) -> HttpResponse:
    # Reconcile this thread before reading status: a worker that died without
    # writing a terminal status would otherwise leave the page in "streaming"
    # mode forever, since the EventSource wouldn't reach an end event. The
    # global sweep stays debounced, but this exact session must be fresh.
    codex_pool.reconcile_dead_for_thread(session_id)
    codex_pool.reconcile_dead_if_due()
    initial_settings = _stored_settings(request)
    active_instance = _active_instance_for(session_id)
    active_system_workflow = system_agents.active_workflow_for_thread(session_id)
    metadata = _session_detail_metadata(session_id)
    # Capture the rollout mtime *before* any entries are read (the resume helper
    # below reads them off disk), so a concurrent append surfaces as a cache
    # miss on the next read rather than being masked behind a post-read stat.
    # See the matching rule in ``token_usage_snapshot`` and
    # ``_attach_session_stage_context``.
    stage_cache_mtime_ns = (
        _rollout_mtime_ns(_rollout_path_from_value(metadata.codex_path))
        if metadata is not None
        else 0
    )
    metadata_resume = _metadata_resume_for_inactive_session(
        session_id,
        metadata,
        active_instance=active_instance,
        active_system_workflow=active_system_workflow,
        require_system_agent_thread=require_system_agent_thread,
    )
    resumed: Any
    thread: Any

    if metadata_resume is not None:
        resumed = metadata_resume
        thread = metadata_resume.thread
        raw_entries = list(metadata_resume.entries)
        rollout_data = metadata_resume.rollout_data
        models_data = caches._cached_models_for_session_detail(
            enable_memories=initial_settings.enable_memories
        )
        resolved_settings = _resolved_settings(request, models_data)
        settings = resolved_settings.values
        cookie_updates = resolved_settings.cookie_updates
        plan_model = _plan_mode_model_from_models(resumed, settings, models_data)
    else:

        def _resume_for_detail(codex: Codex) -> tuple[Any, Any, list[Any], Any, Any]:
            # ``thread/read`` only works for threads already loaded into the
            # app-server's in-memory map. A cold open spawns a fresh app-server
            # subprocess, so newly-created threads (or any thread persisted by a
            # different worker) need ``thread/resume`` to read them off disk.
            # The resume response already carries the full thread including turns,
            # so a follow-up ``thread/read`` would just be a redundant round-trip.
            try:
                resumed = codex._client.thread_resume(session_id)
            except InvalidRequestError as exc:
                if require_system_agent_thread and _thread_resume_missing_or_invalid(exc):
                    raise Http404("system session not found") from None
                raise
            thread = resumed.thread
            if require_system_agent_thread and not system_agents.hitch_system_agent_thread(
                thread
            ):
                raise Http404("system session not found")
            models_data = _models_for_plan_mode_fallback(codex)
            resolved_settings = _resolved_settings(request, models_data)
            plan_model = _plan_mode_model_from_models(
                resumed, resolved_settings.values, models_data
            )
            return resumed, thread, models_data, resolved_settings, plan_model

        # Prefer a warm pooled app-server: cold-opening here re-runs the
        # CODEX_HOME init write that contends on the state-DB writer lock, which
        # is what surfaced "failed to initialize sqlite state runtime ...
        # database is locked" on this page while a worker held the lock. A warm
        # server is already initialized, so the resume does no init write; only
        # an empty pool falls back to a retrying cold open. ``thread_resume``
        # lazily migrates a foreign thread's rows and is idempotent, so a retry
        # (or the warm->cold fallback re-running it) is safe.
        resumed, thread, models_data, resolved_settings, plan_model = (
            codex_pool.run_borrowed_op_with_retry(
                Codex,
                _resume_for_detail,
                enable_memories=initial_settings.enable_memories,
            )
        )
        settings = resolved_settings.values
        cookie_updates = resolved_settings.cookie_updates
        # Capture the rollout mtime before reading entries; see the
        # metadata-resume branch above for why the order matters.
        stage_cache_mtime_ns = _rollout_mtime_ns(_rollout_path_for(thread))
        raw_entries = list(_entries_for(thread))
        rollout_data = None
    is_archived = _thread_is_archived(thread)
    entries = _apply_system_authors(raw_entries, session_id)
    entries = _apply_qa_approval_messages(entries, session_id)
    if hide_demo_agent_entries:
        entries = _filter_demo_agent_entries(entries, session_id)
    name_value = getattr(thread, "name", None) or ""
    projects = list(Project.objects.all())
    metadata_by_thread = _metadata_by_thread_id([thread])
    if metadata is not None:
        metadata_by_thread[session_id] = metadata
    session_project = _project_for_thread(thread, metadata_by_thread, projects)
    latest_pr_url = rollout_data.latest_pr_url if rollout_data is not None else None
    pr_observation = (
        rollout_data.pr_observation
        if rollout_data is not None
        else _pr_observation_result_for_thread(thread)
    )
    latest_pr_workflow = pr_stage._latest_pr_workflow_for_thread(session_id)
    stage_workflow = active_system_workflow or latest_pr_workflow
    stage_pr_workflow = (
        active_system_workflow
        if active_system_workflow is not None
        and active_system_workflow.kind == SystemWorkflow.KIND_PR_QA
        else latest_pr_workflow
    )
    main_updated_at = getattr(thread, "updated_at", None)
    stage_workflow = _workflow_after_main_lifecycle(
        stage_workflow, pr_observation, main_updated_at=main_updated_at
    )
    stage_pr_workflow = _workflow_after_main_lifecycle(
        stage_pr_workflow, pr_observation, main_updated_at=main_updated_at
    )
    pr_url = _current_pr_url_for_thread(
        thread,
        pr_observation=pr_observation,
        stage_pr_workflow=stage_pr_workflow,
        latest_pr_url=latest_pr_url,
        latest_pr_url_loaded=rollout_data is not None,
    )
    stage_context: dict[str, Any] | None = None
    if not read_only:
        awaiting_user_input = session_id in _thread_ids_awaiting_input([session_id])
        # Serve the last-known PR stage now and refresh off-request when due.
        # A synchronous ``gh pr view`` here shelled out on every detail render
        # (up to a 5s timeout) and dominated page latency; instead the badge is
        # flagged as refreshing and the actual gh call runs in the background,
        # persisting the result for a later render to read back.
        workflow_pr_snapshot = system_agents.pr_handoff_for_workflow(stage_pr_workflow)
        # Only flag refreshing when the PR stage is the one actually displayed.
        # An active worker or a waiting-for-input session shows its own stage, so
        # marking that live badge refreshing would let the reload script tear
        # down the running EventSource transcript.
        pr_stage_displayed = active_instance is None and not awaiting_user_input
        stage_refreshing = pr_stage_displayed and (
            system_agents.pr_handoff_stage_refresh_due(stage_pr_workflow)
            or system_agents.pr_monitor_backoff_stage_refresh_due(stage_pr_workflow)
        )
        log_pr_snapshot = pr_observation.snapshot
        if (
            pr_stage_displayed
            and stage_pr_workflow is None
            and log_pr_snapshot is not None
        ):
            detail_cwd = (
                metadata.cwd
                if metadata is not None and metadata.cwd
                else _thread_cwd(thread) or ""
            )
            if system_agents.pr_snapshot_stage_refresh_due(
                cwd=detail_cwd,
                snapshot=log_pr_snapshot,
                attempted_at=(
                    metadata.derived_stage_pr_refresh_attempted_at
                    if metadata is not None
                    else None
                ),
            ):
                stage_refreshing = True
        if stage_refreshing:
            _schedule_pr_stage_refresh(session_id)
        stage = session_stage.derive_stage(
            entries=entries,
            active_instance=active_instance,
            workflow=stage_workflow,
            awaiting_user_input=awaiting_user_input,
            pr_snapshot=log_pr_snapshot,
            workflow_pr_snapshot=workflow_pr_snapshot,
        )
        # A background PR refresh persists a terminal stage to the mtime-keyed
        # cache, but the detail render otherwise re-derives from the (still-open)
        # rollout. Prefer the cached terminal stage when it matches the current
        # rollout so the async result surfaces on reload instead of reverting to
        # the open-PR badge while the gh refresh stays throttled.
        if (
            stage.key == session_stage.PR.key
            and metadata is not None
            and metadata.derived_stage_source_mtime_ns == stage_cache_mtime_ns
        ):
            cached_terminal = session_stage.stage_for_key(metadata.derived_stage)
            if cached_terminal is not None and cached_terminal.key in (
                session_stage.DONE_MERGED.key,
                session_stage.DONE_CLOSED.key,
            ):
                stage = cached_terminal
                stage_refreshing = False
        # Only persist a rollout-derived stage; see _attach_session_stage_context
        # for why active-instance/workflow-forced stages must not enter the
        # mtime-keyed cache. The post-lifecycle ``stage_workflow``/
        # ``stage_pr_workflow`` being ``None`` means no live owner influenced the
        # result (a stale workflow stripped by ``_workflow_after_main_lifecycle``
        # leaves the stage purely rollout-derived and therefore cacheable).
        if (
            active_instance is None
            and stage_workflow is None
            and stage_pr_workflow is None
            and not awaiting_user_input
            and not stage_refreshing
        ):
            # Best-effort like the session-list path: this runs while rendering
            # the session detail page, so a contended write lock must skip the
            # cache refresh rather than 500 the page (the next render retries).
            pr_stage._update_cached_stage_best_effort(session_id, stage, stage_cache_mtime_ns)
        stage_context = dict(stage.as_context())
        if stage_refreshing:
            stage_context["refreshing"] = True
    show_active_worker_transcript = _show_active_worker_transcript(active_instance)
    active_demo_worker = (
        active_instance is not None and active_instance.agent_kind == demo.DEMO_AGENT_KIND
    )
    # While a worker is running, drop the entries that belong to its
    # in-progress turn — the SSE stream replays them from byte 0 of the
    # events file, so leaving the rollout-rendered copy in place would
    # double up every entry in the live DOM. The page reload on stream end
    # restores the canonical view.
    entries = _trim_in_progress_turn(entries, active_instance)
    plan_mode_state = _thread_plan_mode_state(
        session_id,
        thread,
        entries,
        active_instance=active_instance,
        latest_collaboration_mode=(
            rollout_data.latest_collaboration_mode
            if rollout_data is not None
            else _ROLLOUT_COLLABORATION_MODE_NOT_PROVIDED
        ),
    )
    default_plan_mode = plan_mode_state.active
    _mark_pending_plan_actions(entries, enabled=plan_mode_state.awaiting_approval)
    if rollout_data is not None:
        session_token_usage = (
            token_usage._format_session_token_usage(rollout_data.latest_token_usage)
            if rollout_data.latest_token_usage is not None
            else None
        )
    else:
        session_token_usage = token_usage._token_usage_for(thread)
    _attach_lazy_intermediate_context(
        entries,
        session_id=session_id,
        enabled=rollout_data is not None,
        hide_demo_agent_entries=hide_demo_agent_entries,
        demo_entries_run_id=demo_entries_run_id,
        rollout_state=_rollout_file_state_from_value(getattr(thread, "path", None)),
    )
    goal_objective = codex_events.latest_goal_for_thread(session_id)
    # Scope the plan to the running worker, or to the latest worker on reload
    # when none is running, so a turn that finished without emitting its own
    # plan does not inherit an earlier turn's.
    task_plan = _task_plan_context(
        codex_events.latest_task_plan_for_instance(active_instance)
        if active_instance is not None
        else codex_events.latest_task_plan_for_thread(session_id)
    )
    thread_cwd = _thread_cwd(thread)
    diff_view = build_worktree_diff(thread_cwd)
    active_session_demo = demo.active_demo_for(session_id)
    session_demo = demo.latest_demo_for(session_id)
    demo_system_session_url = _demo_system_session_url(session_id)
    demo_url = (
        demo.demo_url_for_request(request, session_id)
        if active_session_demo is not None
        else ""
    )
    settings_context = _settings_context(settings, models_data)
    active_worker_status_text = _active_worker_status_text(active_instance)
    workflow_status_text = _workflow_status_text(active_system_workflow)
    pr_workflow_progress = streaming.pr_workflow_progress(active_system_workflow)
    workflow_accepts_steering = _workflow_accepts_qa_pause_steering(
        active_system_workflow
    ) or _workflow_accepts_active_turn_steering(
        active_system_workflow, active_instance
    )
    workflow_composer_locked = active_system_workflow is not None and not (
        workflow_accepts_steering
    )
    live_status_text = active_worker_status_text or (
        workflow_status_text
        if active_system_workflow is not None and active_instance is None
        else ""
    )
    debug_chat_url = _debug_chat_new_session_url(
        session_id, session_project, projects, cwd=thread_cwd
    )
    approval_mode = _effective_approval_mode_for_session(
        settings, session_id, metadata
    )
    response = render(
        request,
        "session.html",
        {
            "thread": _session_template_thread(thread),
            "entries": entries,
            "display_title": display_title or _display_title(thread),
            "read_only": read_only,
            "system_prompt": system_prompt,
            "name_value": name_value,
            "name_max_len": _NAME_MAX_LEN,
            "set_name_url": reverse("set_session_name", kwargs={"session_id": session_id}),
            "set_archived_url": reverse(
                "set_session_archived", kwargs={"session_id": session_id}
            ),
            "start_demo_url": reverse(
                "start_session_demo", kwargs={"session_id": session_id}
            ),
            "set_project_url": reverse(
                "set_session_project", kwargs={"session_id": session_id}
            ),
            "is_archived": is_archived,
            "send_message_url": reverse("send_message", kwargs={"session_id": session_id}),
            "stop_url": reverse("stop_session", kwargs={"session_id": session_id}),
            # Pin the stream to the specific worker shown on this page
            # so a newer turn starting between render and EventSource
            # connect can't divert the live view away from the worker
            # the Stop button is wired to.
            "stream_url": _stream_url_for(
                session_id, active_instance, active_system_workflow
            ),
            # The JS swaps the trailing ``0`` for the real ApprovalRequest
            # pk on each POST. Templating the URL server-side (rather than
            # building it in JS from a base path) keeps Django's URL
            # resolver authoritative even when the route changes.
            "approval_url_template": reverse(
                "resolve_approval", kwargs={"approval_id": 0}
            ),
            "input_url_template": reverse(
                "resolve_input_request", kwargs={"input_id": 0}
            ),
            "active_worker": active_instance is not None,
            "active_demo_worker": active_demo_worker,
            "show_active_worker_transcript": show_active_worker_transcript,
            "active_system_workflow": active_system_workflow,
            "workflow_composer_label": _workflow_composer_label(active_system_workflow),
            "workflow_accepts_steering": workflow_accepts_steering,
            "workflow_composer_locked": workflow_composer_locked,
            "demo_start_disabled": (
                active_system_workflow is not None or active_instance is not None
            ),
            "workflow_status_text": workflow_status_text,
            "pr_workflow_progress": pr_workflow_progress,
            "active_worker_status_text": active_worker_status_text,
            "live_status_text": live_status_text,
            # Carried into the Stop button so the click targets the
            # specific worker the page is streaming, not "whichever
            # worker is latest at click time" — overlapping turns can
            # stack two active workers on the same thread.
            "active_instance": active_instance,
            # The in-progress turn is trimmed from ``entries`` above, so the
            # user wouldn't see their own message at all without a pending
            # bubble while the stream catches up.
            "pending_user_prompt": _pending_user_prompt(active_instance),
            "pending_user_author": _pending_user_author(active_instance),
            "pending_user_timestamp": _pending_user_timestamp(active_instance),
            "token_usage": session_token_usage,
            "next_message_config": _next_message_config(
                settings,
                resumed,
                plan_model,
                cwd=thread_cwd or "",
                approval_mode=approval_mode,
            ),
            "input_image_accept": _INPUT_IMAGE_ACCEPT,
            "pr_slash_prompt": _PR_SLASH_PROMPT,
            "fix_pr_slash_command": _FIX_PR_SLASH_COMMAND,
            "qa_slash_prompt": _QA_SLASH_PROMPT,
            "default_plan_mode": default_plan_mode,
            "plan_approval_prompt": _PLAN_APPROVAL_PROMPT,
            "plan_revision_prompt": _PLAN_REVISION_PROMPT,
            "pr_url": pr_url,
            "session_stage": stage_context,
            "goal_objective": goal_objective,
            "task_plan": task_plan,
            "diff_view": diff_view,
            "session_demo": session_demo,
            "active_session_demo": active_session_demo,
            "demo_url": demo_url,
            "demo_system_session_url": demo_system_session_url,
            "projects": projects,
            "session_project": session_project,
            "session_project_id": session_project.pk if session_project is not None else "",
            "debug_chat_url": debug_chat_url,
            **_session_approval_mode_context(settings, session_id, metadata),
            **settings_context,
        },
    )
    _apply_cookie_updates(response, cookie_updates)
    return _prevent_stale_cache(response)


def _thread_resume_missing_or_invalid(exc: InvalidRequestError) -> bool:
    message = exc.message.lower()
    return (
        "invalid thread id" in message
        or "invalid session id" in message
        or bool(
            re.search(
                r"\bthread(?:\s+id)?(?:\s+\S+)?\s+not found\b",
                message,
            )
        )
    )


def _valid_codex_session_id(session_id: str) -> bool:
    value = session_id.removeprefix("urn:uuid:")
    try:
        uuid.UUID(value)
    except ValueError:
        return False
    return True


def _debug_chat_new_session_url(
    session_id: str,
    project: Project | None,
    projects: Iterable[Project],
    *,
    cwd: str | None,
) -> str:
    server_cwd = Path(django_settings.BASE_DIR)
    database_path = _debug_chat_database_path()
    query_params = {
        "prompt": _DEBUG_CHAT_PROMPT_TEMPLATE.format(
            session_id=session_id,
            server_cwd=str(server_cwd),
            database_path=str(database_path),
            wal_path=f"{database_path}-wal",
            shm_path=f"{database_path}-shm",
        )
    }
    repo_set = {str(path) for path in repos_module.discover_repos()}
    project = _debug_chat_project(project, projects, repo_set=repo_set)
    if project is not None:
        query_params["project"] = str(project.pk)
    elif cwd and cwd in repo_set:
        query_params["cwd"] = cwd
    return f"{reverse('new_session')}?{urlencode(query_params)}"


def _debug_chat_database_path() -> Path:
    database_path = Path(str(django_settings.DATABASES["default"]["NAME"]))
    if database_path.is_absolute():
        return database_path
    return Path(django_settings.BASE_DIR) / database_path


def _debug_chat_project(
    fallback_project: Project | None,
    projects: Iterable[Project],
    *,
    repo_set: set[str],
) -> Project | None:
    hitch_project = next(
        (
            project
            for project in projects
            if project.name.casefold() == _DEBUG_CHAT_PROJECT_NAME
            and project.repo_path in repo_set
        ),
        None,
    )
    if hitch_project is not None:
        return hitch_project
    if fallback_project is not None and fallback_project.repo_path in repo_set:
        return fallback_project
    return None


@require_http_methods(["GET", "POST"])
def register(request: HttpRequest) -> HttpResponse:
    if _authenticated_user(request) is not None:
        return redirect("index")
    form: Any
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            _import_cookie_settings_to_user(request, user)
            auth_login(request, user)
            response = redirect("index")
            _apply_cookie_updates(
                response, _settings_cookie_updates(_stored_settings(request))
            )
            return response
    else:
        form = UserCreationForm()
    return render(
        request,
        "register.html",
        {"form": form, "login_url": reverse("login")},
    )


@require_http_methods(["GET", "POST"])
def login(request: HttpRequest) -> HttpResponse:
    if _authenticated_user(request) is not None:
        return redirect("index")
    next_url = _safe_next_url(request)
    form: Any
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            _import_cookie_settings_to_user(request, user)
            auth_login(request, user)
            response = redirect(next_url or "index")
            _apply_cookie_updates(
                response, _settings_cookie_updates(_stored_settings(request))
            )
            return response
    else:
        form = AuthenticationForm(request)
    return render(
        request,
        "login.html",
        {
            "form": form,
            "next": next_url,
            "register_url": reverse("register"),
        },
    )


@require_http_methods(["GET"])
def profile(request: HttpRequest) -> HttpResponse:
    user = _authenticated_user(request)
    usage_context = _profile_usage_context(request)
    profile_name = user.get_username() if user is not None else "anonymous"
    response = render(
        request,
        "profile.html",
        {
            "profile_name": profile_name,
            "profile_status": "Signed in" if user is not None else "Signed out",
            "logout_url": reverse("logout") if user is not None else "",
            "nuke_codex_url": reverse("nuke_codex") if user is not None else "",
            "health_url": reverse("health_dashboard") if user is not None else "",
            "nuked_count": _parse_nuked_count(request.GET.get("nuked")),
            **usage_context.template_context,
        },
    )
    _apply_cookie_updates(response, usage_context.cookie_updates)
    return response


def _parse_nuked_count(raw: str | None) -> int | None:
    """Parse the ``?nuked=N`` confirmation count the nuke action redirects with.

    Returns ``None`` (render no confirmation) for a missing or malformed value
    so a hand-edited URL cannot inject arbitrary text into the page.
    """
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def _profile_usage_context(request: HttpRequest) -> UsageContext:
    try:
        return _usage_context(request)
    except Exception:
        logger.exception("failed to load profile usage context; showing empty usage state")
    settings_context = _settings_context(_stored_settings(request), [])
    return UsageContext(
        template_context={
            "login_url": reverse("login"),
            "register_url": reverse("register"),
            "rate_limits": None,
            "lifetime_usage": None,
            **settings_context,
        },
        cookie_updates={},
    )


@require_http_methods(["GET"])
def health_dashboard(request: HttpRequest) -> HttpResponse:
    """Hitch health dashboard: leak and backlog signals on one page.

    Linked from the bottom of the profile page. Requires authentication since
    it exposes operational internals. The copy block is built to be long-pressed
    and pasted into a chat with the assistant when diagnosing issues.
    """
    if _authenticated_user(request) is None:
        return redirect(f"{reverse('login')}?next={reverse('health_dashboard')}")
    report = health.collect_health_report()
    return render(
        request,
        "health.html",
        {
            "report": report,
            "copy_text": report.copy_text(),
            "profile_url": reverse("profile"),
        },
    )


@require_http_methods(["POST"])
def logout(request: HttpRequest) -> HttpResponse:
    values = _stored_settings(request) if _authenticated_user(request) is not None else None
    auth_logout(request)
    response = redirect("index")
    if values is not None:
        _apply_cookie_updates(response, _settings_cookie_updates(values))
    return response


@require_http_methods(["POST"])
def nuke_codex(request: HttpRequest) -> HttpResponse:
    """SIGKILL every Codex app-server Hitch started, then return to the profile.

    Manual cleanup for leaked app-servers contending on the shared CODEX_HOME
    state-DB lock. The killed count is round-tripped through a query param so
    the profile page can confirm the outcome.
    """
    if _authenticated_user(request) is None:
        return HttpResponseForbidden("authentication required")
    killed = codex_pool.nuke_codex_app_servers()
    return redirect(f"{reverse('profile')}?nuked={killed}")


def _metadata_cwd_is_disallowed(metadata: SessionMetadata | None) -> bool:
    return (
        metadata is not None
        and bool(metadata.cwd)
        and not _is_allowed_session_cwd(metadata.cwd)
    )


def _attach_lazy_intermediate_context(
    entries: list[dict[str, Any]],
    *,
    session_id: str,
    enabled: bool,
    hide_demo_agent_entries: bool,
    demo_entries_run_id: int | None,
    rollout_state: _RolloutFileState | None,
) -> None:
    if not enabled or rollout_state is None:
        return
    query_params: dict[str, str] = {}
    if not hide_demo_agent_entries and demo_entries_run_id is not None:
        query_params["demo_context"] = _session_intermediate_demo_context(
            session_id, demo_entries_run_id
        )
    query = f"?{urlencode(query_params)}" if query_params else ""
    for entry_index, entry in enumerate(entries):
        if entry.get("kind") != "intermediate":
            continue
        _cache_intermediate_detail(
            session_id=session_id,
            rollout_state=rollout_state,
            hide_demo_agent_entries=hide_demo_agent_entries,
            entry_index=entry_index,
            entry=entry,
        )
        entry["lazy_url"] = (
            reverse(
                "session_intermediate",
                kwargs={"session_id": session_id, "entry_index": entry_index},
            )
            + query
        )
        entry["item_count"] = len(entry.get("items", []))
        entry["items"] = []


def _intermediate_detail_cache_key(
    *,
    session_id: str,
    rollout_state: _RolloutFileState,
    hide_demo_agent_entries: bool,
    entry_index: int,
) -> tuple[str, str, int, bool, int]:
    return (
        session_id,
        str(rollout_state.path),
        rollout_state.mtime_ns,
        hide_demo_agent_entries,
        entry_index,
    )


def _cache_intermediate_detail(
    *,
    session_id: str,
    rollout_state: _RolloutFileState,
    hide_demo_agent_entries: bool,
    entry_index: int,
    entry: dict[str, Any],
) -> None:
    key = _intermediate_detail_cache_key(
        session_id=session_id,
        rollout_state=rollout_state,
        hide_demo_agent_entries=hide_demo_agent_entries,
        entry_index=entry_index,
    )
    cached_entry = {
        "kind": "intermediate",
        "thinking_count": entry.get("thinking_count", 0),
        "tool_call_count": entry.get("tool_call_count", 0),
        "items": entry.get("items", []),
    }
    with _INTERMEDIATE_DETAIL_CACHE_LOCK:
        _INTERMEDIATE_DETAIL_CACHE[key] = cached_entry
        _INTERMEDIATE_DETAIL_CACHE.move_to_end(key)
        while len(_INTERMEDIATE_DETAIL_CACHE) > _INTERMEDIATE_DETAIL_CACHE_MAX_SIZE:
            _INTERMEDIATE_DETAIL_CACHE.popitem(last=False)


def _cached_intermediate_detail(
    *,
    session_id: str,
    rollout_state: _RolloutFileState,
    hide_demo_agent_entries: bool,
    entry_index: int,
) -> dict[str, Any] | None:
    key = _intermediate_detail_cache_key(
        session_id=session_id,
        rollout_state=rollout_state,
        hide_demo_agent_entries=hide_demo_agent_entries,
        entry_index=entry_index,
    )
    with _INTERMEDIATE_DETAIL_CACHE_LOCK:
        entry = _INTERMEDIATE_DETAIL_CACHE.get(key)
        if entry is not None:
            _INTERMEDIATE_DETAIL_CACHE.move_to_end(key)
        return entry


@require_http_methods(["GET"])
def session_intermediate(
    request: HttpRequest, session_id: str, entry_index: int
) -> HttpResponse:
    if entry_index < 0:
        raise Http404("intermediate entry not found")
    hide_demo_agent_entries = not _session_intermediate_allows_demo_entries(
        session_id, request.GET.get("demo_context", "")
    )
    entry = _rollout_intermediate_entry_for_detail(
        session_id,
        entry_index=entry_index,
        hide_demo_agent_entries=hide_demo_agent_entries,
    )
    response = render(request, "_session_intermediate_body.html", {"entry": entry})
    # The body depends on the current rollout contents; with no validators a
    # browser may heuristically cache this lazily-fetched fragment and show a
    # stale block after the rollout entry changes.
    return _prevent_stale_cache(response)


def _session_intermediate_demo_context(session_id: str, run_id: int) -> str:
    return signing.dumps(
        {"session_id": session_id, "run_id": run_id},
        salt=_SESSION_INTERMEDIATE_DEMO_CONTEXT_SALT,
    )


def _session_intermediate_allows_demo_entries(
    session_id: str, raw_context: str | None
) -> bool:
    if not raw_context:
        return False
    try:
        context = signing.loads(
            raw_context,
            salt=_SESSION_INTERMEDIATE_DEMO_CONTEXT_SALT,
        )
    except signing.BadSignature:
        return False
    if not isinstance(context, dict) or context.get("session_id") != session_id:
        return False
    run_id = context.get("run_id")
    if not isinstance(run_id, int) or run_id <= 0:
        return False
    run = _system_agent_run_for_thread(session_id, run_id=run_id)
    return run is not None and run.agent_kind == demo.DEMO_AGENT_KIND


def _rollout_intermediate_entry_for_detail(
    session_id: str, *, entry_index: int, hide_demo_agent_entries: bool
) -> dict[str, Any]:
    metadata = _session_detail_metadata(session_id)
    if metadata is None:
        raise Http404("session not found")
    rollout_state = _rollout_file_state_from_value(metadata.codex_path)
    if rollout_state is None:
        raise Http404("session not found")
    cached = _cached_intermediate_detail(
        session_id=session_id,
        rollout_state=rollout_state,
        hide_demo_agent_entries=hide_demo_agent_entries,
        entry_index=entry_index,
    )
    if cached is not None:
        return cached
    try:
        rollout_data = rollout.session_detail_data(rollout_state.path)
    except Exception as exc:
        logger.exception(
            "failed to parse rollout %s for intermediate detail", rollout_state.path
        )
        raise Http404("intermediate entry not found") from exc
    if rollout_data is None:
        raise Http404("session not found")
    entries = list(collapse_flat_entries(list(rollout_data.flat_entries)))
    if not _entries_include_transcript(entries):
        raise Http404("session not found")
    entries = _apply_system_authors(entries, session_id)
    entries = _apply_qa_approval_messages(entries, session_id)
    if hide_demo_agent_entries:
        entries = _filter_demo_agent_entries(entries, session_id)
    if entry_index >= len(entries):
        raise Http404("intermediate entry not found")
    entry = entries[entry_index]
    if entry.get("kind") != "intermediate":
        raise Http404("intermediate entry not found")
    _cache_intermediate_detail(
        session_id=session_id,
        rollout_state=rollout_state,
        hide_demo_agent_entries=hide_demo_agent_entries,
        entry_index=entry_index,
        entry=entry,
    )
    return entry


def _system_agent_runs_by_thread_id(
    thread_ids: Iterable[str],
) -> dict[str, SystemAgentRun]:
    ids = [thread_id for thread_id in thread_ids if thread_id]
    if not ids:
        return {}
    runs = (
        SystemAgentRun.objects.filter(thread_id__in=ids)
        .exclude(thread_id="")
        .select_related("instance")
        .only(
            "id",
            "workflow",
            "agent_kind",
            "thread_id",
            "instance",
            "status",
            "created_at",
            "instance__id",
            "instance__thread_id",
            "instance__display_author",
            "instance__agent_kind",
            "instance__status",
            "instance__started_at",
        )
        .order_by("thread_id", "-created_at", "-pk")
    )
    by_thread_id: dict[str, SystemAgentRun] = {}
    for run in runs:
        by_thread_id.setdefault(run.thread_id, run)
    return by_thread_id


def _system_agent_instances_by_thread_id(
    thread_ids: Iterable[str],
) -> dict[str, CodexInstance]:
    ids = [thread_id for thread_id in thread_ids if thread_id]
    if not ids:
        return {}
    instances = (
        CodexInstance.objects.filter(
            thread_id__in=ids,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
        )
        .exclude(thread_id="")
        .exclude(agent_kind=demo.DEMO_AGENT_KIND)
        .only(
            "id",
            "thread_id",
            "display_author",
            "agent_kind",
            "status",
            "started_at",
        )
        .order_by("thread_id", "-started_at", "-pk")
    )
    by_thread_id: dict[str, CodexInstance] = {}
    for instance in instances:
        by_thread_id.setdefault(instance.thread_id, instance)
    return by_thread_id


def _qa_activity_updated_at_by_main_thread_id(
    threads: Iterable[Any], hidden_thread_ids: set[str]
) -> dict[str, Any]:
    current_thread_ids = {
        thread_id
        for thread in threads
        if isinstance((thread_id := getattr(thread, "id", None)), str)
    }
    current_main_thread_ids = current_thread_ids - hidden_thread_ids
    if not current_main_thread_ids:
        return {}

    hidden_updated_at_by_thread_id: dict[str, Any] = {}
    for thread in threads:
        thread_id = getattr(thread, "id", None)
        if isinstance(thread_id, str) and thread_id in hidden_thread_ids:
            hidden_updated_at_by_thread_id[thread_id] = getattr(
                thread, "updated_at", None
            )

    runs = (
        SystemAgentRun.objects.filter(
            workflow__kind=SystemWorkflow.KIND_PR_QA,
            workflow__main_thread_id__in=current_main_thread_ids,
        )
        .exclude(thread_id="")
        .select_related("workflow")
    )
    updated_at_by_main_thread: dict[str, Any] = {}
    for run in runs:
        main_thread_id = run.workflow.main_thread_id
        if not main_thread_id:
            continue
        run_updated_at = hidden_updated_at_by_thread_id.get(run.thread_id)
        if updated_at_seconds(run_updated_at) is None:
            run_updated_at = latest_updated_at(run.updated_at, run.workflow.updated_at)
        updated_at_by_main_thread[main_thread_id] = latest_updated_at(
            updated_at_by_main_thread.get(main_thread_id),
            run_updated_at,
        )
    return updated_at_by_main_thread


def _session_updated_at(
    thread: Any, qa_updated_at_by_main_thread: Mapping[str, Any]
) -> Any:
    return latest_updated_at(
        getattr(thread, "updated_at", None),
        qa_updated_at_by_main_thread.get(getattr(thread, "id", "")),
    )


def _updated_at_sort_key(updated_at: Any) -> float:
    seconds = updated_at_seconds(updated_at)
    return seconds if seconds is not None else 0.0


def _demo_system_thread_ids() -> set[str]:
    return set(
        SystemAgentRun.objects.filter(agent_kind=demo.DEMO_AGENT_KIND)
        .exclude(thread_id="")
        .values_list("thread_id", flat=True)
        .distinct()
    )


def _demo_system_session_url(session_id: str) -> str:
    if not session_id:
        return ""
    run = (
        SystemAgentRun.objects.filter(
            thread_id=session_id,
            agent_kind=demo.DEMO_AGENT_KIND,
        )
        .order_by("-created_at", "-pk")
        .first()
    )
    if run is None:
        return ""
    path = reverse("system_session", kwargs={"session_id": session_id})
    return f"{path}?{urlencode({'run_id': run.pk})}"


def _system_agent_run_for_thread(
    thread_id: str, *, run_id: int | None = None
) -> SystemAgentRun | None:
    if not thread_id:
        return None
    if run_id is not None:
        return (
            SystemAgentRun.objects.filter(pk=run_id, thread_id=thread_id)
            .select_related("instance", "workflow")
            .first()
        )
    return (
        SystemAgentRun.objects.filter(thread_id=thread_id)
        .select_related("instance", "workflow")
        .order_by("-created_at", "-pk")
        .first()
    )


def _system_agent_instance_for_thread(thread_id: str) -> CodexInstance | None:
    if not thread_id:
        return None
    return (
        CodexInstance.objects.filter(
            thread_id=thread_id,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
        )
        .exclude(agent_kind=demo.DEMO_AGENT_KIND)
        .order_by("-started_at", "-pk")
        .first()
    )


def _system_agent_kind(
    run: SystemAgentRun | None, instance: CodexInstance | None = None
) -> str:
    if run is not None:
        return run.agent_kind
    if instance is not None:
        return instance.agent_kind
    return ""


def _system_agent_run_label(
    run: SystemAgentRun | None, instance: CodexInstance | None = None
) -> str:
    source_instance = run.instance if run is not None else instance
    display_author = source_instance.display_author.strip() if source_instance else ""
    if display_author:
        return display_author
    agent_kind = _system_agent_kind(run, instance)
    return agent_kind.replace("_", " ") if agent_kind else "system agent"


def _system_agent_status(
    run: SystemAgentRun | None, instance: CodexInstance | None = None
) -> str:
    if run is not None:
        return run.status
    return instance.status if instance is not None else ""


def _system_agent_run_detail_title(
    run: SystemAgentRun | None, instance: CodexInstance | None = None
) -> str:
    label = _system_agent_run_label(run, instance)
    return f"{label} log" if label else "System session"


def _metadata_rows_for_usage() -> list[SessionMetadata]:
    return list(
        SessionMetadata.objects.exclude(codex_updated_at__isnull=True).only(
            "thread_id",
            "codex_path",
            "codex_thread_source",
            "usage_last_checked_at",
        )
    )


def _usage_session_index_state() -> UsageSessionIndexState:
    # Source coverage is the availability contract: complete-but-empty is valid
    # zero usage. Pending cursors still need a full refresh, but they do not
    # make the previously complete coverage unavailable.
    active_complete = session_index.is_complete(archived=False)
    archived_complete = session_index.is_complete(archived=True)
    refresh_active = _usage_session_index_refresh_needed(archived=False)
    refresh_archived = _usage_session_index_refresh_needed(archived=True)
    return UsageSessionIndexState(
        active_complete=active_complete,
        archived_complete=archived_complete,
        refresh_active=refresh_active,
        refresh_archived=refresh_archived,
        totals_available=active_complete and archived_complete,
    )


def _schedule_usage_session_index_refresh_if_needed(
    *,
    enable_memories: bool,
    index_state: UsageSessionIndexState | None = None,
) -> None:
    if index_state is None:
        index_state = _usage_session_index_state()
    refresh_active = index_state.refresh_active
    refresh_archived = index_state.refresh_archived
    if not refresh_active and not refresh_archived:
        return
    _schedule_session_index_refresh(
        enable_memories=enable_memories,
        include_active=refresh_active,
        include_archived=refresh_archived,
    )


def _schedule_session_index_refresh(
    *, enable_memories: bool, include_active: bool, include_archived: bool
) -> None:
    if not include_active and not include_archived:
        return
    transaction.on_commit(
        lambda: _start_usage_session_index_refresh_thread(
            enable_memories=enable_memories,
            include_active=include_active,
            include_archived=include_archived,
        )
    )


def _usage_session_index_refresh_needed(*, archived: bool) -> bool:
    return (
        session_index.has_pending_pages(archived=archived)
        or session_index.should_refresh(archived=archived)
    )


def _start_usage_session_index_refresh_thread(
    *, enable_memories: bool, include_active: bool, include_archived: bool
) -> None:
    global _USAGE_SESSION_INDEX_REFRESH_IN_FLIGHT
    with _USAGE_SESSION_INDEX_REFRESH_LOCK:
        if _USAGE_SESSION_INDEX_REFRESH_IN_FLIGHT:
            return
        _USAGE_SESSION_INDEX_REFRESH_IN_FLIGHT = True
    try:
        threading.Thread(
            target=_refresh_usage_session_index_best_effort,
            kwargs={
                "enable_memories": enable_memories,
                "include_active": include_active,
                "include_archived": include_archived,
            },
            name="usage-session-index-refresh",
            daemon=True,
        ).start()
    except Exception:
        with _USAGE_SESSION_INDEX_REFRESH_LOCK:
            _USAGE_SESSION_INDEX_REFRESH_IN_FLIGHT = False
        logger.exception("failed to start usage session index refresh thread")


def _refresh_usage_session_index_best_effort(
    *, enable_memories: bool, include_active: bool, include_archived: bool
) -> None:
    global _USAGE_SESSION_INDEX_REFRESH_IN_FLIGHT
    try:
        close_old_connections()
        refresh_active = include_active and _usage_session_index_refresh_needed(
            archived=False
        )
        refresh_archived = include_archived and _usage_session_index_refresh_needed(
            archived=True
        )
        if not refresh_active and not refresh_archived:
            return
        with codex_pool.borrow_codex(
            Codex, enable_memories=enable_memories
        ) as codex:
            # Web-triggered catch-up must not ask Codex to scan rollouts: on a
            # large CODEX_HOME that backfill can hold Codex's SQLite writer lock
            # long enough to make detached workers exhaust their startup retry.
            # Since state-only data may exclude rollout-only sessions, this
            # warms rows without claiming source coverage is complete.
            session_index.refresh_from_codex(
                codex,
                projects=list(Project.objects.all()),
                include_active=refresh_active,
                include_archived=refresh_archived,
                use_state_db_only=True,
                max_pages=None,
                allow_completion=False,
            )
    except Exception:
        logger.exception("failed to refresh usage session index")
    finally:
        close_old_connections()
        with _USAGE_SESSION_INDEX_REFRESH_LOCK:
            _USAGE_SESSION_INDEX_REFRESH_IN_FLIGHT = False


def _next_message_config(
    settings: SettingsValues,
    resumed: Any,
    plan_model: str | None,
    *,
    cwd: str,
    approval_mode: str | None = None,
) -> list[dict[str, str]]:
    """Return the settings that will govern the next submitted message."""
    model = string_value(getattr(resumed, "model", None))
    reasoning = string_value(getattr(resumed, "reasoning_effort", None))
    plan_model_value = plan_model or "Unknown"
    sandbox_value = _option_label(
        _SANDBOX_POLICY_OPTIONS,
        _effective_sandbox_policy_for_cwd(settings, cwd),
        default="Codex default",
    )
    approval_value = _option_label(
        _APPROVAL_MODE_OPTIONS, approval_mode or _effective_approval_mode(settings)
    )
    web_search_value = _web_search_mode_label(settings.web_search_mode)
    return [
        {"label": "model", "value": model or "Unknown", "plan_value": plan_model_value},
        {
            "label": "reasoning",
            "value": reasoning or "Model default",
            "plan_value": _PLAN_MODE_REASONING_EFFORT,
        },
        {
            "label": "sandbox",
            "value": sandbox_value,
            "plan_value": sandbox_value,
        },
        {
            "label": "approval",
            "value": approval_value,
            "plan_value": approval_value,
        },
        {
            "label": "web search",
            "value": web_search_value,
            "plan_value": web_search_value,
        },
    ]


@dataclass(frozen=True)
class _ApprovalResolvedEvent:
    events_path: str
    request_id: int
    method: str
    decision: str


def _apply_live_approval_mode_to_instances(
    instances: QuerySet[CodexInstance], approval_mode: str
) -> None:
    if approval_mode not in _LIVE_HANDLER_APPROVAL_MODES:
        return
    instances = instances.filter(approval_mode_live_editable=True)
    instance_ids = list(instances.values_list("pk", flat=True))
    if not instance_ids:
        return
    resolved_events: list[_ApprovalResolvedEvent] = []
    with transaction.atomic():
        CodexInstance.objects.filter(pk__in=instance_ids).update(
            approval_mode=approval_mode
        )
        pending_decision = _LIVE_PENDING_APPROVAL_DECISIONS_BY_MODE.get(approval_mode)
        if pending_decision is not None:
            resolved_events = _settle_live_pending_approval_requests(
                instance_ids, pending_decision
            )
    _append_approval_resolved_events(resolved_events)


def _settle_live_pending_approval_requests(
    instance_ids: list[int], decision: str
) -> list[_ApprovalResolvedEvent]:
    pending = list(
        ApprovalRequest.objects.filter(
            instance_id__in=instance_ids,
            decision=ApprovalRequest.DECISION_PENDING,
        )
        .order_by("created_at", "pk")
        .values("pk", "method", "instance__events_path")
    )
    decided_at = timezone.now()
    resolved_events: list[_ApprovalResolvedEvent] = []
    for row in pending:
        request_id = row["pk"]
        updated = ApprovalRequest.objects.filter(
            pk=request_id,
            decision=ApprovalRequest.DECISION_PENDING,
        ).update(
            decision=decision,
            decided_at=decided_at,
        )
        if not updated:
            continue
        events_path = row["instance__events_path"]
        method = row["method"]
        if isinstance(events_path, str) and isinstance(method, str) and events_path:
            resolved_events.append(
                _ApprovalResolvedEvent(
                    events_path=events_path,
                    request_id=request_id,
                    method=method,
                    decision=decision,
                )
            )
    return resolved_events


def _append_approval_resolved_events(events: Iterable[_ApprovalResolvedEvent]) -> None:
    for event in events:
        try:
            codex_events.append_event(
                event.events_path,
                "approval/resolved",
                {
                    "id": event.request_id,
                    "method": event.method,
                    "decision": event.decision,
                },
            )
        except OSError:
            logger.warning(
                "failed to append live approval resolved event for request %s",
                event.request_id,
                exc_info=True,
            )


def _apply_live_session_approval_mode(
    session_id: str, effective_approval_mode: str
) -> None:
    _apply_live_approval_mode_to_instances(
        CodexInstance.objects.filter(
            thread_id=session_id,
            status__in=CodexInstance.ACTIVE_STATUSES,
        ),
        effective_approval_mode,
    )


def _apply_live_global_approval_mode(effective_approval_mode: str) -> None:
    explicit_override_thread_ids = SessionMetadata.objects.filter(
        approval_mode__in=_VALID_APPROVAL_MODES
    ).values("thread_id")
    _apply_live_approval_mode_to_instances(
        CodexInstance.objects.filter(
            purpose=CodexInstance.PURPOSE_USER,
            status__in=CodexInstance.ACTIVE_STATUSES,
        ).exclude(thread_id__in=explicit_override_thread_ids),
        effective_approval_mode,
    )


def _session_approval_mode_context(
    settings: SettingsValues,
    session_id: str,
    metadata: SessionMetadata | None,
) -> dict[str, Any]:
    global_label = _option_label(
        _APPROVAL_MODE_OPTIONS,
        _effective_approval_mode(settings),
    )
    override = _session_approval_mode_override(session_id, metadata)
    return {
        "set_approval_mode_url": reverse(
            "set_session_approval_mode", kwargs={"session_id": session_id}
        ),
        "session_approval_options": [
            {
                "id": "",
                "display_name": f"Follow global ({global_label})",
            },
            *[
                {"id": value, "display_name": label}
                for value, label in _APPROVAL_MODE_OPTIONS
            ],
        ],
        "current_session_approval_mode": override,
    }


def _base_instructions_for_settings(
    settings: SettingsValues, *, explicit_default: bool = False
) -> str | None:
    agent = _effective_coding_agent(settings)
    if agent == coding_agents.CODING_AGENT_CODEX:
        if explicit_default and settings.coding_agent == coding_agents.CODING_AGENT_CODEX:
            return coding_agents.default_codex_base_instructions()
        return None
    return coding_agents.base_instructions_for(agent)


def _parse_disk_usage_max_percent(raw: str) -> tuple[float | None, str | None]:
    value = raw.strip()
    if not value:
        return None, "disk usage limit is required"
    try:
        percent = float(value)
    except ValueError:
        return None, "invalid disk usage limit"
    if not math.isfinite(percent) or percent < 0.1 or percent > 100:
        return None, "invalid disk usage limit"
    rounded_tenths = round(percent * 10)
    if not math.isclose(percent * 10, rounded_tenths, abs_tol=1e-9):
        return None, "invalid disk usage limit"
    return rounded_tenths / 10, None


def _save_disk_usage_max_percent(value: float) -> None:
    settings, created = GlobalSettings.objects.get_or_create(
        pk=GlobalSettings.SINGLETON_PK,
        defaults={"disk_usage_max_percent": value},
    )
    if created or settings.disk_usage_max_percent == value:
        return
    settings.disk_usage_max_percent = value
    settings.save(update_fields=["disk_usage_max_percent", "updated_at"])


def _settings_with_visible_selected_project(
    values: SettingsValues, project: Project | None, *, cookie_required: bool
) -> SettingsValues:
    if project is None or values.visible_session_project_ids is None:
        return values
    if project.pk in values.visible_session_project_ids:
        return values
    visible_project_ids = (*values.visible_session_project_ids, project.pk)
    if cookie_required and not _visible_session_project_ids_cookie_fits(
        visible_project_ids
    ):
        return values._replace(visible_session_project_ids=None)
    return values._replace(visible_session_project_ids=visible_project_ids)


def _session_project_is_visible(
    project: Project | None, visibility: SessionProjectVisibility
) -> bool:
    if project is None:
        return visibility.include_no_project
    return visibility.project_ids is None or project.pk in visibility.project_ids


def _filter_session_metadata_by_project_visibility(
    rows: QuerySet[SessionMetadata], visibility: SessionProjectVisibility
) -> QuerySet[SessionMetadata]:
    if visibility.project_ids is None:
        if visibility.include_no_project:
            return rows
        return rows.exclude(project__isnull=True)
    project_filter = Q(project_id__in=visibility.project_ids)
    if visibility.include_no_project:
        project_filter |= Q(project__isnull=True)
    return rows.filter(project_filter)


def _filter_proposed_sessions_by_project_visibility(
    rows: QuerySet[ProposedSession], visibility: SessionProjectVisibility
) -> QuerySet[ProposedSession]:
    if visibility.project_ids is None:
        if visibility.include_no_project:
            return rows
        return rows.exclude(project__isnull=True)
    project_filter = Q(project_id__in=visibility.project_ids)
    if visibility.include_no_project:
        project_filter |= Q(project__isnull=True)
    return rows.filter(project_filter)


def _session_project_visibility_context(
    visibility: SessionProjectVisibility, projects: list[Project]
) -> dict[str, Any]:
    return {
        "visible_session_projects_url": reverse("update_visible_session_projects"),
        "visible_session_projects": [
            {
                "id": project.pk,
                "name": project.name,
                "visible": (
                    visibility.project_ids is None or project.pk in visibility.project_ids
                ),
            }
            for project in projects
        ],
        "visible_session_no_project": visibility.include_no_project,
    }


def _session_list_title(
    visibility: SessionProjectVisibility, projects: list[Project]
) -> str:
    if visibility.project_ids is None:
        return "Codex sessions"
    if len(visibility.project_ids) == 1 and not visibility.include_no_project:
        project_id = next(iter(visibility.project_ids))
        project = next(
            (project for project in projects if project.pk == project_id), None
        )
        if project is not None:
            return f"{project.name} sessions"
    return "Codex sessions"


def _project_visibility_label(
    visibility: SessionProjectVisibility, projects: list[Project]
) -> str:
    if visibility.project_ids is None:
        return "All projects"
    if len(visibility.project_ids) == 1 and not visibility.include_no_project:
        project_id = next(iter(visibility.project_ids))
        project = next(
            (project for project in projects if project.pk == project_id), None
        )
        if project is not None:
            return project.name
    if visibility.project_ids:
        return "Visible projects"
    if visibility.include_no_project:
        return "No repo"
    return "No projects"


def _project_visibility_shows_project_names(
    visibility: SessionProjectVisibility,
) -> bool:
    if visibility.project_ids is None:
        return True
    return visibility.include_no_project or len(visibility.project_ids) != 1


def _metadata_by_thread_id(threads: list[Any]) -> dict[str, SessionMetadata]:
    thread_ids = [
        thread.id
        for thread in threads
        if isinstance(getattr(thread, "id", None), str) and thread.id
    ]
    if not thread_ids:
        return {}
    return SessionMetadata.objects.select_related("project").in_bulk(
        thread_ids, field_name="thread_id"
    )


def _project_for_thread(
    thread: Any,
    metadata_by_thread: dict[str, SessionMetadata],
    projects: list[Project],
) -> Project | None:
    thread_id = getattr(thread, "id", "")
    metadata = metadata_by_thread.get(thread_id) if isinstance(thread_id, str) else None
    if metadata is not None and (metadata.project_id is not None or metadata.project_cleared):
        return metadata.project
    cwd = _thread_cwd(thread)
    if not cwd:
        return None
    return _project_for_cwd(cwd, projects)


def _project_for_cwd(cwd: str, projects: list[Project]) -> Project | None:
    return next(
        (
            project
            for project in projects
            if same_repo_or_worktree(cwd, project.repo_path, project.git_common_dir)
        ),
        None,
    )


def _developer_instructions_for_project(
    settings: SettingsValues, project: Project | None
) -> str:
    return "\n\n".join(
        part
        for part in (
            settings.extra_system_prompt.strip(),
            project.extra_system_prompt.strip() if project is not None else "",
        )
        if part
    )


def _associate_existing_sessions_with_project(project: Project, request: HttpRequest) -> None:
    settings = _stored_settings(request)
    try:
        with codex_pool.borrow_codex(
            Codex, enable_memories=settings.enable_memories
        ) as codex:
            threads = _all_threads(codex)
            try:
                threads.extend(_all_threads(codex, archived=True))
            except AppServerError:
                logger.warning("failed to list archived sessions while creating project")
    except AppServerError:
        logger.warning("failed to list sessions while creating project")
        return
    hidden_thread_ids = system_agents.hidden_thread_ids()
    seen: set[str] = set()
    for thread in threads:
        thread_id = getattr(thread, "id", None)
        if not isinstance(thread_id, str) or not thread_id or thread_id in seen:
            continue
        seen.add(thread_id)
        if thread_id in hidden_thread_ids:
            continue
        cwd = _thread_cwd(thread)
        if not cwd or not same_repo_or_worktree(cwd, project.repo_path, project.git_common_dir):
            continue
        metadata = SessionMetadata.objects.filter(thread_id=thread_id).first()
        if metadata is not None and metadata.project_cleared:
            continue
        SessionMetadata.objects.update_or_create(
            thread_id=thread_id,
            defaults={"cwd": cwd, "project": project, "project_cleared": False},
        )


def _matching_project_exists(repo_path: str, repo_common_dir: str) -> bool:
    for project in Project.objects.all():
        if project.repo_path == repo_path:
            return True
        if repo_common_dir and project.git_common_dir == repo_common_dir:
            return True
        if same_repo_or_worktree(repo_path, project.repo_path, project.git_common_dir):
            return True
    return False


def _creatable_project_repos(discovered_repos: list[str]) -> list[str]:
    creatable: list[str] = []
    for repo_path in discovered_repos:
        repo_common_dir = str(git_common_dir(repo_path) or "")
        if _matching_project_exists(repo_path, repo_common_dir):
            continue
        creatable.append(repo_path)
    return creatable


def _active_instance_for(session_id: str) -> CodexInstance | None:
    """Return the latest *active* CodexInstance for ``session_id``, or None.

    Selecting on status first (rather than picking the newest row and
    checking its status) means a quickly-terminal newer row doesn't mask
    an older worker that's still mid-turn — ``send_message`` can stack
    workers, so the page must stay in streaming mode as long as any one
    of them is alive.
    """
    return codex_pool.latest_active_for_thread(session_id)


def _trim_in_progress_turn(
    entries: list[dict[str, Any]], active: CodexInstance | None
) -> list[dict[str, Any]]:
    """Drop the in-progress turn's entries from the tail of ``entries``.

    The SSE stream re-emits every event from the start of the worker's
    events file, including the user message and any agent / tool items
    the rollout has already captured. Without this trim those entries
    render twice on the live page — once from the server-side rollout
    pass, once by the streaming JS that can't dedupe against DOM nodes
    it didn't create.

    The in-progress turn is identified by the most recent user-message entry
    whose text matches the active worker's original prompt plus its initial
    image markers. Mid-turn steer images live in the attachment ledger but do
    not change this identity; anything from the original user message onward
    is owned by the stream until the turn ends.
    """
    active_text = _active_user_message_text(active)
    if not active_text:
        return entries
    for i in range(len(entries) - 1, -1, -1):
        entry = entries[i]
        if entry.get("kind") == "user" and entry.get("text") == active_text:
            return entries[:i]
    return entries


def _show_active_worker_transcript(active: CodexInstance | None) -> bool:
    return active is not None and active.agent_kind != demo.DEMO_AGENT_KIND


def _pending_user_prompt(active: CodexInstance | None) -> str:
    """Surface the active worker's prompt as a pending user bubble.

    Pairs with ``_trim_in_progress_turn``: that helper strips the
    in-progress turn from the rollout-rendered entries (so the stream
    owns rendering it), which means the user wouldn't see their own
    message at all between Send and the first stream event without this
    placeholder. The streaming JS removes the bubble as soon as the
    real ``userMessage`` event lands.
    """
    if active is None or active.agent_kind == demo.DEMO_AGENT_KIND:
        return ""
    return _active_user_message_text(active)


def _active_user_message_text(active: CodexInstance | None) -> str:
    if active is None:
        return ""
    parts: list[str] = []
    if active.prompt:
        parts.append(active.prompt)
    parts.extend(
        "[image]" for _path in _normalized_json_string_list(active.input_image_paths)
    )
    return "\n".join(parts)


def _normalized_json_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _task_plan_context(
    snapshot: codex_events.TaskPlanSnapshot | None,
) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    current = _current_task_text(snapshot.steps) or snapshot.explanation
    return {
        "visible": bool(current or snapshot.steps),
        "current": current,
        "explanation": snapshot.explanation,
        "recorded_at": snapshot.order[0],
        "event_seq": snapshot.order[1],
        "fallback_order": snapshot.order[2],
        "steps": [
            {"step": step.step, "status": step.status}
            for step in snapshot.steps
        ],
    }


def _current_task_text(steps: tuple[codex_events.TaskPlanStep, ...]) -> str:
    for status in ("inProgress", "pending"):
        for step in steps:
            if step.status == status:
                return step.step
    return steps[-1].step if steps else ""


def _pending_user_author(active: CodexInstance | None) -> str:
    if active is None:
        return ""
    if active.agent_kind == demo.DEMO_AGENT_KIND:
        return active.display_author
    return active.display_author if active.purpose == CodexInstance.PURPOSE_SYSTEM_FEEDBACK else ""


def _pending_user_timestamp(active: CodexInstance | None) -> int:
    if active is None or active.agent_kind == demo.DEMO_AGENT_KIND:
        return 0
    return int(active.started_at.timestamp())


def _workflow_status_text(workflow: Any | None) -> str:
    return streaming.system_workflow_status_text(workflow)


def _workflow_composer_label(workflow: SystemWorkflow | None) -> str:
    if workflow is not None and workflow.kind == SystemWorkflow.KIND_PR_QA:
        return "QA workflow"
    return "Hitch workflow"


def _workflow_accepts_qa_pause_steering(workflow: SystemWorkflow | None) -> bool:
    return (
        workflow is not None
        and workflow.kind == SystemWorkflow.KIND_PR_QA
        and workflow.is_active
        and workflow.step == system_agents.STEP_QA_RUNNING
    )


def _workflow_accepts_active_turn_steering(
    workflow: SystemWorkflow | None, active: CodexInstance | None
) -> bool:
    return (
        workflow is not None
        and active is not None
        and workflow.kind == SystemWorkflow.KIND_PR_QA
        and workflow.is_active
        and workflow.step == system_agents.STEP_USER_STEERING_RUNNING
        and active.workflow_id == workflow.pk
        and active.purpose == CodexInstance.PURPOSE_USER
        and active.thread_id == workflow.main_thread_id
        and active.is_active
    )


def _active_worker_status_text(active: CodexInstance | None) -> str:
    if active is not None and active.agent_kind == demo.DEMO_AGENT_KIND:
        return "Demo agent is working"
    return streaming.qa_agent_status_text_for_instance(active)


def _apply_system_authors(
    entries: list[dict[str, Any]], session_id: str
) -> list[dict[str, Any]]:
    system_authors: dict[int, str] = {}
    for user_message_index, author in CodexInstance.objects.filter(
        thread_id=session_id,
        purpose=CodexInstance.PURPOSE_SYSTEM_FEEDBACK,
        user_message_index__isnull=False,
    ).values_list("user_message_index", "display_author"):
        if isinstance(user_message_index, int) and author:
            system_authors[user_message_index] = author
    if not system_authors:
        return entries
    user_message_index = 0
    for entry in entries:
        user_message_index = _apply_system_author(
            entry, system_authors, user_message_index
        )
    return entries


def _apply_system_author(
    entry: dict[str, Any], system_authors: dict[int, str], user_message_index: int
) -> int:
    if entry.get("kind") == "user":
        author = system_authors.get(user_message_index)
        if author:
            entry["display_author"] = author
        return user_message_index + 1
    if entry.get("kind") == "intermediate":
        for item in entry.get("items", []):
            user_message_index = _apply_system_author(
                item, system_authors, user_message_index
            )
    return user_message_index


def _filter_demo_agent_entries(
    entries: list[dict[str, Any]], session_id: str
) -> list[dict[str, Any]]:
    hidden_prompts: set[str] = set()
    for prompt in CodexInstance.objects.filter(
        thread_id=session_id,
        agent_kind=demo.DEMO_AGENT_KIND,
    ).values_list("prompt", flat=True):
        if isinstance(prompt, str) and prompt:
            hidden_prompts.add(prompt)
    if not hidden_prompts:
        return entries

    filtered: list[dict[str, Any]] = []
    suppress_turn = False
    for entry in entries:
        if entry.get("kind") == "user":
            text = entry.get("text")
            suppress_turn = isinstance(text, str) and text in hidden_prompts
        if suppress_turn and _preserve_during_hidden_demo_turn(entry):
            filtered.append(entry)
            continue
        if not suppress_turn:
            filtered.append(entry)
    return filtered


def _preserve_during_hidden_demo_turn(entry: dict[str, Any]) -> bool:
    return entry.get("kind") == "agent" and bool(entry.get("display_author"))


def _apply_qa_approval_messages(
    entries: list[dict[str, Any]], session_id: str
) -> list[dict[str, Any]]:
    approvals = sorted(_qa_approval_entries(session_id), key=lambda item: item[0])
    if not approvals:
        return entries
    result: list[dict[str, Any]] = []
    user_message_index = 0
    pending = approvals.copy()
    for entry in entries:
        if entry.get("kind") == "user":
            while pending and pending[0][0] == user_message_index:
                _index, approval = pending.pop(0)
                result.append(approval)
            user_message_index += 1
        result.append(entry)
    result.extend(approval for _index, approval in pending)
    return result


def _qa_approval_entries(session_id: str) -> Iterator[tuple[int, dict[str, Any]]]:
    workflows = (
        SystemWorkflow.objects.filter(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id=session_id,
            status=SystemWorkflow.STATUS_COMPLETED,
            step__in=[
                system_agents.STEP_PR_PROMPT_SPAWNED,
                system_agents.STEP_QA_APPROVED,
                system_agents.STEP_PR_READY,
                system_agents.STEP_PR_CLOSED,
                system_agents.STEP_LOCAL_BRANCH_MERGED,
            ],
        )
        .order_by("created_at")
        .prefetch_related("agent_runs")
    )
    for workflow in workflows:
        next_user_message_index = workflow.state.get("next_user_message_index")
        if not isinstance(next_user_message_index, int):
            continue
        run = _approved_qa_run(workflow)
        if run is None:
            continue
        feedback = _qa_feedback_text(workflow, run)
        text = _qa_approval_text(workflow)
        if feedback:
            text = f"{text}\n\n{feedback}"
        if workflow.step in {
            system_agents.STEP_QA_APPROVED,
            system_agents.STEP_LOCAL_BRANCH_MERGED,
        }:
            insert_index = next_user_message_index
        else:
            prompt_index = workflow.state.get(
                system_agents.QA_APPROVAL_INSERT_INDEX_STATE_KEY
            )
            insert_index = (
                prompt_index
                if isinstance(prompt_index, int) and not isinstance(prompt_index, bool)
                else max(next_user_message_index - 1, 0)
            )
        # ``_finalize_agent_entry`` would skip single-finding feedback
        # (``looks_like_markdown`` needs two bullets), so render the body
        # directly: QA feedback per the agent prompt carries structured
        # findings and must reach the user formatted even for one finding.
        yield insert_index, {
            "kind": "agent",
            "display_author": system_agents.QA_DISPLAY_AUTHOR,
            "text": text,
            "html": render_markdown(text),
            "timestamp": int(workflow.updated_at.timestamp()),
        }


def _qa_approval_text(workflow: SystemWorkflow) -> str:
    if workflow.step != system_agents.STEP_LOCAL_BRANCH_MERGED:
        return "QA agent approved the diff."
    result = workflow.state.get("auto_merge_result")
    if not isinstance(result, dict):
        return "QA agent approved the diff and merged it to the local branch."
    branch = result.get("branch")
    commit_sha = result.get("commit_sha")
    changed = result.get("changed")
    if not isinstance(branch, str) or not branch.strip():
        return "QA agent approved the diff and merged it to the local branch."
    action = "merged it into"
    if changed is False:
        action = "found it already applied to"
    text = f"QA agent approved the diff and {action} {branch.strip()}."
    if isinstance(commit_sha, str) and commit_sha.strip():
        text = f"{text}\n\nCommit: {commit_sha.strip()}"
    return text


def _approved_qa_run(workflow: SystemWorkflow) -> SystemAgentRun | None:
    runs = sorted(
        workflow.agent_runs.all(),
        key=lambda item: item.created_at,
        reverse=True,
    )
    for run in runs:
        if run.status != SystemAgentRun.STATUS_COMPLETED:
            continue
        output = run.output if isinstance(run.output, dict) else {}
        if output.get("lgtm") is True:
            return run
    return None


def _qa_feedback_text(workflow: SystemWorkflow, run: SystemAgentRun) -> str:
    feedback = workflow.state.get("last_feedback")
    if not isinstance(feedback, str) or not feedback.strip():
        output = run.output if isinstance(run.output, dict) else {}
        feedback = output.get("feedback")
    return feedback.strip() if isinstance(feedback, str) else ""


@require_http_methods(["GET"])
def session_stream(request: HttpRequest, session_id: str) -> StreamingHttpResponse:
    """SSE endpoint that mirrors the active worker's events file to the browser.

    When no worker is active the connection stays open emitting heartbeat
    frames so the page's connection indicator can show ``connected, idle``,
    and reloads itself when a worker is later spawned out-of-band.

    The page passes its render-time view of the session state on the URL
    (``baseline`` = latest ``CodexInstance.pk``, ``active`` = the active
    worker's pk if any). If either differs from what the database shows
    when SSE opens, the page is by definition stale (e.g. a worker was
    spawned/completed in the gap, or has already finished by the time
    the browser opens SSE) — we force an immediate reload so the DOM
    matches reality before any item events start flowing.
    """
    baseline_param = request.GET.get("baseline", "")
    active_param = request.GET.get("active", "")
    workflow_param = request.GET.get("workflow", "")
    demo_param = request.GET.get("demo", "")
    codex_pool.reconcile_dead_for_thread(session_id)
    codex_pool.reconcile_dead_if_due()
    current_latest = codex_pool.latest_id_for_thread(session_id)
    current_latest_str = str(current_latest) if current_latest is not None else ""
    active = _active_instance_for(session_id)
    current_active_str = str(active.pk) if active is not None else ""
    active_workflow = system_agents.active_workflow_for_thread(session_id)
    current_workflow_str = str(active_workflow.pk) if active_workflow is not None else ""
    current_demo_str = streaming.demo_stream_token(session_id)

    if (
        baseline_param != current_latest_str
        or active_param != current_active_str
        or workflow_param != current_workflow_str
        or demo_param != current_demo_str
    ):
        response = StreamingHttpResponse(
            streaming.reload_stream(), content_type="text/event-stream"
        )
    elif active is not None:
        response = StreamingHttpResponse(
            streaming.stream_for_instance(active, demo_baseline=current_demo_str),
            content_type="text/event-stream",
        )
    elif active_workflow is not None:
        response = StreamingHttpResponse(
            streaming.system_workflow_stream(
                session_id, current_latest, active_workflow.pk
            ),
            content_type="text/event-stream",
        )
    else:
        response = StreamingHttpResponse(
            streaming.idle_stream(session_id, current_latest, current_demo_str),
            content_type="text/event-stream",
        )
    # Discourage proxies from buffering: SSE depends on every frame reaching
    # the client immediately, not coalesced into a single response body.
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


def _stream_url_for(
    session_id: str,
    active_instance: CodexInstance | None,
    active_workflow: Any | None = None,
) -> str:
    """Build the SSE URL for the session view, tagging it with the page's
    render-time view of the session state.

    Pairs with ``session_stream``: those query params are how the SSE
    endpoint detects that the page was rendered against a stale DB view
    (worker spawned/completed in the gap) and forces a reload. The
    params are always emitted (even when empty) so a malformed direct
    hit without them is treated as ``no state known`` and routes to the
    reload path on any non-empty DB state.
    """
    baseline_id = codex_pool.latest_id_for_thread(session_id)
    active_id = active_instance.pk if active_instance is not None else None
    workflow_id = active_workflow.pk if active_workflow is not None else None
    demo_token = streaming.demo_stream_token(session_id)
    qs = urlencode(
        {
            "baseline": str(baseline_id) if baseline_id is not None else "",
            "active": str(active_id) if active_id is not None else "",
            "workflow": str(workflow_id) if workflow_id is not None else "",
            "demo": demo_token,
        }
    )
    return f"{reverse('session_stream', kwargs={'session_id': session_id})}?{qs}"


def _display_title(thread: Any) -> str:
    """Return a short, single-line title for a thread.

    Falls back through `name` -> first line of `preview` -> `id`, clipping
    to ``_DISPLAY_TITLE_MAX_LEN`` so a long auto-fallback preview cannot
    overflow the row. Threads without any usable text degrade to the id
    rather than to a blank link.
    """
    name = getattr(thread, "name", None)
    candidate = name.strip() if isinstance(name, str) else ""
    if not candidate:
        preview = getattr(thread, "preview", None) or ""
        candidate = preview.split("\n", 1)[0].strip()
    if not candidate:
        return getattr(thread, "id", "") or ""
    if len(candidate) > _DISPLAY_TITLE_MAX_LEN:
        return candidate[:_DISPLAY_TITLE_MAX_LEN].rstrip() + "..."
    return candidate


def _entries_for(thread: Any) -> Iterator[dict[str, Any]]:
    """Prefer the on-disk rollout so commandExecution rows surface.

    ``thread/read`` rebuilds turns through codex's Limited-mode persistence
    filter, which drops every commandExecution item. When ``Thread.path``
    points at a rollout file we can read, parse it ourselves to recover the
    dropped entries; otherwise (ephemeral threads, unreadable paths, parser
    failures, or an empty rollout) fall back to rebuilding from
    ``Thread.turns`` so the page is never empty just because the rollout
    layer misbehaved.
    """
    flat = _entries_from_rollout(thread)
    if flat is not None:
        yield from collapse_flat_entries(flat)
        return
    yield from render_entries(thread)


def _entries_from_rollout(thread: Any) -> list[dict[str, Any]] | None:
    """Materialise entries from the on-disk rollout, or return None to fall back.

    Returning ``None`` (vs. an empty list) is what triggers the SDK fallback;
    an empty list is treated as "the rollout exists and is genuinely empty,"
    matching the behaviour of an empty ``Thread.turns``.
    """
    path = getattr(thread, "path", None)
    if not isinstance(path, str) or not path:
        return None
    rollout_path = Path(path)
    if not rollout_path.is_file():
        logger.warning("thread.path %s is not a readable file; falling back to SDK turns", path)
        return None
    try:
        entries = list(rollout.iter_entries(rollout_path))
    except Exception:
        logger.exception("failed to parse rollout %s; falling back to SDK turns", path)
        return None
    # If the rollout reconstructs no conversation at all but the SDK has
    # turns, prefer the SDK output — that combination almost always means
    # the rollout schema drifted under us (renamed event tag, new wrapper)
    # so our parser silently skipped the user/agent messages even though
    # they are present on disk. A rollout with only tool-call entries is
    # treated the same way: the SDK path may know how to surface the user
    # request, and we'd rather render the conversation without commands
    # than render commands without the conversation. A truly empty parse
    # against an equally empty Thread.turns falls through so the page can
    # show its empty-state placeholder.
    if getattr(thread, "turns", None) and not any(
        entry["kind"] in ("user", "agent") for entry in entries
    ):
        logger.warning(
            "rollout %s yielded no user/agent entries; falling back to SDK turns", path
        )
        return None
    return entries


def _import_cookie_settings_to_user(request: HttpRequest, user: Any) -> UserSettings:
    settings = _settings_for_user(user)
    updates: list[str] = []
    for field, value in _valid_cookie_setting_updates(request).items():
        if getattr(settings, field) != value:
            setattr(settings, field, value)
            updates.append(field)
    if updates:
        settings.save(update_fields=[*updates, "updated_at"])
    return settings


def _safe_next_url(request: HttpRequest) -> str:
    candidate = request.POST.get("next", "").strip() or request.GET.get(
        "next", ""
    ).strip()
    if not candidate:
        return ""
    if url_has_allowed_host_and_scheme(
        candidate,
        allowed_hosts={request.get_host()},
        require_https=request.is_secure(),
    ):
        return candidate
    return ""


def _new_session_post_settings(request: HttpRequest) -> ResolvedSettings:
    stored_settings = _stored_settings(request)
    enable_memories = stored_settings.enable_memories
    if caches._models_cache_has_value(
        enable_memories=enable_memories
    ) and not caches._models_refresh_needed(enable_memories=enable_memories):
        models_data = caches._cached_models_data(enable_memories=enable_memories)
        if models_data:
            return _resolved_settings(request, models_data)

    with codex_pool.borrow_codex(Codex, enable_memories=enable_memories) as codex:
        models_data = list(codex.models().data)
    caches._store_models_cache(enable_memories=enable_memories, models_data=models_data)
    return _resolved_settings(request, models_data)


@require_http_methods(["GET", "POST"])
def update_settings(request: HttpRequest) -> HttpResponse:
    if request.method == "GET":
        models_data, resolved_settings = _cached_models_and_settings(request)
        next_url = _safe_next_url(request) or reverse("index")
        response = render(
            request,
            "settings.html",
            {
                "settings_next_url": next_url,
                "settings_cancel_url": next_url,
                **_settings_context(
                    resolved_settings.values,
                    models_data,
                ),
            },
        )
        _apply_cookie_updates(response, resolved_settings.cookie_updates)
        return response

    model = request.POST.get("model", "").strip()
    effort = request.POST.get("reasoning_effort", "").strip()
    sandbox = request.POST.get("sandbox_policy", "").strip()
    approval = request.POST.get("approval_mode", "").strip()
    coding_agent = request.POST.get("coding_agent", "").strip()
    extra_system_prompt = request.POST.get("extra_system_prompt", "").strip()
    use_worktrees = request.POST.get("use_worktrees", "").strip()
    auto_pr = request.POST.get("auto_pr", "").strip()
    auto_qa = request.POST.get("auto_qa", "").strip()
    spec_critic = request.POST.get("spec_critic", "").strip()
    web_search_mode = request.POST.get("web_search_mode", "").strip()
    posted_disk_usage_max_percent = request.POST.get("disk_usage_max_percent")
    posted_initial_disk_usage_max_percent = request.POST.get(
        "initial_disk_usage_max_percent"
    )
    posted_show_archived = request.POST.get("show_archived_sessions")
    show_archived = (
        posted_show_archived.strip() if posted_show_archived is not None else None
    )
    selected_project, selected_project_error = _posted_project(
        request.POST.get("selected_project", "")
    )
    if selected_project_error is not None:
        return HttpResponseBadRequest(selected_project_error)
    enable_memories = request.POST.get("enable_memories", "").strip()
    user = _authenticated_user(request)
    if len(model) > _MODEL_MAX_LEN:
        return HttpResponseBadRequest("model id is too long")
    if len(extra_system_prompt) > _EXTRA_SYSTEM_PROMPT_MAX_LEN:
        return HttpResponseBadRequest("extra system prompt is too long")
    # The character cap above does not bound the encoded cookie size, so a
    # multibyte prompt can still overflow the browser cookie limit and be
    # silently dropped. For anonymous users the cookie is the only store, so
    # that means the setting is lost — reject it up front. Authenticated users
    # persist to the DB (the cookie is just a best-effort mirror), so a value
    # too big for the cookie still saves correctly; don't block them on it.
    if user is None and not _extra_system_prompt_cookie_fits(extra_system_prompt):
        return HttpResponseBadRequest("extra system prompt is too long")
    valid_efforts = {e.value for e in ReasoningEffort}
    if effort and effort not in valid_efforts:
        return HttpResponseBadRequest("invalid reasoning effort")
    if sandbox and sandbox not in _VALID_SANDBOX_POLICIES:
        return HttpResponseBadRequest("invalid sandbox policy")
    # Approval mode always carries one of the dialog's values. An empty
    # form post is treated as "user picked nothing", which we snap to the
    # safe default.
    if approval and approval not in _VALID_APPROVAL_MODES:
        return HttpResponseBadRequest("invalid approval mode")
    if not approval:
        approval = _DEFAULT_APPROVAL_MODE
    if coding_agent and coding_agent not in coding_agents.VALID_CODING_AGENTS:
        return HttpResponseBadRequest("invalid coding agent")
    if not coding_agent:
        coding_agent = coding_agents.DEFAULT_CODING_AGENT
    if use_worktrees not in {"", "true"}:
        return HttpResponseBadRequest("invalid worktree setting")
    use_worktrees = "true" if use_worktrees == "true" else "false"
    if auto_pr not in {"", "true"}:
        return HttpResponseBadRequest("invalid auto-PR setting")
    auto_pr = "true" if auto_pr == "true" else "false"
    if auto_qa not in {"", "true"}:
        return HttpResponseBadRequest("invalid auto-QA setting")
    auto_qa = "true" if auto_qa == "true" else "false"
    if spec_critic not in {"", "true"}:
        return HttpResponseBadRequest("invalid Spec Critic setting")
    spec_critic = "true" if spec_critic == "true" else "false"
    if web_search_mode and web_search_mode not in _VALID_WEB_SEARCH_MODES:
        return HttpResponseBadRequest("invalid web search setting")
    disk_usage_max_percent: float | None = None
    if posted_disk_usage_max_percent is not None:
        disk_usage_max_percent, disk_usage_error = _parse_disk_usage_max_percent(
            posted_disk_usage_max_percent
        )
        if disk_usage_error is not None:
            return HttpResponseBadRequest(disk_usage_error)
        if posted_initial_disk_usage_max_percent is not None:
            initial_disk_usage_max_percent, initial_disk_usage_error = (
                _parse_disk_usage_max_percent(posted_initial_disk_usage_max_percent)
            )
            if initial_disk_usage_error is not None:
                return HttpResponseBadRequest(initial_disk_usage_error)
            if disk_usage_max_percent == initial_disk_usage_max_percent:
                disk_usage_max_percent = None
    if show_archived is not None and show_archived not in {"", "true"}:
        return HttpResponseBadRequest("invalid archived sessions visibility")
    if enable_memories not in {"", "true"}:
        return HttpResponseBadRequest("invalid memories setting")
    enable_memories = "true" if enable_memories == "true" else "false"
    if model or effort:
        # Cross-check the posted (model, effort) pair against what Codex
        # actually offers so a malformed POST (typo, stale model id, effort
        # the chosen model doesn't support) gets a clean 400 instead of
        # quietly poisoning every subsequent turn at runtime.
        enable_memories_value = enable_memories == "true"
        cache_has_value = caches._models_cache_has_value(enable_memories=enable_memories_value)
        models_data = caches._cached_models_data(enable_memories=enable_memories_value)
        if cache_has_value:
            caches._schedule_models_refresh(enable_memories=enable_memories_value)
        else:
            with codex_pool.borrow_codex(
                Codex, enable_memories=enable_memories_value
            ) as codex:
                models_data = list(codex.models().data)
        compat_error = _validate_settings_against_models(model, effort, models_data)
        if compat_error:
            return HttpResponseBadRequest(compat_error)
    stored = _stored_settings(request)
    values = SettingsValues(
        model=model,
        reasoning_effort=effort,
        sandbox_policy=sandbox,
        approval_mode=approval,
        coding_agent=coding_agent,
        extra_system_prompt=extra_system_prompt,
        use_worktrees=use_worktrees == "true",
        auto_pr_enabled=auto_pr == "true",
        auto_qa_enabled=auto_qa == "true",
        spec_critic_enabled=spec_critic == "true",
        web_search_mode=web_search_mode,
        show_archived_sessions=(
            stored.show_archived_sessions
            if show_archived is None
            else show_archived == "true"
        ),
        last_selected_repo=stored.last_selected_repo,
        selected_project_id=selected_project.pk if selected_project is not None else None,
        visible_session_project_ids=stored.visible_session_project_ids,
        show_no_project_sessions=stored.show_no_project_sessions,
        enable_memories=enable_memories == "true",
    )
    values = _settings_with_visible_selected_project(
        values, selected_project, cookie_required=user is None
    )
    if disk_usage_max_percent is not None:
        _save_disk_usage_max_percent(disk_usage_max_percent)
    if user is not None:
        _save_user_settings(user, values)
    _apply_live_global_approval_mode(values.approval_mode)
    response = redirect(_safe_next_url(request) or "index")
    _apply_cookie_updates(response, _settings_cookie_updates(values))
    return response


@require_http_methods(["POST"])
def update_archived_session_visibility(request: HttpRequest) -> HttpResponse:
    show_archived = request.POST.get("show_archived_sessions", "").strip()
    if show_archived not in {"", "true"}:
        return HttpResponseBadRequest("invalid archived sessions visibility")
    stored = _stored_settings(request)
    values = stored._replace(show_archived_sessions=show_archived == "true")
    user = _authenticated_user(request)
    if user is not None:
        _save_user_settings(user, values)
    response = redirect("index")
    _apply_cookie_updates(response, _settings_cookie_updates(values))
    return response


@require_http_methods(["POST"])
def update_visible_session_projects(request: HttpRequest) -> HttpResponse:
    projects = list(Project.objects.all())
    valid_project_ids = {project.pk for project in projects}
    posted_project_ids: set[int] = set()
    for raw_project_id in request.POST.getlist("visible_project"):
        try:
            project_id = int(raw_project_id)
        except ValueError:
            return HttpResponseBadRequest("invalid visible project")
        if project_id not in valid_project_ids:
            return HttpResponseBadRequest("invalid visible project")
        posted_project_ids.add(project_id)
    show_no_project = request.POST.get("show_no_project_sessions", "").strip()
    if show_no_project not in {"", "true"}:
        return HttpResponseBadRequest("invalid no repo visibility")
    visible_project_ids = tuple(
        project.pk for project in projects if project.pk in posted_project_ids
    )
    user = _authenticated_user(request)
    if user is None and not _visible_session_project_ids_cookie_fits(
        visible_project_ids
    ):
        return HttpResponseBadRequest("visible project selection is too large")
    stored = _stored_settings(request)
    values = stored._replace(
        visible_session_project_ids=visible_project_ids,
        show_no_project_sessions=show_no_project == "true",
    )
    if user is not None:
        _save_user_settings(user, values)
    response = redirect(_safe_next_url(request) or "index")
    _apply_cookie_updates(response, _settings_cookie_updates(values))
    return response


@require_http_methods(["GET", "POST"])
def new_project(request: HttpRequest) -> HttpResponse:
    discovered_repos = [str(p) for p in repos_module.discover_repos()]
    repos = _creatable_project_repos(discovered_repos)
    if request.method == "GET":
        return render(
            request,
            "project_form.html",
            {
                "repos": repos,
                "name_max_len": _PROJECT_NAME_MAX_LEN,
                "index_url": reverse("index"),
            },
        )

    name = request.POST.get("name", "").strip()
    repo_path = request.POST.get("repo_path", "").strip()
    if not name:
        return HttpResponseBadRequest("project name is required")
    if len(name) > _PROJECT_NAME_MAX_LEN:
        return HttpResponseBadRequest("project name is too long")
    if not repo_path:
        return HttpResponseBadRequest("repository is required")
    if repo_path not in set(discovered_repos):
        return HttpResponseBadRequest("repository must be a discovered repository")
    repo_common_dir = str(git_common_dir(repo_path) or "")
    if _matching_project_exists(repo_path, repo_common_dir):
        return HttpResponseBadRequest("project already exists for repository")

    project = Project.objects.create(
        name=name,
        repo_path=repo_path,
        git_common_dir=repo_common_dir,
    )
    _associate_existing_sessions_with_project(project, request)

    stored = _stored_settings(request)
    values = stored._replace(selected_project_id=project.pk, last_selected_repo=repo_path)
    user = _authenticated_user(request)
    values = _settings_with_visible_selected_project(
        values, project, cookie_required=user is None
    )
    if user is not None:
        _save_user_settings(user, values)
    response = redirect("index")
    _apply_cookie_updates(response, _settings_cookie_updates(values))
    return response


@require_http_methods(["POST"])
def edit_project(request: HttpRequest) -> HttpResponse:
    project, project_error = _posted_project(request.POST.get("project", ""))
    if project_error is not None:
        return HttpResponseBadRequest(project_error)
    if project is None:
        return HttpResponseBadRequest("project is required")
    name = request.POST.get("name", "").strip()
    extra_system_prompt = request.POST.get("extra_system_prompt", "").strip()
    auto_pr_mode = request.POST.get("auto_pr_mode", "").strip()
    if not name:
        return HttpResponseBadRequest("project name is required")
    if len(name) > _PROJECT_NAME_MAX_LEN:
        return HttpResponseBadRequest("project name is too long")
    if len(extra_system_prompt) > _EXTRA_SYSTEM_PROMPT_MAX_LEN:
        return HttpResponseBadRequest("extra system prompt is too long")
    if auto_pr_mode not in _VALID_PROJECT_AUTO_PR_MODES:
        return HttpResponseBadRequest("invalid project auto-PR setting")

    updates: list[str] = []
    if project.name != name:
        project.name = name
        updates.append("name")
    if project.extra_system_prompt != extra_system_prompt:
        project.extra_system_prompt = extra_system_prompt
        updates.append("extra_system_prompt")
    if project.auto_pr_mode != auto_pr_mode:
        project.auto_pr_mode = auto_pr_mode
        updates.append("auto_pr_mode")
    if updates:
        project.save(update_fields=[*updates, "updated_at"])
    return redirect(_safe_next_url(request) or "index")


@require_http_methods(["POST"])
def set_session_project(request: HttpRequest, session_id: str) -> HttpResponse:
    project, error = _posted_project(request.POST.get("project", ""))
    if error is not None:
        return HttpResponseBadRequest(error)
    metadata = SessionMetadata.objects.filter(thread_id=session_id).first()
    cwd = metadata.cwd if metadata is not None and metadata.cwd else ""
    if not cwd:
        settings = _stored_settings(request)
        resumed = codex_pool.run_borrowed_op_with_retry(
            Codex,
            lambda codex: codex._client.thread_resume(session_id),
            enable_memories=settings.enable_memories,
        )
        cwd = _thread_cwd(resumed.thread) or ""
    SessionMetadata.objects.update_or_create(
        thread_id=session_id,
        defaults={
            "cwd": cwd,
            "project": project,
            "project_cleared": project is None,
        },
    )
    return redirect("session", session_id=session_id)


@require_http_methods(["POST"])
def set_session_approval_mode(request: HttpRequest, session_id: str) -> HttpResponse:
    approval_mode = request.POST.get("approval_mode", "").strip()
    if approval_mode and approval_mode not in _VALID_APPROVAL_MODES:
        return HttpResponseBadRequest("invalid approval mode")
    metadata = SessionMetadata.objects.filter(thread_id=session_id).first()
    cwd = metadata.cwd if metadata is not None and metadata.cwd else ""
    if not cwd:
        settings = _stored_settings(request)
        resumed = codex_pool.run_borrowed_op_with_retry(
            Codex,
            lambda codex: codex._client.thread_resume(session_id),
            enable_memories=settings.enable_memories,
        )
        cwd = _thread_cwd(resumed.thread) or ""
    SessionMetadata.objects.update_or_create(
        thread_id=session_id,
        defaults={
            "cwd": cwd,
            "approval_mode": approval_mode,
        },
    )
    effective_approval_mode = approval_mode or _effective_approval_mode(
        _stored_settings(request)
    )
    _apply_live_session_approval_mode(session_id, effective_approval_mode)
    return redirect("session", session_id=session_id)


def _validate_settings_against_models(
    model: str, effort: str, models_data: list[Any]
) -> str | None:
    """Return an error message for an invalid (model, effort) pair, or None.

    Empty ``models_data`` (transport hiccup, pre-provider state, mock in
    tests) means we can't validate; trust the caller in that case so a
    temporary Codex outage doesn't block the user from saving.

    When ``model`` is blank the effort is checked against the provider's
    default model — the one Codex will fall back to inside ``new_session``
    — so an empty model can't quietly bypass the supported-effort check.
    """
    if not models_data:
        return None
    valid_ids = {m.id for m in models_data}
    if model and model not in valid_ids:
        return f"model {model!r} is not available"
    if effort:
        effective = (
            next((m for m in models_data if m.id == model), None)
            if model
            else next((m for m in models_data if m.is_default), models_data[0])
        )
        if effective is not None:
            supported = _supported_effort_values(effective)
            if supported and effort not in supported:
                return (
                    f"reasoning effort {effort!r} is not supported by "
                    f"model {effective.id!r}"
                )
    return None


def _posted_project(raw: str | None) -> tuple[Project | None, str | None]:
    value = (raw or "").strip()
    if not value:
        return None, None
    try:
        project_id = int(value)
    except ValueError:
        return None, "invalid project"
    if project_id < 1 or project_id > _MAX_BIGAUTOFIELD:
        return None, "invalid project"
    project = Project.objects.filter(pk=project_id).first()
    if project is None:
        return None, "invalid project"
    return project, None


def _posted_new_session_target(
    request: HttpRequest, projects: list[Project]
) -> tuple[_NewSessionTarget | None, str | None]:
    raw_project = request.POST.get("project")
    if raw_project is None:
        cwd = request.POST.get("cwd", "").strip()
        return (
            _NewSessionTarget(
                cwd,
                _project_for_cwd(cwd, projects),
                False,
                True,
            ),
            None,
        )

    value = raw_project.strip()
    if value == _BARE_REPO_PROJECT_VALUE:
        cwd = request.POST.get("cwd", "").strip()
        return _NewSessionTarget(cwd, None, True, True), None
    if not value:
        return None, "project is required"

    project, error = _posted_project(value)
    if error is not None or project is None:
        return None, error or "invalid project"
    return _NewSessionTarget(project.repo_path, project, False, False), None


def _posted_proposed_session_for_new_session(
    request: HttpRequest, target: _NewSessionTarget
) -> tuple[ProposedSession | None, str | None]:
    raw_session_id = request.POST.get("proposed_session", "").strip()
    if not raw_session_id:
        return None, None
    try:
        session_id = int(raw_session_id)
    except ValueError:
        return None, "proposed session is required"
    if session_id < 1 or session_id > _MAX_BIGAUTOFIELD:
        return None, "proposed session is required"
    _recover_stale_new_session_proposal_start_claims()
    proposed_session = (
        ProposedSession.objects.select_related(
            "project", "autonomous_goal__project", "candidate_session"
        )
        .filter(
            pk=session_id,
            inbox_kind=ProposedSession.INBOX_KIND_PROPOSAL,
            outcome_status=ProposedSession.OUTCOME_UNSET,
        )
        .first()
    )
    if proposed_session is None:
        return None, "proposed session is required"
    session_project = _project_for_proposed_session(proposed_session)
    if session_project is None:
        return None, "proposed session is required"
    if target.project is not None and target.project != session_project:
        return None, "proposed session does not match project"
    if target.project is None and target.cwd != session_project.repo_path:
        return None, "proposed session does not match project"
    return proposed_session, None


def _candidate_session_to_continue_from_proposal(
    proposed_session: ProposedSession | None,
) -> SessionMetadata | None:
    if proposed_session is None or proposed_session.candidate_session is None:
        return None
    candidate_session = proposed_session.candidate_session
    if not candidate_session.cwd:
        return None
    project = _project_for_proposed_session(proposed_session)
    if project is not None and candidate_session.cwd == project.repo_path:
        return None
    return candidate_session


def _accept_proposed_session_for_session(
    proposed_session: ProposedSession | None, session_metadata: SessionMetadata
) -> bool:
    """Record acceptance of ``proposed_session`` into ``session_metadata``.

    Returns whether this call won the one-way transition. ``False`` means the
    proposal was already resolved (e.g. a concurrent inbox reject/dismiss), so
    callers that adopt the candidate worktree must abort rather than present it.
    """
    if proposed_session is None:
        return False
    outcome_metadata = _proposal_outcome_metadata(
        proposed_session,
        {
            "accepted_by": "user",
            "resolved_by": "user",
            "accepted_session_id": session_metadata.pk,
            "accepted_thread_id": session_metadata.thread_id,
        },
    )
    # Gate the accept on the proposal still being undecided, mirroring the
    # conditional UPDATE in update_proposed_session_outcome, so exactly one
    # transition wins across both endpoints. In a stale-tab race where the inbox
    # endpoint rejects/dismisses this proposal (cleaning up the candidate
    # worktree) while new_session is accepting it, an unconditional save here
    # would overwrite the resolved status and leave accepted_session pointing at
    # a removed worktree. The loser of the race updates nothing.
    applied = ProposedSession.objects.filter(
        pk=proposed_session.pk,
        outcome_status=ProposedSession.OUTCOME_UNSET,
    ).update(
        outcome_status=ProposedSession.OUTCOME_ACCEPTED,
        accepted_session=session_metadata,
        outcome_metadata=outcome_metadata,
        updated_at=timezone.now(),
    )
    if not applied:
        return False
    proposed_session.outcome_status = ProposedSession.OUTCOME_ACCEPTED
    proposed_session.accepted_session = session_metadata
    proposed_session.outcome_metadata = outcome_metadata
    return True


def _claim_candidate_proposal_start(
    *,
    proposed_session: ProposedSession,
    candidate_session: SessionMetadata,
    cookie_updates: dict[str, str],
) -> HttpResponse | None:
    if _accept_proposed_session_for_session(proposed_session, candidate_session):
        return None
    response = redirect("inbox")
    _apply_cookie_updates(response, cookie_updates)
    return response


def _reset_candidate_proposal_start_claim(
    proposed_session: ProposedSession, candidate_session: SessionMetadata
) -> None:
    outcome_metadata = _proposal_outcome_metadata(
        proposed_session,
        {
            "accepted_by": None,
            "resolved_by": None,
            "accepted_session_id": None,
            "accepted_thread_id": None,
        },
    )
    applied = ProposedSession.objects.filter(
        pk=proposed_session.pk,
        outcome_status=ProposedSession.OUTCOME_ACCEPTED,
        accepted_session=candidate_session,
    ).update(
        outcome_status=ProposedSession.OUTCOME_UNSET,
        accepted_session=None,
        outcome_metadata=outcome_metadata,
        updated_at=timezone.now(),
    )
    if not applied:
        return
    proposed_session.outcome_status = ProposedSession.OUTCOME_UNSET
    proposed_session.accepted_session = None
    proposed_session.outcome_metadata = outcome_metadata


def _claim_new_session_proposal_start(
    *,
    proposed_session: ProposedSession,
    cookie_updates: dict[str, str],
) -> HttpResponse | None:
    claimed_at = timezone.now()
    outcome_metadata = _proposal_outcome_metadata(
        proposed_session,
        {
            "accepted_by": "user",
            "resolved_by": "user",
            "accepted_session_id": None,
            "accepted_thread_id": "",
            ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY: (
                claimed_at.isoformat()
            ),
        },
    )
    applied = ProposedSession.objects.filter(
        pk=proposed_session.pk,
        outcome_status=ProposedSession.OUTCOME_UNSET,
    ).update(
        outcome_status=ProposedSession.OUTCOME_ACCEPTED,
        accepted_session=None,
        outcome_metadata=outcome_metadata,
        updated_at=claimed_at,
    )
    if applied:
        proposed_session.outcome_status = ProposedSession.OUTCOME_ACCEPTED
        proposed_session.accepted_session = None
        proposed_session.outcome_metadata = outcome_metadata
        return None
    response = redirect("inbox")
    _apply_cookie_updates(response, cookie_updates)
    return response


def _reset_new_session_proposal_start_claim(proposed_session: ProposedSession) -> None:
    claim_filter = _new_session_proposal_start_claim_filter(proposed_session)
    if claim_filter is None:
        return
    outcome_metadata = _proposal_outcome_metadata(
        proposed_session,
        {
            "accepted_by": None,
            "resolved_by": None,
            "accepted_session_id": None,
            "accepted_thread_id": None,
            ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY: None,
        },
    )
    applied = ProposedSession.objects.filter(
        pk=proposed_session.pk,
        outcome_status=ProposedSession.OUTCOME_ACCEPTED,
        accepted_session__isnull=True,
        **claim_filter,
    ).update(
        outcome_status=ProposedSession.OUTCOME_UNSET,
        accepted_session=None,
        outcome_metadata=outcome_metadata,
        updated_at=timezone.now(),
    )
    if not applied:
        return
    proposed_session.outcome_status = ProposedSession.OUTCOME_UNSET
    proposed_session.accepted_session = None
    proposed_session.outcome_metadata = outcome_metadata


def _finish_new_session_proposal_start_claim(
    proposed_session: ProposedSession | None, session_metadata: SessionMetadata
) -> None:
    if proposed_session is None:
        return
    claim_filter = _new_session_proposal_start_claim_filter(proposed_session)
    if claim_filter is None:
        return
    outcome_metadata = _proposal_outcome_metadata(
        proposed_session,
        {
            "accepted_by": "user",
            "resolved_by": "user",
            "accepted_session_id": session_metadata.pk,
            "accepted_thread_id": session_metadata.thread_id,
            ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY: None,
        },
    )
    applied = ProposedSession.objects.filter(
        pk=proposed_session.pk,
        outcome_status=ProposedSession.OUTCOME_ACCEPTED,
        accepted_session__isnull=True,
        **claim_filter,
    ).update(
        accepted_session=session_metadata,
        outcome_metadata=outcome_metadata,
        updated_at=timezone.now(),
    )
    if not applied:
        return
    proposed_session.accepted_session = session_metadata
    proposed_session.outcome_metadata = outcome_metadata
    _stop_autonomous_goal_stack_after_proposal_resolution(proposed_session)


def _new_session_proposal_start_claim_filter(
    proposed_session: ProposedSession,
) -> dict[str, object] | None:
    metadata = (
        proposed_session.outcome_metadata
        if isinstance(proposed_session.outcome_metadata, dict)
        else {}
    )
    claim_key = ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY
    claim_value = metadata.get(claim_key)
    if claim_value is None:
        return None
    return {f"outcome_metadata__{claim_key}": claim_value}


def _recover_stale_new_session_proposal_start_claims() -> None:
    claim_key = ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY
    claim_lookup = f"outcome_metadata__{claim_key}__isnull"
    now = timezone.now()
    claimed_proposals = ProposedSession.objects.filter(
        inbox_kind=ProposedSession.INBOX_KIND_PROPOSAL,
        outcome_status=ProposedSession.OUTCOME_ACCEPTED,
        accepted_session__isnull=True,
        **{claim_lookup: False},
    ).only("pk", "outcome_metadata")
    for proposed_session in claimed_proposals:
        claim_filter = _new_session_proposal_start_claim_filter(proposed_session)
        if claim_filter is None:
            continue
        if ProposedSession.accepted_session_start_claim_is_active(
            proposed_session.outcome_metadata, now=now
        ):
            continue
        outcome_metadata = _proposal_outcome_metadata(
            proposed_session,
            {
                "accepted_by": None,
                "resolved_by": None,
                "accepted_session_id": None,
                "accepted_thread_id": None,
                claim_key: None,
            },
        )
        ProposedSession.objects.filter(
            pk=proposed_session.pk,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            accepted_session__isnull=True,
            **claim_filter,
        ).update(
            outcome_status=ProposedSession.OUTCOME_UNSET,
            accepted_session=None,
            outcome_metadata=outcome_metadata,
            updated_at=now,
        )


def _proposed_session_thread_title(proposed_session: ProposedSession) -> str:
    return proposed_session.title.strip()[:_NAME_MAX_LEN].rstrip()


def _apply_proposed_session_title_to_session_metadata(
    proposed_session: ProposedSession,
    session_metadata: SessionMetadata,
) -> None:
    title = _proposed_session_thread_title(proposed_session)
    if not title:
        return
    session_index.update_cached_name(session_metadata.thread_id, title)


def _rename_codex_thread_from_proposal(
    *,
    proposed_session: ProposedSession,
    session_metadata: SessionMetadata,
    settings: SettingsValues,
) -> bool:
    title = _proposed_session_thread_title(proposed_session)
    if not title:
        return False
    try:
        with codex_pool.borrow_codex(
            Codex, enable_memories=settings.enable_memories
        ) as codex:
            codex._client.thread_set_name(session_metadata.thread_id, title)
    except Exception:
        logger.exception(
            "failed to rename accepted proposed session thread %s",
            session_metadata.thread_id,
        )
        return False
    _apply_proposed_session_title_to_session_metadata(
        proposed_session, session_metadata
    )
    return True


def _proposal_outcome_metadata(
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


def _posted_auto_pr_override(raw: str | None, *, default: bool) -> tuple[bool, str | None]:
    if raw is None:
        return default, None
    value = raw.strip().lower()
    if value in {"", "false"}:
        return False, None
    if value == "true":
        return True, None
    return False, "invalid auto-PR setting"


def _posted_auto_qa_override(raw: str | None, *, default: bool) -> tuple[bool, str | None]:
    if raw is None:
        return default, None
    value = raw.strip().lower()
    if value in {"", "false"}:
        return False, None
    if value == "true":
        return True, None
    return False, "invalid auto-QA setting"


def _posted_use_worktree_override(
    raw: str | None, *, default: bool
) -> tuple[bool, str | None]:
    if raw is None:
        return default, None
    value = raw.strip().lower()
    if value in {"", "false"}:
        return False, None
    if value == "true":
        return True, None
    return False, "invalid worktree setting"


def _posted_web_search_override(
    raw: str | None, *, default: str
) -> tuple[str, str | None]:
    if raw is None:
        return default, None
    value = raw.strip()
    if not value:
        return default, None
    if value in _VALID_WEB_SEARCH_MODES:
        return value, None
    return "", "invalid web search setting"


def _posted_new_session_coding_agent(raw: str | None) -> tuple[str, str | None]:
    value = (raw or "").strip()
    if not value:
        return "", None
    if value in coding_agents.VALID_CODING_AGENTS:
        return value, None
    return "", "invalid coding agent"


@require_http_methods(["POST"])
def set_session_name(request: HttpRequest, session_id: str) -> HttpResponse:
    name = request.POST.get("name", "").strip()
    if not name:
        return HttpResponseBadRequest("name is required")
    if len(name) > _NAME_MAX_LEN:
        return HttpResponseBadRequest("name is too long")
    settings = _stored_settings(request)
    with codex_pool.borrow_codex(
        Codex, enable_memories=settings.enable_memories
    ) as codex:
        codex._client.thread_set_name(session_id, name)
    session_index.update_cached_name(session_id, name)
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return HttpResponse(status=204)
    if request.POST.get("next", "").strip() == "index":
        return redirect("index")
    return redirect("session", session_id=session_id)


@require_http_methods(["POST"])
def set_session_archived(request: HttpRequest, session_id: str) -> HttpResponse:
    archived = request.POST.get("archived", "").strip()
    if archived not in {"true", "false"}:
        return HttpResponseBadRequest("archived must be true or false")
    settings = _stored_settings(request)
    with codex_pool.borrow_codex(
        Codex, enable_memories=settings.enable_memories
    ) as codex:
        if archived == "true":
            codex.thread_archive(session_id)
        else:
            codex.thread_unarchive(session_id)
    if archived == "true":
        demo.cleanup_demo_for_session(session_id)
    session_index.update_cached_archived(session_id, archived=archived == "true")
    # Codex moves this thread's rollout in/out of ``archived_sessions/`` when
    # the archive bit flips, which invalidates *this* thread's cached usage
    # row. Other threads' caches still match their rollouts, so leave them
    # alone — a blanket wipe forces /profile and /usage to re-parse every
    # archived rollout file the next time they render.
    ArchivedSessionTokenUsage.objects.filter(thread_id=session_id).delete()
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return HttpResponse(status=204)
    if request.POST.get("next", "").strip() == "index":
        return redirect("index")
    if archived == "true":
        return redirect("index")
    return redirect("session", session_id=session_id)


def _mark_workflow_failed(workflow: SystemWorkflow) -> None:
    SystemWorkflow.objects.filter(pk=workflow.pk).update(
        status=SystemWorkflow.STATUS_FAILED,
        updated_at=timezone.now(),
    )


@require_http_methods(["POST"])
def start_session_demo(request: HttpRequest, session_id: str) -> HttpResponse:
    if system_agents.active_workflow_for_thread(session_id) is not None:
        return HttpResponseBadRequest("PR workflow is running for this session")
    active_instance = codex_pool.latest_active_for_thread(session_id)
    if active_instance is not None:
        if active_instance.agent_kind == demo.DEMO_AGENT_KIND:
            return HttpResponseBadRequest("demo setup is already running")
        return HttpResponseBadRequest("Codex is already working for this session")
    try:
        demo.demo_runtime()
    except demo.DemoError as exc:
        return HttpResponse(str(exc), status=500, content_type="text/plain")
    if SystemWorkflow.objects.filter(
        kind=demo.DEMO_WORKFLOW_KIND,
        main_thread_id=session_id,
        status=SystemWorkflow.STATUS_RUNNING,
    ).exists():
        return HttpResponseBadRequest("demo setup workflow is already running")
    settings = _stored_settings(request)
    resumed = codex_pool.run_borrowed_op_with_retry(
        Codex,
        lambda codex: codex._client.thread_resume(session_id),
        enable_memories=settings.enable_memories,
    )
    thread = resumed.thread
    cwd = _thread_cwd(thread)
    if not cwd:
        return HttpResponseBadRequest("thread has no cwd")
    if cwd not in _allowed_session_cwds():
        return HttpResponseBadRequest("thread cwd is not an allowed repository")
    sandbox_policy = _effective_sandbox_policy_for_cwd(settings, cwd)
    try:
        with transaction.atomic():
            workflow = SystemWorkflow.objects.create(
                kind=demo.DEMO_WORKFLOW_KIND,
                main_thread_id=session_id,
                cwd=cwd,
                status=SystemWorkflow.STATUS_RUNNING,
                step="demo_running",
                state={},
            )
    except IntegrityError:
        return HttpResponseBadRequest("demo setup workflow is already running")
    try:
        session_demo = demo.request_demo_start(session_id)
    except demo.DemoAlreadyRunningError as exc:
        _mark_workflow_failed(workflow)
        return HttpResponseBadRequest(str(exc))
    except demo.DemoError as exc:
        _mark_workflow_failed(workflow)
        return HttpResponse(str(exc), status=500, content_type="text/plain")
    except Exception:
        _mark_workflow_failed(workflow)
        raise
    try:
        workflow.state = {"session_demo_id": session_demo.pk}
        workflow.save(update_fields=["state", "updated_at"])
        prompt = demo.start_demo_prompt_for(
            request=request,
            session_id=session_id,
            cwd=cwd,
            demo=session_demo,
        )
        instance = codex_pool.spawn_turn(
            thread_id=session_id,
            cwd=cwd,
            prompt=prompt,
            sandbox_policy=sandbox_policy or None,
            approval_mode=_effective_approval_mode_for_session(settings, session_id),
            web_search_mode=_valid_web_search_mode_or_default(
                settings.web_search_mode
            )
            or None,
            enable_memories=settings.enable_memories,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=demo.DEMO_AGENT_KIND,
            display_author=demo.DEMO_DISPLAY_AUTHOR,
            user_message_index=None,
        )
        SystemAgentRun.objects.get_or_create(
            instance=instance,
            defaults={
                "workflow": workflow,
                "agent_kind": demo.DEMO_AGENT_KIND,
                "thread_id": instance.thread_id,
                "status": SystemAgentRun.STATUS_RUNNING,
                "input": {"cwd": cwd, "session_id": session_id},
            },
        )
    except Exception:
        _mark_workflow_failed(workflow)
        demo.cleanup_demo_for_session(session_id)
        raise
    return redirect("session", session_id=session_id)


@csrf_exempt
@require_http_methods(["POST"])
def register_session_demo(request: HttpRequest, session_id: str) -> HttpResponse:
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return HttpResponseBadRequest("invalid JSON")
    if not isinstance(payload, dict):
        return HttpResponseBadRequest("invalid JSON")
    try:
        session_demo = demo.register_demo_container(session_id, payload)
    except demo.DemoError as exc:
        return HttpResponse(str(exc), status=400, content_type="text/plain")
    return demo.registration_response(session_demo)


@csrf_exempt
def session_demo_proxy_root(request: HttpRequest, session_id: str) -> HttpResponse:
    return session_demo_proxy(request, session_id, "")


@csrf_exempt
def session_demo_proxy(
    request: HttpRequest, session_id: str, path: str
) -> HttpResponse:
    prefix = reverse("session_demo_proxy_root", kwargs={"session_id": session_id})
    return demo.proxy_demo_request(request, session_id, path, path_prefix=prefix)


@_limit_input_image_uploads
@require_http_methods(["POST"])
def send_message(request: HttpRequest, session_id: str) -> HttpResponse:
    intent = _message_intent(request)
    pr_activation = _is_pr_activation(request)
    fix_pr_activation = _is_fix_pr_activation(request)
    qa_activation = _is_qa_activation(request)
    qa_workflow_activation = pr_activation or qa_activation or fix_pr_activation
    prompt = intent.prompt
    plan_mode = intent.plan_mode
    has_input_images = _has_input_image_uploads(request)
    if not prompt and not has_input_images:
        return HttpResponseBadRequest("prompt is required")
    collaboration_mode = request.POST.get("collaboration_mode", "").strip().lower()
    plan_action = request.POST.get("plan_action", "").strip().lower()
    if plan_action not in _VALID_PLAN_ACTIONS:
        return HttpResponseBadRequest("invalid plan action")
    if plan_action == _PLAN_ACTION_APPROVE:
        collaboration_mode = _DEFAULT_COLLABORATION_MODE
        plan_mode = False
    elif plan_action == _PLAN_ACTION_REVISE:
        collaboration_mode = ""
        plan_mode = True
    if collaboration_mode and collaboration_mode != _DEFAULT_COLLABORATION_MODE:
        return HttpResponseBadRequest("invalid collaboration mode")
    if collaboration_mode and plan_mode and intent.explicit_plan_mode:
        return HttpResponseBadRequest("collaboration mode conflicts with plan mode")
    if qa_workflow_activation and collaboration_mode:
        return HttpResponseBadRequest("PR workflow conflicts with collaboration mode")
    if collaboration_mode:
        plan_mode = False
    if qa_workflow_activation:
        plan_mode = False
    active_system_workflow = system_agents.active_workflow_for_thread(session_id)
    if qa_workflow_activation and has_input_images:
        return HttpResponseBadRequest(
            "image attachments are not supported for PR workflow requests"
        )
    settings = _stored_settings(request)
    raw_active = request.POST.get("active_instance", "").strip()
    active_instance = None
    instance_id: int | None = None
    if raw_active:
        if qa_workflow_activation:
            return HttpResponseBadRequest("PR workflow requires an idle session")
        instance_id, error = _parse_instance_id(raw_active)
        if error is not None or instance_id is None:
            return HttpResponseBadRequest(error or "invalid instance id")
    else:
        active_instance = codex_pool.latest_active_for_thread(session_id)
        if active_instance is not None and qa_workflow_activation:
            return HttpResponseBadRequest("PR workflow requires an idle session")
    if active_system_workflow is not None and qa_workflow_activation:
        return redirect("session", session_id=session_id)
    workflow_active_instance = active_instance
    if active_system_workflow is not None and raw_active and instance_id is not None:
        workflow_active_instance = CodexInstance.objects.filter(pk=instance_id).first()
    if active_system_workflow is not None:
        if has_input_images:
            return HttpResponseBadRequest(
                "image attachments are not supported while QA workflow is running"
            )
        workflow_accepts_active_steering = _workflow_accepts_active_turn_steering(
            active_system_workflow, workflow_active_instance
        )
        workflow_accepts_qa_pause = (
            active_instance is None
            and not raw_active
            and _workflow_accepts_qa_pause_steering(active_system_workflow)
        )
        if not (workflow_accepts_active_steering or workflow_accepts_qa_pause):
            return HttpResponseBadRequest("PR workflow is running for this session")

    input_image_paths, input_image_error = _save_posted_input_images(request)
    if input_image_error is not None:
        return HttpResponseBadRequest(input_image_error)

    input_images_owned = False
    steer_image_paths: list[str] = []
    session_unarchived_for_turn = False

    def restore_archived_session_for_rejected_turn() -> None:
        if session_unarchived_for_turn:
            _restore_archived_session_for_rejected_turn(session_id, settings)

    try:
        if raw_active:
            assert instance_id is not None
            steer_kwargs: dict[str, Any] = {
                "expected_thread_id": session_id,
                "prompt": prompt,
            }
            if input_image_paths:
                steer_image_paths = _duplicate_saved_input_images(input_image_paths)
                steer_kwargs["input_image_paths"] = steer_image_paths
            steered = codex_pool.steer_instance(
                instance_id,
                **steer_kwargs,
            )
            if steered is not None:
                _cleanup_saved_input_images(input_image_paths)
                steer_image_paths = []
                input_images_owned = True
                return redirect("session", session_id=session_id)
            _cleanup_saved_input_images(steer_image_paths)
            steer_image_paths = []
        elif active_instance is not None:
            active_steer_kwargs: dict[str, Any] = {
                "expected_thread_id": session_id,
                "prompt": prompt,
            }
            if input_image_paths:
                steer_image_paths = _duplicate_saved_input_images(input_image_paths)
                active_steer_kwargs["input_image_paths"] = steer_image_paths
            steered = codex_pool.steer_instance(
                active_instance.pk,
                **active_steer_kwargs,
            )
            if steered is not None:
                _cleanup_saved_input_images(input_image_paths)
                steer_image_paths = []
                input_images_owned = True
                return redirect("session", session_id=session_id)
            _cleanup_saved_input_images(steer_image_paths)
            steer_image_paths = []
        if active_system_workflow is not None:
            if (
                active_instance is None
                and not raw_active
                and _workflow_accepts_qa_pause_steering(active_system_workflow)
            ):
                started = system_agents.start_user_steering_turn(
                    active_system_workflow,
                    prompt=prompt,
                )
                if started is not None:
                    return redirect("session", session_id=session_id)
                _cleanup_saved_input_images(input_image_paths)
                return HttpResponseBadRequest("QA workflow could not be paused")
            _cleanup_saved_input_images(input_image_paths)
            return HttpResponseBadRequest("PR workflow is running for this session")
        # If steering is unavailable or races a terminal worker, preserve the
        # submitted prompt by treating it as an ordinary follow-up turn.
        # ``raw_active`` posts still do not retarget a different active worker.
        # ``Thread.cwd`` is an ``AbsolutePathBuf`` pydantic RootModel, so unwrap
        # ``.root`` to get the underlying string the worker subprocess expects;
        # also accept a plain str so a future SDK schema change does not break us.
        # Resolve the thread's state (entries, plan-mode, cwd, last model) to
        # decide how to spawn the turn. Prefer reading SessionMetadata + the
        # rollout file from disk: the detached worker resumes the thread itself
        # moments later, so a live ``thread_resume`` here only duplicates that
        # rollout read (and its lazy state-DB migration) on the request path.
        # Fall back to a live resume for active/workflow/uncached-cwd threads.
        metadata = _session_detail_metadata(session_id)

        def record_session_unarchived_for_accepted_turn() -> None:
            if not session_unarchived_for_turn:
                return
            _record_session_unarchived(session_id)
            if metadata is not None:
                metadata.codex_archived = False
                metadata.codex_archived_at = None
                metadata.codex_path = ""

        should_unarchive_for_turn = _metadata_indicates_archived(metadata)
        force_live_resume = _metadata_rollout_path_indicates_archived(metadata)
        if should_unarchive_for_turn:
            if _metadata_cwd_is_disallowed(metadata):
                _cleanup_saved_input_images(input_image_paths)
                return HttpResponseBadRequest("thread cwd is not an allowed repository")
            _unarchive_session_for_turn(session_id, settings)
            session_unarchived_for_turn = True
            force_live_resume = True
        metadata_resume = (
            None
            if force_live_resume
            else _metadata_resume_for_inactive_session(
                session_id,
                metadata,
                active_instance=active_instance,
                active_system_workflow=active_system_workflow,
                require_system_agent_thread=False,
            )
        )
        resumed: Any
        thread: Any
        if metadata_resume is not None and _thread_cwd(metadata_resume.thread):
            used_disk_resume = True
            resumed = metadata_resume
            thread = metadata_resume.thread
            thread_entries = list(metadata_resume.entries)
            models_data = caches._cached_models_for_session_detail(
                enable_memories=settings.enable_memories
            )
        else:
            used_disk_resume = False
            with codex_pool.borrow_codex(
                Codex, enable_memories=settings.enable_memories
            ) as codex:
                try:
                    resumed = codex._client.thread_resume(session_id)
                except InvalidRequestError as exc:
                    if not _thread_resume_archived_error(exc):
                        raise
                    if _metadata_cwd_is_disallowed(metadata):
                        _cleanup_saved_input_images(input_image_paths)
                        return HttpResponseBadRequest(
                            "thread cwd is not an allowed repository"
                        )
                    _unarchive_session_for_turn(session_id, settings, codex=codex)
                    session_unarchived_for_turn = True
                    resumed = codex._client.thread_resume(session_id)
                thread = resumed.thread
                thread_entries = list(_entries_for(thread))
                models_data = _models_for_plan_mode_fallback(codex)
        thread_plan_state = _thread_plan_mode_state(
            session_id,
            thread,
            thread_entries,
            active_instance=active_instance,
        )
        thread_awaits_plan_approval = thread_plan_state.awaiting_approval
        if (
            not collaboration_mode
            and intent.allow_pending_plan_default
            and thread_plan_state.active
            and not thread_awaits_plan_approval
            and intent.explicit_plan_mode
            and not plan_mode
        ):
            collaboration_mode = _DEFAULT_COLLABORATION_MODE
        elif (
            not collaboration_mode
            and intent.allow_pending_plan_default
            and thread_plan_state.active
            and (thread_awaits_plan_approval or not intent.explicit_plan_mode)
        ):
            plan_mode = True
        elif (
            not collaboration_mode
            and intent.allow_pending_plan_default
            and not intent.explicit_plan_mode
        ):
            plan_mode = False
        if (
            thread_awaits_plan_approval
            and not collaboration_mode
            and intent.allow_pending_plan_default
            and not intent.explicit_plan_mode
            and prompt == _PLAN_APPROVAL_PROMPT
        ):
            collaboration_mode = _DEFAULT_COLLABORATION_MODE
            plan_mode = False
        # A disk resume carries the thread's model/effort only when Hitch
        # recorded a prior CodexInstance for it. For model-sensitive turns
        # (plan, default collaboration, QA/PR) on threads Hitch never tracked --
        # imported or CLI-created -- recover the thread's actual model with a
        # one-off live resume, matching the old path that used the resumed model
        # (and the live models catalog) in preference to the request's cookie.
        # This also covers a cold (empty) models cache. Plain follow-ups never
        # reach this and keep the disk fast path.
        if (
            used_disk_resume
            and (
                plan_mode
                or collaboration_mode == _DEFAULT_COLLABORATION_MODE
                or qa_workflow_activation
            )
            and not string_value(getattr(resumed, "model", None))
        ):
            with codex_pool.borrow_codex(
                Codex, enable_memories=settings.enable_memories
            ) as codex:
                resumed = codex._client.thread_resume(session_id)
                models_data = _models_for_plan_mode_fallback(codex)
        collaboration_model = (
            _plan_mode_model_from_models(resumed, settings, models_data)
            if plan_mode or collaboration_mode == _DEFAULT_COLLABORATION_MODE
            else None
        )
        if plan_mode and not collaboration_model and not intent.explicit_plan_mode:
            plan_mode = False
        cwd = _thread_cwd(thread)
        if not cwd:
            restore_archived_session_for_rejected_turn()
            _cleanup_saved_input_images(input_image_paths)
            return HttpResponseBadRequest("thread has no cwd")
        # The session list surfaces every thread the app-server knows about, not
        # just those created via ``new_session``, so the resumed ``cwd`` is not
        # automatically inside the discover_repos() allowlist. Re-validate before
        # spawning so a follow-up cannot run a worker in an unintended directory.
        if not _is_allowed_session_cwd(cwd):
            restore_archived_session_for_rejected_turn()
            _cleanup_saved_input_images(input_image_paths)
            return HttpResponseBadRequest("thread cwd is not an allowed repository")
        # Sandbox policy and approval mode are applied per-turn rather than
        # persisted on the thread, so follow-up messages have to re-forward
        # the cookies or every turn after the first silently reverts to Codex
        # defaults — which breaks multi-turn workflows that depend on
        # elevated permissions or stricter escalation handling.
        sandbox_policy = _effective_sandbox_policy_for_cwd(settings, cwd)
        approval_mode = _effective_approval_mode_for_session(
            settings, session_id, metadata
        )
        previous_instance = codex_pool.latest_for_thread(session_id)
        session_project = None
        if previous_instance is None:
            # ``metadata`` was already fetched (with its project) above.
            if metadata is not None and (
                metadata.project_id is not None or metadata.project_cleared
            ):
                session_project = metadata.project
            else:
                session_project = _project_for_cwd(cwd, list(Project.objects.all()))
        developer_instructions = (
            previous_instance.developer_instructions
            if previous_instance is not None
            else _developer_instructions_for_project(settings, session_project)
        )
        configured_web_search_mode = _valid_web_search_mode_or_default(
            settings.web_search_mode
        )
        previous_web_search_mode = (
            _valid_web_search_mode_or_default(previous_instance.web_search_mode)
            if previous_instance is not None
            else ""
        )
        web_search_mode = (
            previous_web_search_mode
            if qa_workflow_activation and not configured_web_search_mode
            else configured_web_search_mode
        )
        should_forward_web_search_mode = bool(web_search_mode) or bool(
            previous_web_search_mode
        )
        base_instructions = _base_instructions_for_settings(
            settings, explicit_default=True
        )
        auto_pr_enabled = _auto_pr_enabled_for_session(session_id)
        auto_qa_enabled = (
            False if auto_pr_enabled else _auto_qa_enabled_for_session(session_id)
        )
        auto_merge_to_local_branch, auto_merge_branch = (
            _auto_merge_to_local_branch_for_session(session_id)
        )
        # ``auto_merge_branch`` is the gated value used by the auto-review
        # spawn path (only forwarded when auto_qa is enabled). The manual
        # ``/qa`` and ``/pr`` activations should honor the session-configured
        # merge target regardless of the auto_qa flag, since the user is
        # explicitly opting into the QA workflow at that moment.
        session_auto_merge_branch = auto_merge_branch
        if not auto_qa_enabled:
            auto_merge_to_local_branch = False
            auto_merge_branch = ""
        if qa_workflow_activation:
            workflow_model = (
                string_value(getattr(resumed, "model", None)) or settings.model
            )
            workflow_reasoning_effort = (
                string_value(getattr(resumed, "reasoning_effort", None))
                or settings.reasoning_effort
            )
            workflow_kwargs: dict[str, Any] = {
                "main_thread_id": session_id,
                "cwd": cwd,
                "sandbox_policy": sandbox_policy or None,
                "approval_mode": approval_mode,
                "model": workflow_model or None,
                "reasoning_effort": workflow_reasoning_effort or None,
                "developer_instructions": developer_instructions or None,
                "enable_memories": settings.enable_memories,
                "initial_user_message_index": _count_user_entries(thread_entries),
            }
            if should_forward_web_search_mode:
                workflow_kwargs["web_search_mode"] = web_search_mode
            if base_instructions:
                workflow_kwargs["base_instructions"] = base_instructions
            if fix_pr_activation:
                pr_url = _fix_pr_url_for_thread(session_id, thread)
                if not pr_url:
                    restore_archived_session_for_rejected_turn()
                    _cleanup_saved_input_images(input_image_paths)
                    return HttpResponseBadRequest(
                        "fix-pr requires an opened PR for this session"
                    )
                system_agents.start_pr_monitor_workflow(
                    pr_url=pr_url,
                    **workflow_kwargs,
                )
                record_session_unarchived_for_accepted_turn()
                return redirect("session", session_id=session_id)
            if qa_activation:
                workflow_kwargs["open_pr_on_lgtm"] = False
            # Honor the session's auto-merge target the same way auto_qa /
            # auto_pr workflows do, so manual /qa and /pr respect the user's
            # "merge into a local branch instead of opening a PR" setting
            # rather than silently dropping it.
            if session_auto_merge_branch:
                workflow_kwargs["auto_merge_branch"] = session_auto_merge_branch
            system_agents.start_pr_qa_workflow(**workflow_kwargs)
            record_session_unarchived_for_accepted_turn()
            return redirect("session", session_id=session_id)
        spawn_kwargs: dict[str, Any] = {
            "thread_id": session_id,
            "cwd": cwd,
            "prompt": prompt,
            "sandbox_policy": sandbox_policy or None,
            "approval_mode": approval_mode,
        }
        if input_image_paths:
            spawn_kwargs["input_image_paths"] = input_image_paths
        if should_forward_web_search_mode:
            spawn_kwargs["web_search_mode"] = web_search_mode
        if base_instructions:
            spawn_kwargs["base_instructions"] = base_instructions
        if previous_instance is None and developer_instructions:
            spawn_kwargs["developer_instructions"] = developer_instructions
        if settings.enable_memories:
            spawn_kwargs["enable_memories"] = True
        if auto_pr_enabled or auto_qa_enabled:
            auto_review_model = (
                string_value(getattr(resumed, "model", None)) or settings.model
            )
            auto_review_reasoning_effort = (
                string_value(getattr(resumed, "reasoning_effort", None))
                or settings.reasoning_effort
            )
            if auto_pr_enabled:
                spawn_kwargs["auto_pr_enabled"] = True
            if auto_qa_enabled:
                spawn_kwargs["auto_qa_enabled"] = True
            spawn_kwargs["user_message_index"] = _count_user_entries(thread_entries)
            spawn_kwargs["stored_model"] = auto_review_model or None
            spawn_kwargs["stored_reasoning_effort"] = auto_review_reasoning_effort or None
            if auto_merge_to_local_branch:
                spawn_kwargs["auto_merge_to_local_branch"] = True
                spawn_kwargs["auto_merge_branch"] = auto_merge_branch
        if plan_mode:
            if not collaboration_model:
                restore_archived_session_for_rejected_turn()
                _cleanup_saved_input_images(input_image_paths)
                return HttpResponseBadRequest("plan mode requires a model")
            spawn_kwargs["model"] = collaboration_model
            spawn_kwargs["plan_mode"] = True
        elif collaboration_mode == _DEFAULT_COLLABORATION_MODE:
            if not collaboration_model:
                restore_archived_session_for_rejected_turn()
                _cleanup_saved_input_images(input_image_paths)
                return HttpResponseBadRequest(
                    "default collaboration mode requires a model"
                )
            spawn_kwargs["model"] = collaboration_model
            spawn_kwargs["collaboration_mode"] = collaboration_mode
        # The should-run classifier runs inside the workflow on a background
        # thread, so sending a message never blocks on it.
        if (
            settings.spec_critic_enabled
            and not input_image_paths
            and not plan_mode
            and not collaboration_mode
        ):
            workflow_model = (
                string_value(getattr(resumed, "model", None)) or settings.model
            )
            workflow_reasoning_effort = (
                string_value(getattr(resumed, "reasoning_effort", None))
                or settings.reasoning_effort
            )
            spec_workflow_kwargs: dict[str, Any] = {
                "main_thread_id": session_id,
                "cwd": cwd,
                "prompt": prompt,
                "sandbox_policy": sandbox_policy or None,
                "approval_mode": approval_mode,
                "model": workflow_model or None,
                "reasoning_effort": workflow_reasoning_effort or None,
                "developer_instructions": developer_instructions or None,
                "enable_memories": settings.enable_memories,
                "initial_user_message_index": _count_user_entries(thread_entries),
                "auto_pr_enabled": auto_pr_enabled,
                "auto_qa_enabled": auto_qa_enabled,
            }
            if auto_merge_to_local_branch:
                spec_workflow_kwargs["auto_merge_to_local_branch"] = True
                spec_workflow_kwargs["auto_merge_branch"] = auto_merge_branch
            if base_instructions:
                spec_workflow_kwargs["base_instructions"] = base_instructions
            if should_forward_web_search_mode:
                spec_workflow_kwargs["web_search_mode"] = web_search_mode
            system_agents.start_spec_critic_workflow(**spec_workflow_kwargs)
            record_session_unarchived_for_accepted_turn()
            return redirect("session", session_id=session_id)
        codex_pool.spawn_turn(**spawn_kwargs)
        record_session_unarchived_for_accepted_turn()
        input_images_owned = True
        return redirect("session", session_id=session_id)
    except codex_pool.InputAttachmentLimitExceededError as exc:
        restore_archived_session_for_rejected_turn()
        _cleanup_saved_input_images(steer_image_paths)
        _cleanup_saved_input_images(input_image_paths)
        return HttpResponseBadRequest(str(exc))
    except Exception:
        restore_archived_session_for_rejected_turn()
        _cleanup_saved_input_images(steer_image_paths)
        if not input_images_owned:
            _cleanup_saved_input_images(input_image_paths)
        raise

def _thread_awaits_plan_approval(thread: Any) -> bool:
    return _entries_await_plan_approval(list(_entries_for(thread)))


def _posted_input_image_uploads(request: HttpRequest) -> list[UploadedFile]:
    return [
        upload
        for upload in request.FILES.getlist(_INPUT_IMAGE_FIELD)
        if isinstance(upload, UploadedFile)
    ]


def _has_input_image_uploads(request: HttpRequest) -> bool:
    return bool(_posted_input_image_uploads(request))


def _save_posted_input_images(request: HttpRequest) -> tuple[list[str], str | None]:
    uploads = _posted_input_image_uploads(request)
    if not uploads:
        return [], None
    if len(uploads) > _INPUT_IMAGE_MAX_COUNT:
        return [], f"at most {_INPUT_IMAGE_MAX_COUNT} image attachments are allowed"

    extensions: list[str] = []
    for upload in uploads:
        extension, error = _uploaded_input_image_extension(upload)
        if error is not None or extension is None:
            return [], error or "image attachment is invalid"
        extensions.append(extension)

    saved_paths: list[str] = []
    current_path: Path | None = None
    try:
        attachments_dir = codex_pool.input_attachments_dir()
        _ensure_private_dir(attachments_dir)
        target_dir = attachments_dir / uuid.uuid4().hex
        target_dir.mkdir(mode=0o700)
        target_dir.chmod(0o700)
        for index, (upload, extension) in enumerate(
            zip(uploads, extensions, strict=True), start=1
        ):
            upload.seek(0)
            path = target_dir / f"{index}{extension}"
            current_path = path
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as fh:
                for chunk in upload.chunks():
                    fh.write(chunk)
            path.chmod(0o600)
            saved_paths.append(str(path))
            current_path = None
    except Exception as exc:  # noqa: BLE001 - cleanup partial writes before re-raising
        cleanup_paths = [*saved_paths]
        if current_path is not None:
            cleanup_paths.append(str(current_path))
        _cleanup_saved_input_images(cleanup_paths)
        if not isinstance(exc, OSError):
            raise
        logger.exception("failed to save uploaded image attachment")
        return [], "failed to save image attachment"
    return saved_paths, None


def _ensure_private_dir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)


def _uploaded_input_image_extension(
    upload: UploadedFile,
) -> tuple[str | None, str | None]:
    size = upload.size
    if size is None or size <= 0:
        return None, "image attachment is empty"
    if size > _INPUT_IMAGE_MAX_BYTES:
        return None, "image attachment is too large"
    try:
        upload.seek(0)
        header = upload.read(16)
        upload.seek(0)
    except OSError:
        logger.exception("failed to read uploaded image attachment")
        return None, "failed to read image attachment"
    extension = _input_image_extension_from_header(header)
    if extension is None:
        return None, "image attachment must be PNG, JPEG, GIF, or WebP"
    return extension, None


def _input_image_extension_from_header(header: bytes) -> str | None:
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if header.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if header.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if len(header) >= 12 and header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return ".webp"
    return None


def _cleanup_saved_input_images(paths: Iterable[str]) -> None:
    for path in paths:
        image_path = Path(path)
        try:
            image_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("failed to clean up uploaded image attachment %s", path)
        else:
            with contextlib.suppress(OSError):
                image_path.parent.rmdir()


def _duplicate_saved_input_images(paths: Iterable[str]) -> list[str]:
    source_paths = [Path(path) for path in paths if path]
    if not source_paths:
        return []
    saved_paths: list[str] = []
    current_path: Path | None = None
    try:
        attachments_dir = codex_pool.input_attachments_dir()
        _ensure_private_dir(attachments_dir)
        target_dir = attachments_dir / uuid.uuid4().hex
        target_dir.mkdir(mode=0o700)
        target_dir.chmod(0o700)
        for index, source_path in enumerate(source_paths, start=1):
            target_path = target_dir / f"{index}{source_path.suffix}"
            current_path = target_path
            fd = -1
            try:
                with source_path.open("rb") as source:
                    fd = os.open(
                        target_path,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                    )
                    with os.fdopen(fd, "wb") as target:
                        fd = -1
                        while chunk := source.read(1024 * 1024):
                            target.write(chunk)
            finally:
                if fd != -1:
                    os.close(fd)
            target_path.chmod(0o600)
            saved_paths.append(str(target_path))
            current_path = None
    except Exception:
        cleanup_paths = [*saved_paths]
        if current_path is not None:
            cleanup_paths.append(str(current_path))
        _cleanup_saved_input_images(cleanup_paths)
        raise
    return saved_paths


def _message_intent(request: HttpRequest) -> _MessageIntent:
    prompt = request.POST.get("prompt", "").strip()
    plan_mode = request.POST.get("plan_mode", "").strip().lower() == "true"
    default_plan_mode_raw = request.POST.get("default_plan_mode")
    default_plan_mode = (
        default_plan_mode_raw.strip().lower() == "true"
        if default_plan_mode_raw is not None
        else False
    )
    default_plan_mode_posted = default_plan_mode_raw is not None
    plan_mode_changed = (
        plan_mode != default_plan_mode if default_plan_mode_posted else plan_mode
    )
    explicit_plan_mode = (
        request.POST.get("plan_mode_explicit", "").strip().lower() == "true"
        or plan_mode_changed
    )
    parts = prompt.split(maxsplit=1)
    if not parts:
        return _MessageIntent(prompt, plan_mode, True, explicit_plan_mode)
    command = parts[0].lower()
    if command == _PLAN_SLASH_COMMAND:
        return _MessageIntent(
            parts[1].strip() if len(parts) > 1 else "",
            True,
            True,
            True,
        )
    if command == _PR_SLASH_COMMAND:
        return _MessageIntent(_PR_SLASH_PROMPT, False, False, False)
    if command == _FIX_PR_SLASH_COMMAND:
        return _MessageIntent(_FIX_PR_SLASH_COMMAND, False, False, False)
    if command == _QA_SLASH_COMMAND:
        return _MessageIntent(_QA_SLASH_PROMPT, False, False, False)
    if not plan_mode and prompt in _PR_PROMPT_ALIASES:
        return _MessageIntent(prompt, False, False, False)
    if not plan_mode and prompt == _QA_SLASH_PROMPT:
        return _MessageIntent(prompt, False, False, False)
    return _MessageIntent(prompt, plan_mode, True, explicit_plan_mode)


def _is_pr_activation(request: HttpRequest) -> bool:
    prompt = request.POST.get("prompt", "").strip()
    parts = prompt.split(maxsplit=1)
    return (
        bool(parts and parts[0].lower() == _PR_SLASH_COMMAND)
        or prompt in _PR_PROMPT_ALIASES
    )


def _is_fix_pr_activation(request: HttpRequest) -> bool:
    prompt = request.POST.get("prompt", "").strip()
    parts = prompt.split(maxsplit=1)
    return bool(parts and parts[0].lower() == _FIX_PR_SLASH_COMMAND)


def _is_qa_activation(request: HttpRequest) -> bool:
    prompt = request.POST.get("prompt", "").strip()
    parts = prompt.split(maxsplit=1)
    return (
        bool(parts and parts[0].lower() == _QA_SLASH_COMMAND)
        or prompt == _QA_SLASH_PROMPT
    )


def _plan_mode_model_from_models(
    resumed: Any, settings: SettingsValues, models_data: list[Any]
) -> str | None:
    resumed_model = getattr(resumed, "model", "")
    if isinstance(resumed_model, str) and resumed_model.strip():
        return resumed_model.strip()
    if settings.model:
        valid_ids = {m.id for m in models_data}
        if not valid_ids or settings.model in valid_ids:
            return settings.model
    default_model = next((m for m in models_data if m.is_default), None)
    if default_model is None and models_data:
        default_model = models_data[0]
    model_id = getattr(default_model, "id", "") if default_model is not None else ""
    return model_id if isinstance(model_id, str) and model_id else None


def _models_for_plan_mode_fallback(codex: Codex) -> list[Any]:
    try:
        return list(codex.models().data)
    except Exception:
        logger.exception("failed to fetch models for plan mode fallback")
        return []


def _thread_cwd(thread: Any) -> str | None:
    raw = getattr(thread, "cwd", None)
    if isinstance(raw, str):
        return raw or None
    root = getattr(raw, "root", None)
    return root if isinstance(root, str) and root else None


def _session_template_thread(thread: Any) -> _SessionTemplateThread:
    updated_at = getattr(thread, "updated_at", "")
    return _SessionTemplateThread(
        id=string_value(getattr(thread, "id", "")),
        cwd=_thread_cwd(thread) or "",
        updated_at="" if updated_at is None else updated_at,
    )


def _is_allowed_session_cwd(cwd: str) -> bool:
    if worktrees_module.is_managed_worktree_path(cwd):
        return True
    return cwd in _allowed_session_cwds()


def _candidate_thread_user_message_index(
    thread_id: str, settings: SettingsValues
) -> int:
    resumed = codex_pool.run_borrowed_op_with_retry(
        Codex,
        lambda codex: codex._client.thread_resume(thread_id),
        enable_memories=settings.enable_memories,
    )
    return _count_user_entries(list(_entries_for(resumed.thread)))


def _next_user_message_index_for_candidate_thread(
    thread_id: str, settings: SettingsValues
) -> int:
    latest_instance = (
        CodexInstance.objects.filter(
            thread_id=thread_id,
            user_message_index__isnull=False,
        )
        .order_by("-user_message_index", "-pk")
        .values("status", "user_message_index")
        .first()
    )
    if latest_instance is None:
        return _candidate_thread_user_message_index(thread_id, settings)
    if latest_instance["status"] == CodexInstance.STATUS_FAILED:
        return _candidate_thread_user_message_index(thread_id, settings)
    latest_index = latest_instance["user_message_index"]
    if latest_index is None:
        return _candidate_thread_user_message_index(thread_id, settings)
    return max(int(latest_index) + 1, 0)


def _finish_candidate_proposal_start(
    *,
    request: HttpRequest,
    proposed_session: ProposedSession,
    candidate_session: SessionMetadata,
    cwd: str,
    target: _NewSessionTarget,
    settings: SettingsValues,
    cookie_updates: dict[str, str],
    auto_pr_enabled: bool,
    auto_qa_enabled: bool,
) -> HttpResponse:
    candidate_cwd = candidate_session.cwd
    auto_merge_to_local_branch, auto_merge_branch = (
        _auto_merge_to_local_branch_for_proposal(
            proposed_session,
            auto_qa_enabled=auto_qa_enabled,
        )
    )
    session_project = (
        None
        if target.project_cleared
        else candidate_session.project or target.project
    )
    _rename_codex_thread_from_proposal(
        proposed_session=proposed_session,
        session_metadata=candidate_session,
        settings=settings,
    )
    SessionMetadata.objects.filter(pk=candidate_session.pk).update(
        cwd=candidate_cwd,
        project=session_project,
        project_cleared=target.project_cleared,
        auto_pr_enabled=auto_pr_enabled,
        auto_qa_enabled=auto_qa_enabled,
        auto_merge_to_local_branch=auto_merge_to_local_branch,
        auto_merge_branch=auto_merge_branch,
        is_hidden_system_session=False,
    )
    candidate_session.refresh_from_db()
    remembered_values = settings._replace(last_selected_repo=cwd)
    user = _authenticated_user(request)
    if user is not None:
        _save_user_settings(user, remembered_values)
        cookie_updates = _settings_cookie_updates(remembered_values)
    else:
        cookie_updates = {**cookie_updates, _LAST_SELECTED_REPO_COOKIE: cwd}
    response = redirect("session", session_id=candidate_session.thread_id)
    _apply_cookie_updates(response, cookie_updates)
    return response


def _start_candidate_proposal_session(
    *,
    request: HttpRequest,
    proposed_session: ProposedSession,
    candidate_session: SessionMetadata,
    prompt: str,
    plan_mode: bool,
    qa_activation: bool,
    qa_workflow_activation: bool,
    cwd: str,
    target: _NewSessionTarget,
    settings: SettingsValues,
    spawn_settings: SettingsValues,
    cookie_updates: dict[str, str],
    auto_pr_enabled: bool,
    auto_qa_enabled: bool,
    web_search_mode: str,
) -> HttpResponse:
    """Start a proposal on its existing candidate thread before accepting it."""
    candidate_cwd = candidate_session.cwd
    if not candidate_cwd:
        return HttpResponseBadRequest("candidate session has no cwd")
    if not _is_allowed_session_cwd(candidate_cwd):
        return HttpResponseBadRequest(
            "candidate session cwd is not an allowed repository"
        )
    prompt = _candidate_proposal_continuation_prompt(prompt)
    base_instructions = _base_instructions_for_settings(spawn_settings)
    project = None if target.project_cleared else candidate_session.project or target.project
    developer_instructions = _developer_instructions_for_project(settings, project)
    auto_merge_to_local_branch, auto_merge_branch = (
        _auto_merge_to_local_branch_for_proposal(
            proposed_session,
            auto_qa_enabled=auto_qa_enabled,
        )
    )
    sandbox_policy = _effective_sandbox_policy_for_cwd(settings, candidate_cwd)
    approval_mode = _effective_approval_mode_for_session(
        settings,
        candidate_session.thread_id,
        candidate_session,
    )
    if qa_workflow_activation:
        workflow_kwargs: dict[str, Any] = {
            "main_thread_id": candidate_session.thread_id,
            "cwd": candidate_cwd,
            "sandbox_policy": sandbox_policy or None,
            "approval_mode": approval_mode,
            "model": settings.model or None,
            "reasoning_effort": settings.reasoning_effort or None,
            "developer_instructions": developer_instructions or None,
            "enable_memories": settings.enable_memories,
            "initial_user_message_index": _next_user_message_index_for_candidate_thread(
                candidate_session.thread_id, settings
            ),
        }
        if web_search_mode:
            workflow_kwargs["web_search_mode"] = web_search_mode
        if base_instructions:
            workflow_kwargs["base_instructions"] = base_instructions
        if qa_activation:
            workflow_kwargs["open_pr_on_lgtm"] = False
        if auto_merge_branch:
            workflow_kwargs["auto_merge_branch"] = auto_merge_branch
        claim_response = _claim_candidate_proposal_start(
            proposed_session=proposed_session,
            candidate_session=candidate_session,
            cookie_updates=cookie_updates,
        )
        if claim_response is not None:
            return claim_response
        try:
            system_agents.start_pr_qa_workflow(**workflow_kwargs)
        except Exception:
            _reset_candidate_proposal_start_claim(proposed_session, candidate_session)
            raise
        # Persist the proposal-derived auto-review configuration so subsequent
        # turns in this session keep honoring it. Hardcoding ``False`` here would
        # silently drop a goal's auto-QA/auto-merge settings after the first turn.
        response = _finish_candidate_proposal_start(
            request=request,
            proposed_session=proposed_session,
            candidate_session=candidate_session,
            cwd=cwd,
            target=target,
            settings=settings,
            cookie_updates=cookie_updates,
            auto_pr_enabled=auto_pr_enabled,
            auto_qa_enabled=auto_qa_enabled,
        )
        _stop_autonomous_goal_stack_after_proposal_resolution(proposed_session)
        return response

    input_image_paths, input_image_error = _save_posted_input_images(request)
    if input_image_error is not None:
        return HttpResponseBadRequest(input_image_error)
    spawn_kwargs: dict[str, Any] = {
        "thread_id": candidate_session.thread_id,
        "cwd": candidate_cwd,
        "prompt": prompt,
        "developer_instructions": developer_instructions or None,
        "model": settings.model or None,
        "reasoning_effort": settings.reasoning_effort or None,
        "sandbox_policy": sandbox_policy or None,
        "approval_mode": approval_mode,
    }
    if input_image_paths:
        spawn_kwargs["input_image_paths"] = input_image_paths
    if web_search_mode:
        spawn_kwargs["web_search_mode"] = web_search_mode
    if base_instructions:
        spawn_kwargs["base_instructions"] = base_instructions
    if settings.enable_memories:
        spawn_kwargs["enable_memories"] = True
    if plan_mode:
        spawn_kwargs["plan_mode"] = True
    if auto_pr_enabled:
        spawn_kwargs["auto_pr_enabled"] = True
    if auto_qa_enabled:
        spawn_kwargs["auto_qa_enabled"] = True
    if auto_pr_enabled or auto_qa_enabled:
        spawn_kwargs["stored_model"] = settings.model or None
        spawn_kwargs["stored_reasoning_effort"] = settings.reasoning_effort or None
        spawn_kwargs["user_message_index"] = _next_user_message_index_for_candidate_thread(
            candidate_session.thread_id, settings
        )
        if auto_merge_to_local_branch:
            spawn_kwargs["auto_merge_to_local_branch"] = True
            spawn_kwargs["auto_merge_branch"] = auto_merge_branch

    input_images_owned = False
    claim_response = _claim_candidate_proposal_start(
        proposed_session=proposed_session,
        candidate_session=candidate_session,
        cookie_updates=cookie_updates,
    )
    if claim_response is not None:
        _cleanup_saved_input_images(input_image_paths)
        return claim_response
    try:
        codex_pool.spawn_turn(**spawn_kwargs)
        input_images_owned = True
    except codex_pool.InputAttachmentLimitExceededError as exc:
        _cleanup_saved_input_images(input_image_paths)
        _reset_candidate_proposal_start_claim(proposed_session, candidate_session)
        return HttpResponseBadRequest(str(exc))
    except Exception:
        if not input_images_owned:
            _cleanup_saved_input_images(input_image_paths)
            _reset_candidate_proposal_start_claim(proposed_session, candidate_session)
        raise

    response = _finish_candidate_proposal_start(
        request=request,
        proposed_session=proposed_session,
        candidate_session=candidate_session,
        cwd=cwd,
        target=target,
        settings=settings,
        cookie_updates=cookie_updates,
        auto_pr_enabled=auto_pr_enabled,
        auto_qa_enabled=auto_qa_enabled,
    )
    _stop_autonomous_goal_stack_after_proposal_resolution(proposed_session)
    return response


def _candidate_proposal_continuation_prompt(prompt: str) -> str:
    rebase_instruction = (
        "First, rebase or otherwise update this worktree onto the current project "
        "base branch before continuing. Resolve any conflicts, then continue with "
        "the user's instructions."
    )
    prompt = prompt.strip()
    if not prompt:
        return rebase_instruction
    return f"{rebase_instruction}\n\n{prompt}"


def _auto_merge_to_local_branch_for_proposal(
    proposed_session: ProposedSession,
    *,
    auto_qa_enabled: bool,
) -> tuple[bool, str]:
    if not auto_qa_enabled:
        return False, ""
    metadata = _proposal_metadata(proposed_session)
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


def _auto_review_settings_for_proposed_session(
    proposed_session: ProposedSession,
) -> tuple[bool, bool]:
    metadata = _proposal_metadata(proposed_session)
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


def _proposal_metadata(proposed_session: ProposedSession) -> dict[str, object]:
    return (
        proposed_session.outcome_metadata
        if isinstance(proposed_session.outcome_metadata, dict)
        else {}
    )


def _proposed_session_for_new_session_page(
    request: HttpRequest,
    *,
    repo_set: set[str],
) -> ProposedSession | None:
    raw_session_id = request.GET.get("proposed_session", "").strip()
    if not raw_session_id:
        return None
    try:
        session_id = int(raw_session_id)
    except ValueError as exc:
        raise Http404("proposed session not found") from exc
    if session_id < 1 or session_id > _MAX_BIGAUTOFIELD:
        raise Http404("proposed session not found")
    _recover_stale_new_session_proposal_start_claims()
    proposed_session = (
        ProposedSession.objects.select_related(
            "project", "autonomous_goal__project", "candidate_session"
        )
        .filter(
            pk=session_id,
            inbox_kind=ProposedSession.INBOX_KIND_PROPOSAL,
            outcome_status=ProposedSession.OUTCOME_UNSET,
        )
        .first()
    )
    project = _project_for_proposed_session(proposed_session)
    if (
        proposed_session is None
        or project is None
        or project.repo_path not in repo_set
    ):
        raise Http404("proposed session not found")
    _attach_proposed_session_display_state([proposed_session])
    return proposed_session


def _prefill_project_for_new_session_page(
    request: HttpRequest, projects: list[Project], *, repo_set: set[str]
) -> Project | None:
    raw_project_id = request.GET.get("project")
    if raw_project_id is None:
        return None
    raw_project_id = raw_project_id.strip()
    if not raw_project_id:
        return None
    try:
        project_id = int(raw_project_id)
    except ValueError as exc:
        raise Http404("project not found") from exc
    project = next(
        (
            project
            for project in projects
            if project.pk == project_id and project.repo_path in repo_set
        ),
        None,
    )
    if project is None:
        raise Http404("project not found")
    return project


def _prefill_bare_repo_cwd_for_new_session_page(
    request: HttpRequest, *, repo_set: set[str]
) -> str:
    cwd = request.GET.get("cwd", "").strip()
    if not cwd:
        return ""
    if cwd not in repo_set:
        raise Http404("repository not found")
    return cwd


def _render_new_session_page(request: HttpRequest) -> HttpResponse:
    codex_pool.reconcile_dead_if_due()
    repos = [str(p) for p in repos_module.discover_repos()]
    repo_set = set(repos)
    proposed_session = _proposed_session_for_new_session_page(
        request, repo_set=repo_set
    )
    models_data, resolved_settings = _cached_models_and_settings(request)
    current_settings = resolved_settings.values
    cookie_updates = resolved_settings.cookie_updates
    projects = list(Project.objects.all())
    current_project = _selected_project_for_settings(current_settings, projects)
    prefill_bare_repo_cwd = ""
    if proposed_session is None:
        prefill_project = _prefill_project_for_new_session_page(
            request, projects, repo_set=repo_set
        )
        if prefill_project is not None:
            current_project = prefill_project
        else:
            prefill_bare_repo_cwd = _prefill_bare_repo_cwd_for_new_session_page(
                request, repo_set=repo_set
            )
            if prefill_bare_repo_cwd:
                current_project = None
    settings_context = _settings_context(current_settings, models_data)
    new_session_context = _new_session_form_context(
        current_settings,
        current_project,
        settings_context["projects"],
        initial_prompt=(
            _proposed_session_prompt(proposed_session)
            if proposed_session is not None
            else request.GET.get("prompt", "")
        ),
        proposed_session=proposed_session,
        prefill_bare_repo_cwd=prefill_bare_repo_cwd,
        repos=repos,
    )
    response = render(
        request,
        "new_session.html",
        {
            "login_url": reverse("login"),
            "register_url": reverse("register"),
            **settings_context,
            **new_session_context,
        },
    )
    _apply_cookie_updates(response, cookie_updates)
    return response


def _post_new_session(request: HttpRequest) -> HttpResponse:
    intent = _message_intent(request)
    pr_activation = _is_pr_activation(request)
    fix_pr_activation = _is_fix_pr_activation(request)
    qa_activation = _is_qa_activation(request)
    qa_workflow_activation = pr_activation or qa_activation or fix_pr_activation
    prompt = intent.prompt
    plan_mode = False if qa_workflow_activation else intent.plan_mode
    has_input_images = _has_input_image_uploads(request)
    if fix_pr_activation:
        return HttpResponseBadRequest("fix-pr requires an existing session with a PR")
    projects = list(Project.objects.all())
    target, target_error = _posted_new_session_target(request, projects)
    if target_error is not None or target is None:
        return HttpResponseBadRequest(target_error or "invalid project")
    proposed_session, proposed_session_error = _posted_proposed_session_for_new_session(
        request, target
    )
    if proposed_session_error is not None:
        return HttpResponseBadRequest(proposed_session_error)
    coding_agent_override, coding_agent_error = _posted_new_session_coding_agent(
        request.POST.get("coding_agent")
    )
    if coding_agent_error is not None:
        return HttpResponseBadRequest(coding_agent_error)
    cwd = target.cwd
    if not prompt and not has_input_images:
        return HttpResponseBadRequest("prompt is required")
    if not cwd:
        return HttpResponseBadRequest("cwd is required")
    # Raw cwd posts still need discovery validation. Project-id posts use the
    # server-side Project.repo_path, so they do not need a home-directory scan
    # on the hot Start path.
    if target.requires_discovered_repo:
        allowed = {str(p) for p in repos_module.discover_repos()}
        if cwd not in allowed:
            return HttpResponseBadRequest("cwd must be a discovered repository")

    # Re-reconcile the cookies against Codex's current model list before
    # spawning. A long-lived tab might still be carrying a model the index
    # render would have snapped away from; without this, a stale value
    # would ride straight into ``thread_start(model=...)`` and 500 the
    # new-session click.
    resolved_settings = _new_session_post_settings(request)
    settings = resolved_settings.values
    spawn_settings = (
        settings._replace(coding_agent=coding_agent_override)
        if coding_agent_override
        else settings
    )
    use_worktrees, use_worktrees_error = _posted_use_worktree_override(
        request.POST.get("use_worktrees"), default=settings.use_worktrees
    )
    if use_worktrees_error is not None:
        return HttpResponseBadRequest(use_worktrees_error)
    cookie_updates = resolved_settings.cookie_updates
    source_project = target.project
    source_developer_instructions = _developer_instructions_for_project(
        settings, None if target.project_cleared else source_project
    )
    default_auto_pr_enabled = _effective_auto_pr_enabled(
        None if target.project_cleared else source_project,
        global_enabled=settings.auto_pr_enabled,
    )
    auto_pr_enabled, auto_pr_error = _posted_auto_pr_override(
        request.POST.get("auto_pr"), default=default_auto_pr_enabled
    )
    if auto_pr_error is not None:
        return HttpResponseBadRequest(auto_pr_error)
    auto_qa_enabled, auto_qa_error = _posted_auto_qa_override(
        request.POST.get("auto_qa"), default=settings.auto_qa_enabled
    )
    if auto_qa_error is not None:
        return HttpResponseBadRequest(auto_qa_error)
    if auto_pr_enabled:
        auto_qa_enabled = False
    if proposed_session is not None and proposed_session.autonomous_goal is not None:
        auto_pr_enabled, auto_qa_enabled = _auto_review_settings_for_proposed_session(
            proposed_session
        )
    auto_merge_to_local_branch = False
    auto_merge_branch = ""
    if proposed_session is not None:
        auto_merge_to_local_branch, auto_merge_branch = (
            _auto_merge_to_local_branch_for_proposal(
                proposed_session, auto_qa_enabled=auto_qa_enabled
            )
        )
    web_search_mode, web_search_error = _posted_web_search_override(
        request.POST.get("web_search_mode"),
        default=settings.web_search_mode,
    )
    if web_search_error is not None:
        return HttpResponseBadRequest(web_search_error)
    if plan_mode and not settings.model:
        return HttpResponseBadRequest("plan mode requires a model")
    if qa_workflow_activation and has_input_images:
        return HttpResponseBadRequest(
            "image attachments are not supported for PR workflow requests"
        )
    candidate_session = _candidate_session_to_continue_from_proposal(proposed_session)
    if candidate_session is not None:
        assert proposed_session is not None
        return _start_candidate_proposal_session(
            request=request,
            proposed_session=proposed_session,
            candidate_session=candidate_session,
            prompt=prompt,
            plan_mode=plan_mode,
            qa_activation=qa_activation,
            qa_workflow_activation=qa_workflow_activation,
            cwd=cwd,
            target=target,
            settings=settings,
            spawn_settings=spawn_settings,
            cookie_updates=cookie_updates,
            auto_pr_enabled=auto_pr_enabled,
            auto_qa_enabled=auto_qa_enabled,
            web_search_mode=web_search_mode,
        )

    session_cwd = cwd
    sandbox_policy = _effective_sandbox_policy_for_cwd(settings, session_cwd)
    # QA workflows review the selected repo's current diff; a fresh managed
    # worktree would be clean and miss uncommitted changes.
    if qa_workflow_activation:
        if proposed_session is not None:
            thread_name = proposed_session.title
        else:
            thread_name = _PR_SLASH_PROMPT if pr_activation else _QA_SLASH_PROMPT
        base_instructions = _base_instructions_for_settings(spawn_settings)
        create_thread_kwargs: dict[str, Any] = {
            "cwd": session_cwd,
            "name": thread_name,
            "developer_instructions": source_developer_instructions or None,
            "model": settings.model or None,
            "enable_memories": settings.enable_memories,
        }
        if web_search_mode:
            create_thread_kwargs["web_search_mode"] = web_search_mode
        if base_instructions:
            create_thread_kwargs["base_instructions"] = base_instructions
        proposal_claimed = False
        if proposed_session is not None:
            claim_response = _claim_new_session_proposal_start(
                proposed_session=proposed_session,
                cookie_updates=cookie_updates,
            )
            if claim_response is not None:
                return claim_response
            proposal_claimed = True
        try:
            thread_id = codex_pool.create_session_thread(**create_thread_kwargs)
        except Exception:
            if proposal_claimed:
                assert proposed_session is not None
                _reset_new_session_proposal_start_claim(proposed_session)
            raise
        # Only proposal acceptances carry forward auto-review/auto-merge, and
        # only the settings the proposal itself requested. A bare ``/qa`` or
        # ``/pr`` (no proposal) is a one-off review, and a coding-agent proposal
        # leaves these inputs empty, so in both cases the resolved
        # ``auto_*_enabled`` here are just the user's global/form defaults.
        # Persisting those would silently auto-review every later follow-up in
        # the session, so derive the stored flags from the proposal only.
        if proposed_session is not None:
            session_auto_pr_enabled, session_auto_qa_enabled = (
                _auto_review_settings_for_proposed_session(proposed_session)
            )
        else:
            session_auto_pr_enabled = False
            session_auto_qa_enabled = False
            auto_merge_to_local_branch, auto_merge_branch = False, ""
        workflow_kwargs: dict[str, Any] = {
            "main_thread_id": thread_id,
            "cwd": session_cwd,
            "sandbox_policy": sandbox_policy or None,
            "approval_mode": settings.approval_mode,
            "model": settings.model or None,
            "reasoning_effort": settings.reasoning_effort or None,
            "developer_instructions": source_developer_instructions or None,
            "enable_memories": settings.enable_memories,
            "initial_user_message_index": 0,
        }
        if web_search_mode:
            workflow_kwargs["web_search_mode"] = web_search_mode
        if base_instructions:
            workflow_kwargs["base_instructions"] = base_instructions
        if qa_activation:
            workflow_kwargs["open_pr_on_lgtm"] = False
        if auto_merge_branch:
            workflow_kwargs["auto_merge_branch"] = auto_merge_branch
        try:
            system_agents.start_pr_qa_workflow(**workflow_kwargs)
        except Exception:
            if proposal_claimed:
                assert proposed_session is not None
                _reset_new_session_proposal_start_claim(proposed_session)
            raise
        # Persist the proposal-derived auto-review configuration so subsequent
        # turns keep honoring it instead of reverting to manual review.
        session_metadata = session_index.upsert_local_session(
            thread_id=thread_id,
            cwd=session_cwd,
            project=source_project,
            project_cleared=target.project_cleared,
            name=thread_name,
            auto_pr_enabled=session_auto_pr_enabled,
            auto_qa_enabled=session_auto_qa_enabled,
            auto_merge_to_local_branch=auto_merge_to_local_branch,
            auto_merge_branch=auto_merge_branch,
        )
        _finish_new_session_proposal_start_claim(proposed_session, session_metadata)
        remembered_values = settings._replace(last_selected_repo=cwd)
        user = _authenticated_user(request)
        if user is not None:
            _save_user_settings(user, remembered_values)
            cookie_updates = _settings_cookie_updates(remembered_values)
        else:
            cookie_updates = {**cookie_updates, _LAST_SELECTED_REPO_COOKIE: cwd}
        response = redirect("session", session_id=thread_id)
        _apply_cookie_updates(response, cookie_updates)
        return response

    managed_worktree = None
    if use_worktrees:
        try:
            managed_worktree = create_worktree_for_session(cwd)
        except WorktreeCreationError as exc:
            return HttpResponseBadRequest(str(exc))
        session_cwd = str(managed_worktree.path)
        sandbox_policy = _effective_sandbox_policy_for_cwd(
            settings,
            session_cwd,
            managed_worktree=True,
        )
    session_project = (
        None
        if target.project_cleared
        else _project_for_cwd(session_cwd, projects) or source_project
    )
    developer_instructions = _developer_instructions_for_project(
        settings, session_project
    )
    input_image_paths, input_image_error = _save_posted_input_images(request)
    if input_image_error is not None:
        if managed_worktree is not None:
            try:
                cleanup_worktree(managed_worktree)
            except WorktreeCleanupError:
                logger.exception(
                    "failed to clean up managed worktree %s", managed_worktree.path
                )
        return HttpResponseBadRequest(input_image_error)

    # Detach a worker subprocess so the initial turn keeps running past a
    # Django restart. The thread itself is created synchronously to give the
    # caller a stable id to redirect to.
    spawn_kwargs: dict[str, Any] = {
        "cwd": session_cwd,
        "prompt": prompt,
        "developer_instructions": developer_instructions or None,
        "model": settings.model or None,
        "reasoning_effort": settings.reasoning_effort or None,
        "sandbox_policy": sandbox_policy or None,
        "approval_mode": settings.approval_mode,
    }
    if input_image_paths:
        spawn_kwargs["input_image_paths"] = input_image_paths
    if web_search_mode:
        spawn_kwargs["web_search_mode"] = web_search_mode
    if proposed_session is not None:
        spawn_kwargs["thread_name"] = proposed_session.title
    base_instructions = _base_instructions_for_settings(spawn_settings)
    if base_instructions:
        spawn_kwargs["base_instructions"] = base_instructions
    if settings.enable_memories:
        spawn_kwargs["enable_memories"] = True
    if plan_mode:
        spawn_kwargs["plan_mode"] = True
    if auto_pr_enabled:
        spawn_kwargs["auto_pr_enabled"] = True
    if auto_qa_enabled:
        spawn_kwargs["auto_qa_enabled"] = True
    if auto_merge_to_local_branch:
        spawn_kwargs["auto_merge_to_local_branch"] = True
        spawn_kwargs["auto_merge_branch"] = auto_merge_branch
    # Proposed sessions already represent reviewed work for the user to start, so
    # they bypass Spec Critic entirely. For everything else the should-run
    # classifier runs inside the workflow on a background thread, so creating a
    # new session never blocks on that LLM call.
    if (
        proposed_session is None
        and settings.spec_critic_enabled
        and not input_image_paths
        and not plan_mode
    ):
        spec_create_thread_kwargs: dict[str, Any] = {
            "cwd": session_cwd,
            "name": (
                proposed_session.title
                if proposed_session is not None
                else prompt.split("\n", 1)[0]
            ),
            "developer_instructions": developer_instructions or None,
            "model": settings.model or None,
            "enable_memories": settings.enable_memories,
        }
        if web_search_mode:
            spec_create_thread_kwargs["web_search_mode"] = web_search_mode
        if base_instructions:
            spec_create_thread_kwargs["base_instructions"] = base_instructions
        try:
            thread_id = codex_pool.create_session_thread(**spec_create_thread_kwargs)
        except Exception:
            if managed_worktree is not None:
                try:
                    cleanup_worktree(managed_worktree)
                except WorktreeCleanupError:
                    logger.exception(
                        "failed to clean up managed worktree %s", managed_worktree.path
                    )
            raise
        spec_workflow_kwargs: dict[str, Any] = {
            "main_thread_id": thread_id,
            "cwd": session_cwd,
            "prompt": prompt,
            "sandbox_policy": sandbox_policy or None,
            "approval_mode": settings.approval_mode,
            "model": settings.model or None,
            "reasoning_effort": settings.reasoning_effort or None,
            "developer_instructions": developer_instructions or None,
            "enable_memories": settings.enable_memories,
            "initial_user_message_index": 0,
            "auto_pr_enabled": auto_pr_enabled,
            "auto_qa_enabled": auto_qa_enabled,
        }
        if base_instructions:
            spec_workflow_kwargs["base_instructions"] = base_instructions
        if web_search_mode:
            spec_workflow_kwargs["web_search_mode"] = web_search_mode
        try:
            system_agents.start_spec_critic_workflow(**spec_workflow_kwargs)
        except Exception:
            # The worktree is only referenced by the not-yet-started workflow, so
            # reclaim it before bubbling up rather than leaking it on disk (and
            # into the cwd allowlist) on every failed-then-retried new session.
            if managed_worktree is not None:
                try:
                    cleanup_worktree(managed_worktree)
                except WorktreeCleanupError:
                    logger.exception(
                        "failed to clean up managed worktree %s", managed_worktree.path
                    )
            raise
        spec_thread_name = (
            proposed_session.title
            if proposed_session is not None
            else prompt.split("\n", 1)[0]
        )
        session_metadata = session_index.upsert_local_session(
            thread_id=thread_id,
            cwd=session_cwd,
            project=session_project,
            project_cleared=target.project_cleared,
            name=spec_thread_name,
            preview=prompt,
            auto_pr_enabled=auto_pr_enabled,
            auto_qa_enabled=auto_qa_enabled,
        )
        _accept_proposed_session_for_session(proposed_session, session_metadata)
        remembered_values = settings._replace(last_selected_repo=cwd)
        user = _authenticated_user(request)
        if user is not None:
            _save_user_settings(user, remembered_values)
            cookie_updates = _settings_cookie_updates(remembered_values)
        else:
            cookie_updates = {**cookie_updates, _LAST_SELECTED_REPO_COOKIE: cwd}
        response = redirect("session", session_id=thread_id)
        _apply_cookie_updates(response, cookie_updates)
        return response
    input_images_owned = False
    proposal_claimed = False
    if proposed_session is not None:
        claim_response = _claim_new_session_proposal_start(
            proposed_session=proposed_session,
            cookie_updates=cookie_updates,
        )
        if claim_response is not None:
            _cleanup_saved_input_images(input_image_paths)
            if managed_worktree is not None:
                try:
                    cleanup_worktree(managed_worktree)
                except WorktreeCleanupError:
                    logger.exception(
                        "failed to clean up managed worktree %s", managed_worktree.path
                    )
            return claim_response
        proposal_claimed = True
    try:
        instance = codex_pool.spawn_new_session(**spawn_kwargs)
        input_images_owned = True
    except Exception:
        if not input_images_owned:
            _cleanup_saved_input_images(input_image_paths)
        if proposal_claimed:
            assert proposed_session is not None
            _reset_new_session_proposal_start_claim(proposed_session)
        if managed_worktree is not None:
            try:
                cleanup_worktree(managed_worktree)
            except WorktreeCleanupError:
                logger.exception(
                    "failed to clean up managed worktree %s", managed_worktree.path
                )
        raise
    session_metadata = session_index.upsert_local_session(
        thread_id=instance.thread_id,
        cwd=session_cwd,
        project=session_project,
        project_cleared=target.project_cleared,
        name=proposed_session.title if proposed_session is not None else "",
        preview=prompt,
        auto_pr_enabled=auto_pr_enabled,
        auto_qa_enabled=auto_qa_enabled,
        auto_merge_to_local_branch=auto_merge_to_local_branch,
        auto_merge_branch=auto_merge_branch,
        codex_path=codex_pool.thread_path_for_instance(instance),
    )
    _finish_new_session_proposal_start_claim(proposed_session, session_metadata)
    remembered_values = settings._replace(last_selected_repo=cwd)
    user = _authenticated_user(request)
    if user is not None:
        _save_user_settings(user, remembered_values)
        cookie_updates = _settings_cookie_updates(remembered_values)
    else:
        cookie_updates = {**cookie_updates, _LAST_SELECTED_REPO_COOKIE: cwd}
    response = redirect("session", session_id=instance.thread_id)
    _apply_cookie_updates(response, cookie_updates)
    return response


@_limit_input_image_uploads
@require_http_methods(["GET", "POST"])
def new_session(request: HttpRequest) -> HttpResponse:
    if request.method == "GET":
        return _render_new_session_page(request)
    return _post_new_session(request)
