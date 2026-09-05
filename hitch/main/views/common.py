"""Shared helpers for the view modules: settings context, the session-list
query machinery, cookie plumbing, and cross-domain utilities."""
import contextlib
import logging
import os
import re
import threading
import uuid
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlencode

from django.conf import settings as django_settings
from django.core.files.uploadedfile import UploadedFile
from django.db import close_old_connections, transaction
from django.db.models import QuerySet
from django.http import (
    Http404,
    HttpRequest,
    HttpResponse,
)
from django.shortcuts import render
from django.urls import reverse
from django.utils import timezone
from django.utils.http import url_has_allowed_host_and_scheme
from openai_codex import Codex as Codex
from openai_codex.errors import InternalRpcError, InvalidRequestError
from openai_codex.generated.v2_all import (
    SortDirection,
    ThreadSortKey,
)

from hitch.main import caches
from hitch.main import repos as repos_module
from hitch.main import worktrees as worktrees_module
from hitch.main.diffs import DiffView, build_worktree_diff
from hitch.main.goals.autonomous_goal_proposal_stack import _proposal_outcome_metadata
from hitch.main.models import (
    ApprovalRequest,
    CodexInstance,
    Project,
    ProposedSession,
    SessionMetadata,
)
from hitch.main.repos import git_common_dir as git_common_dir
from hitch.main.repos import same_repo_or_worktree
from hitch.main.runtime import (
    app_server_pool,
    codex_events,
    codex_pool,
    reconciliation,
    rollout,
)
from hitch.main.runtime.input_images import (
    _INPUT_IMAGE_ACCEPT,
    _INPUT_IMAGE_FIELD,
    _INPUT_IMAGE_MAX_BYTES,
    _INPUT_IMAGE_MAX_COUNT,
)
from hitch.main.runtime.rollout_state import (
    _rollout_file_state_from_value,
    _rollout_path_for,
    _rollout_path_from_value,
    _RolloutFileState,
    _thread_is_archived,
)
from hitch.main.runtime.sdk_values import (
    string_value,
)
from hitch.main.sessions import agent_tasks, session_index, session_stage, token_usage
from hitch.main.sessions.message_intent import (
    _FIX_PR_SLASH_COMMAND,
)
from hitch.main.sessions.pr_prompts import PR_SLASH_DISPLAY_PROMPT
from hitch.main.sessions.project_visibility import (
    _filter_proposed_sessions_by_project_visibility,
)
from hitch.main.sessions.project_visibility import (
    _metadata_by_thread_id as _metadata_by_thread_id,
)
from hitch.main.sessions.session_entry_display import (
    _accepted_proposal_context,
    _active_history_user_identity,
    _active_instance_for,
    _active_stream_owns_turn,
    _active_worker_status_text,
    _apply_system_authors,
    _attach_accepted_proposal_context,
    _display_title,
    _entries_for_with_source,
    _entries_include_active_turn,
    _latest_user_turn_failure,
    _mark_active_history_user_entries,
    _pending_user_author,
    _pending_user_prompt,
    _pending_user_timestamp,
    _show_active_worker_transcript,
    _task_plan_context,
    _trim_in_progress_turn,
)
from hitch.main.sessions.session_pr_plan import (
    _ROLLOUT_COLLABORATION_MODE_NOT_PROVIDED,
    _mark_pending_plan_actions,
    _registered_pr_url,
    _thread_plan_mode_state,
)
from hitch.main.sessions.session_resume import (
    _metadata_resume_for_inactive_session,
    _pending_resume_for_active_session,
    _rollout_path_for_session_detail,
    _session_detail_metadata,
    _stored_model_config_for_session,
)
from hitch.main.sessions.session_settings import (
    _PLAN_MODE_REASONING_EFFORT,
    _QA_SLASH_PROMPT,
    _allowed_session_cwds,
    _current_disk_usage_max_percent,
    _effective_approval_mode,
    _effective_approval_mode_for_session,
    _effective_sandbox_policy_for_cwd,
    _format_disk_usage_max_percent,
    _reasoning_effort_values,
    _resolved_settings,
    _selected_project_for_settings,
    _session_approval_mode_override,
    _session_project_visibility_for_settings,
    _stored_settings,
    _supported_effort_values,
)
from hitch.main.sessions.session_stage_refresh import (
    _thread_ids_awaiting_input,
)
from hitch.main.sessions.settings_cookies import (
    _APPROVAL_MODE_OPTIONS,
    _EXTRA_SYSTEM_PROMPT_MAX_LEN,
    _LIVE_HANDLER_APPROVAL_MODES,
    _LIVE_PENDING_APPROVAL_DECISIONS_BY_MODE,
    _SANDBOX_POLICY_OPTIONS,
    _WEB_SEARCH_MODE_OPTIONS,
    SessionProjectVisibility,
    SettingsValues,
    _apply_cookie_updates,
    _option_label,
    _web_search_mode_label,
)
from hitch.main.workflows import autonomous_goals as goal_workflows
from hitch.main.workflows import pr_stage, pr_tracking, system_agents
from hitch.main.worktrees import cleanup_managed_worktree_path as cleanup_managed_worktree_path
from hitch.main.worktrees import cleanup_worktree as cleanup_worktree
from hitch.main.worktrees import create_worktree_for_session as create_worktree_for_session

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

class _SessionTemplateThread(NamedTuple):
    id: str
    cwd: str
    updated_at: Any

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

_THREAD_LIST_FETCH_LIMIT = 100

_THREAD_LIST_USE_STATE_DB_ONLY = True

_NAME_MAX_LEN = session_index.SESSION_NAME_MAX_LEN

_PROJECT_NAME_MAX_LEN = 200

_MAX_BIGAUTOFIELD = 2**63 - 1

_PLAN_APPROVAL_PROMPT = "Implement the plan."

_PLAN_REVISION_PROMPT = "Revise the plan."

_SESSION_HISTORY_MIN_BYTES = 2 * 1024 * 1024

_SESSION_HISTORY_MESSAGE_TARGET = 40

_ACTIVE_TRANSCRIPT_OWNER_STREAM = "stream"

_ACTIVE_TRANSCRIPT_OWNER_ROLLOUT = "rollout"

_INTERMEDIATE_DETAIL_CACHE_LOCK = threading.Lock()

_INTERMEDIATE_DETAIL_CACHE_MAX_SIZE = 1024

_INTERMEDIATE_DETAIL_CACHE: OrderedDict[
    tuple[str, str, int, int], dict[str, Any]
] = OrderedDict()

def _settings_context(
    current_settings: SettingsValues,
    models_data: list[Any],
    *,
    preserve_current_choices: bool = False,
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
    # ``_validate_model_and_effort_against_models``).
    supported_by_model = {m.id: _supported_effort_values(m) for m in models_data}
    if (
        preserve_current_choices
        and current_settings.model in supported_by_model
        and current_settings.reasoning_effort
        and supported_by_model[current_settings.model]
    ):
        supported_by_model[current_settings.model].add(
            current_settings.reasoning_effort
        )
    current_supported = supported_by_model.get(current_settings.model, set())
    model_options = [
        {
            "id": m.id,
            "display_name": m.display_name,
            # Space-separated so the template can drop it into a single
            # data attribute the effort-filter script splits on whitespace.
            "supported_efforts": " ".join(sorted(supported_by_model[m.id])),
        }
        for m in models_data
    ]
    if (
        preserve_current_choices
        and current_settings.model not in supported_by_model
        and (current_settings.model or models_data)
    ):
        model_options.insert(
            0,
            {
                "id": current_settings.model,
                "display_name": current_settings.model or "Model default",
                "supported_efforts": "",
            },
        )
    return {
        "settings_url": reverse("update_settings"),
        "new_project_url": reverse("new_project"),
        "edit_project_url": reverse("edit_project"),
        "model_options": model_options,
        "effort_options": [
            {
                "value": effort,
                "supported": not current_supported or effort in current_supported,
            }
            for effort in _reasoning_effort_values(
                models_data, current_effort=current_settings.reasoning_effort
            )
        ],
        "sandbox_options": [
            {"id": value, "display_name": label}
            for value, label in _SANDBOX_POLICY_OPTIONS
        ],
        "approval_options": [
            {"id": value, "display_name": label}
            for value, label in _APPROVAL_MODE_OPTIONS
        ],
        "web_search_options": [
            {"id": value, "display_name": label}
            for value, label in _WEB_SEARCH_MODE_OPTIONS
        ],
        "current_model": current_settings.model,
        "current_effort": current_settings.reasoning_effort,
        "current_sandbox": current_settings.sandbox_policy,
        "current_approval": current_settings.approval_mode,
        "current_extra_system_prompt": current_settings.extra_system_prompt,
        "extra_system_prompt_max_len": _EXTRA_SYSTEM_PROMPT_MAX_LEN,
        "current_use_worktrees": current_settings.use_worktrees,
        "current_auto_pr": current_settings.auto_pr_enabled,
        "current_auto_qa": current_settings.auto_qa_enabled,
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

def _prevent_stale_cache(response: HttpResponse) -> HttpResponse:
    # These pages render live session state (running workers, stage badges, and
    # names/archive bits that may change from another page). no-store keeps them
    # out of the browser's back/forward and heuristic caches so a Back
    # navigation re-renders against current state instead of a frozen snapshot.
    response["Cache-Control"] = "no-store"
    return response

def _usage_context(request: HttpRequest) -> UsageContext:
    stored_settings = _stored_settings(request)
    models_data = caches._cached_models_data(enable_memories=stored_settings.enable_memories)
    caches._schedule_models_refresh(enable_memories=stored_settings.enable_memories)
    resolved_settings = _resolved_settings(request, models_data)
    current_settings = resolved_settings.values
    cookie_updates = resolved_settings.cookie_updates
    rate_limits_state = caches._rate_limits_for_usage_context(
        enable_memories=current_settings.enable_memories
    )
    session_index_state = _usage_session_index_state()
    _schedule_usage_session_index_refresh_if_needed(
        enable_memories=current_settings.enable_memories,
        index_state=session_index_state,
    )
    settings_context = _settings_context(current_settings, models_data)
    current_project = settings_context["current_project"]
    usage_metadata = (
        _metadata_rows_for_usage() if session_index_state.totals_available else []
    )
    lifetime_usage = (
        token_usage._lifetime_token_usage_for_metadata(
            usage_metadata,
            selected_project_id=(
                current_project.pk if current_project is not None else None
            ),
        )
        if session_index_state.totals_available
        else None
    )
    if lifetime_usage is not None and lifetime_usage["refresh_pending"]:
        token_usage._schedule_usage_token_refresh(usage_metadata)
    return UsageContext(
        template_context={
            "login_url": reverse("login"),
            "register_url": reverse("register"),
            "rate_limits": rate_limits_state.rate_limits,
            "rate_limits_refresh_pending": rate_limits_state.refresh_pending,
            "lifetime_usage": lifetime_usage,
            **settings_context,
        },
        cookie_updates=cookie_updates,
    )

def _stop_autonomous_goal_stack_after_proposal_resolution(
    proposed_session: ProposedSession,
) -> bool:
    if proposed_session.autonomous_goal_id is None:
        return True
    return goal_workflows.stop_running_autonomous_goal_stack_after_proposal_resolution(
        proposed_session.autonomous_goal_id,
        proposed_session.pk,
        proposed_session.outcome_status,
    )

def _render_session_detail(
    request: HttpRequest,
    session_id: str,
    *,
    read_only: bool = False,
    display_title: str | None = None,
    system_prompt: str = "",
    require_system_agent_thread: bool = False,
) -> HttpResponse:
    # Reconcile this thread before reading status: a worker that died without
    # writing a terminal status would otherwise leave the page in "streaming"
    # mode forever, since the EventSource wouldn't reach an end event. The
    # global sweep stays debounced, but this exact session must be fresh.
    reconciliation.reconcile_dead_for_thread(session_id)
    reconciliation.reconcile_dead_if_due()
    initial_settings = _stored_settings(request)
    active_instance = _active_instance_for(session_id)
    unstarted_instance: CodexInstance | None = None
    metadata = _session_detail_metadata(session_id)
    # Capture the rollout mtime *before* any entries are read (the resume helper
    # below reads them off disk), so a concurrent append surfaces as a cache
    # miss on the next read rather than being masked behind a post-read stat.
    # See the matching rule in ``token_usage_snapshot``, the stage cache, and
    # the lazy intermediate-detail cache.
    detail_rollout_path = _rollout_path_for_session_detail(session_id, metadata)
    detail_rollout_state = _rollout_file_state_from_value(
        str(detail_rollout_path) if detail_rollout_path is not None else None
    )
    stage_cache_mtime_ns = (
        detail_rollout_state.mtime_ns if detail_rollout_state is not None else 0
    )
    full_history_requested = request.GET.get("history") == "all"
    active_history_user = _active_history_user_identity(active_instance)
    paginate_history = False
    if (
        detail_rollout_state is not None
        and not full_history_requested
    ):
        with contextlib.suppress(OSError):
            paginate_history = (
                detail_rollout_state.path.stat().st_size
                >= _SESSION_HISTORY_MIN_BYTES
            )
    if (
        (paginate_history or full_history_requested)
        and metadata is None
        and detail_rollout_state is not None
    ):
        metadata = SessionMetadata(
            thread_id=session_id,
            codex_path=str(detail_rollout_state.path),
        )
    metadata_resume = _metadata_resume_for_inactive_session(
        session_id,
        metadata,
        active_instance=active_instance,
        require_system_agent_thread=require_system_agent_thread,
        history_message_target=(
            _SESSION_HISTORY_MESSAGE_TARGET if paginate_history else None
        ),
        allow_active_rollout=full_history_requested,
        active_user_identity=active_history_user,
    )
    resumed: Any
    thread: Any
    entries_backed_by_rollout = False
    history_page: rollout.SessionHistoryPage | None = None

    if metadata_resume is not None:
        resumed = metadata_resume
        thread = metadata_resume.thread
        raw_entries = list(metadata_resume.entries)
        rollout_data = metadata_resume.rollout_data
        history_page = metadata_resume.history_page
        models_data = caches._cached_models_for_session_detail(
            enable_memories=initial_settings.enable_memories
        )
        resolved_settings = _resolved_settings(request, models_data)
        settings = resolved_settings.values
        cookie_updates = resolved_settings.cookie_updates
        plan_model = _plan_mode_model_from_models(resumed, settings, models_data)
    else:

        def _read_for_detail(codex: Codex) -> tuple[Any, Any, list[Any], Any, Any]:
            nonlocal unstarted_instance
            # Display reads must not resume the thread: newer Codex runtimes
            # grant resume an exclusive writer lease needed by the worker.
            try:
                resumed = codex._client.thread_read(session_id, include_turns=True)
            except (InternalRpcError, InvalidRequestError) as exc:
                if (
                    not require_system_agent_thread
                    and _thread_read_temporarily_unavailable(exc)
                ):
                    if active_instance is None:
                        latest = codex_pool.latest_for_thread(session_id)
                        if latest is not None and latest.status == CodexInstance.STATUS_FAILED:
                            unstarted_instance = latest
                    pending_resume = _pending_resume_for_active_session(
                        session_id,
                        metadata,
                        active_instance=active_instance or unstarted_instance,
                    )
                    if pending_resume is not None:
                        logger.warning(
                            "rendering saved session detail while history is temporarily "
                            "unavailable: %s",
                            session_id,
                        )
                        models_data = caches._cached_models_for_session_detail(
                            enable_memories=initial_settings.enable_memories
                        )
                        resolved_settings = _resolved_settings(request, models_data)
                        plan_model = _plan_mode_model_from_models(
                            pending_resume, resolved_settings.values, models_data
                        )
                        return (
                            pending_resume,
                            pending_resume.thread,
                            models_data,
                            resolved_settings,
                            plan_model,
                        )
                if (
                    require_system_agent_thread
                    and isinstance(exc, InvalidRequestError)
                    and _thread_read_missing_or_invalid(exc)
                ):
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

        # Reuse a warm reader without retaining a thread writer in the pool.
        resumed, thread, models_data, resolved_settings, plan_model = (
            app_server_pool.run_borrowed_op_with_retry(
                Codex,
                _read_for_detail,
                enable_memories=initial_settings.enable_memories,
            )
        )
        settings = resolved_settings.values
        cookie_updates = resolved_settings.cookie_updates
        # Capture the rollout mtime before reading entries; see the
        # metadata-resume branch above for why the order matters.
        detail_rollout_state = _rollout_file_state_from_value(
            getattr(thread, "path", None)
            or (metadata.codex_path if metadata is not None else None)
        )
        stage_cache_mtime_ns = (
            detail_rollout_state.mtime_ns if detail_rollout_state is not None else 0
        )
        raw_entries, entries_backed_by_rollout = _entries_for_with_source(
            thread,
            fallback_rollout_path=(
                metadata.codex_path if metadata is not None else None
            ),
        )
        rollout_data = None
    if not raw_entries and active_instance is None and not require_system_agent_thread:
        latest = codex_pool.latest_for_thread(session_id)
        if latest is not None and latest.status == CodexInstance.STATUS_FAILED:
            unstarted_instance = latest
    is_archived = (
        metadata.codex_archived
        if metadata is not None and metadata.archive_local_only
        else _thread_is_archived(thread)
    )
    history_paginated = history_page is not None
    if history_paginated:
        entries = raw_entries
    else:
        entries = _apply_system_authors(raw_entries, session_id)
        if full_history_requested:
            _mark_active_history_user_entries(entries, active_instance)
    name_value = getattr(thread, "name", None) or ""
    projects = list(Project.objects.all())
    metadata_by_thread = _metadata_by_thread_id([thread])
    if metadata is not None:
        metadata_by_thread[session_id] = metadata
    session_project = _project_for_thread(thread, metadata_by_thread, projects)
    stored_pr = pr_tracking.stored_record_for_thread(session_id)
    registered_pr = stored_pr if pr_tracking.record_is_current(stored_pr) else None
    publishing_before_registration = bool(
        active_instance is not None
        and active_instance.agent_kind == agent_tasks.PR_PUBLISH_AGENT_KIND
        and not pr_tracking.watch_registered_by_instance(
            registered_pr, active_instance.pk
        )
    )
    pr_url = None if publishing_before_registration else _registered_pr_url(registered_pr)
    stage_context: dict[str, Any] | None = None
    if not read_only:
        awaiting_user_input = session_id in _thread_ids_awaiting_input([session_id])
        pr_snapshot = (
            {}
            if publishing_before_registration
            else pr_tracking.pr_handoff_for_record(registered_pr)
        )
        stage = session_stage.derive_stage(
            entries=entries,
            active_instance=active_instance,
            awaiting_user_input=awaiting_user_input,
            pr_snapshot=pr_snapshot,
        )
        if (
            history_paginated
            and metadata is not None
            and metadata.derived_stage_source_mtime_ns == stage_cache_mtime_ns
            and registered_pr is None
            and active_instance is None
            and not awaiting_user_input
        ):
            cached_stage = session_stage.stage_for_key(metadata.derived_stage)
            if cached_stage is not None:
                stage = cached_stage
        # Only persist a rollout-derived stage; see _attach_session_stage_context
        # for why active-instance-forced stages must not enter the
        # mtime-keyed cache. Active turns and pending input remain transient and
        # therefore cannot be cached under a rollout-only key.
        if (
            active_instance is None
            and not awaiting_user_input
            and not history_paginated
        ):
            # Best-effort like the session-list path: this runs while rendering
            # the session detail page, so a contended write lock must skip the
            # cache refresh rather than 500 the page (the next render retries).
            pr_stage._update_cached_stage_best_effort(session_id, stage, stage_cache_mtime_ns)
        stage_context = dict(stage.as_context())
    # While a worker is running, drop the entries that belong to its
    # in-progress turn when SSE has claimed that turn. A worker kept alive
    # across a deploy can lack the user item in its event log even while its
    # rollout advances; that rollout remains the only available transcript.
    rollout_history_can_own_turn = history_page is not None or full_history_requested
    active_stream_owns_turn = bool(
        not rollout_history_can_own_turn
        or _active_stream_owns_turn(active_instance)
    )
    rollout_owns_active_turn = bool(
        rollout_history_can_own_turn
        and active_instance is not None
        and not active_stream_owns_turn
        and (
            bool(history_page is not None and history_page.active_turn_unresolved)
            or _entries_include_active_turn(entries, active_instance)
        )
    )
    active_turn_unresolved = bool(
        history_page is not None and history_page.active_turn_unresolved
    )
    # Rollout and worker event files are independently flushed, so there is no
    # lossless shared byte cursor between them. In fallback mode the rollout is
    # the transcript source for this page lifecycle; SSE still replays from the
    # beginning for controls and state, while the live root hides its transcript
    # items to prevent a second rendering of the same turn.
    show_active_worker_transcript = bool(
        _show_active_worker_transcript(active_instance)
        and not rollout_owns_active_turn
    )
    active_transcript_owner = ""
    if active_instance is not None:
        active_transcript_owner = (
            _ACTIVE_TRANSCRIPT_OWNER_ROLLOUT
            if rollout_owns_active_turn
            else _ACTIVE_TRANSCRIPT_OWNER_STREAM
        )
    entries = _trim_in_progress_turn(
        entries,
        active_instance,
        active_turn_unresolved=active_turn_unresolved,
        active_stream_owns_turn=active_stream_owns_turn,
    )
    accepted_proposal_context = _accepted_proposal_context(session_id)
    if accepted_proposal_context is not None and (
        history_page is None or not history_page.has_older
    ):
        _attach_accepted_proposal_context(entries, accepted_proposal_context)
    active_accepted_proposal_context = (
        accepted_proposal_context
        if active_instance is not None and active_instance.user_message_index == 0
        else None
    )
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
        # SDK-fallback details must stay inline: the fragment endpoint cannot
        # reconstruct entries from the rollout when the same parser failed.
        enabled=rollout_data is not None
        or (entries_backed_by_rollout and metadata is not None),
        cache_entries=rollout_data is not None,
        rollout_state=detail_rollout_state,
    )
    goal_objective = (
        "" if history_paginated else codex_events.latest_goal_for_thread(session_id)
    )
    # Scope the plan to the running worker, or to the latest worker on reload
    # when none is running, so a turn that finished without emitting its own
    # plan does not inherit an earlier turn's.
    task_plan = _task_plan_context(
        None
        if history_paginated
        else (
            codex_events.latest_task_plan_for_instance(active_instance)
            if active_instance is not None
            else codex_events.latest_task_plan_for_thread(session_id)
        )
    )
    thread_cwd = _thread_cwd(thread)
    # A running worker can change the diff continuously. The stream
    # reload after completion supplies a stable preview; until then, avoid
    # parsing and highlighting a large snapshot that the page does not expose.
    diff_view = (
        DiffView(files=[])
        if active_instance is not None
        else build_worktree_diff(thread_cwd)
    )
    settings_context = _settings_context(settings, models_data)
    active_worker_status_text = _active_worker_status_text(active_instance)
    latest_user_turn_failure = _latest_user_turn_failure(session_id)
    pr_watch_progress = pr_tracking.pr_watch_progress(
        registered_pr.state
        if registered_pr is not None and not publishing_before_registration
        else None
    )
    live_status_text = active_worker_status_text
    debug_chat_url = _debug_chat_new_session_url(
        session_id, session_project, projects, cwd=thread_cwd
    )
    approval_mode = _effective_approval_mode_for_session(
        settings, session_id, metadata
    )
    rollout_model_config = (
        rollout_data.latest_model_config if rollout_data is not None else None
    )
    resumed_model = string_value(getattr(resumed, "model", None))
    resumed_reasoning = string_value(getattr(resumed, "reasoning_effort", None))
    active_model = string_value(getattr(active_instance, "model", None))
    active_reasoning = string_value(
        getattr(active_instance, "reasoning_effort", None)
    )
    active_config_present = active_instance is not None and bool(
        active_model
        or active_reasoning
        or getattr(active_instance, "plan_mode", False)
    )
    resumed_config_present = bool(resumed_model or resumed_reasoning)
    needs_recorded_model_config = not (
        active_config_present or resumed_config_present
    )
    active_default_model_unresolved = bool(
        active_instance is not None
        and active_reasoning
        and not active_model
        and not resumed_model
        and not getattr(active_instance, "plan_mode", False)
    )
    if not history_paginated and rollout_model_config is None and (
        needs_recorded_model_config or active_default_model_unresolved
    ):
        rollout_path = _rollout_path_for(thread)
        if rollout_path is None and metadata is not None:
            rollout_path = _rollout_path_from_value(metadata.codex_path)
        if rollout_path is not None:
            rollout_model_config = rollout.latest_model_config(rollout_path)
    stored_model_config = None
    if rollout_model_config is None and needs_recorded_model_config:
        stored_model_config = _stored_model_config_for_session(session_id)
    session_model, session_reasoning = _session_model_and_reasoning(
        resumed,
        active_instance=active_instance,
        rollout_config=rollout_model_config,
        stored_config=stored_model_config,
    )
    response = render(
        request,
        "session.html",
        {
            "thread": _session_template_thread(thread),
            "entries": entries,
            "history_partial": history_paginated,
            "history_next_url": (
                _session_history_url(
                    session_id,
                    before_offset=history_page.start_offset,
                    partial_record_end=history_page.partial_record_end,
                    newer_turn_continues=bool(
                        history_page.flat_entries
                        and history_page.flat_entries[0].get("kind") != "user"
                    ),
                    active_transcript_owner=active_transcript_owner,
                )
                if history_page is not None and history_page.has_older
                else ""
            ),
            "display_title": display_title or _display_title(thread),
            "read_only": read_only,
            "system_prompt": system_prompt,
            "name_value": name_value,
            "name_max_len": _NAME_MAX_LEN,
            "display_title_max_len": session_index.DISPLAY_TITLE_MAX_LEN,
            "set_name_url": reverse("set_session_name", kwargs={"session_id": session_id}),
            "set_archived_url": reverse(
                "set_session_archived", kwargs={"session_id": session_id}
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
                session_id,
                active_instance,
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
            "show_active_worker_transcript": show_active_worker_transcript,
            "rollout_owns_active_turn": rollout_owns_active_turn,
            "pr_watch_progress": pr_watch_progress,
            "active_worker_status_text": active_worker_status_text,
            "latest_user_turn_failure": latest_user_turn_failure,
            "live_status_text": live_status_text,
            # Carried into the Stop button so the click targets the
            # specific worker the page is streaming, not "whichever
            # worker is latest at click time" — overlapping turns can
            # stack two active workers on the same thread.
            "active_instance": active_instance,
            # The in-progress turn is trimmed from ``entries`` above, so the
            # user wouldn't see their own message at all without a pending
            # bubble while the stream catches up.
            "pending_user_prompt": (
                ""
                if rollout_owns_active_turn
                else _pending_user_prompt(active_instance or unstarted_instance)
            ),
            "pending_user_author": _pending_user_author(active_instance or unstarted_instance),
            "pending_user_timestamp": _pending_user_timestamp(active_instance or unstarted_instance),
            "pending_accepted_proposal_context": (
                active_accepted_proposal_context
                if not rollout_owns_active_turn
                else None
            ),
            "live_accepted_proposal_context": active_accepted_proposal_context,
            "token_usage": session_token_usage,
            "session_model": session_model,
            "session_reasoning": session_reasoning,
            "next_message_config": _next_message_config(
                settings,
                session_model,
                session_reasoning,
                plan_model,
                cwd=thread_cwd or "",
                approval_mode=approval_mode,
            ),
            "input_image_accept": _INPUT_IMAGE_ACCEPT,
            "pr_slash_prompt": PR_SLASH_DISPLAY_PROMPT,
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


def _session_history_url(
    session_id: str,
    *,
    before_offset: int,
    partial_record_end: int | None = None,
    newer_turn_continues: bool = False,
    active_transcript_owner: str = "",
) -> str:
    params: dict[str, str | int] = {"before": before_offset}
    if partial_record_end is not None:
        params["record_end"] = partial_record_end
    if newer_turn_continues:
        params["newer_turn"] = "continued"
    if active_transcript_owner:
        params["transcript_owner"] = active_transcript_owner
    return "{}?{}".format(
        reverse("session_history", kwargs={"session_id": session_id}),
        urlencode(params),
    )

def _thread_read_missing_or_invalid(exc: InvalidRequestError) -> bool:
    message = exc.message.lower()
    return (
        "invalid thread id" in message
        or "invalid session id" in message
        or "thread not loaded:" in message
        or bool(
            re.search(
                r"\bthread(?:\s+id)?(?:\s+\S+)?\s+not found\b",
                message,
            )
        )
    )

def _thread_read_temporarily_unavailable(
    exc: InternalRpcError | InvalidRequestError,
) -> bool:
    message = exc.message.lower()
    return (
        "no rollout found for thread id" in message
        or "thread not loaded:" in message
        or (
            "invalid paginated history lineage for " in message
            and "missing source rollout" in message
        )
        or "already has an active writer" in message
        or (
            "failed to read thread" in message
            and "rollout at " in message
            and (
                " is empty" in message
                or "does not start with session metadata" in message
            )
        )
    )

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

def _attach_lazy_intermediate_context(
    entries: list[dict[str, Any]],
    *,
    session_id: str,
    enabled: bool,
    cache_entries: bool,
    rollout_state: _RolloutFileState | None,
) -> None:
    if not enabled or rollout_state is None:
        return
    for entry_index, entry in enumerate(entries):
        if entry.get("kind") != "intermediate":
            continue
        # An active rollout changes throughout the turn. Caching its full
        # intermediate blocks would retain a new, potentially very large copy
        # for every observed mtime until the process-wide LRU evicted it.
        if cache_entries:
            _cache_intermediate_detail(
                session_id=session_id,
                rollout_state=rollout_state,
                entry_index=entry_index,
                entry=entry,
            )
        entry["lazy_url"] = (
            reverse(
                "session_intermediate",
                kwargs={"session_id": session_id, "entry_index": entry_index},
            )
        )
        entry["item_count"] = len(entry.get("items", []))
        entry["items"] = []
        entry["earlier_items"] = []

def _intermediate_detail_cache_key(
    *,
    session_id: str,
    rollout_state: _RolloutFileState,
    entry_index: int,
) -> tuple[str, str, int, int]:
    return (
        session_id,
        str(rollout_state.path),
        rollout_state.mtime_ns,
        entry_index,
    )

def _cache_intermediate_detail(
    *,
    session_id: str,
    rollout_state: _RolloutFileState,
    entry_index: int,
    entry: dict[str, Any],
) -> None:
    key = _intermediate_detail_cache_key(
        session_id=session_id,
        rollout_state=rollout_state,
        entry_index=entry_index,
    )
    cached_entry = {
        "kind": "intermediate",
        "summary": entry.get("summary", ""),
        "reasoning_count": entry.get("reasoning_count", 0),
        "command_count": entry.get("command_count", 0),
        "web_search_count": entry.get("web_search_count", 0),
        "item_count": entry.get("item_count", 0),
        "items": entry.get("items", []),
        "earlier_items": entry.get("earlier_items", []),
        "latest_item": entry.get("latest_item"),
    }
    with _INTERMEDIATE_DETAIL_CACHE_LOCK:
        _INTERMEDIATE_DETAIL_CACHE[key] = cached_entry
        _INTERMEDIATE_DETAIL_CACHE.move_to_end(key)
        while len(_INTERMEDIATE_DETAIL_CACHE) > _INTERMEDIATE_DETAIL_CACHE_MAX_SIZE:
            _INTERMEDIATE_DETAIL_CACHE.popitem(last=False)

def _metadata_rows_for_usage() -> list[SessionMetadata]:
    return list(
        SessionMetadata.objects.exclude(codex_updated_at__isnull=True).only(
            "thread_id",
            "codex_path",
            "codex_thread_source",
            "project",
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
        with app_server_pool.borrow_codex(
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
    model: str,
    reasoning: str,
    plan_model: str | None,
    *,
    cwd: str,
    approval_mode: str | None = None,
) -> list[dict[str, str]]:
    """Return the settings that will govern the next submitted message."""
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
        {"label": "model", "value": model, "plan_value": plan_model_value},
        {
            "label": "reasoning",
            "value": reasoning,
            "plan_value": _PLAN_MODE_REASONING_EFFORT.value,
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


def _session_model_and_reasoning(
    resumed: Any,
    *,
    active_instance: Any | None = None,
    rollout_config: rollout.SessionModelConfig | None = None,
    stored_config: rollout.SessionModelConfig | None = None,
) -> tuple[str, str]:
    """Return the effective session model configuration from known state."""
    resumed_model = string_value(getattr(resumed, "model", None))
    resumed_reasoning = string_value(getattr(resumed, "reasoning_effort", None))
    active_model = string_value(getattr(active_instance, "model", None))
    active_reasoning = string_value(
        getattr(active_instance, "reasoning_effort", None)
    )
    if active_instance is not None and getattr(active_instance, "plan_mode", False):
        return active_model or "Unknown", _PLAN_MODE_REASONING_EFFORT.value
    if active_model or active_reasoning:
        # A blank active model means Codex resolves its default for this turn.
        if active_reasoning and not active_model:
            active_model = resumed_model or string_value(
                getattr(rollout_config, "model", None)
            )
        return (
            active_model or "Unknown",
            active_reasoning or "Model default",
        )

    if resumed_model or resumed_reasoning:
        return resumed_model or "Unknown", resumed_reasoning or "Model default"
    if rollout_config is not None:
        return (
            rollout_config.model or "Unknown",
            rollout_config.reasoning_effort or "Model default",
        )
    if stored_config is not None:
        return (
            stored_config.model or "Unknown",
            stored_config.reasoning_effort or "Model default",
        )
    return "Unknown", "Model default"

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

def _stream_url_for(
    session_id: str,
    active_instance: CodexInstance | None,
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
    query = {
        "baseline": str(baseline_id) if baseline_id is not None else "",
        "active": str(active_id) if active_id is not None else "",
    }
    qs = urlencode(query)
    return f"{reverse('session_stream', kwargs={'session_id': session_id})}?{qs}"

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
        with app_server_pool.borrow_codex(
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
        return caches._models_data_from_codex(codex)
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
