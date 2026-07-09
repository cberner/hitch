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
from django.core import signing
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
from openai_codex.errors import InvalidRequestError
from openai_codex.generated.v2_all import (
    ReasoningEffort,
    SortDirection,
    ThreadSortKey,
)

from hitch.main import caches, coding_agents, demo
from hitch.main import repos as repos_module
from hitch.main import worktrees as worktrees_module
from hitch.main.diffs import build_worktree_diff
from hitch.main.goals.autonomous_goal_proposal_stack import _proposal_outcome_metadata
from hitch.main.local_merges import local_branch_names as local_branch_names
from hitch.main.models import (
    ApprovalRequest,
    CodexInstance,
    Project,
    ProposedSession,
    SessionMetadata,
    SystemWorkflow,
)
from hitch.main.repos import git_common_dir as git_common_dir
from hitch.main.repos import same_repo_or_worktree
from hitch.main.runtime import app_server_pool, codex_events, codex_pool, reconciliation, streaming
from hitch.main.runtime.input_images import (
    _INPUT_IMAGE_ACCEPT,
    _INPUT_IMAGE_FIELD,
    _INPUT_IMAGE_MAX_BYTES,
    _INPUT_IMAGE_MAX_COUNT,
)
from hitch.main.runtime.rollout_state import (
    _rollout_file_state_from_value,
    _rollout_mtime_ns,
    _rollout_path_for,
    _rollout_path_from_value,
    _RolloutFileState,
    _thread_is_archived,
)
from hitch.main.runtime.sdk_values import (
    string_value,
)
from hitch.main.sessions import session_index, session_stage, token_usage
from hitch.main.sessions.message_intent import (
    _FIX_PR_SLASH_COMMAND,
)
from hitch.main.sessions.project_visibility import (
    _filter_proposed_sessions_by_project_visibility,
)
from hitch.main.sessions.project_visibility import (
    _metadata_by_thread_id as _metadata_by_thread_id,
)
from hitch.main.sessions.session_entry_display import (
    _active_instance_for,
    _active_worker_status_text,
    _apply_qa_approval_messages,
    _apply_system_authors,
    _display_title,
    _entries_for,
    _filter_demo_agent_entries,
    _pending_user_author,
    _pending_user_prompt,
    _pending_user_timestamp,
    _show_active_worker_transcript,
    _task_plan_context,
    _trim_in_progress_turn,
    _workflow_accepts_active_turn_steering,
    _workflow_accepts_qa_pause_steering,
    _workflow_composer_label,
    _workflow_status_text,
)
from hitch.main.sessions.session_pr_plan import (
    _PR_SLASH_PROMPT,
    _ROLLOUT_COLLABORATION_MODE_NOT_PROVIDED,
    _current_pr_url_for_thread,
    _mark_pending_plan_actions,
    _pr_observation_result_for_thread,
    _thread_plan_mode_state,
    _workflow_after_main_lifecycle,
)
from hitch.main.sessions.session_resume import (
    _metadata_resume_for_inactive_session,
    _session_detail_metadata,
)
from hitch.main.sessions.session_settings import (
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
    _schedule_pr_stage_refresh,
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
    _effective_coding_agent,
    _option_label,
    _web_search_mode_label,
)
from hitch.main.sessions.system_agent_summary import (
    _demo_system_session_url,
)
from hitch.main.workflows import autonomous_goals as goal_workflows
from hitch.main.workflows import pr_qa, pr_stage, system_agents
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

_NAME_MAX_LEN = 200

_PROJECT_NAME_MAX_LEN = 200

_MAX_BIGAUTOFIELD = 2**63 - 1

_PLAN_APPROVAL_PROMPT = "Implement the plan."

_PLAN_REVISION_PROMPT = "Revise the plan."

_PLAN_MODE_REASONING_EFFORT = ReasoningEffort.medium.value

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
    rate_limits = caches._rate_limits_for_usage_context(
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
    if session_index_state.totals_available:
        token_usage._schedule_usage_token_refresh(usage_metadata)
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
    hide_demo_agent_entries: bool = True,
    demo_entries_run_id: int | None = None,
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
            app_server_pool.run_borrowed_op_with_retry(
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
        workflow_pr_snapshot = pr_qa.pr_handoff_for_workflow(stage_pr_workflow)
        # Only flag refreshing when the PR stage is the one actually displayed.
        # An active worker or a waiting-for-input session shows its own stage, so
        # marking that live badge refreshing would let the reload script tear
        # down the running EventSource transcript.
        pr_stage_displayed = active_instance is None and not awaiting_user_input
        stage_refreshing = pr_stage_displayed and (
            pr_qa.pr_handoff_stage_refresh_due(stage_pr_workflow)
            or pr_qa.pr_monitor_backoff_stage_refresh_due(stage_pr_workflow)
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
            if pr_qa.pr_snapshot_stage_refresh_due(
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
            "display_title_max_len": session_index.DISPLAY_TITLE_MAX_LEN,
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

def _session_intermediate_demo_context(session_id: str, run_id: int) -> str:
    return signing.dumps(
        {"session_id": session_id, "run_id": run_id},
        salt=_SESSION_INTERMEDIATE_DEMO_CONTEXT_SALT,
    )

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


def _base_instructions_for_follow_up(
    settings: SettingsValues, previous_instance: CodexInstance | None
) -> str | None:
    if previous_instance is not None:
        return previous_instance.base_instructions or None
    return _base_instructions_for_settings(settings, explicit_default=True)


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
