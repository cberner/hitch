import base64
import binascii
import json
import logging
import re
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlencode

from django.contrib.auth import login as auth_login
from django.contrib.auth import logout as auth_logout
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.db.models import Prefetch
from django.http import (
    Http404,
    HttpRequest,
    HttpResponse,
    HttpResponseBadRequest,
    StreamingHttpResponse,
)
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils import timezone
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from openai_codex import AppServerError, Codex
from openai_codex.generated.v2_all import (
    GetAccountRateLimitsResponse,
    RateLimitSnapshot,
    ReasoningEffort,
)

from hitch.main import (
    codex_events,
    codex_pool,
    coding_agents,
    demo,
    rollout,
    streaming,
    system_agents,
)
from hitch.main.diffs import build_worktree_diff
from hitch.main.formatting import looks_like_markdown, render_markdown
from hitch.main.models import (
    ApprovalRequest,
    ArchivedSessionTokenUsage,
    CodexInstance,
    KeyResult,
    Objective,
    Project,
    ProposedSession,
    ProposedTask,
    SessionMetadata,
    StandingOrder,
    SystemAgentRun,
    SystemWorkflow,
    UserInputRequest,
    UserSettings,
)
from hitch.main.repos import discover_repos, git_common_dir, same_repo_or_worktree
from hitch.main.worktrees import (
    WorktreeCleanupError,
    WorktreeCreationError,
    cleanup_worktree,
    create_worktree_for_session,
    discover_managed_worktrees,
)

logger = logging.getLogger(__name__)


class SettingsValues(NamedTuple):
    model: str
    reasoning_effort: str
    sandbox_policy: str
    approval_mode: str
    coding_agent: str
    extra_system_prompt: str
    use_worktrees: bool
    auto_pr_enabled: bool
    show_archived_sessions: bool
    last_selected_repo: str
    selected_project_id: int | None
    enable_memories: bool


class ResolvedSettings(NamedTuple):
    values: SettingsValues
    cookie_updates: dict[str, str]


class StandingOrderValues(NamedTuple):
    title: str
    goal: str
    ambition: str
    confidence_threshold: str


class _MessageIntent(NamedTuple):
    prompt: str
    plan_mode: bool
    allow_pending_plan_default: bool
    explicit_plan_mode: bool


class _NewSessionTarget(NamedTuple):
    cwd: str
    project: Project | None
    project_cleared: bool


# Sandbox-policy variants offered in the settings dialog. Stored as the
# SandboxPolicy ``type`` discriminator string so the cookie value can map
# 1:1 onto a constructed SandboxPolicy in the worker without any further
# translation. We omit ``externalSandbox`` because it requires a host
# sandbox the in-process worker doesn't speak; the three values below are
# the union of variants the codex CLI ships out of the box.
_SANDBOX_POLICY_OPTIONS: tuple[tuple[str, str], ...] = (
    ("readOnly", "Read only"),
    ("workspaceWrite", "Workspace write"),
    ("dangerFullAccess", "Danger - full access"),
)
_VALID_SANDBOX_POLICIES = {value for value, _ in _SANDBOX_POLICY_OPTIONS}

# Approval modes the dialog offers. ``auto_review`` and ``deny_all`` map 1:1
# onto the SDK's ``ApprovalMode`` enum. The ``prompt_user`` and
# ``approve_all`` modes are custom worker modes that force escalations to
# the client transport with the ``user`` reviewer; the worker either surfaces
# a browser prompt or rubber-stamps the request depending on the selected
# mode. ``auto_review`` is also the SDK's own default; keeping it first here
# makes it the safe default the dialog selects when no cookie has been
# written yet.
_APPROVAL_MODE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("auto_review", "Auto review (default)"),
    ("prompt_user", "Always prompt for approval"),
    ("deny_all", "Deny all escalations"),
    ("approve_all", "Approve all (dangerous)"),
)
_VALID_APPROVAL_MODES = {value for value, _ in _APPROVAL_MODE_OPTIONS}
# Safe default: ``auto_review`` matches the SDK default and keeps an
# automated reviewer in the loop. Tampered/legacy cookie values fall back
# to this rather than to ``deny_all``, which would silently block the
# agent from ever escalating.
_DEFAULT_APPROVAL_MODE = "auto_review"

# Dedicated signed cookies for the settings dialog. Kept separate from
# Django's session cookie so the (non-revocable, long-lived) settings
# state never rides alongside an auth session — admin auth is allowed to
# stay DB-backed and therefore revocable on logout.
_MODEL_COOKIE = "hitch_model"
_EFFORT_COOKIE = "hitch_reasoning_effort"
_SANDBOX_COOKIE = "hitch_sandbox_policy"
_APPROVAL_COOKIE = "hitch_approval_mode"
_CODING_AGENT_COOKIE = "hitch_coding_agent"
_EXTRA_SYSTEM_PROMPT_COOKIE = "hitch_extra_system_prompt"
_USE_WORKTREES_COOKIE = "hitch_use_worktrees"
_AUTO_PR_COOKIE = "hitch_auto_pr"
_SHOW_ARCHIVED_COOKIE = "hitch_show_archived_sessions"
_LAST_SELECTED_REPO_COOKIE = "hitch_last_selected_repo"
_SELECTED_PROJECT_COOKIE = "hitch_selected_project_id"
_ENABLE_MEMORIES_COOKIE = "hitch_enable_memories"
_BARE_REPO_PROJECT_VALUE = "__bare_repo__"

# Roughly one year. Long enough that a user's pick survives across
# sessions without ever needing a manual revisit; short enough that the
# browser eventually evicts a stale value if the user stops using the app.
_COOKIE_MAX_AGE = 60 * 60 * 24 * 365

# Cap on the posted model id so a crafted oversized POST can't push the
# cookie past the browser's 4KB limit (which would cause the browser to
# silently drop the cookie). Real Codex model ids are tens of chars; 256
# is comfortably more than that without leaving room for abuse.
_MODEL_MAX_LEN = 256

# The prompt is base64-encoded inside a signed cookie, so keep enough room for
# encoding/signing overhead and browser cookie limits.
_EXTRA_SYSTEM_PROMPT_MAX_LEN = 2500

# Upper bound on what we render inline as a session's title. Codex does not
# generate its own thread summaries, so for unnamed threads `Thread.preview`
# (the full first user message) is what we get; that is often paragraphs
# long and would overflow the list rows without a clip.
_DISPLAY_TITLE_MAX_LEN = 80
_ARCHIVED_SESSIONS_DIR = "archived_sessions"
_MINUTES_PER_HOUR = 60
_MINUTES_PER_DAY = 24 * _MINUTES_PER_HOUR

# Server-side cap on user-supplied thread names. Matches the `maxlength` we
# set on the edit form so a client without HTML validation cannot push an
# unbounded blob through.
_NAME_MAX_LEN = 200
_PROJECT_NAME_MAX_LEN = 200
_OKR_TITLE_MAX_LEN = 200
_STANDING_ORDER_TITLE_MAX_LEN = 200
_LAST_SELECTED_REPO_MAX_LEN = 4096
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
_PR_SLASH_PROMPT = system_agents.PR_SLASH_DISPLAY_PROMPT
_PR_SLASH_FINAL_PROMPT = system_agents.PR_SLASH_PROMPT
_LEGACY_PR_SLASH_PROMPT = (
    "Do a thorough review of the diff. Rebase on master, clean it up, "
    "and then open a PR"
)
_LEGACY_PR_SLASH_FINAL_PROMPT = (
    f"{_LEGACY_PR_SLASH_PROMPT}. After opening it, poll the PR every 2 minutes "
    "until you have CI status and at least one review signal: code review "
    "comments, a thumbs up emoji on the PR, or an explicit review approval. "
    "On each poll, check whether the PR has merge conflicts. Address CI "
    "failures, review comments, merge conflicts, and any other blocking issues; "
    "push fixes and keep looping until CI, review, and mergeability are all clean. "
    "Stop and report back if any single polling iteration has no results after "
    "30 minutes."
)
_PR_PROMPT_ALIASES = frozenset(
    {
        _PR_SLASH_PROMPT,
        _PR_SLASH_FINAL_PROMPT,
        _LEGACY_PR_SLASH_PROMPT,
        _LEGACY_PR_SLASH_FINAL_PROMPT,
    }
)
_QA_SLASH_COMMAND = "/qa"
_QA_SLASH_PROMPT = system_agents.QA_SLASH_DISPLAY_PROMPT
_PLAN_MODE_REASONING_EFFORT = ReasoningEffort.medium.value
_DEFAULT_COLLABORATION_MODE = "default"
_GITHUB_PR_TOOL_RE = re.compile(
    r"(?i)(?:^|[/:\s._-])(?:github|mcp__codex_apps__github)(?:$|[/:\s._-]).*"
    r"(?:_?create[_\s-]?(?:pr|pull[_\s-]?request)|open[_\s-]?(?:pr|pull[_\s-]?request))"
)
_GITHUB_PR_URL_RE = re.compile(
    r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/pull/[0-9]+"
)

# Friendly labels for non-message thread item types. Anything not in this map
# falls back to the raw type tag so we never silently drop an item from the UI.
_NON_MESSAGE_LABELS = {
    "commandExecution": "Command",
    "mcpToolCall": "MCP tool call",
    "dynamicToolCall": "Tool call",
    "fileChange": "File change",
    "webSearch": "Web search",
    "collabAgentToolCall": "Collab agent call",
    "imageView": "Image view",
    "imageGeneration": "Image generation",
    "reasoning": "Reasoning",
    "plan": "Plan",
    "hookPrompt": "Hook prompt",
    "enteredReviewMode": "Entered review mode",
    "exitedReviewMode": "Exited review mode",
    "contextCompaction": "Context compaction",
}

_TOKEN_USAGE_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "total_tokens",
    "context_tokens",
    "model_context_window",
)


def _settings_dialog_context(
    current_settings: SettingsValues, models_data: list[Any]
) -> dict[str, Any]:
    projects = list(Project.objects.all())
    current_project = _selected_project_for_settings(current_settings, projects)
    return {
        "settings_url": reverse("update_settings"),
        "new_project_url": reverse("new_project"),
        "edit_project_url": reverse("edit_project"),
        "model_options": [
            {"id": m.id, "display_name": m.display_name} for m in models_data
        ],
        "effort_options": [effort.value for effort in ReasoningEffort],
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
        "current_model": current_settings.model,
        "current_effort": current_settings.reasoning_effort,
        "current_sandbox": current_settings.sandbox_policy,
        "current_approval": current_settings.approval_mode,
        "current_coding_agent": _effective_coding_agent(current_settings),
        "current_extra_system_prompt": current_settings.extra_system_prompt,
        "extra_system_prompt_max_len": _EXTRA_SYSTEM_PROMPT_MAX_LEN,
        "current_use_worktrees": current_settings.use_worktrees,
        "current_auto_pr": current_settings.auto_pr_enabled,
        "current_enable_memories": current_settings.enable_memories,
        "projects": projects,
        "current_project": current_project,
        "current_project_id": current_project.pk if current_project is not None else "",
        "project_name_max_len": _PROJECT_NAME_MAX_LEN,
        "project_auto_pr_options": [
            {"id": value, "display_name": label}
            for value, label in Project.AUTO_PR_CHOICES
        ],
        "project_auto_pr_follow_global": Project.AUTO_PR_FOLLOW_GLOBAL,
        "project_auto_pr_on": Project.AUTO_PR_ON,
        "project_auto_pr_off": Project.AUTO_PR_OFF,
    }


def _new_session_dialog_context(
    current_settings: SettingsValues,
    current_project: Project | None,
    projects: list[Project],
) -> dict[str, Any]:
    repos = [str(p) for p in discover_repos()]
    repo_set = set(repos)
    saved_repo = (
        current_settings.last_selected_repo
        if current_settings.last_selected_repo in repo_set
        else ""
    )
    new_session_projects = [
        project for project in projects if project.repo_path in repo_set
    ]
    current_new_session_project = _new_session_project_for_dialog(
        current_project, saved_repo, new_session_projects
    )
    current_new_session_auto_pr = _effective_auto_pr_enabled(
        current_new_session_project,
        global_enabled=current_settings.auto_pr_enabled,
    )
    return {
        "repos": repos,
        "new_session_projects": new_session_projects,
        "new_session_url": reverse("new_session"),
        "current_repo": _selected_repo_for_dialog(
            saved_repo, repos, current_new_session_project
        ),
        "current_new_session_project_id": (
            current_new_session_project.pk
            if current_new_session_project is not None
            else ""
        ),
        "current_new_session_auto_pr": current_new_session_auto_pr,
        "bare_repo_project_value": _BARE_REPO_PROJECT_VALUE,
        "pr_slash_prompt": _PR_SLASH_PROMPT,
        "qa_slash_prompt": _QA_SLASH_PROMPT,
    }


def _all_threads(codex: Codex, *, archived: bool = False) -> list[Any]:
    """Return every thread from Codex's paginated thread list."""
    threads: list[Any] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()
    while True:
        kwargs: dict[str, Any] = {}
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


def index(request: HttpRequest) -> HttpResponse:
    # Sweep workers whose pid is gone: a Popen that crashed before a worker
    # could record its terminal status (or a row stuck in ``starting``)
    # otherwise stays pending forever, since we don't run a periodic task.
    codex_pool.reconcile_dead()
    initial_settings = _stored_settings(request)
    config = codex_pool.app_server_config(
        enable_memories=initial_settings.enable_memories
    )
    with Codex(config=config) as codex:
        models_data = list(codex.models().data)
        resolved_settings = _resolved_settings(request, models_data)
        current_settings = resolved_settings.values
        cookie_updates = resolved_settings.cookie_updates
        threads = _all_threads(codex)
        if current_settings.show_archived_sessions:
            threads.extend(_all_threads(codex, archived=True))
    hidden_thread_ids = system_agents.hidden_thread_ids()
    threads = sorted(threads, key=lambda s: s.updated_at, reverse=True)
    projects = list(Project.objects.all())
    current_project = _selected_project_for_settings(current_settings, projects)
    metadata_by_thread = _metadata_by_thread_id(threads)
    sessions = []
    for thread in threads:
        if thread.id in hidden_thread_ids:
            continue
        is_archived = _thread_is_archived(thread)
        if is_archived and not current_settings.show_archived_sessions:
            continue
        session_project = _project_for_thread(thread, metadata_by_thread, projects)
        if current_project is not None and session_project != current_project:
            continue
        sessions.append(
            {
                "id": thread.id,
                "cwd": thread.cwd,
                "updated_at": thread.updated_at,
                "display_title": _display_title(thread),
                "name_value": getattr(thread, "name", None) or "",
                "is_archived": is_archived,
                "project": session_project,
            }
        )
    settings_dialog_context = _settings_dialog_context(current_settings, models_data)
    new_session_dialog_context = _new_session_dialog_context(
        current_settings, current_project, projects
    )
    response = render(
        request,
        "index.html",
        {
            "sessions": sessions,
            "has_projects": bool(projects),
            "archived_visibility_url": reverse("update_archived_session_visibility"),
            "login_url": reverse("login"),
            "register_url": reverse("register"),
            "current_show_archived_sessions": current_settings.show_archived_sessions,
            "current_project": current_project,
            "name_max_len": _NAME_MAX_LEN,
            "show_new_session_controls": True,
            **settings_dialog_context,
            **new_session_dialog_context,
        },
    )
    _apply_cookie_updates(response, cookie_updates)
    return response


@require_http_methods(["GET"])
def system_sessions(request: HttpRequest) -> HttpResponse:
    codex_pool.reconcile_dead()
    initial_settings = _stored_settings(request)
    config = codex_pool.app_server_config(
        enable_memories=initial_settings.enable_memories
    )
    with Codex(config=config) as codex:
        models_data = list(codex.models().data)
        resolved_settings = _resolved_settings(request, models_data)
        current_settings = resolved_settings.values
        cookie_updates = resolved_settings.cookie_updates
        threads = _all_threads(codex)
        if current_settings.show_archived_sessions:
            threads.extend(_all_threads(codex, archived=True))
    hidden_thread_ids = system_agents.hidden_thread_ids()
    runs_by_thread_id = _system_agent_runs_by_thread_id(hidden_thread_ids)
    threads = sorted(threads, key=lambda s: s.updated_at, reverse=True)
    projects = list(Project.objects.all())
    current_project = _selected_project_for_settings(current_settings, projects)
    metadata_by_thread = _metadata_by_thread_id(threads)
    sessions = []
    for thread in threads:
        thread_id = getattr(thread, "id", None)
        if not isinstance(thread_id, str) or thread_id not in hidden_thread_ids:
            continue
        run = runs_by_thread_id.get(thread_id)
        session_project = _project_for_thread(thread, metadata_by_thread, projects)
        if current_project is not None and session_project != current_project:
            continue
        sessions.append(
            {
                "id": thread.id,
                "cwd": thread.cwd,
                "updated_at": thread.updated_at,
                "display_title": _display_title(thread),
                "name_value": getattr(thread, "name", None) or "",
                "is_archived": _thread_is_archived(thread),
                "project": session_project,
                "detail_url": reverse(
                    "system_session", kwargs={"session_id": thread_id}
                ),
                "system_kind": _system_agent_run_label(run),
                "system_status": run.status if run is not None else "",
            }
        )
    settings_dialog_context = _settings_dialog_context(current_settings, models_data)
    response = render(
        request,
        "index.html",
        {
            "sessions": sessions,
            "has_projects": bool(projects),
            "archived_visibility_url": reverse("update_archived_session_visibility"),
            "login_url": reverse("login"),
            "register_url": reverse("register"),
            "current_show_archived_sessions": current_settings.show_archived_sessions,
            "name_max_len": _NAME_MAX_LEN,
            "system_session_list": True,
            "show_new_session_controls": False,
            **settings_dialog_context,
        },
    )
    _apply_cookie_updates(response, cookie_updates)
    return response


@require_http_methods(["GET"])
def system_session(request: HttpRequest, session_id: str) -> HttpResponse:
    run = _system_agent_run_for_thread(session_id)
    if run is None:
        raise Http404("system session not found")
    return _render_session_detail(
        request,
        session_id,
        read_only=True,
        display_title=_system_agent_run_detail_title(run),
        system_prompt=run.instance.prompt,
    )


@require_http_methods(["GET"])
def usage(request: HttpRequest) -> HttpResponse:
    initial_settings = _stored_settings(request)
    config = codex_pool.app_server_config(
        enable_memories=initial_settings.enable_memories
    )
    with Codex(config=config) as codex:
        models_data = _models_for_plan_mode_fallback(codex)
        resolved_settings = _resolved_settings(request, models_data)
        current_settings = resolved_settings.values
        cookie_updates = resolved_settings.cookie_updates
        rate_limits = _fetch_rate_limits(codex)
        usage_threads = _threads_for_usage(codex)
        lifetime_usage = (
            _lifetime_token_usage_for(usage_threads)
            if usage_threads is not None
            else None
        )
    settings_dialog_context = _settings_dialog_context(current_settings, models_data)
    response = render(
        request,
        "usage.html",
        {
            "login_url": reverse("login"),
            "register_url": reverse("register"),
            "rate_limits": rate_limits,
            "lifetime_usage": lifetime_usage,
            **settings_dialog_context,
        },
    )
    _apply_cookie_updates(response, cookie_updates)
    return response


@require_http_methods(["GET"])
def okrs(request: HttpRequest) -> HttpResponse:
    codex_pool.reconcile_dead()
    show_hidden_tasks = request.GET.get("show_hidden_tasks", "").strip() in {
        "1",
        "true",
        "on",
    }
    initial_settings = _stored_settings(request)
    config = codex_pool.app_server_config(
        enable_memories=initial_settings.enable_memories
    )
    with Codex(config=config) as codex:
        models_data = _models_for_plan_mode_fallback(codex)
        resolved_settings = _resolved_settings(request, models_data)
        current_settings = resolved_settings.values
        cookie_updates = resolved_settings.cookie_updates
        current_project = _selected_project_for_settings(current_settings)
        objectives = (
            list(
                Objective.objects.filter(project=current_project).prefetch_related(
                    Prefetch(
                        "key_results__proposed_tasks",
                        queryset=ProposedTask.objects.select_related("session"),
                    )
                )
            )
            if current_project is not None
            else []
        )
        _refresh_proposed_task_pr_state(codex, objectives)
    _attach_proposed_task_display_state(objectives, show_hidden_tasks)
    _attach_okr_task_generation_state(objectives)
    settings_dialog_context = _settings_dialog_context(current_settings, models_data)
    new_session_dialog_context = _new_session_dialog_context(
        current_settings, current_project, settings_dialog_context["projects"]
    )
    response = render(
        request,
        "okrs.html",
        {
            "login_url": reverse("login"),
            "register_url": reverse("register"),
            "objectives": objectives,
            "objective_create_url": reverse("create_objective"),
            "proposed_task_rejected_status": ProposedTask.OUTCOME_REJECTED,
            "show_hidden_tasks": show_hidden_tasks,
            "title_max_len": _OKR_TITLE_MAX_LEN,
            **settings_dialog_context,
            **new_session_dialog_context,
        },
    )
    _apply_cookie_updates(response, cookie_updates)
    return response


@require_http_methods(["GET"])
def okr_task_generation_log(request: HttpRequest, workflow_id: int) -> HttpResponse:
    workflow = _okr_task_generation_workflow_for_log(request, workflow_id)
    run = (
        workflow.agent_runs.exclude(thread_id="")
        .order_by("-created_at")
        .first()
    )
    if run is None:
        raise Http404("task generation log not found")
    return _render_session_detail(
        request,
        run.thread_id,
        read_only=True,
        display_title="Task generation log",
    )


def _validated_okr_title(raw_title: str) -> tuple[str, str | None]:
    title = raw_title.strip()
    if not title:
        return "", "title is required"
    if len(title) > _OKR_TITLE_MAX_LEN:
        return "", "title is too long"
    return title, None


def _proposed_task_session_prompt(
    task: ProposedTask, key_result: KeyResult, objective: Objective
) -> str:
    parts = [
        "Do this ProposedTask.",
        "",
        "This task is part of the following Key Result (KR), which is part of "
        "the following Objective.",
        "",
        f"Objective: {objective.title}",
    ]
    if objective.description:
        parts.extend(["", f"Objective description:\n{objective.description}"])
    parts.extend(["", f"Key Result: {key_result.title}"])
    if key_result.description:
        parts.extend(["", f"Key Result description:\n{key_result.description}"])
    if key_result.work_instructions:
        parts.extend(
            ["", f"Key Result work instructions:\n{key_result.work_instructions}"]
        )
    parts.extend(
        [
            "",
            "There will be other tasks to complete the rest of this Key Result. "
            "Only do this part, even if the result seems incomplete without the "
            "other tasks.",
            "",
            f"Title: {task.title}",
        ]
    )
    if task.description:
        parts.extend(["", f"Description:\n{task.description}"])
    if task.success_criteria:
        parts.extend(["", f"Success criteria:\n{task.success_criteria}"])
    if task.rationale:
        parts.extend(["", f"Rationale:\n{task.rationale}"])
    return "\n".join(parts)


def _attach_proposed_task_display_state(
    objectives: list[Objective], show_hidden_tasks: bool
) -> None:
    for objective in objectives:
        for key_result in objective.key_results.all():
            tasks = list(key_result.proposed_tasks.all())
            for task in tasks:
                task.session_prompt = _proposed_task_session_prompt(  # type: ignore[attr-defined]
                    task, key_result, objective
                )
            visible_tasks = (
                tasks
                if show_hidden_tasks
                else [task for task in tasks if not task.outcome_status]
            )
            key_result.visible_proposed_tasks = visible_tasks  # type: ignore[attr-defined]
            key_result.hidden_proposed_task_count = len(tasks) - len(  # type: ignore[attr-defined]
                visible_tasks
            )


def _refresh_proposed_task_pr_state(codex: Codex, objectives: list[Objective]) -> None:
    tasks_by_thread: dict[str, list[ProposedTask]] = {}
    for objective in objectives:
        for key_result in objective.key_results.all():
            for task in key_result.proposed_tasks.all():
                if (
                    task.session_id is None
                    or task.outcome_status
                    not in {
                        ProposedTask.OUTCOME_ACCEPTED,
                        ProposedTask.OUTCOME_PR_OPENED,
                    }
                ):
                    continue
                session = task.session
                if session is None:
                    continue
                tasks_by_thread.setdefault(session.thread_id, []).append(task)
    for thread_id, tasks in tasks_by_thread.items():
        try:
            thread = codex._client.thread_resume(thread_id).thread
        except AppServerError:
            logger.info("could not refresh PR state for proposed task session %s", thread_id)
            continue
        pr_url = _pr_url_for_thread(thread)
        _mark_proposed_tasks_pr_opened(thread_id, pr_url, tasks)


def _mark_proposed_tasks_pr_opened(
    session_id: str, pr_url: str | None, tasks: list[ProposedTask] | None = None
) -> None:
    if not pr_url:
        return
    if tasks is None:
        ProposedTask.objects.filter(
            session__thread_id=session_id,
            outcome_status__in=[
                ProposedTask.OUTCOME_ACCEPTED,
                ProposedTask.OUTCOME_PR_OPENED,
            ],
        ).update(
            outcome_status=ProposedTask.OUTCOME_PR_OPENED,
            pr_url=pr_url,
            updated_at=timezone.now(),
        )
        return
    for task in tasks:
        if task.outcome_status not in {
            ProposedTask.OUTCOME_ACCEPTED,
            ProposedTask.OUTCOME_PR_OPENED,
        }:
            continue
        if (
            task.outcome_status == ProposedTask.OUTCOME_PR_OPENED
            and task.pr_url == pr_url
        ):
            continue
        task.outcome_status = ProposedTask.OUTCOME_PR_OPENED
        task.pr_url = pr_url
        task.save(update_fields=["outcome_status", "pr_url", "updated_at"])


def _redirect_to_okrs_from_post(request: HttpRequest) -> HttpResponse:
    if request.POST.get("show_hidden_tasks", "").strip() in {"1", "true", "on"}:
        return redirect(f"{reverse('okrs')}?{urlencode({'show_hidden_tasks': '1'})}")
    return redirect("okrs")


@require_http_methods(["GET"])
def standing_orders(request: HttpRequest) -> HttpResponse:
    codex_pool.reconcile_dead()
    initial_settings = _stored_settings(request)
    config = codex_pool.app_server_config(
        enable_memories=initial_settings.enable_memories
    )
    with Codex(config=config) as codex:
        models_data = list(codex.models().data)
        resolved_settings = _resolved_settings(request, models_data)
        current_settings = resolved_settings.values
        cookie_updates = resolved_settings.cookie_updates
    projects = list(Project.objects.all())
    current_project = _selected_project_for_settings(current_settings, projects)
    orders = (
        list(StandingOrder.objects.filter(project=current_project))
        if current_project is not None
        else []
    )
    inbox = (
        list(
            ProposedSession.objects.filter(
                standing_order__project=current_project,
                outcome_status=ProposedSession.OUTCOME_UNSET,
            )
            .select_related(
                "standing_order",
                "candidate_session",
                "judge_session",
                "source_workflow",
            )
            .order_by("created_at", "id")
        )
        if current_project is not None
        else []
    )
    _attach_standing_order_run_state(orders)
    _attach_proposed_session_display_state(inbox)
    settings_dialog_context = _settings_dialog_context(current_settings, models_data)
    new_session_dialog_context = _new_session_dialog_context(
        current_settings, current_project, settings_dialog_context["projects"]
    )
    response = render(
        request,
        "standing_orders.html",
        {
            "login_url": reverse("login"),
            "register_url": reverse("register"),
            "current_project": current_project,
            "standing_orders": orders,
            "proposed_sessions": inbox,
            "standing_order_create_url": reverse("create_standing_order"),
            "standing_order_run_all_url": reverse("run_standing_orders"),
            "ambition_choices": StandingOrder.AMBITION_CHOICES,
            "default_ambition": StandingOrder.AMBITION_INCREMENTAL,
            "confidence_choices": StandingOrder.CONFIDENCE_CHOICES,
            "default_confidence": StandingOrder.CONFIDENCE_HIGH,
            "proposed_session_rejected_status": ProposedSession.OUTCOME_REJECTED,
            "proposed_session_dismissed_status": ProposedSession.OUTCOME_DISMISSED,
            "title_max_len": _STANDING_ORDER_TITLE_MAX_LEN,
            **settings_dialog_context,
            **new_session_dialog_context,
        },
    )
    _apply_cookie_updates(response, cookie_updates)
    return response


@require_http_methods(["POST"])
def create_standing_order(request: HttpRequest) -> HttpResponse:
    project = _active_project_from_request(request)
    if project is None:
        return HttpResponseBadRequest("active project is required")
    values, error = _validated_standing_order_values(request)
    if error is not None:
        return HttpResponseBadRequest(error)
    assert values is not None
    StandingOrder.objects.create(
        project=project,
        title=values.title,
        goal=values.goal,
        ambition=values.ambition,
        confidence_threshold=values.confidence_threshold,
    )
    return redirect("standing_orders")


@require_http_methods(["POST"])
def edit_standing_order(request: HttpRequest, standing_order_id: int) -> HttpResponse:
    project = _active_project_from_request(request)
    if project is None:
        return HttpResponseBadRequest("active project is required")
    standing_order = StandingOrder.objects.filter(
        pk=standing_order_id,
        project=project,
    ).first()
    if standing_order is None:
        raise Http404("standing order not found")
    values, error = _validated_standing_order_values(request)
    if error is not None:
        return HttpResponseBadRequest(error)
    assert values is not None

    updates: list[str] = []
    for field in ("title", "goal", "ambition", "confidence_threshold"):
        value = getattr(values, field)
        if getattr(standing_order, field) != value:
            setattr(standing_order, field, value)
            updates.append(field)
    if updates:
        standing_order.save(update_fields=[*updates, "updated_at"])
    return redirect("standing_orders")


@require_http_methods(["POST"])
def run_standing_orders(request: HttpRequest) -> HttpResponse:
    project = _active_project_from_request(request)
    if project is None:
        return HttpResponseBadRequest("active project is required")
    for standing_order in StandingOrder.objects.filter(project=project):
        system_agents.start_standing_order_workflow(standing_order=standing_order)
    return redirect("standing_orders")


@require_http_methods(["GET"])
def standing_order_run_log(request: HttpRequest, workflow_id: int) -> HttpResponse:
    workflow = _standing_order_workflow_for_log(request, workflow_id)
    run = workflow.agent_runs.exclude(thread_id="").order_by("-created_at").first()
    if run is None:
        raise Http404("standing order run log not found")
    return _render_session_detail(
        request,
        run.thread_id,
        read_only=True,
        display_title="Standing order run log",
    )


@require_http_methods(["POST"])
def update_proposed_session_outcome(
    request: HttpRequest, proposed_session_id: int
) -> HttpResponse:
    project = _active_project_from_request(request)
    if project is None:
        return HttpResponseBadRequest("active project is required")
    if proposed_session_id < 1 or proposed_session_id > _MAX_BIGAUTOFIELD:
        return HttpResponseBadRequest("proposed session is required")
    proposed_session = (
        ProposedSession.objects.select_related(
            "standing_order__project",
            "candidate_session",
        )
        .filter(pk=proposed_session_id, standing_order__project=project)
        .first()
    )
    if proposed_session is None:
        return HttpResponseBadRequest("proposed session is required")
    outcome_status = request.POST.get("outcome_status", "")
    valid_statuses = {choice[0] for choice in ProposedSession.OUTCOME_CHOICES}
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
    if (
        proposed_session.inbox_kind == ProposedSession.INBOX_KIND_PROPOSAL
        and outcome_status == ProposedSession.OUTCOME_DISMISSED
    ):
        return HttpResponseBadRequest("outcome status is invalid")
    proposed_session.outcome_status = outcome_status
    proposed_session.outcome_notes = outcome_notes
    update_fields = ["outcome_status", "outcome_notes", "updated_at"]
    if outcome_status == ProposedSession.OUTCOME_ACCEPTED:
        proposed_session.accepted_session = proposed_session.candidate_session
        update_fields.append("accepted_session")
    proposed_session.save(update_fields=update_fields)
    return redirect("standing_orders")


def _validated_standing_order_title(raw_title: str) -> tuple[str, str | None]:
    title = raw_title.strip()
    if not title:
        return "", "title is required"
    if len(title) > _STANDING_ORDER_TITLE_MAX_LEN:
        return "", "title is too long"
    return title, None


def _validated_standing_order_values(
    request: HttpRequest,
) -> tuple[StandingOrderValues | None, str | None]:
    title, error = _validated_standing_order_title(request.POST.get("title", ""))
    if error is not None:
        return None, error
    goal = request.POST.get("goal", "").strip()
    if not goal:
        return None, "goal is required"
    ambition = request.POST.get("ambition", "").strip()
    valid_ambitions = {value for value, _label in StandingOrder.AMBITION_CHOICES}
    if ambition not in valid_ambitions:
        return None, "ambition is invalid"
    threshold = request.POST.get("confidence_threshold", "").strip()
    valid_thresholds = {value for value, _label in StandingOrder.CONFIDENCE_CHOICES}
    if threshold not in valid_thresholds:
        return None, "confidence threshold is invalid"
    return StandingOrderValues(
        title=title,
        goal=goal,
        ambition=ambition,
        confidence_threshold=threshold,
    ), None


@require_http_methods(["POST"])
def create_objective(request: HttpRequest) -> HttpResponse:
    project = _active_project_from_request(request)
    if project is None:
        return HttpResponseBadRequest("active project is required")
    title, error = _validated_okr_title(request.POST.get("title", ""))
    if error is not None:
        return HttpResponseBadRequest(error)
    Objective.objects.create(
        project=project,
        title=title,
        description=request.POST.get("description", "").strip(),
    )
    return redirect("okrs")


@require_http_methods(["POST"])
def create_key_result(request: HttpRequest, objective_id: int) -> HttpResponse:
    project = _active_project_from_request(request)
    if project is None:
        return HttpResponseBadRequest("active project is required")
    if objective_id < 1 or objective_id > _MAX_BIGAUTOFIELD:
        return HttpResponseBadRequest("objective is required")
    objective = Objective.objects.filter(pk=objective_id, project=project).first()
    if objective is None:
        return HttpResponseBadRequest("objective is required")
    title, error = _validated_okr_title(request.POST.get("title", ""))
    if error is not None:
        return HttpResponseBadRequest(error)
    KeyResult.objects.create(
        objective=objective,
        title=title,
        description=request.POST.get("description", "").strip(),
        work_instructions=request.POST.get("work_instructions", "").strip(),
    )
    return redirect("okrs")


@require_http_methods(["POST"])
def generate_key_result_tasks(request: HttpRequest, key_result_id: int) -> HttpResponse:
    project = _active_project_from_request(request)
    if project is None:
        return HttpResponseBadRequest("active project is required")
    if key_result_id < 1 or key_result_id > _MAX_BIGAUTOFIELD:
        return HttpResponseBadRequest("key result is required")
    key_result = (
        KeyResult.objects.select_related("objective__project")
        .filter(pk=key_result_id, objective__project=project)
        .first()
    )
    if key_result is None:
        return HttpResponseBadRequest("key result is required")
    try:
        system_agents.start_okr_task_generation_workflow(key_result=key_result)
    except KeyResult.DoesNotExist:
        return HttpResponseBadRequest("key result is required")
    return redirect("okrs")


@require_http_methods(["POST"])
def update_proposed_task_outcome(request: HttpRequest, task_id: int) -> HttpResponse:
    project = _active_project_from_request(request)
    if project is None:
        return HttpResponseBadRequest("active project is required")
    if task_id < 1 or task_id > _MAX_BIGAUTOFIELD:
        return HttpResponseBadRequest("proposed task is required")
    task = (
        ProposedTask.objects.select_related("key_result__objective__project")
        .filter(pk=task_id, key_result__objective__project=project)
        .first()
    )
    if task is None:
        return HttpResponseBadRequest("proposed task is required")
    outcome_status = request.POST.get("outcome_status", "")
    valid_statuses = {choice[0] for choice in ProposedTask.OUTCOME_CHOICES}
    if outcome_status not in valid_statuses:
        return HttpResponseBadRequest("outcome status is invalid")
    outcome_notes = request.POST.get(
        "reason", request.POST.get("outcome_notes", "")
    ).strip()
    if outcome_status == ProposedTask.OUTCOME_REJECTED and not outcome_notes:
        return HttpResponseBadRequest("reason is required")
    task.outcome_status = outcome_status
    task.outcome_notes = outcome_notes
    task.save(update_fields=["outcome_status", "outcome_notes", "updated_at"])
    return _redirect_to_okrs_from_post(request)


def _attach_okr_task_generation_state(objectives: list[Objective]) -> None:
    key_result_ids = [
        key_result.pk
        for objective in objectives
        for key_result in objective.key_results.all()
    ]
    if not key_result_ids:
        return
    workflows = (
        SystemWorkflow.objects.filter(
            kind=system_agents.OKR_TASK_AGENT_KIND,
            main_thread_id__in=[
                system_agents._okr_task_main_thread_id(key_result_id)
                for key_result_id in key_result_ids
            ],
        )
        .order_by("main_thread_id", "-created_at")
    )
    workflows_by_thread: dict[str, SystemWorkflow] = {}
    for workflow in workflows:
        workflows_by_thread.setdefault(workflow.main_thread_id, workflow)
    log_urls_by_workflow_id = _okr_task_generation_log_urls(
        workflows_by_thread.values()
    )
    for objective in objectives:
        for key_result in objective.key_results.all():
            latest_workflow: SystemWorkflow | None = workflows_by_thread.get(
                system_agents._okr_task_main_thread_id(key_result.pk)
            )
            key_result.task_generation_workflow = latest_workflow  # type: ignore[attr-defined]
            key_result.task_generation_running = (  # type: ignore[attr-defined]
                latest_workflow is not None
                and latest_workflow.status == SystemWorkflow.STATUS_RUNNING
            )
            key_result.task_generation_log_url = (  # type: ignore[attr-defined]
                log_urls_by_workflow_id.get(latest_workflow.pk)
                if latest_workflow is not None
                else ""
            )


def _okr_task_generation_log_urls(
    workflows: Iterable[SystemWorkflow],
) -> dict[int, str]:
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
            reverse("okr_task_generation_log", kwargs={"workflow_id": run.workflow_id}),
        )
    return urls


def _okr_task_generation_workflow_for_log(
    request: HttpRequest, workflow_id: int
) -> SystemWorkflow:
    if workflow_id < 1 or workflow_id > _MAX_BIGAUTOFIELD:
        raise Http404("task generation log not found")
    project = _active_project_from_request(request)
    if project is None:
        raise Http404("task generation log not found")
    workflow = (
        SystemWorkflow.objects.filter(
            pk=workflow_id,
            kind=system_agents.OKR_TASK_AGENT_KIND,
        )
        .first()
    )
    if workflow is None:
        raise Http404("task generation log not found")
    key_result_id = _workflow_state_int(workflow, "key_result_id")
    key_result = (
        KeyResult.objects.select_related("objective__project")
        .filter(pk=key_result_id, objective__project=project)
        .first()
    )
    if key_result is None:
        raise Http404("task generation log not found")
    return workflow


def _attach_standing_order_run_state(orders: list[StandingOrder]) -> None:
    order_ids = [order.pk for order in orders]
    if not order_ids:
        return
    workflows = (
        SystemWorkflow.objects.filter(
            kind=system_agents.STANDING_ORDER_AGENT_KIND,
            main_thread_id__in=[
                system_agents._standing_order_main_thread_id(order_id)
                for order_id in order_ids
            ],
        )
        .order_by("main_thread_id", "-created_at")
    )
    workflows_by_thread: dict[str, SystemWorkflow] = {}
    for workflow in workflows:
        workflows_by_thread.setdefault(workflow.main_thread_id, workflow)
    log_urls_by_workflow_id = _standing_order_log_urls(workflows_by_thread.values())
    for order in orders:
        latest_workflow = workflows_by_thread.get(
            system_agents._standing_order_main_thread_id(order.pk)
        )
        order.latest_workflow = latest_workflow  # type: ignore[attr-defined]
        order.run_running = (  # type: ignore[attr-defined]
            latest_workflow is not None
            and latest_workflow.status == SystemWorkflow.STATUS_RUNNING
        )
        order.run_log_url = (  # type: ignore[attr-defined]
            log_urls_by_workflow_id.get(latest_workflow.pk)
            if latest_workflow is not None
            else ""
        )


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


def _proposed_session_prompt(proposed_session: ProposedSession) -> str:
    parts = [
        "Go ahead and implement this proposed session.",
        "",
        f"Standing order: {proposed_session.standing_order.title}",
    ]
    if proposed_session.standing_order.goal:
        parts.extend(["", f"Standing order goal:\n{proposed_session.standing_order.goal}"])
    parts.extend(["", f"Proposed session: {proposed_session.title}"])
    if proposed_session.summary:
        parts.extend(["", f"Summary:\n{proposed_session.summary}"])
    files = proposed_session.display_files  # type: ignore[attr-defined]
    if files:
        parts.extend(["", "Relevant files:", *[f"- {file}" for file in files]])
    return "\n".join(parts)


def _standing_order_log_urls(workflows: Iterable[SystemWorkflow]) -> dict[int, str]:
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
            reverse("standing_order_run_log", kwargs={"workflow_id": run.workflow_id}),
        )
    return urls


def _standing_order_workflow_for_log(
    request: HttpRequest, workflow_id: int
) -> SystemWorkflow:
    if workflow_id < 1 or workflow_id > _MAX_BIGAUTOFIELD:
        raise Http404("standing order run log not found")
    project = _active_project_from_request(request)
    if project is None:
        raise Http404("standing order run log not found")
    workflow = (
        SystemWorkflow.objects.filter(
            pk=workflow_id,
            kind=system_agents.STANDING_ORDER_AGENT_KIND,
        )
        .first()
    )
    if workflow is None:
        raise Http404("standing order run log not found")
    standing_order_id = _workflow_state_int(workflow, "standing_order_id")
    standing_order = StandingOrder.objects.filter(
        pk=standing_order_id,
        project=project,
    ).first()
    if standing_order is None:
        raise Http404("standing order run log not found")
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


def session(request: HttpRequest, session_id: str) -> HttpResponse:
    return _render_session_detail(request, session_id)


def _render_session_detail(
    request: HttpRequest,
    session_id: str,
    *,
    read_only: bool = False,
    display_title: str | None = None,
    system_prompt: str = "",
) -> HttpResponse:
    # Sweep stuck workers before reading status: a worker that died without
    # writing a terminal status would otherwise leave the page in "streaming"
    # mode forever, since the EventSource wouldn't reach an end event.
    codex_pool.reconcile_dead()
    initial_settings = _stored_settings(request)
    config = codex_pool.app_server_config(
        enable_memories=initial_settings.enable_memories
    )
    with Codex(config=config) as codex:
        # ``thread/read`` only works for threads already loaded into the
        # app-server's in-memory map. Each request spawns a fresh app-server
        # subprocess, so newly-created threads (or any thread persisted by a
        # different worker) need ``thread/resume`` to read them off disk.
        # The resume response already carries the full thread including turns,
        # so a follow-up ``thread/read`` would just be a redundant round-trip.
        resumed = codex._client.thread_resume(session_id)
        thread = resumed.thread
        models_data = _models_for_plan_mode_fallback(codex)
        resolved_settings = _resolved_settings(request, models_data)
        settings = resolved_settings.values
        cookie_updates = resolved_settings.cookie_updates
        plan_model = _plan_mode_model_from_models(resumed, settings, models_data)
    is_archived = _thread_is_archived(thread)
    entries = _apply_system_authors(list(_entries_for(thread)), session_id)
    entries = _apply_qa_approval_messages(entries, session_id)
    entries = _filter_demo_agent_entries(entries, session_id)
    name_value = getattr(thread, "name", None) or ""
    projects = list(Project.objects.all())
    metadata_by_thread = _metadata_by_thread_id([thread])
    session_project = _project_for_thread(thread, metadata_by_thread, projects)
    pr_url = _pr_url_for_thread(thread)
    _mark_proposed_tasks_pr_opened(session_id, pr_url)
    active_instance = _active_instance_for(session_id)
    show_active_worker_transcript = _show_active_worker_transcript(active_instance)
    active_demo_worker = (
        active_instance is not None and active_instance.agent_kind == demo.DEMO_AGENT_KIND
    )
    active_system_workflow = system_agents.active_workflow_for_thread(session_id)
    # While a worker is running, drop the entries that belong to its
    # in-progress turn — the SSE stream replays them from byte 0 of the
    # events file, so leaving the rollout-rendered copy in place would
    # double up every entry in the live DOM. The page reload on stream end
    # restores the canonical view.
    entries = _trim_in_progress_turn(entries, active_instance)
    default_plan_mode = _entries_await_plan_approval(entries)
    _mark_pending_plan_actions(entries)
    token_usage = _token_usage_for(thread)
    goal_objective = codex_events.latest_goal_for_thread(session_id)
    task_plan = _task_plan_context(
        codex_events.latest_task_plan_for_instance(active_instance)
    )
    diff_view = build_worktree_diff(_thread_cwd(thread))
    active_session_demo = demo.active_demo_for(session_id)
    session_demo = demo.latest_demo_for(session_id)
    demo_url = (
        demo.demo_url_for_request(request, session_id)
        if active_session_demo is not None
        else ""
    )
    settings_dialog_context = _settings_dialog_context(settings, models_data)
    active_worker_status_text = _active_worker_status_text(active_instance)
    workflow_status_text = _workflow_status_text(active_system_workflow)
    live_status_text = active_worker_status_text or (
        workflow_status_text
        if active_system_workflow is not None and active_instance is None
        else ""
    )
    response = render(
        request,
        "session.html",
        {
            "thread": thread,
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
            "demo_start_disabled": (
                active_system_workflow is not None or active_instance is not None
            ),
            "workflow_status_text": workflow_status_text,
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
            "token_usage": token_usage,
            "next_message_config": _next_message_config(settings, resumed, plan_model),
            "pr_slash_prompt": _PR_SLASH_PROMPT,
            "qa_slash_prompt": _QA_SLASH_PROMPT,
            "default_plan_mode": default_plan_mode,
            "plan_approval_prompt": _PLAN_APPROVAL_PROMPT,
            "plan_revision_prompt": _PLAN_REVISION_PROMPT,
            "pr_url": pr_url,
            "goal_objective": goal_objective,
            "task_plan": task_plan,
            "diff_view": diff_view,
            "session_demo": session_demo,
            "active_session_demo": active_session_demo,
            "demo_url": demo_url,
            "projects": projects,
            "session_project": session_project,
            "session_project_id": session_project.pk if session_project is not None else "",
            **settings_dialog_context,
        },
    )
    _apply_cookie_updates(response, cookie_updates)
    return response


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


@require_http_methods(["POST"])
def logout(request: HttpRequest) -> HttpResponse:
    values = _stored_settings(request) if _authenticated_user(request) is not None else None
    auth_logout(request)
    response = redirect("index")
    if values is not None:
        _apply_cookie_updates(response, _settings_cookie_updates(values))
    return response


def _thread_is_archived(thread: Any) -> bool:
    """Return whether Codex resumed this thread from archived rollout storage."""
    path = getattr(thread, "path", None)
    if not isinstance(path, str) or not path:
        return False
    return _ARCHIVED_SESSIONS_DIR in Path(path).parts


def _token_usage_for(thread: Any) -> dict[str, str] | None:
    """Return formatted input/cached/output token counts, or None.

    Token usage is only persisted by Codex in the on-disk rollout file (as
    ``TokenCount`` event_msg entries); the SDK ``Thread`` does not carry it.
    Archived sessions use ``ArchivedSessionTokenUsage`` as a cache once the
    value has been parsed. Live sessions always parse the current rollout so
    the page reflects the latest turn.
    """
    usage = _token_usage_numbers_for(thread)
    if usage is None:
        return None
    formatted = {
        "input": _format_token_count(_non_cached_input_tokens(usage)),
        "cached": _format_token_count(usage["cached_input_tokens"]),
        "output": _format_token_count(usage["output_tokens"]),
    }
    context_tokens = usage.get("context_tokens", 0)
    context_window = usage.get("model_context_window", 0)
    if context_tokens > 0 and context_window > 0:
        percent = round((context_tokens / context_window) * 100)
        percent = min(100, max(0, percent))
        formatted.update(
            {
                "context": f"{percent}%",
                "context_title": f"{context_tokens:,} of {context_window:,} tokens in current context",
            }
        )
    return formatted


def _token_usage_numbers_for(thread: Any) -> dict[str, int] | None:
    if _thread_is_archived(thread):
        return _archived_token_usage_numbers_for(thread)
    return _latest_token_usage_numbers_for(thread)


def _latest_token_usage_numbers_for(thread: Any) -> dict[str, int] | None:
    rollout_path = _rollout_path_for(thread)
    if rollout_path is None:
        return None
    usage = rollout.latest_token_usage(rollout_path)
    if usage is None:
        return None
    return {key: usage.get(key, 0) for key in _TOKEN_USAGE_KEYS}


def _rollout_path_for(thread: Any) -> Path | None:
    path = getattr(thread, "path", None)
    if not isinstance(path, str) or not path:
        return None
    rollout_path = Path(path)
    if not rollout_path.is_file():
        return None
    return rollout_path


def _rollout_mtime_ns(rollout_path: Path | None) -> int:
    if rollout_path is None:
        return 0
    try:
        return rollout_path.stat().st_mtime_ns
    except OSError:
        return 0


def _archived_token_usage_numbers_for(thread: Any) -> dict[str, int] | None:
    thread_id = getattr(thread, "id", None)
    if not isinstance(thread_id, str) or not thread_id:
        return _latest_token_usage_numbers_for(thread)
    rollout_path = _rollout_path_for(thread)
    cached = ArchivedSessionTokenUsage.objects.filter(thread_id=thread_id).first()
    if cached is not None and _cached_token_usage_is_current(cached, rollout_path):
        return _token_usage_from_cache(cached)
    usage = _latest_token_usage_numbers_for(thread)
    if usage is None:
        return _token_usage_from_cache(cached) if cached is not None else None
    cached, _created = ArchivedSessionTokenUsage.objects.update_or_create(
        thread_id=thread_id,
        defaults=_token_usage_cache_defaults(rollout_path, usage),
    )
    return _token_usage_from_cache(cached)


def _cached_token_usage_is_current(
    cache: ArchivedSessionTokenUsage, rollout_path: Path | None
) -> bool:
    if rollout_path is None:
        return True
    return (
        cache.rollout_path == str(rollout_path)
        and cache.rollout_mtime_ns == _rollout_mtime_ns(rollout_path)
    )


def _token_usage_cache_defaults(
    rollout_path: Path | None, usage: dict[str, int]
) -> dict[str, str | int]:
    return {
        "rollout_path": str(rollout_path) if rollout_path is not None else "",
        "rollout_mtime_ns": _rollout_mtime_ns(rollout_path),
        "input_tokens": usage["input_tokens"],
        "cached_input_tokens": usage["cached_input_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
        "context_tokens": usage["context_tokens"],
        "model_context_window": usage["model_context_window"],
    }


def _token_usage_from_cache(cache: ArchivedSessionTokenUsage) -> dict[str, int]:
    return {
        "input_tokens": cache.input_tokens,
        "cached_input_tokens": cache.cached_input_tokens,
        "output_tokens": cache.output_tokens,
        "total_tokens": cache.total_tokens,
        "context_tokens": cache.context_tokens,
        "model_context_window": cache.model_context_window,
    }


def _format_token_count(value: int) -> str:
    return f"{value:,}"


# Codex reports cached input as part of input_tokens and total_tokens; keep
# cache as a breakdown rather than adding it back into displayed totals.
def _non_cached_input_tokens(usage: Mapping[str, int]) -> int:
    return max(usage.get("input_tokens", 0) - usage.get("cached_input_tokens", 0), 0)


def _display_total_tokens(usage: Mapping[str, int]) -> int:
    return max(usage.get("total_tokens", 0) - usage.get("cached_input_tokens", 0), 0)


def _threads_for_usage(codex: Codex) -> list[Any] | None:
    threads: list[Any] = []
    failed = False
    try:
        threads.extend(_all_threads(codex))
    except AppServerError:
        logger.warning("failed to list active sessions for usage page")
        failed = True
    try:
        threads.extend(_all_threads(codex, archived=True))
    except AppServerError:
        logger.warning("failed to list archived sessions for usage page")
        failed = True
    if failed:
        return None
    return _dedupe_usage_threads(threads)


def _dedupe_usage_threads(threads: list[Any]) -> list[Any]:
    hidden_thread_ids = system_agents.hidden_thread_ids()
    seen: set[str] = set()
    deduped: list[Any] = []
    for thread in threads:
        thread_id = getattr(thread, "id", None)
        if isinstance(thread_id, str):
            if thread_id in hidden_thread_ids or thread_id in seen:
                continue
            seen.add(thread_id)
        deduped.append(thread)
    return deduped


def _system_agent_runs_by_thread_id(
    thread_ids: Iterable[str],
) -> dict[str, SystemAgentRun]:
    ids = [thread_id for thread_id in thread_ids if thread_id]
    if not ids:
        return {}
    runs = (
        SystemAgentRun.objects.filter(thread_id__in=ids)
        .exclude(thread_id="")
        .select_related("instance", "workflow")
        .order_by("thread_id", "-created_at", "-pk")
    )
    by_thread_id: dict[str, SystemAgentRun] = {}
    for run in runs:
        by_thread_id.setdefault(run.thread_id, run)
    return by_thread_id


def _system_agent_run_for_thread(thread_id: str) -> SystemAgentRun | None:
    if not thread_id:
        return None
    return (
        SystemAgentRun.objects.filter(thread_id=thread_id)
        .select_related("instance", "workflow")
        .order_by("-created_at", "-pk")
        .first()
    )


def _system_agent_run_label(run: SystemAgentRun | None) -> str:
    if run is None:
        return ""
    display_author = run.instance.display_author.strip()
    if display_author:
        return display_author
    return run.agent_kind.replace("_", " ")


def _system_agent_run_detail_title(run: SystemAgentRun) -> str:
    label = _system_agent_run_label(run)
    return f"{label} log" if label else "System session"


def _lifetime_token_usage_for(threads: list[Any]) -> dict[str, str]:
    totals = {key: 0 for key in _TOKEN_USAGE_KEYS}
    display_total = 0
    display_input = 0
    for thread in threads:
        usage = _token_usage_numbers_for(thread)
        if usage is None:
            continue
        for key in _TOKEN_USAGE_KEYS:
            totals[key] += usage.get(key, 0)
        display_total += _display_total_tokens(usage)
        display_input += _non_cached_input_tokens(usage)
    return {
        "total": _format_token_count(display_total),
        "input": _format_token_count(display_input),
        "output": _format_token_count(totals["output_tokens"]),
        "cached": _format_token_count(totals["cached_input_tokens"]),
    }


def _next_message_config(
    settings: SettingsValues, resumed: Any, plan_model: str | None
) -> list[dict[str, str]]:
    """Return the settings that will govern the next submitted message."""
    model = _string_value(getattr(resumed, "model", None))
    reasoning = _string_value(getattr(resumed, "reasoning_effort", None))
    plan_model_value = plan_model or "Unknown"
    sandbox_value = _option_label(
        _SANDBOX_POLICY_OPTIONS,
        _effective_sandbox_policy(settings),
        default="Codex default",
    )
    approval_value = _option_label(
        _APPROVAL_MODE_OPTIONS, _effective_approval_mode(settings)
    )
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
    ]


def _effective_sandbox_policy(settings: SettingsValues) -> str:
    sandbox_policy = settings.sandbox_policy
    if sandbox_policy and sandbox_policy not in _VALID_SANDBOX_POLICIES:
        return ""
    return sandbox_policy


def _effective_approval_mode(settings: SettingsValues) -> str:
    if settings.approval_mode not in _VALID_APPROVAL_MODES:
        return _DEFAULT_APPROVAL_MODE
    return settings.approval_mode


def _effective_coding_agent(settings: SettingsValues) -> str:
    if settings.coding_agent in coding_agents.VALID_CODING_AGENTS:
        return settings.coding_agent
    return coding_agents.DEFAULT_CODING_AGENT


def _base_instructions_for_settings(
    settings: SettingsValues, *, explicit_default: bool = False
) -> str | None:
    agent = _effective_coding_agent(settings)
    if agent == coding_agents.CODING_AGENT_CODEX:
        if explicit_default and settings.coding_agent == coding_agents.CODING_AGENT_CODEX:
            return coding_agents.default_codex_base_instructions()
        return None
    return coding_agents.base_instructions_for(agent)


def _string_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return raw.strip() if isinstance(raw, str) else ""


def _option_label(
    options: tuple[tuple[str, str], ...], value: str, *, default: str | None = None
) -> str:
    if not value and default is not None:
        return default
    return next((label for option_value, label in options if option_value == value), value)


def _selected_repo_for_dialog(
    saved_repo: str, repos: list[str], selected_project: Project | None = None
) -> str:
    if selected_project is not None and selected_project.repo_path in repos:
        return selected_project.repo_path
    return saved_repo if saved_repo in repos else ""


def _new_session_project_for_dialog(
    selected_project: Project | None,
    saved_repo: str,
    projects: list[Project],
) -> Project | None:
    if selected_project is not None and selected_project in projects:
        return selected_project
    if saved_repo:
        return next((project for project in projects if project.repo_path == saved_repo), None)
    return projects[0] if projects else None


def _selected_project_for_settings(
    settings: SettingsValues, projects: list[Project] | None = None
) -> Project | None:
    if settings.selected_project_id is None:
        return None
    candidates = projects if projects is not None else list(Project.objects.all())
    return next(
        (project for project in candidates if project.pk == settings.selected_project_id),
        None,
    )


def _active_project_from_request(request: HttpRequest) -> Project | None:
    return _selected_project_for_settings(_stored_settings(request))


def _metadata_by_thread_id(threads: list[Any]) -> dict[str, SessionMetadata]:
    thread_ids = [
        thread.id
        for thread in threads
        if isinstance(getattr(thread, "id", None), str) and thread.id
    ]
    if not thread_ids:
        return {}
    return SessionMetadata.objects.in_bulk(thread_ids, field_name="thread_id")


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


def _effective_auto_pr_enabled(
    project: Project | None, *, global_enabled: bool
) -> bool:
    if project is None:
        return global_enabled
    if project.auto_pr_mode == Project.AUTO_PR_ON:
        return True
    if project.auto_pr_mode == Project.AUTO_PR_OFF:
        return False
    return global_enabled


def _associate_existing_sessions_with_project(project: Project, request: HttpRequest) -> None:
    settings = _stored_settings(request)
    config = codex_pool.app_server_config(enable_memories=settings.enable_memories)
    try:
        with Codex(config=config) as codex:
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

    The in-progress turn is identified by the most recent user-message
    entry whose text matches the active worker's prompt; anything from
    that point onward is owned by the stream until the turn ends.
    """
    if active is None or not active.prompt:
        return entries
    for i in range(len(entries) - 1, -1, -1):
        entry = entries[i]
        if entry.get("kind") == "user" and entry.get("text") == active.prompt:
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
    if active is None or not active.prompt or active.agent_kind == demo.DEMO_AGENT_KIND:
        return ""
    return active.prompt


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


def _workflow_status_text(workflow: Any | None) -> str:
    return streaming.system_workflow_status_text(workflow)


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
        text = "QA agent approved the diff."
        if feedback:
            text = f"{text}\n\n{feedback}"
        if workflow.step == system_agents.STEP_QA_APPROVED:
            insert_index = next_user_message_index
        else:
            insert_index = max(next_user_message_index - 1, 0)
        yield insert_index, {
            "kind": "agent",
            "display_author": system_agents.QA_DISPLAY_AUTHOR,
            "text": text,
            "timestamp": int(workflow.updated_at.timestamp()),
        }


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
        yield from _collapse_flat_entries(flat)
        return
    yield from _render_entries(thread)


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


def _resolved_settings(request: HttpRequest, models_data: list[Any]) -> ResolvedSettings:
    """Read the dialog state from storage and reconcile against Codex.

    The returned ``cookie_updates`` map must be persisted on the response
    (via ``_apply_cookie_updates``) so corrected state takes effect on the
    next request.

    Two stale-state cases handled here:
      1. The saved model id is no longer offered → snap to the provider's
         default model *and* that model's default effort, since the
         supported-effort set can differ between providers.
      2. The model is still offered but its
         ``supported_reasoning_efforts`` has narrowed under us so the
         saved effort no longer fits → snap effort to that model's
         default while leaving the model alone.

    Authenticated users read from ``UserSettings`` and get a full cookie
    mirror back on each resolution. Anonymous users continue to read and
    write the signed cookies directly.

    Empty ``models_data`` (transport hiccup, mock in tests) means we can't
    validate model compatibility; return the saved values untouched.

    Sandbox is validated against our own static enum rather than Codex's
    model list (it's not a model-scoped setting), so a tampered/legacy
    cookie value falls through to the empty "model default" state.

    Approval mode is validated against our own static enum and falls back
    to ``_DEFAULT_APPROVAL_MODE`` (a safe default with an automated
    reviewer in the loop) when the cookie is missing or invalid, so the
    UI is never left in an ambiguous "no policy picked" state.
    """
    saved = _stored_settings(request)
    saved_sandbox = saved.sandbox_policy
    saved_approval = saved.approval_mode
    if saved_sandbox and saved_sandbox not in _VALID_SANDBOX_POLICIES:
        saved_sandbox = ""
    if saved_approval not in _VALID_APPROVAL_MODES:
        saved_approval = _DEFAULT_APPROVAL_MODE
    saved = saved._replace(
        sandbox_policy=saved_sandbox,
        approval_mode=saved_approval,
    )
    if not models_data:
        return _resolved_settings_result(request, saved, {})

    valid_ids = {m.id for m in models_data}
    if saved.model and saved.model in valid_ids:
        model_obj = next(m for m in models_data if m.id == saved.model)
        if saved.reasoning_effort:
            supported = _supported_effort_values(model_obj)
            if supported and saved.reasoning_effort not in supported:
                new_effort = _model_default_effort(model_obj)
                return _resolved_settings_result(
                    request,
                    saved._replace(reasoning_effort=new_effort),
                    {_EFFORT_COOKIE: new_effort},
                )
        return _resolved_settings_result(request, saved, {})

    default_model = next((m for m in models_data if m.is_default), models_data[0])
    new_effort = _model_default_effort(default_model)
    return _resolved_settings_result(
        request,
        saved._replace(model=default_model.id, reasoning_effort=new_effort),
        {_MODEL_COOKIE: default_model.id, _EFFORT_COOKIE: new_effort},
    )


def _resolved_settings_result(
    request: HttpRequest, values: SettingsValues, cookie_updates: dict[str, str]
) -> ResolvedSettings:
    user = _authenticated_user(request)
    if user is not None:
        _save_user_settings(user, values)
        cookie_updates = _settings_cookie_updates(values)
    return ResolvedSettings(values=values, cookie_updates=cookie_updates)


def _authenticated_user(request: HttpRequest) -> Any | None:
    user = request.user
    return user if user.is_authenticated else None


def _stored_settings(request: HttpRequest) -> SettingsValues:
    user = _authenticated_user(request)
    if user is not None:
        return _settings_values_for_user(_settings_for_user(user))
    return SettingsValues(
        model=_read_cookie(request, _MODEL_COOKIE),
        reasoning_effort=_read_cookie(request, _EFFORT_COOKIE),
        sandbox_policy=_read_cookie(request, _SANDBOX_COOKIE),
        approval_mode=_read_cookie(request, _APPROVAL_COOKIE),
        coding_agent=_read_cookie(request, _CODING_AGENT_COOKIE),
        extra_system_prompt=_read_extra_system_prompt_cookie(request),
        use_worktrees=_read_cookie(request, _USE_WORKTREES_COOKIE) == "true",
        auto_pr_enabled=_read_cookie(request, _AUTO_PR_COOKIE) == "true",
        show_archived_sessions=_read_cookie(request, _SHOW_ARCHIVED_COOKIE) == "true",
        last_selected_repo=_read_cookie(request, _LAST_SELECTED_REPO_COOKIE),
        selected_project_id=_read_selected_project_cookie(request),
        enable_memories=_read_cookie(request, _ENABLE_MEMORIES_COOKIE) == "true",
    )


def _settings_for_user(user: Any) -> UserSettings:
    settings, _created = UserSettings.objects.get_or_create(user=user)
    return settings


def _settings_values_for_user(settings: UserSettings) -> SettingsValues:
    return SettingsValues(
        model=settings.model,
        reasoning_effort=settings.reasoning_effort,
        sandbox_policy=settings.sandbox_policy,
        approval_mode=settings.approval_mode,
        coding_agent=settings.coding_agent,
        extra_system_prompt=settings.extra_system_prompt,
        use_worktrees=settings.use_worktrees,
        auto_pr_enabled=settings.auto_pr_enabled,
        show_archived_sessions=settings.show_archived_sessions,
        last_selected_repo=settings.last_selected_repo,
        selected_project_id=settings.selected_project_id,
        enable_memories=settings.enable_memories,
    )


def _save_user_settings(user: Any, values: SettingsValues) -> UserSettings:
    settings = _settings_for_user(user)
    updates: list[str] = []
    for field, value in (
        ("model", values.model),
        ("reasoning_effort", values.reasoning_effort),
        ("sandbox_policy", values.sandbox_policy),
        ("approval_mode", values.approval_mode),
        ("coding_agent", values.coding_agent),
        ("extra_system_prompt", values.extra_system_prompt),
        ("use_worktrees", values.use_worktrees),
        ("auto_pr_enabled", values.auto_pr_enabled),
        ("show_archived_sessions", values.show_archived_sessions),
        ("last_selected_repo", values.last_selected_repo),
        ("selected_project_id", values.selected_project_id),
        ("enable_memories", values.enable_memories),
    ):
        if getattr(settings, field) != value:
            setattr(settings, field, value)
            updates.append(field)
    if updates:
        settings.save(update_fields=[*updates, "updated_at"])
    return settings


def _settings_cookie_updates(values: SettingsValues) -> dict[str, str]:
    return {
        _MODEL_COOKIE: values.model,
        _EFFORT_COOKIE: values.reasoning_effort,
        _SANDBOX_COOKIE: values.sandbox_policy,
        _APPROVAL_COOKIE: values.approval_mode,
        _CODING_AGENT_COOKIE: _effective_coding_agent(values),
        _EXTRA_SYSTEM_PROMPT_COOKIE: _encode_extra_system_prompt_cookie(
            values.extra_system_prompt
        ),
        _USE_WORKTREES_COOKIE: "true" if values.use_worktrees else "false",
        _AUTO_PR_COOKIE: "true" if values.auto_pr_enabled else "false",
        _SHOW_ARCHIVED_COOKIE: "true" if values.show_archived_sessions else "false",
        _LAST_SELECTED_REPO_COOKIE: values.last_selected_repo,
        _SELECTED_PROJECT_COOKIE: (
            str(values.selected_project_id) if values.selected_project_id is not None else ""
        ),
        _ENABLE_MEMORIES_COOKIE: "true" if values.enable_memories else "false",
    }


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


def _valid_cookie_setting_updates(request: HttpRequest) -> dict[str, str | bool | int | None]:
    updates: dict[str, str | bool | int | None] = {}
    model = _read_signed_cookie_if_present(request, _MODEL_COOKIE)
    if model is not None and len(model) <= _MODEL_MAX_LEN:
        updates["model"] = model
    effort = _read_signed_cookie_if_present(request, _EFFORT_COOKIE)
    if effort is not None and (not effort or effort in {e.value for e in ReasoningEffort}):
        updates["reasoning_effort"] = effort
    sandbox = _read_signed_cookie_if_present(request, _SANDBOX_COOKIE)
    if sandbox is not None:
        updates["sandbox_policy"] = sandbox if sandbox in _VALID_SANDBOX_POLICIES else ""
    approval = _read_signed_cookie_if_present(request, _APPROVAL_COOKIE)
    if approval is not None:
        updates["approval_mode"] = (
            approval if approval in _VALID_APPROVAL_MODES else _DEFAULT_APPROVAL_MODE
        )
    coding_agent = _read_signed_cookie_if_present(request, _CODING_AGENT_COOKIE)
    if coding_agent is not None:
        updates["coding_agent"] = (
            coding_agent
            if coding_agent in coding_agents.VALID_CODING_AGENTS
            else coding_agents.DEFAULT_CODING_AGENT
        )
    extra_prompt = _read_signed_cookie_if_present(request, _EXTRA_SYSTEM_PROMPT_COOKIE)
    if extra_prompt is not None:
        decoded = _decode_extra_system_prompt_value(extra_prompt)
        if len(decoded) <= _EXTRA_SYSTEM_PROMPT_MAX_LEN:
            updates["extra_system_prompt"] = decoded
    show_archived = _read_signed_cookie_if_present(request, _SHOW_ARCHIVED_COOKIE)
    if show_archived in {"true", "false"}:
        updates["show_archived_sessions"] = show_archived == "true"
    use_worktrees = _read_signed_cookie_if_present(request, _USE_WORKTREES_COOKIE)
    if use_worktrees in {"true", "false"}:
        updates["use_worktrees"] = use_worktrees == "true"
    auto_pr = _read_signed_cookie_if_present(request, _AUTO_PR_COOKIE)
    if auto_pr in {"true", "false"}:
        updates["auto_pr_enabled"] = auto_pr == "true"
    last_selected_repo = _read_signed_cookie_if_present(
        request, _LAST_SELECTED_REPO_COOKIE
    )
    if (
        last_selected_repo is not None
        and len(last_selected_repo) <= _LAST_SELECTED_REPO_MAX_LEN
    ):
        updates["last_selected_repo"] = last_selected_repo
    selected_project_raw = _read_signed_cookie_if_present(request, _SELECTED_PROJECT_COOKIE)
    if selected_project_raw is not None:
        updates["selected_project_id"] = _valid_selected_project_id(selected_project_raw)
    enable_memories = _read_signed_cookie_if_present(request, _ENABLE_MEMORIES_COOKIE)
    if enable_memories in {"true", "false"}:
        updates["enable_memories"] = enable_memories == "true"
    return updates


def _read_signed_cookie_if_present(request: HttpRequest, name: str) -> str | None:
    if name not in request.COOKIES:
        return None
    try:
        value = request.get_signed_cookie(name)
    except Exception:
        return None
    return (value or "").strip()


def _read_selected_project_cookie(request: HttpRequest) -> int | None:
    return _valid_selected_project_id(
        _read_signed_cookie_if_present(request, _SELECTED_PROJECT_COOKIE)
    )


def _valid_selected_project_id(raw: str | None) -> int | None:
    if raw is None or raw == "":
        return None
    try:
        project_id = int(raw)
    except ValueError:
        return None
    if project_id < 1 or project_id > _MAX_BIGAUTOFIELD:
        return None
    return project_id if Project.objects.filter(pk=project_id).exists() else None


def _read_cookie(request: HttpRequest, name: str) -> str:
    """Read a signed cookie; return ``""`` on missing or tampered value.

    ``request.get_signed_cookie`` raises on a missing/invalid signature;
    we treat both as "no value" so a corrupt cookie just falls through to
    the reconcile path instead of 500ing the index render.
    """
    try:
        value = request.get_signed_cookie(name, default="")
    except Exception:
        return ""
    return (value or "").strip()


def _read_extra_system_prompt_cookie(request: HttpRequest) -> str:
    encoded = _read_cookie(request, _EXTRA_SYSTEM_PROMPT_COOKIE)
    if not encoded:
        return ""
    return _decode_extra_system_prompt_value(encoded)


def _decode_extra_system_prompt_value(encoded: str) -> str:
    try:
        decoded = base64.urlsafe_b64decode(encoded.encode("ascii")).decode()
    except (binascii.Error, UnicodeDecodeError, ValueError):
        # Fallback for pre-encoding local cookies from development builds.
        decoded = encoded
    return decoded.strip()


def _encode_extra_system_prompt_cookie(value: str) -> str:
    if not value:
        return ""
    return base64.urlsafe_b64encode(value.encode()).decode("ascii")


def _apply_cookie_updates(response: HttpResponse, updates: dict[str, str]) -> None:
    for name, value in updates.items():
        response.set_signed_cookie(
            name, value, max_age=_COOKIE_MAX_AGE, samesite="Lax"
        )


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


def _fetch_rate_limits(codex: Codex) -> dict[str, Any] | None:
    """Fetch the account/rateLimits/read snapshot, or None if unavailable.

    The endpoint is meaningful only when Codex is talking to a real OpenAI
    account; local-dev (no auth, custom provider via ollama) and older
    Codex builds will fail with MethodNotFound or an auth error. The
    usage page must still render in those modes, so any failure here
    swallows into None and the page shows its empty state.
    """
    try:
        response = codex._client.request(
            "account/rateLimits/read",
            None,
            response_model=GetAccountRateLimitsResponse,
        )
    except AppServerError:
        return None
    except Exception:
        logger.exception("failed to fetch account rate limits; showing usage empty state")
        return None
    return _format_rate_limit_snapshot(response.rate_limits)


def _format_rate_limit_snapshot(snapshot: RateLimitSnapshot) -> dict[str, Any] | None:
    """Project a RateLimitSnapshot into a template-friendly dict.

    Returns None when neither the primary nor secondary window is set; the
    template hides the section entirely in that case rather than render an
    empty header. ``used_percent`` from Codex describes consumption, so we
    expose ``remaining_percent`` (the more intuitive framing for a user
    looking at their remaining budget) alongside it.
    """
    windows: list[dict[str, Any]] = []
    for label, window in (("Primary", snapshot.primary), ("Secondary", snapshot.secondary)):
        if window is None:
            continue
        used = max(0, min(100, window.used_percent))
        windows.append(
            {
                "label": label,
                "used_percent": used,
                "remaining_percent": 100 - used,
                "resets_at": window.resets_at,
                "window_duration_label": _format_window_duration(window.window_duration_mins),
            }
        )
    if not windows:
        return None
    plan_type = snapshot.plan_type.value if snapshot.plan_type is not None else None
    return {
        "windows": windows,
        "limit_name": snapshot.limit_name,
        "plan_type": plan_type,
    }


def _format_window_duration(window_duration_mins: int | None) -> str | None:
    if window_duration_mins is None or window_duration_mins <= 0:
        return None
    if window_duration_mins % _MINUTES_PER_DAY == 0:
        days = window_duration_mins // _MINUTES_PER_DAY
        return f"{days}-day"
    if window_duration_mins % _MINUTES_PER_HOUR == 0:
        hours = window_duration_mins // _MINUTES_PER_HOUR
        return f"{hours}-hour"
    return f"{window_duration_mins}-min"


def _supported_effort_values(model_obj: Any) -> set[str]:
    """Return the set of effort enum string values ``model_obj`` accepts."""
    return {
        getattr(opt.reasoning_effort, "value", str(opt.reasoning_effort))
        for opt in (getattr(model_obj, "supported_reasoning_efforts", None) or [])
    }


def _model_default_effort(model_obj: Any) -> str:
    default = getattr(model_obj, "default_reasoning_effort", None)
    if default is None:
        return ""
    return getattr(default, "value", str(default))


@require_http_methods(["POST"])
def update_settings(request: HttpRequest) -> HttpResponse:
    model = request.POST.get("model", "").strip()
    effort = request.POST.get("reasoning_effort", "").strip()
    sandbox = request.POST.get("sandbox_policy", "").strip()
    approval = request.POST.get("approval_mode", "").strip()
    coding_agent = request.POST.get("coding_agent", "").strip()
    extra_system_prompt = request.POST.get("extra_system_prompt", "").strip()
    use_worktrees = request.POST.get("use_worktrees", "").strip()
    auto_pr = request.POST.get("auto_pr", "").strip()
    posted_show_archived = request.POST.get("show_archived_sessions")
    show_archived = (
        posted_show_archived.strip() if posted_show_archived is not None else None
    )
    selected_project, selected_project_error = _posted_project(request.POST.get("selected_project", ""))
    if selected_project_error is not None:
        return HttpResponseBadRequest(selected_project_error)
    enable_memories = request.POST.get("enable_memories", "").strip()
    if len(model) > _MODEL_MAX_LEN:
        return HttpResponseBadRequest("model id is too long")
    if len(extra_system_prompt) > _EXTRA_SYSTEM_PROMPT_MAX_LEN:
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
        config = codex_pool.app_server_config(enable_memories=enable_memories == "true")
        with Codex(config=config) as codex:
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
        show_archived_sessions=(
            stored.show_archived_sessions
            if show_archived is None
            else show_archived == "true"
        ),
        last_selected_repo=stored.last_selected_repo,
        selected_project_id=selected_project.pk if selected_project is not None else None,
        enable_memories=enable_memories == "true",
    )
    user = _authenticated_user(request)
    if user is not None:
        _save_user_settings(user, values)
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


@require_http_methods(["GET", "POST"])
def new_project(request: HttpRequest) -> HttpResponse:
    discovered_repos = [str(p) for p in discover_repos()]
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
    auto_pr_mode = request.POST.get("auto_pr_mode", "").strip()
    if not name:
        return HttpResponseBadRequest("project name is required")
    if len(name) > _PROJECT_NAME_MAX_LEN:
        return HttpResponseBadRequest("project name is too long")
    if auto_pr_mode not in _VALID_PROJECT_AUTO_PR_MODES:
        return HttpResponseBadRequest("invalid project auto-PR setting")

    updates: list[str] = []
    if project.name != name:
        project.name = name
        updates.append("name")
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
    settings = _stored_settings(request)
    config = codex_pool.app_server_config(enable_memories=settings.enable_memories)
    with Codex(config=config) as codex:
        resumed = codex._client.thread_resume(session_id)
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
        return _NewSessionTarget(cwd, _project_for_cwd(cwd, projects), False), None

    value = raw_project.strip()
    if value == _BARE_REPO_PROJECT_VALUE:
        cwd = request.POST.get("cwd", "").strip()
        return _NewSessionTarget(cwd, None, True), None
    if not value:
        return None, "project is required"

    project, error = _posted_project(value)
    if error is not None or project is None:
        return None, error or "invalid project"
    return _NewSessionTarget(project.repo_path, project, False), None


def _posted_proposed_task_for_new_session(
    request: HttpRequest, target: _NewSessionTarget
) -> tuple[ProposedTask | None, str | None]:
    raw_task_id = request.POST.get("proposed_task", "").strip()
    if not raw_task_id:
        return None, None
    try:
        task_id = int(raw_task_id)
    except ValueError:
        return None, "proposed task is required"
    if task_id < 1 or task_id > _MAX_BIGAUTOFIELD:
        return None, "proposed task is required"
    task = (
        ProposedTask.objects.select_related("key_result__objective__project")
        .filter(pk=task_id)
        .first()
    )
    if task is None:
        return None, "proposed task is required"
    task_project = task.key_result.objective.project
    if target.project is not None and target.project != task_project:
        return None, "proposed task does not match project"
    if target.project_cleared and target.cwd != task_project.repo_path:
        return None, "proposed task does not match project"
    return task, None


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
    proposed_session = (
        ProposedSession.objects.select_related("standing_order__project")
        .filter(
            pk=session_id,
            inbox_kind=ProposedSession.INBOX_KIND_PROPOSAL,
            outcome_status=ProposedSession.OUTCOME_UNSET,
        )
        .first()
    )
    if proposed_session is None:
        return None, "proposed session is required"
    session_project = proposed_session.standing_order.project
    if target.project is not None and target.project != session_project:
        return None, "proposed session does not match project"
    if target.project is None and target.cwd != session_project.repo_path:
        return None, "proposed session does not match project"
    return proposed_session, None


def _accept_proposed_task_for_session(
    task: ProposedTask | None, session_metadata: SessionMetadata
) -> None:
    if task is None:
        return
    task.outcome_status = ProposedTask.OUTCOME_ACCEPTED
    task.session = session_metadata
    task.pr_url = ""
    task.save(update_fields=["outcome_status", "session", "pr_url", "updated_at"])


def _accept_proposed_session_for_session(
    proposed_session: ProposedSession | None, session_metadata: SessionMetadata
) -> None:
    if proposed_session is None:
        return
    proposed_session.outcome_status = ProposedSession.OUTCOME_ACCEPTED
    proposed_session.accepted_session = session_metadata
    proposed_session.save(
        update_fields=["outcome_status", "accepted_session", "updated_at"]
    )


def _posted_auto_pr_override(raw: str | None, *, default: bool) -> tuple[bool, str | None]:
    if raw is None:
        return default, None
    value = raw.strip().lower()
    if value in {"", "false"}:
        return False, None
    if value == "true":
        return True, None
    return False, "invalid auto-PR setting"


@require_http_methods(["POST"])
def set_session_name(request: HttpRequest, session_id: str) -> HttpResponse:
    name = request.POST.get("name", "").strip()
    if not name:
        return HttpResponseBadRequest("name is required")
    if len(name) > _NAME_MAX_LEN:
        return HttpResponseBadRequest("name is too long")
    settings = _stored_settings(request)
    config = codex_pool.app_server_config(enable_memories=settings.enable_memories)
    with Codex(config=config) as codex:
        codex._client.thread_set_name(session_id, name)
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
    if archived == "true":
        demo.cleanup_demo_for_session(session_id)
    settings = _stored_settings(request)
    config = codex_pool.app_server_config(enable_memories=settings.enable_memories)
    with Codex(config=config) as codex:
        if archived == "true":
            codex.thread_archive(session_id)
        else:
            codex.thread_unarchive(session_id)
    ArchivedSessionTokenUsage.objects.all().delete()
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return HttpResponse(status=204)
    if request.POST.get("next", "").strip() == "index":
        return redirect("index")
    if archived == "true":
        return redirect("index")
    return redirect("session", session_id=session_id)


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
    settings = _stored_settings(request)
    config = codex_pool.app_server_config(enable_memories=settings.enable_memories)
    with Codex(config=config) as codex:
        resumed = codex._client.thread_resume(session_id)
        thread = resumed.thread
    cwd = _thread_cwd(thread)
    if not cwd:
        return HttpResponseBadRequest("thread has no cwd")
    if cwd not in _allowed_session_cwds():
        return HttpResponseBadRequest("thread cwd is not an allowed repository")
    try:
        session_demo = demo.request_demo_start(session_id)
    except demo.DemoAlreadyRunningError as exc:
        return HttpResponseBadRequest(str(exc))
    except demo.DemoError as exc:
        return HttpResponse(str(exc), status=500, content_type="text/plain")
    prompt = demo.start_demo_prompt_for(
        request=request,
        session_id=session_id,
        cwd=cwd,
        demo=session_demo,
    )
    try:
        codex_pool.spawn_turn(
            thread_id=session_id,
            cwd=cwd,
            prompt=prompt,
            sandbox_policy=_effective_sandbox_policy(settings) or None,
            approval_mode=_effective_approval_mode(settings),
            enable_memories=settings.enable_memories,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=demo.DEMO_AGENT_KIND,
            display_author=demo.DEMO_DISPLAY_AUTHOR,
            user_message_index=None,
        )
    except Exception:
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


@require_http_methods(["POST"])
def send_message(request: HttpRequest, session_id: str) -> HttpResponse:
    intent = _message_intent(request)
    pr_activation = _is_pr_activation(request)
    qa_activation = _is_qa_activation(request)
    qa_workflow_activation = pr_activation or qa_activation
    prompt = intent.prompt
    plan_mode = intent.plan_mode
    if not prompt:
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
    if active_system_workflow is not None:
        if qa_workflow_activation:
            return redirect("session", session_id=session_id)
        return HttpResponseBadRequest("PR workflow is running for this session")
    settings = _stored_settings(request)
    raw_active = request.POST.get("active_instance", "").strip()
    if raw_active:
        if qa_workflow_activation:
            return HttpResponseBadRequest("PR workflow requires an idle session")
        instance_id, error = _parse_instance_id(raw_active)
        if error is not None or instance_id is None:
            return HttpResponseBadRequest(error or "invalid instance id")
        steered = codex_pool.steer_instance(
            instance_id,
            expected_thread_id=session_id,
            prompt=prompt,
        )
        if steered is not None:
            return redirect("session", session_id=session_id)
    else:
        active_instance = codex_pool.latest_active_for_thread(session_id)
        if active_instance is not None:
            if qa_workflow_activation:
                return HttpResponseBadRequest("PR workflow requires an idle session")
            steered = codex_pool.steer_instance(
                active_instance.pk,
                expected_thread_id=session_id,
                prompt=prompt,
            )
            if steered is not None:
                return redirect("session", session_id=session_id)
    # If steering is unavailable or races a terminal worker, preserve the
    # submitted prompt by treating it as an ordinary follow-up turn.
    # ``raw_active`` posts still do not retarget a different active worker.
    # ``Thread.cwd`` is an ``AbsolutePathBuf`` pydantic RootModel, so unwrap
    # ``.root`` to get the underlying string the worker subprocess expects;
    # also accept a plain str so a future SDK schema change does not break us.
    config = codex_pool.app_server_config(enable_memories=settings.enable_memories)
    with Codex(config=config) as codex:
        resumed = codex._client.thread_resume(session_id)
        thread = resumed.thread
        thread_entries = list(_entries_for(thread))
        thread_awaits_plan_approval = _entries_await_plan_approval(thread_entries)
        if (
            not collaboration_mode
            and intent.allow_pending_plan_default
            and not intent.explicit_plan_mode
        ):
            plan_mode = thread_awaits_plan_approval
        if (
            thread_awaits_plan_approval
            and not collaboration_mode
            and intent.allow_pending_plan_default
            and not intent.explicit_plan_mode
            and prompt == _PLAN_APPROVAL_PROMPT
        ):
            collaboration_mode = _DEFAULT_COLLABORATION_MODE
            plan_mode = False
        collaboration_model = (
            _plan_mode_model(codex, resumed, settings)
            if plan_mode or collaboration_mode == _DEFAULT_COLLABORATION_MODE
            else None
        )
        if plan_mode and not collaboration_model and not intent.explicit_plan_mode:
            plan_mode = False
    cwd = _thread_cwd(thread)
    if not cwd:
        return HttpResponseBadRequest("thread has no cwd")
    # The session list surfaces every thread the app-server knows about, not
    # just those created via ``new_session``, so the resumed ``cwd`` is not
    # automatically inside the discover_repos() allowlist. Re-validate before
    # spawning so a follow-up cannot run a worker in an unintended directory.
    if cwd not in _allowed_session_cwds():
        return HttpResponseBadRequest("thread cwd is not an allowed repository")
    # Sandbox policy and approval mode are applied per-turn rather than
    # persisted on the thread, so follow-up messages have to re-forward
    # the cookies or every turn after the first silently reverts to Codex
    # defaults — which breaks multi-turn workflows that depend on
    # elevated permissions or stricter escalation handling.
    sandbox_policy = _effective_sandbox_policy(settings)
    approval_mode = _effective_approval_mode(settings)
    previous_instance = codex_pool.latest_for_thread(session_id)
    base_instructions = _base_instructions_for_settings(
        settings, explicit_default=True
    )
    auto_pr_enabled = _auto_pr_enabled_for_session(session_id)
    if qa_workflow_activation:
        developer_instructions = (
            previous_instance.developer_instructions
            if previous_instance is not None
            else settings.extra_system_prompt
        )
        workflow_model = _string_value(getattr(resumed, "model", None)) or settings.model
        workflow_reasoning_effort = (
            _string_value(getattr(resumed, "reasoning_effort", None))
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
        if base_instructions:
            workflow_kwargs["base_instructions"] = base_instructions
        if qa_activation:
            workflow_kwargs["open_pr_on_lgtm"] = False
        system_agents.start_pr_qa_workflow(**workflow_kwargs)
        return redirect("session", session_id=session_id)
    spawn_kwargs: dict[str, Any] = {
        "thread_id": session_id,
        "cwd": cwd,
        "prompt": prompt,
        "sandbox_policy": sandbox_policy or None,
        "approval_mode": approval_mode,
    }
    if base_instructions:
        spawn_kwargs["base_instructions"] = base_instructions
    if settings.enable_memories:
        spawn_kwargs["enable_memories"] = True
    if auto_pr_enabled:
        auto_pr_model = _string_value(getattr(resumed, "model", None)) or settings.model
        auto_pr_reasoning_effort = (
            _string_value(getattr(resumed, "reasoning_effort", None))
            or settings.reasoning_effort
        )
        spawn_kwargs["auto_pr_enabled"] = True
        spawn_kwargs["user_message_index"] = _count_user_entries(thread_entries)
        spawn_kwargs["stored_model"] = auto_pr_model or None
        spawn_kwargs["stored_reasoning_effort"] = auto_pr_reasoning_effort or None
    if plan_mode:
        if not collaboration_model:
            return HttpResponseBadRequest("plan mode requires a model")
        spawn_kwargs["model"] = collaboration_model
        spawn_kwargs["plan_mode"] = True
    elif collaboration_mode == _DEFAULT_COLLABORATION_MODE:
        if not collaboration_model:
            return HttpResponseBadRequest("default collaboration mode requires a model")
        spawn_kwargs["model"] = collaboration_model
        spawn_kwargs["collaboration_mode"] = collaboration_mode
    codex_pool.spawn_turn(**spawn_kwargs)
    return redirect("session", session_id=session_id)


def _thread_awaits_plan_approval(thread: Any) -> bool:
    return _entries_await_plan_approval(list(_entries_for(thread)))


def _pr_url_for_thread(thread: Any) -> str | None:
    """Return the PR opened by the latest completed /pr turn, if any."""
    for turn in reversed(getattr(thread, "turns", []) or []):
        items = [thread_item.root for thread_item in getattr(turn, "items", []) or []]
        if not _is_pr_prompt_turn(items):
            continue
        final_idx = _find_final_agent_idx(items)
        if final_idx == -1:
            continue
        urls: list[str] = []
        for item in items[:final_idx]:
            if _github_pr_tool_call_used(item):
                urls.extend(_pr_urls_from_value(_value_for(item, "result")))
        return urls[-1] if urls else None
    return None


def _is_pr_prompt_turn(items: list[Any]) -> bool:
    for item in items:
        if _value_for(item, "type") != "userMessage":
            continue
        if _user_message_text(item).strip() in _PR_PROMPT_ALIASES:
            return True
    return False


def _github_pr_tool_call_used(item: Any) -> bool:
    if _value_for(item, "type") != "mcpToolCall":
        return False
    server = _string_value(_value_for(item, "server"))
    tool = _string_value(_value_for(item, "tool"))
    detail = f"{server} / {tool}".strip()
    return _GITHUB_PR_TOOL_RE.search(detail) is not None


def _pr_urls_from_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return _GITHUB_PR_URL_RE.findall(value)
    if isinstance(value, dict):
        urls: list[str] = []
        for child in value.values():
            urls.extend(_pr_urls_from_value(child))
        return urls
    if isinstance(value, list | tuple):
        urls = []
        for child in value:
            urls.extend(_pr_urls_from_value(child))
        return urls
    text = _string_value(_value_for(value, "text"))
    if text:
        return _GITHUB_PR_URL_RE.findall(text)
    urls = []
    for attr in ("url", "display_url", "displayUrl", "structured_content", "content"):
        urls.extend(_pr_urls_from_value(_value_for(value, attr)))
    return urls


def _entries_await_plan_approval(entries: list[dict[str, Any]]) -> bool:
    return _pending_plan_entry(entries) is not None


def _pending_plan_entry(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    for entry in reversed(entries):
        kind = entry.get("kind")
        if kind in {"intermediate", "approval_declined", "tool_call", "thinking", "user"}:
            continue
        if kind == "plan":
            return entry
        if kind == "agent":
            return None
    return None


def _mark_pending_plan_actions(entries: list[dict[str, Any]]) -> None:
    _clear_plan_actions(entries)
    pending_plan = _pending_plan_entry(entries)
    if pending_plan is not None:
        pending_plan["show_plan_actions"] = True


def _clear_plan_actions(entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        if entry.get("kind") == "plan":
            entry["show_plan_actions"] = False
        elif entry.get("kind") == "intermediate":
            _clear_plan_actions(entry.get("items", []))


def _count_user_entries(entries: list[dict[str, Any]]) -> int:
    count = 0
    for entry in entries:
        if entry.get("kind") == "user":
            count += 1
        elif entry.get("kind") == "intermediate":
            count += _count_user_entries(entry.get("items", []))
    return count


def _auto_pr_enabled_for_session(session_id: str) -> bool:
    return SessionMetadata.objects.filter(
        thread_id=session_id, auto_pr_enabled=True
    ).exists()


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


def _is_qa_activation(request: HttpRequest) -> bool:
    prompt = request.POST.get("prompt", "").strip()
    parts = prompt.split(maxsplit=1)
    return (
        bool(parts and parts[0].lower() == _QA_SLASH_COMMAND)
        or prompt == _QA_SLASH_PROMPT
    )


def _plan_mode_model(codex: Codex, resumed: Any, settings: SettingsValues) -> str | None:
    models_data = _models_for_plan_mode_fallback(codex)
    return _plan_mode_model_from_models(resumed, settings, models_data)


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


def _allowed_session_cwds() -> set[str]:
    return {str(p) for p in [*discover_repos(), *discover_managed_worktrees()]}


def _parse_instance_id(raw: str) -> tuple[int | None, str | None]:
    try:
        instance_id = int(raw)
    except ValueError:
        return None, "invalid instance id"
    # Cross-check against the column type up front so a tampered value past
    # the BigAutoField range can't leak a backend-specific OverflowError or
    # DataError out as a 500 from ``objects.get``.
    if instance_id < 1 or instance_id > _MAX_BIGAUTOFIELD:
        return None, "instance id out of range"
    return instance_id, None


# Decisions the approval endpoint accepts. The wire string the worker writes
# back to codex's app-server is taken straight from this set, so the constants
# must stay aligned with app-server's approval response schema
# (``accept`` / ``decline`` / ``cancel``). UI labels live in the template;
# this layer only validates the wire value.
_VALID_APPROVAL_DECISIONS = frozenset(
    {
        ApprovalRequest.DECISION_ACCEPT,
        ApprovalRequest.DECISION_DECLINE,
        ApprovalRequest.DECISION_CANCEL,
    }
)


@require_http_methods(["POST"])
def resolve_approval(request: HttpRequest, approval_id: int) -> HttpResponse:
    """Record the user's decision on a pending command/file approval.

    The worker's polling loop wakes on the row update and answers the
    SDK's JSON-RPC request with the recorded ``decision``. The response is
    intentionally minimal (200 with the recorded decision string) so the
    browser-side fetch can surface success without parsing JSON.

    Returns 409 if the approval has already been resolved — racing two
    clicks shouldn't silently overwrite an earlier choice that the worker
    has already returned to codex.
    """
    raw_decision = request.POST.get("decision", "").strip()
    decision = ApprovalRequest.normalize_decision(raw_decision)
    if decision not in _VALID_APPROVAL_DECISIONS:
        return HttpResponseBadRequest("invalid decision")
    try:
        approval = ApprovalRequest.objects.get(pk=approval_id)
    except ApprovalRequest.DoesNotExist:
        return HttpResponse("approval not found", status=404)
    if approval.decision:
        return HttpResponse("approval already resolved", status=409)
    # Filter on ``decision=""`` so two concurrent POSTs can't both succeed
    # in flipping the row away from pending.
    updated = ApprovalRequest.objects.filter(pk=approval_id, decision="").update(
        decision=decision,
        decided_at=timezone.now(),
    )
    if not updated:
        return HttpResponse("approval already resolved", status=409)
    return HttpResponse(decision, content_type="text/plain")


@require_http_methods(["POST"])
def resolve_input_request(request: HttpRequest, input_id: int) -> HttpResponse:
    raw_answers = request.POST.get("answers", "").strip()
    try:
        parsed = json.loads(raw_answers) if raw_answers else {}
    except json.JSONDecodeError:
        return HttpResponseBadRequest("invalid answers")
    if not isinstance(parsed, dict):
        return HttpResponseBadRequest("invalid answers")
    answers: dict[str, Any] = {}
    for key, value in parsed.items():
        key = key.strip()
        if isinstance(value, str):
            value = value.strip()
        if key:
            answers[key] = value
    response: dict[str, Any] = {"answers": answers}
    try:
        input_request = UserInputRequest.objects.get(pk=input_id)
    except UserInputRequest.DoesNotExist:
        return HttpResponse("input request not found", status=404)
    if input_request.response is not None:
        return HttpResponse("input request already resolved", status=409)
    updated = UserInputRequest.objects.filter(pk=input_id, response__isnull=True).update(
        response=response,
        responded_at=timezone.now(),
    )
    if not updated:
        return HttpResponse("input request already resolved", status=409)
    return HttpResponse(json.dumps(response), content_type="application/json")


@require_http_methods(["POST"])
def stop_session(request: HttpRequest, session_id: str) -> HttpResponse:
    """Interrupt the in-progress turn for ``session_id``.

    The Stop button posts the active worker's id (as ``instance``) so a
    stale tab can't accidentally abort a newer overlapping worker the
    user can't see. When the form value is missing (older cached page,
    direct POST) we fall back to "latest active worker for this thread".

    No-ops cleanly when no worker is active so a double-click after the
    turn already finished still lands on the session page rather than 404.
    """
    raw = request.POST.get("instance", "").strip()
    if raw:
        instance_id, error = _parse_instance_id(raw)
        if error is not None or instance_id is None:
            return HttpResponseBadRequest(error or "invalid instance id")
        codex_pool.interrupt_instance(instance_id, expected_thread_id=session_id)
    else:
        if not system_agents.stop_active_workflow(session_id):
            codex_pool.interrupt_active(session_id)
    return redirect("session", session_id=session_id)


@require_http_methods(["POST"])
def new_session(request: HttpRequest) -> HttpResponse:
    intent = _message_intent(request)
    pr_activation = _is_pr_activation(request)
    qa_activation = _is_qa_activation(request)
    qa_workflow_activation = pr_activation or qa_activation
    prompt = intent.prompt
    plan_mode = False if qa_workflow_activation else intent.plan_mode
    projects = list(Project.objects.all())
    target, target_error = _posted_new_session_target(request, projects)
    if target_error is not None or target is None:
        return HttpResponseBadRequest(target_error or "invalid project")
    proposed_task, proposed_task_error = _posted_proposed_task_for_new_session(
        request, target
    )
    if proposed_task_error is not None:
        return HttpResponseBadRequest(proposed_task_error)
    proposed_session, proposed_session_error = _posted_proposed_session_for_new_session(
        request, target
    )
    if proposed_session_error is not None:
        return HttpResponseBadRequest(proposed_session_error)
    cwd = target.cwd
    if not prompt:
        return HttpResponseBadRequest("prompt is required")
    if not cwd:
        return HttpResponseBadRequest("cwd is required")
    # Restrict cwd to a discovered repo so an arbitrary path can't be injected
    # via the form post.
    allowed = {str(p) for p in discover_repos()}
    if cwd not in allowed:
        return HttpResponseBadRequest("cwd must be a discovered repository")

    # Re-reconcile the cookies against Codex's current model list before
    # spawning. A long-lived tab might still be carrying a model the index
    # render would have snapped away from; without this, a stale value
    # would ride straight into ``thread_start(model=...)`` and 500 the
    # new-session click.
    initial_settings = _stored_settings(request)
    config = codex_pool.app_server_config(
        enable_memories=initial_settings.enable_memories
    )
    with Codex(config=config) as codex:
        models_data = list(codex.models().data)
    resolved_settings = _resolved_settings(request, models_data)
    settings = resolved_settings.values
    cookie_updates = resolved_settings.cookie_updates
    source_project = target.project
    default_auto_pr_enabled = _effective_auto_pr_enabled(
        None if target.project_cleared else source_project,
        global_enabled=settings.auto_pr_enabled,
    )
    auto_pr_enabled, auto_pr_error = _posted_auto_pr_override(
        request.POST.get("auto_pr"), default=default_auto_pr_enabled
    )
    if auto_pr_error is not None:
        return HttpResponseBadRequest(auto_pr_error)
    if plan_mode and not settings.model:
        return HttpResponseBadRequest("plan mode requires a model")

    session_cwd = cwd
    # QA workflows review the selected repo's current diff; a fresh managed
    # worktree would be clean and miss uncommitted changes.
    if qa_workflow_activation:
        thread_name = _PR_SLASH_PROMPT if pr_activation else _QA_SLASH_PROMPT
        base_instructions = _base_instructions_for_settings(settings)
        create_thread_kwargs: dict[str, Any] = {
            "cwd": session_cwd,
            "name": thread_name,
            "developer_instructions": settings.extra_system_prompt or None,
            "model": settings.model or None,
            "enable_memories": settings.enable_memories,
        }
        if base_instructions:
            create_thread_kwargs["base_instructions"] = base_instructions
        thread_id = codex_pool.create_session_thread(**create_thread_kwargs)
        workflow_kwargs: dict[str, Any] = {
            "main_thread_id": thread_id,
            "cwd": session_cwd,
            "sandbox_policy": settings.sandbox_policy or None,
            "approval_mode": settings.approval_mode,
            "model": settings.model or None,
            "reasoning_effort": settings.reasoning_effort or None,
            "developer_instructions": settings.extra_system_prompt or None,
            "enable_memories": settings.enable_memories,
            "initial_user_message_index": 0,
        }
        if base_instructions:
            workflow_kwargs["base_instructions"] = base_instructions
        if qa_activation:
            workflow_kwargs["open_pr_on_lgtm"] = False
        system_agents.start_pr_qa_workflow(**workflow_kwargs)
        session_metadata, _created = SessionMetadata.objects.update_or_create(
            thread_id=thread_id,
            defaults={
                "cwd": session_cwd,
                "project": source_project,
                "project_cleared": target.project_cleared,
                "auto_pr_enabled": False,
            },
        )
        _accept_proposed_task_for_session(proposed_task, session_metadata)
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

    managed_worktree = None
    if settings.use_worktrees:
        try:
            managed_worktree = create_worktree_for_session(cwd)
        except WorktreeCreationError as exc:
            return HttpResponseBadRequest(str(exc))
        session_cwd = str(managed_worktree.path)
    session_project = (
        None
        if target.project_cleared
        else _project_for_cwd(session_cwd, projects) or source_project
    )

    # Detach a worker subprocess so the initial turn keeps running past a
    # Django restart. The thread itself is created synchronously to give the
    # caller a stable id to redirect to.
    spawn_kwargs: dict[str, Any] = {
        "cwd": session_cwd,
        "prompt": prompt,
        "developer_instructions": settings.extra_system_prompt or None,
        "model": settings.model or None,
        "reasoning_effort": settings.reasoning_effort or None,
        "sandbox_policy": settings.sandbox_policy or None,
        "approval_mode": settings.approval_mode,
    }
    base_instructions = _base_instructions_for_settings(settings)
    if base_instructions:
        spawn_kwargs["base_instructions"] = base_instructions
    if settings.enable_memories:
        spawn_kwargs["enable_memories"] = True
    if plan_mode:
        spawn_kwargs["plan_mode"] = True
    if auto_pr_enabled:
        spawn_kwargs["auto_pr_enabled"] = True
    try:
        instance = codex_pool.spawn_new_session(**spawn_kwargs)
    except Exception:
        if managed_worktree is not None:
            try:
                cleanup_worktree(managed_worktree)
            except WorktreeCleanupError:
                logger.exception(
                    "failed to clean up managed worktree %s", managed_worktree.path
                )
        raise
    session_metadata, _created = SessionMetadata.objects.update_or_create(
        thread_id=instance.thread_id,
        defaults={
            "cwd": session_cwd,
            "project": session_project,
            "project_cleared": target.project_cleared,
            "auto_pr_enabled": auto_pr_enabled,
        },
    )
    _accept_proposed_task_for_session(proposed_task, session_metadata)
    _accept_proposed_session_for_session(proposed_session, session_metadata)
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


def _collapse_flat_entries(flat: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Apply the same intermediate-collapsing as ``_render_entries`` to the
    flat per-item entries produced by the rollout parser.

    Turn boundaries are detected via the user kind because the rollout file
    is a chronological log without per-item turn ids exposed at the entry
    level. This matches the SDK's `handle_user_message` fallback for streams
    that did not open turns explicitly.
    """
    turn: list[dict[str, Any]] = []
    for entry in flat:
        if entry["kind"] == "user" and turn:
            yield from _emit_collapsed_turn(turn)
            turn = []
        turn.append(entry)
    if turn:
        yield from _emit_collapsed_turn(turn)


def _emit_collapsed_turn(turn: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    final_idx = _final_agent_idx_in_flat(turn)
    intermediate: list[dict[str, Any]] = []
    for i, entry in enumerate(turn):
        if i == final_idx:
            if intermediate:
                yield _make_intermediate_entry(intermediate)
                intermediate = []
            yield _finalize_agent_entry(_strip_phase(entry))
        elif entry["kind"] == "user":
            # `_collapse_flat_entries` splits on every user past the first, so
            # any user reaching this branch is the leading entry of the turn
            # and intermediate is empty.
            yield entry
        elif entry["kind"] == "agent":
            intermediate.append({**_strip_phase(entry), "kind": "thinking"})
        elif entry["kind"] in {"approval_declined", "plan"}:
            if intermediate:
                yield _make_intermediate_entry(intermediate)
                intermediate = []
            yield entry
        else:
            intermediate.append(entry)
    if intermediate:
        yield _make_intermediate_entry(intermediate)


def _final_agent_idx_in_flat(entries: list[dict[str, Any]]) -> int:
    for i in range(len(entries) - 1, -1, -1):
        entry = entries[i]
        if entry["kind"] == "agent" and entry.get("phase") == "final_answer":
            return i
    for i in range(len(entries) - 1, -1, -1):
        entry = entries[i]
        if entry["kind"] != "agent":
            continue
        if entry.get("phase") == "commentary":
            continue
        return i
    return -1


def _strip_phase(entry: dict[str, Any]) -> dict[str, Any]:
    """Return ``entry`` minus its ``phase`` key.

    Called on every agent entry just before it leaves the collapsing pass so
    the template never sees the internal phase marker. The rollout parser
    sets ``phase`` on every agent entry (to ``None`` when absent), so this
    function never has to consult the dict before stripping.
    """
    return {k: v for k, v in entry.items() if k != "phase"}


def _finalize_agent_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Annotate a turn's final agent entry with rendered-markdown HTML.

    Only the final agent message of a turn is ever fed through this
    function, so collapsed "thinking" messages stay plain-text. A failure
    to recognise the text as markdown leaves the entry untouched, and the
    template falls back to the plain-text body.
    """
    text = entry.get("text")
    if isinstance(text, str) and looks_like_markdown(text):
        entry["html"] = render_markdown(text)
    return entry


def _render_entries(thread: Any) -> Iterator[dict[str, Any]]:
    """Walk every turn's items in order, surfacing the user message and the
    final agent reply as top-level entries and folding everything else
    (intermediate agent commentary plus every tool-call variant) into a
    single collapsible "intermediate" entry per run so long sessions don't
    bury the actual answer behind dozens of tool rows.

    The SDK marks final responses with MessagePhase.final_answer when known;
    for sessions where phase is unset (older data or an in-progress turn)
    the last agentMessage in the turn is treated as final. Each entry
    carries the turn's started_at timestamp; per-item timestamps are not
    exposed by the SDK.
    """
    for turn in getattr(thread, "turns", []) or []:
        timestamp = getattr(turn, "started_at", None)
        items = [thread_item.root for thread_item in turn.items]
        final_idx = _find_final_agent_idx(items)
        intermediate: list[dict[str, Any]] = []

        for i, item in enumerate(items):
            if i == final_idx:
                if intermediate:
                    yield _make_intermediate_entry(intermediate)
                    intermediate = []
                agent_entry: dict[str, Any] = {
                    "kind": "agent",
                    "text": item.text,
                    "timestamp": timestamp,
                }
                memory_citation = _memory_citation_from_item(item)
                if memory_citation is not None:
                    agent_entry["memory_citation"] = memory_citation
                yield _finalize_agent_entry(agent_entry)
            elif item.type == "userMessage":
                if intermediate:
                    yield _make_intermediate_entry(intermediate)
                    intermediate = []
                yield {
                    "kind": "user",
                    "text": _user_message_text(item),
                    "timestamp": timestamp,
                }
            elif item.type == "agentMessage":
                thinking_entry: dict[str, Any] = {
                    "kind": "thinking",
                    "text": item.text,
                    "timestamp": timestamp,
                }
                memory_citation = _memory_citation_from_item(item)
                if memory_citation is not None:
                    thinking_entry["memory_citation"] = memory_citation
                intermediate.append(thinking_entry)
            elif item.type == "plan":
                if intermediate:
                    yield _make_intermediate_entry(intermediate)
                    intermediate = []
                yield {
                    "kind": "plan",
                    "text": getattr(item, "text", "") or "",
                    "timestamp": timestamp,
                }
            else:
                intermediate.append(_make_tool_call_entry(item, timestamp))

        if intermediate:
            yield _make_intermediate_entry(intermediate)


def _find_final_agent_idx(items: list[Any]) -> int:
    """Index of the agent message to display as this turn's final response, or
    -1 if there is no agent message that could be the final.

    An explicit MessagePhase.final_answer always wins (the last one if
    multiple). Otherwise the last agent message whose phase is not
    MessagePhase.commentary is treated as final — i.e. unset phases (older
    sessions / in-progress turns) are eligible to be the final, but explicit
    commentary never is.
    """
    for i in range(len(items) - 1, -1, -1):
        item = items[i]
        if item.type == "agentMessage" and _phase_value(item) == "final_answer":
            return i
    for i in range(len(items) - 1, -1, -1):
        item = items[i]
        if item.type != "agentMessage":
            continue
        if _phase_value(item) == "commentary":
            continue
        return i
    return -1


def _phase_value(item: Any) -> str | None:
    # The SDK normally deserializes `phase` into a MessagePhase enum instance
    # (with `.value`), but accept a raw wire string too so this stays robust
    # against thread data that bypasses pydantic deserialization.
    phase = getattr(item, "phase", None)
    if phase is None:
        return None
    if isinstance(phase, str):
        return phase
    value = getattr(phase, "value", None)
    return value if isinstance(value, str) else None


def _memory_citation_from_item(item: Any) -> dict[str, Any] | None:
    value = _value_for(item, "memory_citation")
    if value is None:
        value = _value_for(item, "memoryCitation")
    if value is None:
        return None

    entries: list[dict[str, Any]] = []
    for raw_entry in _sequence_value(_value_for(value, "entries")):
        path = _string_value(_value_for(raw_entry, "path"))
        line_start = _int_value(_value_for(raw_entry, "line_start"))
        if line_start == 0:
            line_start = _int_value(_value_for(raw_entry, "lineStart"))
        line_end = _int_value(_value_for(raw_entry, "line_end"))
        if line_end == 0:
            line_end = _int_value(_value_for(raw_entry, "lineEnd"))
        if not path or line_start == 0 or line_end == 0:
            continue
        entries.append(
            {
                "path": path,
                "line_start": line_start,
                "line_end": line_end,
                "note": _string_value(_value_for(raw_entry, "note")),
            }
        )

    thread_ids = [
        thread_id
        for raw_id in _sequence_value(
            _value_for(value, "thread_ids") or _value_for(value, "threadIds")
        )
        if (thread_id := _string_value(raw_id))
    ]
    count = len(entries) if entries else len(thread_ids)
    if count == 0:
        return None
    return {"count": count, "entries": entries, "thread_ids": thread_ids}


def _value_for(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _sequence_value(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def _make_tool_call_entry(item: Any, timestamp: Any) -> dict[str, Any]:
    item_type = item.type
    return {
        "kind": "tool_call",
        "type": item_type,
        "label": _NON_MESSAGE_LABELS.get(item_type, item_type),
        "detail": _tool_call_detail(item, item_type),
        "status": _tool_call_status(item),
        "timestamp": timestamp,
    }


def _make_intermediate_entry(items: list[dict[str, Any]]) -> dict[str, Any]:
    thinking_count = sum(1 for e in items if e["kind"] == "thinking")
    tool_call_count = sum(1 for e in items if e["kind"] == "tool_call")
    return {
        "kind": "intermediate",
        "thinking_count": thinking_count,
        "tool_call_count": tool_call_count,
        "items": items,
    }


def _user_message_text(item: Any) -> str:
    parts: list[str] = []
    for input_item in item.content:
        inner = input_item.root
        match inner.type:
            case "text":
                parts.append(inner.text)
            case "mention":
                parts.append(f"@{inner.name}")
            case "skill":
                parts.append(f"/{inner.name}")
            case "image":
                parts.append("[image]")
            case "localImage":
                parts.append(f"[image: {inner.path}]")
    return "\n".join(parts)


def _tool_call_detail(item: Any, item_type: str) -> str:
    """Return a short, human-readable description of a tool-call item.

    Returns an empty string for item types that do not carry useful inline
    detail; the label alone is enough to surface them in the UI.
    """
    match item_type:
        case "commandExecution":
            return getattr(item, "command", "") or ""
        case "mcpToolCall":
            return f"{item.server} / {item.tool}"
        case "dynamicToolCall":
            namespace = getattr(item, "namespace", None)
            return f"{namespace}::{item.tool}" if namespace else item.tool
        case "fileChange":
            paths = [str(change.path) for change in getattr(item, "changes", []) or []]
            if not paths:
                return ""
            if len(paths) == 1:
                return paths[0]
            return f"{paths[0]} (+{len(paths) - 1} more)"
        case "webSearch":
            return getattr(item, "query", "") or ""
        case "plan":
            text = getattr(item, "text", "") or ""
            return text.split("\n", 1)[0]
        case "imageView":
            return getattr(item, "path", "") or ""
        case "imageGeneration":
            return (
                getattr(item, "revised_prompt", None)
                or getattr(item, "saved_path", None)
                or ""
            )
        case "collabAgentToolCall":
            tool = getattr(item, "tool", None)
            tool_name = getattr(tool, "value", None) or (tool if isinstance(tool, str) else "")
            receivers = getattr(item, "receiver_thread_ids", None) or []
            if receivers:
                return f"{tool_name} → {receivers[0]}"
            return str(tool_name)
        case _:
            return ""


def _tool_call_status(item: Any) -> str | None:
    """Return a non-success status string (e.g. ``failed``) or None.

    Hides ``completed`` so the common case stays uncluttered; surfaces
    in-progress, failed, and declined so unusual outcomes are visible.
    """
    status = getattr(item, "status", None)
    if status is None:
        return None
    value = getattr(status, "value", status)
    if not isinstance(value, str) or value == "completed":
        return None
    return value
