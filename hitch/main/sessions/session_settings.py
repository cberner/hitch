"""Shared settings, model-config, and session-config helper foundation.

These non-endpoint helpers own the stored/resolved user settings, model
reasoning-effort reconciliation, sandbox/approval-mode resolution, and the
settings/new-session dialog context builders read by the settings, new-session,
and session endpoints in ``views``. This module must not import ``views``.
"""

from __future__ import annotations

from typing import Any

from django.http import HttpRequest
from django.urls import reverse

from hitch.main import caches, coding_agents
from hitch.main import repos as repos_module
from hitch.main import worktrees as worktrees_module
from hitch.main.models import (
    Project,
    ProposedSession,
    SessionMetadata,
    UserSettings,
)
from hitch.main.runtime import disk_cleanup
from hitch.main.runtime.input_images import _INPUT_IMAGE_ACCEPT
from hitch.main.sessions.session_pr_plan import _PR_SLASH_PROMPT
from hitch.main.sessions.settings_cookies import (
    _APPROVAL_COOKIE,
    _AUTO_PR_COOKIE,
    _AUTO_QA_COOKIE,
    _CODING_AGENT_COOKIE,
    _DEFAULT_APPROVAL_MODE,
    _EFFORT_COOKIE,
    _ENABLE_MEMORIES_COOKIE,
    _LAST_SELECTED_REPO_COOKIE,
    _MANAGED_WORKTREE_DEFAULT_SANDBOX_POLICY,
    _MODEL_COOKIE,
    _SANDBOX_COOKIE,
    _SHOW_ARCHIVED_COOKIE,
    _SHOW_NO_PROJECT_SESSIONS_COOKIE,
    _SPEC_CRITIC_COOKIE,
    _USE_WORKTREES_COOKIE,
    _VALID_APPROVAL_MODES,
    _VALID_SANDBOX_POLICIES,
    _VALID_WEB_SEARCH_MODES,
    _WEB_SEARCH_COOKIE,
    _WEB_SEARCH_MODE_OPTIONS,
    ResolvedSettings,
    SessionProjectVisibility,
    SettingsValues,
    _effective_coding_agent,
    _option_label,
    _read_cookie,
    _read_extra_system_prompt_cookie,
    _read_selected_project_cookie,
    _read_visible_session_project_ids_cookie,
    _settings_cookie_updates,
    _valid_visible_session_project_ids,
    _web_search_mode_label,
)
from hitch.main.workflows import system_agents

_BARE_REPO_PROJECT_VALUE = "__bare_repo__"
_QA_SLASH_PROMPT = system_agents.QA_SLASH_DISPLAY_PROMPT


def _new_session_form_context(
    current_settings: SettingsValues,
    current_project: Project | None,
    projects: list[Project],
    *,
    initial_prompt: str = "",
    proposed_session: ProposedSession | None = None,
    prefill_bare_repo_cwd: str = "",
    repos: list[str] | None = None,
) -> dict[str, Any]:
    if repos is None:
        repos = [str(p) for p in repos_module.discover_repos()]
    repo_set = set(repos)
    if prefill_bare_repo_cwd not in repo_set:
        prefill_bare_repo_cwd = ""
    saved_repo = ""
    if prefill_bare_repo_cwd:
        saved_repo = prefill_bare_repo_cwd
    elif current_settings.last_selected_repo in repo_set:
        saved_repo = current_settings.last_selected_repo
    new_session_projects = [
        project for project in projects if project.repo_path in repo_set
    ]
    selected_project = (
        _project_for_proposed_session(proposed_session)
        if proposed_session is not None
        else current_project
    )
    current_new_session_project = (
        None
        if prefill_bare_repo_cwd
        else _new_session_project_for_dialog(
            selected_project, saved_repo, new_session_projects
        )
    )
    current_new_session_auto_pr = _effective_auto_pr_enabled(
        current_new_session_project,
        global_enabled=current_settings.auto_pr_enabled,
    )
    current_new_session_auto_qa = (
        current_settings.auto_qa_enabled and not current_new_session_auto_pr
    )
    current_coding_agent = _effective_coding_agent(current_settings)
    return {
        "repos": repos,
        "new_session_projects": new_session_projects,
        "new_session_url": reverse("new_session"),
        "new_session_cancel_url": (
            reverse("inbox") if proposed_session is not None else reverse("index")
        ),
        "initial_new_session_prompt": initial_prompt,
        "initial_proposed_session_id": (
            proposed_session.pk if proposed_session is not None else ""
        ),
        "current_repo": _selected_repo_for_dialog(
            saved_repo, repos, current_new_session_project
        ),
        "current_new_session_project_id": (
            current_new_session_project.pk
            if current_new_session_project is not None
            else ""
        ),
        "current_new_session_use_worktrees": current_settings.use_worktrees,
        "current_new_session_auto_pr": current_new_session_auto_pr,
        "current_new_session_auto_qa": current_new_session_auto_qa,
        "bare_repo_project_value": _BARE_REPO_PROJECT_VALUE,
        "new_session_coding_agent_options": [
            {"id": value, "display_name": label}
            for value, label in coding_agents.CODING_AGENT_OPTIONS
        ],
        "new_session_default_coding_agent_label": _option_label(
            coding_agents.CODING_AGENT_OPTIONS, current_coding_agent
        ),
        "new_session_web_search_options": [
            {"id": value, "display_name": label}
            for value, label in _WEB_SEARCH_MODE_OPTIONS
            if value
        ],
        "new_session_default_web_search_label": _web_search_mode_label(
            current_settings.web_search_mode
        ),
        "input_image_accept": _INPUT_IMAGE_ACCEPT,
        "pr_slash_prompt": _PR_SLASH_PROMPT,
        "qa_slash_prompt": _QA_SLASH_PROMPT,
    }


def _effective_sandbox_policy(settings: SettingsValues) -> str:
    sandbox_policy = settings.sandbox_policy
    if sandbox_policy and sandbox_policy not in _VALID_SANDBOX_POLICIES:
        return ""
    return sandbox_policy


def _effective_sandbox_policy_for_cwd(
    settings: SettingsValues,
    cwd: str,
    *,
    managed_worktree: bool = False,
) -> str:
    sandbox_policy = _effective_sandbox_policy(settings)
    if sandbox_policy:
        return sandbox_policy
    if managed_worktree or _is_managed_session_cwd(cwd):
        return _MANAGED_WORKTREE_DEFAULT_SANDBOX_POLICY
    return ""


def _is_managed_session_cwd(cwd: str) -> bool:
    if worktrees_module.is_managed_worktree_path(cwd):
        return True
    return cwd in {str(path) for path in worktrees_module.discover_managed_worktrees()}


def _effective_approval_mode(settings: SettingsValues) -> str:
    if settings.approval_mode not in _VALID_APPROVAL_MODES:
        return _DEFAULT_APPROVAL_MODE
    return settings.approval_mode


def _session_approval_mode_override(
    session_id: str, metadata: SessionMetadata | None = None
) -> str:
    if metadata is None:
        value = (
            SessionMetadata.objects.filter(thread_id=session_id)
            .values_list("approval_mode", flat=True)
            .first()
            or ""
        )
    else:
        value = metadata.approval_mode
    return value if value in _VALID_APPROVAL_MODES else ""


def _effective_approval_mode_for_session(
    settings: SettingsValues,
    session_id: str,
    metadata: SessionMetadata | None = None,
) -> str:
    override = _session_approval_mode_override(session_id, metadata)
    return override or _effective_approval_mode(settings)


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


def _current_disk_usage_max_percent() -> float:
    return disk_cleanup._max_allowed_percent()


def _format_disk_usage_max_percent(value: float) -> str:
    value = round(value, 1)
    if value.is_integer():
        return str(int(value))
    return f"{value:.1f}"


def _session_project_visibility_for_settings(
    settings: SettingsValues, projects: list[Project]
) -> SessionProjectVisibility:
    project_ids = {project.pk for project in projects}
    if settings.visible_session_project_ids is None:
        if (
            settings.selected_project_id is not None
            and settings.selected_project_id in project_ids
        ):
            return SessionProjectVisibility(
                project_ids=frozenset({settings.selected_project_id}),
                include_no_project=False,
            )
        return SessionProjectVisibility(project_ids=None, include_no_project=True)
    return SessionProjectVisibility(
        project_ids=frozenset(
            project_id
            for project_id in settings.visible_session_project_ids
            if project_id in project_ids
        ),
        include_no_project=settings.show_no_project_sessions,
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
    saved_web_search = saved.web_search_mode
    if saved_sandbox and saved_sandbox not in _VALID_SANDBOX_POLICIES:
        saved_sandbox = ""
    if saved_approval not in _VALID_APPROVAL_MODES:
        saved_approval = _DEFAULT_APPROVAL_MODE
    if saved_web_search and saved_web_search not in _VALID_WEB_SEARCH_MODES:
        saved_web_search = ""
    saved = saved._replace(
        sandbox_policy=saved_sandbox,
        approval_mode=saved_approval,
        web_search_mode=saved_web_search,
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
        auto_qa_enabled=_read_cookie(request, _AUTO_QA_COOKIE) == "true",
        spec_critic_enabled=_read_cookie(request, _SPEC_CRITIC_COOKIE) == "true",
        web_search_mode=_read_cookie(request, _WEB_SEARCH_COOKIE),
        show_archived_sessions=_read_cookie(request, _SHOW_ARCHIVED_COOKIE) == "true",
        last_selected_repo=_read_cookie(request, _LAST_SELECTED_REPO_COOKIE),
        selected_project_id=_read_selected_project_cookie(request),
        visible_session_project_ids=_read_visible_session_project_ids_cookie(request),
        show_no_project_sessions=(
            _read_cookie(request, _SHOW_NO_PROJECT_SESSIONS_COOKIE) != "false"
        ),
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
        auto_qa_enabled=settings.auto_qa_enabled,
        spec_critic_enabled=settings.spec_critic_enabled,
        web_search_mode=settings.web_search_mode,
        show_archived_sessions=settings.show_archived_sessions,
        last_selected_repo=settings.last_selected_repo,
        selected_project_id=settings.selected_project_id,
        visible_session_project_ids=_valid_visible_session_project_ids(
            settings.visible_session_project_ids
        ),
        show_no_project_sessions=settings.show_no_project_sessions,
        enable_memories=settings.enable_memories,
    )


def _save_user_settings(user: Any, values: SettingsValues) -> UserSettings:
    settings = _settings_for_user(user)
    updates: list[str] = []
    visible_session_project_ids = (
        list(values.visible_session_project_ids)
        if values.visible_session_project_ids is not None
        else None
    )
    for field, value in (
        ("model", values.model),
        ("reasoning_effort", values.reasoning_effort),
        ("sandbox_policy", values.sandbox_policy),
        ("approval_mode", values.approval_mode),
        ("coding_agent", values.coding_agent),
        ("extra_system_prompt", values.extra_system_prompt),
        ("use_worktrees", values.use_worktrees),
        ("auto_pr_enabled", values.auto_pr_enabled),
        ("auto_qa_enabled", values.auto_qa_enabled),
        ("spec_critic_enabled", values.spec_critic_enabled),
        ("web_search_mode", values.web_search_mode),
        ("show_archived_sessions", values.show_archived_sessions),
        ("last_selected_repo", values.last_selected_repo),
        ("selected_project_id", values.selected_project_id),
        ("visible_session_project_ids", visible_session_project_ids),
        ("show_no_project_sessions", values.show_no_project_sessions),
        ("enable_memories", values.enable_memories),
    ):
        if getattr(settings, field) != value:
            setattr(settings, field, value)
            updates.append(field)
    if updates:
        settings.save(update_fields=[*updates, "updated_at"])
    return settings


def _cached_models_and_settings(request: HttpRequest) -> tuple[list[Any], ResolvedSettings]:
    stored_settings = _stored_settings(request)
    models_data = caches._cached_models_data(enable_memories=stored_settings.enable_memories)
    caches._schedule_models_refresh(enable_memories=stored_settings.enable_memories)
    return models_data, _resolved_settings(request, models_data)


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


def _project_for_proposed_session(
    proposed_session: ProposedSession | None,
) -> Project | None:
    if proposed_session is None:
        return None
    if proposed_session.project is not None:
        return proposed_session.project
    if proposed_session.autonomous_goal is not None:
        return proposed_session.autonomous_goal.project
    return None


def _allowed_session_cwds() -> set[str]:
    return {
        str(p)
        for p in [
            *repos_module.discover_repos(),
            *worktrees_module.discover_managed_worktrees(),
        ]
    }
