"""Settings and project management endpoints."""
import math
from typing import Any

from django.http import (
    HttpRequest,
    HttpResponse,
    HttpResponseBadRequest,
)
from django.shortcuts import redirect, render
from django.urls import reverse
from django.views.decorators.http import require_http_methods
from openai_codex import AppServerError
from openai_codex.generated.v2_all import (
    ReasoningEffort,
)

from hitch.main import caches, coding_agents
from hitch.main import repos as repos_module
from hitch.main.models import (
    CodexInstance,
    GlobalSettings,
    Project,
    SessionMetadata,
)
from hitch.main.repos import same_repo_or_worktree
from hitch.main.runtime import app_server_pool
from hitch.main.sessions.project_visibility import (
    _metadata_by_thread_id as _metadata_by_thread_id,
)
from hitch.main.sessions.project_visibility import (
    _settings_with_visible_selected_project,
)
from hitch.main.sessions.session_settings import (
    _authenticated_user,
    _cached_models_and_settings,
    _save_user_settings,
    _stored_settings,
    _supported_effort_values,
)
from hitch.main.sessions.settings_cookies import (
    _DEFAULT_APPROVAL_MODE,
    _EXTRA_SYSTEM_PROMPT_MAX_LEN,
    _MODEL_MAX_LEN,
    _VALID_APPROVAL_MODES,
    _VALID_SANDBOX_POLICIES,
    _VALID_WEB_SEARCH_MODES,
    SettingsValues,
    _apply_cookie_updates,
    _extra_system_prompt_cookie_fits,
    _settings_cookie_updates,
    _visible_session_project_ids_cookie_fits,
)
from hitch.main.views import common
from hitch.main.workflows import system_agents

_VALID_PROJECT_AUTO_PR_MODES = {value for value, _label in Project.AUTO_PR_CHOICES}

def _apply_live_global_approval_mode(effective_approval_mode: str) -> None:
    explicit_override_thread_ids = SessionMetadata.objects.filter(
        approval_mode__in=_VALID_APPROVAL_MODES
    ).values("thread_id")
    common._apply_live_approval_mode_to_instances(
        CodexInstance.objects.filter(
            purpose=CodexInstance.PURPOSE_USER,
            status__in=CodexInstance.ACTIVE_STATUSES,
        ).exclude(thread_id__in=explicit_override_thread_ids),
        effective_approval_mode,
    )

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

def _associate_existing_sessions_with_project(project: Project, request: HttpRequest) -> None:
    settings = _stored_settings(request)
    try:
        with app_server_pool.borrow_codex(
            common.Codex, enable_memories=settings.enable_memories
        ) as codex:
            threads = common._all_threads(codex)
            try:
                threads.extend(common._all_threads(codex, archived=True))
            except AppServerError:
                common.logger.warning("failed to list archived sessions while creating project")
    except AppServerError:
        common.logger.warning("failed to list sessions while creating project")
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
        cwd = common._thread_cwd(thread)
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
        repo_common_dir = str(common.git_common_dir(repo_path) or "")
        if _matching_project_exists(repo_path, repo_common_dir):
            continue
        creatable.append(repo_path)
    return creatable

@require_http_methods(["GET", "POST"])
def update_settings(request: HttpRequest) -> HttpResponse:
    if request.method == "GET":
        models_data, resolved_settings = _cached_models_and_settings(request)
        next_url = common._safe_next_url(request) or reverse("index")
        response = render(
            request,
            "settings.html",
            {
                "settings_next_url": next_url,
                "settings_cancel_url": next_url,
                **common._settings_context(
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
    selected_project, selected_project_error = common._posted_project(
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
            with app_server_pool.borrow_codex(
                common.Codex, enable_memories=enable_memories_value
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
    response = redirect(common._safe_next_url(request) or "index")
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
    response = redirect(common._safe_next_url(request) or "index")
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
                "name_max_len": common._PROJECT_NAME_MAX_LEN,
                "index_url": reverse("index"),
            },
        )

    name = request.POST.get("name", "").strip()
    repo_path = request.POST.get("repo_path", "").strip()
    if not name:
        return HttpResponseBadRequest("project name is required")
    if len(name) > common._PROJECT_NAME_MAX_LEN:
        return HttpResponseBadRequest("project name is too long")
    if not repo_path:
        return HttpResponseBadRequest("repository is required")
    if repo_path not in set(discovered_repos):
        return HttpResponseBadRequest("repository must be a discovered repository")
    repo_common_dir = str(common.git_common_dir(repo_path) or "")
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
    project, project_error = common._posted_project(request.POST.get("project", ""))
    if project_error is not None:
        return HttpResponseBadRequest(project_error)
    if project is None:
        return HttpResponseBadRequest("project is required")
    name = request.POST.get("name", "").strip()
    extra_system_prompt = request.POST.get("extra_system_prompt", "").strip()
    auto_pr_mode = request.POST.get("auto_pr_mode", "").strip()
    auto_pull = request.POST.get("auto_pull", "").strip()
    if not name:
        return HttpResponseBadRequest("project name is required")
    if len(name) > common._PROJECT_NAME_MAX_LEN:
        return HttpResponseBadRequest("project name is too long")
    if len(extra_system_prompt) > _EXTRA_SYSTEM_PROMPT_MAX_LEN:
        return HttpResponseBadRequest("extra system prompt is too long")
    if auto_pr_mode not in _VALID_PROJECT_AUTO_PR_MODES:
        return HttpResponseBadRequest("invalid project auto-PR setting")
    if auto_pull not in {"", "true"}:
        return HttpResponseBadRequest("invalid project auto-pull setting")
    auto_pull_enabled = auto_pull == "true"

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
    if project.auto_pull_enabled != auto_pull_enabled:
        project.auto_pull_enabled = auto_pull_enabled
        updates.append("auto_pull_enabled")
    if updates:
        project.save(update_fields=[*updates, "updated_at"])
    return redirect(common._safe_next_url(request) or "index")

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
