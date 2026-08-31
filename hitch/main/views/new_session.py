"""The new-session page and start flow, including proposal acceptance."""

import re
from typing import Any, NamedTuple

from django.http import (
    Http404,
    HttpRequest,
    HttpResponse,
    HttpResponseBadRequest,
)
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_http_methods

from hitch.main import caches
from hitch.main import repos as repos_module
from hitch.main.goals.autonomous_goal_proposal_stack import (
    AUTONOMOUS_GOAL_ACCEPTED_SNAPSHOT_METADATA_KEY,
    AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_METADATA_KEY,
    AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_REF_METADATA_KEY,
    AUTONOMOUS_GOAL_TOOL_PROTOCOL_METADATA_KEY,
    _proposal_outcome_metadata,
)
from hitch.main.goals.autonomous_goal_run_display import (
    _accepted_proposal_prompt,
    _attach_proposed_session_display_state,
    _auto_review_settings_for_proposed_session,
    _proposed_session_prompt,
)
from hitch.main.models import (
    AutonomousGoal,
    Project,
    ProposedSession,
    SessionMetadata,
)
from hitch.main.runtime import codex_pool, reconciliation
from hitch.main.runtime.input_images import (
    _limit_input_image_uploads,
)
from hitch.main.sessions import agent_tasks, session_index
from hitch.main.sessions.message_intent import (
    _message_intent,
)
from hitch.main.sessions.pr_prompts import PR_SLASH_DISPLAY_PROMPT
from hitch.main.sessions.project_visibility import (
    _metadata_by_thread_id as _metadata_by_thread_id,
)
from hitch.main.sessions.session_settings import (
    _BARE_REPO_PROJECT_VALUE,
    _PLAN_MODE_REASONING_EFFORT,
    _QA_SLASH_PROMPT,
    _authenticated_user,
    _cached_models_and_settings,
    _effective_auto_pr_enabled,
    _effective_sandbox_policy_for_cwd,
    _new_session_form_context,
    _project_for_proposed_session,
    _reasoning_effort_values,
    _rendered_settings_with_guest_effort_default,
    _resolved_settings,
    _save_user_settings,
    _selected_project_for_settings,
    _stored_settings,
    _target_cwd_for_proposed_session,
    _validate_model_and_effort_against_models,
)
from hitch.main.sessions.settings_cookies import (
    _LAST_SELECTED_REPO_COOKIE,
    _MODEL_MAX_LEN,
    _REASONING_EFFORT_MAX_LEN,
    _VALID_WEB_SEARCH_MODES,
    ResolvedSettings,
    SettingsValues,
    _apply_cookie_updates,
    _settings_cookie_updates,
)
from hitch.main.views import common
from hitch.main.worktrees import (
    ManagedWorktree,
    WorktreeCleanupError,
    WorktreeCreationError,
    release_snapshot_commit_ref,
    snapshot_worktree_to_commit,
)


class _NewSessionTarget(NamedTuple):
    cwd: str
    project: Project | None
    project_cleared: bool
    requires_discovered_repo: bool


_GIT_OBJECT_ID_RE = re.compile(r"[0-9a-f]{40,64}")
_UPGRADE_RECOVERY_METADATA_KEY = "resume_source_session"


def _is_upgrade_recovery_proposal(
    proposed_session: ProposedSession | None,
) -> bool:
    return bool(
        proposed_session is not None
        and proposed_session.source_session is not None
        and isinstance(proposed_session.outcome_metadata, dict)
        and proposed_session.outcome_metadata.get(_UPGRADE_RECOVERY_METADATA_KEY)
        is True
    )


def _proposal_has_usable_stored_target(
    proposed_session: ProposedSession | None,
    cwd: str,
) -> bool:
    """Allow migrated proposals to start fresh from their stored repository."""
    return bool(
        _is_upgrade_recovery_proposal(proposed_session)
        and cwd
        and cwd == _target_cwd_for_proposed_session(proposed_session)
        and repos_module.repo_root(cwd) is not None
    )


def _proposal_has_explicit_auto_review_settings(
    proposed_session: ProposedSession,
) -> bool:
    metadata = proposed_session.outcome_metadata
    return bool(
        proposed_session.autonomous_goal is not None
        or (
            isinstance(metadata, dict)
            and ("auto_pr_enabled" in metadata or "auto_qa_enabled" in metadata)
        )
    )


def _new_session_post_models_and_settings(
    request: HttpRequest,
) -> tuple[list[Any], ResolvedSettings]:
    stored_settings = _stored_settings(request)
    enable_memories = stored_settings.enable_memories
    if caches._models_cache_has_value(enable_memories=enable_memories) and not caches._models_refresh_needed(
        enable_memories=enable_memories
    ):
        models_data = caches._cached_models_data(enable_memories=enable_memories)
        if models_data:
            return models_data, _resolved_settings(request, models_data)

    models_data = caches._fetch_models_data(enable_memories=enable_memories, codex_cls=common.Codex)
    return models_data, _resolved_settings(request, models_data)


def _posted_new_session_target(
    request: HttpRequest, projects: list[Project]
) -> tuple[_NewSessionTarget | None, str | None]:
    raw_project = request.POST.get("project")
    if raw_project is None:
        cwd = request.POST.get("cwd", "").strip()
        return (
            _NewSessionTarget(
                cwd,
                common._project_for_cwd(cwd, projects),
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

    project, error = common._posted_project(value)
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
    if session_id < 1 or session_id > common._MAX_BIGAUTOFIELD:
        return None, "proposed session is required"
    common._recover_stale_new_session_proposal_start_claims()
    proposed_session = (
        ProposedSession.objects.select_related(
            "project",
            "autonomous_goal__project",
            "candidate_session",
            "source_session",
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
    session_cwd = _target_cwd_for_proposed_session(proposed_session)
    if session_project is not None and target.project is not None:
        target_matches = target.project == session_project
    else:
        target_matches = bool(session_cwd) and target.cwd == session_cwd
    if not target_matches:
        return None, "proposed session does not match project"
    return proposed_session, None


def _approved_snapshot_for_proposal(
    proposed_session: ProposedSession | None,
) -> str:
    if proposed_session is None or proposed_session.autonomous_goal_id is None:
        return ""
    metadata = proposed_session.outcome_metadata
    if not isinstance(metadata, dict):
        return ""
    value = metadata.get(AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_METADATA_KEY)
    if not isinstance(value, str) or _GIT_OBJECT_ID_RE.fullmatch(value) is None:
        return ""
    return value


def _is_tool_protocol_ag_proposal(proposed_session: ProposedSession) -> bool:
    return (
        proposed_session.autonomous_goal_id is not None
        and isinstance(proposed_session.outcome_metadata, dict)
        and proposed_session.outcome_metadata.get(
            AUTONOMOUS_GOAL_TOOL_PROTOCOL_METADATA_KEY
        )
        is True
    )


def _release_approved_snapshot_ref(proposed_session: ProposedSession) -> None:
    metadata = proposed_session.outcome_metadata
    if not isinstance(metadata, dict):
        return
    ref = metadata.get(AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_REF_METADATA_KEY)
    project = _project_for_proposed_session(proposed_session)
    if not isinstance(ref, str) or not ref or project is None:
        return
    try:
        release_snapshot_commit_ref(project.repo_path, ref)
    except WorktreeCleanupError:
        common.logger.exception(
            "failed to release snapshot ref for proposed session %s",
            proposed_session.pk,
        )


def _cleanup_hidden_ag_candidate_worktree(
    proposed_session: ProposedSession,
) -> None:
    if (
        not _is_tool_protocol_ag_proposal(proposed_session)
        and _legacy_ag_candidate_for_snapshot(proposed_session) is None
    ):
        return
    candidate = proposed_session.candidate_session
    if candidate is None or not candidate.cwd:
        return
    try:
        common.cleanup_managed_worktree_path(candidate.cwd)
    except WorktreeCleanupError:
        common.logger.exception(
            "failed to clean up hidden candidate worktree for proposed session %s",
            proposed_session.pk,
        )


def _ag_proposal_requires_snapshot(
    proposed_session: ProposedSession,
) -> bool:
    metadata = proposed_session.outcome_metadata
    if not isinstance(metadata, dict):
        return True
    return (
        metadata.get("autonomous_goal_autonomy")
        != AutonomousGoal.AUTONOMY_PROPOSE_ONLY
    )


def _legacy_ag_candidate_for_snapshot(
    proposed_session: ProposedSession | None,
) -> SessionMetadata | None:
    if (
        proposed_session is None
        or proposed_session.autonomous_goal_id is None
        or _is_tool_protocol_ag_proposal(proposed_session)
        or not _ag_proposal_requires_snapshot(proposed_session)
    ):
        return None
    candidate = proposed_session.candidate_session
    if candidate is None or not candidate.cwd.strip():
        return None
    return candidate


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
            ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY: (claimed_at.isoformat()),
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
    claim_filter = common._new_session_proposal_start_claim_filter(proposed_session)
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
    proposed_session: ProposedSession | None,
    session_metadata: SessionMetadata,
    *,
    approved_snapshot: str,
) -> None:
    if proposed_session is None:
        return
    claim_filter = common._new_session_proposal_start_claim_filter(proposed_session)
    if claim_filter is None:
        return
    updates: dict[str, object] = {
        "accepted_by": "user",
        "resolved_by": "user",
        "accepted_session_id": session_metadata.pk,
        "accepted_thread_id": session_metadata.thread_id,
        ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY: None,
    }
    if proposed_session.autonomous_goal_id is not None:
        updates[AUTONOMOUS_GOAL_ACCEPTED_SNAPSHOT_METADATA_KEY] = (
            approved_snapshot or None
        )
    outcome_metadata = _proposal_outcome_metadata(proposed_session, updates)
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
    common._stop_autonomous_goal_stack_after_proposal_resolution(proposed_session)
    _release_approved_snapshot_ref(proposed_session)
    _cleanup_hidden_ag_candidate_worktree(proposed_session)


def _posted_bool_override(raw: str | None, *, default: bool, error: str) -> tuple[bool, str | None]:
    """Parse an optional posted checkbox override: absent keeps the default,
    ""/"false" disables, "true" enables, anything else is rejected."""
    if raw is None:
        return default, None
    value = raw.strip().lower()
    if value in {"", "false"}:
        return False, None
    if value == "true":
        return True, None
    return False, error


def _posted_web_search_override(raw: str | None, *, default: str) -> tuple[str, str | None]:
    if raw is None:
        return default, None
    value = raw.strip()
    if not value:
        return default, None
    if value in _VALID_WEB_SEARCH_MODES:
        return value, None
    return "", "invalid web search setting"


def _posted_model_settings_override(
    request: HttpRequest,
    *,
    default: SettingsValues,
    models_data: list[Any],
    plan_mode: bool,
) -> tuple[SettingsValues, str | None]:
    raw_model = request.POST.get("model")
    raw_effort = request.POST.get("reasoning_effort")
    if raw_model is None and raw_effort is None and not plan_mode:
        return default, None

    posted_model = None if raw_model is None else raw_model.strip()
    posted_effort = None if raw_effort is None else raw_effort.strip()
    rendered_model = request.POST.get("rendered_model")
    rendered_effort = request.POST.get("rendered_reasoning_effort")
    rendered_model_value = None if rendered_model is None else rendered_model.strip()
    rendered_effort_value = None if rendered_effort is None else rendered_effort.strip()
    has_rendered_snapshot = rendered_model_value is not None and rendered_effort_value is not None
    model_changed = posted_model is not None and (rendered_model_value is None or posted_model != rendered_model_value)
    effort_changed = posted_effort is not None and (
        rendered_effort_value is None or posted_effort != rendered_effort_value
    )
    if has_rendered_snapshot and not model_changed and not effort_changed and not plan_mode:
        return default, None

    model = posted_model if model_changed and posted_model is not None else default.model
    # Outside Plan mode, a changed model makes its accompanying effort an
    # explicit pair, even when it matches the rendered snapshot or is blank.
    effort_is_explicit = effort_changed or (model_changed and posted_effort is not None)
    effort = default.reasoning_effort
    if not plan_mode and effort_is_explicit and posted_effort is not None:
        effort = posted_effort
    effective_effort = _PLAN_MODE_REASONING_EFFORT.value if plan_mode else effort
    if len(model) > _MODEL_MAX_LEN:
        return default, "model id is too long"
    if len(effective_effort) > _REASONING_EFFORT_MAX_LEN:
        return default, "invalid reasoning effort"
    valid_efforts = set(
        _reasoning_effort_values(
            models_data,
            current_effort=default.reasoning_effort,
        )
    )
    if effective_effort and effective_effort not in valid_efforts:
        return default, "invalid reasoning effort"
    compatibility_error = _validate_model_and_effort_against_models(model, effective_effort, models_data)
    if compatibility_error is not None:
        return default, compatibility_error
    return default._replace(model=model, reasoning_effort=effort), None


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
    if session_id < 1 or session_id > common._MAX_BIGAUTOFIELD:
        raise Http404("proposed session not found")
    common._recover_stale_new_session_proposal_start_claims()
    proposed_session = (
        ProposedSession.objects.select_related(
            "project",
            "autonomous_goal__project",
            "candidate_session",
            "source_session",
        )
        .filter(
            pk=session_id,
            inbox_kind=ProposedSession.INBOX_KIND_PROPOSAL,
            outcome_status=ProposedSession.OUTCOME_UNSET,
        )
        .first()
    )
    target_cwd = _target_cwd_for_proposed_session(proposed_session)
    if (
        proposed_session is None
        or not target_cwd
        or (
            target_cwd not in repo_set
            and not _proposal_has_usable_stored_target(
                proposed_session,
                target_cwd,
            )
        )
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
        (project for project in projects if project.pk == project_id and project.repo_path in repo_set),
        None,
    )
    if project is None:
        raise Http404("project not found")
    return project


def _prefill_bare_repo_cwd_for_new_session_page(request: HttpRequest, *, repo_set: set[str]) -> str:
    cwd = request.GET.get("cwd", "").strip()
    if not cwd:
        return ""
    if cwd not in repo_set:
        raise Http404("repository not found")
    return cwd


def _render_new_session_page(request: HttpRequest) -> HttpResponse:
    reconciliation.reconcile_dead_if_due()
    repos = [str(p) for p in repos_module.discover_repos()]
    repo_set = set(repos)
    proposed_session = _proposed_session_for_new_session_page(request, repo_set=repo_set)
    proposed_session_cwd = _target_cwd_for_proposed_session(proposed_session)
    if (
        proposed_session_cwd
        and proposed_session_cwd not in repo_set
        and _proposal_has_usable_stored_target(
            proposed_session,
            proposed_session_cwd,
        )
    ):
        repos.append(proposed_session_cwd)
        repo_set.add(proposed_session_cwd)
    models_data, resolved_settings = _cached_models_and_settings(request)
    current_settings = resolved_settings.values
    rendered_settings = _rendered_settings_with_guest_effort_default(
        request,
        current_settings,
    )
    cookie_updates = resolved_settings.cookie_updates
    projects = list(Project.objects.all())
    current_project = _selected_project_for_settings(current_settings, projects)
    prefill_bare_repo_cwd = ""
    if proposed_session is not None:
        if _project_for_proposed_session(proposed_session) is None:
            prefill_bare_repo_cwd = _target_cwd_for_proposed_session(proposed_session)
            current_project = None
    else:
        prefill_project = _prefill_project_for_new_session_page(request, projects, repo_set=repo_set)
        if prefill_project is not None:
            current_project = prefill_project
        else:
            prefill_bare_repo_cwd = _prefill_bare_repo_cwd_for_new_session_page(request, repo_set=repo_set)
            if prefill_bare_repo_cwd:
                current_project = None
    settings_context = common._settings_context(rendered_settings, models_data)
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
            "plan_mode_reasoning_effort": _PLAN_MODE_REASONING_EFFORT.value,
            **settings_context,
            **new_session_context,
        },
    )
    _apply_cookie_updates(response, cookie_updates)
    return response


def _cleanup_worktree_quietly(managed_worktree: ManagedWorktree | None) -> None:
    if managed_worktree is None:
        return
    try:
        common.cleanup_worktree(managed_worktree)
    except WorktreeCleanupError:
        common.logger.exception("failed to clean up managed worktree %s", managed_worktree.path)


def _remember_repo_and_redirect(
    request: HttpRequest,
    cookie_updates: dict[str, str],
    *,
    cwd: str,
    thread_id: str,
) -> HttpResponse:
    """Persist the chosen repo as the last-selected one and redirect to the
    new session. Authenticated users get it saved on their settings row;
    anonymous users get the signed cookie."""
    user = _authenticated_user(request)
    if user is not None:
        # Re-read the persisted defaults so one-off new-session choices do not
        # become account-wide while remembering the repository.
        remembered_values = _stored_settings(request)._replace(last_selected_repo=cwd)
        _save_user_settings(user, remembered_values)
        cookie_updates = _settings_cookie_updates(remembered_values)
    else:
        cookie_updates = {**cookie_updates, _LAST_SELECTED_REPO_COOKIE: cwd}
    response = redirect("session", session_id=thread_id)
    _apply_cookie_updates(response, cookie_updates)
    return response


def _post_new_session(request: HttpRequest) -> HttpResponse:
    intent = _message_intent(request)
    pr_activation = intent.pr_activation
    pr_now_activation = intent.pr_now_activation
    fix_pr_activation = intent.fix_pr_activation
    qa_activation = intent.qa_activation
    agent_task_activation = (
        pr_activation or pr_now_activation or qa_activation or fix_pr_activation
    )
    prompt = intent.prompt
    plan_mode = False if agent_task_activation else intent.plan_mode
    has_input_images = common._has_input_image_uploads(request)
    if fix_pr_activation:
        return HttpResponseBadRequest("fix-pr requires an existing session with a PR")
    projects = list(Project.objects.all())
    target, target_error = _posted_new_session_target(request, projects)
    if target_error is not None or target is None:
        return HttpResponseBadRequest(target_error or "invalid project")
    proposed_session, proposed_session_error = (
        _posted_proposed_session_for_new_session(request, target)
    )
    if proposed_session_error is not None:
        return HttpResponseBadRequest(proposed_session_error)
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
        if cwd not in allowed and not _proposal_has_usable_stored_target(
            proposed_session,
            cwd,
        ):
            return HttpResponseBadRequest("cwd must be a discovered repository")

    # Re-reconcile the cookies against Codex's current model list before
    # spawning. A long-lived tab might still be carrying a model the index
    # render would have snapped away from; without this, a stale value
    # would ride straight into ``thread_start(model=...)`` and 500 the
    # new-session click.
    models_data, resolved_settings = _new_session_post_models_and_settings(request)
    default_settings = resolved_settings.values
    if request.POST.get("rendered_model") is not None and request.POST.get("rendered_reasoning_effort") is not None:
        default_settings = _rendered_settings_with_guest_effort_default(
            request,
            default_settings,
        )
    settings, model_settings_error = _posted_model_settings_override(
        request,
        default=default_settings,
        models_data=models_data,
        plan_mode=plan_mode,
    )
    if model_settings_error is not None:
        return HttpResponseBadRequest(model_settings_error)
    use_worktrees, use_worktrees_error = _posted_bool_override(
        request.POST.get("use_worktrees"),
        default=settings.use_worktrees,
        error="invalid worktree setting",
    )
    if use_worktrees_error is not None:
        return HttpResponseBadRequest(use_worktrees_error)
    cookie_updates = resolved_settings.cookie_updates
    source_project = target.project
    source_developer_instructions = common._developer_instructions_for_project(
        settings, None if target.project_cleared else source_project
    )
    default_auto_pr_enabled = _effective_auto_pr_enabled(
        None if target.project_cleared else source_project,
        global_enabled=settings.auto_pr_enabled,
    )
    auto_pr_enabled, auto_pr_error = _posted_bool_override(
        request.POST.get("auto_pr"),
        default=default_auto_pr_enabled,
        error="invalid auto-PR setting",
    )
    if auto_pr_error is not None:
        return HttpResponseBadRequest(auto_pr_error)
    auto_qa_enabled, auto_qa_error = _posted_bool_override(
        request.POST.get("auto_qa"),
        default=settings.auto_qa_enabled,
        error="invalid auto-QA setting",
    )
    if auto_qa_error is not None:
        return HttpResponseBadRequest(auto_qa_error)
    if auto_pr_enabled:
        auto_qa_enabled = False
    if proposed_session is not None and _proposal_has_explicit_auto_review_settings(
        proposed_session
    ):
        auto_pr_enabled, auto_qa_enabled = _auto_review_settings_for_proposed_session(proposed_session)
    web_search_mode, web_search_error = _posted_web_search_override(
        request.POST.get("web_search_mode"),
        default=settings.web_search_mode,
    )
    if web_search_error is not None:
        return HttpResponseBadRequest(web_search_error)
    if plan_mode and not settings.model:
        return HttpResponseBadRequest("plan mode requires a model")
    if agent_task_activation and has_input_images:
        return HttpResponseBadRequest(
            "image attachments are not supported for review or PR tasks"
        )
    if not agent_task_activation and not plan_mode:
        prompt = agent_tasks.with_automatic_review_guidance(
            prompt,
            auto_pr_enabled=auto_pr_enabled,
            auto_qa_enabled=auto_qa_enabled,
            pr_title=(proposed_session.title if proposed_session is not None else ""),
        )
    if (
        not agent_task_activation
        and source_project is not None
        and source_project.auto_pull_enabled
    ):
        try:
            repos_module.pull_default_branch_from_origin(source_project.repo_path)
        except repos_module.AutoPullError as exc:
            return HttpResponseBadRequest(f"could not update project before session: {exc}")

    session_cwd = cwd
    managed_worktree: ManagedWorktree | None = None
    approved_snapshot = _approved_snapshot_for_proposal(proposed_session)
    legacy_candidate = _legacy_ag_candidate_for_snapshot(proposed_session)
    if legacy_candidate is not None:
        try:
            approved_snapshot = snapshot_worktree_to_commit(
                legacy_candidate.cwd,
                message="Snapshot legacy autonomous-goal proposal",
            )
        except WorktreeCreationError as exc:
            return HttpResponseBadRequest(str(exc))
    if (
        proposed_session is not None
        and _is_tool_protocol_ag_proposal(proposed_session)
        and not _ag_proposal_requires_snapshot(proposed_session)
    ):
        approved_snapshot = ""
    if (
        proposed_session is not None
        and _is_tool_protocol_ag_proposal(proposed_session)
        and not approved_snapshot
        and _ag_proposal_requires_snapshot(proposed_session)
    ):
        return HttpResponseBadRequest(
            "approved autonomous-goal snapshot is missing or invalid"
        )
    if approved_snapshot:
        try:
            managed_worktree = common.create_worktree_for_session(
                cwd, base_ref=approved_snapshot
            )
        except WorktreeCreationError as exc:
            return HttpResponseBadRequest(str(exc))
        session_cwd = str(managed_worktree.path)
    if proposed_session is not None:
        prompt = _accepted_proposal_prompt(
            proposed_session,
            prompt,
            approved_snapshot=approved_snapshot,
        )
    sandbox_policy = _effective_sandbox_policy_for_cwd(
        settings,
        session_cwd,
        managed_worktree=managed_worktree is not None,
    )
    # Review tasks inspect the selected repo's current diff; a fresh managed
    # worktree would be clean and miss uncommitted changes.
    if agent_task_activation:
        if proposed_session is not None:
            thread_name = proposed_session.title
        else:
            thread_name = PR_SLASH_DISPLAY_PROMPT if pr_activation or pr_now_activation else _QA_SLASH_PROMPT
        create_thread_kwargs: dict[str, Any] = {
            "cwd": session_cwd,
            "name": thread_name,
            "developer_instructions": source_developer_instructions or None,
            "model": settings.model or None,
            "enable_memories": settings.enable_memories,
        }
        if web_search_mode:
            create_thread_kwargs["web_search_mode"] = web_search_mode
        proposal_claimed = False
        if proposed_session is not None:
            claim_response = _claim_new_session_proposal_start(
                proposed_session=proposed_session,
                cookie_updates=cookie_updates,
            )
            if claim_response is not None:
                _cleanup_worktree_quietly(managed_worktree)
                return claim_response
            proposal_claimed = True
        try:
            thread_id = codex_pool.create_session_thread(**create_thread_kwargs)
        except Exception:
            if proposal_claimed:
                assert proposed_session is not None
                _reset_new_session_proposal_start_claim(proposed_session)
            _cleanup_worktree_quietly(managed_worktree)
            raise
        # Only proposal acceptances carry forward auto-review, and only the
        # settings the proposal itself requested. A bare ``/qa`` or
        # ``/pr`` (no proposal) is a one-off review, and a coding-agent proposal
        # leaves these inputs empty, so in both cases the resolved
        # ``auto_*_enabled`` here are just the user's global/form defaults.
        # Persisting those would silently auto-review every later follow-up in
        # the session, so derive the stored flags from the proposal only.
        if proposed_session is not None:
            session_auto_pr_enabled, session_auto_qa_enabled = _auto_review_settings_for_proposed_session(
                proposed_session
            )
        else:
            session_auto_pr_enabled = False
            session_auto_qa_enabled = False
        task = (
            agent_tasks.publish_pr_task()
            if pr_now_activation
            else agent_tasks.review_task(
                prepare_pull_request=not qa_activation,
                pr_title=(
                    proposed_session.title
                    if proposed_session is not None and not qa_activation
                    else ""
                ),
            )
        )
        task_kwargs: dict[str, Any] = {
            "thread_id": thread_id,
            "cwd": session_cwd,
            "prompt": _accepted_proposal_prompt(
                proposed_session,
                task.prompt,
                approved_snapshot=approved_snapshot,
            )
            if proposed_session is not None
            else task.prompt,
            "sandbox_policy": sandbox_policy or None,
            "approval_mode": settings.approval_mode,
            "model": settings.model or None,
            "reasoning_effort": settings.reasoning_effort or None,
            "developer_instructions": source_developer_instructions or None,
            "enable_memories": settings.enable_memories,
            "user_message_index": 0,
            "agent_kind": task.agent_kind,
        }
        if web_search_mode:
            task_kwargs["web_search_mode"] = web_search_mode
        try:
            codex_pool.spawn_turn(**task_kwargs)
        except Exception:
            if proposal_claimed:
                assert proposed_session is not None
                _reset_new_session_proposal_start_claim(proposed_session)
            _cleanup_worktree_quietly(managed_worktree)
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
        )
        _finish_new_session_proposal_start_claim(
            proposed_session,
            session_metadata,
            approved_snapshot=approved_snapshot,
        )
        return _remember_repo_and_redirect(request, cookie_updates, cwd=cwd, thread_id=thread_id)

    if use_worktrees and managed_worktree is None:
        try:
            managed_worktree = common.create_worktree_for_session(cwd)
        except WorktreeCreationError as exc:
            return HttpResponseBadRequest(str(exc))
        session_cwd = str(managed_worktree.path)
        sandbox_policy = _effective_sandbox_policy_for_cwd(
            settings,
            session_cwd,
            managed_worktree=True,
        )
    session_project = (
        None if target.project_cleared else common._project_for_cwd(session_cwd, projects) or source_project
    )
    developer_instructions = common._developer_instructions_for_project(settings, session_project)
    input_image_paths, input_image_error = common._save_posted_input_images(request)
    if input_image_error is not None:
        _cleanup_worktree_quietly(managed_worktree)
        return HttpResponseBadRequest(input_image_error)

    # Detach a worker subprocess so the initial turn keeps running past a
    # Django restart. The thread itself is created synchronously to give the
    # caller a stable id to redirect to.
    spawn_kwargs: dict[str, Any] = {
        "cwd": session_cwd,
        "prompt": prompt,
        "developer_instructions": developer_instructions or None,
        "model": settings.model or None,
        "reasoning_effort": None if plan_mode else settings.reasoning_effort or None,
        "sandbox_policy": sandbox_policy or None,
        "approval_mode": settings.approval_mode,
    }
    if input_image_paths:
        spawn_kwargs["input_image_paths"] = input_image_paths
    if web_search_mode:
        spawn_kwargs["web_search_mode"] = web_search_mode
    if proposed_session is not None:
        spawn_kwargs["thread_name"] = proposed_session.title
    if settings.enable_memories:
        spawn_kwargs["enable_memories"] = True
    if plan_mode:
        spawn_kwargs["plan_mode"] = True
    if auto_pr_enabled and not plan_mode:
        spawn_kwargs["agent_kind"] = agent_tasks.PR_PUBLISH_AGENT_KIND
        spawn_kwargs["user_message_index"] = 0
    input_images_owned = False
    proposal_claimed = False
    if proposed_session is not None:
        claim_response = _claim_new_session_proposal_start(
            proposed_session=proposed_session,
            cookie_updates=cookie_updates,
        )
        if claim_response is not None:
            common._cleanup_saved_input_images(input_image_paths)
            _cleanup_worktree_quietly(managed_worktree)
            return claim_response
        proposal_claimed = True
    try:
        instance = codex_pool.spawn_new_session(**spawn_kwargs)
        input_images_owned = True
    except Exception:
        if not input_images_owned:
            common._cleanup_saved_input_images(input_image_paths)
        if proposal_claimed:
            assert proposed_session is not None
            _reset_new_session_proposal_start_claim(proposed_session)
        _cleanup_worktree_quietly(managed_worktree)
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
        codex_path=codex_pool.thread_path_for_instance(instance),
    )
    _finish_new_session_proposal_start_claim(
        proposed_session,
        session_metadata,
        approved_snapshot=approved_snapshot,
    )
    return _remember_repo_and_redirect(request, cookie_updates, cwd=cwd, thread_id=instance.thread_id)


@_limit_input_image_uploads
@require_http_methods(["GET", "POST"])
def new_session(request: HttpRequest) -> HttpResponse:
    if request.method == "GET":
        return _render_new_session_page(request)
    return _post_new_session(request)
