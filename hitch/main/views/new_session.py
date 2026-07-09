"""The new-session page and start flow, including proposal acceptance."""
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

from hitch.main import caches, coding_agents
from hitch.main import repos as repos_module
from hitch.main.goals.autonomous_goal_proposal_stack import _proposal_outcome_metadata
from hitch.main.goals.autonomous_goal_run_display import (
    _attach_proposed_session_display_state,
    _auto_review_settings_for_proposed_session,
    _proposal_metadata,
    _proposed_session_prompt,
)
from hitch.main.models import (
    CodexInstance,
    Project,
    ProposedSession,
    SessionMetadata,
)
from hitch.main.runtime import app_server_pool, codex_pool, reconciliation
from hitch.main.runtime.input_images import (
    _limit_input_image_uploads,
)
from hitch.main.sessions import session_index
from hitch.main.sessions.message_intent import (
    _is_fix_pr_activation,
    _is_pr_activation,
    _is_qa_activation,
    _message_intent,
)
from hitch.main.sessions.project_visibility import (
    _metadata_by_thread_id as _metadata_by_thread_id,
)
from hitch.main.sessions.session_entry_display import (
    _entries_for,
)
from hitch.main.sessions.session_pr_plan import (
    _PR_SLASH_PROMPT,
    _count_user_entries,
)
from hitch.main.sessions.session_settings import (
    _BARE_REPO_PROJECT_VALUE,
    _QA_SLASH_PROMPT,
    _authenticated_user,
    _cached_models_and_settings,
    _effective_approval_mode_for_session,
    _effective_auto_pr_enabled,
    _effective_sandbox_policy_for_cwd,
    _new_session_form_context,
    _project_for_proposed_session,
    _resolved_settings,
    _save_user_settings,
    _selected_project_for_settings,
    _stored_settings,
)
from hitch.main.sessions.settings_cookies import (
    _LAST_SELECTED_REPO_COOKIE,
    _VALID_WEB_SEARCH_MODES,
    ResolvedSettings,
    SettingsValues,
    _apply_cookie_updates,
    _settings_cookie_updates,
)
from hitch.main.views import common
from hitch.main.workflows import pr_qa, spec_critic
from hitch.main.worktrees import (
    ManagedWorktree,
    WorktreeCleanupError,
    WorktreeCreationError,
)


class _NewSessionTarget(NamedTuple):
    cwd: str
    project: Project | None
    project_cleared: bool
    requires_discovered_repo: bool

def _new_session_post_settings(request: HttpRequest) -> ResolvedSettings:
    stored_settings = _stored_settings(request)
    enable_memories = stored_settings.enable_memories
    if caches._models_cache_has_value(
        enable_memories=enable_memories
    ) and not caches._models_refresh_needed(enable_memories=enable_memories):
        models_data = caches._cached_models_data(enable_memories=enable_memories)
        if models_data:
            return _resolved_settings(request, models_data)

    models_data = caches._fetch_models_data(
        enable_memories=enable_memories, codex_cls=common.Codex
    )
    return _resolved_settings(request, models_data)

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
    proposed_session: ProposedSession | None, session_metadata: SessionMetadata
) -> None:
    if proposed_session is None:
        return
    claim_filter = common._new_session_proposal_start_claim_filter(proposed_session)
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
    common._stop_autonomous_goal_stack_after_proposal_resolution(proposed_session)

def _posted_bool_override(
    raw: str | None, *, default: bool, error: str
) -> tuple[bool, str | None]:
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

def _candidate_thread_user_message_index(
    thread_id: str, settings: SettingsValues
) -> int:
    resumed = app_server_pool.run_borrowed_op_with_retry(
        common.Codex,
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
    common._rename_codex_thread_from_proposal(
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
    return _remember_repo_and_redirect(
        request, settings, cookie_updates, cwd=cwd, thread_id=candidate_session.thread_id
    )

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
    if not common._is_allowed_session_cwd(candidate_cwd):
        return HttpResponseBadRequest(
            "candidate session cwd is not an allowed repository"
        )
    prompt = _candidate_proposal_continuation_prompt(prompt)
    base_instructions = common._base_instructions_for_settings(spawn_settings)
    project = None if target.project_cleared else candidate_session.project or target.project
    developer_instructions = common._developer_instructions_for_project(settings, project)
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
            pr_qa.start_pr_qa_workflow(**workflow_kwargs)
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
        common._stop_autonomous_goal_stack_after_proposal_resolution(proposed_session)
        return response

    input_image_paths, input_image_error = common._save_posted_input_images(request)
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
        common._cleanup_saved_input_images(input_image_paths)
        return claim_response
    try:
        codex_pool.spawn_turn(**spawn_kwargs)
        input_images_owned = True
    except codex_pool.InputAttachmentLimitExceededError as exc:
        common._cleanup_saved_input_images(input_image_paths)
        _reset_candidate_proposal_start_claim(proposed_session, candidate_session)
        return HttpResponseBadRequest(str(exc))
    except Exception:
        if not input_images_owned:
            common._cleanup_saved_input_images(input_image_paths)
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
    common._stop_autonomous_goal_stack_after_proposal_resolution(proposed_session)
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
    reconciliation.reconcile_dead_if_due()
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
    settings_context = common._settings_context(current_settings, models_data)
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

def _cleanup_worktree_quietly(managed_worktree: ManagedWorktree | None) -> None:
    if managed_worktree is None:
        return
    try:
        common.cleanup_worktree(managed_worktree)
    except WorktreeCleanupError:
        common.logger.exception(
            "failed to clean up managed worktree %s", managed_worktree.path
        )

def _remember_repo_and_redirect(
    request: HttpRequest,
    settings: SettingsValues,
    cookie_updates: dict[str, str],
    *,
    cwd: str,
    thread_id: str,
) -> HttpResponse:
    """Persist the chosen repo as the last-selected one and redirect to the
    new session. Authenticated users get it saved on their settings row;
    anonymous users get the signed cookie."""
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

def _post_new_session(request: HttpRequest) -> HttpResponse:
    intent = _message_intent(request)
    pr_activation = _is_pr_activation(request)
    fix_pr_activation = _is_fix_pr_activation(request)
    qa_activation = _is_qa_activation(request)
    qa_workflow_activation = pr_activation or qa_activation or fix_pr_activation
    prompt = intent.prompt
    plan_mode = False if qa_workflow_activation else intent.plan_mode
    has_input_images = common._has_input_image_uploads(request)
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
        base_instructions = common._base_instructions_for_settings(spawn_settings)
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
            pr_qa.start_pr_qa_workflow(**workflow_kwargs)
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
        return _remember_repo_and_redirect(
            request, settings, cookie_updates, cwd=cwd, thread_id=thread_id
        )

    managed_worktree = None
    if use_worktrees:
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
        None
        if target.project_cleared
        else common._project_for_cwd(session_cwd, projects) or source_project
    )
    developer_instructions = common._developer_instructions_for_project(
        settings, session_project
    )
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
    base_instructions = common._base_instructions_for_settings(spawn_settings)
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
            _cleanup_worktree_quietly(managed_worktree)
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
            spec_critic.start_spec_critic_workflow(**spec_workflow_kwargs)
        except Exception:
            # The worktree is only referenced by the not-yet-started workflow, so
            # reclaim it before bubbling up rather than leaking it on disk (and
            # into the cwd allowlist) on every failed-then-retried new session.
            _cleanup_worktree_quietly(managed_worktree)
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
        return _remember_repo_and_redirect(
            request, settings, cookie_updates, cwd=cwd, thread_id=thread_id
        )
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
        auto_merge_to_local_branch=auto_merge_to_local_branch,
        auto_merge_branch=auto_merge_branch,
        codex_path=codex_pool.thread_path_for_instance(instance),
    )
    _finish_new_session_proposal_start_claim(proposed_session, session_metadata)
    return _remember_repo_and_redirect(
        request, settings, cookie_updates, cwd=cwd, thread_id=instance.thread_id
    )

@_limit_input_image_uploads
@require_http_methods(["GET", "POST"])
def new_session(request: HttpRequest) -> HttpResponse:
    if request.method == "GET":
        return _render_new_session_page(request)
    return _post_new_session(request)
