"""The new-session page and start flow, including proposal acceptance."""

from collections.abc import Iterator
from contextlib import contextmanager
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
from hitch.main.goals.autonomous_goal_proposal_stack import _proposal_outcome_metadata
from hitch.main.goals.autonomous_goal_run_display import (
    _attach_proposed_session_display_state,
    _auto_review_settings_for_proposed_session,
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
from hitch.main.sessions import agent_tasks, session_index
from hitch.main.sessions import lifecycle as session_lifecycle
from hitch.main.sessions.message_intent import (
    _message_intent,
)
from hitch.main.sessions.pr_prompts import PR_SLASH_DISPLAY_PROMPT
from hitch.main.sessions.project_visibility import (
    _metadata_by_thread_id as _metadata_by_thread_id,
)
from hitch.main.sessions.session_entry_display import (
    _entries_for,
)
from hitch.main.sessions.session_pr_plan import (
    _count_user_entries,
)
from hitch.main.sessions.session_resume import (
    _metadata_rollout_path_indicates_archived,
    _record_session_unarchived,
    _restore_archived_session_for_rejected_turn,
    _session_detail_metadata,
    _unarchive_session_for_turn,
    thread_has_dynamic_tool,
)
from hitch.main.sessions.session_settings import (
    _BARE_REPO_PROJECT_VALUE,
    _PLAN_MODE_REASONING_EFFORT,
    _QA_SLASH_PROMPT,
    _authenticated_user,
    _cached_models_and_settings,
    _effective_approval_mode_for_session,
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
)


class _NewSessionTarget(NamedTuple):
    cwd: str
    project: Project | None
    project_cleared: bool
    requires_discovered_repo: bool


_RESUME_SOURCE_SESSION_METADATA_KEY = "resume_source_session"
_RECOVERY_SOURCE_BUSY_MESSAGE = "source session is already running work"
_RECOVERY_TARGET_UNAVAILABLE_MESSAGE = "recovery repository is unavailable"


class _RecoverySourceBusyError(RuntimeError):
    pass


def _proposal_resumes_source_session(proposed_session: ProposedSession) -> bool:
    metadata = proposed_session.outcome_metadata
    return (
        isinstance(metadata, dict)
        and metadata.get(_RESUME_SOURCE_SESSION_METADATA_KEY) is True
        and proposed_session.source_session_id is not None
    )


def _proposal_has_trusted_source_target(
    proposed_session: ProposedSession | None, cwd: str
) -> bool:
    if (
        proposed_session is None
        or not _proposal_resumes_source_session(proposed_session)
        or not cwd
    ):
        return False
    trusted_cwds = {_target_cwd_for_proposed_session(proposed_session)}
    source_session = proposed_session.source_session
    if source_session is not None:
        trusted_cwds.add(source_session.cwd.strip())
    return cwd in trusted_cwds and _recovery_cwd_is_usable(cwd)


def _recovery_cwd_is_usable(cwd: str) -> bool:
    return bool(cwd and repos_module.repo_root(cwd) is not None)


def _recovery_proposal_has_usable_target(
    proposed_session: ProposedSession | None,
) -> bool:
    if (
        proposed_session is None
        or not _proposal_resumes_source_session(proposed_session)
    ):
        return True
    source_session = proposed_session.source_session
    if source_session is not None and _recovery_cwd_is_usable(
        source_session.cwd.strip()
    ):
        return True
    project = _project_for_proposed_session(proposed_session)
    return project is not None and _recovery_cwd_is_usable(project.repo_path)


@contextmanager
def _recovery_source_lifecycle_for_turn(
    proposed_session: ProposedSession,
    source_session: SessionMetadata,
    settings: SettingsValues,
) -> Iterator[bool]:
    """Own source-session archive state through a recovery turn start."""
    if not _proposal_resumes_source_session(proposed_session):
        yield False
        return
    with session_lifecycle.hold(source_session.thread_id):
        if session_lifecycle.archive_has_active_work(source_session.thread_id):
            raise _RecoverySourceBusyError(_RECOVERY_SOURCE_BUSY_MESSAGE)
        unarchived = False
        try:
            archive_metadata = _session_detail_metadata(source_session.thread_id) or source_session
            if archive_metadata.codex_archived or _metadata_rollout_path_indicates_archived(archive_metadata):
                _unarchive_session_for_turn(source_session.thread_id, settings)
                unarchived = True
                _record_session_unarchived(source_session.thread_id)
                source_session.codex_archived = False
                source_session.codex_archived_at = None
                source_session.codex_path = ""
            yield True
        except Exception:
            if unarchived:
                _restore_archived_session_for_rejected_turn(
                    source_session.thread_id,
                    settings,
                )
            raise


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


def _candidate_session_to_continue_from_proposal(
    proposed_session: ProposedSession | None,
) -> SessionMetadata | None:
    if proposed_session is None:
        return None
    if _proposal_resumes_source_session(proposed_session):
        source_session = proposed_session.source_session
        if source_session is None or not _recovery_cwd_is_usable(
            source_session.cwd.strip()
        ):
            return None
        return source_session
    if proposed_session.candidate_session is None:
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


def _candidate_thread_user_message_index(thread_id: str, settings: SettingsValues) -> int:
    resumed = app_server_pool.run_borrowed_op_with_retry(
        common.Codex,
        lambda codex: codex._client.thread_resume(thread_id),
        enable_memories=settings.enable_memories,
    )
    return _count_user_entries(list(_entries_for(resumed.thread)))


def _next_user_message_index_for_candidate_thread(thread_id: str, settings: SettingsValues) -> int:
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
    session_project = (
        None
        if target.project_cleared
        else candidate_session.project or target.project
    )
    if not _proposal_resumes_source_session(proposed_session):
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
        is_hidden_system_session=False,
    )
    candidate_session.refresh_from_db()
    return _remember_repo_and_redirect(
        request, cookie_updates, cwd=cwd, thread_id=candidate_session.thread_id
    )


def _start_candidate_proposal_session(
    *,
    request: HttpRequest,
    proposed_session: ProposedSession,
    candidate_session: SessionMetadata,
    prompt: str,
    plan_mode: bool,
    pr_now_activation: bool,
    qa_activation: bool,
    agent_task_activation: bool,
    cwd: str,
    target: _NewSessionTarget,
    settings: SettingsValues,
    cookie_updates: dict[str, str],
    auto_pr_enabled: bool,
    auto_qa_enabled: bool,
    web_search_mode: str,
) -> HttpResponse:
    """Start a proposal on its existing candidate thread before accepting it."""
    candidate_cwd = candidate_session.cwd
    if not candidate_cwd:
        return HttpResponseBadRequest("candidate session has no cwd")
    if (
        not common._is_allowed_session_cwd(candidate_cwd)
        and not _proposal_has_trusted_source_target(
            proposed_session,
            candidate_cwd,
        )
    ):
        return HttpResponseBadRequest(
            "candidate session cwd is not an allowed repository"
        )
    if not _proposal_resumes_source_session(proposed_session):
        prompt = _candidate_proposal_continuation_prompt(prompt)
    project = None if target.project_cleared else candidate_session.project or target.project
    developer_instructions = common._developer_instructions_for_project(settings, project)
    sandbox_policy = _effective_sandbox_policy_for_cwd(settings, candidate_cwd)
    approval_mode = _effective_approval_mode_for_session(
        settings,
        candidate_session.thread_id,
        candidate_session,
    )
    if agent_task_activation:
        claim_response = _claim_candidate_proposal_start(
            proposed_session=proposed_session,
            candidate_session=candidate_session,
            cookie_updates=cookie_updates,
        )
        if claim_response is not None:
            return claim_response
        try:
            with _recovery_source_lifecycle_for_turn(
                proposed_session,
                candidate_session,
                settings,
            ):
                task = (
                    agent_tasks.publish_pr_task()
                    if pr_now_activation
                    else agent_tasks.review_task(
                        prepare_pull_request=not qa_activation,
                        pr_title=(
                            proposed_session.title if not qa_activation else ""
                        ),
                    )
                )
                if task.requires_pr_watch and not thread_has_dynamic_tool(
                    candidate_session.thread_id,
                    namespace="hitch",
                    name="watch_pr",
                ):
                    raise agent_tasks.PrWatchUnavailableError(
                        "hitch.watch_pr is unavailable for this session; "
                        "start a new session before publishing a PR"
                    )
                task_kwargs: dict[str, Any] = {
                    "thread_id": candidate_session.thread_id,
                    "cwd": candidate_cwd,
                    "prompt": task.prompt,
                    "sandbox_policy": sandbox_policy or None,
                    "approval_mode": approval_mode,
                    "model": settings.model or None,
                    "reasoning_effort": settings.reasoning_effort or None,
                    "developer_instructions": developer_instructions or None,
                    "enable_memories": settings.enable_memories,
                    "user_message_index": (
                        _next_user_message_index_for_candidate_thread(
                            candidate_session.thread_id,
                            settings,
                        )
                    ),
                    "agent_kind": task.agent_kind,
                }
                if web_search_mode:
                    task_kwargs["web_search_mode"] = web_search_mode
                codex_pool.spawn_turn(**task_kwargs)
        except _RecoverySourceBusyError:
            _reset_candidate_proposal_start_claim(proposed_session, candidate_session)
            return HttpResponseBadRequest(_RECOVERY_SOURCE_BUSY_MESSAGE)
        except agent_tasks.PrWatchUnavailableError as exc:
            _reset_candidate_proposal_start_claim(proposed_session, candidate_session)
            return HttpResponseBadRequest(str(exc))
        except Exception:
            _reset_candidate_proposal_start_claim(proposed_session, candidate_session)
            raise
        # Persist the proposal-derived auto-review configuration so subsequent
        # turns in this session keep honoring it.
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
        with _recovery_source_lifecycle_for_turn(
            proposed_session,
            candidate_session,
            settings,
        ):
            spawn_kwargs: dict[str, Any] = {
                "thread_id": candidate_session.thread_id,
                "cwd": candidate_cwd,
                "prompt": prompt,
                "developer_instructions": developer_instructions or None,
                "model": settings.model or None,
                "reasoning_effort": (
                    None if plan_mode else settings.reasoning_effort or None
                ),
                "sandbox_policy": sandbox_policy or None,
                "approval_mode": approval_mode,
            }
            if input_image_paths:
                spawn_kwargs["input_image_paths"] = input_image_paths
            if web_search_mode:
                spawn_kwargs["web_search_mode"] = web_search_mode
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
                spawn_kwargs["stored_reasoning_effort"] = (
                    settings.reasoning_effort or None
                )
                spawn_kwargs["user_message_index"] = (
                    _next_user_message_index_for_candidate_thread(
                        candidate_session.thread_id,
                        settings,
                    )
                )
            codex_pool.spawn_turn(**spawn_kwargs)
            input_images_owned = True
    except _RecoverySourceBusyError:
        common._cleanup_saved_input_images(input_image_paths)
        _reset_candidate_proposal_start_claim(proposed_session, candidate_session)
        return HttpResponseBadRequest(_RECOVERY_SOURCE_BUSY_MESSAGE)
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
        or (target_cwd not in repo_set and not _proposal_has_trusted_source_target(proposed_session, target_cwd))
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
    if proposed_session_cwd and proposed_session_cwd not in repo_set:
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
    if proposed_session is not None and _proposal_resumes_source_session(proposed_session):
        recovery_auto_pr, recovery_auto_qa = _auto_review_settings_for_proposed_session(proposed_session)
        new_session_context["current_new_session_auto_pr"] = recovery_auto_pr
        new_session_context["current_new_session_auto_qa"] = recovery_auto_qa
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
    if not _recovery_proposal_has_usable_target(proposed_session):
        return HttpResponseBadRequest(_RECOVERY_TARGET_UNAVAILABLE_MESSAGE)
    cwd = target.cwd
    if not prompt and not has_input_images:
        return HttpResponseBadRequest("prompt is required")
    if not cwd:
        return HttpResponseBadRequest("cwd is required")
    # Raw cwd posts still need discovery validation. Project-id posts use the
    # server-side Project.repo_path, so they do not need a home-directory scan
    # on the hot Start path.
    trusted_resume_target = _proposal_has_trusted_source_target(
        proposed_session, target.cwd
    )
    if target.requires_discovered_repo and not trusted_resume_target:
        allowed = {str(p) for p in repos_module.discover_repos()}
        if cwd not in allowed:
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
    if proposed_session is not None and (
        proposed_session.autonomous_goal is not None or _proposal_resumes_source_session(proposed_session)
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
    if (
        not agent_task_activation
        and source_project is not None
        and source_project.auto_pull_enabled
        and not (proposed_session is not None and _proposal_resumes_source_session(proposed_session))
    ):
        try:
            repos_module.pull_default_branch_from_origin(source_project.repo_path)
        except repos_module.AutoPullError as exc:
            return HttpResponseBadRequest(f"could not update project before session: {exc}")

    candidate_session = _candidate_session_to_continue_from_proposal(proposed_session)
    if candidate_session is not None:
        assert proposed_session is not None
        return _start_candidate_proposal_session(
            request=request,
            proposed_session=proposed_session,
            candidate_session=candidate_session,
            prompt=prompt,
            plan_mode=plan_mode,
            pr_now_activation=pr_now_activation,
            qa_activation=qa_activation,
            agent_task_activation=agent_task_activation,
            cwd=cwd,
            target=target,
            settings=settings,
            cookie_updates=cookie_updates,
            auto_pr_enabled=auto_pr_enabled,
            auto_qa_enabled=auto_qa_enabled,
            web_search_mode=web_search_mode,
        )

    session_cwd = cwd
    sandbox_policy = _effective_sandbox_policy_for_cwd(settings, session_cwd)
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
                return claim_response
            proposal_claimed = True
        try:
            thread_id = codex_pool.create_session_thread(**create_thread_kwargs)
        except Exception:
            if proposal_claimed:
                assert proposed_session is not None
                _reset_new_session_proposal_start_claim(proposed_session)
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
            "prompt": task.prompt,
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
        _finish_new_session_proposal_start_claim(proposed_session, session_metadata)
        return _remember_repo_and_redirect(request, cookie_updates, cwd=cwd, thread_id=thread_id)

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
    if auto_pr_enabled:
        spawn_kwargs["auto_pr_enabled"] = True
    if auto_qa_enabled:
        spawn_kwargs["auto_qa_enabled"] = True
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
    _finish_new_session_proposal_start_claim(proposed_session, session_metadata)
    return _remember_repo_and_redirect(request, cookie_updates, cwd=cwd, thread_id=instance.thread_id)


@_limit_input_image_uploads
@require_http_methods(["GET", "POST"])
def new_session(request: HttpRequest) -> HttpResponse:
    if request.method == "GET":
        return _render_new_session_page(request)
    return _post_new_session(request)
