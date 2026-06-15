"""send_message: steering, workflow activation, and turn spawning."""
import logging
import os
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from django.http import (
    HttpRequest,
    HttpResponse,
    HttpResponseBadRequest,
)
from django.shortcuts import redirect
from django.utils import timezone
from django.views.decorators.http import require_http_methods
from openai_codex.errors import InvalidRequestError

from hitch.main import caches, claude_options
from hitch.main.models import (
    CodexInstance,
    Project,
    SessionMetadata,
)
from hitch.main.runtime import app_server_pool, codex_pool, reconciliation
from hitch.main.runtime.db import run_ignoring_database_locks
from hitch.main.runtime.input_images import (
    _limit_input_image_uploads,
)
from hitch.main.runtime.sdk_values import (
    string_value,
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
from hitch.main.sessions.session_approval import _parse_instance_id
from hitch.main.sessions.session_entry_display import (
    _entries_for,
    _workflow_accepts_active_turn_steering,
    _workflow_accepts_qa_pause_steering,
)
from hitch.main.sessions.session_pr_plan import (
    _auto_merge_to_local_branch_for_session,
    _auto_pr_enabled_for_session,
    _auto_qa_enabled_for_session,
    _count_user_entries,
    _fix_pr_url_for_thread,
    _thread_plan_mode_state,
)
from hitch.main.sessions.session_resume import (
    _metadata_indicates_archived,
    _metadata_resume_for_inactive_session,
    _metadata_rollout_path_indicates_archived,
    _record_session_unarchived,
    _restore_archived_session_for_rejected_turn,
    _session_detail_metadata,
    _session_is_claude,
    _thread_resume_archived_error,
    _unarchive_session_for_turn,
)
from hitch.main.sessions.session_settings import (
    _effective_approval_mode_for_session,
    _effective_sandbox_policy,
    _effective_sandbox_policy_for_cwd,
    _model_for_thread_backend,
)
from hitch.main.sessions.settings_cookies import (
    SettingsValues,
    _valid_web_search_mode_or_default,
)
from hitch.main.views import common
from hitch.main.workflows import pr_qa, spec_critic, system_agents

logger = logging.getLogger(__name__)


def _views() -> Any:
    """Return the ``hitch.main.views`` package object.

    The Claude follow-up/workflow helpers below resolve a handful of their
    collaborators through this package so that ``patch.object(views, "...")`` --
    the way the former monolith's tests intercept these dependencies -- keeps
    working now that the helpers live in this split module. Imported lazily to
    avoid the import cycle (``views/__init__`` imports this module).
    """
    from hitch.main import views

    return views


_PLAN_ACTION_APPROVE = "approve"

_PLAN_ACTION_REVISE = "revise"

_VALID_PLAN_ACTIONS = frozenset({"", _PLAN_ACTION_APPROVE, _PLAN_ACTION_REVISE})

_DEFAULT_COLLABORATION_MODE = "default"

def _metadata_cwd_is_disallowed(metadata: SessionMetadata | None) -> bool:
    return (
        metadata is not None
        and bool(metadata.cwd)
        and not common._is_allowed_session_cwd(metadata.cwd)
    )

class _TurnRejectedError(Exception):
    """Reject a send_message turn with this response.

    Raised by the spawn pipeline inside send_message's try block so every
    rejection path funnels through one cleanup site (restore an unarchived
    session, delete saved input images) instead of each branch repeating it.
    """

    def __init__(self, response: HttpResponse) -> None:
        super().__init__()
        self.response = response

def _codex_followup_model(resumed: Any, settings: SettingsValues) -> str | None:
    """Resumed Codex thread's model, falling back to the settings model.

    The settings (cookie) model can hold a ``claude-*`` id when the user
    switches the global provider to Claude while a Codex session stays open.
    This path only runs for Codex-backed sessions, so the Codex normalization in
    ``_model_for_thread_backend`` drops a Claude id rather than queue a Codex
    worker/workflow with a model the app-server would reject. The thread's own
    model keeps priority over the cookie.
    """
    resumed_model = string_value(getattr(resumed, "model", None))
    return _model_for_thread_backend(
        backend=CodexInstance.BACKEND_CODEX,
        model=resumed_model or settings.model,
        codex_fallback_model=resumed_model or None,
    )


def _stored_model_and_effort(resumed: Any, settings: SettingsValues) -> tuple[str, str]:
    """Thread's recorded model/effort, falling back to the request settings.

    The model is normalized to the Codex backend so a ``claude-*`` cookie id
    (provider switched to Claude with a Codex session open) is dropped rather
    than queued onto a Codex worker the app-server would reject.
    """
    model = _codex_followup_model(resumed, settings) or ""
    effort = (
        string_value(getattr(resumed, "reasoning_effort", None))
        or settings.reasoning_effort
    )
    return model, effort

def _start_claude_qa_workflow(
    *,
    session_id: str,
    qa_activation: bool,
    settings: SettingsValues,
    input_image_paths: list[str],
) -> HttpResponse:
    """Start a PR/QA workflow on an existing Claude session (manual /qa or /pr).

    The Claude analog of the Codex follow-up activation: the workflow records the
    thread's (Claude) backend and spawns its sub-agents and the PR-prompt turn as
    Claude workers; the PR itself is opened by hitch via ``gh``. cwd and per-turn
    settings come from local rows since the thread has no Codex rollout to resume.
    """
    # ``/qa`` and ``/pr`` carry no image attachments (rejected earlier), so the
    # saved temp copies are not needed.
    common._cleanup_saved_input_images(input_image_paths)
    common_result = _views()._claude_workflow_common(session_id, settings)
    if isinstance(common_result, HttpResponse):
        return common_result
    cwd, model, developer_instructions = common_result
    auto_merge_to_local_branch, auto_merge_branch = (
        _views()._auto_merge_to_local_branch_for_session(session_id)
    )
    workflow_kwargs: dict[str, Any] = {
        "main_thread_id": session_id,
        "cwd": cwd,
        "sandbox_policy": _effective_sandbox_policy(settings) or None,
        "approval_mode": _effective_approval_mode_for_session(settings, session_id),
        "model": model,
        "reasoning_effort": settings.reasoning_effort or None,
        "developer_instructions": developer_instructions or None,
        "enable_memories": settings.enable_memories,
        "initial_user_message_index": _views()._claude_user_message_index(session_id),
    }
    web_search_mode = _valid_web_search_mode_or_default(settings.web_search_mode)
    if web_search_mode:
        workflow_kwargs["web_search_mode"] = web_search_mode
    # No base instructions: this is a Claude workflow (the thread's backend, not
    # the current global provider, decides). Claude ships its own system prompt,
    # so Hitch's Codex/HITCH base-instruction variants must never reach a Claude
    # QA/PR agent -- even when the global provider was switched back to Codex.
    if qa_activation:
        workflow_kwargs["open_pr_on_lgtm"] = False
    if auto_merge_to_local_branch and auto_merge_branch:
        workflow_kwargs["auto_merge_branch"] = auto_merge_branch
    system_agents.start_pr_qa_workflow(**workflow_kwargs)
    return redirect("session", session_id=session_id)


def _start_claude_fix_pr_workflow(
    *,
    session_id: str,
    settings: SettingsValues,
    input_image_paths: list[str],
) -> HttpResponse:
    """Start PR-follow-up monitoring for an existing Claude session (``/fix-pr``).

    The Claude analog of the Codex ``fix_pr`` route: it targets the session's
    already-open PR via ``start_pr_monitor_workflow`` (which skips the QA step
    and never opens a second PR) rather than the generic PR/QA activation.
    """
    # ``/fix-pr`` carries no image attachments (rejected earlier), so drop the
    # saved temp copies.
    common._cleanup_saved_input_images(input_image_paths)
    pr_url = _views()._claude_fix_pr_url(session_id)
    if not pr_url:
        return HttpResponseBadRequest("fix-pr requires an opened PR for this session")
    common_result = _views()._claude_workflow_common(session_id, settings)
    if isinstance(common_result, HttpResponse):
        return common_result
    cwd, model, developer_instructions = common_result
    workflow_kwargs: dict[str, Any] = {
        "main_thread_id": session_id,
        "cwd": cwd,
        "pr_url": pr_url,
        "sandbox_policy": _effective_sandbox_policy(settings) or None,
        "approval_mode": _effective_approval_mode_for_session(settings, session_id),
        "model": model,
        "reasoning_effort": settings.reasoning_effort or None,
        "developer_instructions": developer_instructions or None,
        "enable_memories": settings.enable_memories,
        "initial_user_message_index": _views()._claude_user_message_index(session_id),
    }
    web_search_mode = _valid_web_search_mode_or_default(settings.web_search_mode)
    if web_search_mode:
        workflow_kwargs["web_search_mode"] = web_search_mode
    # No base instructions: a Claude workflow ships its own system prompt, so
    # Hitch's Codex base-instruction variants must not reach the Claude monitor
    # agent even if the global provider was switched back to Codex.
    system_agents.start_pr_monitor_workflow(**workflow_kwargs)
    return redirect("session", session_id=session_id)


def _start_claude_spec_critic_follow_up(
    *,
    session_id: str,
    prompt: str,
    settings: SettingsValues,
    input_image_paths: list[str],
) -> HttpResponse:
    """Run the Spec Critic preflight on an existing Claude session follow-up.

    Mirrors the Codex follow-up preflight on the local Claude thread: the hidden
    analysis/synthesizer agents run as Claude workers, then the implementation
    turn spawns on the same thread carrying the session's Auto-PR/Auto-QA config.
    """
    common_result = _views()._claude_workflow_common(session_id, settings)
    if isinstance(common_result, HttpResponse):
        common._cleanup_saved_input_images(input_image_paths)
        return common_result
    cwd, model, developer_instructions = common_result
    auto_pr_enabled = _views()._auto_pr_enabled_for_session(session_id)
    auto_qa_enabled = (
        False if auto_pr_enabled else _views()._auto_qa_enabled_for_session(session_id)
    )
    auto_merge_to_local_branch, auto_merge_branch = (
        _views()._auto_merge_to_local_branch_for_session(session_id)
        if auto_qa_enabled
        else (False, "")
    )
    spec_workflow_kwargs: dict[str, Any] = {
        "main_thread_id": session_id,
        "cwd": cwd,
        "prompt": prompt,
        "sandbox_policy": _effective_sandbox_policy(settings) or None,
        "approval_mode": _effective_approval_mode_for_session(settings, session_id),
        "model": model,
        "reasoning_effort": settings.reasoning_effort or None,
        "developer_instructions": developer_instructions or None,
        "enable_memories": settings.enable_memories,
        "initial_user_message_index": _views()._claude_user_message_index(session_id),
        "auto_pr_enabled": auto_pr_enabled,
        "auto_qa_enabled": auto_qa_enabled,
    }
    web_search_mode = _valid_web_search_mode_or_default(settings.web_search_mode)
    if web_search_mode:
        spec_workflow_kwargs["web_search_mode"] = web_search_mode
    # No base instructions: a Claude workflow ships its own system prompt, so
    # Hitch's Codex base-instruction variants must not reach the Claude Spec
    # Critic agents even if the global provider was switched back to Codex.
    if auto_merge_to_local_branch and auto_merge_branch:
        spec_workflow_kwargs["auto_merge_to_local_branch"] = True
        spec_workflow_kwargs["auto_merge_branch"] = auto_merge_branch
    common._cleanup_saved_input_images(input_image_paths)
    system_agents.start_spec_critic_workflow(**spec_workflow_kwargs)
    return redirect("session", session_id=session_id)


def _send_claude_follow_up(
    *,
    session_id: str,
    prompt: str,
    plan_mode: bool,
    settings: SettingsValues,
    input_image_paths: list[str],
) -> HttpResponse:
    """Run a follow-up turn on a Claude session without a Codex resume.

    Claude threads are not known to the Codex app-server, so the normal
    follow-up path's ``thread_resume`` would fail. cwd and per-turn settings are
    taken from local rows instead; ``spawn_turn`` inherits the backend and the
    stored Claude session id from the thread's history.
    """
    previous_instance = codex_pool.latest_for_thread(session_id)
    cwd = previous_instance.cwd if previous_instance is not None else ""
    if not cwd:
        metadata = SessionMetadata.objects.filter(thread_id=session_id).first()
        cwd = metadata.cwd if metadata is not None else ""
    if not cwd:
        common._cleanup_saved_input_images(input_image_paths)
        return HttpResponseBadRequest("session has no cwd")
    if cwd not in _views()._allowed_session_cwds():
        common._cleanup_saved_input_images(input_image_paths)
        return HttpResponseBadRequest("session cwd is not an allowed repository")
    model = settings.model
    if model not in claude_options.VALID_CLAUDE_MODELS:
        # The settings cookie may hold a Codex model id (provider switched back).
        # Prefer the session's own prior Claude model so a follow-up keeps the
        # same model instead of silently jumping to the default.
        prior_model = previous_instance.model if previous_instance is not None else ""
        model = (
            prior_model
            if prior_model in claude_options.VALID_CLAUDE_MODELS
            else claude_options.DEFAULT_CLAUDE_MODEL
        )
    web_search_mode = _valid_web_search_mode_or_default(settings.web_search_mode)
    developer_instructions = (
        previous_instance.developer_instructions
        if previous_instance is not None
        else common._developer_instructions_for_project(
            settings, common._project_for_cwd(cwd, list(Project.objects.all()))
        )
    )
    spawn_kwargs: dict[str, Any] = {
        "thread_id": session_id,
        "cwd": cwd,
        "prompt": prompt,
        "model": model,
        "stored_model": model,
        "reasoning_effort": settings.reasoning_effort or None,
        "sandbox_policy": _effective_sandbox_policy(settings) or None,
        # Honor a per-session approval override (set from the session header), as
        # the Codex follow-up path does -- otherwise a Claude thread pinned to
        # deny_all/approve_all in the session UI would spawn follow-ups under the
        # global default instead.
        "approval_mode": _effective_approval_mode_for_session(settings, session_id),
        "plan_mode": plan_mode,
    }
    if input_image_paths:
        spawn_kwargs["input_image_paths"] = input_image_paths
    if web_search_mode:
        spawn_kwargs["web_search_mode"] = web_search_mode
    if developer_instructions:
        # Set on every turn, not just the first: developer guidance now rides in
        # the per-turn system prompt (not the user prompt), so each follow-up
        # worker must carry it. It is read back from the previous instance above,
        # so it propagates forward across the session.
        spawn_kwargs["developer_instructions"] = developer_instructions
    # Carry the session's Auto-PR/Auto-QA configuration onto every follow-up
    # turn. ``on_codex_instance_finished`` fires off the completed instance's
    # ``auto_pr_enabled``/``auto_qa_enabled`` flags, so without this a Claude
    # session would only auto-review/open-a-PR after its initial turn and skip it
    # on every follow-up. Auto-PR supersedes Auto-QA, and plan turns never
    # auto-review.
    auto_pr_enabled = not plan_mode and _views()._auto_pr_enabled_for_session(session_id)
    auto_qa_enabled = (
        not plan_mode
        and not auto_pr_enabled
        and _views()._auto_qa_enabled_for_session(session_id)
    )
    if auto_pr_enabled or auto_qa_enabled:
        spawn_kwargs["user_message_index"] = _views()._claude_user_message_index(
            session_id
        )
    if auto_pr_enabled:
        spawn_kwargs["auto_pr_enabled"] = True
    elif auto_qa_enabled:
        spawn_kwargs["auto_qa_enabled"] = True
        auto_merge_to_local_branch, auto_merge_branch = (
            _views()._auto_merge_to_local_branch_for_session(session_id)
        )
        if auto_merge_to_local_branch:
            spawn_kwargs["auto_merge_to_local_branch"] = True
            spawn_kwargs["auto_merge_branch"] = auto_merge_branch
    try:
        codex_pool.spawn_turn(**spawn_kwargs)
    except Exception:
        common._cleanup_saved_input_images(input_image_paths)
        raise
    # Claude sessions have no app-server sync to refresh the index timestamp, so
    # bump it here or a multi-turn session stays sorted at its creation time.
    # Best-effort: the worker records turn activity again on completion, so a
    # transient SQLite lock must not 500 an already-launched turn -- that would
    # re-run the cleanup that deletes the worker's input images and invite a
    # duplicate retry.
    now = timezone.now()
    run_ignoring_database_locks(
        lambda: SessionMetadata.objects.filter(thread_id=session_id).update(
            codex_updated_at=now, codex_last_synced_at=now
        ),
        description="claude follow-up recency bump",
    )
    return redirect("session", session_id=session_id)


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
    has_input_images = common._has_input_image_uploads(request)
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
    run_ignoring_database_locks(
        lambda: reconciliation.reconcile_dead_for_thread(session_id),
        description="send-message dead-worker reconcile",
    )
    active_system_workflow = system_agents.active_workflow_for_thread(session_id)
    if qa_workflow_activation and has_input_images:
        return HttpResponseBadRequest(
            "image attachments are not supported for PR workflow requests"
        )
    settings = _views()._stored_settings(request)
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

    input_image_paths, input_image_error = common._save_posted_input_images(request)
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
                common._cleanup_saved_input_images(input_image_paths)
                steer_image_paths = []
                input_images_owned = True
                return redirect("session", session_id=session_id)
            common._cleanup_saved_input_images(steer_image_paths)
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
                common._cleanup_saved_input_images(input_image_paths)
                steer_image_paths = []
                input_images_owned = True
                return redirect("session", session_id=session_id)
            common._cleanup_saved_input_images(steer_image_paths)
            steer_image_paths = []
        if active_system_workflow is not None:
            if (
                active_instance is None
                and not raw_active
                and _workflow_accepts_qa_pause_steering(active_system_workflow)
            ):
                started = pr_qa.start_user_steering_turn(
                    active_system_workflow,
                    prompt=prompt,
                )
                if started is not None:
                    return redirect("session", session_id=session_id)
                raise _TurnRejectedError(
                    HttpResponseBadRequest("QA workflow could not be paused")
                )
            raise _TurnRejectedError(
                HttpResponseBadRequest("PR workflow is running for this session")
            )
        # Claude threads have no Codex rollout to resume, so route their
        # follow-up turns around the app-server entirely.
        if _session_is_claude(session_id):
            # An archived Claude session has no app-server thread to unarchive, so
            # clear the local archived state directly before routing -- otherwise
            # the accepted turn/workflow stays hidden from the session list (the
            # Claude helpers only bump recency, never the archive bit). Mirrors the
            # Codex unarchive below, but keeps the authoritative Claude usage row
            # (``update_cached_archived`` does not drop ``ArchivedSessionTokenUsage``).
            claude_metadata = _session_detail_metadata(session_id)
            if _metadata_indicates_archived(claude_metadata):
                if _metadata_cwd_is_disallowed(claude_metadata):
                    raise _TurnRejectedError(
                        HttpResponseBadRequest(
                            "thread cwd is not an allowed repository"
                        )
                    )
                session_index.update_cached_archived(session_id, archived=False)
            # ``/fix-pr`` targets the session's already-open PR, so route it to the
            # PR-monitor workflow (no second PR on LGTM) instead of the generic
            # QA/PR activation below -- mirroring the Codex follow-up path.
            if fix_pr_activation:
                fix_pr_response: HttpResponse = _views()._start_claude_fix_pr_workflow(
                    session_id=session_id,
                    settings=settings,
                    input_image_paths=input_image_paths,
                )
                return fix_pr_response
            if qa_workflow_activation:
                qa_response: HttpResponse = _views()._start_claude_qa_workflow(
                    session_id=session_id,
                    qa_activation=qa_activation,
                    settings=settings,
                    input_image_paths=input_image_paths,
                )
                return qa_response
            # ``start_spec_critic_workflow`` runs the should-run classifier on a
            # background thread, so do not pre-classify on the request path here:
            # that would stream a synchronous classifier turn (and classify the
            # prompt twice). Route in whenever Spec Critic is eligible, exactly
            # like the new-session path.
            if (
                settings.spec_critic_enabled
                and not plan_mode
                and not input_image_paths
            ):
                spec_response: HttpResponse = (
                    _views()._start_claude_spec_critic_follow_up(
                        session_id=session_id,
                        prompt=prompt,
                        settings=settings,
                        input_image_paths=input_image_paths,
                    )
                )
                return spec_response
            follow_up_response: HttpResponse = _views()._send_claude_follow_up(
                session_id=session_id,
                prompt=prompt,
                plan_mode=plan_mode,
                settings=settings,
                input_image_paths=input_image_paths,
            )
            return follow_up_response
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
            # Runs after the turn is already spawned: a transient failure here
            # (e.g. "database is locked") must not 500 the request -- that
            # would re-archive the session underneath the live worker and
            # delete the input images it was handed.
            try:
                _record_session_unarchived(session_id)
            except Exception:
                logger.exception(
                    "failed to record session %s unarchived for accepted turn",
                    session_id,
                )
                return
            if metadata is not None:
                metadata.codex_archived = False
                metadata.codex_archived_at = None
                metadata.codex_path = ""

        should_unarchive_for_turn = _metadata_indicates_archived(metadata)
        force_live_resume = _metadata_rollout_path_indicates_archived(metadata)
        if should_unarchive_for_turn:
            if _metadata_cwd_is_disallowed(metadata):
                raise _TurnRejectedError(
                    HttpResponseBadRequest("thread cwd is not an allowed repository")
                )
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
        if metadata_resume is not None and common._thread_cwd(metadata_resume.thread):
            used_disk_resume = True
            resumed = metadata_resume
            thread = metadata_resume.thread
            thread_entries = list(metadata_resume.entries)
            models_data = caches._cached_models_for_session_detail(
                enable_memories=settings.enable_memories
            )
        else:
            used_disk_resume = False
            with app_server_pool.borrow_codex(
                common.Codex, enable_memories=settings.enable_memories
            ) as codex:
                try:
                    resumed = codex._client.thread_resume(session_id)
                except InvalidRequestError as exc:
                    if not _thread_resume_archived_error(exc):
                        raise
                    if _metadata_cwd_is_disallowed(metadata):
                        raise _TurnRejectedError(
                            HttpResponseBadRequest(
                                "thread cwd is not an allowed repository"
                            )
                        ) from exc
                    _unarchive_session_for_turn(session_id, settings, codex=codex)
                    session_unarchived_for_turn = True
                    resumed = codex._client.thread_resume(session_id)
                thread = resumed.thread
                thread_entries = list(_entries_for(thread))
                models_data = common._models_for_plan_mode_fallback(codex)
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
            and prompt == common._PLAN_APPROVAL_PROMPT
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
            with app_server_pool.borrow_codex(
                common.Codex, enable_memories=settings.enable_memories
            ) as codex:
                resumed = codex._client.thread_resume(session_id)
                models_data = common._models_for_plan_mode_fallback(codex)
        collaboration_model = (
            common._plan_mode_model_from_models(resumed, settings, models_data)
            if plan_mode or collaboration_mode == _DEFAULT_COLLABORATION_MODE
            else None
        )
        if plan_mode and not collaboration_model and not intent.explicit_plan_mode:
            plan_mode = False
        cwd = common._thread_cwd(thread)
        if not cwd:
            raise _TurnRejectedError(HttpResponseBadRequest("thread has no cwd"))
        # The session list surfaces every thread the app-server knows about, not
        # just those created via ``new_session``, so the resumed ``cwd`` is not
        # automatically inside the discover_repos() allowlist. Re-validate before
        # spawning so a follow-up cannot run a worker in an unintended directory.
        if not common._is_allowed_session_cwd(cwd):
            raise _TurnRejectedError(
                HttpResponseBadRequest("thread cwd is not an allowed repository")
            )
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
                session_project = common._project_for_cwd(cwd, list(Project.objects.all()))
        developer_instructions = (
            previous_instance.developer_instructions
            if previous_instance is not None
            else common._developer_instructions_for_project(settings, session_project)
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
        base_instructions = common._base_instructions_for_settings(
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
            workflow_model, workflow_reasoning_effort = _stored_model_and_effort(
                resumed, settings
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
                    raise _TurnRejectedError(
                        HttpResponseBadRequest(
                            "fix-pr requires an opened PR for this session"
                        )
                    )
                pr_qa.start_pr_monitor_workflow(
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
            pr_qa.start_pr_qa_workflow(**workflow_kwargs)
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
            auto_review_model, auto_review_reasoning_effort = _stored_model_and_effort(
                resumed, settings
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
                raise _TurnRejectedError(
                    HttpResponseBadRequest("plan mode requires a model")
                )
            spawn_kwargs["model"] = collaboration_model
            spawn_kwargs["plan_mode"] = True
        elif collaboration_mode == _DEFAULT_COLLABORATION_MODE:
            if not collaboration_model:
                raise _TurnRejectedError(
                    HttpResponseBadRequest(
                        "default collaboration mode requires a model"
                    )
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
            workflow_model, workflow_reasoning_effort = _stored_model_and_effort(
                resumed, settings
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
            spec_critic.start_spec_critic_workflow(**spec_workflow_kwargs)
            record_session_unarchived_for_accepted_turn()
            return redirect("session", session_id=session_id)
        codex_pool.spawn_turn(**spawn_kwargs)
        # Ownership transfers the moment the spawn succeeds (matching
        # new_session): any bookkeeping failure after this point must not
        # delete files the worker was handed.
        input_images_owned = True
        record_session_unarchived_for_accepted_turn()
        return redirect("session", session_id=session_id)
    except _TurnRejectedError as rejected:
        restore_archived_session_for_rejected_turn()
        common._cleanup_saved_input_images(steer_image_paths)
        common._cleanup_saved_input_images(input_image_paths)
        return rejected.response
    except codex_pool.InputAttachmentLimitExceededError as exc:
        restore_archived_session_for_rejected_turn()
        common._cleanup_saved_input_images(steer_image_paths)
        common._cleanup_saved_input_images(input_image_paths)
        return HttpResponseBadRequest(str(exc))
    except Exception:
        restore_archived_session_for_rejected_turn()
        common._cleanup_saved_input_images(steer_image_paths)
        if not input_images_owned:
            common._cleanup_saved_input_images(input_image_paths)
        raise

def _duplicate_saved_input_images(paths: Iterable[str]) -> list[str]:
    source_paths = [Path(path) for path in paths if path]
    if not source_paths:
        return []
    saved_paths: list[str] = []
    current_path: Path | None = None
    try:
        attachments_dir = codex_pool.input_attachments_dir()
        common._ensure_private_dir(attachments_dir)
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
        common._cleanup_saved_input_images(cleanup_paths)
        raise
    return saved_paths
