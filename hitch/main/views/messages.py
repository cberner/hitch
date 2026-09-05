"""Send-message intent parsing, steering, and ordinary turn spawning."""
import logging
import os
import uuid
from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from django.http import (
    HttpRequest,
    HttpResponse,
    HttpResponseBadRequest,
)
from django.shortcuts import redirect
from django.views.decorators.http import require_http_methods
from openai_codex.errors import InvalidRequestError

from hitch.main import caches
from hitch.main.models import (
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
from hitch.main.sessions import agent_tasks
from hitch.main.sessions import lifecycle as session_lifecycle
from hitch.main.sessions.message_intent import (
    _message_intent,
    _MessageIntent,
)
from hitch.main.sessions.project_visibility import (
    _metadata_by_thread_id as _metadata_by_thread_id,
)
from hitch.main.sessions.session_approval import _parse_instance_id
from hitch.main.sessions.session_entry_display import (
    _entries_for,
)
from hitch.main.sessions.session_pr_plan import (
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
    _thread_resume_archived_error,
    _unarchive_session_for_turn,
    thread_has_dynamic_tool,
)
from hitch.main.sessions.session_settings import (
    _effective_approval_mode_for_session,
    _effective_sandbox_policy_for_cwd,
    _stored_settings,
)
from hitch.main.sessions.settings_cookies import (
    SettingsValues,
    _valid_web_search_mode_or_default,
)
from hitch.main.views import common

logger = logging.getLogger(__name__)

_PLAN_ACTION_APPROVE = "approve"

_PLAN_ACTION_REVISE = "revise"

_VALID_PLAN_ACTIONS = frozenset({"", _PLAN_ACTION_APPROVE, _PLAN_ACTION_REVISE})

_DEFAULT_COLLABORATION_MODE = "default"


@dataclass(frozen=True)
class _MessageSubmission:
    intent: _MessageIntent
    collaboration_mode: str
    pr_now_activation: bool
    fix_pr_activation: bool
    qa_activation: bool
    agent_task_activation: bool
    has_input_images: bool


def _message_submission(
    request: HttpRequest,
) -> tuple[_MessageSubmission | None, str | None]:
    intent = _message_intent(request)
    pr_activation = intent.pr_activation
    pr_now_activation = intent.pr_now_activation
    fix_pr_activation = intent.fix_pr_activation
    qa_activation = intent.qa_activation
    agent_task_activation = (
        pr_activation or pr_now_activation or qa_activation or fix_pr_activation
    )
    has_input_images = common._has_input_image_uploads(request)
    if not intent.prompt and not has_input_images:
        return None, "prompt is required"

    collaboration_mode = request.POST.get("collaboration_mode", "").strip().lower()
    plan_action = request.POST.get("plan_action", "").strip().lower()
    if plan_action not in _VALID_PLAN_ACTIONS:
        return None, "invalid plan action"
    if plan_action == _PLAN_ACTION_APPROVE:
        collaboration_mode = _DEFAULT_COLLABORATION_MODE
        intent = intent._replace(plan_mode=False)
    elif plan_action == _PLAN_ACTION_REVISE:
        collaboration_mode = ""
        intent = intent._replace(plan_mode=True)
    if collaboration_mode and collaboration_mode != _DEFAULT_COLLABORATION_MODE:
        return None, "invalid collaboration mode"
    if collaboration_mode and intent.plan_mode and intent.explicit_plan_mode:
        return None, "collaboration mode conflicts with plan mode"
    if agent_task_activation and collaboration_mode:
        return None, "review and PR tasks conflict with collaboration mode"
    if collaboration_mode or agent_task_activation:
        intent = intent._replace(plan_mode=False)
    return (
        _MessageSubmission(
            intent=intent,
            collaboration_mode=collaboration_mode,
            pr_now_activation=pr_now_activation,
            fix_pr_activation=fix_pr_activation,
            qa_activation=qa_activation,
            agent_task_activation=agent_task_activation,
            has_input_images=has_input_images,
        ),
        None,
    )


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


def _stored_model_and_effort(resumed: Any, settings: SettingsValues) -> tuple[str, str]:
    """Return one atomic recorded pair, or the request settings pair."""
    recorded = getattr(resumed, "model_config", None)
    if recorded is not None:
        model = string_value(recorded.model)
        effort = string_value(recorded.reasoning_effort)
        if model or effort:
            return model, effort
    model = string_value(getattr(resumed, "model", None))
    effort = string_value(getattr(resumed, "reasoning_effort", None))
    if model or effort:
        return model, effort
    return settings.model, settings.reasoning_effort


def _steer_instance_with_input_images(
    instance_id: int,
    *,
    session_id: str,
    prompt: str,
    input_image_paths: list[str],
) -> bool:
    """Try to steer one worker while keeping image-copy ownership explicit."""
    steer_image_paths = _duplicate_saved_input_images(input_image_paths)
    steer_kwargs: dict[str, Any] = {
        "expected_thread_id": session_id,
        "prompt": prompt,
    }
    if steer_image_paths:
        steer_kwargs["input_image_paths"] = steer_image_paths
    try:
        steered = codex_pool.steer_instance(instance_id, **steer_kwargs)
    except Exception:
        common._cleanup_saved_input_images(steer_image_paths)
        raise
    if steered is None:
        common._cleanup_saved_input_images(steer_image_paths)
        return False
    # The duplicated files now belong to the worker's steering request. The
    # originals were saved for a possible fallback turn and are no longer used.
    common._cleanup_saved_input_images(input_image_paths)
    return True


@_limit_input_image_uploads
@require_http_methods(["POST"])
def send_message(request: HttpRequest, session_id: str) -> HttpResponse:
    submission, submission_error = _message_submission(request)
    if submission is None:
        return HttpResponseBadRequest(submission_error or "invalid message")
    intent = submission.intent
    pr_now_activation = submission.pr_now_activation
    fix_pr_activation = submission.fix_pr_activation
    qa_activation = submission.qa_activation
    agent_task_activation = submission.agent_task_activation
    prompt = intent.prompt
    plan_mode = intent.plan_mode
    collaboration_mode = submission.collaboration_mode
    has_input_images = submission.has_input_images
    run_ignoring_database_locks(
        lambda: reconciliation.reconcile_dead_for_thread(session_id),
        description="send-message dead-worker reconcile",
    )
    if agent_task_activation and has_input_images:
        return HttpResponseBadRequest(
            "image attachments are not supported for review or PR tasks"
        )
    settings = _stored_settings(request)
    raw_active = request.POST.get("active_instance", "").strip()
    active_instance = None
    instance_id: int | None = None
    if raw_active:
        if agent_task_activation:
            return HttpResponseBadRequest("review and PR tasks require an idle session")
        instance_id, error = _parse_instance_id(raw_active)
        if error is not None or instance_id is None:
            return HttpResponseBadRequest(error or "invalid instance id")
    else:
        active_instance = codex_pool.latest_active_for_thread(session_id)
        if active_instance is not None and agent_task_activation:
            return HttpResponseBadRequest("review and PR tasks require an idle session")
    input_image_paths, input_image_error = common._save_posted_input_images(request)
    if input_image_error is not None:
        return HttpResponseBadRequest(input_image_error)

    input_images_owned = False
    session_unarchived_for_turn = False
    session_unarchive_recorded = False
    lifecycle_lock_held = False
    lifecycle_stack = ExitStack()

    def restore_archived_session_for_rejected_turn() -> None:
        if session_unarchived_for_turn:
            _restore_archived_session_for_rejected_turn(session_id, settings)

    def hold_lifecycle_lock() -> None:
        nonlocal lifecycle_lock_held
        if lifecycle_lock_held:
            return
        lifecycle_stack.enter_context(session_lifecycle.hold(session_id))
        lifecycle_lock_held = True

    try:
        steer_instance_id: int | None = None
        if raw_active:
            assert instance_id is not None
            steer_instance_id = instance_id
        elif active_instance is not None:
            steer_instance_id = active_instance.pk
        if steer_instance_id is not None and _steer_instance_with_input_images(
            steer_instance_id,
            session_id=session_id,
            prompt=prompt,
            input_image_paths=input_image_paths,
        ):
            input_images_owned = True
            return redirect("session", session_id=session_id)
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
        # A fallback resume owns a private app-server that closes before spawning
        # the worker, releasing its exclusive thread writer lease.
        hold_lifecycle_lock()
        metadata = _session_detail_metadata(session_id)

        def record_session_unarchived_for_accepted_turn(
            *, best_effort: bool = True
        ) -> None:
            nonlocal session_unarchive_recorded
            if not session_unarchived_for_turn or session_unarchive_recorded:
                return
            # Post-spawn bookkeeping is best effort: a transient failure must
            # not re-archive underneath a live worker. Task starts record first
            # because their durable-state guard reads this metadata.
            try:
                _record_session_unarchived(session_id, thread=thread)
            except Exception:
                if not best_effort:
                    raise
                logger.exception(
                    "failed to record session %s unarchived for accepted turn",
                    session_id,
                )
                return
            session_unarchive_recorded = True
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
            hold_lifecycle_lock()
            metadata = _session_detail_metadata(session_id)
            should_unarchive_for_turn = _metadata_indicates_archived(metadata)
            force_live_resume = _metadata_rollout_path_indicates_archived(metadata)
        if should_unarchive_for_turn:
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
            with app_server_pool.open_codex(
                lambda: common.Codex(
                    config=codex_pool.app_server_config(
                        enable_memories=settings.enable_memories
                    )
                )
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
                    hold_lifecycle_lock()
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
                or agent_task_activation
            )
            and not string_value(getattr(resumed, "model", None))
        ):
            with app_server_pool.open_codex(
                lambda: common.Codex(
                    config=codex_pool.app_server_config(
                        enable_memories=settings.enable_memories
                    )
                )
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
        # defaults — which breaks multi-turn sessions that depend on
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
            if agent_task_activation and not configured_web_search_mode
            else configured_web_search_mode
        )
        should_forward_web_search_mode = bool(web_search_mode) or bool(
            previous_web_search_mode
        )
        auto_pr_enabled = _auto_pr_enabled_for_session(session_id)
        auto_qa_enabled = (
            False if auto_pr_enabled else _auto_qa_enabled_for_session(session_id)
        )
        if agent_task_activation:
            task_model, task_reasoning_effort = _stored_model_and_effort(
                resumed, settings
            )
            if fix_pr_activation:
                pr_url = _fix_pr_url_for_thread(session_id)
                if not pr_url:
                    raise _TurnRejectedError(
                        HttpResponseBadRequest(
                            "fix-pr requires an opened PR for this session"
                        )
                    )
                task = agent_tasks.watch_pr_task(pr_url)
            elif pr_now_activation:
                task = agent_tasks.publish_pr_task()
            else:
                task = agent_tasks.review_task(
                    prepare_pull_request=not qa_activation
                )
            if task.requires_pr_watch and not thread_has_dynamic_tool(
                session_id,
                namespace="hitch",
                name="watch_pr",
            ):
                raise _TurnRejectedError(
                    HttpResponseBadRequest(
                        "hitch.watch_pr is unavailable for this session; "
                        "start a new session before publishing or watching a PR"
                    )
                )
            task_kwargs: dict[str, Any] = {
                "thread_id": session_id,
                "cwd": cwd,
                "prompt": task.prompt,
                "sandbox_policy": sandbox_policy or None,
                "approval_mode": approval_mode,
                "model": task_model or None,
                "reasoning_effort": task_reasoning_effort or None,
                "developer_instructions": developer_instructions or None,
                "enable_memories": settings.enable_memories,
                "user_message_index": _count_user_entries(thread_entries),
                "agent_kind": task.agent_kind,
            }
            if should_forward_web_search_mode:
                task_kwargs["web_search_mode"] = web_search_mode
            codex_pool.spawn_turn(**task_kwargs)
            record_session_unarchived_for_accepted_turn()
            return redirect("session", session_id=session_id)
        automatic_pr_available = bool(
            auto_pr_enabled
            and not plan_mode
            and thread_has_dynamic_tool(
                session_id,
                namespace="hitch",
                name="watch_pr",
            )
        )
        if not plan_mode:
            prompt = agent_tasks.with_automatic_review_guidance(
                prompt,
                auto_pr_enabled=automatic_pr_available,
                auto_qa_enabled=auto_qa_enabled,
            )
        spawn_kwargs: dict[str, Any] = {
            "thread_id": session_id,
            "cwd": cwd,
            "prompt": prompt,
            "sandbox_policy": sandbox_policy or None,
            "approval_mode": approval_mode,
        }
        if automatic_pr_available:
            spawn_kwargs["agent_kind"] = agent_tasks.PR_PUBLISH_AGENT_KIND
            spawn_kwargs["user_message_index"] = _count_user_entries(thread_entries)
        if input_image_paths:
            spawn_kwargs["input_image_paths"] = input_image_paths
        if should_forward_web_search_mode:
            spawn_kwargs["web_search_mode"] = web_search_mode
        if previous_instance is None and developer_instructions:
            spawn_kwargs["developer_instructions"] = developer_instructions
        if settings.enable_memories:
            spawn_kwargs["enable_memories"] = True
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
        codex_pool.spawn_turn(**spawn_kwargs)
        # Ownership transfers the moment the spawn succeeds (matching
        # new_session): any bookkeeping failure after this point must not
        # delete files the worker was handed.
        input_images_owned = True
        record_session_unarchived_for_accepted_turn()
        return redirect("session", session_id=session_id)
    except _TurnRejectedError as rejected:
        restore_archived_session_for_rejected_turn()
        common._cleanup_saved_input_images(input_image_paths)
        return rejected.response
    except codex_pool.InputAttachmentLimitExceededError as exc:
        restore_archived_session_for_rejected_turn()
        common._cleanup_saved_input_images(input_image_paths)
        return HttpResponseBadRequest(str(exc))
    except Exception:
        restore_archived_session_for_rejected_turn()
        if not input_images_owned:
            common._cleanup_saved_input_images(input_image_paths)
        raise
    finally:
        lifecycle_stack.close()


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
