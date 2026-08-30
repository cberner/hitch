"""Per-session action endpoints: rename, archive, project, and approval mode."""

from django.contrib import messages
from django.db import transaction
from django.http import (
    HttpRequest,
    HttpResponse,
    HttpResponseBadRequest,
)
from django.shortcuts import redirect
from django.views.decorators.http import require_http_methods
from openai_codex.errors import InvalidRequestError

from hitch.main.models import (
    ArchivedSessionTokenUsage,
    CodexInstance,
    SessionMetadata,
)
from hitch.main.runtime import app_server_pool, reconciliation
from hitch.main.runtime.db import run_ignoring_database_locks
from hitch.main.sessions import lifecycle as session_lifecycle
from hitch.main.sessions import session_index
from hitch.main.sessions.project_visibility import (
    _metadata_by_thread_id as _metadata_by_thread_id,
)
from hitch.main.sessions.session_resume import _stored_rollout_path_for_thread
from hitch.main.sessions.session_settings import (
    _effective_approval_mode,
    _stored_settings,
)
from hitch.main.sessions.settings_cookies import (
    _VALID_APPROVAL_MODES,
)
from hitch.main.views import common
from hitch.main.workflows import system_agents

_ARCHIVE_ACTIVE_WORK_MESSAGE = (
    "Stop the active turn before archiving this session."
)
_ARCHIVE_BUSY_MESSAGE = "This session is changing. Try archiving again."


def _archive_conflict_response(
    request: HttpRequest, session_id: str, message: str
) -> HttpResponse:
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return HttpResponse(message, status=409, content_type="text/plain")
    messages.error(request, message)
    return redirect("session", session_id=session_id)


def _apply_live_session_approval_mode(
    session_id: str, effective_approval_mode: str
) -> None:
    common._apply_live_approval_mode_to_instances(
        CodexInstance.objects.filter(
            thread_id=session_id,
            status__in=CodexInstance.ACTIVE_STATUSES,
        ),
        effective_approval_mode,
    )


def _resumed_thread_cwd(request: HttpRequest, session_id: str) -> str | None:
    """Resolve a session's cwd via thread_resume; None for archived/unknown.

    The app-server raises InvalidRequestError for archived and nonexistent
    threads -- expected states the rest of the codebase handles, so these
    endpoints must answer 400 instead of 500ing on it.
    """
    settings = _stored_settings(request)
    try:
        resumed = app_server_pool.run_borrowed_op_with_retry(
            common.Codex,
            lambda codex: codex._client.thread_resume(session_id),
            enable_memories=settings.enable_memories,
        )
    except InvalidRequestError:
        return None
    return common._thread_cwd(resumed.thread) or ""

@require_http_methods(["POST"])
def set_session_project(request: HttpRequest, session_id: str) -> HttpResponse:
    project, error = common._posted_project(request.POST.get("project", ""))
    if error is not None:
        return HttpResponseBadRequest(error)
    metadata = SessionMetadata.objects.filter(thread_id=session_id).first()
    cwd = metadata.cwd if metadata is not None and metadata.cwd else ""
    if not cwd:
        resumed_cwd = _resumed_thread_cwd(request, session_id)
        if resumed_cwd is None:
            return HttpResponseBadRequest("session is archived or unknown")
        cwd = resumed_cwd
    SessionMetadata.objects.update_or_create(
        thread_id=session_id,
        defaults={
            "cwd": cwd,
            "project": project,
            "project_cleared": project is None,
        },
    )
    return redirect("session", session_id=session_id)

@require_http_methods(["POST"])
def set_session_approval_mode(request: HttpRequest, session_id: str) -> HttpResponse:
    approval_mode = request.POST.get("approval_mode", "").strip()
    if approval_mode and approval_mode not in _VALID_APPROVAL_MODES:
        return HttpResponseBadRequest("invalid approval mode")
    metadata = SessionMetadata.objects.filter(thread_id=session_id).first()
    cwd = metadata.cwd if metadata is not None and metadata.cwd else ""
    if not cwd:
        resumed_cwd = _resumed_thread_cwd(request, session_id)
        if resumed_cwd is None:
            return HttpResponseBadRequest("session is archived or unknown")
        cwd = resumed_cwd
    SessionMetadata.objects.update_or_create(
        thread_id=session_id,
        defaults={
            "cwd": cwd,
            "approval_mode": approval_mode,
        },
    )
    effective_approval_mode = approval_mode or _effective_approval_mode(
        _stored_settings(request)
    )
    _apply_live_session_approval_mode(session_id, effective_approval_mode)
    return redirect("session", session_id=session_id)

@require_http_methods(["POST"])
def set_session_name(request: HttpRequest, session_id: str) -> HttpResponse:
    name = request.POST.get("name", "").strip()
    if not name:
        return HttpResponseBadRequest("name is required")
    if len(name) > common._NAME_MAX_LEN:
        return HttpResponseBadRequest("name is too long")
    settings = _stored_settings(request)
    try:
        with app_server_pool.borrow_codex(
            common.Codex, enable_memories=settings.enable_memories
        ) as codex:
            codex._client.thread_set_name(session_id, name)
    except InvalidRequestError:
        # The app-server raises this for archived/nonexistent threads (e.g. a
        # rename from a stale tab, or right after archiving). Like the sibling
        # endpoints, answer 400 instead of 500ing on an expected state.
        return HttpResponseBadRequest("session is archived or unknown")
    session_index.update_cached_name(session_id, name)
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
    with session_lifecycle.hold(session_id, blocking=False) as acquired:
        if not acquired:
            return _archive_conflict_response(
                request, session_id, _ARCHIVE_BUSY_MESSAGE
            )
        if archived == "true":
            run_ignoring_database_locks(
                lambda: reconciliation.reconcile_dead_for_thread(session_id),
                description="archive dead-worker reconcile",
            )
            if session_lifecycle.archive_has_active_work(session_id):
                return _archive_conflict_response(
                    request, session_id, _ARCHIVE_ACTIVE_WORK_MESSAGE
                )
        settings = _stored_settings(request)
        is_archived = archived == "true"
        thread_for_metadata = None
        with app_server_pool.borrow_codex(
            common.Codex, enable_memories=settings.enable_memories
        ) as codex:
            metadata_exists = SessionMetadata.objects.filter(
                thread_id=session_id
            ).exists()
            if not metadata_exists and is_archived:
                thread_for_metadata = codex._client.thread_resume(session_id).thread
            if is_archived:
                codex.thread_archive(session_id)
            else:
                codex.thread_unarchive(session_id)
                if not metadata_exists:
                    thread_for_metadata = codex._client.thread_resume(session_id).thread
            rollout_path = _stored_rollout_path_for_thread(
                session_id, archived=is_archived
            )
            try:
                with transaction.atomic():
                    session_index.update_cached_archived(
                        session_id,
                        archived=is_archived,
                        thread=thread_for_metadata,
                    )
                    SessionMetadata.objects.filter(thread_id=session_id).update(
                        codex_path=str(rollout_path) if rollout_path is not None else ""
                    )
            except Exception:
                # Keep Codex and local metadata on the same side when a
                # transient database failure follows the RPC.
                try:
                    if is_archived:
                        codex.thread_unarchive(session_id)
                    else:
                        codex.thread_archive(session_id)
                except Exception:
                    common.logger.exception(
                        "failed to restore archive state for session %s",
                        session_id,
                    )
                raise
        if not is_archived:
            system_agents.retry_deferred_auto_review_for_thread(
                session_id, lifecycle_lock_held=True
            )
        if rollout_path is not None:
            # This thread's rollout moved, so only its file-keyed cache is stale.
            ArchivedSessionTokenUsage.objects.filter(thread_id=session_id).delete()
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return HttpResponse(status=204)
    if request.POST.get("next", "").strip() == "index":
        return redirect("index")
    if archived == "true":
        return redirect("index")
    return redirect("session", session_id=session_id)
