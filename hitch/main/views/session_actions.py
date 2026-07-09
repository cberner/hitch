"""Per-session action endpoints: rename, archive, project, approval mode, demo."""
import json

from django.db import IntegrityError, transaction
from django.http import (
    HttpRequest,
    HttpResponse,
    HttpResponseBadRequest,
)
from django.shortcuts import redirect
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from openai_codex.errors import InvalidRequestError

from hitch.main import demo
from hitch.main.models import (
    ArchivedSessionTokenUsage,
    CodexInstance,
    SessionMetadata,
    SystemAgentRun,
    SystemWorkflow,
)
from hitch.main.runtime import app_server_pool, codex_pool
from hitch.main.sessions import session_index
from hitch.main.sessions.project_visibility import (
    _metadata_by_thread_id as _metadata_by_thread_id,
)
from hitch.main.sessions.session_settings import (
    _allowed_session_cwds,
    _effective_approval_mode,
    _effective_approval_mode_for_session,
    _effective_sandbox_policy_for_cwd,
    _stored_settings,
)
from hitch.main.sessions.settings_cookies import (
    _VALID_APPROVAL_MODES,
    _valid_web_search_mode_or_default,
)
from hitch.main.views import common
from hitch.main.workflows import system_agents


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
            lambda codex: app_server_pool.thread_resume_response_tolerating_sdk_metadata(
                codex, thread_id=session_id
            ),
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
    settings = _stored_settings(request)
    with app_server_pool.borrow_codex(
        common.Codex, enable_memories=settings.enable_memories
    ) as codex:
        if archived == "true":
            codex.thread_archive(session_id)
        else:
            codex.thread_unarchive(session_id)
    if archived == "true":
        demo.cleanup_demo_for_session(session_id)
    session_index.update_cached_archived(session_id, archived=archived == "true")
    # Codex moves this thread's rollout in/out of ``archived_sessions/`` when
    # the archive bit flips, which invalidates *this* thread's cached usage
    # row. Other threads' caches still match their rollouts, so leave them
    # alone — a blanket wipe forces /profile and /usage to re-parse every
    # archived rollout file the next time they render.
    ArchivedSessionTokenUsage.objects.filter(thread_id=session_id).delete()
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return HttpResponse(status=204)
    if request.POST.get("next", "").strip() == "index":
        return redirect("index")
    if archived == "true":
        return redirect("index")
    return redirect("session", session_id=session_id)

def _mark_workflow_failed(workflow: SystemWorkflow) -> None:
    SystemWorkflow.objects.filter(pk=workflow.pk).update(
        status=SystemWorkflow.STATUS_FAILED,
        updated_at=timezone.now(),
    )

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
    if SystemWorkflow.objects.filter(
        kind=demo.DEMO_WORKFLOW_KIND,
        main_thread_id=session_id,
        status=SystemWorkflow.STATUS_RUNNING,
    ).exists():
        return HttpResponseBadRequest("demo setup workflow is already running")
    settings = _stored_settings(request)
    try:
        resumed = app_server_pool.run_borrowed_op_with_retry(
            common.Codex,
            lambda codex: app_server_pool.thread_resume_response_tolerating_sdk_metadata(
                codex, thread_id=session_id
            ),
            enable_memories=settings.enable_memories,
        )
    except InvalidRequestError:
        return HttpResponseBadRequest("session is archived or unknown")
    thread = resumed.thread
    cwd = common._thread_cwd(thread)
    if not cwd:
        return HttpResponseBadRequest("thread has no cwd")
    if cwd not in _allowed_session_cwds():
        return HttpResponseBadRequest("thread cwd is not an allowed repository")
    sandbox_policy = _effective_sandbox_policy_for_cwd(settings, cwd)
    try:
        with transaction.atomic():
            workflow = SystemWorkflow.objects.create(
                kind=demo.DEMO_WORKFLOW_KIND,
                main_thread_id=session_id,
                cwd=cwd,
                status=SystemWorkflow.STATUS_RUNNING,
                step="demo_running",
                state={},
            )
    except IntegrityError:
        return HttpResponseBadRequest("demo setup workflow is already running")
    try:
        session_demo = demo.request_demo_start(session_id)
    except demo.DemoAlreadyRunningError as exc:
        _mark_workflow_failed(workflow)
        return HttpResponseBadRequest(str(exc))
    except demo.DemoError as exc:
        _mark_workflow_failed(workflow)
        return HttpResponse(str(exc), status=500, content_type="text/plain")
    except Exception:
        _mark_workflow_failed(workflow)
        raise
    try:
        workflow.state = {"session_demo_id": session_demo.pk}
        workflow.save(update_fields=["state", "updated_at"])
        prompt = demo.start_demo_prompt_for(
            request=request,
            session_id=session_id,
            cwd=cwd,
            demo=session_demo,
        )
        instance = codex_pool.spawn_turn(
            thread_id=session_id,
            cwd=cwd,
            prompt=prompt,
            sandbox_policy=sandbox_policy or None,
            approval_mode=_effective_approval_mode_for_session(settings, session_id),
            web_search_mode=_valid_web_search_mode_or_default(
                settings.web_search_mode
            )
            or None,
            enable_memories=settings.enable_memories,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=demo.DEMO_AGENT_KIND,
            display_author=demo.DEMO_DISPLAY_AUTHOR,
            user_message_index=None,
        )
        SystemAgentRun.objects.get_or_create(
            instance=instance,
            defaults={
                "workflow": workflow,
                "agent_kind": demo.DEMO_AGENT_KIND,
                "thread_id": instance.thread_id,
                "status": SystemAgentRun.STATUS_RUNNING,
                "input": {"cwd": cwd, "session_id": session_id},
            },
        )
    except Exception:
        _mark_workflow_failed(workflow)
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
