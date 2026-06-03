"""Per-session web demo registration, cleanup, and proxying."""

from __future__ import annotations

import html
import http.client
import json
import logging
import re
import secrets
import shlex
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Final, cast
from urllib.parse import urlsplit

from django.conf import settings
from django.db import transaction
from django.http import Http404, HttpRequest, HttpResponse, JsonResponse, StreamingHttpResponse
from django.urls import reverse
from django.utils.text import slugify

from hitch.main.models import CodexInstance, SessionDemo, SystemAgentRun, SystemWorkflow

logger = logging.getLogger(__name__)

DEFAULT_RUNTIME: Final = "podman"
DEFAULT_CONTAINER_PORT: Final = 3000
DEFAULT_HOST: Final = "127.0.0.1"
DEMO_AGENT_KIND: Final = "demo"
DEMO_DISPLAY_AUTHOR: Final = "Demo agent"
DEMO_WORKFLOW_KIND: Final = "demo_deployment"
LOCAL_DEMO_SUFFIX: Final = ".demo.localhost"
MAX_LOG_CHARS: Final = 20_000
TOKEN_BYTES: Final = 24
HITCH_DEMO_LABEL: Final = "io.hitch.managed"
HITCH_DEMO_LABEL_VALUE: Final = "demo"
HITCH_SESSION_LABEL: Final = "io.hitch.session"
HITCH_TOKEN_LABEL: Final = "io.hitch.demo_token"
HITCH_NAME_LABEL: Final = "io.hitch.container_name"
LOCAL_BIND_HOSTS: Final = frozenset({DEFAULT_HOST, "::1", "localhost"})
DEMO_REGISTRATION_TRANSITIONS: Final[dict[str, frozenset[str]]] = {
    SessionDemo.STATUS_REQUESTED: frozenset(
        {
            SessionDemo.STATUS_PREPARING,
            SessionDemo.STATUS_ACTIVE,
            SessionDemo.STATUS_FAILED,
        }
    ),
    SessionDemo.STATUS_PREPARING: frozenset(
        {
            SessionDemo.STATUS_PREPARING,
            SessionDemo.STATUS_ACTIVE,
            SessionDemo.STATUS_FAILED,
        }
    ),
}
HOP_BY_HOP_HEADERS: Final = {
    "connection",
    "keep-alive",
    "proxy-connection",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
CONTROLLED_REQUEST_HEADERS: Final = {
    "accept-encoding",
    "content-length",
    "expect",
    "forwarded",
    "host",
}
SENSITIVE_REQUEST_HEADERS: Final = {
    "authorization",
    "cookie",
    "x-csrftoken",
    "x-csrf-token",
}
# A browser that cached a previous demo build still revalidates against it. If
# the rebuilt container answers 304 Not Modified, the proxy would relay the
# bodiless 304 and the browser would reuse the stale asset despite the no-store
# response header. Strip only the cache-revalidation validators from safe
# requests so the container always returns a fresh 200 with a body. Other
# conditional headers are preserved: If-Match/If-Unmodified-Since are write
# preconditions, and If-Range must stay paired with Range or the container could
# answer 206 against a rebuilt resource and corrupt a resumed download.
REVALIDATION_REQUEST_HEADERS: Final = {
    "if-modified-since",
    "if-none-match",
}
SENSITIVE_RESPONSE_HEADERS: Final = {
    "clear-site-data",
    "set-cookie",
    "set-cookie2",
}
# The proxy URL is stable across demo rebuilds (a new demo generation reuses the
# same ``/sessions/<id>/demo/`` path), so the container's cache-freshness
# directives would let the browser serve a previous build's HTML/JS/CSS after
# the demo is restarted. Strip them and force ``no-store`` instead so a reload
# always reflects the current container. ETag/Last-Modified are intentionally
# left intact: with no-store and the request revalidators stripped they can no
# longer trigger a stale 304, and an app's JS still needs them as representation
# validators (e.g. to populate If-Match on a later write).
CACHE_RESPONSE_HEADERS: Final = {
    "age",
    "cache-control",
    "expires",
    "pragma",
}
TEXT_REWRITE_TYPES: Final = ("text/html", "text/css", "application/javascript", "text/javascript")
DNS_LABEL_RE: Final = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
CONTAINER_NAME_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
PROMPT_REGISTRATION_TOKEN_RE: Final = re.compile(r"^Registration token: (\S+)$", re.MULTILINE)


class DemoError(Exception):
    """Raised when a demo registration or cleanup operation fails."""


class DemoAlreadyRunningError(DemoError):
    """Raised when a demo setup request is already in progress."""


class DemoContainerLabelMismatchError(DemoError):
    """Raised when a registered container target is not verified for its demo."""


def active_demo_for(session_id: str) -> SessionDemo | None:
    return SessionDemo.objects.filter(
        thread_id=session_id,
        status=SessionDemo.STATUS_ACTIVE,
    ).first()


def latest_demo_for(session_id: str) -> SessionDemo | None:
    return SessionDemo.objects.filter(thread_id=session_id).order_by("-updated_at").first()


def demo_url_for_request(request: HttpRequest, session_id: str) -> str:
    host = request.get_host()
    host_name, _sep, port = host.partition(":")
    demo_host_label = _demo_host_label_for(session_id)
    if host_name in {"localhost", "127.0.0.1"} and demo_host_label is not None:
        suffix = f":{port}" if port else ""
        return f"{request.scheme}://{demo_host_label}.demo.localhost{suffix}/"
    return request.build_absolute_uri(
        reverse("session_demo_proxy_root", kwargs={"session_id": session_id})
    )


def start_demo_prompt_for(
    *,
    request: HttpRequest,
    session_id: str,
    cwd: str,
    demo: SessionDemo,
) -> str:
    demo_url = demo_url_for_request(request, session_id)
    container_prefix = _container_name_prefix(session_id)
    register_command = (
        '$HITCH_MANAGE_COMMAND run --project "$HITCH_PROJECT_DIR" '
        '"$HITCH_MANAGE_PY" register_demo '
        f"--session-id={shlex.quote(session_id)} "
        f"--token={shlex.quote(demo.registration_token)}"
    )
    return (
        "Start an interactive web demo for this session.\n\n"
        "You own the container startup. Use normal shell commands and the user's "
        "configured command approvals. Do not return JSON for Hitch to execute.\n\n"
        f"Repository cwd: {cwd}\n"
        f"Demo URL for the user: {demo_url}\n"
        f"Registration command prefix: {register_command}\n"
        f"Registration token: {demo.registration_token}\n\n"
        "Register demo state with the management command from the worker; do not "
        "route registration through the public browser host.\n\n"
        "Process:\n"
        f"1. Choose a unique container name beginning with {container_prefix}.\n"
        "2. Before creating or running the container, register it as preparing:\n"
        f"   {register_command} --status preparing "
        "--container-name CONTAINER_NAME --logs 'starting demo container'\n"
        "3. Run Podman yourself. Label the container with:\n"
        f"   --label {HITCH_DEMO_LABEL}={HITCH_DEMO_LABEL_VALUE}\n"
        f"   --label {HITCH_SESSION_LABEL}={session_id}\n"
        f"   --label {HITCH_TOKEN_LABEL}={demo.registration_token}\n"
        f"   --label {HITCH_NAME_LABEL}=CONTAINER_NAME\n"
        "4. Bind the web server to 0.0.0.0 inside the container and publish it "
        "on 127.0.0.1 on the host.\n"
        "5. Inspect logs and retry in this same turn until the demo responds.\n"
        "6. When it works, replace CONTAINER_NAME, HOST_PORT, CONTAINER_ID, "
        "and logs, then register it as active:\n"
        f"   {register_command} --status active "
        "--container-name CONTAINER_NAME --container-id CONTAINER_ID "
        f"--host {DEFAULT_HOST} --port HOST_PORT --logs 'concise startup logs'\n"
        "   If you cannot make it work, register failed with the relevant "
        "error and logs:\n"
        f"   {register_command} --status failed "
        "--error 'ERROR' --logs 'relevant failure logs'\n"
        "   For long logs, write them to a file and pass --logs-file PATH instead."
    )


def demo_runtime() -> str:
    runtime = _setting("HITCH_DEMO_RUNTIME", DEFAULT_RUNTIME)
    if runtime != DEFAULT_RUNTIME:
        raise DemoError("only podman demo runtime is supported")
    return runtime


def request_demo_start(session_id: str) -> SessionDemo:
    runtime = demo_runtime()
    protected_targets: set[tuple[str, str]] = set()
    demo, created = SessionDemo.objects.get_or_create(
        thread_id=session_id,
        defaults={
            "host": DEFAULT_HOST,
            "port": DEFAULT_CONTAINER_PORT,
            "runtime": runtime,
            "status": SessionDemo.STATUS_REQUESTED,
            "generation": 1,
            "registration_token": _new_registration_token(),
        },
    )
    if not created:
        if demo.status in {
            SessionDemo.STATUS_REQUESTED,
            SessionDemo.STATUS_PREPARING,
        }:
            raise DemoAlreadyRunningError("demo setup is already running")
        original_generation = demo.generation
        original_registration_token = demo.registration_token
        try:
            _remove_container(demo, ignore_missing=True)
        except DemoContainerLabelMismatchError:
            logger.warning(
                "resetting demo %s despite unverified prior container %s",
                demo.thread_id,
                demo.container_name or demo.container_id,
            )
            if demo.container_id:
                protected_targets.add(("id", demo.container_id))
            if demo.container_name:
                protected_targets.add(("name", demo.container_name))
        with transaction.atomic():
            demo = SessionDemo.objects.select_for_update().get(thread_id=session_id)
            row_changed = (
                demo.generation != original_generation
                or demo.registration_token != original_registration_token
            )
            if row_changed or demo.status in {
                SessionDemo.STATUS_REQUESTED,
                SessionDemo.STATUS_PREPARING,
            }:
                raise DemoAlreadyRunningError("demo setup is already running")
            demo.generation += 1
            demo.host = DEFAULT_HOST
            demo.port = DEFAULT_CONTAINER_PORT
            demo.container_id = ""
            demo.container_name = ""
            demo.runtime = runtime
            demo.status = SessionDemo.STATUS_REQUESTED
            demo.last_error = ""
            demo.logs = ""
            demo.registration_token = _new_registration_token()
            demo.save(
                update_fields=[
                    "generation",
                    "host",
                    "port",
                    "container_id",
                    "container_name",
                    "runtime",
                    "status",
                    "last_error",
                    "logs",
                    "registration_token",
                    "updated_at",
                ]
            )
    cleanup_unregistered_demo_containers(protected_targets=protected_targets)
    return demo


def register_demo_container(session_id: str, payload: dict[str, Any]) -> SessionDemo:
    status = str(payload.get("status") or "").strip()
    if status not in {
        SessionDemo.STATUS_PREPARING,
        SessionDemo.STATUS_ACTIVE,
        SessionDemo.STATUS_FAILED,
    }:
        raise DemoError("invalid demo status")
    token = str(payload.get("token") or "")
    demo = _validate_demo_registration(
        SessionDemo.objects.filter(thread_id=session_id).first(),
        token=token,
        status=status,
    )
    snapshot = _demo_registration_snapshot(demo)
    if status == SessionDemo.STATUS_PREPARING:
        _apply_preparing_registration(demo, payload)
    elif status == SessionDemo.STATUS_ACTIVE:
        _apply_active_registration(demo, payload)
    else:
        _apply_failed_registration(demo, payload)
    with transaction.atomic():
        current = _validate_demo_registration(
            SessionDemo.objects.select_for_update().filter(pk=snapshot.pk).first(),
            token=token,
            status=status,
        )
        if _demo_registration_snapshot(current) != snapshot:
            raise DemoError("demo registration changed")
        _copy_demo_registration_fields(current, demo)
        current.save()
        demo = current
    if status == SessionDemo.STATUS_ACTIVE:
        cleanup_unregistered_demo_containers()
    elif status == SessionDemo.STATUS_FAILED:
        demo = _cleanup_failed_registration(demo)
        if not (demo.container_id or demo.container_name):
            cleanup_unregistered_demo_containers()
    return demo


@dataclass(frozen=True)
class _DemoRegistrationSnapshot:
    pk: int
    generation: int
    registration_token: str
    status: str
    container_id: str
    container_name: str


def _demo_registration_snapshot(demo: SessionDemo) -> _DemoRegistrationSnapshot:
    return _DemoRegistrationSnapshot(
        pk=demo.pk,
        generation=demo.generation,
        registration_token=demo.registration_token,
        status=demo.status,
        container_id=demo.container_id,
        container_name=demo.container_name,
    )


def _validate_demo_registration(
    demo: SessionDemo | None, *, token: str, status: str
) -> SessionDemo:
    if demo is None or not demo.registration_token or not secrets.compare_digest(
        token, demo.registration_token
    ):
        raise DemoError("invalid demo registration token")
    if demo.status == SessionDemo.STATUS_STOPPED:
        raise DemoError("demo has been stopped")
    allowed_statuses = DEMO_REGISTRATION_TRANSITIONS.get(demo.status)
    if allowed_statuses is None or status not in allowed_statuses:
        raise DemoError("demo registration is already complete")
    return demo


def _copy_demo_registration_fields(target: SessionDemo, source: SessionDemo) -> None:
    target.status = source.status
    target.host = source.host
    target.port = source.port
    target.container_id = source.container_id
    target.container_name = source.container_name
    target.runtime = source.runtime
    target.logs = source.logs
    target.last_error = source.last_error


def _apply_preparing_registration(demo: SessionDemo, payload: dict[str, Any]) -> None:
    container_name = _clean_container_name(payload.get("container_name"), demo.thread_id)
    container_id = _clean_container_id(payload.get("container_id"))
    if not container_name:
        raise DemoError("preparing demo registration requires container_name")
    demo.status = SessionDemo.STATUS_PREPARING
    demo.host = DEFAULT_HOST
    demo.port = _optional_port(payload.get("port")) or demo.port or DEFAULT_CONTAINER_PORT
    demo.container_name = container_name
    demo.container_id = container_id
    demo.runtime = _clean_runtime(payload.get("runtime"))
    demo.logs = _bounded_text(payload.get("logs"))
    demo.last_error = ""


def _apply_active_registration(demo: SessionDemo, payload: dict[str, Any]) -> None:
    container_name = _clean_container_name(payload.get("container_name"), demo.thread_id)
    container_id = _clean_container_id(payload.get("container_id"))
    host = _clean_host(payload.get("host"))
    port = _required_port(payload.get("port"))
    if not container_name:
        raise DemoError("active demo registration requires container_name")
    _verify_registered_container_labels(
        target=container_id or container_name,
        thread_id=demo.thread_id,
        token=demo.registration_token,
        container_name=container_name,
        port=port,
    )
    demo.status = SessionDemo.STATUS_ACTIVE
    demo.host = host
    demo.port = port
    demo.container_name = container_name
    demo.container_id = container_id
    demo.runtime = _clean_runtime(payload.get("runtime"))
    demo.logs = _bounded_text(payload.get("logs"))
    demo.last_error = ""


def _apply_failed_registration(demo: SessionDemo, payload: dict[str, Any]) -> None:
    error = _bounded_text(payload.get("error")) or "demo setup failed"
    demo.status = SessionDemo.STATUS_FAILED
    demo.last_error = error
    demo.logs = _bounded_text(payload.get("logs")) or error
    demo.runtime = _clean_runtime(payload.get("runtime"))


def _cleanup_failed_registration(demo_snapshot: SessionDemo) -> SessionDemo:
    if not (demo_snapshot.container_id or demo_snapshot.container_name):
        return demo_snapshot
    try:
        _remove_container(demo_snapshot, ignore_missing=True)
    except DemoError as exc:
        demo_snapshot.last_error = f"{demo_snapshot.last_error}; cleanup failed: {exc}"
        _record_demo_cleanup_result_if_current(
            demo_snapshot,
            last_error=demo_snapshot.last_error,
        )
        logger.exception(
            "failed to remove failed demo container for %s", demo_snapshot.thread_id
        )
    else:
        _record_demo_cleanup_result_if_current(
            demo_snapshot,
            clear_container=True,
        )
        demo_snapshot.container_id = ""
        demo_snapshot.container_name = ""
    return demo_snapshot


def cleanup_demo_for_session(session_id: str) -> None:
    demo = SessionDemo.objects.filter(thread_id=session_id).first()
    if demo is None:
        return
    try:
        _remove_container(demo, ignore_missing=True)
    except DemoError as exc:
        demo.status = SessionDemo.STATUS_FAILED
        demo.last_error = str(exc)
        demo.save(update_fields=["status", "last_error", "updated_at"])
        logger.exception("failed to clean up demo container for session %s", session_id)
        return
    demo.status = SessionDemo.STATUS_STOPPED
    demo.container_id = ""
    demo.container_name = ""
    demo.save(update_fields=["status", "container_id", "container_name", "updated_at"])
    cleanup_unregistered_demo_containers()


def cleanup_unregistered_demo_containers(
    *, protected_targets: set[tuple[str, str]] | None = None
) -> int:
    registered = _registered_demo_targets()
    if protected_targets:
        for kind, target in protected_targets:
            if target:
                registered.add((kind, "", target))
    containers = _hitch_demo_containers()
    removed = 0
    for container in containers:
        if _container_registered(container, registered):
            continue
        target = container.get("id", "") or container.get("name", "")
        if not target:
            continue
        try:
            subprocess.run(
                [DEFAULT_RUNTIME, "rm", "-f", target],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            removed += 1
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            logger.exception("failed to remove unregistered demo container %s", target)
    return removed


def on_codex_instance_finished(instance: CodexInstance) -> None:
    if instance.agent_kind != DEMO_AGENT_KIND:
        return
    instance_token = _registration_token_from_instance(instance)
    run_status = SystemAgentRun.STATUS_COMPLETED
    run_error = ""
    workflow_status = SystemWorkflow.STATUS_COMPLETED
    demo_to_clean: SessionDemo | None = None
    with transaction.atomic():
        demo = SessionDemo.objects.select_for_update().filter(
            thread_id=instance.thread_id
        ).first()
        if (
            demo is not None
            and demo.status in {
                SessionDemo.STATUS_REQUESTED,
                SessionDemo.STATUS_PREPARING,
            }
            and instance_token
            and secrets.compare_digest(instance_token, demo.registration_token)
        ):
            if instance.status == CodexInstance.STATUS_COMPLETED:
                demo.status = SessionDemo.STATUS_FAILED
                demo.last_error = "demo agent finished without registering a container"
            else:
                demo.status = SessionDemo.STATUS_FAILED
                demo.last_error = instance.error or "demo agent did not complete"
            run_status = SystemAgentRun.STATUS_FAILED
            run_error = demo.last_error
            workflow_status = SystemWorkflow.STATUS_FAILED
            demo_to_clean = demo
            demo.save(update_fields=["status", "last_error", "updated_at"])
        elif demo is not None and demo.status == SessionDemo.STATUS_FAILED:
            run_status = SystemAgentRun.STATUS_FAILED
            run_error = demo.last_error or "demo setup failed"
            workflow_status = SystemWorkflow.STATUS_FAILED
        elif instance.status != CodexInstance.STATUS_COMPLETED:
            run_status = SystemAgentRun.STATUS_FAILED
            run_error = instance.error or "demo agent did not complete"
            workflow_status = SystemWorkflow.STATUS_FAILED
    if demo_to_clean is not None:
        try:
            _remove_container(demo_to_clean, ignore_missing=True)
        except DemoError as exc:
            run_error = f"{run_error}; cleanup failed: {exc}"
            _record_demo_cleanup_result_if_current(
                demo_to_clean,
                last_error=run_error,
            )
            logger.exception("failed to remove demo container after agent exit")
        else:
            _record_demo_cleanup_result_if_current(
                demo_to_clean,
                clear_container=True,
            )
    _finish_demo_system_run(
        instance,
        run_status=run_status,
        run_error=run_error,
        workflow_status=workflow_status,
    )
    cleanup_unregistered_demo_containers()


def _record_demo_cleanup_result_if_current(
    demo_snapshot: SessionDemo,
    *,
    last_error: str = "",
    clear_container: bool = False,
) -> None:
    with transaction.atomic():
        current = SessionDemo.objects.select_for_update().filter(
            pk=demo_snapshot.pk,
            generation=demo_snapshot.generation,
            registration_token=demo_snapshot.registration_token,
            status=SessionDemo.STATUS_FAILED,
        ).first()
        if current is None:
            return
        update_fields = ["updated_at"]
        if last_error:
            current.last_error = last_error
            update_fields.append("last_error")
        if clear_container:
            current.container_id = ""
            current.container_name = ""
            update_fields.extend(["container_id", "container_name"])
        if len(update_fields) > 1:
            current.save(update_fields=update_fields)


def _finish_demo_system_run(
    instance: CodexInstance,
    *,
    run_status: str,
    run_error: str,
    workflow_status: str,
) -> None:
    if instance.workflow_id is None:
        return
    workflow = SystemWorkflow.objects.filter(pk=instance.workflow_id).first()
    if workflow is None:
        return
    run, _created = SystemAgentRun.objects.get_or_create(
        instance=instance,
        defaults={
            "workflow": workflow,
            "agent_kind": DEMO_AGENT_KIND,
            "thread_id": instance.thread_id,
            "status": SystemAgentRun.STATUS_RUNNING,
            "input": {"cwd": instance.cwd},
        },
    )
    if run.status not in (SystemAgentRun.STATUS_COMPLETED, SystemAgentRun.STATUS_FAILED):
        run.status = run_status
        run.error = run_error
        run.save(update_fields=["status", "error", "updated_at"])
    if workflow.kind == DEMO_WORKFLOW_KIND and workflow.status == SystemWorkflow.STATUS_RUNNING:
        workflow.status = workflow_status
        workflow.save(update_fields=["status", "updated_at"])


def _registration_token_from_instance(instance: CodexInstance) -> str:
    match = PROMPT_REGISTRATION_TOKEN_RE.search(instance.prompt or "")
    return match.group(1) if match is not None else ""


def proxy_demo_request(
    request: HttpRequest, session_id: str, path: str, *, path_prefix: str
) -> HttpResponse:
    demo = active_demo_for(session_id)
    if demo is None:
        raise Http404("demo not found")

    upstream_path = _upstream_path(path, request.META.get("QUERY_STRING", ""))
    method = request.method or "GET"
    body = request.body if method not in {"GET", "HEAD"} else None
    connection = http.client.HTTPConnection(demo.host, demo.port, timeout=None)
    try:
        connection.request(
            method,
            upstream_path,
            body=body,
            headers=_proxy_request_headers(request, demo, path_prefix=path_prefix),
        )
        upstream = connection.getresponse()
    except (OSError, http.client.HTTPException) as exc:
        # A socket error (OSError) or a malformed status line
        # (http.client.HTTPException, e.g. BadStatusLine) both mean the target
        # misbehaved. Close the connection so its socket is not leaked for the
        # life of the process, and surface either as a 502 rather than letting
        # an HTTPException escape as an unhandled 500.
        connection.close()
        return HttpResponse(f"demo target unavailable: {exc}", status=502)

    response_headers = _proxy_response_headers(
        upstream,
        path_prefix,
        upstream_netloc=f"{demo.host}:{demo.port}",
    )
    content_type = upstream.getheader("Content-Type", "")
    content_encoding = upstream.getheader("Content-Encoding", "").lower()
    if _should_rewrite_body(content_type, path_prefix) and content_encoding in {"", "identity"}:
        # The streaming path closes the connection in _stream_upstream's
        # finally; the buffered path must do the same even when the upstream
        # drops mid-read, or the socket leaks for the life of the process.
        try:
            body_bytes = upstream.read()
        except (OSError, http.client.HTTPException) as exc:
            # A socket drop (OSError) or an early body close such as a short
            # Content-Length (http.client.IncompleteRead, an HTTPException) both
            # mean the target misbehaved; surface either as the same 502 the
            # connect path returns rather than letting it escape as a 500.
            return HttpResponse(f"demo target unavailable: {exc}", status=502)
        finally:
            connection.close()
        response = HttpResponse(
            _rewrite_body(body_bytes, content_type, path_prefix),
            status=upstream.status,
        )
        for key, value in response_headers.items():
            response[key] = value
        return response

    stream_response = StreamingHttpResponse(
        _stream_upstream(upstream, connection),
        status=upstream.status,
    )
    for key, value in response_headers.items():
        stream_response[key] = value
    return cast(HttpResponse, stream_response)


def registration_response(demo: SessionDemo) -> JsonResponse:
    return JsonResponse(
        {
            "status": demo.status,
            "demo_url": reverse(
                "session_demo_proxy_root",
                kwargs={"session_id": demo.thread_id},
            ),
        }
    )


def session_id_from_demo_host(host: str) -> str | None:
    host_name = host.split(":", 1)[0].lower()
    if not host_name.endswith(LOCAL_DEMO_SUFFIX):
        return None
    session_id = host_name[: -len(LOCAL_DEMO_SUFFIX)]
    return session_id if _demo_host_label_for(session_id) == session_id else None


def _demo_host_label_for(session_id: str) -> str | None:
    if session_id != session_id.lower():
        return None
    return session_id if DNS_LABEL_RE.fullmatch(session_id) else None


def _setting(name: str, default: str) -> str:
    value = getattr(settings, name, default)
    return value if isinstance(value, str) and value else default


def _new_registration_token() -> str:
    return secrets.token_urlsafe(TOKEN_BYTES)


def _container_name_prefix(session_id: str) -> str:
    slug = slugify(session_id)[:48] or "session"
    return f"hitch-demo-{slug}-"


def _clean_container_name(value: object, session_id: str | None = None) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if not value:
        return ""
    if not CONTAINER_NAME_RE.fullmatch(value):
        raise DemoError("invalid container name")
    if session_id is not None:
        prefix = _container_name_prefix(session_id)
        if not value.startswith(prefix):
            raise DemoError(f"container name must start with {prefix}")
    return value


def _clean_container_id(value: object) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if not value:
        return ""
    if (
        value.startswith("-")
        or len(value) > 128
        or not re.fullmatch(r"[A-Za-z0-9_.:-]+", value)
    ):
        raise DemoError("invalid container id")
    return value


def _clean_runtime(value: object) -> str:
    runtime = str(value or DEFAULT_RUNTIME).strip() or DEFAULT_RUNTIME
    if runtime != DEFAULT_RUNTIME:
        raise DemoError("only podman demo runtime is supported")
    return runtime


def _clean_host(value: object) -> str:
    host = str(value or DEFAULT_HOST).strip()
    if host not in {DEFAULT_HOST, "localhost"}:
        raise DemoError("demo host must be localhost")
    return DEFAULT_HOST if host == "localhost" else host


def _optional_port(value: object) -> int | None:
    if value in {None, ""}:
        return None
    return _required_port(value)


def _required_port(value: object) -> int:
    if isinstance(value, bool):
        raise DemoError("invalid demo port")
    if not isinstance(value, str | int):
        raise DemoError("invalid demo port")
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise DemoError("invalid demo port") from exc
    if not 1 <= port <= 65535:
        raise DemoError("invalid demo port")
    return port


def _bounded_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value[-MAX_LOG_CHARS:]


def _registered_demo_targets() -> set[tuple[str, str, str]]:
    registered: set[tuple[str, str, str]] = set()
    demos = SessionDemo.objects.filter(
        status__in=(
            SessionDemo.STATUS_PREPARING,
            SessionDemo.STATUS_ACTIVE,
            SessionDemo.STATUS_FAILED,
        )
    ).exclude(registration_token="")
    for demo in demos:
        # Failed demos can retain targets after label verification failed.
        # Keep those targets out of the raw sweep without trusting their labels.
        token = "" if demo.status == SessionDemo.STATUS_FAILED else demo.registration_token
        if demo.container_id:
            registered.add(("id", token, demo.container_id))
        if demo.container_name:
            registered.add(("name", token, demo.container_name))
    return registered


def _hitch_demo_containers() -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            [
                DEFAULT_RUNTIME,
                "ps",
                "-a",
                "--filter",
                f"label={HITCH_DEMO_LABEL}={HITCH_DEMO_LABEL_VALUE}",
                "--format",
                "json",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        logger.exception("failed to list demo containers")
        return []
    return _parse_podman_ps_json(result.stdout)


def _parse_podman_ps_json(output: str) -> list[dict[str, str]]:
    if not isinstance(output, str):
        return []
    if not output.strip():
        return []
    try:
        raw = json.loads(output)
    except json.JSONDecodeError:
        raw = []
        for line in output.splitlines():
            if not line.strip():
                continue
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError:
                return []
    if not isinstance(raw, list):
        raw = [raw]
    containers: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        labels = item.get("Labels") or item.get("labels") or {}
        if isinstance(labels, str):
            labels = _parse_label_string(labels)
        if not isinstance(labels, dict):
            labels = {}
        names = item.get("Names") or item.get("names") or item.get("Name") or item.get("name") or ""
        name = names[0] if isinstance(names, list) and names else str(names or "")
        containers.append(
            {
                "id": str(item.get("ID") or item.get("Id") or item.get("id") or ""),
                "name": name,
                "token": str(labels.get(HITCH_TOKEN_LABEL) or ""),
            }
        )
    return containers


def _parse_label_string(value: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for part in value.split(","):
        key, sep, raw = part.partition("=")
        if sep:
            labels[key.strip()] = raw.strip()
    return labels


def _container_registered(
    container: dict[str, str],
    registered: set[tuple[str, str, str]],
) -> bool:
    token = container.get("token", "")
    container_id = container.get("id", "")
    name = container.get("name", "")
    return (
        (bool(token) and ("id", token, container_id) in registered)
        or ("id", "", container_id) in registered
        or (bool(token) and ("name", token, name) in registered)
        or ("name", "", name) in registered
    )


def _verify_registered_container_labels(
    *,
    target: str,
    thread_id: str,
    token: str,
    container_name: str,
    port: int | None = None,
) -> None:
    inspected = _container_inspect(target)
    if inspected is None:
        raise DemoError("registered demo container was not found")
    labels = _container_labels_from_inspect(inspected)
    expected = {
        HITCH_DEMO_LABEL: HITCH_DEMO_LABEL_VALUE,
        HITCH_SESSION_LABEL: thread_id,
        HITCH_TOKEN_LABEL: token,
        HITCH_NAME_LABEL: container_name,
    }
    for key, value in expected.items():
        if labels.get(key) != value:
            raise DemoError(f"registered demo container is missing label {key}")
    if port is not None and port not in _published_local_host_ports(inspected):
        raise DemoError("registered demo port is not published on localhost by container")


def _container_labels(target: str) -> dict[str, str] | None:
    inspected = _container_inspect(target)
    if inspected is None:
        return None
    return _container_labels_from_inspect(inspected)


def _container_inspect(target: str) -> dict[str, Any] | None:
    try:
        result = subprocess.run(
            [DEFAULT_RUNTIME, "inspect", target],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.CalledProcessError as exc:
        if _container_missing_error(exc):
            return None
        raise DemoError(
            f"failed to inspect demo container {target}: {_subprocess_error_message(exc)}"
        ) from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DemoError(
            f"failed to inspect demo container {target}: {_subprocess_error_message(exc)}"
        ) from exc
    try:
        raw = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}
    if isinstance(raw, list):
        item = raw[0] if raw else {}
    else:
        item = raw
    if not isinstance(item, dict):
        return {}
    return item


def _container_labels_from_inspect(item: dict[str, Any]) -> dict[str, str]:
    config = item.get("Config")
    labels: object = {}
    if isinstance(config, dict):
        labels = config.get("Labels") or {}
    if not isinstance(labels, dict):
        labels = item.get("Labels") or {}
    if not isinstance(labels, dict):
        return {}
    return {str(key): str(value) for key, value in labels.items()}


def _published_local_host_ports(item: dict[str, Any]) -> set[int]:
    ports: set[int] = set()
    for bindings in _container_port_binding_sources(item):
        for entries in bindings.values():
            if isinstance(entries, dict):
                entries = [entries]
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                host_ip = str(entry.get("HostIp") or entry.get("HostIP") or "")
                if host_ip not in LOCAL_BIND_HOSTS:
                    continue
                try:
                    host_port = int(str(entry.get("HostPort") or ""))
                except ValueError:
                    continue
                ports.add(host_port)
    return ports


def _container_port_binding_sources(item: dict[str, Any]) -> Iterator[dict[Any, Any]]:
    network_settings = item.get("NetworkSettings")
    if isinstance(network_settings, dict):
        ports = network_settings.get("Ports")
        if isinstance(ports, dict):
            yield ports
    host_config = item.get("HostConfig")
    if isinstance(host_config, dict):
        bindings = host_config.get("PortBindings")
        if isinstance(bindings, dict):
            yield bindings


def _demo_labels_match(
    demo: SessionDemo, labels: dict[str, str], *, require_name: bool
) -> bool:
    if labels.get(HITCH_DEMO_LABEL) != HITCH_DEMO_LABEL_VALUE:
        return False
    if labels.get(HITCH_SESSION_LABEL) != demo.thread_id:
        return False
    if demo.registration_token and labels.get(HITCH_TOKEN_LABEL) != demo.registration_token:
        return False
    if require_name and demo.container_name:
        return labels.get(HITCH_NAME_LABEL) == demo.container_name
    return True


def _remove_container(demo: SessionDemo, *, ignore_missing: bool = False) -> None:
    if not demo.container_id and not demo.container_name:
        return
    target = demo.container_id or demo.container_name
    if demo.registration_token:
        labels = _container_labels(target)
        if labels is None:
            if ignore_missing:
                return
            raise DemoError(f"demo container {target} could not be inspected")
        if not _demo_labels_match(demo, labels, require_name=not demo.container_id):
            logger.warning("refusing to remove unverified demo container %s", target)
            raise DemoContainerLabelMismatchError(
                "demo container labels did not match registration"
            )
    try:
        subprocess.run(
            [demo.runtime or DEFAULT_RUNTIME, "rm", "-f", target],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        if ignore_missing and _container_missing_error(exc):
            return
        raise DemoError(_subprocess_error_message(exc)) from exc


def _container_missing_error(
    exc: OSError | subprocess.CalledProcessError | subprocess.TimeoutExpired,
) -> bool:
    if not isinstance(exc, subprocess.CalledProcessError):
        return False
    detail = f"{exc.stderr or ''}\n{exc.stdout or ''}".lower()
    return any(
        marker in detail
        for marker in (
            "does not exist",
            "no such container",
            "not found",
            "no container with name or id",
        )
    )


def _subprocess_error_message(
    exc: OSError | subprocess.CalledProcessError | subprocess.TimeoutExpired,
) -> str:
    if isinstance(exc, subprocess.CalledProcessError):
        detail = (exc.stderr or exc.stdout or "").strip()
        return detail or f"podman exited with status {exc.returncode}"
    if isinstance(exc, subprocess.TimeoutExpired):
        return "podman command timed out"
    return str(exc)


def _upstream_path(path: str, query_string: str) -> str:
    normalized = "/" + path.lstrip("/")
    if query_string:
        return f"{normalized}?{query_string}"
    return normalized


def _proxy_request_headers(
    request: HttpRequest, demo: SessionDemo, *, path_prefix: str
) -> dict[str, str]:
    blocked_headers = (
        HOP_BY_HOP_HEADERS
        | CONTROLLED_REQUEST_HEADERS
        | SENSITIVE_REQUEST_HEADERS
        | _connection_header_tokens(request.headers.get("Connection", ""))
    )
    # Only drop cache revalidators from safe requests. On unsafe methods
    # (PUT/PATCH/DELETE) If-None-Match/If-Modified-Since can act as write
    # preconditions (e.g. If-None-Match: * for create-if-absent), so leave them
    # intact there.
    if request.method in {"GET", "HEAD"}:
        blocked_headers = blocked_headers | REVALIDATION_REQUEST_HEADERS
    headers = {
        key: value
        for key, value in request.headers.items()
        if not _blocked_request_header(key, blocked_headers)
    }
    headers["Host"] = f"{demo.host}:{demo.port}"
    headers["Accept-Encoding"] = "identity"
    headers["X-Forwarded-Host"] = request.headers.get("Host", "")
    headers["X-Forwarded-Proto"] = request.scheme or "http"
    headers["X-Forwarded-For"] = request.META.get("REMOTE_ADDR", "")
    headers["X-Forwarded-Prefix"] = path_prefix.rstrip("/") or "/"
    return headers


def _proxy_response_headers(
    upstream: http.client.HTTPResponse,
    path_prefix: str,
    *,
    upstream_netloc: str,
) -> dict[str, str]:
    upstream_headers = upstream.getheaders()
    blocked_headers = (
        HOP_BY_HOP_HEADERS
        | SENSITIVE_RESPONSE_HEADERS
        | CACHE_RESPONSE_HEADERS
        | {"content-length"}
        | {
            token
            for key, value in upstream_headers
            if key.lower() == "connection"
            for token in _connection_header_tokens(value)
        }
    )
    headers: dict[str, str] = {}
    for key, value in upstream_headers:
        lower = key.lower()
        if lower in blocked_headers:
            continue
        if lower == "location":
            value = _rewrite_location(value, path_prefix, upstream_netloc=upstream_netloc)
        if _header_has_control_chars(key) or _header_has_control_chars(value):
            # Obsolete RFC 7230 line folding (and outright header injection from a
            # misbehaving demo container) arrives with embedded CR/LF. Copying it
            # onto the Django response raises BadHeaderError, which 500s the proxy
            # request and -- on the streaming path -- leaks the upstream socket
            # because the generator's close() never runs. Drop such headers.
            continue
        headers[key] = value
    headers["Cache-Control"] = "no-store"
    return headers


def _header_has_control_chars(value: str) -> bool:
    return "\n" in value or "\r" in value


def _connection_header_tokens(value: str) -> set[str]:
    return {token.strip().lower() for token in value.split(",") if token.strip()}


def _blocked_request_header(key: str, blocked_headers: set[str]) -> bool:
    lower = key.lower()
    return lower in blocked_headers or lower.startswith("x-forwarded-")


def _rewrite_location(value: str, path_prefix: str, *, upstream_netloc: str) -> str:
    parsed = urlsplit(value)
    if parsed.netloc and parsed.netloc != upstream_netloc:
        return value
    path = parsed.path
    if not path.startswith("/"):
        return value
    rewritten = f"{path_prefix}{path.lstrip('/')}"
    if parsed.query:
        rewritten = f"{rewritten}?{parsed.query}"
    if parsed.fragment:
        rewritten = f"{rewritten}#{parsed.fragment}"
    return rewritten


def _should_rewrite_body(content_type: str, path_prefix: str) -> bool:
    if path_prefix == "/":
        return False
    return any(content_type.lower().startswith(item) for item in TEXT_REWRITE_TYPES)


def _rewrite_body(body: bytes, content_type: str, path_prefix: str) -> bytes:
    encoding = "utf-8"
    match = re.search(r"charset=([^;\s]+)", content_type, re.IGNORECASE)
    if match:
        # Charset values may be quoted (charset="utf-8"); strip so the lookup
        # doesn't fail and silently skip rewriting.
        encoding = match.group(1).strip().strip("\"'") or "utf-8"
    try:
        text = body.decode(encoding)
    except (LookupError, UnicodeDecodeError):
        return body
    escaped = html.escape(path_prefix.rstrip("/"), quote=True)
    text = re.sub(r'(?P<attr>\b(?:href|src|action|poster)=["\'])/(?!/)', rf"\g<attr>{escaped}/", text)
    text = re.sub(
        r'(?P<attr>\bsrcset=["\'])(?P<value>[^"\']*)',
        lambda match: f"{match.group('attr')}{_rewrite_srcset(match.group('value'), escaped)}",
        text,
    )
    text = re.sub(r"url\((?P<quote>['\"]?)/(?!/)", rf"url(\g<quote>{escaped}/", text)
    text = re.sub(
        r"(?P<call>\b(?:fetch|import|EventSource|WebSocket|Worker|SharedWorker)\(\s*(?P<quote>['\"]))/(?!/)",
        rf"\g<call>{escaped}/",
        text,
    )
    text = re.sub(
        r"(?P<stmt>\b(?:import|export)\s+(?:[^'\"]+\s+from\s+)?(?P<quote>['\"]))/(?!/)",
        rf"\g<stmt>{escaped}/",
        text,
    )
    return text.encode(encoding)


def _rewrite_srcset(value: str, path_prefix: str) -> str:
    candidates: list[str] = []
    for candidate in value.split(","):
        stripped = candidate.lstrip()
        leading = candidate[: len(candidate) - len(stripped)]
        if stripped.startswith("/") and not stripped.startswith("//"):
            stripped = f"{path_prefix}{stripped}"
        candidates.append(f"{leading}{stripped}")
    return ",".join(candidates)


def _stream_upstream(
    upstream: http.client.HTTPResponse,
    connection: http.client.HTTPConnection,
) -> Iterator[bytes]:
    try:
        while True:
            chunk = upstream.read(64 * 1024)
            if not chunk:
                break
            yield chunk
    finally:
        connection.close()
