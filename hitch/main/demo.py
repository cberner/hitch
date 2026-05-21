"""Per-session web demo containers and proxying."""

from __future__ import annotations

import html
import http.client
import logging
import re
import socket
import subprocess
import uuid
from collections.abc import Iterator
from typing import Final, cast
from urllib.parse import urlsplit

from django.conf import settings
from django.db import transaction
from django.http import Http404, HttpRequest, HttpResponse, StreamingHttpResponse
from django.urls import reverse
from django.utils.text import slugify

from hitch.main.models import SessionDemo

logger = logging.getLogger(__name__)

DEFAULT_RUNTIME: Final = "podman"
DEFAULT_IMAGE: Final = "node:22-bookworm"
DEFAULT_CONTAINER_PORT: Final = 3000
DEFAULT_HOST: Final = "127.0.0.1"
LOCAL_DEMO_SUFFIX: Final = ".demo.localhost"
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
SENSITIVE_RESPONSE_HEADERS: Final = {
    "clear-site-data",
    "set-cookie",
    "set-cookie2",
}
TEXT_REWRITE_TYPES: Final = ("text/html", "text/css", "application/javascript", "text/javascript")
DNS_LABEL_RE: Final = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


class DemoError(Exception):
    """Raised when a demo container cannot be started or stopped."""


def active_demo_for(session_id: str) -> SessionDemo | None:
    return SessionDemo.objects.filter(
        thread_id=session_id,
        status=SessionDemo.STATUS_ACTIVE,
    ).first()


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


def demo_prompt_for(
    *,
    request: HttpRequest,
    session_id: str,
    demo: SessionDemo,
    container_port: int,
) -> str:
    demo_url = demo_url_for_request(request, session_id)
    target = f"{demo.host}:{demo.port}"
    return (
        "Start an interactive web demo for this session.\n\n"
        f"Hitch has started a Podman container for you:\n"
        f"- container id: {demo.container_id}\n"
        f"- container name: {demo.container_name}\n"
        f"- internal web port: {container_port}\n"
        f"- host target: {target}\n"
        f"- browser demo URL: {demo_url}\n\n"
        "Copy or prepare the feature code inside the container, install what the "
        f"frontend needs, and run the web server on 0.0.0.0:{container_port}. "
        "When it is ready, tell the user to open the browser demo URL."
    )


def start_demo_container(session_id: str) -> tuple[SessionDemo, int]:
    runtime = _setting("HITCH_DEMO_RUNTIME", DEFAULT_RUNTIME)
    if runtime != DEFAULT_RUNTIME:
        raise DemoError("only podman demo runtime is supported")
    image = _setting("HITCH_DEMO_IMAGE", DEFAULT_IMAGE)
    container_port = _int_setting("HITCH_DEMO_CONTAINER_PORT", DEFAULT_CONTAINER_PORT)
    host_port = _reserve_port()
    container_name = _container_name_for(session_id)
    failure: tuple[str, BaseException] | None = None

    with transaction.atomic():
        existing = SessionDemo.objects.select_for_update().filter(thread_id=session_id).first()
        if existing is not None:
            _remove_container(existing, ignore_missing=True)
            existing.status = SessionDemo.STATUS_STOPPED
            existing.container_id = ""
            existing.container_name = ""
            existing.save(
                update_fields=[
                    "status",
                    "container_id",
                    "container_name",
                    "updated_at",
                ]
            )

        cmd = [
            runtime,
            "run",
            "-d",
            "--name",
            container_name,
            "-p",
            f"{DEFAULT_HOST}:{host_port}:{container_port}",
            image,
            "sleep",
            "infinity",
        ]
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            message = _subprocess_error_message(exc)
            SessionDemo.objects.update_or_create(
                thread_id=session_id,
                defaults={
                    "host": DEFAULT_HOST,
                    "port": host_port,
                    "container_id": "",
                    "container_name": container_name,
                    "runtime": runtime,
                    "status": SessionDemo.STATUS_FAILED,
                    "last_error": message,
                },
            )
            failure = (message, exc)
        else:
            demo, _created = SessionDemo.objects.update_or_create(
                thread_id=session_id,
                defaults={
                    "host": DEFAULT_HOST,
                    "port": host_port,
                    "container_id": result.stdout.strip(),
                    "container_name": container_name,
                    "runtime": runtime,
                    "status": SessionDemo.STATUS_ACTIVE,
                    "last_error": "",
                },
            )

    if failure is not None:
        message, cause = failure
        raise DemoError(message) from cause
    return demo, container_port


def cleanup_demo_for_session(session_id: str) -> None:
    demo = SessionDemo.objects.filter(thread_id=session_id).first()
    if demo is None or demo.status != SessionDemo.STATUS_ACTIVE:
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
    demo.save(update_fields=["status", "updated_at"])


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
    except OSError as exc:
        return HttpResponse(f"demo target unavailable: {exc}", status=502)

    response_headers = _proxy_response_headers(
        upstream,
        path_prefix,
        upstream_netloc=f"{demo.host}:{demo.port}",
    )
    content_type = upstream.getheader("Content-Type", "")
    content_encoding = upstream.getheader("Content-Encoding", "").lower()
    if _should_rewrite_body(content_type, path_prefix) and content_encoding in {"", "identity"}:
        body_bytes = upstream.read()
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


def _int_setting(name: str, default: int) -> int:
    value = getattr(settings, name, default)
    return value if isinstance(value, int) and value > 0 else default


def _reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((DEFAULT_HOST, 0))
        return int(sock.getsockname()[1])


def _container_name_for(session_id: str) -> str:
    slug = slugify(session_id)[:48] or "session"
    return f"hitch-demo-{slug}-{uuid.uuid4().hex[:8]}"


def _remove_container(demo: SessionDemo, *, ignore_missing: bool = False) -> None:
    if not demo.container_id and not demo.container_name:
        return
    target = demo.container_id or demo.container_name
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
        headers[key] = value
    return headers


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
        encoding = match.group(1)
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
