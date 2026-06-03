from __future__ import annotations

import http.client
import json
import subprocess
import tempfile
import threading
from collections.abc import Iterable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast, override
from unittest.mock import MagicMock, call, patch

from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import connection
from django.http import StreamingHttpResponse
from django.test import RequestFactory, TestCase, override_settings
from django.urls import reverse

from hitch.main import demo
from hitch.main.management.commands import register_demo as register_demo_command
from hitch.main.models import CodexInstance, SessionDemo, SystemAgentRun, SystemWorkflow


def _inspect_stdout(
    *,
    token: str,
    session: str = "thread-1",
    name: str,
    host_port: int | None = None,
    host_ip: str = "127.0.0.1",
) -> str:
    item: dict[str, object] = {
        "Config": {
            "Labels": {
                "io.hitch.managed": "demo",
                "io.hitch.session": session,
                "io.hitch.demo_token": token,
                "io.hitch.container_name": name,
            }
        }
    }
    if host_port is not None:
        item["NetworkSettings"] = {
            "Ports": {"3000/tcp": [{"HostIp": host_ip, "HostPort": str(host_port)}]}
        }
    return json.dumps([item])


def _response_body(response: object) -> bytes:
    streaming_response = cast(StreamingHttpResponse, response)
    return b"".join(cast(Iterable[bytes], streaming_response.streaming_content))


class _DemoHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    seen: dict[str, object] = {}

    @override
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return

    def do_GET(self) -> None:  # noqa: N802
        self.close_connection = True
        headers = {key.lower(): value for key, value in self.headers.items()}
        type(self).seen = {
            "method": "GET",
            "path": self.path,
            "host": self.headers.get("Host", ""),
            "headers": headers,
            "cookie": self.headers.get("Cookie", ""),
            "authorization": self.headers.get("Authorization", ""),
        }
        if self.path.startswith("/connection-header"):
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Connection", "close, X-Debug")
            self.send_header("X-Debug", "secret")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/events"):
            body = b"data: one\n\ndata: two\n\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/redirect"):
            self.send_response(302)
            self.send_header("Location", "/next")
            self.send_header("Connection", "close")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path.startswith("/same-upstream-redirect"):
            server = cast(ThreadingHTTPServer, self.server)
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{server.server_port}/next")
            self.send_header("Connection", "close")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path.startswith("/external-redirect"):
            self.send_response(302)
            self.send_header("Location", "https://accounts.example.com/login")
            self.send_header("Connection", "close")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path.startswith("/module.js"):
            body = b'import "/dep.js"; fetch("/api/status"); import("/dynamic.js");'
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript; charset=utf-8")
            self.send_header("Connection", "close")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = b'<a href="/asset.css"><img src="/logo.png"></a>'
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Set-Cookie", "demo_session=bad")
        self.send_header("Connection", "close")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        self.close_connection = True
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        type(self).seen = {
            "method": "POST",
            "path": self.path,
            "headers": {key.lower(): value for key, value in self.headers.items()},
            "body": body,
        }
        response = b"posted"
        self.send_response(201)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Connection", "close")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


class DemoProxyTests(TestCase):
    @override
    def setUp(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _DemoHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.thread.join, 2)
        self.addCleanup(self.server.shutdown)
        SessionDemo.objects.create(
            thread_id="thread-1",
            host="127.0.0.1",
            port=int(self.server.server_port),
            status=SessionDemo.STATUS_ACTIVE,
        )

    def test_proxy_forwards_get_and_rewrites_html_paths(self) -> None:
        response = self.client.get(
            reverse(
                "session_demo_proxy",
                kwargs={"session_id": "thread-1", "path": "app/index.html"},
            )
            + "?x=1",
            HTTP_AUTHORIZATION="Bearer secret",
            HTTP_COOKIE="sessionid=secret",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(_DemoHandler.seen["path"], "/app/index.html?x=1")
        self.assertEqual(_DemoHandler.seen["host"], f"127.0.0.1:{self.server.server_port}")
        self.assertEqual(_DemoHandler.seen["authorization"], "")
        self.assertEqual(_DemoHandler.seen["cookie"], "")
        self.assertNotIn("Set-Cookie", response.headers)
        self.assertIn(b'href="/sessions/thread-1/demo/asset.css"', response.content)
        self.assertIn(b'src="/sessions/thread-1/demo/logo.png"', response.content)

    def test_proxy_root_forwards_to_upstream_root(self) -> None:
        response = self.client.get(
            reverse("session_demo_proxy_root", kwargs={"session_id": "thread-1"})
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(_DemoHandler.seen["path"], "/")

    def test_proxy_rewrites_javascript_root_urls_for_path_fallback(self) -> None:
        response = self.client.get(
            reverse(
                "session_demo_proxy",
                kwargs={"session_id": "thread-1", "path": "module.js"},
            )
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'import "/sessions/thread-1/demo/dep.js"', response.content)
        self.assertIn(b'fetch("/sessions/thread-1/demo/api/status")', response.content)
        self.assertIn(b'import("/sessions/thread-1/demo/dynamic.js")', response.content)

    def test_proxy_forwards_post_body(self) -> None:
        response = self.client.post(
            reverse(
                "session_demo_proxy",
                kwargs={"session_id": "thread-1", "path": "api/items"},
            ),
            data=b'{"ok":true}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(_response_body(response), b"posted")
        self.assertEqual(_DemoHandler.seen["method"], "POST")
        self.assertEqual(_DemoHandler.seen["path"], "/api/items")
        self.assertEqual(_DemoHandler.seen["body"], b'{"ok":true}')
        headers = cast(dict[str, str], _DemoHandler.seen["headers"])
        self.assertEqual(headers["content-length"], "11")

    def test_proxy_strips_response_connection_token_headers(self) -> None:
        response = self.client.get(
            reverse(
                "session_demo_proxy",
                kwargs={"session_id": "thread-1", "path": "connection-header"},
            )
        )

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("Connection", response.headers)
        self.assertNotIn("X-Debug", response.headers)

    def test_proxy_rewrites_redirect_location(self) -> None:
        response = self.client.get(
            reverse(
                "session_demo_proxy",
                kwargs={"session_id": "thread-1", "path": "redirect"},
            )
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response["Location"], "/sessions/thread-1/demo/next")

    def test_proxy_rewrites_same_upstream_absolute_redirect_location(self) -> None:
        response = self.client.get(
            reverse(
                "session_demo_proxy",
                kwargs={"session_id": "thread-1", "path": "same-upstream-redirect"},
            )
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response["Location"], "/sessions/thread-1/demo/next")

    def test_proxy_preserves_external_absolute_redirect_location(self) -> None:
        response = self.client.get(
            reverse(
                "session_demo_proxy",
                kwargs={"session_id": "thread-1", "path": "external-redirect"},
            )
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response["Location"], "https://accounts.example.com/login")

    def test_proxy_request_headers_filter_browser_and_proxy_control_headers(self) -> None:
        request = RequestFactory().get(
            "/sessions/thread-1/demo/app",
            headers={
                "Accept-Encoding": "gzip, br",
                "Accept-Language": "en-US",
                "Authorization": "Bearer secret",
                "Connection": "keep-alive, X-Client-Only",
                "Content-Length": "",
                "Cookie": "sessionid=secret",
                "Expect": "100-continue",
                "Forwarded": "for=198.51.100.1",
                "Host": "hitch.example.test",
                "If-Modified-Since": "Wed, 21 Oct 2020 07:28:00 GMT",
                "If-None-Match": '"stale-etag"',
                "If-Range": '"range-validator"',
                "Proxy-Connection": "keep-alive",
                "Range": "bytes=0-5",
                "X-Client-Only": "drop me",
                "X-CSRFToken": "secret",
                "X-Forwarded-For": "198.51.100.1",
                "X-Forwarded-Host": "evil.example.test",
                "X-Forwarded-Prefix": "/evil",
                "X-Forwarded-Port": "443",
                "X-Forwarded-Proto": "https",
            },
            REMOTE_ADDR="203.0.113.9",
        )
        target = SessionDemo(host="127.0.0.1", port=12345)

        headers = demo._proxy_request_headers(
            request,
            target,
            path_prefix="/sessions/thread-1/demo/",
        )
        normalized = {key.lower(): value for key, value in headers.items()}

        self.assertEqual(normalized["host"], "127.0.0.1:12345")
        self.assertEqual(normalized["accept-encoding"], "identity")
        self.assertEqual(normalized["x-forwarded-host"], "hitch.example.test")
        self.assertEqual(normalized["x-forwarded-proto"], "http")
        self.assertEqual(normalized["x-forwarded-for"], "203.0.113.9")
        self.assertEqual(normalized["x-forwarded-prefix"], "/sessions/thread-1/demo")
        self.assertEqual(normalized["accept-language"], "en-US")
        self.assertEqual(normalized["range"], "bytes=0-5")
        # If-Range stays paired with Range so the container can decide between a
        # 206 and a full 200 against the rebuilt resource.
        self.assertEqual(normalized["if-range"], '"range-validator"')
        for blocked in (
            "authorization",
            "connection",
            "content-length",
            "cookie",
            "expect",
            "forwarded",
            # Conditional validators must not reach the container, or a rebuilt
            # demo could answer 304 and the browser would reuse a stale asset.
            "if-modified-since",
            "if-none-match",
            "proxy-connection",
            "x-client-only",
            "x-csrftoken",
            "x-forwarded-port",
        ):
            self.assertNotIn(blocked, normalized)

    def test_proxy_keeps_write_preconditions_on_unsafe_methods(self) -> None:
        # On unsafe methods the conditional headers are optimistic-concurrency
        # preconditions, not cache validators (e.g. If-None-Match: * for
        # create-if-absent), so they must reach the container intact.
        request = RequestFactory().put(
            "/sessions/thread-1/demo/resource",
            headers={
                "Host": "hitch.example.test",
                "If-Match": '"v1"',
                "If-None-Match": "*",
                "If-Unmodified-Since": "Wed, 21 Oct 2020 07:28:00 GMT",
            },
        )
        target = SessionDemo(host="127.0.0.1", port=12345)

        headers = demo._proxy_request_headers(
            request,
            target,
            path_prefix="/sessions/thread-1/demo/",
        )
        normalized = {key.lower(): value for key, value in headers.items()}

        self.assertEqual(normalized["if-match"], '"v1"')
        self.assertEqual(normalized["if-none-match"], "*")
        self.assertEqual(
            normalized["if-unmodified-since"], "Wed, 21 Oct 2020 07:28:00 GMT"
        )

    def test_proxy_drops_response_headers_with_embedded_crlf(self) -> None:
        # Obsolete header line folding / header injection from a misbehaving demo
        # container arrives with embedded CR/LF; copying it onto a Django response
        # would raise BadHeaderError (500 + leaked upstream socket). It must be
        # dropped while well-formed headers pass through.
        upstream = MagicMock(spec=http.client.HTTPResponse)
        upstream.getheaders.return_value = [
            ("X-Good", "fine"),
            ("X-Folded", "bar\r\n\tInjected: 1"),
            ("X-Bare-CR", "a\rb"),
        ]

        headers = demo._proxy_response_headers(
            upstream,
            "/sessions/thread-1/demo/",
            upstream_netloc="127.0.0.1:12345",
        )

        self.assertEqual(headers.get("X-Good"), "fine")
        self.assertNotIn("X-Folded", headers)
        self.assertNotIn("X-Bare-CR", headers)

    def test_proxy_replaces_cache_freshness_headers_but_keeps_validators(self) -> None:
        # The demo proxy URL is stable across rebuilds, so the container's
        # cache-freshness directives must be dropped and replaced with no-store.
        # ETag/Last-Modified are representation validators an app may need, and
        # no-store plus stripped request revalidators stop them from triggering a
        # stale 304, so they are preserved.
        upstream = MagicMock(spec=http.client.HTTPResponse)
        upstream.getheaders.return_value = [
            ("Content-Type", "text/html"),
            ("Cache-Control", "max-age=31536000, immutable"),
            ("ETag", '"abc123"'),
            ("Expires", "Wed, 21 Oct 2099 07:28:00 GMT"),
            ("Last-Modified", "Wed, 21 Oct 2020 07:28:00 GMT"),
            ("Pragma", "cache"),
        ]

        headers = demo._proxy_response_headers(
            upstream,
            "/sessions/thread-1/demo/",
            upstream_netloc="127.0.0.1:12345",
        )

        self.assertEqual(headers.get("Cache-Control"), "no-store")
        for absent in ("Expires", "Pragma"):
            self.assertNotIn(absent, headers)
        self.assertEqual(headers.get("ETag"), '"abc123"')
        self.assertEqual(headers.get("Last-Modified"), "Wed, 21 Oct 2020 07:28:00 GMT")
        self.assertEqual(headers.get("Content-Type"), "text/html")

    def test_rewrite_body_preserves_protocol_relative_urls_and_rewrites_srcset(self) -> None:
        body = (
            b'<link href="//cdn.example.com/app.css">'
            b'<img srcset="/small.png 1x, //cdn.example.com/big.png 2x" '
            b'poster="/poster.png">'
            b"<style>.hero{background:url(//cdn.example.com/bg.png)}</style>"
            b'<script>fetch("//api.example.com/status"); '
            b'import x from "//cdn.example.com/mod.js";</script>'
        )

        rewritten = demo._rewrite_body(
            body,
            "text/html; charset=utf-8",
            "/sessions/thread-1/demo/",
        )

        self.assertIn(b'href="//cdn.example.com/app.css"', rewritten)
        self.assertIn(
            b'srcset="/sessions/thread-1/demo/small.png 1x, //cdn.example.com/big.png 2x"',
            rewritten,
        )
        self.assertIn(b'poster="/sessions/thread-1/demo/poster.png"', rewritten)
        self.assertIn(b"url(//cdn.example.com/bg.png)", rewritten)
        self.assertIn(b'fetch("//api.example.com/status")', rewritten)
        self.assertIn(b'import x from "//cdn.example.com/mod.js"', rewritten)

    def test_rewrite_body_handles_quoted_charset(self) -> None:
        # A quoted charset token must not break decoding and skip rewriting.
        rewritten = demo._rewrite_body(
            b'<a href="/page">link</a>',
            'text/html; charset="utf-8"',
            "/sessions/thread-1/demo/",
        )
        self.assertIn(b'href="/sessions/thread-1/demo/page"', rewritten)

    def test_proxy_streams_sse(self) -> None:
        response = self.client.get(
            reverse(
                "session_demo_proxy",
                kwargs={"session_id": "thread-1", "path": "events"},
            )
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/event-stream")
        self.assertEqual(_response_body(response), b"data: one\n\ndata: two\n\n")

    def test_host_based_proxy_uses_root_path_without_html_rewrite(self) -> None:
        response = self.client.get(
            "/asset",
            headers={"host": "thread-1.demo.localhost"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'href="/asset.css"', _response_body(response))

    def test_proxy_without_active_demo_returns_404(self) -> None:
        SessionDemo.objects.all().delete()

        response = self.client.get(
            reverse(
                "session_demo_proxy",
                kwargs={"session_id": "thread-1", "path": "asset"},
            )
        )

        self.assertEqual(response.status_code, 404)

    @patch("hitch.main.demo.http.client.HTTPConnection")
    def test_proxy_reports_unavailable_demo_target(self, mock_connection: MagicMock) -> None:
        # A socket error on request() and a malformed status line on
        # getresponse() (http.client.HTTPException, not an OSError) must both
        # surface as a 502 and close the connection so its socket is not leaked.
        cases = (
            ("request", OSError("refused"), b"refused"),
            ("getresponse", http.client.BadStatusLine("garbage"), b"demo target unavailable"),
        )
        for failing_call, exc, expected in cases:
            with self.subTest(failing_call=failing_call):
                conn = mock_connection.return_value
                conn.reset_mock()
                conn.request.side_effect = None
                conn.getresponse.side_effect = None
                getattr(conn, failing_call).side_effect = exc
                request = RequestFactory().get("/sessions/thread-1/demo/asset")

                response = demo.proxy_demo_request(
                    request,
                    "thread-1",
                    "asset",
                    path_prefix="/sessions/thread-1/demo/",
                )

                self.assertEqual(response.status_code, 502)
                self.assertIn(expected, response.content)
                conn.close.assert_called_once()

    @patch("hitch.main.demo.http.client.HTTPConnection")
    def test_proxy_closes_connection_when_body_read_fails(self, mock_connection: MagicMock) -> None:
        # Both a socket drop (OSError) and an early body close such as a short
        # Content-Length (http.client.IncompleteRead, an HTTPException -- not an
        # OSError) must close the connection and surface as a 502.
        for exc in (
            ConnectionResetError("peer reset"),
            http.client.IncompleteRead(b"partial"),
        ):
            with self.subTest(exc=type(exc).__name__):
                conn = mock_connection.return_value
                conn.reset_mock()
                upstream = conn.getresponse.return_value
                upstream.getheader.side_effect = lambda name, default="": {
                    "Content-Type": "text/html"
                }.get(name, default)
                upstream.status = 200
                upstream.read.side_effect = exc
                request = RequestFactory().get("/sessions/thread-1/demo/index.html")

                response = demo.proxy_demo_request(
                    request,
                    "thread-1",
                    "index.html",
                    path_prefix="/sessions/thread-1/demo/",
                )

                self.assertEqual(response.status_code, 502)
                self.assertIn(b"demo target unavailable", response.content)
                conn.close.assert_called_once()

    @patch("hitch.main.demo.http.client.HTTPConnection")
    def test_streaming_body_tolerates_read_timeouts(
        self, mock_connection: MagicMock
    ) -> None:
        # A read timeout on an idle-but-open streaming/SSE response must not
        # abort the response after headers are sent; _stream_upstream retries the
        # read instead. This holds even when getresponse() has detached
        # connection.sock (the will_close case), so the guard cannot depend on it.
        conn = mock_connection.return_value
        conn.sock = None
        upstream = conn.getresponse.return_value
        upstream.getheader.side_effect = lambda name, default="": {
            "Content-Type": "application/octet-stream"
        }.get(name, default)
        upstream.status = 200
        upstream.read.side_effect = [TimeoutError("idle"), b"chunk", b""]
        request = RequestFactory().get("/sessions/thread-1/demo/download.bin")

        response = demo.proxy_demo_request(
            request,
            "thread-1",
            "download.bin",
            path_prefix="/sessions/thread-1/demo/",
        )

        self.assertEqual(response.status_code, 200)
        # The body flows through after the idle interval is retried.
        self.assertEqual(_response_body(response), b"chunk")

    def test_localhost_demo_url_uses_isolated_demo_host_for_loopback(self) -> None:
        request = RequestFactory().get("/", headers={"host": "127.0.0.1:8000"})

        url = demo.demo_url_for_request(request, "thread-1")

        self.assertEqual(url, "http://thread-1.demo.localhost:8000/")

    def test_unsafe_session_id_uses_path_demo_url_for_loopback(self) -> None:
        request = RequestFactory().get("/", headers={"host": "127.0.0.1:8000"})

        self.assertEqual(
            demo.demo_url_for_request(request, "thread_1"),
            "http://127.0.0.1:8000/sessions/thread_1/demo/",
        )
        self.assertIsNone(demo.session_id_from_demo_host("thread_1.demo.localhost"))

    def test_start_demo_prompt_uses_management_command_registration(self) -> None:
        request = RequestFactory().get(
            "/",
            SERVER_PORT="8000",
            headers={"host": "testserver"},
        )
        session_demo = SessionDemo(thread_id="thread-1", registration_token="-token")

        prompt = demo.start_demo_prompt_for(
            request=request,
            session_id="thread-1",
            cwd="/repo",
            demo=session_demo,
        )

        self.assertIn(
            'Registration command prefix: $HITCH_MANAGE_COMMAND run --project '
            '"$HITCH_PROJECT_DIR" "$HITCH_MANAGE_PY" register_demo '
            "--session-id=thread-1 --token=-token",
            prompt,
        )
        self.assertIn(
            "$HITCH_MANAGE_COMMAND run --project "
            '"$HITCH_PROJECT_DIR" "$HITCH_MANAGE_PY" register_demo '
            "--session-id=thread-1 --token=-token --status preparing "
            "--container-name CONTAINER_NAME --logs 'starting demo container'",
            prompt,
        )
        self.assertIn(
            "replace CONTAINER_NAME, HOST_PORT, CONTAINER_ID, and logs",
            prompt,
        )
        self.assertIn("--status active", prompt)
        self.assertIn("--status failed", prompt)
        self.assertNotIn("curl -fsS", prompt)


class DemoRegistrationTests(TestCase):
    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    def test_request_demo_start_creates_registration_token(self, _cleanup: MagicMock) -> None:
        session_demo = demo.request_demo_start("thread-1")

        self.assertEqual(session_demo.status, SessionDemo.STATUS_REQUESTED)
        self.assertEqual(session_demo.generation, 1)
        self.assertTrue(session_demo.registration_token)
        self.assertEqual(session_demo.host, "127.0.0.1")
        self.assertEqual(session_demo.port, 3000)

    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    def test_request_demo_start_rejects_in_progress_demo(self, mock_cleanup: MagicMock) -> None:
        SessionDemo.objects.create(
            thread_id="thread-1",
            host="127.0.0.1",
            port=3000,
            runtime="podman",
            status=SessionDemo.STATUS_REQUESTED,
            registration_token="token",
            generation=1,
        )

        with self.assertRaisesRegex(demo.DemoAlreadyRunningError, "already running"):
            demo.request_demo_start("thread-1")

        mock_cleanup.assert_not_called()

    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    @patch("hitch.main.demo.subprocess.run")
    def test_request_demo_start_supersedes_existing_container(
        self, mock_run: MagicMock, _cleanup: MagicMock
    ) -> None:
        mock_run.side_effect = [
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=_inspect_stdout(token="old-token", name="hitch-demo-thread-1-old"),
                stderr="",
            ),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
        ]
        existing = SessionDemo.objects.create(
            thread_id="thread-1",
            host="127.0.0.1",
            port=4567,
            container_id="old-container",
            container_name="hitch-demo-thread-1-old",
            runtime="podman",
            status=SessionDemo.STATUS_ACTIVE,
            registration_token="old-token",
            generation=1,
        )

        session_demo = demo.request_demo_start("thread-1")

        self.assertEqual(mock_run.call_args_list[1], call(
            ["podman", "rm", "-f", "old-container"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ))
        self.assertEqual(session_demo.pk, existing.pk)
        self.assertEqual(session_demo.status, SessionDemo.STATUS_REQUESTED)
        self.assertEqual(session_demo.generation, 2)
        self.assertNotEqual(session_demo.registration_token, "old-token")
        self.assertEqual(session_demo.container_id, "")

    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    @patch("hitch.main.demo._remove_container")
    def test_request_demo_start_cleans_existing_container_outside_transaction(
        self, mock_remove: MagicMock, _cleanup: MagicMock
    ) -> None:
        SessionDemo.objects.create(
            thread_id="thread-1",
            host="127.0.0.1",
            port=4567,
            container_id="old-container",
            container_name="hitch-demo-thread-1-old",
            runtime="podman",
            status=SessionDemo.STATUS_ACTIVE,
            registration_token="old-token",
            generation=1,
        )
        baseline_savepoints = list(connection.savepoint_ids)

        def assert_unlocked(_demo: SessionDemo, *, ignore_missing: bool) -> None:
            self.assertEqual(connection.savepoint_ids, baseline_savepoints)
            self.assertTrue(ignore_missing)

        mock_remove.side_effect = assert_unlocked

        session_demo = demo.request_demo_start("thread-1")

        mock_remove.assert_called_once()
        self.assertEqual(session_demo.status, SessionDemo.STATUS_REQUESTED)

    def test_register_demo_management_command_marks_preparing(self) -> None:
        SessionDemo.objects.create(
            thread_id="thread-1",
            host="127.0.0.1",
            port=3000,
            runtime="podman",
            status=SessionDemo.STATUS_REQUESTED,
            registration_token="-token",
            generation=1,
        )

        output = call_command(
            "register_demo",
            "--session-id",
            "thread-1",
            "--token=-token",
            "--status",
            "preparing",
            "--container-name",
            "hitch-demo-thread-1-cli",
            "--logs",
            "starting",
            "--json",
        )

        self.assertEqual(
            json.loads(output),
            {
                "demo_url": "/sessions/thread-1/demo/",
                "status": SessionDemo.STATUS_PREPARING,
            },
        )
        session_demo = SessionDemo.objects.get(thread_id="thread-1")
        self.assertEqual(session_demo.status, SessionDemo.STATUS_PREPARING)
        self.assertEqual(session_demo.container_name, "hitch-demo-thread-1-cli")
        self.assertEqual(session_demo.logs, "starting")

    def test_register_demo_management_command_reads_logs_file(self) -> None:
        SessionDemo.objects.create(
            thread_id="thread-1",
            host="127.0.0.1",
            port=3000,
            runtime="podman",
            status=SessionDemo.STATUS_REQUESTED,
            registration_token="token",
            generation=1,
        )
        with tempfile.TemporaryDirectory() as raw:
            logs_file = Path(raw) / "logs.txt"
            logs_file.write_text("from file", encoding="utf-8")

            output = call_command(
                "register_demo",
                "--session-id=thread-1",
                "--token=token",
                "--status=preparing",
                "--container-name=hitch-demo-thread-1-cli",
                "--port=4567",
                "--logs-file",
                str(logs_file),
            )

        self.assertEqual(output, "Registered demo as preparing\nDemo: /sessions/thread-1/demo/")
        session_demo = SessionDemo.objects.get(thread_id="thread-1")
        self.assertEqual(session_demo.logs, "from file")
        self.assertEqual(session_demo.port, 4567)

    def test_register_demo_management_command_bounds_logs_file(self) -> None:
        SessionDemo.objects.create(
            thread_id="thread-1",
            host="127.0.0.1",
            port=3000,
            runtime="podman",
            status=SessionDemo.STATUS_REQUESTED,
            registration_token="token",
            generation=1,
        )
        with tempfile.TemporaryDirectory() as raw:
            logs_file = Path(raw) / "logs.txt"
            logs_file.write_text(
                "prefix" + ("x" * (demo.MAX_LOG_CHARS + 20)),
                encoding="utf-8",
            )

            call_command(
                "register_demo",
                "--session-id=thread-1",
                "--token=token",
                "--status=preparing",
                "--container-name=hitch-demo-thread-1-cli",
                "--logs-file",
                str(logs_file),
            )

        session_demo = SessionDemo.objects.get(thread_id="thread-1")
        self.assertEqual(session_demo.logs, "x" * demo.MAX_LOG_CHARS)

    def test_register_demo_management_command_rejects_conflicting_log_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            logs_file = Path(raw) / "logs.txt"
            logs_file.write_text("from file", encoding="utf-8")

            with self.assertRaisesRegex(CommandError, "use either --logs or --logs-file"):
                call_command(
                    "register_demo",
                    "--session-id=thread-1",
                    "--token=token",
                    "--status=preparing",
                    "--container-name=hitch-demo-thread-1-cli",
                    "--logs",
                    "inline",
                    "--logs-file",
                    str(logs_file),
                )

    def test_register_demo_management_command_reports_unreadable_logs_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            missing_logs = Path(raw) / "missing.log"

            with self.assertRaisesRegex(CommandError, "failed to read --logs-file"):
                call_command(
                    "register_demo",
                    "--session-id=thread-1",
                    "--token=token",
                    "--status=preparing",
                    "--container-name=hitch-demo-thread-1-cli",
                    "--logs-file",
                    str(missing_logs),
                )

    def test_register_demo_management_command_reports_non_utf8_logs_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            logs_file = Path(raw) / "logs.bin"
            logs_file.write_bytes(b"\xff")

            with self.assertRaisesRegex(CommandError, "file is not valid UTF-8"):
                call_command(
                    "register_demo",
                    "--session-id=thread-1",
                    "--token=token",
                    "--status=preparing",
                    "--container-name=hitch-demo-thread-1-cli",
                    "--logs-file",
                    str(logs_file),
                )

    def test_register_demo_management_command_decodes_empty_file_tail(self) -> None:
        self.assertEqual(
            register_demo_command._decode_bounded_utf8(b"", truncated=False),
            "",
        )

    def test_register_demo_management_command_decodes_split_utf8_tail(self) -> None:
        self.assertEqual(
            register_demo_command._decode_bounded_utf8(b"\xa9tail", truncated=True),
            "tail",
        )

    def test_register_demo_management_command_reports_registration_errors(self) -> None:
        SessionDemo.objects.create(
            thread_id="thread-1",
            host="127.0.0.1",
            port=3000,
            runtime="podman",
            status=SessionDemo.STATUS_REQUESTED,
            registration_token="token",
            generation=1,
        )

        with self.assertRaisesRegex(CommandError, "invalid demo registration token"):
            call_command(
                "register_demo",
                "--session-id=thread-1",
                "--token=wrong",
                "--status=preparing",
                "--container-name=hitch-demo-thread-1-cli",
            )

    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    @patch("hitch.main.demo.subprocess.run")
    def test_request_demo_start_keeps_container_target_on_inspect_failure(
        self, mock_run: MagicMock, mock_cleanup: MagicMock
    ) -> None:
        mock_run.side_effect = subprocess.CalledProcessError(
            125, ["podman", "inspect"], stderr="permission denied"
        )
        SessionDemo.objects.create(
            thread_id="thread-1",
            host="127.0.0.1",
            port=4567,
            container_id="old-container",
            container_name="hitch-demo-thread-1-old",
            runtime="podman",
            status=SessionDemo.STATUS_ACTIVE,
            registration_token="old-token",
            generation=1,
        )

        with self.assertRaisesRegex(demo.DemoError, "permission denied"):
            demo.request_demo_start("thread-1")

        session_demo = SessionDemo.objects.get(thread_id="thread-1")
        self.assertEqual(session_demo.status, SessionDemo.STATUS_ACTIVE)
        self.assertEqual(session_demo.container_id, "old-container")
        self.assertEqual(session_demo.container_name, "hitch-demo-thread-1-old")
        self.assertEqual(session_demo.registration_token, "old-token")
        mock_cleanup.assert_not_called()
        mock_run.assert_called_once_with(
            ["podman", "inspect", "old-container"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

    @patch("hitch.main.demo.logger")
    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    @patch("hitch.main.demo.subprocess.run")
    def test_request_demo_start_recovers_from_prior_label_mismatch(
        self, mock_run: MagicMock, mock_cleanup: MagicMock, _logger: MagicMock
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='[{"Config":{"Labels":{"io.hitch.managed":"other"}}}]',
            stderr="",
        )
        SessionDemo.objects.create(
            thread_id="thread-1",
            host="127.0.0.1",
            port=4567,
            container_id="old-container",
            container_name="hitch-demo-thread-1-old",
            runtime="podman",
            status=SessionDemo.STATUS_FAILED,
            registration_token="old-token",
            generation=1,
        )

        session_demo = demo.request_demo_start("thread-1")

        self.assertEqual(session_demo.status, SessionDemo.STATUS_REQUESTED)
        self.assertEqual(session_demo.generation, 2)
        self.assertNotEqual(session_demo.registration_token, "old-token")
        self.assertEqual(session_demo.container_id, "")
        self.assertEqual(session_demo.container_name, "")
        mock_cleanup.assert_called_once_with(
            protected_targets={
                ("id", "old-container"),
                ("name", "hitch-demo-thread-1-old"),
            }
        )
        mock_run.assert_called_once_with(
            ["podman", "inspect", "old-container"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

    @patch("hitch.main.demo.logger")
    @patch("hitch.main.demo.subprocess.run")
    def test_request_demo_start_protects_unverified_container_from_sweep(
        self, mock_run: MagicMock, _logger: MagicMock
    ) -> None:
        mock_run.side_effect = [
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=_inspect_stdout(
                    token="old-token",
                    session="other-thread",
                    name="hitch-demo-thread-1-old",
                ),
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=(
                    '[{"ID":"old-container","Names":["hitch-demo-thread-1-old"],'
                    '"Labels":{"io.hitch.demo_token":"old-token"}}]'
                ),
                stderr="",
            ),
        ]
        SessionDemo.objects.create(
            thread_id="thread-1",
            host="127.0.0.1",
            port=4567,
            container_id="old-container",
            container_name="hitch-demo-thread-1-old",
            runtime="podman",
            status=SessionDemo.STATUS_FAILED,
            registration_token="old-token",
            generation=1,
        )

        session_demo = demo.request_demo_start("thread-1")

        self.assertEqual(session_demo.status, SessionDemo.STATUS_REQUESTED)
        self.assertEqual(mock_run.call_count, 2)
        self.assertNotIn("rm", mock_run.call_args_list[-1].args[0])

    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    @patch("hitch.main.demo.subprocess.run")
    def test_request_demo_start_removes_legacy_unlabeled_container(
        self, mock_run: MagicMock, _cleanup: MagicMock
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        SessionDemo.objects.create(
            thread_id="thread-1",
            host="127.0.0.1",
            port=4567,
            container_name="legacy-demo-container",
            runtime="podman",
            status=SessionDemo.STATUS_ACTIVE,
            registration_token="",
            generation=1,
        )

        session_demo = demo.request_demo_start("thread-1")

        mock_run.assert_called_once_with(
            ["podman", "rm", "-f", "legacy-demo-container"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(session_demo.status, SessionDemo.STATUS_REQUESTED)
        self.assertEqual(session_demo.container_name, "")

    @patch("hitch.main.demo.logger")
    @patch("hitch.main.demo.subprocess.run")
    def test_cleanup_demo_for_session_keeps_unverified_registered_container(
        self, mock_run: MagicMock, _logger: MagicMock
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='[{"Config":{"Labels":{"io.hitch.managed":"other"}}}]',
            stderr="",
        )
        SessionDemo.objects.create(
            thread_id="thread-1",
            host="127.0.0.1",
            port=4567,
            container_name="hitch-demo-thread-1-old",
            runtime="podman",
            status=SessionDemo.STATUS_ACTIVE,
            registration_token="token",
            generation=1,
        )

        demo.cleanup_demo_for_session("thread-1")

        session_demo = SessionDemo.objects.get(thread_id="thread-1")
        self.assertEqual(session_demo.status, SessionDemo.STATUS_FAILED)
        self.assertEqual(session_demo.container_name, "hitch-demo-thread-1-old")
        self.assertIn("labels did not match", session_demo.last_error)
        mock_run.assert_called_once_with(
            ["podman", "inspect", "hitch-demo-thread-1-old"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

    @patch("hitch.main.demo.subprocess.run")
    def test_cleanup_demo_for_session_uses_verified_container_id_with_stale_name(
        self, mock_run: MagicMock
    ) -> None:
        mock_run.side_effect = [
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=_inspect_stdout(token="token", name="new-container-name"),
                stderr="",
            ),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="[]", stderr=""),
        ]
        SessionDemo.objects.create(
            thread_id="thread-1",
            host="127.0.0.1",
            port=4567,
            container_id="container-1",
            container_name="stale-container-name",
            runtime="podman",
            status=SessionDemo.STATUS_ACTIVE,
            registration_token="token",
            generation=1,
        )

        demo.cleanup_demo_for_session("thread-1")

        session_demo = SessionDemo.objects.get(thread_id="thread-1")
        self.assertEqual(session_demo.status, SessionDemo.STATUS_STOPPED)
        self.assertEqual(session_demo.container_id, "")
        self.assertEqual(session_demo.container_name, "")
        self.assertEqual(mock_run.call_args_list[1], call(
            ["podman", "rm", "-f", "container-1"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ))

    @patch("hitch.main.demo.logger")
    @patch("hitch.main.demo.subprocess.run")
    def test_cleanup_demo_for_session_keeps_target_on_inspect_failure(
        self, mock_run: MagicMock, _logger: MagicMock
    ) -> None:
        mock_run.side_effect = subprocess.CalledProcessError(
            125, ["podman", "inspect"], stderr="permission denied"
        )
        SessionDemo.objects.create(
            thread_id="thread-1",
            host="127.0.0.1",
            port=4567,
            container_id="old-container",
            container_name="hitch-demo-thread-1-old",
            runtime="podman",
            status=SessionDemo.STATUS_ACTIVE,
            registration_token="token",
            generation=1,
        )

        demo.cleanup_demo_for_session("thread-1")

        session_demo = SessionDemo.objects.get(thread_id="thread-1")
        self.assertEqual(session_demo.status, SessionDemo.STATUS_FAILED)
        self.assertEqual(session_demo.container_id, "old-container")
        self.assertEqual(session_demo.container_name, "hitch-demo-thread-1-old")
        self.assertIn("permission denied", session_demo.last_error)
        mock_run.assert_called_once_with(
            ["podman", "inspect", "old-container"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    def test_register_preparing_protects_container_name(self, _cleanup: MagicMock) -> None:
        session_demo = demo.request_demo_start("thread-1")

        updated = demo.register_demo_container(
            "thread-1",
            {
                "token": session_demo.registration_token,
                "status": "preparing",
                "container_name": "hitch-demo-thread-1-abcd",
                "logs": "building image",
            },
        )

        self.assertEqual(updated.status, SessionDemo.STATUS_PREPARING)
        self.assertEqual(updated.container_name, "hitch-demo-thread-1-abcd")
        self.assertEqual(updated.logs, "building image")

    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    @patch("hitch.main.demo.subprocess.run")
    def test_register_active_enables_proxy_target(
        self, mock_run: MagicMock, _cleanup: MagicMock
    ) -> None:
        session_demo = demo.request_demo_start("thread-1")
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=_inspect_stdout(
                token=session_demo.registration_token,
                name="hitch-demo-thread-1-abcd",
                host_port=45678,
            ),
            stderr="",
        )

        updated = demo.register_demo_container(
            "thread-1",
            {
                "token": session_demo.registration_token,
                "status": "active",
                "container_name": "hitch-demo-thread-1-abcd",
                "container_id": "container123",
                "host": "localhost",
                "port": 45678,
                "logs": "ready",
            },
        )

        self.assertEqual(updated.status, SessionDemo.STATUS_ACTIVE)
        self.assertEqual(updated.host, "127.0.0.1")
        self.assertEqual(updated.port, 45678)
        self.assertEqual(updated.container_id, "container123")
        self.assertEqual(demo.active_demo_for("thread-1"), updated)
        mock_run.assert_called_once_with(
            ["podman", "inspect", "container123"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    @patch("hitch.main.demo._verify_registered_container_labels")
    def test_register_active_verifies_container_outside_transaction(
        self, mock_verify: MagicMock, _cleanup: MagicMock
    ) -> None:
        session_demo = demo.request_demo_start("thread-1")
        baseline_savepoints = list(connection.savepoint_ids)

        def assert_unlocked(**kwargs: object) -> None:
            self.assertEqual(connection.savepoint_ids, baseline_savepoints)
            self.assertEqual(kwargs["target"], "container123")
            self.assertEqual(kwargs["thread_id"], "thread-1")
            self.assertEqual(kwargs["token"], session_demo.registration_token)
            self.assertEqual(kwargs["container_name"], "hitch-demo-thread-1-abcd")
            self.assertEqual(kwargs["port"], 45678)

        mock_verify.side_effect = assert_unlocked

        updated = demo.register_demo_container(
            "thread-1",
            {
                "token": session_demo.registration_token,
                "status": "active",
                "container_name": "hitch-demo-thread-1-abcd",
                "container_id": "container123",
                "host": "127.0.0.1",
                "port": 45678,
                "logs": "ready",
            },
        )

        self.assertEqual(updated.status, SessionDemo.STATUS_ACTIVE)
        mock_verify.assert_called_once()

    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    @patch("hitch.main.demo._verify_registered_container_labels")
    def test_register_active_rechecks_snapshot_after_verify(
        self, mock_verify: MagicMock, _cleanup: MagicMock
    ) -> None:
        session_demo = demo.request_demo_start("thread-1")

        def replace_demo(**_kwargs: object) -> None:
            SessionDemo.objects.filter(thread_id="thread-1").update(
                status=SessionDemo.STATUS_REQUESTED,
                generation=2,
                registration_token=session_demo.registration_token,
                container_id="replacement-id",
                container_name="hitch-demo-thread-1-replacement",
                last_error="",
                logs="replacement logs",
            )

        mock_verify.side_effect = replace_demo

        with self.assertRaisesRegex(demo.DemoError, "demo registration changed"):
            demo.register_demo_container(
                "thread-1",
                {
                    "token": session_demo.registration_token,
                    "status": "active",
                    "container_name": "hitch-demo-thread-1-abcd",
                    "container_id": "container123",
                    "host": "127.0.0.1",
                    "port": 45678,
                    "logs": "ready",
                },
            )

        replacement = SessionDemo.objects.get(thread_id="thread-1")
        self.assertEqual(replacement.status, SessionDemo.STATUS_REQUESTED)
        self.assertEqual(replacement.generation, 2)
        self.assertEqual(replacement.registration_token, session_demo.registration_token)
        self.assertEqual(replacement.container_id, "replacement-id")
        self.assertEqual(replacement.container_name, "hitch-demo-thread-1-replacement")
        self.assertEqual(replacement.logs, "replacement logs")

    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    @patch("hitch.main.demo._remove_container")
    @patch("hitch.main.demo._verify_registered_container_labels")
    def test_register_failed_does_not_remove_after_active_interleaving(
        self,
        _mock_verify: MagicMock,
        mock_remove: MagicMock,
        _mock_cleanup: MagicMock,
    ) -> None:
        session_demo = demo.request_demo_start("thread-1")
        demo.register_demo_container(
            "thread-1",
            {
                "token": session_demo.registration_token,
                "status": "preparing",
                "container_name": "hitch-demo-thread-1-abcd",
            },
        )
        apply_failed_registration = demo._apply_failed_registration

        def active_wins(demo_to_fail: SessionDemo, payload: dict[str, Any]) -> None:
            apply_failed_registration(demo_to_fail, payload)
            demo.register_demo_container(
                "thread-1",
                {
                    "token": session_demo.registration_token,
                    "status": "active",
                    "container_name": "hitch-demo-thread-1-abcd",
                    "container_id": "container123",
                    "host": "127.0.0.1",
                    "port": 45678,
                    "logs": "ready",
                },
            )

        with (
            patch("hitch.main.demo._apply_failed_registration", side_effect=active_wins),
            self.assertRaisesRegex(demo.DemoError, "already complete"),
        ):
            demo.register_demo_container(
                "thread-1",
                {
                    "token": session_demo.registration_token,
                    "status": "failed",
                    "error": "server crashed",
                },
            )

        session_demo.refresh_from_db()
        self.assertEqual(session_demo.status, SessionDemo.STATUS_ACTIVE)
        self.assertEqual(session_demo.container_id, "container123")
        self.assertEqual(session_demo.container_name, "hitch-demo-thread-1-abcd")
        mock_remove.assert_not_called()

    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    @patch("hitch.main.demo.subprocess.run")
    def test_register_active_rejects_unpublished_port(
        self, mock_run: MagicMock, _cleanup: MagicMock
    ) -> None:
        session_demo = demo.request_demo_start("thread-1")
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=_inspect_stdout(
                token=session_demo.registration_token,
                name="hitch-demo-thread-1-abcd",
                host_port=45678,
            ),
            stderr="",
        )

        with self.assertRaisesRegex(demo.DemoError, "port is not published"):
            demo.register_demo_container(
                "thread-1",
                {
                    "token": session_demo.registration_token,
                    "status": "active",
                    "container_name": "hitch-demo-thread-1-abcd",
                    "container_id": "container123",
                    "host": "127.0.0.1",
                    "port": 9999,
                },
            )

    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    @patch("hitch.main.demo.subprocess.run")
    def test_register_rejects_late_updates_after_active(
        self, mock_run: MagicMock, _cleanup: MagicMock
    ) -> None:
        session_demo = demo.request_demo_start("thread-1")
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=_inspect_stdout(
                token=session_demo.registration_token,
                name="hitch-demo-thread-1-abcd",
                host_port=45678,
            ),
            stderr="",
        )
        demo.register_demo_container(
            "thread-1",
            {
                "token": session_demo.registration_token,
                "status": "active",
                "container_name": "hitch-demo-thread-1-abcd",
                "container_id": "container123",
                "host": "127.0.0.1",
                "port": 45678,
                "logs": "ready",
            },
        )

        with self.assertRaisesRegex(demo.DemoError, "already complete"):
            demo.register_demo_container(
                "thread-1",
                {
                    "token": session_demo.registration_token,
                    "status": "preparing",
                    "container_name": "hitch-demo-thread-1-late",
                    "logs": "late preparing",
                },
            )
        with self.assertRaisesRegex(demo.DemoError, "already complete"):
            demo.register_demo_container(
                "thread-1",
                {
                    "token": session_demo.registration_token,
                    "status": "failed",
                    "error": "late failure",
                },
            )

        session_demo.refresh_from_db()
        self.assertEqual(session_demo.status, SessionDemo.STATUS_ACTIVE)
        self.assertEqual(session_demo.container_name, "hitch-demo-thread-1-abcd")
        self.assertEqual(session_demo.logs, "ready")
        mock_run.assert_called_once()

    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    def test_register_rejects_late_updates_after_failed(
        self, _cleanup: MagicMock
    ) -> None:
        session_demo = demo.request_demo_start("thread-1")
        demo.register_demo_container(
            "thread-1",
            {
                "token": session_demo.registration_token,
                "status": "failed",
                "error": "setup failed",
            },
        )

        with self.assertRaisesRegex(demo.DemoError, "already complete"):
            demo.register_demo_container(
                "thread-1",
                {
                    "token": session_demo.registration_token,
                    "status": "preparing",
                    "container_name": "hitch-demo-thread-1-late",
                },
            )

        session_demo.refresh_from_db()
        self.assertEqual(session_demo.status, SessionDemo.STATUS_FAILED)
        self.assertEqual(session_demo.last_error, "setup failed")

    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    def test_register_rejects_stale_token(self, _cleanup: MagicMock) -> None:
        first = demo.request_demo_start("thread-1")
        SessionDemo.objects.filter(thread_id="thread-1").update(
            status=SessionDemo.STATUS_ACTIVE
        )
        demo.request_demo_start("thread-1")

        with self.assertRaisesRegex(demo.DemoError, "invalid demo registration token"):
            demo.register_demo_container(
                "thread-1",
                {
                    "token": first.registration_token,
                    "status": "active",
                    "container_name": "hitch-demo-thread-1-abcd",
                    "port": 45678,
                },
            )

    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    def test_register_rejects_container_name_without_session_prefix(
        self, _cleanup: MagicMock
    ) -> None:
        session_demo = demo.request_demo_start("thread-1")

        with self.assertRaisesRegex(demo.DemoError, "container name must start"):
            demo.register_demo_container(
                "thread-1",
                {
                    "token": session_demo.registration_token,
                    "status": "preparing",
                    "container_name": "other-container",
                },
            )

    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    def test_register_rejects_option_shaped_container_id(self, _cleanup: MagicMock) -> None:
        session_demo = demo.request_demo_start("thread-1")

        with self.assertRaisesRegex(demo.DemoError, "invalid container id"):
            demo.register_demo_container(
                "thread-1",
                {
                    "token": session_demo.registration_token,
                    "status": "preparing",
                    "container_name": "hitch-demo-thread-1-abcd",
                    "container_id": "--latest",
                },
            )

    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    @patch("hitch.main.demo.subprocess.run")
    def test_register_active_rejects_container_without_hitch_labels(
        self, mock_run: MagicMock, _cleanup: MagicMock
    ) -> None:
        session_demo = demo.request_demo_start("thread-1")
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='[{"Config":{"Labels":{"io.hitch.managed":"other"}}}]',
            stderr="",
        )

        with self.assertRaisesRegex(demo.DemoError, "missing label"):
            demo.register_demo_container(
                "thread-1",
                {
                    "token": session_demo.registration_token,
                    "status": "active",
                    "container_name": "hitch-demo-thread-1-abcd",
                    "container_id": "container123",
                    "host": "127.0.0.1",
                    "port": 45678,
                },
            )

    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    @patch("hitch.main.demo.subprocess.run")
    def test_register_failed_records_error_and_removes_registered_container(
        self, mock_run: MagicMock, _cleanup: MagicMock
    ) -> None:
        session_demo = demo.request_demo_start("thread-1")
        demo.register_demo_container(
            "thread-1",
            {
                "token": session_demo.registration_token,
                "status": "preparing",
                "container_name": "hitch-demo-thread-1-abcd",
            },
        )
        mock_run.side_effect = [
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=_inspect_stdout(
                    token=session_demo.registration_token,
                    name="hitch-demo-thread-1-abcd",
                ),
                stderr="",
            ),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
        ]

        updated = demo.register_demo_container(
            "thread-1",
            {
                "token": session_demo.registration_token,
                "status": "failed",
                "error": "server crashed",
                "logs": "traceback",
            },
        )

        self.assertEqual(updated.status, SessionDemo.STATUS_FAILED)
        self.assertEqual(updated.last_error, "server crashed")
        self.assertEqual(updated.logs, "traceback")
        self.assertEqual(updated.container_name, "")
        self.assertEqual(mock_run.call_args_list[1], call(
            ["podman", "rm", "-f", "hitch-demo-thread-1-abcd"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ))

    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    @patch("hitch.main.demo._remove_container")
    def test_register_failed_removes_container_outside_transaction(
        self, mock_remove: MagicMock, _cleanup: MagicMock
    ) -> None:
        session_demo = demo.request_demo_start("thread-1")
        demo.register_demo_container(
            "thread-1",
            {
                "token": session_demo.registration_token,
                "status": "preparing",
                "container_name": "hitch-demo-thread-1-abcd",
            },
        )
        baseline_savepoints = list(connection.savepoint_ids)

        def assert_unlocked(_demo: SessionDemo, *, ignore_missing: bool) -> None:
            self.assertEqual(connection.savepoint_ids, baseline_savepoints)
            self.assertTrue(ignore_missing)
            self.assertEqual(_demo.container_name, "hitch-demo-thread-1-abcd")

        mock_remove.side_effect = assert_unlocked

        updated = demo.register_demo_container(
            "thread-1",
            {
                "token": session_demo.registration_token,
                "status": "failed",
                "error": "server crashed",
            },
        )

        self.assertEqual(updated.status, SessionDemo.STATUS_FAILED)
        self.assertEqual(updated.container_name, "")
        mock_remove.assert_called_once()

    @patch("hitch.main.demo.logger")
    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    @patch("hitch.main.demo.subprocess.run")
    def test_register_failed_preserves_unverified_container_without_sweep(
        self, mock_run: MagicMock, mock_cleanup: MagicMock, _logger: MagicMock
    ) -> None:
        session_demo = demo.request_demo_start("thread-1")
        demo.register_demo_container(
            "thread-1",
            {
                "token": session_demo.registration_token,
                "status": "preparing",
                "container_name": "hitch-demo-thread-1-abcd",
            },
        )
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=_inspect_stdout(
                token=session_demo.registration_token,
                name="hitch-demo-thread-1-other",
            ),
            stderr="",
        )

        updated = demo.register_demo_container(
            "thread-1",
            {
                "token": session_demo.registration_token,
                "status": "failed",
                "error": "server crashed",
            },
        )

        self.assertEqual(updated.status, SessionDemo.STATUS_FAILED)
        self.assertEqual(updated.container_name, "hitch-demo-thread-1-abcd")
        self.assertIn("cleanup failed", updated.last_error)
        self.assertEqual(mock_cleanup.call_count, 1)

    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    @patch("hitch.main.demo.subprocess.run")
    def test_on_codex_instance_finished_marks_unregistered_demo_failed_and_cleans_up(
        self, mock_run: MagicMock, mock_cleanup: MagicMock
    ) -> None:
        SessionDemo.objects.create(
            thread_id="thread-1",
            host="127.0.0.1",
            port=3000,
            status=SessionDemo.STATUS_PREPARING,
            container_name="hitch-demo-thread-1-abcd",
            registration_token="token",
        )
        mock_run.side_effect = [
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=_inspect_stdout(token="token", name="hitch-demo-thread-1-abcd"),
                stderr="",
            ),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
        ]
        instance = CodexInstance.objects.create(
            thread_id="thread-1",
            cwd="/repo",
            prompt="Registration token: token\n",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_COMPLETED,
            pid=1,
            agent_kind=demo.DEMO_AGENT_KIND,
        )

        demo.on_codex_instance_finished(instance)

        session_demo = SessionDemo.objects.get(thread_id="thread-1")
        self.assertEqual(session_demo.status, SessionDemo.STATUS_FAILED)
        self.assertIn("without registering", session_demo.last_error)
        self.assertEqual(session_demo.container_name, "")
        self.assertEqual(mock_run.call_args_list[1], call(
            ["podman", "rm", "-f", "hitch-demo-thread-1-abcd"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ))
        mock_cleanup.assert_called_once()

    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    @patch("hitch.main.demo._remove_container")
    def test_on_codex_instance_finished_cleans_outside_transaction(
        self, mock_remove: MagicMock, _mock_cleanup: MagicMock
    ) -> None:
        SessionDemo.objects.create(
            thread_id="thread-1",
            host="127.0.0.1",
            port=3000,
            status=SessionDemo.STATUS_PREPARING,
            container_name="hitch-demo-thread-1-abcd",
            registration_token="token",
        )
        instance = CodexInstance.objects.create(
            thread_id="thread-1",
            cwd="/repo",
            prompt="Registration token: token\n",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_COMPLETED,
            pid=1,
            agent_kind=demo.DEMO_AGENT_KIND,
        )

        baseline_savepoints = list(connection.savepoint_ids)

        def assert_unlocked(_demo: SessionDemo, *, ignore_missing: bool) -> None:
            self.assertEqual(connection.savepoint_ids, baseline_savepoints)
            self.assertTrue(ignore_missing)

        mock_remove.side_effect = assert_unlocked

        demo.on_codex_instance_finished(instance)

        mock_remove.assert_called_once()

    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    @patch("hitch.main.demo._remove_container")
    def test_on_codex_instance_finished_does_not_clear_replacement_demo(
        self, mock_remove: MagicMock, _mock_cleanup: MagicMock
    ) -> None:
        SessionDemo.objects.create(
            thread_id="thread-1",
            host="127.0.0.1",
            port=3000,
            status=SessionDemo.STATUS_PREPARING,
            container_id="old-container",
            container_name="hitch-demo-thread-1-old",
            registration_token="old-token",
            generation=1,
        )
        instance = CodexInstance.objects.create(
            thread_id="thread-1",
            cwd="/repo",
            prompt="Registration token: old-token\n",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_COMPLETED,
            pid=1,
            agent_kind=demo.DEMO_AGENT_KIND,
        )

        def replace_demo(_demo: SessionDemo, *, ignore_missing: bool) -> None:
            self.assertTrue(ignore_missing)
            SessionDemo.objects.filter(thread_id="thread-1").update(
                status=SessionDemo.STATUS_ACTIVE,
                container_id="new-container",
                container_name="hitch-demo-thread-1-new",
                registration_token="new-token",
                generation=2,
                last_error="",
            )

        mock_remove.side_effect = replace_demo

        demo.on_codex_instance_finished(instance)

        session_demo = SessionDemo.objects.get(thread_id="thread-1")
        self.assertEqual(session_demo.status, SessionDemo.STATUS_ACTIVE)
        self.assertEqual(session_demo.container_id, "new-container")
        self.assertEqual(session_demo.container_name, "hitch-demo-thread-1-new")
        self.assertEqual(session_demo.registration_token, "new-token")
        self.assertEqual(session_demo.last_error, "")

    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    @patch("hitch.main.demo._remove_container")
    def test_on_codex_instance_finished_does_not_overwrite_replacement_error(
        self, mock_remove: MagicMock, _mock_cleanup: MagicMock
    ) -> None:
        SessionDemo.objects.create(
            thread_id="thread-1",
            host="127.0.0.1",
            port=3000,
            status=SessionDemo.STATUS_PREPARING,
            container_name="hitch-demo-thread-1-old",
            registration_token="old-token",
            generation=1,
        )
        instance = CodexInstance.objects.create(
            thread_id="thread-1",
            cwd="/repo",
            prompt="Registration token: old-token\n",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_COMPLETED,
            pid=1,
            agent_kind=demo.DEMO_AGENT_KIND,
        )

        def replace_and_fail(_demo: SessionDemo, *, ignore_missing: bool) -> None:
            self.assertTrue(ignore_missing)
            SessionDemo.objects.filter(thread_id="thread-1").update(
                status=SessionDemo.STATUS_ACTIVE,
                container_name="hitch-demo-thread-1-new",
                registration_token="new-token",
                generation=2,
                last_error="",
            )
            raise demo.DemoError("podman timed out")

        mock_remove.side_effect = replace_and_fail

        demo.on_codex_instance_finished(instance)

        session_demo = SessionDemo.objects.get(thread_id="thread-1")
        self.assertEqual(session_demo.status, SessionDemo.STATUS_ACTIVE)
        self.assertEqual(session_demo.container_name, "hitch-demo-thread-1-new")
        self.assertEqual(session_demo.registration_token, "new-token")
        self.assertEqual(session_demo.last_error, "")

    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    @patch("hitch.main.demo.subprocess.run")
    def test_on_codex_instance_finished_fails_demo_system_run_when_unregistered(
        self, mock_run: MagicMock, _mock_cleanup: MagicMock
    ) -> None:
        SessionDemo.objects.create(
            thread_id="thread-1",
            host="127.0.0.1",
            port=3000,
            status=SessionDemo.STATUS_PREPARING,
            container_name="hitch-demo-thread-1-abcd",
            registration_token="token",
        )
        mock_run.side_effect = [
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=_inspect_stdout(token="token", name="hitch-demo-thread-1-abcd"),
                stderr="",
            ),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
        ]
        workflow = SystemWorkflow.objects.create(
            kind=demo.DEMO_WORKFLOW_KIND,
            main_thread_id="thread-1",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
        )
        instance = CodexInstance.objects.create(
            thread_id="thread-1",
            cwd="/repo",
            prompt="Registration token: token\n",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_COMPLETED,
            pid=1,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=demo.DEMO_AGENT_KIND,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=demo.DEMO_AGENT_KIND,
            thread_id="thread-1",
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )

        demo.on_codex_instance_finished(instance)

        run.refresh_from_db()
        workflow.refresh_from_db()
        self.assertEqual(run.status, SystemAgentRun.STATUS_FAILED)
        self.assertIn("without registering", run.error)
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_FAILED)

    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    @patch("hitch.main.demo.subprocess.run")
    def test_on_codex_instance_finished_creates_missing_demo_system_run(
        self, mock_run: MagicMock, _mock_cleanup: MagicMock
    ) -> None:
        SessionDemo.objects.create(
            thread_id="thread-1",
            host="127.0.0.1",
            port=3000,
            status=SessionDemo.STATUS_PREPARING,
            container_name="hitch-demo-thread-1-abcd",
            registration_token="token",
        )
        mock_run.side_effect = [
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=_inspect_stdout(token="token", name="hitch-demo-thread-1-abcd"),
                stderr="",
            ),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
        ]
        workflow = SystemWorkflow.objects.create(
            kind=demo.DEMO_WORKFLOW_KIND,
            main_thread_id="thread-1",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
        )
        instance = CodexInstance.objects.create(
            thread_id="thread-1",
            cwd="/repo",
            prompt="Registration token: token\n",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_COMPLETED,
            pid=1,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=demo.DEMO_AGENT_KIND,
        )

        demo.on_codex_instance_finished(instance)

        run = SystemAgentRun.objects.get(instance=instance)
        workflow.refresh_from_db()
        self.assertEqual(run.agent_kind, demo.DEMO_AGENT_KIND)
        self.assertEqual(run.status, SystemAgentRun.STATUS_FAILED)
        self.assertIn("without registering", run.error)
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_FAILED)

    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    def test_on_codex_instance_finished_ignores_stale_demo_token(
        self, mock_cleanup: MagicMock
    ) -> None:
        first = demo.request_demo_start("thread-1")
        SessionDemo.objects.filter(thread_id="thread-1").update(
            status=SessionDemo.STATUS_ACTIVE
        )
        second = demo.request_demo_start("thread-1")
        instance = CodexInstance.objects.create(
            thread_id="thread-1",
            cwd="/repo",
            prompt=f"Registration token: {first.registration_token}\n",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_COMPLETED,
            pid=1,
            agent_kind=demo.DEMO_AGENT_KIND,
        )

        demo.on_codex_instance_finished(instance)

        session_demo = SessionDemo.objects.get(thread_id="thread-1")
        self.assertEqual(session_demo.status, SessionDemo.STATUS_REQUESTED)
        self.assertEqual(session_demo.registration_token, second.registration_token)
        self.assertEqual(session_demo.last_error, "")
        self.assertEqual(mock_cleanup.call_count, 3)

    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    def test_on_codex_instance_finished_does_not_fail_active_demo(
        self, mock_cleanup: MagicMock
    ) -> None:
        SessionDemo.objects.create(
            thread_id="thread-1",
            host="127.0.0.1",
            port=3000,
            status=SessionDemo.STATUS_ACTIVE,
            registration_token="token",
            container_name="hitch-demo-thread-1-abcd",
        )
        workflow = SystemWorkflow.objects.create(
            kind=demo.DEMO_WORKFLOW_KIND,
            main_thread_id="thread-1",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
        )
        instance = CodexInstance.objects.create(
            thread_id="thread-1",
            cwd="/repo",
            prompt="Registration token: token\n",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_COMPLETED,
            pid=1,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=demo.DEMO_AGENT_KIND,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=demo.DEMO_AGENT_KIND,
            thread_id="thread-1",
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )

        demo.on_codex_instance_finished(instance)

        session_demo = SessionDemo.objects.get(thread_id="thread-1")
        run.refresh_from_db()
        workflow.refresh_from_db()
        self.assertEqual(session_demo.status, SessionDemo.STATUS_ACTIVE)
        self.assertEqual(session_demo.container_name, "hitch-demo-thread-1-abcd")
        self.assertEqual(run.status, SystemAgentRun.STATUS_COMPLETED)
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        mock_cleanup.assert_called_once()

    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    def test_on_codex_instance_finished_cleans_unregistered_after_stop(
        self, mock_cleanup: MagicMock
    ) -> None:
        SessionDemo.objects.create(
            thread_id="thread-1",
            host="127.0.0.1",
            port=3000,
            status=SessionDemo.STATUS_STOPPED,
            registration_token="token",
        )
        instance = CodexInstance.objects.create(
            thread_id="thread-1",
            cwd="/repo",
            prompt="Registration token: token\n",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_COMPLETED,
            pid=1,
            agent_kind=demo.DEMO_AGENT_KIND,
        )

        demo.on_codex_instance_finished(instance)

        mock_cleanup.assert_called_once()

    @patch("hitch.main.demo.subprocess.run")
    def test_cleanup_unregistered_demo_containers_preserves_registered_current_token(
        self, mock_run: MagicMock
    ) -> None:
        SessionDemo.objects.create(
            thread_id="thread-1",
            host="127.0.0.1",
            port=45678,
            container_id="keep-id",
            container_name="keep-name",
            status=SessionDemo.STATUS_ACTIVE,
            registration_token="token-1",
        )
        mock_run.side_effect = [
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=(
                    '[{"ID":"keep-id","Names":["keep-name"],'
                    '"Labels":{"io.hitch.demo_token":"token-1"}},'
                    '{"ID":"old-id","Names":["old-name"],'
                    '"Labels":{"io.hitch.demo_token":"old-token"}}]'
                ),
                stderr="",
            ),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
        ]

        removed = demo.cleanup_unregistered_demo_containers()

        self.assertEqual(removed, 1)
        self.assertEqual(mock_run.call_count, 2)
        self.assertEqual(mock_run.call_args_list[1], call(
            ["podman", "rm", "-f", "old-id"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ))

    @patch("hitch.main.demo.subprocess.run")
    def test_cleanup_unregistered_demo_containers_preserves_failed_retained_target(
        self, mock_run: MagicMock
    ) -> None:
        SessionDemo.objects.create(
            thread_id="thread-1",
            host="127.0.0.1",
            port=45678,
            container_id="keep-id",
            container_name="keep-name",
            status=SessionDemo.STATUS_FAILED,
            registration_token="token-1",
            last_error="cleanup failed: label mismatch",
        )
        mock_run.side_effect = [
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=(
                    '[{"ID":"keep-id","Names":["keep-name"],'
                    '"Labels":{"io.hitch.demo_token":"wrong-token"}},'
                    '{"ID":"old-id","Names":["old-name"],'
                    '"Labels":{"io.hitch.demo_token":"old-token"}}]'
                ),
                stderr="",
            ),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
        ]

        removed = demo.cleanup_unregistered_demo_containers()

        self.assertEqual(removed, 1)
        self.assertEqual(mock_run.call_count, 2)
        self.assertEqual(mock_run.call_args_list[1], call(
            ["podman", "rm", "-f", "old-id"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ))

    @patch("hitch.main.demo.subprocess.run")
    def test_cleanup_unregistered_demo_containers_ignores_malformed_ps_json(
        self, mock_run: MagicMock
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"ID":"first"}\nnot-json\n',
            stderr="",
        )

        removed = demo.cleanup_unregistered_demo_containers()

        self.assertEqual(removed, 0)
        mock_run.assert_called_once()

    @override_settings(HITCH_DEMO_RUNTIME="docker")
    def test_demo_runtime_rejects_unsupported_runtime(self) -> None:
        with self.assertRaisesRegex(demo.DemoError, "only podman"):
            demo.demo_runtime()
