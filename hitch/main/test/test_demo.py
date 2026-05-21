from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Iterable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast, override
from unittest.mock import MagicMock, call, patch

from django.http import StreamingHttpResponse
from django.test import RequestFactory, TestCase, override_settings
from django.urls import reverse

from hitch.main import demo
from hitch.main.models import CodexInstance, SessionDemo


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
        for blocked in (
            "authorization",
            "connection",
            "content-length",
            "cookie",
            "expect",
            "forwarded",
            "proxy-connection",
            "x-client-only",
            "x-csrftoken",
            "x-forwarded-port",
        ):
            self.assertNotIn(blocked, normalized)

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
        mock_connection.return_value.request.side_effect = OSError("refused")
        request = RequestFactory().get("/sessions/thread-1/demo/asset")

        response = demo.proxy_demo_request(
            request,
            "thread-1",
            "asset",
            path_prefix="/sessions/thread-1/demo/",
        )

        self.assertEqual(response.status_code, 502)
        self.assertIn(b"demo target unavailable: refused", response.content)

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
        self.assertEqual(session_demo.status, SessionDemo.STATUS_ACTIVE)
        self.assertEqual(session_demo.container_name, "hitch-demo-thread-1-abcd")
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
