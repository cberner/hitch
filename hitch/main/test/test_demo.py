from __future__ import annotations

import subprocess
import threading
from collections.abc import Iterable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import cast, override
from unittest.mock import MagicMock, patch

from django.http import StreamingHttpResponse
from django.test import RequestFactory, TestCase, override_settings
from django.urls import reverse

from hitch.main import demo
from hitch.main.models import SessionDemo


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
        type(self).seen = {
            "method": "GET",
            "path": self.path,
            "host": self.headers.get("Host", ""),
            "forwarded": self.headers.get("X-Forwarded-Host", ""),
            "cookie": self.headers.get("Cookie", ""),
            "authorization": self.headers.get("Authorization", ""),
        }
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
        if self.path.startswith("/events"):
            body = b"data: one\n\ndata: two\n\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
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
        type(self).seen = {"method": "POST", "path": self.path, "body": body}
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
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True
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
        self.assertIn(
            b'import "/sessions/thread-1/demo/dep.js"',
            response.content,
        )
        self.assertIn(
            b'fetch("/sessions/thread-1/demo/api/status")',
            response.content,
        )
        self.assertIn(
            b'import("/sessions/thread-1/demo/dynamic.js")',
            response.content,
        )

    def test_rewrite_body_preserves_protocol_relative_urls_and_rewrites_srcset(self) -> None:
        body = (
            b'<link href="//cdn.example.com/app.css">'
            b'<img srcset="/small.png 1x, //cdn.example.com/big.png 2x" '
            b'poster="/poster.png">'
            b"<style>.hero{background:url(//cdn.example.com/bg.png)}</style>"
            b'<script>fetch("//api.example.com/status"); import x from "//cdn.example.com/mod.js";</script>'
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

    def test_rewrite_location_preserves_relative_and_adds_query_fragment(self) -> None:
        self.assertEqual(
            demo._rewrite_location("asset.css", "/sessions/thread-1/demo/", upstream_netloc="127.0.0.1:1"),
            "asset.css",
        )
        self.assertEqual(
            demo._rewrite_location(
                "/next?ok=1#section",
                "/sessions/thread-1/demo/",
                upstream_netloc="127.0.0.1:1",
            ),
            "/sessions/thread-1/demo/next?ok=1#section",
        )

    def test_proxy_streams_sse(self) -> None:
        response = self.client.get(
            reverse(
                "session_demo_proxy",
                kwargs={"session_id": "thread-1", "path": "events"},
            )
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/event-stream")
        self.assertEqual(
            _response_body(response),
            b"data: one\n\ndata: two\n\n",
        )

    def test_host_based_proxy_uses_root_path_without_html_rewrite(self) -> None:
        response = self.client.get(
            "/asset",
            headers={"host": "thread-1.demo.localhost"},
        )

        self.assertEqual(response.status_code, 200)
        body = _response_body(response)
        self.assertIn(b'href="/asset.css"', body)
        self.assertNotIn(b"/sessions/thread-1/demo", body)

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
        response = self.client.get(
            reverse(
                "session_demo_proxy",
                kwargs={"session_id": "thread-1", "path": "events"},
            ),
            HTTP_HOST="127.0.0.1:8000",
        )

        self.assertEqual(response.status_code, 200)
        url = demo.demo_url_for_request(response.wsgi_request, "thread-1")
        self.assertEqual(url, "http://thread-1.demo.localhost:8000/")

    def test_unsafe_session_id_uses_path_demo_url_for_loopback(self) -> None:
        request = RequestFactory().get("/", headers={"host": "127.0.0.1:8000"})

        url = demo.demo_url_for_request(request, "thread_1")

        self.assertEqual(url, "http://127.0.0.1:8000/sessions/thread_1/demo/")
        self.assertEqual(
            demo.demo_url_for_request(request, "Thread-1"),
            "http://127.0.0.1:8000/sessions/Thread-1/demo/",
        )
        self.assertIsNone(demo.session_id_from_demo_host("thread_1.demo.localhost"))


class DemoContainerTests(TestCase):
    def test_session_demo_str_includes_target_and_status(self) -> None:
        session_demo = SessionDemo(
            thread_id="thread-1",
            host="127.0.0.1",
            port=12345,
            status=SessionDemo.STATUS_ACTIVE,
        )

        self.assertEqual(
            str(session_demo),
            "SessionDemo(thread_id=thread-1, target=127.0.0.1:12345, status=active)",
        )

    @override_settings(HITCH_DEMO_RUNTIME="docker")
    def test_start_demo_container_rejects_unsupported_runtime(self) -> None:
        with self.assertRaisesRegex(demo.DemoError, "only podman"):
            demo.start_demo_container("thread-1")

    @patch("hitch.main.demo.subprocess.run", side_effect=OSError("no podman"))
    @patch("hitch.main.demo._reserve_port", return_value=45678)
    @patch("hitch.main.demo._container_name_for", return_value="hitch-demo-thread")
    def test_failed_start_records_failed_demo_row(
        self,
        _mock_name: MagicMock,
        _mock_port: MagicMock,
        _mock_run: MagicMock,
    ) -> None:
        with self.assertRaisesRegex(demo.DemoError, "no podman"):
            demo.start_demo_container("thread-1")

        session_demo = SessionDemo.objects.get(thread_id="thread-1")
        self.assertEqual(session_demo.status, SessionDemo.STATUS_FAILED)
        self.assertEqual(session_demo.container_id, "")
        self.assertEqual(session_demo.container_name, "hitch-demo-thread")
        self.assertEqual(session_demo.last_error, "no podman")

    @patch("hitch.main.demo.subprocess.run")
    @patch("hitch.main.demo._reserve_port", return_value=45678)
    @patch("hitch.main.demo._container_name_for", return_value="hitch-demo-thread")
    def test_start_demo_container_uses_podman_and_replaces_existing(
        self,
        _mock_name: MagicMock,
        _mock_port: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        SessionDemo.objects.create(
            thread_id="thread-1",
            host="127.0.0.1",
            port=12345,
            container_id="old-container",
            runtime="podman",
            status=SessionDemo.STATUS_ACTIVE,
        )
        mock_run.side_effect = [
            SimpleNamespace(stdout="removed\n"),
            SimpleNamespace(stdout="new-container\n"),
        ]

        session_demo, container_port = demo.start_demo_container("thread-1")

        self.assertEqual(container_port, 3000)
        self.assertEqual(session_demo.container_id, "new-container")
        self.assertEqual(session_demo.port, 45678)
        self.assertEqual(session_demo.status, SessionDemo.STATUS_ACTIVE)
        self.assertEqual(mock_run.call_args_list[0].args[0], ["podman", "rm", "-f", "old-container"])
        self.assertEqual(
            mock_run.call_args_list[1].args[0],
            [
                "podman",
                "run",
                "-d",
                "--name",
                "hitch-demo-thread",
                "-p",
                "127.0.0.1:45678:3000",
                "node:22-bookworm",
                "sleep",
                "infinity",
            ],
        )

    @patch("hitch.main.demo.subprocess.run")
    def test_cleanup_demo_for_session_removes_active_container(self, mock_run: MagicMock) -> None:
        SessionDemo.objects.create(
            thread_id="thread-1",
            host="127.0.0.1",
            port=12345,
            container_id="container-1",
            runtime="podman",
            status=SessionDemo.STATUS_ACTIVE,
        )
        mock_run.return_value = SimpleNamespace(stdout="removed\n")

        demo.cleanup_demo_for_session("thread-1")

        session_demo = SessionDemo.objects.get(thread_id="thread-1")
        self.assertEqual(session_demo.status, SessionDemo.STATUS_STOPPED)
        mock_run.assert_called_once_with(
            ["podman", "rm", "-f", "container-1"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

    @patch("hitch.main.demo._remove_container", side_effect=demo.DemoError("boom"))
    def test_cleanup_demo_for_session_marks_failed_when_remove_fails(
        self, _mock_remove: MagicMock
    ) -> None:
        SessionDemo.objects.create(
            thread_id="thread-1",
            host="127.0.0.1",
            port=12345,
            container_id="container-1",
            runtime="podman",
            status=SessionDemo.STATUS_ACTIVE,
        )

        demo.cleanup_demo_for_session("thread-1")

        session_demo = SessionDemo.objects.get(thread_id="thread-1")
        self.assertEqual(session_demo.status, SessionDemo.STATUS_FAILED)
        self.assertEqual(session_demo.last_error, "boom")

    @patch("hitch.main.demo.subprocess.run")
    @patch("hitch.main.demo._reserve_port", return_value=45678)
    @patch("hitch.main.demo._container_name_for", return_value="hitch-demo-thread")
    def test_retry_after_failed_start_ignores_missing_stale_container_name(
        self,
        _mock_name: MagicMock,
        _mock_port: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        SessionDemo.objects.create(
            thread_id="thread-1",
            host="127.0.0.1",
            port=12345,
            container_name="hitch-demo-thread-old",
            runtime="podman",
            status=SessionDemo.STATUS_FAILED,
        )
        mock_run.side_effect = [
            subprocess.CalledProcessError(
                125,
                ["podman", "rm", "-f", "hitch-demo-thread-old"],
                stderr="no such container",
            ),
            SimpleNamespace(stdout="new-container\n"),
        ]

        session_demo, _container_port = demo.start_demo_container("thread-1")

        self.assertEqual(session_demo.container_id, "new-container")
        self.assertEqual(session_demo.status, SessionDemo.STATUS_ACTIVE)
        self.assertEqual(mock_run.call_count, 2)

    def test_reserve_port_returns_available_loopback_port(self) -> None:
        self.assertGreater(demo._reserve_port(), 0)

    def test_container_name_for_slugs_session_id_and_has_fallback(self) -> None:
        self.assertRegex(
            demo._container_name_for("Feature ABC"),
            r"^hitch-demo-feature-abc-[0-9a-f]{8}$",
        )
        self.assertRegex(
            demo._container_name_for("!!!"),
            r"^hitch-demo-session-[0-9a-f]{8}$",
        )

    @patch("hitch.main.demo.subprocess.run")
    def test_remove_container_raises_demo_error(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.CalledProcessError(
            125,
            ["podman"],
            stderr="permission denied",
        )
        session_demo = SessionDemo(
            host="127.0.0.1",
            port=12345,
            container_id="container-1",
            runtime="podman",
        )

        with self.assertRaisesRegex(demo.DemoError, "permission denied"):
            demo._remove_container(session_demo)

    @patch("hitch.main.demo.subprocess.run")
    def test_remove_container_without_target_is_noop(self, mock_run: MagicMock) -> None:
        session_demo = SessionDemo(host="127.0.0.1", port=12345)

        demo._remove_container(session_demo)

        mock_run.assert_not_called()

    def test_container_missing_error_and_subprocess_message_fallbacks(self) -> None:
        self.assertFalse(demo._container_missing_error(OSError("no runtime")))
        self.assertEqual(
            demo._subprocess_error_message(
                subprocess.CalledProcessError(125, ["podman"], stderr="")
            ),
            "podman exited with status 125",
        )
        self.assertEqual(
            demo._subprocess_error_message(
                subprocess.TimeoutExpired(["podman"], timeout=30)
            ),
            "podman command timed out",
        )

    def test_rewrite_body_returns_original_for_unknown_encoding(self) -> None:
        body = b'<a href="/asset.css">'

        self.assertEqual(
            demo._rewrite_body(
                body,
                "text/html; charset=not-a-real-codec",
                "/sessions/thread-1/demo/",
            ),
            body,
        )
