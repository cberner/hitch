import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.parse import urlencode

from django.conf import settings
from django.template.loader import render_to_string
from django.test import TestCase
from django.urls import reverse

from hitch.main.models import SessionMetadata
from hitch.main.test.support import _rollout_line, _setup_codex
from hitch.main.test.views_helpers import _make_rollout, _session


class SessionCommandTests(TestCase):
    def command_url(self, call_id: str) -> str:
        return f"{reverse('session_command', args=['commands'])}?{urlencode({'id': call_id})}"

    def make_commands(self) -> tuple[Path, str]:
        command = "printf " + "long command " * 2000 + "<script>window.injected=true</script>"
        path = _make_rollout(self, [
            _rollout_line("event_msg", {"type": "user_message", "message": "Work"}),
            self.command_line("single", command),
            _rollout_line("event_msg", {"type": "agent_message", "message": "Still working"}),
            self.command_line("earlier", "printf earlier"),
            self.command_line("latest", command),
        ])
        SessionMetadata.objects.create(thread_id="commands", codex_path=str(path))
        return path, command

    def command_line(self, call_id: str, command: str) -> str:
        return _rollout_line("response_item", {
            "type": "function_call", "name": "exec_command", "call_id": call_id,
            "arguments": json.dumps({"cmd": command}),
        })

    @patch("hitch.main.caches._start_models_refresh_thread")
    def test_long_commands_defer_full_text_and_read_fresh_rollout(self, _refresh: MagicMock) -> None:
        path, command = self.make_commands()
        session_url = reverse("session", args=["commands"])
        with patch("hitch.main.views.common._SESSION_HISTORY_MIN_BYTES", 1), patch(
            "hitch.main.views.common._SESSION_HISTORY_MESSAGE_TARGET", 1,
        ):
            preview = self.client.get(session_url)
            self.assertContains(preview, "data-history-all")
            self.assertNotContains(preview, "data-command-url=")
            self.assertNotContains(preview, "data-expandable-command")
            response = self.client.get(session_url, {"history": "all"})
        self.assertContains(response, 'data-command-url=', count=2)
        self.assertNotContains(response, "window.injected")
        for call_id in ("single", "latest"):
            url = self.command_url(call_id)
            full = self.client.get(url)
            self.assertEqual(full.content.decode(), command)
            self.assertIn("no-store", full["Cache-Control"])
            self.assertTrue(full["Content-Type"].startswith("text/plain"))
            self.assertEqual(self.client.post(url).status_code, 405)
        group = self.client.get(reverse("session_intermediate", args=["commands", 3]))
        self.assertContains(group, "printf earlier")
        self.assertContains(group, 'data-command-url=', count=1)
        self.assertNotContains(group, "window.injected")
        for url in (
            self.command_url(""), self.command_url("missing"), reverse("session_intermediate", args=["commands", 1]),
        ):
            self.assertEqual(self.client.get(url).status_code, 404)
        path.write_text(path.read_text().replace("window.injected=true", "window.changed=true"))
        updated = self.client.get(self.command_url("latest"))
        self.assertContains(updated, "window.changed=true")
        self.assertNotContains(updated, "window.injected=true")

    @patch("hitch.main.caches._start_models_refresh_thread")
    @patch("hitch.main.views.common.Codex")
    def test_command_identity_survives_appends_and_unindexed_reads(
        self, mock_codex: MagicMock, _refresh: MagicMock,
    ) -> None:
        path, command = self.make_commands()
        url = self.command_url("latest")
        self.client.get(reverse("session", args=["commands"]))
        with path.open("a") as handle:
            handle.write("\n" + self.command_line("new latest", "different command " * 100) + "\n")
        self.assertEqual(self.client.get(url).content.decode(), command)
        SessionMetadata.objects.filter(thread_id="commands").delete()
        client = _setup_codex(mock_codex)
        client._client.thread_read.return_value = SimpleNamespace(thread=_session("commands", path=str(path)))
        with patch("hitch.main.sessions.session_resume._stored_rollout_path_for_thread", return_value=path):
            for query in ({}, {"history": "all"}):
                full = self.client.get(reverse("session", args=["commands"]), query)
                self.assertContains(full, "data-command-url=")
                self.assertNotContains(full, "window.injected")
                self.assertEqual(self.client.get(url).content.decode(), command)
                group = self.client.get(reverse("session_intermediate", args=["commands", 3]))
                self.assertContains(group, "printf earlier")
                self.assertNotContains(group, "window.injected")
        client._client.thread_read.reset_mock()
        with patch("hitch.main.sessions.session_resume._stored_rollout_path_for_thread", return_value=None):
            self.assertEqual(self.client.get(url).content.decode(), command)
        client._client.thread_read.assert_called_once_with("commands", include_turns=False)
        client.thread_resume.assert_not_called()
        self.assertFalse(SessionMetadata.objects.filter(thread_id="commands").exists())

    @patch("hitch.main.caches._start_models_refresh_thread")
    def test_command_browser_expansion_retry_and_collapse(self, _refresh: MagicMock) -> None:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import Route, sync_playwright

        self.make_commands()
        html = self.client.get(reverse("session", args=["commands"])).content.decode()

        def route_request(route: Route) -> None:
            url = route.request.url
            if "/assets/" in url:
                asset = url.split("/assets/", 1)[1]
                route.fulfill(body=render_to_string("assets/" + asset), content_type=(
                    "text/css" if asset.endswith(".css") else "text/javascript"
                ))
            elif "/static/" in url:
                route.fulfill(path=str(Path(settings.BASE_DIR) / "hitch/main/static" / url.split("/static/", 1)[1]))
            elif route.request.resource_type == "document":
                route.fulfill(body=html, content_type="text/html")
            else:
                route.fulfill(json={"agents": []})

        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
            except PlaywrightError as exc:
                self.skipTest(f"playwright browser unavailable: {exc}")
            try:
                page = browser.new_page()
                page.route("**/*", route_request)
                page.add_init_script("""
                    window.EventSource = class { addEventListener() {} close() {} };
                    window.commands = [];
                    const originalFetch = window.fetch;
                    window.fetch = (url, options) => String(url).includes('/command/')
                        ? new Promise(resolve => commands.push({resolve, options})) : originalFetch(url, options);
                """)
                page.goto("http://hitch.test/sessions/commands/")
                button = page.locator("[data-command-url]").first
                preview = button.inner_text()
                self.assertEqual(page.evaluate("commands.length"), 0)
                button.click()
                page.evaluate("commands[0].resolve(new Response('', {status: 503}))")
                page.wait_for_function(
                    "document.querySelector('[data-command-url]').textContent.includes('Click to retry')"
                )
                button.click()
                button.click()
                self.assertEqual(button.inner_text(), preview)
                button.click()
                page.evaluate("commands[2].resolve(new Response('<script>window.injected=true</script>'))")
                page.evaluate("commands[1].resolve(new Response('obsolete'))")
                page.wait_for_function("document.querySelector('[data-command-url]').textContent.includes('window.injected')")
                self.assertFalse(page.evaluate("Boolean(window.injected)"))
                self.assertEqual(page.evaluate("commands.map(c => c.options.cache)"), ["no-store"] * 3)
                button.click()
                self.assertEqual(button.inner_text(), preview)
            finally:
                browser.close()
