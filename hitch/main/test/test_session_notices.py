"""Browser coverage for model warnings and their integration with session controls."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from django.conf import settings
from django.test import TestCase
from django.urls import reverse

from hitch.main.models import CodexInstance, SessionMetadata
from hitch.main.test.support import _make_model, _setup_codex
from hitch.main.test.views_helpers import _basic_session_rollout_lines, _make_rollout
from hitch.main.views import common


class SessionNoticeTests(TestCase):
    @patch("hitch.main.runtime.codex_pool.worker_is_alive", return_value=True)
    @patch("hitch.main.caches._cached_models_for_session_detail")
    @patch("hitch.main.views.common.Codex")
    def test_notices_recover_and_capacity_action_preserves_draft(
        self, codex: MagicMock, catalog: MagicMock, _alive: MagicMock,
    ) -> None:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import Route, sync_playwright

        catalog.return_value = [_make_model("first")]
        client = _setup_codex(codex, models=catalog.return_value)
        client._client.thread_read.return_value.thread = SimpleNamespace(
            id="session", name="Sample session", preview="Prompt", cwd="/repo", updated_at=1700000000, turns=[],
        )
        SessionMetadata.objects.create(
            thread_id="session", cwd="/repo",
            codex_path=str(_make_rollout(self, _basic_session_rollout_lines("Prompt", "Response"))),
        )
        instance = CodexInstance.objects.create(
            thread_id="session", status=CodexInstance.STATUS_FAILED, pid=0,
            codex_error_info="serverOverloaded", error="Selected model is at capacity.",
        )
        url = reverse("session", args=["session"])
        response = self.client.get(url)
        html = response.content.decode()
        read_only = common._render_session_detail(response.wsgi_request, "session", read_only=True)
        self.assertContains(read_only, "Model at capacity")
        self.assertNotContains(read_only, "Change model")
        instance.status, instance.pid = CodexInstance.STATUS_RUNNING, 1
        instance.save()
        active_html = self.client.get(url).content.decode()

        def route_request(route: Route) -> None:
            if "/static/" in route.request.url:
                path = route.request.url.split("/static/", 1)[1]
                route.fulfill(path=str(Path(settings.BASE_DIR) / "hitch/main/static" / path))
            elif "/agents/" in route.request.url:
                route.fulfill(json={"agents": [{"id": "child", "name": "Reviewer", "parent_id": "session"}],
                                    "selected": "child", "name": "Reviewer", "html": "Child response"})
            else:
                route.fulfill(content_type="text/html", body=html)

        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
            except PlaywrightError as exc:
                self.skipTest(f"playwright browser unavailable: {exc}")
            try:
                page = browser.new_page(viewport={"width": 375, "height": 812})
                errors: list[str] = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.add_init_script("""
                    window.EventSource = class {
                        constructor() { window.streamHandlers = {}; }
                        addEventListener(name, callback) { window.streamHandlers[name] = callback; }
                        close() {}
                    };
                """)
                page.route("**/*", route_request)
                page.goto("http://hitch.test" + url)
                draft = page.locator("[data-composer-input]")
                draft.fill("Keep this draft")
                page.get_by_role("button", name="Change model", exact=True).click()
                dialog = page.locator("[data-model-settings-dialog]")
                self.assertTrue(dialog.is_visible())
                dialog.get_by_role("button", name="Cancel", exact=True).click()
                self.assertEqual(draft.input_value(), "Keep this draft")

                html = active_html
                page.goto("http://hitch.test" + url)
                draft.fill("Keep this draft")
                notice = page.locator("[data-turn-notice]")

                def emit(method: str, **payload: Any) -> None:
                    page.evaluate("event => streamHandlers.message({data: JSON.stringify(event)})", {
                        "method": method, "payload": {"threadId": "session", "turnId": "turn", **payload},
                    })

                def retry() -> None:
                    emit("error", willRetry=True, error={"message": "Reconnecting... 2/5",
                         "additionalDetails": "<img src=x onerror=alert(1)>"})

                retry()
                notice.locator("summary").click()
                self.assertIn("Reconnecting... 2/5", notice.inner_text())
                self.assertEqual(notice.locator("img").count(), 0)
                page.evaluate("streamHandlers.error()")
                self.assertEqual(page.locator("[data-live-status]").get_attribute("data-state"), "reconnecting")
                self.assertTrue(notice.is_visible())
                page.evaluate("streamHandlers.heartbeat({data: JSON.stringify({working: true})})")
                emit("item/commandExecution/outputDelta", delta="Background output")
                emit("model/safetyBuffering/updated", showBufferingUi=False)
                emit("turn/completed", turnId="older-turn", turn={"id": "older-turn"})
                self.assertTrue(notice.is_visible())
                agent = page.locator("[data-agent-select]")
                agent.locator("option[value='child']").wait_for(state="attached")
                agent.select_option("child")
                self.assertFalse(notice.is_visible())
                agent.select_option("")
                self.assertTrue(notice.is_visible())
                self.assertEqual(draft.input_value(), "Keep this draft")

                for _ in range(3):
                    emit("model/safetyBuffering/updated", showBufferingUi=True, useCases=["bio"], reasons=["probes"])
                emit("thread/goal/updated", goal={"objective": "Check the result and write up the proof"})
                self.assertEqual(notice.locator("summary").inner_text(), "Checking response…")
                notice.locator("summary").click()
                emit("model/safetyBuffering/updated", showBufferingUi=True)
                self.assertTrue(notice.evaluate("el => el.open"))
                self.assertNotIn("bio", notice.inner_text())
                self.assertLessEqual(page.evaluate("document.documentElement.scrollWidth"), 375)
                emit("item/agentMessage/delta", threadId="child", delta="Other agent")
                self.assertTrue(notice.is_visible())
                emit("item/agentMessage/delta", delta="Recovered")
                self.assertFalse(notice.is_visible())

                for method, recovery_payload in (
                    ("item/reasoning/summaryTextDelta", {"delta": "Thinking"}),
                    ("item/reasoning/textDelta", {"delta": "Thinking"}),
                    ("item/plan/delta", {"itemId": "plan-recovery", "delta": "Next step"}),
                    ("item/started", {"item": {"id": "reasoning-recovery", "type": "reasoning"}}),
                    ("item/started", {"item": {"id": "plan-recovery", "type": "plan"}}),
                    ("item/completed", {"item": {"id": "reasoning-recovery", "type": "reasoning"}}),
                    ("item/completed", {"item": {"id": "plan-recovery", "type": "plan", "text": "Next step"}}),
                ):
                    retry()
                    emit(method, **recovery_payload)
                    self.assertFalse(notice.is_visible(), method)
                    emit("model/safetyBuffering/updated", showBufferingUi=True)
                    emit(method, **recovery_payload)
                    self.assertTrue(notice.is_visible(), method)

                for method, payload in (
                    ("item/completed", {"item": {"id": "reply", "type": "agentMessage", "text": "Done"}}),
                    ("item/started", {"item": {"id": "tool", "type": "commandExecution", "command": "pwd"}}),
                    ("item/started", {"item": {"id": "image-view", "type": "imageView"}}),
                    ("item/started", {"item": {"id": "image-generation", "type": "imageGeneration"}}),
                    ("turn/started", {"turn": {"id": "next-turn"}}),
                    ("turn/completed", {"turn": {"id": "turn"}}),
                    ("error", {"willRetry": False}),
                ):
                    retry()
                    emit(method, **payload)
                    self.assertFalse(notice.is_visible(), method)
                    emit("model/safetyBuffering/updated", showBufferingUi=True)
                    emit(method, **payload)
                    self.assertFalse(notice.is_visible(), method)
                emit("model/safetyBuffering/updated", showBufferingUi=True)
                emit("model/safetyBuffering/updated", showBufferingUi=False)
                self.assertFalse(notice.is_visible())
                retry()
                page.evaluate("document.addEventListener('hitch:session-ended', e => e.preventDefault())")
                page.evaluate("streamHandlers.end()")
                self.assertFalse(notice.is_visible())
                self.assertEqual(errors, [])
            finally:
                browser.close()
