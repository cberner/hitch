"""Goal controls stay compact and preserve conversation state."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from django.conf import settings
from django.test import TestCase
from django.urls import reverse

from hitch.main.models import CodexInstance, SessionMetadata
from hitch.main.test.support import _make_model, _setup_codex
from hitch.main.test.test_session_goals import goal_snapshot
from hitch.main.test.views_helpers import _basic_session_rollout_lines, _make_rollout
from hitch.main.views import common


class SessionGoalUiTests(TestCase):
    @patch.object(common, "_SESSION_HISTORY_MIN_BYTES", 1)
    @patch("hitch.main.runtime.codex_pool.worker_is_alive", return_value=True)
    @patch("hitch.main.runtime.session_goals.current_goal")
    @patch("hitch.main.caches._cached_models_for_session_detail")
    @patch("hitch.main.views.common.Codex")
    def test_goal_controls_mobile_desktop_and_streaming(
        self, codex: MagicMock, catalog: MagicMock, goal: MagicMock, _alive: MagicMock,
    ) -> None:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import Route, sync_playwright

        catalog.return_value = [_make_model("first")]
        _setup_codex(codex, models=catalog.return_value)
        SessionMetadata.objects.create(
            thread_id="session", cwd="/repo",
            codex_path=str(_make_rollout(self, _basic_session_rollout_lines("Prompt", "Response"))),
        )
        snapshot = goal_snapshot(objective="Finish the Chen formalization and verify every remaining estimate. " * 5)
        goal.return_value = snapshot
        url = reverse("session", args=["session"])
        response = self.client.get(url)
        self.assertTrue(response.context["history_partial"])
        self.assertEqual(response.context["session_goal"], snapshot)
        html = idle_html = response.content.decode()
        read_only = common._render_session_detail(response.wsgi_request, "session", read_only=True)
        self.assertNotContains(read_only, "data-goal-form>")
        self.assertNotContains(read_only, 'data-goal-action="pause" hidden>Pause')
        CodexInstance.objects.create(thread_id="session", status=CodexInstance.STATUS_RUNNING, pid=1)
        active_html = self.client.get(url).content.decode()
        goal.return_value = None
        unpreserved_html = self.client.get(url).content.decode()
        goal.return_value = snapshot
        result: dict[str, Any] = {"goal": snapshot}
        fail = False

        def route_request(route: Route) -> None:
            if "/static/" in route.request.url:
                path = route.request.url.split("/static/", 1)[1]
                route.fulfill(path=str(Path(settings.BASE_DIR) / "hitch/main/static" / path))
            elif "/goal/" in route.request.url:
                route.fulfill(status=409 if fail else 200, json={"error": "Try again shortly."} if fail else result)
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
                panel = page.locator("[data-live-goal]")
                self.assertTrue(panel.is_visible())
                self.assertFalse(page.locator("[data-goal-new]").is_visible())
                draft = page.locator("[data-composer-input]")
                draft.fill("Keep this draft")
                panel.locator("summary").click()
                self.assertIn("2h 0m worked · 120 / 1,000 tokens", panel.inner_text())
                for width in [375, 1280]:
                    page.set_viewport_size({"width": width, "height": 812})
                    box = panel.bounding_box()
                    composer = page.locator(".composer").bounding_box()
                    assert box is not None and composer is not None
                    self.assertGreaterEqual(box["x"], 0)
                    self.assertLessEqual(box["x"] + box["width"], width)
                    self.assertLessEqual(box["y"] + box["height"], composer["y"])
                    page.screenshot(path=f"/tmp/hitch-goal-ui-{width}.png")
                panel.get_by_role("button", name="Edit budget", exact=True).click()
                dialog = page.locator("[data-goal-dialog]")
                dialog.locator("[name='token_budget']").fill("2000")
                fail = True
                dialog.get_by_role("button", name="Save changes").click()
                dialog.get_by_role("alert").wait_for(state="visible")
                self.assertEqual(dialog.locator("[name='token_budget']").input_value(), "2000")
                fail = False
                result = {"goal": {**snapshot, "tokenBudget": 2000}}
                dialog.get_by_role("button", name="Save changes").click()
                dialog.wait_for(state="hidden")
                self.assertTrue(panel.evaluate("el => el.open"))
                self.assertIn("2,000 tokens", panel.inner_text())
                self.assertEqual(draft.input_value(), "Keep this draft")
                page.reload()
                self.assertTrue(panel.evaluate("el => el.open"))
                draft.fill("Keep this draft")
                agent = page.locator("[data-agent-select]")
                agent.locator("option[value='child']").wait_for(state="attached")
                agent.select_option("child")
                page.get_by_text("Child response", exact=True).wait_for()
                self.assertFalse(panel.is_visible())
                agent.select_option("")
                self.assertTrue(panel.is_visible())
                panel.get_by_role("button", name="Clear", exact=True).click()
                result = {"goal": None}
                dialog.get_by_role("button", name="Clear goal").click()
                panel.wait_for(state="hidden")
                page.locator("[data-session-menu-open]").click()
                page.locator("[data-goal-new]").click()
                self.assertEqual(dialog.locator("[name='objective']").input_value(), "")
                dialog.get_by_role("button", name="Cancel").click()
                html = active_html
                page.reload()
                draft.fill("Keep this draft")
                panel.locator("summary").click()
                panel.get_by_role("button", name="Edit budget", exact=True).click()
                self.assertFalse(dialog.locator("[name='objective']").is_visible())
                self.assertTrue(dialog.locator("[name='objective']").is_disabled())
                dialog.locator("[name='token_budget']").fill("3000")
                updated = goal_snapshot(status="budgetLimited", tokensUsed=1000)
                page.evaluate("""payload => {
                    window.streamHandlers.message({data: JSON.stringify(payload)});
                }""", {
                    "method": "thread/goal/updated", "recordedAt": 300000000,
                    "payload": {"threadId": "session", "goal": updated},
                })
                self.assertEqual(dialog.locator("[name='token_budget']").input_value(), "3000")
                self.assertIn("Token budget reached", panel.inner_text())
                self.assertTrue(panel.get_by_role("button", name="Resume", exact=True).is_disabled())
                page.evaluate("window.streamHandlers.end({})")
                self.assertTrue(dialog.is_visible())
                self.assertEqual(dialog.locator("[name='token_budget']").input_value(), "3000")
                self.assertTrue(dialog.locator("[name='objective']").is_disabled())
                page.set_viewport_size({"width": 375, "height": 812})
                box = dialog.bounding_box()
                assert box is not None
                self.assertGreaterEqual(box["x"], 0)
                self.assertLessEqual(box["x"] + box["width"], 375)
                page.screenshot(path="/tmp/hitch-goal-dialog.png")
                html = idle_html
                with page.expect_navigation():
                    dialog.get_by_role("button", name="Cancel").click()
                self.assertTrue(panel.evaluate("el => el.open"))
                self.assertEqual(draft.input_value(), "Keep this draft")
                page.evaluate("goal => document.dispatchEvent(new CustomEvent('hitch:goal-updated', {detail: goal}))",
                              {**snapshot, "status": "complete"})
                self.assertEqual(panel.locator("button:visible").all_text_contents(), ["Clear"])
                html = unpreserved_html
                page.reload()
                page.evaluate("goal => document.dispatchEvent(new CustomEvent('hitch:goal-updated', {detail: goal}))",
                              {**snapshot, "status": "active"})
                panel.locator("summary").click()
                self.assertIn("Goal controls will be available after this turn finishes.", panel.inner_text())
                self.assertEqual(panel.locator("button:visible").count(), 0)
                self.assertEqual(errors, [])
            finally:
                browser.close()
