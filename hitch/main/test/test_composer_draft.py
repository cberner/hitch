from unittest.mock import MagicMock, patch

from django.template.loader import render_to_string
from django.test import TestCase
from django.urls import reverse

from hitch.main.models import SessionMetadata
from hitch.main.test.support import _rollout_line
from hitch.main.test.views_helpers import _make_rollout


class ComposerDraftTests(TestCase):
    @patch("hitch.main.caches._start_models_refresh_thread")
    def test_drafts_survive_updates_and_clear_on_submission(self, _refresh: MagicMock) -> None:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import Route, sync_playwright

        path = _make_rollout(self, [_rollout_line("event_msg", {"type": "user_message", "message": "Work"})])
        SessionMetadata.objects.create(thread_id="draft", codex_path=str(path))
        html = self.client.get(reverse("session", args=["draft"])).content.decode()

        def route_request(route: Route) -> None:
            if "/assets/session.js" in route.request.url:
                route.fulfill(body=render_to_string("assets/session.js"), content_type="text/javascript")
            elif route.request.resource_type == "document":
                body = html.replace("/draft/", "/other/") if "/other/" in route.request.url else html
                route.fulfill(body=body, content_type="text/html")
            else:
                route.fulfill(body="")

        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
            except PlaywrightError as exc:
                self.skipTest(f"playwright browser unavailable: {exc}")
            try:
                context = browser.new_context()
                context.route("**/*", route_request)
                context.add_init_script("""
                    window.EventSource = class extends EventTarget {
                        constructor() { super(); window.stream = this; }
                        close() {}
                    };
                """)
                page = context.new_page()
                errors: list[str] = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                url = "http://hitch.test/sessions/draft/"
                page.goto(url)
                field = page.locator("[data-composer-input]")
                draft = "Please keep this draft\n  including whitespace  "
                field.fill(draft)
                # The SSE end event is also used when Hitch starts background work.
                with page.expect_navigation():
                    page.evaluate("stream.dispatchEvent(new Event('end'))")
                self.assertEqual(field.input_value(), draft)
                self.assertTrue(page.locator("[data-composer-submit]").is_enabled())
                other = context.new_page()
                other.goto(url)
                self.assertEqual(other.locator("[data-composer-input]").input_value(), "")
                other.close()
                page.goto("http://hitch.test/sessions/other/")
                self.assertEqual(field.input_value(), "")
                field.fill("Another session")
                page.goto(url)
                self.assertEqual(field.input_value(), draft)
                page.reload()
                self.assertEqual(field.input_value(), draft)
                field.fill("")
                page.reload()
                self.assertEqual(field.input_value(), "")
                field.fill(draft)
                with page.expect_navigation():
                    page.locator("[data-composer-submit]").click()
                self.assertEqual(field.input_value(), "")
                page.reload()
                self.assertEqual(field.input_value(), "")
                self.assertEqual(errors, [])
                # Restricted storage must not prevent ordinary composing/sending.
                page.add_init_script("""
                    Object.defineProperty(window, 'sessionStorage', {
                        get() { throw new Error('Storage blocked'); }
                    });
                """)
                page.reload()
                field.fill("Still usable")
                self.assertTrue(page.locator("[data-composer-submit]").is_enabled())
                self.assertEqual(errors, [])
            finally:
                browser.close()
