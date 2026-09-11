"""Session model selection, worker delivery, and persistence across turns."""

import json
import signal
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, override
from unittest.mock import MagicMock, patch

from django.conf import settings
from django.test import TestCase, override_settings
from django.urls import reverse

from hitch.main.management.commands import codex_worker
from hitch.main.models import CodexInstance, SessionMetadata
from hitch.main.runtime import codex_pool
from hitch.main.test.support import _make_model, _setup_codex
from hitch.main.test.views_helpers import _basic_session_rollout_lines, _make_rollout
from hitch.main.views import common


class SessionModelTests(TestCase):
    @override
    def setUp(self) -> None:
        self.metadata = SessionMetadata.objects.create(thread_id="session", cwd="/repo")
        self.url = reverse("set_session_model", args=["session"])
        self.models = [_make_model("first"), _make_model("second")]
        catalog = patch("hitch.main.caches._fetch_models_data", return_value=self.models)
        self.catalog = catalog.start()
        self.addCleanup(catalog.stop)

    def test_validates_and_saves_atomic_pair_without_changing_defaults(self) -> None:
        self.assertEqual(self.client.get(self.url).status_code, 405)
        for model, effort in (("missing", "high"), ("first", "ultra"), ("", "low"), ("x" * 257, "low")):
            with self.subTest(model=model, effort=effort):
                self.assertEqual(self.client.post(self.url, {
                    "model": model, "reasoning_effort": effort,
                }).status_code, 400)
                self.metadata.refresh_from_db()
                self.assertEqual(self.metadata.model, "")
        response = self.client.post(self.url, {"model": " second ", "reasoning_effort": ""})
        self.assertEqual(response.status_code, 302)
        self.metadata.refresh_from_db()
        self.assertEqual((self.metadata.model, self.metadata.reasoning_effort), ("second", "medium"))
        self.assertNotIn("hitch_model", response.cookies)

    @patch("hitch.main.caches._cached_models_data", return_value=[])
    def test_catalog_outage_does_not_save_unvalidated_pair(self, _cache: MagicMock) -> None:
        self.catalog.side_effect = RuntimeError("offline")
        self.assertEqual(self.client.post(self.url, {"model": "first"}).status_code, 503)
        self.metadata.refresh_from_db()
        self.assertEqual(self.metadata.model, "")

    @patch("hitch.main.views.session_actions._read_thread_cwd", return_value=None)
    def test_unknown_session_is_not_created(self, read_cwd: MagicMock) -> None:
        self.metadata.delete()
        self.assertEqual(self.client.post(self.url, {"model": "first"}).status_code, 400)
        read_cwd.assert_called_once()
        self.assertFalse(SessionMetadata.objects.exists())

    @patch("hitch.main.runtime.codex_pool.update_instance_model")
    def test_live_update_reports_confirmation_and_retains_pending_settings(self, update: MagicMock) -> None:
        active = CodexInstance.objects.create(thread_id="session", status=CodexInstance.STATUS_RUNNING, pid=1)
        CodexInstance.objects.create(thread_id="another", status=CodexInstance.STATUS_RUNNING, pid=2)
        for applied in (True, False):
            with self.subTest(applied=applied):
                update.reset_mock()
                update.return_value = applied
                response = self.client.post(self.url, {"model": "first", "reasoning_effort": "high"},
                                            HTTP_X_REQUESTED_WITH="XMLHttpRequest")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["pending"], not applied)
                update.assert_called_once_with(active, model="first", reasoning_effort="high")
                self.metadata.refresh_from_db()
                self.assertEqual((self.metadata.model, self.metadata.reasoning_effort), ("first", "high"))

    @patch("hitch.main.runtime.codex_pool._launch_worker_process", return_value=SimpleNamespace(pid=0))
    def test_override_reaches_normal_plan_and_review_workers(self, launch: MagicMock) -> None:
        self.metadata.model, self.metadata.reasoning_effort = "second", "high"
        self.metadata.save()
        with tempfile.TemporaryDirectory() as raw, override_settings(CODEX_EVENTS_DIR=Path(raw)):
            for extra in ({}, {"plan_mode": True}, {"agent_kind": "code_review"}):
                with self.subTest(extra=extra):
                    instance = codex_pool.spawn_turn(
                        thread_id="session", cwd="/repo", prompt="Continue", model="old",
                        reasoning_effort="low", stored_model="stale", stored_reasoning_effort="medium", **extra,
                    )
                    self.assertEqual((instance.model, instance.reasoning_effort), ("second", "high"))
                    self.assertEqual(launch.call_args.kwargs["model"], "second")
                    self.assertEqual(launch.call_args.kwargs["reasoning_effort"], "high")

    @patch("hitch.main.caches._start_models_refresh_thread")
    @patch("hitch.main.runtime.codex_pool.worker_is_alive", return_value=True)
    @patch("hitch.main.views.common.Codex")
    def test_page_distinguishes_pending_and_active_pair(self, codex: MagicMock, *_mocks: MagicMock) -> None:
        _setup_codex(codex, models=self.models)
        path = _make_rollout(self, _basic_session_rollout_lines("Prompt", "Response"))
        self.metadata.codex_path = str(path)
        self.metadata.model, self.metadata.reasoning_effort = "second", "high"
        self.metadata.save()
        active = CodexInstance.objects.create(
            thread_id="session", status=CodexInstance.STATUS_RUNNING,
            model="first", reasoning_effort="low", pid=123,
        )
        url = reverse("session", args=["session"])
        response = self.client.get(url)
        self.assertContains(response, "data-model-settings-open")
        self.assertEqual((response.context["session_model"], response.context["session_reasoning"]), ("first", "low"))
        self.assertTrue(response.context["session_model_pending"])
        next_message = response.context["next_message_config"]
        self.assertEqual((next_message[0]["value"], next_message[1]["plan_value"]), ("second", "high"))
        readonly = common._render_session_detail(response.wsgi_request, "session", read_only=True)
        self.assertNotContains(readonly, "data-model-settings-form>")
        active.model, active.reasoning_effort = "second", "high"
        active.save()
        self.assertFalse(self.client.get(url).context["session_model_pending"])
        active.status = CodexInstance.STATUS_COMPLETED
        active.save()
        response = self.client.get(url)
        self.assertEqual((response.context["session_model"], response.context["session_reasoning"]), ("second", "high"))

    @patch("hitch.main.caches._cached_models_for_session_detail")
    @patch("hitch.main.views.common.Codex")
    def test_dialog_browser_filters_effort_and_preserves_draft(self, codex: MagicMock, catalog: MagicMock) -> None:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import Route, sync_playwright

        _setup_codex(codex, models=self.models)
        self.models[1].supported_reasoning_efforts = self.models[1].supported_reasoning_efforts[:2]
        catalog.return_value = self.models
        self.metadata.codex_path = str(_make_rollout(self, _basic_session_rollout_lines("Prompt", "Response")))
        self.metadata.model, self.metadata.reasoning_effort = "first", "high"
        self.metadata.save()
        html = self.client.get(reverse("session", args=["session"])).content.decode()
        state = {"fail": True, "pending": False}
        submissions: list[str] = []

        def route_request(route: Route) -> None:
            url = route.request.url
            if "/static/" in url:
                route.fulfill(path=str(Path(settings.BASE_DIR) / "hitch/main/static" / url.split("/static/", 1)[1]))
            elif "/model/" in url:
                submissions.append(route.request.post_data or "")
                if state["fail"]:
                    route.fulfill(status=503, body="Model list unavailable. Try again shortly.")
                else:
                    route.fulfill(json={"model": "second", "effort": "medium", "pending": state["pending"],
                                        "message": "Saved for this session."})
            elif "/agents/" in url:
                data: dict[str, Any] = {"agents": [
                    {"id": "child", "name": "Reviewer", "role": "reviewer", "parent_id": "session"},
                ]}
                if "agent=child" in url:
                    data.update(selected="child", name="Reviewer", role="reviewer", html="Child response",
                                next_url="", partial=False)
                route.fulfill(json=data)
            else:
                route.fulfill(content_type="text/html", body=html)

        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
            except PlaywrightError as exc:
                self.skipTest(f"playwright browser unavailable: {exc}")
            try:
                page = browser.new_page(viewport={"width": 375, "height": 812})
                page.add_init_script("window.EventSource = class { addEventListener() {} close() {} };")
                page.route("**/*", route_request)
                page.goto("http://hitch.test/sessions/session/")
                draft = page.locator(".composer textarea[name='prompt']")
                draft.fill("Keep this draft")
                agent = page.locator("[data-agent-select]")
                agent.locator("option[value='child']").wait_for(state="attached")
                agent.select_option("child")
                page.get_by_text("Child response", exact=True).wait_for()
                page.locator("[data-session-menu-open]").click()
                page.locator("[data-model-settings-open]").click()
                dialog = page.locator("[data-model-settings-dialog]")
                model = dialog.locator("[name='model']")
                effort = dialog.locator("[name='reasoning_effort']")
                self.assertEqual((model.input_value(), effort.input_value()), ("first", "high"))
                model.select_option("second")
                self.assertEqual(effort.input_value(), "")
                self.assertTrue(effort.locator("option[value='high']").evaluate("el => el.disabled && el.hidden"))
                dialog.get_by_role("button", name="Save", exact=True).click()
                dialog.get_by_role("alert").wait_for(state="visible")
                self.assertIn("unavailable", dialog.get_by_role("alert").inner_text())
                state["fail"] = False
                state["pending"] = True
                dialog.get_by_role("button", name="Save", exact=True).click()
                dialog.wait_for(state="hidden")
                self.assertEqual(page.locator("[data-session-model]").inner_text(), "first")
                self.assertTrue(page.locator("[data-model-settings-pending]").is_visible())
                self.assertTrue(page.locator("[data-model-settings-status]").is_visible())
                page.locator("[data-session-menu-open]").click()
                page.locator("[data-model-settings-open]").click()
                state["pending"] = False
                dialog.get_by_role("button", name="Save", exact=True).click()
                dialog.wait_for(state="hidden")
                self.assertEqual(page.locator("[data-session-model]").inner_text(), "second")
                self.assertFalse(page.locator("[data-model-settings-pending]").is_visible())
                self.assertTrue(page.locator("[data-model-settings-status]").is_visible())
                self.assertEqual(agent.input_value(), "child")
                self.assertTrue(page.get_by_text("Child response", exact=True).is_visible())
                self.assertEqual(draft.input_value(), "Keep this draft")
                agent.select_option("")
                draft.fill("/plan Keep this draft")
                for setting, value in (("model", "second"), ("reasoning", "medium")):
                    self.assertEqual(
                        page.locator(f"[data-setting='{setting}'] [data-next-message-value]").inner_text(), value,
                    )
                self.assertEqual(len(submissions), 3)
                self.assertLessEqual(page.evaluate("document.documentElement.scrollWidth"), 375)
                self.assertTrue(page.get_by_text("Response", exact=True).is_visible())
            finally:
                browser.close()


class ModelControlTests(TestCase):
    @patch("hitch.main.runtime.codex_pool._pid_is_instance_worker", return_value=True)
    @patch("hitch.main.runtime.codex_pool.os.kill")
    def test_worker_applies_queued_settings_and_acknowledges_outcome(self, kill: MagicMock, _pid: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as raw:
            instance = CodexInstance.objects.create(
                thread_id="session", status=CodexInstance.STATUS_RUNNING, pid=4321,
                events_path=str(Path(raw) / "events.jsonl"), model="old", reasoning_effort="low",
            )
            turn = MagicMock(thread_id="session", id="turn-1")
            append = codex_pool._append_control_request

            def deliver(instance: CodexInstance, payload: dict[str, Any]) -> None:
                append(instance, payload)
                codex_worker._drain_steer_requests(
                    turn, instance=instance, control_path=codex_pool.control_path_for(instance), control_offset=0,
                )

            for response in ({"status": "applied"}, {"status": "targetUnavailable"}, RuntimeError("unsupported")):
                with (
                    self.subTest(response=response),
                    patch.object(codex_pool, "_append_control_request", side_effect=deliver),
                ):
                    instance.model, instance.reasoning_effort = "old", "low"
                    instance.save()
                    codex_pool.control_path_for(instance).write_text("")
                    turn._client._request_raw.reset_mock(side_effect=True)
                    if isinstance(response, Exception):
                        turn._client._request_raw.side_effect = response
                    else:
                        turn._client._request_raw.return_value = response
                    applied = codex_pool.update_instance_model(instance, model="new", reasoning_effort="ultra")
                    self.assertEqual(applied, response == {"status": "applied"})
                    turn._client._request_raw.assert_called_once_with("turn/settings/update", {
                        "threadId": "session", "turnId": "turn-1", "model": "new", "effort": "ultra",
                    })
                    instance.refresh_from_db()
                    self.assertEqual(instance.model, "new" if applied else "old")
                    kill.assert_called_with(4321, signal.SIGUSR1)
                    turn.steer.assert_not_called()

    @patch("hitch.main.runtime.codex_pool.time.monotonic", side_effect=[0, 3])
    @patch("hitch.main.runtime.codex_pool.os.kill")
    def test_starting_worker_queues_without_signal_or_false_confirmation(
        self, kill: MagicMock, _time: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            instance = CodexInstance.objects.create(
                thread_id="session", status=CodexInstance.STATUS_STARTING, pid=0,
                events_path=str(Path(raw) / "events.jsonl"),
            )
            self.assertFalse(codex_pool.update_instance_model(instance, model="new", reasoning_effort="high"))
            payload = json.loads(codex_pool.control_path_for(instance).read_text())
            self.assertEqual((payload["model"], payload["effort"]), ("new", "high"))
            kill.assert_not_called()
