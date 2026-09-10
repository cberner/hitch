from datetime import timedelta
from typing import Any, override
from unittest.mock import patch

from django.core import signing
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from hitch.main import caches
from hitch.main.models import ArchivedSessionTokenUsage, SessionMetadata
from hitch.main.sessions import session_index, token_usage
from hitch.main.test.support import _setup_codex
from hitch.main.test.views_helpers import (
    _cache_token_usage,
    _make_rollout,
    _seed_usage_metadata,
    _session,
    _token_count_line,
)
from hitch.main.views import common


class UsageRefreshTests(TestCase):
    @override
    def setUp(self) -> None:
        super().setUp()
        self.enterContext(patch.object(caches, "_start_models_refresh_thread"))
        self.enterContext(
            patch.object(
                caches, "_rate_limits_for_usage_context", return_value=caches._RateLimitsUsageState(None, False)
            )
        )
        self.start_tokens = self.enterContext(patch.object(token_usage, "_start_usage_token_refresh_thread"))
        self.start_index = self.enterContext(patch.object(common, "_start_usage_session_index_refresh_thread"))
        self.enterContext(patch.object(common, "_USAGE_SESSION_INDEX_REFRESH_RESULTS", {}))
        self.enterContext(patch.object(token_usage, "_USAGE_TOKEN_REFRESH_FINISHED_AT", None))
        self.enterContext(patch.object(token_usage, "_USAGE_TOKEN_REFRESH_THREAD_IDS", frozenset()))
        self.enterContext(patch.object(token_usage, "_USAGE_TOKEN_REFRESH_FAILED", False))
        self.enterContext(patch.object(token_usage, "_USAGE_TOKEN_REFRESH_IN_FLIGHT", False))

    def poll(self, cursor: str, *, profile: bool = False) -> dict[str, Any]:
        with self.captureOnCommitCallbacks(execute=True):
            response = self.client.get(reverse("usage_refresh"), {"cursor": cursor, "profile": "1" if profile else "0"})
        self.assertEqual(response["Cache-Control"], "no-store")
        result: dict[str, Any] = response.json()
        return result

    def test_poll_updates_real_token_counts_and_stops_after_sweep(self) -> None:
        path = _make_rollout(
            self, [_token_count_line(input_tokens=400, cached_input_tokens=50, output_tokens=600, total_tokens=1000)]
        )
        row = _seed_usage_metadata("usage", path=path)
        _cache_token_usage("usage", input_tokens=10, cached_input_tokens=0, output_tokens=20, total_tokens=30, path="")
        with (
            self.captureOnCommitCallbacks(execute=True),
            patch.object(token_usage, "_rollout_file_state_from_value") as stat,
        ):
            initial = self.client.get(reverse("usage"))
        stat.assert_not_called()
        self.assertTrue(initial.context["usage_should_poll"])
        cursor = initial.context["usage_cursor"]
        pending = self.poll(cursor)
        self.assertTrue(pending["should_poll"])
        token_usage._refresh_usage_token_cache_best_effort(token_usage._usage_token_refresh_candidates([row]))
        result = self.poll(pending["cursor"])
        self.assertFalse(result["should_poll"])
        self.assertIn("600", result["html"])
        self.assertNotIn("Refreshing session token usage", result["html"])
        self.assertEqual(ArchivedSessionTokenUsage.objects.get(thread_id="usage").total_tokens, 1000)
        # A slow poll or a later quota refresh cannot restart the completed sweep.
        self.start_tokens.reset_mock()
        SessionMetadata.objects.update(usage_last_checked_at=timezone.now() - timedelta(minutes=5))
        self.assertFalse(self.poll(result["cursor"])["should_poll"])
        self.start_tokens.assert_not_called()

    def test_index_poll_waits_for_both_sources_then_refreshes_new_rows(self) -> None:
        initial = self.client.get(reverse("usage"))
        cursor = initial.context["usage_cursor"]
        common._USAGE_SESSION_INDEX_REFRESH_RESULTS[False] = (timezone.now(), False)
        self.assertTrue(self.poll(cursor)["should_poll"])
        self.start_index.assert_called_with(enable_memories=False, include_active=False, include_archived=True)
        path = _make_rollout(self, ["{}"])
        row = _seed_usage_metadata("newly-indexed", path=path)
        common._USAGE_SESSION_INDEX_REFRESH_RESULTS[True] = (timezone.now(), False)
        pending = self.poll(cursor)
        self.assertTrue(pending["should_poll"])
        self.start_tokens.assert_called()
        token_usage._refresh_usage_token_cache_best_effort(token_usage._usage_token_refresh_candidates([row]))
        self.assertFalse(self.poll(pending["cursor"])["should_poll"])

    def test_completion_during_render_cannot_certify_older_counts(self) -> None:
        path = _make_rollout(
            self,
            [
                _token_count_line(
                    input_tokens=400,
                    cached_input_tokens=50,
                    output_tokens=600,
                    total_tokens=1000,
                )
            ],
        )
        row = _seed_usage_metadata("racing", path=path)
        _cache_token_usage("racing", input_tokens=10, cached_input_tokens=0, output_tokens=20, total_tokens=30)
        initial = self.client.get(reverse("usage"))
        aggregate = token_usage._lifetime_token_usage_for_metadata

        def finish_after_read(*args: Any, **kwargs: Any) -> dict[str, Any]:
            old = aggregate(*args, **kwargs)
            token_usage._refresh_usage_token_cache_best_effort(token_usage._usage_token_refresh_candidates([row]))
            return old

        with patch.object(token_usage, "_lifetime_token_usage_for_metadata", side_effect=finish_after_read):
            result = self.poll(initial.context["usage_cursor"])
        self.assertTrue(result["should_poll"])
        result = self.poll(result["cursor"])
        self.assertFalse(result["should_poll"])
        self.assertIn("600", result["html"])

    def test_cold_index_worker_establishes_coverage_and_starts_tokens(self) -> None:
        path = _make_rollout(
            self,
            [
                _token_count_line(
                    input_tokens=400,
                    cached_input_tokens=50,
                    output_tokens=600,
                    total_tokens=1000,
                )
            ],
        )
        initial = self.client.get(reverse("usage"))
        self.assertIsNone(initial.context["lifetime_usage"])
        with patch("hitch.main.views.common.Codex") as codex:
            client = _setup_codex(codex, threads=[_session("cold", path=str(path))])
            common._refresh_usage_session_index_best_effort(
                enable_memories=False,
                include_active=True,
                include_archived=True,
            )
        self.assertTrue(all(call.kwargs["use_state_db_only"] for call in client.thread_list.call_args_list))
        self.assertTrue(session_index.is_complete(archived=False))
        self.assertTrue(session_index.is_complete(archived=True))
        pending = self.poll(initial.context["usage_cursor"])
        self.assertTrue(pending["should_poll"])
        token_usage._refresh_usage_token_cache_best_effort(self.start_tokens.call_args.args[0])
        result = self.poll(pending["cursor"])
        self.assertFalse(result["should_poll"])
        self.assertIn("600", result["html"])

    def test_expired_cursor_restarts_stale_usage_but_invalid_signature_fails(self) -> None:
        row = _seed_usage_metadata("expired")
        _cache_token_usage(row.thread_id, input_tokens=12, cached_input_tokens=0, output_tokens=23, total_tokens=35)
        with patch("django.core.signing.time.time", return_value=timezone.now().timestamp() - 7200):
            cursor = signing.dumps({"phase": "done", "status": "fresh"}, salt="usage-refresh")
        pending = self.poll(cursor)
        self.assertTrue(pending["should_poll"])
        self.start_tokens.assert_called_once()
        self.start_tokens.reset_mock()
        self.assertFalse(self.poll("invalid-signature")["should_poll"])
        self.start_tokens.assert_not_called()

    def test_historical_failed_sweep_does_not_poison_fresh_view(self) -> None:
        path = _make_rollout(self, ["{}"])
        row = _seed_usage_metadata("fresh", path=path)
        _cache_token_usage(
            row.thread_id, input_tokens=0, cached_input_tokens=0, output_tokens=0, total_tokens=0, path=path
        )
        SessionMetadata.objects.update(usage_last_checked_at=timezone.now())
        token_usage._USAGE_TOKEN_REFRESH_FAILED = True
        token_usage._USAGE_TOKEN_REFRESH_FINISHED_AT = timezone.now() - timedelta(seconds=5)
        response = self.client.get(reverse("usage"))
        self.assertEqual(response.context["usage_status"], "fresh")
        self.assertFalse(response.context["usage_should_poll"])

    def test_joined_sweep_rechecks_previously_observed_omitted_rows(self) -> None:
        path = _make_rollout(self, ["{}"])
        included = _seed_usage_metadata("included", path=path)
        omitted = _seed_usage_metadata("omitted", path=path)
        SessionMetadata.objects.update(usage_last_checked_at=timezone.now() - timedelta(minutes=5))
        # Hide an existing row from the index without clearing its check time.
        SessionMetadata.objects.filter(pk=omitted.pk).update(codex_updated_at=None)
        initial = self.client.get(reverse("usage"))
        token_usage._refresh_usage_token_cache_best_effort(token_usage._usage_token_refresh_candidates([included]))
        SessionMetadata.objects.filter(pk=omitted.pk).update(codex_updated_at=timezone.now())
        pending = self.poll(initial.context["usage_cursor"])
        self.assertTrue(pending["should_poll"])
        self.assertEqual({item.thread_id for item in self.start_tokens.call_args.args[0]}, {"included", "omitted"})
        token_usage._refresh_usage_token_cache_best_effort(self.start_tokens.call_args.args[0])
        # Earlier rows in a slow sweep need not be rechecked merely because
        # their ordinary TTL expired while later rows were being processed.
        with patch("hitch.main.views.common.timezone.now", return_value=timezone.now() + timedelta(minutes=5)):
            result = self.poll(pending["cursor"])
        self.assertFalse(result["should_poll"])

    def test_failed_joined_sweep_still_covers_omitted_rows_and_stops(self) -> None:
        broken = _seed_usage_metadata("broken", path=_make_rollout(self, ["broken json"]))
        omitted = _seed_usage_metadata(
            "omitted",
            path=_make_rollout(
                self,
                [_token_count_line(input_tokens=400, cached_input_tokens=50, output_tokens=600, total_tokens=1000)],
            ),
        )
        SessionMetadata.objects.filter(pk=omitted.pk).update(
            codex_updated_at=None, usage_last_checked_at=timezone.now() - timedelta(minutes=5)
        )
        initial = self.client.get(reverse("usage"))
        token_usage._refresh_usage_token_cache_best_effort(token_usage._usage_token_refresh_candidates([broken]))
        self.assertTrue(token_usage._USAGE_TOKEN_REFRESH_FAILED)
        SessionMetadata.objects.filter(pk=omitted.pk).update(codex_updated_at=timezone.now())
        pending = self.poll(initial.context["usage_cursor"])
        self.assertTrue(pending["should_poll"])
        token_usage._refresh_usage_token_cache_best_effort(self.start_tokens.call_args.args[0])
        result = self.poll(pending["cursor"])
        self.assertFalse(result["should_poll"])
        self.assertEqual(signing.loads(result["cursor"], salt="usage-refresh")["status"], "failed")
        self.assertEqual(ArchivedSessionTokenUsage.objects.get(thread_id="omitted").total_tokens, 1000)
        self.assertIn("600", result["html"])
        self.start_tokens.reset_mock()
        self.assertFalse(self.poll(result["cursor"])["should_poll"])
        self.start_tokens.assert_not_called()

    def test_sweep_failure_before_row_checks_is_terminal(self) -> None:
        row = _seed_usage_metadata("failed")
        initial = self.client.get(reverse("usage"))
        with patch.object(token_usage, "_usage_token_refresh_work_batches", side_effect=RuntimeError("unavailable")):
            token_usage._refresh_usage_token_cache_best_effort(token_usage._usage_token_refresh_candidates([row]))
        result = self.poll(initial.context["usage_cursor"])
        self.assertFalse(result["should_poll"])
        self.assertEqual(signing.loads(result["cursor"], salt="usage-refresh")["status"], "failed")

    def test_failed_index_and_unavailable_coverage_stop_polling(self) -> None:
        initial = self.client.get(reverse("usage"))
        cursor = initial.context["usage_cursor"]
        for failed, status in ((False, "unavailable"), (True, "failed")):
            with self.subTest(failed=failed):
                common._USAGE_SESSION_INDEX_REFRESH_RESULTS.update(
                    {source: (timezone.now(), failed) for source in (False, True)}
                )
                result = self.poll(cursor)
                self.assertFalse(result["should_poll"])
                state = signing.loads(result["cursor"], salt="usage-refresh")
                self.assertEqual(state["status"], status)

    def test_missing_and_corrupt_rollouts_finish_with_partial_or_failed_status(self) -> None:
        for text, missing, status in (("{}", True, "partial-checked"), ("broken json", False, "failed")):
            with self.subTest(status=status):
                path = _make_rollout(self, [text])
                row = _seed_usage_metadata(status, path=path)
                _cache_token_usage(
                    status, input_tokens=12, cached_input_tokens=0, output_tokens=23, total_tokens=35, path=""
                )
                initial = self.client.get(reverse("usage"))
                if missing:
                    path.unlink()
                with (
                    patch.object(token_usage, "_refresh_missing_usage_metadata_path", return_value=None),
                    patch("hitch.main.sessions.token_usage.app_server_pool.borrow_codex"),
                ):
                    token_usage._refresh_usage_token_cache_best_effort(
                        token_usage._usage_token_refresh_candidates([row])
                    )
                result = self.poll(initial.context["usage_cursor"])
                self.assertFalse(result["should_poll"])
                self.assertEqual(signing.loads(result["cursor"], salt="usage-refresh")["status"], status)
                self.assertIn("last-known or partial", result["html"])
                ArchivedSessionTokenUsage.objects.all().delete()
                SessionMetadata.objects.all().delete()

    def test_profile_fragment_keeps_project_scope_and_quota_completion(self) -> None:
        from hitch.main.test.support import _make_project, _seed_cookies
        from hitch.main.test.views_helpers import _SELECTED_PROJECT_COOKIE

        project = _make_project()
        row = _seed_usage_metadata("project", project=project)
        _cache_token_usage(row.thread_id, input_tokens=12, cached_input_tokens=0, output_tokens=23, total_tokens=35)
        SessionMetadata.objects.update(usage_last_checked_at=timezone.now())
        _seed_cookies(self.client, **{_SELECTED_PROJECT_COOKIE: str(project.pk)})
        with patch.object(
            caches, "_rate_limits_for_usage_context", return_value=caches._RateLimitsUsageState(None, True)
        ):
            initial = self.client.get(reverse("profile"))
        self.assertTrue(initial.context["usage_should_poll"])
        result = self.poll(initial.context["usage_cursor"], profile=True)
        self.assertFalse(result["should_poll"])
        self.assertIn('data-profile="1"', result["html"])
        self.assertIn("project-usage-card", result["html"])
        self.assertNotIn("project-usage-card", self.poll(initial.context["usage_cursor"])["html"])

    def test_list_fragment_updates_names_and_omits_page_scripts(self) -> None:
        row = _seed_usage_metadata("list")
        session_index.update_cached_name(row.thread_id, "Before refresh")
        response = self.client.get(reverse("index"), HTTP_X_HITCH_REFRESH="sessions")
        self.assertContains(response, "Before refresh")
        session_index.update_cached_name(row.thread_id, "After refresh")
        response = self.client.get(reverse("index"), HTTP_X_HITCH_REFRESH="sessions")
        self.assertContains(response, "After refresh")
        self.assertNotContains(response, "<script>")
        self.assertEqual(response["Cache-Control"], "no-store")

    def test_browser_refreshes_usage_and_session_rows_without_interrupting_edits(self) -> None:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright

        from hitch.main.test.support import _seed_cookies
        from hitch.main.test.views_helpers import _SHOW_ARCHIVED_COOKIE

        path = _make_rollout(
            self, [_token_count_line(input_tokens=400, cached_input_tokens=50, output_tokens=600, total_tokens=1000)]
        )
        row = _seed_usage_metadata("browser", path=path)
        _cache_token_usage(
            "browser", input_tokens=10, cached_input_tokens=0, output_tokens=20, total_tokens=30, path=""
        )
        usage = self.client.get(reverse("usage"))
        token_usage._refresh_usage_token_cache_best_effort(token_usage._usage_token_refresh_candidates([row]))
        usage_result = self.poll(usage.context["usage_cursor"])
        other = _seed_usage_metadata("other-browser", path=path)
        other.codex_updated_at = (row.codex_updated_at or timezone.now()) - timedelta(days=1)
        other.save(update_fields=["codex_updated_at"])
        session_index.update_cached_name(other.thread_id, "Other original")
        session_index.update_cached_name(row.thread_id, "Original title")
        index_html = self.client.get(reverse("index")).content.decode()
        session_index.update_cached_name(row.thread_id, "Updated title")
        session_index.update_cached_name(other.thread_id, "Other updated")
        list_html = self.client.get(reverse("index"), HTTP_X_HITCH_REFRESH="sessions").content.decode()
        session_index.update_cached_name(other.thread_id, "Other during Undo")
        session_index.update_cached_archived(row.thread_id, archived=True)
        _seed_cookies(self.client, **{_SHOW_ARCHIVED_COOKIE: "false"})
        undo_list_html = self.client.get(reverse("index"), HTTP_X_HITCH_REFRESH="sessions").content.decode()
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
            except PlaywrightError as exc:
                self.skipTest(f"playwright browser unavailable: {exc}")
            try:
                page = browser.new_page()
                errors: list[str] = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.clock.install()
                page.route("http://hitch.test/usage/", lambda route: route.fulfill(body=usage.content.decode()))
                usage_requests: list[str] = []

                def serve_usage(route: Any) -> None:
                    usage_requests.append(route.request.url)
                    route.fulfill(json=usage_result)

                page.route("http://hitch.test/usage/refresh/**", serve_usage)
                page.goto("http://hitch.test/usage/")
                page.locator("[data-lifetime-total-toggle]").click()
                page.clock.run_for(1000)
                page.wait_for_function("!document.body.textContent.includes('Refreshing session token usage')")
                self.assertIn("600", page.locator("[data-usage-root]").inner_text())
                self.assertEqual(page.locator("[data-lifetime-total-toggle]").get_attribute("aria-expanded"), "true")
                page.clock.run_for(5000)
                self.assertEqual(len(usage_requests), 1)

                list_requests: list[str] = []
                held_lists: list[Any] = []
                hold_lists = False

                def serve_list(route: Any) -> None:
                    if route.request.headers.get("x-hitch-refresh") == "sessions":
                        list_requests.append(route.request.url)
                        if hold_lists:
                            held_lists.append(route)
                        else:
                            route.fulfill(body=list_html)
                    else:
                        route.fulfill(body=index_html)

                page.route("http://hitch.test/", serve_list)
                page.route("**/archive/", lambda route: route.fulfill(status=200, body=""))
                page.goto("http://hitch.test/")
                page.locator("[data-session-menu-open]").first.click()
                page.locator("[data-session-rename-open]").first.click()
                page.locator("input[name='name']").first.fill("Unsaved edit")
                page.clock.run_for(6000)
                self.assertEqual(page.locator("input[name='name']").first.input_value(), "Unsaved edit")
                page.wait_for_function("document.body.textContent.includes('Other updated')")
                self.assertGreater(len(list_requests), 0)
                self.assertEqual(page.locator(":focus").input_value(), "Unsaved edit")
                page.evaluate("""() => {
                    document.querySelector('[data-session-rename]').hidden = true;
                    document.querySelector('[data-session-row]').classList.remove('rename-open');
                    document.activeElement.blur();
                }""")
                page.locator(".session-link").first.focus()
                page.clock.run_for(5000)
                page.wait_for_function(
                    "document.querySelector('[data-session-title]').textContent.includes('Updated title')"
                )
                self.assertEqual(page.locator(":focus").get_attribute("class"), "session-link")
                # The list reads old data after Save, but responds after the
                # rename succeeds. It must not replace the confirmed title.
                held_renames: list[Any] = []
                rename_url = "http://hitch.test" + reverse("set_session_name", kwargs={"session_id": row.thread_id})
                page.route(rename_url, lambda route: held_renames.append(route))
                page.locator("[data-session-menu-open]").first.click()
                page.locator("[data-session-rename-open]").first.click()
                page.locator("input[name='name']").first.fill("Saved title")
                page.locator("[data-session-rename-form] button").first.click()
                page.wait_for_timeout(100)
                self.assertEqual(len(held_renames), 1)
                hold_lists = True
                page.evaluate("document.dispatchEvent(new Event('visibilitychange'))")
                page.wait_for_timeout(100)
                self.assertEqual(len(held_lists), 1)
                held_renames.pop().fulfill(status=200, body="")
                page.wait_for_function("document.querySelector('[data-session-title]').textContent === 'Saved title'")
                held_lists.pop().fulfill(body=list_html)
                page.wait_for_timeout(100)
                self.assertEqual(page.locator("[data-session-title]").first.inner_text(), "Saved title")
                hold_lists = False
                # Handlers on newly inserted rows must still support archive/Undo.
                page.locator("[data-session-menu-open]").first.click()
                page.locator("[data-session-archive-form] button").first.click()
                page.wait_for_function("!document.querySelector('[data-archive-toast]').hidden")
                count = len(list_requests)
                # Refresh while the archived row is absent from the server's
                # result; its retained DOM node must still support Undo.
                list_html = undo_list_html
                page.evaluate("document.dispatchEvent(new Event('visibilitychange'))")
                page.wait_for_function("document.body.textContent.includes('Other during Undo')")
                self.assertGreater(len(list_requests), count)
                self.assertFalse(page.locator("[data-archive-toast]").is_hidden())
                page.locator("[data-archive-undo]").click()
                page.wait_for_function(
                    "!document.querySelector('[data-session-row]').classList.contains('pending-archive')"
                )
                self.assertEqual(errors, [])
            finally:
                browser.close()
