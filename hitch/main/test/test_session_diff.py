import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from django.conf import settings
from django.template.loader import render_to_string
from django.test import TestCase
from django.urls import reverse
from openai_codex.errors import InvalidRequestError

from hitch.main.git_support import GitCommandError, run_git
from hitch.main.models import CodexInstance, SessionMetadata
from hitch.main.test.support import _git, _setup_codex
from hitch.main.test.views_helpers import _basic_session_rollout_lines, _make_rollout, _session


class SessionDiffTests(TestCase):
    @patch("hitch.main.caches._start_models_refresh_thread")
    def test_diff_is_loaded_on_demand_and_reads_current_worktree(self, _refresh: MagicMock) -> None:
        path = _make_rollout(self, _basic_session_rollout_lines("Work", "Done"))
        repo = path.parent / "repo"
        repo.mkdir()
        _git(repo, "init")
        changed = repo / "example.txt"
        changed.write_text("original\n")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "Create original file for diff test")
        _git(repo, "update-ref", "refs/remotes/origin/master", "HEAD")
        SessionMetadata.objects.create(thread_id="diff", codex_path=str(path), cwd=str(repo))
        with patch("hitch.main.views.session_detail.build_worktree_diff") as build:
            response = self.client.get(reverse("session", args=["diff"]))
            self.assertContains(response, 'data-diff-url="/sessions/diff/diff/"')
            self.assertNotContains(response, 'class="diff-table"')
            build.assert_not_called()
        url = reverse("session_diff", args=["diff"])
        self.assertContains(self.client.get(url), "No changes in this working tree.")
        for command in (
            ["rev-parse", "--show-toplevel"], ["rev-parse", "--verify"], ["merge-base"],
            ["rev-list"], ["diff"], ["ls-files"],
        ):
            for failure in (GitCommandError("timed out"), subprocess.CompletedProcess([], 128, b"", b"failed")):
                with self.subTest(command=command, failure=failure):
                    def failed_read(
                        cwd: Path, args: list[str], *, command: list[str] = command,
                        failure: GitCommandError | subprocess.CompletedProcess[bytes] = failure,
                        **kwargs: Any,
                    ) -> subprocess.CompletedProcess[bytes]:
                        if args[:len(command)] == command:
                            if isinstance(failure, GitCommandError):
                                raise failure
                            return failure
                        return run_git(cwd, args, **kwargs)

                    with patch("hitch.main.diffs.run_git", side_effect=failed_read):
                        response = self.client.get(url)
                    self.assertContains(response, "Unable to read the working tree", status_code=503)
                    self.assertNotContains(response, "No changes", status_code=503)
                    self.assertIn("no-store", response.headers["Cache-Control"])
        self.assertContains(self.client.get(url), "No changes in this working tree.")
        for text in ("first change", "second <script>change</script>"):
            changed.write_text(text + "\n")
            response = self.client.get(url)
            self.assertContains(response, "example.txt")
            self.assertContains(response, text.replace("<", "&lt;").replace(">", "&gt;"))
            self.assertIn("no-store", response.headers["Cache-Control"])
        self.assertNotContains(response, "first change")
        self.assertEqual(self.client.post(url).status_code, 405)
        with patch("hitch.main.runtime.codex_pool.worker_is_alive", return_value=True), patch(
            "hitch.main.views.session_detail.build_worktree_diff"
        ) as build:
            CodexInstance.objects.create(thread_id="diff", status=CodexInstance.STATUS_RUNNING, pid=1)
            self.assertEqual(self.client.get(url).status_code, 409)
            build.assert_not_called()

    @patch("hitch.main.views.common.Codex")
    def test_unindexed_diff_uses_a_metadata_only_reader(self, mock_codex: MagicMock) -> None:
        client = _setup_codex(mock_codex)
        client._client.thread_read.return_value.thread = _session("unindexed", cwd="/missing")
        url = reverse("session_diff", args=["unindexed"])
        self.assertContains(self.client.get(url), "Unable to read the working tree", status_code=503)
        client._client.thread_read.assert_called_once_with("unindexed", include_turns=False)
        client.thread_resume.assert_not_called()
        client._client.thread_read.side_effect = InvalidRequestError(-32600, "session not found")
        self.assertEqual(self.client.get(url).status_code, 404)

    @patch("hitch.main.caches._start_models_refresh_thread")
    def test_diff_dialog_refresh_retry_and_obsolete_responses(self, _refresh: MagicMock) -> None:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import Route, sync_playwright

        path = _make_rollout(self, _basic_session_rollout_lines("Work", "Done"))
        SessionMetadata.objects.create(thread_id="diff", codex_path=str(path))
        html = self.client.get(reverse("session", args=["diff"])).content.decode()
        failed_diff = self.client.get(reverse("session_diff", args=["diff"]))
        self.assertEqual(failed_diff.status_code, 503)

        def route_request(route: Route) -> None:
            if "/assets/" in route.request.url:
                asset = route.request.url.split("/assets/", 1)[1]
                route.fulfill(body=render_to_string("assets/" + asset), content_type=(
                    "text/css" if asset.endswith(".css") else "text/javascript"
                ))
                return
            if "/static/" in route.request.url:
                asset = route.request.url.split("/static/", 1)[1]
                route.fulfill(path=str(Path(settings.BASE_DIR) / "hitch/main/static" / asset))
            elif route.request.resource_type == "document":
                route.fulfill(content_type="text/html", body=html)
            else:
                route.fulfill(json={"agents": []})

        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
            except PlaywrightError as exc:
                self.skipTest(f"playwright browser unavailable: {exc}")
            try:
                page = browser.new_page(viewport={"width": 375, "height": 812})
                page.route("**/*", route_request)
                page.add_init_script("""
                    window.EventSource = class { addEventListener() {} close() {} };
                    window.diffRequests = [];
                    const originalFetch = window.fetch;
                    window.fetch = (url, options) => String(url).endsWith('/diff/')
                        ? new Promise(resolve => diffRequests.push({resolve, options}))
                        : originalFetch(url, options);
                """)
                page.goto("http://hitch.test/sessions/diff/")
                self.assertEqual(page.evaluate("diffRequests.length"), 0)
                opener = page.locator(".diff-fab")
                body = page.locator("[data-diff-body]")

                def answer(index: int, text: str, status: int = 200) -> None:
                    page.evaluate("args => diffRequests[args[0]].resolve(new Response(args[1], {status: args[2]}))",
                                  [index, text, status])

                opener.click()
                page.wait_for_function("diffRequests.length === 1")
                self.assertIn("Loading diff", body.inner_text())
                answer(0, "First diff")
                page.wait_for_function("document.querySelector('[data-diff-body]').textContent === 'First diff'")
                page.keyboard.press("Escape")
                opener.click()
                page.wait_for_function("diffRequests.length === 2")
                answer(1, failed_diff.content.decode(), failed_diff.status_code)
                page.wait_for_function("document.querySelector('[data-diff-retry]').hidden === false")
                self.assertIn("Unable to load the diff", body.inner_text())
                self.assertNotIn("No changes", body.inner_text())
                page.locator("[data-diff-retry]").click()
                page.wait_for_function("diffRequests.length === 3")
                page.keyboard.press("Escape")
                opener.click()
                page.wait_for_function("diffRequests.length === 4")
                answer(3, "Current diff")
                answer(2, "Obsolete diff")
                page.wait_for_function("document.querySelector('[data-diff-body]').textContent === 'Current diff'")
                requests: list[dict[str, Any]] = page.evaluate("diffRequests.map(r => ({cache: r.options.cache}))")
                self.assertTrue(all(r["cache"] == "no-store" for r in requests))
                self.assertFalse(page.locator("[data-diff-retry]").is_visible())
            finally:
                browser.close()
