"""Native subagent discovery and the session transcript selector."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from django.conf import settings
from django.core.cache import cache
from django.test import SimpleTestCase, TestCase
from django.urls import reverse
from openai_codex.errors import MethodNotFoundError

from hitch.main.models import SessionMetadata
from hitch.main.sessions import subagents
from hitch.main.test.support import _rollout_line
from hitch.main.test.views_helpers import _basic_session_rollout_lines, _make_rollout
from hitch.main.views import common


class SubagentDiscoveryTests(SimpleTestCase):
    @patch("hitch.main.sessions.subagents.app_server_pool.run_borrowed_op_with_retry")
    def test_paged_archived_and_nested_agents_validate_ancestry_and_cache(self, run: MagicMock) -> None:
        cache.clear()
        self.addCleanup(cache.clear)
        codex = MagicMock()
        run.side_effect = lambda _factory, operation: operation(codex)
        codex._client._request_raw.side_effect = [
            {"data": [
                {"id": "grandchild", "parentThreadId": "child", "agentNickname": "Scout"},
                {"id": "unrelated", "parentThreadId": "another-session"},
                {"id": "system", "source": "subagent"},
                {"id": "self", "parentThreadId": "self"},
                {},
            ], "nextCursor": "page-2"},
            {"data": [{"id": "child", "path": "/child.jsonl", "source": {"subAgent": {"thread_spawn": {
                "parent_thread_id": "parent", "agent_nickname": "Ada", "agent_role": "reviewer",
            }}}}], "nextCursor": None},
            {"data": [{"id": "archived", "parentThreadId": "parent", "name": "Old reviewer"}]},
        ]
        agents = subagents.list_subagents("parent")
        self.assertEqual([a.id for a in agents], ["child", "archived", "grandchild"])
        self.assertEqual((agents[0].name, agents[0].role, agents[0].path), ("Ada", "reviewer", "/child.jsonl"))
        self.assertEqual(subagents.list_subagents("parent"), agents)
        self.assertEqual(run.call_count, 1)
        calls = codex._client._request_raw.call_args_list
        self.assertTrue(all(call.args[0] == "thread/list" for call in calls))
        self.assertTrue(all(call.args[1]["ancestorThreadId"] == "parent" for call in calls))
        self.assertTrue(all(call.args[1]["useStateDbOnly"] for call in calls))
        self.assertEqual(calls[1].args[1]["cursor"], "page-2")
        self.assertTrue(calls[2].args[1]["archived"])
        codex._client.thread_resume.assert_not_called()

    def test_invalid_response_and_repeated_cursor(self) -> None:
        codex = MagicMock()
        codex._client._request_raw.return_value = {"data": "invalid"}
        with self.assertRaises(ValueError):
            subagents._read_subagents(codex, "parent")
        codex._client._request_raw.return_value = {
            "data": [{"id": "child", "parentThreadId": "parent"}], "nextCursor": "same",
        }
        self.assertEqual(subagents._read_subagents(codex, "parent")[0].name, "Subagent child")
        self.assertEqual(codex._client._request_raw.call_count, 5)


class SubagentViewTests(TestCase):
    @patch("hitch.main.sessions.subagents.list_subagents")
    def test_transcript_discovery_errors_and_unrelated_selection(self, listing: MagicMock) -> None:
        lines = _basic_session_rollout_lines("Review the code", "<script>unsafe</script> **Result**")
        lines.insert(1, _rollout_line("response_item", {
            "type": "function_call", "name": "exec_command", "call_id": "command",
            "arguments": json.dumps({"cmd": "git status"}),
        }))
        path = _make_rollout(self, lines)
        listing.return_value = [subagents.Subagent("child", "parent", "Ada", "reviewer", str(path))]
        url = reverse("session_agents", args=["parent"])
        self.assertEqual(self.client.get(url).json()["agents"][0]["name"], "Ada")
        response = self.client.get(url, {"agent": "child"})
        data = response.json()
        self.assertContains(response, "Review the code")
        self.assertIn("git status", data["html"])
        self.assertNotIn("<script>", data["html"])
        self.assertNotIn("send_message_url", data["html"])
        self.assertEqual(data["selected"], "child")
        self.assertEqual(data["next_url"], "")
        self.assertIn("no-store", response["Cache-Control"])
        for query in ({"agent": "unrelated"}, {"agent": "child", "before": "bad"},
                      {"agent": "child", "before": "-1"}, {"agent": "child", "before": "99999999"},
                      {"agent": "child", "record_end": "1"},
                      {"agent": "child", "before": "1", "record_end": "1"}):
            with self.subTest(query=query):
                self.assertEqual(self.client.get(url, query).status_code, 404)
        for minimum in (1, 1024 * 1024):
            with (
                self.subTest(minimum=minimum),
                patch.object(common, "_SESSION_HISTORY_MIN_BYTES", minimum),
                patch.object(Path, "open", side_effect=PermissionError("unreadable")),
                self.assertLogs("hitch.main.runtime.rollout", level="WARNING"),
            ):
                self.assertEqual(self.client.get(url, {"agent": "child"}).status_code, 503)
        listing.side_effect = MethodNotFoundError(-32601, "unavailable", None)
        with self.assertLogs("hitch.main.views.session_detail", level="ERROR"):
            self.assertEqual(self.client.get(url).status_code, 503)

    @patch.object(common, "_SESSION_HISTORY_MIN_BYTES", 1)
    @patch("hitch.main.sessions.subagents.list_subagents")
    def test_older_history_and_missing_rollout(self, listing: MagicMock) -> None:
        path = _make_rollout(self, [
            line for index in range(45)
            for line in [
                *_basic_session_rollout_lines(f"Prompt {index}", f"Reply {index}"),
                _rollout_line("event_msg", {"type": "agent_message", "message": f"Reply {index}"}),
            ]
        ])
        with path.open("a") as handle:
            handle.write("\n" + _rollout_line("response_item", {
                "type": "function_call", "name": "exec_command", "call_id": "activity",
                "arguments": json.dumps({"cmd": "git status"}),
            }) + "\n")
        listing.return_value = [subagents.Subagent("child", "parent", "Ada", "", str(path))]
        url = reverse("session_agents", args=["parent"])
        latest = self.client.get(url, {"agent": "child"}).json()
        self.assertIn("Prompt 44", latest["html"])
        self.assertNotIn("Prompt 0<", latest["html"])
        older = self.client.get(latest["next_url"]).json()
        self.assertIn("Prompt 24", older["html"])
        self.assertNotIn("Prompt 44", older["html"])
        self.assertTrue(latest["partial"])
        self.assertNotIn("git status", latest["html"])
        full = self.client.get(url, {"agent": "child", "history": "all"}).json()
        self.assertIn("git status", full["html"])
        self.assertIn("Prompt 0<", full["html"])
        self.assertFalse(full["partial"])
        self.assertEqual(full["next_url"], "")
        with (
            patch.object(Path, "open", side_effect=PermissionError("unreadable")),
            self.assertLogs("hitch.main.runtime.rollout", level="WARNING"),
        ):
            self.assertEqual(self.client.get(latest["next_url"]).status_code, 503)
        listing.return_value = [subagents.Subagent("child", "parent", "Ada", "", "")]
        with patch("hitch.main.views.session_detail._stored_rollout_path_for_thread", return_value=None):
            self.assertEqual(self.client.get(url, {"agent": "child"}).json()["html"].strip(), "")

    @patch("hitch.main.caches._start_models_refresh_thread")
    def test_picker_browser_preserves_draft_and_recovers_from_errors(self, _models: MagicMock) -> None:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import Route, sync_playwright

        path = _make_rollout(self, _basic_session_rollout_lines("Main prompt", "Main response"))
        SessionMetadata.objects.create(thread_id="parent", codex_path=str(path))
        response = self.client.get(reverse("session", args=["parent"]))
        html = response.content.decode()
        self.assertIn("data-agent-select", html)
        read_only = common._render_session_detail(
            response.wsgi_request, "parent", read_only=True,
        )
        self.assertContains(read_only, "data-agent-select")
        self.assertNotContains(read_only, '<form class="composer"')
        state: dict[str, Any] = {
            "fail": False, "text": "Child response", "next_url": "/sessions/parent/agents/?agent=child&before=20",
        }

        def route_request(route: Route) -> None:
            url = route.request.url
            if "/agents/" in url:
                if state["fail"]:
                    route.fulfill(status=503)
                    return
                data: dict[str, Any] = {
                    "agents": [{"id": "child", "name": "Ada", "role": "reviewer", "parent_id": "parent"}],
                }
                if "agent=child" in url:
                    data.update(selected="child", name="Ada", role="reviewer",
                                next_url=state["next_url"], html=state["text"], partial=True)
                    if "before=" in url:
                        state["history_url"] = url
                        data.update(next_url="", html="Older child response")
                    if "history=all" in url:
                        data.update(next_url="", html="Command: git status", partial=False)
                route.fulfill(json=data)
            elif "/static/" in url:
                asset = Path(settings.BASE_DIR) / "hitch/main/static" / url.split("/static/", 1)[1]
                route.fulfill(path=str(asset))
            elif "/stream/" in url:
                route.fulfill(content_type="text/event-stream", body="")
            else:
                route.fulfill(content_type="text/html", body=html)

        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
            except PlaywrightError as exc:
                self.skipTest(f"playwright browser unavailable: {exc}")
            try:
                page = browser.new_page(viewport={"width": 375, "height": 812})
                page.add_init_script("""window.EventSource = class {
                    constructor() { this.handlers = {}; window.stream = this; }
                    addEventListener(name, handler) { this.handlers[name] = handler; }
                    close() {}
                };""")
                page.route("**/*", route_request)
                page.goto("http://hitch.test/sessions/parent/")
                page.locator('[data-agent-select] option[value="child"]').wait_for(state="attached")
                composer = page.locator("[data-composer-input]")
                composer.fill("Keep my draft")
                page.set_viewport_size({"width": 1280, "height": 900})
                layout = page.evaluate("""() => {
                    const main = document.querySelector('main');
                    const plan = document.querySelector('[data-task-plan]');
                    const card = plan.querySelector('[data-task-plan-card]');
                    main.classList.add('has-task-plan');
                    plan.hidden = false;
                    card.style.minHeight = '650px';
                    const body = document.querySelector('.session-body').getBoundingClientRect();
                    const sidebar = plan.getBoundingClientRect();
                    const result = {
                        bodyTop: body.top, bodyRight: body.right, planTop: sidebar.top, planLeft: sidebar.left,
                    };
                    main.classList.remove('has-task-plan');
                    plan.hidden = true;
                    card.style.minHeight = '';
                    return result;
                }""")
                self.assertLess(layout["bodyTop"], layout["planTop"] + 300)
                self.assertLess(layout["bodyRight"], layout["planLeft"])
                page.set_viewport_size({"width": 375, "height": 812})
                page.locator("[data-agent-select]").select_option("child")
                page.get_by_text("Child response", exact=True).wait_for()
                self.assertFalse(composer.is_visible())
                self.assertIn("agent=child", page.url)
                page.locator("[data-agent-full]").click()
                page.get_by_text("Command: git status", exact=True).wait_for()
                self.assertFalse(page.locator("[data-agent-full]").is_visible())
                self.assertFalse(page.locator("[data-agent-earlier]").is_visible())
                page.locator("[data-agent-latest]").click()
                page.get_by_text("Child response", exact=True).wait_for()
                page.locator("[data-agent-select]").select_option("")
                self.assertTrue(composer.is_visible())
                self.assertEqual(composer.input_value(), "Keep my draft")
                state["fail"] = True
                page.locator("[data-agents-refresh]").click()
                page.locator("[data-agent-feedback]").wait_for(state="visible")
                self.assertTrue(composer.is_visible())
                state["fail"] = False
                page.locator("[data-agents-refresh]").click()
                page.locator("[data-agent-feedback]").wait_for(state="hidden")
                page.locator("[data-agent-select]").select_option("child")
                page.get_by_text("Child response", exact=True).wait_for()
                self.assertFalse(page.locator("[data-agent-latest]").is_visible())
                page.evaluate("""() => {
                    window.originalAgentOption = document.querySelector('[data-agent-select] option[value="child"]');
                    window.getSelection().selectAllChildren(document.querySelector('[data-agent-entries]'));
                }""")
                state["text"] = "Updated child response"
                state["next_url"] = "/sessions/parent/agents/?agent=child&before=30"
                with page.expect_response("**/agents/?agent=child"):
                    page.locator("[data-agents-refresh]").evaluate("button => button.click()")
                page.wait_for_function("!document.querySelector('[data-agent-transcript]').hasAttribute('aria-busy')")
                self.assertEqual(page.evaluate("window.getSelection().toString()"), "Child response")
                self.assertTrue(page.evaluate("""window.originalAgentOption ===
                    document.querySelector('[data-agent-select] option[value="child"]')"""))
                page.evaluate("window.getSelection().removeAllRanges()")
                page.locator("[data-agent-earlier]").click()
                page.get_by_text("Older child response", exact=True).wait_for()
                self.assertIn("before=20", state["history_url"])
                self.assertFalse(page.locator("[data-agent-earlier]").is_visible())
                with page.expect_response("**/agents/?agent=child"):
                    page.locator("[data-agents-refresh]").click()
                page.wait_for_function("!document.querySelector('[data-agent-transcript]').hasAttribute('aria-busy')")
                self.assertTrue(page.get_by_text("Child response", exact=True).is_visible())
                page.locator("[data-agent-latest]").click()
                page.get_by_text("Updated child response", exact=True).wait_for()
                self.assertFalse(page.get_by_text("Older child response", exact=True).is_visible())
                self.assertLessEqual(page.evaluate("document.documentElement.scrollWidth"), 375)
                page.locator("[data-agents-refresh]").click()
                page.get_by_text("Updated child response", exact=True).wait_for()
                page.reload()
                page.get_by_text("Updated child response", exact=True).wait_for()
                self.assertEqual(page.locator("[data-agent-select]").input_value(), "child")
                self.assertLessEqual(page.evaluate("document.documentElement.scrollWidth"), 375)
                for return_to in ("main", "latest"):
                    page.locator("[data-agent-earlier]").click()
                    page.get_by_text("Older child response", exact=True).wait_for()
                    selected_text = page.evaluate("""() => {
                        window.getSelection().selectAllChildren(document.querySelector('[data-agent-older]'));
                        return window.getSelection().toString();
                    }""")
                    page.evaluate("window.stream.handlers.end()")
                    self.assertEqual(page.evaluate("window.getSelection().toString()"), selected_text)
                    self.assertTrue(page.get_by_text("Older child response", exact=True).is_visible())
                    with page.expect_response(lambda response: response.request.is_navigation_request()):
                        if return_to == "main":
                            page.locator("[data-agent-select]").select_option("")
                        else:
                            page.locator("[data-agent-latest]").click()
                    page.wait_for_load_state()
                    if return_to == "main":
                        self.assertTrue(composer.is_visible())
                        page.locator('[data-agent-select] option[value="child"]').wait_for(state="attached")
                        page.locator("[data-agent-select]").select_option("child")
                    page.get_by_text("Updated child response", exact=True).wait_for()
            finally:
                browser.close()
