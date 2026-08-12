import json
import os
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast, override
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

from django.conf import settings as django_settings
from django.http import HttpResponse
from django.test import Client, TestCase
from django.urls import reverse
from django.utils import timezone
from openai_codex import Codex
from openai_codex.errors import CodexError

from hitch.main import caches, demo
from hitch.main.diffs import DiffFile, DiffLine, DiffView
from hitch.main.models import (
    CodexInstance,
    SessionDemo,
    SessionMetadata,
    SystemAgentRun,
    SystemWorkflow,
    WorkflowSteeringMessage,
)
from hitch.main.runtime import codex_events
from hitch.main.sessions import session_entry_display
from hitch.main.sessions.entry_render import tool_call_detail, tool_call_status
from hitch.main.sessions.session_pr_plan import (
    _pr_snapshot_for_thread,
    _pr_url_for_thread,
)
from hitch.main.test.support import (
    _make_project,
    _rollout_line,
    _seed_cookies,
)
from hitch.main.workflows import system_agents

# Used for active-worker rendering tests so the session view's
# ``reconcile_dead`` sweep doesn't mark the row failed before the assertions
# run; the current process pid is by definition alive.
_LIVE_PID = os.getpid()


def _worker_is_live_for_test(instance: CodexInstance) -> bool:
    return instance.pid == _LIVE_PID


class AutoPullTextTests(TestCase):
    def test_formats_up_to_date_and_failure_results(self) -> None:
        self.assertEqual(
            session_entry_display._auto_pull_text(
                {"status": "up_to_date", "branch": "main"}
            ),
            "Auto-pull: the default repo was already up to date with origin/main.",
        )
        self.assertEqual(
            session_entry_display._auto_pull_text(
                {"status": "failed", "error": "project repository is dirty"}
            ),
            "Auto-pull failed: project repository is dirty",
        )
        self.assertEqual(
            session_entry_display._auto_pull_text({"status": "failed"}),
            "Auto-pull failed.",
        )
        self.assertEqual(
            session_entry_display._auto_pull_text(
                {
                    "status": "skipped",
                    "reason": "default checkout is the active session checkout",
                }
            ),
            "Auto-pull skipped: default checkout is the active session checkout",
        )
        self.assertEqual(
            session_entry_display._auto_pull_text({"status": "skipped"}),
            "Auto-pull skipped.",
        )
        self.assertEqual(
            session_entry_display._auto_pull_text({"status": "running"}),
            "Auto-pull started but did not finish.",
        )
        self.assertEqual(session_entry_display._auto_pull_text("missing"), "")
        self.assertEqual(
            session_entry_display._auto_pull_text({"status": "pulled"}), ""
        )


def _root(item: SimpleNamespace) -> SimpleNamespace:
    """Wrap an item to look like a pydantic RootModel from the codex SDK."""
    return SimpleNamespace(root=item)


def _user_message(*texts: str) -> SimpleNamespace:
    return _root(
        SimpleNamespace(
            type="userMessage",
            content=[_root(SimpleNamespace(type="text", text=t)) for t in texts],
        )
    )


def _agent_message(
    text: str,
    phase: str | None = None,
    memory_citation: SimpleNamespace | dict[str, object] | None = None,
) -> SimpleNamespace:
    # The SDK surfaces phase as a MessagePhase enum (with `.value`); mirror
    # that shape so tests match production deserialization.
    phase_obj = SimpleNamespace(value=phase) if phase is not None else None
    return _root(
        SimpleNamespace(
            type="agentMessage",
            text=text,
            phase=phase_obj,
            memory_citation=memory_citation,
        )
    )


def _command(command: str, status: str = "completed") -> SimpleNamespace:
    return _root(
        SimpleNamespace(
            type="commandExecution",
            command=command,
            status=SimpleNamespace(value=status),
        )
    )


def _file_change(*paths: str, status: str = "completed") -> SimpleNamespace:
    return _root(
        SimpleNamespace(
            type="fileChange",
            changes=[SimpleNamespace(path=p) for p in paths],
            status=SimpleNamespace(value=status),
        )
    )


def _tool_call(item_type: str) -> SimpleNamespace:
    return _root(SimpleNamespace(type=item_type))


def _mcp_tool_call(
    server: str,
    tool: str,
    result: object | None = None,
) -> SimpleNamespace:
    return _root(
        SimpleNamespace(
            type="mcpToolCall",
            server=server,
            tool=tool,
            result=result,
            status=SimpleNamespace(value="completed"),
        )
    )


def _turn(items: list[SimpleNamespace], started_at: int | None = 1700000000) -> SimpleNamespace:
    return SimpleNamespace(items=items, started_at=started_at)


def _thread(turns: list[SimpleNamespace], **overrides: object) -> SimpleNamespace:
    defaults: dict[str, object] = {
        "id": "thread-1",
        "name": "Demo session",
        "preview": "first message",
        "cwd": "/tmp/demo",
        "updated_at": 1700000000,
        "turns": turns,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _write_rollout_tempfile(lines: list[str], *, binary: bytes | None = None) -> Path:
    if binary is not None:
        with tempfile.NamedTemporaryFile(
            prefix="rollout-", suffix=".jsonl", mode="wb", delete=False
        ) as fh:
            fh.write(binary)
            return Path(fh.name)
    with tempfile.NamedTemporaryFile(
        prefix="rollout-", suffix=".jsonl", mode="w", delete=False
    ) as fh:
        fh.write("\n".join(lines))
        if lines:
            fh.write("\n")
        return Path(fh.name)


def _patch_thread(test: TestCase, mock_codex: MagicMock, thread: SimpleNamespace) -> None:
    client = mock_codex.return_value.__enter__.return_value
    client._client.thread_resume.return_value.thread = thread


def _clear_models_cache() -> None:
    with caches._MODELS_REFRESH_LOCK:
        caches._MODELS_CACHE_VALUE = {}
        caches._MODELS_CACHE_FETCHED_AT = {}
        caches._MODELS_REFRESH_IN_FLIGHT = set()


def _seed_models_cache(models: list[SimpleNamespace]) -> None:
    with caches._MODELS_REFRESH_LOCK:
        caches._MODELS_CACHE_VALUE[False] = list(models)
        caches._MODELS_CACHE_FETCHED_AT[False] = timezone.now()
        caches._MODELS_REFRESH_IN_FLIGHT.discard(False)


def _make_rollout(test: TestCase, lines: list[str], *, binary: bytes | None = None) -> Path:
    path = _write_rollout_tempfile(lines, binary=binary)
    test.addCleanup(path.unlink, missing_ok=True)
    return path


def _get_session(client: Client, session_id: str = "thread-1") -> HttpResponse:
    response = client.get(reverse("session", kwargs={"session_id": session_id}))
    assert isinstance(response, HttpResponse)
    return response


def _diff_view() -> DiffView:
    return DiffView(
        files=[
            DiffFile(
                path="hitch/main/views.py",
                old_path="hitch/main/views.py",
                status="Modified",
                additions=1,
                deletions=1,
                lines=[
                    DiffLine("hunk", None, None, "@@ -1 +1 @@"),
                    DiffLine("remove", 1, None, '<span class="k">return</span> 1'),
                    DiffLine("add", None, 1, '<span class="k">return</span> 2'),
                ],
            )
        ],
        additions=1,
        deletions=1,
    )


class PrUrlDetectionTests(TestCase):
    LEGACY_PR_PROMPT = (
        "Do a thorough review of the diff. Rebase on master, clean it up, "
        "and then open a PR"
    )
    LEGACY_PR_FINAL_PROMPT = (
        f"{LEGACY_PR_PROMPT}. After opening it, poll the PR every 2 minutes "
        "until you have CI status and at least one review signal: code review "
        "comments, a thumbs up emoji on the PR, or an explicit review approval. "
        "On each poll, check whether the PR has merge conflicts. Address CI "
        "failures, review comments, merge conflicts, and any other blocking issues; "
        "push fixes and keep looping until CI, review, and mergeability are all clean. "
        "Stop and report back if any single polling iteration has no results after "
        "30 minutes."
    )

    def test_detects_pr_url_from_latest_pr_turn_github_mcp_result(self) -> None:
        earlier = "https://github.com/cberner/hitch/pull/93"
        latest = "https://github.com/cberner/hitch/pull/94"
        thread = _thread(
            [
                _turn(
                    [
                        _user_message("ordinary follow-up"),
                        _mcp_tool_call(
                            "github",
                            "_create_pull_request",
                            {"structuredContent": {"display_url": earlier}},
                        ),
                        _agent_message("Done."),
                    ]
                ),
                _turn(
                    [
                        _user_message(system_agents.PR_SLASH_PROMPT),
                        _mcp_tool_call(
                            "github",
                            "_create_pull_request",
                            {
                                "content": [
                                    {
                                        "text": json.dumps(
                                            {"url": earlier, "display_url": latest}
                                        )
                                    }
                                ],
                            },
                        ),
                        _agent_message("Opened the PR."),
                    ]
                ),
            ]
        )

        self.assertEqual(_pr_url_for_thread(thread), latest)

    def test_detects_pr_url_from_legacy_pr_prompt_strings(self) -> None:
        display_url = "https://github.com/cberner/hitch/pull/93"
        final_url = "https://github.com/cberner/hitch/pull/94"

        for prompt, url in (
            (self.LEGACY_PR_PROMPT, display_url),
            (self.LEGACY_PR_FINAL_PROMPT, final_url),
        ):
            with self.subTest(prompt=prompt):
                thread = _thread(
                    [
                        _turn(
                            [
                                _user_message(prompt),
                                _mcp_tool_call(
                                    "github", "_create_pull_request", {"url": url}
                                ),
                                _agent_message("Opened the PR."),
                            ]
                        )
                    ]
                )

                self.assertEqual(_pr_url_for_thread(thread), url)

    def test_ignores_non_pr_turns_and_non_github_tools(self) -> None:
        url = "https://github.com/cberner/hitch/pull/94"
        non_pr_thread = _thread(
            [
                _turn(
                    [
                        _user_message("ordinary follow-up"),
                        _mcp_tool_call("github", "_create_pull_request", {"url": url}),
                        _agent_message("Done."),
                    ]
                )
            ]
        )
        non_github_thread = _thread(
            [
                _turn(
                    [
                        _user_message(system_agents.PR_SLASH_PROMPT),
                        _mcp_tool_call("linear", "create_issue", {"url": url}),
                        _agent_message("Done."),
                    ]
                )
            ]
        )

        self.assertIsNone(_pr_url_for_thread(non_pr_thread))
        self.assertIsNone(_pr_url_for_thread(non_github_thread))

    def test_ignores_incomplete_pr_turns(self) -> None:
        url = "https://github.com/cberner/hitch/pull/94"
        thread = _thread(
            [
                _turn(
                    [
                        _user_message(system_agents.PR_SLASH_PROMPT),
                        _mcp_tool_call("github", "_create_pull_request", {"url": url}),
                    ]
                )
            ]
        )

        self.assertIsNone(_pr_url_for_thread(thread))

    def test_latest_completed_pr_turn_without_url_stops_search(self) -> None:
        stale_url = "https://github.com/cberner/hitch/pull/93"
        thread = _thread(
            [
                _turn(
                    [
                        _user_message(system_agents.PR_SLASH_PROMPT),
                        _mcp_tool_call("github", "_create_pull_request", {"url": stale_url}),
                        _agent_message("Opened the PR."),
                    ]
                ),
                _turn(
                    [
                        _user_message(system_agents.PR_SLASH_PROMPT),
                        _mcp_tool_call("github", "_create_pull_request", {"content": []}),
                        _agent_message("No PR was opened."),
                    ]
                ),
            ]
        )

        self.assertIsNone(_pr_url_for_thread(thread))

    def test_pr_workflow_notice_without_observation_keeps_existing_pr(self) -> None:
        url = "https://github.com/cberner/hitch/pull/94"
        thread = _thread(
            [
                _turn(
                    [
                        _user_message(system_agents.PR_SLASH_PROMPT),
                        _mcp_tool_call(
                            "github",
                            "_create_pull_request",
                            {"url": url, "state": "open"},
                        ),
                        _agent_message("Opened the PR."),
                    ]
                ),
                _turn(
                    [
                        _user_message(
                            "Hitch QA agent could not complete the PR workflow.\n\n"
                            "Status: Hitch checked the PR gates and is waiting on "
                            "external PR state.\n\n"
                            "Tell the user the PR workflow needs attention before "
                            "continuing."
                        ),
                        _agent_message("PR workflow needs attention."),
                    ]
                ),
            ]
        )

        self.assertEqual(_pr_url_for_thread(thread), url)
        snapshot = _pr_snapshot_for_thread(thread)
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot["url"], url)

    def test_detects_pr_url_when_mcp_tool_call_follows_final_message(self) -> None:
        # The model can emit the create_pull_request MCP call in the same
        # response that carries the final-answer message: the tool runs after
        # that response, so the completed ``mcpToolCall`` item lands in
        # ``turn.items`` AFTER the final-answer message for the same turn.
        # ``items[:final_idx]`` would silently drop that result and the
        # session page would render no PR pill for the PR the user just
        # opened. Mirrors the ``rollout.latest_pr_url`` regression test for
        # the function_call_output-after-final-answer shape.
        url = "https://github.com/cberner/hitch/pull/94"
        thread = _thread(
            [
                _turn(
                    [
                        _user_message(system_agents.PR_SLASH_PROMPT),
                        _agent_message("Opening the PR now.", phase="final_answer"),
                        _mcp_tool_call(
                            "github",
                            "_create_pull_request",
                            {"structuredContent": {"url": url}},
                        ),
                    ]
                )
            ]
        )

        self.assertEqual(_pr_url_for_thread(thread), url)

    def test_pr_snapshot_when_mcp_tool_call_follows_final_message(self) -> None:
        # ``_pr_observation_result_for_thread`` reads the PR identity that
        # the session-stage badge and the cached ``derived_stage`` both
        # depend on. With the post-final ``mcpToolCall`` dropped, the URL
        # pill could still render (from ``_pr_url_for_thread``) while the
        # snapshot stayed empty -- so the stage fell back to
        # ``IMPLEMENTATION`` and any ``closed``/``merged`` state from
        # ``structuredContent`` was lost.
        url = "https://github.com/cberner/hitch/pull/95"
        thread = _thread(
            [
                _turn(
                    [
                        _user_message(system_agents.PR_SLASH_PROMPT),
                        _agent_message("Closed it.", phase="final_answer"),
                        _mcp_tool_call(
                            "github",
                            "_create_pull_request",
                            {
                                "structuredContent": {
                                    "url": url,
                                    "state": "closed",
                                    "merged": False,
                                }
                            },
                        ),
                    ]
                )
            ]
        )

        snapshot = _pr_snapshot_for_thread(thread)

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot["url"], url)
        self.assertEqual(snapshot["state"], "closed")
        self.assertIs(snapshot["merged"], False)


class SessionViewTests(TestCase):
    @override
    def setUp(self) -> None:
        super().setUp()
        patcher = patch(
            "hitch.main.runtime.codex_pool.worker_is_alive",
            side_effect=_worker_is_live_for_test,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(_clear_models_cache)

    @patch("hitch.main.views.common.Codex")
    def test_renders_primary_nav_menu_instead_of_back_link(
        self, mock_codex: MagicMock
    ) -> None:
        thread = _thread([_turn([_user_message("hi")])])
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "data-nav-menu")
        self.assertContains(response, "data-nav-menu-open")
        self.assertContains(response, "data-nav-menu-panel")
        self.assertContains(response, 'aria-label="Navigation menu"')
        body = response.content.decode()
        nav_start = body.index('<nav class="primary-nav"')
        nav_end = body.index("</nav>", nav_start)
        nav_html = body[nav_start:nav_end]
        self.assertIn(f'href="{reverse("new_session")}"', nav_html)
        self.assertIn('class="primary-nav-new-session"', nav_html)
        self.assertIn('aria-label="New session"', nav_html)
        self.assertNotIn(">new session<", nav_html)
        self.assertContains(response, f'href="{reverse("index")}"')
        self.assertNotContains(
            response, f'href="{reverse("index")}" aria-current="page"'
        )
        self.assertNotIn(f'href="{reverse("usage")}"', nav_html)
        self.assertIn(f'href="{reverse("profile")}"', nav_html)
        self.assertIn(">anonymous</a>", nav_html)
        self.assertContains(response, ">settings<")
        self.assertIn(reverse("update_settings"), nav_html)
        self.assertNotContains(response, "data-settings-dialog")
        self.assertContains(response, 'classList.add("primary-nav-js")')
        self.assertNotContains(response, "html:not(.js) .primary-nav-toggle")
        self.assertNotContains(response, 'class="back-link"')

    @patch("hitch.main.views.common.Codex")
    def test_selected_project_session_nav_includes_project_links(
        self, mock_codex: MagicMock
    ) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        thread = _thread([_turn([_user_message("hi")])], cwd="/repo")
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        body = response.content.decode()
        nav_start = body.index('<nav class="primary-nav"')
        nav_end = body.index("</nav>", nav_start)
        nav_html = body[nav_start:nav_end]
        self.assertIn(f'href="{reverse("autonomous_goals")}"', nav_html)
        self.assertIn(">auto goals</a>", nav_html)
        self.assertContains(response, "@media (max-width: 900px)")

    @patch("hitch.main.views.common.Codex")
    def test_settings_page_preserves_saved_choices_missing_from_cache(
        self, mock_codex: MagicMock
    ) -> None:
        models = [
            SimpleNamespace(
                id="gpt-current",
                display_name="GPT Current",
                is_default=True,
                default_reasoning_effort=SimpleNamespace(value="medium"),
                supported_reasoning_efforts=[],
            )
        ]
        _seed_cookies(
            self.client,
            hitch_model="stale-model",
            hitch_reasoning_effort="high",
        )
        _seed_models_cache(models)

        response = cast(HttpResponse, self.client.get(reverse("update_settings")))

        self.assertEqual(response.status_code, 200)
        mock_codex.assert_not_called()
        self.assertContains(response, 'value="stale-model" selected')
        self.assertContains(response, 'value="high" selected')
        self.assertNotIn("hitch_model", response.cookies)
        self.assertNotIn("hitch_reasoning_effort", response.cookies)

    @patch("hitch.main.views.common.Codex")
    def test_renders_edit_title_form(self, mock_codex: MagicMock) -> None:
        """The edit form is pre-populated with the current name when set, and
        empty when not — so the user can revise without retyping from scratch."""
        for name, expected_value in (("Custom title", "Custom title"), (None, "")):
            with self.subTest(name=name):
                thread = _thread([_turn([_user_message("hi")])], name=name)
                _patch_thread(self, mock_codex, thread)
                response = _get_session(self.client)
                self.assertEqual(response.status_code, 200)
                self.assertContains(response, 'name="name"')
                self.assertContains(response, f'value="{expected_value}"')
                self.assertContains(
                    response,
                    reverse("set_session_name", kwargs={"session_id": "thread-1"}),
                )
                self.assertContains(
                    response,
                    reverse("set_session_archived", kwargs={"session_id": "thread-1"}),
                )
                self.assertContains(response, 'aria-label="Session actions"')
                self.assertContains(response, 'role="menuitem" data-edit-title-open>Rename')
                self.assertContains(response, 'name="archived" value="true"')
                self.assertContains(response, 'role="menuitem">Archive</button>')
                self.assertNotContains(response, ">Edit</button>")

    @patch("hitch.main.repos.discover_repos", return_value=[Path("/tmp/demo")])
    @patch("hitch.main.views.common.Codex")
    def test_action_menu_includes_debug_chat_link(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        project = _make_project(repo_path="/tmp/demo")
        SessionMetadata.objects.create(
            thread_id="thread-1", cwd="/tmp/demo", project=project
        )
        thread = _thread([_turn([_user_message("hi")])], cwd="/tmp/demo")
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, ">Debug chat</a>")
        debug_url = cast(str, cast(Any, response).context["debug_chat_url"])
        parsed = urlparse(debug_url)
        query = parse_qs(parsed.query)
        self.assertEqual(parsed.path, reverse("new_session"))
        self.assertEqual(query["project"], [str(project.pk)])
        server_cwd = Path(django_settings.BASE_DIR)
        database_path = Path(str(django_settings.DATABASES["default"]["NAME"]))
        if not database_path.is_absolute():
            database_path = server_cwd / database_path
        self.assertEqual(
            query["prompt"],
            [
                "Debug and fix the user's issue from session UID thread-1.\n\n"
                f"Hitch server working directory: {server_cwd}\n"
                f"Configured Hitch SQLite database path: {database_path}\n"
                "If you need to inspect it, copy the database first and use the copy; "
                "do not modify the main database file. When copying files directly, include "
                f"the WAL sidecars {database_path}-wal and {database_path}-shm if they exist "
                "so recent rows are included. A SQLite .backup snapshot is also acceptable.\n\n"
                "User issue: "
            ],
        )

    @patch(
        "hitch.main.repos.discover_repos",
        return_value=[Path("/tmp/other"), Path("/tmp/hitch")],
    )
    @patch("hitch.main.views.common.Codex")
    def test_action_menu_prefers_hitch_project_for_debug_chat_link(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        session_project = _make_project(name="Other", repo_path="/tmp/other")
        hitch_project = _make_project(repo_path="/tmp/hitch")
        SessionMetadata.objects.create(
            thread_id="thread-1", cwd="/tmp/other", project=session_project
        )
        thread = _thread([_turn([_user_message("hi")])], cwd="/tmp/other")
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        debug_url = cast(str, cast(Any, response).context["debug_chat_url"])
        query = parse_qs(urlparse(debug_url).query)
        self.assertEqual(query["project"], [str(hitch_project.pk)])
        self.assertNotEqual(query["project"], [str(session_project.pk)])

    @patch("hitch.main.repos.discover_repos", return_value=[Path("/tmp/other")])
    @patch("hitch.main.views.common.Codex")
    def test_action_menu_ignores_undiscovered_hitch_project_for_debug_chat_link(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        session_project = _make_project(name="Other", repo_path="/tmp/other")
        hitch_project = _make_project(
            name="Hitch", repo_path="/tmp/missing-hitch"
        )
        SessionMetadata.objects.create(
            thread_id="thread-1", cwd="/tmp/other", project=session_project
        )
        thread = _thread([_turn([_user_message("hi")])], cwd="/tmp/other")
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        debug_url = cast(str, cast(Any, response).context["debug_chat_url"])
        query = parse_qs(urlparse(debug_url).query)
        self.assertEqual(query["project"], [str(session_project.pk)])
        self.assertNotEqual(query["project"], [str(hitch_project.pk)])

    @patch("hitch.main.repos.discover_repos", return_value=[Path("/tmp/demo")])
    @patch("hitch.main.views.common.Codex")
    def test_action_menu_includes_cwd_for_bare_repo_debug_chat_link(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        _make_project(name="Other", repo_path="/tmp/demo")
        SessionMetadata.objects.create(
            thread_id="thread-1", cwd="/tmp/demo", project_cleared=True
        )
        thread = _thread([_turn([_user_message("hi")])], cwd="/tmp/demo")
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        debug_url = cast(str, cast(Any, response).context["debug_chat_url"])
        query = parse_qs(urlparse(debug_url).query)
        self.assertNotIn("project", query)
        self.assertEqual(query["cwd"], ["/tmp/demo"])

    @patch("hitch.main.views.common.Codex")
    def test_renders_move_to_project_menu_and_dialog(self, mock_codex: MagicMock) -> None:
        project = _make_project(repo_path="/tmp/demo")
        SessionMetadata.objects.create(thread_id="thread-1", cwd="/tmp/demo", project=project)
        thread = _thread([_turn([_user_message("hi")])])
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Move to project")
        self.assertContains(
            response,
            reverse("set_session_project", kwargs={"session_id": "thread-1"}),
        )
        self.assertContains(response, 'name="project"')
        self.assertContains(response, f'value="{project.pk}" selected')

    @patch("hitch.main.views.common.Codex")
    def test_set_session_project_moves_and_clears_project(
        self, mock_codex: MagicMock
    ) -> None:
        project = _make_project(repo_path="/tmp/demo")
        thread = _thread([_turn([_user_message("hi")])])
        _patch_thread(self, mock_codex, thread)

        response = self.client.post(
            reverse("set_session_project", kwargs={"session_id": "thread-1"}),
            data={"project": str(project.pk)},
        )

        self.assertEqual(response.status_code, 302)
        metadata = SessionMetadata.objects.get(thread_id="thread-1")
        self.assertEqual(metadata.project, project)
        self.assertEqual(metadata.cwd, "/tmp/demo")
        self.assertFalse(metadata.project_cleared)

        response = self.client.post(
            reverse("set_session_project", kwargs={"session_id": "thread-1"}),
            data={"project": ""},
        )

        self.assertEqual(response.status_code, 302)
        metadata.refresh_from_db()
        self.assertIsNone(metadata.project)
        self.assertTrue(metadata.project_cleared)

    @patch("hitch.main.views.common.Codex")
    def test_set_session_project_uses_metadata_cwd_without_resume(
        self, mock_codex: MagicMock
    ) -> None:
        project = _make_project(repo_path="/tmp/demo")
        SessionMetadata.objects.create(thread_id="thread-1", cwd="/tmp/demo")

        response = self.client.post(
            reverse("set_session_project", kwargs={"session_id": "thread-1"}),
            data={"project": str(project.pk)},
        )

        self.assertEqual(response.status_code, 302)
        metadata = SessionMetadata.objects.get(thread_id="thread-1")
        self.assertEqual(metadata.cwd, "/tmp/demo")
        self.assertEqual(metadata.project, project)
        self.assertFalse(metadata.project_cleared)
        mock_codex.assert_not_called()

    @patch("hitch.main.runtime.app_server_pool.run_borrowed_op_with_retry")
    def test_set_session_project_falls_back_without_metadata_cwd(
        self, mock_run_borrowed: MagicMock
    ) -> None:
        project = _make_project(repo_path="/tmp/demo")
        SessionMetadata.objects.create(thread_id="thread-1")
        client = SimpleNamespace(
            _client=SimpleNamespace(
                thread_resume=MagicMock(
                    return_value=SimpleNamespace(
                        thread=SimpleNamespace(cwd="/tmp/demo")
                    )
                )
            )
        )

        def run_side_effect(
            _factory: object, operation: Callable[[Any], object], **_kwargs: object
        ) -> object:
            return operation(client)

        mock_run_borrowed.side_effect = run_side_effect

        response = self.client.post(
            reverse("set_session_project", kwargs={"session_id": "thread-1"}),
            data={"project": str(project.pk)},
        )

        self.assertEqual(response.status_code, 302)
        metadata = SessionMetadata.objects.get(thread_id="thread-1")
        self.assertEqual(metadata.cwd, "/tmp/demo")
        self.assertEqual(metadata.project, project)
        client._client.thread_resume.assert_called_once_with("thread-1")
        mock_run_borrowed.assert_called_once()
        self.assertIs(mock_run_borrowed.call_args.args[0], Codex)
        self.assertEqual(
            mock_run_borrowed.call_args.kwargs,
            {"enable_memories": False},
        )

    @patch("hitch.main.views.common.Codex")
    def test_renders_open_pr_menu_link_when_detected(
        self, mock_codex: MagicMock
    ) -> None:
        url = "https://github.com/cberner/hitch/pull/94"
        thread = _thread(
            [
                _turn(
                    [
                        _user_message(system_agents.PR_SLASH_PROMPT),
                        _mcp_tool_call("github", "_create_pull_request", {"url": url}),
                        _agent_message("Opened the PR."),
                    ]
                )
            ]
        )
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            f'<a href="{url}" role="menuitem" target="_blank" rel="noopener noreferrer">Open PR</a>',
            html=True,
        )

    @patch("hitch.main.views.common.Codex")
    def test_stage_reads_sdk_mcp_result_model(self, mock_codex: MagicMock) -> None:
        url = "https://github.com/cberner/hitch/pull/94"
        sdk_result = SimpleNamespace(
            model_dump=lambda by_alias=False: {
                "structuredContent": {
                    "url": url,
                    "state": "closed",
                    "merged": False,
                }
            }
        )
        thread = _thread(
            [
                _turn(
                    [
                        _user_message(system_agents.PR_SLASH_PROMPT),
                        _mcp_tool_call("github", "_create_pull_request", sdk_result),
                        _agent_message("Closed."),
                    ]
                )
            ]
        )
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            f'<a href="{url}" role="menuitem" target="_blank" rel="noopener noreferrer">Open PR</a>',
            html=True,
        )
        self.assertContains(
            response,
            '<span class="stage-badge" data-tone="done">Done: Closed</span>',
        )

    @patch("hitch.main.views.common.Codex")
    def test_hides_open_pr_menu_link_without_detected_pr(
        self, mock_codex: MagicMock
    ) -> None:
        thread = _thread([_turn([_user_message("hi"), _agent_message("Hello.")])])
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "Open PR")

    @patch("hitch.main.views.common.Codex")
    def test_stage_clears_sdk_pr_snapshot_when_latest_pr_turn_has_no_pr(
        self, mock_codex: MagicMock
    ) -> None:
        stale_url = "https://github.com/cberner/hitch/pull/93"
        thread = _thread(
            [
                _turn(
                    [
                        _user_message(system_agents.PR_SLASH_PROMPT),
                        _mcp_tool_call(
                            "github",
                            "_create_pull_request",
                            {
                                "url": stale_url,
                                "state": "closed",
                                "merged": False,
                            },
                        ),
                        _agent_message("Closed."),
                    ]
                ),
                _turn(
                    [
                        _user_message(system_agents.PR_SLASH_PROMPT),
                        _mcp_tool_call(
                            "github",
                            "_create_pull_request",
                            {"content": []},
                        ),
                        _agent_message("No PR."),
                    ]
                ),
            ]
        )
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "Open PR")
        self.assertNotContains(response, "Done: Closed")
        self.assertContains(
            response,
            '<span class="stage-badge" data-tone="active">Implementation</span>',
        )

    @patch("hitch.main.views.common.Codex")
    def test_local_image_entries_redact_server_paths(self, mock_codex: MagicMock) -> None:
        local_image_message = _root(
            SimpleNamespace(
                type="userMessage",
                content=[
                    _root(SimpleNamespace(type="text", text="see attached")),
                    _root(
                        SimpleNamespace(
                            type="localImage",
                            path="/tmp/private/screen.png",
                        )
                    ),
                ],
            )
        )
        thread = _thread([_turn([local_image_message, _agent_message("Done.")])])
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertContains(response, "see attached")
        self.assertContains(response, "[image]")
        self.assertNotContains(response, "/tmp/private/screen.png")

    @patch("hitch.main.views.common.Codex")
    def test_next_message_settings_render_under_title(
        self, mock_codex: MagicMock
    ) -> None:
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=_thread([_turn([_user_message("hi")])]),
            model="gpt-5.4",
            reasoning_effort=SimpleNamespace(value="high"),
        )
        _seed_cookies(
            self.client,
            hitch_sandbox_policy="workspaceWrite",
            hitch_approval_mode="prompt_user",
        )

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'aria-label="Settings for the next message"')
        self.assertContains(response, ">model<")
        self.assertContains(response, "gpt-5.4")
        self.assertContains(response, ">reasoning<")
        self.assertContains(response, "high")
        self.assertContains(response, 'data-normal-value="high"')
        self.assertContains(response, 'data-plan-value="medium"')
        self.assertContains(response, ">sandbox<")
        self.assertContains(response, "Workspace write")
        self.assertContains(response, ">approval<")
        self.assertContains(response, "Always prompt for approval")

    @patch("hitch.main.views.common.Codex")
    def test_system_feedback_renders_with_display_author(
        self, mock_codex: MagicMock
    ) -> None:
        prompt = "Feedback from Hitch QA agent:\n\nFix the failing flow."
        thread = _thread([_turn([_user_message(prompt), _agent_message("fixed")])])
        _patch_thread(self, mock_codex, thread)
        CodexInstance.objects.create(
            pid=1,
            thread_id="thread-1",
            cwd="/tmp/demo",
            prompt=prompt,
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_FEEDBACK,
            display_author="QA agent",
            user_message_index=0,
        )

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<span class="role">QA agent</span>')
        self.assertNotContains(response, '<span class="role">User</span>')

    @patch("hitch.main.views.common.Codex")
    def test_system_feedback_author_uses_turn_index_not_text(
        self, mock_codex: MagicMock
    ) -> None:
        prompt = "Feedback from Hitch QA agent:\n\nFix the failing flow."
        thread = _thread(
            [
                _turn([_user_message(prompt), _agent_message("fixed")]),
                _turn([_user_message(prompt), _agent_message("explained")]),
            ]
        )
        _patch_thread(self, mock_codex, thread)
        CodexInstance.objects.create(
            pid=1,
            thread_id="thread-1",
            cwd="/tmp/demo",
            prompt=prompt,
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_FEEDBACK,
            display_author="QA agent",
            user_message_index=0,
        )

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<span class="role">QA agent</span>', count=1)
        self.assertContains(response, '<span class="role">User</span>', count=1)

    @patch("hitch.main.views.common.Codex")
    def test_demo_agent_turn_is_hidden_from_transcript(
        self, mock_codex: MagicMock
    ) -> None:
        prompt = "Start an interactive web demo for this session.\n\nRegistration token: secret"
        thread = _thread(
            [
                _turn([_user_message("Show the feature"), _agent_message("Done")]),
                _turn(
                    [
                        _user_message(prompt),
                        _agent_message("Registered the demo container."),
                    ]
                ),
            ]
        )
        _patch_thread(self, mock_codex, thread)
        CodexInstance.objects.create(
            pid=1,
            thread_id="thread-1",
            cwd="/tmp/demo",
            prompt=prompt,
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=demo.DEMO_AGENT_KIND,
            display_author=demo.DEMO_DISPLAY_AUTHOR,
            user_message_index=1,
        )

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Show the feature")
        self.assertContains(response, "Done")
        self.assertNotContains(response, "Start an interactive web demo")
        self.assertNotContains(response, "Registration token: secret")
        self.assertNotContains(response, "Registered the demo container")

    @patch("hitch.main.views.common.Codex")
    def test_demo_agent_filter_ignores_stale_user_message_index(
        self, mock_codex: MagicMock
    ) -> None:
        prompt = "Start an interactive web demo for this session.\n\nRegistration token: secret"
        thread = _thread([_turn([_user_message("Show the feature"), _agent_message("Done")])])
        _patch_thread(self, mock_codex, thread)
        CodexInstance.objects.create(
            pid=1,
            thread_id="thread-1",
            cwd="/tmp/demo",
            prompt=prompt,
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=demo.DEMO_AGENT_KIND,
            display_author=demo.DEMO_DISPLAY_AUTHOR,
            user_message_index=0,
        )

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Show the feature")
        self.assertContains(response, "Done")
        self.assertNotContains(response, "Registration token: secret")

    @patch("hitch.main.views.common.Codex")
    def test_demo_agent_filter_preserves_inserted_qa_approval(
        self, mock_codex: MagicMock
    ) -> None:
        prompt = "Start an interactive web demo for this session.\n\nRegistration token: secret"
        thread = _thread(
            [
                _turn([_user_message("Show the feature"), _agent_message("Done")]),
                _turn([_user_message(prompt), _agent_message("Registered container")]),
                _turn([_user_message("Now explain it"), _agent_message("Explained")]),
            ]
        )
        _patch_thread(self, mock_codex, thread)
        CodexInstance.objects.create(
            pid=1,
            thread_id="thread-1",
            cwd="/tmp/demo",
            prompt=prompt,
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=demo.DEMO_AGENT_KIND,
            display_author=demo.DEMO_DISPLAY_AUTHOR,
            user_message_index=1,
        )
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="thread-1",
            cwd="/tmp/demo",
            status=SystemWorkflow.STATUS_COMPLETED,
            step=system_agents.STEP_QA_APPROVED,
            state={
                "next_user_message_index": 2,
                "last_feedback": "No qualifying findings.",
            },
        )
        qa_instance = CodexInstance.objects.create(
            pid=1,
            thread_id="qa-thread",
            cwd="/tmp/demo",
            prompt="review",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            display_author=system_agents.QA_DISPLAY_AUTHOR,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            thread_id="qa-thread",
            instance=qa_instance,
            status=SystemAgentRun.STATUS_COMPLETED,
            output={"feedback": "No qualifying findings.", "lgtm": True},
        )

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "QA agent approved the diff.")
        self.assertContains(response, "Now explain it")
        self.assertNotContains(response, "Registration token: secret")
        self.assertNotContains(response, "Registered container")

    @patch("hitch.main.views.common.Codex")
    def test_active_demo_worker_renders_live_transcript_without_prompt(
        self, mock_codex: MagicMock
    ) -> None:
        prompt = "Start an interactive web demo for this session.\n\nRegistration token: secret"
        thread = _thread([])
        _patch_thread(self, mock_codex, thread)
        CodexInstance.objects.create(
            pid=_LIVE_PID,
            thread_id="thread-1",
            cwd="/tmp/demo",
            prompt=prompt,
            events_path="/tmp/demo-events.jsonl",
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=demo.DEMO_AGENT_KIND,
            display_author=demo.DEMO_DISPLAY_AUTHOR,
            user_message_index=0,
        )

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "Registration token: secret")
        self.assertContains(response, "Demo agent</label>")
        self.assertContains(response, 'id="session-tab-demo-agent" checked')
        self.assertContains(response, "data-live-root")
        self.assertContains(response, 'data-hide-transcript="false"')
        self.assertContains(response, 'data-hide-user-message="true"')
        self.assertContains(response, 'data-sanitize-live-details="true"')
        self.assertContains(response, "data-composer-stop")
        self.assertContains(response, "Demo agent is working")
        html = response.content.decode()
        self.assertIn("Demo setup command approval requested.", html)
        self.assertIn(
            "detail.textContent = HIDE_LIVE_TRANSCRIPT || SANITIZE_LIVE_DETAILS\n"
            "                    ? sanitizedApprovalDetail(payload.method, payload.params)",
            html,
        )
        self.assertNotIn(
            "function handleApprovalRequested(payload) {\n"
            "                if (HIDE_LIVE_TRANSCRIPT) return;",
            html,
        )

    @patch("hitch.main.views.common.Codex")
    def test_active_demo_worker_sanitizes_sensitive_live_details(
        self, mock_codex: MagicMock
    ) -> None:
        prompt = "Start an interactive web demo for this session.\n\nRegistration token: token-secret"
        thread = _thread([])
        _patch_thread(self, mock_codex, thread)
        CodexInstance.objects.create(
            pid=_LIVE_PID,
            thread_id="thread-1",
            cwd="/tmp/demo",
            prompt=prompt,
            events_path="/tmp/demo-events.jsonl",
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=demo.DEMO_AGENT_KIND,
            display_author=demo.DEMO_DISPLAY_AUTHOR,
            user_message_index=0,
        )

        response = _get_session(self.client)
        html = response.content.decode()

        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            self.skipTest(f"playwright unavailable: {exc}")

        timestamp_rows: list[dict[str, str]] = []
        plan_timestamps: list[str] = []
        header_timestamp_count = 0
        tool_box_timestamp_count = 0
        expandable_command_count = 0
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
            except PlaywrightError as exc:
                self.skipTest(f"playwright browser unavailable: {exc}")
            try:
                page = browser.new_page()
                page.evaluate(
                    """
                    () => {
                    class MockEventSource {
                        constructor(url) {
                            this.url = url;
                            this.listeners = {};
                            window.__eventSource = this;
                        }
                        addEventListener(type, callback) {
                            this.listeners[type] = callback;
                        }
                        close() {}
                        emit(type, data) {
                            this.listeners[type]({ data: JSON.stringify(data) });
                        }
                    }
                    window.EventSource = MockEventSource;
                    }
                    """
                )
                page.set_content(html, wait_until="load")
                page.wait_for_function("window.__eventSource !== undefined")
                page.evaluate(
                    """
                    () => {
                        const tokenCommand = [
                            "$HITCH_MANAGE_COMMAND run --project $HITCH_PROJECT_DIR " +
                                "$HITCH_MANAGE_PY register_demo --session-id thread-1 " +
                                "--token=token-secret --status preparing",
                            "$HITCH_MANAGE_COMMAND run --project $HITCH_PROJECT_DIR " +
                                "$HITCH_MANAGE_PY register_demo --session-id thread-1 " +
                                "--token token-secret --status active",
                            "curl -H 'Authorization: Bearer token-secret' https://token-secret.example/start",
                            "podman run --label io.hitch.demo_token=token-secret",
                        ].join(" && ");
                        const unknownCommand = "AWS_SECRET_ACCESS_KEY=token-secret ./run-demo --password token-secret";
                        const sensitiveChanges = [
                            { path: "/tmp/token-secret.txt" },
                            { path: "/home/user/.ssh/id_rsa" },
                        ];
                        window.__eventSource.emit("message", {
                            method: "item/started",
                            recordedAt: 1700000123000000,
                            eventSeq: 1,
                            payload: {
                                item: {
                                    id: "agent-1",
                                    type: "agentMessage",
                                    text: "Working.",
                                },
                            },
                        });
                        window.__eventSource.emit("message", {
                            method: "item/plan/delta",
                            recordedAt: 1700000124000000,
                            eventSeq: 2,
                            payload: {
                                itemId: "plan-1",
                                turnId: "turn-1",
                                delta: "Step 1",
                            },
                        });
                        window.__eventSource.emit("message", {
                            method: "item/started",
                            recordedAt: 1700000125000000,
                            eventSeq: 3,
                            payload: {
                                turnId: "turn-1",
                                item: {
                                    id: "plan-1",
                                    type: "plan",
                                    text: "Step 1",
                                },
                            },
                        });
                        window.__eventSource.emit("message", {
                            method: "item/completed",
                            recordedAt: 1700000126000000,
                            eventSeq: 4,
                            payload: {
                                turnId: "turn-1",
                                item: {
                                    id: "plan-1",
                                    type: "plan",
                                    text: "Step 1\\nStep 2",
                                },
                            },
                        });
                        window.__eventSource.emit("message", {
                            method: "approval/requested",
                            payload: {
                                id: 77,
                                method: "item/commandExecution/requestApproval",
                                params: { item: { command: tokenCommand } },
                            },
                        });
                        window.__eventSource.emit("message", {
                            method: "item/started",
                            payload: {
                                item: {
                                    id: "cmd-1",
                                    type: "commandExecution",
                                    command: tokenCommand,
                                },
                            },
                        });
                        window.__eventSource.emit("message", {
                            method: "item/started",
                            payload: {
                                item: {
                                    id: "cmd-2",
                                    type: "commandExecution",
                                    command: unknownCommand,
                                },
                            },
                        });
                        window.__eventSource.emit("message", {
                            method: "approval/requested",
                            payload: {
                                id: 78,
                                method: "item/fileChange/requestApproval",
                                params: { item: { changes: sensitiveChanges } },
                            },
                        });
                        window.__eventSource.emit("message", {
                            method: "item/started",
                            payload: {
                                item: {
                                    id: "file-1",
                                    type: "fileChange",
                                    changes: sensitiveChanges,
                                },
                            },
                        });
                        window.__eventSource.emit("message", {
                            method: "item/completed",
                            payload: {
                                item: {
                                    id: "cmd-1",
                                    type: "commandExecution",
                                    command: tokenCommand,
                                    status: "completed",
                                },
                            },
                        });
                        window.__eventSource.emit("message", {
                            method: "item/completed",
                            payload: {
                                item: {
                                    id: "cmd-2",
                                    type: "commandExecution",
                                    command: unknownCommand,
                                    status: "completed",
                                },
                            },
                        });
                        window.__eventSource.emit("message", {
                            method: "item/completed",
                            payload: {
                                item: {
                                    id: "file-1",
                                    type: "fileChange",
                                    changes: sensitiveChanges,
                                    status: "completed",
                                },
                            },
                        });
                    }
                    """
                )
                timestamp_rows = cast(
                    list[dict[str, str]],
                    page.locator(".entry time[data-ts]").evaluate_all(
                        """(els) => els.map((el) => ({
                            className: el.className,
                            dateTime: el.dateTime,
                            text: el.textContent,
                            ts: el.dataset.ts,
                        }))"""
                    ),
                )
                plan_timestamps = cast(
                    list[str],
                    page.locator(".plan-header time[data-ts]").evaluate_all(
                        "(els) => els.map((el) => el.dataset.ts || '')"
                    ),
                )
                header_timestamp_count = page.locator(".entry-header time[data-ts]").count()
                tool_box_timestamp_count = page.locator(".tool-call time[data-ts]").count()
                expandable_command_count = page.locator(
                    "[data-expandable-command]"
                ).count()
                body = page.locator("body").inner_text()
            finally:
                browser.close()

        self.assertGreaterEqual(len(timestamp_rows), 5)
        self.assertTrue(any(row["ts"] == "1700000123" for row in timestamp_rows))
        self.assertEqual(plan_timestamps, ["1700000124"])
        self.assertGreaterEqual(header_timestamp_count, 4)
        self.assertEqual(tool_box_timestamp_count, 0)
        self.assertEqual(expandable_command_count, 2)
        self.assertEqual(sum(row["className"] == "timestamp" for row in timestamp_rows), 0)
        for row in timestamp_rows:
            self.assertTrue(row["dateTime"])
            self.assertNotEqual(row["text"], row["ts"])
        self.assertNotIn("token-secret", body)
        self.assertNotIn("--token", body)
        self.assertNotIn("Authorization", body)
        self.assertNotIn("https://token-secret.example", body)
        self.assertNotIn("io.hitch.demo_token", body)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", body)
        self.assertNotIn("run-demo", body)
        self.assertNotIn("--password", body)
        self.assertNotIn("register_demo --status", body)
        self.assertNotIn("podman run", body)
        self.assertNotIn("curl request", body)
        self.assertNotIn("/tmp/token-secret.txt", body)
        self.assertNotIn("/home/user/.ssh/id_rsa", body)
        self.assertIn("Demo setup command approval requested.", body)
        self.assertIn("Command details hidden in demo panel.", body)
        self.assertIn("Demo setup file change approval requested. 2 files", body)
        self.assertIn("Demo setup file change", body)

    @patch("hitch.main.views.common.Codex")
    def test_input_request_focusing_other_field_keeps_selection(
        self, mock_codex: MagicMock
    ) -> None:
        thread = _thread([])
        _patch_thread(self, mock_codex, thread)
        CodexInstance.objects.create(
            pid=_LIVE_PID,
            thread_id="thread-1",
            cwd="/tmp/demo",
            prompt="hello",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_USER,
            user_message_index=0,
        )

        response = _get_session(self.client)
        self.assertEqual(response.status_code, 200)
        html = response.content.decode()
        self.assertIn('data-read-only="false"', html)

        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            self.skipTest(f"playwright unavailable: {exc}")

        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
            except PlaywrightError as exc:
                self.skipTest(f"playwright browser unavailable: {exc}")
            try:
                page = browser.new_page()
                page.evaluate(
                    """
                    () => {
                        class MockEventSource {
                            constructor(url) {
                                this.url = url;
                                this.listeners = {};
                                window.__eventSource = this;
                            }
                            addEventListener(type, cb) { this.listeners[type] = cb; }
                            close() {}
                            emit(type, data) {
                                this.listeners[type]({ data: JSON.stringify(data) });
                            }
                        }
                        window.EventSource = MockEventSource;
                    }
                    """
                )
                page.set_content(html, wait_until="load")
                page.wait_for_function("window.__eventSource !== undefined")
                page.evaluate(
                    """
                    () => window.__eventSource.emit("message", {
                        method: "input/requested",
                        eventSeq: 1,
                        payload: {
                            id: "req-1",
                            params: { questions: [{
                                id: "q1",
                                question: "Pick one",
                                options: [{ label: "Alpha" }, { label: "Beta" }],
                            }] },
                        },
                    })
                    """
                )
                page.wait_for_function(
                    "document.querySelectorAll('.input-options button').length === 2"
                )
                # Select the second option explicitly.
                page.evaluate(
                    "() => document.querySelectorAll('.input-options button')[1].click()"
                )
                pressed = (
                    "() => document.querySelector("
                    "\"[data-input-request-id] .input-options button[aria-pressed='true']\")"
                )
                page.wait_for_function(
                    f"{pressed}.querySelector('.input-option-label').textContent === 'Beta'"
                )
                # Focusing the empty Other field must not wipe the selection.
                page.evaluate("() => document.querySelector('.input-other').focus()")
                self.assertEqual(
                    page.evaluate(
                        "() => document.querySelectorAll("
                        "\".input-options button[aria-pressed='true']\").length"
                    ),
                    1,
                )
                self.assertEqual(
                    page.evaluate(
                        f"{pressed}.querySelector('.input-option-label').textContent"
                    ),
                    "Beta",
                )
            finally:
                browser.close()

    @patch("hitch.main.views.common.Codex")
    def test_goal_updates_ignore_stale_ordering(
        self, mock_codex: MagicMock
    ) -> None:
        # Exercises the shared event-order comparator: a goal update with an
        # older recordedAt must not overwrite a newer one.
        thread = _thread([])
        _patch_thread(self, mock_codex, thread)
        CodexInstance.objects.create(
            pid=_LIVE_PID,
            thread_id="thread-1",
            cwd="/tmp/demo",
            prompt="hello",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_USER,
            user_message_index=0,
        )

        html = _get_session(self.client).content.decode()

        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            self.skipTest(f"playwright unavailable: {exc}")

        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
            except PlaywrightError as exc:
                self.skipTest(f"playwright browser unavailable: {exc}")
            try:
                page = browser.new_page()
                page.evaluate(
                    """
                    () => {
                        class MockEventSource {
                            constructor(url) {
                                this.url = url;
                                this.listeners = {};
                                window.__eventSource = this;
                            }
                            addEventListener(type, cb) { this.listeners[type] = cb; }
                            close() {}
                            emit(type, data) {
                                this.listeners[type]({ data: JSON.stringify(data) });
                            }
                        }
                        window.EventSource = MockEventSource;
                    }
                    """
                )
                page.set_content(html, wait_until="load")
                page.wait_for_function("window.__eventSource !== undefined")

                def emit_goal(recorded_at: int, objective: str) -> None:
                    page.evaluate(
                        """
                        (args) => window.__eventSource.emit("message", {
                            method: "thread/goal/updated",
                            recordedAt: args.recordedAt,
                            eventSeq: args.recordedAt,
                            payload: { goal: { objective: args.objective } },
                        })
                        """,
                        {"recordedAt": recorded_at, "objective": objective},
                    )

                goal_text = (
                    "() => document.querySelector('[data-live-goal-text]').textContent"
                )
                emit_goal(200, "newer goal")
                page.wait_for_function(f"{goal_text} === 'newer goal'")
                # A stale (older recordedAt) update is ignored.
                emit_goal(100, "stale goal")
                self.assertEqual(page.evaluate(goal_text), "newer goal")
                # A genuinely newer update applies.
                emit_goal(300, "latest goal")
                page.wait_for_function(f"{goal_text} === 'latest goal'")
            finally:
                browser.close()

    @patch("hitch.main.views.common.Codex")
    def test_registered_demo_status_renders_logs(
        self, mock_codex: MagicMock
    ) -> None:
        thread = _thread([])
        _patch_thread(self, mock_codex, thread)
        SessionDemo.objects.create(
            thread_id="thread-1",
            port=3000,
            status=SessionDemo.STATUS_PREPARING,
            generation=1,
            container_name="hitch-demo-thread-1-abcd",
            logs="installing dependencies",
        )

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Demo preparing")
        self.assertContains(response, "hitch-demo-thread-1-abcd")
        self.assertContains(response, "installing dependencies")

    @patch("hitch.main.views.common.Codex")
    def test_failed_demo_status_renders_error_without_approval_ui(
        self, mock_codex: MagicMock
    ) -> None:
        thread = _thread([])
        _patch_thread(self, mock_codex, thread)
        SessionDemo.objects.create(
            thread_id="thread-1",
            port=3000,
            status=SessionDemo.STATUS_FAILED,
            generation=1,
            last_error="server crashed",
            logs="traceback",
        )

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Demo failed")
        self.assertContains(response, "server crashed")
        self.assertContains(response, "traceback")
        self.assertNotContains(response, "Start demo?")

    @patch("hitch.main.views.common.Codex")
    def test_failed_demo_links_to_system_session_with_full_demo_history(
        self, mock_codex: MagicMock
    ) -> None:
        prompt = "Start an interactive web demo for this session.\n\nRegistration token: secret"
        thread = _thread(
            [
                _turn([_user_message("Show the feature"), _agent_message("Done")]),
                _turn([_user_message(prompt), _agent_message("container failed")]),
            ]
        )
        _patch_thread(self, mock_codex, thread)
        SessionDemo.objects.create(
            thread_id="thread-1",
            port=3000,
            status=SessionDemo.STATUS_FAILED,
            generation=1,
            last_error="demo agent finished without registering a container",
        )
        workflow = SystemWorkflow.objects.create(
            kind=demo.DEMO_WORKFLOW_KIND,
            main_thread_id="thread-1",
            cwd="/tmp/demo",
            status=SystemWorkflow.STATUS_FAILED,
        )
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="thread-1",
            cwd="/tmp/demo",
            prompt=prompt,
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=demo.DEMO_AGENT_KIND,
            display_author=demo.DEMO_DISPLAY_AUTHOR,
        )
        demo_run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=demo.DEMO_AGENT_KIND,
            thread_id="thread-1",
            instance=instance,
            status=SystemAgentRun.STATUS_FAILED,
        )
        qa_workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="thread-1",
            cwd="/tmp/demo",
            status=SystemWorkflow.STATUS_COMPLETED,
        )
        qa_instance = CodexInstance.objects.create(
            pid=1,
            thread_id="thread-1",
            cwd="/tmp/demo",
            prompt="Review the diff",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=qa_workflow.pk,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            display_author=system_agents.QA_DISPLAY_AUTHOR,
        )
        SystemAgentRun.objects.create(
            workflow=qa_workflow,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            thread_id="thread-1",
            instance=qa_instance,
            status=SystemAgentRun.STATUS_COMPLETED,
        )

        response = _get_session(self.client)
        demo_log_url = (
            f"{reverse('system_session', kwargs={'session_id': 'thread-1'})}"
            f"?run_id={demo_run.pk}"
        )
        system_response = self.client.get(
            demo_log_url,
        )

        self.assertContains(response, "View demo agent session")
        self.assertContains(response, demo_log_url)
        self.assertNotContains(response, "Registration token: secret")
        self.assertNotContains(response, "container failed")
        self.assertEqual(system_response.status_code, 200)
        self.assertContains(system_response, "Demo agent log")
        self.assertContains(system_response, "Registration token: secret")
        self.assertContains(system_response, "container failed")

    @patch("hitch.main.views.common.Codex")
    def test_next_message_model_comes_only_from_resumed_thread(
        self, mock_codex: MagicMock
    ) -> None:
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=_thread([_turn([_user_message("hi")])]),
            reasoning_effort=SimpleNamespace(value="medium"),
        )
        _seed_cookies(self.client, hitch_model="gpt-stale")

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, ">model<")
        self.assertContains(response, 'data-normal-value="Unknown"')
        self.assertContains(response, 'data-plan-value="gpt-stale"')

    @patch("hitch.main.views.common.build_worktree_diff")
    @patch("hitch.main.views.common.Codex")
    def test_renders_diff_viewer_entry_points_and_modal(
        self, mock_codex: MagicMock, mock_diff: MagicMock
    ) -> None:
        mock_diff.return_value = _diff_view()
        _patch_thread(self, mock_codex, _thread([_turn([_user_message("hi")])]))

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'role="menuitem" data-diff-open')
        self.assertContains(response, 'class="diff-fab"')
        self.assertContains(response, '<dialog class="diff-modal"', html=False)
        self.assertContains(response, "Working tree diff")
        self.assertContains(response, "hitch/main/views.py")
        self.assertContains(response, '<span class="k">return</span> 2', html=False)

    @patch("hitch.main.views.common.build_worktree_diff")
    @patch("hitch.main.views.common.Codex")
    def test_diff_menu_item_is_disabled_without_changes(
        self, mock_codex: MagicMock, mock_diff: MagicMock
    ) -> None:
        mock_diff.return_value = DiffView(files=[])
        _patch_thread(self, mock_codex, _thread([_turn([_user_message("hi")])]))

        response = _get_session(self.client)

        self.assertContains(response, "data-diff-open disabled")
        self.assertNotContains(response, 'class="diff-fab"')
        self.assertNotContains(response, '<dialog class="diff-modal"', html=False)

    @patch("hitch.main.views.common.Codex")
    def test_archived_session_menu_offers_unarchive(self, mock_codex: MagicMock) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        rollout_path = (
            Path(temp_dir.name)
            / "archived_sessions"
            / "2026"
            / "05"
            / "15"
            / "rollout-thread-1.jsonl"
        )
        rollout_path.parent.mkdir(parents=True)
        rollout_path.write_text(
            _rollout_line("event_msg", {"type": "user_message", "message": "hi"}) + "\n"
        )
        thread = _thread(
            [_turn([_user_message("hi")])],
            path=str(rollout_path),
        )
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'name="archived" value="false"')
        self.assertContains(response, 'role="menuitem">Unarchive</button>')
        self.assertNotContains(response, 'role="menuitem">Archive</button>')

    @patch("hitch.main.views.common.Codex")
    def test_archived_menu_label_does_not_depend_on_thread_list(
        self, mock_codex: MagicMock
    ) -> None:
        thread = _thread([_turn([_user_message("hi")])])
        _patch_thread(self, mock_codex, thread)
        client = mock_codex.return_value.__enter__.return_value
        client.thread_list.side_effect = CodexError("archived list unavailable")

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'role="menuitem">Archive</button>')
        client.thread_list.assert_not_called()

    @patch("hitch.main.views.common.Codex")
    def test_session_menu_offers_demo_actions(self, mock_codex: MagicMock) -> None:
        SessionDemo.objects.create(
            thread_id="thread-1",
            host="127.0.0.1",
            port=45678,
            status=SessionDemo.STATUS_ACTIVE,
        )
        _patch_thread(self, mock_codex, _thread([_turn([_user_message("hi")])]))

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            f'action="{reverse("start_session_demo", kwargs={"session_id": "thread-1"})}"',
        )
        self.assertContains(response, "Start demo</button>")
        self.assertNotContains(response, "disabled>Start demo</button>")
        self.assertContains(response, 'href="http://testserver/sessions/thread-1/demo/"')
        self.assertContains(response, 'role="menuitem" target="_blank" rel="noopener noreferrer">Open demo</a>')

    @patch("hitch.main.views.common.Codex")
    def test_start_demo_menu_item_disabled_during_system_workflow(
        self, mock_codex: MagicMock
    ) -> None:
        SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="thread-1",
            cwd="/tmp/demo",
            status=SystemWorkflow.STATUS_RUNNING,
        )
        _patch_thread(self, mock_codex, _thread([_turn([_user_message("hi")])]))

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "disabled>Start demo</button>")

    @patch("hitch.main.views.common.Codex")
    def test_start_demo_menu_item_disabled_during_active_turn(
        self, mock_codex: MagicMock
    ) -> None:
        CodexInstance.objects.create(
            pid=_LIVE_PID,
            thread_id="thread-1",
            cwd="/tmp/demo",
            prompt="user turn",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_RUNNING,
        )
        _patch_thread(self, mock_codex, _thread([_turn([_user_message("hi")])]))

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "disabled>Start demo</button>")

    @patch("hitch.main.views.common.Codex")
    def test_topbar_title_truncates_long_preview(self, mock_codex: MagicMock) -> None:
        # The session page topbar title uses the same `_display_title` as the
        # index, so an unnamed thread with a long preview gets clipped here too.
        thread = _thread([_turn([_user_message("x" * 200)])], name=None, preview="x" * 200)
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)
        body = response.content.decode()

        self.assertEqual(response.status_code, 200)
        # The topbar title contains the clipped title, not the full 200-char preview.
        self.assertIn('<div class="topbar-title">' + "x" * 80 + "...</div>", body)

    @patch("hitch.main.views.common.Codex")
    def test_session_template_receives_slim_thread_context(
        self, mock_codex: MagicMock
    ) -> None:
        thread = _thread([_turn([_user_message("hi")])])
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        template_thread = cast(Any, response).context["thread"]
        self.assertEqual(template_thread.id, "thread-1")
        self.assertEqual(template_thread.cwd, "/tmp/demo")
        self.assertEqual(template_thread.updated_at, 1700000000)
        self.assertFalse(hasattr(template_thread, "turns"))

    @patch("hitch.main.views.common.Codex")
    def test_renders_messages_tool_calls_and_timestamps(
        self, mock_codex: MagicMock
    ) -> None:
        """A representative turn: user prompt, agent reply, several tool
        calls (each rendered as its own row, including an unknown type),
        and the turn's started_at flowing through as ``data-ts``."""
        thread = _thread(
            [
                _turn(
                    [
                        _user_message("Refactor the login flow"),
                        _command("./scripts/build.sh"),
                        _command("./scripts/test.sh"),
                        _file_change("hitch/main/views.py"),
                        _tool_call("brandNewTool"),
                        _agent_message("Sure, here is the plan."),
                    ],
                    started_at=1700000123,
                ),
            ]
        )
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)
        body = response.content.decode()

        self.assertEqual(response.status_code, 200)
        # ``thread/read`` requires the thread to already be loaded into the
        # codex app-server; the view uses ``thread/resume`` so a brand-new
        # session (or any thread the request's app-server hasn't seen) can
        # be loaded from disk in the same call that fetches the turns.
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.assert_called_once_with("thread-1")
        self.assertContains(response, "Demo session")
        self.assertContains(response, "Refactor the login flow")
        self.assertContains(response, "Sure, here is the plan.")
        self.assertContains(response, ">User<")
        self.assertContains(response, ">Agent<")
        self.assertContains(response, "./scripts/build.sh")
        self.assertContains(response, "./scripts/test.sh")
        self.assertContains(response, "hitch/main/views.py")
        # Unmapped types fall back to the raw type tag so nothing is hidden.
        self.assertContains(response, "brandNewTool")
        # Detailed tool calls render as separate boxes, while label-only
        # unknown tool calls still remain visible in the header row.
        self.assertEqual(body.count('class="tool-call"'), 3)
        self.assertContains(response, 'data-ts="1700000123"')
        self.assertEqual(
            body.count('<time data-ts="1700000123">1700000123</time>'), 6
        )
        self.assertNotContains(response, '<time class="timestamp"')
        self.assertNotContains(response, 'data-format="time"')
        self.assertContains(response, 'timeZoneName: "short"', count=2)
        self.assertContains(response, "formatTimestamps(document);")
        self.assertContains(response, "formatTimestamps(body);")

    @patch("hitch.main.views.common.Codex")
    def test_command_detail_expands_from_single_line(
        self, mock_codex: MagicMock
    ) -> None:
        long_command = "printf first-line\nprintf second-line " + "argument " * 80
        thread = _thread(
            [
                _turn(
                    [
                        _user_message("Run the command"),
                        _command(long_command),
                        _agent_message("Done."),
                    ]
                )
            ]
        )
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)
        html = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'class="detail command-detail"')
        self.assertContains(response, "data-expandable-command")
        self.assertContains(response, 'aria-expanded="false"')
        self.assertContains(response, 'title="Expand command"')
        self.assertIn(long_command, html)

        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            self.skipTest(f"playwright unavailable: {exc}")

        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
            except PlaywrightError as exc:
                self.skipTest(f"playwright browser unavailable: {exc}")
            try:
                page = browser.new_page(viewport={"width": 480, "height": 800})
                page.set_content(html, wait_until="load")
                page.locator("details.intermediate").evaluate(
                    "(el) => { el.open = true; }"
                )
                command = page.locator("[data-expandable-command]")

                self.assertEqual(command.get_attribute("aria-expanded"), "false")
                self.assertEqual(
                    command.evaluate("(el) => getComputedStyle(el).whiteSpace"),
                    "nowrap",
                )
                collapsed_height = command.evaluate(
                    "(el) => el.getBoundingClientRect().height"
                )
                self.assertTrue(
                    command.evaluate("(el) => el.scrollWidth > el.clientWidth")
                )

                command.click()

                self.assertEqual(command.get_attribute("aria-expanded"), "true")
                self.assertEqual(command.get_attribute("title"), "Collapse command")
                self.assertEqual(
                    command.evaluate("(el) => getComputedStyle(el).whiteSpace"),
                    "pre-wrap",
                )
                self.assertGreater(
                    command.evaluate("(el) => el.getBoundingClientRect().height"),
                    collapsed_height,
                )

                command.click()
                self.assertEqual(command.get_attribute("aria-expanded"), "false")
            finally:
                browser.close()

    @patch("hitch.main.views.common.Codex")
    def test_sdk_memory_citation_renders_details(self, mock_codex: MagicMock) -> None:
        memory_citation = SimpleNamespace(
            entries=[
                SimpleNamespace(
                    path="MEMORY.md",
                    line_start=1,
                    line_end=2,
                    note="project convention",
                )
            ],
            thread_ids=["019cc2ea-1dff-7902-8d40-c8f6e5d83cc4"],
        )
        thread = _thread(
            [
                _turn(
                    [
                        _user_message("Use prior context"),
                        _agent_message(
                            "Applied the remembered convention.",
                            memory_citation=memory_citation,
                        ),
                    ]
                )
            ]
        )
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Applied the remembered convention.")
        # 1 file-line entry + 1 prior-session id; both render in the popover,
        # so the summary count must include both kinds.
        self.assertContains(response, "Memories used: 2")
        self.assertContains(response, "MEMORY.md:1-2")
        self.assertContains(response, "project convention")
        self.assertContains(response, "019cc2ea-1dff-7902-8d40-c8f6e5d83cc4")
        self.assertContains(response, "function renderMemoryCitation")
        self.assertContains(response, "item.memoryCitation || item.memory_citation")

    @patch("hitch.main.views.common.Codex")
    def test_messages_are_copyable_by_long_press(self, mock_codex: MagicMock) -> None:
        thread = _thread(
            [
                _turn(
                    [
                        _user_message("Copy my prompt"),
                        _agent_message("Thinking", phase="commentary"),
                        _agent_message("Copy my answer"),
                    ]
                )
            ]
        )
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertContains(
            response,
            '<div class="message user" data-copyable-message>',
            html=False,
        )
        self.assertContains(
            response,
            '<div class="message thinking" data-copyable-message>',
            html=False,
        )
        self.assertContains(
            response,
            '<div class="message agent" data-copyable-message>',
            html=False,
        )
        self.assertContains(response, 'target.closest("[data-copyable-message]")')
        self.assertContains(response, "navigator.clipboard.writeText(text)")
        self.assertContains(response, 'document.addEventListener("pointerdown"')
        self.assertContains(response, 'msg.dataset.copyableMessage = "";')

    @patch("hitch.main.views.common.Codex")
    def test_trailing_tool_calls_are_rendered(self, mock_codex: MagicMock) -> None:
        # Mid-turn agent commentary followed by a tool call (no final agent
        # message) still surfaces the tool call.
        thread = _thread(
            [
                _turn(
                    [
                        _user_message("Investigate"),
                        _agent_message("Looking into it."),
                        _command("rg --files"),
                    ]
                ),
            ]
        )
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "rg --files")

    @patch("hitch.main.views.common.Codex")
    def test_failed_tool_calls_show_status(self, mock_codex: MagicMock) -> None:
        thread = _thread(
            [
                _turn(
                    [
                        _user_message("Run tests"),
                        _command("just test", status="failed"),
                    ]
                ),
            ]
        )
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "failed")

    @patch("hitch.main.views.common.Codex")
    def test_empty_session_shows_placeholder(self, mock_codex: MagicMock) -> None:
        _patch_thread(self, mock_codex, _thread([]))

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No messages in this session yet.")


class RolloutFileViewTests(TestCase):
    """When ``thread.path`` points at a rollout file, the view parses it
    directly to recover the commandExecution items codex strips from
    ``thread/read``; failure modes (missing, malformed, empty, or
    contentless rollouts) fall back to the SDK-derived entries instead of
    bubbling the error to the response.
    """

    @patch("hitch.main.views.common.Codex")
    def test_prefers_rollout_file_when_path_is_set(self, mock_codex: MagicMock) -> None:
        rollout_lines = [
            _rollout_line(
                "event_msg",
                {"type": "user_message", "message": "build it"},
                timestamp="2026-04-14T23:05:00Z",
            ),
            _rollout_line(
                "event_msg",
                {"type": "agent_message", "message": "looking into it"},
                timestamp="2026-04-14T23:05:05Z",
            ),
            _rollout_line(
                "response_item",
                {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": json.dumps({"cmd": "cargo build --release"}),
                    "call_id": "c1",
                },
                timestamp="2026-04-14T23:05:10Z",
            ),
            _rollout_line(
                "event_msg",
                {"type": "agent_message", "message": "done"},
                timestamp="2026-04-14T23:05:30Z",
            ),
        ]
        rollout_path = _make_rollout(self, rollout_lines)

        _patch_thread(self, mock_codex, _thread([], path=str(rollout_path)))

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "build it")
        self.assertContains(response, "cargo build --release")
        self.assertContains(response, "done")
        self.assertContains(response, '<details class="intermediate">')
        self.assertContains(response, "1 thinking message and 1 tool call")

    @patch("hitch.main.views.common.Codex")
    def test_rollout_memory_citation_renders_details(self, mock_codex: MagicMock) -> None:
        raw_text = (
            "Used prior context."
            "<oai-mem-citation>"
            "<citation_entries>\n"
            "MEMORY.md:3-4|note=[repo preference]\n"
            "</citation_entries>\n"
            "<rollout_ids>\n"
            "019cc2ea-1dff-7902-8d40-c8f6e5d83cc4\n"
            "</rollout_ids>"
            "</oai-mem-citation>"
        )
        rollout_lines = [
            _rollout_line("event_msg", {"type": "user_message", "message": "remember?"}),
            _rollout_line(
                "response_item",
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": raw_text}],
                },
            ),
        ]
        rollout_path = _make_rollout(self, rollout_lines)
        _patch_thread(self, mock_codex, _thread([], path=str(rollout_path)))

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Used prior context.")
        # 1 file-line entry + 1 prior-session id; both render in the popover,
        # so the summary count must include both kinds.
        self.assertContains(response, "Memories used: 2")
        self.assertContains(response, "MEMORY.md:3-4")
        self.assertContains(response, "repo preference")
        self.assertNotContains(response, "oai-mem-citation")

    @patch("hitch.main.views.common.Codex")
    def test_rejected_command_renders_approval_choice(self, mock_codex: MagicMock) -> None:
        rejection = (
            "exec_command failed for `/bin/bash -lc 'printf Reason: command && git push origin master'`: "
            'CreateProcess { message: "Rejected(\\"This action was rejected'
            "\\\\nReason: Pushing directly to origin/master is risky."
            '\\\\nStop and request user input.\\\\")" }'
        )
        rollout_lines = [
            _rollout_line("event_msg", {"type": "user_message", "message": "push it"}),
            _rollout_line(
                "response_item",
                {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": json.dumps({"cmd": "git push origin master"}),
                    "call_id": "push",
                },
            ),
            _rollout_line(
                "response_item",
                {
                    "type": "function_call_output",
                    "call_id": "push",
                    "output": rejection,
                },
            ),
            _rollout_line(
                "event_msg",
                {
                    "type": "agent_message",
                    "message": "Please confirm explicitly.",
                    "phase": "final_answer",
                },
            ),
        ]
        rollout_path = _make_rollout(self, rollout_lines)
        _patch_thread(self, mock_codex, _thread([], path=str(rollout_path)))

        response = _get_session(self.client)
        body = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Approval declined")
        self.assertContains(response, "git push origin master")
        self.assertContains(response, "Pushing directly to origin/master is risky.")
        self.assertNotContains(response, "I explicitly approve")
        self.assertNotContains(response, "No, do not run this command")
        self.assertLess(body.index("</details>"), body.index("Approval declined"))
        self.assertLess(body.index("Approval declined"), body.index("Please confirm explicitly."))

    @patch("hitch.main.views.common.Codex")
    def test_plan_mode_rollout_renders_final_plan_card(
        self, mock_codex: MagicMock
    ) -> None:
        plan = "# Fix Login CSRF Redirect Failure\n\n## Summary\nRead the CSRF cookie at submit time."
        rollout_lines = [
            _rollout_line(
                "event_msg",
                {"type": "user_message", "message": "Debug the login CSRF issue"},
            ),
            _rollout_line(
                "response_item",
                {
                    "type": "function_call",
                    "name": "request_user_input",
                    "arguments": json.dumps(
                        {
                            "questions": [
                                {
                                    "id": "failure_point",
                                    "header": "Failure",
                                    "question": "Where do you see the CSRF failure?",
                                    "options": [
                                        {
                                            "label": "Main-page action (Recommended)",
                                            "description": "The next unsafe action fails.",
                                        },
                                        {
                                            "label": "Immediate page",
                                            "description": "Login lands on the failure page.",
                                        },
                                    ],
                                }
                            ]
                        }
                    ),
                    "call_id": "ask",
                },
            ),
            _rollout_line(
                "response_item",
                {
                    "type": "function_call_output",
                    "call_id": "ask",
                    "output": json.dumps({"answers": {}}),
                },
            ),
            _rollout_line(
                "event_msg",
                {
                    "type": "item_completed",
                    "item": {"type": "Plan", "id": "turn-plan", "text": plan},
                },
                timestamp="1970-01-01T00:01:23Z",
            ),
            _rollout_line(
                "response_item",
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": f"<proposed_plan>\n{plan}\n</proposed_plan>",
                        }
                    ],
                    "phase": "final_answer",
                },
            ),
        ]
        rollout_path = _make_rollout(self, rollout_lines)
        _patch_thread(self, mock_codex, _thread([], path=str(rollout_path)))

        response = _get_session(self.client)
        body = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'class="plan-card"')
        self.assertContains(response, "Fix Login CSRF Redirect Failure")
        self.assertContains(response, "Read the CSRF cookie at submit time.")
        self.assertContains(response, "Approve plan")
        self.assertContains(response, 'name="plan_action" value="approve"')
        self.assertContains(response, 'name="collaboration_mode" value="default"')
        self.assertContains(response, "Implement the plan.")
        self.assertContains(response, 'name="plan_action" value="revise"')
        self.assertContains(response, 'data-initial-plan-mode="true"')
        self.assertContains(response, 'name="plan_mode" value="true" data-plan-mode-input')
        self.assertContains(response, 'name="default_plan_mode" value="true"')
        self.assertContains(response, '<time data-ts="83">83</time>')
        self.assertNotContains(response, '<time data-ts="83" data-format="time">83</time>')
        self.assertContains(
            response,
            'name="plan_mode_explicit" value="" data-plan-mode-explicit-input',
        )
        self.assertNotContains(response, "&lt;proposed_plan&gt;")
        self.assertNotContains(response, "request_user_input")
        self.assertNotContains(response, '<details class="intermediate">')
        self.assertIn("Debug the login CSRF issue", body)

    @patch("hitch.main.views.common.Codex")
    def test_resolved_plan_card_hides_approval_actions(
        self, mock_codex: MagicMock
    ) -> None:
        plan = "# Fix Login CSRF Redirect Failure\n\n## Summary\nRead the CSRF cookie at submit time."
        rollout_lines = [
            _rollout_line(
                "event_msg",
                {"type": "user_message", "message": "Debug the login CSRF issue"},
            ),
            _rollout_line(
                "event_msg",
                {
                    "type": "item_completed",
                    "item": {"type": "Plan", "id": "turn-plan", "text": plan},
                },
            ),
            _rollout_line(
                "response_item",
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": f"<proposed_plan>\n{plan}\n</proposed_plan>",
                        }
                    ],
                    "phase": "final_answer",
                },
            ),
            _rollout_line(
                "event_msg",
                {"type": "user_message", "message": "Implement the plan."},
            ),
            _rollout_line(
                "response_item",
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Done."}],
                    "phase": "final_answer",
                },
            ),
        ]
        rollout_path = _make_rollout(self, rollout_lines)
        _patch_thread(self, mock_codex, _thread([], path=str(rollout_path)))

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'class="plan-card"')
        self.assertContains(response, "Fix Login CSRF Redirect Failure")
        self.assertNotContains(response, "Approve plan")
        self.assertNotContains(response, 'name="plan_action" value="approve"')
        self.assertContains(response, 'data-initial-plan-mode="false"')
        self.assertContains(response, 'name="default_plan_mode" value=""')

    @patch("hitch.main.views.common.Codex")
    def test_falls_back_to_sdk_on_rollout_failure_modes(
        self, mock_codex: MagicMock
    ) -> None:
        """Unreadable / unparseable / message-less / empty rollouts each fall
        through to the SDK-derived turns rather than crashing or dropping
        the page back to an empty state."""
        sdk_turns = [_turn([_user_message("hi"), _agent_message("hello")])]

        # (label, thread.path) — each path is a different failure mode.
        cases: list[tuple[str, str]] = [
            ("missing path", "/nonexistent/rollout.jsonl"),
            (
                "binary garbage that survives line splitting",
                str(_make_rollout(self, [], binary=b"\xff\xfe\x00not json\n")),
            ),
            (
                "tool-only rollout (no user/agent messages)",
                str(
                    _make_rollout(
                        self,
                        [
                            _rollout_line(
                                "response_item",
                                {
                                    "type": "function_call",
                                    "name": "exec_command",
                                    "arguments": json.dumps({"cmd": "uname -a"}),
                                    "call_id": "c1",
                                },
                            ),
                        ],
                    )
                ),
            ),
            ("empty rollout file", str(_make_rollout(self, []))),
        ]

        for label, path in cases:
            with self.subTest(label=label):
                _patch_thread(self, mock_codex, _thread(sdk_turns, path=path))
                response = _get_session(self.client)
                self.assertEqual(response.status_code, 200)
                self.assertContains(response, "hi")
                self.assertContains(response, "hello")
                # The tool-only rollout's commandExecution must NOT leak into
                # the response — the SDK path took over entirely.
                if "tool-only" in label:
                    self.assertNotContains(response, "uname -a")

    @patch("hitch.main.views.common.Codex")
    def test_empty_rollout_with_no_sdk_turns_renders_placeholder(
        self, mock_codex: MagicMock
    ) -> None:
        # Both the rollout and Thread.turns are empty — the page should still
        # render its empty-state placeholder rather than fall back to a
        # second SDK call.
        rollout_path = _make_rollout(self, [])
        _patch_thread(self, mock_codex, _thread([], path=str(rollout_path)))

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No messages in this session yet.")

    @patch("hitch.main.views.common.Codex")
    def test_token_usage_renders_under_title(self, mock_codex: MagicMock) -> None:
        # The most recent token_count event's non-cached input and output
        # surface under the session title with thousands separators, while
        # context usage comes from the non-cumulative active context token
        # count. Threads with no token_count event in their rollout hide the
        # section entirely.
        rollout_lines = [
            _rollout_line("event_msg", {"type": "user_message", "message": "go"}),
            _rollout_line(
                "event_msg",
                {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {
                            "input_tokens": 12345,
                            "cached_input_tokens": 678,
                            "output_tokens": 9012,
                            "reasoning_output_tokens": 3,
                            "total_tokens": 22038,
                        },
                        "last_token_usage": {
                            "input_tokens": 10000,
                            "cached_input_tokens": 500,
                            "output_tokens": 8000,
                            "reasoning_output_tokens": 3,
                            "total_tokens": 22038,
                        },
                        "model_context_window": 200000,
                    },
                },
            ),
        ]
        path = _make_rollout(self, rollout_lines)
        _patch_thread(self, mock_codex, _thread([], path=str(path)))

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'class="usage"')
        self.assertContains(response, ">context<")
        self.assertContains(response, "11%")
        self.assertContains(response, "22,038 of 200,000 tokens in current context")
        self.assertContains(response, ">in<")
        self.assertContains(response, ">out<")
        self.assertContains(response, ">cached<")
        self.assertContains(response, "11,667")
        self.assertNotContains(response, "12,345")
        self.assertContains(response, "9,012")
        self.assertContains(response, "678")

        # Same view with no token_count events: usage row is omitted.
        empty_path = _make_rollout(
            self,
            [_rollout_line("event_msg", {"type": "user_message", "message": "hi"})],
        )
        _patch_thread(self, mock_codex, _thread([], path=str(empty_path)))
        response = _get_session(self.client)
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, 'class="usage"')

    @patch("hitch.main.views.common.Codex")
    def test_rollout_groups_multiple_turns_separately(self, mock_codex: MagicMock) -> None:
        # Two user messages in the rollout should produce two independent
        # intermediate blocks — one per turn — rather than one giant block.
        rollout_lines = [
            _rollout_line("event_msg", {"type": "user_message", "message": "first ask"}),
            _rollout_line(
                "response_item",
                {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": json.dumps({"cmd": "echo a"}),
                    "call_id": "c1",
                },
            ),
            _rollout_line("event_msg", {"type": "agent_message", "message": "first reply"}),
            _rollout_line("event_msg", {"type": "user_message", "message": "second ask"}),
            _rollout_line(
                "response_item",
                {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": json.dumps({"cmd": "echo b"}),
                    "call_id": "c2",
                },
            ),
            _rollout_line("event_msg", {"type": "agent_message", "message": "second reply"}),
        ]
        rollout_path = _make_rollout(self, rollout_lines)
        _patch_thread(self, mock_codex, _thread([], path=str(rollout_path)))

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content.decode().count('<details class="intermediate">'), 2)

    @patch("hitch.main.views.common.Codex")
    def test_rollout_turn_with_only_tool_calls_has_no_final_agent(
        self, mock_codex: MagicMock
    ) -> None:
        # An interrupted turn that produced no agent reply should still
        # render, with the tool call(s) inside the intermediate block and no
        # top-level agent message.
        rollout_lines = [
            _rollout_line("event_msg", {"type": "user_message", "message": "try this"}),
            _rollout_line(
                "response_item",
                {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": json.dumps({"cmd": "sleep 1"}),
                    "call_id": "c1",
                },
            ),
        ]
        rollout_path = _make_rollout(self, rollout_lines)
        _patch_thread(self, mock_codex, _thread([], path=str(rollout_path)))

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "try this")
        self.assertContains(response, "sleep 1")
        self.assertContains(response, '<details class="intermediate">')
        self.assertNotContains(response, ">Agent<")

    @patch("hitch.main.views.common.Codex")
    def test_rollout_commentary_phase_never_treated_as_final(
        self, mock_codex: MagicMock
    ) -> None:
        # Scanning from the end must skip the trailing tool call AND both
        # commentary messages, falling through to "no final agent".
        rollout_lines = [
            _rollout_line("event_msg", {"type": "user_message", "message": "go"}),
            _rollout_line(
                "event_msg",
                {"type": "agent_message", "message": "preamble", "phase": "commentary"},
            ),
            _rollout_line(
                "event_msg",
                {"type": "agent_message", "message": "narrating", "phase": "commentary"},
            ),
            _rollout_line(
                "response_item",
                {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": json.dumps({"cmd": "ls"}),
                    "call_id": "c1",
                },
            ),
        ]
        rollout_path = _make_rollout(self, rollout_lines)
        _patch_thread(self, mock_codex, _thread([], path=str(rollout_path)))

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "preamble")
        self.assertContains(response, "narrating")
        self.assertNotContains(response, ">Agent<")
        self.assertContains(response, "2 thinking messages and 1 tool call")

    @patch("hitch.main.views.common.Codex")
    def test_rollout_final_answer_phase_wins_over_later_unphased(
        self, mock_codex: MagicMock
    ) -> None:
        # The explicit final_answer is the final agent reply even when an
        # un-phased agent message follows it; the trailing message folds
        # into the post-final intermediate block.
        rollout_lines = [
            _rollout_line("event_msg", {"type": "user_message", "message": "go"}),
            _rollout_line(
                "event_msg",
                {"type": "agent_message", "message": "the answer", "phase": "final_answer"},
            ),
            _rollout_line(
                "event_msg",
                {"type": "agent_message", "message": "post-answer note"},
            ),
        ]
        rollout_path = _make_rollout(self, rollout_lines)
        _patch_thread(self, mock_codex, _thread([], path=str(rollout_path)))

        response = _get_session(self.client)
        body = response.content.decode()

        self.assertEqual(response.status_code, 200)
        # The final_answer text renders outside the <details> block...
        self.assertLess(body.index("the answer"), body.index('<details class="intermediate"'))
        # ...while the trailing un-phased message goes inside it.
        self.assertContains(response, "post-answer note")
        self.assertContains(response, "1 thinking message")


class IntermediateCollapseTests(TestCase):
    """Non-final agent messages and tool calls fold into a <details> block so
    the page renders only the user/final-agent conversation by default."""

    @patch("hitch.main.views.common.Codex")
    def test_intermediate_thinking_and_tool_calls_collapse(
        self, mock_codex: MagicMock
    ) -> None:
        thread = _thread(
            [
                _turn(
                    [
                        _user_message("Help me out"),
                        _agent_message("Let me look at this."),
                        _command("./scripts/check.sh"),
                        _agent_message("Trying something else."),
                        _agent_message("Here is the answer."),
                    ]
                ),
            ]
        )
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)
        body = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Here is the answer.")
        self.assertContains(response, '<details class="intermediate">')
        self.assertContains(response, "2 thinking messages and 1 tool call")
        # Intermediate content stays in the document (collapsed, not removed).
        self.assertContains(response, "Let me look at this.")
        self.assertContains(response, "Trying something else.")
        self.assertContains(response, "./scripts/check.sh")
        # <details> appears before the final agent message in source order.
        self.assertLess(body.index("<details"), body.index("Here is the answer."))

    @patch("hitch.main.views.common.Codex")
    def test_no_collapse_when_nothing_intermediate(
        self, mock_codex: MagicMock
    ) -> None:
        thread = _thread([_turn([_user_message("Hi"), _agent_message("Hello.")])])
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, '<details class="intermediate"')

    @patch("hitch.main.views.common.Codex")
    def test_summary_pluralization(self, mock_codex: MagicMock) -> None:
        """The summary shows just the relevant kind(s) and pluralizes
        correctly: 1 tool call, 1 thinking message, etc."""
        # The "must_not_contain" pattern is anchored to ``</summary>`` so it
        # matches only the rendered summary text — the streaming script
        # also mentions ``tool call`` and ``thinking`` in its tool-label
        # map and would otherwise trigger a false positive.
        cases: list[tuple[list[SimpleNamespace], str, str]] = [
            (
                [_user_message("Run it"), _command("./scripts/run.sh"), _agent_message("Done.")],
                "<summary>1 tool call</summary>",
                "thinking message</summary>",
            ),
            (
                [_user_message("Think it through"), _agent_message("Step 1."), _agent_message("Final.")],
                "<summary>1 thinking message</summary>",
                "tool call</summary>",
            ),
        ]
        for items, expected, must_not_contain in cases:
            with self.subTest(expected=expected):
                _patch_thread(self, mock_codex, _thread([_turn(items)]))
                response = _get_session(self.client)
                self.assertEqual(response.status_code, 200)
                self.assertContains(response, expected, html=False)
                self.assertNotContains(response, must_not_contain)

    @patch("hitch.main.views.common.Codex")
    def test_phase_final_answer_wins_over_position(
        self, mock_codex: MagicMock
    ) -> None:
        """An explicit final_answer phase is the final reply even if later
        agent messages have no phase set."""
        thread = _thread(
            [
                _turn(
                    [
                        _user_message("Question"),
                        _agent_message("The answer is 42.", phase="final_answer"),
                        _agent_message("post-answer noise"),
                    ]
                ),
            ]
        )
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)
        body = response.content.decode()

        self.assertEqual(response.status_code, 200)
        agent_pos = body.index(">Agent<")
        answer_pos = body.index("The answer is 42.")
        noise_pos = body.index("post-answer noise")
        self.assertLess(agent_pos, answer_pos)
        self.assertLess(answer_pos, noise_pos)
        # The trailing message lives inside an intermediate block.
        self.assertContains(response, ">Agent (thinking)<")

    @patch("hitch.main.views.common.Codex")
    def test_commentary_phase_is_never_final(self, mock_codex: MagicMock) -> None:
        """All-commentary turns (in-progress) collapse entirely; no top-level
        Agent block."""
        thread = _thread(
            [
                _turn(
                    [
                        _user_message("Working on it"),
                        _agent_message("Step 1", phase="commentary"),
                        _agent_message("Step 2", phase="commentary"),
                    ]
                ),
            ]
        )
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<details class="intermediate">')
        self.assertContains(response, "2 thinking messages")
        self.assertNotContains(response, ">Agent<")
        self.assertContains(response, ">Agent (thinking)<")

    @patch("hitch.main.views.common.Codex")
    def test_phase_accepts_raw_string_shape(self, mock_codex: MagicMock) -> None:
        """Robustness: phase as a raw wire string (e.g. data deserialized
        without pydantic) is still recognized."""
        commentary = _root(
            SimpleNamespace(type="agentMessage", text="Thinking.", phase="commentary")
        )
        final = _root(
            SimpleNamespace(type="agentMessage", text="42.", phase="final_answer")
        )
        thread = _thread([_turn([_user_message("Q"), commentary, final])])
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)
        body = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertLess(body.index(">Agent<"), body.index("42."))
        self.assertContains(response, ">Agent (thinking)<")


class FinalAgentMarkdownTests(TestCase):
    """The turn's final agent reply is rendered as markdown when the body
    contains a high-confidence markdown construct; thinking entries and user
    messages always stay plain-text.
    """

    @patch("hitch.main.views.common.Codex")
    def test_final_agent_markdown_is_rendered(self, mock_codex: MagicMock) -> None:
        thread = _thread(
            [
                _turn(
                    [
                        _user_message("Plan it"),
                        _agent_message("# Plan\n\n- step one\n- step two"),
                    ]
                ),
            ]
        )
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "<h1>Plan</h1>", html=False)
        self.assertContains(response, "<li>step one</li>", html=False)
        self.assertContains(response, '<div class="body markdown">', html=False)

    @patch("hitch.main.views.common.Codex")
    def test_plain_final_agent_is_not_treated_as_markdown(
        self, mock_codex: MagicMock
    ) -> None:
        thread = _thread([_turn([_user_message("Hi"), _agent_message("Hello, friend.")])])
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Hello, friend.")
        self.assertNotContains(response, '<div class="body markdown">')

    @patch("hitch.main.views.common.Codex")
    def test_markdown_styling_is_only_for_final_agent(
        self, mock_codex: MagicMock
    ) -> None:
        """Mid-turn agent commentary and user messages keep their literal
        markdown characters even when the text looks like a heading: only
        the turn's final agent reply gets markdown rendering."""
        thread = _thread(
            [
                _turn(
                    [
                        _user_message("# my heading"),
                        _agent_message("# Thinking..."),
                        _agent_message("ok"),
                    ]
                ),
            ]
        )
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "# my heading")
        self.assertContains(response, "# Thinking...")
        self.assertNotContains(response, "<h1>my heading</h1>")
        self.assertNotContains(response, "<h1>Thinking")

    @patch("hitch.main.views.common.Codex")
    def test_agent_html_is_escaped(self, mock_codex: MagicMock) -> None:
        # Even when the body is detected as markdown, raw HTML in the
        # source is escaped, so an agent reply can't smuggle a <script>
        # tag into the page.
        thread = _thread(
            [
                _turn(
                    [
                        _user_message("Q"),
                        _agent_message("# Heading\n\n<script>alert(1)</script>"),
                    ]
                ),
            ]
        )
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)
        body = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("<script>alert(1)</script>", body)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", body)


class ToolCallDetailTests(TestCase):
    """Exercise every branch of ``tool_call_detail`` and ``tool_call_status``
    so the per-type description and status badge stay stable.
    """

    def test_detail_per_tool_type(self) -> None:
        cases: list[tuple[str, str, SimpleNamespace, str]] = [
            ("commandExecution", "with command", SimpleNamespace(command="ls -la"), "ls -la"),
            ("commandExecution", "missing command", SimpleNamespace(command=None), ""),
            (
                "mcpToolCall",
                "server and tool",
                SimpleNamespace(server="github", tool="create_pr"),
                "github / create_pr",
            ),
            (
                "dynamicToolCall",
                "with namespace",
                SimpleNamespace(namespace="codex", tool="apply_patch"),
                "codex::apply_patch",
            ),
            (
                "dynamicToolCall",
                "without namespace",
                SimpleNamespace(namespace=None, tool="apply_patch"),
                "apply_patch",
            ),
            ("fileChange", "empty changes", SimpleNamespace(changes=[]), ""),
            (
                "fileChange",
                "single change",
                SimpleNamespace(changes=[SimpleNamespace(path="a.py")]),
                "a.py",
            ),
            (
                "fileChange",
                "multiple changes",
                SimpleNamespace(
                    changes=[SimpleNamespace(path="a.py"), SimpleNamespace(path="b.py")]
                ),
                "a.py (+1 more)",
            ),
            ("webSearch", "query", SimpleNamespace(query="how to django"), "how to django"),
            (
                "plan",
                "first line only",
                SimpleNamespace(text="Step 1\nStep 2\nStep 3"),
                "Step 1",
            ),
            ("imageView", "path", SimpleNamespace(path="/tmp/x.png"), "/tmp/x.png"),
            (
                "imageGeneration",
                "prefers revised_prompt",
                SimpleNamespace(revised_prompt="a cat", saved_path="/tmp/y.png"),
                "a cat",
            ),
            (
                "imageGeneration",
                "falls back to saved_path",
                SimpleNamespace(revised_prompt=None, saved_path="/tmp/y.png"),
                "/tmp/y.png",
            ),
            ("somethingNew", "unknown type", SimpleNamespace(), ""),
        ]
        for tool_type, label, item, expected in cases:
            with self.subTest(tool_type=tool_type, case=label):
                self.assertEqual(tool_call_detail(item, tool_type), expected)

    def test_collab_agent_detail(self) -> None:
        # Two cases share an assertion shape but differ in what they expose.
        with_receiver = SimpleNamespace(
            tool=SimpleNamespace(value="spawn"), receiver_thread_ids=["child-thread"]
        )
        without_receiver = SimpleNamespace(
            tool=SimpleNamespace(value="spawn"), receiver_thread_ids=[]
        )
        result = tool_call_detail(with_receiver, "collabAgentToolCall")
        self.assertIn("child-thread", result)
        self.assertIn("spawn", result)
        self.assertEqual(
            tool_call_detail(without_receiver, "collabAgentToolCall"), "spawn"
        )

    def test_status_badge(self) -> None:
        # ``completed`` and missing status both render no badge; ``failed``
        # surfaces verbatim so the UI can highlight it.
        self.assertIsNone(tool_call_status(SimpleNamespace(status=SimpleNamespace(value="completed"))))
        self.assertIsNone(tool_call_status(SimpleNamespace()))
        self.assertEqual(
            tool_call_status(SimpleNamespace(status=SimpleNamespace(value="failed"))),
            "failed",
        )


def _make_codex_instance(**kwargs: object) -> CodexInstance:
    """Helper used by the active-worker session view tests."""
    defaults: dict[str, object] = {
        "pid": 0,
        "thread_id": "thread-1",
        "cwd": "/repo",
        "prompt": "do work",
        "events_path": "/tmp/some.jsonl",
        "status": CodexInstance.STATUS_RUNNING,
    }
    defaults.update(kwargs)
    return CodexInstance.objects.create(**defaults)


class SessionViewActiveWorkerTests(TestCase):
    """How the session detail view surfaces an in-progress turn: the
    streaming UI guard, the pending user bubble, the in-progress turn
    trim, and the dead-worker reconciliation sweep.
    """

    @override
    def setUp(self) -> None:
        super().setUp()
        patcher = patch(
            "hitch.main.runtime.codex_pool.worker_is_alive",
            side_effect=_worker_is_live_for_test,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    @patch("hitch.main.views.common.Codex")
    def test_inactive_thread_renders_status_pill_idle_with_no_live_root(
        self, mock_codex: MagicMock
    ) -> None:
        # Without an active worker the live-streaming insertion anchor
        # (``data-live-root``) must not be in the DOM, so streamed item
        # events have nowhere to land — but the live-status pill still
        # renders (in its hidden idle state) so the JS heartbeat handler
        # can surface reconnecting/fatal connection states if frames stop
        # arriving.
        # The pill-text checks use the ``>...</span`` anchor so they
        # don't false-match the same string literal inside the JS map.
        _patch_thread(self, mock_codex, _thread([]))

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "data-live-root></div>")
        self.assertContains(response, "data-live-status")
        self.assertContains(response, 'data-state="idle"')
        self.assertContains(response, ">Connected</span>")
        self.assertNotContains(response, ">Codex is working")
        self.assertNotContains(response, 'class="jump-latest" data-jump-latest')

    @patch("hitch.main.views.common.Codex")
    def test_inactive_thread_surfaces_latest_user_turn_failure(
        self, mock_codex: MagicMock
    ) -> None:
        _patch_thread(self, mock_codex, _thread([]))
        ended_at = timezone.now().replace(microsecond=0)
        instance = _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_FAILED,
            error="This content was flagged for possible cybersecurity risk.",
            purpose=CodexInstance.PURPOSE_USER,
        )
        CodexInstance.objects.filter(pk=instance.pk).update(ended_at=ended_at)

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertContains(response, "data-turn-failure")
        self.assertContains(response, "Agent turn failed")
        self.assertContains(
            response,
            "This content was flagged for possible cybersecurity risk.",
        )
        self.assertContains(response, f'data-ts="{int(ended_at.timestamp())}"')

    @patch("hitch.main.views.common.Codex")
    def test_completed_user_turn_supersedes_prior_failure(
        self, mock_codex: MagicMock
    ) -> None:
        _patch_thread(self, mock_codex, _thread([]))
        now = timezone.now()
        failed = _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_FAILED,
            error="old failure",
            purpose=CodexInstance.PURPOSE_USER,
        )
        completed = _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_USER,
        )
        CodexInstance.objects.filter(pk=failed.pk).update(
            started_at=now - timedelta(minutes=2),
            ended_at=now - timedelta(minutes=1),
        )
        CodexInstance.objects.filter(pk=completed.pk).update(
            started_at=now - timedelta(seconds=30),
            ended_at=now,
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertNotContains(response, "data-turn-failure")
        self.assertNotContains(response, "old failure")

    def test_new_active_user_turn_supersedes_prior_failure(self) -> None:
        now = timezone.now()
        failed = _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_FAILED,
            error="old failure",
            purpose=CodexInstance.PURPOSE_USER,
        )
        active = _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_USER,
        )
        CodexInstance.objects.filter(pk=failed.pk).update(
            started_at=now - timedelta(minutes=2),
            ended_at=now - timedelta(minutes=1),
        )
        CodexInstance.objects.filter(pk=active.pk).update(started_at=now)

        self.assertIsNone(session_entry_display._latest_user_turn_failure("thread-1"))

    @patch("hitch.main.views.common.Codex")
    def test_later_ending_overlapping_user_turn_failure_is_visible(
        self, mock_codex: MagicMock
    ) -> None:
        _patch_thread(self, mock_codex, _thread([]))
        now = timezone.now().replace(microsecond=0)
        failed = _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_FAILED,
            error="older turn failed last",
            purpose=CodexInstance.PURPOSE_USER,
        )
        completed = _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_USER,
        )
        CodexInstance.objects.filter(pk=failed.pk).update(
            started_at=now - timedelta(minutes=2),
            ended_at=now,
        )
        CodexInstance.objects.filter(pk=completed.pk).update(
            started_at=now - timedelta(minutes=1),
            ended_at=now - timedelta(seconds=30),
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertContains(response, "data-turn-failure")
        self.assertContains(response, "older turn failed last")
        self.assertContains(response, f'data-ts="{int(now.timestamp())}"')

    @patch("hitch.main.views.common.Codex")
    def test_connection_indicator_retries_before_showing_fatal_loss(
        self, mock_codex: MagicMock
    ) -> None:
        _patch_thread(self, mock_codex, _thread([]))

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertContains(response, 'reconnecting: "Reconnecting')
        self.assertContains(response, "const RECONNECT_FATAL_MS")
        self.assertContains(response, '"disconnected" : "reconnecting"')
        self.assertContains(response, 'indicator.addEventListener("click"')
        self.assertContains(response, "function retryNow()")
        self.assertContains(response, "function clearRetryTimer()")
        self.assertContains(response, "function handleReconnect(event)")
        self.assertContains(response, 'source.addEventListener("reconnect", handleReconnect)')
        self.assertContains(response, "new EventSource(streamUrl)")
        self.assertContains(response, "const seenEvents = new Set()")
        self.assertContains(response, "if (seenEvents.has(key)) return;")

    @patch("hitch.main.views.common.build_worktree_diff")
    @patch("hitch.main.views.common.Codex")
    def test_active_worker_renders_status_without_volatile_diff_preview(
        self, mock_codex: MagicMock, mock_diff: MagicMock
    ) -> None:
        # With an active worker the pill renders in its "working" state
        # so the user sees the pulsing-green indicator immediately, even
        # before the first heartbeat lands.
        mock_diff.return_value = _diff_view()
        _patch_thread(self, mock_codex, _thread([]))
        _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_RUNNING,
            prompt="warming up",
            pid=_LIVE_PID,
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertContains(response, "data-live-status")
        self.assertContains(response, 'data-state="working"')
        self.assertContains(response, ">Codex is working")
        self.assertContains(response, 'class="jump-latest" data-jump-latest')
        self.assertContains(response, 'aria-label="Jump to latest message"')
        self.assertContains(response, "data-diff-open disabled")
        self.assertNotContains(response, '<dialog class="diff-modal"', html=False)
        self.assertNotContains(response, 'class="diff-fab"')
        self.assertNotContains(response, 'aria-label="Settings for the next message"')
        mock_diff.assert_not_called()

    @patch("hitch.main.views.common.build_worktree_diff")
    @patch("hitch.main.views.common.Codex")
    def test_active_qa_worker_renders_token_progress(
        self, mock_codex: MagicMock, mock_diff: MagicMock
    ) -> None:
        mock_diff.return_value = _diff_view()
        _patch_thread(self, mock_codex, _thread([]))
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as fh:
            fh.write(
                json.dumps(
                    {
                        "method": codex_events.GOAL_UPDATED_METHOD,
                        "payload": {
                            "threadId": "thread-1",
                            "goal": {
                                "objective": "Apply QA feedback",
                                "tokens_used": 1234,
                            },
                        },
                    }
                )
                + "\n"
            )
            events_path = fh.name
        self.addCleanup(Path(events_path).unlink, missing_ok=True)
        _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_SYSTEM_FEEDBACK,
            display_author=system_agents.QA_DISPLAY_AUTHOR,
            events_path=events_path,
            pid=_LIVE_PID,
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertContains(response, "QA agent working...1.2K tokens")
        self.assertContains(response, 'data-working-text="QA agent working...1.2K tokens"')

    @patch("hitch.main.views.common.build_worktree_diff")
    @patch("hitch.main.views.common.Codex")
    def test_active_qa_feedback_message_renders_timestamp(
        self, mock_codex: MagicMock, mock_diff: MagicMock
    ) -> None:
        mock_diff.return_value = _diff_view()
        _patch_thread(self, mock_codex, _thread([]))
        instance = _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_RUNNING,
            prompt="Feedback from Hitch QA agent:\n\nFix this.",
            purpose=CodexInstance.PURPOSE_SYSTEM_FEEDBACK,
            display_author=system_agents.QA_DISPLAY_AUTHOR,
            pid=_LIVE_PID,
        )
        CodexInstance.objects.filter(pk=instance.pk).update(
            started_at=datetime.fromtimestamp(1700000456, UTC)
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertContains(response, '<span class="role">QA agent</span>')
        self.assertContains(
            response,
            '<time data-ts="1700000456">1700000456</time>',
            count=1,
        )

    @patch("hitch.main.views.common.build_worktree_diff")
    @patch("hitch.main.views.common.Codex")
    def test_active_worker_renders_latest_goal_near_status_pill(
        self, mock_codex: MagicMock, mock_diff: MagicMock
    ) -> None:
        mock_diff.return_value = _diff_view()
        _patch_thread(self, mock_codex, _thread([]))
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as fh:
            fh.write(
                json.dumps(
                    {
                        "method": codex_events.GOAL_UPDATED_METHOD,
                        "payload": {
                            "threadId": "thread-1",
                            "goal": {"objective": "Keep the live goal visible"},
                        },
                    }
                )
                + "\n"
            )
            events_path = fh.name
        self.addCleanup(Path(events_path).unlink, missing_ok=True)
        _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_RUNNING,
            prompt="warming up",
            pid=_LIVE_PID,
            events_path=events_path,
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertContains(response, 'data-live-work data-state="working"')
        self.assertContains(response, 'data-live-goal')
        self.assertContains(response, ">Goal<")
        self.assertContains(response, "Keep the live goal visible")

    @patch("hitch.main.views.common.build_worktree_diff")
    @patch("hitch.main.views.common.Codex")
    def test_active_worker_renders_latest_task_plan_after_refresh(
        self, mock_codex: MagicMock, mock_diff: MagicMock
    ) -> None:
        mock_diff.return_value = _diff_view()
        _patch_thread(self, mock_codex, _thread([]))
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as fh:
            fh.write(
                json.dumps(
                    {
                        "method": codex_events.TASK_PLAN_UPDATED_METHOD,
                        "payload": {
                            "threadId": "thread-1",
                            "explanation": "Keep progress visible after refresh",
                            "plan": [
                                {"step": "Inspect current live UI", "status": "completed"},
                                {"step": "Render saved task plan", "status": "in_progress"},
                                {"step": "Run the refresh tests", "status": "pending"},
                            ],
                        },
                        "recordedAt": 20,
                        "eventSeq": 2,
                    }
                )
                + "\n"
            )
            events_path = fh.name
        self.addCleanup(Path(events_path).unlink, missing_ok=True)
        _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_RUNNING,
            prompt="warming up",
            pid=_LIVE_PID,
            events_path=events_path,
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))
        body = response.content.decode()

        self.assertContains(response, 'data-task-plan')
        self.assertContains(response, 'data-stream-url="')
        self.assertContains(response, 'class="has-task-plan"')
        self.assertContains(response, 'data-recorded-at="20"')
        self.assertContains(response, 'data-event-seq="2"')
        self.assertContains(response, 'data-fallback-order="1"')
        self.assertNotIn('aria-label="Current task plan" hidden', body)
        self.assertContains(response, "Keep progress visible after refresh")
        self.assertContains(response, "Render saved task plan")
        self.assertContains(response, 'class="plan-step inProgress"')
        self.assertContains(
            response,
            '<span class="task-plan-current" data-task-plan-current>Render saved task plan</span>',
            html=False,
        )

    @patch("hitch.main.views.common.build_worktree_diff")
    @patch("hitch.main.views.common.Codex")
    def test_finished_worker_rebuilds_task_plan_on_page_load(
        self, mock_codex: MagicMock, mock_diff: MagicMock
    ) -> None:
        # The task-plan widget must rebuild from the thread's persisted worker
        # logs after a reload, when no worker is running, just as the goal
        # objective does.
        mock_diff.return_value = _diff_view()
        _patch_thread(self, mock_codex, _thread([]))
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as fh:
            fh.write(
                json.dumps(
                    {
                        "method": codex_events.TASK_PLAN_UPDATED_METHOD,
                        "payload": {
                            "threadId": "thread-1",
                            "explanation": "Persist plan past the worker",
                            "plan": [
                                {"step": "Inspect current live UI", "status": "completed"},
                                {"step": "Render saved task plan", "status": "in_progress"},
                            ],
                        },
                        "recordedAt": 30,
                        "eventSeq": 3,
                    }
                )
                + "\n"
            )
            events_path = fh.name
        self.addCleanup(Path(events_path).unlink, missing_ok=True)
        _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_COMPLETED,
            prompt="finished work",
            pid=0,
            events_path=events_path,
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))
        body = response.content.decode()

        # No active worker: the page renders without a live stream root.
        self.assertNotContains(response, "data-live-root></div>")
        self.assertContains(response, 'class="has-task-plan"')
        self.assertContains(response, "Render saved task plan")
        self.assertContains(response, 'class="plan-step inProgress"')
        self.assertNotIn('aria-label="Current task plan" hidden', body)
        self.assertContains(
            response,
            '<span class="task-plan-current" data-task-plan-current>Render saved task plan</span>',
            html=False,
        )

    @patch("hitch.main.views.common.build_worktree_diff")
    @patch("hitch.main.views.common.Codex")
    def test_planless_latest_turn_does_not_resurrect_prior_task_plan(
        self, mock_codex: MagicMock, mock_diff: MagicMock
    ) -> None:
        # A later turn that finished without emitting a plan must leave the
        # widget empty on reload rather than resurrecting the earlier turn's
        # plan; the lookup scopes to the most recently started worker.
        mock_diff.return_value = _diff_view()
        _patch_thread(self, mock_codex, _thread([]))
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as fh:
            fh.write(
                json.dumps(
                    {
                        "method": codex_events.TASK_PLAN_UPDATED_METHOD,
                        "payload": {
                            "threadId": "thread-1",
                            "plan": [
                                {"step": "Stale prior-turn task", "status": "in_progress"},
                            ],
                        },
                        "recordedAt": 30,
                        "eventSeq": 3,
                    }
                )
                + "\n"
            )
            planned_path = fh.name
        self.addCleanup(Path(planned_path).unlink, missing_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=False
        ) as planless_fh:
            planless_path = planless_fh.name
        self.addCleanup(Path(planless_path).unlink, missing_ok=True)
        planned = _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_COMPLETED,
            prompt="earlier turn",
            pid=0,
            events_path=planned_path,
        )
        planless = _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_COMPLETED,
            prompt="later planless turn",
            pid=0,
            events_path=planless_path,
        )
        # ``started_at`` is auto-set on create; pin an explicit ordering so the
        # planless worker is unambiguously the most recent.
        CodexInstance.objects.filter(pk=planned.pk).update(
            started_at=timezone.now() - timedelta(minutes=5)
        )
        CodexInstance.objects.filter(pk=planless.pk).update(started_at=timezone.now())

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))
        body = response.content.decode()

        self.assertNotIn("Stale prior-turn task", body)
        self.assertNotContains(response, 'class="has-task-plan"')
        self.assertRegex(
            body,
            r'<aside class="task-plan"[\s\S]*?aria-label="Current task plan"\s+hidden>',
        )

    @patch("hitch.main.views.common.build_worktree_diff")
    @patch("hitch.main.views.common.Codex")
    def test_active_worker_does_not_inherit_prior_worker_task_plan(
        self, mock_codex: MagicMock, mock_diff: MagicMock
    ) -> None:
        # A new turn whose worker has not emitted its own plan must not seed the
        # widget from an earlier completed worker's plan; the thread-wide scan
        # only applies when no worker is running.
        mock_diff.return_value = _diff_view()
        _patch_thread(self, mock_codex, _thread([]))
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as fh:
            fh.write(
                json.dumps(
                    {
                        "method": codex_events.TASK_PLAN_UPDATED_METHOD,
                        "payload": {
                            "threadId": "thread-1",
                            "plan": [
                                {"step": "Stale prior-turn task", "status": "in_progress"},
                            ],
                        },
                        "recordedAt": 30,
                        "eventSeq": 3,
                    }
                )
                + "\n"
            )
            finished_path = fh.name
        self.addCleanup(Path(finished_path).unlink, missing_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=False
        ) as active_fh:
            active_path = active_fh.name
        self.addCleanup(Path(active_path).unlink, missing_ok=True)
        _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_COMPLETED,
            prompt="finished work",
            pid=0,
            events_path=finished_path,
        )
        _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_RUNNING,
            prompt="new turn",
            pid=_LIVE_PID,
            events_path=active_path,
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))
        body = response.content.decode()

        self.assertContains(response, 'data-stream-url="')
        self.assertNotIn("Stale prior-turn task", body)
        self.assertNotContains(response, 'class="has-task-plan"')
        self.assertRegex(
            body,
            r'<aside class="task-plan"[\s\S]*?aria-label="Current task plan"\s+hidden>',
        )

    @patch("hitch.main.views.common.build_worktree_diff")
    @patch("hitch.main.views.common.Codex")
    def test_active_worker_preserves_cleared_task_plan_order_after_refresh(
        self, mock_codex: MagicMock, mock_diff: MagicMock
    ) -> None:
        mock_diff.return_value = _diff_view()
        _patch_thread(self, mock_codex, _thread([]))
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as fh:
            fh.write(
                json.dumps(
                    {
                        "method": codex_events.TASK_PLAN_UPDATED_METHOD,
                        "payload": {
                            "threadId": "thread-1",
                            "plan": [
                                {"step": "Stale task", "status": "in_progress"},
                            ],
                        },
                        "recordedAt": 10,
                        "eventSeq": 1,
                    }
                )
                + "\n"
            )
            fh.write(
                json.dumps(
                    {
                        "method": codex_events.TASK_PLAN_UPDATED_METHOD,
                        "payload": {"threadId": "thread-1", "plan": []},
                        "recordedAt": 20,
                        "eventSeq": 2,
                    }
                )
                + "\n"
            )
            events_path = fh.name
        self.addCleanup(Path(events_path).unlink, missing_ok=True)
        _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_RUNNING,
            prompt="warming up",
            pid=_LIVE_PID,
            events_path=events_path,
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))
        body = response.content.decode()

        self.assertContains(response, 'data-task-plan')
        self.assertContains(response, 'data-stream-url="')
        self.assertContains(response, 'class="">')
        self.assertContains(response, 'data-recorded-at="20"')
        self.assertContains(response, 'data-event-seq="2"')
        self.assertContains(response, 'data-fallback-order="2"')
        self.assertRegex(
            body,
            r'<aside class="task-plan"[\s\S]*?data-recorded-at="20"'
            r'[\s\S]*?data-fallback-order="2"'
            r'[\s\S]*?aria-label="Current task plan"\s+hidden>',
        )
        self.assertNotIn("Stale task", body)
        self.assertContains(
            response,
            '<span class="task-plan-current" data-task-plan-current></span>',
            html=False,
        )

    @patch("hitch.main.views.common.build_worktree_diff")
    @patch("hitch.main.views.common.Codex")
    def test_active_worker_preserves_fallback_task_plan_order_after_refresh(
        self, mock_codex: MagicMock, mock_diff: MagicMock
    ) -> None:
        mock_diff.return_value = _diff_view()
        _patch_thread(self, mock_codex, _thread([]))
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as fh:
            fh.write(
                json.dumps(
                    {
                        "method": codex_events.TASK_PLAN_UPDATED_METHOD,
                        "payload": {
                            "threadId": "thread-1",
                            "plan": [
                                {"step": "Fallback stale task", "status": "in_progress"},
                            ],
                        },
                    }
                )
                + "\n"
            )
            fh.write(
                json.dumps(
                    {
                        "method": codex_events.TASK_PLAN_UPDATED_METHOD,
                        "payload": {
                            "threadId": "thread-1",
                            "plan": [
                                {"step": "Fallback latest task", "status": "pending"},
                            ],
                        },
                    }
                )
                + "\n"
            )
            events_path = fh.name
        self.addCleanup(Path(events_path).unlink, missing_ok=True)
        _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_RUNNING,
            prompt="warming up",
            pid=_LIVE_PID,
            events_path=events_path,
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))
        body = response.content.decode()

        self.assertContains(response, 'data-recorded-at="0"')
        self.assertContains(response, 'data-event-seq="0"')
        self.assertContains(response, 'data-fallback-order="2"')
        self.assertContains(response, "Fallback latest task")
        self.assertNotIn("Fallback stale task", body)

    @patch("hitch.main.views.common.Codex")
    def test_active_worker_renders_live_section_and_stream_url(
        self, mock_codex: MagicMock
    ) -> None:
        _patch_thread(self, mock_codex, _thread([]))
        _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_RUNNING,
            prompt="please refactor",
            pid=_LIVE_PID,
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "data-live-root")
        instance = CodexInstance.objects.get(thread_id="thread-1")
        self.assertContains(
            response, reverse("session_stream", kwargs={"session_id": "thread-1"})
        )
        # The pending user message is surfaced as a regular user bubble so
        # the user sees their own prompt immediately, even before the
        # worker's user_message event reaches the rollout.
        self.assertContains(response, "please refactor")
        self.assertContains(response, "data-pending-user>")
        self.assertContains(
            response,
            '<div class="message user pending" data-copyable-message>',
            html=False,
        )
        # Stop uses a separate non-multipart form so selected image
        # attachments cannot delay or block cancellation.
        # The regular composer submit carries the same active instance in a
        # hidden field so it steers this running turn instead of spawning an
        # overlapping follow-up worker.
        # The stop form carries the *specific* worker the page is showing —
        # clicking Stop on a stale tab must not accidentally abort a newer
        # overlapping worker the user can't see.
        stop_url = reverse("stop_session", kwargs={"session_id": "thread-1"})
        self.assertContains(response, f'name="active_instance" value="{instance.id}"')
        self.assertContains(response, ">Steer</button>")
        self.assertContains(response, f'action="{stop_url}"')
        self.assertContains(response, 'form="stop-session-form"')
        self.assertContains(response, f'name="instance" value="{instance.id}"')
        self.assertContains(response, 'aria-label="Stop the running turn"')
        self.assertContains(response, "!commandPrompt && !hasImages()")
        self.assertContains(response, "let stopSubmitting = false")
        self.assertContains(response, 'document.querySelector("[data-stop-form]")')
        self.assertContains(response, 'document.querySelectorAll("[data-composer-stop]")')
        self.assertContains(response, "if (inner.text) parts.push(inner.text)")

    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_turn")
    @patch("hitch.main.workflows.system_agents.codex_pool.interrupt_instance")
    @patch("hitch.main.views.common.Codex")
    def test_rendered_stop_cancels_workflow_and_queued_steering(
        self,
        mock_codex: MagicMock,
        mock_interrupt: MagicMock,
        mock_spawn: MagicMock,
    ) -> None:
        _patch_thread(self, mock_codex, _thread([]))
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="thread-1",
            cwd="/tmp/demo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_PR_PROMPT_RUNNING,
            state={"next_user_message_index": 1},
        )
        instance = _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_USER,
            workflow_id=workflow.pk,
            prompt="first request",
            developer_instructions="PR workflow continuation requirements",
            user_message_index=0,
            pid=_LIVE_PID,
        )
        WorkflowSteeringMessage.objects.create(
            workflow=workflow, prompt="second request"
        )
        mock_interrupt.return_value = instance

        rendered = self.client.get(
            reverse("session", kwargs={"session_id": "thread-1"})
        )
        self.assertContains(
            rendered, f'name="instance" value="{instance.pk}"'
        )
        self.assertContains(rendered, "first request")
        self.assertContains(rendered, 'aria-label="Stop the QA workflow"')
        self.assertNotContains(rendered, 'aria-label="Stop the running turn"')
        self.assertNotContains(
            rendered, "PR workflow continuation requirements"
        )

        response = self.client.post(
            reverse("stop_session", kwargs={"session_id": "thread-1"}),
            data={"instance": str(instance.pk)},
        )

        self.assertEqual(response.status_code, 302)
        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
        self.assertFalse(workflow.steering_messages.exists())
        mock_interrupt.assert_called_once_with(
            instance.pk, expected_thread_id="thread-1"
        )

        instance.status = CodexInstance.STATUS_FAILED
        instance.error = "interrupted by user"
        instance.save(update_fields=["status", "error"])
        system_agents.on_codex_instance_finished(instance)

        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
        self.assertEqual(mock_spawn.call_count, 1)
        self.assertNotIn(
            "second request", mock_spawn.call_args.kwargs["prompt"]
        )

    @patch("hitch.main.views.common.Codex")
    def test_inactive_thread_omits_stop_button(
        self, mock_codex: MagicMock
    ) -> None:
        # Stop is only meaningful while a turn is in progress; otherwise it
        # would just be dead chrome cluttering the composer.
        _patch_thread(self, mock_codex, _thread([]))

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(
            response, reverse("stop_session", kwargs={"session_id": "thread-1"})
        )

    @patch("hitch.main.views.common.Codex")
    def test_hidden_system_workflow_renders_busy_state(
        self, mock_codex: MagicMock
    ) -> None:
        _patch_thread(self, mock_codex, _thread([]))
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="thread-1",
            cwd="/tmp/demo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_QA_RUNNING,
        )
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as fh:
            fh.write(
                json.dumps(
                    {
                        "method": codex_events.GOAL_UPDATED_METHOD,
                        "payload": {
                            "threadId": "hidden-thread",
                            "goal": {
                                "objective": "Review the diff",
                                "tokensUsed": 4200,
                            },
                        },
                    }
                )
                + "\n"
            )
            events_path = fh.name
        self.addCleanup(Path(events_path).unlink, missing_ok=True)
        instance = _make_codex_instance(
            thread_id="hidden-thread",
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            display_author=system_agents.QA_DISPLAY_AUTHOR,
            events_path=events_path,
            pid=_LIVE_PID,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            thread_id="hidden-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )
        WorkflowSteeringMessage.objects.create(
            workflow=workflow,
            prompt="also update the release notes",
        )
        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))
        stream_path = reverse("session_stream", kwargs={"session_id": "thread-1"})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "QA agent working...4.2K tokens")
        self.assertContains(response, 'data-workflow-locked="false"')
        self.assertContains(response, "Add instructions")
        self.assertContains(response, ">Steer</button>")
        self.assertContains(response, "User · Queued")
        self.assertContains(response, "also update the release notes")
        self.assertContains(response, "data-queued-workflow-user")
        self.assertContains(response, 'aria-label="Stop the QA workflow"')
        self.assertContains(
            response,
            f'data-stream-url="{stream_path}?baseline=&amp;active=&amp;workflow={workflow.pk}&amp;steering=0&amp;demo="',
        )

        workflow.steering_messages.all().delete()
        workflow.step = system_agents.STEP_USER_STEERING_RUNNING
        workflow.state = {
            "next_user_message_index": 1,
            "user_steering_prompt": "also update the release notes",
            "user_steering_resume_step": system_agents.STEP_QA_RUNNING,
            "user_steering_message_index": 1,
        }
        workflow.save(update_fields=["step", "state", "updated_at"])

        claimed_response = self.client.get(
            reverse("session", kwargs={"session_id": "thread-1"})
        )

        self.assertContains(claimed_response, "User · Queued")
        self.assertContains(claimed_response, "also update the release notes")
        self.assertContains(claimed_response, "data-queued-workflow-user")

    @patch("hitch.main.views.common.Codex")
    def test_active_prompt_renders_before_later_queued_steering(
        self, mock_codex: MagicMock
    ) -> None:
        _patch_thread(self, mock_codex, _thread([]))
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="thread-1",
            cwd="/tmp/demo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_USER_STEERING_RUNNING,
            state={
                "next_user_message_index": 1,
                "user_steering_prompt": "first request",
                "user_steering_resume_step": system_agents.STEP_QA_RUNNING,
                "user_steering_message_index": 0,
            },
        )
        _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_USER,
            workflow_id=workflow.pk,
            prompt="first request",
            user_message_index=0,
            pid=_LIVE_PID,
        )
        WorkflowSteeringMessage.objects.create(
            workflow=workflow,
            prompt="second request",
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))
        body = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "first request")
        self.assertContains(response, "second request")
        self.assertLess(body.index("first request"), body.index("second request"))

        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            self.skipTest(f"playwright unavailable: {exc}")

        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
            except PlaywrightError as exc:
                self.skipTest(f"playwright browser unavailable: {exc}")
            try:
                page = browser.new_page()
                page.evaluate(
                    """
                    () => {
                        class MockEventSource {
                            constructor(url) {
                                this.url = url;
                                this.listeners = {};
                                window.__eventSource = this;
                            }
                            addEventListener(type, callback) {
                                this.listeners[type] = callback;
                            }
                            close() {}
                            emit(type, data) {
                                this.listeners[type]({ data: JSON.stringify(data) });
                            }
                        }
                        window.EventSource = MockEventSource;
                    }
                    """
                )
                page.set_content(body, wait_until="load")
                page.wait_for_function("window.__eventSource !== undefined")
                user_messages = "[data-thread] > .entry .message.user .body"
                self.assertEqual(
                    page.locator(user_messages).all_text_contents(),
                    ["first request", "second request"],
                )

                page.evaluate(
                    """
                    () => window.__eventSource.emit("message", {
                        method: "item/started",
                        recordedAt: 1700000123000000,
                        eventSeq: 1,
                        payload: {
                            item: {
                                id: "streamed-user",
                                type: "userMessage",
                                content: [{ type: "text", text: "first request" }],
                            },
                        },
                    })
                    """
                )
                page.wait_for_selector('[data-item-id="streamed-user"]')

                self.assertEqual(
                    page.locator(user_messages).all_text_contents(),
                    ["first request", "second request"],
                )
            finally:
                browser.close()

    @patch("hitch.main.views.common.Codex")
    def test_workflow_system_feedback_worker_accepts_steering(
        self, mock_codex: MagicMock
    ) -> None:
        _patch_thread(self, mock_codex, _thread([]))
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="thread-1",
            cwd="/tmp/demo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_FEEDBACK_RUNNING,
        )
        _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_SYSTEM_FEEDBACK,
            workflow_id=workflow.pk,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            display_author=system_agents.QA_DISPLAY_AUTHOR,
            pid=_LIVE_PID,
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'data-workflow-locked="false"')
        self.assertContains(response, "Add instructions")
        self.assertContains(response, ">Steer</button>")

    @patch("hitch.main.views.common.Codex")
    def test_completed_qa_approval_is_shown_in_transcript(
        self, mock_codex: MagicMock
    ) -> None:
        _patch_thread(
            self,
            mock_codex,
            _thread(
                [
                    _turn([_user_message("Change it"), _agent_message("Done")]),
                    _turn(
                        [
                            _user_message(system_agents.PR_SLASH_PROMPT),
                            _agent_message("Opened PR"),
                        ],
                        started_at=1700000010,
                    ),
                ]
            ),
        )
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="thread-1",
            cwd="/tmp/demo",
            status=SystemWorkflow.STATUS_COMPLETED,
            step=system_agents.STEP_PR_PROMPT_SPAWNED,
            state={
                "next_user_message_index": 2,
                "last_feedback": "No qualifying findings.",
            },
        )
        instance = _make_codex_instance(
            thread_id="hidden-thread",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            display_author=system_agents.QA_DISPLAY_AUTHOR,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            thread_id="hidden-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_COMPLETED,
            output={"feedback": "No qualifying findings.", "lgtm": True},
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))
        body = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<span class="role">QA agent</span>')
        self.assertContains(response, "QA agent approved the diff.")
        self.assertContains(response, "No qualifying findings.")
        self.assertLess(
            body.index("QA agent approved the diff."),
            body.index(system_agents.PR_SLASH_PROMPT),
        )

    @patch("hitch.main.views.common.Codex")
    def test_monitored_pr_qa_approval_keeps_original_prompt_order(
        self, mock_codex: MagicMock
    ) -> None:
        _patch_thread(
            self,
            mock_codex,
            _thread(
                [
                    _turn([_user_message("Change it"), _agent_message("Done")]),
                    _turn(
                        [
                            _user_message(system_agents.PR_SLASH_PROMPT),
                            _agent_message("Opened PR"),
                        ],
                        started_at=1700000010,
                    ),
                    _turn(
                        [
                            _user_message("Address monitor feedback"),
                            _agent_message("Pushed fix"),
                        ],
                        started_at=1700000020,
                    ),
                ]
            ),
        )
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="thread-1",
            cwd="/tmp/demo",
            status=SystemWorkflow.STATUS_COMPLETED,
            step=system_agents.STEP_PR_READY,
            state={
                "next_user_message_index": 4,
                system_agents.QA_APPROVAL_INSERT_INDEX_STATE_KEY: 1,
                "last_feedback": "No qualifying findings.",
            },
        )
        instance = _make_codex_instance(
            thread_id="hidden-thread",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            display_author=system_agents.QA_DISPLAY_AUTHOR,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            thread_id="hidden-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_COMPLETED,
            output={"feedback": "No qualifying findings.", "lgtm": True},
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))
        body = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "QA agent approved the diff.")
        self.assertLess(
            body.index("QA agent approved the diff."),
            body.index(system_agents.PR_SLASH_PROMPT),
        )
        self.assertLess(
            body.index("QA agent approved the diff."),
            body.index("Address monitor feedback"),
        )

    @patch("hitch.main.views.common.Codex")
    def test_completed_qa_only_approval_is_appended_to_transcript(
        self, mock_codex: MagicMock
    ) -> None:
        _patch_thread(
            self,
            mock_codex,
            _thread([_turn([_user_message("Change it"), _agent_message("Done")])]),
        )
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="thread-1",
            cwd="/tmp/demo",
            status=SystemWorkflow.STATUS_COMPLETED,
            step=system_agents.STEP_QA_APPROVED,
            state={
                "next_user_message_index": 1,
                "last_feedback": "No qualifying findings.",
            },
        )
        instance = _make_codex_instance(
            thread_id="hidden-thread",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            display_author=system_agents.QA_DISPLAY_AUTHOR,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            thread_id="hidden-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_COMPLETED,
            output={"feedback": "No qualifying findings.", "lgtm": True},
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))
        body = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<span class="role">QA agent</span>')
        self.assertContains(response, "QA agent approved the diff.")
        self.assertContains(response, "No qualifying findings.")
        self.assertLess(body.index("Done"), body.index("QA agent approved the diff."))

    @patch("hitch.main.views.common.Codex")
    def test_qa_approval_feedback_renders_markdown_findings(
        self, mock_codex: MagicMock
    ) -> None:
        # Multi-finding feedback: lists, bold severity tags, inline-code paths
        # must reach the user formatted rather than as raw markdown syntax.
        _patch_thread(
            self,
            mock_codex,
            _thread([_turn([_user_message("Change it"), _agent_message("Done")])]),
        )
        feedback = (
            "Findings:\n\n"
            "- **CRITICAL**: SQL injection in `users.py:42`\n"
            "- **MAJOR**: Missing test for `delete_user`\n"
        )
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="thread-1",
            cwd="/tmp/demo",
            status=SystemWorkflow.STATUS_COMPLETED,
            step=system_agents.STEP_QA_APPROVED,
            state={"next_user_message_index": 1, "last_feedback": feedback},
        )
        instance = _make_codex_instance(
            thread_id="hidden-thread",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            display_author=system_agents.QA_DISPLAY_AUTHOR,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            thread_id="hidden-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_COMPLETED,
            output={"feedback": feedback, "lgtm": True},
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))
        body = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertIn('<div class="body markdown">', body)
        self.assertIn("<strong>CRITICAL</strong>", body)
        self.assertIn("<strong>MAJOR</strong>", body)
        self.assertIn("<code>users.py:42</code>", body)
        # The raw markdown syntax must not leak through to the rendered page.
        self.assertNotIn("**CRITICAL**", body)
        self.assertNotIn("- **MAJOR**", body)

    @patch("hitch.main.views.common.Codex")
    def test_qa_approval_feedback_renders_single_finding_markdown(
        self, mock_codex: MagicMock
    ) -> None:
        # ``looks_like_markdown`` needs two bullets, so single-finding
        # feedback would otherwise stay as raw ``-``/``**``/backticks; the
        # transcript must still surface the formatted finding.
        _patch_thread(
            self,
            mock_codex,
            _thread([_turn([_user_message("Change it"), _agent_message("Done")])]),
        )
        feedback = "- **P1**: issue in `views.py`"
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="thread-1",
            cwd="/tmp/demo",
            status=SystemWorkflow.STATUS_COMPLETED,
            step=system_agents.STEP_QA_APPROVED,
            state={"next_user_message_index": 1, "last_feedback": feedback},
        )
        instance = _make_codex_instance(
            thread_id="hidden-thread",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            display_author=system_agents.QA_DISPLAY_AUTHOR,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            thread_id="hidden-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_COMPLETED,
            output={"feedback": feedback, "lgtm": True},
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))
        body = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertIn('<div class="body markdown">', body)
        self.assertIn("<strong>P1</strong>", body)
        self.assertIn("<code>views.py</code>", body)
        self.assertNotIn("**P1**", body)

    @patch("hitch.main.views.common.Codex")
    def test_completed_local_merge_approval_shows_branch_and_commit(
        self, mock_codex: MagicMock
    ) -> None:
        _patch_thread(
            self,
            mock_codex,
            _thread([_turn([_user_message("Change it"), _agent_message("Done")])]),
        )
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="thread-1",
            cwd="/tmp/demo",
            status=SystemWorkflow.STATUS_COMPLETED,
            step=system_agents.STEP_LOCAL_BRANCH_MERGED,
            state={
                "next_user_message_index": 1,
                "last_feedback": "No qualifying findings.",
                "auto_merge_result": {
                    "branch": "main",
                    "commit_sha": "abc123",
                    "target_worktree": "/tmp/demo",
                    "changed": True,
                },
            },
        )
        instance = _make_codex_instance(
            thread_id="hidden-thread",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            display_author=system_agents.QA_DISPLAY_AUTHOR,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            thread_id="hidden-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_COMPLETED,
            output={"feedback": "No qualifying findings.", "lgtm": True},
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))
        body = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response, "QA agent approved the diff and merged it into main."
        )
        self.assertContains(
            response, '<span class="stage-badge" data-tone="done">Done: Merged</span>'
        )
        self.assertContains(response, "Commit: abc123")
        self.assertContains(response, "No qualifying findings.")
        self.assertLess(
            body.index("Done"),
            body.index("QA agent approved the diff and merged it into main."),
        )

    @patch("hitch.main.repos.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.common.Codex")
    def test_system_session_detail_is_read_only_and_shows_system_prompt(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        prompt = "You are Hitch's QA agent.\nReview <diff>."
        _patch_thread(
            self,
            mock_codex,
            _thread([], id="qa-thread", name="QA thread", cwd="/repo"),
        )
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="thread-1",
            cwd="/repo",
        )
        instance = _make_codex_instance(
            thread_id="qa-thread",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            display_author=system_agents.QA_DISPLAY_AUTHOR,
            prompt=prompt,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            thread_id="qa-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_COMPLETED,
        )

        response = self.client.get(
            reverse("system_session", kwargs={"session_id": "qa-thread"})
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<body class="read-only">')
        self.assertContains(response, "QA agent log")
        self.assertContains(response, '<details class="system-prompt">', html=False)
        self.assertContains(response, "<summary>System prompt</summary>", html=False)
        self.assertContains(response, "Review &lt;diff&gt;.")
        self.assertContains(response, 'aria-label="Session actions"')
        self.assertContains(response, ">Debug chat</a>")
        debug_url = cast(str, cast(Any, response).context["debug_chat_url"])
        parsed = urlparse(debug_url)
        query = parse_qs(parsed.query)
        self.assertEqual(parsed.path, reverse("new_session"))
        self.assertEqual(query["cwd"], ["/repo"])
        self.assertIn("session UID qa-thread", query["prompt"][0])
        self.assertNotContains(response, '<span class="meta-label">stage</span>')
        self.assertNotContains(response, "No messages in this session yet.")
        self.assertNotContains(response, 'class="composer"')
        self.assertNotContains(
            response,
            '<button type="button" role="menuitem" data-edit-title-open>Rename</button>',
            html=False,
        )
        self.assertNotContains(
            response,
            '<button type="button" role="menuitem" data-move-project-open>Move to project</button>',
            html=False,
        )
        self.assertNotContains(response, 'name="archived"')

    def test_system_session_detail_requires_system_run(self) -> None:
        response = self.client.get(
            reverse("system_session", kwargs={"session_id": "thread-1"})
        )

        self.assertEqual(response.status_code, 404)

    @patch("hitch.main.views.common.Codex")
    def test_indicator_stream_url_lives_on_composer_form(
        self, mock_codex: MagicMock
    ) -> None:
        # The connection indicator's EventSource reads ``data-stream-url``
        # off the composer form so it works whether or not a live worker
        # is active. The URL is tagged with the page's render-time view
        # of the session state so the SSE endpoint can detect a stale
        # render and force a reload before any item events flow.
        _patch_thread(self, mock_codex, _thread([]))

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))
        stream_path = reverse("session_stream", kwargs={"session_id": "thread-1"})
        # No workers exist for this session, so both query params are
        # empty — empty is the canonical encoding of "page knows of no
        # prior state".
        self.assertContains(response, "data-composer")
        self.assertContains(response, 'enctype="multipart/form-data"')
        self.assertContains(response, 'name="input_images"')
        self.assertContains(response, "data-image-button")
        self.assertContains(response, "data-image-clear")
        self.assertContains(response, 'data-count')
        self.assertContains(response, "const hasImages = ()")
        self.assertContains(response, "clearImages()")
        self.assertContains(response, ".composer-attachment[data-has-images")
        self.assertNotContains(response, "[image:")
        self.assertContains(response, "flex-wrap: wrap;")
        self.assertContains(response, "min-width: 100%;")
        self.assertContains(
            response,
            f'data-stream-url="{stream_path}?baseline=&amp;active=&amp;workflow=&amp;steering=&amp;demo="',
        )

    @patch("hitch.main.views.common.Codex")
    def test_composer_supports_super_enter_submit(self, mock_codex: MagicMock) -> None:
        _patch_thread(self, mock_codex, _thread([]))

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertContains(response, "event.metaKey")
        self.assertContains(response, 'event.key === "Enter"')
        self.assertContains(response, 'event.getModifierState("Meta")')
        self.assertContains(response, 'event.getModifierState("OS")')
        self.assertContains(response, "requestSubmit(composer, submit)")

    @patch("hitch.main.views.common.Codex")
    def test_composer_adjusts_for_mobile_keyboard(self, mock_codex: MagicMock) -> None:
        _patch_thread(self, mock_codex, _thread([]))

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertContains(response, "--composer-keyboard-offset")
        self.assertContains(response, "--composer-height")
        self.assertContains(response, "window.visualViewport")
        self.assertContains(response, 'window.matchMedia("(max-width: 640px)")')
        self.assertContains(response, "viewport.scale")
        self.assertContains(response, "font-size: 16px;")
        self.assertContains(response, "interactive-widget=resizes-content")
        self.assertContains(response, "composerKeyboardSettleDelays")
        self.assertContains(response, "syncComposerHeight")
        self.assertContains(response, "new ResizeObserver")
        self.assertContains(response, "document.documentElement.style.setProperty")
        self.assertContains(response, "scheduleComposerKeyboardOffset")

    @patch("hitch.main.views.common.Codex")
    def test_composer_exposes_plan_slash_command(self, mock_codex: MagicMock) -> None:
        _patch_thread(self, mock_codex, _thread([]))

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertContains(response, 'class="slash-trigger"')
        self.assertContains(response, 'name="plan_mode"')
        self.assertContains(response, "Plan mode")
        self.assertContains(response, "PR")
        self.assertContains(response, "QA")
        self.assertContains(response, "data-slash-pr")
        self.assertContains(response, "data-slash-pr-now")
        self.assertContains(response, "data-slash-qa")
        self.assertContains(
            response,
            "Rebase on the default branch, clean it up, and then open a PR",
        )
        self.assertContains(
            response,
            "Run the QA agent on the current diff and fix anything it finds",
        )
        self.assertContains(response, "syncNextMessageConfig")
        self.assertContains(response, 'parsePlanCommand() !== null')
        self.assertContains(response, "parsePrCommand")
        self.assertContains(response, "parsePrNowCommand")
        self.assertContains(response, "parseQaCommand")
        self.assertContains(response, '"turn/plan/updated":')
        self.assertContains(response, '"item/plan/delta":')
        # Stream dispatch must only honour own properties so a wire method
        # named "constructor"/"__proto__" can't resolve to a prototype member.
        self.assertContains(
            response,
            "Object.prototype.hasOwnProperty.call(STREAM_HANDLERS, parsed.method)",
        )
        self.assertContains(response, "activateFinalPlanText")
        self.assertContains(response, "input-option-description")
        self.assertContains(response, 'other.placeholder = "Other"')
        self.assertContains(response, "delete entry.answers[key]")

    @patch("hitch.main.views.common.Codex")
    def test_live_task_plan_uses_responsive_panel(self, mock_codex: MagicMock) -> None:
        _patch_thread(self, mock_codex, _thread([]))

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertContains(response, 'class="task-plan"')
        self.assertContains(response, 'data-task-plan')
        self.assertContains(response, 'data-task-plan-current')
        self.assertContains(response, 'data-fallback-order=')
        self.assertContains(response, "@media (min-width: 1120px)")
        self.assertContains(response, "main.has-task-plan")
        self.assertContains(
            response,
            "grid-template-columns: minmax(0, 900px) minmax(260px, 320px)",
        )
        self.assertContains(response, "function updateTaskPlan")
        self.assertContains(response, "function applyTaskPlanOrder")
        self.assertContains(response, "let streamFallbackOrder = 0")
        self.assertContains(response, "Number(taskPlan.dataset.fallbackOrder)")
        self.assertContains(response, "streamFallbackOrder += 1")
        # Task-plan ordering compares all three tuple levels (the fallback-order
        # tiebreak) via the shared comparator.
        self.assertContains(response, "isOrderNewer(order, latestTaskPlanOrder, 3)")
        self.assertContains(
            response,
            '"turn/plan/updated": (payload, parsed, order) => handlePlanUpdated(payload, order)',
        )
        self.assertContains(response, 'status === "in_progress"')
        self.assertContains(response, "taskPlan.dataset.expanded")
        self.assertNotContains(response, "ensurePlanView(turnId, null)")

    @patch("hitch.main.views.common.build_worktree_diff")
    @patch("hitch.main.views.common.Codex")
    def test_open_slash_menu_offsets_live_status_pill(
        self, mock_codex: MagicMock, mock_diff: MagicMock
    ) -> None:
        mock_diff.return_value = DiffView(files=[])
        _patch_thread(self, mock_codex, _thread([]))
        _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_RUNNING,
            prompt="warming up",
            pid=_LIVE_PID,
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertContains(response, 'data-live-work data-state="working"')
        self.assertContains(response, 'data-slash-menu-open="true"')
        self.assertContains(response, "syncSlashMenuStatusOffset")
        self.assertContains(response, 'liveWork.dataset.slashMenuOpen = "true"')

    @patch("hitch.main.views.common.Codex")
    def test_stream_url_carries_baseline_and_active_when_worker_present(
        self, mock_codex: MagicMock
    ) -> None:
        # When a worker is running, the page emits its pk on both query
        # params so ``session_stream`` can confirm the SSE-time DB state
        # still matches what was rendered. A mismatch on either param
        # is the trigger for the reload path.
        _patch_thread(self, mock_codex, _thread([]))
        instance = _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_RUNNING,
            prompt="warming up",
            pid=_LIVE_PID,
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))
        stream_path = reverse("session_stream", kwargs={"session_id": "thread-1"})
        self.assertContains(
            response,
            f'data-stream-url="{stream_path}?baseline={instance.pk}&amp;active={instance.pk}&amp;workflow=&amp;steering=&amp;demo="',
        )

    @patch("hitch.main.views.common.Codex")
    def test_in_progress_turn_is_trimmed_when_worker_active(
        self, mock_codex: MagicMock
    ) -> None:
        # The rollout may already have the in-progress turn's user (and
        # possibly some early agent commentary) by the time the session
        # page loads. The view must hide that range from the rollout-
        # rendered entries so the SSE stream — which replays from byte 0
        # of the events file — doesn't render every item twice.
        prior_user = _user_message("earlier")
        prior_agent = _agent_message("earlier reply")
        in_progress_user = _user_message("run tests")
        partial_agent = _agent_message("working on it")
        thread = _thread(
            [
                _turn([prior_user, prior_agent], started_at=1700000000),
                _turn([in_progress_user, partial_agent], started_at=1700000100),
            ]
        )
        _patch_thread(self, mock_codex, thread)
        _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_RUNNING,
            prompt="run tests",
            pid=_LIVE_PID,
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        body = response.content.decode()
        # The earlier (completed) turn is still rendered server-side.
        self.assertIn("earlier", body)
        self.assertIn("earlier reply", body)
        # The in-progress turn's content is *not* server-rendered. The
        # stream owns that range; the pending bubble covers the user's
        # message until the userMessage event arrives.
        self.assertNotIn("working on it", body)
        self.assertContains(response, "data-pending-user>")
        self.assertContains(response, "run tests")
        self.assertContains(response, "data-live-root")

    @patch("hitch.main.views.common.Codex")
    def test_in_progress_image_only_turn_is_trimmed_when_worker_active(
        self, mock_codex: MagicMock
    ) -> None:
        image_user = _root(
            SimpleNamespace(
                type="userMessage",
                content=[
                    _root(
                        SimpleNamespace(
                            type="localImage",
                            path="/tmp/private/screen.png",
                        )
                    )
                ],
            )
        )
        thread = _thread(
            [
                _turn([_user_message("earlier"), _agent_message("earlier reply")]),
                _turn([image_user, _agent_message("working on image")]),
            ]
        )
        _patch_thread(self, mock_codex, thread)
        _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_RUNNING,
            prompt="",
            input_image_paths=["/tmp/private/screen.png"],
            pid=_LIVE_PID,
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        body = response.content.decode()
        self.assertIn("earlier", body)
        self.assertIn("earlier reply", body)
        self.assertNotIn("working on image", body)
        self.assertNotIn("/tmp/private/screen.png", body)
        self.assertContains(response, "data-pending-user>")
        self.assertContains(response, "[image]")
        self.assertContains(response, "data-live-root")

    @patch("hitch.main.views.common.Codex")
    def test_active_worker_picked_over_newer_terminal_row(
        self, mock_codex: MagicMock
    ) -> None:
        # ``send_message`` can stack workers: a newer row may flip to
        # FAILED quickly while an older row is still RUNNING. The session
        # must stay in streaming mode until *all* workers are terminal.
        _patch_thread(self, mock_codex, _thread([]))
        older = _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_RUNNING,
            prompt="still running",
            pid=_LIVE_PID,
        )
        newer = _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_FAILED,
            prompt="bailed fast",
            pid=_LIVE_PID,
        )
        # Force a strictly later started_at on the newer terminal row so
        # ``latest_for_thread`` would return it; the active-aware helper
        # must skip past it.
        CodexInstance.objects.filter(pk=newer.pk).update(
            started_at=older.started_at + timedelta(seconds=1)
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "data-live-root")
        # The pending bubble carries the still-running worker's prompt,
        # not the bailed one — proves we picked the active row.
        self.assertContains(response, "still running")
        self.assertNotContains(response, "bailed fast")

    @patch("hitch.main.runtime.reconciliation.reconcile_dead_if_due", return_value=0)
    @patch("hitch.main.views.common.Codex")
    def test_dead_worker_is_reconciled_before_render(
        self, mock_codex: MagicMock, mock_global_reconcile: MagicMock
    ) -> None:
        # A worker that died without writing a terminal status would leave
        # the page in "streaming" mode permanently. The session view
        # reconciles this exact thread before reading status so the live UI
        # doesn't appear even when the global sweep is debounced.
        _patch_thread(self, mock_codex, _thread([]))
        instance = _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_RUNNING,
            pid=99999999,
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "data-live-root></div>")
        mock_global_reconcile.assert_called_once()
