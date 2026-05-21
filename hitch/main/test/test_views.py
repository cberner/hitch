"""View-layer tests: index, new_session, send_message, set_session_name,
session_stream.

Shared helpers configure the Codex mock and seed signed cookies so each
test stays focused on the behavior under examination.
"""

import base64
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.core import signing
from django.test import Client, TestCase
from django.urls import reverse
from openai_codex.errors import AppServerError, MethodNotFoundError

from hitch.main import demo, system_agents, views
from hitch.main.models import (
    ApprovalRequest,
    ArchivedSessionTokenUsage,
    CodexInstance,
    KeyResult,
    Objective,
    Project,
    ProposedSession,
    ProposedTask,
    SessionDemo,
    SessionMetadata,
    StandingOrder,
    SystemAgentRun,
    SystemWorkflow,
    UserInputRequest,
    UserSettings,
)
from hitch.main.worktrees import (
    ManagedWorktree,
    WorktreeCleanupError,
    WorktreeCreationError,
)

_SHOW_ARCHIVED_COOKIE = "hitch_show_archived_sessions"
_MODEL_COOKIE = "hitch_model"
_EXTRA_SYSTEM_PROMPT_COOKIE = "hitch_extra_system_prompt"
_USE_WORKTREES_COOKIE = "hitch_use_worktrees"
_AUTO_PR_COOKIE = "hitch_auto_pr"
_LAST_SELECTED_REPO_COOKIE = "hitch_last_selected_repo"
_ENABLE_MEMORIES_COOKIE = "hitch_enable_memories"
_PR_PROMPT = (
    "Do a thorough review of the diff. Rebase on master, clean it up, "
    "and then open a PR"
)
_QA_PROMPT = "Run the QA agent on the current diff and fix anything it finds"


def _rollout_line(
    line_type: str,
    payload: dict[str, object],
    *,
    timestamp: str = "2025-01-05T12:00:00Z",
) -> str:
    return json.dumps({"timestamp": timestamp, "type": line_type, "payload": payload})


def _token_count_line(
    *,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    reasoning_output_tokens: int = 0,
    context_tokens: int = 0,
    model_context_window: int = 0,
) -> str:
    return _rollout_line(
        "event_msg",
        {
            "type": "token_count",
            "info": {
                "total_token_usage": {
                    "input_tokens": input_tokens,
                    "cached_input_tokens": cached_input_tokens,
                    "output_tokens": output_tokens,
                    "reasoning_output_tokens": reasoning_output_tokens,
                    "total_tokens": total_tokens,
                },
                "last_token_usage": {
                    "total_tokens": context_tokens,
                },
                "model_context_window": model_context_window,
            },
        },
    )


def _make_rollout(
    testcase: TestCase, lines: list[str], *, archived: bool = False
) -> Path:
    temp_dir = tempfile.TemporaryDirectory()
    testcase.addCleanup(temp_dir.cleanup)
    parent = Path(temp_dir.name)
    if archived:
        parent = parent / "archived_sessions"
        parent.mkdir()
    path = parent / "rollout.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _setup_codex(
    mock_codex: MagicMock,
    *,
    threads: list[Any] | None = None,
    archived_threads: list[Any] | None = None,
    models: list[Any] | None = None,
) -> MagicMock:
    """Configure the Codex mock with ``thread_list`` and ``models``.

    The index view reads both active and, when enabled, archived thread
    lists. Also stubs ``_client.request`` to raise
    MethodNotFound so the rate-limits fetch falls through its
    unsupported-endpoint branch — tests that care set their own value."""
    ctx: MagicMock = mock_codex.return_value.__enter__.return_value

    def thread_list(*, archived: bool | None = None, **_: Any) -> SimpleNamespace:
        data = archived_threads if archived else threads
        return SimpleNamespace(data=data or [])

    ctx.thread_list.side_effect = thread_list
    ctx.models.return_value.data = models or []
    ctx._client.request.side_effect = MethodNotFoundError(
        -32601, "method not found", None
    )
    return ctx


def _make_model(model_id: str, *, is_default: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        id=model_id,
        display_name=model_id,
        is_default=is_default,
        default_reasoning_effort=SimpleNamespace(value="medium"),
        supported_reasoning_efforts=[
            SimpleNamespace(reasoning_effort=SimpleNamespace(value=v), description=v)
            for v in ("low", "medium", "high")
        ],
    )


def _sign(name: str, value: str) -> str:
    return signing.get_cookie_signer(salt=name).sign(value)


def _encode_extra_system_prompt(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode()).decode("ascii")


def _seed_cookies(client: Client, **values: str) -> None:
    for name, value in values.items():
        client.cookies[name] = _sign(name, value)


def _cookie_value(response: object, name: str) -> str:
    raw = response.cookies[name].value  # type: ignore[attr-defined]
    return signing.get_cookie_signer(salt=name).unsign(raw)


def _session(
    session_id: str = "sess",
    *,
    name: str | None = None,
    preview: str = "",
    cwd: str = "/repo",
    path: str | None = None,
    updated_at: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=session_id,
        name=name,
        preview=preview,
        cwd=cwd,
        path=path,
        updated_at=updated_at,
    )


class PendingPlanStateTests(TestCase):
    def test_approval_declined_does_not_clear_pending_plan(self) -> None:
        entries = [
            {"kind": "user", "text": "Plan it"},
            {"kind": "plan", "text": "# Plan"},
            {"kind": "user", "text": "Try command"},
            {"kind": "approval_declined", "detail": "git push"},
        ]

        self.assertTrue(views._entries_await_plan_approval(entries))

    def test_agent_answer_clears_pending_plan(self) -> None:
        entries = [
            {"kind": "user", "text": "Plan it"},
            {"kind": "plan", "text": "# Plan"},
            {"kind": "user", "text": "Implement the plan"},
            {"kind": "agent", "text": "Done"},
        ]

        self.assertFalse(views._entries_await_plan_approval(entries))

    def test_only_latest_pending_plan_is_actionable(self) -> None:
        entries = [
            {"kind": "user", "text": "Plan it"},
            {"kind": "plan", "text": "# Old Plan"},
            {"kind": "user", "text": "Revise"},
            {"kind": "plan", "text": "# Current Plan"},
        ]

        views._mark_pending_plan_actions(entries)

        self.assertFalse(entries[1]["show_plan_actions"])
        self.assertTrue(entries[3]["show_plan_actions"])

    def test_agent_answer_clears_plan_actions(self) -> None:
        entries = [
            {"kind": "user", "text": "Plan it"},
            {"kind": "plan", "text": "# Plan"},
            {"kind": "user", "text": "Implement the plan"},
            {"kind": "agent", "text": "Done"},
        ]

        views._mark_pending_plan_actions(entries)

        self.assertFalse(entries[1]["show_plan_actions"])


class IndexViewTests(TestCase):
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_renders_empty_state_and_new_session_button(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _setup_codex(mock_codex)
        mock_discover.return_value = []
        response = self.client.get(reverse("index"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "HITCH")
        self.assertContains(response, "No sessions found.")
        self.assertContains(response, "New session")
        self.assertContains(response, "No git repositories found")
        self.assertContains(response, "Create project")

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_hides_project_banner_when_project_exists(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        Project.objects.create(name="Hitch", repo_path="/repo")
        _setup_codex(mock_codex)
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "Create a project to group sessions")

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_hides_system_agent_threads(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        visible = _session("visible", preview="Visible")
        hidden = _session("qa-thread", preview="Hidden QA")
        _setup_codex(mock_codex, threads=[visible, hidden])
        mock_discover.return_value = []
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="visible",
            cwd="/repo",
        )
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="qa-thread",
            cwd="/repo",
            prompt="qa",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind="pr_qa",
            thread_id="qa-thread",
            instance=instance,
        )

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Visible")
        self.assertNotContains(response, "Hidden QA")

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_paginates_sessions_before_hiding_system_agent_threads(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        hidden = _session("qa-thread", preview="Hidden QA", updated_at=2000)
        visible = _session(
            "visible-next-page", preview="Visible next page", updated_at=1000
        )
        client = _setup_codex(mock_codex)

        def thread_list(
            *, archived: bool | None = None, cursor: str | None = None, **_: Any
        ) -> SimpleNamespace:
            if archived:
                return SimpleNamespace(data=[])
            if cursor == "page-2":
                return SimpleNamespace(data=[visible])
            return SimpleNamespace(data=[hidden], next_cursor="page-2")

        client.thread_list.side_effect = thread_list
        mock_discover.return_value = []
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="visible-next-page",
            cwd="/repo",
        )
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="qa-thread",
            cwd="/repo",
            prompt="qa",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind="pr_qa",
            thread_id="qa-thread",
            instance=instance,
        )

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Visible next page")
        self.assertNotContains(response, "Hidden QA")

    @patch("hitch.main.views.Codex")
    def test_system_sessions_lists_hidden_threads_as_read_only_links(
        self, mock_codex: MagicMock
    ) -> None:
        visible = _session("visible", preview="Visible")
        hidden = _session("qa-thread", preview="Hidden QA")
        _setup_codex(mock_codex, threads=[visible, hidden])
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="visible",
            cwd="/repo",
        )
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="qa-thread",
            cwd="/repo",
            prompt="qa",
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
            instance=instance,
            status=SystemAgentRun.STATUS_COMPLETED,
        )

        response = self.client.get(reverse("system_sessions"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "System sessions")
        self.assertContains(response, "Hidden QA")
        self.assertContains(
            response, reverse("system_session", kwargs={"session_id": "qa-thread"})
        )
        self.assertContains(response, "QA agent")
        self.assertContains(response, "completed")
        self.assertNotContains(response, "Visible")
        self.assertNotContains(response, 'aria-label="Session actions"')
        self.assertNotContains(response, "data-session-archive-url")
        self.assertNotContains(
            response, '<dialog class="new-session" data-new-session-dialog', html=False
        )

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_new_session_dialog_adjusts_for_mobile_keyboard(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _setup_codex(mock_codex)
        mock_discover.return_value = ["/repo"]

        response = self.client.get(reverse("index"))

        self.assertContains(response, "keyboard-adjusted")
        self.assertContains(response, "--dialog-keyboard-top")
        self.assertContains(response, "window.visualViewport")
        self.assertContains(response, 'window.matchMedia("(max-width: 640px)")')
        self.assertContains(response, "scheduleKeyboardAdjustedDialog();")

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_lists_sessions_sorted_descending(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        sessions = [
            _session("older", name="Older session", updated_at=1000),
            _session("newer", name="Newer session", updated_at=2000),
            _session("middle", name="Middle session", updated_at=1500),
        ]
        _setup_codex(mock_codex, threads=sessions)
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))
        body = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response, reverse("session", kwargs={"session_id": "newer"})
        )
        self.assertContains(
            response,
            'data-session-archive-url="'
            + reverse("set_session_archived", kwargs={"session_id": "newer"})
            + '"',
        )
        self.assertContains(
            response,
            'data-session-name-url="'
            + reverse("set_session_name", kwargs={"session_id": "newer"})
            + '"',
        )
        self.assertContains(response, 'data-session-archived="false"')
        self.assertContains(response, 'aria-label="Session actions"')
        self.assertContains(response, 'data-session-rename-open')
        self.assertContains(response, 'name="name" value="Newer session" maxlength="200"')
        self.assertContains(response, 'name="next" value="index"')
        self.assertContains(response, 'data-session-archive-label>Archive</button>')
        self.assertContains(response, "data-archive-undo")
        self.assertLess(body.index("Newer session"), body.index("Middle session"))
        self.assertLess(body.index("Middle session"), body.index("Older session"))

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_selected_project_filters_sessions(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        other = Project.objects.create(name="Other", repo_path="/other")
        sessions = [
            _session("matching", name="Matching", cwd="/repo"),
            _session("other", name="Other session", cwd="/other"),
        ]
        _setup_codex(mock_codex, threads=sessions)
        mock_discover.return_value = [Path("/repo"), Path("/other")]
        SessionMetadata.objects.create(thread_id="matching", cwd="/repo", project=project)
        SessionMetadata.objects.create(thread_id="other", cwd="/other", project=other)
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Matching")
        self.assertContains(response, "Hitch sessions")
        self.assertContains(response, 'name="selected_project"')
        self.assertNotContains(response, "Other session")

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_no_project_metadata_prevents_cwd_project_inference(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _setup_codex(mock_codex, threads=[_session("cleared", name="Cleared", cwd="/repo")])
        mock_discover.return_value = [Path("/repo")]
        SessionMetadata.objects.create(
            thread_id="cleared", cwd="/repo", project=None, project_cleared=True
        )
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "Cleared")

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_session_list_omits_token_usage(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        rollout_path = _make_rollout(
            self,
            [
                _token_count_line(
                    input_tokens=400_000,
                    cached_input_tokens=25_000,
                    output_tokens=562_654,
                    total_tokens=987_654,
                )
            ],
        )
        active = _session("active", name="Active session", path=str(rollout_path))
        _setup_codex(mock_codex, threads=[active])
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Active session")
        self.assertNotContains(response, 'aria-label="Token usage"')
        self.assertNotContains(response, "987,654")
        self.assertEqual(ArchivedSessionTokenUsage.objects.count(), 0)

        rollout_path.write_text(
            _token_count_line(
                input_tokens=500_000,
                cached_input_tokens=30_000,
                output_tokens=704_567,
                total_tokens=1_234_567,
            ),
            encoding="utf-8",
        )

        response = self.client.get(reverse("index"))

        self.assertNotContains(response, "1,234,567")
        self.assertNotContains(response, "987,654")
        self.assertEqual(ArchivedSessionTokenUsage.objects.count(), 0)

    @patch("hitch.main.views.Codex")
    def test_usage_page_archived_token_usage_refreshes_when_rollout_changes(
        self, mock_codex: MagicMock
    ) -> None:
        rollout_path = _make_rollout(
            self,
            [
                _token_count_line(
                    input_tokens=100_000,
                    cached_input_tokens=10_000,
                    output_tokens=23_456,
                    total_tokens=123_456,
                )
            ],
            archived=True,
        )
        os.utime(rollout_path, ns=(1_000_000_000, 1_000_000_000))
        archived = _session(
            "archived",
            name="Archived session",
            path=str(rollout_path),
        )
        _setup_codex(mock_codex, archived_threads=[archived])

        response = self.client.get(reverse("usage"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "113,456")
        self.assertNotContains(response, "123,456")
        cache = ArchivedSessionTokenUsage.objects.get(thread_id="archived")
        self.assertEqual(cache.total_tokens, 123_456)
        self.assertEqual(cache.rollout_mtime_ns, 1_000_000_000)

        rollout_path.write_text(
            _token_count_line(
                input_tokens=900_000,
                cached_input_tokens=90_000,
                output_tokens=99_999,
                total_tokens=999_999,
            ),
            encoding="utf-8",
        )
        os.utime(rollout_path, ns=(2_000_000_000, 2_000_000_000))

        response = self.client.get(reverse("usage"))

        self.assertContains(response, "909,999")
        self.assertNotContains(response, "999,999")
        self.assertNotContains(response, "123,456")
        cache.refresh_from_db()
        self.assertEqual(cache.total_tokens, 999_999)
        self.assertEqual(cache.rollout_mtime_ns, 2_000_000_000)

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_hides_archived_sessions_by_default(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        active = _session("active", name="Active session")
        archived = _session(
            "archived",
            name="Archived session",
            path="/home/user/.codex/archived_sessions/archived.jsonl",
        )
        _setup_codex(mock_codex, threads=[active], archived_threads=[archived])
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertContains(response, "Active session")
        self.assertNotContains(response, "Archived session")
        client = mock_codex.return_value.__enter__.return_value
        client.thread_list.assert_called_once_with()

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_shows_archived_sessions_when_setting_enabled(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _seed_cookies(self.client, **{_SHOW_ARCHIVED_COOKIE: "true"})
        active = _session("active", name="Active session", updated_at=1000)
        archived = _session(
            "archived",
            name="Archived session",
            path="/home/user/.codex/archived_sessions/archived.jsonl",
            updated_at=2000,
        )
        _setup_codex(mock_codex, threads=[active], archived_threads=[archived])
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))
        body = response.content.decode()

        self.assertContains(response, "Active session")
        self.assertContains(response, "Archived session")
        self.assertContains(response, '<span class="archive-badge">Archived</span>')
        self.assertContains(response, 'data-session-archived="true"')
        self.assertContains(response, 'data-session-archive-label>Unarchive</button>')
        self.assertContains(response, 'name="archived" value="false"')
        self.assertLess(body.index("Archived session"), body.index("Active session"))
        client = mock_codex.return_value.__enter__.return_value
        client.thread_list.assert_any_call()
        client.thread_list.assert_any_call(archived=True)

    @patch("hitch.main.views.Codex")
    def test_usage_page_sums_lifetime_token_usage_without_cached_double_count(
        self, mock_codex: MagicMock
    ) -> None:
        active_path = _make_rollout(
            self,
            [
                _token_count_line(
                    input_tokens=400,
                    cached_input_tokens=50,
                    output_tokens=600,
                    reasoning_output_tokens=40,
                    total_tokens=1_040,
                )
            ],
        )
        archived_path = _make_rollout(
            self,
            [
                _token_count_line(
                    input_tokens=1_000,
                    cached_input_tokens=200,
                    output_tokens=1_500,
                    reasoning_output_tokens=300,
                    total_tokens=3_000,
                )
            ],
            archived=True,
        )
        _setup_codex(
            mock_codex,
            threads=[_session("active", name="Active session", path=str(active_path))],
            archived_threads=[
                _session("archived", name="Archived session", path=str(archived_path))
            ],
        )

        response = self.client.get(reverse("usage"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Lifetime")
        self.assertContains(response, "All sessions")
        self.assertContains(response, "3,790")
        self.assertContains(response, "1,150")
        self.assertContains(response, "2,100")
        self.assertContains(response, "250")
        self.assertNotContains(response, "4,040")
        self.assertNotContains(response, "3,250")
        self.assertNotContains(response, "1,400")
        cache = ArchivedSessionTokenUsage.objects.get(thread_id="archived")
        self.assertEqual(cache.total_tokens, 3_000)
        client = mock_codex.return_value.__enter__.return_value
        client.thread_list.assert_any_call()
        client.thread_list.assert_any_call(archived=True)

    @patch("hitch.main.views.Codex")
    def test_usage_page_marks_lifetime_unavailable_when_session_list_fails(
        self, mock_codex: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")

        response = self.client.get(reverse("usage"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Lifetime")
        self.assertContains(response, "Lifetime usage unavailable.")
        self.assertNotContains(response, "All sessions")

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_new_session_dialog_populates_project_and_bare_repo_selectors(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        project_a = Project.objects.create(
            name="Project A",
            repo_path="/home/user/proj-a",
            auto_pr_mode=Project.AUTO_PR_ON,
        )
        Project.objects.create(name="Project B", repo_path="/home/user/proj-b")
        _setup_codex(mock_codex)
        mock_discover.return_value = [Path("/home/user/proj-a"), Path("/home/user/proj-b")]

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'name="project"')
        self.assertContains(response, "Project A")
        self.assertContains(response, "Project B")
        self.assertContains(response, "&lt;bare repo&gt;")
        self.assertContains(response, f'value="{project_a.pk}" selected')
        self.assertContains(response, 'data-auto-pr-default="true"')
        self.assertContains(response, "data-new-session-auto-pr checked")
        self.assertContains(response, "data-new-session-repo-field hidden")
        self.assertContains(response, "/home/user/proj-a")
        self.assertContains(response, "/home/user/proj-b")
        self.assertContains(response, 'name="cwd"')

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_repo_dropdown_selects_saved_repo(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _seed_cookies(
            self.client,
            **{_LAST_SELECTED_REPO_COOKIE: "/home/user/proj-b"},
        )
        _setup_codex(mock_codex)
        mock_discover.return_value = [Path("/home/user/proj-a"), Path("/home/user/proj-b")]

        response = self.client.get(reverse("index"))

        self.assertContains(response, 'value="/home/user/proj-b" selected')
        self.assertNotContains(response, 'value="/home/user/proj-a" selected')

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_project_dropdown_selects_saved_repo_project(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        Project.objects.create(name="Project A", repo_path="/home/user/proj-a")
        project_b = Project.objects.create(name="Project B", repo_path="/home/user/proj-b")
        _seed_cookies(
            self.client,
            **{_LAST_SELECTED_REPO_COOKIE: "/home/user/proj-b"},
        )
        _setup_codex(mock_codex)
        mock_discover.return_value = [Path("/home/user/proj-a"), Path("/home/user/proj-b")]

        response = self.client.get(reverse("index"))

        self.assertContains(response, f'value="{project_b.pk}" selected')
        self.assertContains(response, "data-new-session-repo-field hidden")

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_project_dropdown_keeps_saved_unprojected_repo_on_bare_option(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        Project.objects.create(name="Project A", repo_path="/home/user/proj-a")
        _seed_cookies(
            self.client,
            **{_LAST_SELECTED_REPO_COOKIE: "/home/user/bare"},
        )
        _setup_codex(mock_codex)
        mock_discover.return_value = [Path("/home/user/proj-a"), Path("/home/user/bare")]

        response = self.client.get(reverse("index"))

        self.assertContains(
            response, f'value="{views._BARE_REPO_PROJECT_VALUE}" selected'
        )
        self.assertNotContains(response, "data-new-session-repo-field hidden")
        self.assertContains(response, 'value="/home/user/bare" selected')

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_project_dropdown_ignores_stale_saved_repo(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        project = Project.objects.create(name="Project A", repo_path="/home/user/proj-a")
        _seed_cookies(
            self.client,
            **{_LAST_SELECTED_REPO_COOKIE: "/home/user/missing"},
        )
        _setup_codex(mock_codex)
        mock_discover.return_value = [Path("/home/user/proj-a")]

        response = self.client.get(reverse("index"))

        self.assertContains(response, f'value="{project.pk}" selected')
        self.assertContains(response, "data-new-session-repo-field hidden")

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_new_session_dialog_supports_super_enter_submit(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _setup_codex(mock_codex)
        mock_discover.return_value = [Path("/home/user/proj")]

        response = self.client.get(reverse("index"))

        self.assertContains(response, "data-new-session-submit")
        self.assertContains(response, "event.metaKey")
        self.assertContains(response, 'event.key === "Enter"')
        self.assertContains(response, 'event.getModifierState("Meta")')
        self.assertContains(response, 'event.getModifierState("OS")')
        self.assertContains(response, "requestSubmit(newSessionForm, newSessionSubmit)")

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_new_session_dialog_exposes_plan_slash_command(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _setup_codex(mock_codex)
        mock_discover.return_value = [Path("/home/user/proj")]

        response = self.client.get(reverse("index"))

        self.assertContains(response, 'class="slash-trigger"')
        self.assertContains(response, 'name="plan_mode"')
        self.assertContains(response, "Plan mode")
        self.assertNotContains(response, "data-slash-pr")
        self.assertContains(response, _PR_PROMPT)
        self.assertContains(response, _QA_PROMPT)
        self.assertContains(response, "parseNewSessionPlanCommand")
        self.assertContains(response, "parseNewSessionPrCommand")
        self.assertContains(response, "parseNewSessionQaCommand")
        self.assertContains(response, 'toLowerCase() !== "/plan"')
        self.assertContains(response, 'toLowerCase() !== "/qa"')

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_title_rendering(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        """Per-row title display: user-set name wins, otherwise the preview's
        first line clipped to 80 chars, otherwise the bare id."""
        long_text = "x" * 200
        sessions = [
            _session("long-preview", preview=long_text),
            _session("multiline", preview="first line\nsecond line\nthird line"),
            _session("named", name="Short title", preview="ignored long preview " * 20),
            _session("bare-id"),
        ]
        _setup_codex(mock_codex, threads=sessions)
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        # Long preview is clipped, untruncated form must not leak.
        self.assertContains(response, "x" * 80 + "...")
        self.assertNotContains(response, "x" * 120)
        # Multiline preview collapses to first line.
        self.assertContains(response, "first line")
        self.assertNotContains(response, "second line")
        # Named row uses the name, not the preview.
        self.assertContains(response, "Short title")
        self.assertNotContains(response, "ignored long preview")
        # No name + no preview → fall back to the id.
        self.assertContains(response, ">bare-id<")


class ProjectViewTests(TestCase):
    def test_projects_default_to_follow_global_auto_pr(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")

        self.assertEqual(project.auto_pr_mode, Project.AUTO_PR_FOLLOW_GLOBAL)

    @patch("hitch.main.views.discover_repos")
    def test_new_project_form_lists_discovered_repos(self, mock_discover: MagicMock) -> None:
        mock_discover.return_value = [Path("/repo")]

        response = self.client.get(reverse("new_project"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Create project")
        self.assertContains(response, 'name="repo_path"')
        self.assertContains(response, "/repo")

    @patch("hitch.main.views.discover_repos")
    def test_new_project_form_hides_repos_that_already_have_projects(
        self, mock_discover: MagicMock
    ) -> None:
        Project.objects.create(name="Existing", repo_path="/repo")
        mock_discover.return_value = [Path("/repo"), Path("/other")]

        response = self.client.get(reverse("new_project"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, '<option value="/repo">/repo</option>', html=True)
        self.assertContains(response, '<option value="/other">/other</option>', html=True)

    @patch("hitch.main.views.git_common_dir")
    @patch("hitch.main.views.discover_repos")
    def test_new_project_form_hides_repos_with_existing_git_common_dir(
        self, mock_discover: MagicMock, mock_common_dir: MagicMock
    ) -> None:
        Project.objects.create(
            name="Existing",
            repo_path="/repo",
            git_common_dir="/repo/.git",
        )
        mock_discover.return_value = [Path("/repo-worktree")]
        mock_common_dir.return_value = Path("/repo/.git")

        response = self.client.get(reverse("new_project"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "/repo-worktree")
        self.assertContains(response, "No git repositories without projects")

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.discover_repos")
    def test_creates_project_selects_it_and_associates_existing_sessions(
        self, mock_discover: MagicMock, mock_codex: MagicMock
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        _setup_codex(
            mock_codex,
            threads=[
                _session("match", name="Match", cwd="/repo"),
                _session("miss", name="Miss", cwd="/other"),
            ],
        )

        response = self.client.post(
            reverse("new_project"),
            data={"name": "Hitch", "repo_path": "/repo"},
        )

        self.assertEqual(response.status_code, 302)
        project = Project.objects.get()
        self.assertEqual(project.name, "Hitch")
        self.assertEqual(project.repo_path, "/repo")
        self.assertEqual(_cookie_value(response, "hitch_selected_project_id"), str(project.pk))
        self.assertEqual(SessionMetadata.objects.get(thread_id="match").project, project)
        self.assertFalse(SessionMetadata.objects.filter(thread_id="miss").exists())

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.git_common_dir")
    @patch("hitch.main.views.discover_repos")
    def test_rejects_project_with_existing_git_common_dir(
        self,
        mock_discover: MagicMock,
        mock_common_dir: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        Project.objects.create(
            name="Source",
            repo_path="/repo",
            git_common_dir="/repo/.git",
        )
        mock_discover.return_value = [Path("/repo-worktree")]
        mock_common_dir.return_value = Path("/repo/.git")
        _setup_codex(mock_codex)

        response = self.client.post(
            reverse("new_project"),
            data={"name": "Worktree", "repo_path": "/repo-worktree"},
        )

        self.assertContains(
            response,
            "project already exists for repository",
            status_code=400,
        )
        self.assertEqual(Project.objects.count(), 1)

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.discover_repos")
    def test_project_creation_preserves_manually_cleared_sessions(
        self, mock_discover: MagicMock, mock_codex: MagicMock
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        _setup_codex(
            mock_codex,
            threads=[
                _session("cleared", name="Cleared", cwd="/repo"),
                _session("ordinary", name="Ordinary", cwd="/repo"),
            ],
        )
        SessionMetadata.objects.create(
            thread_id="cleared",
            cwd="/repo",
            project=None,
            project_cleared=True,
        )
        SessionMetadata.objects.create(thread_id="ordinary", cwd="/repo", project=None)

        response = self.client.post(
            reverse("new_project"),
            data={"name": "Hitch", "repo_path": "/repo"},
        )

        self.assertEqual(response.status_code, 302)
        project = Project.objects.get()
        self.assertIsNone(SessionMetadata.objects.get(thread_id="cleared").project)
        self.assertTrue(SessionMetadata.objects.get(thread_id="cleared").project_cleared)
        self.assertEqual(SessionMetadata.objects.get(thread_id="ordinary").project, project)

    def test_edit_project_updates_name_and_auto_pr_mode(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")

        response = self.client.post(
            reverse("edit_project"),
            data={
                "project": str(project.pk),
                "name": "Renamed",
                "auto_pr_mode": Project.AUTO_PR_ON,
            },
        )

        self.assertEqual(response.status_code, 302)
        project.refresh_from_db()
        self.assertEqual(project.name, "Renamed")
        self.assertEqual(project.auto_pr_mode, Project.AUTO_PR_ON)

    def test_edit_project_rejects_invalid_posts(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")

        for data, message in (
            (
                {
                    "project": "",
                    "name": "Renamed",
                    "auto_pr_mode": Project.AUTO_PR_ON,
                },
                "project is required",
            ),
            (
                {
                    "project": str(project.pk),
                    "name": "",
                    "auto_pr_mode": Project.AUTO_PR_ON,
                },
                "project name is required",
            ),
            (
                {
                    "project": str(project.pk),
                    "name": "Renamed",
                    "auto_pr_mode": "maybe",
                },
                "invalid project auto-PR setting",
            ),
        ):
            with self.subTest(message=message):
                response = self.client.post(reverse("edit_project"), data=data)
                self.assertContains(response, message, status_code=400)

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.discover_repos")
    def test_rejects_invalid_project_posts(
        self, mock_discover: MagicMock, mock_codex: MagicMock
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        _setup_codex(mock_codex)

        for data, message in (
            ({"name": "", "repo_path": "/repo"}, "project name is required"),
            ({"name": "Hitch", "repo_path": "/etc"}, "repository must be a discovered repository"),
        ):
            with self.subTest(message=message):
                response = self.client.post(reverse("new_project"), data=data)
                self.assertContains(response, message, status_code=400)


class OKRModelTests(TestCase):
    def test_project_objectives_and_key_results_cascade(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        objective = Objective.objects.create(project=project, title="Ship OKRs")
        KeyResult.objects.create(objective=objective, title="Create UI")

        project.delete()

        self.assertEqual(Objective.objects.count(), 0)
        self.assertEqual(KeyResult.objects.count(), 0)

    def test_objective_and_key_result_string_values_are_titles(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        objective = Objective.objects.create(project=project, title="Ship OKRs")
        key_result = KeyResult.objects.create(objective=objective, title="Create UI")
        task = ProposedTask.objects.create(
            key_result=key_result,
            title="Build task generation",
        )

        self.assertEqual(str(objective), "Ship OKRs")
        self.assertEqual(str(key_result), "Create UI")
        self.assertEqual(str(task), "Build task generation")

    def test_proposed_tasks_cascade_with_key_result(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        objective = Objective.objects.create(project=project, title="Ship OKRs")
        key_result = KeyResult.objects.create(objective=objective, title="Create UI")
        ProposedTask.objects.create(key_result=key_result, title="Build task generation")

        key_result.delete()

        self.assertEqual(ProposedTask.objects.count(), 0)


class OKRViewTests(TestCase):
    def _select_project(self, project: Project) -> None:
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))

    @patch("hitch.main.views.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.Codex")
    def test_okrs_page_lists_selected_project_objectives_with_nested_key_results(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _setup_codex(mock_codex)
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        other = Project.objects.create(name="Other", repo_path="/other")
        objective = Objective.objects.create(
            project=project,
            title="Improve review flow",
            description="Make PR reviews faster.",
        )
        KeyResult.objects.create(
            objective=objective,
            title="Open PR from session",
            description="The workflow creates a pull request.",
            work_instructions="Use the GitHub connector.",
        )
        Objective.objects.create(project=other, title="Hidden objective")
        self._select_project(project)

        response = self.client.get(reverse("okrs"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, f'href="{reverse("okrs")}" aria-current="page"')
        self.assertContains(response, "Improve review flow")
        self.assertContains(response, "Make PR reviews faster.")
        self.assertContains(response, "Open PR from session")
        self.assertContains(response, "Use the GitHub connector.")
        self.assertContains(response, "Extra instructions to help generate and perform tasks.")
        self.assertContains(response, "Generate tasks")
        self.assertContains(response, "Show hidden tasks")
        self.assertNotContains(response, "Hidden objective")

    @patch("hitch.main.views.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.reconcile_dead")
    def test_okrs_page_reconciles_dead_workers_before_rendering_state(
        self,
        mock_reconcile: MagicMock,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        _setup_codex(mock_codex)
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        self._select_project(project)

        response = self.client.get(reverse("okrs"))

        self.assertEqual(response.status_code, 200)
        mock_reconcile.assert_called_once()

    @patch("hitch.main.views.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.Codex")
    def test_okrs_page_lists_proposed_tasks_and_generation_status(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _setup_codex(mock_codex)
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        objective = Objective.objects.create(project=project, title="Improve planning")
        key_result = KeyResult.objects.create(objective=objective, title="Draft plan")
        ProposedTask.objects.create(
            key_result=key_result,
            title="Add task model",
            description="Store generated tasks.",
            success_criteria="Tasks render on the OKR page.",
            rationale="The KR needs persisted proposals.",
            outcome_status=ProposedTask.OUTCOME_ACCEPTED,
            outcome_notes="Useful scope.",
        )
        SystemWorkflow.objects.create(
            kind=system_agents.OKR_TASK_AGENT_KIND,
            main_thread_id=system_agents._okr_task_main_thread_id(key_result.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_BLOCKED,
            step="blocked",
            state={"error": "task planning output was not valid JSON"},
        )
        self._select_project(project)

        response = self.client.get(reverse("okrs"), data={"show_hidden_tasks": "1"})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Add task model")
        self.assertContains(response, "Tasks render on the OKR page.")
        self.assertContains(response, "Outcome:")
        self.assertContains(response, "Accepted")
        self.assertContains(response, "Useful scope.")
        self.assertContains(response, "Last generation: blocked")
        self.assertContains(response, "task planning output was not valid JSON")

    @patch("hitch.main.views.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.Codex")
    def test_okrs_do_it_prompt_includes_okr_context(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _setup_codex(mock_codex)
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        objective = Objective.objects.create(
            project=project,
            title="Improve planning",
            description="Make plans concrete.",
        )
        key_result = KeyResult.objects.create(
            objective=objective,
            title="Draft plan",
            description="Capture the work needed to ship planning.",
            work_instructions="Use concise task prompts.",
        )
        ProposedTask.objects.create(
            key_result=key_result,
            title="Add task model",
            description="Store generated tasks.",
        )
        self._select_project(project)

        response = self.client.get(reverse("okrs"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Do this ProposedTask.")
        self.assertContains(
            response,
            "This task is part of the following Key Result (KR), which is part of "
            "the following Objective.",
        )
        self.assertContains(response, "Objective: Improve planning")
        self.assertContains(response, "Objective description:")
        self.assertContains(response, "Make plans concrete.")
        self.assertContains(response, "Key Result: Draft plan")
        self.assertContains(response, "Key Result description:")
        self.assertContains(response, "Capture the work needed to ship planning.")
        self.assertContains(response, "Key Result work instructions:")
        self.assertContains(response, "Use concise task prompts.")
        self.assertContains(
            response,
            "There will be other tasks to complete the rest of this Key Result. "
            "Only do this part, even if the result seems incomplete without the "
            "other tasks.",
        )

    @patch("hitch.main.views.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.Codex")
    def test_okrs_page_hides_completed_proposed_tasks_by_default(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _setup_codex(mock_codex)
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        objective = Objective.objects.create(project=project, title="Improve planning")
        key_result = KeyResult.objects.create(objective=objective, title="Draft plan")
        ProposedTask.objects.create(key_result=key_result, title="Pending task")
        ProposedTask.objects.create(
            key_result=key_result,
            title="Rejected task",
            outcome_status=ProposedTask.OUTCOME_REJECTED,
            outcome_notes="Not useful.",
        )
        self._select_project(project)

        response = self.client.get(reverse("okrs"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Pending task")
        self.assertContains(response, "Do it")
        self.assertContains(response, "Reject")
        self.assertContains(response, "1 hidden task")
        self.assertNotContains(response, "Rejected task")

    @patch("hitch.main.views.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.Codex")
    def test_okrs_page_links_running_task_generation_log(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _setup_codex(mock_codex)
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        objective = Objective.objects.create(project=project, title="Improve planning")
        key_result = KeyResult.objects.create(objective=objective, title="Draft plan")
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.OKR_TASK_AGENT_KIND,
            main_thread_id=system_agents._okr_task_main_thread_id(key_result.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_OKR_TASKS_RUNNING,
            state={"key_result_id": key_result.pk},
        )
        instance = CodexInstance.objects.create(
            pid=os.getpid(),
            thread_id="task-thread",
            cwd="/repo",
            prompt="generate tasks",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.OKR_TASK_AGENT_KIND,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.OKR_TASK_AGENT_KIND,
            thread_id="task-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )
        self._select_project(project)

        response = self.client.get(reverse("okrs"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Generating...")
        self.assertContains(response, "View log")
        log_url = reverse(
            "okr_task_generation_log", kwargs={"workflow_id": workflow.pk}
        )
        self.assertContains(
            response,
            f'href="{log_url}"',
        )

    @patch("hitch.main.views.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.Codex")
    def test_okrs_page_without_selected_project_has_no_create_forms(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _setup_codex(mock_codex)
        Project.objects.create(name="Hitch", repo_path="/repo")

        response = self.client.get(reverse("okrs"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No active project selected.")
        self.assertNotContains(response, 'action="/okrs/objectives/"')
        self.assertNotContains(response, ">OKRs</a>")

    def test_create_objective_for_selected_project(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        self._select_project(project)

        response = self.client.post(
            reverse("create_objective"),
            data={"title": "Improve planning", "description": "Make plans concrete."},
        )

        self.assertEqual(response.status_code, 302)
        objective = Objective.objects.get()
        self.assertEqual(objective.project, project)
        self.assertEqual(objective.title, "Improve planning")
        self.assertEqual(objective.description, "Make plans concrete.")

    def test_create_key_result_for_selected_project_objective(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        objective = Objective.objects.create(project=project, title="Improve planning")
        self._select_project(project)

        response = self.client.post(
            reverse("create_key_result", kwargs={"objective_id": objective.pk}),
            data={
                "title": "Draft OKR plan",
                "description": "The plan captures scope.",
                "work_instructions": "Use concise task prompts.",
            },
        )

        self.assertEqual(response.status_code, 302)
        key_result = KeyResult.objects.get()
        self.assertEqual(key_result.objective, objective)
        self.assertEqual(key_result.title, "Draft OKR plan")
        self.assertEqual(key_result.description, "The plan captures scope.")
        self.assertEqual(key_result.work_instructions, "Use concise task prompts.")

    def test_create_key_result_rejects_objective_from_another_project(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        other = Project.objects.create(name="Other", repo_path="/other")
        objective = Objective.objects.create(project=other, title="Other objective")
        self._select_project(project)

        response = self.client.post(
            reverse("create_key_result", kwargs={"objective_id": objective.pk}),
            data={"title": "Should fail"},
        )

        self.assertContains(response, "objective is required", status_code=400)
        self.assertEqual(KeyResult.objects.count(), 0)

    @patch("hitch.main.views.system_agents.start_okr_task_generation_workflow")
    def test_generate_key_result_tasks_starts_workflow(self, mock_start: MagicMock) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        objective = Objective.objects.create(project=project, title="Improve planning")
        key_result = KeyResult.objects.create(objective=objective, title="Draft plan")
        self._select_project(project)

        response = self.client.post(
            reverse(
                "generate_key_result_tasks",
                kwargs={"key_result_id": key_result.pk},
            )
        )

        self.assertEqual(response.status_code, 302)
        mock_start.assert_called_once()
        self.assertEqual(mock_start.call_args.kwargs["key_result"], key_result)

    @patch("hitch.main.views.system_agents.start_okr_task_generation_workflow")
    def test_generate_key_result_tasks_rejects_other_project(
        self, mock_start: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        other = Project.objects.create(name="Other", repo_path="/other")
        objective = Objective.objects.create(project=other, title="Hidden")
        key_result = KeyResult.objects.create(objective=objective, title="Hidden KR")
        self._select_project(project)

        response = self.client.post(
            reverse(
                "generate_key_result_tasks",
                kwargs={"key_result_id": key_result.pk},
            )
        )

        self.assertContains(response, "key result is required", status_code=400)
        mock_start.assert_not_called()

    @patch("hitch.main.views.system_agents.start_okr_task_generation_workflow")
    def test_generate_key_result_tasks_rejects_without_active_project(
        self, mock_start: MagicMock
    ) -> None:
        response = self.client.post(
            reverse("generate_key_result_tasks", kwargs={"key_result_id": 1})
        )

        self.assertContains(response, "active project is required", status_code=400)
        mock_start.assert_not_called()

    @patch("hitch.main.views.system_agents.start_okr_task_generation_workflow")
    def test_generate_key_result_tasks_rejects_out_of_range_id(
        self, mock_start: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        self._select_project(project)

        response = self.client.post(
            reverse(
                "generate_key_result_tasks",
                kwargs={"key_result_id": views._MAX_BIGAUTOFIELD + 1},
            )
        )

        self.assertContains(response, "key result is required", status_code=400)
        mock_start.assert_not_called()

    @patch("hitch.main.views.system_agents.start_okr_task_generation_workflow")
    def test_generate_key_result_tasks_handles_stale_key_result(
        self, mock_start: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        objective = Objective.objects.create(project=project, title="Improve planning")
        key_result = KeyResult.objects.create(objective=objective, title="Draft plan")
        self._select_project(project)
        mock_start.side_effect = KeyResult.DoesNotExist

        response = self.client.post(
            reverse(
                "generate_key_result_tasks",
                kwargs={"key_result_id": key_result.pk},
            )
        )

        self.assertContains(response, "key result is required", status_code=400)

    def test_update_proposed_task_outcome(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        objective = Objective.objects.create(project=project, title="Improve planning")
        key_result = KeyResult.objects.create(objective=objective, title="Draft plan")
        task = ProposedTask.objects.create(key_result=key_result, title="Add model")
        self._select_project(project)

        response = self.client.post(
            reverse("update_proposed_task_outcome", kwargs={"task_id": task.pk}),
            data={
                "outcome_status": ProposedTask.OUTCOME_COMPLETED,
                "outcome_notes": "Worked well.",
            },
        )

        self.assertEqual(response.status_code, 302)
        task.refresh_from_db()
        self.assertEqual(task.outcome_status, ProposedTask.OUTCOME_COMPLETED)
        self.assertEqual(task.outcome_notes, "Worked well.")

    def test_update_proposed_task_outcome_saves_reject_reason(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        objective = Objective.objects.create(project=project, title="Improve planning")
        key_result = KeyResult.objects.create(objective=objective, title="Draft plan")
        task = ProposedTask.objects.create(key_result=key_result, title="Add model")
        self._select_project(project)

        response = self.client.post(
            reverse("update_proposed_task_outcome", kwargs={"task_id": task.pk}),
            data={
                "outcome_status": ProposedTask.OUTCOME_REJECTED,
                "reason": "Too broad.",
            },
        )

        self.assertEqual(response.status_code, 302)
        task.refresh_from_db()
        self.assertEqual(task.outcome_status, ProposedTask.OUTCOME_REJECTED)
        self.assertEqual(task.outcome_notes, "Too broad.")

    def test_update_proposed_task_outcome_requires_reject_reason(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        objective = Objective.objects.create(project=project, title="Improve planning")
        key_result = KeyResult.objects.create(objective=objective, title="Draft plan")
        task = ProposedTask.objects.create(key_result=key_result, title="Add model")
        self._select_project(project)

        response = self.client.post(
            reverse("update_proposed_task_outcome", kwargs={"task_id": task.pk}),
            data={"outcome_status": ProposedTask.OUTCOME_REJECTED},
        )

        self.assertContains(response, "reason is required", status_code=400)

    def test_update_proposed_task_outcome_rejects_invalid_status(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        objective = Objective.objects.create(project=project, title="Improve planning")
        key_result = KeyResult.objects.create(objective=objective, title="Draft plan")
        task = ProposedTask.objects.create(key_result=key_result, title="Add model")
        self._select_project(project)

        response = self.client.post(
            reverse("update_proposed_task_outcome", kwargs={"task_id": task.pk}),
            data={"outcome_status": "invalid"},
        )

        self.assertContains(response, "outcome status is invalid", status_code=400)

    def test_update_proposed_task_outcome_rejects_without_active_project(self) -> None:
        response = self.client.post(
            reverse("update_proposed_task_outcome", kwargs={"task_id": 1}),
            data={"outcome_status": ProposedTask.OUTCOME_ACCEPTED},
        )

        self.assertContains(response, "active project is required", status_code=400)

    def test_update_proposed_task_outcome_rejects_out_of_range_id(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        self._select_project(project)

        response = self.client.post(
            reverse(
                "update_proposed_task_outcome",
                kwargs={"task_id": views._MAX_BIGAUTOFIELD + 1},
            ),
            data={"outcome_status": ProposedTask.OUTCOME_ACCEPTED},
        )

        self.assertContains(response, "proposed task is required", status_code=400)

    def test_update_proposed_task_outcome_rejects_other_project_task(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        other = Project.objects.create(name="Other", repo_path="/other")
        objective = Objective.objects.create(project=other, title="Hidden")
        key_result = KeyResult.objects.create(objective=objective, title="Hidden KR")
        task = ProposedTask.objects.create(key_result=key_result, title="Hidden task")
        self._select_project(project)

        response = self.client.post(
            reverse("update_proposed_task_outcome", kwargs={"task_id": task.pk}),
            data={"outcome_status": ProposedTask.OUTCOME_ACCEPTED},
        )

        self.assertContains(response, "proposed task is required", status_code=400)

    def test_marks_associated_proposed_task_when_pr_opens(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        objective = Objective.objects.create(project=project, title="Improve planning")
        key_result = KeyResult.objects.create(objective=objective, title="Draft plan")
        session = SessionMetadata.objects.create(thread_id="thread-xyz", cwd="/repo")
        task = ProposedTask.objects.create(
            key_result=key_result,
            title="Add model",
            outcome_status=ProposedTask.OUTCOME_ACCEPTED,
            session=session,
        )
        completed = ProposedTask.objects.create(
            key_result=key_result,
            title="Completed task",
            outcome_status=ProposedTask.OUTCOME_COMPLETED,
            session=session,
        )

        views._mark_proposed_tasks_pr_opened(
            "thread-xyz", "https://github.com/cberner/hitch/pull/94"
        )

        task.refresh_from_db()
        completed.refresh_from_db()
        self.assertEqual(task.outcome_status, ProposedTask.OUTCOME_PR_OPENED)
        self.assertEqual(task.pr_url, "https://github.com/cberner/hitch/pull/94")
        self.assertEqual(completed.outcome_status, ProposedTask.OUTCOME_COMPLETED)
        self.assertEqual(completed.pr_url, "")

    def test_rejects_invalid_okr_posts(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        objective = Objective.objects.create(project=project, title="Objective")
        self._select_project(project)

        for url, data, message in (
            (reverse("create_objective"), {"title": ""}, "title is required"),
            (
                reverse("create_objective"),
                {"title": "x" * 201},
                "title is too long",
            ),
            (
                reverse("create_key_result", kwargs={"objective_id": objective.pk}),
                {"title": ""},
                "title is required",
            ),
        ):
            with self.subTest(message=message):
                response = self.client.post(url, data=data)
                self.assertContains(response, message, status_code=400)

    def test_rejects_create_without_active_project(self) -> None:
        response = self.client.post(
            reverse("create_objective"),
            data={"title": "Improve planning"},
        )

        self.assertContains(response, "active project is required", status_code=400)

    def test_rejects_key_result_create_without_active_project(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        objective = Objective.objects.create(project=project, title="Objective")

        response = self.client.post(
            reverse("create_key_result", kwargs={"objective_id": objective.pk}),
            data={"title": "Improve planning"},
        )

        self.assertContains(response, "active project is required", status_code=400)

    def test_rejects_out_of_range_objective_id_before_orm_lookup(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        self._select_project(project)

        response = self.client.post(
            reverse(
                "create_key_result",
                kwargs={"objective_id": views._MAX_BIGAUTOFIELD + 1},
            ),
            data={"title": "Should fail"},
        )

        self.assertContains(response, "objective is required", status_code=400)


class NewSessionViewTests(TestCase):
    REPO = "/home/user/proj"

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_spawns_worker_and_redirects(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        # No models from Codex → reconcile is a no-op; spawn sees None/None.
        _setup_codex(mock_codex, models=[])

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "Refactor the login flow", "cwd": self.REPO},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            response.headers["Location"],
            reverse("session", kwargs={"session_id": "thread-xyz"}),
        )
        mock_spawn.assert_called_once_with(
            cwd=self.REPO,
            prompt="Refactor the login flow",
            developer_instructions=None,
            model=None,
            reasoning_effort=None,
            sandbox_policy=None,
            approval_mode="auto_review",
        )

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_accepts_and_associates_proposed_task(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path=self.REPO)
        objective = Objective.objects.create(project=project, title="Improve planning")
        key_result = KeyResult.objects.create(objective=objective, title="Draft plan")
        task = ProposedTask.objects.create(key_result=key_result, title="Add model")
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        _setup_codex(mock_codex, models=[])

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": "Do this ProposedTask.",
                "cwd": self.REPO,
                "proposed_task": str(task.pk),
            },
        )

        self.assertEqual(response.status_code, 302)
        task.refresh_from_db()
        metadata = SessionMetadata.objects.get(thread_id="thread-xyz")
        self.assertEqual(task.outcome_status, ProposedTask.OUTCOME_ACCEPTED)
        self.assertEqual(task.session, metadata)

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_remembers_selected_repo_in_cookie(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        other_repo = "/home/user/other"
        mock_discover.return_value = [Path(self.REPO), Path(other_repo)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        _setup_codex(mock_codex, models=[])

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "do thing", "cwd": other_repo},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(_cookie_value(response, _LAST_SELECTED_REPO_COOKIE), other_repo)

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_remembers_selected_repo_in_account_settings(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        user_model = get_user_model()
        user = user_model.objects.create_user("dev@example.com", password="StrongPass123!")
        self.client.force_login(user)
        other_repo = "/home/user/other"
        mock_discover.return_value = [Path(self.REPO), Path(other_repo)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        _setup_codex(mock_codex, models=[])

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "do thing", "cwd": other_repo},
        )

        self.assertEqual(response.status_code, 302)
        settings = UserSettings.objects.get(user=user)
        self.assertEqual(settings.last_selected_repo, other_repo)
        self.assertEqual(_cookie_value(response, _LAST_SELECTED_REPO_COOKIE), other_repo)

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_auto_pr_override_marks_new_session_and_spawn(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        _setup_codex(mock_codex, models=[])

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "do thing", "cwd": self.REPO, "auto_pr": "true"},
        )

        self.assertEqual(response.status_code, 302)
        metadata = SessionMetadata.objects.get(thread_id="thread-xyz")
        self.assertTrue(metadata.auto_pr_enabled)
        mock_spawn.assert_called_once_with(
            cwd=self.REPO,
            prompt="do thing",
            developer_instructions=None,
            model=None,
            reasoning_effort=None,
            sandbox_policy=None,
            approval_mode="auto_review",
            auto_pr_enabled=True,
        )

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_auto_pr_override_can_disable_global_setting(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        _setup_codex(mock_codex, models=[])
        _seed_cookies(self.client, **{_AUTO_PR_COOKIE: "true"})

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "do thing", "cwd": self.REPO, "auto_pr": "false"},
        )

        self.assertEqual(response.status_code, 302)
        metadata = SessionMetadata.objects.get(thread_id="thread-xyz")
        self.assertFalse(metadata.auto_pr_enabled)
        mock_spawn.assert_called_once_with(
            cwd=self.REPO,
            prompt="do thing",
            developer_instructions=None,
            model=None,
            reasoning_effort=None,
            sandbox_policy=None,
            approval_mode="auto_review",
        )

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_project_auto_pr_on_sets_new_session_default(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        project = Project.objects.create(
            name="Hitch",
            repo_path=self.REPO,
            auto_pr_mode=Project.AUTO_PR_ON,
        )
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        _setup_codex(mock_codex, models=[])

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "do thing", "project": str(project.pk)},
        )

        self.assertEqual(response.status_code, 302)
        metadata = SessionMetadata.objects.get(thread_id="thread-xyz")
        self.assertTrue(metadata.auto_pr_enabled)
        mock_spawn.assert_called_once_with(
            cwd=self.REPO,
            prompt="do thing",
            developer_instructions=None,
            model=None,
            reasoning_effort=None,
            sandbox_policy=None,
            approval_mode="auto_review",
            auto_pr_enabled=True,
        )

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_project_auto_pr_off_overrides_global_new_session_default(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        project = Project.objects.create(
            name="Hitch",
            repo_path=self.REPO,
            auto_pr_mode=Project.AUTO_PR_OFF,
        )
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        _setup_codex(mock_codex, models=[])
        _seed_cookies(self.client, **{_AUTO_PR_COOKIE: "true"})

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "do thing", "project": str(project.pk)},
        )

        self.assertEqual(response.status_code, 302)
        metadata = SessionMetadata.objects.get(thread_id="thread-xyz")
        self.assertFalse(metadata.auto_pr_enabled)
        mock_spawn.assert_called_once_with(
            cwd=self.REPO,
            prompt="do thing",
            developer_instructions=None,
            model=None,
            reasoning_effort=None,
            sandbox_policy=None,
            approval_mode="auto_review",
        )

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_auto_pr_override_can_disable_project_setting(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        project = Project.objects.create(
            name="Hitch",
            repo_path=self.REPO,
            auto_pr_mode=Project.AUTO_PR_ON,
        )
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        _setup_codex(mock_codex, models=[])

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": "do thing",
                "project": str(project.pk),
                "auto_pr": "false",
            },
        )

        self.assertEqual(response.status_code, 302)
        metadata = SessionMetadata.objects.get(thread_id="thread-xyz")
        self.assertFalse(metadata.auto_pr_enabled)
        mock_spawn.assert_called_once_with(
            cwd=self.REPO,
            prompt="do thing",
            developer_instructions=None,
            model=None,
            reasoning_effort=None,
            sandbox_policy=None,
            approval_mode="auto_review",
        )

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_assigns_new_session_to_selected_project(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path=self.REPO)
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        _setup_codex(mock_codex, models=[])
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "do thing", "cwd": self.REPO},
        )

        self.assertEqual(response.status_code, 302)
        metadata = SessionMetadata.objects.get(thread_id="thread-xyz")
        self.assertEqual(metadata.cwd, self.REPO)
        self.assertEqual(metadata.project, project)

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_project_comes_from_posted_repository(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        repo_a = self.REPO
        repo_b = "/home/user/other"
        project_a = Project.objects.create(name="Project A", repo_path=repo_a)
        project_b = Project.objects.create(name="Project B", repo_path=repo_b)
        mock_discover.return_value = [Path(repo_a), Path(repo_b)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        _setup_codex(mock_codex, models=[])
        _seed_cookies(self.client, hitch_selected_project_id=str(project_a.pk))

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "do thing", "cwd": repo_b},
        )

        self.assertEqual(response.status_code, 302)
        metadata = SessionMetadata.objects.get(thread_id="thread-xyz")
        self.assertEqual(metadata.cwd, repo_b)
        self.assertEqual(metadata.project, project_b)

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_project_comes_from_posted_project(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        repo_b = "/home/user/other"
        project = Project.objects.create(name="Project B", repo_path=repo_b)
        mock_discover.return_value = [Path(self.REPO), Path(repo_b)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        _setup_codex(mock_codex, models=[])

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "do thing", "project": str(project.pk)},
        )

        self.assertEqual(response.status_code, 302)
        mock_spawn.assert_called_once_with(
            cwd=repo_b,
            prompt="do thing",
            developer_instructions=None,
            model=None,
            reasoning_effort=None,
            sandbox_policy=None,
            approval_mode="auto_review",
        )
        metadata = SessionMetadata.objects.get(thread_id="thread-xyz")
        self.assertEqual(metadata.cwd, repo_b)
        self.assertEqual(metadata.project, project)
        self.assertFalse(metadata.project_cleared)

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_bare_repo_does_not_set_matching_project(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        Project.objects.create(name="Hitch", repo_path=self.REPO)
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        _setup_codex(mock_codex, models=[])

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": "do thing",
                "project": views._BARE_REPO_PROJECT_VALUE,
                "cwd": self.REPO,
            },
        )

        self.assertEqual(response.status_code, 302)
        metadata = SessionMetadata.objects.get(thread_id="thread-xyz")
        self.assertEqual(metadata.cwd, self.REPO)
        self.assertIsNone(metadata.project)
        self.assertTrue(metadata.project_cleared)

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_in_unprojected_repo_ignores_selected_project(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        repo_b = "/home/user/other"
        project = Project.objects.create(name="Hitch", repo_path=self.REPO)
        mock_discover.return_value = [Path(self.REPO), Path(repo_b)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        _setup_codex(mock_codex, models=[])
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "do thing", "cwd": repo_b},
        )

        self.assertEqual(response.status_code, 302)
        self.assertIsNone(SessionMetadata.objects.get(thread_id="thread-xyz").project)

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_forwards_extra_system_prompt_cookie_to_spawn(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        _setup_codex(mock_codex, models=[])
        _seed_cookies(
            self.client,
            **{
                _EXTRA_SYSTEM_PROMPT_COOKIE: _encode_extra_system_prompt(
                    "  Always run focused tests.  "
                )
            },
        )

        self.client.post(
            reverse("new_session"),
            data={"prompt": "do thing", "cwd": self.REPO},
        )

        mock_spawn.assert_called_once_with(
            cwd=self.REPO,
            prompt="do thing",
            developer_instructions="Always run focused tests.",
            model=None,
            reasoning_effort=None,
            sandbox_policy=None,
            approval_mode="auto_review",
        )

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_forwards_cookie_settings_to_spawn(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        """Cookie-driven model/effort/sandbox flow into ``spawn_new_session``."""
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        _setup_codex(mock_codex, models=[_make_model("gpt-5", is_default=True)])
        _seed_cookies(
            self.client,
            hitch_model="gpt-5",
            hitch_reasoning_effort="high",
            hitch_sandbox_policy="workspaceWrite",
        )

        self.client.post(
            reverse("new_session"),
            data={"prompt": "do thing", "cwd": self.REPO},
        )

        mock_spawn.assert_called_once_with(
            cwd=self.REPO,
            prompt="do thing",
            developer_instructions=None,
            model="gpt-5",
            reasoning_effort="high",
            sandbox_policy="workspaceWrite",
            approval_mode="auto_review",
        )

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_forwards_memories_cookie_to_new_session_spawn(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        _setup_codex(mock_codex, models=[])
        _seed_cookies(self.client, **{_ENABLE_MEMORIES_COOKIE: "true"})

        self.client.post(
            reverse("new_session"),
            data={"prompt": "do thing", "cwd": self.REPO},
        )

        mock_spawn.assert_called_once_with(
            cwd=self.REPO,
            prompt="do thing",
            developer_instructions=None,
            model=None,
            reasoning_effort=None,
            sandbox_policy=None,
            approval_mode="auto_review",
            enable_memories=True,
        )

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_forwards_approval_mode_cookie_to_spawn(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        """An explicit approval-mode cookie must reach the spawn call; the
        SDK default is the safe fallback otherwise, but a user who picked a
        stricter/user-prompting mode expects it to take effect on session
        start."""
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        _setup_codex(mock_codex, models=[])

        for mode in ("deny_all", "prompt_user"):
            with self.subTest(mode=mode):
                mock_spawn.reset_mock()
                _seed_cookies(self.client, hitch_approval_mode=mode)

                self.client.post(
                    reverse("new_session"),
                    data={"prompt": "do thing", "cwd": self.REPO},
                )

                mock_spawn.assert_called_once_with(
                    cwd=self.REPO,
                    prompt="do thing",
                    developer_instructions=None,
                    model=None,
                    reasoning_effort=None,
                    sandbox_policy=None,
                    approval_mode=mode,
                )

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_forwards_plan_mode_for_new_session(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        _setup_codex(mock_codex, models=[_make_model("gpt-default", is_default=True)])

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": "make a migration plan",
                "cwd": self.REPO,
                "plan_mode": "true",
            },
        )

        self.assertEqual(response.status_code, 302)
        mock_spawn.assert_called_once_with(
            cwd=self.REPO,
            prompt="make a migration plan",
            developer_instructions=None,
            model="gpt-default",
            reasoning_effort="medium",
            sandbox_policy=None,
            approval_mode="auto_review",
            plan_mode=True,
        )

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_plan_slash_command_starts_new_session_in_plan_mode(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        _setup_codex(mock_codex, models=[_make_model("gpt-5", is_default=True)])
        _seed_cookies(self.client, **{_MODEL_COOKIE: "gpt-5"})

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "/plan make a migration plan", "cwd": self.REPO},
        )

        self.assertEqual(response.status_code, 302)
        mock_spawn.assert_called_once_with(
            cwd=self.REPO,
            prompt="make a migration plan",
            developer_instructions=None,
            model="gpt-5",
            reasoning_effort=None,
            sandbox_policy=None,
            approval_mode="auto_review",
            plan_mode=True,
        )

    @patch("hitch.main.views.codex_pool.spawn_new_session")
    def test_plan_slash_command_without_prompt_is_rejected(
        self, mock_spawn: MagicMock
    ) -> None:
        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "/plan", "cwd": self.REPO},
        )

        self.assertEqual(response.status_code, 400)
        self.assertContains(response, "prompt is required", status_code=400)
        mock_spawn.assert_not_called()

    @patch("hitch.main.views.system_agents.start_pr_qa_workflow")
    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.create_session_thread")
    @patch("hitch.main.views.discover_repos")
    def test_pr_slash_command_starts_new_session_with_qa_workflow(
        self,
        mock_discover: MagicMock,
        mock_create_thread: MagicMock,
        mock_codex: MagicMock,
        mock_start_workflow: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        mock_create_thread.return_value = "thread-xyz"
        _setup_codex(mock_codex, models=[_make_model("gpt-5.4", is_default=True)])
        _seed_cookies(
            self.client,
            hitch_model="gpt-5.4",
            hitch_reasoning_effort="high",
            hitch_extra_system_prompt=_encode_extra_system_prompt(
                "Use repo conventions."
            ),
        )

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "/PR", "cwd": self.REPO, "plan_mode": "true"},
        )

        self.assertEqual(response.status_code, 302)
        mock_create_thread.assert_called_once_with(
            cwd=self.REPO,
            name=_PR_PROMPT,
            developer_instructions="Use repo conventions.",
            model="gpt-5.4",
            enable_memories=False,
        )
        mock_start_workflow.assert_called_once_with(
            main_thread_id="thread-xyz",
            cwd=self.REPO,
            sandbox_policy=None,
            approval_mode="auto_review",
            model="gpt-5.4",
            reasoning_effort="high",
            developer_instructions="Use repo conventions.",
            enable_memories=False,
            initial_user_message_index=0,
        )

    @patch("hitch.main.views.system_agents.start_pr_qa_workflow")
    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.create_session_thread")
    @patch("hitch.main.views.discover_repos")
    def test_qa_slash_command_starts_new_session_without_pr_prompt(
        self,
        mock_discover: MagicMock,
        mock_create_thread: MagicMock,
        mock_codex: MagicMock,
        mock_start_workflow: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        mock_create_thread.return_value = "thread-xyz"
        _setup_codex(mock_codex, models=[_make_model("gpt-5.4", is_default=True)])

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "/QA", "cwd": self.REPO, "plan_mode": "true"},
        )

        self.assertEqual(response.status_code, 302)
        mock_create_thread.assert_called_once_with(
            cwd=self.REPO,
            name=_QA_PROMPT,
            developer_instructions=None,
            model="gpt-5.4",
            enable_memories=False,
        )
        mock_start_workflow.assert_called_once_with(
            main_thread_id="thread-xyz",
            cwd=self.REPO,
            sandbox_policy=None,
            approval_mode="auto_review",
            model="gpt-5.4",
            reasoning_effort="medium",
            developer_instructions=None,
            enable_memories=False,
            initial_user_message_index=0,
            open_pr_on_lgtm=False,
        )

    @patch("hitch.main.views.system_agents.start_pr_qa_workflow")
    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.create_session_thread")
    @patch("hitch.main.views.create_worktree_for_session")
    @patch("hitch.main.views.discover_repos")
    def test_pr_slash_command_uses_selected_repo_when_worktrees_are_enabled(
        self,
        mock_discover: MagicMock,
        mock_create_worktree: MagicMock,
        mock_create_thread: MagicMock,
        mock_codex: MagicMock,
        mock_start_workflow: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        mock_create_thread.return_value = "thread-xyz"
        _setup_codex(mock_codex, models=[])
        _seed_cookies(self.client, **{_USE_WORKTREES_COOKIE: "true"})

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "/pr", "cwd": self.REPO},
        )

        self.assertEqual(response.status_code, 302)
        mock_create_worktree.assert_not_called()
        mock_create_thread.assert_called_once_with(
            cwd=self.REPO,
            name=_PR_PROMPT,
            developer_instructions=None,
            model=None,
            enable_memories=False,
        )
        mock_start_workflow.assert_called_once_with(
            main_thread_id="thread-xyz",
            cwd=self.REPO,
            sandbox_policy=None,
            approval_mode="auto_review",
            model=None,
            reasoning_effort=None,
            developer_instructions=None,
            enable_memories=False,
            initial_user_message_index=0,
        )

    @patch("hitch.main.views.system_agents.start_pr_qa_workflow")
    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.create_session_thread")
    @patch("hitch.main.views.discover_repos")
    def test_pr_new_session_project_comes_from_posted_repository(
        self,
        mock_discover: MagicMock,
        mock_create_thread: MagicMock,
        mock_codex: MagicMock,
        mock_start_workflow: MagicMock,
    ) -> None:
        repo_b = "/home/user/other"
        project_a = Project.objects.create(name="Project A", repo_path=self.REPO)
        project_b = Project.objects.create(name="Project B", repo_path=repo_b)
        mock_discover.return_value = [Path(self.REPO), Path(repo_b)]
        mock_create_thread.return_value = "thread-xyz"
        _setup_codex(mock_codex, models=[])
        _seed_cookies(self.client, hitch_selected_project_id=str(project_a.pk))

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "/pr", "cwd": repo_b},
        )

        self.assertEqual(response.status_code, 302)
        metadata = SessionMetadata.objects.get(thread_id="thread-xyz")
        self.assertEqual(metadata.cwd, repo_b)
        self.assertEqual(metadata.project, project_b)
        mock_start_workflow.assert_called_once()

    @patch("hitch.main.views.system_agents.start_pr_qa_workflow")
    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.create_session_thread")
    @patch("hitch.main.views.discover_repos")
    def test_pr_new_session_bare_repo_does_not_set_matching_project(
        self,
        mock_discover: MagicMock,
        mock_create_thread: MagicMock,
        mock_codex: MagicMock,
        mock_start_workflow: MagicMock,
    ) -> None:
        Project.objects.create(name="Hitch", repo_path=self.REPO)
        mock_discover.return_value = [Path(self.REPO)]
        mock_create_thread.return_value = "thread-xyz"
        _setup_codex(mock_codex, models=[])

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": "/pr",
                "project": views._BARE_REPO_PROJECT_VALUE,
                "cwd": self.REPO,
            },
        )

        self.assertEqual(response.status_code, 302)
        metadata = SessionMetadata.objects.get(thread_id="thread-xyz")
        self.assertEqual(metadata.cwd, self.REPO)
        self.assertIsNone(metadata.project)
        self.assertTrue(metadata.project_cleared)
        mock_start_workflow.assert_called_once()

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_plan_mode_returns_bad_request_when_model_cannot_be_resolved(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        _setup_codex(mock_codex, models=[])

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": "make a migration plan",
                "cwd": self.REPO,
                "plan_mode": "true",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertContains(response, "plan mode requires a model", status_code=400)
        mock_spawn.assert_not_called()

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_reconciles_stale_model_before_spawning(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        """A long-lived tab can POST with a session that names a model the
        running Codex no longer offers; reconcile catches it so
        ``thread_start(model=...)`` doesn't get a stale id."""
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        _setup_codex(mock_codex, models=[_make_model("gpt-5", is_default=True)])
        _seed_cookies(
            self.client, hitch_model="ancient-model", hitch_reasoning_effort="low"
        )

        self.client.post(
            reverse("new_session"),
            data={"prompt": "do thing", "cwd": self.REPO},
        )

        mock_spawn.assert_called_once_with(
            cwd=self.REPO,
            prompt="do thing",
            developer_instructions=None,
            model="gpt-5",
            reasoning_effort="medium",
            sandbox_policy=None,
            approval_mode="auto_review",
        )

    @patch("hitch.main.views.create_worktree_for_session")
    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_uses_managed_worktree_when_setting_enabled(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
        mock_create_worktree: MagicMock,
    ) -> None:
        worktree = Path("/home/user/.hitch/worktrees/proj/20260516120000-abcdef12")
        mock_discover.return_value = [Path(self.REPO)]
        mock_create_worktree.return_value = ManagedWorktree(
            path=worktree,
            branch="hitch/proj/20260516120000-abcdef12",
            source_repo=Path(self.REPO),
        )
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        _setup_codex(mock_codex, models=[])
        _seed_cookies(self.client, **{_USE_WORKTREES_COOKIE: "true"})

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "do thing", "cwd": self.REPO},
        )

        self.assertEqual(response.status_code, 302)
        mock_create_worktree.assert_called_once_with(self.REPO)
        mock_spawn.assert_called_once_with(
            cwd=str(worktree),
            prompt="do thing",
            developer_instructions=None,
            model=None,
            reasoning_effort=None,
            sandbox_policy=None,
            approval_mode="auto_review",
        )

    @patch("hitch.main.views.cleanup_worktree")
    @patch("hitch.main.views.create_worktree_for_session")
    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_cleans_up_managed_worktree_when_spawn_fails(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
        mock_create_worktree: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        worktree = ManagedWorktree(
            path=Path("/home/user/.hitch/worktrees/proj/20260516120000-abcdef12"),
            branch="hitch/proj/20260516120000-abcdef12",
            source_repo=Path(self.REPO),
        )
        mock_discover.return_value = [Path(self.REPO)]
        mock_create_worktree.return_value = worktree
        mock_spawn.side_effect = RuntimeError("spawn failed")
        _setup_codex(mock_codex, models=[])
        _seed_cookies(self.client, **{_USE_WORKTREES_COOKIE: "true"})

        with self.assertRaisesRegex(RuntimeError, "spawn failed"):
            self.client.post(
                reverse("new_session"),
                data={"prompt": "do thing", "cwd": self.REPO},
            )

        mock_cleanup.assert_called_once_with(worktree)

    @patch("hitch.main.views.cleanup_worktree")
    @patch("hitch.main.views.create_worktree_for_session")
    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_preserves_spawn_error_when_managed_worktree_cleanup_fails(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
        mock_create_worktree: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        worktree = ManagedWorktree(
            path=Path("/home/user/.hitch/worktrees/proj/20260516120000-abcdef12"),
            branch="hitch/proj/20260516120000-abcdef12",
            source_repo=Path(self.REPO),
        )
        mock_discover.return_value = [Path(self.REPO)]
        mock_create_worktree.return_value = worktree
        mock_spawn.side_effect = RuntimeError("spawn failed")
        mock_cleanup.side_effect = WorktreeCleanupError("cleanup failed")
        _setup_codex(mock_codex, models=[])
        _seed_cookies(self.client, **{_USE_WORKTREES_COOKIE: "true"})

        with (
            self.assertLogs("hitch.main.views", level="ERROR") as logs,
            self.assertRaisesRegex(RuntimeError, "spawn failed"),
        ):
            self.client.post(
                reverse("new_session"),
                data={"prompt": "do thing", "cwd": self.REPO},
            )

        mock_cleanup.assert_called_once_with(worktree)
        self.assertIn("failed to clean up managed worktree", logs.output[0])

    @patch("hitch.main.views.create_worktree_for_session")
    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_reports_worktree_creation_failure(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
        mock_create_worktree: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        mock_create_worktree.side_effect = WorktreeCreationError("boom")
        _setup_codex(mock_codex, models=[])
        _seed_cookies(self.client, **{_USE_WORKTREES_COOKIE: "true"})

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "do thing", "cwd": self.REPO},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.content, b"boom")
        mock_spawn.assert_not_called()

    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_rejects_invalid_input(
        self, mock_discover: MagicMock, mock_spawn: MagicMock
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]

        cases = [
            ({"prompt": "", "cwd": self.REPO}, "empty prompt"),
            ({"prompt": "hello", "cwd": ""}, "missing cwd"),
            ({"prompt": "hello", "cwd": "/etc"}, "cwd outside allowed list"),
        ]
        for data, label in cases:
            with self.subTest(label=label):
                mock_spawn.reset_mock()
                response = self.client.post(reverse("new_session"), data=data)
                self.assertEqual(response.status_code, 400)
                mock_spawn.assert_not_called()

    @patch("hitch.main.views.discover_repos")
    def test_rejects_get(self, mock_discover: MagicMock) -> None:
        mock_discover.return_value = []
        response = self.client.get(reverse("new_session"))
        self.assertEqual(response.status_code, 405)


class SendMessageViewTests(TestCase):
    def _patch_codex(
        self,
        mock_codex: MagicMock,
        *,
        cwd: object = "/repo",
        model: str | None = "gpt-5",
        reasoning_effort: str | None = None,
        models: list[Any] | None = None,
        path: str | None = None,
        turns: list[Any] | None = None,
    ) -> None:
        client = mock_codex.return_value.__enter__.return_value
        thread = SimpleNamespace(cwd=cwd, turns=turns or [])
        if path is not None:
            thread.path = path
        resumed = SimpleNamespace(thread=thread)
        if model is not None:
            resumed.model = model
        if reasoning_effort is not None:
            resumed.reasoning_effort = SimpleNamespace(value=reasoning_effort)
        client._client.thread_resume.return_value = resumed
        client.models.return_value.data = models or []

    def _make_rollout(self, lines: list[str]) -> Path:
        with tempfile.NamedTemporaryFile(
            prefix="rollout-",
            suffix=".jsonl",
            mode="w",
            delete=False,
        ) as fh:
            fh.write("\n".join(lines))
            if lines:
                fh.write("\n")
            path = Path(fh.name)
        self.addCleanup(path.unlink, missing_ok=True)
        return path

    def _make_pending_plan_rollout(
        self,
        plan: str = "# Pending Plan\n\nReady to implement.",
    ) -> Path:
        return self._make_rollout(
            [
                _rollout_line("turn_context", {"collaboration_mode": {"mode": "plan"}}),
                _rollout_line("event_msg", {"type": "user_message", "message": "Plan it"}),
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
        )

    def _make_resolved_plan_rollout(self) -> Path:
        return self._make_rollout(
            [
                _rollout_line("turn_context", {"collaboration_mode": {"mode": "plan"}}),
                _rollout_line("event_msg", {"type": "user_message", "message": "Plan it"}),
                _rollout_line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": (
                                    "<proposed_plan>\n# Plan\n\nImplement it.\n"
                                    "</proposed_plan>"
                                ),
                            }
                        ],
                        "phase": "final_answer",
                    },
                ),
                _rollout_line("turn_context", {"collaboration_mode": {"mode": "default"}}),
                _rollout_line(
                    "event_msg",
                    {"type": "user_message", "message": "Implement the plan."},
                ),
                _rollout_line(
                    "event_msg",
                    {
                        "type": "agent_message",
                        "message": "Implemented.",
                        "phase": "final_answer",
                    },
                ),
            ]
        )

    @patch("hitch.main.views.codex_pool.steer_instance")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_steers_posted_active_instance_without_spawning(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_steer: MagicMock,
    ) -> None:
        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "  also update docs  ", "active_instance": "42"},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            response.headers["Location"],
            reverse("session", kwargs={"session_id": "abc"}),
        )
        mock_steer.assert_called_once_with(
            42,
            expected_thread_id="abc",
            prompt="also update docs",
        )
        mock_spawn.assert_not_called()
        mock_codex.assert_not_called()

    @patch("hitch.main.views.codex_pool.steer_instance")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_steers_latest_active_when_form_has_no_instance(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_steer: MagicMock,
    ) -> None:
        instance = CodexInstance.objects.create(
            pid=123,
            thread_id="abc",
            cwd="/repo",
            prompt="already running",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_RUNNING,
        )

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "also lint"},
        )

        self.assertEqual(response.status_code, 302)
        mock_steer.assert_called_once_with(
            instance.pk,
            expected_thread_id="abc",
            prompt="also lint",
        )
        mock_spawn.assert_not_called()
        mock_codex.assert_not_called()

    @patch("hitch.main.views.codex_pool.steer_instance")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_rejects_invalid_active_instance(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_steer: MagicMock,
    ) -> None:
        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "also lint", "active_instance": "not-a-number"},
        )

        self.assertEqual(response.status_code, 400)
        mock_steer.assert_not_called()
        mock_spawn.assert_not_called()
        mock_codex.assert_not_called()

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.steer_instance", return_value=None)
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_stale_finished_active_instance_falls_back_to_spawn(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_steer: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex)
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "follow up", "active_instance": "42"},
        )

        self.assertEqual(response.status_code, 302)
        mock_steer.assert_called_once_with(
            42,
            expected_thread_id="abc",
            prompt="follow up",
        )
        mock_spawn.assert_called_once_with(
            thread_id="abc",
            cwd="/repo",
            prompt="follow up",
            sandbox_policy=None,
            approval_mode="auto_review",
        )

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.steer_instance", return_value=None)
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_stale_active_instance_recomputes_plan_mode_before_spawn(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_steer: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        rollout_path = self._make_resolved_plan_rollout()
        self._patch_codex(mock_codex, path=str(rollout_path))
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={
                "prompt": "follow up",
                "active_instance": "42",
                "plan_mode": "true",
                "default_plan_mode": "true",
            },
        )

        self.assertEqual(response.status_code, 302)
        mock_steer.assert_called_once_with(
            42,
            expected_thread_id="abc",
            prompt="follow up",
        )
        mock_spawn.assert_called_once_with(
            thread_id="abc",
            cwd="/repo",
            prompt="follow up",
            sandbox_policy=None,
            approval_mode="auto_review",
        )

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.steer_instance", return_value=None)
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_explicit_plan_toggle_survives_stale_active_fallback(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_steer: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        rollout_path = self._make_resolved_plan_rollout()
        self._patch_codex(mock_codex, path=str(rollout_path))
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={
                "prompt": "make another plan",
                "active_instance": "42",
                "plan_mode": "true",
                "plan_mode_explicit": "true",
            },
        )

        self.assertEqual(response.status_code, 302)
        mock_steer.assert_called_once_with(
            42,
            expected_thread_id="abc",
            prompt="make another plan",
        )
        mock_spawn.assert_called_once_with(
            thread_id="abc",
            cwd="/repo",
            prompt="make another plan",
            sandbox_policy=None,
            approval_mode="auto_review",
            model="gpt-5",
            plan_mode=True,
        )

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_pending_plan_default_recomputes_before_spawn(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        rollout_path = self._make_resolved_plan_rollout()
        self._patch_codex(mock_codex, path=str(rollout_path))
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={
                "prompt": "follow up",
                "plan_mode": "true",
                "default_plan_mode": "true",
            },
        )

        self.assertEqual(response.status_code, 302)
        mock_spawn.assert_called_once_with(
            thread_id="abc",
            cwd="/repo",
            prompt="follow up",
            sandbox_policy=None,
            approval_mode="auto_review",
        )

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.steer_instance", return_value=None)
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_stale_active_instance_failed_steer_falls_back_to_spawn(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_steer: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex)
        mock_discover.return_value = [Path("/repo")]
        CodexInstance.objects.create(
            pid=123,
            thread_id="abc",
            cwd="/repo",
            prompt="newer work",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_RUNNING,
        )

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "follow up", "active_instance": "42"},
        )

        self.assertEqual(response.status_code, 302)
        mock_steer.assert_called_once_with(
            42,
            expected_thread_id="abc",
            prompt="follow up",
        )
        mock_spawn.assert_called_once_with(
            thread_id="abc",
            cwd="/repo",
            prompt="follow up",
            sandbox_policy=None,
            approval_mode="auto_review",
        )

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.steer_instance", return_value=None)
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_latest_active_failed_steer_falls_back_to_spawn(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_steer: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex)
        mock_discover.return_value = [Path("/repo")]
        instance = CodexInstance.objects.create(
            pid=0,
            thread_id="abc",
            cwd="/repo",
            prompt="launching",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_STARTING,
        )

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "also lint"},
        )

        self.assertEqual(response.status_code, 302)
        mock_steer.assert_called_once_with(
            instance.pk,
            expected_thread_id="abc",
            prompt="also lint",
        )
        mock_spawn.assert_called_once_with(
            thread_id="abc",
            cwd="/repo",
            prompt="also lint",
            sandbox_policy=None,
            approval_mode="auto_review",
        )

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_spawns_turn_and_redirects(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex)
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "  follow-up question  "},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            response.headers["Location"],
            reverse("session", kwargs={"session_id": "abc"}),
        )
        # Whitespace is trimmed before forwarding.
        mock_spawn.assert_called_once_with(
            thread_id="abc",
            cwd="/repo",
            prompt="follow-up question",
            sandbox_policy=None,
            approval_mode="auto_review",
        )

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_unwraps_pydantic_rootmodel_cwd(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        # The SDK's Thread.cwd is an AbsolutePathBuf (pydantic RootModel[str]),
        # not a bare str, so the view has to unwrap ``.root``.
        self._patch_codex(mock_codex, cwd=SimpleNamespace(root="/repo"))
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "hi"},
        )

        self.assertEqual(response.status_code, 302)
        mock_spawn.assert_called_once_with(
            thread_id="abc",
            cwd="/repo",
            prompt="hi",
            sandbox_policy=None,
            approval_mode="auto_review",
        )

    @patch("hitch.main.views.discover_managed_worktrees")
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_allows_follow_up_turns_in_managed_worktrees(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
        mock_managed_worktrees: MagicMock,
    ) -> None:
        worktree = "/home/user/.hitch/worktrees/proj/20260516120000-abcdef12"
        self._patch_codex(mock_codex, cwd=worktree)
        mock_discover.return_value = [Path("/repo")]
        mock_managed_worktrees.return_value = [Path(worktree)]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "hi"},
        )

        self.assertEqual(response.status_code, 302)
        mock_spawn.assert_called_once_with(
            thread_id="abc",
            cwd=worktree,
            prompt="hi",
            sandbox_policy=None,
            approval_mode="auto_review",
        )

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_forwards_sandbox_policy_cookie_to_spawn_turn(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        """Sandbox policy is applied per-turn, not persisted on the thread,
        so follow-up messages must re-forward the cookie or every turn
        after the first silently reverts to Codex defaults — which breaks
        multi-turn workflows that depend on elevated permissions."""
        self._patch_codex(mock_codex)
        mock_discover.return_value = [Path("/repo")]
        _seed_cookies(self.client, hitch_sandbox_policy="workspaceWrite")

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "follow-up"},
        )

        self.assertEqual(response.status_code, 302)
        mock_spawn.assert_called_once_with(
            thread_id="abc",
            cwd="/repo",
            prompt="follow-up",
            sandbox_policy="workspaceWrite",
            approval_mode="auto_review",
        )

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_invalid_sandbox_cookie_is_treated_as_empty(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        """A tampered or post-SDK-upgrade cookie value must fall through to
        ``None`` rather than ride a bogus string into the worker."""
        self._patch_codex(mock_codex)
        mock_discover.return_value = [Path("/repo")]
        _seed_cookies(self.client, hitch_sandbox_policy="phantomPolicy")

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "follow-up"},
        )

        self.assertEqual(response.status_code, 302)
        mock_spawn.assert_called_once_with(
            thread_id="abc",
            cwd="/repo",
            prompt="follow-up",
            sandbox_policy=None,
            approval_mode="auto_review",
        )

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_forwards_memories_cookie_to_spawn_turn(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex)
        mock_discover.return_value = [Path("/repo")]
        _seed_cookies(self.client, **{_ENABLE_MEMORIES_COOKIE: "true"})

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "follow-up"},
        )

        self.assertEqual(response.status_code, 302)
        mock_spawn.assert_called_once_with(
            thread_id="abc",
            cwd="/repo",
            prompt="follow-up",
            sandbox_policy=None,
            approval_mode="auto_review",
            enable_memories=True,
        )

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_forwards_approval_mode_cookie_to_spawn_turn(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        """Approval mode is applied per-turn just like sandbox policy, so
        the cookie has to ride into every follow-up turn or an explicit
        stricter/user-prompting choice silently reverts to the SDK default."""
        self._patch_codex(mock_codex)
        mock_discover.return_value = [Path("/repo")]

        for mode in ("deny_all", "prompt_user"):
            with self.subTest(mode=mode):
                mock_spawn.reset_mock()
                _seed_cookies(self.client, hitch_approval_mode=mode)

                response = self.client.post(
                    reverse("send_message", kwargs={"session_id": "abc"}),
                    data={"prompt": "follow-up"},
                )

                self.assertEqual(response.status_code, 302)
                mock_spawn.assert_called_once_with(
                    thread_id="abc",
                    cwd="/repo",
                    prompt="follow-up",
                    sandbox_policy=None,
                    approval_mode=mode,
                )

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_auto_pr_session_marks_follow_up_turn(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        self._patch_codex(
            mock_codex,
            model="gpt-5.4",
            reasoning_effort="high",
        )
        mock_discover.return_value = [Path("/repo")]
        SessionMetadata.objects.create(
            thread_id="abc",
            cwd="/repo",
            auto_pr_enabled=True,
        )

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "follow-up"},
        )

        self.assertEqual(response.status_code, 302)
        mock_spawn.assert_called_once_with(
            thread_id="abc",
            cwd="/repo",
            prompt="follow-up",
            sandbox_policy=None,
            approval_mode="auto_review",
            auto_pr_enabled=True,
            user_message_index=0,
            stored_model="gpt-5.4",
            stored_reasoning_effort="high",
        )

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_forwards_plan_mode_for_one_turn(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex, model="gpt-5.4")
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "make a migration plan", "plan_mode": "true"},
        )

        self.assertEqual(response.status_code, 302)
        mock_spawn.assert_called_once_with(
            thread_id="abc",
            cwd="/repo",
            prompt="make a migration plan",
            sandbox_policy=None,
            approval_mode="auto_review",
            model="gpt-5.4",
            plan_mode=True,
        )

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_follow_up_after_pending_plan_stays_in_plan_mode(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        rollout_path = self._make_pending_plan_rollout("# Revised Plan\n\nKeep planning.")
        self._patch_codex(mock_codex, model="gpt-5.4", path=str(rollout_path))
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "tighten the QA part"},
        )

        self.assertEqual(response.status_code, 302)
        mock_spawn.assert_called_once_with(
            thread_id="abc",
            cwd="/repo",
            prompt="tighten the QA part",
            sandbox_policy=None,
            approval_mode="auto_review",
            model="gpt-5.4",
            plan_mode=True,
        )

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_explicit_plan_toggle_off_after_pending_plan_stays_default_mode(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        rollout_path = self._make_pending_plan_rollout()
        self._patch_codex(mock_codex, model="gpt-5.4", path=str(rollout_path))
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={
                "prompt": "ship it without more planning",
                "default_plan_mode": "true",
                "plan_mode_explicit": "true",
            },
        )

        self.assertEqual(response.status_code, 302)
        mock_spawn.assert_called_once_with(
            thread_id="abc",
            cwd="/repo",
            prompt="ship it without more planning",
            sandbox_policy=None,
            approval_mode="auto_review",
        )

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_pending_plan_default_without_model_falls_back_to_default_mode(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        rollout_path = self._make_pending_plan_rollout()
        self._patch_codex(mock_codex, model=None, models=[], path=str(rollout_path))
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={
                "prompt": "tighten the QA part",
                "plan_mode": "true",
                "default_plan_mode": "true",
            },
        )

        self.assertEqual(response.status_code, 302)
        mock_spawn.assert_called_once_with(
            thread_id="abc",
            cwd="/repo",
            prompt="tighten the QA part",
            sandbox_policy=None,
            approval_mode="auto_review",
        )

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.system_agents.start_pr_qa_workflow")
    @patch("hitch.main.views.Codex")
    def test_pr_slash_command_after_pending_plan_stays_default_mode(
        self,
        mock_codex: MagicMock,
        mock_start_workflow: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        rollout_path = self._make_pending_plan_rollout()
        self._patch_codex(mock_codex, model="gpt-5.4", path=str(rollout_path))
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "/pr", "plan_mode": "true"},
        )

        self.assertEqual(response.status_code, 302)
        mock_start_workflow.assert_called_once_with(
            main_thread_id="abc",
            cwd="/repo",
            sandbox_policy=None,
            approval_mode="auto_review",
            model="gpt-5.4",
            reasoning_effort=None,
            developer_instructions=None,
            enable_memories=False,
            initial_user_message_index=1,
        )

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.system_agents.start_pr_qa_workflow")
    @patch("hitch.main.views.Codex")
    def test_pr_menu_prompt_after_pending_plan_stays_default_mode(
        self,
        mock_codex: MagicMock,
        mock_start_workflow: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        rollout_path = self._make_pending_plan_rollout()
        self._patch_codex(mock_codex, model="gpt-5.4", path=str(rollout_path))
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": _PR_PROMPT},
        )

        self.assertEqual(response.status_code, 302)
        mock_start_workflow.assert_called_once_with(
            main_thread_id="abc",
            cwd="/repo",
            sandbox_policy=None,
            approval_mode="auto_review",
            model="gpt-5.4",
            reasoning_effort=None,
            developer_instructions=None,
            enable_memories=False,
            initial_user_message_index=1,
        )

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_default_collaboration_mode_switches_to_default_mode(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        rollout_path = self._make_pending_plan_rollout()
        self._patch_codex(mock_codex, model="gpt-5.4", path=str(rollout_path))
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={
                "prompt": "Implement the plan.",
                "collaboration_mode": "default",
                "plan_mode": "true",
                "default_plan_mode": "true",
            },
        )

        self.assertEqual(response.status_code, 302)
        mock_spawn.assert_called_once_with(
            thread_id="abc",
            cwd="/repo",
            prompt="Implement the plan.",
            sandbox_policy=None,
            approval_mode="auto_review",
            model="gpt-5.4",
            collaboration_mode="default",
        )

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_plan_approve_action_switches_to_default_mode(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        rollout_path = self._make_pending_plan_rollout()
        self._patch_codex(mock_codex, model="gpt-5.4", path=str(rollout_path))
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={
                "prompt": "Implement the plan.",
                "plan_action": "approve",
                "plan_mode": "true",
                "default_plan_mode": "true",
            },
        )

        self.assertEqual(response.status_code, 302)
        mock_spawn.assert_called_once_with(
            thread_id="abc",
            cwd="/repo",
            prompt="Implement the plan.",
            sandbox_policy=None,
            approval_mode="auto_review",
            model="gpt-5.4",
            collaboration_mode="default",
        )

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_auto_pr_marks_plan_implementation_turn(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        rollout_path = self._make_pending_plan_rollout()
        self._patch_codex(mock_codex, model="gpt-5.4", path=str(rollout_path))
        mock_discover.return_value = [Path("/repo")]
        SessionMetadata.objects.create(
            thread_id="abc",
            cwd="/repo",
            auto_pr_enabled=True,
        )

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={
                "prompt": "Implement the plan.",
                "plan_action": "approve",
                "plan_mode": "true",
                "default_plan_mode": "true",
            },
        )

        self.assertEqual(response.status_code, 302)
        mock_spawn.assert_called_once_with(
            thread_id="abc",
            cwd="/repo",
            prompt="Implement the plan.",
            sandbox_policy=None,
            approval_mode="auto_review",
            auto_pr_enabled=True,
            user_message_index=1,
            stored_model="gpt-5.4",
            stored_reasoning_effort=None,
            model="gpt-5.4",
            collaboration_mode="default",
        )

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_pending_plan_approval_prompt_switches_to_default_mode(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        rollout_path = self._make_pending_plan_rollout()
        self._patch_codex(mock_codex, model="gpt-5.4", path=str(rollout_path))
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={
                "prompt": "Implement the plan.",
                "plan_mode": "true",
                "default_plan_mode": "true",
            },
        )

        self.assertEqual(response.status_code, 302)
        mock_spawn.assert_called_once_with(
            thread_id="abc",
            cwd="/repo",
            prompt="Implement the plan.",
            sandbox_policy=None,
            approval_mode="auto_review",
            model="gpt-5.4",
            collaboration_mode="default",
        )

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_plan_revise_action_stays_in_plan_mode(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        rollout_path = self._make_pending_plan_rollout()
        self._patch_codex(mock_codex, model="gpt-5.4", path=str(rollout_path))
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "Revise the plan.", "plan_action": "revise"},
        )

        self.assertEqual(response.status_code, 302)
        mock_spawn.assert_called_once_with(
            thread_id="abc",
            cwd="/repo",
            prompt="Revise the plan.",
            sandbox_policy=None,
            approval_mode="auto_review",
            model="gpt-5.4",
            plan_mode=True,
        )

    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_rejects_invalid_plan_action(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
    ) -> None:
        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "Implement the plan.", "plan_action": "ship"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertContains(response, "invalid plan action", status_code=400)
        mock_codex.assert_not_called()
        mock_spawn.assert_not_called()

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_plan_slash_command_strips_command_prefix(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex, model="gpt-5.4")
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "/plan make a migration plan"},
        )

        self.assertEqual(response.status_code, 302)
        mock_spawn.assert_called_once_with(
            thread_id="abc",
            cwd="/repo",
            prompt="make a migration plan",
            sandbox_policy=None,
            approval_mode="auto_review",
            model="gpt-5.4",
            plan_mode=True,
        )

    @patch("hitch.main.views.system_agents.start_pr_qa_workflow")
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_pr_slash_command_starts_qa_workflow(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_start_workflow: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex, model="gpt-5.4")
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "/pr"},
        )

        self.assertEqual(response.status_code, 302)
        mock_start_workflow.assert_called_once_with(
            main_thread_id="abc",
            cwd="/repo",
            sandbox_policy=None,
            approval_mode="auto_review",
            model="gpt-5.4",
            reasoning_effort=None,
            developer_instructions=None,
            enable_memories=False,
            initial_user_message_index=0,
        )

    @patch("hitch.main.views.system_agents.start_pr_qa_workflow")
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_qa_slash_command_starts_qa_workflow_without_pr_prompt(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_start_workflow: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex, model="gpt-5.4", reasoning_effort="high")
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "/qa", "plan_mode": "true"},
        )

        self.assertEqual(response.status_code, 302)
        mock_start_workflow.assert_called_once_with(
            main_thread_id="abc",
            cwd="/repo",
            sandbox_policy=None,
            approval_mode="auto_review",
            model="gpt-5.4",
            reasoning_effort="high",
            developer_instructions=None,
            enable_memories=False,
            initial_user_message_index=0,
            open_pr_on_lgtm=False,
        )

    @patch("hitch.main.views.system_agents.start_pr_qa_workflow")
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_qa_menu_prompt_starts_qa_workflow_without_pr_prompt(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_start_workflow: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex, model="gpt-5.4")
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": _QA_PROMPT},
        )

        self.assertEqual(response.status_code, 302)
        mock_start_workflow.assert_called_once_with(
            main_thread_id="abc",
            cwd="/repo",
            sandbox_policy=None,
            approval_mode="auto_review",
            model="gpt-5.4",
            reasoning_effort=None,
            developer_instructions=None,
            enable_memories=False,
            initial_user_message_index=0,
            open_pr_on_lgtm=False,
        )

    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_running_pr_workflow_blocks_normal_follow_up(
        self, mock_codex: MagicMock, mock_spawn: MagicMock
    ) -> None:
        SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="abc",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step="qa_running",
        )

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "please also do this"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertContains(
            response, "PR workflow is running for this session", status_code=400
        )
        mock_codex.assert_not_called()
        mock_spawn.assert_not_called()

    @patch("hitch.main.views.system_agents.start_pr_qa_workflow")
    @patch("hitch.main.views.Codex")
    def test_duplicate_pr_command_during_running_workflow_redirects(
        self, mock_codex: MagicMock, mock_start_workflow: MagicMock
    ) -> None:
        SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="abc",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step="qa_running",
        )

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "/pr"},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            response.headers["Location"],
            reverse("session", kwargs={"session_id": "abc"}),
        )
        mock_codex.assert_not_called()
        mock_start_workflow.assert_not_called()

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_plan_mode_uses_saved_model_when_resume_omits_model(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex, model=None)
        mock_discover.return_value = [Path("/repo")]
        _seed_cookies(self.client, **{_MODEL_COOKIE: "gpt-saved"})

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "make a migration plan", "plan_mode": "true"},
        )

        self.assertEqual(response.status_code, 302)
        mock_spawn.assert_called_once_with(
            thread_id="abc",
            cwd="/repo",
            prompt="make a migration plan",
            sandbox_policy=None,
            approval_mode="auto_review",
            model="gpt-saved",
            plan_mode=True,
        )

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_plan_mode_uses_default_model_when_resume_omits_model(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        self._patch_codex(
            mock_codex,
            model=None,
            models=[_make_model("gpt-default", is_default=True)],
        )
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "make a migration plan", "plan_mode": "true"},
        )

        self.assertEqual(response.status_code, 302)
        mock_spawn.assert_called_once_with(
            thread_id="abc",
            cwd="/repo",
            prompt="make a migration plan",
            sandbox_policy=None,
            approval_mode="auto_review",
            model="gpt-default",
            plan_mode=True,
        )

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_plan_mode_returns_bad_request_when_model_cannot_be_resolved(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex, model=None, models=[])
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "make a migration plan", "plan_mode": "true"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertContains(response, "plan mode requires a model", status_code=400)
        mock_spawn.assert_not_called()

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_invalid_approval_cookie_falls_back_to_safe_default(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        """A tampered or post-SDK-upgrade cookie value must snap back to
        the safe default rather than ride a bogus string into the worker
        (which would map to ``None`` and silently drop the policy)."""
        self._patch_codex(mock_codex)
        mock_discover.return_value = [Path("/repo")]
        _seed_cookies(self.client, hitch_approval_mode="phantomMode")

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "follow-up"},
        )

        self.assertEqual(response.status_code, 302)
        mock_spawn.assert_called_once_with(
            thread_id="abc",
            cwd="/repo",
            prompt="follow-up",
            sandbox_policy=None,
            approval_mode="auto_review",
        )

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_rejects_invalid_input(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]

        # cwd-missing and cwd-outside-allowlist need the resumed thread set up;
        # the empty-prompt cases never reach Codex, but stubbing it is cheap.
        cases = [
            ({"prompt": ""}, "/repo", "empty prompt"),
            ({"prompt": "   \n  "}, "/repo", "whitespace-only prompt"),
            ({"prompt": "hi"}, None, "thread without cwd"),
            # The session list shows every thread the app-server knows about,
            # so a resumed thread's cwd can point outside the discover_repos()
            # allowlist (e.g. for threads created by another tool). The
            # composer must refuse to spawn a worker in such a directory.
            ({"prompt": "hi"}, "/etc", "cwd outside allowed list"),
        ]
        for data, cwd, label in cases:
            with self.subTest(label=label):
                self._patch_codex(mock_codex, cwd=cwd)
                mock_spawn.reset_mock()
                response = self.client.post(
                    reverse("send_message", kwargs={"session_id": "abc"}),
                    data=data,
                )
                self.assertEqual(response.status_code, 400)
                mock_spawn.assert_not_called()

    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_rejects_get(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
    ) -> None:
        response = self.client.get(
            reverse("send_message", kwargs={"session_id": "abc"})
        )

        self.assertEqual(response.status_code, 405)
        mock_codex.assert_not_called()
        mock_spawn.assert_not_called()


class SetSessionNameViewTests(TestCase):
    @patch("hitch.main.views.Codex")
    def test_updates_name_and_response_shape(self, mock_codex: MagicMock) -> None:
        client = mock_codex.return_value.__enter__.return_value
        cases: list[tuple[str, dict[str, str], bool, int, str | None]] = [
            (
                "session redirect trims whitespace",
                {"name": "  New title  "},
                False,
                302,
                reverse("session", kwargs={"session_id": "abc"}),
            ),
            (
                "index redirect",
                {"name": "New title", "next": "index"},
                False,
                302,
                reverse("index"),
            ),
            (
                "ajax",
                {"name": "New title"},
                True,
                204,
                None,
            ),
        ]
        for label, data, ajax, status, location in cases:
            with self.subTest(label=label):
                client._client.thread_set_name.reset_mock()
                url = reverse("set_session_name", kwargs={"session_id": "abc"})
                if ajax:
                    response = self.client.post(
                        url,
                        data=data,
                        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
                    )
                else:
                    response = self.client.post(url, data=data)

                self.assertEqual(response.status_code, status)
                if location is None:
                    self.assertNotIn("Location", response.headers)
                else:
                    self.assertEqual(response.headers["Location"], location)
                client._client.thread_set_name.assert_called_once_with(
                    "abc", "New title"
                )

    @patch("hitch.main.views.Codex")
    def test_rejects_invalid_requests(self, mock_codex: MagicMock) -> None:
        # The form caps input client-side; the view enforces the same bounds
        # so a hand-crafted POST can't bypass them.
        cases: list[tuple[str, dict[str, str], str, int]] = [
            ("post", {"name": ""}, "empty", 400),
            ("post", {"name": "   "}, "whitespace only", 400),
            ("post", {"name": "x" * 201}, "over length cap", 400),
            ("get", {}, "method", 405),
        ]
        for method, data, label, status in cases:
            with self.subTest(label=label):
                url = reverse("set_session_name", kwargs={"session_id": "abc"})
                if method == "post":
                    response = self.client.post(url, data=data)
                else:
                    response = self.client.get(url)
                self.assertEqual(response.status_code, status)
        mock_codex.assert_not_called()


class StartSessionDemoViewTests(TestCase):
    @patch("hitch.main.views.demo.start_demo_container")
    @patch("hitch.main.views.system_agents.active_workflow_for_thread")
    def test_rejects_start_while_system_workflow_is_active(
        self, mock_active_workflow: MagicMock, mock_start_demo: MagicMock
    ) -> None:
        mock_active_workflow.return_value = SimpleNamespace()

        response = self.client.post(
            reverse("start_session_demo", kwargs={"session_id": "abc"})
        )

        self.assertEqual(response.status_code, 400)
        self.assertContains(
            response,
            "PR workflow is running for this session",
            status_code=400,
        )
        mock_start_demo.assert_not_called()

    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.codex_pool.latest_active_for_thread", return_value=None)
    @patch("hitch.main.views.demo.start_demo_container")
    @patch("hitch.main.views.system_agents.active_workflow_for_thread", return_value=None)
    @patch("hitch.main.views.discover_managed_worktrees", return_value=[])
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_starts_container_and_sends_agent_setup_prompt(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        _mock_managed: MagicMock,
        _mock_workflow: MagicMock,
        mock_start_demo: MagicMock,
        _mock_active: MagicMock,
        mock_spawn: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(cwd="/repo", turns=[])
        )
        session_demo = SessionDemo(
            thread_id="abc",
            host="127.0.0.1",
            port=45678,
            container_id="container-1",
            container_name="hitch-demo-abc",
            status=SessionDemo.STATUS_ACTIVE,
        )
        mock_start_demo.return_value = (session_demo, 3000)

        response = self.client.post(
            reverse("start_session_demo", kwargs={"session_id": "abc"})
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], reverse("session", kwargs={"session_id": "abc"}))
        mock_start_demo.assert_called_once_with("abc")
        mock_spawn.assert_called_once()
        kwargs = mock_spawn.call_args.kwargs
        self.assertEqual(kwargs["thread_id"], "abc")
        self.assertEqual(kwargs["cwd"], "/repo")
        self.assertIn("container id: container-1", kwargs["prompt"])
        self.assertIn("internal web port: 3000", kwargs["prompt"])
        self.assertIn("http://testserver/sessions/abc/demo/", kwargs["prompt"])

    @patch("hitch.main.views.demo.cleanup_demo_for_session")
    @patch("hitch.main.views.codex_pool.spawn_turn", side_effect=RuntimeError("spawn failed"))
    @patch("hitch.main.views.codex_pool.latest_active_for_thread", return_value=None)
    @patch("hitch.main.views.demo.start_demo_container")
    @patch("hitch.main.views.system_agents.active_workflow_for_thread", return_value=None)
    @patch("hitch.main.views.discover_managed_worktrees", return_value=[])
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_cleans_up_demo_when_worker_dispatch_fails(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        _mock_managed: MagicMock,
        _mock_workflow: MagicMock,
        mock_start_demo: MagicMock,
        _mock_active: MagicMock,
        _mock_spawn: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(cwd="/repo", turns=[])
        )
        mock_start_demo.return_value = (
            SessionDemo(
                thread_id="abc",
                host="127.0.0.1",
                port=45678,
                container_id="container-1",
                status=SessionDemo.STATUS_ACTIVE,
            ),
            3000,
        )

        with self.assertRaisesRegex(RuntimeError, "spawn failed"):
            self.client.post(reverse("start_session_demo", kwargs={"session_id": "abc"}))

        mock_cleanup.assert_called_once_with("abc")

    @patch("hitch.main.views.codex_pool.steer_instance", return_value=True)
    @patch("hitch.main.views.codex_pool.latest_active_for_thread")
    @patch("hitch.main.views.demo.start_demo_container")
    @patch("hitch.main.views.system_agents.active_workflow_for_thread", return_value=None)
    @patch("hitch.main.views.discover_managed_worktrees", return_value=[])
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_steers_active_worker_after_starting_demo(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        _mock_managed: MagicMock,
        _mock_workflow: MagicMock,
        mock_start_demo: MagicMock,
        mock_active: MagicMock,
        mock_steer: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(cwd="/repo", turns=[])
        )
        mock_active.return_value = SimpleNamespace(pk=123)
        mock_start_demo.return_value = (
            SessionDemo(
                thread_id="abc",
                host="127.0.0.1",
                port=45678,
                container_id="container-1",
                status=SessionDemo.STATUS_ACTIVE,
            ),
            3000,
        )

        response = self.client.post(reverse("start_session_demo", kwargs={"session_id": "abc"}))

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], reverse("session", kwargs={"session_id": "abc"}))
        mock_steer.assert_called_once()
        self.assertEqual(mock_steer.call_args.kwargs["expected_thread_id"], "abc")

    @patch("hitch.main.views.demo.start_demo_container")
    @patch("hitch.main.views.system_agents.active_workflow_for_thread", return_value=None)
    @patch("hitch.main.views.discover_managed_worktrees", return_value=[])
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_rejects_missing_cwd_before_starting_container(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        _mock_managed: MagicMock,
        _mock_workflow: MagicMock,
        mock_start_demo: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(cwd="", turns=[])
        )

        response = self.client.post(reverse("start_session_demo", kwargs={"session_id": "abc"}))

        self.assertEqual(response.status_code, 400)
        self.assertContains(response, "thread has no cwd", status_code=400)
        mock_start_demo.assert_not_called()

    @patch("hitch.main.views.demo.start_demo_container", side_effect=demo.DemoError("no podman"))
    @patch("hitch.main.views.system_agents.active_workflow_for_thread", return_value=None)
    @patch("hitch.main.views.discover_managed_worktrees", return_value=[])
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_reports_demo_start_failure(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        _mock_managed: MagicMock,
        _mock_workflow: MagicMock,
        _mock_start_demo: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(cwd="/repo", turns=[])
        )

        response = self.client.post(reverse("start_session_demo", kwargs={"session_id": "abc"}))

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.content, b"no podman")

    @patch("hitch.main.views.demo.start_demo_container")
    @patch("hitch.main.views.system_agents.active_workflow_for_thread", return_value=None)
    @patch("hitch.main.views.discover_managed_worktrees", return_value=[])
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_rejects_unallowed_cwd_before_starting_container(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        _mock_managed: MagicMock,
        _mock_workflow: MagicMock,
        mock_start_demo: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(cwd="/elsewhere", turns=[])
        )

        response = self.client.post(
            reverse("start_session_demo", kwargs={"session_id": "abc"})
        )

        self.assertEqual(response.status_code, 400)
        mock_start_demo.assert_not_called()


class SetSessionArchivedViewTests(TestCase):
    @patch("hitch.main.views.Codex")
    def test_updates_archive_state_and_response_shape(
        self, mock_codex: MagicMock
    ) -> None:
        client = mock_codex.return_value.__enter__.return_value
        cases: list[
            tuple[str, dict[str, str], bool, int, str | None, str, bool]
        ] = [
            (
                "archive",
                {"archived": "true"},
                False,
                302,
                reverse("index"),
                "thread_archive",
                True,
            ),
            (
                "archive ajax",
                {"archived": "true"},
                True,
                204,
                None,
                "thread_archive",
                False,
            ),
            (
                "unarchive to session",
                {"archived": "false"},
                False,
                302,
                reverse("session", kwargs={"session_id": "abc"}),
                "thread_unarchive",
                True,
            ),
            (
                "unarchive to index",
                {"archived": "false", "next": "index"},
                False,
                302,
                reverse("index"),
                "thread_unarchive",
                False,
            ),
        ]
        for label, data, ajax, status, location, expected_call, seed_cache in cases:
            with self.subTest(label=label):
                ArchivedSessionTokenUsage.objects.all().delete()
                client.thread_archive.reset_mock()
                client.thread_unarchive.reset_mock()
                if seed_cache:
                    ArchivedSessionTokenUsage.objects.create(
                        thread_id="abc",
                        total_tokens=100,
                    )
                    ArchivedSessionTokenUsage.objects.create(
                        thread_id="abc-child",
                        total_tokens=200,
                    )

                url = reverse("set_session_archived", kwargs={"session_id": "abc"})
                if ajax:
                    response = self.client.post(
                        url,
                        data=data,
                        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
                    )
                else:
                    response = self.client.post(url, data=data)

                self.assertEqual(response.status_code, status)
                if location is None:
                    self.assertNotIn("Location", response.headers)
                else:
                    self.assertEqual(response.headers["Location"], location)
                if expected_call == "thread_archive":
                    client.thread_archive.assert_called_once_with("abc")
                    client.thread_unarchive.assert_not_called()
                else:
                    client.thread_unarchive.assert_called_once_with("abc")
                    client.thread_archive.assert_not_called()
                if seed_cache:
                    self.assertEqual(ArchivedSessionTokenUsage.objects.count(), 0)

    @patch("hitch.main.demo.subprocess.run")
    @patch("hitch.main.views.Codex")
    def test_archive_cleans_up_active_demo_container(
        self, mock_codex: MagicMock, mock_run: MagicMock
    ) -> None:
        SessionDemo.objects.create(
            thread_id="abc",
            host="127.0.0.1",
            port=45678,
            container_id="container-1",
            runtime="podman",
            status=SessionDemo.STATUS_ACTIVE,
        )

        response = self.client.post(
            reverse("set_session_archived", kwargs={"session_id": "abc"}),
            data={"archived": "true"},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(SessionDemo.objects.get(thread_id="abc").status, SessionDemo.STATUS_STOPPED)
        mock_run.assert_called_once_with(
            ["podman", "rm", "-f", "container-1"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        mock_codex.return_value.__enter__.return_value.thread_archive.assert_called_once_with("abc")

    @patch("hitch.main.views.Codex")
    def test_rejects_invalid_archive_requests(self, mock_codex: MagicMock) -> None:
        cases: list[tuple[str, dict[str, str], int]] = [
            ("post", {}, 400),
            ("post", {"archived": ""}, 400),
            ("post", {"archived": "yes"}, 400),
            ("get", {}, 405),
        ]
        for method, data, status in cases:
            with self.subTest(method=method, data=data):
                url = reverse("set_session_archived", kwargs={"session_id": "abc"})
                if method == "post":
                    response = self.client.post(url, data=data)
                else:
                    response = self.client.get(url)
                self.assertEqual(response.status_code, status)
        mock_codex.assert_not_called()


class StopSessionViewTests(TestCase):
    @patch("hitch.main.views.codex_pool.interrupt_instance")
    @patch("hitch.main.views.codex_pool.interrupt_active")
    def test_targets_instance_from_form_value(
        self,
        mock_interrupt_active: MagicMock,
        mock_interrupt_instance: MagicMock,
    ) -> None:
        # The Stop button posts the active worker's pk so a stale tab
        # cannot accidentally abort a newer overlapping worker. The
        # view forwards the id (and the URL's session id, as a
        # cross-thread guard) to ``interrupt_instance``.
        response = self.client.post(
            reverse("stop_session", kwargs={"session_id": "abc"}),
            data={"instance": "42"},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            response.headers["Location"],
            reverse("session", kwargs={"session_id": "abc"}),
        )
        mock_interrupt_instance.assert_called_once_with(42, expected_thread_id="abc")
        mock_interrupt_active.assert_not_called()

    @patch(
        "hitch.main.views.system_agents.stop_active_workflow",
        wraps=system_agents.stop_active_workflow,
    )
    @patch("hitch.main.views.codex_pool.interrupt_instance")
    @patch("hitch.main.views.codex_pool.interrupt_active")
    def test_falls_back_to_latest_active_without_instance(
        self,
        mock_interrupt_active: MagicMock,
        mock_interrupt_instance: MagicMock,
        mock_stop_workflow: MagicMock,
    ) -> None:
        # Older cached page (or a direct curl POST) won't carry the
        # instance field; fall back to "latest active worker for this
        # thread" so the stop click still has a chance to do something.
        # ``None`` models a double-click after the worker already finished;
        # the view should still redirect instead of surfacing an error.
        mock_interrupt_active.return_value = None
        response = self.client.post(
            reverse("stop_session", kwargs={"session_id": "abc"})
        )

        self.assertEqual(response.status_code, 302)
        mock_stop_workflow.assert_called_once_with("abc")
        mock_interrupt_active.assert_called_once_with("abc")
        mock_interrupt_instance.assert_not_called()

    @patch("hitch.main.views.codex_pool.interrupt_active")
    @patch("hitch.main.views.system_agents.stop_active_workflow", return_value=True)
    def test_stops_active_system_workflow_without_instance(
        self, mock_stop_workflow: MagicMock, mock_interrupt_active: MagicMock
    ) -> None:
        response = self.client.post(
            reverse("stop_session", kwargs={"session_id": "abc"})
        )

        self.assertEqual(response.status_code, 302)
        mock_stop_workflow.assert_called_once_with("abc")
        mock_interrupt_active.assert_not_called()

    @patch("hitch.main.views.codex_pool.interrupt_instance")
    @patch("hitch.main.views.codex_pool.interrupt_active")
    def test_rejects_invalid_requests(
        self, mock_interrupt_active: MagicMock, mock_interrupt_instance: MagicMock
    ) -> None:
        # Tampered/oversized values must be rejected at the view boundary so
        # they never reach ``objects.get`` (which would raise backend-specific
        # OverflowError/DataError and surface as a 500 instead of a clean 400).
        url = reverse("stop_session", kwargs={"session_id": "abc"})
        cases: list[tuple[str, dict[str, str], str, int]] = [
            ("post", {"instance": "not-a-number"}, "non-integer", 400),
            ("post", {"instance": "0"}, "zero", 400),
            ("post", {"instance": "-1"}, "negative", 400),
            ("post", {"instance": str(2**63)}, "above BigAutoField max", 400),
            ("get", {}, "method", 405),
        ]
        for method, data, label, status in cases:
            with self.subTest(label=label):
                if method == "post":
                    response = self.client.post(url, data=data)
                else:
                    response = self.client.get(url)
                self.assertEqual(response.status_code, status)
        mock_interrupt_active.assert_not_called()
        mock_interrupt_instance.assert_not_called()


class SessionStreamViewTests(TestCase):
    """The SSE endpoint that mirrors a worker's events file to the browser."""

    def _make(self, **kwargs: Any) -> CodexInstance:
        defaults: dict[str, Any] = {
            "pid": 0,
            "thread_id": "thread-1",
            "cwd": "/repo",
            "prompt": "do work",
            "events_path": "/dev/null",
            "status": CodexInstance.STATUS_RUNNING,
        }
        defaults.update(kwargs)
        return CodexInstance.objects.create(**defaults)

    def _stream_url(
        self,
        session_id: str,
        *,
        baseline: str = "",
        active: str = "",
        workflow: str = "",
    ) -> str:
        # Helper that builds the SSE URL with the page-render-time state
        # the view expects on every legitimate request. Tests that want
        # to exercise the stale-reload path pass an empty/wrong value.
        return (
            reverse("session_stream", kwargs={"session_id": session_id})
            + f"?baseline={baseline}&active={active}&workflow={workflow}"
        )

    @patch("hitch.main.streaming._IDLE_MAX_STREAM_SECONDS", 0.001)
    @patch("hitch.main.streaming._IDLE_POLL_INTERVAL", 0.001)
    def test_returns_idle_heartbeat_stream_without_active_worker(self) -> None:
        # Without an active worker the SSE channel stays open emitting
        # heartbeat events with ``working: false`` so the page's connection
        # indicator can show ``connected, idle``. The cap is patched down
        # so the test doesn't sit in the recycle loop.
        response = self.client.get(self._stream_url("thread-1"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/event-stream")
        self.assertEqual(response["Cache-Control"], "no-cache")
        self.assertEqual(response["X-Accel-Buffering"], "no")
        body = b"".join(response.streaming_content)  # type: ignore[attr-defined]
        self.assertIn(b"event: heartbeat", body)
        self.assertIn(b'"working": false', body)

        # A terminal worker counts as ``no active worker`` for routing
        # purposes; the stream should stay idle without re-tailing old events.
        with tempfile.TemporaryDirectory() as raw:
            events_path = str(Path(raw) / "events.jsonl")
            Path(events_path).touch()
            inst = self._make(
                thread_id="thread-done",
                status=CodexInstance.STATUS_COMPLETED,
                events_path=events_path,
            )

            response = self.client.get(
                self._stream_url("thread-done", baseline=str(inst.pk))
            )
            body = b"".join(response.streaming_content)  # type: ignore[attr-defined]

        self.assertIn(b"event: heartbeat", body)
        self.assertIn(b'"working": false', body)

    @patch("hitch.main.streaming._IDLE_MAX_STREAM_SECONDS", 0.001)
    @patch("hitch.main.streaming._IDLE_POLL_INTERVAL", 0.001)
    def test_returns_working_heartbeat_stream_for_active_system_workflow(self) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="thread-workflow",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step="qa_running",
        )

        response = self.client.get(
            self._stream_url("thread-workflow", workflow=str(workflow.pk))
        )
        body = b"".join(response.streaming_content)  # type: ignore[attr-defined]

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"event: heartbeat", body)
        self.assertIn(b'"working": true', body)

    def test_reloads_when_page_render_state_is_stale(self) -> None:
        # The classic out-of-band-spawn race: page rendered with no
        # worker (empty baseline / active), but by the time SSE opens a
        # worker has shown up in the DB. The endpoint must reload the
        # page so the DOM gets the live-streaming UI before any item
        # events start arriving.
        self._make(thread_id="thread-1", status=CodexInstance.STATUS_RUNNING)
        response = self.client.get(self._stream_url("thread-1"))
        body = b"".join(response.streaming_content)  # type: ignore[attr-defined]
        self.assertIn(b'"status": "stale"', body)

        # Inverse race: page rendered expecting a live worker (passes
        # ``active=N`` and ``baseline=N``) but by the time SSE opens the
        # worker has gone terminal. Without the reload the page would
        # show a permanent "Codex is working…" pill and a stale pending
        # bubble for the just-completed turn.
        inst = self._make(
            thread_id="thread-completed-before-open",
            status=CodexInstance.STATUS_COMPLETED,
        )
        response = self.client.get(
            self._stream_url(
                "thread-completed-before-open",
                baseline=str(inst.pk),
                active=str(inst.pk),
            )
        )
        body = b"".join(response.streaming_content)  # type: ignore[attr-defined]
        self.assertIn(b'"status": "stale"', body)

    @patch("hitch.main.streaming._POLL_INTERVAL", 0.01)
    def test_forwards_worker_events_through_view(self) -> None:
        # End-to-end through the URL routing: a RUNNING instance with
        # events on disk gets tailed, and once the status flips before the
        # response is iterated the stream drains and closes.
        with tempfile.TemporaryDirectory() as raw:
            events_path = str(Path(raw) / "events.jsonl")
            with open(events_path, "w", encoding="utf-8") as fh:
                fh.write(
                    json.dumps({"method": "item/started", "payload": {"item": {"id": "a"}}})
                    + "\n"
                )
            instance = self._make(
                thread_id="thread-live",
                status=CodexInstance.STATUS_RUNNING,
                events_path=events_path,
            )
            response = self.client.get(
                self._stream_url(
                    "thread-live", baseline=str(instance.pk), active=str(instance.pk)
                )
            )
            # Flip the row terminal before iterating so the generator's
            # _is_done() check exits the read loop cleanly.
            instance.status = CodexInstance.STATUS_COMPLETED
            instance.save(update_fields=["status"])
            body = b"".join(response.streaming_content)  # type: ignore[attr-defined]

        self.assertIn(b"item/started", body)
        self.assertIn(b'"status": "completed"', body)


class ResolveApprovalViewTests(TestCase):
    """The ``POST /approval/<id>/`` endpoint that records the user's pick on
    a pending command/file approval. The worker's polling loop wakes on the
    row update and answers codex's JSON-RPC request with the recorded
    decision — see ``hitch.main.management.commands.codex_worker``."""

    def _make_approval(
        self, *, decision: str = ApprovalRequest.DECISION_PENDING
    ) -> ApprovalRequest:
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="thread-1",
            cwd="/repo",
            prompt="hi",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
        )
        return ApprovalRequest.objects.create(
            instance=instance,
            method="item/commandExecution/requestApproval",
            params={"item": {"command": "ls"}},
            decision=decision,
        )

    def test_accepts_each_valid_decision(self) -> None:
        """Pin the wire-string contract — these three values are what
        app-server's approval response schema accepts (``accept`` /
        ``decline`` / ``cancel``). A regression that drops one of them
        would silently break that decision in the UI."""
        for decision in ("accept", "decline", "cancel"):
            with self.subTest(decision=decision):
                approval = self._make_approval()
                response = self.client.post(
                    reverse("resolve_approval", kwargs={"approval_id": approval.pk}),
                    data={"decision": decision},
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.content, decision.encode())
                approval.refresh_from_db()
                self.assertEqual(approval.decision, decision)
                self.assertIsNotNone(approval.decided_at)

    def test_normalizes_legacy_decision_values(self) -> None:
        """Tabs loaded before a deploy may still POST the old UI values.
        Normalize them at the boundary so a click doesn't poison the row
        with a value app-server treats as a declined request."""
        aliases = {
            "approved": "accept",
            "denied": "decline",
            "abort": "cancel",
        }
        for posted, stored in aliases.items():
            with self.subTest(posted=posted):
                approval = self._make_approval()
                response = self.client.post(
                    reverse("resolve_approval", kwargs={"approval_id": approval.pk}),
                    data={"decision": posted},
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.content, stored.encode())
                approval.refresh_from_db()
                self.assertEqual(approval.decision, stored)

    def test_rejects_invalid_or_stale_requests(self) -> None:
        """A POST with a value outside the app-server-accepted set must 400
        rather than poison the row — the worker would otherwise round-trip
        the bogus string into a JSON-RPC response codex rejects. Already
        resolved rows must stay locked so two tabs cannot clobber a choice."""
        cases: list[
            tuple[str, ApprovalRequest | None, str, dict[str, str], int, str | None]
        ] = [
            (
                "invalid decision",
                self._make_approval(),
                "post",
                {"decision": "yes please"},
                400,
                "",
            ),
            ("missing row", None, "post", {"decision": "accept"}, 404, None),
            (
                "already resolved",
                self._make_approval(decision="accept"),
                "post",
                {"decision": "decline"},
                409,
                "accept",
            ),
            ("method", self._make_approval(), "get", {}, 405, ""),
        ]
        for label, approval, method, data, status, expected_decision in cases:
            with self.subTest(label=label):
                approval_id = approval.pk if approval is not None else 99999999
                url = reverse("resolve_approval", kwargs={"approval_id": approval_id})
                if method == "post":
                    response = self.client.post(url, data=data)
                else:
                    response = self.client.get(url)

                self.assertEqual(response.status_code, status)
                if approval is not None:
                    approval.refresh_from_db()
                    self.assertEqual(approval.decision, expected_decision)


class ResolveInputRequestViewTests(TestCase):
    """The ``POST /input/<id>/`` endpoint records structured answers for
    app-server ``request_user_input`` prompts.
    """

    def _make_input_request(
        self, *, response: dict[str, object] | None = None
    ) -> UserInputRequest:
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="thread-1",
            cwd="/repo",
            prompt="hi",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
        )
        return UserInputRequest.objects.create(
            instance=instance,
            method="request_user_input",
            params={"questions": [{"id": "scope"}]},
            response=response,
        )

    def test_records_answer_payloads(self) -> None:
        structured_answers = {
            "scope": ["UI", "CLI"],
            "details": {"choice": "Other", "notes": ["keep history"]},
            "confirmed": True,
            "priority": 2,
            "optional": None,
        }
        cases = [
            (
                "string answer",
                {"answers": json.dumps({"scope": "Management command"})},
                {"answers": {"scope": "Management command"}},
            ),
            ("omitted payload", {}, {"answers": {}}),
            (
                "trimmed strings",
                {"answers": json.dumps({" scope ": " UI ", " ": "ignored"})},
                {"answers": {"scope": "UI"}},
            ),
            (
                "structured values",
                {"answers": json.dumps(structured_answers)},
                {"answers": structured_answers},
            ),
        ]
        for label, data, expected_response in cases:
            with self.subTest(label=label):
                input_request = self._make_input_request()

                response = self.client.post(
                    reverse(
                        "resolve_input_request", kwargs={"input_id": input_request.pk}
                    ),
                    data=data,
                )

                self.assertEqual(response.status_code, 200)
                input_request.refresh_from_db()
                self.assertEqual(input_request.response, expected_response)
                self.assertIsNotNone(input_request.responded_at)

    def test_rejects_invalid_answers_payload(self) -> None:
        input_request = self._make_input_request()

        for answers in ("not-json", json.dumps(["not", "object"])):
            with self.subTest(answers=answers):
                response = self.client.post(
                    reverse(
                        "resolve_input_request", kwargs={"input_id": input_request.pk}
                    ),
                    data={"answers": answers},
                )
                self.assertEqual(response.status_code, 400)

    def test_returns_409_when_already_resolved(self) -> None:
        input_request = self._make_input_request(response={"answers": {"scope": "UI"}})

        response = self.client.post(
            reverse("resolve_input_request", kwargs={"input_id": input_request.pk}),
            data={"answers": json.dumps({"scope": "CLI"})},
        )

        self.assertEqual(response.status_code, 409)
        input_request.refresh_from_db()
        self.assertEqual(input_request.response, {"answers": {"scope": "UI"}})

    def test_returns_404_when_input_request_is_missing(self) -> None:
        response = self.client.post(
            reverse("resolve_input_request", kwargs={"input_id": 999_999}),
            data={"answers": json.dumps({"scope": "UI"})},
        )

        self.assertEqual(response.status_code, 404)

    def test_returns_409_when_update_loses_race(self) -> None:
        input_request = self._make_input_request()
        original_filter = UserInputRequest.objects.filter

        class _RacingUpdate:
            def update(self, **kwargs: Any) -> int:
                original_filter(pk=input_request.pk).update(
                    response={"answers": {"scope": "already answered"}}
                )
                return 0

        def _filter(*args: Any, **kwargs: Any) -> Any:
            if kwargs == {"pk": input_request.pk, "response__isnull": True}:
                return _RacingUpdate()
            return original_filter(*args, **kwargs)

        with patch.object(UserInputRequest.objects, "filter", side_effect=_filter):
            response = self.client.post(
                reverse("resolve_input_request", kwargs={"input_id": input_request.pk}),
                data={"answers": json.dumps({"scope": "UI"})},
            )

        self.assertEqual(response.status_code, 409)
        input_request.refresh_from_db()
        self.assertEqual(
            input_request.response,
            {"answers": {"scope": "already answered"}},
        )

    def test_string_representation_reflects_response_state(self) -> None:
        pending = self._make_input_request()
        answered = self._make_input_request(response={"answers": {"scope": "UI"}})

        self.assertIn("state=pending", str(pending))
        self.assertIn("state=answered", str(answered))


class SessionViewApprovalContextTests(TestCase):
    """The session detail view exposes POST URL templates for live
    browser prompts. Pin them so a URL refactor can't quietly break the
    streaming approval or structured-input loops."""

    @patch("hitch.main.views.Codex")
    def test_session_template_renders_prompt_url_templates(
        self, mock_codex: MagicMock
    ) -> None:
        ctx: MagicMock = mock_codex.return_value.__enter__.return_value
        ctx._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(
                id="thread-1",
                cwd="/repo",
                name="Demo",
                preview="",
                turns=[],
                path=None,
                updated_at=1,
            )
        )
        # The approval-url template only renders inside the
        # ``active_worker`` block (an idle session has no SSE stream and so
        # no client-side approval prompts to wire up).
        CodexInstance.objects.create(
            pid=1,
            thread_id="thread-1",
            cwd="/repo",
            prompt="hi",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
        )

        response = self.client.get(
            reverse("session", kwargs={"session_id": "thread-1"})
        )

        self.assertEqual(response.status_code, 200)
        # The placeholder pk is ``0`` — the JS swaps it for the real
        # ApprovalRequest id when posting a decision.
        self.assertContains(
            response,
            'data-approval-url-template="' + reverse(
                "resolve_approval", kwargs={"approval_id": 0}
            ),
        )
        self.assertContains(
            response,
            'data-input-url-template="' + reverse(
                "resolve_input_request", kwargs={"input_id": 0}
            ),
        )


class StandingOrderViewTests(TestCase):
    @patch("hitch.main.views.Codex")
    def test_page_lists_inbox_and_orders_for_selected_project(
        self, mock_codex: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        other_project = Project.objects.create(name="Other", repo_path="/other")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        order = StandingOrder.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        StandingOrder.objects.create(
            project=other_project,
            title="Other order",
            goal="Should not render.",
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo",
            project=project,
        )
        ProposedSession.objects.create(
            standing_order=order,
            title="Add parser coverage",
            summary="This adds focused parser coverage.",
            confidence=StandingOrder.CONFIDENCE_HIGH,
            relevant_files=["hitch/main/rollout.py"],
            candidate_session=candidate,
        )

        response = self.client.get(reverse("standing_orders"))

        self.assertEqual(response.status_code, 200)
        body = response.content.decode()
        nav_start = body.index('<nav class="primary-nav"')
        nav_end = body.index("</nav>", nav_start)
        nav_html = body[nav_start:nav_end]
        self.assertIn(
            f'href="{reverse("standing_orders")}" aria-current="page"', nav_html
        )
        self.assertIn(">standing orders</a>", nav_html)
        self.assertIn(f'href="{reverse("okrs")}"', nav_html)
        self.assertContains(response, "--accent-soft")
        self.assertContains(response, "--shadow-lg")
        self.assertContains(response, "Improve tests")
        self.assertContains(response, "Add parser coverage")
        self.assertContains(response, "This adds focused parser coverage.")
        self.assertContains(response, "hitch/main/rollout.py")
        self.assertNotContains(response, "Other order")

    def test_create_standing_order_for_selected_project(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))

        response = self.client.post(
            reverse("create_standing_order"),
            {
                "title": "Improve tests",
                "goal": "Find useful test coverage increments.",
                "confidence_threshold": StandingOrder.CONFIDENCE_VERY_HIGH,
            },
        )

        self.assertEqual(response.status_code, 302)
        order = StandingOrder.objects.get()
        self.assertEqual(order.project, project)
        self.assertEqual(order.title, "Improve tests")
        self.assertEqual(
            order.confidence_threshold,
            StandingOrder.CONFIDENCE_VERY_HIGH,
        )

    @patch("hitch.main.views.system_agents.start_standing_order_workflow")
    def test_run_all_starts_each_selected_project_order(
        self, mock_start: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        other_project = Project.objects.create(name="Other", repo_path="/other")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        first = StandingOrder.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        second = StandingOrder.objects.create(
            project=project,
            title="Improve docs",
            goal="Find useful docs increments.",
        )
        StandingOrder.objects.create(
            project=other_project,
            title="Other order",
            goal="Should not run.",
        )

        response = self.client.post(reverse("run_standing_orders"))

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            [call.kwargs["standing_order"] for call in mock_start.call_args_list],
            [first, second],
        )

    def test_reject_proposed_session_requires_reason(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        order = StandingOrder.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        proposal = ProposedSession.objects.create(
            standing_order=order,
            title="Add parser coverage",
        )

        response = self.client.post(
            reverse("update_proposed_session_outcome", args=[proposal.pk]),
            {"outcome_status": ProposedSession.OUTCOME_REJECTED},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.content, b"reason is required")

    def test_accept_proposed_session_links_candidate_session(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        order = StandingOrder.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo",
            project=project,
        )
        proposal = ProposedSession.objects.create(
            standing_order=order,
            candidate_session=candidate,
            title="Add parser coverage",
        )

        response = self.client.post(
            reverse("update_proposed_session_outcome", args=[proposal.pk]),
            {"outcome_status": ProposedSession.OUTCOME_ACCEPTED},
        )

        self.assertEqual(response.status_code, 302)
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_ACCEPTED)
        self.assertEqual(proposal.accepted_session, candidate)
