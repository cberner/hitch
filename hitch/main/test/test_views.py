"""View-layer tests: index, new_session, send_message, set_session_name,
session_stream.

Shared helpers configure the Codex mock and seed signed cookies so each
test stays focused on the behavior under examination.
"""

import base64
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from django.core import signing
from django.test import Client, TestCase
from django.urls import reverse
from openai_codex.errors import MethodNotFoundError

from hitch.main.models import ApprovalRequest, CodexInstance, UserInputRequest
from hitch.main.worktrees import (
    ManagedWorktree,
    WorktreeCleanupError,
    WorktreeCreationError,
)

_SHOW_ARCHIVED_COOKIE = "hitch_show_archived_sessions"
_MODEL_COOKIE = "hitch_model"
_EXTRA_SYSTEM_PROMPT_COOKIE = "hitch_extra_system_prompt"
_USE_WORKTREES_COOKIE = "hitch_use_worktrees"
_PR_PROMPT = "Do a thorough review of the diff. Clean it up, and then open a PR"


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
        self.assertContains(response, 'data-session-archived="false"')
        self.assertContains(response, "data-archive-undo")
        self.assertLess(body.index("Newer session"), body.index("Middle session"))
        self.assertLess(body.index("Middle session"), body.index("Older session"))

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
        self.assertLess(body.index("Archived session"), body.index("Active session"))
        client = mock_codex.return_value.__enter__.return_value
        client.thread_list.assert_any_call()
        client.thread_list.assert_any_call(archived=True)

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_populates_repo_dropdown(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _setup_codex(mock_codex)
        mock_discover.return_value = [Path("/home/user/proj-a"), Path("/home/user/proj-b")]

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "/home/user/proj-a")
        self.assertContains(response, "/home/user/proj-b")
        self.assertContains(response, 'name="cwd"')

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
        self.assertContains(response, "parseNewSessionPlanCommand")
        self.assertContains(response, "parseNewSessionPrCommand")
        self.assertContains(response, 'toLowerCase() !== "/plan"')

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

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_pr_slash_command_starts_new_session_with_review_prompt(
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
            data={"prompt": "/PR", "cwd": self.REPO, "plan_mode": "true"},
        )

        self.assertEqual(response.status_code, 302)
        mock_spawn.assert_called_once_with(
            cwd=self.REPO,
            prompt=_PR_PROMPT,
            developer_instructions=None,
            model=None,
            reasoning_effort=None,
            sandbox_policy=None,
            approval_mode="auto_review",
        )

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
        models: list[Any] | None = None,
    ) -> None:
        client = mock_codex.return_value.__enter__.return_value
        resumed = SimpleNamespace(thread=SimpleNamespace(cwd=cwd))
        if model is not None:
            resumed.model = model
        client._client.thread_resume.return_value = resumed
        client.models.return_value.data = models or []

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
    def test_default_collaboration_mode_switches_to_default_mode(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex, model="gpt-5.4")
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={
                "prompt": "Implement the plan.",
                "collaboration_mode": "default",
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_pr_slash_command_sends_review_prompt(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex, model="gpt-5.4")
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "/pr", "plan_mode": "true"},
        )

        self.assertEqual(response.status_code, 302)
        mock_spawn.assert_called_once_with(
            thread_id="abc",
            cwd="/repo",
            prompt=_PR_PROMPT,
            sandbox_policy=None,
            approval_mode="auto_review",
        )

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
    def test_updates_name_and_redirects(self, mock_codex: MagicMock) -> None:
        response = self.client.post(
            reverse("set_session_name", kwargs={"session_id": "abc"}),
            data={"name": "  New title  "},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            response.headers["Location"],
            reverse("session", kwargs={"session_id": "abc"}),
        )
        # Whitespace is trimmed before saving.
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_set_name.assert_called_once_with("abc", "New title")

    @patch("hitch.main.views.Codex")
    def test_rejects_invalid_input(self, mock_codex: MagicMock) -> None:
        # The form caps input client-side; the view enforces the same bounds
        # so a hand-crafted POST can't bypass them.
        cases = [
            ({"name": ""}, "empty"),
            ({"name": "   "}, "whitespace only"),
            ({"name": "x" * 201}, "over length cap"),
        ]
        for data, label in cases:
            with self.subTest(label=label):
                response = self.client.post(
                    reverse("set_session_name", kwargs={"session_id": "abc"}),
                    data=data,
                )
                self.assertEqual(response.status_code, 400)
        mock_codex.assert_not_called()

    @patch("hitch.main.views.Codex")
    def test_rejects_get(self, mock_codex: MagicMock) -> None:
        response = self.client.get(
            reverse("set_session_name", kwargs={"session_id": "abc"})
        )

        self.assertEqual(response.status_code, 405)
        mock_codex.assert_not_called()


class SetSessionArchivedViewTests(TestCase):
    @patch("hitch.main.views.Codex")
    def test_archives_session_and_redirects_to_index(
        self, mock_codex: MagicMock
    ) -> None:
        response = self.client.post(
            reverse("set_session_archived", kwargs={"session_id": "abc"}),
            data={"archived": "true"},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], reverse("index"))
        client = mock_codex.return_value.__enter__.return_value
        client.thread_archive.assert_called_once_with("abc")
        client.thread_unarchive.assert_not_called()

    @patch("hitch.main.views.Codex")
    def test_ajax_archive_returns_no_content(self, mock_codex: MagicMock) -> None:
        response = self.client.post(
            reverse("set_session_archived", kwargs={"session_id": "abc"}),
            data={"archived": "true"},
            HTTP_X_REQUESTED_WITH="XMLHttpRequest",
        )

        self.assertEqual(response.status_code, 204)
        self.assertNotIn("Location", response.headers)
        client = mock_codex.return_value.__enter__.return_value
        client.thread_archive.assert_called_once_with("abc")

    @patch("hitch.main.views.Codex")
    def test_unarchives_session_and_redirects_to_session(
        self, mock_codex: MagicMock
    ) -> None:
        response = self.client.post(
            reverse("set_session_archived", kwargs={"session_id": "abc"}),
            data={"archived": "false"},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            response.headers["Location"],
            reverse("session", kwargs={"session_id": "abc"}),
        )
        client = mock_codex.return_value.__enter__.return_value
        client.thread_unarchive.assert_called_once_with("abc")
        client.thread_archive.assert_not_called()

    @patch("hitch.main.views.Codex")
    def test_rejects_invalid_input(self, mock_codex: MagicMock) -> None:
        for data in ({}, {"archived": ""}, {"archived": "yes"}):
            with self.subTest(data=data):
                response = self.client.post(
                    reverse("set_session_archived", kwargs={"session_id": "abc"}),
                    data=data,
                )
                self.assertEqual(response.status_code, 400)
        mock_codex.assert_not_called()

    @patch("hitch.main.views.Codex")
    def test_rejects_get(self, mock_codex: MagicMock) -> None:
        response = self.client.get(
            reverse("set_session_archived", kwargs={"session_id": "abc"})
        )

        self.assertEqual(response.status_code, 405)
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

    @patch("hitch.main.views.codex_pool.interrupt_instance")
    @patch("hitch.main.views.codex_pool.interrupt_active")
    def test_falls_back_to_latest_active_without_instance(
        self,
        mock_interrupt_active: MagicMock,
        mock_interrupt_instance: MagicMock,
    ) -> None:
        # Older cached page (or a direct curl POST) won't carry the
        # instance field; fall back to "latest active worker for this
        # thread" so the stop click still has a chance to do something.
        response = self.client.post(
            reverse("stop_session", kwargs={"session_id": "abc"})
        )

        self.assertEqual(response.status_code, 302)
        mock_interrupt_active.assert_called_once_with("abc")
        mock_interrupt_instance.assert_not_called()

    @patch("hitch.main.views.codex_pool.interrupt_instance")
    def test_rejects_non_integer_instance(
        self, mock_interrupt_instance: MagicMock
    ) -> None:
        response = self.client.post(
            reverse("stop_session", kwargs={"session_id": "abc"}),
            data={"instance": "not-a-number"},
        )

        self.assertEqual(response.status_code, 400)
        mock_interrupt_instance.assert_not_called()

    @patch("hitch.main.views.codex_pool.interrupt_instance")
    def test_rejects_out_of_range_instance(
        self, mock_interrupt_instance: MagicMock
    ) -> None:
        # Tampered/oversized values must be rejected at the view
        # boundary so they never reach ``objects.get`` (which would
        # raise backend-specific OverflowError/DataError and surface
        # as a 500 instead of a clean 400).
        cases = [
            ("0", "zero"),
            ("-1", "negative"),
            (str(2**63), "above BigAutoField max"),
        ]
        for value, label in cases:
            with self.subTest(label=label):
                response = self.client.post(
                    reverse("stop_session", kwargs={"session_id": "abc"}),
                    data={"instance": value},
                )
                self.assertEqual(response.status_code, 400)
        mock_interrupt_instance.assert_not_called()

    @patch("hitch.main.views.codex_pool.interrupt_active")
    def test_no_active_worker_still_redirects(
        self, mock_interrupt: MagicMock
    ) -> None:
        # A double-click after the agent already finished must not 500.
        mock_interrupt.return_value = None

        response = self.client.post(
            reverse("stop_session", kwargs={"session_id": "abc"})
        )

        self.assertEqual(response.status_code, 302)

    @patch("hitch.main.views.codex_pool.interrupt_active")
    def test_rejects_get(self, mock_interrupt: MagicMock) -> None:
        response = self.client.get(
            reverse("stop_session", kwargs={"session_id": "abc"})
        )

        self.assertEqual(response.status_code, 405)
        mock_interrupt.assert_not_called()


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
        self, session_id: str, *, baseline: str = "", active: str = ""
    ) -> str:
        # Helper that builds the SSE URL with the page-render-time state
        # the view expects on every legitimate request. Tests that want
        # to exercise the stale-reload path pass an empty/wrong value.
        return (
            reverse("session_stream", kwargs={"session_id": session_id})
            + f"?baseline={baseline}&active={active}"
        )

    @patch("hitch.main.streaming._IDLE_MAX_STREAM_SECONDS", 0.001)
    @patch("hitch.main.streaming._IDLE_POLL_INTERVAL", 0.001)
    def test_returns_idle_heartbeat_stream_when_no_worker(self) -> None:
        # Without an active worker the SSE channel stays open emitting
        # heartbeat events with ``working: false`` so the page's connection
        # indicator can show ``connected, idle``. The cap is patched down
        # so the test doesn't sit in the recycle loop.
        response = self.client.get(self._stream_url("thread-1"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/event-stream")
        body = b"".join(response.streaming_content)  # type: ignore[attr-defined]
        self.assertIn(b"event: heartbeat", body)
        self.assertIn(b'"working": false', body)

    @patch("hitch.main.streaming._IDLE_MAX_STREAM_SECONDS", 0.001)
    @patch("hitch.main.streaming._IDLE_POLL_INTERVAL", 0.001)
    def test_idle_stream_when_only_completed_worker_exists(self) -> None:
        # A terminal worker counts as ``no active worker`` for routing
        # purposes — the idle heartbeat stream is what we serve so the
        # connection indicator stays accurate without re-tailing the old
        # events file. The baseline reflects the page's render-time view
        # (the completed worker's pk, no active).
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
    def test_sets_no_buffering_headers(self) -> None:
        # SSE needs frame-by-frame delivery; proxies (and Django's own
        # middleware stack) honour these headers to disable coalescing.
        response = self.client.get(self._stream_url("thread-1"))
        self.assertEqual(response["Cache-Control"], "no-cache")
        self.assertEqual(response["X-Accel-Buffering"], "no")
        b"".join(response.streaming_content)  # type: ignore[attr-defined]

    def test_reloads_when_worker_appeared_after_page_render(self) -> None:
        # The classic out-of-band-spawn race: page rendered with no
        # worker (empty baseline / active), but by the time SSE opens a
        # worker has shown up in the DB. The endpoint must reload the
        # page so the DOM gets the live-streaming UI before any item
        # events start arriving.
        self._make(thread_id="thread-1", status=CodexInstance.STATUS_RUNNING)
        response = self.client.get(self._stream_url("thread-1"))
        body = b"".join(response.streaming_content)  # type: ignore[attr-defined]
        self.assertIn(b'"status": "stale"', body)

    def test_reloads_when_active_worker_completed_before_sse_opens(self) -> None:
        # Inverse race: page rendered expecting a live worker (passes
        # ``active=N`` and ``baseline=N``) but by the time SSE opens the
        # worker has gone terminal. Without the reload the page would
        # show a permanent "Codex is working…" pill and a stale pending
        # bubble for the just-completed turn.
        inst = self._make(thread_id="thread-1", status=CodexInstance.STATUS_COMPLETED)
        response = self.client.get(
            self._stream_url("thread-1", baseline=str(inst.pk), active=str(inst.pk))
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

    def test_records_decision_and_marks_decided_at(self) -> None:
        approval = self._make_approval()

        response = self.client.post(
            reverse("resolve_approval", kwargs={"approval_id": approval.pk}),
            data={"decision": "accept"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"accept")
        approval.refresh_from_db()
        self.assertEqual(approval.decision, "accept")
        self.assertIsNotNone(approval.decided_at)

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
                approval.refresh_from_db()
                self.assertEqual(approval.decision, decision)

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

    def test_rejects_invalid_decision(self) -> None:
        """A POST with a value outside the app-server-accepted set must 400
        rather than poison the row — the worker would otherwise round-trip
        the bogus string into a JSON-RPC response codex rejects."""
        approval = self._make_approval()

        response = self.client.post(
            reverse("resolve_approval", kwargs={"approval_id": approval.pk}),
            data={"decision": "yes please"},
        )

        self.assertEqual(response.status_code, 400)
        approval.refresh_from_db()
        self.assertEqual(approval.decision, "")

    def test_returns_404_for_missing_approval(self) -> None:
        response = self.client.post(
            reverse("resolve_approval", kwargs={"approval_id": 99999999}),
            data={"decision": "accept"},
        )

        self.assertEqual(response.status_code, 404)

    def test_returns_409_when_already_resolved(self) -> None:
        """Two browser tabs racing the same approval must not silently
        clobber each other. The first POST wins; later POSTs get 409 so
        the UI knows the choice is locked in."""
        approval = self._make_approval(decision="accept")

        response = self.client.post(
            reverse("resolve_approval", kwargs={"approval_id": approval.pk}),
            data={"decision": "decline"},
        )

        self.assertEqual(response.status_code, 409)
        approval.refresh_from_db()
        self.assertEqual(approval.decision, "accept")

    def test_rejects_get(self) -> None:
        approval = self._make_approval()
        response = self.client.get(
            reverse("resolve_approval", kwargs={"approval_id": approval.pk})
        )
        self.assertEqual(response.status_code, 405)


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

    def test_records_answers_and_marks_responded_at(self) -> None:
        input_request = self._make_input_request()

        response = self.client.post(
            reverse("resolve_input_request", kwargs={"input_id": input_request.pk}),
            data={"answers": json.dumps({"scope": "Management command"})},
        )

        self.assertEqual(response.status_code, 200)
        input_request.refresh_from_db()
        self.assertEqual(
            input_request.response,
            {"answers": {"scope": "Management command"}},
        )
        self.assertIsNotNone(input_request.responded_at)

    def test_records_empty_answers_when_payload_is_omitted(self) -> None:
        input_request = self._make_input_request()

        response = self.client.post(
            reverse("resolve_input_request", kwargs={"input_id": input_request.pk})
        )

        self.assertEqual(response.status_code, 200)
        input_request.refresh_from_db()
        self.assertEqual(input_request.response, {"answers": {}})

    def test_trims_string_answers_and_ignores_blank_keys(self) -> None:
        input_request = self._make_input_request()

        response = self.client.post(
            reverse("resolve_input_request", kwargs={"input_id": input_request.pk}),
            data={"answers": json.dumps({" scope ": " UI ", " ": "ignored"})},
        )

        self.assertEqual(response.status_code, 200)
        input_request.refresh_from_db()
        self.assertEqual(input_request.response, {"answers": {"scope": "UI"}})

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

    def test_accepts_structured_answer_values(self) -> None:
        input_request = self._make_input_request()
        answers = {
            "scope": ["UI", "CLI"],
            "details": {"choice": "Other", "notes": ["keep history"]},
            "confirmed": True,
            "priority": 2,
            "optional": None,
        }

        response = self.client.post(
            reverse("resolve_input_request", kwargs={"input_id": input_request.pk}),
            data={"answers": json.dumps(answers)},
        )

        self.assertEqual(response.status_code, 200)
        input_request.refresh_from_db()
        self.assertEqual(input_request.response, {"answers": answers})

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
