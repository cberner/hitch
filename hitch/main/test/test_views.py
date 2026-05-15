"""View-layer tests: index, new_session, send_message, set_session_name.

Shared helpers configure the Codex mock and seed signed cookies so each
test stays focused on the behavior under examination.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from django.core import signing
from django.test import Client, TestCase
from django.urls import reverse
from openai_codex.errors import MethodNotFoundError


def _setup_codex(
    mock_codex: MagicMock,
    *,
    threads: list[Any] | None = None,
    models: list[Any] | None = None,
) -> MagicMock:
    """Configure the Codex mock with both ``thread_list`` and ``models``;
    the index view reads both. Also stubs ``_client.request`` to raise
    MethodNotFound so the rate-limits fetch falls through its
    unsupported-endpoint branch — tests that care set their own value."""
    ctx: MagicMock = mock_codex.return_value.__enter__.return_value
    ctx.thread_list.return_value.data = threads or []
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


def _seed_cookies(client: Client, **values: str) -> None:
    for name, value in values.items():
        client.cookies[name] = _sign(name, value)


def _session(
    session_id: str = "sess",
    *,
    name: str | None = None,
    preview: str = "",
    cwd: str = "/repo",
    updated_at: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=session_id, name=name, preview=preview, cwd=cwd, updated_at=updated_at
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
        self.assertLess(body.index("Newer session"), body.index("Middle session"))
        self.assertLess(body.index("Middle session"), body.index("Older session"))

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
            model=None,
            reasoning_effort=None,
            sandbox_policy=None,
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
            model="gpt-5",
            reasoning_effort="high",
            sandbox_policy="workspaceWrite",
        )

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
            model="gpt-5",
            reasoning_effort="medium",
            sandbox_policy=None,
        )

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
    def _patch_codex(self, mock_codex: MagicMock, *, cwd: object = "/repo") -> None:
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(cwd=cwd)
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
            thread_id="abc", cwd="/repo", prompt="hi", sandbox_policy=None
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
