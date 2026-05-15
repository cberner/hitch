from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.test import TestCase
from django.urls import reverse


class SendMessageViewTests(TestCase):
    def _patch_codex(self, mock_codex: MagicMock, *, cwd: str | None = "/repo") -> None:
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
            data={"prompt": "follow-up question"},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            response.headers["Location"],
            reverse("session", kwargs={"session_id": "abc"}),
        )
        mock_spawn.assert_called_once_with(
            thread_id="abc", cwd="/repo", prompt="follow-up question"
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
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(cwd=SimpleNamespace(root="/repo"))
        )
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "hi"},
        )

        self.assertEqual(response.status_code, 302)
        mock_spawn.assert_called_once_with(thread_id="abc", cwd="/repo", prompt="hi")

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_trims_whitespace_before_spawning(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex)
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "   hi   "},
        )

        self.assertEqual(response.status_code, 302)
        mock_spawn.assert_called_once_with(thread_id="abc", cwd="/repo", prompt="hi")

    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_rejects_empty_prompt(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
    ) -> None:
        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": ""},
        )

        self.assertEqual(response.status_code, 400)
        mock_codex.assert_not_called()
        mock_spawn.assert_not_called()

    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_rejects_whitespace_only_prompt(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
    ) -> None:
        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "   \n  "},
        )

        self.assertEqual(response.status_code, 400)
        mock_codex.assert_not_called()
        mock_spawn.assert_not_called()

    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_rejects_thread_without_cwd(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex, cwd=None)

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "hi"},
        )

        self.assertEqual(response.status_code, 400)
        mock_spawn.assert_not_called()

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_rejects_cwd_outside_discovered_repos(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        # The session list shows every thread the app-server knows about, so
        # a resumed thread's cwd can point outside the discover_repos()
        # allowlist (e.g. for threads created by another tool). The composer
        # must refuse to spawn a worker in such a directory.
        self._patch_codex(mock_codex, cwd="/etc")
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "hi"},
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
