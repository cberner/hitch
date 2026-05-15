from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.test import TestCase
from django.urls import reverse


class NewSessionViewTests(TestCase):
    def _allowed_repo(self) -> str:
        return "/home/user/proj"

    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_spawns_worker_and_redirects(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self._allowed_repo())]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "Refactor the login flow", "cwd": self._allowed_repo()},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            response.headers["Location"],
            reverse("session", kwargs={"session_id": "thread-xyz"}),
        )
        mock_spawn.assert_called_once_with(
            cwd=self._allowed_repo(), prompt="Refactor the login flow"
        )

    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_rejects_missing_prompt(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self._allowed_repo())]

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "", "cwd": self._allowed_repo()},
        )

        self.assertEqual(response.status_code, 400)
        mock_spawn.assert_not_called()

    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_rejects_missing_cwd(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self._allowed_repo())]

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "hello", "cwd": ""},
        )

        self.assertEqual(response.status_code, 400)
        mock_spawn.assert_not_called()

    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_rejects_unknown_cwd(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self._allowed_repo())]

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "hello", "cwd": "/etc"},
        )

        self.assertEqual(response.status_code, 400)
        mock_spawn.assert_not_called()

    @patch("hitch.main.views.discover_repos")
    def test_new_session_rejects_get(self, mock_discover: MagicMock) -> None:
        mock_discover.return_value = []
        response = self.client.get(reverse("new_session"))
        self.assertEqual(response.status_code, 405)
