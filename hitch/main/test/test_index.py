from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.test import TestCase
from django.urls import reverse


class IndexViewTests(TestCase):
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_index_renders_hitch(self, mock_codex: MagicMock, mock_discover: MagicMock) -> None:
        mock_codex.return_value.__enter__.return_value.thread_list.return_value.data = []
        mock_discover.return_value = []
        response = self.client.get(reverse("index"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "HITCH")
        self.assertContains(response, "No sessions found.")
        self.assertContains(response, "New session")

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_index_lists_sessions(self, mock_codex: MagicMock, mock_discover: MagicMock) -> None:
        session = SimpleNamespace(
            id="abc123",
            name="Refactor login flow",
            preview="Let's start refactoring the login flow.",
            cwd="/home/user/proj",
            updated_at=1234567890,
        )
        mock_codex.return_value.__enter__.return_value.thread_list.return_value.data = [session]
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Refactor login flow")
        self.assertContains(response, "abc123")
        self.assertContains(response, "/home/user/proj")
        self.assertContains(response, reverse("session", kwargs={"session_id": "abc123"}))

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_index_sorts_sessions_by_updated_at_descending(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        older = SimpleNamespace(
            id="older",
            name="Older session",
            preview="",
            cwd="/home/user/proj",
            updated_at=1000,
        )
        newer = SimpleNamespace(
            id="newer",
            name="Newer session",
            preview="",
            cwd="/home/user/proj",
            updated_at=2000,
        )
        middle = SimpleNamespace(
            id="middle",
            name="Middle session",
            preview="",
            cwd="/home/user/proj",
            updated_at=1500,
        )
        mock_codex.return_value.__enter__.return_value.thread_list.return_value.data = [
            older,
            newer,
            middle,
        ]
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        body = response.content.decode()
        newer_pos = body.index("Newer session")
        middle_pos = body.index("Middle session")
        older_pos = body.index("Older session")
        self.assertLess(newer_pos, middle_pos)
        self.assertLess(middle_pos, older_pos)

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_index_populates_repo_dropdown(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        from pathlib import Path

        mock_codex.return_value.__enter__.return_value.thread_list.return_value.data = []
        mock_discover.return_value = [Path("/home/user/proj-a"), Path("/home/user/proj-b")]

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "/home/user/proj-a")
        self.assertContains(response, "/home/user/proj-b")
        self.assertContains(response, 'name="cwd"')

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_index_shows_empty_repo_message(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        mock_codex.return_value.__enter__.return_value.thread_list.return_value.data = []
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No git repositories found")
