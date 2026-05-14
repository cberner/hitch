from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.test import TestCase
from django.urls import reverse


class IndexViewTests(TestCase):
    @patch("hitch.main.views.Codex")
    def test_index_renders_hitch(self, mock_codex: MagicMock) -> None:
        mock_codex.return_value.__enter__.return_value.thread_list.return_value.data = []
        response = self.client.get(reverse("index"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "HITCH")
        self.assertContains(response, "No sessions found.")

    @patch("hitch.main.views.Codex")
    def test_index_lists_sessions(self, mock_codex: MagicMock) -> None:
        session = SimpleNamespace(
            id="abc123",
            name="Refactor login flow",
            preview="Let's start refactoring the login flow.",
            cwd="/home/user/proj",
            updated_at=1234567890,
        )
        mock_codex.return_value.__enter__.return_value.thread_list.return_value.data = [session]

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Refactor login flow")
        self.assertContains(response, "abc123")
        self.assertContains(response, "/home/user/proj")
