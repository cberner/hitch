from unittest.mock import MagicMock, patch

from django.test import TestCase
from django.urls import reverse


class SetSessionNameViewTests(TestCase):
    @patch("hitch.main.views.Codex")
    def test_updates_name_and_redirects(self, mock_codex: MagicMock) -> None:
        response = self.client.post(
            reverse("set_session_name", kwargs={"session_id": "abc"}),
            data={"name": "New title"},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            response.headers["Location"],
            reverse("session", kwargs={"session_id": "abc"}),
        )
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_set_name.assert_called_once_with("abc", "New title")

    @patch("hitch.main.views.Codex")
    def test_trims_whitespace_before_saving(self, mock_codex: MagicMock) -> None:
        response = self.client.post(
            reverse("set_session_name", kwargs={"session_id": "abc"}),
            data={"name": "   Spaced   "},
        )

        self.assertEqual(response.status_code, 302)
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_set_name.assert_called_once_with("abc", "Spaced")

    @patch("hitch.main.views.Codex")
    def test_rejects_empty_name(self, mock_codex: MagicMock) -> None:
        response = self.client.post(
            reverse("set_session_name", kwargs={"session_id": "abc"}),
            data={"name": ""},
        )

        self.assertEqual(response.status_code, 400)
        mock_codex.assert_not_called()

    @patch("hitch.main.views.Codex")
    def test_rejects_whitespace_only_name(self, mock_codex: MagicMock) -> None:
        response = self.client.post(
            reverse("set_session_name", kwargs={"session_id": "abc"}),
            data={"name": "   "},
        )

        self.assertEqual(response.status_code, 400)
        mock_codex.assert_not_called()

    @patch("hitch.main.views.Codex")
    def test_rejects_overly_long_name(self, mock_codex: MagicMock) -> None:
        # The form caps input client-side; the view enforces the same bound
        # so a hand-crafted POST can't bypass it.
        response = self.client.post(
            reverse("set_session_name", kwargs={"session_id": "abc"}),
            data={"name": "x" * 201},
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
