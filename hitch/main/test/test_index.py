from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from django.test import TestCase
from django.urls import reverse


def _setup_codex(
    mock_codex: MagicMock,
    *,
    threads: list[Any] | None = None,
    models: list[Any] | None = None,
) -> MagicMock:
    """Configure the Codex mock with the calls the index view makes.

    The view now reads both ``thread_list`` and ``models``; both attributes
    must be set or the view's ``list(codex.models().data)`` will iterate a
    bare MagicMock and explode.
    """
    ctx: MagicMock = mock_codex.return_value.__enter__.return_value
    ctx.thread_list.return_value.data = threads or []
    ctx.models.return_value.data = models or []
    return ctx


class IndexViewTests(TestCase):
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_index_renders_hitch(self, mock_codex: MagicMock, mock_discover: MagicMock) -> None:
        _setup_codex(mock_codex)
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
        _setup_codex(mock_codex, threads=[session])
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
        _setup_codex(mock_codex, threads=[older, newer, middle])
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

        _setup_codex(mock_codex)
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
        _setup_codex(mock_codex)
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No git repositories found")

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_index_truncates_long_preview(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        # Unnamed threads fall back to `preview` (the full first user
        # message), which is often a long paragraph. The list row must clip
        # it so the title stays compact.
        long_text = "x" * 200
        session = SimpleNamespace(
            id="sess",
            name=None,
            preview=long_text,
            cwd="/repo",
            updated_at=1,
        )
        _setup_codex(mock_codex, threads=[session])
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "x" * 80 + "...")
        # The untruncated 200-char preview must not leak through.
        self.assertNotContains(response, "x" * 120)

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_index_collapses_multiline_preview_to_first_line(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        session = SimpleNamespace(
            id="sess",
            name=None,
            preview="first line\nsecond line\nthird line",
            cwd="/repo",
            updated_at=1,
        )
        _setup_codex(mock_codex, threads=[session])
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "first line")
        self.assertNotContains(response, "second line")

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_index_uses_name_when_set(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        # A user-set name wins over the preview fallback.
        session = SimpleNamespace(
            id="sess",
            name="Short title",
            preview="ignored long preview " * 20,
            cwd="/repo",
            updated_at=1,
        )
        _setup_codex(mock_codex, threads=[session])
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Short title")
        self.assertNotContains(response, "ignored long preview")

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_index_falls_back_to_id_when_no_title_text(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        session = SimpleNamespace(
            id="bare-id",
            name=None,
            preview="",
            cwd="/repo",
            updated_at=1,
        )
        _setup_codex(mock_codex, threads=[session])
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, ">bare-id<")
