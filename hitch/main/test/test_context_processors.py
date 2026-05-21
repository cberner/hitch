import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import override
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase, override_settings

from hitch.main import context_processors


class ServerVersionContextTests(SimpleTestCase):
    @override
    def setUp(self) -> None:
        context_processors._server_git_hash.cache_clear()

    @override
    def tearDown(self) -> None:
        context_processors._server_git_hash.cache_clear()

    @override_settings(BASE_DIR=Path("/srv/hitch"))
    @patch("hitch.main.context_processors.subprocess.run")
    def test_server_git_hash_uses_first_six_head_digits(self, mock_run: MagicMock) -> None:
        mock_run.return_value = SimpleNamespace(stdout="abcdef123456\n")

        self.assertEqual(context_processors._server_git_hash(), "abcdef")
        mock_run.assert_called_once_with(
            ["git", "-C", "/srv/hitch", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=1,
        )

    @patch("hitch.main.context_processors.subprocess.run")
    def test_server_git_hash_is_empty_when_git_is_unavailable(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.CalledProcessError(1, ["git"])

        self.assertEqual(context_processors._server_git_hash(), "")
