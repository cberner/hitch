import os
import tempfile
from unittest.mock import patch

from django.test import TestCase, tag
from django.urls import reverse


@tag("integration")
class IndexViewIntegrationTests(TestCase):
    """Drive the index view against a real `codex app-server` subprocess.

    Requires the `codex` binary on PATH; the CI workflow installs it via npm.
    Each test runs with an empty CODEX_HOME so the listing is deterministic.
    """

    def test_index_renders_empty_session_list(self) -> None:
        with (
            tempfile.TemporaryDirectory(prefix="codex-test-") as codex_home,
            patch.dict(os.environ, {"CODEX_HOME": codex_home}),
        ):
            response = self.client.get(reverse("index"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No sessions found.")
