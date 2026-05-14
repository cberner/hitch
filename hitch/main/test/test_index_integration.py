import os
import shutil
import tempfile
from unittest.mock import patch

from django.test import TestCase, tag
from django.urls import reverse
from openai_codex import AppServerConfig, Codex


@tag("integration")
class IndexViewIntegrationTests(TestCase):
    """End-to-end coverage against a real `codex app-server` subprocess.

    Requires the `codex` binary on PATH plus a local ollama instance with
    `qwen2.5-coder:0.5b` pulled; CI installs both. Tests run with a fresh
    CODEX_HOME so the listing and rollout state are deterministic.
    """

    def test_index_renders_empty_session_list(self) -> None:
        with (
            tempfile.TemporaryDirectory(prefix="codex-test-") as codex_home,
            patch.dict(os.environ, {"CODEX_HOME": codex_home}),
        ):
            response = self.client.get(reverse("index"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No sessions found.")

    def test_sdk_runs_turn_via_ollama(self) -> None:
        with (
            tempfile.TemporaryDirectory(prefix="codex-test-") as codex_home,
            patch.dict(os.environ, {"CODEX_HOME": codex_home}),
        ):
            config = AppServerConfig(codex_bin=shutil.which("codex"))
            with Codex(config=config) as codex:
                thread = codex.thread_start(
                    model="qwen2.5-coder:0.5b",
                    model_provider="ollama",
                )
                result = thread.run("Hi")
        assert result.final_response is not None
        self.assertGreater(len(result.final_response), 0)
