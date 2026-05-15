import os
import shutil
import tempfile
from unittest.mock import patch

from django.test import TestCase, tag
from django.urls import reverse
from openai_codex import AppServerConfig, Codex

from hitch.main import codex_pool


@tag("integration")
class SessionViewIntegrationTests(TestCase):
    """End-to-end coverage of the session detail view against a real
    `codex app-server` subprocess.

    Requires the `codex` binary on PATH plus a local ollama instance with
    `qwen2.5-coder:0.5b` pulled; CI installs both. Tests run with a fresh
    CODEX_HOME so the listing and rollout state are deterministic.
    """

    def test_session_view_loads_thread_persisted_to_disk(self) -> None:
        """Render the session page for a thread that lives only on disk.

        Reproduces the "thread not loaded" failure the new-session button
        produced: ``new_session`` started a thread in one transient codex
        subprocess and redirected to ``/sessions/<id>/``, where the view
        opened a *different* transient subprocess that had no thread in
        its in-memory map. ``thread/read`` requires a loaded thread, so the
        view has to call ``thread/resume`` (which reads the rollout off
        disk) before it can render the page. This test creates the thread
        in one Codex context, runs a turn through ollama, exits the
        context, and then asks the view to render the result — the GET
        only succeeds when the view loads the thread from disk.
        """
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
                thread_id = thread.id
                result = thread.run("Hi")
            assert result.final_response is not None
            self.assertGreater(len(result.final_response), 0)

            response = self.client.get(
                reverse("session", kwargs={"session_id": thread_id})
            )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, thread_id)

    def test_session_view_loads_thread_before_any_turn_runs(self) -> None:
        """Render the session page immediately after thread/start, no turn.

        This is the exact race the "new session" flow hits: ``new_session``
        starts the thread and redirects to ``/sessions/<id>/`` before the
        detached worker has had a chance to submit any prompt. ``thread/start``
        on its own defers rollout materialisation, so an in-process
        ``thread/resume`` from the session view's *next* Codex subprocess
        fails with "no rollout found for thread id" unless the new-session
        flow has forced the rollout file to be written. Verifies that the
        codex_pool wires up that materialisation step on every spawn.
        """
        with (
            tempfile.TemporaryDirectory(prefix="codex-test-") as codex_home,
            patch.dict(os.environ, {"CODEX_HOME": codex_home}),
            patch(
                "hitch.main.codex_pool._launch_worker_process",
                return_value=type("_Stub", (), {"pid": 1})(),
            ),
        ):
            instance = codex_pool.spawn_new_session(
                cwd=os.getcwd(),
                prompt="What time is it",
            )

            response = self.client.get(
                reverse("session", kwargs={"session_id": instance.thread_id})
            )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, instance.thread_id)
