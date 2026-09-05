"""End-to-end coverage against a real `codex app-server` subprocess.

Persistence tests run without a model, credentials, or network access. Tests
with the integration tag require local Ollama with `qwen2.5-coder:0.5b` pulled.
All tests use a fresh CODEX_HOME; Hitch selects the bundled or newer PATH runtime.
"""

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from django.test import TestCase, override_settings, tag
from django.urls import reverse
from openai_codex import Codex, CodexConfig

from hitch.main.models import CodexInstance
from hitch.main.runtime import codex_pool
from hitch.main.sessions.session_settings import (
    _effective_sandbox_policy_for_cwd,
    _stored_settings,
)


@contextmanager
def _fresh_codex_home() -> Iterator[None]:
    with (
        tempfile.TemporaryDirectory(prefix="codex-test-") as codex_home,
        patch.dict(os.environ, {"CODEX_HOME": codex_home}),
    ):
        # Marketplace sync can outlive app-server shutdown and race temp cleanup.
        Path(codex_home, "config.toml").write_text(
            "[features]\nplugins = false\n", encoding="utf-8"
        )
        yield


def _start_codex() -> Codex:
    return Codex(config=CodexConfig())


class CodexPersistenceTests(TestCase):
    def test_session_page_does_not_claim_worker_writer(self) -> None:
        with (
            _fresh_codex_home(),
            patch(
                "hitch.main.runtime.codex_pool._launch_worker_process",
                return_value=SimpleNamespace(pid=1),
            ),
            patch("hitch.main.runtime.codex_pool.worker_is_alive", return_value=True),
            patch("hitch.main.views.common._models_for_plan_mode_fallback", return_value=[]),
        ):
            instance = codex_pool.spawn_new_session(cwd=os.getcwd(), prompt="Hi")
            url = reverse("session", kwargs={"session_id": instance.thread_id})
            with (
                Codex(config=codex_pool.app_server_config()) as reader,
                patch(
                    "hitch.main.runtime.app_server_pool.run_borrowed_op_with_retry",
                    side_effect=lambda _factory, operation, **_kwargs: operation(reader),
                ),
            ):
                self.assertEqual(self.client.get(url).status_code, 200)
                with Codex(config=codex_pool.app_server_config()) as worker:
                    worker._client.thread_resume(instance.thread_id)
                    self.assertEqual(self.client.get(url).status_code, 200)
                instance.status = CodexInstance.STATUS_FAILED
                instance.error = "Startup failed"
                instance.save(update_fields=["status", "error"])
                response = self.client.get(url)
                self.assertEqual(response.status_code, 200)
                self.assertContains(response, "Startup failed")
                self.assertEqual(response.context["pending_user_prompt"], "Hi")

    def test_new_threads_resume_in_another_process_before_first_turn(self) -> None:
        # This needs a real runtime but no model, credentials, or network access.
        for spawn_worker in (False, True):
            with (
                self.subTest(spawn_worker=spawn_worker),
                _fresh_codex_home(),
                patch(
                    "hitch.main.runtime.codex_pool._launch_worker_process",
                    return_value=SimpleNamespace(pid=1),
                ),
            ):
                if spawn_worker:
                    instance = codex_pool.spawn_new_session(cwd=os.getcwd(), prompt="Hi")
                    thread_id = instance.thread_id
                    thread_path = codex_pool.thread_path_for_instance(instance)
                else:
                    thread_id, thread_path = codex_pool.create_session_thread_with_path(
                        cwd=os.getcwd(), name="Hi"
                    )

                self.assertTrue(thread_path)
                self.assertTrue(Path(thread_path).is_file())
                with Codex(config=codex_pool.app_server_config()) as codex:
                    resumed = codex._client.thread_resume(thread_id)
                    self.assertEqual(resumed.thread.id, thread_id)
                    self.assertEqual(resumed.thread.turns, [])


@tag("integration")
class CodexIntegrationTests(TestCase):
    def test_index_renders_empty_session_list(self) -> None:
        with _fresh_codex_home():
            response = self.client.get(reverse("index"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No sessions found.")

    def test_sdk_runs_turn_via_ollama(self) -> None:
        with _fresh_codex_home(), _start_codex() as codex:
            thread = codex.thread_start(
                model="qwen2.5-coder:0.5b",
                model_provider="ollama",
            )
            result = thread.run("Hi")
        assert result.final_response is not None
        self.assertGreater(len(result.final_response), 0)

    def test_session_view_loads_thread_persisted_to_disk(self) -> None:
        """Render the session page for a thread that lives only on disk.

        Reproduces the "thread not loaded" failure the new-session button
        produced: ``new_session`` started a thread in one transient codex
        subprocess and redirected to ``/sessions/<id>/``, where the view
        opened a *different* transient subprocess that had no thread in
        its in-memory map. ``thread/read`` requires a loaded thread, so the
        view has to call ``thread/resume`` (which reads the rollout off
        disk) before it can render the page.
        """
        with _fresh_codex_home():
            with _start_codex() as codex:
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

        ``thread/start`` on its own defers rollout materialisation, so an
        in-process ``thread/resume`` from the session view's *next* Codex
        subprocess fails with "no rollout found for thread id" unless the
        new-session flow has forced the rollout file to be written.
        Verifies that codex_pool wires up that materialisation step on
        every spawn.
        """
        with (
            _fresh_codex_home(),
            patch(
                "hitch.main.runtime.codex_pool._launch_worker_process",
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

    def test_plan_approval_spawns_default_collaboration_worker(self) -> None:
        """Plan approval must enqueue a default collaboration turn.

        This exercises the real view and ``codex_pool.spawn_turn`` path, while
        stubbing only the final detached subprocess launch. That is the boundary
        where Hitch must preserve the SDK mode switch for the worker.
        """
        repo = os.getcwd()
        with _fresh_codex_home():
            with _start_codex() as codex:
                thread = codex.thread_start(cwd=repo)
                thread_id = thread.id
                codex._client.thread_set_name(thread_id, "Plan approval integration")
                expected_model = codex._client.thread_resume(thread_id).model

            with (
                tempfile.TemporaryDirectory(prefix="hitch-events-") as events_dir,
                override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
                patch("hitch.main.repos.discover_repos", return_value=[Path(repo)]),
                patch(
                    "hitch.main.runtime.codex_pool._launch_worker_process",
                    return_value=SimpleNamespace(pid=4321),
                ) as mock_launch,
            ):
                response = self.client.post(
                    reverse("send_message", kwargs={"session_id": thread_id}),
                    data={
                        "prompt": "Implement the plan.",
                        "collaboration_mode": "default",
                    },
                )

        self.assertEqual(response.status_code, 302)
        instance = CodexInstance.objects.get(thread_id=thread_id)
        self.assertEqual(instance.prompt, "Implement the plan.")
        self.assertEqual(instance.cwd, repo)
        self.assertEqual(instance.pid, 4321)
        expected_sandbox = _effective_sandbox_policy_for_cwd(
            _stored_settings(response.wsgi_request), repo
        )
        mock_launch.assert_called_once_with(
            instance_id=instance.pk,
            model=expected_model,
            reasoning_effort=None,
            sandbox_policy=expected_sandbox or None,
            approval_mode="auto_review",
            collaboration_mode="default",
        )
