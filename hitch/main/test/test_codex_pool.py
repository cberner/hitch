import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.test import TestCase, override_settings

from hitch.main import codex_pool
from hitch.main.models import CodexInstance


class SpawnNewSessionTests(TestCase):
    @patch("hitch.main.codex_pool._launch_worker_process")
    @patch("hitch.main.codex_pool.Codex")
    def test_creates_thread_then_spawns_worker(
        self, mock_codex: MagicMock, mock_launch: MagicMock
    ) -> None:
        mock_codex.return_value.__enter__.return_value.thread_start.return_value = (
            SimpleNamespace(id="thread-abc")
        )
        mock_launch.return_value = SimpleNamespace(pid=4242)

        with (
            tempfile.TemporaryDirectory() as events_dir,
            override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
        ):
            instance = codex_pool.spawn_new_session(cwd="/repo", prompt="hi")

        self.assertEqual(instance.thread_id, "thread-abc")
        self.assertEqual(instance.cwd, "/repo")
        self.assertEqual(instance.prompt, "hi")
        self.assertEqual(instance.pid, 4242)
        self.assertEqual(instance.status, CodexInstance.STATUS_STARTING)
        self.assertTrue(instance.events_path.endswith(f"{instance.pk}.jsonl"))
        mock_codex.return_value.__enter__.return_value.thread_start.assert_called_once_with(
            cwd="/repo"
        )
        # Worker subprocess only receives the row id; prompt is read from the
        # row to avoid argparse misinterpreting prompts that begin with '-'.
        mock_launch.assert_called_once_with(instance_id=instance.pk)


class SpawnLaunchFailureTests(TestCase):
    @patch("hitch.main.codex_pool._launch_worker_process")
    def test_marks_instance_failed_when_launch_raises(
        self, mock_launch: MagicMock
    ) -> None:
        mock_launch.side_effect = OSError("boom")

        with (
            tempfile.TemporaryDirectory() as events_dir,
            override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
            self.assertRaises(OSError),
        ):
            codex_pool.spawn_turn(thread_id="t", cwd="/repo", prompt="hi")

        # The exception propagates to the caller, but the row is left in a
        # terminal state so it isn't treated as forever-pending.
        instance = CodexInstance.objects.latest("started_at")
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertEqual(instance.pid, 0)
        self.assertIn("boom", instance.error)
        self.assertIsNotNone(instance.ended_at)


class SpawnTurnTests(TestCase):
    @patch("hitch.main.codex_pool._launch_worker_process")
    def test_resumes_existing_thread_without_calling_codex(
        self, mock_launch: MagicMock
    ) -> None:
        """``spawn_turn`` operates entirely on a known thread id — it must
        never reach out to the Codex app-server, which keeps it cheap on the
        request-serving path."""
        mock_launch.return_value = SimpleNamespace(pid=1234)

        with (
            tempfile.TemporaryDirectory() as events_dir,
            override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
        ):
            instance = codex_pool.spawn_turn(
                thread_id="thread-xyz", cwd="/repo", prompt="follow-up"
            )

        self.assertEqual(instance.thread_id, "thread-xyz")
        self.assertEqual(instance.prompt, "follow-up")
        self.assertEqual(instance.pid, 1234)
        mock_launch.assert_called_once_with(instance_id=instance.pk)


class LaunchWorkerProcessTests(TestCase):
    @patch("hitch.main.codex_pool.subprocess.Popen")
    def test_launches_manage_command_in_new_session(self, mock_popen: MagicMock) -> None:
        mock_popen.return_value = SimpleNamespace(pid=999)

        codex_pool._launch_worker_process(instance_id=7)

        args, kwargs = mock_popen.call_args
        argv = args[0]
        self.assertEqual(argv[1].endswith("manage.py"), True)
        self.assertEqual(argv[2], "codex_worker")
        self.assertIn("--instance-id", argv)
        self.assertEqual(argv[argv.index("--instance-id") + 1], "7")
        # The prompt is *not* passed on the command line — it lives on the
        # CodexInstance row so a leading '-' can't be reinterpreted as an
        # argparse option.
        self.assertNotIn("--prompt", argv)
        # The worker must outlive the Django process, so it gets its own
        # session and all stdio is redirected to /dev/null.
        self.assertTrue(kwargs["start_new_session"])
        from subprocess import DEVNULL

        self.assertEqual(kwargs["stdin"], DEVNULL)
        self.assertEqual(kwargs["stdout"], DEVNULL)
        self.assertEqual(kwargs["stderr"], DEVNULL)
        self.assertTrue(kwargs["close_fds"])
        # Settings module is propagated so the child can bootstrap Django.
        self.assertEqual(
            kwargs["env"]["DJANGO_SETTINGS_MODULE"],
            "hitch.settings.dev",
        )


class IsAliveTests(TestCase):
    def test_current_process_is_alive(self) -> None:
        self.assertTrue(codex_pool.is_alive(os.getpid()))

    def test_pid_zero_is_dead(self) -> None:
        self.assertFalse(codex_pool.is_alive(0))

    def test_negative_pid_is_dead(self) -> None:
        self.assertFalse(codex_pool.is_alive(-1))

    def test_definitely_unused_pid_is_dead(self) -> None:
        # 2**22 is well above the default pid_max on Linux/macOS, so it cannot
        # match a running process.
        self.assertFalse(codex_pool.is_alive(2**22))

    @patch("hitch.main.codex_pool.os.kill")
    def test_permission_error_means_alive(self, mock_kill: MagicMock) -> None:
        # ``os.kill(pid, 0)`` raises PermissionError when the pid exists but is
        # owned by another user; the process is still alive in that case.
        mock_kill.side_effect = PermissionError
        self.assertTrue(codex_pool.is_alive(1234))

    @patch("hitch.main.codex_pool.os.kill")
    def test_other_os_error_is_treated_as_dead(self, mock_kill: MagicMock) -> None:
        mock_kill.side_effect = OSError
        self.assertFalse(codex_pool.is_alive(1234))


class CodexInstanceStrTests(TestCase):
    def test_str_includes_pid_thread_and_status(self) -> None:
        instance = CodexInstance(
            pid=42,
            thread_id="abc",
            cwd="/r",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
        )
        rendered = str(instance)
        self.assertIn("pid=42", rendered)
        self.assertIn("thread_id=abc", rendered)
        self.assertIn("status=running", rendered)


class ReconcileDeadTests(TestCase):
    def _make(self, pid: int, status: str) -> CodexInstance:
        return CodexInstance.objects.create(
            pid=pid,
            thread_id="t",
            cwd="/r",
            events_path="/dev/null",
            status=status,
        )

    @patch("hitch.main.codex_pool.is_alive")
    def test_marks_only_dead_pending_rows_failed(self, mock_alive: MagicMock) -> None:
        dead_running = self._make(pid=10, status=CodexInstance.STATUS_RUNNING)
        live_running = self._make(pid=11, status=CodexInstance.STATUS_RUNNING)
        completed = self._make(pid=12, status=CodexInstance.STATUS_COMPLETED)
        mock_alive.side_effect = lambda pid: pid == 11

        n = codex_pool.reconcile_dead()

        self.assertEqual(n, 1)
        dead_running.refresh_from_db()
        live_running.refresh_from_db()
        completed.refresh_from_db()
        self.assertEqual(dead_running.status, CodexInstance.STATUS_FAILED)
        self.assertIsNotNone(dead_running.ended_at)
        self.assertEqual(live_running.status, CodexInstance.STATUS_RUNNING)
        self.assertIsNone(live_running.ended_at)
        self.assertEqual(completed.status, CodexInstance.STATUS_COMPLETED)
        self.assertIn("exited", dead_running.error)


class LookupTests(TestCase):
    def _make(self, thread_id: str) -> CodexInstance:
        return CodexInstance.objects.create(
            pid=1,
            thread_id=thread_id,
            cwd="/r",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
        )

    def test_list_for_thread_returns_newest_first(self) -> None:
        first = self._make("t1")
        second = self._make("t1")
        self._make("t2")

        result = codex_pool.list_for_thread("t1")

        self.assertEqual([r.pk for r in result], [second.pk, first.pk])

    def test_latest_for_thread_returns_newest(self) -> None:
        self._make("t1")
        newest = self._make("t1")

        self.assertEqual(codex_pool.latest_for_thread("t1"), newest)

    def test_latest_for_thread_returns_none_when_missing(self) -> None:
        self.assertIsNone(codex_pool.latest_for_thread("nothing"))


class EventsDirTests(TestCase):
    def test_uses_setting_when_configured(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_EVENTS_DIR=Path(raw)),
        ):
            self.assertEqual(codex_pool.events_dir(), Path(raw))

    def test_falls_back_to_home_dir(self) -> None:
        with override_settings(CODEX_EVENTS_DIR=None):
            self.assertEqual(
                codex_pool.events_dir(),
                Path.home() / ".hitch" / "codex_events",
            )
