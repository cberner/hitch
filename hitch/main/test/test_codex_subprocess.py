"""Coverage for the detached Codex worker plumbing: spawning, launching the
subprocess, the worker management command, and the bookkeeping that keeps
the CodexInstance row in sync with the OS process.
"""

import dataclasses
import json
import os
import signal
import tempfile
from collections.abc import Iterator
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, override
from unittest.mock import MagicMock, patch

from django.core.management import call_command
from django.test import TestCase, override_settings
from django.utils import timezone
from openai_codex import ApprovalMode
from openai_codex.generated.v2_all import (
    ApprovalsReviewer,
    AskForApprovalValue,
    DangerFullAccessSandboxPolicy,
    ReasoningEffort,
    SandboxPolicy,
    Turn,
    TurnCompletedNotification,
    TurnError,
    TurnStartParams,
    TurnStatus,
    WorkspaceWriteSandboxPolicy,
)
from pydantic import BaseModel

from hitch.main import codex_pool, streaming
from hitch.main.management.commands import codex_worker as codex_worker_module
from hitch.main.management.commands.codex_worker import (
    _make_approval_handler,
    _serialize_event,
)
from hitch.main.models import ApprovalRequest, CodexInstance


def _events_dir() -> tempfile.TemporaryDirectory[str]:
    return tempfile.TemporaryDirectory()


def _stub_codex_thread_start(mock_codex: MagicMock, thread_id: str = "t") -> MagicMock:
    codex: MagicMock = mock_codex.return_value.__enter__.return_value
    codex.thread_start.return_value = SimpleNamespace(id=thread_id)
    return codex


def _completed_event(turn_id: str, status: TurnStatus, error_message: str | None = None) -> SimpleNamespace:
    """Build a turn/completed event whose payload is a real
    TurnCompletedNotification — the worker's status logic narrows on the
    SDK type, not on duck-typed shapes, so the test must use the real model.
    """
    return SimpleNamespace(
        method="turn/completed",
        payload=TurnCompletedNotification(
            thread_id="thread-1",
            turn=Turn(
                id=turn_id,
                items=[],
                status=status,
                error=TurnError(message=error_message) if error_message else None,
            ),
        ),
    )


def _stub_thread_resume(events: list[SimpleNamespace], turn_id: str = "turn-1") -> object:
    """Return an object shaped like ``thread_resume(...).turn(...).stream()``."""
    return SimpleNamespace(
        turn=lambda _input: SimpleNamespace(id=turn_id, stream=lambda: iter(events)),
    )


class SpawnNewSessionTests(TestCase):
    @patch("hitch.main.codex_pool._launch_worker_process")
    @patch("hitch.main.codex_pool.Codex")
    def test_creates_thread_then_spawns_worker(
        self, mock_codex: MagicMock, mock_launch: MagicMock
    ) -> None:
        codex = _stub_codex_thread_start(mock_codex, "thread-abc")
        mock_launch.return_value = SimpleNamespace(pid=4242)

        with (
            _events_dir() as events_dir,
            override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
        ):
            instance = codex_pool.spawn_new_session(cwd="/repo", prompt="hi")

        self.assertEqual(instance.thread_id, "thread-abc")
        self.assertEqual(instance.cwd, "/repo")
        self.assertEqual(instance.prompt, "hi")
        self.assertEqual(instance.pid, 4242)
        self.assertEqual(instance.status, CodexInstance.STATUS_STARTING)
        self.assertTrue(instance.events_path.endswith(f"{instance.pk}.jsonl"))
        # ``model=None`` means "fall back to whatever Codex's config picks".
        codex.thread_start.assert_called_once_with(cwd="/repo", model=None)
        # ``thread/start`` defers writing the rollout file to disk, so the
        # cross-process ``thread/resume`` the worker and the session view both
        # rely on would fail with "no rollout found" without an explicit
        # metadata write to materialise the rollout. ``thread/set-name`` is the
        # cheapest such write; it must happen inside the same Codex context as
        # ``thread/start`` so the in-memory thread is still loaded.
        codex._client.thread_set_name.assert_called_once_with("thread-abc", "hi")
        # Worker subprocess only receives the row id and (when set) the
        # reasoning effort, sandbox policy, and approval mode; the prompt is
        # read from the row to avoid argparse misinterpreting prompts that
        # begin with '-'.
        mock_launch.assert_called_once_with(
            instance_id=instance.pk,
            reasoning_effort=None,
            sandbox_policy=None,
            approval_mode=None,
        )

    @patch("hitch.main.codex_pool._launch_worker_process")
    @patch("hitch.main.codex_pool.Codex")
    def test_initial_thread_name_derivation(
        self, mock_codex: MagicMock, mock_launch: MagicMock
    ) -> None:
        """The initial name is the first line of the prompt, trimmed of
        whitespace, clipped to 200 chars; whitespace-only prompts fall back
        to a static placeholder rather than being passed through verbatim
        (Codex rejects whitespace-only names)."""
        codex = _stub_codex_thread_start(mock_codex)
        mock_launch.return_value = SimpleNamespace(pid=1)

        cases = [
            ("  Refactor the parser \nthen write tests\n", "Refactor the parser"),
            ("a" * 500, "a" * 200),
            ("   \n\t  ", "New session"),
        ]
        for prompt, expected in cases:
            with self.subTest(prompt=prompt[:30]):
                codex._client.thread_set_name.reset_mock()
                with (
                    _events_dir() as events_dir,
                    override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
                ):
                    codex_pool.spawn_new_session(cwd="/repo", prompt=prompt)
                codex._client.thread_set_name.assert_called_once_with("t", expected)

    @patch("hitch.main.codex_pool._launch_worker_process")
    @patch("hitch.main.codex_pool.Codex")
    def test_forwards_model_effort_sandbox_and_approval(
        self, mock_codex: MagicMock, mock_launch: MagicMock
    ) -> None:
        """The settings dialog's model selector flows into
        ``thread_start(model=...)``; effort, sandbox policy and approval
        mode flow into the worker as CLI args. Pin every wiring so a
        refactor can't quietly drop one of them.
        """
        codex = _stub_codex_thread_start(mock_codex)
        mock_launch.return_value = SimpleNamespace(pid=1)

        with (
            _events_dir() as events_dir,
            override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
        ):
            instance = codex_pool.spawn_new_session(
                cwd="/repo",
                prompt="hi",
                model="gpt-5",
                reasoning_effort="high",
                sandbox_policy="workspaceWrite",
                approval_mode="deny_all",
            )

        codex.thread_start.assert_called_once_with(cwd="/repo", model="gpt-5")
        mock_launch.assert_called_once_with(
            instance_id=instance.pk,
            reasoning_effort="high",
            sandbox_policy="workspaceWrite",
            approval_mode="deny_all",
        )


class SpawnFailureTests(TestCase):
    @patch("hitch.main.codex_pool._launch_worker_process")
    def test_marks_instance_failed_when_launch_raises(
        self, mock_launch: MagicMock
    ) -> None:
        mock_launch.side_effect = OSError("boom")

        with (
            _events_dir() as events_dir,
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
            _events_dir() as events_dir,
            override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
        ):
            instance = codex_pool.spawn_turn(
                thread_id="thread-xyz", cwd="/repo", prompt="follow-up"
            )

        self.assertEqual(instance.thread_id, "thread-xyz")
        self.assertEqual(instance.prompt, "follow-up")
        self.assertEqual(instance.pid, 1234)
        mock_launch.assert_called_once_with(
            instance_id=instance.pk,
            reasoning_effort=None,
            sandbox_policy=None,
            approval_mode=None,
        )


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
        self.assertEqual(
            kwargs["env"]["DJANGO_SETTINGS_MODULE"],
            "hitch.settings.dev",
        )
        # No effort passed → no --reasoning-effort on the CLI; Codex's own
        # default takes over inside the worker.
        self.assertNotIn("--reasoning-effort", argv)

    @patch("hitch.main.codex_pool.subprocess.Popen")
    def test_forwards_reasoning_effort_as_cli_arg(self, mock_popen: MagicMock) -> None:
        mock_popen.return_value = SimpleNamespace(pid=999)

        codex_pool._launch_worker_process(instance_id=7, reasoning_effort="high")

        argv = mock_popen.call_args.args[0]
        self.assertIn("--reasoning-effort", argv)
        self.assertEqual(argv[argv.index("--reasoning-effort") + 1], "high")

    @patch("hitch.main.codex_pool.subprocess.Popen")
    def test_forwards_sandbox_policy_as_cli_arg(self, mock_popen: MagicMock) -> None:
        mock_popen.return_value = SimpleNamespace(pid=999)

        codex_pool._launch_worker_process(
            instance_id=7, sandbox_policy="workspaceWrite"
        )

        argv = mock_popen.call_args.args[0]
        self.assertIn("--sandbox-policy", argv)
        self.assertEqual(
            argv[argv.index("--sandbox-policy") + 1], "workspaceWrite"
        )

    @patch("hitch.main.codex_pool.subprocess.Popen")
    def test_forwards_approval_mode_as_cli_arg(self, mock_popen: MagicMock) -> None:
        mock_popen.return_value = SimpleNamespace(pid=999)

        codex_pool._launch_worker_process(instance_id=7, approval_mode="deny_all")

        argv = mock_popen.call_args.args[0]
        self.assertIn("--approval-mode", argv)
        self.assertEqual(argv[argv.index("--approval-mode") + 1], "deny_all")


class IsAliveTests(TestCase):
    def test_known_pid_states(self) -> None:
        self.assertTrue(codex_pool.is_alive(os.getpid()))
        self.assertFalse(codex_pool.is_alive(0))
        self.assertFalse(codex_pool.is_alive(-1))
        # 2**22 is well above the default pid_max on Linux/macOS.
        self.assertFalse(codex_pool.is_alive(2**22))

    @patch("hitch.main.codex_pool.os.kill")
    def test_permission_error_means_alive(self, mock_kill: MagicMock) -> None:
        # ``os.kill(pid, 0)`` raises PermissionError when the pid exists but
        # is owned by another user; the process is still alive in that case.
        mock_kill.side_effect = PermissionError
        self.assertTrue(codex_pool.is_alive(1234))

    @patch("hitch.main.codex_pool.os.kill")
    def test_other_os_error_is_treated_as_dead(self, mock_kill: MagicMock) -> None:
        mock_kill.side_effect = OSError
        self.assertFalse(codex_pool.is_alive(1234))


class CodexInstanceModelTests(TestCase):
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


class ApprovalRequestModelTests(TestCase):
    def test_str_surfaces_method_and_pending_state(self) -> None:
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="t",
            cwd="/r",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
        )
        approval = ApprovalRequest.objects.create(
            instance=instance,
            method="item/commandExecution/requestApproval",
            params={"item": {"command": "ls"}},
        )
        rendered = str(approval)
        self.assertIn(f"pk={approval.pk}", rendered)
        self.assertIn("method=item/commandExecution/requestApproval", rendered)
        # Empty decision renders as "pending" so a glance at logs/admin
        # makes the row's open state obvious.
        self.assertIn("decision=pending", rendered)

        approval.decision = "approved"
        approval.save(update_fields=["decision"])
        self.assertIn("decision=approved", str(approval))


class ReconcileAndLookupTests(TestCase):
    def _make(self, *, pid: int = 1, thread_id: str = "t", status: str | None = None) -> CodexInstance:
        return CodexInstance.objects.create(
            pid=pid,
            thread_id=thread_id,
            cwd="/r",
            events_path="/dev/null",
            status=status or CodexInstance.STATUS_COMPLETED,
        )

    @patch("hitch.main.codex_pool.is_alive")
    def test_reconcile_marks_only_dead_pending_rows_failed(
        self, mock_alive: MagicMock
    ) -> None:
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

    def test_list_and_latest_for_thread(self) -> None:
        first = self._make(thread_id="t1")
        second = self._make(thread_id="t1")
        self._make(thread_id="t2")

        self.assertEqual(
            [r.pk for r in codex_pool.list_for_thread("t1")],
            [second.pk, first.pk],
        )
        self.assertEqual(codex_pool.latest_for_thread("t1"), second)
        self.assertIsNone(codex_pool.latest_for_thread("nothing"))

    def test_latest_active_for_thread(self) -> None:
        # ``send_message`` can stack workers on a thread, so a newer
        # terminal row must not mask an older still-running one — the
        # streaming UI needs to stay up as long as *any* worker for the
        # thread is in progress.
        older = self._make(thread_id="t-active", status=CodexInstance.STATUS_RUNNING)
        newer_terminal = self._make(
            thread_id="t-active", status=CodexInstance.STATUS_FAILED
        )
        self.assertEqual(codex_pool.latest_active_for_thread("t-active"), older)

        # When multiple actives exist, return the newest by started_at.
        newer_active = self._make(
            thread_id="t-active", status=CodexInstance.STATUS_STARTING
        )
        self.assertEqual(codex_pool.latest_active_for_thread("t-active"), newer_active)

        # All terminal → None; never-existed thread → None.
        older.status = CodexInstance.STATUS_COMPLETED
        older.save(update_fields=["status"])
        newer_active.status = CodexInstance.STATUS_COMPLETED
        newer_active.save(update_fields=["status"])
        self.assertIsNone(codex_pool.latest_active_for_thread("t-active"))
        self.assertIsNone(codex_pool.latest_active_for_thread("never-existed"))
        # newer_terminal was already terminal; referenced here so it isn't
        # flagged as unused by future readers.
        self.assertEqual(newer_terminal.status, CodexInstance.STATUS_FAILED)


    def test_latest_id_for_thread(self) -> None:
        # ``latest_id_for_thread`` is the baseline the idle SSE stream
        # uses to spot any out-of-band turn — including fast turns that
        # complete between two polls. It must return the highest pk for
        # the thread regardless of status and ``None`` when the thread
        # has never had a worker.
        self.assertIsNone(codex_pool.latest_id_for_thread("never-existed"))
        first = self._make(thread_id="t-id", status=CodexInstance.STATUS_RUNNING)
        self.assertEqual(codex_pool.latest_id_for_thread("t-id"), first.pk)
        second = self._make(thread_id="t-id", status=CodexInstance.STATUS_COMPLETED)
        self.assertEqual(codex_pool.latest_id_for_thread("t-id"), second.pk)
        # Workers on other threads must not bleed into this thread's pk.
        self._make(thread_id="other", status=CodexInstance.STATUS_RUNNING)
        self.assertEqual(codex_pool.latest_id_for_thread("t-id"), second.pk)


class InterruptActiveTests(TestCase):
    def _make(
        self,
        *,
        pid: int = 1,
        thread_id: str = "t",
        status: str = CodexInstance.STATUS_RUNNING,
    ) -> CodexInstance:
        return CodexInstance.objects.create(
            pid=pid,
            thread_id=thread_id,
            cwd="/r",
            events_path="/dev/null",
            status=status,
        )

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.killpg")
    @patch("hitch.main.codex_pool.os.kill")
    def test_first_stop_sends_sigterm_and_records_timestamp(
        self,
        mock_kill: MagicMock,
        mock_killpg: MagicMock,
        mock_identity: MagicMock,
    ) -> None:
        # Polite interrupt: SIGTERM is sent to the worker pid alone (not
        # the group) so the worker's handler can call the SDK's
        # ``turn.interrupt()``. The row's status is left for the worker
        # to update — flipping it here would be overwritten when the
        # worker's stream loop finishes and saves its own terminal state.
        instance = self._make(pid=4321, status=CodexInstance.STATUS_RUNNING)

        result = codex_pool.interrupt_active("t")

        self.assertIsNotNone(result)
        mock_kill.assert_called_once_with(4321, signal.SIGTERM)
        mock_killpg.assert_not_called()
        mock_identity.assert_called_once_with(4321, instance.pk)
        instance.refresh_from_db()
        # Status untouched — worker writes it when the SDK interrupt
        # surfaces as a turn/completed event with status=interrupted.
        self.assertEqual(instance.status, CodexInstance.STATUS_RUNNING)
        self.assertEqual(instance.error, "")
        # Timestamp recorded so the next click can detect "polite
        # already issued" and escalate to SIGKILL.
        self.assertIsNotNone(instance.interrupt_requested_at)

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.killpg")
    @patch("hitch.main.codex_pool.os.kill")
    def test_second_stop_escalates_to_sigkill(
        self,
        mock_kill: MagicMock,
        mock_killpg: MagicMock,
        mock_identity: MagicMock,
    ) -> None:
        # Worker didn't honour the polite stop; user clicks again.
        # Escalate to SIGKILL on the whole process group (so the codex
        # app-server child dies with the worker) and write status
        # ourselves since the worker no longer has the chance to.
        instance = self._make(pid=4321, status=CodexInstance.STATUS_RUNNING)
        CodexInstance.objects.filter(pk=instance.pk).update(
            interrupt_requested_at=timezone.now()
        )

        result = codex_pool.interrupt_active("t")

        self.assertIsNotNone(result)
        mock_kill.assert_not_called()
        mock_killpg.assert_called_once_with(4321, signal.SIGKILL)
        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertEqual(instance.error, "forcibly stopped by user")
        self.assertIsNotNone(instance.ended_at)

    @patch("hitch.main.codex_pool.os.kill")
    @patch("hitch.main.codex_pool.os.killpg")
    def test_no_active_worker_is_a_noop(
        self, mock_killpg: MagicMock, mock_kill: MagicMock
    ) -> None:
        # An already-completed turn must not 500 the stop endpoint; the user
        # may have raced the agent finishing.
        self._make(pid=99, status=CodexInstance.STATUS_COMPLETED)

        self.assertIsNone(codex_pool.interrupt_active("t"))
        mock_kill.assert_not_called()
        mock_killpg.assert_not_called()

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=False)
    @patch("hitch.main.codex_pool.os.killpg")
    @patch("hitch.main.codex_pool.os.kill")
    def test_dead_or_recycled_pid_skips_signal_but_marks_row_failed(
        self,
        mock_kill: MagicMock,
        mock_killpg: MagicMock,
        mock_identity: MagicMock,
    ) -> None:
        # Identity check rejects the pid: no safe target for SIGTERM or
        # SIGKILL, but the leftover RUNNING row is still flipped to
        # failed so the UI exits streaming mode.
        instance = self._make(pid=12345, status=CodexInstance.STATUS_RUNNING)

        result = codex_pool.interrupt_active("t")

        self.assertIsNotNone(result)
        mock_kill.assert_not_called()
        mock_killpg.assert_not_called()
        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertEqual(instance.error, "interrupted by user")

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.kill")
    def test_worker_exit_between_identity_and_sigterm_marks_failed(
        self, mock_kill: MagicMock, mock_identity: MagicMock
    ) -> None:
        # TOCTOU: worker exited between the cmdline check and the signal.
        # ESRCH is tolerated and the leftover row is flipped to failed.
        mock_kill.side_effect = ProcessLookupError
        instance = self._make(pid=4321, status=CodexInstance.STATUS_RUNNING)

        result = codex_pool.interrupt_active("t")

        self.assertIsNotNone(result)
        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.kill")
    def test_eperm_from_sigterm_leaves_row_active(
        self, mock_kill: MagicMock, mock_identity: MagicMock
    ) -> None:
        # Real signal failure: the worker is still running. Don't claim
        # the turn was stopped — leave the row active so the user can
        # retry rather than seeing a phantom "failed" state.
        mock_kill.side_effect = PermissionError
        instance = self._make(pid=4321, status=CodexInstance.STATUS_RUNNING)

        result = codex_pool.interrupt_active("t")

        self.assertIsNone(result)
        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_RUNNING)
        self.assertEqual(instance.error, "")
        self.assertIsNone(instance.interrupt_requested_at)

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.killpg")
    def test_worker_exit_between_identity_and_sigkill_marks_failed(
        self, mock_killpg: MagicMock, mock_identity: MagicMock
    ) -> None:
        # Same TOCTOU as the SIGTERM case, but on the escalation path:
        # ESRCH from SIGKILL is tolerated and we still flip the row.
        mock_killpg.side_effect = ProcessLookupError
        instance = self._make(pid=4321, status=CodexInstance.STATUS_RUNNING)
        CodexInstance.objects.filter(pk=instance.pk).update(
            interrupt_requested_at=timezone.now()
        )

        result = codex_pool.interrupt_active("t")

        self.assertIsNotNone(result)
        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertEqual(instance.error, "forcibly stopped by user")

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.killpg")
    def test_completed_status_under_race_is_preserved_on_sigkill(
        self, mock_killpg: MagicMock, mock_identity: MagicMock
    ) -> None:
        # Escalation path: worker raced to completion between the row
        # read and the killpg. ``_mark_failed`` uses a conditional
        # UPDATE so the genuine completed status is preserved and the
        # helper returns None rather than rewriting it.
        instance = self._make(pid=4321, status=CodexInstance.STATUS_RUNNING)
        CodexInstance.objects.filter(pk=instance.pk).update(
            interrupt_requested_at=timezone.now()
        )

        def flip_to_completed(*_args: object, **_kwargs: object) -> None:
            CodexInstance.objects.filter(pk=instance.pk).update(
                status=CodexInstance.STATUS_COMPLETED
            )

        mock_killpg.side_effect = flip_to_completed

        result = codex_pool.interrupt_active("t")

        self.assertIsNone(result)
        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_COMPLETED)

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.killpg")
    def test_eperm_from_sigkill_leaves_row_active(
        self, mock_killpg: MagicMock, mock_identity: MagicMock
    ) -> None:
        # Same as above, but on the escalation path (second click): a
        # SIGKILL that we can't deliver must not flip the row to
        # failed, because the worker is still running.
        mock_killpg.side_effect = PermissionError
        instance = self._make(pid=4321, status=CodexInstance.STATUS_RUNNING)
        CodexInstance.objects.filter(pk=instance.pk).update(
            interrupt_requested_at=timezone.now()
        )

        result = codex_pool.interrupt_active("t")

        self.assertIsNone(result)
        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_RUNNING)
        self.assertEqual(instance.error, "")

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.kill")
    def test_completed_status_under_race_is_preserved_on_first_stop(
        self, mock_kill: MagicMock, mock_identity: MagicMock
    ) -> None:
        # The worker can legitimately finish in the window between the
        # row read and the signal. The first-click path no longer
        # writes a terminal status, but it does record a timestamp;
        # the timestamp write must not resurrect an already-terminal
        # row by re-enabling the Stop button.
        instance = self._make(pid=4321, status=CodexInstance.STATUS_RUNNING)

        def flip_to_completed(*_args: object, **_kwargs: object) -> None:
            CodexInstance.objects.filter(pk=instance.pk).update(
                status=CodexInstance.STATUS_COMPLETED
            )

        mock_kill.side_effect = flip_to_completed

        result = codex_pool.interrupt_active("t")

        # Helper returns the refreshed instance (with whatever state
        # the DB now reflects); the test cares that ``status`` is
        # preserved.
        self.assertIsNotNone(result)
        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_COMPLETED)
        self.assertEqual(instance.error, "")

    @patch("hitch.main.codex_pool._pid_is_our_worker")
    @patch("hitch.main.codex_pool.os.kill")
    @patch("hitch.main.codex_pool.os.killpg")
    def test_unset_pid_is_noop(
        self,
        mock_killpg: MagicMock,
        mock_kill: MagicMock,
        mock_identity: MagicMock,
    ) -> None:
        # The codex_worker subprocess may already be alive and will set
        # status=RUNNING any moment now. Flipping the row to failed
        # here would be silently undone, so the helper refuses.
        instance = self._make(pid=0, status=CodexInstance.STATUS_STARTING)

        result = codex_pool.interrupt_active("t")

        self.assertIsNone(result)
        mock_kill.assert_not_called()
        mock_killpg.assert_not_called()
        mock_identity.assert_not_called()
        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_STARTING)
        self.assertEqual(instance.error, "")


class InterruptInstanceTests(TestCase):
    """Targeted-interrupt entry point used by the Stop button.

    ``interrupt_instance`` differs from ``interrupt_active`` by stopping
    the *specific* worker the page is showing, not "whichever worker is
    latest at click time" — protecting against a stale tab aborting an
    overlapping newer turn.
    """

    def _make(
        self,
        *,
        pid: int = 1,
        thread_id: str = "t",
        status: str = CodexInstance.STATUS_RUNNING,
    ) -> CodexInstance:
        return CodexInstance.objects.create(
            pid=pid,
            thread_id=thread_id,
            cwd="/r",
            events_path="/dev/null",
            status=status,
        )

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.kill")
    def test_stops_specific_instance(
        self, mock_kill: MagicMock, mock_identity: MagicMock
    ) -> None:
        target = self._make(pid=4321, status=CodexInstance.STATUS_RUNNING)
        # A second, newer active worker that would be picked by
        # ``latest_active_for_thread`` is left untouched: the targeted
        # entry point must hit the exact instance the page asked for.
        bystander = self._make(pid=9999, status=CodexInstance.STATUS_RUNNING)

        result = codex_pool.interrupt_instance(target.pk, expected_thread_id="t")

        self.assertIsNotNone(result)
        # First-click path: polite SIGTERM to the worker pid only.
        mock_kill.assert_called_once_with(4321, signal.SIGTERM)
        target.refresh_from_db()
        bystander.refresh_from_db()
        # Status is left for the worker to update; the timestamp marks
        # that polite interrupt was issued so a re-click can escalate.
        self.assertIsNotNone(target.interrupt_requested_at)
        self.assertIsNone(bystander.interrupt_requested_at)
        self.assertEqual(bystander.status, CodexInstance.STATUS_RUNNING)

    def test_unknown_instance_returns_none(self) -> None:
        # A stale form value for a row that's been deleted (or never
        # existed) must not 500 the stop endpoint.
        self.assertIsNone(
            codex_pool.interrupt_instance(99999, expected_thread_id="t")
        )

    @patch("hitch.main.codex_pool.os.killpg")
    @patch("hitch.main.codex_pool.os.kill")
    def test_thread_id_mismatch_refuses(
        self, mock_kill: MagicMock, mock_killpg: MagicMock
    ) -> None:
        # A tampered/stale form post that targets a worker belonging to
        # a different thread must not stop it.
        instance = self._make(thread_id="other", status=CodexInstance.STATUS_RUNNING)

        result = codex_pool.interrupt_instance(instance.pk, expected_thread_id="t")

        self.assertIsNone(result)
        mock_kill.assert_not_called()
        mock_killpg.assert_not_called()
        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_RUNNING)

    @patch("hitch.main.codex_pool.os.killpg")
    @patch("hitch.main.codex_pool.os.kill")
    def test_already_terminal_returns_none(
        self, mock_kill: MagicMock, mock_killpg: MagicMock
    ) -> None:
        # The page may be stale; clicking Stop on a worker that already
        # finished must be a clean no-op, not an overwrite.
        instance = self._make(
            pid=4321, thread_id="t", status=CodexInstance.STATUS_COMPLETED
        )

        result = codex_pool.interrupt_instance(instance.pk, expected_thread_id="t")

        self.assertIsNone(result)
        mock_kill.assert_not_called()
        mock_killpg.assert_not_called()
        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_COMPLETED)


class PidIsOurWorkerTests(TestCase):
    """The cmdline-based identity guard that protects against PID reuse."""

    @patch("hitch.main.codex_pool.Path")
    @patch("hitch.main.codex_pool.os.getsid")
    def test_matches_when_cmdline_carries_instance_id(
        self, mock_getsid: MagicMock, mock_path: MagicMock
    ) -> None:
        mock_getsid.return_value = 4321
        cmdline = b"/usr/bin/python\x00manage.py\x00codex_worker\x00--instance-id\x0042\x00"
        mock_path.return_value.__truediv__.return_value.__truediv__.return_value.read_bytes.return_value = cmdline

        self.assertTrue(codex_pool._pid_is_our_worker(4321, 42))

    @patch("hitch.main.codex_pool.Path")
    @patch("hitch.main.codex_pool.os.getsid")
    def test_rejects_when_cmdline_lacks_codex_worker(
        self, mock_getsid: MagicMock, mock_path: MagicMock
    ) -> None:
        # An unrelated session leader has inherited the recycled pid:
        # session-leader check passes, cmdline check rules it out.
        mock_getsid.return_value = 4321
        mock_path.return_value.__truediv__.return_value.__truediv__.return_value.read_bytes.return_value = (
            b"/usr/bin/bash\x00-l\x00"
        )

        self.assertFalse(codex_pool._pid_is_our_worker(4321, 42))

    @patch("hitch.main.codex_pool.Path")
    @patch("hitch.main.codex_pool.os.getsid")
    def test_rejects_when_instance_id_flag_missing(
        self, mock_getsid: MagicMock, mock_path: MagicMock
    ) -> None:
        # Defensive: ``codex_worker`` is always invoked with
        # ``--instance-id`` today, but a malformed cmdline (truncated,
        # different worker variant) must not pass identity.
        mock_getsid.return_value = 4321
        mock_path.return_value.__truediv__.return_value.__truediv__.return_value.read_bytes.return_value = (
            b"python\x00manage.py\x00codex_worker\x00"
        )

        self.assertFalse(codex_pool._pid_is_our_worker(4321, 42))

    @patch("hitch.main.codex_pool.Path")
    @patch("hitch.main.codex_pool.os.getsid")
    def test_rejects_wrong_instance_id(
        self, mock_getsid: MagicMock, mock_path: MagicMock
    ) -> None:
        # cmdline names a codex_worker but for a different instance —
        # another worker, not ours.
        mock_getsid.return_value = 4321
        mock_path.return_value.__truediv__.return_value.__truediv__.return_value.read_bytes.return_value = (
            b"python\x00manage.py\x00codex_worker\x00--instance-id\x0099\x00"
        )

        self.assertFalse(codex_pool._pid_is_our_worker(4321, 42))

    @patch("hitch.main.codex_pool.os.getsid")
    def test_rejects_when_not_session_leader(
        self, mock_getsid: MagicMock
    ) -> None:
        mock_getsid.return_value = 999  # different from pid

        self.assertFalse(codex_pool._pid_is_our_worker(4321, 42))

    @patch("hitch.main.codex_pool.os.getsid")
    def test_rejects_when_pid_gone(self, mock_getsid: MagicMock) -> None:
        mock_getsid.side_effect = ProcessLookupError

        self.assertFalse(codex_pool._pid_is_our_worker(4321, 42))

    @patch("hitch.main.codex_pool.Path")
    @patch("hitch.main.codex_pool.os.getsid")
    def test_falls_back_to_getsid_when_proc_missing(
        self, mock_getsid: MagicMock, mock_path: MagicMock
    ) -> None:
        # macOS dev: /proc doesn't exist. Cmdline layer is unavailable;
        # trust the session-leader check rather than refuse to stop.
        mock_getsid.return_value = 4321
        mock_path.return_value.exists.return_value = False
        mock_path.return_value.__truediv__.return_value.__truediv__.return_value.read_bytes.side_effect = (
            FileNotFoundError
        )

        self.assertTrue(codex_pool._pid_is_our_worker(4321, 42))

    @patch("hitch.main.codex_pool.Path")
    @patch("hitch.main.codex_pool.os.getsid")
    def test_rejects_when_pid_vanishes_between_getsid_and_cmdline(
        self, mock_getsid: MagicMock, mock_path: MagicMock
    ) -> None:
        # Linux TOCTOU: getsid says the pid is a session leader, but
        # ``/proc/<pid>/cmdline`` is gone moments later because the
        # worker exited. The pid is now at risk of being recycled to an
        # unrelated process; falling back to the session-leader check
        # could let us signal a stranger's group. ``/proc`` itself
        # still exists, so we must refuse rather than trust the cheap
        # check that's no longer authoritative.
        mock_getsid.return_value = 4321
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.__truediv__.return_value.__truediv__.return_value.read_bytes.side_effect = (
            FileNotFoundError
        )

        self.assertFalse(codex_pool._pid_is_our_worker(4321, 42))

    @patch("hitch.main.codex_pool.Path")
    @patch("hitch.main.codex_pool.os.getsid")
    def test_rejects_on_other_cmdline_read_error(
        self, mock_getsid: MagicMock, mock_path: MagicMock
    ) -> None:
        # Permission error or any other non-ENOENT failure: be
        # conservative — refuse rather than signaling something we
        # could not identify.
        mock_getsid.return_value = 4321
        mock_path.return_value.__truediv__.return_value.__truediv__.return_value.read_bytes.side_effect = (
            PermissionError
        )

        self.assertFalse(codex_pool._pid_is_our_worker(4321, 42))


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


class _FakePayload(BaseModel):
    method_kind: str = "demo"
    detail: str


class SerializeEventTests(TestCase):
    def test_serializes_payload_shapes(self) -> None:
        @dataclasses.dataclass
        class Params:
            params: dict[str, str]

        # pydantic, dataclass, plain dict all flatten to a JSON payload field.
        cases = [
            (_FakePayload(detail="hello"), {"method_kind": "demo", "detail": "hello"}),
            (Params(params={"k": "v"}), {"params": {"k": "v"}}),
            ({"k": 1}, {"k": 1}),
        ]
        for payload, expected in cases:
            with self.subTest(payload=type(payload).__name__):
                parsed = json.loads(_serialize_event("m", payload))
                self.assertEqual(parsed["method"], "m")
                self.assertEqual(parsed["payload"], expected)


class CodexWorkerCommandTests(TestCase):
    def _make_instance(self, events_dir: Path, *, prompt: str = "hi") -> CodexInstance:
        return CodexInstance.objects.create(
            pid=12345,
            thread_id="thread-1",
            cwd="/repo",
            prompt=prompt,
            events_path=str(events_dir / "events.jsonl"),
            status=CodexInstance.STATUS_STARTING,
        )

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_streams_notifications_and_marks_completed(self, mock_codex: MagicMock) -> None:
        events = [
            SimpleNamespace(
                method="item/agentMessage/delta",
                payload=_FakePayload(detail="chunk-1"),
            ),
            _completed_event("turn-1", TurnStatus.completed),
        ]
        codex_ctx = mock_codex.return_value.__enter__.return_value
        codex_ctx.thread_resume.return_value = _stub_thread_resume(events)

        with tempfile.TemporaryDirectory() as raw:
            instance = self._make_instance(Path(raw))
            call_command("codex_worker", "--instance-id", str(instance.pk))

            with open(instance.events_path, encoding="utf-8") as fh:
                lines = [json.loads(line) for line in fh]

        codex_ctx.thread_resume.assert_called_once_with("thread-1")
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["method"], "item/agentMessage/delta")
        self.assertEqual(lines[0]["payload"]["detail"], "chunk-1")
        self.assertEqual(lines[1]["method"], "turn/completed")

        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_COMPLETED)
        self.assertIsNotNone(instance.ended_at)
        self.assertEqual(instance.error, "")

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_reads_prompt_from_instance_row(self, mock_codex: MagicMock) -> None:
        """The prompt isn't a CLI arg — verify it round-trips via the row,
        including a leading-dash value that argparse would reject."""
        captured: dict[str, object] = {}

        def _capture_turn(input_obj: object) -> object:
            captured["input"] = input_obj
            return SimpleNamespace(
                id="turn-1",
                stream=lambda: iter([_completed_event("turn-1", TurnStatus.completed)]),
            )

        codex_ctx = mock_codex.return_value.__enter__.return_value
        codex_ctx.thread_resume.return_value = SimpleNamespace(turn=_capture_turn)

        with tempfile.TemporaryDirectory() as raw:
            instance = self._make_instance(Path(raw), prompt="- a markdown bullet")
            call_command("codex_worker", "--instance-id", str(instance.pk))

        self.assertEqual(getattr(captured["input"], "text", None), "- a markdown bullet")

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_marks_failed_for_non_completed_outcomes(self, mock_codex: MagicMock) -> None:
        """Failed / interrupted turn statuses and a stream that ends without
        a turn/completed event all leave the row in STATUS_FAILED with a
        human-readable error blurb."""
        codex_ctx = mock_codex.return_value.__enter__.return_value

        cases = [
            (
                [_completed_event("turn-1", TurnStatus.failed, error_message="model said no")],
                "model said no",
            ),
            ([_completed_event("turn-1", TurnStatus.interrupted)], "interrupted"),
            ([], "turn/completed"),
        ]
        for events, expected_in_error in cases:
            with self.subTest(case=expected_in_error):
                codex_ctx.thread_resume.reset_mock()
                codex_ctx.thread_resume.return_value = _stub_thread_resume(events)
                with tempfile.TemporaryDirectory() as raw:
                    instance = self._make_instance(Path(raw))
                    call_command("codex_worker", "--instance-id", str(instance.pk))
                    instance.refresh_from_db()
                self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
                self.assertIn(expected_in_error, instance.error)
                self.assertIsNotNone(instance.ended_at)

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_records_failure_when_codex_raises(self, mock_codex: MagicMock) -> None:
        mock_codex.return_value.__enter__.side_effect = RuntimeError("boom")

        with tempfile.TemporaryDirectory() as raw:
            instance = self._make_instance(Path(raw))
            with self.assertRaises(RuntimeError):
                call_command(
                    "codex_worker",
                    "--instance-id",
                    str(instance.pk),
                    stderr=StringIO(),
                )

        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertIn("boom", instance.error)
        self.assertIsNotNone(instance.ended_at)

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_reasoning_effort_cli_arg_round_trip(self, mock_codex: MagicMock) -> None:
        """The effort rides into the worker as a CLI arg (no DB or cookie
        lookup in the subprocess). A known value reaches ``turn(effort=)``;
        an unknown one (e.g. stale enum after SDK upgrade) is silently
        dropped so Codex's own default takes over."""
        captured: dict[str, object] = {}

        def _capture_turn(input_obj: object, **kwargs: object) -> object:
            captured.update(kwargs)
            return SimpleNamespace(
                id="turn-1",
                stream=lambda: iter([_completed_event("turn-1", TurnStatus.completed)]),
            )

        codex_ctx = mock_codex.return_value.__enter__.return_value
        codex_ctx.thread_resume.return_value = SimpleNamespace(turn=_capture_turn)

        cases = [("high", ReasoningEffort.high), ("ludicrous", None)]
        for cli_value, expected in cases:
            with self.subTest(cli_value=cli_value):
                captured.clear()
                with tempfile.TemporaryDirectory() as raw:
                    instance = self._make_instance(Path(raw))
                    call_command(
                        "codex_worker",
                        "--instance-id",
                        str(instance.pk),
                        "--reasoning-effort",
                        cli_value,
                    )
                self.assertEqual(captured.get("effort"), expected)

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_sandbox_policy_cli_arg_round_trip(self, mock_codex: MagicMock) -> None:
        """The sandbox policy rides into the worker as a CLI arg, same as
        reasoning effort. A known value reaches ``turn(sandbox_policy=)`` as
        the matching SandboxPolicy variant; an unknown value (stale cookie
        after SDK upgrade) is silently dropped so the turn runs under
        Codex's default policy."""
        captured: dict[str, object] = {}

        def _capture_turn(input_obj: object, **kwargs: object) -> object:
            captured.update(kwargs)
            return SimpleNamespace(
                id="turn-1",
                stream=lambda: iter([_completed_event("turn-1", TurnStatus.completed)]),
            )

        codex_ctx = mock_codex.return_value.__enter__.return_value
        codex_ctx.thread_resume.return_value = SimpleNamespace(turn=_capture_turn)

        cases = [
            ("workspaceWrite", WorkspaceWriteSandboxPolicy),
            ("dangerFullAccess", DangerFullAccessSandboxPolicy),
            ("phantomPolicy", None),
        ]
        for cli_value, expected_variant in cases:
            with self.subTest(cli_value=cli_value):
                captured.clear()
                with tempfile.TemporaryDirectory() as raw:
                    instance = self._make_instance(Path(raw))
                    call_command(
                        "codex_worker",
                        "--instance-id",
                        str(instance.pk),
                        "--sandbox-policy",
                        cli_value,
                    )
                if expected_variant is None:
                    self.assertNotIn("sandbox_policy", captured)
                else:
                    policy = captured.get("sandbox_policy")
                    assert isinstance(policy, SandboxPolicy)
                    self.assertIsInstance(policy.root, expected_variant)

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_approval_mode_cli_arg_round_trip(self, mock_codex: MagicMock) -> None:
        """Approval mode rides in as a CLI arg just like sandbox policy. A
        known value reaches ``turn(approval_mode=)`` as the matching enum
        member; an unknown value (stale cookie after SDK upgrade) is
        silently dropped so the turn runs under Codex's default rather
        than crashing the worker."""
        captured: dict[str, object] = {}

        def _capture_turn(input_obj: object, **kwargs: object) -> object:
            captured.update(kwargs)
            return SimpleNamespace(
                id="turn-1",
                stream=lambda: iter([_completed_event("turn-1", TurnStatus.completed)]),
            )

        codex_ctx = mock_codex.return_value.__enter__.return_value
        codex_ctx.thread_resume.return_value = SimpleNamespace(turn=_capture_turn)

        cases = [
            ("deny_all", ApprovalMode.deny_all),
            ("auto_review", ApprovalMode.auto_review),
            ("phantom_mode", None),
        ]
        for cli_value, expected in cases:
            with self.subTest(cli_value=cli_value):
                captured.clear()
                with tempfile.TemporaryDirectory() as raw:
                    instance = self._make_instance(Path(raw))
                    call_command(
                        "codex_worker",
                        "--instance-id",
                        str(instance.pk),
                        "--approval-mode",
                        cli_value,
                    )
                if expected is None:
                    self.assertNotIn("approval_mode", captured)
                else:
                    self.assertEqual(captured.get("approval_mode"), expected)

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_approve_all_bypasses_thread_turn(self, mock_codex: MagicMock) -> None:
        """``approve_all`` is not in the SDK's ``ApprovalMode`` enum, so
        the worker has to bypass ``Thread.turn(approval_mode=)`` and post
        wire-level ``TurnStartParams`` directly: an on-request approval
        policy paired with ``ApprovalsReviewer.user`` routes every
        escalation to the client transport, where the default approval
        handler rubber-stamps it. Pin the wire call so a refactor cannot
        quietly downgrade the mode to one of the typed SDK values, or
        drop the explicit reviewer (which would let server-side routing
        send approvals to the auto-reviewer instead of the client)."""
        captured_params: dict[str, object] = {}
        codex_ctx = mock_codex.return_value.__enter__.return_value

        def _capture_turn_start(
            _thread_id: str, _input: object, *, params: object
        ) -> object:
            captured_params["params"] = params
            return SimpleNamespace(turn=SimpleNamespace(id="turn-1"))

        codex_ctx._client.turn_start.side_effect = _capture_turn_start
        codex_ctx._client.next_turn_notification.return_value = _completed_event(
            "turn-1", TurnStatus.completed
        )
        codex_ctx.thread_resume.return_value = SimpleNamespace(
            id="thread-1", turn=MagicMock()
        )

        with tempfile.TemporaryDirectory() as raw:
            instance = self._make_instance(Path(raw))
            call_command(
                "codex_worker",
                "--instance-id",
                str(instance.pk),
                "--approval-mode",
                "approve_all",
            )

        # ``Thread.turn`` (the typed SDK entry point) must NOT be used —
        # otherwise the call routes through ``ApprovalMode`` and the
        # rubber-stamp pairing is unreachable.
        codex_ctx.thread_resume.return_value.turn.assert_not_called()
        params = captured_params["params"]
        assert isinstance(params, TurnStartParams)
        # On-request approval policy + ``user`` reviewer means every
        # escalation is routed to the client transport, where the
        # default auto-approve handler answers it unconditionally.
        # ``reviewer=None`` would defer to server-side routing and is
        # NOT a safe substitute.
        approval_policy = params.approval_policy
        assert approval_policy is not None
        self.assertEqual(approval_policy.root, AskForApprovalValue.on_request)
        self.assertEqual(params.approvals_reviewer, ApprovalsReviewer.user)

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_marks_running_before_streaming(self, mock_codex: MagicMock) -> None:
        """The row flips to ``running`` before the first event so a slow
        codex initialization is visible to observers (UI / reconciliation)."""
        observed_status: dict[str, str] = {}

        def _capture_thread_resume(*_args: object, **_kwargs: object) -> object:
            observed_status["value"] = CodexInstance.objects.get(pk=instance.pk).status
            return _stub_thread_resume([_completed_event("turn-1", TurnStatus.completed)])

        codex_ctx = mock_codex.return_value.__enter__.return_value
        codex_ctx.thread_resume.side_effect = _capture_thread_resume

        with tempfile.TemporaryDirectory() as raw:
            instance = self._make_instance(Path(raw))
            call_command("codex_worker", "--instance-id", str(instance.pk))

        self.assertEqual(observed_status["value"], CodexInstance.STATUS_RUNNING)

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_installs_interactive_approval_handler_before_streaming(
        self, mock_codex: MagicMock
    ) -> None:
        """The handler must be hooked onto ``_client._approval_handler``
        *before* ``thread_resume`` runs, otherwise an early in-stream
        approval request would still hit the SDK's broken default handler
        (which sends ``{"decision": "accept"}`` — a value codex's
        ``ReviewDecision`` enum no longer accepts)."""
        observed_handler: dict[str, object] = {}
        codex_ctx = mock_codex.return_value.__enter__.return_value

        def _capture_thread_resume(*_args: object, **_kwargs: object) -> object:
            observed_handler["value"] = codex_ctx._client._approval_handler
            return _stub_thread_resume([_completed_event("turn-1", TurnStatus.completed)])

        codex_ctx.thread_resume.side_effect = _capture_thread_resume

        with tempfile.TemporaryDirectory() as raw:
            instance = self._make_instance(Path(raw))
            call_command("codex_worker", "--instance-id", str(instance.pk))

        handler = observed_handler["value"]
        # The handler is a closure (callable but not bound to the SDK's
        # default rubber-stamp method). Unrecognised methods short-circuit
        # to ``{}`` per the SDK's previous default-handler contract.
        assert callable(handler)
        self.assertEqual(handler("custom/method", None), {})


class ApprovalHandlerTests(TestCase):
    """``_make_approval_handler`` produces the closure the worker installs on
    ``AppServerClient._approval_handler``. The closure is called from the
    SDK's reader thread when codex escalates a command/file action.

    These tests exercise the closure directly rather than running a full
    worker turn — the surrounding worker plumbing is covered in
    ``CodexWorkerCommandTests``. Patching the polling sleep keeps the wait
    loop tight enough for unit tests.
    """

    def _make_instance(self) -> CodexInstance:
        return CodexInstance.objects.create(
            pid=12345,
            thread_id="thread-approval",
            cwd="/repo",
            prompt="hi",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
        )

    def test_approve_all_handler_rubber_stamps_known_methods(self) -> None:
        """``approve_all`` must keep its "approve everything" promise even
        though we replaced the SDK's default handler. Crucially the wire
        value is ``approved`` (not ``accept``) — the SDK's prior default
        sent ``accept``, which codex's ``ReviewDecision`` enum no longer
        accepts."""
        instance = self._make_instance()
        events: list[tuple[str, object]] = []

        def _record(method: str, payload: object) -> None:
            events.append((method, payload))

        handler = _make_approval_handler(
            instance=instance, write_event=_record, approval_mode="approve_all"
        )

        self.assertEqual(
            handler("item/commandExecution/requestApproval", {"item": {"command": "ls"}}),
            {"decision": "approved"},
        )
        self.assertEqual(
            handler("item/fileChange/requestApproval", {"item": {"changes": []}}),
            {"decision": "approved"},
        )
        # Unknown methods fall through to ``{}`` — the SDK uses that for
        # any server-to-client call we don't explicitly handle.
        self.assertEqual(handler("item/something/else", None), {})
        # Rubber-stamp mode emits no synthetic events; the UI doesn't
        # need an approval prompt to render in this mode.
        self.assertEqual(events, [])
        self.assertFalse(ApprovalRequest.objects.exists())

    def test_interactive_handler_creates_row_and_emits_events(self) -> None:
        """The interactive handler creates an ApprovalRequest row, emits an
        ``approval/requested`` event so the SSE stream surfaces the prompt,
        blocks until the row's ``decision`` is set, then emits an
        ``approval/resolved`` event with the chosen wire value.

        The polling wait is mocked because the production handler runs in
        the SDK reader thread while the view writes the decision from a
        request handler — sqlite under the test runner can't cleanly model
        that cross-thread race, so we exercise the orchestration directly
        and cover the polling loop in a separate test."""
        instance = self._make_instance()
        events: list[tuple[str, dict[str, Any]]] = []

        def _record(method: str, payload: object) -> None:
            assert isinstance(payload, dict)
            events.append((method, payload))

        handler = _make_approval_handler(
            instance=instance, write_event=_record, approval_mode="auto_review"
        )

        with patch(
            "hitch.main.management.commands.codex_worker._wait_for_decision",
            return_value="approved",
        ):
            result = handler(
                "item/commandExecution/requestApproval",
                {"item": {"command": "rm -rf /"}},
            )

        self.assertEqual(result, {"decision": "approved"})
        row = ApprovalRequest.objects.get(instance=instance)
        self.assertEqual(row.method, "item/commandExecution/requestApproval")
        self.assertEqual(row.params, {"item": {"command": "rm -rf /"}})
        # The pk is the link between the SSE event and the
        # ``POST /approval/<id>/`` URL the browser POSTs to.
        methods_to_payload = {m: p for m, p in events}
        self.assertIn("approval/requested", methods_to_payload)
        self.assertIn("approval/resolved", methods_to_payload)
        self.assertEqual(methods_to_payload["approval/requested"]["id"], row.pk)
        self.assertEqual(methods_to_payload["approval/requested"]["method"], row.method)
        self.assertEqual(methods_to_payload["approval/resolved"]["decision"], "approved")

    @patch(
        "hitch.main.management.commands.codex_worker._APPROVAL_POLL_INTERVAL", 0.001
    )
    def test_wait_for_decision_returns_recorded_decision(self) -> None:
        """Once the view records a decision on the row, the polling loop
        wakes up on the next interval and returns that wire value
        verbatim — the handler then forwards it into the SDK response."""
        from hitch.main.management.commands.codex_worker import _wait_for_decision

        approval = ApprovalRequest.objects.create(
            instance=self._make_instance(),
            method="item/commandExecution/requestApproval",
            params={},
            decision="abort",
        )

        self.assertEqual(_wait_for_decision(approval.pk), "abort")

    @patch(
        "hitch.main.management.commands.codex_worker._APPROVAL_POLL_INTERVAL", 0.001
    )
    @patch(
        "hitch.main.management.commands.codex_worker._APPROVAL_WAIT_SECONDS", 0.02
    )
    def test_wait_for_decision_defaults_to_denied_on_timeout(self) -> None:
        """A stuck approval (browser tab closed, user away) must release the
        worker rather than hang the turn forever. The timeout writes
        ``denied`` to the row so the UI doesn't show a perpetually-pending
        prompt after the next page reload."""
        from hitch.main.management.commands.codex_worker import _wait_for_decision

        approval = ApprovalRequest.objects.create(
            instance=self._make_instance(),
            method="item/fileChange/requestApproval",
            params={},
        )

        self.assertEqual(_wait_for_decision(approval.pk), "denied")
        approval.refresh_from_db()
        self.assertEqual(approval.decision, "denied")
        self.assertIsNotNone(approval.decided_at)

    @patch(
        "hitch.main.management.commands.codex_worker._APPROVAL_POLL_INTERVAL", 0.001
    )
    @patch(
        "hitch.main.management.commands.codex_worker._APPROVAL_WAIT_SECONDS", 0.0
    )
    def test_wait_for_decision_honours_user_pick_at_timeout_boundary(self) -> None:
        """When a user click lands in the window between the last empty
        read and the timeout's conditional UPDATE, the UPDATE matches
        zero rows. The handler must round-trip the user's recorded
        decision rather than overwriting it with ``denied`` (which
        would diverge the executed action from the click)."""
        from hitch.main.management.commands.codex_worker import _wait_for_decision

        approval = ApprovalRequest.objects.create(
            instance=self._make_instance(),
            method="item/commandExecution/requestApproval",
            params={},
        )

        # Simulate the race: the user POSTs the decision after the
        # initial values_list() read sees an empty decision but before
        # the timeout UPDATE runs. Patching ``values_list().get`` to
        # return ``""`` once and then handing control to the UPDATE
        # path with the real ``approved`` row already written is what
        # the production race looks like.
        original_values_list = ApprovalRequest.objects.values_list
        call_count = {"n": 0}

        def _values_list(*args: Any, **kwargs: Any) -> Any:
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First read: row is still pending — flip it to
                # ``approved`` *after* we hand the empty value back, so
                # the timeout UPDATE will see ``decision != ""`` and
                # match zero rows.
                qs = original_values_list(*args, **kwargs)

                class _Wrap:
                    def get(self, **q: Any) -> str:
                        result: str = qs.get(**q)
                        ApprovalRequest.objects.filter(pk=approval.pk).update(
                            decision="approved"
                        )
                        return result

                return _Wrap()
            return original_values_list(*args, **kwargs)

        with patch.object(
            ApprovalRequest.objects, "values_list", side_effect=_values_list
        ):
            self.assertEqual(_wait_for_decision(approval.pk), "approved")
        approval.refresh_from_db()
        self.assertEqual(approval.decision, "approved")

    def test_interactive_handler_ignores_unknown_methods(self) -> None:
        """Approval methods we don't recognise (future SDK additions) must
        return ``{}`` without creating a stray row — the SDK treats ``{}``
        as "no opinion", which is what the previous default handler did."""
        instance = self._make_instance()
        events: list[tuple[str, object]] = []
        handler = _make_approval_handler(
            instance=instance,
            write_event=lambda m, p: events.append((m, p)),
            approval_mode="auto_review",
        )

        self.assertEqual(handler("custom/escalation", None), {})
        self.assertFalse(ApprovalRequest.objects.exists())
        self.assertEqual(events, [])


class WorkerCancellationTests(TestCase):
    """The worker's graceful-stop path: SIGTERM handler + SDK interrupt.

    The stop endpoint signals the worker with SIGTERM; the worker turns
    that into the SDK's ``turn.interrupt()`` between stream events
    rather than dying abruptly. We test the handler (sets the flag),
    the interrupt helper (calls SDK, swallows errors), and the
    integration where a flag set during streaming triggers exactly one
    SDK interrupt.
    """

    @override
    def setUp(self) -> None:
        # Module-level flag persists across tests; reset to a known
        # state so previous tests don't bleed into this one.
        codex_worker_module._cancel_requested = False

    @override
    def tearDown(self) -> None:
        codex_worker_module._cancel_requested = False

    def test_sigterm_handler_sets_flag(self) -> None:
        self.assertFalse(codex_worker_module._cancel_requested)
        codex_worker_module._on_sigterm(15, None)
        self.assertTrue(codex_worker_module._cancel_requested)

    def test_try_interrupt_calls_sdk_and_reports_sent(self) -> None:
        turn = MagicMock()

        sent = codex_worker_module._try_interrupt(turn)

        self.assertTrue(sent)
        turn.interrupt.assert_called_once_with()

    def test_try_interrupt_swallows_sdk_errors(self) -> None:
        # A failed SDK call (turn already done, transport hiccup) must
        # still report "sent" so the loop doesn't retry forever — the
        # user's escalation is SIGKILL via a second click, not a retry.
        turn = MagicMock()
        turn.interrupt.side_effect = RuntimeError("turn already terminated")

        sent = codex_worker_module._try_interrupt(turn)

        self.assertTrue(sent)
        turn.interrupt.assert_called_once_with()

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_interrupt_fires_when_flag_set_during_stream(
        self, mock_codex: MagicMock
    ) -> None:
        # Drive the stream loop through a flag-set transition: yield
        # one event, then arrange for the cancel flag to be observed
        # on the next iteration, then yield the turn/completed.
        # Verify exactly one ``turn.interrupt()`` call.
        first_event = SimpleNamespace(
            method="item/agentMessage/delta",
            payload=_FakePayload(detail="chunk"),
        )
        completion = _completed_event("turn-1", TurnStatus.completed)

        def gen() -> Iterator[Any]:
            yield first_event
            codex_worker_module._cancel_requested = True
            yield completion

        captured: dict[str, Any] = {}

        def thread_resume_side_effect(*_args: object, **_kwargs: object) -> object:
            turn_mock = MagicMock()
            turn_mock.id = "turn-1"
            turn_mock.stream.return_value = gen()
            captured["turn"] = turn_mock
            return SimpleNamespace(turn=lambda _input: turn_mock)

        codex_ctx = mock_codex.return_value.__enter__.return_value
        codex_ctx.thread_resume.side_effect = thread_resume_side_effect

        with tempfile.TemporaryDirectory() as raw:
            instance = CodexInstance.objects.create(
                pid=12345,
                thread_id="thread-1",
                cwd="/repo",
                prompt="hi",
                events_path=str(Path(raw) / "events.jsonl"),
                status=CodexInstance.STATUS_STARTING,
            )
            call_command("codex_worker", "--instance-id", str(instance.pk))

        captured["turn"].interrupt.assert_called_once_with()

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_interrupt_called_exactly_once_when_flag_stays_set(
        self, mock_codex: MagicMock
    ) -> None:
        # If multiple SIGTERMs arrive (or the flag was never cleared),
        # the loop must not re-send ``turn.interrupt()`` on every event.
        # The escalation lever is SIGKILL via a fresh click, not
        # spamming the SDK with cancellation requests.
        events = [
            SimpleNamespace(
                method="item/agentMessage/delta",
                payload=_FakePayload(detail="a"),
            ),
            SimpleNamespace(
                method="item/agentMessage/delta",
                payload=_FakePayload(detail="b"),
            ),
            _completed_event("turn-1", TurnStatus.completed),
        ]

        # Set the flag before the worker starts so it observes on the
        # initial pre-loop check; then more events arrive — interrupt
        # must not be re-called on each.
        codex_worker_module._cancel_requested = True

        captured: dict[str, Any] = {}

        def thread_resume_side_effect(*_args: object, **_kwargs: object) -> object:
            turn_mock = MagicMock()
            turn_mock.id = "turn-1"
            turn_mock.stream.return_value = iter(events)
            captured["turn"] = turn_mock
            return SimpleNamespace(turn=lambda _input: turn_mock)

        codex_ctx = mock_codex.return_value.__enter__.return_value
        codex_ctx.thread_resume.side_effect = thread_resume_side_effect

        with tempfile.TemporaryDirectory() as raw:
            instance = CodexInstance.objects.create(
                pid=12345,
                thread_id="thread-1",
                cwd="/repo",
                prompt="hi",
                events_path=str(Path(raw) / "events.jsonl"),
                status=CodexInstance.STATUS_STARTING,
            )
            call_command("codex_worker", "--instance-id", str(instance.pk))

        captured["turn"].interrupt.assert_called_once_with()

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_no_interrupt_when_flag_never_set(
        self, mock_codex: MagicMock
    ) -> None:
        # Sanity check: the normal turn flow doesn't call interrupt.
        completion = _completed_event("turn-1", TurnStatus.completed)

        captured: dict[str, Any] = {}

        def thread_resume_side_effect(*_args: object, **_kwargs: object) -> object:
            turn_mock = MagicMock()
            turn_mock.id = "turn-1"
            turn_mock.stream.return_value = iter([completion])
            captured["turn"] = turn_mock
            return SimpleNamespace(turn=lambda _input: turn_mock)

        codex_ctx = mock_codex.return_value.__enter__.return_value
        codex_ctx.thread_resume.side_effect = thread_resume_side_effect

        with tempfile.TemporaryDirectory() as raw:
            instance = CodexInstance.objects.create(
                pid=12345,
                thread_id="thread-1",
                cwd="/repo",
                prompt="hi",
                events_path=str(Path(raw) / "events.jsonl"),
                status=CodexInstance.STATUS_STARTING,
            )
            call_command("codex_worker", "--instance-id", str(instance.pk))

        captured["turn"].interrupt.assert_not_called()


# A pid we know is alive (this Python process) lets the streaming tests
# create CodexInstance rows that survive ``reconcile_dead`` without faking
# the ``is_alive`` helper.
_LIVE_PID = os.getpid()


def _make_streaming_instance(
    events_path: str,
    *,
    thread_id: str = "thread-1",
    status: str = CodexInstance.STATUS_RUNNING,
    prompt: str = "do work",
    pid: int = 0,
) -> CodexInstance:
    return CodexInstance.objects.create(
        pid=pid,
        thread_id=thread_id,
        cwd="/repo",
        prompt=prompt,
        events_path=events_path,
        status=status,
    )


class StreamForInstanceTests(TestCase):
    """The SSE generator that re-emits a worker's JSONL events file frame-by-
    frame. Pairs with ``streaming.stream_for_instance`` / ``empty_stream``.
    """

    def test_idle_stream_emits_initial_heartbeat_and_recycles(self) -> None:
        # The "no active worker" path keeps the SSE channel open so the
        # session page's connection indicator can show ``connected, idle``.
        # Force the cap down so the loop returns promptly without an end
        # event — EventSource will reconnect transparently.
        with (
            patch("hitch.main.streaming._IDLE_MAX_STREAM_SECONDS", 0.001),
            patch("hitch.main.streaming._IDLE_POLL_INTERVAL", 0.001),
        ):
            frames = list(streaming.idle_stream("thread-none", baseline_id=None))
        self.assertEqual(frames[0], b"retry: 2000\n\n")
        heartbeats = [f for f in frames if f.startswith(b"event: heartbeat")]
        self.assertGreaterEqual(len(heartbeats), 1)
        self.assertIn(b'"working": false', heartbeats[0])
        # No worker ever showed up, so the stream closes silently rather
        # than firing an ``end`` event that would force a page reload.
        self.assertFalse(any(f.startswith(b"event: end") for f in frames))

    def test_idle_stream_ends_when_worker_appears(self) -> None:
        # If a worker is spawned out-of-band (e.g. another tab), the idle
        # stream should fire ``event: end`` so the client reloads into the
        # live-streaming UI rather than waiting for the per-stream cap.
        # The baseline (``None``) reflects what the page saw at render
        # time; a fresh pk on the first poll proves an out-of-band turn
        # has landed since then.
        with (
            patch("hitch.main.streaming._IDLE_POLL_INTERVAL", 0.001),
            patch(
                "hitch.main.streaming.codex_pool.latest_id_for_thread",
                return_value=42,
            ),
        ):
            frames = list(streaming.idle_stream("thread-active", baseline_id=None))
        self.assertTrue(frames[-1].startswith(b"event: end"))
        self.assertIn(b'"active"', frames[-1])

    def test_idle_stream_ends_on_fast_completed_out_of_band_turn(self) -> None:
        # Reload-detection has to fire even for an out-of-band turn that
        # already finished by the time we look — pk tracking catches it
        # where a "still active?" check would have missed it. Page saw
        # baseline=7 at render; one poll later the DB shows pk=9 even
        # though no row is currently active.
        with (
            patch("hitch.main.streaming._IDLE_POLL_INTERVAL", 0.001),
            patch(
                "hitch.main.streaming.codex_pool.latest_id_for_thread",
                return_value=9,
            ),
        ):
            frames = list(
                streaming.idle_stream("thread-completed-fast", baseline_id=7)
            )
        self.assertTrue(frames[-1].startswith(b"event: end"))
        self.assertIn(b'"active"', frames[-1])

    @patch("hitch.main.streaming._HEARTBEAT_INTERVAL", 0.0)
    @patch("hitch.main.streaming._IDLE_POLL_INTERVAL", 0.001)
    @patch("hitch.main.streaming._IDLE_MAX_STREAM_SECONDS", 0.005)
    def test_idle_stream_resends_heartbeats_at_cadence(self) -> None:
        # With the heartbeat cadence collapsed to zero we should observe
        # multiple heartbeat frames before the per-stream cap closes the
        # stream — confirming the periodic refresh path actually runs.
        frames = list(streaming.idle_stream("thread-none", baseline_id=None))
        heartbeats = [f for f in frames if f.startswith(b"event: heartbeat")]
        self.assertGreater(len(heartbeats), 1)
        for frame in heartbeats:
            self.assertIn(b'"working": false', frame)

    def test_reload_stream_yields_immediate_end(self) -> None:
        # ``session_stream`` returns this when it detects the page is
        # stale (worker spawned/completed between page render and SSE
        # open). It needs to fire ``event: end`` immediately so the
        # client reloads — no heartbeats, no waiting on a poll.
        frames = list(streaming.reload_stream())
        self.assertEqual(frames[0], b"retry: 2000\n\n")
        self.assertTrue(frames[-1].startswith(b"event: end"))
        self.assertIn(b'"stale"', frames[-1])

    def test_streams_existing_lines_and_terminates_when_done(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            events_path = str(Path(raw) / "events.jsonl")
            with open(events_path, "w", encoding="utf-8") as fh:
                fh.write(
                    json.dumps({"method": "item/started", "payload": {"item": {"id": "x"}}})
                    + "\n"
                )
                fh.write(json.dumps({"method": "turn/completed", "payload": {}}) + "\n")

            instance = _make_streaming_instance(events_path, status=CodexInstance.STATUS_COMPLETED)
            frames = list(streaming.stream_for_instance(instance))

        data_frames = [f for f in frames if f.startswith(b"data: ")]
        self.assertEqual(len(data_frames), 2)
        self.assertIn(b"item/started", data_frames[0])
        self.assertIn(b"turn/completed", data_frames[1])
        self.assertTrue(frames[-1].startswith(b"event: end"))
        self.assertIn(b'"completed"', frames[-1])

    def test_terminates_when_status_flips_to_failed(self) -> None:
        # A worker that ended with a failure status still flushes its events
        # file, but the end frame should carry the actual terminal status so
        # the client can surface the failure UI.
        with tempfile.TemporaryDirectory() as raw:
            events_path = str(Path(raw) / "events.jsonl")
            with open(events_path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"method": "item/started", "payload": {}}) + "\n")

            instance = _make_streaming_instance(events_path, status=CodexInstance.STATUS_FAILED)
            frames = list(streaming.stream_for_instance(instance))

        self.assertIn(b'"failed"', frames[-1])

    def test_ignores_partial_trailing_line(self) -> None:
        # The worker is line-buffered, so a tailer that opens the file at the
        # exact moment a half-line is on disk must not emit a malformed JSON
        # frame to the client.
        with tempfile.TemporaryDirectory() as raw:
            events_path = str(Path(raw) / "events.jsonl")
            with open(events_path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"method": "item/started", "payload": {}}) + "\n")
                # Half a JSON object, no trailing newline.
                fh.write('{"method":"item/partial')

            instance = _make_streaming_instance(events_path, status=CodexInstance.STATUS_COMPLETED)
            frames = list(streaming.stream_for_instance(instance))

        data_frames = [f for f in frames if f.startswith(b"data: ")]
        self.assertEqual(len(data_frames), 1)
        self.assertIn(b"item/started", data_frames[0])
        self.assertNotIn(b"item/partial", b"".join(frames))

    @patch("hitch.main.streaming._POLL_INTERVAL", 0.01)
    def test_missing_events_file_with_dead_worker_ends_promptly(self) -> None:
        # If the events file never appears but the worker process is also
        # gone, the tailer must bail rather than wait the full appearance
        # timeout. ``pid=99999999`` is well above the kernel's pid_max so
        # ``os.kill(pid, 0)`` raises ProcessLookupError → is_alive False.
        with tempfile.TemporaryDirectory() as raw:
            events_path = str(Path(raw) / "never-created.jsonl")
            instance = _make_streaming_instance(
                events_path, status=CodexInstance.STATUS_RUNNING, pid=99999999
            )
            frames = list(streaming.stream_for_instance(instance))

        self.assertTrue(frames[-1].startswith(b"event: end"))
        self.assertNotIn(b'"missing"', frames[-1])

    @patch("hitch.main.streaming._POLL_INTERVAL", 0.005)
    @patch("hitch.main.streaming._FILE_APPEAR_TIMEOUT", 0.001)
    def test_appearance_timeout_when_file_never_arrives(self) -> None:
        # A live worker that's stuck before its first write should bail via
        # the appearance timeout rather than hanging the request handler.
        instance = _make_streaming_instance(
            "/tmp/hitch-test-never-created.jsonl",
            status=CodexInstance.STATUS_RUNNING,
            pid=_LIVE_PID,
        )
        frames = list(streaming.stream_for_instance(instance))
        self.assertIn(b'"missing"', frames[-1])

    @patch("hitch.main.streaming._POLL_INTERVAL", 0.001)
    @patch("hitch.main.streaming._HEARTBEAT_INTERVAL", 0.0)
    @patch("hitch.main.streaming._FILE_APPEAR_TIMEOUT", 0.02)
    def test_heartbeat_yielded_while_waiting_for_events_file(self) -> None:
        # The pre-file-open wait loop also has to refresh the heartbeat so
        # the page's connection indicator stays green during a worker's
        # startup latency, not just once the file is being tailed.
        instance = _make_streaming_instance(
            "/tmp/hitch-test-startup-heartbeat.jsonl",
            status=CodexInstance.STATUS_RUNNING,
            pid=_LIVE_PID,
        )
        frames = list(streaming.stream_for_instance(instance))
        heartbeats = [f for f in frames if f.startswith(b"event: heartbeat")]
        # One initial heartbeat plus at least one from the appearance wait.
        self.assertGreaterEqual(len(heartbeats), 2)
        for frame in heartbeats:
            self.assertIn(b'"working": true', frame)
        self.assertIn(b'"missing"', frames[-1])

    def test_emit_skips_blank_lines(self) -> None:
        # A stray blank line on the events file must not produce an empty
        # SSE ``data:`` frame.
        with tempfile.TemporaryDirectory() as raw:
            events_path = str(Path(raw) / "events.jsonl")
            with open(events_path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"method": "a", "payload": {}}) + "\n")
                fh.write("\n")
                fh.write(json.dumps({"method": "b", "payload": {}}) + "\n")
            instance = _make_streaming_instance(events_path, status=CodexInstance.STATUS_COMPLETED)
            frames = list(streaming.stream_for_instance(instance))
        data_frames = [f for f in frames if f.startswith(b"data: ")]
        self.assertEqual(len(data_frames), 2)

    @patch("hitch.main.streaming._POLL_INTERVAL", 0.005)
    @patch("hitch.main.streaming._MAX_STREAM_SECONDS", 0.001)
    def test_read_loop_hits_stream_timeout(self) -> None:
        # A live worker that goes silent for longer than the per-stream cap
        # must release the request thread; the browser's EventSource will
        # reconnect if the user is still on the page.
        with tempfile.TemporaryDirectory() as raw:
            events_path = str(Path(raw) / "events.jsonl")
            Path(events_path).touch()
            instance = _make_streaming_instance(
                events_path, status=CodexInstance.STATUS_RUNNING, pid=_LIVE_PID
            )
            frames = list(streaming.stream_for_instance(instance))
        self.assertIn(b'"timeout"', frames[-1])

    @patch("hitch.main.streaming._POLL_INTERVAL", 0.001)
    @patch("hitch.main.streaming._HEARTBEAT_INTERVAL", 0.0)
    def test_heartbeat_yielded_while_worker_is_idle(self) -> None:
        # Idle SSE connections get periodic heartbeat events so the page's
        # connection indicator can refresh and the channel stays open past
        # idle proxies. Force _is_done False on the first poll and True on
        # the second so we observe at least one heartbeat frame past the
        # initial one before the stream finishes cleanly.
        with tempfile.TemporaryDirectory() as raw:
            events_path = str(Path(raw) / "events.jsonl")
            Path(events_path).touch()
            instance = _make_streaming_instance(
                events_path, status=CodexInstance.STATUS_RUNNING, pid=_LIVE_PID
            )

            done_calls = [0]

            def fake_is_done(_pk: int) -> bool:
                done_calls[0] += 1
                return done_calls[0] >= 2

            with patch("hitch.main.streaming._is_done", side_effect=fake_is_done):
                frames = list(streaming.stream_for_instance(instance))

        heartbeats = [f for f in frames if f.startswith(b"event: heartbeat")]
        # One initial heartbeat plus at least one from the idle poll loop.
        self.assertGreaterEqual(len(heartbeats), 2)
        for frame in heartbeats:
            self.assertIn(b'"working": true', frame)
        self.assertTrue(frames[-1].startswith(b"event: end"))

    def test_late_flush_after_status_flip_is_picked_up(self) -> None:
        # The worker's status transition and its final file write are not
        # atomic: a turn/completed event can land on disk *after* the row
        # already shows COMPLETED. The post-done re-read catches that line.
        fake_file = MagicMock()
        fake_file.__enter__.return_value = fake_file
        fake_file.__exit__.return_value = False
        fake_file.read.side_effect = [
            "",
            json.dumps({"method": "turn/completed", "payload": {}}) + "\n",
            "",
        ]

        instance = _make_streaming_instance(
            "/tmp/hitch-test-late.jsonl", status=CodexInstance.STATUS_COMPLETED
        )

        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "open", return_value=fake_file),
        ):
            frames = list(streaming.stream_for_instance(instance))

        data_frames = [f for f in frames if f.startswith(b"data: ")]
        self.assertEqual(len(data_frames), 1)
        self.assertIn(b"turn/completed", data_frames[0])

    def test_is_done_treats_missing_instance_as_terminal(self) -> None:
        # A row deleted out from under the tailer (cleanup race) ends the
        # stream rather than crashing the generator with DoesNotExist.
        self.assertTrue(streaming._is_done(99999999))

    def test_current_status_returns_unknown_for_missing_instance(self) -> None:
        # Symmetric to _is_done: the end-frame status falls back to a
        # stable sentinel when the row is gone.
        self.assertEqual(streaming._current_status(99999999), "unknown")

    @patch("hitch.main.streaming._POLL_INTERVAL", 0.01)
    def test_running_instance_with_dead_pid_terminates(self) -> None:
        # The events file exists but the worker died before flipping its
        # status; ``is_alive`` short-circuits the otherwise-infinite read.
        with tempfile.TemporaryDirectory() as raw:
            events_path = str(Path(raw) / "events.jsonl")
            with open(events_path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"method": "item/started", "payload": {}}) + "\n")

            instance = _make_streaming_instance(
                events_path, status=CodexInstance.STATUS_RUNNING, pid=99999999
            )
            frames = list(streaming.stream_for_instance(instance))

        data_frames = [f for f in frames if f.startswith(b"data: ")]
        self.assertEqual(len(data_frames), 1)
        self.assertTrue(frames[-1].startswith(b"event: end"))
