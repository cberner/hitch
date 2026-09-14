import errno
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.db import connections
from django.test import TestCase, TransactionTestCase, override_settings

from hitch.main import checkouts
from hitch.main.models import CodexInstance, SessionMetadata
from hitch.main.runtime import codex_pool
from hitch.main.runtime.file_locks import FileLease
from hitch.main.sessions import lifecycle, turn_startup
from hitch.main.sessions.execution_settings import RequestApproval
from hitch.main.workflows import pr_tracking


class SessionLifecycleLockTests(TestCase):
    def test_worktree_lease_needs_no_new_files_when_disk_is_full(self) -> None:
        with tempfile.TemporaryDirectory() as raw, override_settings(HITCH_WORKTREES_DIR=Path(raw)):
            path = Path(raw) / "checkout"
            path.mkdir()
            original_open = os.open

            def open_without_space(path: str | Path, flags: int, mode: int = 0o777) -> int:
                if flags & os.O_CREAT:
                    raise OSError(errno.ENOSPC, "No space left on device")
                return original_open(path, flags, mode)

            with (
                patch.object(os, "open", side_effect=open_without_space),
                checkouts.hold(str(path)) as acquired,
            ):
                self.assertTrue(acquired)

    def test_session_lease_protects_worktree_and_allows_nested_worker_start(self) -> None:
        with tempfile.TemporaryDirectory() as raw, override_settings(
            HITCH_WORKTREES_DIR=Path(raw), CODEX_EVENTS_DIR=Path(raw) / "events",
        ), ThreadPoolExecutor(max_workers=1) as executor:
            cwd = str(Path(raw) / "checkout")
            Path(cwd).mkdir()
            SessionMetadata.objects.create(thread_id="thread-1", cwd=cwd)

            def competing_claim() -> bool:
                with checkouts.hold(cwd, blocking=False) as acquired:
                    return acquired

            with turn_startup.claim("thread-1") as startup, patch.object(
                codex_pool, "_launch_worker_process", return_value=SimpleNamespace(pid=0),
            ):
                self.assertFalse(executor.submit(competing_claim).result(timeout=5))
                assert startup is not None
                worker = startup.spawn(cwd=cwd, prompt="Continue", approval=RequestApproval("deny_all"))
                self.assertEqual(worker.status, CodexInstance.STATUS_STARTING)
            self.assertTrue(executor.submit(competing_claim).result(timeout=5))

            with (
                self.assertRaises(FileNotFoundError),
                patch.object(codex_pool, "_launch_worker_process") as launch,
                turn_startup.claim("removed") as startup,
            ):
                assert startup is not None
                startup.spawn(cwd=str(Path(raw) / "removed"), prompt="Continue", approval=RequestApproval("deny_all"))
            launch.assert_not_called()
            self.assertFalse(CodexInstance.objects.filter(thread_id="removed").exists())

    def test_nonblocking_claim_reports_busy_lock(self) -> None:
        with lifecycle.hold("thread-1") as acquired:
            self.assertTrue(acquired)
            with lifecycle.hold("thread-1", blocking=False) as competing:
                self.assertFalse(competing)
            with lifecycle.hold("thread-1", allow_nested=True) as nested:
                self.assertTrue(nested)

        with lifecycle.hold("thread-1", blocking=False) as reacquired:
            self.assertTrue(reacquired)

    @patch("hitch.main.sessions.lifecycle.os.close")
    @patch("hitch.main.runtime.file_locks.fcntl.flock")
    @patch("hitch.main.sessions.lifecycle.os.open", return_value=123)
    def test_acquire_closes_descriptor_when_flock_fails(
        self, _open: MagicMock, flock: MagicMock, close: MagicMock
    ) -> None:
        flock.side_effect = OSError("flock failed")

        with self.assertRaisesRegex(OSError, "flock failed"):
            lifecycle._acquire("thread-1", blocking=True)

        close.assert_called_once_with(123)


class PrCompletionLockTests(TransactionTestCase):
    def test_completion_waits_for_terminal_delivery_lease(self) -> None:
        instance = CodexInstance.objects.create(
            thread_id="terminal-thread", pid=0, cwd="/tmp", status=CodexInstance.STATUS_COMPLETED,
        )
        acquiring = threading.Event()
        acquire = lifecycle._acquire

        def signal_acquire(thread_id: str, *, blocking: bool) -> FileLease | None:
            acquiring.set()
            return acquire(thread_id, blocking=blocking)

        def complete() -> None:
            try:
                pr_tracking.supersede_pr_after_turn(instance)
            finally:
                connections.close_all()

        with ThreadPoolExecutor(max_workers=1) as executor:
            with lifecycle.hold(instance.thread_id), patch.object(lifecycle, "_acquire", side_effect=signal_acquire):
                future = executor.submit(complete)
                self.assertTrue(acquiring.wait(timeout=5))
                self.assertFalse(future.done())
            future.result(timeout=5)
