"""Contracts shared by message, review, and background turn startup."""

import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from django.db import connections
from django.test import TestCase, TransactionTestCase, override_settings

from hitch.main.models import CodexInstance, SessionMetadata
from hitch.main.runtime import codex_pool
from hitch.main.sessions import lifecycle, turn_startup
from hitch.main.sessions.execution_settings import PreviousTurnApproval, RequestApproval


class TurnStartupTests(TestCase):
    def test_busy_workers_and_leases_do_not_launch(self) -> None:
        with patch.object(codex_pool, "_spawn_turn") as spawn:
            for status in CodexInstance.ACTIVE_STATUSES:
                with self.subTest(status=status):
                    CodexInstance.objects.create(thread_id=status, cwd="/tmp", pid=0, status=status)
                    with turn_startup.claim(status) as startup:
                        self.assertIsNone(startup)
            with lifecycle.hold("locked"), turn_startup.claim("locked", blocking=False) as startup:
                self.assertIsNone(startup)
        spawn.assert_not_called()

    def test_claim_cannot_be_reused_or_used_after_release(self) -> None:
        with patch.object(codex_pool, "_spawn_turn") as spawn:
            with self.assertRaisesRegex(RuntimeError, "no longer available"):
                turn_startup.TurnStartup("unclaimed").spawn(
                    cwd="/tmp", prompt="Unclaimed", approval=RequestApproval("deny_all"),
                )
            with turn_startup.claim("once") as startup:
                assert startup is not None
                startup.spawn(cwd="/tmp", prompt="Continue", approval=RequestApproval("deny_all"))
                with self.assertRaisesRegex(RuntimeError, "no longer available"):
                    startup.spawn(cwd="/tmp", prompt="Duplicate", approval=RequestApproval("deny_all"))
            with turn_startup.claim("expired") as expired:
                assert expired is not None
            with self.assertRaisesRegex(RuntimeError, "no longer available"):
                expired.spawn(cwd="/tmp", prompt="Late", approval=RequestApproval("deny_all"))
        spawn.assert_called_once()

    def test_failed_launch_releases_claim_and_allows_retry(self) -> None:
        with tempfile.TemporaryDirectory() as raw, override_settings(CODEX_EVENTS_DIR=Path(raw)):
            with (
                patch.object(codex_pool, "_launch_worker_process", side_effect=OSError("launch failed")),
                self.assertRaisesRegex(OSError, "launch failed"),
                turn_startup.claim("retry") as startup,
            ):
                assert startup is not None
                startup.spawn(cwd="/tmp", prompt="First", approval=RequestApproval("deny_all"))
            self.assertEqual(CodexInstance.objects.get(thread_id="retry").status, CodexInstance.STATUS_FAILED)
            with (
                patch.object(codex_pool, "_launch_worker_process", return_value=SimpleNamespace(pid=0)),
                turn_startup.claim("retry") as startup,
            ):
                assert startup is not None
                worker = startup.spawn(cwd="/tmp", prompt="Retry", approval=RequestApproval("deny_all"))
            self.assertEqual(worker.status, CodexInstance.STATUS_STARTING)

    def test_request_and_background_settings_are_resolved_before_launch(self) -> None:
        metadata = SessionMetadata.objects.create(
            thread_id="settings", cwd="/tmp", model="chosen-model", reasoning_effort="high",
            approval_snapshot_mode="deny_all", approval_snapshot_instance_id=42,
        )
        with tempfile.TemporaryDirectory() as raw, override_settings(CODEX_EVENTS_DIR=Path(raw)):
            for defaults, expected in (
                (RequestApproval("auto_review"), "auto_review"),
                (PreviousTurnApproval(42, "auto_review"), "deny_all"),
            ):
                with (
                    self.subTest(defaults=defaults),
                    patch.object(codex_pool, "_launch_worker_process", return_value=SimpleNamespace(pid=0)) as launch,
                    turn_startup.claim("settings") as startup,
                ):
                    assert startup is not None
                    worker = startup.spawn(cwd="/tmp", prompt="Continue", approval=defaults, model="old-model")
                    self.assertEqual(worker.approval_mode, expected)
                    self.assertEqual(worker.model, "chosen-model")
                    self.assertEqual(worker.reasoning_effort, "high")
                    self.assertEqual(launch.call_args.kwargs["approval_mode"], expected)
                    self.assertEqual(launch.call_args.kwargs["model"], "chosen-model")
                worker.status = CodexInstance.STATUS_COMPLETED
                worker.save(update_fields=["status"])
            metadata.approval_mode = "approve_all"
            metadata.save(update_fields=["approval_mode"])
            with patch.object(codex_pool, "_spawn_turn") as spawn, turn_startup.claim("settings") as startup:
                assert startup is not None
                startup.spawn(cwd="/tmp", prompt="Override", approval=PreviousTurnApproval(42, "deny_all"))
            self.assertEqual(spawn.call_args.kwargs["approval_mode"], "approve_all")


class ConcurrentTurnStartupTests(TransactionTestCase):
    def test_waiting_sender_observes_worker_created_by_winning_claim(self) -> None:
        acquiring = threading.Event()
        acquire = lifecycle._acquire

        def signal_acquire(thread_id: str, *, blocking: bool) -> lifecycle._Lease | None:
            acquiring.set()
            return acquire(thread_id, blocking=blocking)

        def competing_start() -> int | None:
            try:
                with turn_startup.claim("contended") as startup:
                    if startup is None:
                        return None
                    return startup.spawn(
                        cwd="/tmp", prompt="Competing", approval=RequestApproval("deny_all"),
                    ).pk
            finally:
                connections.close_all()

        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_EVENTS_DIR=Path(raw)),
            patch.object(codex_pool, "_launch_worker_process", return_value=SimpleNamespace(pid=0)) as launch,
            ThreadPoolExecutor(max_workers=1) as executor,
        ):
            with turn_startup.claim("contended") as startup:
                assert startup is not None
                with patch.object(lifecycle, "_acquire", side_effect=signal_acquire):
                    future = executor.submit(competing_start)
                    self.assertTrue(acquiring.wait(timeout=5))
                    self.assertFalse(future.done())
                    worker = startup.spawn(cwd="/tmp", prompt="Winner", approval=RequestApproval("deny_all"))
            self.assertIsNone(future.result(timeout=5))
            launch.assert_called_once()
            self.assertEqual(list(CodexInstance.objects.values_list("pk", flat=True)), [worker.pk])
