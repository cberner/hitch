from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.test import TestCase, override_settings
from django.utils import timezone

from hitch.main import disk_cleanup
from hitch.main.models import (
    CodexInstance,
    GlobalSettings,
    ProposedSession,
    SessionMetadata,
    SystemWorkflow,
)


class DiskCleanupTests(TestCase):
    def _managed_path(self, root: Path, name: str) -> str:
        return str(root / "managed" / "repo" / name)

    def _session(
        self,
        *,
        thread_id: str,
        cwd: str,
        archived: bool = False,
        archived_at: datetime | None = None,
        hidden_system: bool = False,
        stage: str = "",
    ) -> SessionMetadata:
        now = timezone.now()
        return SessionMetadata.objects.create(
            thread_id=thread_id,
            cwd=cwd,
            codex_archived=archived,
            codex_archived_at=archived_at,
            codex_created_at=now,
            codex_updated_at=now,
            codex_last_synced_at=now,
            is_hidden_system_session=hidden_system,
            derived_stage=stage,
        )

    def _run_cleanup(
        self,
        *,
        root: Path,
        sizes: list[int],
        mock_cleanup: MagicMock,
    ) -> int:
        hitch_home = root / ".hitch"
        managed = root / "managed"
        hitch_home.mkdir()
        managed.mkdir()
        with (
            override_settings(
                HITCH_HOME_DIR=hitch_home,
                HITCH_WORKTREES_DIR=managed,
                HITCH_MAX_ALLOWED_DISK_SPACE_PERCENT=20,
            ),
            patch(
                "hitch.main.disk_cleanup.shutil.disk_usage",
                return_value=SimpleNamespace(total=1000),
            ),
            patch("hitch.main.disk_cleanup._directory_size", side_effect=sizes),
        ):
            return disk_cleanup.cleanup_hitch_disk_usage_if_needed()

    def test_default_max_allowed_percent_is_twenty(self) -> None:
        self.assertEqual(disk_cleanup._max_allowed_percent(), 20.0)

    @override_settings(HITCH_MAX_ALLOWED_DISK_SPACE_PERCENT=20)
    def test_saved_global_max_allowed_percent_overrides_env(self) -> None:
        GlobalSettings.objects.create(
            pk=GlobalSettings.SINGLETON_PK, disk_usage_max_percent=35.5
        )

        self.assertEqual(disk_cleanup._max_allowed_percent(), 35.5)

    def test_cleanup_orders_system_then_archived_pr_then_old_archived(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            patch(
                "hitch.main.disk_cleanup.cleanup_managed_worktree_path",
                return_value=True,
            ) as mock_cleanup,
        ):
            root = Path(raw)
            system_path = self._managed_path(root, "system")
            pr_path = self._managed_path(root, "pr")
            old_path = self._managed_path(root, "old")
            self._session(
                thread_id="system",
                cwd=system_path,
                hidden_system=True,
            )
            self._session(
                thread_id="pr",
                cwd=pr_path,
                archived=True,
                archived_at=timezone.now(),
                stage="done_merged",
            )
            self._session(
                thread_id="old",
                cwd=old_path,
                archived=True,
                archived_at=timezone.now() - disk_cleanup.ARCHIVED_USER_SESSION_MIN_AGE,
            )

            cleaned = self._run_cleanup(
                root=root,
                sizes=[300, 250, 150],
                mock_cleanup=mock_cleanup,
            )

        self.assertEqual(cleaned, 2)
        self.assertEqual(
            [call.args[0] for call in mock_cleanup.call_args_list],
            [system_path, pr_path],
        )

    def test_pending_proposal_candidate_worktree_is_preserved(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            patch(
                "hitch.main.disk_cleanup.cleanup_managed_worktree_path",
                return_value=True,
            ) as mock_cleanup,
        ):
            root = Path(raw)
            pending_path = self._managed_path(root, "pending")
            old_path = self._managed_path(root, "old")
            pending = self._session(
                thread_id="pending",
                cwd=pending_path,
                hidden_system=True,
            )
            ProposedSession.objects.create(title="Pending", candidate_session=pending)
            self._session(
                thread_id="old",
                cwd=old_path,
                archived=True,
                archived_at=timezone.now() - disk_cleanup.ARCHIVED_USER_SESSION_MIN_AGE,
            )

            cleaned = self._run_cleanup(
                root=root,
                sizes=[300, 100],
                mock_cleanup=mock_cleanup,
            )

        self.assertEqual(cleaned, 1)
        mock_cleanup.assert_called_once_with(old_path)

    def test_cleanup_rechecks_size_after_candidate_was_already_removed(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            patch(
                "hitch.main.disk_cleanup.cleanup_managed_worktree_path",
                side_effect=[False, True],
            ) as mock_cleanup,
        ):
            root = Path(raw)
            first_path = self._managed_path(root, "first")
            second_path = self._managed_path(root, "second")
            archived_at = timezone.now() - disk_cleanup.ARCHIVED_USER_SESSION_MIN_AGE
            self._session(
                thread_id="first",
                cwd=first_path,
                archived=True,
                archived_at=archived_at,
            )
            self._session(
                thread_id="second",
                cwd=second_path,
                archived=True,
                archived_at=archived_at,
            )

            cleaned = self._run_cleanup(
                root=root,
                sizes=[300, 100],
                mock_cleanup=mock_cleanup,
            )

        self.assertEqual(cleaned, 0)
        mock_cleanup.assert_called_once_with(first_path)

    def test_unarchived_user_session_protects_shared_worktree(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            patch(
                "hitch.main.disk_cleanup.cleanup_managed_worktree_path",
                return_value=True,
            ) as mock_cleanup,
        ):
            root = Path(raw)
            shared_path = self._managed_path(root, "shared")
            self._session(thread_id="user", cwd=shared_path)
            self._session(
                thread_id="system",
                cwd=shared_path,
                hidden_system=True,
            )

            cleaned = self._run_cleanup(
                root=root,
                sizes=[300],
                mock_cleanup=mock_cleanup,
            )

        self.assertEqual(cleaned, 0)
        mock_cleanup.assert_not_called()

    def test_recent_archived_user_session_protects_shared_worktree(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            patch(
                "hitch.main.disk_cleanup.cleanup_managed_worktree_path",
                return_value=True,
            ) as mock_cleanup,
        ):
            root = Path(raw)
            shared_path = self._managed_path(root, "shared")
            self._session(
                thread_id="user",
                cwd=shared_path,
                archived=True,
                archived_at=timezone.now(),
            )
            self._session(
                thread_id="system",
                cwd=shared_path,
                hidden_system=True,
            )

            cleaned = self._run_cleanup(
                root=root,
                sizes=[300],
                mock_cleanup=mock_cleanup,
            )

        self.assertEqual(cleaned, 0)
        mock_cleanup.assert_not_called()

    def test_archived_user_without_pr_must_be_archived_for_48_hours(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            patch(
                "hitch.main.disk_cleanup.cleanup_managed_worktree_path",
                return_value=True,
            ) as mock_cleanup,
        ):
            root = Path(raw)
            recent_path = self._managed_path(root, "recent")
            old_path = self._managed_path(root, "old")
            self._session(
                thread_id="recent",
                cwd=recent_path,
                archived=True,
                archived_at=timezone.now(),
            )
            self._session(
                thread_id="old",
                cwd=old_path,
                archived=True,
                archived_at=timezone.now() - disk_cleanup.ARCHIVED_USER_SESSION_MIN_AGE,
            )

            cleaned = self._run_cleanup(
                root=root,
                sizes=[300, 100],
                mock_cleanup=mock_cleanup,
            )

        self.assertEqual(cleaned, 1)
        mock_cleanup.assert_called_once_with(old_path)

    def test_active_system_session_is_not_finished_for_cleanup(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            patch(
                "hitch.main.disk_cleanup.cleanup_managed_worktree_path",
                return_value=True,
            ) as mock_cleanup,
        ):
            root = Path(raw)
            system_path = self._managed_path(root, "system")
            old_path = self._managed_path(root, "old")
            self._session(
                thread_id="system",
                cwd=system_path,
                hidden_system=True,
            )
            CodexInstance.objects.create(
                pid=123,
                thread_id="system",
                cwd=system_path,
                events_path="/tmp/events.jsonl",
                status=CodexInstance.STATUS_RUNNING,
                purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            )
            SystemWorkflow.objects.create(
                kind=SystemWorkflow.KIND_AUTONOMOUS_GOAL_RUN,
                main_thread_id="system",
                cwd=system_path,
                status=SystemWorkflow.STATUS_RUNNING,
            )
            self._session(
                thread_id="old",
                cwd=old_path,
                archived=True,
                archived_at=timezone.now() - disk_cleanup.ARCHIVED_USER_SESSION_MIN_AGE,
            )

            cleaned = self._run_cleanup(
                root=root,
                sizes=[300, 100],
                mock_cleanup=mock_cleanup,
            )

        self.assertEqual(cleaned, 1)
        mock_cleanup.assert_called_once_with(old_path)
