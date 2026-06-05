from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta
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
        used: int = 1000,
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
                return_value=SimpleNamespace(total=1000, used=used),
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

    @override_settings(HITCH_MAX_ALLOWED_DISK_SPACE_PERCENT=35)
    @patch("hitch.main.disk_cleanup.logger.exception")
    def test_max_allowed_percent_falls_back_when_saved_global_read_fails(
        self, mock_log_exception: MagicMock
    ) -> None:
        with patch(
            "hitch.main.disk_cleanup.GlobalSettings.objects.filter",
            side_effect=RuntimeError("database unavailable"),
        ):
            self.assertEqual(disk_cleanup._max_allowed_percent(), 35.0)

        mock_log_exception.assert_called_once_with(
            "failed to load saved Hitch disk usage setting"
        )

    @override_settings(HITCH_MAX_ALLOWED_DISK_SPACE_PERCENT=35)
    @patch("hitch.main.disk_cleanup.logger.warning")
    def test_invalid_saved_global_max_allowed_percent_uses_default(
        self, mock_warning: MagicMock
    ) -> None:
        GlobalSettings.objects.create(
            pk=GlobalSettings.SINGLETON_PK, disk_usage_max_percent=100.1
        )

        self.assertEqual(
            disk_cleanup._max_allowed_percent(),
            disk_cleanup.DEFAULT_MAX_ALLOWED_DISK_SPACE_PERCENT,
        )
        mock_warning.assert_called_once()

    def test_global_settings_str(self) -> None:
        self.assertEqual(str(GlobalSettings()), "GlobalSettings")

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
                sizes=[300, 50, 50, 50],
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
                sizes=[300, 150],
                mock_cleanup=mock_cleanup,
            )

        self.assertEqual(cleaned, 1)
        mock_cleanup.assert_called_once_with(old_path)

    def test_partition_prefilter_skips_walk_when_under_limit(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            patch("hitch.main.disk_cleanup._directory_size") as mock_size,
            patch(
                "hitch.main.disk_cleanup.cleanup_managed_worktree_path",
            ) as mock_cleanup,
        ):
            root = Path(raw)
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
                    return_value=SimpleNamespace(total=1000, used=200),
                ),
            ):
                cleaned = disk_cleanup.cleanup_hitch_disk_usage_if_needed()

        self.assertEqual(cleaned, 0)
        mock_size.assert_not_called()
        mock_cleanup.assert_not_called()

    def test_noop_removal_counts_only_successful_planned_deletions(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            patch(
                "hitch.main.disk_cleanup.cleanup_managed_worktree_path",
                side_effect=[False, True],
            ) as mock_cleanup,
            patch(
                "hitch.main.disk_cleanup._directory_size",
                side_effect=[300, 150, 150],
            ) as mock_size,
        ):
            root = Path(raw)
            hitch_home = root / ".hitch"
            managed = root / "managed"
            hitch_home.mkdir()
            managed.mkdir()
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
            with (
                override_settings(
                    HITCH_HOME_DIR=hitch_home,
                    HITCH_WORKTREES_DIR=managed,
                    HITCH_MAX_ALLOWED_DISK_SPACE_PERCENT=20,
                ),
                patch(
                    "hitch.main.disk_cleanup.shutil.disk_usage",
                    return_value=SimpleNamespace(total=1000, used=1000),
                ),
            ):
                cleaned = disk_cleanup.cleanup_hitch_disk_usage_if_needed()

        self.assertEqual(cleaned, 1)
        self.assertEqual(mock_size.call_count, 3)
        self.assertEqual(
            [call.args[0] for call in mock_cleanup.call_args_list],
            [first_path, second_path],
        )

    def test_cleanup_plans_enough_deletions_from_worktree_usage_snapshot(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            patch(
                "hitch.main.disk_cleanup.cleanup_managed_worktree_path",
                return_value=True,
            ) as mock_cleanup,
            patch(
                "hitch.main.disk_cleanup._directory_size",
                side_effect=[500, 125, 125, 125, 125],
            ) as mock_size,
        ):
            root = Path(raw)
            hitch_home = root / ".hitch"
            managed = root / "managed"
            hitch_home.mkdir()
            managed.mkdir()
            archived_at = timezone.now() - disk_cleanup.ARCHIVED_USER_SESSION_MIN_AGE
            paths: list[str] = []
            for index in range(4):
                path = self._managed_path(root, f"old-{index}")
                paths.append(path)
                self._session(
                    thread_id=f"old-{index}",
                    cwd=path,
                    archived=True,
                    archived_at=archived_at,
                )
            with (
                override_settings(
                    HITCH_HOME_DIR=hitch_home,
                    HITCH_WORKTREES_DIR=managed,
                    HITCH_MAX_ALLOWED_DISK_SPACE_PERCENT=20,
                ),
                patch(
                    "hitch.main.disk_cleanup.shutil.disk_usage",
                    return_value=SimpleNamespace(total=1000, used=1000),
                ),
            ):
                cleaned = disk_cleanup.cleanup_hitch_disk_usage_if_needed()

        self.assertEqual(cleaned, 3)
        self.assertEqual(mock_size.call_count, 5)
        self.assertEqual(
            [call.args[0] for call in mock_cleanup.call_args_list],
            paths[:3],
        )

    def test_failed_removal_falls_back_to_later_candidate(self) -> None:
        from hitch.main.worktrees import WorktreeCleanupError

        with (
            tempfile.TemporaryDirectory() as raw,
            patch(
                "hitch.main.disk_cleanup.cleanup_managed_worktree_path",
                side_effect=[WorktreeCleanupError("boom"), True],
            ) as mock_cleanup,
            patch(
                "hitch.main.disk_cleanup._directory_size",
                side_effect=[300, 150, 150],
            ) as mock_size,
            patch("hitch.main.disk_cleanup.logger.exception"),
        ):
            root = Path(raw)
            hitch_home = root / ".hitch"
            managed = root / "managed"
            hitch_home.mkdir()
            managed.mkdir()
            old_path = self._managed_path(root, "old")
            fallback_path = self._managed_path(root, "fallback")
            archived_at = timezone.now() - disk_cleanup.ARCHIVED_USER_SESSION_MIN_AGE
            self._session(
                thread_id="old",
                cwd=old_path,
                archived=True,
                archived_at=archived_at,
            )
            self._session(
                thread_id="fallback",
                cwd=fallback_path,
                archived=True,
                archived_at=archived_at,
            )
            with (
                override_settings(
                    HITCH_HOME_DIR=hitch_home,
                    HITCH_WORKTREES_DIR=managed,
                    HITCH_MAX_ALLOWED_DISK_SPACE_PERCENT=20,
                ),
                patch(
                    "hitch.main.disk_cleanup.shutil.disk_usage",
                    return_value=SimpleNamespace(total=1000, used=1000),
                ),
            ):
                cleaned = disk_cleanup.cleanup_hitch_disk_usage_if_needed()

        self.assertEqual(cleaned, 1)
        self.assertEqual(mock_size.call_count, 3)
        self.assertEqual(
            [call.args[0] for call in mock_cleanup.call_args_list],
            [old_path, fallback_path],
        )

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
                sizes=[300, 150],
                mock_cleanup=mock_cleanup,
            )

        self.assertEqual(cleaned, 1)
        mock_cleanup.assert_called_once_with(old_path)

    def test_old_orphaned_managed_worktree_is_cleanup_candidate(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            patch(
                "hitch.main.disk_cleanup.cleanup_managed_worktree_path",
                return_value=True,
            ) as mock_cleanup,
        ):
            root = Path(raw)
            old_created_at = (
                timezone.now()
                - disk_cleanup.ARCHIVED_USER_SESSION_MIN_AGE
                - timedelta(hours=1)
            )
            orphan_path = (
                root
                / "managed"
                / "repo"
                / f"{old_created_at.strftime('%Y%m%d%H%M%S')}-abcdef12"
            )
            with patch(
                "hitch.main.disk_cleanup.discover_managed_worktrees",
                return_value=[orphan_path],
            ):
                cleaned = self._run_cleanup(
                    root=root,
                    sizes=[300, 150],
                    mock_cleanup=mock_cleanup,
                )

        self.assertEqual(cleaned, 1)
        mock_cleanup.assert_called_once_with(str(orphan_path))

    def test_recent_orphaned_managed_worktree_is_preserved(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            patch(
                "hitch.main.disk_cleanup.cleanup_managed_worktree_path",
                return_value=True,
            ) as mock_cleanup,
        ):
            root = Path(raw)
            orphan_path = (
                root
                / "managed"
                / "repo"
                / f"{timezone.now().strftime('%Y%m%d%H%M%S')}-abcdef12"
            )
            with patch(
                "hitch.main.disk_cleanup.discover_managed_worktrees",
                return_value=[orphan_path],
            ):
                cleaned = self._run_cleanup(
                    root=root,
                    sizes=[300],
                    mock_cleanup=mock_cleanup,
                )

        self.assertEqual(cleaned, 0)
        mock_cleanup.assert_not_called()

    def test_active_orphaned_managed_worktree_is_preserved(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            patch(
                "hitch.main.disk_cleanup.cleanup_managed_worktree_path",
                return_value=True,
            ) as mock_cleanup,
        ):
            root = Path(raw)
            old_created_at = (
                timezone.now()
                - disk_cleanup.ARCHIVED_USER_SESSION_MIN_AGE
                - timedelta(hours=1)
            )
            orphan_path = (
                root
                / "managed"
                / "repo"
                / f"{old_created_at.strftime('%Y%m%d%H%M%S')}-abcdef12"
            )
            CodexInstance.objects.create(
                pid=123,
                thread_id="active",
                cwd=str(orphan_path),
                events_path="/tmp/events.jsonl",
                status=CodexInstance.STATUS_RUNNING,
            )
            with patch(
                "hitch.main.disk_cleanup.discover_managed_worktrees",
                return_value=[orphan_path],
            ):
                cleaned = self._run_cleanup(
                    root=root,
                    sizes=[300],
                    mock_cleanup=mock_cleanup,
                )

        self.assertEqual(cleaned, 0)
        mock_cleanup.assert_not_called()

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
                sizes=[300, 150],
                mock_cleanup=mock_cleanup,
            )

        self.assertEqual(cleaned, 1)
        mock_cleanup.assert_called_once_with(old_path)


def _naive_directory_size(path: Path) -> int:
    """Sum allocated size for every entry without inode dedup (pre-fix behavior)."""
    total = disk_cleanup._allocated_size(path.lstat())
    if path.is_dir() and not path.is_symlink():
        for child in path.iterdir():
            total += _naive_directory_size(child)
    return total


class DirectorySizeTests(TestCase):
    def test_hardlinked_file_counted_once(self) -> None:
        # Managed worktrees hardlink shared blobs into a common object store, so
        # a naive walk counts the same inode once per link. Dedupe must drop the
        # duplicate exactly once, leaving directory entries untouched.
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            original = root / "a" / "blob"
            original.parent.mkdir(parents=True)
            original.write_bytes(b"x" * 4096)

            link_dir = root / "b"
            link_dir.mkdir()
            os.link(original, link_dir / "blob")

            allocated = disk_cleanup._allocated_size(original.lstat())
            naive = _naive_directory_size(root)
            total = disk_cleanup._directory_size(root)

        # The duplicate inode is dropped exactly once relative to the naive walk.
        self.assertEqual(total, naive - allocated)

    def test_distinct_files_counted_separately(self) -> None:
        # Two genuinely distinct inodes of the same size must both be summed --
        # dedupe keys on (st_dev, st_ino), not on size.
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "first").write_bytes(b"x" * 4096)
            (root / "second").write_bytes(b"y" * 4096)

            allocated = disk_cleanup._allocated_size((root / "first").lstat())
            total = disk_cleanup._directory_size(root)

        self.assertGreaterEqual(total, 2 * allocated)
