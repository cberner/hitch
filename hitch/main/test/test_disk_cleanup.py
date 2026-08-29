from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import override
from unittest.mock import MagicMock, call, patch

from django.test import SimpleTestCase, TestCase, override_settings
from django.utils import timezone

from hitch.main.models import (
    CodexInstance,
    GlobalSettings,
    ProposedSession,
    SessionMetadata,
    SystemWorkflow,
)
from hitch.main.runtime import disk_cleanup


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
                "hitch.main.runtime.disk_cleanup.shutil.disk_usage",
                return_value=SimpleNamespace(total=1000, used=used),
            ),
            patch("hitch.main.runtime.disk_cleanup._directory_size", side_effect=sizes),
        ):
            return disk_cleanup.cleanup_hitch_disk_usage_if_needed()

    @override_settings(HITCH_MAX_ALLOWED_DISK_SPACE_PERCENT=35)
    @patch("hitch.main.runtime.disk_cleanup.logger.exception")
    def test_max_allowed_percent_falls_back_when_saved_global_read_fails(self, mock_log_exception: MagicMock) -> None:
        with patch(
            "hitch.main.runtime.disk_cleanup.GlobalSettings.objects.filter",
            side_effect=RuntimeError("database unavailable"),
        ):
            self.assertEqual(disk_cleanup._max_allowed_percent(), 35.0)

        mock_log_exception.assert_called_once_with("failed to load saved Hitch disk usage setting")

    @override_settings(HITCH_MAX_ALLOWED_DISK_SPACE_PERCENT=35)
    @patch("hitch.main.runtime.disk_cleanup.logger.warning")
    def test_invalid_saved_global_max_allowed_percent_uses_default(self, mock_warning: MagicMock) -> None:
        GlobalSettings.objects.create(pk=GlobalSettings.SINGLETON_PK, disk_usage_max_percent=100.1)

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
                "hitch.main.runtime.disk_cleanup.cleanup_managed_worktree_path",
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
                sizes=[300, 50, 50, 50, 200],
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
                "hitch.main.runtime.disk_cleanup.cleanup_managed_worktree_path",
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
            patch("hitch.main.runtime.disk_cleanup._directory_size") as mock_size,
            patch(
                "hitch.main.runtime.disk_cleanup.cleanup_managed_worktree_path",
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
                    "hitch.main.runtime.disk_cleanup.shutil.disk_usage",
                    return_value=SimpleNamespace(total=1000, used=200),
                ),
            ):
                cleaned = disk_cleanup.cleanup_hitch_disk_usage_if_needed()

        self.assertEqual(cleaned, 0)
        mock_size.assert_not_called()
        mock_cleanup.assert_not_called()

    def test_prunes_only_oversized_finished_event_logs_in_managed_dir(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            events_dir = root / "events"
            events_dir.mkdir()
            completed_path = events_dir / "completed.jsonl"
            active_path = events_dir / "active.jsonl"
            outside_path = root / "outside.jsonl"
            diff_event = '{"method": "turn/diff/updated", "payload": {"diff": "large"}}\n'
            terminal_event = '{"method": "turn/completed", "payload": {}}\n'
            for path in (completed_path, active_path, outside_path):
                path.write_text(diff_event + terminal_event, encoding="utf-8")
            CodexInstance.objects.create(
                pid=1,
                thread_id="completed",
                cwd="/repo",
                events_path=str(completed_path),
                status=CodexInstance.STATUS_COMPLETED,
            )
            CodexInstance.objects.create(
                pid=2,
                thread_id="active",
                cwd="/repo",
                events_path=str(active_path),
                status=CodexInstance.STATUS_RUNNING,
            )
            CodexInstance.objects.create(
                pid=3,
                thread_id="outside",
                cwd="/repo",
                events_path=str(outside_path),
                status=CodexInstance.STATUS_COMPLETED,
            )

            with (
                override_settings(CODEX_EVENTS_DIR=events_dir),
                patch.object(
                    disk_cleanup,
                    "LEGACY_DIFF_EVENT_COMPACTION_MIN_BYTES",
                    1,
                ),
            ):
                freed = disk_cleanup._prune_oversized_finished_event_logs()

            completed = completed_path.read_text(encoding="utf-8")
            active = active_path.read_text(encoding="utf-8")
            outside = outside_path.read_text(encoding="utf-8")

        self.assertGreater(freed, 0)
        self.assertNotIn("turn/diff/updated", completed)
        self.assertIn("turn/completed", completed)
        self.assertIn("turn/diff/updated", active)
        self.assertIn("turn/diff/updated", outside)

    def test_event_compaction_can_avoid_worktree_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
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
                    "hitch.main.runtime.disk_cleanup.shutil.disk_usage",
                    return_value=SimpleNamespace(total=1000, used=1000),
                ),
                patch(
                    "hitch.main.runtime.disk_cleanup._directory_size",
                    return_value=300,
                ),
                patch(
                    "hitch.main.runtime.disk_cleanup._prune_oversized_finished_event_logs",
                    return_value=150,
                ) as mock_prune,
                patch("hitch.main.runtime.disk_cleanup.invalidate_hitch_home_disk_usage") as mock_invalidate,
                patch("hitch.main.runtime.disk_cleanup.cleanup_managed_worktree_path") as mock_cleanup,
            ):
                cleaned = disk_cleanup.cleanup_hitch_disk_usage_if_needed()

        self.assertEqual(cleaned, 0)
        mock_prune.assert_called_once_with()
        mock_invalidate.assert_called_once_with()
        mock_cleanup.assert_not_called()

    def test_noop_removal_counts_only_successful_planned_deletions(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            patch(
                "hitch.main.runtime.disk_cleanup.cleanup_managed_worktree_path",
                side_effect=[False, True],
            ) as mock_cleanup,
            patch(
                "hitch.main.runtime.disk_cleanup._directory_size",
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
                    "hitch.main.runtime.disk_cleanup.shutil.disk_usage",
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

    def test_duplicate_candidate_worktree_is_removed_once(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            patch(
                "hitch.main.runtime.disk_cleanup.cleanup_managed_worktree_path",
                return_value=True,
            ) as mock_cleanup,
        ):
            root = Path(raw)
            shared_path = self._managed_path(root, "shared")
            archived_at = timezone.now() - disk_cleanup.ARCHIVED_USER_SESSION_MIN_AGE
            self._session(
                thread_id="first",
                cwd=shared_path,
                archived=True,
                archived_at=archived_at,
            )
            self._session(
                thread_id="second",
                cwd=shared_path,
                archived=True,
                archived_at=archived_at,
            )

            cleaned = self._run_cleanup(
                root=root,
                sizes=[300, 50],
                mock_cleanup=mock_cleanup,
            )

        self.assertEqual(cleaned, 1)
        mock_cleanup.assert_called_once_with(shared_path)

    def test_zero_usage_candidate_is_skipped(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            patch(
                "hitch.main.runtime.disk_cleanup.cleanup_managed_worktree_path",
                return_value=True,
            ) as mock_cleanup,
        ):
            root = Path(raw)
            old_path = self._managed_path(root, "empty")
            self._session(
                thread_id="empty",
                cwd=old_path,
                archived=True,
                archived_at=timezone.now() - disk_cleanup.ARCHIVED_USER_SESSION_MIN_AGE,
            )

            cleaned = self._run_cleanup(
                root=root,
                sizes=[300, 0],
                mock_cleanup=mock_cleanup,
            )

        self.assertEqual(cleaned, 0)
        mock_cleanup.assert_not_called()

    def test_shared_hardlink_usage_is_rechecked_before_stopping(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            hitch_home = root / ".hitch"
            managed = hitch_home / "worktrees"
            first = managed / "repo" / "first"
            second = managed / "repo" / "second"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            first_blob = first / "blob"
            first_blob.write_bytes(b"x" * 4096)
            os.link(first_blob, second / "blob")
            archived_at = timezone.now() - disk_cleanup.ARCHIVED_USER_SESSION_MIN_AGE
            self._session(
                thread_id="first",
                cwd=str(first),
                archived=True,
                archived_at=archived_at,
            )
            self._session(
                thread_id="second",
                cwd=str(second),
                archived=True,
                archived_at=archived_at,
            )
            used_bytes = disk_cleanup._directory_size(hitch_home)
            first_unique_bytes = disk_cleanup._directory_size(first) - disk_cleanup._allocated_size(first_blob.lstat())
            limit_bytes = used_bytes - first_unique_bytes - 1
            removed_paths: list[str] = []

            def cleanup_path(path: str) -> bool:
                removed_paths.append(path)
                shutil.rmtree(path)
                return True

            with (
                override_settings(
                    HITCH_HOME_DIR=hitch_home,
                    HITCH_WORKTREES_DIR=managed,
                    HITCH_MAX_ALLOWED_DISK_SPACE_PERCENT=20,
                ),
                patch(
                    "hitch.main.runtime.disk_cleanup.shutil.disk_usage",
                    return_value=SimpleNamespace(
                        total=limit_bytes * 5,
                        used=limit_bytes * 5,
                    ),
                ),
                patch(
                    "hitch.main.runtime.disk_cleanup.cleanup_managed_worktree_path",
                    side_effect=cleanup_path,
                ),
            ):
                cleaned = disk_cleanup.cleanup_hitch_disk_usage_if_needed()

            self.assertEqual(cleaned, 2)
            self.assertEqual(removed_paths, [str(first), str(second)])
            self.assertLessEqual(disk_cleanup._directory_size(hitch_home), limit_bytes)

    def test_failed_removal_falls_back_to_later_candidate(self) -> None:
        from hitch.main.worktrees import WorktreeCleanupError

        with (
            tempfile.TemporaryDirectory() as raw,
            patch(
                "hitch.main.runtime.disk_cleanup.cleanup_managed_worktree_path",
                side_effect=[WorktreeCleanupError("boom"), True],
            ) as mock_cleanup,
            patch(
                "hitch.main.runtime.disk_cleanup._directory_size",
                side_effect=[300, 150, 150],
            ) as mock_size,
            patch("hitch.main.runtime.disk_cleanup.logger.exception"),
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
                    "hitch.main.runtime.disk_cleanup.shutil.disk_usage",
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
                "hitch.main.runtime.disk_cleanup.cleanup_managed_worktree_path",
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

    def test_stale_visible_system_session_does_not_protect_worktree(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            patch(
                "hitch.main.runtime.disk_cleanup.cleanup_managed_worktree_path",
                return_value=True,
            ) as mock_cleanup,
        ):
            root = Path(raw)
            shared_path = self._managed_path(root, "shared")
            self._session(thread_id="system", cwd=shared_path)
            CodexInstance.objects.create(
                pid=123,
                thread_id="system",
                cwd=shared_path,
                events_path="/tmp/events.jsonl",
                status=CodexInstance.STATUS_COMPLETED,
                purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            )

            cleaned = self._run_cleanup(
                root=root,
                sizes=[300, 150],
                mock_cleanup=mock_cleanup,
            )

        self.assertEqual(cleaned, 1)
        mock_cleanup.assert_called_once_with(shared_path)

    def test_recent_archived_user_session_protects_shared_worktree(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            patch(
                "hitch.main.runtime.disk_cleanup.cleanup_managed_worktree_path",
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

    def test_orphan_discovery_skips_unmanaged_duplicate_and_invalid_paths(
        self,
    ) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            patch(
                "hitch.main.runtime.disk_cleanup.cleanup_managed_worktree_path",
                return_value=True,
            ) as mock_cleanup,
        ):
            root = Path(raw)
            old_created_at = timezone.now() - disk_cleanup.ARCHIVED_USER_SESSION_MIN_AGE - timedelta(hours=1)
            metadata_path = self._managed_path(root, "metadata")
            outside_path = root / "outside" / f"{old_created_at.strftime('%Y%m%d%H%M%S')}-abcdef12"
            bad_shape_path = root / "managed" / "repo" / "not-a-managed-name"
            bad_date_path = root / "managed" / "repo" / "20261301121212-abcdef12"
            valid_path = root / "managed" / "repo" / f"{old_created_at.strftime('%Y%m%d%H%M%S')}-12345678"
            self._session(thread_id="visible", cwd=metadata_path)
            with patch(
                "hitch.main.runtime.disk_cleanup.discover_managed_worktrees",
                return_value=[
                    outside_path,
                    Path(metadata_path),
                    bad_shape_path,
                    bad_date_path,
                    valid_path,
                ],
            ):
                cleaned = self._run_cleanup(
                    root=root,
                    sizes=[300, 150],
                    mock_cleanup=mock_cleanup,
                )

        self.assertEqual(cleaned, 1)
        mock_cleanup.assert_called_once_with(str(valid_path))

    def test_recent_orphaned_managed_worktree_is_preserved(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            patch(
                "hitch.main.runtime.disk_cleanup.cleanup_managed_worktree_path",
                return_value=True,
            ) as mock_cleanup,
        ):
            root = Path(raw)
            orphan_path = root / "managed" / "repo" / f"{timezone.now().strftime('%Y%m%d%H%M%S')}-abcdef12"
            with patch(
                "hitch.main.runtime.disk_cleanup.discover_managed_worktrees",
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
                "hitch.main.runtime.disk_cleanup.cleanup_managed_worktree_path",
                return_value=True,
            ) as mock_cleanup,
        ):
            root = Path(raw)
            old_created_at = timezone.now() - disk_cleanup.ARCHIVED_USER_SESSION_MIN_AGE - timedelta(hours=1)
            orphan_path = root / "managed" / "repo" / f"{old_created_at.strftime('%Y%m%d%H%M%S')}-abcdef12"
            CodexInstance.objects.create(
                pid=123,
                thread_id="active",
                cwd=str(orphan_path),
                events_path="/tmp/events.jsonl",
                status=CodexInstance.STATUS_RUNNING,
            )
            with patch(
                "hitch.main.runtime.disk_cleanup.discover_managed_worktrees",
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
                "hitch.main.runtime.disk_cleanup.cleanup_managed_worktree_path",
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

    def test_active_stack_fork_worktree_is_not_finished_for_cleanup(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            patch(
                "hitch.main.runtime.disk_cleanup.cleanup_managed_worktree_path",
                return_value=True,
            ) as mock_cleanup,
        ):
            root = Path(raw)
            fork_path = self._managed_path(root, "fork")
            self._session(
                thread_id="candidate",
                cwd=fork_path,
                hidden_system=True,
            )
            SystemWorkflow.objects.create(
                kind=SystemWorkflow.KIND_AUTONOMOUS_GOAL_RUN,
                main_thread_id="autonomous-goal-1",
                cwd="/repo",
                status=SystemWorkflow.STATUS_RUNNING,
                state={"stacked_diff_fork_from_cwd": fork_path},
            )

            cleaned = self._run_cleanup(
                root=root,
                sizes=[300],
                mock_cleanup=mock_cleanup,
            )

        self.assertEqual(cleaned, 0)
        mock_cleanup.assert_not_called()

    def test_hidden_stack_proposal_worktree_is_not_finished_for_cleanup(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            patch(
                "hitch.main.runtime.disk_cleanup.cleanup_managed_worktree_path",
                return_value=True,
            ) as mock_cleanup,
        ):
            root = Path(raw)
            proposal_path = self._managed_path(root, "proposal")
            proposal_session = self._session(
                thread_id="proposal",
                cwd=proposal_path,
                hidden_system=True,
            )
            ProposedSession.objects.create(
                title="Hidden stack proposal",
                candidate_session=proposal_session,
                outcome_status=ProposedSession.OUTCOME_DISMISSED,
                outcome_metadata={"stacked_diff_hidden_until_complete": True},
            )

            cleaned = self._run_cleanup(
                root=root,
                sizes=[300],
                mock_cleanup=mock_cleanup,
            )

        self.assertEqual(cleaned, 0)
        mock_cleanup.assert_not_called()


def _naive_directory_size(path: Path) -> int:
    """Sum allocated size for every entry without inode dedup (pre-fix behavior)."""
    total = disk_cleanup._allocated_size(path.lstat())
    if path.is_dir() and not path.is_symlink():
        for child in path.iterdir():
            total += _naive_directory_size(child)
    return total


class DirectorySizeTests(TestCase):
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


class DiskUsageSnapshotTests(SimpleTestCase):
    @override
    def setUp(self) -> None:
        self._reset_snapshot()

    @override
    def tearDown(self) -> None:
        self._reset_snapshot()

    @staticmethod
    def _reset_snapshot() -> None:
        with disk_cleanup._disk_usage_snapshot_lock:
            disk_cleanup._disk_usage_snapshot = None
            disk_cleanup._disk_usage_refreshing = False
            disk_cleanup._disk_usage_generation = 0

    @staticmethod
    def _run_scheduled_refresh(mock_thread: MagicMock, index: int = -1) -> None:
        call = mock_thread.call_args_list[index]
        call.kwargs["target"](*call.kwargs["args"])

    def test_refresh_is_single_flight(self) -> None:
        with patch("hitch.main.runtime.disk_cleanup.threading.Thread") as mock_thread:
            self.assertIsNone(disk_cleanup.cached_hitch_home_disk_usage())
            self.assertIsNone(disk_cleanup.cached_hitch_home_disk_usage())

        mock_thread.assert_called_once()
        mock_thread.return_value.start.assert_called_once_with()

    def test_refresh_thread_start_failure_allows_retry(self) -> None:
        with (
            patch("hitch.main.runtime.disk_cleanup.threading.Thread") as mock_thread,
            patch.object(disk_cleanup.logger, "exception") as mock_log,
        ):
            mock_thread.return_value.start.side_effect = [
                RuntimeError("cannot start thread"),
                None,
            ]

            self.assertIsNone(disk_cleanup.cached_hitch_home_disk_usage())
            self.assertFalse(disk_cleanup._disk_usage_refreshing)
            self.assertIsNone(disk_cleanup.cached_hitch_home_disk_usage())

        self.assertEqual(mock_thread.call_count, 2)
        self.assertTrue(disk_cleanup._disk_usage_refreshing)
        mock_log.assert_called_once_with("failed to start Hitch disk usage refresh")

    def test_fresh_snapshot_is_reused_and_stale_snapshot_refreshes(self) -> None:
        now = timezone.now()
        usage = disk_cleanup.HitchDiskUsage(100, 200, 1000)
        disk_cleanup._disk_usage_snapshot = disk_cleanup._DiskUsageSnapshot(
            captured_at=now,
            invalidation_token="token",
            usage=usage,
        )

        with (
            patch("hitch.main.runtime.disk_cleanup.threading.Thread") as mock_thread,
            patch("hitch.main.runtime.disk_cleanup.timezone.now", return_value=now),
            patch.object(disk_cleanup, "_disk_usage_invalidation_token", return_value="token"),
            patch.object(disk_cleanup, "_max_allowed_percent", return_value=20.0),
        ):
            self.assertEqual(disk_cleanup.cached_hitch_home_disk_usage(), usage)
            mock_thread.assert_not_called()

        with (
            patch("hitch.main.runtime.disk_cleanup.threading.Thread") as mock_thread,
            patch(
                "hitch.main.runtime.disk_cleanup.timezone.now",
                return_value=now + disk_cleanup._DISK_USAGE_SNAPSHOT_TTL,
            ),
            patch.object(disk_cleanup, "_disk_usage_invalidation_token", return_value="token"),
            patch.object(disk_cleanup, "_max_allowed_percent", return_value=20.0),
        ):
            self.assertEqual(disk_cleanup.cached_hitch_home_disk_usage(), usage)
            mock_thread.assert_called_once()

    def test_failed_refresh_can_be_retried(self) -> None:
        now = timezone.now()
        stale_usage = disk_cleanup.HitchDiskUsage(100, 200, 1000)
        disk_cleanup._disk_usage_snapshot = disk_cleanup._DiskUsageSnapshot(
            captured_at=now - disk_cleanup._DISK_USAGE_SNAPSHOT_TTL,
            invalidation_token="token",
            usage=stale_usage,
        )
        with (
            patch("hitch.main.runtime.disk_cleanup.threading.Thread") as mock_thread,
            patch("hitch.main.runtime.disk_cleanup.timezone.now", return_value=now),
            patch.object(
                disk_cleanup,
                "hitch_home_disk_usage",
                side_effect=RuntimeError("scan failed"),
            ),
            patch.object(disk_cleanup, "_disk_usage_invalidation_token", return_value="token"),
            patch.object(disk_cleanup, "_max_allowed_percent", return_value=20.0),
            patch("hitch.main.runtime.disk_cleanup.close_old_connections") as mock_close,
            patch.object(disk_cleanup.logger, "exception") as mock_log,
        ):
            self.assertEqual(disk_cleanup.cached_hitch_home_disk_usage(), stale_usage)
            self._run_scheduled_refresh(mock_thread)
            self.assertIsNone(disk_cleanup.cached_hitch_home_disk_usage())

        self.assertEqual(mock_thread.call_count, 2)
        self.assertEqual(mock_close.call_count, 2)
        mock_log.assert_called_once_with("failed to refresh Hitch disk usage snapshot")

    def test_shared_token_expires_snapshot_from_another_process(self) -> None:
        usage = disk_cleanup.HitchDiskUsage(100, 200, 1000)
        with tempfile.TemporaryDirectory() as raw, override_settings(HITCH_HOME_DIR=Path(raw)):
            old_token = disk_cleanup._disk_usage_invalidation_token()
            disk_cleanup._disk_usage_snapshot = disk_cleanup._DiskUsageSnapshot(
                captured_at=timezone.now(),
                invalidation_token=old_token,
                usage=usage,
            )

            disk_cleanup._publish_disk_usage_invalidation()

            with patch("hitch.main.runtime.disk_cleanup.threading.Thread") as mock_thread:
                self.assertIsNone(disk_cleanup.cached_hitch_home_disk_usage())

        self.assertIsNone(disk_cleanup._disk_usage_snapshot)
        mock_thread.assert_called_once()

    def test_invalidate_clears_snapshot_and_advances_generation(self) -> None:
        disk_cleanup._disk_usage_snapshot = disk_cleanup._DiskUsageSnapshot(
            captured_at=timezone.now(),
            invalidation_token="token",
            usage=disk_cleanup.HitchDiskUsage(100, 200, 1000),
        )
        disk_cleanup._disk_usage_generation = 7

        with patch.object(disk_cleanup, "_publish_disk_usage_invalidation") as mock_publish:
            disk_cleanup.invalidate_hitch_home_disk_usage()

        mock_publish.assert_called_once_with()
        self.assertIsNone(disk_cleanup._disk_usage_snapshot)
        self.assertEqual(disk_cleanup._disk_usage_generation, 8)

    def test_shared_token_read_failure_is_nonfatal(self) -> None:
        token_path = MagicMock(spec=Path)
        token_path.read_text.side_effect = OSError("token unreadable")
        with (
            patch.object(
                disk_cleanup,
                "_disk_usage_invalidation_path",
                return_value=token_path,
            ),
            patch.object(disk_cleanup.logger, "exception") as mock_log,
        ):
            self.assertEqual(disk_cleanup._disk_usage_invalidation_token(), "")

        token_path.read_text.assert_called_once_with(encoding="utf-8")
        mock_log.assert_called_once_with("failed to read Hitch disk usage invalidation token")

    def test_shared_token_publish_failures_are_nonfatal(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(HITCH_HOME_DIR=Path(raw)),
            patch.object(Path, "write_text", side_effect=OSError("write failed")),
            patch.object(Path, "unlink", side_effect=OSError("unlink failed")),
            patch.object(disk_cleanup.logger, "exception") as mock_log,
        ):
            disk_cleanup._publish_disk_usage_invalidation()

        self.assertEqual(
            mock_log.call_args_list,
            [
                call("failed to publish Hitch disk usage invalidation"),
                call("failed to remove disk usage invalidation temporary file"),
            ],
        )
