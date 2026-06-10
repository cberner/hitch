import os
import sys
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase, TestCase, override_settings

import hitch.main as main_app
from hitch.main import workflow_maintenance
from hitch.main.apps import MainConfig
from hitch.main.goals import auto_proposals
from hitch.main.models import SessionMetadata


class AutoProposalSchedulerTests(SimpleTestCase):
    @override_settings(TESTING=True)
    @patch.dict(os.environ, {}, clear=True)
    @patch.object(sys, "argv", ["manage.py", "runserver", "--noreload"])
    def test_scheduler_disabled_during_tests_by_default(self) -> None:
        self.assertFalse(auto_proposals._auto_proposal_scheduler_enabled())

    @override_settings(TESTING=False)
    @patch.dict(os.environ, {"RUN_MAIN": "true"}, clear=True)
    @patch.object(sys, "argv", ["manage.py", "runserver"])
    def test_scheduler_enabled_in_runserver_child(self) -> None:
        self.assertTrue(auto_proposals._auto_proposal_scheduler_enabled())

    @override_settings(TESTING=False)
    @patch.dict(os.environ, {}, clear=True)
    @patch.object(sys, "argv", ["manage.py", "runserver", "--noreload"])
    def test_scheduler_enabled_in_runserver_noreload_process(self) -> None:
        self.assertTrue(auto_proposals._auto_proposal_scheduler_enabled())

    @override_settings(TESTING=False)
    @patch.dict(os.environ, {}, clear=True)
    @patch.object(sys, "argv", ["manage.py", "runserver"])
    def test_scheduler_disabled_in_runserver_autoreloader_parent(self) -> None:
        self.assertFalse(auto_proposals._auto_proposal_scheduler_enabled())

    @override_settings(TESTING=False)
    @patch.dict(os.environ, {}, clear=True)
    @patch.object(sys, "argv", ["manage.py", "migrate"])
    def test_scheduler_disabled_for_management_commands(self) -> None:
        self.assertFalse(auto_proposals._auto_proposal_scheduler_enabled())

    @override_settings(TESTING=True)
    @patch.dict(os.environ, {"HITCH_AUTO_PROPOSAL_SCHEDULER": "1"}, clear=True)
    @patch.object(sys, "argv", ["manage.py", "migrate"])
    def test_scheduler_env_override_can_enable(self) -> None:
        self.assertTrue(auto_proposals._auto_proposal_scheduler_enabled())

    @patch(
        "hitch.main.goals.auto_proposals.system_agents.maybe_start_auto_proposal_workflows",
        return_value=2,
    )
    @patch("hitch.main.goals.auto_proposals._refresh_unarchived_session_state_best_effort")
    @patch("hitch.main.goals.auto_proposals.codex_pool.reconcile_dead")
    def test_scheduler_tick_reconciles_and_starts_auto_proposals(
        self,
        mock_reconcile_dead: MagicMock,
        mock_refresh: MagicMock,
        mock_start: MagicMock,
    ) -> None:
        auto_proposals._run_auto_proposal_scheduler_tick()

        mock_reconcile_dead.assert_called_once_with()
        mock_refresh.assert_called_once_with(None, start_cursor="")
        mock_start.assert_called_once_with()

    @patch("hitch.main.goals.auto_proposals.logger.exception")
    @patch("hitch.main.goals.auto_proposals.system_agents.maybe_start_auto_proposal_workflows")
    @patch("hitch.main.goals.auto_proposals._refresh_unarchived_session_state_best_effort")
    @patch("hitch.main.goals.auto_proposals.codex_pool.reconcile_dead")
    def test_scheduler_tick_keeps_running_after_errors(
        self,
        mock_reconcile_dead: MagicMock,
        mock_refresh: MagicMock,
        mock_start: MagicMock,
        mock_log_exception: MagicMock,
    ) -> None:
        mock_reconcile_dead.side_effect = RuntimeError("boom")

        auto_proposals._run_auto_proposal_scheduler_tick()

        mock_refresh.assert_not_called()
        mock_start.assert_not_called()
        mock_log_exception.assert_called_once_with(
            "failed to run auto-proposal scheduler tick"
        )


class WorkflowMaintenanceSchedulerTests(SimpleTestCase):
    @override_settings(TESTING=True)
    @patch.dict(os.environ, {}, clear=True)
    @patch.object(sys, "argv", ["manage.py", "runserver", "--noreload"])
    def test_scheduler_disabled_during_tests_by_default(self) -> None:
        self.assertFalse(
            workflow_maintenance._workflow_maintenance_scheduler_enabled()
        )

    @override_settings(TESTING=False)
    @patch.dict(os.environ, {"RUN_MAIN": "true"}, clear=True)
    @patch.object(sys, "argv", ["manage.py", "runserver"])
    def test_scheduler_enabled_in_runserver_child(self) -> None:
        self.assertTrue(
            workflow_maintenance._workflow_maintenance_scheduler_enabled()
        )

    @override_settings(TESTING=False)
    @patch.dict(os.environ, {}, clear=True)
    @patch.object(sys, "argv", ["manage.py", "runserver", "--noreload"])
    def test_scheduler_enabled_in_runserver_noreload_process(self) -> None:
        self.assertTrue(
            workflow_maintenance._workflow_maintenance_scheduler_enabled()
        )

    @override_settings(TESTING=False)
    @patch.dict(os.environ, {}, clear=True)
    @patch.object(sys, "argv", ["manage.py", "runserver"])
    def test_scheduler_disabled_in_runserver_autoreloader_parent(self) -> None:
        self.assertFalse(
            workflow_maintenance._workflow_maintenance_scheduler_enabled()
        )

    @override_settings(TESTING=False)
    @patch.dict(os.environ, {}, clear=True)
    @patch.object(sys, "argv", ["manage.py", "migrate"])
    def test_scheduler_disabled_for_management_commands(self) -> None:
        self.assertFalse(
            workflow_maintenance._workflow_maintenance_scheduler_enabled()
        )

    @override_settings(TESTING=False)
    @patch.dict(os.environ, {}, clear=True)
    @patch.object(sys, "argv", ["gunicorn", "hitch.wsgi:application"])
    def test_scheduler_enabled_for_wsgi_server_process(self) -> None:
        self.assertTrue(
            workflow_maintenance._workflow_maintenance_scheduler_enabled()
        )

    @patch(
        "hitch.main.workflow_maintenance.disk_cleanup.run_finished_session_disk_cleanup"
    )
    def test_disk_usage_cleanup_runs_every_ten_minutes(
        self, mock_cleanup: MagicMock
    ) -> None:
        next_due = 100.0

        next_due = workflow_maintenance._run_due_disk_usage_cleanup(
            next_due_at=next_due, now=99.0
        )
        self.assertEqual(next_due, 100.0)
        mock_cleanup.assert_not_called()

        next_due = workflow_maintenance._run_due_disk_usage_cleanup(
            next_due_at=next_due, now=100.0
        )
        self.assertEqual(
            next_due,
            100.0 + workflow_maintenance._DISK_USAGE_CLEANUP_INTERVAL_SECONDS,
        )
        mock_cleanup.assert_called_once_with()

        workflow_maintenance._run_due_disk_usage_cleanup(
            next_due_at=next_due, now=699.0
        )
        self.assertEqual(mock_cleanup.call_count, 1)

        workflow_maintenance._run_due_disk_usage_cleanup(
            next_due_at=next_due, now=700.0
        )
        self.assertEqual(mock_cleanup.call_count, 2)

    def test_scheduler_loop_runs_due_disk_cleanup_between_ticks(self) -> None:
        class StopSchedulerError(Exception):
            pass

        fake_stop = MagicMock()
        fake_stop.wait.side_effect = StopSchedulerError
        with (
            patch(
                "hitch.main.workflow_maintenance.threading.Event",
                return_value=fake_stop,
            ),
            patch(
                "hitch.main.workflow_maintenance.time.monotonic",
                return_value=10.0,
            ),
            patch(
                "hitch.main.workflow_maintenance._run_workflow_maintenance_scheduler_tick"
            ) as mock_tick,
            patch(
                "hitch.main.workflow_maintenance._run_due_stale_blocked_archive",
                return_value=999.0,
            ) as mock_stale_archive,
            patch(
                "hitch.main.workflow_maintenance._run_due_disk_usage_cleanup",
                return_value=999.0,
            ) as mock_disk_cleanup,
            self.assertRaises(StopSchedulerError),
        ):
            workflow_maintenance._workflow_maintenance_scheduler_loop()

        mock_tick.assert_called_once_with()
        mock_stale_archive.assert_called_once_with(
            next_due_at=10.0
            + workflow_maintenance._STALE_BLOCKED_ARCHIVE_INTERVAL_SECONDS
        )
        mock_disk_cleanup.assert_called_once_with(
            next_due_at=10.0
            + workflow_maintenance._DISK_USAGE_CLEANUP_INTERVAL_SECONDS
        )
        fake_stop.wait.assert_called_once_with(
            workflow_maintenance._WORKFLOW_MAINTENANCE_INTERVAL_SECONDS
        )

    def test_scheduler_loop_archives_stale_blocked_before_disk_cleanup(self) -> None:
        # Archiving unpins worktrees, so it must run ahead of disk cleanup on the
        # same loop iteration for the freed space to be reclaimable that tick.
        class StopSchedulerError(Exception):
            pass

        order: list[str] = []

        def record_archive(**_: object) -> float:
            order.append("archive")
            return 999.0

        def record_disk(**_: object) -> float:
            order.append("disk")
            return 999.0

        fake_stop = MagicMock()
        fake_stop.wait.side_effect = StopSchedulerError
        with (
            patch(
                "hitch.main.workflow_maintenance.threading.Event",
                return_value=fake_stop,
            ),
            patch(
                "hitch.main.workflow_maintenance.time.monotonic",
                return_value=10.0,
            ),
            patch(
                "hitch.main.workflow_maintenance._run_workflow_maintenance_scheduler_tick"
            ),
            patch(
                "hitch.main.workflow_maintenance._run_due_stale_blocked_archive",
                side_effect=record_archive,
            ),
            patch(
                "hitch.main.workflow_maintenance._run_due_disk_usage_cleanup",
                side_effect=record_disk,
            ),
            self.assertRaises(StopSchedulerError),
        ):
            workflow_maintenance._workflow_maintenance_scheduler_loop()

        self.assertEqual(order, ["archive", "disk"])

    @patch(
        "hitch.main.workflow_maintenance.system_agents.archive_stale_blocked_workflows",
        return_value=[],
    )
    def test_stale_blocked_archive_runs_every_hour(
        self, mock_archive: MagicMock
    ) -> None:
        next_due = 100.0

        next_due = workflow_maintenance._run_due_stale_blocked_archive(
            next_due_at=next_due, now=99.0
        )
        self.assertEqual(next_due, 100.0)
        mock_archive.assert_not_called()

        frozen = datetime(2026, 6, 6, tzinfo=UTC)
        with patch(
            "hitch.main.workflow_maintenance.timezone.now", return_value=frozen
        ):
            next_due = workflow_maintenance._run_due_stale_blocked_archive(
                next_due_at=next_due, now=100.0
            )
        self.assertEqual(
            next_due,
            100.0 + workflow_maintenance._STALE_BLOCKED_ARCHIVE_INTERVAL_SECONDS,
        )
        from hitch.main import system_agents

        mock_archive.assert_called_once_with(
            older_than=frozen - system_agents.STALE_BLOCKED_AGE, apply=True
        )

        # Still on cooldown an hour later minus a second.
        workflow_maintenance._run_due_stale_blocked_archive(
            next_due_at=next_due, now=next_due - 1.0
        )
        self.assertEqual(mock_archive.call_count, 1)

    @patch("hitch.main.workflow_maintenance.logger.exception")
    @patch(
        "hitch.main.workflow_maintenance.system_agents.archive_stale_blocked_workflows",
        side_effect=RuntimeError("archive failed"),
    )
    def test_stale_blocked_archive_failure_is_logged_and_rescheduled(
        self, mock_archive: MagicMock, mock_log_exception: MagicMock
    ) -> None:
        next_due = workflow_maintenance._run_due_stale_blocked_archive(
            next_due_at=100.0, now=100.0
        )

        self.assertEqual(
            next_due,
            100.0 + workflow_maintenance._STALE_BLOCKED_ARCHIVE_INTERVAL_SECONDS,
        )
        mock_archive.assert_called_once()
        mock_log_exception.assert_called_once_with(
            "failed to run scheduled stale blocked workflow archive"
        )

    @patch("hitch.main.workflow_maintenance.logger.exception")
    @patch(
        "hitch.main.workflow_maintenance.disk_cleanup.run_finished_session_disk_cleanup",
        side_effect=RuntimeError("cleanup failed"),
    )
    def test_disk_usage_cleanup_failure_is_logged_and_rescheduled(
        self, mock_cleanup: MagicMock, mock_log_exception: MagicMock
    ) -> None:
        next_due = workflow_maintenance._run_due_disk_usage_cleanup(
            next_due_at=100.0, now=100.0
        )

        self.assertEqual(
            next_due,
            100.0 + workflow_maintenance._DISK_USAGE_CLEANUP_INTERVAL_SECONDS,
        )
        mock_cleanup.assert_called_once_with()
        mock_log_exception.assert_called_once_with(
            "failed to run scheduled Hitch disk cleanup"
        )

    @override_settings(TESTING=False)
    @patch.dict(
        os.environ,
        {"HITCH_AUTO_PROPOSAL_SCHEDULER": "0", "RUN_MAIN": "true"},
        clear=True,
    )
    @patch.object(sys, "argv", ["manage.py", "runserver"])
    def test_scheduler_does_not_follow_auto_proposal_env(self) -> None:
        self.assertTrue(
            workflow_maintenance._workflow_maintenance_scheduler_enabled()
        )

    @patch(
        "hitch.main.workflow_maintenance.disk_cleanup.run_finished_session_disk_cleanup"
    )
    @patch(
        "hitch.main.workflow_maintenance.system_agents.refresh_unarchived_session_pr_stages",
        return_value=1,
    )
    @patch(
        "hitch.main.workflow_maintenance.system_agents.refresh_due_pr_monitor_backoffs",
        return_value=2,
    )
    @patch("hitch.main.workflow_maintenance.codex_pool.reconcile_dead")
    def test_scheduler_tick_reconciles_and_refreshes_pr_monitor_backoffs(
        self,
        mock_reconcile_dead: MagicMock,
        mock_refresh: MagicMock,
        mock_refresh_pr_stages: MagicMock,
        mock_disk_cleanup: MagicMock,
    ) -> None:
        workflow_maintenance._run_workflow_maintenance_scheduler_tick()

        mock_reconcile_dead.assert_called_once_with()
        # Disk cleanup runs only on the separate 10-minute cadence, never as
        # part of the 60-second maintenance tick.
        mock_disk_cleanup.assert_not_called()
        # PR-monitor backoff polling shells out to gh per due monitor, so it is
        # bounded per tick like the PR-stage sweep below.
        mock_refresh.assert_called_once_with(
            limit=workflow_maintenance._PR_MONITOR_BACKOFF_LIMIT_PER_TICK
        )
        # The maintenance scheduler runs under production server commands, so it
        # owns background PR-stage convergence to keep gh out of the request
        # path -- bounded per tick so it can't starve the reconcile sweep.
        mock_refresh_pr_stages.assert_called_once_with(
            limit=workflow_maintenance._PR_STAGE_REFRESH_LIMIT_PER_TICK
        )


class UnarchivedSessionStateRefreshTests(TestCase):
    @patch(
        "hitch.main.goals.auto_proposals.system_agents.refresh_unarchived_session_pr_stages",
        return_value=2,
    )
    @patch("hitch.main.goals.auto_proposals.codex_pool.app_server_config")
    @patch("hitch.main.goals.auto_proposals.Codex")
    def test_refresh_updates_active_codex_metadata_and_pr_stages(
        self,
        mock_codex: MagicMock,
        mock_config: MagicMock,
        mock_refresh_pr_stages: MagicMock,
    ) -> None:
        config = object()
        mock_config.return_value = config
        client = mock_codex.return_value.__enter__.return_value
        client.thread_list.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(
                    id="thread-1",
                    name="Renamed session",
                    preview="Finished the work",
                    cwd="/repo",
                    path="/repo/.codex/rollout.jsonl",
                    created_at=5,
                    updated_at=10,
                    archived=False,
                )
            ],
            next_cursor="",
        )

        result = auto_proposals.refresh_unarchived_session_state()

        self.assertEqual(result.synced, 1)
        self.assertFalse(result.failed)
        self.assertEqual(result.pr_stages_refreshed, 2)
        mock_codex.assert_called_once_with(config=config)
        mock_refresh_pr_stages.assert_called_once_with(
            limit=auto_proposals._PR_STAGE_REFRESH_LIMIT_PER_TICK
        )
        metadata = SessionMetadata.objects.get(thread_id="thread-1")
        self.assertEqual(metadata.codex_display_title, "Renamed session")
        self.assertEqual(metadata.codex_updated_at, datetime.fromtimestamp(10, UTC))

    @patch("hitch.main.goals.auto_proposals.logger.exception")
    @patch(
        "hitch.main.goals.auto_proposals.system_agents.refresh_unarchived_session_pr_stages",
        return_value=1,
    )
    @patch("hitch.main.goals.auto_proposals.codex_pool.app_server_config")
    def test_refresh_still_updates_pr_stages_when_codex_metadata_fails(
        self,
        mock_config: MagicMock,
        mock_refresh_pr_stages: MagicMock,
        mock_log_exception: MagicMock,
    ) -> None:
        mock_config.side_effect = RuntimeError("codex unavailable")

        result = auto_proposals.refresh_unarchived_session_state()

        self.assertEqual(result.synced, 0)
        self.assertTrue(result.failed)
        self.assertEqual(result.pr_stages_refreshed, 1)
        mock_refresh_pr_stages.assert_called_once_with(
            limit=auto_proposals._PR_STAGE_REFRESH_LIMIT_PER_TICK
        )
        mock_log_exception.assert_called_once_with(
            "failed to refresh active Codex session metadata"
        )


class SchedulerCodexReuseTests(SimpleTestCase):
    @patch("hitch.main.goals.auto_proposals.codex_pool.app_server_config")
    @patch("hitch.main.goals.auto_proposals.codex_pool.start_codex")
    def test_get_reuses_one_app_server(
        self, mock_start: MagicMock, mock_config: MagicMock
    ) -> None:
        codex = MagicMock()
        mock_start.return_value = codex
        holder = auto_proposals._SchedulerCodex()

        first = holder.get()
        second = holder.get()

        self.assertIs(first, codex)
        self.assertIs(second, codex)
        mock_start.assert_called_once_with(mock_config.return_value)

    @patch("hitch.main.goals.auto_proposals.codex_pool.app_server_config")
    @patch("hitch.main.goals.auto_proposals.codex_pool.start_codex")
    def test_reset_closes_and_reconnects(
        self, mock_start: MagicMock, _mock_config: MagicMock
    ) -> None:
        first_codex, second_codex = MagicMock(), MagicMock()
        mock_start.side_effect = [first_codex, second_codex]
        holder = auto_proposals._SchedulerCodex()

        self.assertIs(holder.get(), first_codex)
        holder.reset()
        first_codex.close.assert_called_once_with()
        self.assertIs(holder.get(), second_codex)
        self.assertEqual(mock_start.call_count, 2)

    @patch("hitch.main.goals.auto_proposals.refresh_unarchived_session_state")
    @patch("hitch.main.goals.auto_proposals.codex_pool.app_server_config")
    @patch("hitch.main.goals.auto_proposals.codex_pool.start_codex")
    def test_best_effort_reuses_held_codex_across_ticks(
        self,
        mock_start: MagicMock,
        _mock_config: MagicMock,
        mock_refresh: MagicMock,
    ) -> None:
        codex = MagicMock()
        mock_start.return_value = codex
        mock_refresh.return_value = auto_proposals.SessionStateRefreshResult(
            synced=0, failed=False, pr_stages_refreshed=0
        )
        holder = auto_proposals._SchedulerCodex()

        auto_proposals._refresh_unarchived_session_state_best_effort(holder)
        auto_proposals._refresh_unarchived_session_state_best_effort(holder)

        # One app-server initialized once, reused for both ticks.
        self.assertEqual(mock_start.call_count, 1)
        self.assertEqual(mock_refresh.call_count, 2)
        mock_refresh.assert_called_with(codex, start_cursor="")

    @patch("hitch.main.goals.auto_proposals.refresh_unarchived_session_state")
    @patch("hitch.main.goals.auto_proposals.codex_pool.app_server_config")
    @patch("hitch.main.goals.auto_proposals.codex_pool.start_codex")
    def test_best_effort_resets_codex_on_failure(
        self,
        mock_start: MagicMock,
        _mock_config: MagicMock,
        mock_refresh: MagicMock,
    ) -> None:
        first_codex, second_codex = MagicMock(), MagicMock()
        mock_start.side_effect = [first_codex, second_codex]
        mock_refresh.side_effect = [
            auto_proposals.SessionStateRefreshResult(
                synced=0, failed=True, pr_stages_refreshed=0
            ),
            auto_proposals.SessionStateRefreshResult(
                synced=0, failed=False, pr_stages_refreshed=0
            ),
        ]
        holder = auto_proposals._SchedulerCodex()

        auto_proposals._refresh_unarchived_session_state_best_effort(holder)
        # A failed refresh drops the (possibly dead) app-server...
        first_codex.close.assert_called_once_with()
        auto_proposals._refresh_unarchived_session_state_best_effort(holder)

        # ...so the next tick reconnects with a fresh one.
        self.assertEqual(mock_start.call_count, 2)
        self.assertEqual(mock_refresh.call_args_list[0].args, (first_codex,))
        self.assertEqual(mock_refresh.call_args_list[1].args, (second_codex,))


class MainConfigTests(SimpleTestCase):
    @patch("hitch.main.workflow_maintenance.start_workflow_maintenance_scheduler")
    @patch("hitch.main.goals.auto_proposals.start_auto_proposal_scheduler")
    def test_ready_starts_schedulers(
        self, mock_auto_start: MagicMock, mock_workflow_start: MagicMock
    ) -> None:
        config = MainConfig("hitch.main", main_app)

        config.ready()

        mock_workflow_start.assert_called_once_with()
        mock_auto_start.assert_called_once_with()
