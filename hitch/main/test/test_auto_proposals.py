import os
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase, TestCase, override_settings

import hitch.main as main_app
from hitch.main.apps import MainConfig
from hitch.main.goals import auto_proposals
from hitch.main.models import SessionMetadata
from hitch.main.workflows import workflow_maintenance

_SchedulerEnablementCase = tuple[str, bool, dict[str, str], list[str], bool]
_COMMON_SCHEDULER_ENABLEMENT_CASES: tuple[_SchedulerEnablementCase, ...] = (
    ("tests", True, {}, ["manage.py", "runserver", "--noreload"], False),
    (
        "runserver child",
        False,
        {"RUN_MAIN": "true"},
        ["manage.py", "runserver"],
        True,
    ),
    (
        "runserver noreload",
        False,
        {},
        ["manage.py", "runserver", "--noreload"],
        True,
    ),
    ("autoreloader parent", False, {}, ["manage.py", "runserver"], False),
    ("management command", False, {}, ["manage.py", "migrate"], False),
    ("wsgi server", False, {}, ["gunicorn", "hitch.wsgi:application"], True),
)
_AUTO_PROPOSAL_SCHEDULER_OVERRIDE_CASES: tuple[_SchedulerEnablementCase, ...] = (
    (
        "override cannot enable migration",
        False,
        {"HITCH_AUTO_PROPOSAL_SCHEDULER": "1"},
        ["manage.py", "migrate"],
        False,
    ),
    (
        "override disables server",
        False,
        {"HITCH_AUTO_PROPOSAL_SCHEDULER": "0"},
        ["gunicorn", "hitch.wsgi:application"],
        False,
    ),
)


def _assert_scheduler_enablement_cases(
    test_case: SimpleTestCase,
    scheduler_enabled: Callable[[], bool],
    cases: tuple[_SchedulerEnablementCase, ...],
) -> None:
    for name, testing, environ, argv, expected in cases:
        with (
            test_case.subTest(name=name),
            override_settings(TESTING=testing),
            patch.dict(os.environ, environ, clear=True),
            patch.object(sys, "argv", argv),
        ):
            test_case.assertEqual(scheduler_enabled(), expected)


class AutoProposalSchedulerTests(SimpleTestCase):
    def test_scheduler_enablement_cases(self) -> None:
        _assert_scheduler_enablement_cases(
            self,
            auto_proposals._auto_proposal_scheduler_enabled,
            _COMMON_SCHEDULER_ENABLEMENT_CASES + _AUTO_PROPOSAL_SCHEDULER_OVERRIDE_CASES,
        )

    @patch(
        "hitch.main.goals.auto_proposals.autonomous_goals.maybe_start_auto_proposal_workflows",
        return_value=2,
    )
    @patch("hitch.main.goals.auto_proposals._refresh_unarchived_session_state_best_effort")
    @patch("hitch.main.goals.auto_proposals.reconciliation.reconcile_dead")
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

    @patch("hitch.main.runtime.server_lifecycle.logger.exception")
    @patch("hitch.main.goals.auto_proposals.autonomous_goals.maybe_start_auto_proposal_workflows")
    @patch("hitch.main.goals.auto_proposals._refresh_unarchived_session_state_best_effort")
    @patch("hitch.main.goals.auto_proposals.reconciliation.reconcile_dead")
    def test_scheduler_tick_keeps_running_after_errors(
        self,
        mock_reconcile_dead: MagicMock,
        mock_refresh: MagicMock,
        mock_start: MagicMock,
        mock_log_exception: MagicMock,
    ) -> None:
        mock_reconcile_dead.side_effect = RuntimeError("boom")

        result = auto_proposals._run_auto_proposal_scheduler_tick()

        # The shared run_tick wrapper swallows and records the failure so one
        # bad tick cannot kill the scheduler thread; the cursor resets.
        self.assertEqual(result, "")
        mock_refresh.assert_not_called()
        mock_start.assert_not_called()
        mock_log_exception.assert_called_once_with(
            "scheduler %s tick failed", "hitch-auto-proposals"
        )
        status = auto_proposals._scheduler.status()
        self.assertTrue(status.last_tick_errored)
        self.assertIn("boom", status.last_error)
        self.assertIsNotNone(status.last_error_at)


    @patch("hitch.main.goals.auto_proposals.autonomous_goals.maybe_start_auto_proposal_workflows")
    @patch("hitch.main.goals.auto_proposals._refresh_unarchived_session_state_best_effort")
    @patch("hitch.main.goals.auto_proposals.reconciliation.reconcile_dead")
    def test_tick_keeps_refreshed_cursor_when_workflow_start_fails(
        self,
        mock_reconcile_dead: MagicMock,
        mock_refresh: MagicMock,
        mock_start: MagicMock,
    ) -> None:
        # A failure after the refresh must not reset the incremental scan to
        # the front, or a repeated start failure would rescan only the first
        # window of active sessions forever.
        mock_refresh.return_value = "cursor-page-2"
        mock_start.side_effect = RuntimeError("boom")

        result = auto_proposals._run_auto_proposal_scheduler_tick(
            None, start_cursor="cursor-page-1"
        )

        self.assertEqual(result, "cursor-page-2")

class WorkflowMaintenanceSchedulerTests(SimpleTestCase):
    def test_scheduler_enablement_cases(self) -> None:
        _assert_scheduler_enablement_cases(
            self,
            workflow_maintenance._workflow_maintenance_scheduler_enabled,
            _COMMON_SCHEDULER_ENABLEMENT_CASES,
        )

    @patch(
        "hitch.main.workflows.workflow_maintenance.disk_cleanup.run_finished_session_disk_cleanup"
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

    @patch("hitch.main.workflows.workflow_maintenance.logger.exception")
    @patch(
        "hitch.main.workflows.workflow_maintenance.disk_cleanup.run_finished_session_disk_cleanup",
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

    @patch(
        "hitch.main.workflows.workflow_maintenance.disk_cleanup.run_finished_session_disk_cleanup"
    )
    @patch(
        "hitch.main.workflows.workflow_maintenance.pr_tracking.refresh_unarchived_session_pr_stages",
        return_value=1,
    )
    @patch("hitch.main.workflows.workflow_maintenance.reconciliation.reconcile_dead")
    def test_scheduler_tick_reconciles_and_refreshes_pr_stages(
        self,
        mock_reconcile_dead: MagicMock,
        mock_refresh_pr_stages: MagicMock,
        mock_disk_cleanup: MagicMock,
    ) -> None:
        workflow_maintenance._run_workflow_maintenance_scheduler_tick()

        mock_reconcile_dead.assert_called_once_with()
        # Disk cleanup runs only on the separate 10-minute cadence, never as
        # part of the 60-second maintenance tick.
        mock_disk_cleanup.assert_not_called()
        # The maintenance scheduler runs under production server commands, so it
        # owns background PR-stage convergence to keep gh out of the request
        # path -- bounded per tick so it can't starve the reconcile sweep.
        mock_refresh_pr_stages.assert_called_once_with(
            limit=workflow_maintenance._PR_STAGE_REFRESH_LIMIT_PER_TICK
        )


class UnarchivedSessionStateRefreshTests(TestCase):
    @patch(
        "hitch.main.goals.auto_proposals.pr_tracking.refresh_unarchived_session_pr_stages",
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
        "hitch.main.goals.auto_proposals.pr_tracking.refresh_unarchived_session_pr_stages",
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
    @patch("hitch.main.goals.auto_proposals.refresh_unarchived_session_state")
    @patch("hitch.main.goals.auto_proposals.codex_pool.app_server_config")
    @patch("hitch.main.goals.auto_proposals.app_server_pool.start_codex")
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
    @patch("hitch.main.workflows.workflow_maintenance.start_workflow_maintenance_scheduler")
    @patch("hitch.main.goals.auto_proposals.start_auto_proposal_scheduler")
    def test_ready_starts_schedulers(
        self, mock_auto_start: MagicMock, mock_workflow_start: MagicMock
    ) -> None:
        config = MainConfig("hitch.main", main_app)

        config.ready()

        mock_workflow_start.assert_called_once_with()
        mock_auto_start.assert_called_once_with()
