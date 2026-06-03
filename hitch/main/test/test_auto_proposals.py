import os
import sys
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase, TestCase, override_settings

import hitch.main as main_app
from hitch.main import auto_proposals, workflow_maintenance
from hitch.main.apps import MainConfig
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
    @patch.object(sys, "argv", ["manage.py", "migrate"])
    def test_scheduler_disabled_for_management_commands(self) -> None:
        self.assertFalse(auto_proposals._auto_proposal_scheduler_enabled())

    @override_settings(TESTING=True)
    @patch.dict(os.environ, {"HITCH_AUTO_PROPOSAL_SCHEDULER": "1"}, clear=True)
    @patch.object(sys, "argv", ["manage.py", "migrate"])
    def test_scheduler_env_override_can_enable(self) -> None:
        self.assertTrue(auto_proposals._auto_proposal_scheduler_enabled())

    @patch(
        "hitch.main.auto_proposals.system_agents.maybe_start_auto_proposal_workflows",
        return_value=2,
    )
    @patch("hitch.main.auto_proposals._refresh_unarchived_session_state_best_effort")
    @patch("hitch.main.auto_proposals.codex_pool.reconcile_dead")
    def test_scheduler_tick_reconciles_and_starts_auto_proposals(
        self,
        mock_reconcile_dead: MagicMock,
        mock_refresh: MagicMock,
        mock_start: MagicMock,
    ) -> None:
        auto_proposals._run_auto_proposal_scheduler_tick()

        mock_reconcile_dead.assert_called_once_with()
        mock_refresh.assert_called_once_with()
        mock_start.assert_called_once_with()

    @patch("hitch.main.auto_proposals.logger.exception")
    @patch("hitch.main.auto_proposals.system_agents.maybe_start_auto_proposal_workflows")
    @patch("hitch.main.auto_proposals._refresh_unarchived_session_state_best_effort")
    @patch("hitch.main.auto_proposals.codex_pool.reconcile_dead")
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
        "hitch.main.workflow_maintenance.system_agents.refresh_due_pr_monitor_backoffs",
        return_value=2,
    )
    @patch("hitch.main.workflow_maintenance.codex_pool.reconcile_dead")
    def test_scheduler_tick_reconciles_and_refreshes_pr_monitor_backoffs(
        self, mock_reconcile_dead: MagicMock, mock_refresh: MagicMock
    ) -> None:
        workflow_maintenance._run_workflow_maintenance_scheduler_tick()

        mock_reconcile_dead.assert_called_once_with()
        mock_refresh.assert_called_once_with()


class UnarchivedSessionStateRefreshTests(TestCase):
    @patch(
        "hitch.main.auto_proposals.system_agents.refresh_unarchived_session_pr_stages",
        return_value=2,
    )
    @patch("hitch.main.auto_proposals.codex_pool.app_server_config")
    @patch("hitch.main.auto_proposals.Codex")
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
        mock_refresh_pr_stages.assert_called_once_with()
        metadata = SessionMetadata.objects.get(thread_id="thread-1")
        self.assertEqual(metadata.codex_display_title, "Renamed session")
        self.assertEqual(metadata.codex_updated_at, datetime.fromtimestamp(10, UTC))

    @patch("hitch.main.auto_proposals.logger.exception")
    @patch(
        "hitch.main.auto_proposals.system_agents.refresh_unarchived_session_pr_stages",
        return_value=1,
    )
    @patch("hitch.main.auto_proposals.codex_pool.app_server_config")
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
        mock_refresh_pr_stages.assert_called_once_with()
        mock_log_exception.assert_called_once_with(
            "failed to refresh active Codex session metadata"
        )


class MainConfigTests(SimpleTestCase):
    @patch("hitch.main.workflow_maintenance.start_workflow_maintenance_scheduler")
    @patch("hitch.main.auto_proposals.start_auto_proposal_scheduler")
    def test_ready_starts_schedulers(
        self, mock_auto_start: MagicMock, mock_workflow_start: MagicMock
    ) -> None:
        config = MainConfig("hitch.main", main_app)

        config.ready()

        mock_workflow_start.assert_called_once_with()
        mock_auto_start.assert_called_once_with()
