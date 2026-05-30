import os
import sys
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase, override_settings

import hitch.main as main_app
from hitch.main import auto_proposals
from hitch.main.apps import MainConfig


class AutoProposalSchedulerTests(SimpleTestCase):
    @override_settings(TESTING=True)
    @patch.dict(os.environ, {}, clear=True)
    @patch.object(sys, "argv", ["manage.py", "runserver", "--noreload"])
    def test_scheduler_disabled_during_tests_by_default(self) -> None:
        self.assertFalse(auto_proposals._scheduler_thread_enabled())

    @override_settings(TESTING=False)
    @patch.dict(os.environ, {"RUN_MAIN": "true"}, clear=True)
    @patch.object(sys, "argv", ["manage.py", "runserver"])
    def test_scheduler_enabled_in_runserver_child(self) -> None:
        self.assertTrue(auto_proposals._scheduler_thread_enabled())

    @override_settings(TESTING=False)
    @patch.dict(os.environ, {}, clear=True)
    @patch.object(sys, "argv", ["manage.py", "migrate"])
    def test_scheduler_disabled_for_management_commands(self) -> None:
        self.assertFalse(auto_proposals._scheduler_thread_enabled())

    @override_settings(TESTING=True)
    @patch.dict(os.environ, {"HITCH_AUTO_PROPOSAL_SCHEDULER": "1"}, clear=True)
    @patch.object(sys, "argv", ["manage.py", "migrate"])
    def test_scheduler_env_override_can_enable(self) -> None:
        self.assertTrue(auto_proposals._scheduler_thread_enabled())

    @patch("hitch.main.auto_proposals.system_agents.maybe_advance_pr_monitors")
    @patch(
        "hitch.main.auto_proposals.system_agents.maybe_start_auto_proposal_workflows",
        return_value=2,
    )
    @patch("hitch.main.auto_proposals.codex_pool.reconcile_dead")
    def test_scheduler_tick_reconciles_and_starts_auto_proposals(
        self,
        mock_reconcile_dead: MagicMock,
        mock_start: MagicMock,
        mock_advance_monitors: MagicMock,
    ) -> None:
        auto_proposals._run_auto_proposal_scheduler_tick()

        mock_reconcile_dead.assert_called_once_with()
        mock_advance_monitors.assert_called_once_with()
        mock_start.assert_called_once_with()

    @patch("hitch.main.auto_proposals.system_agents.maybe_advance_pr_monitors")
    @patch("hitch.main.auto_proposals.logger.exception")
    @patch("hitch.main.auto_proposals.system_agents.maybe_start_auto_proposal_workflows")
    @patch("hitch.main.auto_proposals.codex_pool.reconcile_dead")
    def test_scheduler_tick_keeps_running_after_errors(
        self,
        mock_reconcile_dead: MagicMock,
        mock_start: MagicMock,
        mock_log_exception: MagicMock,
        mock_advance_monitors: MagicMock,
    ) -> None:
        mock_reconcile_dead.side_effect = RuntimeError("boom")

        auto_proposals._run_auto_proposal_scheduler_tick()

        mock_start.assert_not_called()
        mock_log_exception.assert_called_once_with(
            "failed to run auto-proposal scheduler tick"
        )


class MainConfigTests(SimpleTestCase):
    @patch("hitch.main.auto_proposals.start_auto_proposal_scheduler")
    def test_ready_starts_auto_proposal_scheduler(self, mock_start: MagicMock) -> None:
        config = MainConfig("hitch.main", main_app)

        config.ready()

        mock_start.assert_called_once_with()
