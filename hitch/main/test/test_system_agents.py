from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import override
from unittest.mock import MagicMock, patch

from django.db import transaction
from django.test import TestCase
from django.utils import timezone

from hitch.main.models import (
    AutonomousGoal,
    CodexInstance,
    Project,
    ProposedSession,
    RefreshThrottle,
    SessionMetadata,
    SystemAgentRun,
    SystemWorkflow,
)
from hitch.main.runtime import codex_tools
from hitch.main.workflows import autonomous_goals, system_agents


class DynamicAutonomousGoalToolTests(TestCase):
    def test_candidate_receives_agent_owned_protocol_tools(self) -> None:
        specs = codex_tools.registered_dynamic_tool_specs(
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )

        self.assertEqual(
            {spec["name"] for spec in specs},
            {
                "get_goal",
                "list_goal_sessions",
                "review",
                "propose_session",
                "no_proposal",
            },
        )
        propose = next(spec for spec in specs if spec["name"] == "propose_session")
        self.assertEqual(propose["inputSchema"]["properties"], {})

    def test_reviewer_receives_only_verdict_tools(self) -> None:
        specs = codex_tools.registered_dynamic_tool_specs(
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.AUTONOMOUS_GOAL_REVIEWER_AGENT_KIND,
        )

        self.assertEqual({spec["name"] for spec in specs}, {"approve", "deny"})

    def test_visible_propose_session_keeps_its_full_schema(self) -> None:
        specs = codex_tools.registered_dynamic_tool_specs(
            purpose=CodexInstance.PURPOSE_USER,
        )
        propose = next(spec for spec in specs if spec["name"] == "propose_session")

        self.assertIn("title", propose["inputSchema"]["properties"])

    def test_unknown_role_receives_no_ag_tools(self) -> None:
        specs = codex_tools.registered_dynamic_tool_specs(
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind="unrelated",
        )

        self.assertEqual(specs, [])

    @patch("hitch.main.runtime.codex_tools._autonomous_goals")
    def test_candidate_tools_dispatch_to_agent_owned_operations(self, backend_factory: MagicMock) -> None:
        backend = backend_factory.return_value
        context = codex_tools.ToolContext(
            cwd="/repo",
            thread_id="candidate",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        cases: tuple[tuple[str, dict[str, str], str, tuple[object, ...]], ...] = (
            ("get_goal", {}, "candidate_goal_data", (context,)),
            ("list_goal_sessions", {}, "candidate_goal_sessions", (context,)),
            (
                "review",
                {"title": "Candidate"},
                "candidate_request_review",
                ({"title": "Candidate"}, context),
            ),
            ("propose_session", {}, "candidate_submit_proposal", ({}, context)),
            (
                "no_proposal",
                {"reason": "Done"},
                "candidate_decline_proposal",
                ({"reason": "Done"}, context),
            ),
        )

        for tool_name, arguments, operation_name, expected_arguments in cases:
            with self.subTest(tool=tool_name):
                operation = getattr(backend, operation_name)
                operation.return_value = {"operation": operation_name}

                response = codex_tools.handle_dynamic_tool_call(
                    {"namespace": "hitch", "tool": tool_name, "arguments": arguments},
                    context,
                )

                self.assertTrue(response["success"])
                self.assertEqual(
                    json.loads(response["contentItems"][0]["text"]),
                    {"operation": operation_name},
                )
                operation.assert_called_once_with(*expected_arguments)


class SystemAgentRoutingTests(TestCase):
    @override
    def setUp(self) -> None:
        self.project = Project.objects.create(name="repo", repo_path="/repo")
        self.goal = AutonomousGoal.objects.create(
            project=self.project,
            title="Goal",
            goal="Do useful work.",
        )
        self.workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(self.goal.pk),
            cwd=self.project.repo_path,
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_RUNNING,
            state={"autonomous_goal_id": self.goal.pk},
        )
        self.instance = CodexInstance.objects.create(
            pid=1,
            thread_id="candidate",
            cwd="/repo",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=self.workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        self.agent_run = SystemAgentRun.objects.create(
            workflow=self.workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id=self.instance.thread_id,
            instance=self.instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )

    @patch("hitch.main.workflows.autonomous_goals.on_agent_finished")
    def test_terminal_agent_routes_directly_to_ag_runtime(self, finish: MagicMock) -> None:
        routed = system_agents.on_codex_instance_finished(self.instance)

        self.assertTrue(routed)
        finish.assert_called_once()
        self.instance.refresh_from_db()
        self.assertIsNotNone(self.instance.workflow_routing_started_at)

    @patch("hitch.main.workflows.autonomous_goals.on_agent_finished")
    def test_finish_routing_claim_is_idempotent(self, finish: MagicMock) -> None:
        self.assertTrue(system_agents.on_codex_instance_finished(self.instance))
        self.assertTrue(system_agents.on_codex_instance_finished(self.instance))

        finish.assert_called_once()

    def test_system_agent_without_workflow_is_not_routed(self) -> None:
        self.instance.workflow_id = None
        self.instance.save(update_fields=["workflow_id"])

        self.assertFalse(system_agents.on_codex_instance_finished(self.instance))

    @patch("hitch.main.workflows.autonomous_goals.cleanup_terminal_run")
    def test_terminal_run_is_cleaned_up_without_reprocessing(self, cleanup: MagicMock) -> None:
        self.agent_run.status = SystemAgentRun.STATUS_COMPLETED
        self.agent_run.save(update_fields=["status"])

        self.assertTrue(system_agents.on_codex_instance_finished(self.instance))

        cleanup.assert_called_once_with(self.agent_run)

    def test_unclaimed_terminal_run_still_counts_as_inflight(self) -> None:
        self.assertTrue(system_agents.workflow_has_inflight_instance(self.workflow.pk))

        self.agent_run.status = SystemAgentRun.STATUS_COMPLETED
        self.agent_run.save(update_fields=["status"])

        self.assertFalse(system_agents.workflow_has_inflight_instance(self.workflow.pk))

    @patch("hitch.main.workflows.autonomous_goals.on_agent_finished", side_effect=RuntimeError("boom"))
    def test_failed_routing_releases_claim_for_recovery(self, _finish: MagicMock) -> None:
        with self.assertRaisesRegex(RuntimeError, "boom"):
            system_agents.on_codex_instance_finished(self.instance)

        self.instance.refresh_from_db()
        self.assertIsNone(self.instance.workflow_routing_started_at)

    @patch("hitch.main.workflows.autonomous_goals.on_agent_finished")
    def test_missing_turn_ledger_is_recovered_from_instance(self, finish: MagicMock) -> None:
        self.agent_run.delete()

        self.assertTrue(system_agents.on_codex_instance_finished(self.instance))

        recovered = SystemAgentRun.objects.get(instance=self.instance)
        self.assertEqual(recovered.thread_id, self.instance.thread_id)
        finish.assert_called_once()

    @patch("hitch.main.workflows.autonomous_goals.recover_orphaned_workflows", return_value=1)
    @patch("hitch.main.workflows.system_agents.on_codex_instance_finished", return_value=True)
    def test_reconciler_routes_terminal_rows_then_checks_orphans(self, route: MagicMock, recover: MagicMock) -> None:
        reconciled = system_agents.reconcile_terminal_workflow_instances(workflow_id=self.workflow.pk)

        self.assertEqual(reconciled, 2)
        route.assert_called_once_with(self.instance)
        recover.assert_called_once()

    @patch("hitch.main.workflows.autonomous_goals.recover_orphaned_workflows", return_value=1)
    @patch("hitch.main.workflows.system_agents.on_codex_instance_finished", return_value=True)
    def test_reconciler_includes_legacy_workflow_steps(self, _route: MagicMock, recover: MagicMock) -> None:
        self.workflow.step = "autonomous_goal_candidate_running"
        self.workflow.save(update_fields=["step"])

        self.assertEqual(
            system_agents.reconcile_terminal_workflow_instances(workflow_id=self.workflow.pk),
            2,
        )

        workflows = recover.call_args.args[0]
        self.assertEqual([workflow.pk for workflow in workflows], [self.workflow.pk])

    @patch("hitch.main.workflows.pr_tracking.supersede_pr_after_turn")
    def test_visible_turn_still_routes_pr_lifecycle(self, supersede: MagicMock) -> None:
        visible = CodexInstance.objects.create(
            pid=2,
            thread_id="visible",
            cwd="/repo",
            events_path="/tmp/visible.jsonl",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_USER,
        )

        self.assertFalse(system_agents.on_codex_instance_finished(visible))
        supersede.assert_called_once_with(visible)

    def test_hidden_threads_include_durable_run_metadata(self) -> None:
        SessionMetadata.objects.create(
            thread_id="metadata-only",
            cwd="/repo",
            project=self.project,
            is_hidden_system_session=True,
        )

        self.assertEqual(
            system_agents.hidden_thread_ids(),
            {"candidate", "metadata-only"},
        )

    def test_final_agent_text_ignores_commentary(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "events.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "method": "item/completed",
                                "payload": {
                                    "item": {
                                        "type": "agentMessage",
                                        "phase": "commentary",
                                        "text": "working",
                                    }
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "method": "item/completed",
                                "payload": {
                                    "item": {
                                        "type": "agentMessage",
                                        "phase": "final_answer",
                                        "text": "finished",
                                    }
                                },
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(system_agents.final_agent_text(str(path)), "finished")


class AutonomousGoalAdmissionTests(TestCase):
    @override
    def setUp(self) -> None:
        self.project = Project.objects.create(name="repo", repo_path="/repo")
        self.goal = AutonomousGoal.objects.create(
            project=self.project,
            title="Goal",
            goal="Do useful work.",
            auto_proposal_enabled=True,
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            stacked_diff_depth=3,
            proposal_budget=1_000,
        )

    @patch("hitch.main.workflows.autonomous_goals._spawn_autonomous_goal_candidate_or_finish")
    def test_manual_start_creates_one_thin_run_record(self, spawn: MagicMock) -> None:
        workflow = autonomous_goals.start_autonomous_goal_workflow_if_queue_idle(
            autonomous_goal=self.goal,
            use_worktrees=True,
        )

        assert workflow is not None
        self.assertEqual(workflow.step, system_agents.STEP_AUTONOMOUS_GOAL_RUNNING)
        self.assertEqual(
            set(workflow.state),
            {
                "autonomous_goal_id",
                "auto_proposal",
                "autonomous_goal_updated_at",
                "web_search_mode",
                "use_worktrees",
                "stacked_diff_depth",
                "proposal_budget",
            },
        )
        spawn.assert_called_once()

    @patch("hitch.main.workflows.autonomous_goals._spawn_autonomous_goal_candidate_or_finish")
    def test_global_queue_admits_only_one_goal(self, _spawn: MagicMock) -> None:
        other = AutonomousGoal.objects.create(
            project=self.project,
            title="Other",
            goal="Other work.",
        )
        first = autonomous_goals.start_autonomous_goal_workflow_if_queue_idle(
            autonomous_goal=self.goal,
        )
        second = autonomous_goals.start_autonomous_goal_workflow_if_queue_idle(
            autonomous_goal=other,
        )

        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertEqual(
            SystemWorkflow.objects.filter(status=SystemWorkflow.STATUS_RUNNING).count(),
            1,
        )

    @patch("hitch.main.workflows.autonomous_goals._spawn_autonomous_goal_candidate_or_finish")
    @patch("hitch.main.workflows.autonomous_goals.default_branch_commit_hash", return_value="abc")
    def test_auto_start_skips_unchanged_no_proposal_branch(self, _branch_sha: MagicMock, spawn: MagicMock) -> None:
        self.goal.auto_proposal_last_no_proposal_sha = "abc"
        self.goal.save(update_fields=["auto_proposal_last_no_proposal_sha"])

        self.assertFalse(autonomous_goals._maybe_start_auto_proposal_workflow(self.goal.pk))
        spawn.assert_not_called()

    @patch("hitch.main.workflows.autonomous_goals._spawn_autonomous_goal_candidate_or_finish")
    @patch("hitch.main.workflows.autonomous_goals.default_branch_commit_hash", return_value="abc")
    def test_auto_scheduler_starts_at_most_one_goal(self, _branch_sha: MagicMock, spawn: MagicMock) -> None:
        AutonomousGoal.objects.create(
            project=self.project,
            title="Other",
            goal="Other work.",
            auto_proposal_enabled=True,
        )
        with patch(
            "hitch.main.workflows.autonomous_goals._auto_proposals_paused_by_usage_quota_throttled",
            return_value=False,
        ):
            started = autonomous_goals.maybe_start_auto_proposal_workflows()

        self.assertEqual(started, 1)
        self.assertEqual(spawn.call_count, 1)

    def test_pending_proposal_blocks_auto_start(self) -> None:
        ProposedSession.objects.create(
            project=self.project,
            autonomous_goal=self.goal,
            title="Pending",
        )

        self.assertFalse(autonomous_goals._autonomous_goal_db_allows_start(self.goal, "abc"))

    def test_queue_lock_has_one_durable_database_row(self) -> None:
        with transaction.atomic():
            autonomous_goals._lock_autonomous_goal_queue()
        with transaction.atomic():
            autonomous_goals._lock_autonomous_goal_queue()

        self.assertEqual(
            RefreshThrottle.objects.filter(key="autonomous_goal:auto_proposal_queue").count(),
            1,
        )


class AutonomousGoalQuotaTests(TestCase):
    def test_low_quota_uses_linear_expected_remaining_threshold(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        window = SimpleNamespace(
            used_percent=80,
            resets_at=(now + timedelta(days=3, hours=12)).timestamp(),
            window_duration_mins=7 * 24 * 60,
        )

        self.assertEqual(
            autonomous_goals._rate_limit_window_auto_proposal_quota_status(window, now=now),
            "low",
        )

    def test_unverifiable_quota_fails_closed(self) -> None:
        rate_limits = SimpleNamespace(
            primary=SimpleNamespace(
                used_percent=None,
                resets_at=None,
                window_duration_mins=None,
            ),
            secondary=None,
        )

        self.assertEqual(
            autonomous_goals._auto_proposal_quota_status_from_rate_limits(rate_limits, now=timezone.now()),
            "unavailable",
        )
