from __future__ import annotations

import json
import tempfile
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase, TestCase

from hitch.main.models import CodexInstance, SystemWorkflow
from hitch.main.runtime.codex_tools import (
    ToolContext,
    handle_dynamic_tool_call,
    registered_dynamic_tool_specs,
)
from hitch.main.workflows import pr_qa, pr_watch, system_agents
from hitch.main.workflows.gh_cli import _GhPrOpenError
from hitch.main.workflows.gh_observations import (
    _evaluate_pr_gates,
    _gh_watch_blockers,
    _gh_watch_summary,
)


def _observation(
    update: dict[str, object] | None = None, *, feedback: str = ""
) -> dict[str, object]:
    pr: dict[str, object] = {
        "url": "https://github.com/openai/hitch/pull/42",
        "repository_full_name": "openai/hitch",
        "pr_number": 42,
        "state": "open",
        "merged": False,
        "mergeable": True,
        "draft": False,
        "review_signal": "approved",
        "unresolved_thread_count": 0,
        "ci_status": "success",
    }
    pr.update(update or {})
    gates = _evaluate_pr_gates(pr)
    return {
        "summary": _gh_watch_summary(gates, pr),
        "feedback": feedback,
        "pr": pr,
        "gates": gates,
        "blockers": _gh_watch_blockers(gates),
    }


class PrWatchTests(SimpleTestCase):
    @patch("hitch.main.workflows.pr_watch.observe_pr")
    def test_returns_ready_without_polling(self, mock_observe: MagicMock) -> None:
        mock_observe.return_value = _observation()

        with tempfile.TemporaryDirectory() as cwd:
            result = pr_watch.watch_pr(
                cwd=cwd,
                url="https://github.com/openai/hitch/pull/42",
                poll_seconds=0,
            )

        self.assertEqual(result["status"], "ready")
        mock_observe.assert_called_once()

    @patch("hitch.main.workflows.pr_watch.observe_pr")
    def test_returns_actionable_failure(self, mock_observe: MagicMock) -> None:
        mock_observe.return_value = _observation(
            {
                "ci_status": "failure",
                "failing_jobs": [{"name": "tests", "conclusion": "failure"}],
            },
            feedback="tests failed",
        )

        with tempfile.TemporaryDirectory() as cwd:
            result = pr_watch.watch_pr(
                cwd=cwd,
                url="https://github.com/openai/hitch/pull/42",
                poll_seconds=0,
            )

        self.assertEqual(result["status"], "action_required")
        self.assertIn("CI", result["summary"])

    @patch("hitch.main.workflows.pr_watch.observe_pr")
    def test_new_feedback_interrupts_pending_watch(
        self, mock_observe: MagicMock
    ) -> None:
        pending = _observation(
            {
                "review_signal": "",
                "ci_status": "pending",
                "pending_jobs": [{"name": "tests", "status": "queued"}],
            },
            feedback="A reviewer left a note.",
        )
        mock_observe.return_value = pending

        with tempfile.TemporaryDirectory() as cwd:
            result = pr_watch.watch_pr(
                cwd=cwd,
                url="https://github.com/openai/hitch/pull/42",
                poll_seconds=0,
            )

        self.assertEqual(result["status"], "attention")
        self.assertTrue(result["feedback_fingerprint"])

    @patch("hitch.main.workflows.pr_watch.observe_pr")
    def test_new_feedback_requires_attention_before_ready(
        self, mock_observe: MagicMock
    ) -> None:
        mock_observe.return_value = _observation(
            feedback="A reviewer left a non-blocking note."
        )

        with tempfile.TemporaryDirectory() as cwd:
            result = pr_watch.watch_pr(
                cwd=cwd,
                url="https://github.com/openai/hitch/pull/42",
                poll_seconds=0,
            )

        self.assertEqual(result["status"], "attention")

    @patch("hitch.main.workflows.pr_watch.observe_pr")
    def test_seen_feedback_does_not_create_a_hot_loop(
        self, mock_observe: MagicMock
    ) -> None:
        pending = _observation(
            {"review_signal": "", "ci_status": "pending"},
            feedback="Already assessed.",
        )
        mock_observe.side_effect = [pending, _observation()]
        fingerprint = pr_watch.feedback_fingerprint(pending)

        with tempfile.TemporaryDirectory() as cwd:
            result = pr_watch.watch_pr(
                cwd=cwd,
                url="https://github.com/openai/hitch/pull/42",
                previous_feedback_fingerprint=fingerprint,
                poll_seconds=0,
            )

        self.assertEqual(result["status"], "ready")
        self.assertEqual(mock_observe.call_count, 2)

    @patch("hitch.main.workflows.pr_watch.observe_pr")
    def test_returns_terminal_pr(self, mock_observe: MagicMock) -> None:
        mock_observe.return_value = _observation(
            {"state": "merged", "merged": True}
        )

        with tempfile.TemporaryDirectory() as cwd:
            result = pr_watch.watch_pr(
                cwd=cwd,
                url="https://github.com/openai/hitch/pull/42",
            )

        self.assertEqual(result["status"], "terminal")

    @patch("hitch.main.workflows.pr_watch.observe_pr")
    def test_pending_watch_times_out(self, mock_observe: MagicMock) -> None:
        mock_observe.return_value = _observation(
            {"review_signal": "", "ci_status": "pending"}
        )

        with tempfile.TemporaryDirectory() as cwd:
            result = pr_watch.watch_pr(
                cwd=cwd,
                url="https://github.com/openai/hitch/pull/42",
                timeout_seconds=0,
            )

        self.assertEqual(result["status"], "timed_out")

    @patch("hitch.main.workflows.pr_watch.observe_pr")
    def test_deadline_prevents_another_observation(
        self, mock_observe: MagicMock
    ) -> None:
        mock_observe.return_value = _observation(
            {"review_signal": "", "ci_status": "pending"}
        )
        ticks = iter([0.0, 0.0, 1.0])

        with tempfile.TemporaryDirectory() as cwd:
            result = pr_watch.watch_pr(
                cwd=cwd,
                url="https://github.com/openai/hitch/pull/42",
                timeout_seconds=1,
                monotonic=lambda: next(ticks),
            )

        self.assertEqual(result["status"], "timed_out")
        mock_observe.assert_called_once()

    @patch("hitch.main.workflows.pr_watch.observe_pr")
    def test_cancellation_interrupts_polling_wait(
        self, mock_observe: MagicMock
    ) -> None:
        mock_observe.return_value = _observation(
            {"review_signal": "", "ci_status": "pending"}
        )
        cancelled = False

        def cancel_requested() -> bool:
            return cancelled

        def cancel_during_sleep(_seconds: float) -> None:
            nonlocal cancelled
            cancelled = True

        with (
            tempfile.TemporaryDirectory() as cwd,
            self.assertRaisesRegex(pr_watch.PrWatchError, "cancelled"),
        ):
            pr_watch.watch_pr(
                cwd=cwd,
                url="https://github.com/openai/hitch/pull/42",
                poll_seconds=10,
                sleep=cancel_during_sleep,
                cancel_requested=cancel_requested,
            )

    def test_rejects_non_pr_url(self) -> None:
        with (
            tempfile.TemporaryDirectory() as cwd,
            self.assertRaisesRegex(pr_watch.PrWatchError, "GitHub pull request"),
        ):
            pr_watch.watch_pr(cwd=cwd, url="https://example.com/pr/42")

    @patch("hitch.main.workflows.pr_watch.observe_pr")
    def test_reports_gh_failure_as_tool_error(self, mock_observe: MagicMock) -> None:
        mock_observe.side_effect = _GhPrOpenError("gh auth failed")

        with (
            tempfile.TemporaryDirectory() as cwd,
            self.assertRaisesRegex(pr_watch.PrWatchError, "gh auth failed"),
        ):
            pr_watch.watch_pr(
                cwd=cwd,
                url="https://github.com/openai/hitch/pull/42",
            )


class PrWatchToolTests(TestCase):
    def test_registered_specs_include_watch_pr(self) -> None:
        specs = registered_dynamic_tool_specs()

        watch = next(spec for spec in specs if spec["name"] == "watch_pr")
        self.assertEqual(watch["namespace"], "hitch")
        self.assertEqual(watch["inputSchema"]["required"], ["url"])

    @patch("hitch.main.runtime.codex_tools.pr_watch.watch_pr")
    def test_tool_records_result_on_owning_workflow(
        self, mock_watch: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_PR_WATCH_RUNNING,
            state={
                system_agents._WORKFLOW_TURN_OWNER_STEP_STATE_KEY: (
                    system_agents.STEP_PR_WATCH_RUNNING
                ),
                system_agents._WORKFLOW_TURN_OWNER_INDEX_STATE_KEY: 4,
                pr_watch.PR_WATCH_RESULT_STATE_KEY: {
                    "feedback_fingerprint": "seen-feedback"
                },
            },
        )
        result = {
            "status": "ready",
            "summary": "The PR gates are passing.",
            "feedback": "",
            "feedback_fingerprint": "",
            "pr": _observation()["pr"],
            "gates": _observation()["gates"],
            "blockers": [],
        }
        mock_watch.return_value = result

        response = handle_dynamic_tool_call(
            {
                "tool": "watch_pr",
                "arguments": {
                    "url": "https://github.com/openai/hitch/pull/42",
                },
            },
            ToolContext(
                cwd="/repo",
                thread_id="main-thread",
                workflow_id=workflow.pk,
                user_message_index=4,
            ),
        )

        self.assertTrue(response["success"])
        self.assertEqual(
            mock_watch.call_args.kwargs["previous_feedback_fingerprint"],
            "seen-feedback",
        )
        self.assertEqual(
            json.loads(response["contentItems"][0]["text"])["status"], "ready"
        )
        workflow.refresh_from_db()
        self.assertEqual(
            workflow.state[pr_watch.PR_WATCH_RESULT_STATE_KEY]["status"], "ready"
        )
        self.assertEqual(workflow.state["pr_handoff"]["pr_number"], 42)

    @patch("hitch.main.runtime.codex_tools.pr_watch.watch_pr")
    def test_tool_does_not_record_another_threads_workflow(
        self, mock_watch: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_PR_WATCH_RUNNING,
            state={},
        )
        mock_watch.return_value = {
            "status": "ready",
            "summary": "ready",
            "pr": _observation()["pr"],
            "gates": _observation()["gates"],
        }

        response = handle_dynamic_tool_call(
            {
                "tool": "watch_pr",
                "arguments": {
                    "url": "https://github.com/openai/hitch/pull/42",
                },
            },
            ToolContext(
                cwd="/repo",
                thread_id="other-thread",
                workflow_id=workflow.pk,
                user_message_index=4,
            ),
        )

        self.assertTrue(response["success"])
        workflow.refresh_from_db()
        self.assertNotIn(pr_watch.PR_WATCH_RESULT_STATE_KEY, workflow.state)

    @patch("hitch.main.runtime.codex_tools.pr_watch.watch_pr")
    def test_tool_rejects_another_pr_for_owning_workflow(
        self, mock_watch: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_PR_WATCH_RUNNING,
            state={
                system_agents._PR_HANDOFF_STATE_KEY: _observation()["pr"],
                system_agents._WORKFLOW_TURN_OWNER_STEP_STATE_KEY: (
                    system_agents.STEP_PR_WATCH_RUNNING
                ),
                system_agents._WORKFLOW_TURN_OWNER_INDEX_STATE_KEY: 4,
            },
        )

        response = handle_dynamic_tool_call(
            {
                "tool": "watch_pr",
                "arguments": {
                    "url": "https://github.com/openai/another/pull/42",
                },
            },
            ToolContext(
                cwd="/repo",
                thread_id="main-thread",
                workflow_id=workflow.pk,
                user_message_index=4,
            ),
        )

        self.assertFalse(response["success"])
        mock_watch.assert_not_called()

    @patch("hitch.main.runtime.codex_tools.pr_watch.watch_pr")
    def test_failed_later_call_clears_same_turn_success(
        self, mock_watch: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_PR_WATCH_RUNNING,
            state={
                system_agents._PR_HANDOFF_STATE_KEY: _observation()["pr"],
                system_agents._WORKFLOW_TURN_OWNER_STEP_STATE_KEY: (
                    system_agents.STEP_PR_WATCH_RUNNING
                ),
                system_agents._WORKFLOW_TURN_OWNER_INDEX_STATE_KEY: 4,
                pr_watch.PR_WATCH_RESULT_STATE_KEY: {
                    "status": "ready",
                    "feedback_fingerprint": "seen-feedback",
                },
                pr_watch.PR_WATCH_RESULT_TURN_INDEX_STATE_KEY: 4,
            },
        )
        mock_watch.side_effect = pr_watch.PrWatchError("GitHub unavailable")

        response = handle_dynamic_tool_call(
            {
                "tool": "watch_pr",
                "arguments": {
                    "url": "https://github.com/openai/hitch/pull/42",
                },
            },
            ToolContext(
                cwd="/repo",
                thread_id="main-thread",
                workflow_id=workflow.pk,
                user_message_index=4,
            ),
        )

        self.assertFalse(response["success"])
        self.assertEqual(
            mock_watch.call_args.kwargs["previous_feedback_fingerprint"],
            "seen-feedback",
        )
        workflow.refresh_from_db()
        self.assertNotIn(pr_watch.PR_WATCH_RESULT_STATE_KEY, workflow.state)
        self.assertNotIn(
            pr_watch.PR_WATCH_RESULT_TURN_INDEX_STATE_KEY,
            workflow.state,
        )


class AgentDrivenPrWorkflowTests(TestCase):
    @patch("hitch.main.workflows.system_agents._finish_workflow_block")
    def test_reconciliation_blocks_removed_monitor_steps(
        self, mock_finish_block: MagicMock
    ) -> None:
        for step in ("pr_monitoring", "pr_feedback_running"):
            with self.subTest(step=step):
                workflow = SystemWorkflow.objects.create(
                    kind=SystemWorkflow.KIND_PR_QA,
                    main_thread_id=f"legacy-{step}",
                    cwd="/repo",
                    status=SystemWorkflow.STATUS_RUNNING,
                    step=step,
                )

                reconciled = system_agents.reconcile_terminal_workflow_instances(
                    workflow_id=workflow.pk
                )

                self.assertEqual(reconciled, 1)
                workflow.refresh_from_db()
                self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
                self.assertEqual(workflow.step, system_agents.STEP_BLOCKED)
                self.assertIn("hitch.watch_pr", workflow.state["error"])
                mock_finish_block.assert_called_once()
                mock_finish_block.reset_mock()

    @patch("hitch.main.workflows.system_agents._finish_workflow_block")
    @patch("hitch.main.workflows.system_agents.codex_pool.interrupt_instance")
    def test_reconciliation_interrupts_removed_monitor_worker(
        self,
        mock_interrupt: MagicMock,
        mock_finish_block: MagicMock,
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="legacy-monitor",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step="pr_feedback_running",
        )
        worker = CodexInstance.objects.create(
            pid=123,
            thread_id="legacy-feedback",
            cwd="/repo",
            prompt="hidden feedback",
            events_path="/tmp/legacy-feedback-events.jsonl",
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_SYSTEM_FEEDBACK,
            workflow_id=workflow.pk,
        )
        mock_interrupt.return_value = worker

        reconciled = system_agents.reconcile_terminal_workflow_instances(
            workflow_id=workflow.pk
        )

        self.assertEqual(reconciled, 1)
        mock_interrupt.assert_called_once_with(
            worker.pk,
            expected_thread_id="legacy-feedback",
        )
        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
        self.assertIn("hitch.watch_pr", workflow.state["error"])
        self.assertTrue(workflow.state["deferred_failure_surface"])
        mock_finish_block.assert_not_called()

    def test_steering_turn_defers_workflow_owned_watch(self) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_USER_STEERING_RUNNING,
            state={
                "user_steering_resume_step": system_agents.STEP_PR_WATCH_RUNNING,
            },
        )

        instructions = pr_qa._user_steering_developer_instructions(workflow)

        self.assertIn("Do not invoke `hitch.watch_pr`", instructions)
        self.assertIn("workflow-owned watch turn", instructions)

    def test_existing_thread_without_tool_rejects_pr_workflow(self) -> None:
        with self.assertRaisesRegex(
            pr_qa.PrWatchUnavailableError,
            "Start a new session",
        ):
            pr_qa.start_pr_watch_workflow(
                main_thread_id="old-thread",
                cwd="/repo",
                pr_url="https://github.com/openai/hitch/pull/42",
                sandbox_policy="workspace-write",
                approval_mode="auto_review",
                pr_watch_tool_available=False,
            )

    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_turn")
    def test_fix_pr_starts_one_visible_watch_turn(self, mock_spawn: MagicMock) -> None:
        mock_spawn.return_value = MagicMock(spec=CodexInstance)

        workflow = pr_qa.start_pr_watch_workflow(
            main_thread_id="main-thread",
            cwd="/repo",
            pr_url="https://github.com/openai/hitch/pull/42",
            sandbox_policy="workspace-write",
            approval_mode="auto_review",
            initial_user_message_index=3,
        )

        workflow.refresh_from_db()
        self.assertEqual(workflow.step, system_agents.STEP_PR_WATCH_RUNNING)
        self.assertIn("hitch.watch_pr", mock_spawn.call_args.kwargs["prompt"])
        self.assertEqual(
            mock_spawn.call_args.kwargs["purpose"], CodexInstance.PURPOSE_USER
        )

    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_turn")
    def test_publication_hands_new_workflow_to_agent_watch(
        self, mock_spawn: MagicMock
    ) -> None:
        mock_spawn.return_value = MagicMock(spec=CodexInstance)
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_PR_PROMPT_RUNNING,
            state={
                "next_user_message_index": 1,
            },
        )

        pr_qa._commit_pr_prompt_result(
            workflow,
            snapshot=_observation()["pr"],  # type: ignore[arg-type]
        )

        workflow.refresh_from_db()
        self.assertEqual(workflow.step, system_agents.STEP_PR_WATCH_RUNNING)
        self.assertIn("hitch.watch_pr", mock_spawn.call_args.kwargs["prompt"])

    def test_ready_tool_result_completes_watch_as_pr_ready(self) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_PR_WATCH_RUNNING,
            state={
                pr_watch.PR_WATCH_RESULT_STATE_KEY: {"status": "ready"},
                pr_watch.PR_WATCH_RESULT_TURN_INDEX_STATE_KEY: 2,
                system_agents._WORKFLOW_TURN_OWNER_STEP_STATE_KEY: (
                    system_agents.STEP_PR_WATCH_RUNNING
                ),
                system_agents._WORKFLOW_TURN_OWNER_INDEX_STATE_KEY: 2,
            },
        )
        instance = CodexInstance.objects.create(
            pid=0,
            thread_id="main-thread",
            cwd="/repo",
            prompt="watch",
            events_path="/tmp/watch-events.jsonl",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_USER,
            workflow_id=workflow.pk,
            user_message_index=2,
        )

        pr_qa._handle_pr_watch_finished(instance, workflow)

        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(workflow.step, system_agents.STEP_PR_READY)

    def test_previous_turn_result_does_not_complete_resumed_watch_as_ready(
        self,
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_PR_WATCH_RUNNING,
            state={
                pr_watch.PR_WATCH_RESULT_STATE_KEY: {"status": "ready"},
                pr_watch.PR_WATCH_RESULT_TURN_INDEX_STATE_KEY: 1,
                system_agents._WORKFLOW_TURN_OWNER_STEP_STATE_KEY: (
                    system_agents.STEP_PR_WATCH_RUNNING
                ),
                system_agents._WORKFLOW_TURN_OWNER_INDEX_STATE_KEY: 2,
            },
        )
        instance = CodexInstance.objects.create(
            pid=0,
            thread_id="main-thread",
            cwd="/repo",
            prompt="watch again",
            events_path="/tmp/watch-events.jsonl",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_USER,
            workflow_id=workflow.pk,
            user_message_index=2,
        )

        pr_qa._handle_pr_watch_finished(instance, workflow)

        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(workflow.step, system_agents.STEP_PR_WATCH_COMPLETED)

    def test_previous_terminal_result_does_not_complete_resumed_watch(self) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_PR_WATCH_RUNNING,
            state={
                pr_watch.PR_WATCH_RESULT_STATE_KEY: {"status": "terminal"},
                pr_watch.PR_WATCH_RESULT_TURN_INDEX_STATE_KEY: 1,
                system_agents._PR_HANDOFF_STATE_KEY: {
                    "url": "https://github.com/openai/hitch/pull/42",
                    "state": "closed",
                },
                system_agents._WORKFLOW_TURN_OWNER_STEP_STATE_KEY: (
                    system_agents.STEP_PR_WATCH_RUNNING
                ),
                system_agents._WORKFLOW_TURN_OWNER_INDEX_STATE_KEY: 2,
            },
        )
        instance = CodexInstance.objects.create(
            pid=0,
            thread_id="main-thread",
            cwd="/repo",
            prompt="watch again",
            events_path="/tmp/watch-events.jsonl",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_USER,
            workflow_id=workflow.pk,
            user_message_index=2,
        )

        pr_qa._handle_pr_watch_finished(instance, workflow)

        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(workflow.step, system_agents.STEP_PR_WATCH_COMPLETED)

    @patch(
        "hitch.main.workflows.system_agents."
        "_maybe_auto_pull_default_repo_after_pr_merge"
    )
    def test_terminal_tool_result_completes_and_runs_merged_auto_pull(
        self, mock_auto_pull: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_PR_WATCH_RUNNING,
            state={
                pr_watch.PR_WATCH_RESULT_STATE_KEY: {"status": "terminal"},
                pr_watch.PR_WATCH_RESULT_TURN_INDEX_STATE_KEY: 2,
                system_agents._PR_HANDOFF_STATE_KEY: {
                    "url": "https://github.com/openai/hitch/pull/42",
                    "state": "merged",
                    "merged": True,
                },
                system_agents._WORKFLOW_TURN_OWNER_STEP_STATE_KEY: (
                    system_agents.STEP_PR_WATCH_RUNNING
                ),
                system_agents._WORKFLOW_TURN_OWNER_INDEX_STATE_KEY: 2,
            },
        )
        instance = CodexInstance.objects.create(
            pid=0,
            thread_id="main-thread",
            cwd="/repo",
            prompt="watch",
            events_path="/tmp/watch-events.jsonl",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_USER,
            workflow_id=workflow.pk,
            user_message_index=2,
        )

        pr_qa._handle_pr_watch_finished(instance, workflow)

        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(workflow.step, system_agents.STEP_PR_CLOSED)
        mock_auto_pull.assert_called_once_with(workflow)
