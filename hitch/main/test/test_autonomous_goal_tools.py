from __future__ import annotations

from datetime import timedelta
from tempfile import TemporaryDirectory
from typing import override
from unittest.mock import MagicMock, call, patch

from django.test import TestCase
from django.utils import timezone

from hitch.main.goals.autonomous_goal_prompts import _autonomous_goal_proposed_session_prompt
from hitch.main.models import (
    AutonomousGoal,
    CodexInstance,
    Project,
    ProposedSession,
    SessionMetadata,
    SystemAgentRun,
    SystemWorkflow,
)
from hitch.main.runtime.codex_tools import ToolContext
from hitch.main.workflows import autonomous_goals, system_agents

_CANDIDATE = {
    "title": "Reduce runtime orchestration",
    "summary": "Move AG decisions into role-scoped tools.",
    "impact": "Less runtime framework code.",
    "implemented_changes": "Added a direct review tool.",
    "implementation_direction": "Keep the durable control plane thin.",
    "verification": "Focused tests pass.",
    "rough_edges": "None known.",
    "suggested_continuation": "Review and merge the change.",
    "relevant_files": ["hitch/main/workflows/autonomous_goals.py"],
}


class _AutonomousGoalFixture(TestCase):
    @override
    def setUp(self) -> None:
        self.project = Project.objects.create(name="repo", repo_path="/repo")
        self.goal = AutonomousGoal.objects.create(
            project=self.project,
            title="Simplify AGs",
            goal="Use tools instead of framework state machines.",
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            stacked_diff_depth=2,
            proposal_budget=10_000,
            confidence_threshold=AutonomousGoal.CONFIDENCE_HIGH,
        )
        self.workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(self.goal.pk),
            cwd=self.project.repo_path,
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_RUNNING,
            state={
                "autonomous_goal_id": self.goal.pk,
                "use_worktrees": True,
                "stacked_diff_depth": 2,
                "proposal_budget": 10_000,
            },
        )
        self.candidate_instance = self._instance(
            "candidate-thread",
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            cwd="/candidate-worktree",
        )
        self.candidate_run = SystemAgentRun.objects.create(
            workflow=self.workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id=self.candidate_instance.thread_id,
            instance=self.candidate_instance,
            status=SystemAgentRun.STATUS_RUNNING,
            input={
                "autonomous_goal_id": self.goal.pk,
                "cwd": self.candidate_instance.cwd,
                "managed_candidate_cwd": True,
                "stack_iteration": 1,
            },
        )
        SessionMetadata.objects.create(
            thread_id=self.candidate_instance.thread_id,
            cwd=self.candidate_instance.cwd,
            project=self.project,
            is_hidden_system_session=True,
        )

    def _instance(
        self,
        thread_id: str,
        *,
        agent_kind: str,
        cwd: str = "/repo",
        status: str = CodexInstance.STATUS_RUNNING,
    ) -> CodexInstance:
        return CodexInstance.objects.create(
            pid=100,
            thread_id=thread_id,
            cwd=cwd,
            events_path=f"/tmp/{thread_id}.jsonl",
            status=status,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=self.workflow.pk,
            agent_kind=agent_kind,
        )

    def _candidate_context(self) -> ToolContext:
        return ToolContext(
            cwd=self.candidate_instance.cwd,
            thread_id=self.candidate_instance.thread_id,
            instance_id=self.candidate_instance.pk,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=self.workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )

    def _review(
        self,
        *,
        verdict: str = "approve",
        confidence: str = AutonomousGoal.CONFIDENCE_HIGH,
        status: str = SystemAgentRun.STATUS_COMPLETED,
    ) -> SystemAgentRun:
        instance = self._instance(
            f"review-{SystemAgentRun.objects.count()}",
            agent_kind=system_agents.AUTONOMOUS_GOAL_REVIEWER_AGENT_KIND,
            cwd="/review-worktree",
            status=CodexInstance.STATUS_COMPLETED,
        )
        return SystemAgentRun.objects.create(
            workflow=self.workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_REVIEWER_AGENT_KIND,
            thread_id=instance.thread_id,
            instance=instance,
            status=status,
            input={
                "autonomous_goal_id": self.goal.pk,
                "candidate_instance_id": self.candidate_instance.pk,
                "candidate": dict(_CANDIDATE),
                "snapshot_sha": "abc123",
                "snapshot_ref": "refs/hitch/autonomous-goals/1/review",
                "cwd": instance.cwd,
                "managed_review_cwd": True,
            },
            output={
                "verdict": verdict,
                "confidence": confidence,
                "feedback": "looks good" if verdict == "approve" else "revise it",
            },
        )


class AutonomousGoalToolTests(_AutonomousGoalFixture):
    def test_proposal_prompt_falls_back_to_reviewed_candidate_details(self) -> None:
        candidate = {**_CANDIDATE, "suggested_continuation": ""}

        prompt = _autonomous_goal_proposed_session_prompt(
            self.goal,
            candidate,
            {"feedback": "Keep the boundary narrow."},
        )

        self.assertIn("Reviewer feedback:\nKeep the boundary narrow.", prompt)
        self.assertIn(
            "Implementation guidance:\nKeep the durable control plane thin.",
            prompt,
        )

    def test_get_goal_reads_limits_and_review_feedback(self) -> None:
        self._review(verdict="deny")

        result = autonomous_goals.candidate_goal_data(self._candidate_context())

        self.assertEqual(result["title"], self.goal.title)
        self.assertEqual(result["stack_iteration"], 1)
        self.assertEqual(result["stack_depth"], 2)
        self.assertEqual(result["reviews_used"], 1)
        self.assertEqual(result["last_review_feedback"], "revise it")

    def test_candidate_scope_is_bound_to_exact_turn(self) -> None:
        context = self._candidate_context()
        context = ToolContext(**{**context.__dict__, "instance_id": self.candidate_instance.pk + 1})

        with self.assertRaisesRegex(ValueError, "candidate run no longer exists"):
            autonomous_goals.candidate_goal_data(context)

    @patch("hitch.main.workflows.autonomous_goals._release_autonomous_goal_snapshot_ref")
    @patch("hitch.main.workflows.autonomous_goals._cleanup_autonomous_goal_candidate_cwd")
    @patch("hitch.main.workflows.autonomous_goals._spawn_review_run")
    def test_review_returns_terminal_reviewer_result(
        self,
        spawn_review: MagicMock,
        cleanup_cwd: MagicMock,
        release_ref: MagicMock,
    ) -> None:
        spawn_review.side_effect = lambda *args, **kwargs: self._review()

        result = autonomous_goals.candidate_request_review(dict(_CANDIDATE), self._candidate_context())

        self.assertEqual(result["verdict"], "approve")
        self.assertEqual(result["reviews_remaining"], 1)
        cleanup_cwd.assert_called_once_with("/review-worktree")
        release_ref.assert_not_called()

    @patch("hitch.main.workflows.autonomous_goals._release_autonomous_goal_snapshot_ref")
    @patch("hitch.main.workflows.autonomous_goals._cleanup_autonomous_goal_candidate_cwd")
    @patch("hitch.main.workflows.autonomous_goals._spawn_review_run")
    def test_failed_review_is_returned_as_denial_and_releases_snapshot(
        self,
        spawn_review: MagicMock,
        cleanup_cwd: MagicMock,
        release_ref: MagicMock,
    ) -> None:
        def failed_review(*args: object, **kwargs: object) -> SystemAgentRun:
            review = self._review(status=SystemAgentRun.STATUS_FAILED)
            review.error = "reviewer crashed"
            review.save(update_fields=["error"])
            return review

        spawn_review.side_effect = failed_review

        result = autonomous_goals.candidate_request_review(dict(_CANDIDATE), self._candidate_context())

        self.assertEqual(result["verdict"], "deny")
        self.assertEqual(result["feedback"], "reviewer crashed")
        cleanup_cwd.assert_called_once_with("/review-worktree")
        release_ref.assert_called_once_with(
            self.project.repo_path,
            "refs/hitch/autonomous-goals/1/review",
        )

    def test_candidate_may_review_at_most_twice(self) -> None:
        self._review(verdict="deny")
        self._review(verdict="deny")

        with self.assertRaisesRegex(ValueError, "already used both reviews"):
            autonomous_goals.candidate_request_review(dict(_CANDIDATE), self._candidate_context())

    def test_submit_proposal_uses_exact_approved_review(self) -> None:
        self._review()

        result = autonomous_goals.candidate_submit_proposal({}, self._candidate_context())

        self.candidate_run.refresh_from_db()
        self.assertEqual(result["status"], "proposal_ready")
        self.assertEqual(self.candidate_run.output["candidate"], _CANDIDATE)
        self.assertEqual(self.candidate_run.output["snapshot_sha"], "abc123")
        self.assertEqual(self.candidate_run.output["terminal"], "propose")

    def test_submit_requires_approved_review(self) -> None:
        self._review(verdict="deny")

        with self.assertRaisesRegex(ValueError, "has not been approved"):
            autonomous_goals.candidate_submit_proposal({}, self._candidate_context())

    def test_no_proposal_marks_current_turn_terminal(self) -> None:
        result = autonomous_goals.candidate_decline_proposal(
            {"reason": "No useful work remains."}, self._candidate_context()
        )

        self.candidate_run.refresh_from_db()
        self.assertEqual(result["status"], "no_proposal")
        self.assertEqual(self.candidate_run.output["terminal"], "no_proposal")
        self.assertEqual(self.candidate_run.output["reason"], "No useful work remains.")

    def test_reviewer_verdict_is_bound_to_reviewer_turn(self) -> None:
        review = self._review(status=SystemAgentRun.STATUS_RUNNING)
        review.output = {}
        review.save(update_fields=["output"])
        context = ToolContext(
            cwd=review.instance.cwd,
            thread_id=review.thread_id,
            instance_id=review.instance_id,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=self.workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_REVIEWER_AGENT_KIND,
        )

        result = autonomous_goals.reviewer_record_verdict(
            {"confidence": "high", "feedback": "ship it"},
            context,
            approved=True,
        )

        review.refresh_from_db()
        self.assertEqual(result["verdict"], "approve")
        self.assertEqual(review.output["feedback"], "ship it")

    def test_reviewer_cannot_approve_below_threshold(self) -> None:
        review = self._review(status=SystemAgentRun.STATUS_RUNNING)
        review.output = {}
        review.save(update_fields=["output"])
        context = ToolContext(
            cwd=review.instance.cwd,
            thread_id=review.thread_id,
            instance_id=review.instance_id,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=self.workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_REVIEWER_AGENT_KIND,
        )

        with self.assertRaisesRegex(ValueError, "below this goal"):
            autonomous_goals.reviewer_record_verdict({"confidence": "medium"}, context, approved=True)

    def test_terminal_reviewer_cannot_record_a_verdict(self) -> None:
        review = self._review(status=SystemAgentRun.STATUS_FAILED)
        review.output = {}
        review.save(update_fields=["output"])
        context = ToolContext(
            cwd=review.instance.cwd,
            thread_id=review.thread_id,
            instance_id=review.instance_id,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=self.workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_REVIEWER_AGENT_KIND,
        )

        with self.assertRaisesRegex(ValueError, "no longer active"):
            autonomous_goals.reviewer_record_verdict({"confidence": "high"}, context, approved=True)

    @patch("hitch.main.workflows.autonomous_goals._release_autonomous_goal_snapshot_ref")
    @patch("hitch.main.workflows.autonomous_goals.create_worktree_for_session", side_effect=RuntimeError("disk full"))
    @patch("hitch.main.workflows.autonomous_goals.snapshot_worktree_to_commit", return_value="snapshot-sha")
    def test_review_worktree_failure_releases_retained_snapshot(
        self,
        _snapshot: MagicMock,
        _create_worktree: MagicMock,
        release_ref: MagicMock,
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, "disk full"):
            autonomous_goals._spawn_review_run(
                self.workflow,
                self.goal,
                candidate_run=self.candidate_run,
                candidate=dict(_CANDIDATE),
                attempt=1,
                candidate_cwd=self.candidate_instance.cwd,
            )

        release_ref.assert_called_once()
        repo_path, snapshot_ref = release_ref.call_args.args
        self.assertEqual(repo_path, self.project.repo_path)
        self.assertTrue(snapshot_ref.startswith(f"refs/hitch/autonomous-goals/{self.workflow.pk}/"))

    @patch("hitch.main.workflows.autonomous_goals.time.sleep")
    @patch("hitch.main.workflows.autonomous_goals._release_autonomous_goal_snapshot_ref")
    @patch("hitch.main.workflows.autonomous_goals._cleanup_autonomous_goal_candidate_cwd")
    @patch("hitch.main.workflows.autonomous_goals._spawn_review_run")
    @patch("hitch.main.workflows.autonomous_goals.codex_pool.interrupt_instance")
    def test_cancelled_review_waits_for_terminal_interruption_before_cleanup(
        self,
        interrupt: MagicMock,
        spawn_review: MagicMock,
        cleanup_cwd: MagicMock,
        release_ref: MagicMock,
        _sleep: MagicMock,
    ) -> None:
        review = self._review(status=SystemAgentRun.STATUS_RUNNING)
        review.output = {}
        review.input = {**review.input, "cwd": "/review-worktree", "managed_review_cwd": True}
        review.save(update_fields=["input", "output"])
        review.instance.status = CodexInstance.STATUS_STARTING
        review.instance.pid = 0
        review.instance.save(update_fields=["status", "pid"])
        spawn_review.return_value = review

        def interrupt_review(*args: object, **kwargs: object) -> CodexInstance | None:
            if interrupt.call_count == 1:
                return None
            review.instance.status = CodexInstance.STATUS_FAILED
            review.instance.error = "cancelled"
            review.instance.save(update_fields=["status", "error"])
            return review.instance

        interrupt.side_effect = interrupt_review
        context = self._candidate_context()
        context = ToolContext(**{**context.__dict__, "cancel_requested": lambda: True})

        with self.assertRaisesRegex(ValueError, "candidate stopped"):
            autonomous_goals.candidate_request_review(dict(_CANDIDATE), context)

        review.refresh_from_db()
        self.assertEqual(review.status, SystemAgentRun.STATUS_FAILED)
        self.assertEqual(interrupt.call_count, 2)
        cleanup_cwd.assert_called_once_with("/review-worktree")
        release_ref.assert_called_once()

    @patch("hitch.main.workflows.autonomous_goals.session_index.upsert_local_session")
    @patch(
        "hitch.main.workflows.autonomous_goals.codex_pool.create_session_thread_with_path",
        return_value=("bound-candidate", ""),
    )
    @patch("hitch.main.workflows.autonomous_goals._prepare_autonomous_goal_candidate_cwd", return_value=("/repo", None))
    @patch("hitch.main.workflows.autonomous_goals.codex_pool.spawn_turn")
    def test_candidate_run_is_bound_before_worker_launch(
        self,
        spawn_turn: MagicMock,
        _prepare_cwd: MagicMock,
        _create_thread: MagicMock,
        _upsert: MagicMock,
    ) -> None:
        def launch(**kwargs: object) -> CodexInstance:
            instance = self._instance(
                "bound-candidate",
                agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            )
            callback = kwargs["before_worker_launch"]
            assert callable(callback)
            callback(instance)
            self.assertTrue(SystemAgentRun.objects.filter(instance=instance).exists())
            return instance

        spawn_turn.side_effect = launch

        run = autonomous_goals._spawn_autonomous_goal_candidate_run(self.workflow, self.goal)

        self.assertEqual(run.thread_id, "bound-candidate")

    @patch("hitch.main.workflows.autonomous_goals.session_index.upsert_local_session")
    @patch(
        "hitch.main.workflows.autonomous_goals.codex_pool.create_session_thread_with_path",
        return_value=("bound-reviewer", ""),
    )
    @patch("hitch.main.workflows.autonomous_goals.codex_pool.spawn_turn")
    def test_review_run_is_bound_before_worker_launch(
        self,
        spawn_turn: MagicMock,
        _create_thread: MagicMock,
        _upsert: MagicMock,
    ) -> None:
        self.goal.autonomy = AutonomousGoal.AUTONOMY_PROPOSE_ONLY
        self.goal.save(update_fields=["autonomy"])
        self.candidate_run.input = {**self.candidate_run.input, "cwd": self.project.repo_path}
        self.candidate_run.save(update_fields=["input"])

        def launch(**kwargs: object) -> CodexInstance:
            instance = self._instance(
                "bound-reviewer",
                agent_kind=system_agents.AUTONOMOUS_GOAL_REVIEWER_AGENT_KIND,
            )
            callback = kwargs["before_worker_launch"]
            assert callable(callback)
            callback(instance)
            self.assertTrue(SystemAgentRun.objects.filter(instance=instance).exists())
            return instance

        spawn_turn.side_effect = launch

        run = autonomous_goals._spawn_review_run(
            self.workflow,
            self.goal,
            candidate_run=self.candidate_run,
            candidate=dict(_CANDIDATE),
            attempt=1,
            candidate_cwd=self.candidate_instance.cwd,
        )

        self.assertEqual(run.thread_id, "bound-reviewer")

    @patch("hitch.main.workflows.autonomous_goals.session_index.upsert_local_session")
    @patch(
        "hitch.main.workflows.autonomous_goals.codex_pool.create_session_thread_with_path",
        return_value=("stopped-reviewer", ""),
    )
    @patch("hitch.main.workflows.autonomous_goals.codex_pool.spawn_turn")
    def test_review_binding_rejects_latched_workflow_stop(
        self,
        spawn_turn: MagicMock,
        _create_thread: MagicMock,
        _upsert: MagicMock,
    ) -> None:
        self.goal.autonomy = AutonomousGoal.AUTONOMY_PROPOSE_ONLY
        self.goal.save(update_fields=["autonomy"])
        self.candidate_run.input = {**self.candidate_run.input, "cwd": self.project.repo_path}
        self.candidate_run.save(update_fields=["input"])
        self.workflow.state = {**self.workflow.state, "error": "stopped"}
        self.workflow.save(update_fields=["state"])

        def launch(**kwargs: object) -> CodexInstance:
            instance = self._instance(
                "stopped-reviewer",
                agent_kind=system_agents.AUTONOMOUS_GOAL_REVIEWER_AGENT_KIND,
            )
            callback = kwargs["before_worker_launch"]
            assert callable(callback)
            callback(instance)
            return instance

        spawn_turn.side_effect = launch

        with self.assertRaisesRegex(ValueError, "no longer active"):
            autonomous_goals._spawn_review_run(
                self.workflow,
                self.goal,
                candidate_run=self.candidate_run,
                candidate=dict(_CANDIDATE),
                attempt=1,
                candidate_cwd=self.candidate_instance.cwd,
            )

        self.assertFalse(SystemAgentRun.objects.filter(thread_id="stopped-reviewer").exists())


class AutonomousGoalFinishTests(_AutonomousGoalFixture):
    def _approved_output(self) -> dict[str, object]:
        return {
            "terminal": "propose",
            "candidate": dict(_CANDIDATE),
            "review": {
                "verdict": "approve",
                "confidence": "high",
                "feedback": "ship it",
            },
            "snapshot_sha": "abc123",
            "snapshot_ref": "refs/hitch/autonomous-goals/1/final",
            "review_thread_id": "review-thread",
        }

    @patch("hitch.main.workflows.autonomous_goals._autonomous_goal_instance_tokens_used", return_value=123)
    @patch("hitch.main.workflows.system_agents.final_agent_text", return_value="done")
    @patch("hitch.main.workflows.autonomous_goals._spawn_autonomous_goal_candidate_or_finish")
    def test_approved_candidate_continues_bounded_stack(
        self,
        spawn_candidate: MagicMock,
        _final_text: MagicMock,
        _tokens: MagicMock,
    ) -> None:
        self.candidate_run.output = self._approved_output()
        self.candidate_run.save(update_fields=["output"])
        self.candidate_instance.status = CodexInstance.STATUS_COMPLETED
        self.candidate_instance.save(update_fields=["status"])

        autonomous_goals.on_agent_finished(self.candidate_instance, self.candidate_run, self.workflow)

        self.workflow.refresh_from_db()
        self.candidate_run.refresh_from_db()
        self.assertEqual(self.workflow.status, SystemWorkflow.STATUS_RUNNING)
        self.assertEqual(self.candidate_run.status, SystemAgentRun.STATUS_COMPLETED)
        self.assertEqual(self.candidate_run.output["tokens_used"], 123)
        spawn_candidate.assert_called_once()
        self.assertFalse(ProposedSession.objects.exists())

    @patch("hitch.main.workflows.autonomous_goals._auto_proposal_quota_status", return_value="low")
    @patch("hitch.main.workflows.autonomous_goals._autonomous_goal_instance_tokens_used", return_value=123)
    @patch("hitch.main.workflows.system_agents.final_agent_text", return_value="done")
    @patch("hitch.main.workflows.autonomous_goals._spawn_autonomous_goal_candidate_or_finish")
    def test_auto_stack_publishes_checkpoint_when_fresh_quota_is_low(
        self,
        spawn_candidate: MagicMock,
        _final_text: MagicMock,
        _tokens: MagicMock,
        quota_status: MagicMock,
    ) -> None:
        self.workflow.state = {**self.workflow.state, "auto_proposal": True}
        self.workflow.save(update_fields=["state"])
        self.candidate_run.output = self._approved_output()
        self.candidate_run.save(update_fields=["output"])
        self.candidate_instance.status = CodexInstance.STATUS_COMPLETED
        self.candidate_instance.save(update_fields=["status"])

        autonomous_goals.on_agent_finished(self.candidate_instance, self.candidate_run, self.workflow)

        proposal = ProposedSession.objects.get(inbox_kind=ProposedSession.INBOX_KIND_PROPOSAL)
        self.workflow.refresh_from_db()
        self.assertEqual(self.workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(
            proposal.outcome_metadata["stacked_diff_continuation_stopped_reason"],
            "quota_low",
        )
        quota_status.assert_called_once()
        spawn_candidate.assert_not_called()

    @patch("hitch.main.workflows.autonomous_goals._autonomous_goal_instance_tokens_used", return_value=123)
    @patch("hitch.main.workflows.system_agents.final_agent_text", return_value="done")
    @patch("hitch.main.workflows.autonomous_goals._cleanup_autonomous_goal_candidate_cwd")
    def test_final_stack_candidate_publishes_reviewed_snapshot(
        self,
        cleanup_cwd: MagicMock,
        _final_text: MagicMock,
        _tokens: MagicMock,
    ) -> None:
        self.candidate_run.input = {
            **self.candidate_run.input,
            "stack_iteration": 2,
        }
        self.candidate_run.output = self._approved_output()
        self.candidate_run.save(update_fields=["input", "output"])
        self.candidate_instance.status = CodexInstance.STATUS_COMPLETED
        self.candidate_instance.save(update_fields=["status"])
        SessionMetadata.objects.create(
            thread_id="review-thread",
            cwd="/review",
            project=self.project,
            is_hidden_system_session=True,
        )

        autonomous_goals.on_agent_finished(self.candidate_instance, self.candidate_run, self.workflow)

        proposal = ProposedSession.objects.get(inbox_kind=ProposedSession.INBOX_KIND_PROPOSAL)
        self.workflow.refresh_from_db()
        self.assertEqual(self.workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(self.workflow.step, system_agents.STEP_AUTONOMOUS_GOAL_PROPOSED)
        self.assertEqual(proposal.title, _CANDIDATE["title"])
        self.assertEqual(proposal.outcome_metadata["approved_snapshot_sha"], "abc123")
        self.assertEqual(proposal.outcome_metadata["stacked_diff_iteration"], 2)
        self.assertEqual(proposal.outcome_metadata["proposal_budget_tokens_used"], 123)
        cleanup_cwd.assert_called_once_with(self.candidate_instance.cwd)

    @patch("hitch.main.workflows.autonomous_goals._autonomous_goal_instance_tokens_used", return_value=None)
    @patch("hitch.main.workflows.system_agents.final_agent_text", return_value="done")
    @patch("hitch.main.workflows.autonomous_goals._cleanup_autonomous_goal_candidate_cwd")
    def test_no_proposal_after_checkpoint_publishes_checkpoint(
        self,
        cleanup_cwd: MagicMock,
        _final_text: MagicMock,
        _tokens: MagicMock,
    ) -> None:
        previous_instance = self._instance(
            "previous-candidate",
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            cwd="/previous-worktree",
            status=CodexInstance.STATUS_COMPLETED,
        )
        previous = SystemAgentRun.objects.create(
            workflow=self.workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id=previous_instance.thread_id,
            instance=previous_instance,
            status=SystemAgentRun.STATUS_COMPLETED,
            input={
                "cwd": previous_instance.cwd,
                "managed_candidate_cwd": True,
                "stack_iteration": 1,
            },
            output=self._approved_output(),
        )
        SessionMetadata.objects.create(
            thread_id=previous.thread_id,
            cwd=previous_instance.cwd,
            project=self.project,
            is_hidden_system_session=True,
        )
        self.candidate_run.output = {
            "terminal": "no_proposal",
            "reason": "No further improvement.",
        }
        self.candidate_run.input = {**self.candidate_run.input, "stack_iteration": 2}
        self.candidate_run.save(update_fields=["input", "output"])
        self.candidate_instance.status = CodexInstance.STATUS_COMPLETED
        self.candidate_instance.save(update_fields=["status"])

        autonomous_goals.on_agent_finished(self.candidate_instance, self.candidate_run, self.workflow)

        proposal = ProposedSession.objects.get(inbox_kind=ProposedSession.INBOX_KIND_PROPOSAL)
        self.assertIsNotNone(proposal.candidate_session)
        assert proposal.candidate_session is not None
        self.assertEqual(proposal.candidate_session.thread_id, previous.thread_id)
        self.assertEqual(
            proposal.outcome_metadata["stacked_diff_continuation_stopped_reason"],
            "candidate_no_proposal",
        )
        cleanup_cwd.assert_called_with(self.candidate_instance.cwd)

    @patch("hitch.main.workflows.autonomous_goals._autonomous_goal_instance_tokens_used", return_value=None)
    @patch("hitch.main.workflows.system_agents.final_agent_text", return_value="final prose only")
    @patch("hitch.main.workflows.autonomous_goals._cleanup_autonomous_goal_candidate_cwd")
    def test_final_prose_without_terminal_tool_fails_once(
        self,
        cleanup_cwd: MagicMock,
        _final_text: MagicMock,
        _tokens: MagicMock,
    ) -> None:
        self.candidate_instance.status = CodexInstance.STATUS_COMPLETED
        self.candidate_instance.save(update_fields=["status"])

        autonomous_goals.on_agent_finished(self.candidate_instance, self.candidate_run, self.workflow)

        self.workflow.refresh_from_db()
        self.candidate_run.refresh_from_db()
        notice = ProposedSession.objects.get(inbox_kind=ProposedSession.INBOX_KIND_NOTICE)
        self.assertEqual(self.workflow.status, SystemWorkflow.STATUS_FAILED)
        self.assertIn("without calling", self.candidate_run.error)
        self.assertEqual(notice.outcome_metadata["automation_status"], "failed")
        self.assertEqual(SystemAgentRun.objects.count(), 1)
        cleanup_cwd.assert_called_once_with(self.candidate_instance.cwd)

    @patch("hitch.main.workflows.autonomous_goals._autonomous_goal_instance_tokens_used", return_value=50)
    @patch("hitch.main.workflows.system_agents.final_agent_text", return_value="ignored")
    def test_reviewer_without_verdict_fails_without_reminder_turn(
        self, _final_text: MagicMock, _tokens: MagicMock
    ) -> None:
        review = self._review(status=SystemAgentRun.STATUS_RUNNING)
        review.output = {}
        review.save(update_fields=["output"])

        autonomous_goals.on_agent_finished(review.instance, review, self.workflow)

        review.refresh_from_db()
        self.assertEqual(review.status, SystemAgentRun.STATUS_FAILED)
        self.assertIn("without calling", review.error)
        self.assertEqual(
            SystemAgentRun.objects.filter(agent_kind=system_agents.AUTONOMOUS_GOAL_REVIEWER_AGENT_KIND).count(),
            1,
        )

    @patch("hitch.main.workflows.autonomous_goals._release_autonomous_goal_snapshot_ref")
    @patch("hitch.main.workflows.autonomous_goals._cleanup_autonomous_goal_candidate_cwd")
    @patch("hitch.main.workflows.autonomous_goals._autonomous_goal_instance_tokens_used", return_value=None)
    @patch("hitch.main.workflows.system_agents.final_agent_text", return_value="legacy output")
    def test_finishing_legacy_turn_retires_workflow_and_resources(
        self,
        _final_text: MagicMock,
        _tokens: MagicMock,
        cleanup_cwd: MagicMock,
        release_ref: MagicMock,
    ) -> None:
        self.workflow.step = "autonomous_goal_judge_running"
        self.workflow.state = {
            **self.workflow.state,
            "session_cwd": self.candidate_instance.cwd,
            "judge_snapshot_cwd": "/legacy-review",
            "approved_snapshot_ref": "refs/hitch/autonomous-goals/legacy",
        }
        self.workflow.save(update_fields=["step", "state"])
        self.candidate_instance.status = CodexInstance.STATUS_COMPLETED
        self.candidate_instance.save(update_fields=["status"])

        autonomous_goals.on_agent_finished(self.candidate_instance, self.candidate_run, self.workflow)

        self.workflow.refresh_from_db()
        self.candidate_run.refresh_from_db()
        self.assertEqual(self.workflow.status, SystemWorkflow.STATUS_FAILED)
        self.assertEqual(self.candidate_run.status, SystemAgentRun.STATUS_FAILED)
        self.assertIn("tool-driven protocol upgrade", self.candidate_run.error)
        self.assertTrue(ProposedSession.objects.filter(inbox_kind=ProposedSession.INBOX_KIND_NOTICE).exists())
        cleanup_cwd.assert_has_calls(
            [call(self.candidate_instance.cwd), call("/legacy-review")],
            any_order=True,
        )
        release_ref.assert_called_once_with(
            self.project.repo_path,
            "refs/hitch/autonomous-goals/legacy",
        )

    @patch("hitch.main.workflows.autonomous_goals._release_autonomous_goal_snapshot_ref")
    @patch("hitch.main.workflows.autonomous_goals._cleanup_autonomous_goal_candidate_cwd")
    @patch("hitch.main.workflows.autonomous_goals.codex_pool.interrupt_instance")
    def test_stopping_stack_cleans_completed_checkpoint_resources(
        self,
        interrupt: MagicMock,
        cleanup_cwd: MagicMock,
        release_ref: MagicMock,
    ) -> None:
        interrupt.return_value = self.candidate_instance
        checkpoint_instance = self._instance(
            "checkpoint",
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            cwd="/checkpoint-worktree",
            status=CodexInstance.STATUS_COMPLETED,
        )
        checkpoint = SystemAgentRun.objects.create(
            workflow=self.workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id=checkpoint_instance.thread_id,
            instance=checkpoint_instance,
            status=SystemAgentRun.STATUS_COMPLETED,
            input={"cwd": checkpoint_instance.cwd, "managed_candidate_cwd": True, "stack_iteration": 1},
            output={"terminal": "propose", "snapshot_ref": "refs/hitch/autonomous-goals/checkpoint"},
        )

        self.assertTrue(autonomous_goals.stop_running_autonomous_goal_workflow(self.goal.pk, "stopped"))

        self.workflow.refresh_from_db()
        self.candidate_run.refresh_from_db()
        checkpoint.refresh_from_db()
        self.assertEqual(self.workflow.status, SystemWorkflow.STATUS_BLOCKED)
        self.assertEqual(self.candidate_run.status, SystemAgentRun.STATUS_FAILED)
        self.assertEqual(checkpoint.status, SystemAgentRun.STATUS_COMPLETED)
        cleanup_cwd.assert_has_calls(
            [call(self.candidate_instance.cwd), call(checkpoint_instance.cwd)],
            any_order=True,
        )
        release_ref.assert_called_once_with(
            self.project.repo_path,
            "refs/hitch/autonomous-goals/checkpoint",
        )

    @patch("hitch.main.workflows.autonomous_goals.codex_pool.interrupt_instance", return_value=None)
    def test_stop_is_latched_before_an_uninterruptible_worker_returns(self, _interrupt: MagicMock) -> None:
        self.assertFalse(autonomous_goals.stop_running_autonomous_goal_workflow(self.goal.pk, "stopped"))

        self.workflow.refresh_from_db()
        self.assertEqual(self.workflow.status, SystemWorkflow.STATUS_RUNNING)
        self.assertEqual(self.workflow.state["error"], "stopped")

        self.candidate_instance.status = CodexInstance.STATUS_FAILED
        self.candidate_instance.save(update_fields=["status"])
        with (
            patch("hitch.main.workflows.system_agents.final_agent_text", return_value="stopped"),
            patch("hitch.main.workflows.autonomous_goals._autonomous_goal_instance_tokens_used", return_value=None),
            patch("hitch.main.workflows.autonomous_goals._cleanup_autonomous_goal_candidate_cwd") as cleanup_cwd,
        ):
            autonomous_goals.on_agent_finished(self.candidate_instance, self.candidate_run, self.workflow)

        self.workflow.refresh_from_db()
        self.candidate_run.refresh_from_db()
        self.assertEqual(self.workflow.status, SystemWorkflow.STATUS_BLOCKED)
        self.assertEqual(self.candidate_run.status, SystemAgentRun.STATUS_FAILED)
        cleanup_cwd.assert_called_once_with(self.candidate_instance.cwd)

    @patch("hitch.main.workflows.autonomous_goals._spawn_autonomous_goal_candidate_or_finish")
    def test_fresh_terminal_routing_claim_defers_orphan_recovery(self, spawn: MagicMock) -> None:
        self.candidate_instance.status = CodexInstance.STATUS_COMPLETED
        self.candidate_instance.workflow_routing_started_at = timezone.now()
        self.candidate_instance.save(update_fields=["status", "workflow_routing_started_at"])
        SystemWorkflow.objects.filter(pk=self.workflow.pk).update(
            updated_at=timezone.now() - timedelta(hours=1)
        )
        self.workflow.refresh_from_db()

        self.assertEqual(autonomous_goals.recover_orphaned_workflows([self.workflow]), 0)

        spawn.assert_not_called()

    @patch("hitch.main.workflows.autonomous_goals._spawn_autonomous_goal_candidate_or_finish")
    def test_orphan_recovery_claim_prevents_duplicate_spawn(self, spawn: MagicMock) -> None:
        self.candidate_instance.delete()
        SystemWorkflow.objects.filter(pk=self.workflow.pk).update(
            updated_at=timezone.now() - timedelta(hours=1)
        )
        self.workflow.refresh_from_db()

        self.assertEqual(autonomous_goals.recover_orphaned_workflows([self.workflow]), 1)
        self.assertEqual(autonomous_goals.recover_orphaned_workflows([self.workflow]), 0)

        spawn.assert_called_once()

    @patch("hitch.main.workflows.autonomous_goals._release_autonomous_goal_snapshot_ref")
    @patch("hitch.main.workflows.autonomous_goals._cleanup_autonomous_goal_candidate_cwd")
    def test_orphaned_legacy_workflow_is_retired(
        self,
        cleanup_cwd: MagicMock,
        release_ref: MagicMock,
    ) -> None:
        self.workflow.step = "autonomous_goal_history_summarizing"
        self.workflow.state = {
            **self.workflow.state,
            "session_cwd": self.candidate_instance.cwd,
            "approved_snapshot_ref": "refs/hitch/autonomous-goals/legacy",
        }
        self.workflow.save(update_fields=["step", "state"])
        self.candidate_instance.status = CodexInstance.STATUS_FAILED
        self.candidate_instance.save(update_fields=["status"])

        self.assertEqual(
            system_agents.reconcile_terminal_workflow_instances(workflow_id=self.workflow.pk),
            1,
        )

        self.workflow.refresh_from_db()
        self.candidate_run.refresh_from_db()
        self.assertEqual(self.workflow.status, SystemWorkflow.STATUS_FAILED)
        self.assertEqual(self.candidate_run.status, SystemAgentRun.STATUS_FAILED)
        cleanup_cwd.assert_called_once_with(self.candidate_instance.cwd)
        release_ref.assert_called_once_with(
            self.project.repo_path,
            "refs/hitch/autonomous-goals/legacy",
        )

    @patch("hitch.main.workflows.autonomous_goals._release_autonomous_goal_snapshot_ref")
    @patch("hitch.main.workflows.autonomous_goals._cleanup_autonomous_goal_candidate_cwd")
    @patch("hitch.main.workflows.autonomous_goals._autonomous_goal_instance_tokens_used", return_value=None)
    @patch("hitch.main.workflows.system_agents.final_agent_text", return_value="late finish")
    def test_late_finish_after_goal_deletion_cleans_run_resources(
        self,
        _final_text: MagicMock,
        _tokens: MagicMock,
        cleanup_cwd: MagicMock,
        release_ref: MagicMock,
    ) -> None:
        self.goal.deleted_at = timezone.now()
        self.goal.save(update_fields=["deleted_at"])
        self.workflow.status = SystemWorkflow.STATUS_BLOCKED
        self.workflow.step = system_agents.STEP_BLOCKED
        self.workflow.save(update_fields=["status", "step"])
        self.candidate_run.output = {"snapshot_ref": "refs/hitch/autonomous-goals/late"}
        self.candidate_run.save(update_fields=["output"])
        self.candidate_instance.status = CodexInstance.STATUS_COMPLETED
        self.candidate_instance.save(update_fields=["status"])

        autonomous_goals.on_agent_finished(self.candidate_instance, self.candidate_run, self.workflow)

        cleanup_cwd.assert_called_once_with(self.candidate_instance.cwd)
        release_ref.assert_called_once_with(
            self.workflow.cwd,
            "refs/hitch/autonomous-goals/late",
        )

    @patch("hitch.main.workflows.autonomous_goals._release_autonomous_goal_snapshot_ref")
    @patch("hitch.main.workflows.autonomous_goals._cleanup_autonomous_goal_candidate_cwd")
    @patch("hitch.main.workflows.autonomous_goals._autonomous_goal_instance_tokens_used", return_value=None)
    @patch("hitch.main.workflows.system_agents.final_agent_text", return_value="failed candidate")
    def test_terminal_cleanup_is_retried_from_the_run_ledger(
        self,
        _final_text: MagicMock,
        _tokens: MagicMock,
        cleanup_cwd: MagicMock,
        release_ref: MagicMock,
    ) -> None:
        snapshot_ref = "refs/hitch/autonomous-goals/pending-cleanup"
        self.candidate_run.output = {"snapshot_ref": snapshot_ref}
        self.candidate_run.save(update_fields=["output"])
        self.candidate_instance.status = CodexInstance.STATUS_COMPLETED
        self.candidate_instance.save(update_fields=["status"])

        with (
            patch(
                "hitch.main.workflows.autonomous_goals._apply_finish_action",
                side_effect=RuntimeError("process exited"),
            ),
            self.assertRaisesRegex(RuntimeError, "process exited"),
        ):
            autonomous_goals.on_agent_finished(self.candidate_instance, self.candidate_run, self.workflow)

        self.workflow.refresh_from_db()
        self.candidate_run.refresh_from_db()
        self.assertEqual(self.workflow.status, SystemWorkflow.STATUS_FAILED)
        self.assertIn("pending_cleanup", self.candidate_run.output)

        self.assertEqual(
            system_agents.reconcile_terminal_workflow_instances(workflow_id=self.workflow.pk),
            1,
        )

        self.candidate_run.refresh_from_db()
        self.assertNotIn("pending_cleanup", self.candidate_run.output)
        cleanup_cwd.assert_called_once_with(self.candidate_instance.cwd)
        release_ref.assert_called_once_with(self.project.repo_path, snapshot_ref)

    @patch("hitch.main.workflows.autonomous_goals._cleanup_autonomous_goal_candidate_cwd")
    @patch("hitch.main.workflows.autonomous_goals.codex_pool.interrupt_instance")
    def test_stopping_legacy_stack_preserves_accepted_session_worktree(
        self,
        interrupt: MagicMock,
        cleanup_cwd: MagicMock,
    ) -> None:
        interrupt.return_value = self.candidate_instance
        metadata = SessionMetadata.objects.get(thread_id=self.candidate_instance.thread_id)
        ProposedSession.objects.create(
            project=self.project,
            autonomous_goal=self.goal,
            source_workflow=self.workflow,
            title="Accepted legacy work",
            candidate_session=metadata,
            accepted_session=metadata,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
        )

        self.assertTrue(autonomous_goals.stop_running_autonomous_goal_workflow(self.goal.pk, "accepted"))

        cleanup_cwd.assert_not_called()

    @patch("hitch.main.workflows.autonomous_goals.codex_pool.interrupt_instance")
    def test_resolving_stale_proposal_does_not_stop_newer_workflow(self, interrupt: MagicMock) -> None:
        old_workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=self.workflow.main_thread_id,
            cwd=self.workflow.cwd,
            status=SystemWorkflow.STATUS_COMPLETED,
            step=system_agents.STEP_AUTONOMOUS_GOAL_PROPOSED,
            state={"autonomous_goal_id": self.goal.pk},
        )
        stale = ProposedSession.objects.create(
            project=self.project,
            autonomous_goal=self.goal,
            source_workflow=old_workflow,
            title="Stale proposal",
            outcome_status=ProposedSession.OUTCOME_REJECTED,
        )

        self.assertTrue(
            autonomous_goals.stop_running_autonomous_goal_stack_after_proposal_resolution(
                self.goal.pk,
                stale.pk,
                stale.outcome_status,
            )
        )

        self.workflow.refresh_from_db()
        self.assertEqual(self.workflow.status, SystemWorkflow.STATUS_RUNNING)
        interrupt.assert_not_called()


class AutonomousGoalHistoryTests(TestCase):
    def test_list_sessions_returns_rollout_paths_without_summarizing(self) -> None:
        with TemporaryDirectory() as temp_dir:
            rollout_path = f"{temp_dir}/rollout.jsonl"
            with open(rollout_path, "w", encoding="utf-8") as rollout_file:
                rollout_file.write("{}\n")
            project = Project.objects.create(name="repo", repo_path="/repo")
            goal = AutonomousGoal.objects.create(
                project=project,
                title="Goal",
                goal="Inspect history directly.",
            )
            workflow = SystemWorkflow.objects.create(
                kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
                main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(goal.pk),
                cwd=project.repo_path,
                status=SystemWorkflow.STATUS_RUNNING,
                step=system_agents.STEP_AUTONOMOUS_GOAL_RUNNING,
                state={"autonomous_goal_id": goal.pk},
            )
            prior_metadata = SessionMetadata.objects.create(
                thread_id="prior",
                cwd="/repo",
                project=project,
                codex_path=rollout_path,
                is_hidden_system_session=True,
            )
            current_metadata = SessionMetadata.objects.create(
                thread_id="current",
                cwd="/repo",
                project=project,
                is_hidden_system_session=True,
            )
            prior_instance = CodexInstance.objects.create(
                pid=1,
                thread_id=prior_metadata.thread_id,
                cwd="/repo",
                events_path="/tmp/prior.jsonl",
                status=CodexInstance.STATUS_COMPLETED,
                purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
                workflow_id=workflow.pk,
                agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            )
            current_instance = CodexInstance.objects.create(
                pid=2,
                thread_id=current_metadata.thread_id,
                cwd="/repo",
                events_path="/tmp/current.jsonl",
                status=CodexInstance.STATUS_RUNNING,
                purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
                workflow_id=workflow.pk,
                agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            )
            SystemAgentRun.objects.create(
                workflow=workflow,
                agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
                thread_id=prior_instance.thread_id,
                instance=prior_instance,
                status=SystemAgentRun.STATUS_COMPLETED,
            )
            SystemAgentRun.objects.create(
                workflow=workflow,
                agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
                thread_id=current_instance.thread_id,
                instance=current_instance,
                status=SystemAgentRun.STATUS_RUNNING,
            )
            context = ToolContext(
                cwd="/repo",
                thread_id=current_instance.thread_id,
                instance_id=current_instance.pk,
                purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
                workflow_id=workflow.pk,
                agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            )

            rows = autonomous_goals.candidate_goal_sessions(context)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["session_id"], "prior")
            self.assertEqual(rows[0]["session_file"], rollout_path)
            self.assertTrue(rows[0]["session_file_available"])
