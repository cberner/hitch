"""Role-scoped tools for autonomous-goal candidates and judges."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, cast, override
from unittest.mock import MagicMock, patch

from django.test import TestCase

from hitch.main.models import (
    AutonomousGoal,
    CodexInstance,
    ProposedSession,
    SessionMetadata,
    SystemAgentRun,
    SystemWorkflow,
)
from hitch.main.runtime import codex_events
from hitch.main.runtime.codex_tools import (
    ToolContext,
    handle_dynamic_tool_call,
    registered_dynamic_tool_specs,
)
from hitch.main.test.support import _make_project
from hitch.main.workflows import autonomous_goals, system_agents
from hitch.main.worktrees import ManagedWorktree


def _tool_payload(response: dict[str, Any]) -> Any:
    return json.loads(response["contentItems"][0]["text"])


def _candidate_arguments(*, title: str = "Harden retries") -> dict[str, Any]:
    return {
        "title": title,
        "summary": "Avoid retry loss.",
        "impact": "Fewer stalled workflows.",
        "implemented_changes": "Added an atomic claim.",
        "implementation_direction": "Continue from the judged snapshot.",
        "verification": "Focused tests pass.",
        "rough_edges": "None known.",
        "suggested_continuation": "Review and publish.",
        "relevant_files": ["hitch/main/workflows/autonomous_goals.py"],
    }


def _events_file(test: TestCase, *, thread_id: str, tokens_used: int) -> str:
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as events:
        events.write(
            json.dumps(
                {
                    "method": codex_events.GOAL_UPDATED_METHOD,
                    "payload": {
                        "threadId": thread_id,
                        "goal": {
                            "objective": "Judge candidate",
                            "tokensUsed": tokens_used,
                        },
                    },
                }
            )
            + "\n"
        )
        events.write(
            json.dumps(
                {
                    "method": "item/completed",
                    "payload": {
                        "item": {
                            "id": "judge-message",
                            "type": "agentMessage",
                            "text": "Verdict recorded.",
                        }
                    },
                }
            )
            + "\n"
        )
        path = events.name
    test.addCleanup(Path(path).unlink, missing_ok=True)
    return path


class AutonomousGoalToolTests(TestCase):
    @override
    def setUp(self) -> None:
        self.project = _make_project(repo_path="/repo")
        self.goal = AutonomousGoal.objects.create(
            project=self.project,
            title="Improve reliability",
            goal="Find and fix a meaningful reliability problem.",
            confidence_threshold=AutonomousGoal.CONFIDENCE_HIGH,
        )
        self.candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/candidate",
            project=self.project,
            is_hidden_system_session=True,
        )
        self.workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_AUTONOMOUS_GOAL_RUN,
            main_thread_id=f"autonomous-goal:{self.goal.pk}",
            cwd="/repo",
            step=system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
            state={
                "autonomous_goal_id": self.goal.pk,
                "tool_protocol": True,
                "candidate_session_id": self.candidate.pk,
                "session_cwd": "/candidate",
                "stacked_diff_depth": 1,
                "stacked_diff_iteration": 1,
            },
        )

    def candidate_context(self) -> ToolContext:
        return ToolContext(
            cwd="/candidate",
            thread_id=self.candidate.thread_id,
            instance_id=10,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=self.workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )

    def judge_context(self, *, instance_id: int = 11) -> ToolContext:
        return ToolContext(
            cwd="/candidate",
            thread_id="judge-thread",
            instance_id=instance_id,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=self.workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
        )

    def call_candidate(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return handle_dynamic_tool_call(
            {"namespace": "hitch", "tool": name, "arguments": arguments},
            self.candidate_context(),
        )

    def record_and_finish_judge(
        self, arguments: dict[str, Any], *, approved: bool
    ) -> Any:
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="judge-thread",
            cwd="/candidate",
            prompt="judge",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=self.workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
        )
        run = SystemAgentRun.objects.create(
            workflow=self.workflow,
            instance=instance,
            agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
            thread_id=instance.thread_id,
            status=SystemAgentRun.STATUS_RUNNING,
        )
        autonomous_goals.judge_record_verdict(
            arguments,
            self.judge_context(instance_id=instance.pk),
            approved=approved,
        )
        self.workflow.refresh_from_db()
        self.assertEqual(
            self.workflow.state["judgment_verdict_id"],
            self.workflow.state["judgment_request_id"],
        )
        self.assertNotIn("judgment_result_id", self.workflow.state)
        self.assertEqual(
            self.workflow.step, system_agents.STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING
        )

        return autonomous_goals._handle_tool_judge_finished_locked(
            instance, run, self.workflow, "Judge finished after recording a verdict."
        )

    def test_registration_is_role_scoped(self) -> None:
        visible = {
            spec["name"]
            for spec in registered_dynamic_tool_specs(
                purpose=CodexInstance.PURPOSE_USER
            )
        }
        candidate = {
            spec["name"]
            for spec in registered_dynamic_tool_specs(
                purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
                agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            )
        }
        judge = {
            spec["name"]
            for spec in registered_dynamic_tool_specs(
                purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
                agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
            )
        }

        self.assertEqual(visible, {"propose_session", "rename_session", "watch_pr"})
        self.assertEqual(
            candidate,
            {"get_goal", "list_goal_sessions", "judge", "no_proposal"},
        )
        self.assertEqual(judge, {"approve", "deny"})
        self.assertEqual(
            registered_dynamic_tool_specs(
                purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
                agent_kind="unrelated",
            ),
            [],
        )

    @patch("hitch.main.workflows.autonomous_goals.codex_pool.spawn_turn")
    @patch(
        "hitch.main.workflows.autonomous_goals.codex_pool.create_session_thread_with_path",
        return_value=("new-candidate", "/rollouts/new-candidate.jsonl"),
    )
    @patch(
        "hitch.main.workflows.autonomous_goals._prepare_autonomous_goal_candidate_cwd",
        return_value=("/candidate", None),
    )
    def test_candidate_binding_is_persisted_before_initial_turn_starts(
        self,
        _mock_prepare: MagicMock,
        mock_create_thread: MagicMock,
        mock_spawn_turn: MagicMock,
    ) -> None:
        state = dict(self.workflow.state)
        state.pop("candidate_session_id")
        self.workflow.state = state
        self.workflow.save(update_fields=["state", "updated_at"])

        def spawn(**kwargs: Any) -> CodexInstance:
            bound_workflow = SystemWorkflow.objects.get(pk=self.workflow.pk)
            metadata = SessionMetadata.objects.get(
                pk=bound_workflow.state["candidate_session_id"]
            )
            self.assertEqual(metadata.thread_id, "new-candidate")
            self.assertEqual(metadata.codex_path, "/rollouts/new-candidate.jsonl")
            return CodexInstance.objects.create(
                pid=1,
                thread_id=kwargs["thread_id"],
                cwd=kwargs["cwd"],
                prompt=kwargs["prompt"],
                purpose=kwargs["purpose"],
                workflow_id=kwargs["workflow_id"],
                agent_kind=kwargs["agent_kind"],
            )

        mock_spawn_turn.side_effect = spawn

        run = autonomous_goals._spawn_autonomous_goal_candidate_run(
            self.workflow, self.goal
        )

        self.assertEqual(run.thread_id, "new-candidate")
        self.assertEqual(
            mock_create_thread.call_args.kwargs["agent_kind"],
            system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        self.assertEqual(
            mock_spawn_turn.call_args.kwargs["thread_id"], "new-candidate"
        )

    @patch("hitch.main.workflows.autonomous_goals.codex_pool.spawn_turn")
    @patch("hitch.main.workflows.autonomous_goals.create_worktree_for_session")
    @patch(
        "hitch.main.workflows.autonomous_goals.codex_pool.create_session_thread_with_path",
        return_value=("new-judge", "/rollouts/new-judge.jsonl"),
    )
    def test_judge_binding_is_persisted_before_initial_turn_starts(
        self,
        mock_create_thread: MagicMock,
        mock_create_worktree: MagicMock,
        mock_spawn_turn: MagicMock,
    ) -> None:
        snapshot = "a" * 40
        judge_worktree = ManagedWorktree(
            path=Path("/judge-snapshot"),
            branch="hitch/repo/judge-snapshot",
            source_repo=Path("/repo"),
        )
        mock_create_worktree.return_value = judge_worktree
        self.workflow.step = system_agents.STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING
        self.workflow.state = {
            **self.workflow.state,
            "judgment_request_id": "request",
            "approved_snapshot_sha": snapshot,
            "candidate": _candidate_arguments(),
        }
        self.workflow.save(update_fields=["step", "state", "updated_at"])

        def spawn(**kwargs: Any) -> CodexInstance:
            bound_workflow = SystemWorkflow.objects.get(pk=self.workflow.pk)
            metadata = SessionMetadata.objects.get(
                pk=bound_workflow.state["judge_session_id"]
            )
            self.assertEqual(metadata.thread_id, "new-judge")
            self.assertEqual(metadata.codex_path, "/rollouts/new-judge.jsonl")
            return CodexInstance.objects.create(
                pid=1,
                thread_id=kwargs["thread_id"],
                cwd=kwargs["cwd"],
                prompt=kwargs["prompt"],
                purpose=kwargs["purpose"],
                workflow_id=kwargs["workflow_id"],
                agent_kind=kwargs["agent_kind"],
            )

        mock_spawn_turn.side_effect = spawn

        run = autonomous_goals._spawn_autonomous_goal_judge_run(
            self.workflow,
            self.goal,
            _candidate_arguments(),
        )

        self.assertEqual(run.thread_id, "new-judge")
        self.assertEqual(
            mock_create_thread.call_args.kwargs["agent_kind"],
            system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
        )
        mock_create_worktree.assert_called_once_with("/repo", base_ref=snapshot)
        self.assertEqual(mock_create_thread.call_args.kwargs["cwd"], "/judge-snapshot")
        self.assertEqual(mock_spawn_turn.call_args.kwargs["cwd"], "/judge-snapshot")
        self.assertEqual(mock_spawn_turn.call_args.kwargs["thread_id"], "new-judge")

    def test_goal_and_history_tools_return_scoped_raw_sessions(self) -> None:
        with tempfile.NamedTemporaryFile() as rollout:
            old_candidate = SessionMetadata.objects.create(
                thread_id="old-candidate",
                cwd="/old",
                project=self.project,
                codex_name="Earlier candidate",
                codex_path=rollout.name,
                is_hidden_system_session=True,
            )
            accepted = SessionMetadata.objects.create(
                thread_id="accepted-thread",
                cwd="/accepted",
                project=self.project,
                codex_name="Accepted work",
                codex_path="/missing/rollout.jsonl",
            )
            ProposedSession.objects.create(
                autonomous_goal=self.goal,
                title="Prior proposal",
                candidate_session=old_candidate,
                accepted_session=accepted,
                outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            )

            goal_response = self.call_candidate("get_goal", {})
            history_response = self.call_candidate("list_goal_sessions", {})

        self.assertTrue(goal_response["success"])
        goal = cast(dict[str, Any], _tool_payload(goal_response))
        self.assertEqual(goal["goal"], self.goal.goal)
        self.assertEqual(goal["judgment_attempts_remaining"], 2)
        history = cast(list[dict[str, Any]], _tool_payload(history_response))
        self.assertEqual(
            [row["kind"] for row in history],
            ["candidate", "accepted_work"],
        )
        self.assertEqual(history[0]["session_file"], rollout.name)
        self.assertTrue(history[0]["session_file_available"])
        self.assertFalse(history[1]["session_file_available"])
        self.assertNotIn("candidate-thread", {row["session_id"] for row in history})

    def test_visible_session_cannot_invoke_candidate_tool(self) -> None:
        response = handle_dynamic_tool_call(
            {"namespace": "hitch", "tool": "get_goal", "arguments": {}},
            ToolContext(cwd="/repo", thread_id="visible"),
        )

        self.assertFalse(response["success"])
        self.assertIn("unavailable", response["contentItems"][0]["text"])

    def test_no_proposal_is_a_terminal_tool_state(self) -> None:
        response = self.call_candidate(
            "no_proposal",
            {"reason": "The remaining work is already covered."},
        )

        self.assertTrue(response["success"])
        self.workflow.refresh_from_db()
        self.assertEqual(self.workflow.state["candidate_terminal"], "no_proposal")
        self.assertEqual(
            self.workflow.state["candidate"]["message"],
            "The remaining work is already covered.",
        )

    @patch("hitch.main.workflows.autonomous_goals.snapshot_worktree_to_commit")
    @patch("hitch.main.workflows.autonomous_goals._spawn_autonomous_goal_judge_run")
    def test_judgment_blocks_for_tool_verdict_and_records_snapshot(
        self,
        mock_spawn: MagicMock,
        mock_snapshot: MagicMock,
    ) -> None:
        mock_snapshot.return_value = "a" * 40

        def approve(
            workflow: SystemWorkflow,
            _goal: AutonomousGoal,
            _candidate: dict[str, Any],
        ) -> MagicMock:
            judge = SessionMetadata.objects.create(
                thread_id="judge-thread",
                cwd="/candidate",
                project=self.project,
                is_hidden_system_session=True,
            )
            workflow.state = {**workflow.state, "judge_session_id": judge.pk}
            workflow.save(update_fields=["state", "updated_at"])
            self.record_and_finish_judge(
                {"confidence": "high", "feedback": "Concrete and verified."},
                approved=True,
            )
            return MagicMock(
                workflow_id=workflow.pk,
                input={"autonomous_goal_id": self.goal.pk},
            )

        mock_spawn.side_effect = approve
        response = self.call_candidate("judge", _candidate_arguments())

        self.assertTrue(response["success"])
        result = cast(dict[str, Any], _tool_payload(response))
        self.assertEqual(result["verdict"], "approve")
        self.assertEqual(result["judgment_attempts_remaining"], 1)
        mock_snapshot.assert_called_once()
        self.assertEqual(mock_snapshot.call_args.args, ("/candidate",))
        self.assertEqual(
            mock_snapshot.call_args.kwargs["message"],
            "Snapshot AG judgment attempt 1",
        )
        self.assertRegex(
            mock_snapshot.call_args.kwargs["retain_ref"],
            rf"refs/hitch/autonomous-goals/{self.workflow.pk}/[0-9a-f]{{32}}",
        )
        self.workflow.refresh_from_db()
        self.assertEqual(self.workflow.state["approved_snapshot_sha"], "a" * 40)
        self.assertEqual(self.workflow.state["candidate_terminal"], "approved")

    @patch("hitch.main.workflows.autonomous_goals.snapshot_worktree_to_commit")
    @patch("hitch.main.workflows.autonomous_goals._spawn_autonomous_goal_judge_run")
    def test_two_denials_return_feedback_and_exhaust_judgment_calls(
        self,
        mock_spawn: MagicMock,
        mock_snapshot: MagicMock,
    ) -> None:
        mock_snapshot.side_effect = ["a" * 40, "b" * 40]
        judge = SessionMetadata.objects.create(
            thread_id="judge-thread",
            cwd="/candidate",
            project=self.project,
            is_hidden_system_session=True,
        )
        feedback = iter(
            [
                "Add a regression test for the race.",
                "The revised test still misses cancellation.",
            ]
        )

        def deny(
            workflow: SystemWorkflow,
            _goal: AutonomousGoal,
            _candidate: dict[str, Any],
        ) -> MagicMock:
            workflow.refresh_from_db()
            self.assertNotIn("judge_protocol_recoveries", workflow.state)
            workflow.state = {**workflow.state, "judge_session_id": judge.pk}
            workflow.save(update_fields=["state", "updated_at"])
            self.record_and_finish_judge(
                {"confidence": "medium", "feedback": next(feedback)},
                approved=False,
            )
            return MagicMock(
                workflow_id=workflow.pk,
                input={"autonomous_goal_id": self.goal.pk},
            )

        mock_spawn.side_effect = deny
        first = self.call_candidate("judge", _candidate_arguments())
        first_result = cast(dict[str, Any], _tool_payload(first))

        self.assertTrue(first["success"])
        self.assertEqual(first_result["verdict"], "deny")
        self.assertEqual(
            first_result["feedback"], "Add a regression test for the race."
        )
        goal_state = cast(
            dict[str, Any], _tool_payload(self.call_candidate("get_goal", {}))
        )
        self.assertEqual(
            goal_state["last_judge_feedback"],
            "Add a regression test for the race.",
        )
        self.workflow.refresh_from_db()
        self.workflow.state = {
            **self.workflow.state,
            "judge_protocol_recoveries": 1,
        }
        self.workflow.save(update_fields=["state", "updated_at"])

        second = self.call_candidate(
            "judge", _candidate_arguments(title="Harden retry cancellation")
        )
        second_result = cast(dict[str, Any], _tool_payload(second))
        third = self.call_candidate("judge", _candidate_arguments())

        self.assertTrue(second["success"])
        self.assertEqual(second_result["verdict"], "deny")
        self.assertEqual(second_result["judgment_attempts_remaining"], 0)
        self.assertFalse(third["success"])
        self.assertIn("both judgment attempts", third["contentItems"][0]["text"])
        self.assertEqual(mock_spawn.call_count, 2)
        self.assertEqual(mock_snapshot.call_count, 2)

    def test_judge_tokens_are_accounted_before_candidate_receives_verdict(
        self,
    ) -> None:
        judge = SessionMetadata.objects.create(
            thread_id="judge-thread",
            cwd="/candidate",
            project=self.project,
            is_hidden_system_session=True,
        )
        self.workflow.step = system_agents.STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING
        self.workflow.state = {
            **self.workflow.state,
            "proposal_budget": 1_000,
            "judge_session_id": judge.pk,
            "judgment_request_id": "request",
        }
        self.workflow.save(update_fields=["step", "state", "updated_at"])
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id=judge.thread_id,
            cwd=judge.cwd,
            prompt="judge",
            events_path=_events_file(
                self, thread_id=judge.thread_id, tokens_used=175
            ),
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=self.workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
        )
        SystemAgentRun.objects.create(
            workflow=self.workflow,
            instance=instance,
            agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
            thread_id=judge.thread_id,
            status=SystemAgentRun.STATUS_RUNNING,
        )
        autonomous_goals.judge_record_verdict(
            {"confidence": "high", "feedback": "Approved."},
            self.judge_context(instance_id=instance.pk),
            approved=True,
        )

        self.assertTrue(system_agents.on_codex_instance_finished(instance))

        self.workflow.refresh_from_db()
        self.assertEqual(self.workflow.state["proposal_budget_tokens_used"], 175)
        self.assertEqual(self.workflow.state["judgment_result_id"], "request")
        self.assertNotIn("judgment_verdict_id", self.workflow.state)
        self.assertEqual(
            self.workflow.step,
            system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
        )

    def test_denial_releases_snapshot_and_judge_worktree_after_finish(self) -> None:
        snapshot_ref = (
            f"refs/hitch/autonomous-goals/{self.workflow.pk}/"
            "0123456789abcdef0123456789abcdef"
        )
        judge = SessionMetadata.objects.create(
            thread_id="judge-thread",
            cwd="/judge-snapshot",
            project=self.project,
            is_hidden_system_session=True,
        )
        self.workflow.step = system_agents.STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING
        self.workflow.state = {
            **self.workflow.state,
            "candidate": _candidate_arguments(),
            "judge_session_id": judge.pk,
            "judgment_request_id": "request",
            "approved_snapshot_sha": "a" * 40,
            "approved_snapshot_ref": snapshot_ref,
            "judge_snapshot_cwd": judge.cwd,
        }
        self.workflow.save(update_fields=["step", "state", "updated_at"])

        action = self.record_and_finish_judge(
            {"confidence": "medium", "feedback": "Needs another test."},
            approved=False,
        )

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.cleanup_candidate_cwds, (judge.cwd,))
        self.assertEqual(
            action.release_snapshot_refs,
            ((self.project.repo_path, snapshot_ref),),
        )
        self.workflow.refresh_from_db()
        self.assertNotIn("approved_snapshot_sha", self.workflow.state)
        self.assertNotIn("approved_snapshot_ref", self.workflow.state)
        self.assertNotIn("judge_snapshot_cwd", self.workflow.state)


    @patch("hitch.main.workflows.autonomous_goals.codex_pool.interrupt_instance")
    def test_candidate_failure_force_stops_judge_and_blocks(
        self, mock_interrupt: MagicMock
    ) -> None:
        snapshot_ref = (
            f"refs/hitch/autonomous-goals/{self.workflow.pk}/"
            "0123456789abcdef0123456789abcdef"
        )
        self.workflow.step = system_agents.STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING
        self.workflow.state = {
            **self.workflow.state,
            "candidate": _candidate_arguments(),
            "judgment_request_id": "request",
            "approved_snapshot_sha": "a" * 40,
            "approved_snapshot_ref": snapshot_ref,
            "judge_snapshot_cwd": "/judge-snapshot",
        }
        self.workflow.save(update_fields=["step", "state", "updated_at"])
        candidate_instance = CodexInstance.objects.create(
            pid=1,
            thread_id=self.candidate.thread_id,
            cwd=self.candidate.cwd,
            prompt="candidate",
            status=CodexInstance.STATUS_FAILED,
            error="cancelled while waiting",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=self.workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        candidate_run = SystemAgentRun.objects.create(
            workflow=self.workflow,
            instance=candidate_instance,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id=candidate_instance.thread_id,
        )
        judge_instance = CodexInstance.objects.create(
            pid=2,
            thread_id="judge-thread",
            cwd="/judge-snapshot",
            prompt="judge",
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=self.workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
        )
        SystemAgentRun.objects.create(
            workflow=self.workflow,
            instance=judge_instance,
            agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
            thread_id=judge_instance.thread_id,
            status=SystemAgentRun.STATUS_RUNNING,
        )
        judge_instance.status = CodexInstance.STATUS_FAILED
        mock_interrupt.return_value = judge_instance

        action = autonomous_goals._handle_tool_protocol_agent_finished_locked(
            candidate_instance,
            candidate_run,
            self.workflow,
            self.goal,
            "Candidate cancelled.",
            tokens_used=100,
            token_delta=100,
        )

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.cleanup_candidate_cwds, ("/judge-snapshot",))
        self.assertEqual(
            action.release_snapshot_refs,
            ((self.project.repo_path, snapshot_ref),),
        )
        self.workflow.refresh_from_db()
        self.assertEqual(self.workflow.status, SystemWorkflow.STATUS_BLOCKED)
        mock_interrupt.assert_called_once_with(
            judge_instance.pk,
            expected_thread_id=judge_instance.thread_id,
            force=True,
            error="autonomous goal worker failed: cancelled while waiting",
        )

    def test_failed_turn_preserves_recorded_approval(
        self,
    ) -> None:
        snapshot_ref = (
            f"refs/hitch/autonomous-goals/{self.workflow.pk}/"
            "0123456789abcdef0123456789abcdef"
        )
        self.workflow.state = {
            **self.workflow.state,
            "proposal_budget": 1_000,
            "candidate_terminal": "approved",
            "candidate": _candidate_arguments(),
            "judgment": {
                "verdict": "approve",
                "confidence": "high",
                "feedback": "Approved.",
            },
            "approved_snapshot_sha": "a" * 40,
            "approved_snapshot_ref": snapshot_ref,
        }
        self.workflow.save(update_fields=["state", "updated_at"])
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id=self.candidate.thread_id,
            cwd=self.candidate.cwd,
            prompt="candidate",
            status=CodexInstance.STATUS_FAILED,
            error="worker failed after approval",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=self.workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        run = SystemAgentRun.objects.create(
            workflow=self.workflow,
            instance=instance,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id=instance.thread_id,
        )

        action = autonomous_goals._handle_tool_protocol_agent_finished_locked(
            instance,
            run,
            self.workflow,
            self.goal,
            "Candidate failed.",
            tokens_used=100,
            token_delta=100,
        )

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.kind, "")
        self.assertEqual(action.release_snapshot_refs, ())
        self.workflow.refresh_from_db()
        run.refresh_from_db()
        proposal = ProposedSession.objects.get(source_workflow=self.workflow)
        self.assertEqual(self.workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(
            self.workflow.step, system_agents.STEP_AUTONOMOUS_GOAL_PROPOSED
        )
        self.assertEqual(run.status, SystemAgentRun.STATUS_COMPLETED)
        self.assertEqual(
            proposal.outcome_metadata["approved_snapshot_ref"], snapshot_ref
        )
        self.assertIn(_candidate_arguments()["summary"], proposal.summary)

    def test_failed_initial_judge_spawn_does_not_revive_blocked_workflow(
        self,
    ) -> None:
        self.workflow.step = system_agents.STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING
        self.workflow.state = {
            **self.workflow.state,
            "judgment_attempts": 1,
            "judgment_request_id": "request",
        }
        self.workflow.status = SystemWorkflow.STATUS_BLOCKED
        self.workflow.step = system_agents.STEP_BLOCKED
        self.workflow.state = {**self.workflow.state, "error": "goal deleted"}
        self.workflow.save(
            update_fields=["status", "step", "state", "updated_at"]
        )

        autonomous_goals._restore_failed_judge_spawn(
            workflow_id=self.workflow.pk,
            request_id="request",
            attempts_before=0,
        )

        self.workflow.refresh_from_db()
        self.assertEqual(self.workflow.status, SystemWorkflow.STATUS_BLOCKED)
        self.assertEqual(self.workflow.step, system_agents.STEP_BLOCKED)
        self.assertEqual(self.workflow.state["judgment_attempts"], 1)

    @patch(
        "hitch.main.workflows.autonomous_goals._release_autonomous_goal_snapshot_ref"
    )
    @patch("hitch.main.workflows.autonomous_goals._cleanup_autonomous_goal_candidate_cwd")
    @patch(
        "hitch.main.workflows.autonomous_goals._spawn_autonomous_goal_judge_protocol_recovery_run",
        side_effect=RuntimeError("spawn failed"),
    )
    def test_failed_judge_recovery_returns_denial_and_releases_snapshot(
        self,
        _mock_spawn: MagicMock,
        mock_cleanup: MagicMock,
        mock_release: MagicMock,
    ) -> None:
        snapshot_ref = (
            f"refs/hitch/autonomous-goals/{self.workflow.pk}/"
            "0123456789abcdef0123456789abcdef"
        )
        self.workflow.step = system_agents.STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING
        self.workflow.state = {
            **self.workflow.state,
            "judgment_request_id": "request",
            "approved_snapshot_sha": "a" * 40,
            "approved_snapshot_ref": snapshot_ref,
            "judge_snapshot_cwd": "/judge-snapshot",
        }
        self.workflow.save(update_fields=["step", "state", "updated_at"])

        autonomous_goals._spawn_autonomous_goal_judge_protocol_recovery_or_block(
            self.workflow, self.goal
        )

        self.workflow.refresh_from_db()
        self.assertEqual(
            self.workflow.step, system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING
        )
        self.assertEqual(self.workflow.state["judgment_result_id"], "request")
        self.assertEqual(self.workflow.state["judgment"]["verdict"], "deny")
        self.assertNotIn("approved_snapshot_ref", self.workflow.state)
        mock_cleanup.assert_called_once_with("/judge-snapshot")
        mock_release.assert_called_once_with(self.project.repo_path, snapshot_ref)

    @patch(
        "hitch.main.workflows.autonomous_goals._spawn_autonomous_goal_judge_protocol_recovery_run"
    )
    def test_failed_judge_recovery_does_not_overwrite_concurrent_block(
        self, mock_spawn: MagicMock
    ) -> None:
        self.workflow.step = system_agents.STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING
        self.workflow.state = {
            **self.workflow.state,
            "judgment_request_id": "request",
        }
        self.workflow.save(update_fields=["step", "state", "updated_at"])

        def block_then_fail(*_args: Any, **_kwargs: Any) -> None:
            SystemWorkflow.objects.filter(pk=self.workflow.pk).update(
                status=SystemWorkflow.STATUS_BLOCKED,
                step=system_agents.STEP_BLOCKED,
                state={**self.workflow.state, "error": "goal deleted"},
            )
            raise RuntimeError("spawn failed")

        mock_spawn.side_effect = block_then_fail

        autonomous_goals._spawn_autonomous_goal_judge_protocol_recovery_or_block(
            self.workflow, self.goal
        )

        self.workflow.refresh_from_db()
        self.assertEqual(self.workflow.status, SystemWorkflow.STATUS_BLOCKED)
        self.assertEqual(self.workflow.step, system_agents.STEP_BLOCKED)
        self.assertNotIn("judgment_result_id", self.workflow.state)

    def test_failed_turn_preserves_recorded_no_proposal(self) -> None:
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id=self.candidate.thread_id,
            cwd="/candidate",
            prompt="candidate",
            status=CodexInstance.STATUS_FAILED,
            error="worker failed after no_proposal",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=self.workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        run = SystemAgentRun.objects.create(
            workflow_id=self.workflow.pk,
            instance=instance,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id=self.candidate.thread_id,
        )
        self.call_candidate(
            "no_proposal",
            {"reason": "No worthwhile candidate remains."},
        )
        self.workflow.refresh_from_db()

        action = autonomous_goals._handle_tool_protocol_agent_finished_locked(
            instance,
            run,
            self.workflow,
            self.goal,
            "Finished via no_proposal.",
            tokens_used=100,
            token_delta=100,
        )

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.cleanup_candidate_cwds, (self.candidate.cwd,))
        self.workflow.refresh_from_db()
        run.refresh_from_db()
        self.assertEqual(self.workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(
            self.workflow.step, system_agents.STEP_AUTONOMOUS_GOAL_SKIPPED
        )
        self.assertEqual(
            ProposedSession.objects.get(source_workflow=self.workflow).inbox_kind,
            ProposedSession.INBOX_KIND_NOTICE,
        )
        self.assertEqual(run.status, SystemAgentRun.STATUS_COMPLETED)

    def test_no_proposal_cleans_abandoned_stack_candidate_worktree(self) -> None:
        prior_candidate = SessionMetadata.objects.create(
            thread_id="prior-candidate",
            cwd="/prior-candidate",
            project=self.project,
            is_hidden_system_session=True,
        )
        prior = ProposedSession.objects.create(
            autonomous_goal=self.goal,
            source_workflow=self.workflow,
            title="Prior approved stack",
            candidate_session=prior_candidate,
            outcome_status=ProposedSession.OUTCOME_DISMISSED,
            outcome_metadata={"stacked_diff_hidden_until_complete": True},
        )
        self.workflow.state = {
            **self.workflow.state,
            "proposal_id": prior.pk,
            "stacked_diff_depth": 2,
            "stacked_diff_iteration": 2,
        }
        self.workflow.save(update_fields=["state", "updated_at"])
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id=self.candidate.thread_id,
            cwd=self.candidate.cwd,
            prompt="candidate",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=self.workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        run = SystemAgentRun.objects.create(
            workflow=self.workflow,
            instance=instance,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id=self.candidate.thread_id,
        )
        self.call_candidate(
            "no_proposal", {"reason": "No stronger second stack remains."}
        )
        self.workflow.refresh_from_db()

        action = autonomous_goals._handle_tool_protocol_agent_finished_locked(
            instance,
            run,
            self.workflow,
            self.goal,
            "Finished via no_proposal.",
            tokens_used=100,
            token_delta=100,
        )

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.cleanup_candidate_cwds, (self.candidate.cwd,))
        prior.refresh_from_db()
        self.assertEqual(prior.outcome_status, ProposedSession.OUTCOME_UNSET)
        self.workflow.refresh_from_db()
        self.assertEqual(
            self.workflow.step, system_agents.STEP_AUTONOMOUS_GOAL_PROPOSED
        )

    @patch("hitch.main.workflows.autonomous_goals.codex_pool.spawn_turn")
    def test_candidate_protocol_omission_resumes_same_hidden_thread(
        self, mock_spawn: MagicMock
    ) -> None:
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id=self.candidate.thread_id,
            cwd="/candidate",
            prompt="candidate",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=self.workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        run = SystemAgentRun.objects.create(
            workflow=self.workflow,
            instance=instance,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id=self.candidate.thread_id,
        )
        action = autonomous_goals._handle_tool_protocol_agent_finished_locked(
            instance,
            run,
            self.workflow,
            self.goal,
            "Stopped without a terminal tool.",
            tokens_used=100,
            token_delta=100,
        )
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(
            action.kind, autonomous_goals._AUTONOMOUS_GOAL_PROTOCOL_RECOVERY_ACTION
        )
        self.workflow.refresh_from_db()
        self.assertEqual(self.workflow.state["protocol_recoveries"], 1)
        resumed_instance = CodexInstance.objects.create(
            pid=2,
            thread_id=self.candidate.thread_id,
            cwd="/candidate",
            prompt="resume",
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=self.workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        mock_spawn.return_value = resumed_instance

        autonomous_goals._spawn_autonomous_goal_candidate_protocol_recovery_run(
            self.workflow, self.goal
        )

        self.assertEqual(
            mock_spawn.call_args.kwargs["thread_id"], self.candidate.thread_id
        )

    def test_approve_rejects_confidence_below_threshold(self) -> None:
        judge = SessionMetadata.objects.create(
            thread_id="judge-thread",
            cwd="/candidate",
            project=self.project,
            is_hidden_system_session=True,
        )
        self.workflow.step = system_agents.STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING
        self.workflow.state = {
            **self.workflow.state,
            "judge_session_id": judge.pk,
            "judgment_request_id": "request",
        }
        self.workflow.save(update_fields=["step", "state", "updated_at"])

        response = handle_dynamic_tool_call(
            {
                "namespace": "hitch",
                "tool": "approve",
                "arguments": {"confidence": "medium"},
            },
            self.judge_context(),
        )

        self.assertFalse(response["success"])
        self.assertIn("below", response["contentItems"][0]["text"])


class ApprovedSnapshotMetadataTests(TestCase):
    def test_proposal_records_approved_snapshot(self) -> None:
        project = _make_project(repo_path="/repo")
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Goal",
            goal="Improve things",
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate",
            cwd="/candidate",
            project=project,
            is_hidden_system_session=True,
        )
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_AUTONOMOUS_GOAL_RUN,
            main_thread_id="autonomous-goal:snapshot",
            cwd="/repo",
            step=system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
            state={
                "autonomous_goal_id": goal.pk,
                "tool_protocol": True,
                "candidate_session_id": candidate.pk,
                "approved_snapshot_sha": "b" * 40,
                "approved_snapshot_ref": (
                    "refs/hitch/autonomous-goals/1/"
                    "0123456789abcdef0123456789abcdef"
                ),
            },
        )

        proposal = autonomous_goals._create_autonomous_goal_proposal(
            workflow,
            goal,
            {"title": "Candidate", "relevant_files": []},
            {"confidence": "high", "feedback": ""},
        )

        self.assertEqual(proposal.outcome_metadata["approved_snapshot_sha"], "b" * 40)
        self.assertIs(
            proposal.outcome_metadata["autonomous_goal_tool_protocol"], True
        )
        self.assertEqual(
            proposal.outcome_metadata["approved_snapshot_ref"],
            "refs/hitch/autonomous-goals/1/0123456789abcdef0123456789abcdef",
        )
