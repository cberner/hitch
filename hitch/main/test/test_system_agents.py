import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from django.db import IntegrityError, transaction
from django.test import TestCase
from openai_codex.generated.v2_all import ThreadSource

from hitch.main import system_agents
from hitch.main.models import CodexInstance, SystemAgentRun, SystemWorkflow


def _instance(
    *,
    thread_id: str = "thread-1",
    purpose: str = CodexInstance.PURPOSE_USER,
    workflow_id: int | None = None,
    events_path: str = "/dev/null",
    status: str = CodexInstance.STATUS_COMPLETED,
    agent_kind: str = "",
) -> CodexInstance:
    return CodexInstance.objects.create(
        pid=1,
        thread_id=thread_id,
        cwd="/repo",
        prompt="prompt",
        events_path=events_path,
        status=status,
        purpose=purpose,
        workflow_id=workflow_id,
        agent_kind=agent_kind,
    )


class PrQaWorkflowTests(TestCase):
    @patch("hitch.main.system_agents.build_worktree_diff_text", return_value="diff --git")
    @patch("hitch.main.system_agents.codex_pool.spawn_new_session")
    def test_pr_qa_workflow_starts_hidden_subagent_thread(
        self, mock_spawn: MagicMock, _mock_diff: MagicMock
    ) -> None:
        mock_spawn.return_value = _instance(
            thread_id="qa-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
        )

        workflow = system_agents.start_pr_qa_workflow(
            main_thread_id="main-thread",
            cwd="/repo",
            sandbox_policy="workspaceWrite",
            approval_mode="prompt_user",
            model="gpt-5.4",
            reasoning_effort="high",
            developer_instructions="Use repo conventions.",
            enable_memories=True,
            initial_user_message_index=2,
        )

        self.assertEqual(workflow.step, "qa_running")
        mock_spawn.assert_called_once()
        kwargs = mock_spawn.call_args.kwargs
        self.assertEqual(kwargs["thread_source"], ThreadSource.subagent)
        self.assertEqual(kwargs["purpose"], CodexInstance.PURPOSE_SYSTEM_AGENT)
        self.assertEqual(kwargs["approval_mode"], system_agents.SYSTEM_AGENT_APPROVAL_MODE)
        self.assertEqual(kwargs["sandbox_policy"], "workspaceWrite")
        self.assertEqual(kwargs["model"], "gpt-5.4")
        self.assertEqual(kwargs["reasoning_effort"], "high")
        self.assertEqual(kwargs["developer_instructions"], "Use repo conventions.")
        self.assertTrue(kwargs["enable_memories"])
        self.assertEqual(kwargs["workflow_id"], workflow.pk)
        self.assertEqual(kwargs["agent_kind"], system_agents.PR_QA_AGENT_KIND)
        self.assertEqual(kwargs["display_author"], system_agents.QA_DISPLAY_AUTHOR)
        self.assertIn("output_schema", kwargs)
        self.assertIn("Apply the same review standards as Codex /review", kwargs["prompt"])
        self.assertIn("Do not stop at the first issue", kwargs["prompt"])
        self.assertIn("shortest useful file/line reference", kwargs["prompt"])
        self.assertIn("diff --git", kwargs["prompt"])

        run = SystemAgentRun.objects.get(workflow=workflow)
        self.assertEqual(run.thread_id, "qa-thread")

    @patch("hitch.main.system_agents.codex_pool.spawn_new_session")
    def test_start_returns_existing_running_workflow(self, mock_spawn: MagicMock) -> None:
        existing = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_QA_RUNNING,
        )

        workflow = system_agents.start_pr_qa_workflow(
            main_thread_id="main-thread",
            cwd="/repo",
            sandbox_policy=None,
            approval_mode="auto_review",
        )

        self.assertEqual(workflow, existing)
        mock_spawn.assert_not_called()

    def test_only_one_running_workflow_is_allowed_per_thread_and_kind(self) -> None:
        SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_QA_RUNNING,
        )

        with self.assertRaises(IntegrityError), transaction.atomic():
            SystemWorkflow.objects.create(
                kind=SystemWorkflow.KIND_PR_QA,
                main_thread_id="main-thread",
                cwd="/repo",
                status=SystemWorkflow.STATUS_RUNNING,
                step=system_agents.STEP_QA_RUNNING,
            )

    @patch("hitch.main.system_agents.codex_pool.spawn_turn")
    def test_qa_feedback_spawns_tagged_visible_turn(self, mock_spawn: MagicMock) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step="qa_running",
            state={
                "approval_mode": "prompt_user",
                "sandbox_policy": "workspaceWrite",
                "model": "gpt-5.4",
                "reasoning_effort": "high",
                "developer_instructions": "Use repo conventions.",
                "next_user_message_index": 2,
            },
        )
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as fh:
            fh.write(
                json.dumps(
                    {
                        "method": "item/completed",
                        "payload": {
                            "item": {
                                "id": "a1",
                                "type": "agentMessage",
                                "text": '{"feedback": "Fix it", "lgtm": false}',
                            }
                        },
                    }
                )
                + "\n"
            )
            events_path = fh.name
        self.addCleanup(Path(events_path).unlink, missing_ok=True)
        instance = _instance(
            thread_id="qa-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            events_path=events_path,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            thread_id="qa-thread",
            instance=instance,
        )

        system_agents.on_codex_instance_finished(instance)

        mock_spawn.assert_called_once()
        kwargs = mock_spawn.call_args.kwargs
        self.assertEqual(kwargs["thread_id"], "main-thread")
        self.assertEqual(kwargs["purpose"], CodexInstance.PURPOSE_SYSTEM_FEEDBACK)
        self.assertEqual(kwargs["display_author"], system_agents.QA_DISPLAY_AUTHOR)
        self.assertEqual(kwargs["approval_mode"], "prompt_user")
        self.assertEqual(kwargs["model"], "gpt-5.4")
        self.assertEqual(kwargs["reasoning_effort"], "high")
        self.assertEqual(kwargs["developer_instructions"], "Use repo conventions.")
        self.assertEqual(kwargs["user_message_index"], 2)
        self.assertIn("Feedback from Hitch QA agent", kwargs["prompt"])
        workflow.refresh_from_db()
        self.assertEqual(workflow.step, "feedback_running")
        self.assertEqual(workflow.iteration, 1)
        self.assertEqual(workflow.state["next_user_message_index"], 3)

    @patch("hitch.main.system_agents.codex_pool.spawn_turn")
    def test_qa_lgtm_spawns_pr_prompt(self, mock_spawn: MagicMock) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step="qa_running",
            state={
                "pr_prompt": system_agents.PR_SLASH_PROMPT,
                "model": "gpt-5.4",
                "reasoning_effort": "high",
                "developer_instructions": "Use repo conventions.",
                "next_user_message_index": 4,
            },
        )
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as fh:
            fh.write(
                json.dumps(
                    {
                        "method": "item/completed",
                        "payload": {
                            "item": {
                                "id": "a1",
                                "type": "agentMessage",
                                "text": '{"feedback": "Looks good", "lgtm": true}',
                            }
                        },
                    }
                )
                + "\n"
            )
            events_path = fh.name
        self.addCleanup(Path(events_path).unlink, missing_ok=True)
        instance = _instance(
            thread_id="qa-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            events_path=events_path,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            thread_id="qa-thread",
            instance=instance,
        )

        system_agents.on_codex_instance_finished(instance)

        mock_spawn.assert_called_once()
        self.assertEqual(mock_spawn.call_args.kwargs["prompt"], system_agents.PR_SLASH_PROMPT)
        self.assertEqual(mock_spawn.call_args.kwargs["model"], "gpt-5.4")
        self.assertEqual(mock_spawn.call_args.kwargs["reasoning_effort"], "high")
        self.assertEqual(
            mock_spawn.call_args.kwargs["developer_instructions"],
            "Use repo conventions.",
        )
        self.assertEqual(mock_spawn.call_args.kwargs["user_message_index"], 4)
        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(workflow.step, "pr_prompt_spawned")
        self.assertEqual(workflow.state["next_user_message_index"], 5)

    @patch("hitch.main.system_agents.codex_pool.spawn_turn")
    def test_qa_completion_recovers_when_run_row_does_not_exist_yet(
        self, mock_spawn: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step="qa_running",
            state={"pr_prompt": system_agents.PR_SLASH_PROMPT},
        )
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as fh:
            fh.write(
                json.dumps(
                    {
                        "method": "item/completed",
                        "payload": {
                            "item": {
                                "id": "a1",
                                "type": "agentMessage",
                                "text": '{"feedback": "Looks good", "lgtm": true}',
                            }
                        },
                    }
                )
                + "\n"
            )
            events_path = fh.name
        self.addCleanup(Path(events_path).unlink, missing_ok=True)
        instance = _instance(
            thread_id="qa-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            events_path=events_path,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
        )

        system_agents.on_codex_instance_finished(instance)

        run = SystemAgentRun.objects.get(instance=instance)
        self.assertEqual(run.status, SystemAgentRun.STATUS_COMPLETED)
        mock_spawn.assert_called_once()
        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)

    @patch("hitch.main.system_agents.codex_pool.spawn_turn")
    def test_invalid_qa_output_blocks_workflow_and_surfaces_failure(
        self, mock_spawn: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step="qa_running",
            state={"next_user_message_index": 1},
        )
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as fh:
            fh.write(
                json.dumps(
                    {
                        "method": "item/completed",
                        "payload": {
                            "item": {
                                "id": "a1",
                                "type": "agentMessage",
                                "text": "not json",
                            }
                        },
                    }
                )
                + "\n"
            )
            events_path = fh.name
        self.addCleanup(Path(events_path).unlink, missing_ok=True)
        instance = _instance(
            thread_id="qa-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            events_path=events_path,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            thread_id="qa-thread",
            instance=instance,
        )

        system_agents.on_codex_instance_finished(instance)

        run.refresh_from_db()
        workflow.refresh_from_db()
        self.assertEqual(run.status, SystemAgentRun.STATUS_FAILED)
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
        self.assertTrue(workflow.state["failure_surfaced"])
        mock_spawn.assert_called_once()
        kwargs = mock_spawn.call_args.kwargs
        self.assertEqual(kwargs["thread_id"], "main-thread")
        self.assertEqual(kwargs["purpose"], CodexInstance.PURPOSE_SYSTEM_FEEDBACK)
        self.assertEqual(kwargs["display_author"], system_agents.QA_DISPLAY_AUTHOR)
        self.assertEqual(kwargs["user_message_index"], 1)
        self.assertIn("QA output was not valid JSON", kwargs["prompt"])

    @patch("hitch.main.system_agents.codex_pool.spawn_turn")
    @patch("hitch.main.system_agents.codex_pool.interrupt_instance")
    def test_stop_active_workflow_interrupts_hidden_run_and_blocks(
        self, mock_interrupt: MagicMock, mock_spawn: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step="qa_running",
            state={"next_user_message_index": 0},
        )
        instance = _instance(
            thread_id="qa-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            status=CodexInstance.STATUS_RUNNING,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            thread_id="qa-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )
        mock_interrupt.return_value = instance

        stopped = system_agents.stop_active_workflow("main-thread")

        self.assertTrue(stopped)
        mock_interrupt.assert_called_once_with(
            instance.pk, expected_thread_id="qa-thread"
        )
        run.refresh_from_db()
        workflow.refresh_from_db()
        self.assertEqual(run.status, SystemAgentRun.STATUS_FAILED)
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
        mock_spawn.assert_called_once()

    @patch("hitch.main.system_agents.codex_pool.spawn_turn")
    def test_stop_active_workflow_returns_false_without_hidden_run(
        self, mock_spawn: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_FEEDBACK_RUNNING,
        )

        stopped = system_agents.stop_active_workflow("main-thread")

        self.assertFalse(stopped)
        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_RUNNING)
        mock_spawn.assert_not_called()

    @patch("hitch.main.system_agents.codex_pool.spawn_turn")
    @patch("hitch.main.system_agents.codex_pool.interrupt_instance", return_value=None)
    def test_stop_active_workflow_leaves_running_when_interrupt_fails(
        self, mock_interrupt: MagicMock, mock_spawn: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_QA_RUNNING,
        )
        instance = _instance(
            thread_id="qa-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            status=CodexInstance.STATUS_RUNNING,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            thread_id="qa-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )

        stopped = system_agents.stop_active_workflow("main-thread")

        self.assertFalse(stopped)
        mock_interrupt.assert_called_once_with(
            instance.pk, expected_thread_id="qa-thread"
        )
        run.refresh_from_db()
        workflow.refresh_from_db()
        self.assertEqual(run.status, SystemAgentRun.STATUS_RUNNING)
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_RUNNING)
        mock_spawn.assert_not_called()

    def test_final_agent_text_ignores_commentary_messages(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as fh:
            fh.write(
                json.dumps(
                    {
                        "method": "item/completed",
                        "payload": {
                            "item": {
                                "id": "commentary",
                                "type": "agentMessage",
                                "phase": "commentary",
                                "text": "thinking out loud",
                            }
                        },
                    }
                )
                + "\n"
            )
            fh.write(
                json.dumps(
                    {
                        "method": "item/completed",
                        "payload": {
                            "item": {
                                "id": "final",
                                "type": "agentMessage",
                                "text": '{"feedback": "Done", "lgtm": true}',
                            }
                        },
                    }
                )
                + "\n"
            )
            events_path = fh.name
        self.addCleanup(Path(events_path).unlink, missing_ok=True)

        self.assertEqual(
            system_agents._final_agent_text(events_path),
            '{"feedback": "Done", "lgtm": true}',
        )
