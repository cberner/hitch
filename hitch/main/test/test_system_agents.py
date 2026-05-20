import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from django.db import IntegrityError, transaction
from django.test import TestCase
from openai_codex.generated.v2_all import ThreadSource

from hitch.main import system_agents
from hitch.main.models import (
    CodexInstance,
    KeyResult,
    Objective,
    Project,
    ProposedTask,
    SystemAgentRun,
    SystemWorkflow,
)


def _instance(
    *,
    thread_id: str = "thread-1",
    purpose: str = CodexInstance.PURPOSE_USER,
    workflow_id: int | None = None,
    events_path: str = "/dev/null",
    status: str = CodexInstance.STATUS_COMPLETED,
    agent_kind: str = "",
    auto_pr_enabled: bool = False,
    plan_mode: bool = False,
    model: str = "",
    reasoning_effort: str = "",
    sandbox_policy: str = "",
    approval_mode: str = "",
    developer_instructions: str = "",
    enable_memories: bool = False,
    user_message_index: int | None = None,
    error: str = "",
) -> CodexInstance:
    return CodexInstance.objects.create(
        pid=1,
        thread_id=thread_id,
        cwd="/repo",
        prompt="prompt",
        developer_instructions=developer_instructions,
        enable_memories=enable_memories,
        model=model,
        reasoning_effort=reasoning_effort,
        sandbox_policy=sandbox_policy,
        approval_mode=approval_mode,
        plan_mode=plan_mode,
        auto_pr_enabled=auto_pr_enabled,
        events_path=events_path,
        status=status,
        purpose=purpose,
        workflow_id=workflow_id,
        agent_kind=agent_kind,
        user_message_index=user_message_index,
        error=error,
    )


class OkrTaskGenerationWorkflowTests(TestCase):
    @patch("hitch.main.system_agents.codex_pool.spawn_new_session")
    def test_workflow_starts_hidden_task_planning_thread(
        self, mock_spawn: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        objective = Objective.objects.create(
            project=project,
            title="Improve planning",
            description="Make OKR execution concrete.",
        )
        key_result = KeyResult.objects.create(
            objective=objective,
            title="Generate task proposals",
            description="Produce useful task lists.",
            work_instructions="Prefer small increments.",
        )
        KeyResult.objects.create(
            objective=objective,
            title="Keep existing OKR rendering stable",
        )
        ProposedTask.objects.create(
            key_result=key_result,
            title="Overbroad generated task",
            description="Do everything.",
            outcome_status=ProposedTask.OUTCOME_REJECTED,
            outcome_notes="Too broad.",
        )
        mock_spawn.return_value = _instance(
            thread_id="task-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
        )

        workflow = system_agents.start_okr_task_generation_workflow(
            key_result=key_result
        )

        self.assertEqual(workflow.step, system_agents.STEP_OKR_TASKS_RUNNING)
        mock_spawn.assert_called_once()
        kwargs = mock_spawn.call_args.kwargs
        self.assertEqual(kwargs["thread_source"], ThreadSource.subagent)
        self.assertEqual(kwargs["purpose"], CodexInstance.PURPOSE_SYSTEM_AGENT)
        self.assertEqual(kwargs["approval_mode"], system_agents.SYSTEM_AGENT_APPROVAL_MODE)
        self.assertEqual(kwargs["workflow_id"], workflow.pk)
        self.assertEqual(kwargs["agent_kind"], system_agents.OKR_TASK_AGENT_KIND)
        self.assertEqual(kwargs["display_author"], system_agents.OKR_TASK_DISPLAY_AUTHOR)
        self.assertIn("output_schema", kwargs)
        self.assertIn("senior software engineering manager", kwargs["prompt"])
        self.assertIn("Generate task proposals", kwargs["prompt"])
        self.assertIn("Keep existing OKR rendering stable", kwargs["prompt"])
        self.assertIn("Too broad.", kwargs["prompt"])
        self.assertIn("small, but logically consistent pieces", kwargs["prompt"])
        self.assertIn("tagging system", kwargs["prompt"])
        self.assertIn("basic comment implementation", kwargs["prompt"])
        self.assertIn("rich text to the comments", kwargs["prompt"])

        run = SystemAgentRun.objects.get(workflow=workflow)
        self.assertEqual(run.thread_id, "task-thread")

    @patch("hitch.main.system_agents.codex_pool.spawn_new_session")
    def test_start_returns_existing_running_task_workflow(
        self, mock_spawn: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        objective = Objective.objects.create(project=project, title="Improve planning")
        key_result = KeyResult.objects.create(objective=objective, title="Draft plan")
        existing = SystemWorkflow.objects.create(
            kind=system_agents.OKR_TASK_AGENT_KIND,
            main_thread_id=system_agents._okr_task_main_thread_id(key_result.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_OKR_TASKS_RUNNING,
        )

        workflow = system_agents.start_okr_task_generation_workflow(
            key_result=key_result
        )

        self.assertEqual(workflow, existing)
        mock_spawn.assert_not_called()

    @patch("hitch.main.system_agents._spawn_okr_task_generation_run")
    def test_start_blocks_workflow_when_task_agent_spawn_fails(
        self, mock_spawn_run: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        objective = Objective.objects.create(project=project, title="Improve planning")
        key_result = KeyResult.objects.create(objective=objective, title="Draft plan")
        mock_spawn_run.side_effect = RuntimeError("no worker")

        workflow = system_agents.start_okr_task_generation_workflow(
            key_result=key_result
        )

        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
        self.assertEqual(workflow.step, system_agents.STEP_BLOCKED)
        self.assertIn("failed to start task planning agent", workflow.state["error"])
        self.assertNotIn("failure_surfaced", workflow.state)

    @patch("hitch.main.system_agents.codex_pool.spawn_new_session")
    @patch("hitch.main.system_agents.codex_pool.events_dir")
    @patch.object(system_agents, "_OKR_TASK_INLINE_CONTEXT_CHARS", 1)
    def test_task_agent_spawn_records_overflow_context_files(
        self, mock_events_dir: MagicMock, mock_spawn: MagicMock
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_events_dir.return_value = Path(tmpdir)
            project = Project.objects.create(name="Hitch", repo_path="/repo")
            objective = Objective.objects.create(project=project, title="Improve planning")
            key_result = KeyResult.objects.create(objective=objective, title="Draft plan")
            sibling = KeyResult.objects.create(objective=objective, title="Sibling KR")
            ProposedTask.objects.create(
                key_result=sibling,
                title="Sibling task",
                description="Overflow context.",
            )
            mock_spawn.return_value = _instance(
                thread_id="task-thread",
                purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            )

            workflow = system_agents.start_okr_task_generation_workflow(
                key_result=key_result
            )

            workflow.refresh_from_db()
            self.assertEqual(len(workflow.state["context_files"]), 1)
            self.assertIn("prior_tasks.txt", workflow.state["context_files"][0])
            self.assertIn(
                workflow.state["context_files"][0],
                mock_spawn.call_args.kwargs["prompt"],
            )

    def test_task_agent_completion_saves_proposed_tasks(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        objective = Objective.objects.create(project=project, title="Improve planning")
        key_result = KeyResult.objects.create(objective=objective, title="Draft plan")
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.OKR_TASK_AGENT_KIND,
            main_thread_id=system_agents._okr_task_main_thread_id(key_result.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_OKR_TASKS_RUNNING,
            state={"key_result_id": key_result.pk},
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
                                "text": json.dumps(
                                    {
                                        "tasks": [
                                            {
                                                "title": "Add ProposedTask model",
                                                "description": "Persist generated tasks.",
                                                "success_criteria": "Tasks are queryable.",
                                                "rationale": "Planning needs storage.",
                                            },
                                            {
                                                "title": "Render tasks on OKR page",
                                                "description": "Show tasks under KRs.",
                                                "success_criteria": "The page lists tasks.",
                                                "rationale": "Users need visibility.",
                                            },
                                        ]
                                    }
                                ),
                            }
                        },
                    }
                )
                + "\n"
            )
            events_path = fh.name
        self.addCleanup(Path(events_path).unlink, missing_ok=True)
        instance = _instance(
            thread_id="task-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            events_path=events_path,
            agent_kind=system_agents.OKR_TASK_AGENT_KIND,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.OKR_TASK_AGENT_KIND,
            thread_id="task-thread",
            instance=instance,
        )

        system_agents.on_codex_instance_finished(instance)

        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(workflow.step, system_agents.STEP_OKR_TASKS_SAVED)
        self.assertEqual(workflow.state["saved_task_count"], 2)
        self.assertEqual(
            list(key_result.proposed_tasks.values_list("title", flat=True)),
            ["Add ProposedTask model", "Render tasks on OKR page"],
        )

    def test_invalid_task_agent_output_blocks_without_visible_feedback_turn(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        objective = Objective.objects.create(project=project, title="Improve planning")
        key_result = KeyResult.objects.create(objective=objective, title="Draft plan")
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.OKR_TASK_AGENT_KIND,
            main_thread_id=system_agents._okr_task_main_thread_id(key_result.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_OKR_TASKS_RUNNING,
            state={"key_result_id": key_result.pk},
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
            thread_id="task-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            events_path=events_path,
            agent_kind=system_agents.OKR_TASK_AGENT_KIND,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.OKR_TASK_AGENT_KIND,
            thread_id="task-thread",
            instance=instance,
        )

        system_agents.on_codex_instance_finished(instance)

        run.refresh_from_db()
        workflow.refresh_from_db()
        self.assertEqual(run.status, SystemAgentRun.STATUS_FAILED)
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
        self.assertEqual(ProposedTask.objects.count(), 0)
        self.assertNotIn("failure_surfaced", workflow.state)

    def test_task_agent_worker_failure_blocks_without_visible_feedback_turn(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        objective = Objective.objects.create(project=project, title="Improve planning")
        key_result = KeyResult.objects.create(objective=objective, title="Draft plan")
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.OKR_TASK_AGENT_KIND,
            main_thread_id=system_agents._okr_task_main_thread_id(key_result.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_OKR_TASKS_RUNNING,
            state={"key_result_id": key_result.pk},
        )
        instance = _instance(
            thread_id="task-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            status=CodexInstance.STATUS_FAILED,
            error="worker crashed",
            agent_kind=system_agents.OKR_TASK_AGENT_KIND,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.OKR_TASK_AGENT_KIND,
            thread_id="task-thread",
            instance=instance,
        )

        system_agents.on_codex_instance_finished(instance)

        run.refresh_from_db()
        workflow.refresh_from_db()
        self.assertEqual(run.status, SystemAgentRun.STATUS_FAILED)
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
        self.assertIn("worker crashed", workflow.state["error"])
        self.assertNotIn("failure_surfaced", workflow.state)

    def test_task_agent_completion_ignores_non_running_workflow(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        objective = Objective.objects.create(project=project, title="Improve planning")
        key_result = KeyResult.objects.create(objective=objective, title="Draft plan")
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.OKR_TASK_AGENT_KIND,
            main_thread_id=system_agents._okr_task_main_thread_id(key_result.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_COMPLETED,
            step=system_agents.STEP_OKR_TASKS_SAVED,
            state={"key_result_id": key_result.pk},
        )
        instance = _instance(
            thread_id="task-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.OKR_TASK_AGENT_KIND,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.OKR_TASK_AGENT_KIND,
            thread_id="task-thread",
            instance=instance,
        )

        system_agents.on_codex_instance_finished(instance)

        run.refresh_from_db()
        self.assertEqual(run.status, SystemAgentRun.STATUS_STARTING)
        self.assertEqual(ProposedTask.objects.count(), 0)

    def test_task_agent_completion_blocks_when_key_result_is_missing(self) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.OKR_TASK_AGENT_KIND,
            main_thread_id=system_agents._okr_task_main_thread_id(999),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_OKR_TASKS_RUNNING,
            state={"key_result_id": 999},
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
                                "text": json.dumps(
                                    {
                                        "tasks": [
                                            {
                                                "title": "Plan task",
                                                "description": "",
                                                "success_criteria": "",
                                                "rationale": "",
                                            }
                                        ]
                                    }
                                ),
                            }
                        },
                    }
                )
                + "\n"
            )
            events_path = fh.name
        self.addCleanup(Path(events_path).unlink, missing_ok=True)
        instance = _instance(
            thread_id="task-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            events_path=events_path,
            agent_kind=system_agents.OKR_TASK_AGENT_KIND,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.OKR_TASK_AGENT_KIND,
            thread_id="task-thread",
            instance=instance,
        )

        system_agents.on_codex_instance_finished(instance)

        run.refresh_from_db()
        workflow.refresh_from_db()
        self.assertEqual(run.status, SystemAgentRun.STATUS_FAILED)
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
        self.assertIn("key result no longer exists", workflow.state["error"])

    def test_parse_task_output_accepts_fenced_json_and_rejects_invalid_shapes(self) -> None:
        valid = system_agents._parse_okr_task_output(
            "```json\n"
            '{"tasks": [{"title": " Task ", "description": " Desc ", '
            '"success_criteria": " Done ", "rationale": " Why "}]}'
            "\n```"
        )
        self.assertEqual(
            valid,
            {
                "tasks": [
                    {
                        "title": "Task",
                        "description": "Desc",
                        "success_criteria": "Done",
                        "rationale": "Why",
                    }
                ]
            },
        )
        for raw in (
            "not json",
            "[]",
            '{"tasks": "nope"}',
            '{"tasks": ["nope"]}',
            '{"tasks": [{"title": "", "description": "", '
            '"success_criteria": "", "rationale": ""}]}',
            '{"tasks": [{"title": 1, "description": "", '
            '"success_criteria": "", "rationale": ""}]}',
            '{"tasks": [{"title": "Task", "description": 1, '
            '"success_criteria": "", "rationale": ""}]}',
            '{"tasks": [{"title": "Task", "description": "", '
            '"success_criteria": 1, "rationale": ""}]}',
            '{"tasks": [{"title": "Task", "description": "", '
            '"success_criteria": "", "rationale": 1}]}',
        ):
            with self.subTest(raw=raw):
                self.assertIsNone(system_agents._parse_okr_task_output(raw))

    @patch("hitch.main.system_agents.codex_pool.events_dir")
    @patch.object(system_agents, "_OKR_TASK_INLINE_CONTEXT_CHARS", 1)
    def test_prompt_writes_overflow_task_context_files(
        self, mock_events_dir: MagicMock
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_events_dir.return_value = Path(tmpdir)
            project = Project.objects.create(name="Hitch", repo_path="/repo")
            objective = Objective.objects.create(project=project, title="Improve planning")
            key_result = KeyResult.objects.create(objective=objective, title="Draft plan")
            sibling = KeyResult.objects.create(objective=objective, title="Sibling KR")
            ProposedTask.objects.create(
                key_result=sibling,
                title="Sibling task",
                description="Overflow context.",
            )
            workflow = SystemWorkflow.objects.create(
                kind=system_agents.OKR_TASK_AGENT_KIND,
                main_thread_id=system_agents._okr_task_main_thread_id(key_result.pk),
                cwd="/repo",
                status=SystemWorkflow.STATUS_RUNNING,
            )

            prompt, context_files = system_agents._okr_task_generation_prompt(
                workflow, key_result
            )

            self.assertEqual(len(context_files), 1)
            self.assertIn(context_files[0], prompt)
            self.assertIn("Sibling task", Path(context_files[0]).read_text())

    @patch.object(system_agents, "_OKR_TASK_INLINE_CONTEXT_CHARS", 12)
    def test_split_task_context_caps_important_sections(self) -> None:
        inline, overflow = system_agents._split_task_context(
            [
                (True, "short"),
                (True, "important but too long"),
                (False, "also too long"),
            ]
        )

        self.assertEqual(inline, "short")
        self.assertEqual(overflow, ["important but too long", "also too long"])


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
        self.assertIn("just qa-browser-setup", kwargs["prompt"])
        self.assertIn("Playwright/Chromium", kwargs["prompt"])
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

    @patch("hitch.main.system_agents.build_worktree_diff_text", return_value="diff --git")
    @patch("hitch.main.system_agents.codex_pool.spawn_new_session")
    def test_auto_pr_starts_workflow_after_completed_user_implementation_turn(
        self, mock_spawn: MagicMock, _mock_diff: MagicMock
    ) -> None:
        mock_spawn.return_value = _instance(
            thread_id="qa-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
        )
        instance = _instance(
            thread_id="main-thread",
            auto_pr_enabled=True,
            model="gpt-5.4",
            reasoning_effort="high",
            sandbox_policy="workspaceWrite",
            approval_mode="prompt_user",
            developer_instructions="Use repo conventions.",
            enable_memories=True,
            user_message_index=2,
        )

        system_agents.on_codex_instance_finished(instance)

        instance.refresh_from_db()
        self.assertIsNotNone(instance.auto_pr_triggered_at)
        workflow = SystemWorkflow.objects.get(main_thread_id="main-thread")
        self.assertEqual(workflow.state["sandbox_policy"], "workspaceWrite")
        self.assertEqual(workflow.state["approval_mode"], "prompt_user")
        self.assertEqual(workflow.state["model"], "gpt-5.4")
        self.assertEqual(workflow.state["reasoning_effort"], "high")
        self.assertEqual(workflow.state["developer_instructions"], "Use repo conventions.")
        self.assertTrue(workflow.state["enable_memories"])
        self.assertEqual(workflow.state["next_user_message_index"], 3)
        mock_spawn.assert_called_once()

    @patch("hitch.main.system_agents.start_pr_qa_workflow")
    def test_auto_pr_does_not_stamp_when_workflow_start_fails(
        self, mock_start: MagicMock
    ) -> None:
        mock_start.side_effect = RuntimeError("database unavailable")
        instance = _instance(auto_pr_enabled=True)

        with self.assertRaises(RuntimeError):
            system_agents.on_codex_instance_finished(instance)

        instance.refresh_from_db()
        self.assertIsNone(instance.auto_pr_triggered_at)

    @patch("hitch.main.system_agents.codex_pool.spawn_new_session")
    def test_auto_pr_skips_completed_plan_mode_turn(self, mock_spawn: MagicMock) -> None:
        instance = _instance(auto_pr_enabled=True, plan_mode=True)

        system_agents.on_codex_instance_finished(instance)

        self.assertFalse(SystemWorkflow.objects.exists())
        mock_spawn.assert_not_called()

    @patch("hitch.main.system_agents.codex_pool.spawn_new_session")
    def test_auto_pr_skips_workflow_owned_user_turn(self, mock_spawn: MagicMock) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_COMPLETED,
            step=system_agents.STEP_PR_PROMPT_SPAWNED,
        )
        instance = _instance(
            thread_id="main-thread",
            auto_pr_enabled=True,
            workflow_id=workflow.pk,
        )

        system_agents.on_codex_instance_finished(instance)

        self.assertEqual(SystemWorkflow.objects.count(), 1)
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
        prompt = mock_spawn.call_args.kwargs["prompt"]
        self.assertEqual(prompt, system_agents.PR_SLASH_PROMPT)
        self.assertIn("poll the PR every 2 minutes", prompt)
        self.assertIn("CI status", prompt)
        self.assertIn("thumbs up emoji", prompt)
        self.assertIn("explicit review approval", prompt)
        self.assertIn("merge conflicts", prompt)
        self.assertIn("keep looping until CI, review, and mergeability are all clean", prompt)
        self.assertIn("no results after 30 minutes", prompt)
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
