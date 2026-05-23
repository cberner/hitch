import json
import tempfile
from pathlib import Path
from typing import Any, NamedTuple
from unittest.mock import MagicMock, patch

from django.db import IntegrityError, transaction
from django.test import TestCase
from openai_codex.generated.v2_all import ThreadSource

from hitch.main import demo, system_agents
from hitch.main.models import (
    CodexInstance,
    Project,
    ProposedSession,
    SessionMetadata,
    StandingOrder,
    StandingOrderMemory,
    SystemAgentRun,
    SystemWorkflow,
    UserInputRequest,
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
    qa_panel_enabled: bool = False,
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
        qa_panel_enabled=qa_panel_enabled,
        events_path=events_path,
        status=status,
        purpose=purpose,
        workflow_id=workflow_id,
        agent_kind=agent_kind,
        user_message_index=user_message_index,
        error=error,
    )


def _events_file(test: TestCase, payload: dict[str, object]) -> str:
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as fh:
        fh.write(
            json.dumps(
                {
                    "method": "item/completed",
                    "payload": {
                        "item": {
                            "id": "a1",
                            "type": "agentMessage",
                            "text": json.dumps(payload),
                        }
                    },
                }
            )
            + "\n"
        )
        events_path = fh.name
    test.addCleanup(Path(events_path).unlink, missing_ok=True)
    return events_path


def _assert_response_schema_objects_are_strict(
    test: TestCase, schema: dict[str, Any], *, path: str = "$"
) -> None:
    schema_type = schema.get("type")
    is_object = schema_type == "object" or (
        isinstance(schema_type, list) and "object" in schema_type
    )
    if is_object:
        test.assertIs(schema.get("additionalProperties"), False, path)
        properties = schema.get("properties")
        if isinstance(properties, dict):
            required = schema.get("required")
            if not isinstance(required, list):
                test.fail(path)
            test.assertEqual(set(required), set(properties), path)
            for name, child in properties.items():
                if isinstance(child, dict):
                    _assert_response_schema_objects_are_strict(
                        test, child, path=f"{path}.{name}"
                    )
    items = schema.get("items")
    if isinstance(items, dict):
        _assert_response_schema_objects_are_strict(test, items, path=f"{path}[]")


def _raw_events_file(test: TestCase, events: list[dict[str, object]]) -> str:
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as fh:
        for event in events:
            fh.write(json.dumps(event) + "\n")
        events_path = fh.name
    test.addCleanup(Path(events_path).unlink, missing_ok=True)
    return events_path


def _pr_tool_event(
    *,
    thread_id: str,
    tool: str,
    arguments: dict[str, object] | None = None,
    structured_content: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "method": "item/completed",
        "payload": {
            "threadId": thread_id,
            "item": {
                "type": "mcpToolCall",
                "server": "codex_apps",
                "tool": tool,
                "arguments": arguments or {},
                "result": {
                    "structuredContent": structured_content or {},
                },
            },
        },
    }


class _DesignGateCase(NamedTuple):
    name: str
    current_feedback: str
    expect_gate: bool
    prior_feedback: str | None = None
    prior_lgtm: bool = False
    iteration: int = 1
    expected_categories: tuple[str, ...] = ()
    expected_files: tuple[str, ...] = ()
    prompt_includes: tuple[str, ...] = ()


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
        self.assertEqual(
            workflow.max_iterations, system_agents.PR_QA_WORKFLOW_MAX_ITERATIONS
        )
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


class SpecCriticWorkflowTests(TestCase):
    def test_prompt_classifier_targets_vague_broad_and_high_impact_prompts(self) -> None:
        self.assertTrue(system_agents.spec_critic_should_run("Improve the app"))
        self.assertTrue(
            system_agents.spec_critic_should_run(
                "Implement authentication and permission handling"
            )
        )
        self.assertTrue(
            system_agents.spec_critic_should_run("Change token rotation")
        )
        self.assertFalse(
            system_agents.spec_critic_should_run(
                'Change the settings checkbox label from "Auto-PR" to "Open PR automatically".'
            )
        )
        self.assertFalse(system_agents.spec_critic_should_run("Change tokenizer tests"))
        self.assertFalse(system_agents.spec_critic_should_run("Explain how sessions work"))

    @patch("hitch.main.system_agents.codex_pool.spawn_new_session")
    def test_spec_critic_starts_hidden_specialized_agents(
        self, mock_spawn: MagicMock
    ) -> None:
        def _spawn(**kwargs: Any) -> CodexInstance:
            return _instance(
                thread_id=f"{kwargs['agent_kind']}-thread",
                purpose=kwargs["purpose"],
                status=CodexInstance.STATUS_RUNNING,
                agent_kind=kwargs["agent_kind"],
            )

        mock_spawn.side_effect = _spawn

        workflow = system_agents.start_spec_critic_workflow(
            main_thread_id="main-thread",
            cwd="/repo",
            prompt="Improve onboarding",
            sandbox_policy="workspaceWrite",
            approval_mode="prompt_user",
            model="gpt-5.4",
            reasoning_effort="high",
            developer_instructions="Use repo conventions.",
            enable_memories=True,
            initial_user_message_index=2,
            auto_pr_enabled=True,
            qa_panel_enabled=True,
        )

        self.assertEqual(workflow.kind, system_agents.SPEC_CRITIC_WORKFLOW_KIND)
        self.assertEqual(workflow.step, system_agents.STEP_SPEC_CRITIC_ANALYZING)
        self.assertEqual(mock_spawn.call_count, 3)
        agent_kinds = {call.kwargs["agent_kind"] for call in mock_spawn.call_args_list}
        self.assertEqual(
            agent_kinds,
            {
                system_agents.SPEC_REQUIREMENTS_AGENT_KIND,
                system_agents.SPEC_RISK_AGENT_KIND,
                system_agents.SPEC_TEST_AGENT_KIND,
            },
        )
        for call in mock_spawn.call_args_list:
            self.assertEqual(call.kwargs["thread_source"], ThreadSource.subagent)
            self.assertEqual(call.kwargs["purpose"], CodexInstance.PURPOSE_SYSTEM_AGENT)
            self.assertEqual(call.kwargs["display_author"], "Spec Critic")
            self.assertEqual(call.kwargs["approval_mode"], "auto_review")
            self.assertEqual(call.kwargs["sandbox_policy"], "readOnly")
            self.assertEqual(call.kwargs["model"], "gpt-5.4")
            self.assertEqual(call.kwargs["reasoning_effort"], "high")
            self.assertIn("output_schema", call.kwargs)
        prompts = "\n\n".join(call.kwargs["prompt"] for call in mock_spawn.call_args_list)
        self.assertIn("requirements extractor", prompts)
        self.assertIn("ambiguity and risk agent", prompts)
        self.assertIn("acceptance and test strategist", prompts)

    @patch("hitch.main.system_agents.codex_pool.spawn_new_session")
    def test_spec_critic_gates_on_required_clarification(
        self, mock_spawn: MagicMock
    ) -> None:
        def _spawn(**kwargs: Any) -> CodexInstance:
            return _instance(
                thread_id=f"{kwargs['agent_kind']}-{mock_spawn.call_count}",
                purpose=kwargs["purpose"],
                status=CodexInstance.STATUS_RUNNING,
                agent_kind=kwargs["agent_kind"],
            )

        mock_spawn.side_effect = _spawn
        workflow = system_agents.start_spec_critic_workflow(
            main_thread_id="main-thread",
            cwd="/repo",
            prompt="Improve onboarding",
            sandbox_policy=None,
            approval_mode="auto_review",
        )
        outputs: dict[str, dict[str, object]] = {
            system_agents.SPEC_REQUIREMENTS_AGENT_KIND: {
                "summary": "Onboarding needs work.",
                "requirements": ["Improve onboarding."],
                "assumptions": [],
                "repo_signals": ["Existing session UI."],
            },
            system_agents.SPEC_RISK_AGENT_KIND: {
                "summary": "Scope is unclear.",
                "ambiguities": ["Onboarding surface is not specified."],
                "risks": ["Could expand into unrelated UX work."],
                "questions": [
                    {
                        "id": "scope",
                        "header": "Scope",
                        "question": "Which onboarding scope should this cover?",
                        "required": True,
                        "allow_safe_default": False,
                        "safe_default": None,
                        "options": [
                            {
                                "label": "New session flow",
                                "description": "Focus on first-run session creation.",
                            },
                            {
                                "label": "Settings flow",
                                "description": "Focus on setup and preferences.",
                            },
                        ],
                    },
                    {
                        "id": "tone",
                        "header": "Tone",
                        "question": "Which tone should the UI use?",
                        "required": True,
                        "allow_safe_default": True,
                        "safe_default": "Minimal (Recommended)",
                        "options": [
                            {
                                "label": "Minimal (Recommended)",
                                "description": "Keep the implementation restrained.",
                            },
                            {
                                "label": "Guided",
                                "description": "Add more instructional UI.",
                            },
                        ],
                    },
                ],
            },
            system_agents.SPEC_TEST_AGENT_KIND: {
                "summary": "Test the selected flow.",
                "acceptance_criteria": ["Chosen flow is covered."],
                "test_strategy": ["Add a Django view test."],
                "manual_checks": ["Exercise the form in browser."],
            },
        }
        for run in workflow.agent_runs.order_by("created_at", "id"):
            instance = run.instance
            instance.events_path = _events_file(self, outputs[run.agent_kind])
            instance.status = CodexInstance.STATUS_COMPLETED
            instance.save(update_fields=["events_path", "status"])
            system_agents.on_codex_instance_finished(instance)

        workflow.refresh_from_db()
        self.assertEqual(workflow.step, system_agents.STEP_SPEC_CRITIC_CLARIFYING)
        self.assertEqual(mock_spawn.call_count, 3)
        input_request = UserInputRequest.objects.get(
            method=system_agents.SPEC_CRITIC_CLARIFICATION_METHOD
        )
        self.assertEqual(input_request.params["questions"][0]["id"], "scope")
        self.assertEqual(len(input_request.params["questions"]), 1)
        self.assertTrue(input_request.params["questions"][0]["required"])
        self.assertTrue(
            input_request.params["questions"][0]["requires_explicit_choice"]
        )
        self.assertEqual(
            input_request.params["questions"][0]["options"][0]["label"],
            "New session flow",
        )
        self.assertEqual(
            workflow.state["clarification_answers"], {"tone": "Minimal (Recommended)"}
        )

        input_request.response = {"answers": {"scope": "New session flow"}}
        input_request.save(update_fields=["response"])
        system_agents.on_user_input_resolved(input_request)

        workflow.refresh_from_db()
        self.assertEqual(workflow.step, system_agents.STEP_SPEC_CRITIC_SYNTHESIZING)
        self.assertEqual(
            workflow.state["clarification_answers"],
            {
                "scope": "New session flow",
                "tone": "Minimal (Recommended)",
            },
        )
        self.assertEqual(mock_spawn.call_count, 4)
        self.assertEqual(
            mock_spawn.call_args.kwargs["agent_kind"],
            system_agents.SPEC_SYNTHESIZER_AGENT_KIND,
        )

    @patch("hitch.main.system_agents.codex_pool.spawn_new_session")
    def test_spec_critic_preserves_partial_clarification_answers_across_reprompt(
        self, mock_spawn: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.SPEC_CRITIC_WORKFLOW_KIND,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_SPEC_CRITIC_CLARIFYING,
            state={
                "original_prompt": "Improve onboarding",
                "clarification_questions": [
                    {
                        "id": "scope",
                        "header": "Scope",
                        "question": "Which scope?",
                        "options": [
                            {"label": "New sessions", "description": "Session setup."},
                            {"label": "Settings", "description": "Settings setup."},
                        ],
                    },
                    {
                        "id": "tone",
                        "header": "Tone",
                        "question": "Which tone?",
                        "options": [
                            {"label": "Minimal", "description": "Keep it quiet."},
                            {"label": "Guided", "description": "Add more guidance."},
                        ],
                    },
                ],
                "clarification_safe_defaults": {},
            },
        )
        instance = _instance(
            thread_id="risk-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            status=CodexInstance.STATUS_COMPLETED,
            agent_kind=system_agents.SPEC_RISK_AGENT_KIND,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.SPEC_RISK_AGENT_KIND,
            thread_id=instance.thread_id,
            instance=instance,
            status=SystemAgentRun.STATUS_COMPLETED,
        )
        first_request = UserInputRequest.objects.create(
            instance=instance,
            method=system_agents.SPEC_CRITIC_CLARIFICATION_METHOD,
            params={"questions": workflow.state["clarification_questions"]},
            response={"answers": {"scope": "New sessions", "tone": ""}},
        )

        system_agents.on_user_input_resolved(first_request)

        workflow.refresh_from_db()
        self.assertEqual(workflow.step, system_agents.STEP_SPEC_CRITIC_CLARIFYING)
        self.assertEqual(
            workflow.state["clarification_answers"], {"scope": "New sessions"}
        )
        mock_spawn.assert_not_called()
        follow_up = UserInputRequest.objects.exclude(pk=first_request.pk).get(
            method=system_agents.SPEC_CRITIC_CLARIFICATION_METHOD
        )
        self.assertEqual(follow_up.params["questions"][0]["id"], "tone")

        def _spawn(**kwargs: Any) -> CodexInstance:
            return _instance(
                thread_id=f"{kwargs['agent_kind']}-thread",
                purpose=kwargs["purpose"],
                status=CodexInstance.STATUS_RUNNING,
                agent_kind=kwargs["agent_kind"],
            )

        mock_spawn.side_effect = _spawn
        follow_up.response = {"answers": {"tone": "Minimal"}}
        follow_up.save(update_fields=["response"])

        system_agents.on_user_input_resolved(follow_up)

        workflow.refresh_from_db()
        self.assertEqual(workflow.step, system_agents.STEP_SPEC_CRITIC_SYNTHESIZING)
        self.assertEqual(
            workflow.state["clarification_answers"],
            {"scope": "New sessions", "tone": "Minimal"},
        )
        mock_spawn.assert_called_once()
        self.assertEqual(
            mock_spawn.call_args.kwargs["agent_kind"],
            system_agents.SPEC_SYNTHESIZER_AGENT_KIND,
        )

    @patch("hitch.main.system_agents.codex_pool.spawn_new_session")
    def test_spec_critic_analysis_advance_claims_synthesizer_once(
        self, mock_spawn: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.SPEC_CRITIC_WORKFLOW_KIND,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_SPEC_CRITIC_ANALYZING,
            state={"original_prompt": "Improve onboarding"},
        )
        analysis_outputs: dict[str, dict[str, object]] = {
            system_agents.SPEC_REQUIREMENTS_AGENT_KIND: {
                "summary": "Onboarding needs work.",
                "requirements": ["Improve onboarding."],
                "assumptions": [],
                "repo_signals": [],
            },
            system_agents.SPEC_RISK_AGENT_KIND: {
                "summary": "No required clarification.",
                "ambiguities": [],
                "risks": [],
                "questions": [],
            },
            system_agents.SPEC_TEST_AGENT_KIND: {
                "summary": "Add focused coverage.",
                "acceptance_criteria": ["Onboarding behavior is covered."],
                "test_strategy": ["Add a focused Django test."],
                "manual_checks": [],
            },
        }
        for agent_kind, output in analysis_outputs.items():
            instance = _instance(
                thread_id=f"{agent_kind}-thread",
                purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
                workflow_id=workflow.pk,
                status=CodexInstance.STATUS_COMPLETED,
                agent_kind=agent_kind,
            )
            SystemAgentRun.objects.create(
                workflow=workflow,
                agent_kind=agent_kind,
                thread_id=instance.thread_id,
                instance=instance,
                status=SystemAgentRun.STATUS_COMPLETED,
                output=output,
            )
        mock_spawn.return_value = _instance(
            thread_id="synth-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            status=CodexInstance.STATUS_RUNNING,
            agent_kind=system_agents.SPEC_SYNTHESIZER_AGENT_KIND,
        )

        system_agents._maybe_advance_spec_critic_after_analysis(workflow)
        system_agents._maybe_advance_spec_critic_after_analysis(workflow)

        workflow.refresh_from_db()
        self.assertEqual(workflow.step, system_agents.STEP_SPEC_CRITIC_SYNTHESIZING)
        mock_spawn.assert_called_once()
        self.assertEqual(
            SystemAgentRun.objects.filter(
                workflow=workflow,
                agent_kind=system_agents.SPEC_SYNTHESIZER_AGENT_KIND,
            ).count(),
            1,
        )

    @patch("hitch.main.system_agents.codex_pool.spawn_turn")
    def test_spec_critic_invalid_json_failure_surfaces_to_visible_thread(
        self, mock_spawn: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.SPEC_CRITIC_WORKFLOW_KIND,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_SPEC_CRITIC_ANALYZING,
            state={
                "original_prompt": "Improve onboarding",
                "next_user_message_index": 3,
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
            thread_id="requirements-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            status=CodexInstance.STATUS_COMPLETED,
            events_path=events_path,
            agent_kind=system_agents.SPEC_REQUIREMENTS_AGENT_KIND,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.SPEC_REQUIREMENTS_AGENT_KIND,
            thread_id=instance.thread_id,
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
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
        self.assertEqual(kwargs["agent_kind"], system_agents.SPEC_CRITIC_WORKFLOW_KIND)
        self.assertEqual(kwargs["display_author"], system_agents.SPEC_CRITIC_DISPLAY_AUTHOR)
        self.assertEqual(kwargs["user_message_index"], 3)
        self.assertIn("Spec Critic could not complete", kwargs["prompt"])
        self.assertIn("Improve onboarding", kwargs["prompt"])
        self.assertIn("output was not valid JSON", kwargs["prompt"])

    @patch("hitch.main.system_agents.codex_pool.spawn_turn")
    def test_spec_critic_synthesis_injects_visible_implementation_brief(
        self, mock_spawn_turn: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.SPEC_CRITIC_WORKFLOW_KIND,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_SPEC_CRITIC_SYNTHESIZING,
            state={
                "original_prompt": "Improve onboarding",
                "sandbox_policy": "workspaceWrite",
                "approval_mode": "prompt_user",
                "model": "gpt-5.4",
                "reasoning_effort": "high",
                "base_instructions": "Base",
                "developer_instructions": "Developer",
                "enable_memories": True,
                "next_user_message_index": 4,
                "auto_pr_enabled": True,
                "qa_panel_enabled": True,
                "clarification_answers": {"scope": "New session flow"},
            },
        )
        instance = _instance(
            thread_id="synth-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            status=CodexInstance.STATUS_COMPLETED,
            agent_kind=system_agents.SPEC_SYNTHESIZER_AGENT_KIND,
            events_path=_events_file(
                self,
                {"brief": "Implement a focused onboarding pass for new sessions."},
            ),
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.SPEC_SYNTHESIZER_AGENT_KIND,
            thread_id=instance.thread_id,
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )

        system_agents.on_codex_instance_finished(instance)

        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(
            workflow.step, system_agents.STEP_SPEC_CRITIC_IMPLEMENTATION_SPAWNED
        )
        mock_spawn_turn.assert_called_once()
        kwargs = mock_spawn_turn.call_args.kwargs
        self.assertEqual(kwargs["thread_id"], "main-thread")
        self.assertNotIn("workflow_id", kwargs)
        self.assertTrue(kwargs["auto_pr_enabled"])
        self.assertTrue(kwargs["qa_panel_enabled"])
        self.assertEqual(kwargs["user_message_index"], 4)
        self.assertIn("Hitch Spec Critic synthesized", kwargs["prompt"])
        self.assertIn("Improve onboarding", kwargs["prompt"])
        self.assertIn(
            "Implement a focused onboarding pass for new sessions.", kwargs["prompt"]
        )
        self.assertIn("scope: New session flow", kwargs["prompt"])

    @patch("hitch.main.system_agents.codex_pool.spawn_turn")
    def test_stop_active_workflow_cancels_spec_critic_clarification_without_running_agent(
        self, mock_spawn: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.SPEC_CRITIC_WORKFLOW_KIND,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_SPEC_CRITIC_CLARIFYING,
            state={
                "original_prompt": "Improve onboarding",
                "next_user_message_index": 5,
            },
        )
        instance = _instance(
            thread_id="risk-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            status=CodexInstance.STATUS_COMPLETED,
            agent_kind=system_agents.SPEC_RISK_AGENT_KIND,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.SPEC_RISK_AGENT_KIND,
            thread_id=instance.thread_id,
            instance=instance,
            status=SystemAgentRun.STATUS_COMPLETED,
        )
        input_request = UserInputRequest.objects.create(
            instance=instance,
            method=system_agents.SPEC_CRITIC_CLARIFICATION_METHOD,
            params={"questions": [{"id": "scope"}]},
        )

        stopped = system_agents.stop_active_workflow("main-thread")

        self.assertTrue(stopped)
        workflow.refresh_from_db()
        input_request.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
        self.assertEqual(workflow.step, system_agents.STEP_BLOCKED)
        response = input_request.response
        assert response is not None
        self.assertEqual(response.get("cancelled"), True)
        self.assertIsNotNone(input_request.responded_at)
        mock_spawn.assert_called_once()
        kwargs = mock_spawn.call_args.kwargs
        self.assertEqual(kwargs["purpose"], CodexInstance.PURPOSE_SYSTEM_FEEDBACK)
        self.assertEqual(kwargs["display_author"], system_agents.SPEC_CRITIC_DISPLAY_AUTHOR)
        self.assertEqual(kwargs["user_message_index"], 5)
        self.assertIn("stopped by user", kwargs["prompt"])

    def test_pr_monitor_output_schema_is_strict_for_response_format(self) -> None:
        schema = system_agents._PR_MONITOR_OUTPUT_SCHEMA

        _assert_response_schema_objects_are_strict(self, schema)
        pr_schema = schema["properties"]["pr"]
        self.assertEqual(pr_schema["required"], list(system_agents._PR_HANDOFF_FIELDS))
        self.assertEqual(pr_schema["properties"]["url"]["type"], ["string", "null"])
        self.assertEqual(
            pr_schema["properties"]["pr_number"]["type"], ["integer", "null"]
        )
        self.assertEqual(
            pr_schema["properties"]["merged"]["type"], ["boolean", "null"]
        )
        self.assertEqual(
            pr_schema["properties"]["latest_comments"]["type"], ["array", "null"]
        )
        self.assertEqual(
            pr_schema["properties"]["latest_comments"]["items"]["type"], "string"
        )

    @patch("hitch.main.system_agents.build_worktree_diff_text", return_value="diff --git")
    @patch("hitch.main.system_agents.codex_pool.spawn_new_session")
    def test_qa_only_workflow_keeps_default_iteration_limit(
        self, mock_spawn: MagicMock, _mock_diff: MagicMock
    ) -> None:
        mock_spawn.return_value = _instance(
            thread_id="qa-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
        )

        workflow = system_agents.start_pr_qa_workflow(
            main_thread_id="main-thread",
            cwd="/repo",
            sandbox_policy=None,
            approval_mode="auto_review",
            open_pr_on_lgtm=False,
        )

        self.assertEqual(
            workflow.max_iterations, system_agents.QA_WORKFLOW_MAX_ITERATIONS
        )

    @patch("hitch.main.system_agents.build_worktree_diff_text", return_value="diff --git")
    @patch("hitch.main.system_agents.codex_pool.spawn_new_session")
    def test_parallel_qa_panel_starts_all_hidden_lanes(
        self, mock_spawn: MagicMock, _mock_diff: MagicMock
    ) -> None:
        mock_spawn.side_effect = [
            _instance(
                thread_id=f"qa-lane-{index}",
                purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
                agent_kind=lane.agent_kind,
            )
            for index, lane in enumerate(system_agents._QA_PANEL_LANES)
        ]

        workflow = system_agents.start_pr_qa_workflow(
            main_thread_id="main-thread",
            cwd="/repo",
            sandbox_policy="workspaceWrite",
            approval_mode="prompt_user",
            model="gpt-5.4",
            reasoning_effort="high",
            qa_panel_enabled=True,
        )

        self.assertEqual(workflow.step, system_agents.STEP_QA_RUNNING)
        self.assertTrue(workflow.state["qa_panel_enabled"])
        self.assertEqual(mock_spawn.call_count, len(system_agents._QA_PANEL_LANES))
        agent_kinds = [call.kwargs["agent_kind"] for call in mock_spawn.call_args_list]
        self.assertEqual(agent_kinds, list(system_agents._QA_PANEL_LANE_KINDS))
        for call, lane in zip(
            mock_spawn.call_args_list, system_agents._QA_PANEL_LANES, strict=True
        ):
            kwargs = call.kwargs
            self.assertEqual(kwargs["thread_source"], ThreadSource.subagent)
            self.assertEqual(kwargs["purpose"], CodexInstance.PURPOSE_SYSTEM_AGENT)
            self.assertEqual(kwargs["approval_mode"], system_agents.SYSTEM_AGENT_APPROVAL_MODE)
            self.assertEqual(kwargs["sandbox_policy"], "workspaceWrite")
            self.assertEqual(kwargs["model"], "gpt-5.4")
            self.assertEqual(kwargs["reasoning_effort"], "high")
            self.assertEqual(kwargs["display_author"], system_agents.QA_PANEL_DISPLAY_AUTHOR)
            self.assertEqual(kwargs["output_schema"], system_agents._QA_PANEL_LANE_OUTPUT_SCHEMA)
            self.assertIn(lane.label, kwargs["prompt"])
            self.assertIn(lane.focus, kwargs["prompt"])
            self.assertIn("diff --git", kwargs["prompt"])

        runs = list(SystemAgentRun.objects.filter(workflow=workflow).order_by("created_at"))
        self.assertEqual(len(runs), len(system_agents._QA_PANEL_LANES))
        self.assertEqual([run.agent_kind for run in runs], list(system_agents._QA_PANEL_LANE_KINDS))
        self.assertEqual(runs[0].input["lane"], system_agents._QA_PANEL_LANES[0].label)

    @patch("hitch.main.system_agents.build_worktree_diff_text", return_value="diff --git")
    @patch("hitch.main.system_agents.codex_pool.spawn_new_session")
    def test_parallel_qa_panel_waits_for_lanes_then_starts_synthesizer(
        self, mock_spawn: MagicMock, _mock_diff: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_QA_RUNNING,
            state={"qa_panel_enabled": True},
        )
        lane_instances: list[CodexInstance] = []
        for index, lane in enumerate(system_agents._QA_PANEL_LANES):
            instance = _instance(
                thread_id=f"qa-lane-{index}",
                purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
                workflow_id=workflow.pk,
                events_path=_events_file(
                    self,
                    {
                        "summary": f"{lane.label} summary",
                        "findings": [
                            {
                                "severity": "P2",
                                "location": "hitch/main/system_agents.py:1",
                                "title": f"{lane.label} finding",
                                "description": "The panel should preserve this issue.",
                            }
                        ],
                        "lgtm": False,
                    },
                ),
                agent_kind=lane.agent_kind,
            )
            lane_instances.append(instance)
            SystemAgentRun.objects.create(
                workflow=workflow,
                agent_kind=lane.agent_kind,
                thread_id=instance.thread_id,
                instance=instance,
                status=SystemAgentRun.STATUS_RUNNING,
                input={"lane": lane.label},
            )
        mock_spawn.return_value = _instance(
            thread_id="qa-synth",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.PR_QA_PANEL_SYNTHESIZER_AGENT_KIND,
        )

        for instance in lane_instances[:-1]:
            system_agents.on_codex_instance_finished(instance)
        mock_spawn.assert_not_called()

        system_agents.on_codex_instance_finished(lane_instances[-1])

        mock_spawn.assert_called_once()
        kwargs = mock_spawn.call_args.kwargs
        self.assertEqual(kwargs["agent_kind"], system_agents.PR_QA_PANEL_SYNTHESIZER_AGENT_KIND)
        self.assertEqual(kwargs["output_schema"], system_agents._QA_OUTPUT_SCHEMA)
        self.assertIn("final Parallel QA Panel synthesizer", kwargs["prompt"])
        self.assertIn("Deduplicate overlapping findings", kwargs["prompt"])
        self.assertIn("preserve the highest severity", kwargs["prompt"])
        self.assertIn("Correctness summary", kwargs["prompt"])
        self.assertIn("diff --git", kwargs["prompt"])
        workflow.refresh_from_db()
        self.assertEqual(workflow.state[system_agents._QA_PANEL_SYNTHESIZER_STARTED_KEY], 0)
        synth_run = SystemAgentRun.objects.get(
            workflow=workflow,
            agent_kind=system_agents.PR_QA_PANEL_SYNTHESIZER_AGENT_KIND,
        )
        self.assertEqual(synth_run.thread_id, "qa-synth")

    @patch("hitch.main.system_agents.codex_pool.spawn_turn")
    def test_parallel_qa_panel_synthesizer_verdict_feeds_existing_loop(
        self, mock_spawn: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_QA_RUNNING,
            state={"qa_panel_enabled": True, "next_user_message_index": 2},
        )
        instance = _instance(
            thread_id="qa-synth",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            events_path=_events_file(
                self,
                {"feedback": "[P1] Fix the consolidated issue.", "lgtm": False},
            ),
            agent_kind=system_agents.PR_QA_PANEL_SYNTHESIZER_AGENT_KIND,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.PR_QA_PANEL_SYNTHESIZER_AGENT_KIND,
            thread_id="qa-synth",
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )

        system_agents.on_codex_instance_finished(instance)

        run.refresh_from_db()
        workflow.refresh_from_db()
        self.assertEqual(run.status, SystemAgentRun.STATUS_COMPLETED)
        self.assertEqual(workflow.step, system_agents.STEP_FEEDBACK_RUNNING)
        self.assertEqual(workflow.iteration, 1)
        self.assertEqual(workflow.state["last_feedback"], "[P1] Fix the consolidated issue.")
        self.assertIn("Feedback from Hitch QA agent", mock_spawn.call_args.kwargs["prompt"])
        self.assertEqual(mock_spawn.call_args.kwargs["user_message_index"], 2)

    @patch("hitch.main.system_agents.build_worktree_diff_text", return_value="diff --git")
    @patch("hitch.main.system_agents.codex_pool.spawn_turn")
    @patch("hitch.main.system_agents.codex_pool.interrupt_instance")
    @patch("hitch.main.system_agents.codex_pool.spawn_new_session")
    def test_parallel_qa_panel_interrupts_started_lanes_when_start_fails(
        self,
        mock_spawn_new: MagicMock,
        mock_interrupt: MagicMock,
        _mock_spawn_turn: MagicMock,
        _mock_diff: MagicMock,
    ) -> None:
        started = _instance(
            thread_id="qa-lane-0",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents._QA_PANEL_LANES[0].agent_kind,
        )
        mock_spawn_new.side_effect = [started, RuntimeError("boom")]
        mock_interrupt.return_value = started

        workflow = system_agents.start_pr_qa_workflow(
            main_thread_id="main-thread",
            cwd="/repo",
            sandbox_policy=None,
            approval_mode="auto_review",
            qa_panel_enabled=True,
        )

        workflow.refresh_from_db()
        run = SystemAgentRun.objects.get(workflow=workflow)
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
        self.assertEqual(run.status, SystemAgentRun.STATUS_FAILED)
        self.assertEqual(run.error, "QA panel failed to start")
        mock_interrupt.assert_called_once_with(
            started.pk, expected_thread_id="qa-lane-0"
        )

    def test_parallel_qa_panel_finalizes_lane_after_workflow_blocks(self) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_BLOCKED,
            step=system_agents.STEP_BLOCKED,
            state={"qa_panel_enabled": True},
        )
        lane = system_agents._QA_PANEL_LANES[0]
        instance = _instance(
            thread_id="qa-lane-0",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            events_path=_events_file(
                self,
                {
                    "summary": "Clean after sibling failed.",
                    "findings": [],
                    "lgtm": True,
                },
            ),
            agent_kind=lane.agent_kind,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=lane.agent_kind,
            thread_id=instance.thread_id,
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
            input={"lane": lane.label},
        )

        system_agents.on_codex_instance_finished(instance)

        run.refresh_from_db()
        workflow.refresh_from_db()
        self.assertEqual(run.status, SystemAgentRun.STATUS_COMPLETED)
        self.assertEqual(run.output["lane"], lane.label)
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)

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
    def test_auto_pr_forwards_parallel_qa_panel_setting(
        self, mock_start: MagicMock
    ) -> None:
        instance = _instance(auto_pr_enabled=True, qa_panel_enabled=True)

        system_agents.on_codex_instance_finished(instance)

        self.assertTrue(mock_start.call_args.kwargs["qa_panel_enabled"])

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

    @patch("hitch.main.system_agents.start_pr_qa_workflow")
    def test_auto_pr_claims_turn_before_starting_workflow(
        self, mock_start: MagicMock
    ) -> None:
        instance = _instance(auto_pr_enabled=True)

        def assert_claimed(**_kwargs: object) -> None:
            instance.refresh_from_db()
            self.assertIsNotNone(instance.auto_pr_triggered_at)

        mock_start.side_effect = assert_claimed

        system_agents.on_codex_instance_finished(instance)

        mock_start.assert_called_once()

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

    def test_retired_system_agent_workflow_is_finalized(self) -> None:
        workflow = SystemWorkflow.objects.create(
            kind="retired_kind",
            main_thread_id="retired-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step="retired_running",
        )
        instance = _instance(
            thread_id="retired-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind="retired_kind",
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind="retired_kind",
            thread_id="retired-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )

        system_agents.on_codex_instance_finished(instance)

        run.refresh_from_db()
        workflow.refresh_from_db()
        self.assertEqual(run.status, SystemAgentRun.STATUS_FAILED)
        self.assertIn("no longer supported", run.error)
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
        self.assertEqual(workflow.step, system_agents.STEP_BLOCKED)
        self.assertIn("no longer supported", workflow.state["error"])
        self.assertNotIn("failure_surfaced", workflow.state)

    @patch("hitch.main.system_agents.demo.on_codex_instance_finished")
    def test_demo_system_workflow_is_routed_to_demo_router(
        self, mock_demo_finished: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=demo.DEMO_WORKFLOW_KIND,
            main_thread_id="demo-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step="demo_running",
        )
        instance = _instance(
            thread_id="demo-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=demo.DEMO_AGENT_KIND,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=demo.DEMO_AGENT_KIND,
            thread_id="demo-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )

        system_agents.on_codex_instance_finished(instance)

        mock_demo_finished.assert_called_once_with(instance)
        run.refresh_from_db()
        workflow.refresh_from_db()
        self.assertEqual(run.status, SystemAgentRun.STATUS_RUNNING)
        self.assertEqual(run.error, "")
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_RUNNING)
        self.assertEqual(workflow.step, "demo_running")

    @patch("hitch.main.system_agents.demo.on_codex_instance_finished")
    def test_demo_workflow_requires_demo_agent_kind(
        self, mock_demo_finished: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=demo.DEMO_WORKFLOW_KIND,
            main_thread_id="demo-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step="demo_running",
        )
        instance = _instance(
            thread_id="demo-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind="unexpected",
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind="unexpected",
            thread_id="demo-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )

        handled = system_agents.on_codex_instance_finished(instance)

        self.assertTrue(handled)
        mock_demo_finished.assert_not_called()
        run.refresh_from_db()
        workflow.refresh_from_db()
        self.assertEqual(run.status, SystemAgentRun.STATUS_FAILED)
        self.assertIn("no longer supported", run.error)
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
        self.assertEqual(workflow.step, system_agents.STEP_BLOCKED)

    @patch("hitch.main.system_agents.demo.on_codex_instance_finished")
    def test_demo_workflow_requires_demo_instance_kind(
        self, mock_demo_finished: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=demo.DEMO_WORKFLOW_KIND,
            main_thread_id="demo-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step="demo_running",
        )
        instance = _instance(
            thread_id="demo-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind="unexpected",
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=demo.DEMO_AGENT_KIND,
            thread_id="demo-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )

        handled = system_agents.on_codex_instance_finished(instance)

        self.assertTrue(handled)
        mock_demo_finished.assert_not_called()
        run.refresh_from_db()
        workflow.refresh_from_db()
        self.assertEqual(run.status, SystemAgentRun.STATUS_FAILED)
        self.assertIn("no longer supported", run.error)
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
        self.assertEqual(workflow.step, system_agents.STEP_BLOCKED)

    @patch(
        "hitch.main.system_agents.demo.on_codex_instance_finished",
        side_effect=RuntimeError("boom"),
    )
    def test_demo_system_workflow_fails_if_demo_router_raises(
        self, mock_demo_finished: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=demo.DEMO_WORKFLOW_KIND,
            main_thread_id="demo-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step="demo_running",
        )
        instance = _instance(
            thread_id="demo-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=demo.DEMO_AGENT_KIND,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=demo.DEMO_AGENT_KIND,
            thread_id="demo-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )

        system_agents.on_codex_instance_finished(instance)

        mock_demo_finished.assert_called_once_with(instance)
        run.refresh_from_db()
        workflow.refresh_from_db()
        self.assertEqual(run.status, SystemAgentRun.STATUS_FAILED)
        self.assertIn("demo workflow router failed: boom", run.error)
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_FAILED)

    @patch("hitch.main.system_agents.demo.on_codex_instance_finished")
    def test_demo_agent_kind_does_not_route_non_demo_workflow(
        self, mock_demo_finished: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="demo-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_QA_RUNNING,
        )
        instance = _instance(
            thread_id="demo-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=demo.DEMO_AGENT_KIND,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=demo.DEMO_AGENT_KIND,
            thread_id="demo-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )

        system_agents.on_codex_instance_finished(instance)

        mock_demo_finished.assert_not_called()
        run.refresh_from_db()
        workflow.refresh_from_db()
        self.assertEqual(run.status, SystemAgentRun.STATUS_FAILED)
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
        self.assertEqual(workflow.step, system_agents.STEP_BLOCKED)
        self.assertIn("unsupported PR QA agent kind 'demo'", workflow.state["error"])

    def test_system_agent_without_run_returns_false(self) -> None:
        instance = _instance(
            thread_id="orphan-system-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=12345,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
        )

        handled = system_agents.on_codex_instance_finished(instance)

        self.assertFalse(handled)

    def test_terminal_system_agent_run_returns_true_without_changes(self) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="terminal-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_QA_RUNNING,
        )
        instance = _instance(
            thread_id="terminal-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            thread_id="terminal-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_COMPLETED,
            output={"feedback": "", "lgtm": True},
        )

        handled = system_agents.on_codex_instance_finished(instance)

        self.assertTrue(handled)
        run.refresh_from_db()
        workflow.refresh_from_db()
        self.assertEqual(run.status, SystemAgentRun.STATUS_COMPLETED)
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_RUNNING)
        self.assertEqual(workflow.step, system_agents.STEP_QA_RUNNING)

    def test_retired_system_agent_run_does_not_reopen_terminal_workflow(self) -> None:
        workflow = SystemWorkflow.objects.create(
            kind="retired_kind",
            main_thread_id="retired-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_COMPLETED,
            step="retired_completed",
        )
        instance = _instance(
            thread_id="retired-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind="retired_kind",
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind="retired_kind",
            thread_id="retired-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )

        system_agents.on_codex_instance_finished(instance)

        run.refresh_from_db()
        workflow.refresh_from_db()
        self.assertEqual(run.status, SystemAgentRun.STATUS_FAILED)
        self.assertIn("no longer supported", run.error)
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(workflow.step, "retired_completed")
        self.assertEqual(workflow.state, {})

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
    def test_qa_design_synthesis_gate_cases(self, mock_spawn: MagicMock) -> None:
        cases = (
            _DesignGateCase(
                name="recurring file and state lifecycle feedback",
                prior_feedback=(
                    "[P2] hitch/main/demo.py:230 lets stale superseded attempts "
                    "overwrite the active generation after cleanup."
                ),
                current_feedback=(
                    "[P2] hitch/main/demo.py:287 still has a stale generation "
                    "race where a cancelled attempt overwrites active state."
                ),
                expect_gate=True,
                expected_categories=("state_lifecycle",),
                expected_files=("hitch/main/demo.py",),
                prompt_includes=(
                    "QA Design Synthesis Gate",
                    "pause and simplify",
                    "Prior related QA feedback",
                ),
            ),
            _DesignGateCase(
                name="first unrelated feedback",
                current_feedback=(
                    "[P2] hitch/main/views.py:120 returns the wrong template "
                    "for this single path."
                ),
                expect_gate=False,
                iteration=0,
            ),
            _DesignGateCase(
                name="substring keyword match",
                prior_feedback="Manual QA for hitch/main/views.py: interactive test suite passed.",
                current_feedback=(
                    "[P2] hitch/main/views.py:120 has an interactive test suite "
                    "coverage gap."
                ),
                expect_gate=False,
            ),
            _DesignGateCase(
                name="prior lgtm feedback",
                prior_feedback="No findings. Browser UI status rendered for hitch/main/views.py.",
                prior_lgtm=True,
                current_feedback="[P2] hitch/main/views.py:120 leaves browser UI status stale.",
                expect_gate=False,
            ),
            _DesignGateCase(
                name="category words in file paths",
                prior_feedback=(
                    "[P2] frontend/state_guard.tsx:14 and config/schema.json "
                    "return the wrong value."
                ),
                current_feedback=(
                    "[P2] frontend/state_guard.tsx:20 and config/schema.json "
                    "miss a test assertion."
                ),
                expect_gate=False,
            ),
            _DesignGateCase(
                name="dotted prose as file overlap",
                prior_feedback="State validation is unclear, e.g. around v1.2.3.",
                current_feedback="State validation still fails, e.g. in v1.2.3.",
                expect_gate=False,
            ),
            _DesignGateCase(
                name="urls as file overlap",
                prior_feedback="Browser status is unclear; see https://docs.example.com/spec.v1.",
                current_feedback="Browser status still fails; see https://docs.example.com/spec.v1.",
                expect_gate=False,
            ),
            _DesignGateCase(
                name="extensionless paths before categorizing",
                prior_feedback=(
                    "[P2] services/state/ and hitch/main/migrations/ return "
                    "the wrong value."
                ),
                current_feedback=(
                    "[P2] services/state/ and hitch/main/migrations/ miss "
                    "a test assertion."
                ),
                expect_gate=False,
                iteration=0,
            ),
        )

        for index, case in enumerate(cases):
            with self.subTest(case=case.name):
                mock_spawn.reset_mock()
                workflow = self._finish_design_gate_case(case, index)

                mock_spawn.assert_called_once()
                prompt = mock_spawn.call_args.kwargs["prompt"]
                gate = workflow.state.get(system_agents._QA_DESIGN_SYNTHESIS_STATE_KEY)
                if case.expect_gate:
                    self.assertIsNotNone(gate)
                    self.assertEqual(gate["triggered_at_iteration"], case.iteration + 1)
                    for category in case.expected_categories:
                        self.assertIn(category, gate["recurring_categories"])
                    for file_path in case.expected_files:
                        self.assertIn(file_path, gate["recurring_files"])
                    for expected_text in case.prompt_includes:
                        self.assertIn(expected_text, prompt)
                else:
                    self.assertIsNone(gate)
                    self.assertIn("Feedback from Hitch QA agent", prompt)
                    self.assertNotIn("QA Design Synthesis Gate", prompt)

    def _finish_design_gate_case(self, case: _DesignGateCase, index: int) -> SystemWorkflow:
        slug = f"design-gate-{index}"
        cwd = f"/repo/{slug}"
        if case.prior_feedback is not None:
            self._record_prior_design_feedback(
                cwd=cwd,
                feedback=case.prior_feedback,
                lgtm=case.prior_lgtm,
                slug=slug,
            )
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id=f"{slug}-main-thread",
            cwd=cwd,
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_QA_RUNNING,
            iteration=case.iteration,
            state={"next_user_message_index": 2},
        )
        events_path = _events_file(
            self,
            {"feedback": case.current_feedback, "lgtm": False},
        )
        instance = _instance(
            thread_id=f"{slug}-qa-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            events_path=events_path,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            thread_id=f"{slug}-qa-thread",
            instance=instance,
        )

        system_agents.on_codex_instance_finished(instance)

        workflow.refresh_from_db()
        return workflow

    def _record_prior_design_feedback(
        self, *, cwd: str, feedback: str, lgtm: bool, slug: str
    ) -> None:
        prior_workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id=f"{slug}-prior-thread",
            cwd=cwd,
            status=SystemWorkflow.STATUS_COMPLETED,
            step=system_agents.STEP_PR_PROMPT_SPAWNED,
            iteration=1,
        )
        prior_instance = _instance(
            thread_id=f"{slug}-prior-qa-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=prior_workflow.pk,
        )
        SystemAgentRun.objects.create(
            workflow=prior_workflow,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            thread_id=f"{slug}-prior-qa-thread",
            instance=prior_instance,
            status=SystemAgentRun.STATUS_COMPLETED,
            output={"feedback": feedback, "lgtm": lgtm},
        )

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
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_RUNNING)
        self.assertEqual(workflow.step, system_agents.STEP_PR_PROMPT_RUNNING)
        self.assertEqual(workflow.state["next_user_message_index"], 5)
        self.assertEqual(
            workflow.state[system_agents.QA_APPROVAL_INSERT_INDEX_STATE_KEY], 4
        )

    @patch("hitch.main.system_agents.codex_pool.spawn_turn")
    def test_qa_lgtm_can_complete_without_pr_prompt(self, mock_spawn: MagicMock) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step="qa_running",
            state={"open_pr_on_lgtm": False, "next_user_message_index": 4},
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

        mock_spawn.assert_not_called()
        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(workflow.step, system_agents.STEP_QA_APPROVED)
        self.assertEqual(workflow.state["next_user_message_index"], 4)
        self.assertEqual(workflow.state["last_feedback"], "Looks good")

    @patch("hitch.main.system_agents.codex_pool.spawn_new_session")
    def test_pr_prompt_completion_stores_handoff_and_starts_monitor(
        self, mock_spawn: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_PR_PROMPT_RUNNING,
            state={"next_user_message_index": 5},
        )
        events_path = _raw_events_file(
            self,
            [
                _pr_tool_event(
                    thread_id="main-thread",
                    tool="github_create_pull_request",
                    arguments={"repository_full_name": "cberner/hitch"},
                    structured_content={
                        "url": "https://github.com/cberner/hitch/pull/169",
                        "number": 169,
                        "state": "open",
                        "merged": False,
                        "mergeable": True,
                        "head": "feature",
                        "head_sha": "abc123",
                    },
                )
            ],
        )
        instance = _instance(
            thread_id="main-thread",
            purpose=CodexInstance.PURPOSE_USER,
            workflow_id=workflow.pk,
            events_path=events_path,
        )
        mock_spawn.return_value = _instance(
            thread_id="monitor-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.PR_FOLLOWUP_MONITOR_AGENT_KIND,
        )

        system_agents.on_codex_instance_finished(instance)

        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_RUNNING)
        self.assertEqual(workflow.step, system_agents.STEP_PR_MONITORING)
        handoff = workflow.state[system_agents._PR_HANDOFF_STATE_KEY]
        self.assertEqual(handoff["url"], "https://github.com/cberner/hitch/pull/169")
        self.assertEqual(handoff["repository_full_name"], "cberner/hitch")
        self.assertEqual(handoff["pr_number"], 169)
        self.assertEqual(handoff["head_sha"], "abc123")
        kwargs = mock_spawn.call_args.kwargs
        self.assertEqual(kwargs["purpose"], CodexInstance.PURPOSE_SYSTEM_AGENT)
        self.assertEqual(
            kwargs["agent_kind"], system_agents.PR_FOLLOWUP_MONITOR_AGENT_KIND
        )
        self.assertEqual(kwargs["display_author"], system_agents.PR_MONITOR_DISPLAY_AUTHOR)
        self.assertEqual(kwargs["output_schema"], system_agents._PR_MONITOR_OUTPUT_SCHEMA)
        self.assertIn("Do not edit files", kwargs["prompt"])
        self.assertIn("https://github.com/cberner/hitch/pull/169", kwargs["prompt"])
        run = SystemAgentRun.objects.get(workflow=workflow)
        self.assertEqual(run.thread_id, "monitor-thread")

    @patch("hitch.main.system_agents.codex_pool.spawn_new_session")
    def test_pr_prompt_completion_without_handoff_preserves_completion(
        self, mock_spawn: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_PR_PROMPT_RUNNING,
            state={"next_user_message_index": 5},
        )
        instance = _instance(
            thread_id="main-thread",
            purpose=CodexInstance.PURPOSE_USER,
            workflow_id=workflow.pk,
        )

        system_agents.on_codex_instance_finished(instance)

        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(workflow.step, system_agents.STEP_PR_PROMPT_SPAWNED)
        self.assertNotIn(system_agents._PR_HANDOFF_STATE_KEY, workflow.state)
        mock_spawn.assert_not_called()

    @patch("hitch.main.system_agents.codex_pool.spawn_turn")
    def test_monitor_blocker_spawns_pr_feedback_with_stale_branch_guard(
        self, mock_spawn: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_PR_MONITORING,
            state={
                "next_user_message_index": 5,
                system_agents._PR_HANDOFF_STATE_KEY: {
                    "url": "https://github.com/cberner/hitch/pull/169",
                    "repository_full_name": "cberner/hitch",
                    "pr_number": 169,
                    "head": "feature",
                    "head_sha": "abc123",
                },
            },
        )
        events_path = _events_file(
            self,
            {
                "status": "blocked",
                "summary": "Review feedback arrived.",
                "feedback": "Address the unresolved review thread.",
                "pr": {
                    "url": "https://github.com/cberner/hitch/pull/169",
                    "unresolved_thread_count": 1,
                },
                "blockers": ["1 unresolved review thread"],
            },
        )
        instance = _instance(
            thread_id="monitor-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            events_path=events_path,
            agent_kind=system_agents.PR_FOLLOWUP_MONITOR_AGENT_KIND,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.PR_FOLLOWUP_MONITOR_AGENT_KIND,
            thread_id="monitor-thread",
            instance=instance,
        )

        system_agents.on_codex_instance_finished(instance)

        workflow.refresh_from_db()
        self.assertEqual(workflow.step, system_agents.STEP_PR_FEEDBACK_RUNNING)
        self.assertEqual(workflow.iteration, 1)
        self.assertEqual(
            workflow.state[system_agents._PR_HANDOFF_STATE_KEY][
                "unresolved_thread_count"
            ],
            1,
        )
        kwargs = mock_spawn.call_args.kwargs
        self.assertEqual(kwargs["thread_id"], "main-thread")
        self.assertEqual(kwargs["purpose"], CodexInstance.PURPOSE_SYSTEM_FEEDBACK)
        self.assertEqual(
            kwargs["display_author"], system_agents.PR_MONITOR_DISPLAY_AUTHOR
        )
        self.assertEqual(
            kwargs["agent_kind"], system_agents.PR_FOLLOWUP_MONITOR_AGENT_KIND
        )
        self.assertEqual(kwargs["user_message_index"], 5)
        self.assertIn("Hitch PR monitor found follow-up work", kwargs["prompt"])
        self.assertIn("merged, closed, or its head branch is missing", kwargs["prompt"])
        self.assertIn("Address the unresolved review thread.", kwargs["prompt"])

    @patch("hitch.main.system_agents.codex_pool.spawn_new_session")
    def test_pr_feedback_completion_restarts_monitor_with_updated_handoff(
        self, mock_spawn: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_PR_FEEDBACK_RUNNING,
            state={
                system_agents._PR_HANDOFF_STATE_KEY: {
                    "url": "https://github.com/cberner/hitch/pull/169",
                    "repository_full_name": "cberner/hitch",
                    "pr_number": 169,
                    "head_sha": "oldsha",
                }
            },
        )
        events_path = _raw_events_file(
            self,
            [
                _pr_tool_event(
                    thread_id="main-thread",
                    tool="github_get_pr_info",
                    arguments={
                        "repository_full_name": "cberner/hitch",
                        "pr_number": 169,
                    },
                    structured_content={
                        "url": "https://github.com/cberner/hitch/pull/169",
                        "number": 169,
                        "state": "open",
                        "merged": False,
                        "head_sha": "newsha",
                    },
                )
            ],
        )
        instance = _instance(
            thread_id="main-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_FEEDBACK,
            workflow_id=workflow.pk,
            events_path=events_path,
        )
        mock_spawn.return_value = _instance(
            thread_id="monitor-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.PR_FOLLOWUP_MONITOR_AGENT_KIND,
        )

        system_agents.on_codex_instance_finished(instance)

        workflow.refresh_from_db()
        self.assertEqual(workflow.step, system_agents.STEP_PR_MONITORING)
        self.assertEqual(
            workflow.state[system_agents._PR_HANDOFF_STATE_KEY]["head_sha"],
            "newsha",
        )
        self.assertEqual(
            mock_spawn.call_args.kwargs["agent_kind"],
            system_agents.PR_FOLLOWUP_MONITOR_AGENT_KIND,
        )

    def test_monitor_ready_completes_workflow(self) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_PR_MONITORING,
            state={system_agents._PR_HANDOFF_STATE_KEY: {"pr_number": 169}},
        )
        events_path = _events_file(
            self,
            {
                "status": "ready",
                "summary": "PR is clean.",
                "feedback": "",
                "pr": {"pr_number": 169, "ci_status": "success"},
                "blockers": [],
            },
        )
        instance = _instance(
            thread_id="monitor-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            events_path=events_path,
            agent_kind=system_agents.PR_FOLLOWUP_MONITOR_AGENT_KIND,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.PR_FOLLOWUP_MONITOR_AGENT_KIND,
            thread_id="monitor-thread",
            instance=instance,
        )

        system_agents.on_codex_instance_finished(instance)

        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(workflow.step, system_agents.STEP_PR_READY)

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
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_RUNNING)
        self.assertEqual(workflow.step, system_agents.STEP_PR_PROMPT_RUNNING)

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
    @patch("hitch.main.system_agents.codex_pool.interrupt_instance")
    def test_stop_active_workflow_marks_only_interrupted_runs_failed(
        self, mock_interrupt: MagicMock, mock_spawn: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_QA_RUNNING,
            state={"qa_panel_enabled": True},
        )
        interrupted_instance = _instance(
            thread_id="qa-lane-0",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            status=CodexInstance.STATUS_RUNNING,
        )
        still_running_instance = _instance(
            thread_id="qa-lane-1",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            status=CodexInstance.STATUS_RUNNING,
        )
        interrupted_run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents._QA_PANEL_LANES[0].agent_kind,
            thread_id=interrupted_instance.thread_id,
            instance=interrupted_instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )
        still_running_run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents._QA_PANEL_LANES[1].agent_kind,
            thread_id=still_running_instance.thread_id,
            instance=still_running_instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )
        def interrupt_side_effect(
            _instance_id: int, *, expected_thread_id: str
        ) -> CodexInstance | None:
            if expected_thread_id == interrupted_instance.thread_id:
                return interrupted_instance
            return None

        mock_interrupt.side_effect = interrupt_side_effect

        stopped = system_agents.stop_active_workflow("main-thread")

        self.assertTrue(stopped)
        interrupted_run.refresh_from_db()
        still_running_run.refresh_from_db()
        workflow.refresh_from_db()
        self.assertEqual(interrupted_run.status, SystemAgentRun.STATUS_FAILED)
        self.assertEqual(still_running_run.status, SystemAgentRun.STATUS_RUNNING)
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

    def test_pr_monitor_output_rejects_boolean_numeric_handoff_fields(self) -> None:
        parsed = system_agents._parse_pr_monitor_output(
            json.dumps(
                {
                    "status": "blocked",
                    "summary": "CI pending.",
                    "feedback": "Wait for CI.",
                    "pr": {
                        "url": "https://github.com/cberner/hitch/pull/169",
                        "pr_number": True,
                        "merged": False,
                        "review_count": True,
                    },
                    "blockers": ["CI pending"],
                }
            )
        )

        self.assertIsNotNone(parsed)
        assert parsed is not None
        pr = parsed["pr"]
        self.assertEqual(pr["url"], "https://github.com/cberner/hitch/pull/169")
        self.assertIs(pr["merged"], False)
        self.assertNotIn("pr_number", pr)
        self.assertNotIn("review_count", pr)


class StandingOrderWorkflowTests(TestCase):
    def test_standing_order_candidate_parser_accepts_wrapped_proposal(self) -> None:
        parsed = system_agents._parse_standing_order_candidate_output(
            json.dumps(
                {
                    "proposal": {
                        "title": "Add parser coverage",
                        "summary": "Cover parser edge cases.",
                        "impact": "Fewer regressions.",
                        "implementation_direction": "Add focused tests.",
                        "relevant_files": ["hitch/main/rollout.py"],
                    },
                    "message": "",
                    "next_steps_summary": (
                        "Proposed hitch/main/rollout.py; try parser edges next."
                    ),
                    "memory_relevant_files": ["hitch/main/rollout.py"],
                }
            )
        )

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["proposal"]["title"], "Add parser coverage")
        self.assertEqual(parsed["message"], "")
        self.assertEqual(
            parsed["next_steps_summary"],
            "Proposed hitch/main/rollout.py; try parser edges next.",
        )
        self.assertEqual(parsed["memory_relevant_files"], ["hitch/main/rollout.py"])

    def test_standing_order_candidate_parser_rejects_invalid_wrapped_output(
        self,
    ) -> None:
        self.assertIsNone(
            system_agents._parse_standing_order_candidate_output(
                json.dumps({"proposal": None, "message": "   "})
            )
        )
        self.assertIsNone(
            system_agents._parse_standing_order_candidate_output(
                json.dumps({"proposal": "not an object", "message": ""})
            )
        )
        self.assertIsNone(
            system_agents._parse_standing_order_candidate_output(
                json.dumps({"proposal": {"title": ""}, "message": ""})
            )
        )
        self.assertIsNone(
            system_agents._parse_standing_order_candidate_output(
                json.dumps({"title": "", "summary": "", "impact": ""})
            )
        )

    @patch("hitch.main.system_agents.codex_pool.spawn_new_session")
    def test_workflow_starts_hidden_candidate_thread(
        self, mock_spawn: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        standing_order = StandingOrder.objects.create(
            project=project,
            title="Keep docs current",
            goal="Find small documentation improvements.",
            ambition=StandingOrder.AMBITION_HIGH,
            confidence_threshold=StandingOrder.CONFIDENCE_HIGH,
        )
        mock_spawn.return_value = _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.STANDING_ORDER_AGENT_KIND,
        )

        workflow = system_agents.start_standing_order_workflow(
            standing_order=standing_order
        )

        self.assertEqual(workflow.step, system_agents.STEP_STANDING_ORDER_CANDIDATE_RUNNING)
        kwargs = mock_spawn.call_args.kwargs
        self.assertEqual(kwargs["cwd"], "/repo")
        self.assertEqual(kwargs["approval_mode"], system_agents.SYSTEM_AGENT_APPROVAL_MODE)
        self.assertEqual(kwargs["agent_kind"], system_agents.STANDING_ORDER_AGENT_KIND)
        self.assertEqual(kwargs["display_author"], system_agents.STANDING_ORDER_DISPLAY_AUTHOR)
        schema = kwargs["output_schema"]
        self.assertEqual(
            schema["required"],
            [
                "proposal",
                "message",
                "next_steps_summary",
                "memory_relevant_files",
            ],
        )
        self.assertEqual(schema["properties"]["proposal"]["type"], ["object", "null"])
        self.assertEqual(schema["properties"]["message"]["type"], "string")
        self.assertEqual(
            schema["properties"]["next_steps_summary"]["type"], "string"
        )
        self.assertEqual(
            schema["properties"]["memory_relevant_files"]["type"], "array"
        )
        self.assertIn("Keep docs current", kwargs["prompt"])
        self.assertIn("make high progress", kwargs["prompt"])
        self.assertIn('"proposal" to null', kwargs["prompt"])
        self.assertIn("Standing order memory from previous candidate runs", kwargs["prompt"])
        self.assertIn("next_steps_summary", kwargs["prompt"])
        self.assertTrue(
            SessionMetadata.objects.filter(thread_id="candidate-thread").exists()
        )

    @patch("hitch.main.system_agents.codex_pool.spawn_new_session")
    def test_yolo_workflow_starts_candidate_thread_with_yolo_guidance(
        self, mock_spawn: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        standing_order = StandingOrder.objects.create(
            project=project,
            title="Keep docs current",
            goal="Find substantial documentation improvements.",
            ambition=StandingOrder.AMBITION_YOLO,
        )
        mock_spawn.return_value = _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.STANDING_ORDER_AGENT_KIND,
        )

        system_agents.start_standing_order_workflow(standing_order=standing_order)

        prompt = mock_spawn.call_args.kwargs["prompt"]
        self.assertIn("bold, high-leverage progress", prompt)
        self.assertIn("substantial session", prompt)
        self.assertNotIn("incremental", prompt.lower())

    @patch("hitch.main.system_agents.codex_pool.spawn_new_session")
    def test_candidate_prompt_includes_prior_memory(
        self, mock_spawn: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        standing_order = StandingOrder.objects.create(
            project=project,
            title="Process one test file",
            goal="Pick one test file and improve it.",
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread-old",
            cwd="/repo",
            project=project,
        )
        StandingOrderMemory.objects.create(
            standing_order=standing_order,
            candidate_session=candidate,
            title="Processed rollout tests",
            summary=(
                "Selected hitch/main/test/test_rollout.py; next try a different "
                "test file."
            ),
            relevant_files=["hitch/main/test/test_rollout.py"],
        )
        mock_spawn.return_value = _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.STANDING_ORDER_AGENT_KIND,
        )

        workflow = system_agents.start_standing_order_workflow(
            standing_order=standing_order
        )

        prompt = mock_spawn.call_args.kwargs["prompt"]
        self.assertIn("Standing order memory from previous candidate runs", prompt)
        self.assertIn("Processed rollout tests", prompt)
        self.assertIn("hitch/main/test/test_rollout.py", prompt)
        run = SystemAgentRun.objects.get(workflow=workflow)
        self.assertEqual(run.input["memory_count"], 1)
        self.assertFalse(run.input["memory_compacted"])

    @patch("hitch.main.system_agents.codex_pool.spawn_new_session")
    @patch.object(system_agents, "_STANDING_ORDER_MEMORY_CONTEXT_CHARS", 350)
    def test_candidate_prompt_compacts_large_prior_memory(
        self, mock_spawn: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        standing_order = StandingOrder.objects.create(
            project=project,
            title="Process one test file",
            goal="Pick one test file and improve it.",
        )
        for idx in range(4):
            StandingOrderMemory.objects.create(
                standing_order=standing_order,
                title=f"Processed test file {idx}",
                summary=(
                    (
                        f"Selected hitch/main/test/test_{idx}.py and completed a "
                        "focused pass. Future runs should choose a different file. "
                    )
                    * 6
                ),
                relevant_files=[f"hitch/main/test/test_{idx}.py"],
            )
        mock_spawn.return_value = _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.STANDING_ORDER_AGENT_KIND,
        )

        workflow = system_agents.start_standing_order_workflow(
            standing_order=standing_order
        )

        prompt = mock_spawn.call_args.kwargs["prompt"]
        self.assertIn("Compacted from 4 prior candidate summaries", prompt)
        self.assertIn("Files seen across prior runs", prompt)
        self.assertIn("hitch/main/test/test_3.py", prompt)
        memory_context = system_agents._standing_order_memory_context(standing_order)
        self.assertLessEqual(
            len(memory_context.text), system_agents._STANDING_ORDER_MEMORY_CONTEXT_CHARS
        )
        run = SystemAgentRun.objects.get(workflow=workflow)
        self.assertEqual(run.input["memory_count"], 4)
        self.assertTrue(run.input["memory_compacted"])

    @patch.object(system_agents, "_STANDING_ORDER_MEMORY_CONTEXT_CHARS", 240)
    def test_compacted_memory_context_enforces_budget_with_long_files(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        standing_order = StandingOrder.objects.create(
            project=project,
            title="Process one test file",
            goal="Pick one test file and improve it.",
        )
        for idx in range(5):
            StandingOrderMemory.objects.create(
                standing_order=standing_order,
                title=f"Processed file {idx}",
                summary="Chose one file and left a long next-step summary. " * 12,
                relevant_files=[
                    "hitch/main/test/"
                    + ("very_long_path_segment_" * 8)
                    + f"{idx}.py"
                ],
            )

        memory_context = system_agents._standing_order_memory_context(standing_order)

        self.assertTrue(memory_context.compacted)
        self.assertIn("Compacted from 5 prior candidate summaries", memory_context.text)
        self.assertLessEqual(
            len(memory_context.text), system_agents._STANDING_ORDER_MEMORY_CONTEXT_CHARS
        )

    @patch.object(system_agents, "_STANDING_ORDER_MEMORY_MAX_ROWS", 2)
    def test_memory_context_caps_recent_rows_before_compaction(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        standing_order = StandingOrder.objects.create(
            project=project,
            title="Process one test file",
            goal="Pick one test file and improve it.",
        )
        for idx in range(4):
            StandingOrderMemory.objects.create(
                standing_order=standing_order,
                title=f"Processed file {idx}",
                summary=f"Summary for file {idx}.",
                relevant_files=[f"hitch/main/test/test_{idx}.py"],
            )

        memory_context = system_agents._standing_order_memory_context(standing_order)

        self.assertTrue(memory_context.compacted)
        self.assertEqual(memory_context.count, 4)
        self.assertIn("Compacted from 4 prior candidate summaries", memory_context.text)
        self.assertIn("2 older memory rows are outside this prompt cap", memory_context.text)
        self.assertIn("Processed file 3", memory_context.text)
        self.assertNotIn("Processed file 0", memory_context.text)

    @patch("hitch.main.system_agents.codex_pool.spawn_new_session")
    def test_candidate_completion_starts_judge_thread(
        self, mock_spawn: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        standing_order = StandingOrder.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.STANDING_ORDER_AGENT_KIND,
            main_thread_id=system_agents._standing_order_main_thread_id(
                standing_order.pk
            ),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_STANDING_ORDER_CANDIDATE_RUNNING,
            state={"standing_order_id": standing_order.pk},
        )
        candidate_metadata = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo",
            project=project,
        )
        workflow.state = {
            **workflow.state,
            "candidate_session_id": candidate_metadata.pk,
        }
        workflow.save(update_fields=["state", "updated_at"])
        instance = _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            events_path=_events_file(
                self,
                {
                    "title": "Add parser coverage",
                    "summary": "Cover parser edge cases.",
                    "impact": "Fewer regressions.",
                    "implementation_direction": "Add focused tests.",
                    "relevant_files": ["hitch/main/rollout.py"],
                    "next_steps_summary": (
                        "Selected hitch/main/rollout.py for parser coverage; "
                        "try adjacent rollout tests after this."
                    ),
                    "memory_relevant_files": [
                        "hitch/main/rollout.py",
                        "hitch/main/test/test_rollout.py",
                    ],
                },
            ),
            agent_kind=system_agents.STANDING_ORDER_AGENT_KIND,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.STANDING_ORDER_AGENT_KIND,
            thread_id="candidate-thread",
            instance=instance,
        )
        mock_spawn.return_value = _instance(
            thread_id="judge-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.STANDING_ORDER_JUDGE_AGENT_KIND,
        )

        system_agents.on_codex_instance_finished(instance)

        workflow.refresh_from_db()
        self.assertEqual(workflow.step, system_agents.STEP_STANDING_ORDER_JUDGE_RUNNING)
        self.assertEqual(workflow.state["candidate"]["title"], "Add parser coverage")
        memory = StandingOrderMemory.objects.get()
        self.assertEqual(memory.standing_order, standing_order)
        self.assertEqual(memory.candidate_session, candidate_metadata)
        self.assertEqual(memory.title, "Add parser coverage")
        self.assertEqual(
            memory.summary,
            "Selected hitch/main/rollout.py for parser coverage; try adjacent rollout tests after this.",
        )
        self.assertEqual(
            memory.relevant_files,
            ["hitch/main/rollout.py", "hitch/main/test/test_rollout.py"],
        )
        kwargs = mock_spawn.call_args.kwargs
        self.assertEqual(kwargs["agent_kind"], system_agents.STANDING_ORDER_JUDGE_AGENT_KIND)
        self.assertIn("Add parser coverage", kwargs["prompt"])
        self.assertTrue(SessionMetadata.objects.filter(thread_id="judge-thread").exists())

    @patch("hitch.main.system_agents.codex_pool.spawn_new_session")
    def test_candidate_completion_creates_notice_when_no_proposal(
        self, mock_spawn: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        standing_order = StandingOrder.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        candidate_metadata = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo",
            project=project,
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.STANDING_ORDER_AGENT_KIND,
            main_thread_id=system_agents._standing_order_main_thread_id(
                standing_order.pk
            ),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_STANDING_ORDER_CANDIDATE_RUNNING,
            state={
                "standing_order_id": standing_order.pk,
                "candidate_session_id": candidate_metadata.pk,
            },
        )
        instance = _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            events_path=_events_file(
                self,
                {
                    "proposal": None,
                    "message": "No concrete test increment was worth proposing.",
                    "next_steps_summary": (
                        "Inspected rollout tests and found no clear increment; "
                        "try settings tests next."
                    ),
                    "memory_relevant_files": ["hitch/main/test/test_rollout.py"],
                },
            ),
            agent_kind=system_agents.STANDING_ORDER_AGENT_KIND,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.STANDING_ORDER_AGENT_KIND,
            thread_id="candidate-thread",
            instance=instance,
        )

        system_agents.on_codex_instance_finished(instance)

        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(workflow.step, system_agents.STEP_STANDING_ORDER_SKIPPED)
        notice = ProposedSession.objects.get()
        self.assertEqual(notice.inbox_kind, ProposedSession.INBOX_KIND_NOTICE)
        self.assertEqual(notice.title, "No proposal from Improve tests")
        self.assertEqual(
            notice.summary, "No concrete test increment was worth proposing."
        )
        self.assertEqual(notice.candidate_session, candidate_metadata)
        memory = StandingOrderMemory.objects.get()
        self.assertEqual(memory.title, "No proposal from Improve tests")
        self.assertEqual(
            memory.summary,
            "Inspected rollout tests and found no clear increment; try settings tests next.",
        )
        self.assertEqual(memory.relevant_files, ["hitch/main/test/test_rollout.py"])
        mock_spawn.assert_not_called()

    @patch("hitch.main.system_agents.codex_pool.spawn_new_session")
    def test_yolo_candidate_completion_starts_judge_thread_with_yolo_guidance(
        self, mock_spawn: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        standing_order = StandingOrder.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            ambition=StandingOrder.AMBITION_YOLO,
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.STANDING_ORDER_AGENT_KIND,
            main_thread_id=system_agents._standing_order_main_thread_id(
                standing_order.pk
            ),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_STANDING_ORDER_CANDIDATE_RUNNING,
            state={"standing_order_id": standing_order.pk},
        )
        candidate_metadata = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo",
            project=project,
        )
        workflow.state = {
            **workflow.state,
            "candidate_session_id": candidate_metadata.pk,
        }
        workflow.save(update_fields=["state", "updated_at"])
        instance = _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            events_path=_events_file(
                self,
                {
                    "title": "Consolidate command tests",
                    "summary": "Merge duplicated command-routing tests.",
                    "impact": "Less duplicated test maintenance.",
                    "implementation_direction": "Refactor adjacent tests.",
                    "relevant_files": ["hitch/main/test/test_views.py"],
                },
            ),
            agent_kind=system_agents.STANDING_ORDER_AGENT_KIND,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.STANDING_ORDER_AGENT_KIND,
            thread_id="candidate-thread",
            instance=instance,
        )
        mock_spawn.return_value = _instance(
            thread_id="judge-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.STANDING_ORDER_JUDGE_AGENT_KIND,
        )

        system_agents.on_codex_instance_finished(instance)

        prompt = mock_spawn.call_args.kwargs["prompt"]
        self.assertIn("bold, high-leverage progress", prompt)
        self.assertIn("substantial and high-upside", prompt)
        self.assertNotIn("incremental", prompt.lower())

    def test_judge_creates_proposal_when_confidence_meets_threshold(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        standing_order = StandingOrder.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            confidence_threshold=StandingOrder.CONFIDENCE_HIGH,
        )
        candidate_metadata = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo",
            project=project,
        )
        judge_metadata = SessionMetadata.objects.create(
            thread_id="judge-thread",
            cwd="/repo",
            project=project,
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.STANDING_ORDER_AGENT_KIND,
            main_thread_id=system_agents._standing_order_main_thread_id(
                standing_order.pk
            ),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_STANDING_ORDER_JUDGE_RUNNING,
            state={
                "standing_order_id": standing_order.pk,
                "candidate_session_id": candidate_metadata.pk,
                "judge_session_id": judge_metadata.pk,
                "candidate": {
                    "title": "Add parser coverage",
                    "implementation_direction": (
                        "Add focused rollout parser regression tests before "
                        "touching parser behavior."
                    ),
                    "relevant_files": ["hitch/main/rollout.py"],
                },
            },
        )
        instance = _instance(
            thread_id="judge-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            events_path=_events_file(
                self,
                {
                    "confidence": "high",
                    "summary": "This adds focused parser coverage.",
                    "rationale": "The files are well-scoped.",
                },
            ),
            agent_kind=system_agents.STANDING_ORDER_JUDGE_AGENT_KIND,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.STANDING_ORDER_JUDGE_AGENT_KIND,
            thread_id="judge-thread",
            instance=instance,
        )

        system_agents.on_codex_instance_finished(instance)

        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(workflow.step, system_agents.STEP_STANDING_ORDER_PROPOSED)
        proposal = ProposedSession.objects.get()
        self.assertEqual(proposal.title, "Add parser coverage")
        self.assertEqual(proposal.confidence, StandingOrder.CONFIDENCE_HIGH)
        self.assertIn("Implementation guidance:", proposal.prompt)
        self.assertIn(
            "Add focused rollout parser regression tests before touching parser behavior.",
            proposal.prompt,
        )
        self.assertEqual(proposal.candidate_session, candidate_metadata)
        self.assertEqual(proposal.judge_session, judge_metadata)

    @patch("hitch.main.system_agents.codex_pool.spawn_new_session")
    def test_draft_patch_autonomy_starts_implementation_session(
        self, mock_spawn: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        standing_order = StandingOrder.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            confidence_threshold=StandingOrder.CONFIDENCE_HIGH,
            autonomy=StandingOrder.AUTONOMY_DRAFT_PATCH,
        )
        candidate_metadata = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo",
            project=project,
        )
        judge_metadata = SessionMetadata.objects.create(
            thread_id="judge-thread",
            cwd="/repo",
            project=project,
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.STANDING_ORDER_AGENT_KIND,
            main_thread_id=system_agents._standing_order_main_thread_id(
                standing_order.pk
            ),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_STANDING_ORDER_JUDGE_RUNNING,
            state={
                "standing_order_id": standing_order.pk,
                "candidate_session_id": candidate_metadata.pk,
                "judge_session_id": judge_metadata.pk,
                "candidate": {
                    "title": "Add parser coverage",
                    "implementation_direction": "Add focused tests.",
                    "relevant_files": ["hitch/main/rollout.py"],
                },
            },
        )
        instance = _instance(
            thread_id="judge-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            events_path=_events_file(
                self,
                {
                    "confidence": "high",
                    "summary": "This adds focused parser coverage.",
                    "rationale": "The files are well-scoped.",
                },
            ),
            agent_kind=system_agents.STANDING_ORDER_JUDGE_AGENT_KIND,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.STANDING_ORDER_JUDGE_AGENT_KIND,
            thread_id="judge-thread",
            instance=instance,
        )
        mock_spawn.return_value = _instance(
            thread_id="implementation-thread",
            purpose=CodexInstance.PURPOSE_USER,
        )

        system_agents.on_codex_instance_finished(instance)

        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(
            workflow.step, system_agents.STEP_STANDING_ORDER_DRAFT_STARTED
        )
        proposal = ProposedSession.objects.get()
        implementation = SessionMetadata.objects.get(thread_id="implementation-thread")
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_ACCEPTED)
        self.assertEqual(proposal.accepted_session, implementation)
        self.assertEqual(
            proposal.outcome_metadata["standing_order_autonomy"],
            StandingOrder.AUTONOMY_DRAFT_PATCH,
        )
        self.assertEqual(
            proposal.outcome_metadata["automation_status"],
            "implementation_started",
        )
        self.assertFalse(proposal.outcome_metadata["auto_pr_enabled"])
        self.assertFalse(implementation.auto_pr_enabled)
        kwargs = mock_spawn.call_args.kwargs
        self.assertEqual(kwargs["cwd"], "/repo")
        self.assertEqual(kwargs["thread_name"], "Add parser coverage")
        self.assertEqual(kwargs["approval_mode"], system_agents.SYSTEM_AGENT_APPROVAL_MODE)
        self.assertEqual(
            kwargs["sandbox_policy"],
            system_agents.STANDING_ORDER_IMPLEMENTATION_SANDBOX_POLICY,
        )
        self.assertFalse(kwargs["auto_pr_enabled"])
        self.assertIn("Implementation guidance:\nAdd focused tests.", kwargs["prompt"])

    @patch("hitch.main.system_agents.codex_pool.spawn_new_session")
    def test_draft_pr_autonomy_enables_auto_pr_for_implementation(
        self, mock_spawn: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        standing_order = StandingOrder.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            confidence_threshold=StandingOrder.CONFIDENCE_HIGH,
            autonomy=StandingOrder.AUTONOMY_DRAFT_PR,
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.STANDING_ORDER_AGENT_KIND,
            main_thread_id=system_agents._standing_order_main_thread_id(
                standing_order.pk
            ),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_STANDING_ORDER_JUDGE_RUNNING,
            state={
                "standing_order_id": standing_order.pk,
                "candidate": {
                    "title": "Add parser coverage",
                    "implementation_direction": "Add focused tests.",
                    "relevant_files": [],
                },
            },
        )
        instance = _instance(
            thread_id="judge-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            events_path=_events_file(
                self,
                {
                    "confidence": "high",
                    "summary": "This adds focused parser coverage.",
                    "rationale": "The files are well-scoped.",
                },
            ),
            agent_kind=system_agents.STANDING_ORDER_JUDGE_AGENT_KIND,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.STANDING_ORDER_JUDGE_AGENT_KIND,
            thread_id="judge-thread",
            instance=instance,
        )
        mock_spawn.return_value = _instance(
            thread_id="implementation-thread",
            purpose=CodexInstance.PURPOSE_USER,
            auto_pr_enabled=True,
        )

        system_agents.on_codex_instance_finished(instance)

        proposal = ProposedSession.objects.get()
        implementation = SessionMetadata.objects.get(thread_id="implementation-thread")
        self.assertTrue(mock_spawn.call_args.kwargs["auto_pr_enabled"])
        self.assertTrue(implementation.auto_pr_enabled)
        self.assertTrue(proposal.outcome_metadata["auto_pr_enabled"])
        self.assertIn("Auto-PR will run", proposal.outcome_notes)

    @patch("hitch.main.system_agents.start_pr_qa_workflow")
    def test_draft_pr_implementation_completion_records_pr_workflow(
        self, mock_start: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        implementation = SessionMetadata.objects.create(
            thread_id="implementation-thread",
            cwd="/repo",
            project=project,
            auto_pr_enabled=True,
        )
        proposal = ProposedSession.objects.create(
            project=project,
            title="Add parser coverage",
            accepted_session=implementation,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            outcome_metadata={"standing_order_autonomy": StandingOrder.AUTONOMY_DRAFT_PR},
        )
        pr_workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="implementation-thread",
            cwd="/repo",
        )
        mock_start.return_value = pr_workflow
        instance = _instance(
            thread_id="implementation-thread",
            auto_pr_enabled=True,
            user_message_index=0,
        )

        system_agents.on_codex_instance_finished(instance)

        mock_start.assert_called_once()
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_metadata["auto_pr_status"], "started")
        self.assertEqual(
            proposal.outcome_metadata["auto_pr_workflow_id"], pr_workflow.pk
        )

    @patch("hitch.main.system_agents.codex_pool.spawn_new_session")
    def test_draft_patch_start_failure_leaves_visible_proposal(
        self, mock_spawn: MagicMock
    ) -> None:
        mock_spawn.side_effect = RuntimeError("app-server unavailable")
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        standing_order = StandingOrder.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            confidence_threshold=StandingOrder.CONFIDENCE_HIGH,
            autonomy=StandingOrder.AUTONOMY_DRAFT_PATCH,
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.STANDING_ORDER_AGENT_KIND,
            main_thread_id=system_agents._standing_order_main_thread_id(
                standing_order.pk
            ),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_STANDING_ORDER_JUDGE_RUNNING,
            state={
                "standing_order_id": standing_order.pk,
                "candidate": {
                    "title": "Add parser coverage",
                    "implementation_direction": "Add focused tests.",
                    "relevant_files": [],
                },
            },
        )
        instance = _instance(
            thread_id="judge-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            events_path=_events_file(
                self,
                {
                    "confidence": "high",
                    "summary": "This adds focused parser coverage.",
                    "rationale": "The files are well-scoped.",
                },
            ),
            agent_kind=system_agents.STANDING_ORDER_JUDGE_AGENT_KIND,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.STANDING_ORDER_JUDGE_AGENT_KIND,
            thread_id="judge-thread",
            instance=instance,
        )

        system_agents.on_codex_instance_finished(instance)

        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
        proposal = ProposedSession.objects.get()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_UNSET)
        self.assertIsNone(proposal.accepted_session)
        self.assertIn("failed to start implementation session", proposal.outcome_notes)
        self.assertEqual(
            proposal.outcome_metadata["automation_status"],
            "implementation_start_failed",
        )

    def test_candidate_failure_creates_visible_notice(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        standing_order = StandingOrder.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        candidate_metadata = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo",
            project=project,
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.STANDING_ORDER_AGENT_KIND,
            main_thread_id=system_agents._standing_order_main_thread_id(
                standing_order.pk
            ),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_STANDING_ORDER_CANDIDATE_RUNNING,
            state={
                "standing_order_id": standing_order.pk,
                "candidate_session_id": candidate_metadata.pk,
            },
        )
        instance = _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            events_path=_raw_events_file(
                self,
                [
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
                ],
            ),
            agent_kind=system_agents.STANDING_ORDER_AGENT_KIND,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.STANDING_ORDER_AGENT_KIND,
            thread_id="candidate-thread",
            instance=instance,
        )

        system_agents.on_codex_instance_finished(instance)

        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
        notice = ProposedSession.objects.get()
        self.assertEqual(notice.inbox_kind, ProposedSession.INBOX_KIND_NOTICE)
        self.assertEqual(notice.candidate_session, candidate_metadata)
        self.assertIn("Standing order failed: Improve tests", notice.title)
        self.assertIn("candidate output was not valid JSON", notice.summary)

    def test_judge_skips_proposal_below_threshold(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        standing_order = StandingOrder.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            confidence_threshold=StandingOrder.CONFIDENCE_VERY_HIGH,
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.STANDING_ORDER_AGENT_KIND,
            main_thread_id=system_agents._standing_order_main_thread_id(
                standing_order.pk
            ),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_STANDING_ORDER_JUDGE_RUNNING,
            state={
                "standing_order_id": standing_order.pk,
                "candidate": {"title": "Maybe add tests", "relevant_files": []},
            },
        )
        instance = _instance(
            thread_id="judge-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            events_path=_events_file(
                self,
                {
                    "confidence": "high",
                    "summary": "Useful but not certain.",
                    "rationale": "There is some ambiguity.",
                },
            ),
            agent_kind=system_agents.STANDING_ORDER_JUDGE_AGENT_KIND,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.STANDING_ORDER_JUDGE_AGENT_KIND,
            thread_id="judge-thread",
            instance=instance,
        )

        system_agents.on_codex_instance_finished(instance)

        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(workflow.step, system_agents.STEP_STANDING_ORDER_SKIPPED)
        self.assertFalse(ProposedSession.objects.exists())

    def test_accepted_proposed_session_unhides_candidate_thread(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        standing_order = StandingOrder.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        metadata = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo",
            project=project,
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.STANDING_ORDER_AGENT_KIND,
            main_thread_id=system_agents._standing_order_main_thread_id(
                standing_order.pk
            ),
            cwd="/repo",
        )
        instance = _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.STANDING_ORDER_AGENT_KIND,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.STANDING_ORDER_AGENT_KIND,
            thread_id="candidate-thread",
            instance=instance,
        )

        self.assertIn("candidate-thread", system_agents.hidden_thread_ids())

        ProposedSession.objects.create(
            standing_order=standing_order,
            candidate_session=metadata,
            accepted_session=metadata,
            title="Add parser coverage",
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
        )

        self.assertNotIn("candidate-thread", system_agents.hidden_thread_ids())

    def test_proposed_session_accepted_into_new_thread_keeps_candidate_hidden(
        self,
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        standing_order = StandingOrder.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo",
            project=project,
        )
        accepted = SessionMetadata.objects.create(
            thread_id="implementation-thread",
            cwd="/repo",
            project=project,
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.STANDING_ORDER_AGENT_KIND,
            main_thread_id=system_agents._standing_order_main_thread_id(
                standing_order.pk
            ),
            cwd="/repo",
        )
        instance = _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.STANDING_ORDER_AGENT_KIND,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.STANDING_ORDER_AGENT_KIND,
            thread_id="candidate-thread",
            instance=instance,
        )
        ProposedSession.objects.create(
            standing_order=standing_order,
            candidate_session=candidate,
            accepted_session=accepted,
            title="Add parser coverage",
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
        )

        hidden_ids = system_agents.hidden_thread_ids()
        self.assertIn("candidate-thread", hidden_ids)
        self.assertNotIn("implementation-thread", hidden_ids)
