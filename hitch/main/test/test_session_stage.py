from django.test import SimpleTestCase

from hitch.main import session_stage, system_agents
from hitch.main.models import SystemWorkflow


class SessionStageTests(SimpleTestCase):
    def test_pr_qa_workflow_steps_map_to_display_stage(self) -> None:
        cases = {
            system_agents.STEP_LOCAL_BRANCH_MERGED: session_stage.DONE_MERGED,
            system_agents.STEP_QA_RUNNING: session_stage.QA,
            system_agents.STEP_QA_APPROVED: session_stage.QA,
            system_agents.STEP_FEEDBACK_RUNNING: session_stage.IMPLEMENTATION,
            system_agents.STEP_USER_STEERING_RUNNING: session_stage.IMPLEMENTATION,
            system_agents.STEP_PR_FEEDBACK_RUNNING: session_stage.IMPLEMENTATION,
            system_agents.STEP_PR_PROMPT_SPAWNED: session_stage.PR,
            system_agents.STEP_PR_PROMPT_RUNNING: session_stage.PR,
            system_agents.STEP_PR_MONITORING: session_stage.PR,
            system_agents.STEP_PR_READY: session_stage.PR,
            system_agents.STEP_PR_CLOSED: session_stage.PR,
            system_agents.STEP_BLOCKED: session_stage.BLOCKED,
        }

        for step, expected_stage in cases.items():
            with self.subTest(step=step):
                workflow = SystemWorkflow(
                    kind=SystemWorkflow.KIND_PR_QA,
                    status=SystemWorkflow.STATUS_RUNNING,
                    step=step,
                )

                self.assertEqual(session_stage.derive_stage(workflow=workflow), expected_stage)

    def test_approval_declined_after_plan_stays_plan_stage(self) -> None:
        stage = session_stage.derive_stage(
            entries=[
                {"kind": "user", "text": "Plan it"},
                {"kind": "plan", "text": "# Plan"},
                {"kind": "user", "text": "Try command"},
                {"kind": "approval_declined", "detail": "git push"},
            ],
        )

        self.assertEqual(stage, session_stage.PLAN)

    def test_trailing_commentary_after_plan_stays_plan_stage(self) -> None:
        # The session-list stage runs against raw, un-collapsed rollout
        # entries, where Codex's post-plan narration keeps ``kind="agent"``
        # with a ``commentary`` phase. That intermediate narration must not be
        # mistaken for a final reply that resolves the plan, or the list badge
        # would disagree with the detail view's Approve/Revise card.
        stage = session_stage.derive_stage(
            entries=[
                {"kind": "user", "text": "Plan it"},
                {"kind": "plan", "text": "# Plan"},
                {
                    "kind": "agent",
                    "text": "Let me know if you'd like changes.",
                    "phase": "commentary",
                },
            ],
        )

        self.assertEqual(stage, session_stage.PLAN)

    def test_workflow_pr_identity_wins_over_terminal_log_snapshot(self) -> None:
        workflow = SystemWorkflow(
            kind=SystemWorkflow.KIND_PR_QA,
            status=SystemWorkflow.STATUS_COMPLETED,
            step=system_agents.STEP_PR_READY,
        )

        stage = session_stage.derive_stage(
            entries=[{"kind": "user"}],
            workflow=workflow,
            pr_snapshot={
                "url": "https://github.com/cberner/hitch/pull/93",
                "state": "closed",
            },
            workflow_pr_snapshot={
                "url": "https://github.com/cberner/hitch/pull/94",
                "state": "open",
            },
        )

        self.assertEqual(stage, session_stage.PR)
