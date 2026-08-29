from django.test import SimpleTestCase

from hitch.main.models import SystemWorkflow
from hitch.main.sessions import session_stage
from hitch.main.workflows import system_agents


class SessionStageTests(SimpleTestCase):
    def test_archived_workflow_does_not_derive_blocked_stage(self) -> None:
        # Archived stale-blocked rows must drop out of the Blocked inbox stage.
        workflow = SystemWorkflow(
            kind=SystemWorkflow.KIND_PR_QA,
            status=SystemWorkflow.STATUS_COMPLETED,
            step=system_agents.STEP_ARCHIVED,
        )

        self.assertNotEqual(
            session_stage.derive_stage(workflow=workflow), session_stage.BLOCKED
        )

    def test_qa_guidance_turn_uses_qa_stage_instead_of_pr_stage(self) -> None:
        workflow = SystemWorkflow(
            kind=SystemWorkflow.KIND_PR_QA,
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_PR_PROMPT_RUNNING,
            state={
                "open_pr_on_lgtm": False,
                system_agents.REVIEW_GUIDANCE_STATE_KEY: True,
            },
        )

        self.assertEqual(
            session_stage.derive_stage(workflow=workflow),
            session_stage.QA,
        )
        workflow.state.pop(system_agents.REVIEW_GUIDANCE_STATE_KEY)
        self.assertEqual(
            session_stage.derive_stage(workflow=workflow),
            session_stage.PR,
        )

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

    def test_running_workflow_uses_terminal_main_pr_state_for_same_handoff_pr(
        self,
    ) -> None:
        workflow = SystemWorkflow(
            kind=SystemWorkflow.KIND_PR_QA,
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_PR_PROMPT_RUNNING,
        )

        stage = session_stage.derive_stage(
            entries=[{"kind": "user"}],
            workflow=workflow,
            pr_snapshot={
                "url": "https://github.com/cberner/hitch/pull/93",
                "state": "closed",
                "merged": True,
            },
            workflow_pr_snapshot={
                "url": "https://github.com/cberner/hitch/pull/93",
                "state": "open",
            },
        )

        self.assertEqual(stage, session_stage.DONE_MERGED)
