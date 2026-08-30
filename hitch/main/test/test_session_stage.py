from django.test import SimpleTestCase

from hitch.main.models import CodexInstance
from hitch.main.sessions import agent_tasks, session_stage


def _active_instance(agent_kind: str) -> CodexInstance:
    return CodexInstance(
        pid=1,
        thread_id="thread-1",
        cwd="/repo",
        events_path="/dev/null",
        agent_kind=agent_kind,
    )


class SessionStageTests(SimpleTestCase):
    def test_review_guidance_turn_uses_qa_stage(self) -> None:
        stage = session_stage.derive_stage(
            active_instance=_active_instance(agent_tasks.REVIEW_AGENT_KIND)
        )

        self.assertEqual(stage, session_stage.QA)

    def test_publish_and_watch_turns_use_pr_stage(self) -> None:
        for agent_kind in (
            agent_tasks.PR_PUBLISH_AGENT_KIND,
            agent_tasks.PR_WATCH_AGENT_KIND,
        ):
            with self.subTest(agent_kind=agent_kind):
                stage = session_stage.derive_stage(
                    active_instance=_active_instance(agent_kind)
                )

                self.assertEqual(stage, session_stage.PR)

    def test_completed_review_task_stays_visible_as_qa(self) -> None:
        stage = session_stage.derive_stage(
            entries=[
                {
                    "kind": "user",
                    "text": agent_tasks.review_task(
                        prepare_pull_request=False
                    ).prompt,
                }
            ]
        )

        self.assertEqual(stage, session_stage.QA)

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

    def test_registered_pr_identity_sets_pr_stage(self) -> None:
        stage = session_stage.derive_stage(
            entries=[{"kind": "user"}],
            pr_snapshot={
                "url": "https://github.com/cberner/hitch/pull/94",
                "state": "open",
            },
        )

        self.assertEqual(stage, session_stage.PR)

    def test_terminal_registered_pr_sets_done_stage(self) -> None:
        stage = session_stage.derive_stage(
            entries=[{"kind": "user"}],
            pr_snapshot={
                "url": "https://github.com/cberner/hitch/pull/93",
                "state": "closed",
                "merged": True,
            },
        )

        self.assertEqual(stage, session_stage.DONE_MERGED)
