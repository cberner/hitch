from pathlib import Path
from unittest.mock import Mock, patch

from django.test import SimpleTestCase

from hitch.main.models import CodexInstance, SessionPullRequest
from hitch.main.sessions import agent_tasks, session_stage
from hitch.main.sessions.stage_resolution import StageCache, pr_display_state, read_stage_history, resolve_stage


class StageResolutionTests(SimpleTestCase):
    def test_pr_visibility_requires_a_current_handoff_owned_by_the_publisher(self) -> None:
        record = SessionPullRequest(state={"pr_handoff": {"url": "https://github.com/cberner/hitch/pull/94"}})
        self.assertFalse(pr_display_state(None, None).has_registered_pr)
        self.assertFalse(pr_display_state(SessionPullRequest(state={"unrelated": True}), None).has_registered_pr)
        self.assertTrue(pr_display_state(record, None).has_registered_pr)
        publisher = CodexInstance(pk=8, agent_kind=agent_tasks.PR_PUBLISH_AGENT_KIND)
        self.assertFalse(pr_display_state(record, publisher).has_registered_pr)
        record.state[SessionPullRequest.WATCH_OWNER_INSTANCE_STATE_KEY] = publisher.pk
        self.assertTrue(pr_display_state(record, publisher).has_registered_pr)
        record.state[SessionPullRequest.SUPERSEDED_BY_INSTANCE_STATE_KEY] = 9
        self.assertIsNone(pr_display_state(record, None).record)

    def test_live_state_overrides_cache_and_never_reads_history(self) -> None:
        pr = pr_display_state(SessionPullRequest(state={
            "pr_handoff": {"state": "closed", "url": "https://example.com/pr"},
        }), None)
        worker = CodexInstance(agent_kind="", plan_mode=False)
        for active, waiting, expected in (
            (None, True, session_stage.AWAITING_INPUT),
            (worker, False, session_stage.IMPLEMENTATION),
            (CodexInstance(plan_mode=True), False, session_stage.PLAN),
            (CodexInstance(agent_kind=agent_tasks.REVIEW_AGENT_KIND), False, session_stage.QA),
            (CodexInstance(agent_kind=agent_tasks.PR_WATCH_AGENT_KIND), False, session_stage.PR),
            (None, False, session_stage.DONE_CLOSED),
        ):
            for complete in (True, False):
                with self.subTest(stage=expected.key, complete=complete):
                    history = Mock(side_effect=AssertionError("history must not be read"))
                    entries = (history() for _ in range(1))
                    result = resolve_stage(
                        entries=entries, history_complete=complete, pr=pr,
                        active_instance=active, awaiting_user_input=waiting,
                        cache=StageCache("new", 10), source_mtime_ns=10,
                    )
                    self.assertEqual(result.stage, expected)
                    self.assertEqual(result.should_persist, complete and active is None and not waiting)
                    history.assert_not_called()

    def test_only_fresh_rollout_stages_can_replace_partial_history(self) -> None:
        for key, saved_mtime, current_mtime, expected in (
            ("new", 10, 10, session_stage.NEW),
            ("plan", 10, 10, session_stage.PLAN),
            ("implementation", 10, 10, session_stage.IMPLEMENTATION),
            ("qa", 10, 10, session_stage.QA),
            ("plan", 9, 10, session_stage.IMPLEMENTATION),
            ("plan", 0, None, session_stage.IMPLEMENTATION),
            ("done_closed", 10, 10, session_stage.IMPLEMENTATION),
            ("pr", 10, 10, session_stage.IMPLEMENTATION),
            ("awaiting_input", 10, 10, session_stage.IMPLEMENTATION),
            ("invalid", 10, 10, session_stage.IMPLEMENTATION),
        ):
            with self.subTest(key=key, current_mtime=current_mtime):
                result = resolve_stage(
                    entries=[{"kind": "thinking"}], history_complete=False, leading_user_text="Continue",
                    pr=pr_display_state(None, None), cache=StageCache(key, saved_mtime),
                    source_mtime_ns=current_mtime,
                )
                self.assertEqual(result.stage, expected)
                self.assertFalse(result.should_persist)

    def test_partial_history_cannot_be_persisted_without_a_complete_source(self) -> None:
        for complete in (False, True):
            result = resolve_stage(
                entries=[{"kind": "user"}], history_complete=complete, pr=pr_display_state(None, None),
            )
            self.assertEqual(result.stage, session_stage.IMPLEMENTATION)
            self.assertEqual(result.should_persist, complete)

    def test_partial_history_uses_its_leading_user_context_without_caching(self) -> None:
        result = resolve_stage(
            entries=[{"kind": "thinking"}], history_complete=False,
            leading_user_text=agent_tasks.review_task(prepare_pull_request=False).prompt,
            pr=pr_display_state(None, None),
        )
        self.assertEqual(result.stage, session_stage.QA)
        self.assertFalse(result.should_persist)

    def test_unreadable_history_fallback_does_not_poison_the_cache(self) -> None:
        with patch("hitch.main.runtime.rollout.session_stage_data", return_value=None):
            result = resolve_stage(
                entries=(), history_complete=True, pr=pr_display_state(None, None),
                history_loader=lambda: read_stage_history(Path("/unreadable"), has_activity=True),
            )
        self.assertEqual(result.stage, session_stage.IMPLEMENTATION)
        self.assertFalse(result.should_persist)
