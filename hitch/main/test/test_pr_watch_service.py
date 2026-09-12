from __future__ import annotations

from collections.abc import Callable
from typing import Any, override
from unittest.mock import MagicMock, patch

from django.test import TestCase

from hitch.main.management.commands import codex_worker
from hitch.main.models import CodexInstance, RefreshThrottle, SessionMetadata, SessionPullRequest
from hitch.main.runtime import maintenance
from hitch.main.runtime.codex_tools import ToolContext, handle_dynamic_tool_call
from hitch.main.sessions import agent_tasks
from hitch.main.test.test_pr_watch import _PR_URL, _observation
from hitch.main.workflows import pr_tracking, pr_watch, pr_watch_service
from hitch.main.workflows.gh_observations import _gh_watch_feedback


class PersistentPrWatchTests(TestCase):
    @override
    def setUp(self) -> None:
        allowed = patch("hitch.main.workflows.pr_watch_service._is_allowed_session_cwd", return_value=True)
        self.mock_allowed = allowed.start()
        self.addCleanup(allowed.stop)
        self.owner = CodexInstance.objects.create(
            pid=0,
            thread_id="watched-thread",
            cwd="/tmp",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            model="preferred-model",
            reasoning_effort="high",
            approval_mode="deny_all",
            sandbox_policy="workspaceWrite",
            web_search_mode="disabled",
            developer_instructions="Personal instructions",
            hitch_extra_instructions="Hitch instructions",
            enable_memories=True,
        )
        registration, _ = pr_tracking.begin_pr_watch_invocation(
            thread_id=self.owner.thread_id,
            cwd="/tmp",
            instance_id=self.owner.pk,
            user_message_index=0,
            agent_kind=agent_tasks.PR_PUBLISH_AGENT_KIND,
            requested_pr={"url": _PR_URL},
        )
        assert registration is not None
        self.registration = registration
        self.record = SessionPullRequest.objects.get(pk=registration.record_id)

    def _due(self) -> None:
        self.record.refresh_from_db()
        RefreshThrottle.objects.filter(key__startswith="pr-watch:").delete()

    def _unwatch(self, url: str = _PR_URL, instance_id: int | None = None) -> dict[str, object]:
        return handle_dynamic_tool_call(
            {"namespace": "hitch", "tool": "unwatch_pr", "arguments": {"url": url}},
            ToolContext(
                cwd="/tmp",
                thread_id=self.owner.thread_id,
                instance_id=self.owner.pk if instance_id is None else instance_id,
            ),
        )

    @patch("hitch.main.sessions.session_resume.thread_has_dynamic_tool", return_value=True)
    @patch("hitch.main.workflows.pr_watch_service.codex_pool.spawn_turn")
    @patch("hitch.main.workflows.pr_watch.observe_pr")
    def test_cancelled_tool_leaves_later_feedback_for_background_delivery(
        self, observe: MagicMock, spawn: MagicMock, _capability: MagicMock,
    ) -> None:
        initial = pr_watch._result_from_observation("ready", _observation())
        pr_tracking.record_pr_watch_result(self.registration, initial)
        self.owner.agent_kind = agent_tasks.PR_WATCH_AGENT_KIND
        cancelled = False
        deliveries: list[Callable[[], None]] = []
        handler = codex_worker._make_approval_handler(
            instance=self.owner, write_event=lambda *_args: None, approval_mode="deny_all",
            question_cancelled=lambda: cancelled,
            on_response_sent=deliveries.append,
        )
        later = _observation(feedback="Feedback posted after the turn ended")

        def cancel_during_read(**_kwargs: object) -> dict[str, object]:
            nonlocal cancelled
            cancelled = True
            return later

        observe.side_effect = cancel_during_read
        response = handler("item/tool/call", {
            "namespace": "hitch", "tool": "watch_pr", "arguments": {"url": _PR_URL},
        })
        self.assertFalse(response["success"])
        self.assertIn("cancelled", str(response))
        self.record.refresh_from_db()
        self.assertEqual(self.record.state[pr_watch.PR_WATCH_RESULT_STATE_KEY], initial)
        self.assertTrue(self.record.state[pr_tracking.WATCH_ACTIVE_STATE_KEY])
        self.assertEqual(deliveries, [])
        observe.side_effect = None
        observe.return_value = later

        # A prepared response is not delivered until the transport writes it.
        cancelled = False
        response = handler("item/tool/call", {
            "namespace": "hitch", "tool": "watch_pr", "arguments": {"url": _PR_URL},
        })
        self.assertTrue(response["success"])
        self.assertEqual(len(deliveries), 1)
        self.record.refresh_from_db()
        self.assertEqual(self.record.state[pr_tracking.WATCH_FEEDBACK_STATE_KEY], initial["feedback_fingerprint"])
        pr_watch_service.poll_registered_prs()
        spawn.assert_called_once()

        deliveries[0]()
        self._due()
        pr_watch_service.poll_registered_prs()
        spawn.assert_called_once()

    @patch("hitch.main.sessions.session_resume.thread_has_dynamic_tool", return_value=True)
    @patch("hitch.main.workflows.pr_tracking._maybe_auto_pull_default_repo_after_pr_merge")
    @patch("hitch.main.workflows.pr_watch.observe_pr")
    def test_auto_pull_finishes_before_response_and_delivery_preserves_newer_snapshot(
        self, observe: MagicMock, auto_pull: MagicMock, _capability: MagicMock,
    ) -> None:
        observe.return_value = _observation({"state": "merged", "merged": True})
        deliveries: list[Callable[[], None]] = []

        def defer_delivery(callback: Callable[[], None]) -> None:
            auto_pull.assert_called_once()
            deliveries.append(callback)

        response = handle_dynamic_tool_call(
            {"namespace": "hitch", "tool": "watch_pr", "arguments": {"url": _PR_URL}},
            ToolContext(
                cwd="/tmp", thread_id=self.owner.thread_id, instance_id=self.owner.pk,
                agent_kind=agent_tasks.PR_WATCH_AGENT_KIND, on_response_sent=defer_delivery,
            ),
        )
        self.assertTrue(response["success"])
        self.record.refresh_from_db()
        self.assertFalse(self.record.state[pr_tracking.WATCH_ACTIVE_STATE_KEY])
        self.assertNotIn(pr_tracking.WATCH_DELIVERED_STATE_KEY, self.record.state)
        self.record.state[pr_tracking.AUTO_PULL_RESULT_STATE_KEY] = {"status": "pulled"}
        self.record.save()
        deliveries[0]()
        self.record.refresh_from_db()
        self.assertIn(pr_tracking.WATCH_DELIVERED_STATE_KEY, self.record.state)
        self.assertEqual(self.record.state[pr_tracking.AUTO_PULL_RESULT_STATE_KEY], {"status": "pulled"})
        auto_pull.assert_called_once()
        self._unwatch()
        state = SessionPullRequest.objects.get(pk=self.record.pk).state
        deliveries[0]()
        self.assertEqual(SessionPullRequest.objects.get(pk=self.record.pk).state, state)

    @patch("hitch.main.workflows.pr_watch_service.codex_pool.spawn_turn")
    @patch("hitch.main.workflows.pr_watch.observe_pr")
    def test_feedback_beyond_five_threads_resumes_watch(self, observe: MagicMock, spawn: MagicMock) -> None:
        threads: list[dict[str, Any]] = [
            {"path": f"file-{index}.py", "comments": {"nodes": [{"body": f"Finding {index}"}]}}
            for index in range(6)
        ]
        # Reproduce a registration whose delivered feedback omitted its sixth thread.
        initial = _observation({"unresolved_thread_count": 6}, feedback=_gh_watch_feedback({}, threads[:5], {}))
        pr_tracking.record_pr_watch_result(
            self.registration, pr_watch._result_from_observation("action_required", initial),
        )
        observe.return_value = {**initial, "feedback": _gh_watch_feedback({}, threads, {})}
        pr_watch_service.poll_registered_prs()
        spawn.assert_called_once()
        self.record.refresh_from_db()
        self.assertIn("Finding 5", self.record.state[pr_watch.PR_WATCH_RESULT_STATE_KEY]["feedback"])

        pr_tracking.record_pr_watch_result(
            self.registration, pr_watch._result_from_observation("action_required", observe.return_value),
        )
        self._due()
        threads[-1]["comments"]["nodes"].append({"body": "New reply"})
        observe.return_value = {**initial, "feedback": _gh_watch_feedback({}, threads, {})}
        pr_watch_service.poll_registered_prs()
        self.assertEqual(spawn.call_count, 2)

    @patch("hitch.main.workflows.pr_watch_service.codex_pool.spawn_turn")
    @patch("hitch.main.workflows.pr_watch_service.pr_watch.observe_pr")
    def test_later_comments_after_readiness_resume_with_inherited_settings(
        self,
        observe: MagicMock,
        spawn: MagicMock,
    ) -> None:
        initial = pr_watch._result_from_observation("ready", _observation(feedback="First comment"))
        pr_tracking.record_pr_watch_result(self.registration, initial)
        observe.return_value = _observation(feedback="First comment")
        pr_watch_service.poll_registered_prs()
        spawn.assert_not_called()
        self._due()
        observe.return_value = _observation(feedback="First comment\nSecond comment")
        pr_watch_service.poll_registered_prs()
        spawn.assert_called_once()
        for key in (
            "model",
            "reasoning_effort",
            "approval_mode",
            "sandbox_policy",
            "web_search_mode",
            "developer_instructions",
            "hitch_extra_instructions",
            "enable_memories",
        ):
            self.assertEqual(spawn.call_args.kwargs[key], getattr(self.owner, key))
        self.record.refresh_from_db()
        self.assertTrue(self.record.state[pr_tracking.WATCH_ACTIVE_STATE_KEY])
        self.assertEqual(pr_tracking._previous_feedback_fingerprint(self.record), initial["feedback_fingerprint"])
        # Delivery by the resumed tool acknowledges this batch; subsequent polls stay quiet.
        pr_tracking.record_pr_watch_result(
            self.registration,
            pr_watch._result_from_observation("attention", observe.return_value),
        )
        self._due()
        pr_watch_service.poll_registered_prs()
        spawn.assert_called_once()
        self._due()
        observe.return_value = _observation(feedback="Third comment")
        pr_watch_service.poll_registered_prs()
        self.assertEqual(spawn.call_count, 2)

    @patch("hitch.main.workflows.pr_watch_service.codex_pool.spawn_turn")
    @patch("hitch.main.workflows.pr_watch_service.pr_watch.observe_pr")
    def test_unwatch_invalidates_inflight_result_and_rewatch_restarts(
        self,
        observe: MagicMock,
        spawn: MagicMock,
    ) -> None:
        def unwatch_during_read(**kwargs: object) -> dict[str, object]:
            self.assertTrue(self._unwatch()["success"])
            return _observation(feedback="Late comment")

        observe.side_effect = unwatch_during_read
        pr_watch_service.poll_registered_prs()
        spawn.assert_not_called()
        self.record.refresh_from_db()
        self.assertFalse(self.record.state[pr_tracking.WATCH_ACTIVE_STATE_KEY])
        self.assertEqual(pr_tracking.pr_handoff_for_record(self.record)["url"], _PR_URL)
        self.assertTrue(pr_tracking.watch_registration_cancelled(self.registration))
        self._due()
        pr_watch_service.poll_registered_prs()
        observe.assert_called_once()
        replacement, _ = pr_tracking.begin_pr_watch_invocation(
            thread_id=self.owner.thread_id,
            cwd="/tmp",
            instance_id=self.owner.pk,
            user_message_index=0,
            agent_kind=agent_tasks.PR_WATCH_AGENT_KIND,
            requested_pr={"url": _PR_URL},
        )
        assert replacement is not None
        self.assertNotEqual(replacement.token, self.registration.token)
        pr_tracking.record_pr_watch_result(
            self.registration, pr_watch._result_from_observation("terminal", _observation())
        )
        self.record.refresh_from_db()
        self.assertTrue(self.record.state[pr_tracking.WATCH_ACTIVE_STATE_KEY])

    @patch("hitch.main.workflows.pr_watch_service.codex_pool.spawn_turn")
    @patch("hitch.main.workflows.pr_watch_service.pr_watch.observe_pr")
    def test_terminal_stops_but_errors_and_timeout_do_not(self, observe: MagicMock, spawn: MagicMock) -> None:
        pr_tracking.record_pr_watch_result(self.registration, pr_watch._result_from_observation("timed_out", {}))
        observe.side_effect = RuntimeError("GitHub unavailable")
        with self.assertLogs(pr_watch_service.logger, level="ERROR"):
            pr_watch_service.poll_registered_prs()
        self._due()
        self.assertTrue(self.record.state[pr_tracking.WATCH_ACTIVE_STATE_KEY])
        observe.side_effect = None
        observe.return_value = _observation({"state": "closed"})
        pr_watch_service.poll_registered_prs()
        self.record.refresh_from_db()
        self.assertFalse(self.record.state[pr_tracking.WATCH_ACTIVE_STATE_KEY])
        self.assertEqual(self.record.state[pr_watch.PR_WATCH_RESULT_STATE_KEY]["status"], "terminal")
        self._due()
        pr_watch_service.poll_registered_prs()
        self.assertEqual(observe.call_count, 2)
        spawn.assert_not_called()

    @patch("hitch.main.workflows.pr_watch_service.codex_pool.spawn_turn")
    @patch("hitch.main.workflows.pr_watch_service.pr_watch.observe_pr")
    def test_active_turn_defers_poll_and_archive_defers_delivery(self, observe: MagicMock, spawn: MagicMock) -> None:
        self.owner.status = CodexInstance.STATUS_RUNNING
        self.owner.save()
        pr_watch_service.poll_registered_prs()
        observe.assert_not_called()
        self.owner.status = CodexInstance.STATUS_COMPLETED
        self.owner.save()
        SessionMetadata.objects.create(thread_id=self.owner.thread_id, codex_archived=True)
        observe.return_value = _observation({"ci_status": "failure"})
        pr_watch_service.poll_registered_prs()
        spawn.assert_not_called()
        self._due()
        SessionMetadata.objects.filter(thread_id=self.owner.thread_id).update(codex_archived=False)
        pr_watch_service.poll_registered_prs()
        spawn.assert_called_once()
        pr_watch_service.poll_registered_prs()
        self.assertEqual(observe.call_count, 2)

    def test_unwatch_rejects_wrong_identity_and_stale_owner(self) -> None:
        self.assertFalse(self._unwatch("https://github.com/openai/hitch/pull/43")["success"])
        newer = CodexInstance.objects.create(
            pid=0,
            thread_id=self.owner.thread_id,
            cwd="/tmp",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
        )
        self.assertFalse(self._unwatch()["success"])
        self.assertTrue(self._unwatch(instance_id=newer.pk)["success"])
        self.assertTrue(self._unwatch(instance_id=newer.pk)["success"])

    def test_timeout_without_observation_preserves_delivered_cursor(self) -> None:
        initial = pr_watch._result_from_observation("ready", _observation(feedback="Assessed comment"))
        pr_tracking.record_pr_watch_result(self.registration, initial)
        pr_tracking.record_pr_watch_result(self.registration, pr_watch._result_from_observation("timed_out", {}))
        self.record.refresh_from_db()
        self.assertEqual(pr_tracking._previous_feedback_fingerprint(self.record), initial["feedback_fingerprint"])
        self.assertEqual(pr_tracking.previous_event_fingerprint(self.registration), pr_watch.event_fingerprint(initial))
        self.assertTrue(self.record.state[pr_tracking.WATCH_ACTIVE_STATE_KEY])

    def test_active_watch_requires_unwatch_before_replacement(self) -> None:
        with self.assertRaisesRegex(pr_watch.PrWatchError, "unwatch_pr"):
            pr_tracking.begin_pr_watch_invocation(
                thread_id=self.owner.thread_id,
                cwd="/tmp",
                instance_id=self.owner.pk,
                user_message_index=0,
                agent_kind=agent_tasks.PR_PUBLISH_AGENT_KIND,
                requested_pr={"url": "https://github.com/openai/hitch/pull/43"},
            )
        self.assertTrue(self._unwatch()["success"])
        registration, _ = pr_tracking.begin_pr_watch_invocation(
            thread_id=self.owner.thread_id,
            cwd="/tmp",
            instance_id=self.owner.pk,
            user_message_index=0,
            agent_kind=agent_tasks.PR_PUBLISH_AGENT_KIND,
            requested_pr={"url": "https://github.com/openai/hitch/pull/43"},
        )
        self.assertIsNotNone(registration)

    @patch("hitch.main.workflows.pr_watch_service.codex_pool.spawn_turn")
    @patch("hitch.main.workflows.pr_watch_service.pr_watch.observe_pr")
    def test_gate_progress_with_old_feedback_does_not_resume(self, observe: MagicMock, spawn: MagicMock) -> None:
        initial = pr_watch._result_from_observation(
            "attention",
            _observation({"ci_status": "pending"}, feedback="Assessed comment"),
        )
        pr_tracking.record_pr_watch_result(self.registration, initial)
        observe.return_value = _observation(feedback="Assessed comment")
        pr_watch_service.poll_registered_prs()
        spawn.assert_not_called()
        self.record.refresh_from_db()
        self.assertEqual(self.record.state[pr_watch.PR_WATCH_RESULT_STATE_KEY]["status"], "ready")

    @patch("hitch.main.workflows.pr_watch_service.codex_pool.spawn_turn")
    @patch("hitch.main.workflows.pr_watch_service.pr_watch.observe_pr")
    def test_disallowed_checkout_defers_delivery(self, observe: MagicMock, spawn: MagicMock) -> None:
        self.mock_allowed.return_value = False
        observe.return_value = _observation(feedback="New comment")
        with self.assertLogs(pr_watch_service.logger, level="WARNING"):
            pr_watch_service.poll_registered_prs()
        spawn.assert_not_called()
        self.record.refresh_from_db()
        self.assertTrue(self.record.state[pr_tracking.WATCH_ACTIVE_STATE_KEY])
        self.mock_allowed.return_value = True
        self._due()
        pr_watch_service.poll_registered_prs()
        spawn.assert_called_once()

    @patch("hitch.main.workflows.pr_watch_service.codex_pool.spawn_turn")
    @patch("hitch.main.workflows.pr_watch_service.pr_watch.observe_pr")
    def test_duplicate_pr_registrations_share_poll_and_throttle(self, observe: MagicMock, spawn: MagicMock) -> None:
        other = CodexInstance.objects.create(
            pid=0,
            thread_id="other-thread",
            cwd="/tmp",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
        )
        pr_tracking.begin_pr_watch_invocation(
            thread_id=other.thread_id,
            cwd=other.cwd,
            instance_id=other.pk,
            user_message_index=0,
            agent_kind=agent_tasks.PR_PUBLISH_AGENT_KIND,
            requested_pr={"url": _PR_URL},
        )
        observe.return_value = _observation(feedback="New comment")
        pr_watch_service.poll_registered_prs()
        observe.assert_called_once()
        self.assertEqual(spawn.call_count, 2)
        pr_watch_service.poll_registered_prs()
        observe.assert_called_once()

    @patch("hitch.main.workflows.pr_watch_service.poll_registered_prs")
    @patch("hitch.main.workflows.pr_watch_service._scheduler.start")
    @patch("hitch.main.runtime.maintenance._scheduler.start")
    @patch("hitch.main.runtime.maintenance._maintenance_scheduler_enabled", return_value=True)
    @patch("hitch.main.runtime.maintenance.reconciliation.reconcile_dead")
    def test_watch_scheduler_does_not_block_maintenance(
        self,
        reconcile: MagicMock,
        enabled: MagicMock,
        start_maintenance: MagicMock,
        start_watch: MagicMock,
        poll: MagicMock,
    ) -> None:
        maintenance.start_maintenance_scheduler()
        start_watch.assert_called_once_with(pr_watch_service._scheduler_loop)
        start_maintenance.assert_called_once()
        maintenance._maintenance_tick()
        reconcile.assert_called_once()
        poll.assert_not_called()
        enabled.return_value = False
        start_watch.reset_mock()
        self.assertFalse(maintenance.start_maintenance_scheduler())
        start_watch.assert_not_called()
