from __future__ import annotations

import json
import tempfile
from typing import cast, override
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase, TestCase

from hitch.main.models import CodexInstance, SessionMetadata, SessionPullRequest
from hitch.main.repos import AutoPullError, AutoPullResult
from hitch.main.runtime.codex_tools import (
    ToolContext,
    handle_dynamic_tool_call,
    registered_dynamic_tool_specs,
)
from hitch.main.sessions import agent_tasks
from hitch.main.test.support import _make_project
from hitch.main.workflows import pr_tracking, pr_watch, system_agents
from hitch.main.workflows.gh_observations import (
    _evaluate_pr_gates,
    _gh_watch_blockers,
    _gh_watch_summary,
)

_PR_URL = "https://github.com/openai/hitch/pull/42"


def _observation(
    update: dict[str, object] | None = None, *, feedback: str = ""
) -> dict[str, object]:
    pr: dict[str, object] = {
        "url": _PR_URL,
        "repository_full_name": "openai/hitch",
        "pr_number": 42,
        "state": "open",
        "merged": False,
        "mergeable": True,
        "draft": False,
        "review_signal": "approved",
        "unresolved_thread_count": 0,
        "ci_status": "success",
    }
    pr.update(update or {})
    gates = _evaluate_pr_gates(pr)
    return {
        "summary": _gh_watch_summary(gates, pr),
        "feedback": feedback,
        "pr": pr,
        "gates": gates,
        "blockers": _gh_watch_blockers(gates),
    }


def _tool_result(response: dict[str, object]) -> dict[str, object]:
    items = response["contentItems"]
    assert isinstance(items, list)
    item = items[0]
    assert isinstance(item, dict)
    return cast(dict[str, object], json.loads(str(item["text"])))


class PrWatchTests(SimpleTestCase):
    @patch("hitch.main.workflows.pr_watch.observe_pr")
    def test_returns_actionable_failure(self, mock_observe: MagicMock) -> None:
        mock_observe.return_value = _observation(
            {
                "ci_status": "failure",
                "failing_jobs": [{"name": "tests", "conclusion": "failure"}],
            },
            feedback="tests failed",
        )

        with tempfile.TemporaryDirectory() as cwd:
            result = pr_watch.watch_pr(cwd=cwd, url=_PR_URL, poll_seconds=0)

        self.assertEqual(result["status"], "action_required")
        self.assertIn("CI", result["summary"])

    @patch("hitch.main.workflows.pr_watch.observe_pr")
    def test_new_feedback_interrupts_pending_watch(
        self, mock_observe: MagicMock
    ) -> None:
        mock_observe.return_value = _observation(
            {"review_signal": "", "ci_status": "pending"},
            feedback="A reviewer left a note.",
        )

        with tempfile.TemporaryDirectory() as cwd:
            result = pr_watch.watch_pr(cwd=cwd, url=_PR_URL, poll_seconds=0)

        self.assertEqual(result["status"], "attention")
        self.assertTrue(result["feedback_fingerprint"])

    @patch("hitch.main.workflows.pr_watch.observe_pr")
    def test_seen_feedback_does_not_create_a_hot_loop(
        self, mock_observe: MagicMock
    ) -> None:
        pending = _observation(
            {"review_signal": "", "ci_status": "pending"},
            feedback="Already assessed.",
        )
        mock_observe.side_effect = [pending, _observation()]

        with tempfile.TemporaryDirectory() as cwd:
            result = pr_watch.watch_pr(
                cwd=cwd,
                url=_PR_URL,
                previous_feedback_fingerprint=pr_watch.feedback_fingerprint(pending),
                poll_seconds=0,
            )

        self.assertEqual(result["status"], "ready")
        self.assertEqual(mock_observe.call_count, 2)

    @patch("hitch.main.workflows.pr_watch.observe_pr")
    def test_returns_terminal_pr(self, mock_observe: MagicMock) -> None:
        mock_observe.return_value = _observation(
            {"state": "merged", "merged": True}
        )

        with tempfile.TemporaryDirectory() as cwd:
            result = pr_watch.watch_pr(cwd=cwd, url=_PR_URL)

        self.assertEqual(result["status"], "terminal")

    @patch("hitch.main.workflows.pr_watch.observe_pr")
    def test_pending_watch_times_out(self, mock_observe: MagicMock) -> None:
        mock_observe.return_value = _observation(
            {"review_signal": "", "ci_status": "pending"}
        )

        with tempfile.TemporaryDirectory() as cwd:
            result = pr_watch.watch_pr(cwd=cwd, url=_PR_URL, timeout_seconds=0)

        self.assertEqual(result["status"], "timed_out")

    @patch("hitch.main.workflows.pr_watch.observe_pr")
    def test_cancellation_interrupts_polling_wait(
        self, mock_observe: MagicMock
    ) -> None:
        mock_observe.return_value = _observation(
            {"review_signal": "", "ci_status": "pending"}
        )
        cancelled = False

        def cancel_requested() -> bool:
            return cancelled

        def cancel_during_sleep(_seconds: float) -> None:
            nonlocal cancelled
            cancelled = True

        with (
            tempfile.TemporaryDirectory() as cwd,
            self.assertRaisesRegex(pr_watch.PrWatchError, "cancelled"),
        ):
            pr_watch.watch_pr(
                cwd=cwd,
                url=_PR_URL,
                poll_seconds=10,
                sleep=cancel_during_sleep,
                cancel_requested=cancel_requested,
            )

    def test_rejects_non_pr_url(self) -> None:
        with (
            tempfile.TemporaryDirectory() as cwd,
            self.assertRaisesRegex(pr_watch.PrWatchError, "GitHub pull request"),
        ):
            pr_watch.watch_pr(cwd=cwd, url="https://example.com/pr/42")


class PrWatchToolTests(TestCase):
    @override
    def setUp(self) -> None:
        cwd = tempfile.TemporaryDirectory()
        self.addCleanup(cwd.cleanup)
        self.cwd = cwd.name

    def _call(
        self,
        *,
        agent_kind: str,
        instance_id: int = 7,
        message_index: int = 4,
    ) -> dict[str, object]:
        return handle_dynamic_tool_call(
            {
                "namespace": "hitch",
                "tool": "watch_pr",
                "arguments": {"url": _PR_URL},
            },
            ToolContext(
                cwd=self.cwd,
                thread_id="main-thread",
                instance_id=instance_id,
                agent_kind=agent_kind,
                user_message_index=message_index,
            ),
        )

    def test_registered_specs_include_watch_pr(self) -> None:
        specs = registered_dynamic_tool_specs()

        watch = next(spec for spec in specs if spec["name"] == "watch_pr")
        self.assertEqual(watch["namespace"], "hitch")
        self.assertEqual(watch["inputSchema"]["required"], ["url"])

    @patch("hitch.main.runtime.codex_tools.pr_watch.watch_pr")
    def test_tagged_publish_turn_registers_and_records_pr(
        self, mock_watch: MagicMock
    ) -> None:
        result = {
            **_observation(),
            "status": "ready",
            "feedback_fingerprint": "fingerprint",
        }
        mock_watch.return_value = result

        with patch(
            "hitch.main.runtime.codex_tools.pr_watch.validate_published_pr_checkout"
        ) as validate:
            response = self._call(agent_kind=agent_tasks.PR_PUBLISH_AGENT_KIND)

        self.assertTrue(response["success"])
        self.assertEqual(_tool_result(response)["status"], "ready")
        validate.assert_called_once_with(cwd=self.cwd, url=_PR_URL)
        record = SessionPullRequest.objects.get(thread_id="main-thread")
        self.assertEqual(
            pr_tracking.pr_handoff_for_record(record)["pr_number"], 42
        )
        self.assertEqual(
            record.state[pr_watch.PR_WATCH_RESULT_STATE_KEY]["status"], "ready"
        )
        self.assertEqual(
            record.state[pr_watch.PR_WATCH_RESULT_TURN_INDEX_STATE_KEY], 4
        )

    @patch("hitch.main.runtime.codex_tools.pr_watch.watch_pr")
    def test_follow_up_watch_reuses_feedback_fingerprint(
        self, mock_watch: MagicMock
    ) -> None:
        mock_watch.return_value = {
            **_observation(),
            "status": "attention",
            "feedback_fingerprint": "seen",
        }
        self._call(agent_kind=agent_tasks.PR_WATCH_AGENT_KIND)
        mock_watch.return_value = {
            **_observation(),
            "status": "ready",
            "feedback_fingerprint": "seen",
        }

        self._call(
            agent_kind=agent_tasks.PR_WATCH_AGENT_KIND,
            instance_id=8,
            message_index=5,
        )

        self.assertEqual(
            mock_watch.call_args.kwargs["previous_feedback_fingerprint"], "seen"
        )

    @patch("hitch.main.runtime.codex_tools.pr_watch.watch_pr")
    def test_failed_follow_up_preserves_feedback_suppression(
        self, mock_watch: MagicMock
    ) -> None:
        prior_result = {
            **_observation(),
            "status": "attention",
            "feedback_fingerprint": "seen",
        }
        mock_watch.return_value = prior_result
        self._call(agent_kind=agent_tasks.PR_WATCH_AGENT_KIND)
        mock_watch.side_effect = pr_watch.PrWatchError("GitHub unavailable")

        response = self._call(
            agent_kind=agent_tasks.PR_WATCH_AGENT_KIND,
            instance_id=8,
            message_index=5,
        )

        self.assertFalse(response["success"])
        record = SessionPullRequest.objects.get(thread_id="main-thread")
        self.assertEqual(
            record.state[pr_watch.PR_WATCH_RESULT_STATE_KEY][
                "feedback_fingerprint"
            ],
            "seen",
        )

    @patch("hitch.main.runtime.codex_tools.pr_watch.watch_pr")
    def test_standalone_watch_does_not_claim_session_ui_state(
        self, mock_watch: MagicMock
    ) -> None:
        mock_watch.return_value = {
            **_observation(),
            "status": "ready",
            "feedback_fingerprint": "",
        }

        response = self._call(agent_kind="")

        self.assertTrue(response["success"])
        self.assertFalse(SessionPullRequest.objects.exists())

    @patch("hitch.main.runtime.codex_tools.pr_watch.watch_pr")
    def test_review_only_turn_does_not_claim_session_ui_state(
        self, mock_watch: MagicMock
    ) -> None:
        mock_watch.return_value = {
            **_observation(),
            "status": "ready",
            "feedback_fingerprint": "",
        }

        self._call(agent_kind=agent_tasks.REVIEW_AGENT_KIND)

        self.assertFalse(SessionPullRequest.objects.exists())

    @patch("hitch.main.runtime.codex_tools.pr_watch.watch_pr")
    def test_fix_turn_cannot_replace_registered_pr(
        self, mock_watch: MagicMock
    ) -> None:
        SessionPullRequest.objects.create(
            thread_id="main-thread",
            cwd=self.cwd,
            state={
                pr_tracking.PR_HANDOFF_STATE_KEY: {
                    "url": "https://github.com/openai/hitch/pull/41",
                    "repository_full_name": "openai/hitch",
                    "pr_number": 41,
                }
            },
        )

        response = self._call(agent_kind=agent_tasks.PR_WATCH_AGENT_KIND)

        self.assertFalse(response["success"])
        mock_watch.assert_not_called()

    @patch("hitch.main.workflows.pr_tracking._maybe_auto_pull_default_repo_after_pr_merge")
    def test_terminal_merge_triggers_post_merge_auto_pull(
        self, mock_auto_pull: MagicMock
    ) -> None:
        requested = {
            "url": _PR_URL,
            "repository_full_name": "openai/hitch",
            "pr_number": 42,
        }
        registration, _fingerprint = pr_tracking.begin_pr_watch_invocation(
            thread_id="main-thread",
            cwd=self.cwd,
            instance_id=7,
            user_message_index=4,
            agent_kind=agent_tasks.PR_WATCH_AGENT_KIND,
            requested_pr=requested,
        )
        assert registration is not None

        pr_tracking.record_pr_watch_result(
            registration,
            {
                "status": "terminal",
                "pr": {**requested, "state": "merged", "merged": True},
                "gates": [],
            },
        )

        mock_auto_pull.assert_called_once_with(registration)

    @patch("hitch.main.workflows.pr_tracking._maybe_auto_pull_default_repo_after_pr_merge")
    def test_stale_watch_result_cannot_overwrite_new_owner(
        self, mock_auto_pull: MagicMock
    ) -> None:
        requested = {
            "url": _PR_URL,
            "repository_full_name": "openai/hitch",
            "pr_number": 42,
        }
        stale, _fingerprint = pr_tracking.begin_pr_watch_invocation(
            thread_id="main-thread",
            cwd=self.cwd,
            instance_id=7,
            user_message_index=4,
            agent_kind=agent_tasks.PR_WATCH_AGENT_KIND,
            requested_pr=requested,
        )
        current, _fingerprint = pr_tracking.begin_pr_watch_invocation(
            thread_id="main-thread",
            cwd=self.cwd,
            instance_id=8,
            user_message_index=5,
            agent_kind=agent_tasks.PR_WATCH_AGENT_KIND,
            requested_pr=requested,
        )
        assert stale is not None
        assert current is not None

        pr_tracking.record_pr_watch_result(
            stale,
            {
                "status": "terminal",
                "pr": {**requested, "state": "merged", "merged": True},
                "gates": [],
            },
        )

        record = SessionPullRequest.objects.get(thread_id="main-thread")
        self.assertTrue(pr_tracking.watch_registered_by_instance(record, 8))
        self.assertNotIn(pr_watch.PR_WATCH_RESULT_STATE_KEY, record.state)
        mock_auto_pull.assert_not_called()

    def test_stage_refresh_cannot_overwrite_new_publication_registration(self) -> None:
        old_pr = {
            "url": _PR_URL,
            "repository_full_name": "openai/hitch",
            "pr_number": 42,
            "state": "open",
        }
        record = SessionPullRequest.objects.create(
            thread_id="main-thread",
            cwd=self.cwd,
            state={
                pr_tracking.PR_HANDOFF_STATE_KEY: old_pr,
                "hitch_pr_handoff": {
                    key: old_pr[key]
                    for key in ("url", "repository_full_name", "pr_number")
                },
            },
        )
        new_pr = {
            "url": "https://github.com/openai/hitch/pull/43",
            "repository_full_name": "openai/hitch",
            "pr_number": 43,
            "state": "open",
        }

        def register_new_pr(*_args: object, **_kwargs: object) -> dict[str, object]:
            registration, _fingerprint = pr_tracking.begin_pr_watch_invocation(
                thread_id="main-thread",
                cwd=self.cwd,
                instance_id=8,
                user_message_index=5,
                agent_kind=agent_tasks.PR_PUBLISH_AGENT_KIND,
                requested_pr=new_pr,
            )
            assert registration is not None
            return {**old_pr, "state": "merged", "merged": True}

        with patch(
            "hitch.main.workflows.pr_tracking._gh_pr_view",
            side_effect=register_new_pr,
        ):
            refreshed = pr_tracking.refreshed_pr_handoff_for_stage(
                record, force=True
            )

        self.assertEqual(refreshed["pr_number"], 43)
        record.refresh_from_db()
        self.assertEqual(
            pr_tracking.pr_handoff_for_record(record)["pr_number"], 43
        )
        self.assertTrue(pr_tracking.watch_registered_by_instance(record, 8))

    def test_stage_refresh_cannot_update_a_pr_superseded_during_observation(
        self,
    ) -> None:
        handoff = {
            "url": _PR_URL,
            "repository_full_name": "openai/hitch",
            "pr_number": 42,
            "state": "open",
        }
        record = SessionPullRequest.objects.create(
            thread_id="main-thread",
            cwd=self.cwd,
            state={
                pr_tracking.PR_HANDOFF_STATE_KEY: handoff,
                "hitch_pr_handoff": {
                    key: handoff[key]
                    for key in ("url", "repository_full_name", "pr_number")
                },
            },
        )

        def supersede_pr(*_args: object, **_kwargs: object) -> dict[str, object]:
            current = SessionPullRequest.objects.get(pk=record.pk)
            current.state = {
                **current.state,
                SessionPullRequest.SUPERSEDED_BY_INSTANCE_STATE_KEY: 9,
            }
            current.save(update_fields=["state", "updated_at"])
            return {**handoff, "state": "merged", "merged": True}

        with patch(
            "hitch.main.workflows.pr_tracking._gh_pr_view",
            side_effect=supersede_pr,
        ):
            refreshed = pr_tracking.refreshed_pr_handoff_for_stage(
                record, force=True
            )

        self.assertEqual(refreshed, {})
        record.refresh_from_db()
        self.assertEqual(
            pr_tracking.pr_handoff_for_record(record)["state"], "open"
        )

    def _merged_registration(
        self,
        *,
        project_repo: str = "/default-repo",
        auto_pull_enabled: bool = True,
    ) -> tuple[pr_tracking.PrWatchRegistration, SessionPullRequest]:
        project = _make_project(
            repo_path=project_repo,
            git_common_dir=f"{project_repo}/.git",
            auto_pull_enabled=auto_pull_enabled,
        )
        SessionMetadata.objects.create(
            thread_id="main-thread",
            cwd=self.cwd,
            project=project,
        )
        registration, _fingerprint = pr_tracking.begin_pr_watch_invocation(
            thread_id="main-thread",
            cwd=self.cwd,
            instance_id=7,
            user_message_index=4,
            agent_kind=agent_tasks.PR_PUBLISH_AGENT_KIND,
            requested_pr={
                "url": _PR_URL,
                "repository_full_name": "openai/hitch",
                "pr_number": 42,
            },
        )
        assert registration is not None
        record = SessionPullRequest.objects.get(thread_id="main-thread")
        record.state = {
            **record.state,
            pr_tracking.PR_HANDOFF_STATE_KEY: {
                **pr_tracking.pr_handoff_for_record(record),
                "state": "merged",
                "merged": True,
            },
        }
        record.save(update_fields=["state", "updated_at"])
        return registration, record

    @patch("hitch.main.workflows.pr_tracking.pull_default_branch_from_origin")
    def test_post_merge_auto_pull_records_a_verified_update(
        self, mock_pull: MagicMock
    ) -> None:
        registration, record = self._merged_registration()
        mock_pull.return_value = AutoPullResult(
            branch="main",
            before_sha="a" * 40,
            after_sha="b" * 40,
            changed=True,
        )

        with patch(
            "hitch.main.workflows.pr_tracking.same_repo_or_worktree",
            return_value=True,
        ):
            pr_tracking._maybe_auto_pull_default_repo_after_pr_merge(
                registration
            )

        mock_pull.assert_called_once_with("/default-repo")
        record.refresh_from_db()
        self.assertEqual(
            record.state[pr_tracking.AUTO_PULL_RESULT_STATE_KEY]["status"],
            "pulled",
        )

    @patch("hitch.main.workflows.pr_tracking.pull_default_branch_from_origin")
    def test_post_merge_auto_pull_records_expected_failure(
        self, mock_pull: MagicMock
    ) -> None:
        registration, record = self._merged_registration()
        mock_pull.side_effect = AutoPullError("default checkout is dirty")

        with patch(
            "hitch.main.workflows.pr_tracking.same_repo_or_worktree",
            return_value=True,
        ):
            pr_tracking._maybe_auto_pull_default_repo_after_pr_merge(
                registration
            )

        record.refresh_from_db()
        self.assertEqual(
            record.state[pr_tracking.AUTO_PULL_RESULT_STATE_KEY],
            {"status": "failed", "error": "default checkout is dirty"},
        )

    @patch("hitch.main.workflows.pr_tracking.pull_default_branch_from_origin")
    def test_post_merge_auto_pull_skips_the_active_checkout(
        self, mock_pull: MagicMock
    ) -> None:
        registration, record = self._merged_registration(
            project_repo=self.cwd
        )

        pr_tracking._maybe_auto_pull_default_repo_after_pr_merge(registration)

        mock_pull.assert_not_called()
        record.refresh_from_db()
        self.assertEqual(
            record.state[pr_tracking.AUTO_PULL_RESULT_STATE_KEY],
            {
                "status": "skipped",
                "reason": "default checkout is the active session checkout",
            },
        )

    @patch("hitch.main.workflows.pr_tracking.pull_default_branch_from_origin")
    def test_post_merge_auto_pull_does_not_write_through_new_registration(
        self, mock_pull: MagicMock
    ) -> None:
        registration, record = self._merged_registration()

        def replace_registration(_repo_path: str) -> AutoPullResult:
            replacement, _fingerprint = pr_tracking.begin_pr_watch_invocation(
                thread_id="main-thread",
                cwd=self.cwd,
                instance_id=8,
                user_message_index=5,
                agent_kind=agent_tasks.PR_PUBLISH_AGENT_KIND,
                requested_pr={
                    "url": "https://github.com/openai/hitch/pull/43",
                    "repository_full_name": "openai/hitch",
                    "pr_number": 43,
                },
            )
            assert replacement is not None
            return AutoPullResult(
                branch="main",
                before_sha="a" * 40,
                after_sha="b" * 40,
                changed=True,
            )

        mock_pull.side_effect = replace_registration
        with patch(
            "hitch.main.workflows.pr_tracking.same_repo_or_worktree",
            return_value=True,
        ):
            pr_tracking._maybe_auto_pull_default_repo_after_pr_merge(
                registration
            )

        record.refresh_from_db()
        self.assertEqual(
            pr_tracking.pr_handoff_for_record(record)["pr_number"], 43
        )
        self.assertNotIn(pr_tracking.AUTO_PULL_RESULT_STATE_KEY, record.state)

    def test_unrelated_completed_turn_supersedes_registered_pr(self) -> None:
        owner = CodexInstance.objects.create(
            pid=1,
            thread_id="main-thread",
            cwd=self.cwd,
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            agent_kind=agent_tasks.PR_PUBLISH_AGENT_KIND,
        )
        registration, _fingerprint = pr_tracking.begin_pr_watch_invocation(
            thread_id="main-thread",
            cwd=self.cwd,
            instance_id=owner.pk,
            user_message_index=1,
            agent_kind=agent_tasks.PR_PUBLISH_AGENT_KIND,
            requested_pr={
                "url": _PR_URL,
                "repository_full_name": "openai/hitch",
                "pr_number": 42,
                "state": "open",
            },
        )
        assert registration is not None
        watch_turn = CodexInstance.objects.create(
            pid=2,
            thread_id="main-thread",
            cwd=self.cwd,
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            agent_kind=agent_tasks.PR_WATCH_AGENT_KIND,
        )

        system_agents.on_codex_instance_finished(watch_turn)

        self.assertIsNotNone(pr_tracking.record_for_thread("main-thread"))
        ordinary_turn = CodexInstance.objects.create(
            pid=3,
            thread_id="main-thread",
            cwd=self.cwd,
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
        )

        system_agents.on_codex_instance_finished(ordinary_turn)

        self.assertIsNone(pr_tracking.record_for_thread("main-thread"))
        record = SessionPullRequest.objects.get(thread_id="main-thread")
        self.assertFalse(pr_tracking.pr_handoff_stage_refresh_due(record))

        replacement_turn = CodexInstance.objects.create(
            pid=4,
            thread_id="main-thread",
            cwd=self.cwd,
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
            agent_kind=agent_tasks.PR_PUBLISH_AGENT_KIND,
        )
        replacement, _fingerprint = pr_tracking.begin_pr_watch_invocation(
            thread_id="main-thread",
            cwd=self.cwd,
            instance_id=replacement_turn.pk,
            user_message_index=3,
            agent_kind=agent_tasks.PR_PUBLISH_AGENT_KIND,
            requested_pr={
                "url": "https://github.com/openai/hitch/pull/43",
                "repository_full_name": "openai/hitch",
                "pr_number": 43,
                "state": "open",
            },
        )

        assert replacement is not None
        self.assertIsNotNone(pr_tracking.record_for_thread("main-thread"))
