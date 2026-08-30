from __future__ import annotations

import json
import tempfile
from types import SimpleNamespace
from typing import override
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase, TestCase

from hitch.main.models import CodexInstance, SystemWorkflow
from hitch.main.runtime.codex_tools import (
    ToolContext,
    handle_dynamic_tool_call,
    registered_dynamic_tool_specs,
)
from hitch.main.sessions import pr_prompts
from hitch.main.workflows import pr_qa, pr_watch, system_agents
from hitch.main.workflows.gh_cli import _GhPrOpenError
from hitch.main.workflows.gh_observations import (
    _evaluate_pr_gates,
    _gh_watch_blockers,
    _gh_watch_summary,
)


def _observation(
    update: dict[str, object] | None = None, *, feedback: str = ""
) -> dict[str, object]:
    pr: dict[str, object] = {
        "url": "https://github.com/openai/hitch/pull/42",
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
            result = pr_watch.watch_pr(
                cwd=cwd,
                url="https://github.com/openai/hitch/pull/42",
                poll_seconds=0,
            )

        self.assertEqual(result["status"], "action_required")
        self.assertIn("CI", result["summary"])

    @patch("hitch.main.workflows.pr_watch.observe_pr")
    def test_new_feedback_interrupts_pending_watch(
        self, mock_observe: MagicMock
    ) -> None:
        pending = _observation(
            {
                "review_signal": "",
                "ci_status": "pending",
                "pending_jobs": [{"name": "tests", "status": "queued"}],
            },
            feedback="A reviewer left a note.",
        )
        mock_observe.return_value = pending

        with tempfile.TemporaryDirectory() as cwd:
            result = pr_watch.watch_pr(
                cwd=cwd,
                url="https://github.com/openai/hitch/pull/42",
                poll_seconds=0,
            )

        self.assertEqual(result["status"], "attention")
        self.assertTrue(result["feedback_fingerprint"])

    @patch("hitch.main.workflows.pr_watch.observe_pr")
    def test_new_feedback_requires_attention_before_ready(
        self, mock_observe: MagicMock
    ) -> None:
        mock_observe.return_value = _observation(
            feedback="A reviewer left a non-blocking note."
        )

        with tempfile.TemporaryDirectory() as cwd:
            result = pr_watch.watch_pr(
                cwd=cwd,
                url="https://github.com/openai/hitch/pull/42",
                poll_seconds=0,
            )

        self.assertEqual(result["status"], "attention")

    @patch("hitch.main.workflows.pr_watch.observe_pr")
    def test_seen_feedback_does_not_create_a_hot_loop(
        self, mock_observe: MagicMock
    ) -> None:
        pending = _observation(
            {"review_signal": "", "ci_status": "pending"},
            feedback="Already assessed.",
        )
        mock_observe.side_effect = [pending, _observation()]
        fingerprint = pr_watch.feedback_fingerprint(pending)

        with tempfile.TemporaryDirectory() as cwd:
            result = pr_watch.watch_pr(
                cwd=cwd,
                url="https://github.com/openai/hitch/pull/42",
                previous_feedback_fingerprint=fingerprint,
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
            result = pr_watch.watch_pr(
                cwd=cwd,
                url="https://github.com/openai/hitch/pull/42",
            )

        self.assertEqual(result["status"], "terminal")

    @patch("hitch.main.workflows.pr_watch.observe_pr")
    def test_pending_watch_times_out(self, mock_observe: MagicMock) -> None:
        mock_observe.return_value = _observation(
            {"review_signal": "", "ci_status": "pending"}
        )

        with tempfile.TemporaryDirectory() as cwd:
            result = pr_watch.watch_pr(
                cwd=cwd,
                url="https://github.com/openai/hitch/pull/42",
                timeout_seconds=0,
            )

        self.assertEqual(result["status"], "timed_out")

    @patch("hitch.main.workflows.pr_watch.observe_pr")
    def test_deadline_prevents_another_observation(
        self, mock_observe: MagicMock
    ) -> None:
        mock_observe.return_value = _observation(
            {"review_signal": "", "ci_status": "pending"}
        )
        ticks = iter([0.0, 0.0, 1.0])

        with tempfile.TemporaryDirectory() as cwd:
            result = pr_watch.watch_pr(
                cwd=cwd,
                url="https://github.com/openai/hitch/pull/42",
                timeout_seconds=1,
                monotonic=lambda: next(ticks),
            )

        self.assertEqual(result["status"], "timed_out")
        mock_observe.assert_called_once()

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
                url="https://github.com/openai/hitch/pull/42",
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

    def test_recognizes_all_hitch_published_legacy_prompts(self) -> None:
        legacy_prompts = (
            "Rebase on the default branch, polish it, get it ready, and commit "
            "the final changes. Do not push the branch or open a PR; Hitch will "
            "push and open it after this turn completes.",
            "Rebase on the repository's default branch, polish it, get it ready, "
            "and commit the final changes. Do not push the branch or open a PR; "
            "Hitch will push and open it after this turn completes.",
            "Rebase on master, polish it, get it ready, and commit the final "
            "changes. Do not push the branch or open a PR; Hitch will push and "
            "open it after this turn completes.",
            "Polish it, get it ready, and commit the final changes. Do not push "
            "the branch or open a PR; Hitch will push and open it after this turn "
            "completes.",
            "Polish it, get it ready, commit the final changes, and push the "
            "branch. Do not open a PR; Hitch will open it after this turn "
            "completes.",
        )

        for prompt in legacy_prompts:
            with self.subTest(prompt=prompt):
                self.assertTrue(
                    pr_prompts.is_legacy_hitch_published_pr_prompt(
                        f"Review first.\n\n{prompt}\n\nUse the requested title."
                    )
                )
        self.assertFalse(
            pr_prompts.is_legacy_hitch_published_pr_prompt(
                pr_prompts.PR_SLASH_PROMPT
            )
        )

    @patch("hitch.main.workflows.pr_watch._run_git_cli")
    @patch("hitch.main.workflows.pr_watch._gh_pr_view_payload")
    def test_published_pr_must_match_checkout(
        self,
        mock_view: MagicMock,
        mock_git: MagicMock,
    ) -> None:
        mock_view.return_value = {
            "url": "https://github.com/openai/hitch/pull/42",
            "number": 42,
            "state": "OPEN",
            "headRefName": "feature",
            "headRefOid": "abc123",
            "headRepository": {"name": "hitch"},
            "headRepositoryOwner": {"login": "openai"},
        }
        mock_git.side_effect = [
            SimpleNamespace(
                returncode=0,
                stdout=(
                    "origin\tgit@github.com:openai/hitch.git (fetch)\n"
                    "origin\tgit@github.com:openai/hitch.git (push)\n"
                ),
            ),
            SimpleNamespace(returncode=0, stdout="feature\n"),
            SimpleNamespace(returncode=0, stdout="abc123\n"),
        ]

        with tempfile.TemporaryDirectory() as cwd:
            pr_watch.validate_published_pr_checkout(
                cwd=cwd,
                url="https://github.com/openai/hitch/pull/42",
            )

    @patch("hitch.main.workflows.pr_watch._run_git_cli")
    @patch("hitch.main.workflows.pr_watch._gh_pr_view_payload")
    def test_published_pr_rejects_unrelated_repository(
        self,
        mock_view: MagicMock,
        mock_git: MagicMock,
    ) -> None:
        mock_view.return_value = {
            "url": "https://github.com/openai/another/pull/42",
            "number": 42,
            "state": "OPEN",
            "headRefName": "feature",
            "headRefOid": "abc123",
            "headRepository": {"name": "another"},
            "headRepositoryOwner": {"login": "openai"},
        }
        mock_git.return_value = SimpleNamespace(
            returncode=0,
            stdout="origin\tgit@github.com:openai/hitch.git (fetch)\n",
        )

        with (
            tempfile.TemporaryDirectory() as cwd,
            self.assertRaisesRegex(pr_watch.PrWatchError, "does not match"),
        ):
            pr_watch.validate_published_pr_checkout(
                cwd=cwd,
                url="https://github.com/openai/another/pull/42",
            )

    @patch("hitch.main.workflows.pr_watch._run_git_cli")
    @patch("hitch.main.workflows.pr_watch._gh_pr_view_payload")
    def test_published_pr_must_be_open(
        self,
        mock_view: MagicMock,
        mock_git: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as cwd:
            for state in ("CLOSED", "MERGED"):
                with self.subTest(state=state):
                    mock_view.return_value = {
                        "url": "https://github.com/openai/hitch/pull/42",
                        "number": 42,
                        "state": state,
                        "headRefName": "feature",
                        "headRefOid": "abc123",
                        "headRepository": {"name": "hitch"},
                        "headRepositoryOwner": {"login": "openai"},
                    }

                    with self.assertRaisesRegex(
                        pr_watch.PrWatchError,
                        "must be open",
                    ):
                        pr_watch.validate_published_pr_checkout(
                            cwd=cwd,
                            url="https://github.com/openai/hitch/pull/42",
                        )

        mock_git.assert_not_called()

    @patch("hitch.main.workflows.pr_watch._run_git_cli")
    @patch("hitch.main.workflows.pr_watch._gh_pr_view_payload")
    def test_published_pr_rejects_a_different_head(
        self,
        mock_view: MagicMock,
        mock_git: MagicMock,
    ) -> None:
        cases = (
            ("another-branch", "abc123", "head branch"),
            ("feature", "def456", "head commit"),
        )
        with tempfile.TemporaryDirectory() as cwd:
            for head, head_sha, message in cases:
                with self.subTest(message=message):
                    mock_view.return_value = {
                        "url": "https://github.com/openai/hitch/pull/42",
                        "number": 42,
                        "state": "OPEN",
                        "headRefName": head,
                        "headRefOid": head_sha,
                        "headRepository": {"name": "hitch"},
                        "headRepositoryOwner": {"login": "openai"},
                    }
                    mock_git.side_effect = [
                        SimpleNamespace(
                            returncode=0,
                            stdout=(
                                "origin\tgit@github.com:openai/hitch.git (fetch)\n"
                            ),
                        ),
                        SimpleNamespace(returncode=0, stdout="feature\n"),
                        SimpleNamespace(returncode=0, stdout="abc123\n"),
                    ]

                    with self.assertRaisesRegex(pr_watch.PrWatchError, message):
                        pr_watch.validate_published_pr_checkout(
                            cwd=cwd,
                            url="https://github.com/openai/hitch/pull/42",
                        )

    @patch("hitch.main.workflows.pr_watch._run_git_cli")
    @patch("hitch.main.workflows.pr_watch._gh_pr_view_payload")
    def test_published_fork_pr_matches_head_repository(
        self,
        mock_view: MagicMock,
        mock_git: MagicMock,
    ) -> None:
        mock_view.return_value = {
            "url": "https://github.com/openai/hitch/pull/42",
            "number": 42,
            "state": "OPEN",
            "headRefName": "feature",
            "headRefOid": "abc123",
            "headRepository": {"name": "hitch"},
            "headRepositoryOwner": {"login": "contributor"},
        }
        mock_git.side_effect = [
            SimpleNamespace(
                returncode=0,
                stdout=(
                    "origin\tgit@github.com:contributor/hitch.git (fetch)\n"
                ),
            ),
            SimpleNamespace(returncode=0, stdout="feature\n"),
            SimpleNamespace(returncode=0, stdout="abc123\n"),
        ]

        with tempfile.TemporaryDirectory() as cwd:
            pr_watch.validate_published_pr_checkout(
                cwd=cwd,
                url="https://github.com/openai/hitch/pull/42",
            )

    @patch(
        "hitch.main.workflows.pr_watch._resolved_ssh_hostname",
        return_value="github.com",
    )
    @patch("hitch.main.workflows.pr_watch._run_git_cli")
    @patch("hitch.main.workflows.pr_watch._gh_pr_view_payload")
    def test_published_pr_accepts_github_ssh_host_alias(
        self,
        mock_view: MagicMock,
        mock_git: MagicMock,
        mock_resolve_hostname: MagicMock,
    ) -> None:
        mock_view.return_value = {
            "url": "https://github.com/openai/hitch/pull/42",
            "number": 42,
            "state": "OPEN",
            "headRefName": "feature",
            "headRefOid": "abc123",
            "headRepository": {"name": "hitch"},
            "headRepositoryOwner": {"login": "openai"},
        }
        mock_git.side_effect = [
            SimpleNamespace(
                returncode=0,
                stdout=(
                    "origin\tgit@github-work:openai/hitch.git (fetch)\n"
                    "origin\tgit@github-work:openai/hitch.git (push)\n"
                ),
            ),
            SimpleNamespace(returncode=0, stdout="feature\n"),
            SimpleNamespace(returncode=0, stdout="abc123\n"),
        ]

        with tempfile.TemporaryDirectory() as cwd:
            pr_watch.validate_published_pr_checkout(
                cwd=cwd,
                url="https://github.com/openai/hitch/pull/42",
            )

        mock_resolve_hostname.assert_called_once_with("github-work")

    @patch("hitch.main.workflows.pr_watch.subprocess.run")
    def test_resolves_effective_ssh_hostname(self, mock_run: MagicMock) -> None:
        mock_run.return_value = SimpleNamespace(
            returncode=0,
            stdout="host github-work\nhostname GitHub.COM.\nport 22\n",
        )

        hostname = pr_watch._resolved_ssh_hostname("github-work")

        self.assertEqual(hostname, "github.com")
        mock_run.assert_called_once()

    @patch("hitch.main.workflows.pr_watch.observe_pr")
    def test_reports_gh_failure_as_tool_error(self, mock_observe: MagicMock) -> None:
        mock_observe.side_effect = _GhPrOpenError("gh auth failed")

        with (
            tempfile.TemporaryDirectory() as cwd,
            self.assertRaisesRegex(pr_watch.PrWatchError, "gh auth failed"),
        ):
            pr_watch.watch_pr(
                cwd=cwd,
                url="https://github.com/openai/hitch/pull/42",
            )


class PrWatchToolTests(TestCase):
    @override
    def setUp(self) -> None:
        super().setUp()
        cwd = tempfile.TemporaryDirectory()
        self.addCleanup(cwd.cleanup)
        self.cwd = cwd.name

    def test_registered_specs_include_watch_pr(self) -> None:
        specs = registered_dynamic_tool_specs()

        watch = next(spec for spec in specs if spec["name"] == "watch_pr")
        self.assertEqual(watch["namespace"], "hitch")
        self.assertEqual(watch["inputSchema"]["required"], ["url"])

    @patch("hitch.main.runtime.codex_tools.pr_watch.watch_pr")
    def test_tool_records_result_on_owning_workflow(
        self, mock_watch: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd=self.cwd,
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_PR_WATCH_RUNNING,
            state={
                system_agents._WORKFLOW_TURN_OWNER_STEP_STATE_KEY: (
                    system_agents.STEP_PR_WATCH_RUNNING
                ),
                system_agents._WORKFLOW_TURN_OWNER_INDEX_STATE_KEY: 4,
                pr_watch.PR_WATCH_RESULT_STATE_KEY: {
                    "feedback_fingerprint": "seen-feedback"
                },
            },
        )
        result = {
            "status": "ready",
            "summary": "The PR gates are passing.",
            "feedback": "",
            "feedback_fingerprint": "",
            "pr": _observation()["pr"],
            "gates": _observation()["gates"],
            "blockers": [],
        }
        mock_watch.return_value = result

        response = handle_dynamic_tool_call(
            {
                "tool": "watch_pr",
                "arguments": {
                    "url": "https://github.com/openai/hitch/pull/42",
                },
            },
            ToolContext(
                cwd=self.cwd,
                thread_id="main-thread",
                workflow_id=workflow.pk,
                user_message_index=4,
            ),
        )

        self.assertTrue(response["success"])
        self.assertEqual(
            mock_watch.call_args.kwargs["previous_feedback_fingerprint"],
            "seen-feedback",
        )
        self.assertEqual(
            json.loads(response["contentItems"][0]["text"])["status"], "ready"
        )
        workflow.refresh_from_db()
        self.assertEqual(
            workflow.state[pr_watch.PR_WATCH_RESULT_STATE_KEY]["status"], "ready"
        )
        self.assertEqual(workflow.state["pr_handoff"]["pr_number"], 42)

    @patch("hitch.main.runtime.codex_tools.pr_watch.watch_pr")
    @patch("hitch.main.runtime.codex_tools.pr_watch.validate_published_pr_checkout")
    def test_publishing_turn_registers_pr_before_polling(
        self,
        mock_validate_checkout: MagicMock,
        mock_watch: MagicMock,
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd=self.cwd,
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_PR_PROMPT_RUNNING,
            state={
                "open_pr_on_lgtm": True,
                system_agents._WORKFLOW_TURN_OWNER_STEP_STATE_KEY: (
                    system_agents.STEP_PR_PROMPT_RUNNING
                ),
                system_agents._WORKFLOW_TURN_OWNER_INDEX_STATE_KEY: 4,
            },
        )
        result = {
            "status": "ready",
            "summary": "The PR gates are passing.",
            "feedback": "",
            "feedback_fingerprint": "",
            "pr": _observation()["pr"],
            "gates": _observation()["gates"],
            "blockers": [],
        }

        def assert_registered_before_polling(**_kwargs: object) -> dict[str, object]:
            registered = SystemWorkflow.objects.get(pk=workflow.pk)
            self.assertEqual(
                registered.step, system_agents.STEP_PR_WATCH_RUNNING
            )
            self.assertEqual(registered.state["pr_handoff"]["pr_number"], 42)
            self.assertEqual(
                registered.state["hitch_pr_handoff"]["pr_number"], 42
            )
            self.assertEqual(
                registered.state[
                    system_agents._WORKFLOW_TURN_OWNER_STEP_STATE_KEY
                ],
                system_agents.STEP_PR_WATCH_RUNNING,
            )
            return result

        mock_watch.side_effect = assert_registered_before_polling

        response = handle_dynamic_tool_call(
            {
                "tool": "watch_pr",
                "arguments": {
                    "url": "https://github.com/openai/hitch/pull/42",
                },
            },
            ToolContext(
                cwd=self.cwd,
                thread_id="main-thread",
                workflow_id=workflow.pk,
                user_message_index=4,
            ),
        )

        self.assertTrue(response["success"])
        mock_validate_checkout.assert_called_once_with(
            cwd=self.cwd,
            url="https://github.com/openai/hitch/pull/42",
        )
        workflow.refresh_from_db()
        self.assertEqual(workflow.step, system_agents.STEP_PR_WATCH_RUNNING)
        self.assertEqual(
            workflow.state[pr_watch.PR_WATCH_RESULT_TURN_INDEX_STATE_KEY], 4
        )

        instance = CodexInstance.objects.create(
            pid=0,
            thread_id="main-thread",
            cwd=self.cwd,
            prompt=system_agents.PR_SLASH_PROMPT,
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_USER,
            workflow_id=workflow.pk,
            user_message_index=4,
        )
        system_agents.on_codex_instance_finished(instance)

        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(workflow.step, system_agents.STEP_PR_READY)

    @patch("hitch.main.runtime.codex_tools.pr_watch.watch_pr")
    @patch(
        "hitch.main.runtime.codex_tools.pr_watch.validate_published_pr_checkout",
        side_effect=pr_watch.PrWatchError(
            "published PR repository does not match the publishing checkout"
        ),
    )
    def test_publishing_turn_rejects_unrelated_pr_before_registration(
        self,
        _mock_validate_checkout: MagicMock,
        mock_watch: MagicMock,
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd=self.cwd,
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_PR_PROMPT_RUNNING,
            state={
                "open_pr_on_lgtm": True,
                system_agents._WORKFLOW_TURN_OWNER_STEP_STATE_KEY: (
                    system_agents.STEP_PR_PROMPT_RUNNING
                ),
                system_agents._WORKFLOW_TURN_OWNER_INDEX_STATE_KEY: 4,
            },
        )

        response = handle_dynamic_tool_call(
            {
                "tool": "watch_pr",
                "arguments": {
                    "url": "https://github.com/openai/another/pull/42",
                },
            },
            ToolContext(
                cwd=self.cwd,
                thread_id="main-thread",
                workflow_id=workflow.pk,
                user_message_index=4,
            ),
        )

        self.assertFalse(response["success"])
        self.assertIn(
            "does not match the publishing checkout",
            response["contentItems"][0]["text"],
        )
        mock_watch.assert_not_called()
        workflow.refresh_from_db()
        self.assertEqual(workflow.step, system_agents.STEP_PR_PROMPT_RUNNING)
        self.assertNotIn("pr_handoff", workflow.state)

    @patch("hitch.main.runtime.codex_tools.pr_watch.watch_pr")
    def test_tool_does_not_record_another_threads_workflow(
        self, mock_watch: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd=self.cwd,
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_PR_WATCH_RUNNING,
            state={},
        )
        mock_watch.return_value = {
            "status": "ready",
            "summary": "ready",
            "pr": _observation()["pr"],
            "gates": _observation()["gates"],
        }

        response = handle_dynamic_tool_call(
            {
                "tool": "watch_pr",
                "arguments": {
                    "url": "https://github.com/openai/hitch/pull/42",
                },
            },
            ToolContext(
                cwd=self.cwd,
                thread_id="other-thread",
                workflow_id=workflow.pk,
                user_message_index=4,
            ),
        )

        self.assertTrue(response["success"])
        workflow.refresh_from_db()
        self.assertNotIn(pr_watch.PR_WATCH_RESULT_STATE_KEY, workflow.state)

    @patch("hitch.main.runtime.codex_tools.pr_watch.watch_pr")
    def test_tool_does_not_turn_qa_guidance_into_pr_workflow(
        self, mock_watch: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd=self.cwd,
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_PR_PROMPT_RUNNING,
            state={
                "open_pr_on_lgtm": False,
                system_agents._WORKFLOW_TURN_OWNER_STEP_STATE_KEY: (
                    system_agents.STEP_PR_PROMPT_RUNNING
                ),
                system_agents._WORKFLOW_TURN_OWNER_INDEX_STATE_KEY: 4,
            },
        )
        mock_watch.return_value = {
            "status": "ready",
            "summary": "ready",
            "pr": _observation()["pr"],
            "gates": _observation()["gates"],
        }

        response = handle_dynamic_tool_call(
            {
                "tool": "watch_pr",
                "arguments": {
                    "url": "https://github.com/openai/hitch/pull/42",
                },
            },
            ToolContext(
                cwd=self.cwd,
                thread_id="main-thread",
                workflow_id=workflow.pk,
                user_message_index=4,
            ),
        )

        self.assertTrue(response["success"])
        workflow.refresh_from_db()
        self.assertEqual(workflow.step, system_agents.STEP_PR_PROMPT_RUNNING)
        self.assertNotIn("pr_handoff", workflow.state)

    @patch("hitch.main.runtime.codex_tools.pr_watch.watch_pr")
    def test_invalid_url_does_not_register_publishing_handoff(
        self, mock_watch: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd=self.cwd,
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_PR_PROMPT_RUNNING,
            state={
                "open_pr_on_lgtm": True,
                system_agents._WORKFLOW_TURN_OWNER_STEP_STATE_KEY: (
                    system_agents.STEP_PR_PROMPT_RUNNING
                ),
                system_agents._WORKFLOW_TURN_OWNER_INDEX_STATE_KEY: 4,
            },
        )

        response = handle_dynamic_tool_call(
            {
                "tool": "watch_pr",
                "arguments": {"url": "https://example.com/not-a-pr"},
            },
            ToolContext(
                cwd=self.cwd,
                thread_id="main-thread",
                workflow_id=workflow.pk,
                user_message_index=4,
            ),
        )

        self.assertFalse(response["success"])
        mock_watch.assert_not_called()
        workflow.refresh_from_db()
        self.assertEqual(workflow.step, system_agents.STEP_PR_PROMPT_RUNNING)
        self.assertNotIn("pr_handoff", workflow.state)

    @patch("hitch.main.runtime.codex_tools.pr_watch.watch_pr")
    def test_tool_rejects_another_pr_for_owning_workflow(
        self, mock_watch: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd=self.cwd,
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_PR_WATCH_RUNNING,
            state={
                system_agents._PR_HANDOFF_STATE_KEY: _observation()["pr"],
                system_agents._WORKFLOW_TURN_OWNER_STEP_STATE_KEY: (
                    system_agents.STEP_PR_WATCH_RUNNING
                ),
                system_agents._WORKFLOW_TURN_OWNER_INDEX_STATE_KEY: 4,
            },
        )

        response = handle_dynamic_tool_call(
            {
                "tool": "watch_pr",
                "arguments": {
                    "url": "https://github.com/openai/another/pull/42",
                },
            },
            ToolContext(
                cwd=self.cwd,
                thread_id="main-thread",
                workflow_id=workflow.pk,
                user_message_index=4,
            ),
        )

        self.assertFalse(response["success"])
        mock_watch.assert_not_called()

    @patch("hitch.main.runtime.codex_tools.pr_watch.watch_pr")
    def test_failed_later_call_clears_same_turn_success(
        self, mock_watch: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd=self.cwd,
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_PR_WATCH_RUNNING,
            state={
                system_agents._PR_HANDOFF_STATE_KEY: _observation()["pr"],
                system_agents._WORKFLOW_TURN_OWNER_STEP_STATE_KEY: (
                    system_agents.STEP_PR_WATCH_RUNNING
                ),
                system_agents._WORKFLOW_TURN_OWNER_INDEX_STATE_KEY: 4,
                pr_watch.PR_WATCH_RESULT_STATE_KEY: {
                    "status": "ready",
                    "feedback_fingerprint": "seen-feedback",
                },
                pr_watch.PR_WATCH_RESULT_TURN_INDEX_STATE_KEY: 4,
            },
        )
        mock_watch.side_effect = pr_watch.PrWatchError("GitHub unavailable")

        response = handle_dynamic_tool_call(
            {
                "tool": "watch_pr",
                "arguments": {
                    "url": "https://github.com/openai/hitch/pull/42",
                },
            },
            ToolContext(
                cwd=self.cwd,
                thread_id="main-thread",
                workflow_id=workflow.pk,
                user_message_index=4,
            ),
        )

        self.assertFalse(response["success"])
        self.assertEqual(
            mock_watch.call_args.kwargs["previous_feedback_fingerprint"],
            "seen-feedback",
        )
        workflow.refresh_from_db()
        self.assertNotIn(pr_watch.PR_WATCH_RESULT_STATE_KEY, workflow.state)
        self.assertNotIn(
            pr_watch.PR_WATCH_RESULT_TURN_INDEX_STATE_KEY,
            workflow.state,
        )


class AgentDrivenPrWorkflowTests(TestCase):
    @patch("hitch.main.workflows.system_agents._finish_workflow_block")
    def test_reconciliation_blocks_removed_monitor_steps(
        self, mock_finish_block: MagicMock
    ) -> None:
        for step in ("pr_monitoring", "pr_feedback_running"):
            with self.subTest(step=step):
                workflow = SystemWorkflow.objects.create(
                    kind=SystemWorkflow.KIND_PR_QA,
                    main_thread_id=f"legacy-{step}",
                    cwd="/repo",
                    status=SystemWorkflow.STATUS_RUNNING,
                    step=step,
                )

                reconciled = system_agents.reconcile_terminal_workflow_instances(
                    workflow_id=workflow.pk
                )

                self.assertEqual(reconciled, 1)
                workflow.refresh_from_db()
                self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
                self.assertEqual(workflow.step, system_agents.STEP_BLOCKED)
                self.assertIn("hitch.watch_pr", workflow.state["error"])
                mock_finish_block.assert_called_once()
                mock_finish_block.reset_mock()

    @patch("hitch.main.workflows.system_agents._finish_workflow_block")
    @patch("hitch.main.workflows.system_agents.codex_pool.interrupt_instance")
    def test_reconciliation_interrupts_removed_monitor_worker(
        self,
        mock_interrupt: MagicMock,
        mock_finish_block: MagicMock,
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="legacy-monitor",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step="pr_feedback_running",
        )
        worker = CodexInstance.objects.create(
            pid=123,
            thread_id="legacy-feedback",
            cwd="/repo",
            prompt="hidden feedback",
            events_path="/tmp/legacy-feedback-events.jsonl",
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_SYSTEM_FEEDBACK,
            workflow_id=workflow.pk,
        )
        mock_interrupt.return_value = worker

        reconciled = system_agents.reconcile_terminal_workflow_instances(
            workflow_id=workflow.pk
        )

        self.assertEqual(reconciled, 1)
        mock_interrupt.assert_called_once_with(
            worker.pk,
            expected_thread_id="legacy-feedback",
        )
        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
        self.assertIn("hitch.watch_pr", workflow.state["error"])
        self.assertTrue(workflow.state["deferred_failure_surface"])
        mock_finish_block.assert_not_called()

    def test_steering_turn_defers_workflow_owned_watch(self) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_USER_STEERING_RUNNING,
            state={
                "user_steering_resume_step": system_agents.STEP_PR_WATCH_RUNNING,
            },
        )

        instructions = pr_qa._user_steering_developer_instructions(workflow)

        self.assertIn("Do not invoke `hitch.watch_pr`", instructions)
        self.assertIn("workflow-owned watch turn", instructions)

    def test_existing_thread_without_tool_rejects_pr_workflow(self) -> None:
        with self.assertRaisesRegex(
            pr_qa.PrWatchUnavailableError,
            "Start a new session",
        ):
            pr_qa.start_pr_watch_workflow(
                main_thread_id="old-thread",
                cwd="/repo",
                pr_url="https://github.com/openai/hitch/pull/42",
                sandbox_policy="workspace-write",
                approval_mode="auto_review",
                pr_watch_tool_available=False,
            )

    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_turn")
    def test_fix_pr_starts_one_visible_watch_turn(self, mock_spawn: MagicMock) -> None:
        mock_spawn.return_value = MagicMock(spec=CodexInstance)

        workflow = pr_qa.start_pr_watch_workflow(
            main_thread_id="main-thread",
            cwd="/repo",
            pr_url="https://github.com/openai/hitch/pull/42",
            sandbox_policy="workspace-write",
            approval_mode="auto_review",
            initial_user_message_index=3,
        )

        workflow.refresh_from_db()
        self.assertEqual(workflow.step, system_agents.STEP_PR_WATCH_RUNNING)
        self.assertIn("hitch.watch_pr", mock_spawn.call_args.kwargs["prompt"])
        self.assertEqual(
            mock_spawn.call_args.kwargs["purpose"], CodexInstance.PURPOSE_USER
        )

    def test_pr_prompt_assigns_publication_and_watch_to_codex(self) -> None:
        prompt = system_agents.PR_SLASH_PROMPT

        self.assertIn("Codex's built-in PR publishing tool", prompt)
        self.assertIn("`hitch.watch_pr`", prompt)
        self.assertIn("registers the PR with Hitch", prompt)

    def test_ready_tool_result_completes_watch_as_pr_ready(self) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_PR_WATCH_RUNNING,
            state={
                pr_watch.PR_WATCH_RESULT_STATE_KEY: {"status": "ready"},
                pr_watch.PR_WATCH_RESULT_TURN_INDEX_STATE_KEY: 2,
                system_agents._WORKFLOW_TURN_OWNER_STEP_STATE_KEY: (
                    system_agents.STEP_PR_WATCH_RUNNING
                ),
                system_agents._WORKFLOW_TURN_OWNER_INDEX_STATE_KEY: 2,
            },
        )
        instance = CodexInstance.objects.create(
            pid=0,
            thread_id="main-thread",
            cwd="/repo",
            prompt="watch",
            events_path="/tmp/watch-events.jsonl",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_USER,
            workflow_id=workflow.pk,
            user_message_index=2,
        )

        pr_qa._handle_pr_watch_finished(instance, workflow)

        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(workflow.step, system_agents.STEP_PR_READY)

    def test_previous_terminal_result_does_not_complete_resumed_watch(self) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_PR_WATCH_RUNNING,
            state={
                pr_watch.PR_WATCH_RESULT_STATE_KEY: {"status": "terminal"},
                pr_watch.PR_WATCH_RESULT_TURN_INDEX_STATE_KEY: 1,
                system_agents._PR_HANDOFF_STATE_KEY: {
                    "url": "https://github.com/openai/hitch/pull/42",
                    "state": "closed",
                },
                system_agents._WORKFLOW_TURN_OWNER_STEP_STATE_KEY: (
                    system_agents.STEP_PR_WATCH_RUNNING
                ),
                system_agents._WORKFLOW_TURN_OWNER_INDEX_STATE_KEY: 2,
            },
        )
        instance = CodexInstance.objects.create(
            pid=0,
            thread_id="main-thread",
            cwd="/repo",
            prompt="watch again",
            events_path="/tmp/watch-events.jsonl",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_USER,
            workflow_id=workflow.pk,
            user_message_index=2,
        )

        pr_qa._handle_pr_watch_finished(instance, workflow)

        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(workflow.step, system_agents.STEP_PR_WATCH_COMPLETED)

    @patch(
        "hitch.main.workflows.system_agents."
        "_maybe_auto_pull_default_repo_after_pr_merge"
    )
    def test_terminal_tool_result_completes_and_runs_merged_auto_pull(
        self, mock_auto_pull: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_PR_WATCH_RUNNING,
            state={
                pr_watch.PR_WATCH_RESULT_STATE_KEY: {"status": "terminal"},
                pr_watch.PR_WATCH_RESULT_TURN_INDEX_STATE_KEY: 2,
                system_agents._PR_HANDOFF_STATE_KEY: {
                    "url": "https://github.com/openai/hitch/pull/42",
                    "state": "merged",
                    "merged": True,
                },
                system_agents._WORKFLOW_TURN_OWNER_STEP_STATE_KEY: (
                    system_agents.STEP_PR_WATCH_RUNNING
                ),
                system_agents._WORKFLOW_TURN_OWNER_INDEX_STATE_KEY: 2,
            },
        )
        instance = CodexInstance.objects.create(
            pid=0,
            thread_id="main-thread",
            cwd="/repo",
            prompt="watch",
            events_path="/tmp/watch-events.jsonl",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_USER,
            workflow_id=workflow.pk,
            user_message_index=2,
        )

        pr_qa._handle_pr_watch_finished(instance, workflow)

        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(workflow.step, system_agents.STEP_PR_CLOSED)
        mock_auto_pull.assert_called_once_with(workflow)
