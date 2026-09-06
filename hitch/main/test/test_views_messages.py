"""send_message, stop, and approval/input resolution endpoint tests."""


import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, call, patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import (
    Client,
    TestCase,
    override_settings,
)
from django.urls import reverse
from django.utils import timezone
from openai_codex.errors import InvalidRequestError

from hitch.main import caches
from hitch.main.models import (
    ApprovalRequest,
    ArchivedSessionTokenUsage,
    CodexInstance,
    SessionMetadata,
    SessionPullRequest,
    SystemWorkflow,
    UserInputRequest,
)
from hitch.main.runtime import codex_pool
from hitch.main.runtime import rollout as rollout_module
from hitch.main.sessions import agent_tasks
from hitch.main.sessions.settings_cookies import SettingsValues
from hitch.main.test.support import (
    _encode_extra_system_prompt,
    _make_model,
    _make_project,
    _rollout_line,
    _seed_cookies,
)
from hitch.main.test.views_helpers import (
    _ENABLE_MEMORIES_COOKIE,
    _EXTRA_SYSTEM_PROMPT_COOKIE,
    _GIF_BYTES,
    _JPEG_BYTES,
    _MODEL_COOKIE,
    _PNG_BYTES,
    _PR_PROMPT,
    _QA_PROMPT,
    _WEB_SEARCH_COOKIE,
    _WEBP_BYTES,
)
from hitch.main.views import messages as message_views


class SendMessageViewTests(TestCase):
    def test_stored_model_and_effort_uses_atomic_recorded_or_settings_pair(
        self,
    ) -> None:
        settings = SettingsValues(
            model="gpt-5.5",
            reasoning_effort="xhigh",
            sandbox_policy="",
            approval_mode="auto_review",
            extra_system_prompt="",
            use_worktrees=False,
            auto_pr_enabled=False,
            auto_qa_enabled=False,
            web_search_mode="",
            show_archived_sessions=False,
            last_selected_repo="",
            selected_project_id=None,
            visible_session_project_ids=None,
            show_no_project_sessions=True,
            enable_memories=False,
        )
        resumed = SimpleNamespace(
            model="gpt-5.6-sol",
            reasoning_effort="",
            model_config=rollout_module.SessionModelConfig(
                model="gpt-5.6-sol",
                reasoning_effort="",
            ),
        )

        self.assertEqual(
            message_views._stored_model_and_effort(resumed, settings),
            ("gpt-5.6-sol", ""),
        )

        resumed.model = ""
        resumed.model_config = rollout_module.SessionModelConfig(
            model="",
            reasoning_effort="",
        )
        self.assertEqual(
            message_views._stored_model_and_effort(resumed, settings),
            ("gpt-5.5", "xhigh"),
        )

    def _patch_codex(
        self,
        mock_codex: MagicMock,
        *,
        cwd: object = "/repo",
        model: str | None = "gpt-5",
        reasoning_effort: str | None = None,
        models: list[Any] | None = None,
        path: str | None = None,
        turns: list[Any] | None = None,
    ) -> None:
        session_resume_codex_patcher = patch(
            "hitch.main.sessions.session_resume.Codex", new=mock_codex
        )
        session_resume_codex_patcher.start()
        self.addCleanup(session_resume_codex_patcher.stop)
        client = mock_codex.return_value.__enter__.return_value
        thread = SimpleNamespace(cwd=cwd, turns=turns or [])
        if path is not None:
            thread.path = path
        resumed = SimpleNamespace(thread=thread)
        if model is not None:
            resumed.model = model
        if reasoning_effort is not None:
            resumed.reasoning_effort = SimpleNamespace(value=reasoning_effort)
        client._client.thread_resume.return_value = resumed
        client.models.return_value.data = models or []

    def _make_rollout(self, lines: list[str]) -> Path:
        with tempfile.NamedTemporaryFile(
            prefix="rollout-",
            suffix=".jsonl",
            mode="w",
            delete=False,
        ) as fh:
            fh.write("\n".join(lines))
            if lines:
                fh.write("\n")
            path = Path(fh.name)
        self.addCleanup(path.unlink, missing_ok=True)
        return path

    def _make_pending_plan_rollout(
        self,
        plan: str = "# Pending Plan\n\nReady to implement.",
    ) -> Path:
        return self._make_rollout(
            [
                _rollout_line("turn_context", {"collaboration_mode": {"mode": "plan"}}),
                _rollout_line("event_msg", {"type": "user_message", "message": "Plan it"}),
                _rollout_line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": f"<proposed_plan>\n{plan}\n</proposed_plan>",
                            }
                        ],
                        "phase": "final_answer",
                    },
                ),
            ]
        )

    def _make_resolved_plan_rollout(self) -> Path:
        return self._make_rollout(
            [
                _rollout_line("turn_context", {"collaboration_mode": {"mode": "plan"}}),
                _rollout_line("event_msg", {"type": "user_message", "message": "Plan it"}),
                _rollout_line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": (
                                    "<proposed_plan>\n# Plan\n\nImplement it.\n"
                                    "</proposed_plan>"
                                ),
                            }
                        ],
                        "phase": "final_answer",
                    },
                ),
                _rollout_line("turn_context", {"collaboration_mode": {"mode": "default"}}),
                _rollout_line(
                    "event_msg",
                    {"type": "user_message", "message": "Implement the plan."},
                ),
                _rollout_line(
                    "event_msg",
                    {
                        "type": "agent_message",
                        "message": "Implemented.",
                        "phase": "final_answer",
                    },
                ),
            ]
        )

    def _make_plan_discussion_rollout(self) -> Path:
        return self._make_rollout(
            [
                _rollout_line("turn_context", {"collaboration_mode": {"mode": "plan"}}),
                _rollout_line(
                    "event_msg",
                    {"type": "user_message", "message": "Talk through the shape."},
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "This can work; I need one decision first.",
                            }
                        ],
                        "phase": "final_answer",
                    },
                ),
            ]
        )

    def _make_active_plan_mode_rollout_without_plan(self) -> Path:
        return self._make_rollout(
            [
                _rollout_line("turn_context", {"collaboration_mode": {"mode": "plan"}}),
                _rollout_line(
                    "event_msg", {"type": "user_message", "message": "Plan it"}
                ),
                _rollout_line(
                    "event_msg",
                    {
                        "type": "agent_message",
                        "message": "I need to inspect the code first.",
                        "phase": "final_answer",
                    },
                ),
            ]
        )

    def _assert_follow_up_spawn(
        self,
        mock_spawn: MagicMock,
        *,
        prompt: str = "follow-up",
        cwd: str = "/repo",
        **overrides: Any,
    ) -> None:
        expected = {
            "thread_id": "abc",
            "cwd": cwd,
            "prompt": prompt,
            "sandbox_policy": None,
            "approval_mode": "auto_review",
        }
        expected.update(overrides)
        mock_spawn.assert_called_once_with(**expected)

    @patch("hitch.main.runtime.codex_pool.steer_instance")
    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.views.common.Codex")
    def test_steers_posted_active_instance_without_spawning(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_steer: MagicMock,
    ) -> None:
        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "  also update docs  ", "active_instance": "42"},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            response.headers["Location"],
            reverse("session", kwargs={"session_id": "abc"}),
        )
        mock_steer.assert_called_once_with(
            42,
            expected_thread_id="abc",
            prompt="also update docs",
        )
        mock_spawn.assert_not_called()
        mock_codex.assert_not_called()


    @patch("hitch.main.runtime.codex_pool.steer_instance")
    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.views.common.Codex")
    def test_rejects_invalid_active_instance(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_steer: MagicMock,
    ) -> None:
        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "also lint", "active_instance": "not-a-number"},
        )

        self.assertEqual(response.status_code, 400)
        mock_steer.assert_not_called()
        mock_spawn.assert_not_called()
        mock_codex.assert_not_called()

    @patch("hitch.main.repos.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.runtime.codex_pool.worker_is_alive", return_value=False)
    @patch("hitch.main.views.common.Codex")
    def test_dead_posted_active_instance_is_reconciled_before_follow_up_spawn(
        self,
        mock_codex: MagicMock,
        mock_worker_alive: MagicMock,
        mock_spawn: MagicMock,
        _mock_discover: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex)
        instance = CodexInstance.objects.create(
            pid=99999999,
            thread_id="abc",
            cwd="/repo",
            prompt="stale work",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_RUNNING,
        )

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "still there?", "active_instance": str(instance.pk)},
        )

        self.assertEqual(response.status_code, 302)
        mock_worker_alive.assert_called()
        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertIn("worker process exited", instance.error)
        self._assert_follow_up_spawn(mock_spawn, prompt="still there?")

    @patch("hitch.main.runtime.codex_pool.steer_instance")
    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.runtime.codex_pool.worker_is_alive", return_value=True)
    @patch("hitch.main.views.common.Codex")
    def test_image_steer_attachment_cap_returns_bad_request_without_fallback(
        self,
        mock_codex: MagicMock,
        _mock_worker_alive: MagicMock,
        mock_spawn: MagicMock,
        mock_steer: MagicMock,
    ) -> None:
        mock_steer.side_effect = codex_pool.InputAttachmentLimitExceededError(
            "too many image attachments are queued for this turn"
        )
        instance = CodexInstance.objects.create(
            pid=123,
            thread_id="abc",
            cwd="/repo",
            prompt="already running",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_RUNNING,
        )

        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_EVENTS_DIR=Path(raw)),
        ):
            response = self.client.post(
                reverse("send_message", kwargs={"session_id": "abc"}),
                data={
                    "prompt": "use this screenshot",
                    "input_images": SimpleUploadedFile(
                        "screen.png", _PNG_BYTES, content_type="image/png"
                    ),
                },
            )

            self.assertContains(
                response,
                "too many image attachments are queued for this turn",
                status_code=400,
            )
            attachments = Path(raw) / "attachments"
            self.assertEqual(
                [path for path in attachments.rglob("*") if path.is_file()],
                [],
            )

        mock_steer.assert_called_once()
        self.assertEqual(mock_steer.call_args.args[0], instance.pk)
        mock_spawn.assert_not_called()
        mock_codex.assert_not_called()

    @patch(
        "hitch.main.views.messages.thread_has_dynamic_tool",
        return_value=True,
    )
    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.views.common.Codex")
    def test_review_and_pr_shortcuts_spawn_ordinary_visible_turns(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
        _mock_has_watch_tool: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        cases = (
            ("/qa", agent_tasks.review_task(prepare_pull_request=False)),
            (_QA_PROMPT, agent_tasks.review_task(prepare_pull_request=False)),
            ("/pr", agent_tasks.review_task(prepare_pull_request=True)),
            (_PR_PROMPT, agent_tasks.review_task(prepare_pull_request=True)),
            ("/PR-NOW", agent_tasks.publish_pr_task()),
        )

        for prompt, task in cases:
            with self.subTest(prompt=prompt):
                self._patch_codex(
                    mock_codex,
                    model="gpt-5.4",
                    reasoning_effort="high",
                )
                mock_spawn.reset_mock()

                response = self.client.post(
                    reverse("send_message", kwargs={"session_id": "abc"}),
                    data={"prompt": prompt, "plan_mode": "true"},
                )

                self.assertEqual(response.status_code, 302)
                kwargs = mock_spawn.call_args.kwargs
                self.assertEqual(kwargs["thread_id"], "abc")
                self.assertEqual(kwargs["cwd"], "/repo")
                self.assertEqual(kwargs["prompt"], task.prompt)
                if prompt != "/PR-NOW":
                    self.assertIn("merge base", kwargs["prompt"])
                    self.assertIn(
                        "committed, staged, unstaged, and untracked",
                        kwargs["prompt"],
                    )
                    self.assertIn(
                        "Delegate review to Codex subagents as you see fit",
                        kwargs["prompt"],
                    )
                    self.assertNotIn("agent_type", kwargs["prompt"])
                self.assertEqual(kwargs["agent_kind"], task.agent_kind)
                self.assertEqual(kwargs["model"], "gpt-5.4")
                self.assertEqual(kwargs["reasoning_effort"], "high")
                self.assertNotIn("plan_mode", kwargs)
                self.assertNotIn("workflow_id", kwargs)
                self.assertFalse(
                    SystemWorkflow.objects.filter(main_thread_id="abc").exists()
                )

    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.views.common.Codex")
    def test_review_and_pr_shortcuts_reject_image_uploads_before_side_effects(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
    ) -> None:
        for prompt in ("/pr", "/pr-now", "/qa", "/fix-pr"):
            with self.subTest(prompt=prompt):
                response = self.client.post(
                    reverse("send_message", kwargs={"session_id": "abc"}),
                    data={
                        "prompt": prompt,
                        "input_images": SimpleUploadedFile(
                            "screen.png", _PNG_BYTES, content_type="image/png"
                        ),
                    },
                )

                self.assertContains(
                    response,
                    "image attachments are not supported for review or PR tasks",
                    status_code=400,
                )
                mock_spawn.assert_not_called()
                mock_codex.assert_not_called()

    @patch(
        "hitch.main.views.messages.thread_has_dynamic_tool",
        return_value=True,
    )
    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.views.common.Codex")
    def test_fix_pr_uses_registered_session_pr(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
        _mock_has_watch_tool: MagicMock,
    ) -> None:
        pr_url = "https://github.com/cberner/hitch/pull/594"
        SessionPullRequest.objects.create(
            thread_id="abc",
            cwd="/repo",
            state={
                "pr_handoff": {
                    "url": pr_url,
                    "repository_full_name": "cberner/hitch",
                    "pr_number": 594,
                    "state": "open",
                }
            },
        )
        self._patch_codex(mock_codex, model="gpt-5.4", reasoning_effort="high")
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "/fix-pr"},
        )

        self.assertEqual(response.status_code, 302)
        kwargs = mock_spawn.call_args.kwargs
        self.assertEqual(kwargs["agent_kind"], agent_tasks.PR_WATCH_AGENT_KIND)
        self.assertIn(pr_url, kwargs["prompt"])

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.views.common.Codex")
    def test_fix_pr_requires_registered_pr(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex, model="gpt-5.4")
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "/fix-pr"},
        )

        self.assertContains(
            response,
            "fix-pr requires an opened PR for this session",
            status_code=400,
        )
        mock_spawn.assert_not_called()

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.views.common.Codex")
    def test_qa_shortcut_forwards_web_search_setting(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex, model="gpt-5.4")
        mock_discover.return_value = [Path("/repo")]
        _seed_cookies(self.client, **{_WEB_SEARCH_COOKIE: "cached"})

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "/qa"},
        )

        self.assertEqual(response.status_code, 302)
        kwargs = mock_spawn.call_args.kwargs
        self.assertEqual(kwargs["web_search_mode"], "cached")
        self.assertEqual(kwargs["agent_kind"], agent_tasks.REVIEW_AGENT_KIND)

    @patch(
        "hitch.main.views.messages.thread_has_dynamic_tool",
        return_value=False,
    )
    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.views.common.Codex")
    def test_pr_shortcut_requires_watch_tool(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
        _mock_has_watch_tool: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex, model="gpt-5.4")
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "/pr"},
        )

        self.assertContains(response, "hitch.watch_pr is unavailable", status_code=400)
        mock_spawn.assert_not_called()

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.views.common.Codex")
    def test_first_follow_up_uses_project_developer_prompt(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        project = _make_project(
            extra_system_prompt="Use project fixtures.",
        )
        SessionMetadata.objects.create(thread_id="abc", cwd="/repo", project=project)
        _seed_cookies(
            self.client,
            **{
                _EXTRA_SYSTEM_PROMPT_COOKIE: _encode_extra_system_prompt(
                    "Always run focused tests."
                )
            },
        )
        self._patch_codex(mock_codex)
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "follow-up"},
        )

        self.assertEqual(response.status_code, 302)
        self._assert_follow_up_spawn(
            mock_spawn,
            developer_instructions=(
                "Always run focused tests.\n\nUse project fixtures."
            ),
        )


    @patch("hitch.main.worktrees.discover_managed_worktrees", return_value=[])
    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.views.common.Codex")
    def test_archived_follow_up_rejects_disallowed_cached_cwd_before_unarchive(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
        _mock_managed_worktrees: MagicMock,
    ) -> None:
        archived_path = (
            "/tmp/archived_sessions/rollout-2026-06-07T05-43-07-abc.jsonl"
        )
        metadata = SessionMetadata.objects.create(
            thread_id="abc",
            cwd="/elsewhere",
            codex_path=archived_path,
            codex_archived=True,
            codex_archived_at=timezone.now(),
        )
        ArchivedSessionTokenUsage.objects.create(thread_id="abc", total_tokens=100)
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "follow-up"},
        )

        self.assertContains(
            response,
            "thread cwd is not an allowed repository",
            status_code=400,
        )
        mock_codex.assert_not_called()
        mock_spawn.assert_not_called()
        metadata.refresh_from_db()
        self.assertTrue(metadata.codex_archived)
        self.assertIsNotNone(metadata.codex_archived_at)
        self.assertEqual(metadata.codex_path, archived_path)
        self.assertTrue(
            ArchivedSessionTokenUsage.objects.filter(thread_id="abc").exists()
        )

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.views.common.Codex")
    def test_archived_live_resume_retry_unarchives_before_spawning(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex)
        mock_discover.return_value = [Path("/repo")]
        client = mock_codex.return_value.__enter__.return_value
        resumed = client._client.thread_resume.return_value
        client._client.thread_resume.side_effect = [
            InvalidRequestError(
                -32600,
                "session abc is archived. Run `codex unarchive abc` to unarchive it first.",
            ),
            resumed,
        ]
        ArchivedSessionTokenUsage.objects.create(thread_id="abc", total_tokens=100)
        ArchivedSessionTokenUsage.objects.create(
            thread_id="other", total_tokens=200
        )

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "follow-up"},
        )

        self.assertEqual(response.status_code, 302)
        client.thread_unarchive.assert_called_once_with("abc")
        self.assertEqual(
            client._client.thread_resume.call_args_list,
            [call("abc"), call("abc")],
        )
        self._assert_follow_up_spawn(mock_spawn)
        self.assertFalse(
            ArchivedSessionTokenUsage.objects.filter(thread_id="abc").exists()
        )
        self.assertTrue(
            ArchivedSessionTokenUsage.objects.filter(thread_id="other").exists()
        )

    @patch("hitch.main.worktrees.discover_managed_worktrees", return_value=[])
    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.views.common.Codex")
    def test_archived_live_resume_retry_rejects_cached_cwd_before_unarchive(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
        _mock_managed_worktrees: MagicMock,
    ) -> None:
        SessionMetadata.objects.create(thread_id="abc", cwd="/elsewhere")
        ArchivedSessionTokenUsage.objects.create(thread_id="abc", total_tokens=100)
        self._patch_codex(mock_codex)
        mock_discover.return_value = [Path("/repo")]
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.side_effect = InvalidRequestError(
            -32600,
            "session abc is archived. Run `codex unarchive abc` to unarchive it first.",
        )

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "follow-up"},
        )

        self.assertContains(
            response,
            "thread cwd is not an allowed repository",
            status_code=400,
        )
        client._client.thread_resume.assert_called_once_with("abc")
        client.thread_unarchive.assert_not_called()
        mock_spawn.assert_not_called()
        self.assertTrue(
            ArchivedSessionTokenUsage.objects.filter(thread_id="abc").exists()
        )

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.views.common.Codex")
    def test_archived_follow_up_rearchives_when_default_model_missing(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        metadata = SessionMetadata.objects.create(
            thread_id="abc",
            cwd="/repo",
            codex_path="/tmp/archived_sessions/rollout-2026-06-07T05-43-07-abc.jsonl",
            codex_archived=True,
            codex_archived_at=timezone.now(),
        )
        ArchivedSessionTokenUsage.objects.create(thread_id="abc", total_tokens=100)
        self._patch_codex(mock_codex, model=None, models=[])
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "Implement it.", "collaboration_mode": "default"},
        )

        self.assertContains(
            response,
            "default collaboration mode requires a model",
            status_code=400,
        )
        client = mock_codex.return_value.__enter__.return_value
        client.thread_unarchive.assert_called_once_with("abc")
        client.thread_archive.assert_called_once_with("abc")
        mock_spawn.assert_not_called()
        metadata.refresh_from_db()
        self.assertTrue(metadata.codex_archived)
        self.assertIsNotNone(metadata.codex_archived_at)
        self.assertEqual(metadata.codex_path, "")
        self.assertFalse(
            ArchivedSessionTokenUsage.objects.filter(thread_id="abc").exists()
        )

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.views.common.Codex")
    def test_disk_resume_plan_turn_recovers_thread_model(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        # A plan turn on the disk path for a thread Hitch never recorded a model
        # for (no CodexInstance) must recover the thread's actual model via a
        # one-off live resume -- preferring it over the catalog default -- rather
        # than 400 "requires a model" or sending the wrong model.
        def _clear_models_cache() -> None:
            with caches._MODELS_REFRESH_LOCK:
                caches._MODELS_CACHE_VALUE = {}
                caches._MODELS_CACHE_FETCHED_AT = {}
                caches._MODELS_REFRESH_IN_FLIGHT = set()

        _clear_models_cache()
        self.addCleanup(_clear_models_cache)
        rollout_path = self._make_rollout(
            [
                _rollout_line("event_msg", {"type": "user_message", "message": "Hi"}),
                _rollout_line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Done."}],
                        "phase": "final_answer",
                    },
                ),
            ]
        )
        SessionMetadata.objects.create(
            thread_id="abc", cwd="/repo", codex_path=str(rollout_path)
        )
        # The live resume reports the thread's real model ("gpt-5"); the catalog
        # default ("gpt-default") must not win over it.
        self._patch_codex(
            mock_codex,
            model="gpt-5",
            models=[_make_model("gpt-default", is_default=True)],
        )
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={
                "prompt": "make a plan",
                "plan_mode": "true",
                "plan_mode_explicit": "true",
            },
        )

        self.assertEqual(response.status_code, 302)
        self._assert_follow_up_spawn(
            mock_spawn, prompt="make a plan", model="gpt-5", plan_mode=True
        )
        # The model-sensitive turn recovered the thread model via a live resume.
        mock_codex.assert_called()

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.views.common.Codex")
    def test_spawns_turn_with_multiple_uploaded_image_formats(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex)
        mock_discover.return_value = [Path("/repo")]
        uploads = [
            ("screen.png", _PNG_BYTES, ".png", "image/png"),
            ("photo.jpg", _JPEG_BYTES, ".jpg", "image/jpeg"),
            ("clip.gif", _GIF_BYTES, ".gif", "image/gif"),
            ("mock.webp", _WEBP_BYTES, ".webp", "image/webp"),
        ]

        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_EVENTS_DIR=Path(raw)),
        ):
            response = self.client.post(
                reverse("send_message", kwargs={"session_id": "abc"}),
                data={
                    "prompt": "use these screenshots",
                    "input_images": [
                        SimpleUploadedFile(name, body, content_type=content_type)
                        for name, body, _suffix, content_type in uploads
                    ],
                },
            )

            self.assertEqual(response.status_code, 302)
            image_paths = mock_spawn.call_args.kwargs["input_image_paths"]
            self.assertEqual(
                [Path(path).suffix for path in image_paths],
                [suffix for _name, _body, suffix, _content_type in uploads],
            )
            self.assertEqual(
                [Path(path).read_bytes() for path in image_paths],
                [body for _name, body, _suffix, _content_type in uploads],
            )

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.views.common.Codex")
    def test_send_message_cleans_uploaded_images_when_spawn_handoff_fails(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex)
        mock_discover.return_value = [Path("/repo")]
        mock_spawn.side_effect = RuntimeError("launch failed")

        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_EVENTS_DIR=Path(raw)),
        ):
            with self.assertRaises(RuntimeError):
                self.client.post(
                    reverse("send_message", kwargs={"session_id": "abc"}),
                    data={
                        "prompt": "use this screenshot",
                        "input_images": SimpleUploadedFile(
                            "screen.png", _PNG_BYTES, content_type="image/png"
                        ),
                    },
                )

            attachments = Path(raw) / "attachments"
            self.assertEqual(
                [path for path in attachments.rglob("*") if path.is_file()],
                [],
            )

    @patch("hitch.main.runtime.codex_pool.steer_instance")
    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.views.common.Codex")
    def test_send_message_rejects_invalid_image_uploads_before_side_effects(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_steer: MagicMock,
    ) -> None:
        cases: list[tuple[str, object, str]] = [
            (
                "too many",
                [
                    SimpleUploadedFile(
                        f"screen-{index}.png", _PNG_BYTES, content_type="image/png"
                    )
                    for index in range(5)
                ],
                "at most 4 image attachments are allowed",
            ),
            (
                "empty",
                SimpleUploadedFile("screen.png", b"", content_type="image/png"),
                "image attachment is empty",
            ),
            (
                "bad magic",
                SimpleUploadedFile("screen.png", b"not an image", content_type="image/png"),
                "image attachment must be PNG, JPEG, GIF, or WebP",
            ),
        ]

        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_EVENTS_DIR=Path(raw)),
        ):
            for label, upload, message in cases:
                with self.subTest(label=label):
                    mock_steer.reset_mock()
                    mock_spawn.reset_mock()
                    mock_codex.reset_mock()
                    response = self.client.post(
                        reverse("send_message", kwargs={"session_id": "abc"}),
                        data={
                            "prompt": "use this",
                            "active_instance": "42",
                            "input_images": upload,
                        },
                    )

                    self.assertContains(response, message, status_code=400)
                    mock_steer.assert_not_called()
                    mock_spawn.assert_not_called()
                    mock_codex.assert_not_called()
                    self.assertFalse((Path(raw) / "attachments").exists())

            with patch("hitch.main.runtime.input_images._INPUT_IMAGE_MAX_BYTES", len(_PNG_BYTES) - 1):
                response = self.client.post(
                    reverse("send_message", kwargs={"session_id": "abc"}),
                    data={
                        "prompt": "use this",
                        "active_instance": "42",
                        "input_images": SimpleUploadedFile(
                            "screen.png", _PNG_BYTES, content_type="image/png"
                        ),
                    },
                )

            self.assertContains(response, "image attachment is too large", status_code=400)
            mock_steer.assert_not_called()
            mock_spawn.assert_not_called()
            mock_codex.assert_not_called()
            self.assertFalse((Path(raw) / "attachments").exists())


    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.views.common.Codex")
    def test_forwards_follow_up_cookie_options_to_spawn_turn(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex)
        mock_discover.return_value = [Path("/repo")]
        cases: list[tuple[str, dict[str, str], dict[str, object]]] = [
            (
                "sandbox policy",
                {"hitch_sandbox_policy": "workspaceWrite"},
                {"sandbox_policy": "workspaceWrite", "approval_mode": "auto_review"},
            ),
            (
                "invalid sandbox policy",
                {"hitch_sandbox_policy": "phantomPolicy"},
                {"sandbox_policy": None, "approval_mode": "auto_review"},
            ),
            (
                "memories",
                {_ENABLE_MEMORIES_COOKIE: "true"},
                {
                    "sandbox_policy": None,
                    "approval_mode": "auto_review",
                    "enable_memories": True,
                },
            ),
            (
                "web search",
                {_WEB_SEARCH_COOKIE: "live"},
                {
                    "sandbox_policy": None,
                    "approval_mode": "auto_review",
                    "web_search_mode": "live",
                },
            ),
            (
                "deny all approval mode",
                {"hitch_approval_mode": "deny_all"},
                {"sandbox_policy": None, "approval_mode": "deny_all"},
            ),
            (
                "prompt user approval mode",
                {"hitch_approval_mode": "prompt_user"},
                {"sandbox_policy": None, "approval_mode": "prompt_user"},
            ),
            (
                "invalid approval mode",
                {"hitch_approval_mode": "phantomMode"},
                {"sandbox_policy": None, "approval_mode": "auto_review"},
            ),
        ]

        for label, cookies, expected_options in cases:
            with self.subTest(label=label):
                self.client.cookies.clear()
                mock_spawn.reset_mock()
                _seed_cookies(self.client, **cookies)

                response = self.client.post(
                    reverse("send_message", kwargs={"session_id": "abc"}),
                    data={"prompt": "follow-up"},
                )

                self.assertEqual(response.status_code, 302)
                mock_spawn.assert_called_once_with(
                    thread_id="abc",
                    cwd="/repo",
                    prompt="follow-up",
                    **expected_options,
                )

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.views.common.Codex")
    def test_follow_up_uses_session_approval_mode_override(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex)
        mock_discover.return_value = [Path("/repo")]
        SessionMetadata.objects.create(
            thread_id="abc",
            cwd="/repo",
            approval_mode="deny_all",
        )
        _seed_cookies(self.client, hitch_approval_mode="prompt_user")

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "follow-up"},
        )

        self.assertEqual(response.status_code, 302)
        mock_spawn.assert_called_once_with(
            thread_id="abc",
            cwd="/repo",
            prompt="follow-up",
            sandbox_policy=None,
            approval_mode="deny_all",
        )

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.views.common.Codex")
    @patch(
        "hitch.main.views.messages.thread_has_dynamic_tool",
        return_value=True,
    )
    def test_plan_routing_to_spawn_matrix(
        self,
        _mock_has_watch_tool: MagicMock,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        cases: list[
            tuple[str, dict[str, str], str | None, str | None, bool, dict[str, Any]]
        ] = [
            (
                "explicit plan mode",
                {"prompt": "make a migration plan", "plan_mode": "true"},
                None,
                "gpt-5.4",
                False,
                {
                    "prompt": "make a migration plan",
                    "model": "gpt-5.4",
                    "plan_mode": True,
                },
            ),
            (
                "plan slash strips prefix",
                {"prompt": "/plan make a migration plan"},
                None,
                "gpt-5.4",
                False,
                {
                    "prompt": "make a migration plan",
                    "model": "gpt-5.4",
                    "plan_mode": True,
                },
            ),
            (
                "resolved pending default is recomputed",
                {
                    "prompt": "follow up",
                    "plan_mode": "true",
                    "default_plan_mode": "true",
                },
                "resolved",
                "gpt-5.4",
                False,
                {"prompt": "follow up"},
            ),
            (
                "pending follow-up defaults to plan mode",
                {"prompt": "tighten the QA part"},
                "pending",
                "gpt-5.4",
                False,
                {
                    "prompt": "tighten the QA part",
                    "model": "gpt-5.4",
                    "plan_mode": True,
                },
            ),
            (
                "explicit toggle off does not leave pending plan mode",
                {
                    "prompt": "ship it without more planning",
                    "default_plan_mode": "true",
                    "plan_mode_explicit": "true",
                },
                "pending",
                "gpt-5.4",
                False,
                {
                    "prompt": "ship it without more planning",
                    "model": "gpt-5.4",
                    "plan_mode": True,
                },
            ),
            (
                "pending default keeps plan mode",
                {
                    "prompt": "tighten the QA part",
                    "plan_mode": "true",
                    "default_plan_mode": "true",
                },
                "pending",
                "gpt-5.4",
                False,
                {
                    "prompt": "tighten the QA part",
                    "model": "gpt-5.4",
                    "plan_mode": True,
                },
            ),
            (
                "pending default without model falls back",
                {
                    "prompt": "tighten the QA part",
                    "plan_mode": "true",
                    "default_plan_mode": "true",
                },
                "pending",
                None,
                False,
                {"prompt": "tighten the QA part"},
            ),
            (
                "active plan mode without proposed plan stays in plan mode",
                {"prompt": "now give me the plan"},
                "active",
                "gpt-5.4",
                False,
                {
                    "prompt": "now give me the plan",
                    "model": "gpt-5.4",
                    "plan_mode": True,
                },
            ),
            (
                "explicit toggle off leaves active plan mode without proposed plan",
                {
                    "prompt": "answer directly",
                    "default_plan_mode": "true",
                    "plan_mode_explicit": "true",
                },
                "active",
                "gpt-5.4",
                False,
                {
                    "prompt": "answer directly",
                    "model": "gpt-5.4",
                    "collaboration_mode": "default",
                },
            ),
            (
                "approval prompt enters default collaboration",
                {
                    "prompt": "Implement the plan.",
                    "plan_mode": "true",
                    "default_plan_mode": "true",
                },
                "pending",
                "gpt-5.4",
                False,
                {
                    "prompt": "Implement the plan.",
                    "model": "gpt-5.4",
                    "collaboration_mode": "default",
                },
            ),
            (
                "posted default collaboration wins over plan default",
                {
                    "prompt": "Implement the plan.",
                    "collaboration_mode": "default",
                    "plan_mode": "true",
                    "default_plan_mode": "true",
                },
                "pending",
                "gpt-5.4",
                False,
                {
                    "prompt": "Implement the plan.",
                    "model": "gpt-5.4",
                    "collaboration_mode": "default",
                },
            ),
            (
                "approve action enters default collaboration",
                {
                    "prompt": "Implement the plan.",
                    "plan_action": "approve",
                    "plan_mode": "true",
                    "default_plan_mode": "true",
                },
                "pending",
                "gpt-5.4",
                False,
                {
                    "prompt": "Implement the plan.",
                    "model": "gpt-5.4",
                    "collaboration_mode": "default",
                },
            ),
            (
                "auto-pr approve action marks implementation turn",
                {
                    "prompt": "Implement the plan.",
                    "plan_action": "approve",
                    "plan_mode": "true",
                    "default_plan_mode": "true",
                },
                "pending",
                "gpt-5.4",
                True,
                {
                    "prompt": agent_tasks.with_automatic_review_guidance(
                        "Implement the plan.",
                        auto_pr_enabled=True,
                        auto_qa_enabled=False,
                    ),
                    "model": "gpt-5.4",
                    "collaboration_mode": "default",
                    "agent_kind": agent_tasks.PR_PUBLISH_AGENT_KIND,
                    "user_message_index": 1,
                },
            ),
            (
                "revise action stays in plan mode",
                {"prompt": "Revise the plan.", "plan_action": "revise"},
                "pending",
                "gpt-5.4",
                False,
                {
                    "prompt": "Revise the plan.",
                    "model": "gpt-5.4",
                    "plan_mode": True,
                },
            ),
        ]

        for label, data, rollout, model, auto_pr_enabled, expected in cases:
            with self.subTest(label=label):
                SessionMetadata.objects.all().delete()
                rollout_path = None
                if rollout == "pending":
                    rollout_path = str(self._make_pending_plan_rollout())
                elif rollout == "active":
                    rollout_path = str(self._make_active_plan_mode_rollout_without_plan())
                elif rollout == "resolved":
                    rollout_path = str(self._make_resolved_plan_rollout())
                else:
                    self.assertIsNone(rollout)
                if auto_pr_enabled:
                    SessionMetadata.objects.create(
                        thread_id="abc",
                        cwd="/repo",
                        auto_pr_enabled=True,
                    )
                self._patch_codex(mock_codex, model=model, path=rollout_path)
                mock_spawn.reset_mock()

                response = self.client.post(
                    reverse("send_message", kwargs={"session_id": "abc"}),
                    data=data,
                )

                self.assertEqual(response.status_code, 302)
                self._assert_follow_up_spawn(mock_spawn, **expected)

    @patch(
        "hitch.main.views.messages.thread_has_dynamic_tool",
        return_value=False,
    )
    @patch("hitch.main.repos.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.views.common.Codex")
    def test_auto_pr_follow_up_stays_ordinary_without_watch_tool(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        _mock_discover: MagicMock,
        mock_has_watch_tool: MagicMock,
    ) -> None:
        SessionMetadata.objects.create(
            thread_id="abc",
            cwd="/repo",
            auto_pr_enabled=True,
        )
        self._patch_codex(mock_codex)

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "follow-up"},
        )

        self.assertEqual(response.status_code, 302)
        self._assert_follow_up_spawn(mock_spawn)
        mock_has_watch_tool.assert_called_once_with(
            "abc", namespace="hitch", name="watch_pr"
        )

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.views.common.Codex")
    def test_follow_up_after_plan_mode_discussion_stays_in_plan_mode(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        CodexInstance.objects.create(
            pid=os.getpid(),
            thread_id="abc",
            cwd="/repo",
            prompt="Talk through the shape.",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_COMPLETED,
            plan_mode=True,
        )
        self._patch_codex(
            mock_codex,
            model="gpt-5.4",
            path=str(self._make_plan_discussion_rollout()),
        )

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "yes, make that the plan"},
        )

        self.assertEqual(response.status_code, 302)
        self._assert_follow_up_spawn(
            mock_spawn,
            prompt="yes, make that the plan",
            model="gpt-5.4",
            plan_mode=True,
        )











    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.views.common.Codex")
    def test_plan_mode_model_resolution_matrix(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        cases = [
            ("saved cookie", {_MODEL_COOKIE: "gpt-saved"}, [], "gpt-saved", 302),
            (
                "default model",
                {},
                [_make_model("gpt-default", is_default=True)],
                "gpt-default",
                302,
            ),
            ("unresolved", {}, [], None, 400),
        ]

        for label, cookies, models, expected_model, expected_status in cases:
            with self.subTest(label=label):
                client = Client()
                self._patch_codex(mock_codex, model=None, models=models)
                mock_spawn.reset_mock()
                if cookies:
                    _seed_cookies(client, **cookies)

                response = client.post(
                    reverse("send_message", kwargs={"session_id": "abc"}),
                    data={"prompt": "make a migration plan", "plan_mode": "true"},
                )

                self.assertEqual(response.status_code, expected_status)
                if expected_model is None:
                    self.assertContains(
                        response, "plan mode requires a model", status_code=400
                    )
                    mock_spawn.assert_not_called()
                else:
                    self._assert_follow_up_spawn(
                        mock_spawn,
                        prompt="make a migration plan",
                        model=expected_model,
                        plan_mode=True,
                    )

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.views.common.Codex")
    def test_rejects_invalid_input(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]

        # cwd-missing and cwd-outside-allowlist need the resumed thread set up;
        # the empty-prompt cases never reach Codex, but stubbing it is cheap.
        cases: list[tuple[dict[str, str], str | None, str, str, bool]] = [
            ({"prompt": ""}, "/repo", "empty prompt", "prompt is required", False),
            (
                {"prompt": "   \n  "},
                "/repo",
                "whitespace-only prompt",
                "prompt is required",
                False,
            ),
            (
                {"prompt": "Implement the plan.", "plan_action": "ship"},
                "/repo",
                "invalid plan action",
                "invalid plan action",
                False,
            ),
            (
                {"prompt": "hi", "collaboration_mode": "pair"},
                "/repo",
                "invalid collaboration mode",
                "invalid collaboration mode",
                False,
            ),
            (
                {
                    "prompt": "hi",
                    "collaboration_mode": "default",
                    "plan_mode": "true",
                    "plan_mode_explicit": "true",
                },
                "/repo",
                "collaboration conflicts with explicit plan mode",
                "collaboration mode conflicts with plan mode",
                False,
            ),
            (
                {"prompt": "/pr", "collaboration_mode": "default"},
                "/repo",
                "PR task conflicts with collaboration",
                "review and PR tasks conflict with collaboration mode",
                False,
            ),
            ({"prompt": "hi"}, None, "thread without cwd", "thread has no cwd", True),
            # The session list shows every thread the app-server knows about,
            # so a resumed thread's cwd can point outside the discover_repos()
            # allowlist (e.g. for threads created by another tool). The
            # composer must refuse to spawn a worker in such a directory.
            (
                {"prompt": "hi"},
                "/etc",
                "cwd outside allowed list",
                "thread cwd is not an allowed repository",
                True,
            ),
        ]
        for data, cwd, label, message, codex_called in cases:
            with self.subTest(label=label):
                self._patch_codex(mock_codex, cwd=cwd)
                mock_codex.reset_mock()
                mock_spawn.reset_mock()
                response = self.client.post(
                    reverse("send_message", kwargs={"session_id": "abc"}),
                    data=data,
                )
                self.assertContains(response, message, status_code=400)
                if codex_called:
                    mock_codex.assert_called_once()
                else:
                    mock_codex.assert_not_called()
                mock_spawn.assert_not_called()

class StopSessionViewTests(TestCase):
    @patch("hitch.main.runtime.codex_pool.interrupt_instance")
    @patch("hitch.main.runtime.codex_pool.interrupt_active")
    def test_stop_with_selected_images_still_interrupts_worker(
        self,
        mock_interrupt_active: MagicMock,
        mock_interrupt_instance: MagicMock,
    ) -> None:
        response = self.client.post(
            reverse("stop_session", kwargs={"session_id": "abc"}),
            data={
                "instance": "42",
                "input_images": SimpleUploadedFile(
                    "screen.png", _PNG_BYTES, content_type="image/png"
                ),
            },
        )

        self.assertEqual(response.status_code, 302)
        mock_interrupt_instance.assert_called_once_with(42, expected_thread_id="abc")
        mock_interrupt_active.assert_not_called()

    @patch("hitch.main.runtime.codex_pool.interrupt_instance")
    @patch("hitch.main.runtime.codex_pool.interrupt_active")
    def test_falls_back_to_latest_active_without_instance(
        self,
        mock_interrupt_active: MagicMock,
        mock_interrupt_instance: MagicMock,
    ) -> None:
        mock_interrupt_active.return_value = None

        response = self.client.post(
            reverse("stop_session", kwargs={"session_id": "abc"})
        )

        self.assertEqual(response.status_code, 302)
        mock_interrupt_active.assert_called_once_with("abc")
        mock_interrupt_instance.assert_not_called()

    @patch("hitch.main.runtime.codex_pool.interrupt_instance")
    @patch("hitch.main.runtime.codex_pool.interrupt_active")
    def test_rejects_invalid_requests(
        self, mock_interrupt_active: MagicMock, mock_interrupt_instance: MagicMock
    ) -> None:
        # Tampered/oversized values must be rejected at the view boundary so
        # they never reach ``objects.get`` (which would raise backend-specific
        # OverflowError/DataError and surface as a 500 instead of a clean 400).
        url = reverse("stop_session", kwargs={"session_id": "abc"})
        cases: list[tuple[str, dict[str, str], str, int]] = [
            ("post", {"instance": "not-a-number"}, "non-integer", 400),
            ("post", {"instance": "0"}, "zero", 400),
            ("post", {"instance": "-1"}, "negative", 400),
            ("post", {"instance": str(2**63)}, "above BigAutoField max", 400),
            ("get", {}, "method", 405),
        ]
        for method, data, label, status in cases:
            with self.subTest(label=label):
                if method == "post":
                    response = self.client.post(url, data=data)
                else:
                    response = self.client.get(url)
                self.assertEqual(response.status_code, status)
        mock_interrupt_active.assert_not_called()
        mock_interrupt_instance.assert_not_called()

class ResolveApprovalViewTests(TestCase):
    """The ``POST /approval/<id>/`` endpoint that records the user's pick on
    a pending command/file approval. The worker's polling loop wakes on the
    row update and answers codex's JSON-RPC request with the recorded
    decision — see ``hitch.main.management.commands.codex_worker``."""

    def _make_approval(
        self,
        *,
        decision: str = ApprovalRequest.DECISION_PENDING,
        params: dict[str, object] | None = None,
    ) -> ApprovalRequest:
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="thread-1",
            cwd="/repo",
            prompt="hi",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
        )
        return ApprovalRequest.objects.create(
            instance=instance,
            method="item/commandExecution/requestApproval",
            params=params or {"item": {"command": "ls"}},
            decision=decision,
        )

    def test_accepts_each_valid_decision(self) -> None:
        """Pin the wire-string contract — these three values are what
        app-server's approval response schema accepts (``accept`` /
        ``decline`` / ``cancel``). A regression that drops one of them
        would silently break that decision in the UI."""
        for decision in ("accept", "decline", "cancel"):
            with self.subTest(decision=decision):
                approval = self._make_approval()
                response = self.client.post(
                    reverse("resolve_approval", kwargs={"approval_id": approval.pk}),
                    data={"decision": decision},
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.content, decision.encode())
                approval.refresh_from_db()
                self.assertEqual(approval.decision, decision)
                self.assertIsNotNone(approval.decided_at)

    def test_accepts_structured_execpolicy_amendment_decision(self) -> None:
        """Codex can offer a structured accept decision that both runs the
        command and persists the proposed command-prefix approval."""
        payload = {
            "acceptWithExecpolicyAmendment": {
                "execpolicy_amendment": ["just", "test"]
            }
        }
        approval = self._make_approval(
            params={
                "item": {"command": "just test"},
                "availableDecisions": ["accept", payload, "cancel"],
            }
        )

        response = self.client.post(
            reverse("resolve_approval", kwargs={"approval_id": approval.pk}),
            data={
                "decision": "accept",
                "decision_payload": json.dumps(payload),
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"accept")
        approval.refresh_from_db()
        self.assertEqual(approval.decision, "accept")
        self.assertEqual(approval.decision_payload, payload)
        self.assertIsNotNone(approval.decided_at)

    def test_rejects_unoffered_structured_decision(self) -> None:
        payload = {
            "acceptWithExecpolicyAmendment": {
                "execpolicy_amendment": ["just", "test"]
            }
        }
        approval = self._make_approval(
            params={
                "item": {"command": "just test"},
                "availableDecisions": ["accept", "cancel"],
            }
        )

        response = self.client.post(
            reverse("resolve_approval", kwargs={"approval_id": approval.pk}),
            data={
                "decision": "accept",
                "decision_payload": json.dumps(payload),
            },
        )

        self.assertEqual(response.status_code, 400)
        approval.refresh_from_db()
        self.assertEqual(approval.decision, "")
        self.assertIsNone(approval.decision_payload)

    def test_rejects_invalid_or_stale_requests(self) -> None:
        """A POST with a value outside the app-server-accepted set must 400
        rather than poison the row — the worker would otherwise round-trip
        the bogus string into a JSON-RPC response codex rejects. Already
        resolved rows must stay locked so two tabs cannot clobber a choice."""
        cases: list[
            tuple[str, ApprovalRequest | None, str, dict[str, str], int, str | None]
        ] = [
            (
                "invalid decision",
                self._make_approval(),
                "post",
                {"decision": "yes please"},
                400,
                "",
            ),
            ("missing row", None, "post", {"decision": "accept"}, 404, None),
            (
                "already resolved",
                self._make_approval(decision="accept"),
                "post",
                {"decision": "decline"},
                409,
                "accept",
            ),
            ("method", self._make_approval(), "get", {}, 405, ""),
        ]
        for label, approval, method, data, status, expected_decision in cases:
            with self.subTest(label=label):
                approval_id = approval.pk if approval is not None else 99999999
                url = reverse("resolve_approval", kwargs={"approval_id": approval_id})
                if method == "post":
                    response = self.client.post(url, data=data)
                else:
                    response = self.client.get(url)

                self.assertEqual(response.status_code, status)
                if approval is not None:
                    approval.refresh_from_db()
                    self.assertEqual(approval.decision, expected_decision)

class ResolveInputRequestViewTests(TestCase):
    """The ``POST /input/<id>/`` endpoint records structured answers for
    app-server ``request_user_input`` prompts.
    """

    def _make_input_request(
        self, *, response: dict[str, object] | None = None
    ) -> UserInputRequest:
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="thread-1",
            cwd="/repo",
            prompt="hi",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
        )
        return UserInputRequest.objects.create(
            instance=instance,
            method="request_user_input",
            params={"questions": [{"id": "scope"}]},
            response=response,
        )

    def test_records_answer_payloads(self) -> None:
        structured_answers = {
            "scope": ["UI", "CLI"],
            "details": {"choice": "Other", "notes": ["keep history"]},
            "confirmed": True,
            "priority": 2,
            "optional": None,
        }
        cases = [
            (
                "string answer",
                {"answers": json.dumps({"scope": "Management command"})},
                {"answers": {"scope": "Management command"}},
            ),
            ("omitted payload", {}, {"answers": {}}),
            (
                "trimmed strings",
                {"answers": json.dumps({" scope ": " UI ", " ": "ignored"})},
                {"answers": {"scope": "UI"}},
            ),
            (
                "structured values",
                {"answers": json.dumps(structured_answers)},
                {"answers": structured_answers},
            ),
        ]
        for label, data, expected_response in cases:
            with self.subTest(label=label):
                input_request = self._make_input_request()

                response = self.client.post(
                    reverse(
                        "resolve_input_request", kwargs={"input_id": input_request.pk}
                    ),
                    data=data,
                )

                self.assertEqual(response.status_code, 200)
                input_request.refresh_from_db()
                self.assertEqual(input_request.response, expected_response)
                self.assertIsNotNone(input_request.responded_at)

    def test_rejects_invalid_answers_payload(self) -> None:
        input_request = self._make_input_request()

        for answers in ("not-json", json.dumps(["not", "object"])):
            with self.subTest(answers=answers):
                response = self.client.post(
                    reverse(
                        "resolve_input_request", kwargs={"input_id": input_request.pk}
                    ),
                    data={"answers": answers},
                )
                self.assertEqual(response.status_code, 400)

    def test_returns_409_when_already_resolved(self) -> None:
        input_request = self._make_input_request(response={"answers": {"scope": "UI"}})

        response = self.client.post(
            reverse("resolve_input_request", kwargs={"input_id": input_request.pk}),
            data={"answers": json.dumps({"scope": "CLI"})},
        )

        self.assertEqual(response.status_code, 409)
        input_request.refresh_from_db()
        self.assertEqual(input_request.response, {"answers": {"scope": "UI"}})

    def test_returns_404_when_input_request_is_missing(self) -> None:
        response = self.client.post(
            reverse("resolve_input_request", kwargs={"input_id": 999_999}),
            data={"answers": json.dumps({"scope": "UI"})},
        )

        self.assertEqual(response.status_code, 404)

    def test_returns_409_when_update_loses_race(self) -> None:
        input_request = self._make_input_request()
        original_filter = UserInputRequest.objects.filter

        class _RacingUpdate:
            def update(self, **kwargs: Any) -> int:
                original_filter(pk=input_request.pk).update(
                    response={"answers": {"scope": "already answered"}}
                )
                return 0

        def _filter(*args: Any, **kwargs: Any) -> Any:
            if kwargs == {"pk": input_request.pk, "response__isnull": True}:
                return _RacingUpdate()
            return original_filter(*args, **kwargs)

        with patch.object(UserInputRequest.objects, "filter", side_effect=_filter):
            response = self.client.post(
                reverse("resolve_input_request", kwargs={"input_id": input_request.pk}),
                data={"answers": json.dumps({"scope": "UI"})},
            )

        self.assertEqual(response.status_code, 409)
        input_request.refresh_from_db()
        self.assertEqual(
            input_request.response,
            {"answers": {"scope": "already answered"}},
        )

    def test_string_representation_reflects_response_state(self) -> None:
        pending = self._make_input_request()
        answered = self._make_input_request(response={"answers": {"scope": "UI"}})

        self.assertIn("state=pending", str(pending))
        self.assertIn("state=answered", str(answered))
