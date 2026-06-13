"""Session detail page, plan state, transcript trimming, and SSE stream tests."""


import json
import os
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from django.test import (
    TestCase,
)
from django.urls import reverse

from hitch.main import demo, views
from hitch.main.models import (
    CodexInstance,
    SessionDemo,
    SessionMetadata,
    SystemAgentRun,
    SystemWorkflow,
)
from hitch.main.runtime import rollout as rollout_module
from hitch.main.runtime import streaming
from hitch.main.sessions import (
    entry_render,
    session_entry_display,
    session_pr_plan,
    session_resume,
)
from hitch.main.test.support import (
    _rollout_line,
    _seed_cookies,
    _setup_codex,
)
from hitch.main.test.views_helpers import (
    _basic_session_rollout_lines,
    _due_pr_monitor_state,
    _make_rollout,
    _merged_pr_monitor_observation,
    _session,
    _token_count_line,
    _write_codex_home_rollout,
)
from hitch.main.workflows import system_agents


class SessionDetailFastPathTests(TestCase):
    def test_pr_closed_workflow_surfaces_auto_pull_result(self) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="pr-closed-auto-pull",
            cwd="/repo",
            status=SystemWorkflow.STATUS_COMPLETED,
            step=system_agents.STEP_PR_CLOSED,
            state={
                "next_user_message_index": 1,
                system_agents.AUTO_PULL_RESULT_STATE_KEY: {
                    "status": "failed",
                    "error": "project repository has uncommitted changes",
                },
            },
        )
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="pr-closed-auto-pull",
            cwd="/repo",
            prompt="Run QA",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            display_author=system_agents.QA_DISPLAY_AUTHOR,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            thread_id="pr-closed-auto-pull",
            instance=instance,
            status=SystemAgentRun.STATUS_COMPLETED,
            output={"lgtm": True},
        )

        entries = session_entry_display._apply_qa_approval_messages(
            [{"kind": "user", "text": "Open a PR"}],
            "pr-closed-auto-pull",
        )

        self.assertEqual(entries[0]["display_author"], system_agents.QA_DISPLAY_AUTHOR)
        self.assertEqual(
            entries[-1]["display_author"], system_agents.PR_WORKFLOW_DISPLAY_AUTHOR
        )
        self.assertIn(
            "Auto-pull failed: project repository has uncommitted changes",
            entries[-1]["text"],
        )

    def test_pr_workflow_auto_pull_result_surfaces_without_qa_approval(
        self,
    ) -> None:
        SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="pr-monitor-auto-pull",
            cwd="/repo",
            status=SystemWorkflow.STATUS_COMPLETED,
            step=system_agents.STEP_PR_CLOSED,
            state={
                system_agents.AUTO_PULL_RESULT_STATE_KEY: {
                    "status": "pulled",
                    "branch": "main",
                    "before_sha": "a" * 40,
                    "after_sha": "b" * 40,
                    "changed": True,
                },
            },
        )

        entries = session_entry_display._apply_qa_approval_messages(
            [{"kind": "user", "text": "Fix PR"}],
            "pr-monitor-auto-pull",
        )

        self.assertEqual(
            entries[-1]["display_author"], system_agents.PR_WORKFLOW_DISPLAY_AUTHOR
        )
        self.assertIn(
            "Auto-pull: pulled origin/main into the default repo.",
            entries[-1]["text"],
        )

    @patch("hitch.main.caches._start_models_refresh_thread")
    @patch("hitch.main.views.common.Codex")
    def test_inactive_session_detail_renders_indexed_rollout_without_resume(
        self, mock_codex: MagicMock, _start_models_refresh: MagicMock
    ) -> None:
        pr_url = "https://github.com/cberner/hitch/pull/94"
        rollout_path = _make_rollout(
            self,
            [
                _rollout_line(
                    "event_msg",
                    {"type": "user_message", "message": "Read from rollout"},
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Rollout answer"}],
                        "phase": "final_answer",
                    },
                ),
                _rollout_line(
                    "event_msg",
                    {"type": "user_message", "message": system_agents.PR_SLASH_PROMPT},
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "function_call",
                        "name": "github_create_pull_request",
                        "arguments": "{}",
                        "call_id": "call-pr",
                    },
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "function_call_output",
                        "call_id": "call-pr",
                        "output": json.dumps({"url": pr_url}),
                    },
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Opened the PR."}],
                        "phase": "final_answer",
                    },
                ),
            ],
        )
        now = datetime(2025, 1, 5, tzinfo=UTC)
        SessionMetadata.objects.create(
            thread_id="indexed",
            cwd="/repo",
            codex_path=str(rollout_path),
            codex_name="Indexed session",
            codex_preview="Read from rollout",
            codex_created_at=now,
            codex_updated_at=now,
        )
        CodexInstance.objects.create(
            pid=1,
            thread_id="indexed",
            cwd="/repo",
            prompt="done",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_COMPLETED,
            model="gpt-5.4",
            reasoning_effort="high",
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "indexed"}))

        self.assertEqual(response.status_code, 200)
        # The live session page must stay out of the browser bfcache/heuristic
        # cache so a Back navigation re-renders against current state.
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")
        self.assertContains(response, "Read from rollout")
        self.assertContains(response, "Rollout answer")
        self.assertContains(response, "Indexed session")
        self.assertContains(response, "gpt-5.4")
        self.assertContains(response, "high")
        self.assertContains(response, f'href="{pr_url}"')
        self.assertContains(
            response,
            '<button type="button" role="menuitem" data-slash-fix-pr>',
            count=1,
        )
        self.assertContains(
            response, '<span class="stage-badge" data-tone="active">PR</span>'
        )
        self.assertContains(response, f'data-ts="{now.timestamp()}"')
        self.assertNotContains(response, "Jan. 5, 2025")
        mock_codex.assert_not_called()

    @patch("hitch.main.worktrees.discover_managed_worktrees")
    @patch("hitch.main.caches._start_models_refresh_thread")
    @patch("hitch.main.views.common.Codex")
    def test_session_detail_next_message_config_uses_managed_worktree_sandbox(
        self,
        mock_codex: MagicMock,
        _start_models_refresh: MagicMock,
        mock_managed_worktrees: MagicMock,
    ) -> None:
        worktree = "/repo-worktree"
        rollout_path = _make_rollout(
            self,
            _basic_session_rollout_lines("Edit the app", "Done."),
        )
        now = datetime(2025, 1, 5, tzinfo=UTC)
        SessionMetadata.objects.create(
            thread_id="managed",
            cwd=worktree,
            codex_path=str(rollout_path),
            codex_name="Managed session",
            codex_preview="Edit the app",
            codex_created_at=now,
            codex_updated_at=now,
        )
        CodexInstance.objects.create(
            pid=1,
            thread_id="managed",
            cwd=worktree,
            prompt="done",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_COMPLETED,
        )
        mock_managed_worktrees.return_value = [Path(worktree)]

        response = self.client.get(reverse("session", kwargs={"session_id": "managed"}))

        self.assertEqual(response.status_code, 200)
        config = {
            item["label"]: item["value"]
            for item in response.context["next_message_config"]
        }
        self.assertEqual(config["sandbox"], "Workspace write")
        mock_codex.assert_not_called()

    @patch("hitch.main.caches._start_models_refresh_thread")
    @patch("hitch.main.views.common.Codex")
    def test_session_detail_uses_session_approval_mode_override(
        self, mock_codex: MagicMock, _start_models_refresh: MagicMock
    ) -> None:
        rollout_path = _make_rollout(
            self,
            _basic_session_rollout_lines("Edit the app", "Done."),
        )
        now = datetime(2025, 1, 5, tzinfo=UTC)
        SessionMetadata.objects.create(
            thread_id="approval-session",
            cwd="/repo",
            codex_path=str(rollout_path),
            codex_name="Approval session",
            codex_preview="Edit the app",
            codex_created_at=now,
            codex_updated_at=now,
            approval_mode="prompt_user",
        )
        CodexInstance.objects.create(
            pid=1,
            thread_id="approval-session",
            cwd="/repo",
            prompt="done",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_COMPLETED,
        )
        _seed_cookies(self.client, hitch_approval_mode="deny_all")

        response = self.client.get(
            reverse("session", kwargs={"session_id": "approval-session"})
        )

        self.assertEqual(response.status_code, 200)
        config = {
            item["label"]: item["value"]
            for item in response.context["next_message_config"]
        }
        self.assertEqual(config["approval"], "Always prompt for approval")
        self.assertContains(response, "data-approval-mode-open")
        self.assertContains(
            response,
            f'action="{reverse("set_session_approval_mode", kwargs={"session_id": "approval-session"})}"',
        )
        self.assertContains(response, 'value="prompt_user" selected')
        self.assertContains(response, "Follow global (Deny all escalations)")
        mock_codex.assert_not_called()

    @patch("hitch.main.caches._start_models_refresh_thread")
    @patch("hitch.main.views.common.Codex")
    def test_inactive_session_detail_uses_archived_rollout_for_stale_path(
        self, mock_codex: MagicMock, _start_models_refresh: MagicMock
    ) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        codex_home = Path(temp_dir.name) / ".codex"
        active_path = (
            codex_home
            / "sessions"
            / "2026"
            / "06"
            / "01"
            / "rollout-2026-06-01T12-00-00-stale.jsonl"
        )
        archived_path = codex_home / "archived_sessions" / active_path.name
        archived_path.parent.mkdir(parents=True)
        archived_path.write_text(
            "\n".join(
                [
                    _rollout_line(
                        "event_msg",
                        {"type": "user_message", "message": "Read archived rollout"},
                    ),
                    _rollout_line(
                        "response_item",
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "Archived rollout answer",
                                }
                            ],
                            "phase": "final_answer",
                        },
                    ),
                ]
            ),
            encoding="utf-8",
        )
        now = datetime(2025, 1, 5, tzinfo=UTC)
        SessionMetadata.objects.create(
            thread_id="stale-archived-path",
            cwd="/repo",
            codex_path=str(active_path),
            codex_name="Archived fast path",
            codex_preview="Read archived rollout",
            codex_created_at=now,
            codex_updated_at=now,
        )

        response = self.client.get(
            reverse("session", kwargs={"session_id": "stale-archived-path"})
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Read archived rollout")
        self.assertContains(response, "Archived rollout answer")
        mock_codex.assert_not_called()

    @patch("hitch.main.caches._start_models_refresh_thread")
    @patch("hitch.main.views.common.Codex")
    def test_inactive_session_detail_recovers_missing_rollout_path(
        self, mock_codex: MagicMock, _start_models_refresh: MagicMock
    ) -> None:
        with tempfile.TemporaryDirectory() as codex_home:
            rollout_path = _write_codex_home_rollout(
                codex_home,
                "recovered-thread",
                _basic_session_rollout_lines(
                    "Recovered from disk", "Recovered answer"
                ),
            )
            now = datetime(2025, 1, 5, tzinfo=UTC)
            metadata = SessionMetadata.objects.create(
                thread_id="recovered-thread",
                cwd="/repo",
                codex_path="",
                codex_name="Recovered session",
                codex_preview="Recovered from disk",
                codex_created_at=now,
                codex_updated_at=now,
            )

            with patch.dict(os.environ, {"CODEX_HOME": codex_home}):
                response = self.client.get(
                    reverse("session", kwargs={"session_id": "recovered-thread"})
                )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Recovered from disk")
        self.assertContains(response, "Recovered answer")
        metadata.refresh_from_db()
        self.assertEqual(metadata.codex_path, str(rollout_path))
        mock_codex.assert_not_called()

    def test_session_detail_rollout_recovery_ignores_hyphenated_suffix_match(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as codex_home:
            _write_codex_home_rollout(
                codex_home,
                "xyz-abc-def",
                ["not the requested thread"],
            )
            metadata = SessionMetadata.objects.create(
                thread_id="abc-def",
                cwd="/repo",
                codex_path="",
                codex_created_at=datetime(2025, 1, 5, tzinfo=UTC),
                codex_updated_at=datetime(2025, 1, 5, tzinfo=UTC),
            )

            with patch.dict(os.environ, {"CODEX_HOME": codex_home}):
                recovered = session_resume._session_detail_metadata("abc-def")

        self.assertEqual(recovered, metadata)
        metadata.refresh_from_db()
        self.assertEqual(metadata.codex_path, "")

    @patch("hitch.main.caches._start_models_refresh_thread")
    @patch("hitch.main.views.common.Codex")
    def test_inactive_session_detail_keeps_archived_flag_for_recovered_active_path(
        self, mock_codex: MagicMock, _start_models_refresh: MagicMock
    ) -> None:
        with tempfile.TemporaryDirectory() as codex_home:
            rollout_path = _write_codex_home_rollout(
                codex_home,
                "archived-thread",
                _basic_session_rollout_lines(
                    "Recovered archived thread", "Archived answer"
                ),
            )
            now = datetime(2025, 1, 5, tzinfo=UTC)
            metadata = SessionMetadata.objects.create(
                thread_id="archived-thread",
                cwd="/repo",
                codex_path="",
                codex_name="Archived recovered session",
                codex_preview="Recovered archived thread",
                codex_created_at=now,
                codex_updated_at=now,
                codex_archived=True,
            )

            with patch.dict(os.environ, {"CODEX_HOME": codex_home}):
                response = self.client.get(
                    reverse("session", kwargs={"session_id": "archived-thread"})
                )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Recovered archived thread")
        self.assertContains(response, "Archived answer")
        metadata.refresh_from_db()
        self.assertEqual(metadata.codex_path, str(rollout_path))
        self.assertTrue(metadata.codex_archived)
        mock_codex.assert_not_called()

    @patch("hitch.main.caches._start_models_refresh_thread")
    @patch("hitch.main.views.common.Codex")
    def test_inactive_session_detail_reuses_loaded_rollout_data(
        self, mock_codex: MagicMock, _start_models_refresh: MagicMock
    ) -> None:
        rollout_path = _make_rollout(
            self,
            [
                _rollout_line(
                    "event_msg",
                    {"type": "user_message", "message": "Read once"},
                ),
                _token_count_line(
                    input_tokens=100,
                    cached_input_tokens=20,
                    output_tokens=30,
                    total_tokens=130,
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "One pass"}],
                        "phase": "final_answer",
                    },
                ),
            ],
        )
        now = datetime(2025, 1, 5, tzinfo=UTC)
        SessionMetadata.objects.create(
            thread_id="one-pass-rollout",
            cwd="/repo",
            codex_path=str(rollout_path),
            codex_name="One pass rollout",
            codex_preview="Read once",
            codex_created_at=now,
            codex_updated_at=now,
        )

        with patch(
            "hitch.main.runtime.rollout._load_rollout_lines",
            wraps=rollout_module._load_rollout_lines,
        ) as load_rollout_lines:
            response = self.client.get(
                reverse("session", kwargs={"session_id": "one-pass-rollout"})
            )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "One pass")
        self.assertContains(
            response,
            '<span class="usage-label">in</span><span class="usage-value">80</span>',
        )
        self.assertContains(
            response,
            '<span class="usage-label">out</span><span class="usage-value">30</span>',
        )
        self.assertNotContains(response, "Fix PR")
        self.assertEqual(load_rollout_lines.call_count, 1)
        mock_codex.assert_not_called()

    @patch("hitch.main.caches._start_models_refresh_thread")
    @patch("hitch.main.views.common.Codex")
    def test_inactive_session_detail_hides_fix_pr_after_pr_epoch_clears(
        self, mock_codex: MagicMock, _start_models_refresh: MagicMock
    ) -> None:
        pr_url = "https://github.com/cberner/hitch/pull/94"
        rollout_path = _make_rollout(
            self,
            [
                _rollout_line(
                    "event_msg",
                    {"type": "user_message", "message": system_agents.PR_SLASH_PROMPT},
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "function_call",
                        "name": "github_create_pull_request",
                        "arguments": "{}",
                        "call_id": "call-pr",
                    },
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "function_call_output",
                        "call_id": "call-pr",
                        "output": json.dumps({"url": pr_url}),
                    },
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Opened the PR."}],
                        "phase": "final_answer",
                    },
                ),
                _rollout_line(
                    "event_msg",
                    {"type": "user_message", "message": "Make another change"},
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Implemented."}],
                        "phase": "final_answer",
                    },
                ),
            ],
        )
        now = datetime(2025, 1, 5, tzinfo=UTC)
        SessionMetadata.objects.create(
            thread_id="cleared-pr-epoch",
            cwd="/repo",
            codex_path=str(rollout_path),
            codex_name="Cleared PR epoch",
            codex_preview="Make another change",
            codex_created_at=now,
            codex_updated_at=now,
        )

        response = self.client.get(
            reverse("session", kwargs={"session_id": "cleared-pr-epoch"})
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Implemented.")
        self.assertNotContains(response, "Fix PR")
        mock_codex.assert_not_called()

    @patch("hitch.main.caches._start_models_refresh_thread")
    @patch("hitch.main.views.common.Codex")
    def test_inactive_session_detail_falls_back_when_rollout_fast_path_raises(
        self, mock_codex: MagicMock, _start_models_refresh: MagicMock
    ) -> None:
        rollout_path = _make_rollout(
            self,
            [
                _rollout_line(
                    "event_msg",
                    {"type": "user_message", "message": "Schema drift"},
                ),
            ],
        )
        now = datetime(2025, 1, 5, tzinfo=UTC)
        SessionMetadata.objects.create(
            thread_id="schema-drift",
            cwd="/repo",
            codex_path=str(rollout_path),
            codex_name="Schema drift",
            codex_preview="Fallback",
            codex_created_at=now,
            codex_updated_at=now,
        )
        resumed_thread = _session("schema-drift", path=str(rollout_path))
        codex = mock_codex.return_value.__enter__.return_value
        codex._client.thread_resume.return_value = SimpleNamespace(thread=resumed_thread)
        sdk_entries = [
            {"kind": "user", "text": "SDK user"},
            {"kind": "agent", "text": "SDK answer"},
        ]

        with (
            patch(
                "hitch.main.runtime.rollout.session_detail_data",
                side_effect=ValueError("unexpected rollout shape"),
            ),
            patch("hitch.main.views.common._models_for_plan_mode_fallback", return_value=[]),
            patch("hitch.main.views.common._entries_for", return_value=sdk_entries),
        ):
            response = self.client.get(
                reverse("session", kwargs={"session_id": "schema-drift"})
            )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "SDK answer")
        codex._client.thread_resume.assert_called_once_with("schema-drift")

    @patch("hitch.main.caches._start_models_refresh_thread")
    @patch("hitch.main.views.common.Codex")
    def test_inactive_session_detail_lazy_loads_intermediate_body(
        self, mock_codex: MagicMock, _start_models_refresh: MagicMock
    ) -> None:
        rollout_path = _make_rollout(
            self,
            [
                _rollout_line(
                    "event_msg",
                    {"type": "user_message", "message": "Run a command"},
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "function_call",
                        "name": "exec_command",
                        "arguments": json.dumps({"cmd": "printf lazy-loaded-command"}),
                        "call_id": "call-lazy",
                    },
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Done."}],
                        "phase": "final_answer",
                    },
                ),
            ],
        )
        now = datetime(2025, 1, 5, tzinfo=UTC)
        SessionMetadata.objects.create(
            thread_id="lazy-intermediate",
            cwd="/repo",
            codex_path=str(rollout_path),
            codex_name="Lazy intermediate",
            codex_preview="Run a command",
            codex_created_at=now,
            codex_updated_at=now,
        )

        with patch(
            "hitch.main.runtime.rollout._load_rollout_lines",
            wraps=rollout_module._load_rollout_lines,
        ) as load_rollout_lines:
            response = self.client.get(
                reverse("session", kwargs={"session_id": "lazy-intermediate"})
            )
            fragment = self.client.get(
                reverse(
                    "session_intermediate",
                    kwargs={"session_id": "lazy-intermediate", "entry_index": 1},
                )
            )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "1 tool call")
        self.assertContains(response, "data-lazy-intermediate")
        self.assertContains(
            response,
            reverse(
                "session_intermediate",
                kwargs={"session_id": "lazy-intermediate", "entry_index": 1},
            ),
        )
        self.assertNotContains(response, "printf lazy-loaded-command")
        self.assertEqual(fragment.status_code, 200)
        self.assertContains(fragment, "printf lazy-loaded-command")
        self.assertEqual(load_rollout_lines.call_count, 1)
        mock_codex.assert_not_called()

    @patch("hitch.main.caches._start_models_refresh_thread")
    @patch("hitch.main.views.common.Codex")
    def test_session_intermediate_derives_demo_visibility_from_signed_context(
        self, mock_codex: MagicMock, _start_models_refresh: MagicMock
    ) -> None:
        demo_prompt = "Start an interactive web demo"
        rollout_path = _make_rollout(
            self,
            [
                _rollout_line(
                    "event_msg",
                    {"type": "user_message", "message": demo_prompt},
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "function_call",
                        "name": "exec_command",
                        "arguments": json.dumps({"cmd": "printf demo-only-command"}),
                        "call_id": "call-demo",
                    },
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Demo ready."}],
                        "phase": "final_answer",
                    },
                ),
            ],
        )
        now = datetime(2025, 1, 5, tzinfo=UTC)
        SessionMetadata.objects.create(
            thread_id="demo-fragment",
            cwd="/repo",
            codex_path=str(rollout_path),
            codex_name="Demo fragment",
            codex_preview=demo_prompt,
            codex_created_at=now,
            codex_updated_at=now,
        )
        workflow = SystemWorkflow.objects.create(
            kind=demo.DEMO_WORKFLOW_KIND,
            main_thread_id="demo-fragment",
            cwd="/repo",
            status=SystemWorkflow.STATUS_COMPLETED,
        )
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="demo-fragment",
            cwd="/repo",
            prompt=demo_prompt,
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=demo.DEMO_AGENT_KIND,
            display_author=demo.DEMO_DISPLAY_AUTHOR,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=demo.DEMO_AGENT_KIND,
            thread_id="demo-fragment",
            instance=instance,
            status=SystemAgentRun.STATUS_COMPLETED,
        )
        fragment_url = reverse(
            "session_intermediate",
            kwargs={"session_id": "demo-fragment", "entry_index": 1},
        )

        insecure_fragment = self.client.get(f"{fragment_url}?hide_demo=0")
        system_response = self.client.get(
            reverse("system_session", kwargs={"session_id": "demo-fragment"}),
            {"run_id": run.pk},
        )
        demo_context = views._session_intermediate_demo_context(
            "demo-fragment", run.pk
        )
        signed_fragment = self.client.get(fragment_url, {"demo_context": demo_context})

        self.assertEqual(insecure_fragment.status_code, 404)
        self.assertEqual(system_response.status_code, 200)
        self.assertContains(system_response, "data-lazy-intermediate")
        self.assertContains(system_response, "demo_context=")
        self.assertNotContains(system_response, "hide_demo=0")
        self.assertEqual(signed_fragment.status_code, 200)
        self.assertContains(signed_fragment, "printf demo-only-command")
        mock_codex.assert_not_called()

    @patch("hitch.main.caches._start_models_refresh_thread")
    @patch("hitch.main.views.common.Codex")
    def test_inactive_session_detail_derives_done_stage_from_pr_state(
        self, mock_codex: MagicMock, _start_models_refresh: MagicMock
    ) -> None:
        pr_url = "https://github.com/cberner/hitch/pull/94"
        rollout_path = _make_rollout(
            self,
            [
                _rollout_line(
                    "event_msg",
                    {"type": "user_message", "message": system_agents.PR_SLASH_PROMPT},
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "function_call",
                        "name": "github_create_pull_request",
                        "arguments": "{}",
                        "call_id": "call-pr",
                    },
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "function_call_output",
                        "call_id": "call-pr",
                        "output": json.dumps({"url": pr_url, "state": "open"}),
                    },
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "function_call",
                        "name": "github_fetch_pr",
                        "arguments": json.dumps(
                            {"repo_full_name": "cberner/hitch", "pr_number": 94}
                        ),
                        "call_id": "call-fetch",
                    },
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "function_call_output",
                        "call_id": "call-fetch",
                        "output": json.dumps(
                            {"url": pr_url, "state": "closed", "merged": True}
                        ),
                    },
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Merged."}],
                        "phase": "final_answer",
                    },
                ),
            ],
        )
        now = datetime(2025, 1, 5, tzinfo=UTC)
        metadata = SessionMetadata.objects.create(
            thread_id="merged-pr",
            cwd="/repo",
            codex_path=str(rollout_path),
            codex_name="Merged PR",
            codex_preview="Open a PR",
            codex_created_at=now,
            codex_updated_at=now,
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "merged-pr"}))

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response, '<span class="stage-badge" data-tone="done">Done: Merged</span>'
        )
        metadata.refresh_from_db()
        self.assertEqual(metadata.derived_stage, "done_merged")
        mock_codex.assert_not_called()

    @patch("hitch.main.caches._start_models_refresh_thread")
    @patch("hitch.main.views.common.Codex")
    def test_inactive_session_detail_uses_pr_workflow_failure_observation_for_pr_link(
        self, mock_codex: MagicMock, _start_models_refresh: MagicMock
    ) -> None:
        pr_url = "https://github.com/cberner-ai/raptorq-ai/pull/44"
        rollout_path = _make_rollout(
            self,
            [
                _rollout_line(
                    "event_msg",
                    {"type": "user_message", "message": system_agents.PR_SLASH_PROMPT},
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Committed."}],
                        "phase": "final_answer",
                    },
                ),
                _rollout_line(
                    "event_msg",
                    {
                        "type": "user_message",
                        "message": (
                            "Hitch QA agent could not complete the PR workflow.\n\n"
                            "Status: Hitch checked the PR gates and is waiting on "
                            "external PR state.\n\n"
                            "Tell the user the PR workflow needs attention before "
                            "continuing."
                        ),
                    },
                ),
                _rollout_line(
                    "event_msg",
                    {
                        "type": "mcp_tool_call_end",
                        "invocation": {
                            "server": "codex_apps",
                            "tool": "github_get_pr_info",
                            "arguments": {
                                "repo_full_name": "cberner-ai/raptorq-ai",
                                "pr_number": 44,
                            },
                        },
                        "result": {
                            "Ok": {
                                "structuredContent": {
                                    "url": pr_url,
                                    "number": 44,
                                    "state": "open",
                                    "merged": False,
                                }
                            }
                        },
                    },
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "PR workflow needs attention.",
                            }
                        ],
                        "phase": "final_answer",
                    },
                ),
            ],
        )
        now = datetime(2025, 1, 5, tzinfo=UTC)
        SessionMetadata.objects.create(
            thread_id="pr-workflow-failure-observed-pr",
            cwd="/repo",
            codex_path=str(rollout_path),
            codex_name="Observed PR",
            codex_preview="Open a PR",
            codex_created_at=now,
            codex_updated_at=now,
        )

        response = self.client.get(
            reverse(
                "session", kwargs={"session_id": "pr-workflow-failure-observed-pr"}
            )
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, f'href="{pr_url}"')
        self.assertContains(
            response, '<span class="stage-badge" data-tone="active">PR</span>'
        )
        mock_codex.assert_not_called()

    @patch("hitch.main.workflows.pr_qa._gh_pr_view")
    @patch("hitch.main.caches._start_models_refresh_thread")
    @patch("hitch.main.views.common.Codex")
    def test_inactive_session_detail_refreshes_ready_pr_to_done_merged(
        self,
        mock_codex: MagicMock,
        _start_models_refresh: MagicMock,
        mock_gh_pr_view: MagicMock,
    ) -> None:
        pr_url = "https://github.com/cberner/hitch/pull/344"
        rollout_path = _make_rollout(
            self,
            [
                _rollout_line(
                    "event_msg",
                    {"type": "user_message", "message": "Fix database locks"},
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Fixed."}],
                        "phase": "final_answer",
                    },
                ),
            ],
        )
        now = datetime(2025, 1, 5, tzinfo=UTC)
        SessionMetadata.objects.create(
            thread_id="ready-pr-merged-detail",
            cwd=str(rollout_path.parent),
            codex_path=str(rollout_path),
            codex_name="Ready PR merged detail",
            codex_preview="Fix database locks",
            codex_created_at=now,
            codex_updated_at=now,
        )
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="ready-pr-merged-detail",
            cwd=str(rollout_path.parent),
            status=SystemWorkflow.STATUS_COMPLETED,
            step=system_agents.STEP_PR_READY,
            state={
                "pr_handoff": {
                    "url": pr_url,
                    "repository_full_name": "cberner/hitch",
                    "pr_number": 344,
                    "state": "open",
                },
            },
        )
        mock_gh_pr_view.return_value = {
            "url": pr_url,
            "repository_full_name": "cberner/hitch",
            "pr_number": 344,
            "state": "closed",
            "merged": True,
            "merged_at": "2026-06-02T08:26:51Z",
        }

        # First load serves the last-known (open) PR stage with the refreshing
        # highlight and runs the gh refresh off-request (synchronous under
        # TESTING), which persists the terminal stage onto the workflow.
        response = self.client.get(
            reverse("session", kwargs={"session_id": "ready-pr-merged-detail"})
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            '<span class="stage-badge" data-tone="active" data-refreshing="true">PR</span>',
        )
        workflow.refresh_from_db()
        self.assertEqual(workflow.step, system_agents.STEP_PR_CLOSED)
        self.assertTrue(workflow.state["pr_handoff"]["merged"])
        mock_gh_pr_view.assert_called_once()
        mock_codex.assert_not_called()

        # The next load reflects the refreshed terminal stage without hitting gh
        # again -- the same PR is debounced.
        response = self.client.get(
            reverse("session", kwargs={"session_id": "ready-pr-merged-detail"})
        )
        self.assertContains(
            response, '<span class="stage-badge" data-tone="done">Done: Merged</span>'
        )
        mock_gh_pr_view.assert_called_once()

    @patch("hitch.main.workflows.pr_qa._pr_monitor_observation_from_gh")
    @patch("hitch.main.caches._start_models_refresh_thread")
    @patch("hitch.main.views.common.Codex")
    def test_inactive_session_detail_refreshes_due_pr_monitor_backoff_to_done_merged(
        self,
        mock_codex: MagicMock,
        _start_models_refresh: MagicMock,
        mock_observe: MagicMock,
    ) -> None:
        pr_url = "https://github.com/cberner/hitch/pull/60"
        repo = "cberner/hitch"
        pr_number = 60
        rollout_path = _make_rollout(
            self,
            _basic_session_rollout_lines("Open a PR", "Opened."),
        )
        now = datetime.now(UTC)
        SessionMetadata.objects.create(
            thread_id="monitor-pr-merged-detail",
            cwd=str(rollout_path.parent),
            codex_path=str(rollout_path),
            codex_name="Monitor PR merged detail",
            codex_preview="Open a PR",
            codex_created_at=now,
            codex_updated_at=now,
        )
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="monitor-pr-merged-detail",
            cwd=str(rollout_path.parent),
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_PR_MONITORING,
            state=_due_pr_monitor_state(
                pr_url=pr_url, repo=repo, pr_number=pr_number, now=now
            ),
        )
        mock_observe.return_value = _merged_pr_monitor_observation(
            pr_url=pr_url, repo=repo, pr_number=pr_number
        )

        response = self.client.get(
            reverse("session", kwargs={"session_id": "monitor-pr-merged-detail"})
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            '<span class="stage-badge" data-tone="active" data-refreshing="true">PR</span>',
        )
        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(workflow.step, system_agents.STEP_PR_CLOSED)
        self.assertTrue(workflow.state["pr_handoff"]["merged"])
        mock_observe.assert_called_once()

        response = self.client.get(
            reverse("session", kwargs={"session_id": "monitor-pr-merged-detail"})
        )
        self.assertContains(
            response, '<span class="stage-badge" data-tone="done">Done: Merged</span>'
        )
        mock_observe.assert_called_once()

    @patch("hitch.main.workflows.pr_qa._gh_pr_view")
    @patch("hitch.main.caches._start_models_refresh_thread")
    @patch("hitch.main.views.common.Codex")
    def test_inactive_session_detail_refreshes_cached_pr_stage_to_done_merged(
        self,
        mock_codex: MagicMock,
        _start_models_refresh: MagicMock,
        mock_gh_pr_view: MagicMock,
    ) -> None:
        pr_url = "https://github.com/cberner/hitch/pull/94"
        rollout_path = _make_rollout(
            self,
            [
                _rollout_line(
                    "event_msg",
                    {"type": "user_message", "message": system_agents.PR_SLASH_PROMPT},
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "function_call",
                        "name": "github_fetch_pr",
                        "arguments": "{}",
                        "call_id": "call-pr",
                    },
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "function_call_output",
                        "call_id": "call-pr",
                        "output": json.dumps({"url": pr_url, "state": "open"}),
                    },
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Open."}],
                        "phase": "final_answer",
                    },
                ),
            ],
        )
        now = datetime(2025, 1, 5, tzinfo=UTC)
        metadata = SessionMetadata.objects.create(
            thread_id="cached-pr-merged-detail",
            cwd=str(rollout_path.parent),
            codex_path=str(rollout_path),
            codex_name="Cached PR merged detail",
            codex_preview="Open a PR",
            codex_created_at=now,
            codex_updated_at=now,
            derived_stage="pr",
            derived_stage_source_mtime_ns=rollout_path.stat().st_mtime_ns,
        )
        mock_gh_pr_view.return_value = {
            "url": pr_url,
            "repository_full_name": "cberner/hitch",
            "pr_number": 94,
            "state": "closed",
            "merged": True,
            "merged_at": "2026-06-02T08:26:51Z",
        }

        # First load serves the cached (open) PR stage with the refreshing
        # highlight and runs the gh refresh off-request, persisting the terminal
        # stage to the mtime-keyed cache.
        response = self.client.get(
            reverse("session", kwargs={"session_id": "cached-pr-merged-detail"})
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            '<span class="stage-badge" data-tone="active" data-refreshing="true">PR</span>',
        )
        metadata.refresh_from_db()
        self.assertEqual(metadata.derived_stage, "done_merged")
        self.assertIsNotNone(metadata.derived_stage_pr_refresh_attempted_at)
        mock_gh_pr_view.assert_called_once()
        mock_codex.assert_not_called()

        # The next load surfaces the cached terminal stage without hitting gh
        # again, even though the rollout still shows the PR open.
        response = self.client.get(
            reverse("session", kwargs={"session_id": "cached-pr-merged-detail"})
        )
        self.assertContains(
            response, '<span class="stage-badge" data-tone="done">Done: Merged</span>'
        )
        mock_gh_pr_view.assert_called_once()

    @patch("hitch.main.caches._start_models_refresh_thread")
    @patch("hitch.main.views.common.Codex")
    def test_inactive_session_detail_stamps_stage_cache_with_pre_read_mtime(
        self, mock_codex: MagicMock, _start_models_refresh: MagicMock
    ) -> None:
        # The stage cache is keyed on the rollout mtime. If the detail view
        # re-stats the file *after* reading entries, a worker append that lands
        # during the read would stamp the (pre-append) stage with the post-append
        # mtime -- so a later index render would compare equal mtimes, serve the
        # stale stage, and never re-derive. The mtime must be captured before the
        # entries are read.
        rollout_path = _make_rollout(
            self,
            [
                _rollout_line(
                    "event_msg",
                    {"type": "user_message", "message": "Read from rollout"},
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Implemented."}],
                        "phase": "final_answer",
                    },
                ),
            ],
        )
        pre_read_mtime_ns = rollout_path.stat().st_mtime_ns
        now = datetime(2025, 1, 5, tzinfo=UTC)
        metadata = SessionMetadata.objects.create(
            thread_id="mtime-race",
            cwd="/repo",
            codex_path=str(rollout_path),
            codex_name="Mtime race",
            codex_preview="Read from rollout",
            codex_created_at=now,
            codex_updated_at=now,
        )

        real_session_detail_data = session_resume._session_detail_data_for_metadata_resume

        def _append_during_read(path: Path) -> rollout_module.SessionDetailData | None:
            rollout_data = real_session_detail_data(path)
            # Simulate a worker appending a new turn while the request reads the
            # rollout, advancing the file's mtime past the captured value.
            with path.open("a", encoding="utf-8") as handle:
                handle.write(
                    "\n"
                    + _rollout_line(
                        "event_msg", {"type": "user_message", "message": "Next"}
                    )
                )
            post_read_mtime_ns = pre_read_mtime_ns + 1_000_000_000
            os.utime(path, ns=(post_read_mtime_ns, post_read_mtime_ns))
            return rollout_data

        with patch(
            "hitch.main.sessions.session_resume._session_detail_data_for_metadata_resume",
            side_effect=_append_during_read,
        ):
            response = self.client.get(
                reverse("session", kwargs={"session_id": "mtime-race"})
            )

        self.assertEqual(response.status_code, 200)
        metadata.refresh_from_db()
        # The stage was derived from the pre-append entries, so it must be
        # stamped with the pre-append mtime; otherwise the post-append mtime
        # would mask the staleness and the cache would never invalidate.
        self.assertEqual(metadata.derived_stage_source_mtime_ns, pre_read_mtime_ns)
        self.assertNotEqual(
            metadata.derived_stage_source_mtime_ns, rollout_path.stat().st_mtime_ns
        )
        mock_codex.assert_not_called()

    @patch("hitch.main.caches._start_models_refresh_thread")
    @patch("hitch.main.views.common.Codex")
    def test_active_session_detail_does_not_cache_forced_stage(
        self, mock_codex: MagicMock, _start_models_refresh: MagicMock
    ) -> None:
        # Viewing a session while a worker runs shows the active Implementation
        # stage, but that stage is forced by the running worker rather than the
        # rollout. Writing it to the mtime-keyed cache would let the index serve
        # a stale active badge once the worker stops without rewriting the
        # rollout, so the detail view must not persist it.
        rollout_path = _make_rollout(
            self,
            [
                _rollout_line(
                    "event_msg",
                    {"type": "user_message", "message": system_agents.PR_SLASH_PROMPT},
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "function_call",
                        "name": "github_create_pull_request",
                        "arguments": "{}",
                        "call_id": "call-pr",
                    },
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "function_call_output",
                        "call_id": "call-pr",
                        "output": json.dumps(
                            {"url": "https://github.com/cberner/hitch/pull/94",
                             "state": "closed", "merged": False}
                        ),
                    },
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Closed."}],
                        "phase": "final_answer",
                    },
                ),
            ],
        )
        now = datetime(2025, 1, 5, tzinfo=UTC)
        metadata = SessionMetadata.objects.create(
            thread_id="active-detail",
            cwd="/repo",
            codex_path=str(rollout_path),
            codex_name="Active detail",
            codex_preview="Follow up",
            codex_created_at=now,
            codex_updated_at=now,
        )
        CodexInstance.objects.create(
            pid=os.getpid(),
            thread_id="active-detail",
            cwd="/repo",
            prompt="Follow up",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_RUNNING,
        )
        client = _setup_codex(mock_codex)
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=_session("active-detail", name="Active detail", path=str(rollout_path))
        )

        with patch("hitch.main.runtime.codex_pool.worker_is_alive", return_value=True):
            response = self.client.get(
                reverse("session", kwargs={"session_id": "active-detail"})
            )

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            '<span class="stage-badge" data-tone="active">Implementation</span>',
        )
        metadata.refresh_from_db()
        self.assertEqual(metadata.derived_stage, "")
        self.assertEqual(metadata.derived_stage_source_mtime_ns, 0)

    @patch("hitch.main.sessions.session_stage_refresh._schedule_pr_stage_refresh")
    @patch("hitch.main.caches._start_models_refresh_thread")
    @patch("hitch.main.views.common.Codex")
    def test_active_session_detail_does_not_flag_pr_workflow_refreshing(
        self,
        mock_codex: MagicMock,
        _start_models_refresh: MagicMock,
        mock_schedule: MagicMock,
    ) -> None:
        # A live worker shows its own Implementation stage even when an older
        # completed PR workflow is due for a gh refresh. Flagging that live badge
        # refreshing would let the reload script tear down the EventSource
        # transcript mid-turn, so the PR refresh must stay dormant here.
        pr_url = "https://github.com/cberner/hitch/pull/77"
        rollout_path = _make_rollout(
            self,
            [
                _rollout_line(
                    "event_msg",
                    {"type": "user_message", "message": "Keep going"},
                ),
            ],
        )
        now = datetime(2025, 1, 5, tzinfo=UTC)
        SessionMetadata.objects.create(
            thread_id="active-with-pr-workflow",
            cwd="/repo",
            codex_path=str(rollout_path),
            codex_name="Active with PR workflow",
            codex_preview="Keep going",
            codex_created_at=now,
            codex_updated_at=now,
        )
        handoff = {
            "url": pr_url,
            "repository_full_name": "cberner/hitch",
            "pr_number": 77,
            "state": "open",
        }
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="active-with-pr-workflow",
            cwd="/repo",
            status=SystemWorkflow.STATUS_COMPLETED,
            step=system_agents.STEP_PR_READY,
            state={"pr_handoff": handoff, "hitch_pr_handoff": handoff},
        )
        # Keep the completed PR workflow alive through the main-lifecycle check.
        SystemWorkflow.objects.filter(pk=workflow.pk).update(
            updated_at=now + timedelta(minutes=1)
        )
        CodexInstance.objects.create(
            pid=os.getpid(),
            thread_id="active-with-pr-workflow",
            cwd="/repo",
            prompt="Keep going",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_RUNNING,
        )
        client = _setup_codex(mock_codex)
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=_session(
                "active-with-pr-workflow",
                name="Active with PR workflow",
                path=str(rollout_path),
            )
        )

        with patch("hitch.main.runtime.codex_pool.worker_is_alive", return_value=True):
            response = self.client.get(
                reverse("session", kwargs={"session_id": "active-with-pr-workflow"})
            )

        self.assertEqual(response.status_code, 200)
        # The plain (non-refreshing) active badge -- the refreshing variant would
        # carry data-refreshing between the tone and the '>'.
        self.assertContains(
            response,
            '<span class="stage-badge" data-tone="active">Implementation</span>',
        )
        mock_schedule.assert_not_called()

    @patch("hitch.main.caches._start_models_refresh_thread")
    @patch("hitch.main.views.common.Codex")
    def test_inactive_session_detail_ignores_stale_completed_pr_workflow(
        self, mock_codex: MagicMock, _start_models_refresh: MagicMock
    ) -> None:
        pr_url = "https://github.com/cberner/hitch/pull/98"
        rollout_path = _make_rollout(
            self,
            [
                _rollout_line(
                    "event_msg",
                    {"type": "user_message", "message": system_agents.PR_SLASH_PROMPT},
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "function_call",
                        "name": "github_fetch_pr",
                        "arguments": "{}",
                        "call_id": "call-pr",
                    },
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "function_call_output",
                        "call_id": "call-pr",
                        "output": json.dumps(
                            {"url": pr_url, "state": "closed", "merged": False}
                        ),
                    },
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Closed."}],
                        "phase": "final_answer",
                    },
                ),
                _rollout_line("event_msg", {"type": "user_message", "message": "Next"}),
                _rollout_line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Implemented."}],
                        "phase": "final_answer",
                    },
                ),
            ],
        )
        now = datetime(2025, 1, 5, tzinfo=UTC)
        metadata = SessionMetadata.objects.create(
            thread_id="stale-workflow-detail",
            cwd="/repo",
            codex_path=str(rollout_path),
            codex_name="Stale workflow detail",
            codex_preview="Next",
            codex_created_at=now,
            codex_updated_at=now,
        )
        SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="stale-workflow-detail",
            cwd="/repo",
            status=SystemWorkflow.STATUS_COMPLETED,
            step=system_agents.STEP_PR_READY,
            state={
                "pr_handoff": {"url": pr_url, "state": "closed", "merged": False}
            },
        )

        response = self.client.get(
            reverse("session", kwargs={"session_id": "stale-workflow-detail"})
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            '<span class="stage-badge" data-tone="active">Implementation</span>',
        )
        self.assertNotContains(response, "Done: Closed")
        metadata.refresh_from_db()
        self.assertEqual(metadata.derived_stage, "implementation")
        mock_codex.assert_not_called()

    @patch("hitch.main.caches._start_models_refresh_thread")
    @patch("hitch.main.views.common.Codex")
    def test_inactive_session_detail_prefers_newer_main_pr_over_stale_workflow(
        self, mock_codex: MagicMock, _start_models_refresh: MagicMock
    ) -> None:
        pr_url = "https://github.com/cberner/hitch/pull/94"
        now = datetime.now(UTC)
        rollout_path = _make_rollout(
            self,
            [
                _rollout_line(
                    "event_msg",
                    {"type": "user_message", "message": system_agents.PR_SLASH_PROMPT},
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "function_call",
                        "name": "github_fetch_pr",
                        "arguments": "{}",
                        "call_id": "call-pr",
                    },
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "function_call_output",
                        "call_id": "call-pr",
                        "output": json.dumps({"url": pr_url, "state": "open"}),
                    },
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Opened."}],
                        "phase": "final_answer",
                    },
                ),
            ],
        )
        metadata = SessionMetadata.objects.create(
            thread_id="newer-main-pr-detail",
            cwd="/repo",
            codex_path=str(rollout_path),
            codex_name="Newer main PR detail",
            codex_preview="Open a PR",
            codex_created_at=now,
            codex_updated_at=now,
        )
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="newer-main-pr-detail",
            cwd="/repo",
            status=SystemWorkflow.STATUS_COMPLETED,
            step=system_agents.STEP_PR_READY,
            state={
                "pr_handoff": {
                    "url": "https://github.com/cberner/hitch/pull/93",
                    "state": "closed",
                    "merged": False,
                }
            },
        )
        SystemWorkflow.objects.filter(pk=workflow.pk).update(
            updated_at=now - timedelta(minutes=5)
        )

        response = self.client.get(
            reverse("session", kwargs={"session_id": "newer-main-pr-detail"})
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            '<span class="stage-badge" data-tone="active">PR</span>',
        )
        self.assertNotContains(response, "Done: Closed")
        metadata.refresh_from_db()
        self.assertEqual(metadata.derived_stage, "pr")
        mock_codex.assert_not_called()

    @patch("hitch.main.caches._start_models_refresh_thread")
    @patch("hitch.main.views.common.Codex")
    def test_inactive_session_detail_prefers_newer_main_pr_state_over_workflow(
        self, mock_codex: MagicMock, _start_models_refresh: MagicMock
    ) -> None:
        pr_url = "https://github.com/cberner/hitch/pull/94"
        now = datetime.now(UTC)
        rollout_path = _make_rollout(
            self,
            [
                _rollout_line(
                    "event_msg",
                    {"type": "user_message", "message": system_agents.PR_SLASH_PROMPT},
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "function_call",
                        "name": "github_fetch_pr",
                        "arguments": "{}",
                        "call_id": "call-pr",
                    },
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "function_call_output",
                        "call_id": "call-pr",
                        "output": json.dumps({"url": pr_url, "state": "open"}),
                    },
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Reopened."}],
                        "phase": "final_answer",
                    },
                ),
            ],
        )
        metadata = SessionMetadata.objects.create(
            thread_id="newer-main-pr-state-detail",
            cwd="/repo",
            codex_path=str(rollout_path),
            codex_name="Newer main PR state detail",
            codex_preview="Open a PR",
            codex_created_at=now,
            codex_updated_at=now,
        )
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="newer-main-pr-state-detail",
            cwd="/repo",
            status=SystemWorkflow.STATUS_COMPLETED,
            step=system_agents.STEP_PR_READY,
            state={
                "pr_handoff": {
                    "url": pr_url,
                    "state": "closed",
                    "merged": False,
                }
            },
        )
        SystemWorkflow.objects.filter(pk=workflow.pk).update(
            updated_at=now - timedelta(minutes=5)
        )

        response = self.client.get(
            reverse("session", kwargs={"session_id": "newer-main-pr-state-detail"})
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            '<span class="stage-badge" data-tone="active">PR</span>',
        )
        self.assertNotContains(response, "Done: Closed")
        metadata.refresh_from_db()
        self.assertEqual(metadata.derived_stage, "pr")
        mock_codex.assert_not_called()

    @patch("hitch.main.caches._start_models_refresh_thread")
    @patch("hitch.main.views.common.Codex")
    def test_inactive_session_detail_ignores_workflow_only_stale_pr_handoff(
        self, mock_codex: MagicMock, _start_models_refresh: MagicMock
    ) -> None:
        rollout_path = _make_rollout(
            self,
            [
                _rollout_line("event_msg", {"type": "user_message", "message": "Next"}),
                _rollout_line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Implemented."}],
                        "phase": "final_answer",
                    },
                ),
            ],
        )
        now = datetime(2025, 1, 5, tzinfo=UTC)
        metadata = SessionMetadata.objects.create(
            thread_id="workflow-only-stale-detail",
            cwd="/repo",
            codex_path=str(rollout_path),
            codex_name="Workflow-only stale detail",
            codex_preview="Next",
            codex_created_at=now,
            codex_updated_at=now,
        )
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="workflow-only-stale-detail",
            cwd="/repo",
            status=SystemWorkflow.STATUS_COMPLETED,
            step=system_agents.STEP_PR_READY,
            state={
                "pr_handoff": {
                    "url": "https://github.com/cberner/hitch/pull/99",
                    "state": "closed",
                    "merged": False,
                }
            },
        )
        SystemWorkflow.objects.filter(pk=workflow.pk).update(
            updated_at=now - timedelta(minutes=5)
        )

        response = self.client.get(
            reverse("session", kwargs={"session_id": "workflow-only-stale-detail"})
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            '<span class="stage-badge" data-tone="active">Implementation</span>',
        )
        self.assertNotContains(response, "Done: Closed")
        metadata.refresh_from_db()
        self.assertEqual(metadata.derived_stage, "implementation")
        mock_codex.assert_not_called()

    @patch("hitch.main.caches._start_models_refresh_thread")
    @patch("hitch.main.views.common.Codex")
    def test_inactive_session_detail_keeps_server_created_pr_handoff(
        self, mock_codex: MagicMock, _start_models_refresh: MagicMock
    ) -> None:
        pr_url = "https://github.com/cberner/hitch/pull/100"
        rollout_path = _make_rollout(
            self,
            [
                _rollout_line(
                    "event_msg",
                    {"type": "user_message", "message": system_agents.PR_SLASH_PROMPT},
                ),
                _rollout_line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Ready."}],
                        "phase": "final_answer",
                    },
                ),
            ],
        )
        now = datetime(2025, 1, 5, tzinfo=UTC)
        metadata = SessionMetadata.objects.create(
            thread_id="server-created-pr-detail",
            cwd="/repo",
            codex_path=str(rollout_path),
            codex_name="Server-created PR detail",
            codex_preview="Open a PR",
            codex_created_at=now,
            codex_updated_at=now,
        )
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="server-created-pr-detail",
            cwd="/repo",
            status=SystemWorkflow.STATUS_COMPLETED,
            step=system_agents.STEP_PR_READY,
            state={
                "pr_handoff": {
                    "url": pr_url,
                    "repository_full_name": "cberner/hitch",
                    "pr_number": 100,
                    "state": "open",
                    "source_tool": "fetch_pr",
                },
                "hitch_pr_handoff": {
                    "url": pr_url,
                    "repository_full_name": "cberner/hitch",
                    "pr_number": 100,
                },
            },
        )
        SystemWorkflow.objects.filter(pk=workflow.pk).update(
            updated_at=now + timedelta(minutes=1)
        )

        response = self.client.get(
            reverse("session", kwargs={"session_id": "server-created-pr-detail"})
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, f'href="{pr_url}"')
        self.assertContains(
            response,
            '<span class="stage-badge" data-tone="active">PR</span>',
        )
        metadata.refresh_from_db()
        self.assertEqual(metadata.derived_stage, "")
        mock_codex.assert_not_called()

    @patch("hitch.main.views.common.Codex")
    def test_session_detail_shows_button_for_plan_request_followup(
        self, mock_codex: MagicMock
    ) -> None:
        rollout_path = _make_rollout(
            self,
            [
                _rollout_line("turn_context", {"collaboration_mode": {"mode": "plan"}}),
                _rollout_line("event_msg", {"type": "user_message", "message": "Plan it"}),
                _rollout_line(
                    "event_msg",
                    {
                        "type": "agent_message",
                        "message": "No proposed plan yet.",
                        "phase": "final_answer",
                    },
                ),
                _rollout_line("turn_context", {"collaboration_mode": {"mode": "default"}}),
                _rollout_line(
                    "event_msg",
                    {
                        "type": "user_message",
                        "message": "Give me the plan and I'll approve it",
                    },
                ),
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
            ],
        )
        SessionMetadata.objects.create(
            thread_id="plan-followup",
            cwd="/repo",
            codex_path=str(rollout_path),
            codex_updated_at=datetime(2025, 1, 5, tzinfo=UTC),
        )

        response = self.client.get(
            reverse("session", kwargs={"session_id": "plan-followup"})
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Approve plan")
        self.assertContains(response, 'name="plan_action" value="approve"')
        mock_codex.assert_not_called()

    @patch("hitch.main.views.common.Codex")
    def test_session_detail_falls_back_when_indexed_rollout_has_no_transcript(
        self, mock_codex: MagicMock
    ) -> None:
        rollout_path = _make_rollout(
            self,
            [
                _token_count_line(
                    input_tokens=10,
                    cached_input_tokens=2,
                    output_tokens=3,
                    total_tokens=13,
                )
            ],
        )
        SessionMetadata.objects.create(
            thread_id="indexed-empty",
            cwd="/repo",
            codex_path=str(rollout_path),
            codex_updated_at=datetime(2025, 1, 5, tzinfo=UTC),
        )
        client = _setup_codex(mock_codex)
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=_session("indexed-empty", name="Resumed session")
        )

        response = self.client.get(
            reverse("session", kwargs={"session_id": "indexed-empty"})
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Resumed session")
        client._client.thread_resume.assert_called_once_with("indexed-empty")

    @patch("hitch.main.views.common.Codex")
    def test_session_detail_sdk_fallback_recognizes_previous_pr_prompt(
        self, mock_codex: MagicMock
    ) -> None:
        pr_url = "https://github.com/cberner/hitch/pull/172"
        missing_rollout_path = "/nonexistent/rollout.jsonl"
        SessionMetadata.objects.create(
            thread_id="previous-pr-prompt",
            cwd="/repo",
            codex_path=missing_rollout_path,
            codex_updated_at=datetime(2025, 1, 5, tzinfo=UTC),
        )
        thread = _session(
            "previous-pr-prompt",
            name="Previous PR prompt",
            path=missing_rollout_path,
        )
        thread.turns = [
            SimpleNamespace(
                started_at=datetime(2025, 1, 5, tzinfo=UTC),
                items=[
                    SimpleNamespace(
                        root=SimpleNamespace(
                            type="userMessage",
                            content=[
                                SimpleNamespace(
                                    root=SimpleNamespace(
                                        type="text",
                                        text=(
                                            "Polish it, get it ready, and open or "
                                            "update the PR."
                                        ),
                                    )
                                )
                            ],
                        )
                    ),
                    SimpleNamespace(
                        root=SimpleNamespace(
                            type="agentMessage",
                            text="Opened the PR.",
                            phase="final_answer",
                        )
                    ),
                    SimpleNamespace(
                        root=SimpleNamespace(
                            type="mcpToolCall",
                            server="github",
                            tool="create_pull_request",
                            arguments={},
                            result={"url": pr_url, "state": "open"},
                            status="completed",
                        )
                    ),
                ],
            )
        ]
        client = _setup_codex(mock_codex)
        client._client.thread_resume.return_value = SimpleNamespace(thread=thread)

        response = self.client.get(
            reverse("session", kwargs={"session_id": "previous-pr-prompt"})
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, f'href="{pr_url}"')
        self.assertContains(
            response, '<span class="stage-badge" data-tone="active">PR</span>'
        )
        client._client.thread_resume.assert_called_once_with("previous-pr-prompt")

    @patch("hitch.main.views.common.Codex")
    def test_session_detail_falls_back_when_metadata_missing(
        self, mock_codex: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=_session("missing", name="Missing metadata")
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "missing"}))

        self.assertEqual(response.status_code, 200)
        client._client.thread_resume.assert_called_once_with("missing")

    @patch("hitch.main.views.common.Codex")
    def test_session_detail_falls_back_for_active_session(
        self, mock_codex: MagicMock
    ) -> None:
        rollout_path = _make_rollout(
            self,
            [
                _rollout_line(
                    "event_msg",
                    {"type": "user_message", "message": "Indexed active"},
                )
            ],
        )
        SessionMetadata.objects.create(
            thread_id="active",
            cwd="/repo",
            codex_path=str(rollout_path),
            codex_updated_at=datetime(2025, 1, 5, tzinfo=UTC),
        )
        CodexInstance.objects.create(
            pid=os.getpid(),
            thread_id="active",
            cwd="/repo",
            prompt="still running",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_RUNNING,
        )
        client = _setup_codex(mock_codex)
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=_session("active", name="Active session")
        )

        with patch("hitch.main.runtime.codex_pool.worker_is_alive", return_value=True):
            response = self.client.get(reverse("session", kwargs={"session_id": "active"}))

        self.assertEqual(response.status_code, 200)
        client._client.thread_resume.assert_called_once_with("active")

class PendingPlanStateTests(TestCase):
    def test_approval_declined_does_not_clear_pending_plan(self) -> None:
        entries = [
            {"kind": "user", "text": "Plan it"},
            {"kind": "plan", "text": "# Plan"},
            {"kind": "user", "text": "Try command"},
            {"kind": "approval_declined", "detail": "git push"},
        ]

        self.assertTrue(session_pr_plan._entries_await_plan_approval(entries))

    def test_agent_answer_clears_pending_plan(self) -> None:
        entries = [
            {"kind": "user", "text": "Plan it"},
            {"kind": "plan", "text": "# Plan"},
            {"kind": "user", "text": "Implement the plan"},
            {"kind": "agent", "text": "Done"},
        ]

        self.assertFalse(session_pr_plan._entries_await_plan_approval(entries))

    def test_only_latest_pending_plan_is_actionable(self) -> None:
        entries = [
            {"kind": "user", "text": "Plan it"},
            {"kind": "plan", "text": "# Old Plan"},
            {"kind": "user", "text": "Revise"},
            {"kind": "plan", "text": "# Current Plan"},
        ]

        session_pr_plan._mark_pending_plan_actions(entries)

        self.assertFalse(entries[1]["show_plan_actions"])
        self.assertTrue(entries[3]["show_plan_actions"])

    def test_agent_answer_clears_plan_actions(self) -> None:
        entries = [
            {"kind": "user", "text": "Plan it"},
            {"kind": "plan", "text": "# Plan"},
            {"kind": "user", "text": "Implement the plan"},
            {"kind": "agent", "text": "Done"},
        ]

        session_pr_plan._mark_pending_plan_actions(entries)

        self.assertFalse(entries[1]["show_plan_actions"])

class ActiveTurnTrimTests(TestCase):
    def test_user_message_text_ignores_empty_text_parts_before_images(self) -> None:
        item = SimpleNamespace(
            content=[
                SimpleNamespace(root=SimpleNamespace(type="text", text="")),
                SimpleNamespace(
                    root=SimpleNamespace(
                        type="localImage",
                        path="/tmp/private/screen.png",
                    )
                ),
            ]
        )

        self.assertEqual(entry_render.user_message_text(item), "[image]")

    def test_steer_attachment_ledger_does_not_change_active_turn_marker(self) -> None:
        active = CodexInstance.objects.create(
            pid=1,
            thread_id="thread-1",
            cwd="/repo",
            prompt="initial prompt",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_RUNNING,
            input_attachment_paths=["/tmp/private/steer.png"],
        )
        entries = [
            {"kind": "user", "text": "before"},
            {"kind": "agent", "text": "before reply"},
            {"kind": "user", "text": "initial prompt"},
            {"kind": "agent", "text": "working"},
            {"kind": "user", "text": "[image]"},
            {"kind": "agent", "text": "working after steer"},
        ]

        trimmed = session_entry_display._trim_in_progress_turn(entries, active)

        self.assertEqual(
            trimmed,
            [
                {"kind": "user", "text": "before"},
                {"kind": "agent", "text": "before reply"},
            ],
        )
        self.assertEqual(
            session_entry_display._pending_user_prompt(active), "initial prompt"
        )

class SessionStreamViewTests(TestCase):
    """The SSE endpoint that mirrors a worker's events file to the browser."""

    def _make(self, **kwargs: Any) -> CodexInstance:
        defaults: dict[str, Any] = {
            "pid": 0,
            "thread_id": "thread-1",
            "cwd": "/repo",
            "prompt": "do work",
            "events_path": "/dev/null",
            "status": CodexInstance.STATUS_RUNNING,
        }
        defaults.update(kwargs)
        return CodexInstance.objects.create(**defaults)

    def _stream_url(
        self,
        session_id: str,
        *,
        baseline: str = "",
        active: str = "",
        workflow: str = "",
        demo: str = "",
    ) -> str:
        # Helper that builds the SSE URL with the page-render-time state
        # the view expects on every legitimate request. Tests that want
        # to exercise the stale-reload path pass an empty/wrong value.
        return (
            reverse("session_stream", kwargs={"session_id": session_id})
            + f"?baseline={baseline}&active={active}&workflow={workflow}&demo={demo}"
        )

    @patch("hitch.main.runtime.streaming._IDLE_MAX_STREAM_SECONDS", 0.001)
    @patch("hitch.main.runtime.streaming._IDLE_POLL_INTERVAL", 0.001)
    def test_returns_idle_heartbeat_stream_without_active_worker(self) -> None:
        # Without an active worker the SSE channel stays open emitting
        # heartbeat events with ``working: false`` so the page's connection
        # indicator can show ``connected, idle``. The cap is patched down
        # so the test doesn't sit in the recycle loop.
        response = self.client.get(self._stream_url("thread-1"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/event-stream")
        self.assertEqual(response["Cache-Control"], "no-cache")
        self.assertEqual(response["X-Accel-Buffering"], "no")
        body = b"".join(response.streaming_content)  # type: ignore[attr-defined]
        self.assertIn(b"event: heartbeat", body)
        self.assertIn(b'"working": false', body)

        # A terminal worker counts as ``no active worker`` for routing
        # purposes; the stream should stay idle without re-tailing old events.
        with tempfile.TemporaryDirectory() as raw:
            events_path = str(Path(raw) / "events.jsonl")
            Path(events_path).touch()
            inst = self._make(
                thread_id="thread-done",
                status=CodexInstance.STATUS_COMPLETED,
                events_path=events_path,
            )

            response = self.client.get(
                self._stream_url("thread-done", baseline=str(inst.pk))
            )
            body = b"".join(response.streaming_content)  # type: ignore[attr-defined]

        self.assertIn(b"event: heartbeat", body)
        self.assertIn(b'"working": false', body)

    @patch("hitch.main.runtime.streaming._IDLE_MAX_STREAM_SECONDS", 0.001)
    @patch("hitch.main.runtime.streaming._IDLE_POLL_INTERVAL", 0.001)
    def test_returns_working_heartbeat_stream_for_active_system_workflow(self) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="thread-workflow",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step="qa_running",
            state={
                "pr_gates": [
                    {
                        "key": "ci",
                        "label": "CI",
                        "status": "pending",
                        "summary": "CI is still running.",
                    }
                ]
            },
        )

        response = self.client.get(
            self._stream_url("thread-workflow", workflow=str(workflow.pk))
        )
        body = b"".join(response.streaming_content)  # type: ignore[attr-defined]

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"event: heartbeat", body)
        self.assertIn(b'"working": true', body)
        self.assertIn(b'"prWorkflowProgress"', body)
        self.assertIn(b'"label": "CI"', body)
        self.assertIn(b'"statusLabel": "Pending"', body)

    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_turn")
    @patch("hitch.main.runtime.codex_pool.worker_is_alive", return_value=False)
    def test_stream_reloads_and_blocks_when_hidden_system_worker_died(
        self, mock_worker_alive: MagicMock, mock_spawn: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="thread-dead-workflow",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_QA_RUNNING,
        )
        instance = self._make(
            pid=12345,
            thread_id="qa-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            display_author=system_agents.QA_DISPLAY_AUTHOR,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            thread_id=instance.thread_id,
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )

        response = self.client.get(
            self._stream_url("thread-dead-workflow", workflow=str(workflow.pk))
        )
        body = b"".join(response.streaming_content)  # type: ignore[attr-defined]

        self.assertIn(b'"status": "stale"', body)
        mock_worker_alive.assert_called()
        mock_spawn.assert_called_once()
        instance.refresh_from_db()
        run.refresh_from_db()
        workflow.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertEqual(run.status, SystemAgentRun.STATUS_FAILED)
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)

    @patch("hitch.main.runtime.streaming._IDLE_MAX_STREAM_SECONDS", 0.001)
    @patch("hitch.main.runtime.streaming._IDLE_POLL_INTERVAL", 0.001)
    def test_system_workflow_heartbeat_clears_empty_pr_progress(self) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="thread-workflow-empty-progress",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step="qa_running",
        )

        response = self.client.get(
            self._stream_url(
                "thread-workflow-empty-progress", workflow=str(workflow.pk)
            )
        )
        body = b"".join(response.streaming_content)  # type: ignore[attr-defined]

        self.assertIn(b'"prWorkflowProgress": []', body)

    def test_active_instance_stream_includes_pr_progress(self) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="thread-active-progress",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_PR_FEEDBACK_RUNNING,
            state={
                "pr_gates": [
                    {
                        "key": "review",
                        "label": "Review",
                        "status": "blocked",
                        "summary": "Review changes requested.",
                    }
                ]
            },
        )
        instance = self._make(
            thread_id="thread-active-progress",
            workflow_id=workflow.pk,
            purpose=CodexInstance.PURPOSE_SYSTEM_FEEDBACK,
            agent_kind=system_agents.PR_FOLLOWUP_MONITOR_AGENT_KIND,
            display_author=system_agents.PR_MONITOR_DISPLAY_AUTHOR,
            status=CodexInstance.STATUS_COMPLETED,
        )

        stream = streaming.stream_for_instance(instance)
        body = next(stream) + next(stream)

        self.assertIn(b'"prWorkflowProgress"', body)
        self.assertIn(b'"label": "Review"', body)
        self.assertIn(b'"statusLabel": "Blocked"', body)

    def test_reloads_when_page_render_state_is_stale(self) -> None:
        # The classic out-of-band-spawn race: page rendered with no
        # worker (empty baseline / active), but by the time SSE opens a
        # worker has shown up in the DB. The endpoint must reload the
        # page so the DOM gets the live-streaming UI before any item
        # events start arriving.
        self._make(thread_id="thread-1", status=CodexInstance.STATUS_RUNNING)
        response = self.client.get(self._stream_url("thread-1"))
        body = b"".join(response.streaming_content)  # type: ignore[attr-defined]
        self.assertIn(b'"status": "stale"', body)

        # Inverse race: page rendered expecting a live worker (passes
        # ``active=N`` and ``baseline=N``) but by the time SSE opens the
        # worker has gone terminal. Without the reload the page would
        # show a permanent "Codex is working…" pill and a stale pending
        # bubble for the just-completed turn.
        inst = self._make(
            thread_id="thread-completed-before-open",
            status=CodexInstance.STATUS_COMPLETED,
        )
        response = self.client.get(
            self._stream_url(
                "thread-completed-before-open",
                baseline=str(inst.pk),
                active=str(inst.pk),
            )
        )
        body = b"".join(response.streaming_content)  # type: ignore[attr-defined]
        self.assertIn(b'"status": "stale"', body)

        # Demo status changes are also render-state changes. A page that
        # rendered while a demo was still requested must reload when the
        # background notifier later marks it active.
        SessionDemo.objects.create(
            thread_id="thread-demo",
            host="127.0.0.1",
            port=45678,
            status=SessionDemo.STATUS_ACTIVE,
        )
        response = self.client.get(self._stream_url("thread-demo"))
        body = b"".join(response.streaming_content)  # type: ignore[attr-defined]
        self.assertIn(b'"status": "stale"', body)

    @patch("hitch.main.runtime.reconciliation.reconcile_dead_if_due", return_value=0)
    def test_reconciles_dead_active_worker_before_stream_routing(
        self, mock_global_reconcile: MagicMock
    ) -> None:
        instance = self._make(
            thread_id="thread-dead-before-stream",
            pid=99999999,
            status=CodexInstance.STATUS_RUNNING,
        )

        response = self.client.get(
            self._stream_url(
                "thread-dead-before-stream",
                baseline=str(instance.pk),
                active=str(instance.pk),
            )
        )
        body = b"".join(response.streaming_content)  # type: ignore[attr-defined]

        self.assertIn(b'"status": "stale"', body)
        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        mock_global_reconcile.assert_called_once()

    @patch("hitch.main.runtime.streaming._POLL_INTERVAL", 0.01)
    def test_forwards_worker_events_through_view(self) -> None:
        # End-to-end through the URL routing: a RUNNING instance with
        # events on disk gets tailed, and once the status flips before the
        # response is iterated the stream drains and closes.
        with tempfile.TemporaryDirectory() as raw:
            events_path = str(Path(raw) / "events.jsonl")
            with open(events_path, "w", encoding="utf-8") as fh:
                fh.write(
                    json.dumps({"method": "item/started", "payload": {"item": {"id": "a"}}})
                    + "\n"
                )
            instance = self._make(
                thread_id="thread-live",
                status=CodexInstance.STATUS_RUNNING,
                events_path=events_path,
            )
            response = self.client.get(
                self._stream_url(
                    "thread-live", baseline=str(instance.pk), active=str(instance.pk)
                )
            )
            # Flip the row terminal before iterating so the generator's
            # _is_done() check exits the read loop cleanly.
            instance.status = CodexInstance.STATUS_COMPLETED
            instance.save(update_fields=["status"])
            body = b"".join(response.streaming_content)  # type: ignore[attr-defined]

        self.assertIn(b"item/started", body)
        self.assertIn(b'"status": "completed"', body)

    @patch("hitch.main.runtime.streaming._POLL_INTERVAL", 0.001)
    def test_active_worker_stream_reloads_when_demo_status_changes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            events_path = str(Path(raw) / "events.jsonl")
            Path(events_path).touch()
            instance = self._make(
                thread_id="thread-demo-active",
                agent_kind=demo.DEMO_AGENT_KIND,
                events_path=events_path,
            )
            session_demo = SessionDemo.objects.create(
                thread_id="thread-demo-active",
                host="127.0.0.1",
                port=3000,
                status=SessionDemo.STATUS_REQUESTED,
            )
            demo_token = streaming.demo_stream_token("thread-demo-active")
            response = self.client.get(
                self._stream_url(
                    "thread-demo-active",
                    baseline=str(instance.pk),
                    active=str(instance.pk),
                    demo=demo_token,
                )
            )
            session_demo.status = SessionDemo.STATUS_ACTIVE
            session_demo.save(update_fields=["status", "updated_at"])

            body = b"".join(response.streaming_content)  # type: ignore[attr-defined]

        self.assertIn(b'"status": "demo"', body)

class SessionViewApprovalContextTests(TestCase):
    """The session detail view exposes POST URL templates for live
    browser prompts. Pin them so a URL refactor can't quietly break the
    streaming approval or structured-input loops."""

    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.runtime.codex_pool.worker_is_alive", return_value=True)
    def test_session_template_renders_prompt_url_templates(
        self, _mock_worker_alive: MagicMock, mock_codex: MagicMock
    ) -> None:
        ctx: MagicMock = mock_codex.return_value.__enter__.return_value
        ctx._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(
                id="thread-1",
                cwd="/repo",
                name="Demo",
                preview="",
                turns=[],
                path=None,
                updated_at=1,
            )
        )
        # The approval-url template only renders inside the
        # ``active_worker`` block (an idle session has no SSE stream and so
        # no client-side approval prompts to wire up).
        CodexInstance.objects.create(
            pid=1,
            thread_id="thread-1",
            cwd="/repo",
            prompt="hi",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
        )

        response = self.client.get(
            reverse("session", kwargs={"session_id": "thread-1"})
        )

        self.assertEqual(response.status_code, 200)
        # The placeholder pk is ``0`` — the JS swaps it for the real
        # ApprovalRequest id when posting a decision.
        self.assertContains(
            response,
            'data-approval-url-template="' + reverse(
                "resolve_approval", kwargs={"approval_id": 0}
            ),
        )
        self.assertContains(
            response,
            'data-input-url-template="' + reverse(
                "resolve_input_request", kwargs={"input_id": 0}
            ),
        )
        self.assertContains(response, "requires_explicit_choice")
        self.assertContains(response, "requiredQuestionIds")
