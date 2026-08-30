"""Session index, project views, and PR-stage refresh scheduling tests."""

import base64
import html
import json
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast, override
from unittest.mock import MagicMock, patch

from django.db import connection
from django.test import (
    TestCase,
)
from django.test.utils import CaptureQueriesContext
from django.urls import reverse
from openai_codex.errors import CodexError as CodexError
from openai_codex.errors import InvalidRequestError
from openai_codex.generated.v2_all import (
    SortDirection,
    ThreadSortKey,
    ThreadSource,
)

from hitch.main import caches
from hitch.main.models import (
    ApprovalRequest,
    ArchivedSessionTokenUsage,
    CodexInstance,
    Project,
    ProposedSession,
    SessionIndexSyncState,
    SessionMetadata,
    SessionPullRequest,
    SystemAgentRun,
    SystemWorkflow,
)
from hitch.main.sessions import (
    session_index,
    session_stage_refresh,
    settings_cookies,
    token_usage,
)
from hitch.main.sessions.pr_prompts import PR_SLASH_PROMPT
from hitch.main.test.support import (
    _cookie_value,
    _make_project,
    _rollout_line,
    _seed_cookies,
    _setup_codex,
)
from hitch.main.test.views_helpers import (
    _APPROVAL_COOKIE,
    _LAST_SELECTED_REPO_COOKIE,
    _SELECTED_PROJECT_COOKIE,
    _SHOW_ARCHIVED_COOKIE,
    _SHOW_NO_PROJECT_SESSIONS_COOKIE,
    _VISIBLE_SESSION_PROJECTS_COOKIE,
    _cache_token_usage,
    _make_rollout,
    _seed_usage_metadata,
    _session,
    _token_count_line,
)
from hitch.main.views import common as common_views
from hitch.main.workflows import gh_cli, system_agents


class IndexViewTests(TestCase):
    def _load_more_url(self, response: Any) -> str:
        match = re.search(
            r'<div class="load-more"><a[^>]+href="([^"]+)"',
            response.content.decode(),
        )
        if match is None:
            self.fail("expected a Load more link")
        return html.unescape(match.group(1))

    def _assert_index_cursor_url(self, response: Any) -> str:
        url = self._load_more_url(response)
        self.assertIn("cursor=idx%3A", url)
        return url

    @patch("hitch.main.repos.discover_repos", return_value=[])
    @patch("hitch.main.views.common.Codex")
    def test_cached_session_list_shows_registered_pr_number(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        now = datetime.now(UTC)
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=now,
            is_complete=True,
        )
        SessionMetadata.objects.create(
            thread_id="cached-pr",
            cwd="/repo",
            codex_display_title="Cached PR",
            codex_preview="Open a PR",
            codex_created_at=now,
            codex_updated_at=now,
            codex_last_synced_at=now,
        )
        SessionPullRequest.objects.create(
            thread_id="cached-pr",
            cwd="/repo",
            state={
                "pr_handoff": {
                    "url": "https://github.com/cberner/hitch/pull/94",
                    "pr_number": 94,
                    "state": "open",
                }
            },
        )

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Cached PR")
        self.assertContains(
            response,
            '<span class="stage-badge" data-tone="active">PR #94</span>',
        )
        mock_codex.assert_not_called()




    @patch("hitch.main.runtime.codex_pool.worker_is_alive", return_value=True)
    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_cached_session_list_omits_stale_pr_number_for_new_publish_turn(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        _mock_worker_alive: MagicMock,
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = CodexError("thread list unavailable")
        mock_discover.return_value = []
        now = datetime.now(UTC)
        rollout_path = _make_rollout(
            self,
            [
                _rollout_line(
                    "event_msg",
                    {"type": "user_message", "message": PR_SLASH_PROMPT},
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
                            {
                                "url": "https://github.com/cberner/hitch/pull/94",
                                "state": "open",
                            }
                        ),
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
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=now,
            is_complete=True,
        )
        SessionMetadata.objects.create(
            thread_id="new-pr-workflow",
            cwd="/repo",
            codex_display_title="New PR workflow",
            codex_preview="Open a PR",
            codex_path=str(rollout_path),
            codex_created_at=now,
            codex_updated_at=now,
            codex_last_synced_at=now,
        )
        CodexInstance.objects.create(
            pid=os.getpid(),
            thread_id="new-pr-workflow",
            cwd="/repo",
            prompt="Publish the PR",
            events_path="/tmp/new-pr-events.jsonl",
            status=CodexInstance.STATUS_RUNNING,
            agent_kind="pr_publish",
        )

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "New PR workflow")
        self.assertContains(
            response,
            '<span class="stage-badge" data-tone="active">PR</span>',
        )
        self.assertNotContains(response, "PR #94")
        mock_codex.assert_not_called()
        client.thread_list.assert_not_called()

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_cached_session_list_does_not_resurrect_superseded_pr(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = CodexError("thread list unavailable")
        mock_discover.return_value = []
        now = datetime.now(UTC)
        rollout_path = _make_rollout(
            self,
            [
                _rollout_line(
                    "event_msg",
                    {"type": "user_message", "message": "Continue implementation"},
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
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=now,
            is_complete=True,
        )
        SessionMetadata.objects.create(
            thread_id="superseded-pr",
            cwd="/repo",
            codex_display_title="Superseded PR",
            codex_preview="Continue implementation",
            codex_path=str(rollout_path),
            codex_created_at=now,
            codex_updated_at=now,
            codex_last_synced_at=now,
            derived_stage="done_closed",
            derived_stage_source_mtime_ns=rollout_path.stat().st_mtime_ns,
        )
        SessionPullRequest.objects.create(
            thread_id="superseded-pr",
            cwd="/repo",
            state={
                "pr_handoff": {
                    "url": "https://github.com/cberner/hitch/pull/100",
                    "repository_full_name": "cberner/hitch",
                    "pr_number": 100,
                    "state": "closed",
                },
                SessionPullRequest.SUPERSEDED_BY_INSTANCE_STATE_KEY: 9,
            },
        )

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            '<span class="stage-badge" data-tone="idle">Implementation</span>',
        )
        self.assertNotContains(response, "PR #100")
        self.assertNotContains(response, "Done: Closed")
        mock_codex.assert_not_called()
        client.thread_list.assert_not_called()

    @patch("hitch.main.runtime.codex_pool.worker_is_alive", return_value=True)
    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_cached_session_list_flags_pending_active_approval(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        _worker_is_alive: MagicMock,
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = CodexError("thread list unavailable")
        mock_discover.return_value = []
        now = datetime.now(UTC)
        rollout_path = _make_rollout(
            self,
            [
                _rollout_line(
                    "event_msg",
                    {"type": "user_message", "message": "Continue the work."},
                ),
            ],
        )
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=now,
            is_complete=True,
        )
        metadata = SessionMetadata.objects.create(
            thread_id="approval-needed",
            cwd="/repo",
            codex_display_title="Approval needed",
            codex_preview="Continue the work.",
            codex_path=str(rollout_path),
            codex_created_at=now,
            codex_updated_at=now,
            codex_last_synced_at=now,
            derived_stage="implementation",
            derived_stage_source_mtime_ns=rollout_path.stat().st_mtime_ns,
        )
        instance = CodexInstance.objects.create(
            pid=os.getpid(),
            thread_id="approval-needed",
            cwd="/repo",
            prompt="Continue the work.",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_RUNNING,
        )
        ApprovalRequest.objects.create(
            instance=instance,
            method="item/commandExecution/requestApproval",
            params={"item": {"command": "git rebase --autostash origin/master"}},
            decision=ApprovalRequest.DECISION_PENDING,
        )

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Approval needed")
        self.assertContains(
            response,
            '<span class="stage-badge" data-tone="warning">Awaiting Input</span>',
        )
        metadata.refresh_from_db()
        self.assertEqual(metadata.derived_stage, "implementation")
        mock_codex.assert_not_called()
        client.thread_list.assert_not_called()

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_cached_session_list_survives_malformed_stage_rollout(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = CodexError("thread list unavailable")
        mock_discover.return_value = []
        now = datetime.now(UTC)
        rollout_path = _make_rollout(
            self,
            [
                json.dumps(
                    {
                        "timestamp": "2025-01-05T12:00:00Z",
                        "type": "event_msg",
                        "payload": ["schema drift"],
                    }
                )
            ],
        )
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=now,
            is_complete=True,
        )
        SessionMetadata.objects.create(
            thread_id="malformed-rollout",
            cwd="/repo",
            codex_display_title="Malformed rollout",
            codex_preview="Implement the change",
            codex_path=str(rollout_path),
            codex_created_at=now,
            codex_updated_at=now,
            codex_last_synced_at=now,
        )

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Malformed rollout")
        self.assertContains(
            response,
            '<span class="stage-badge" data-tone="idle">Implementation</span>',
        )
        mock_codex.assert_not_called()
        client.thread_list.assert_not_called()


    @patch("hitch.main.workflows.pr_tracking.logger")
    @patch("hitch.main.workflows.pr_tracking._gh_pr_view")
    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_cached_session_list_backs_off_failed_ready_pr_refresh(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_gh_pr_view: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = CodexError("thread list unavailable")
        mock_discover.return_value = []
        now = datetime.now(UTC)
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
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=now,
            is_complete=True,
        )
        registered_pr = SessionPullRequest.objects.create(
            thread_id="ready-pr-refresh-backoff",
            cwd=str(rollout_path.parent),
            state={
                "pr_handoff": {
                    "url": pr_url,
                    "repository_full_name": "cberner/hitch",
                    "pr_number": 344,
                    "state": "open",
                },
                "hitch_pr_handoff": {
                    "url": pr_url,
                    "repository_full_name": "cberner/hitch",
                    "pr_number": 344,
                },
            },
        )
        SessionMetadata.objects.create(
            thread_id="ready-pr-refresh-backoff",
            cwd=str(rollout_path.parent),
            codex_display_title="Ready PR refresh backoff",
            codex_preview="Fix database locks",
            codex_path=str(rollout_path),
            codex_created_at=now,
            codex_updated_at=now,
            codex_last_synced_at=now,
        )
        mock_gh_pr_view.side_effect = gh_cli._GhPrOpenError("gh unavailable")

        first_response = self.client.get(reverse("index"))
        second_response = self.client.get(reverse("index"))

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(second_response.status_code, 200)
        registered_pr.refresh_from_db()
        self.assertIn("pr_stage_refresh", registered_pr.state)
        mock_gh_pr_view.assert_called_once()
        mock_logger.exception.assert_called_once()
        mock_codex.assert_not_called()
        client.thread_list.assert_not_called()

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_system_sessions_cursor_keeps_same_second_rows_stable(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = CodexError("thread list unavailable")
        mock_discover.return_value = []
        now = datetime.now(UTC)
        same_second = datetime.fromtimestamp(1000, UTC)
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=now,
            is_complete=True,
        )
        for i in range(51):
            SessionMetadata.objects.create(
                thread_id=f"system-{i:02d}",
                cwd="/repo",
                codex_display_title=f"System {i:02d}",
                codex_name=f"System {i:02d}",
                codex_created_at=same_second,
                codex_updated_at=same_second + timedelta(microseconds=50 - i),
                codex_last_synced_at=now,
                is_hidden_system_session=True,
            )

        response = self.client.get(reverse("system_sessions"))
        load_more_url = self._assert_index_cursor_url(response)

        self.assertContains(response, "System 00")
        self.assertContains(response, "System 49")
        self.assertNotContains(response, "System 50")

        response = self.client.get(load_more_url)

        self.assertContains(response, "System 50")
        self.assertNotContains(response, "System 00")
        self.assertNotContains(response, "System 49")
        client.thread_list.assert_not_called()

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_system_sessions_keeps_cold_index_second_precision_across_pages(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        mock_discover.return_value = []
        threads = [
            SimpleNamespace(
                id=f"system-{i:03d}",
                name=f"System {i:03d}",
                preview="",
                cwd="/repo",
                path=None,
                updated_at=1000 + ((119 - i) / 1_000_000),
                thread_source=ThreadSource.subagent,
            )
            for i in range(120)
        ]
        client = _setup_codex(mock_codex, threads=threads)

        response = self.client.get(reverse("system_sessions"))
        page_two_url = self._assert_index_cursor_url(response)

        self.assertContains(response, "System 119")
        self.assertContains(response, "System 070")
        self.assertNotContains(response, "System 069")

        response = self.client.get(page_two_url)
        page_three_url = self._assert_index_cursor_url(response)

        self.assertContains(response, "System 069")
        self.assertContains(response, "System 020")
        self.assertNotContains(response, "System 070")
        self.assertNotContains(response, "System 019")

        response = self.client.get(page_three_url)

        self.assertContains(response, "System 019")
        self.assertContains(response, "System 000")
        self.assertNotContains(response, "System 070")
        self.assertNotContains(response, "System 020")
        self.assertEqual(client.thread_list.call_count, 1)

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_system_sessions_ignores_invalid_index_cursor_timestamps(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = CodexError("thread list unavailable")
        mock_discover.return_value = []
        now = datetime.now(UTC)
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=now,
            is_complete=True,
        )
        SessionMetadata.objects.create(
            thread_id="system",
            cwd="/repo",
            codex_display_title="System",
            codex_name="System",
            codex_created_at=now,
            codex_updated_at=now,
            codex_last_synced_at=now,
            is_hidden_system_session=True,
        )

        cases = (
            ("NaN", ""),
            ("Infinity", ""),
            ("-Infinity", ""),
            ("1e100", ""),
            ("1e100", ',"updated_at_precision":"exact"'),
            ("-1e100", ""),
            ("-1e100", ',"updated_at_precision":"exact"'),
        )
        for updated_at, precision in cases:
            with self.subTest(updated_at=updated_at, precision=precision):
                cursor_payload = f'{{"updated_at":{updated_at},"id":"a"{precision}}}'
                cursor = "idx:" + base64.urlsafe_b64encode(cursor_payload.encode()).decode()

                response = self.client.get(reverse("system_sessions"), {"cursor": cursor})

                self.assertEqual(response.status_code, 200)
                self.assertContains(response, "System")
        client.thread_list.assert_not_called()

    @patch("hitch.main.views.common.Codex")
    def test_full_refresh_invalidates_absent_active_rows(self, mock_codex: MagicMock) -> None:
        now = datetime.now(UTC)
        SessionMetadata.objects.create(
            thread_id="stale-active",
            cwd="/repo",
            codex_display_title="Stale active",
            codex_created_at=now,
            codex_updated_at=now,
            codex_last_synced_at=now,
        )
        fresh = _session("fresh-active", name="Fresh active")
        client = _setup_codex(mock_codex, threads=[fresh])

        session_index.refresh_from_codex(
            client,
            projects=[],
            include_active=True,
            max_pages=None,
            use_state_db_only=False,
        )

        self.assertFalse(
            session_index.indexed_sessions().filter(thread_id="stale-active", codex_archived=False).exists()
        )
        self.assertTrue(session_index.indexed_sessions().filter(thread_id="fresh-active").exists())

    @patch("hitch.main.views.common.Codex")
    def test_background_session_index_refresh_uses_state_db_only(self, mock_codex: MagicMock) -> None:
        active = _session("active", name="Active session")
        archived = _session(
            "archived",
            name="Archived session",
            path="/home/user/.codex/archived_sessions/archived.jsonl",
        )
        client = _setup_codex(
            mock_codex,
            threads=[active],
            archived_threads=[archived],
        )

        common_views._refresh_usage_session_index_best_effort(
            enable_memories=False,
            include_active=True,
            include_archived=True,
        )

        client.thread_list.assert_any_call(
            limit=100,
            sort_key=ThreadSortKey.updated_at,
            sort_direction=SortDirection.desc,
            use_state_db_only=True,
        )
        client.thread_list.assert_any_call(
            limit=100,
            sort_key=ThreadSortKey.updated_at,
            sort_direction=SortDirection.desc,
            archived=True,
            use_state_db_only=True,
        )
        self.assertTrue(
            all(mock_call.kwargs["use_state_db_only"] is True for mock_call in client.thread_list.call_args_list)
        )
        active_state = SessionIndexSyncState.objects.get(source=SessionIndexSyncState.SOURCE_ACTIVE)
        archived_state = SessionIndexSyncState.objects.get(source=SessionIndexSyncState.SOURCE_ARCHIVED)
        self.assertFalse(active_state.is_complete)
        self.assertFalse(archived_state.is_complete)

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_stale_complete_session_index_keeps_load_more_on_cache(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        now = datetime.now(UTC)
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=now - timedelta(minutes=5),
            is_complete=True,
        )
        for index in range(51):
            SessionMetadata.objects.create(
                thread_id=f"cached-{index}",
                cwd="/repo",
                codex_display_title=f"Cached {index}",
                codex_name=f"Cached {index}",
                codex_created_at=datetime.fromtimestamp(1000 - index, UTC),
                codex_updated_at=datetime.fromtimestamp(1000 - index, UTC),
                codex_last_synced_at=now,
            )
        client = _setup_codex(mock_codex)
        mock_discover.return_value = []

        with (
            patch("hitch.main.caches._start_models_refresh_thread"),
            patch("hitch.main.views.common._start_usage_session_index_refresh_thread"),
            self.captureOnCommitCallbacks(execute=True),
        ):
            response = self.client.get(reverse("index"))
        load_more_url = self._load_more_url(response)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Cached 0")
        self.assertIn("cursor=idx%3A", load_more_url)
        self.assertNotIn("cursor=page-2", load_more_url)
        mock_codex.assert_not_called()
        client.thread_list.assert_not_called()
        state = SessionIndexSyncState.objects.get(source=SessionIndexSyncState.SOURCE_ACTIVE)
        self.assertTrue(state.is_complete)
        self.assertEqual(state.next_cursor, "")

        load_more_response = self.client.get(load_more_url)

        self.assertEqual(load_more_response.status_code, 200)
        self.assertContains(load_more_response, "Cached 50")
        mock_codex.assert_not_called()
        client.thread_list.assert_not_called()

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_system_sessions_backfill_missing_cached_metadata(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _setup_codex(mock_codex)
        mock_discover.return_value = []
        now = datetime.now(UTC)
        project = _make_project(name="Repo")
        _seed_cookies(self.client, **{_SELECTED_PROJECT_COOKIE: str(project.pk)})
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=now,
            is_complete=True,
        )
        SessionMetadata.objects.create(
            thread_id="visible",
            cwd="/repo",
            codex_display_title="Visible",
            codex_created_at=now,
            codex_updated_at=now,
            codex_last_synced_at=now,
            project=project,
        )
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_AUTONOMOUS_GOAL_RUN,
            main_thread_id="visible",
            cwd="/repo",
        )
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="system-thread",
            cwd="/repo",
            prompt="Autonomous goal prompt",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            display_author=system_agents.AUTONOMOUS_GOAL_DISPLAY_AUTHOR,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id="system-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_COMPLETED,
        )
        SessionMetadata.objects.create(
            thread_id="system-thread",
            cwd="/repo",
            project=project,
        )

        response = self.client.get(reverse("system_sessions"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, system_agents.AUTONOMOUS_GOAL_DISPLAY_AUTHOR)
        self.assertContains(response, "completed")
        metadata = SessionMetadata.objects.get(thread_id="system-thread")
        self.assertEqual(metadata.project, project)
        self.assertIsNotNone(metadata.codex_updated_at)



    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_hides_system_agent_threads(self, mock_codex: MagicMock, mock_discover: MagicMock) -> None:
        visible = _session("visible", preview="Visible")
        hidden = _session("system-thread", preview="Hidden system")
        _setup_codex(mock_codex, threads=[visible, hidden])
        mock_discover.return_value = []
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_AUTONOMOUS_GOAL_RUN,
            main_thread_id="visible",
            cwd="/repo",
        )
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="system-thread",
            cwd="/repo",
            prompt="autonomous goal",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id="system-thread",
            instance=instance,
        )

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Visible")
        self.assertNotContains(response, "Hidden system")

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_hides_system_agent_instance_threads_without_run_record(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        visible = _session("visible", preview="Visible")
        hidden = _session("autonomous-goal-thread", preview="Hidden autonomous goal")
        hidden.turns = []
        client = _setup_codex(mock_codex, threads=[visible, hidden])
        client._client.thread_resume.return_value = SimpleNamespace(thread=hidden)
        mock_discover.return_value = []
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_AUTONOMOUS_GOAL_RUN,
            main_thread_id="autonomous-goal:1",
            cwd="/repo",
        )
        CodexInstance.objects.create(
            pid=1,
            thread_id="autonomous-goal-thread",
            cwd="/repo",
            prompt="autonomous goal",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Visible")
        self.assertNotContains(response, "Hidden autonomous goal")

        system_response = self.client.get(reverse("system_sessions"))

        self.assertEqual(system_response.status_code, 200)
        self.assertContains(system_response, "Hidden autonomous goal")
        self.assertContains(system_response, "autonomous goal run")
        self.assertContains(system_response, "completed")
        self.assertContains(
            system_response,
            reverse(
                "system_session",
                kwargs={"session_id": "autonomous-goal-thread"},
            ),
        )

        detail_response = self.client.get(
            reverse(
                "system_session",
                kwargs={"session_id": "autonomous-goal-thread"},
            )
        )

        self.assertEqual(detail_response.status_code, 200)
        self.assertContains(detail_response, "autonomous goal run log")
        self.assertContains(detail_response, "System prompt")
        self.assertContains(detail_response, "autonomous goal")

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_hides_legacy_autonomous_goal_prompt_threads_without_source(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        visible = _session("visible", name="Visible")
        candidate = _session(
            "legacy-candidate",
            name=system_agents.AUTONOMOUS_GOAL_AGENT_PROMPT_TITLE,
            preview=(
                f"{system_agents.AUTONOMOUS_GOAL_AGENT_PROMPT_TITLE}\n\n"
                "Analyze the repo.\n\n"
                "Autonomous goal title: Docs\n\n"
                "Autonomous goal objective:\nKeep documentation tidy.\n\n"
                "Return only JSON matching this shape: {}"
            ),
        )
        judge = _session(
            "legacy-judge",
            name=system_agents.AUTONOMOUS_GOAL_JUDGE_PROMPT_TITLE,
            preview=(
                f"{system_agents.AUTONOMOUS_GOAL_JUDGE_PROMPT_TITLE}\n\n"
                "Judge it.\n\n"
                "Autonomous goal title: Docs\n\n"
                "Candidate session JSON:\n{}\n\n"
                "Return only JSON matching this shape: {}"
            ),
        )
        legacy_candidate = _session(
            "legacy-standing-candidate",
            name=session_index.LEGACY_AUTONOMOUS_GOAL_AGENT_PROMPT_TITLE,
            preview=(
                f"{session_index.LEGACY_AUTONOMOUS_GOAL_AGENT_PROMPT_TITLE}\n\n"
                "Analyze the repo.\n\n"
                "Standing order title: Docs\n\n"
                "Standing order goal:\nKeep documentation tidy.\n\n"
                "Return only JSON matching this shape: {}"
            ),
        )
        legacy_judge = _session(
            "legacy-standing-judge",
            name=session_index.LEGACY_AUTONOMOUS_GOAL_JUDGE_PROMPT_TITLE,
            preview=(
                f"{session_index.LEGACY_AUTONOMOUS_GOAL_JUDGE_PROMPT_TITLE}\n\n"
                "Judge it.\n\n"
                "Standing order title: Docs\n\n"
                "Candidate session JSON:\n{}\n\n"
                "Return only JSON matching this shape: {}"
            ),
        )
        _setup_codex(
            mock_codex,
            threads=[visible, candidate, judge, legacy_candidate, legacy_judge],
        )
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Visible")
        self.assertNotContains(response, "You are Hitch&#x27;s autonomous goal agent.")
        self.assertNotContains(response, "You are Hitch&#x27;s autonomous goal confidence judge.")
        self.assertNotContains(response, "You are Hitch&#x27;s standing order agent.")
        self.assertNotContains(response, "You are Hitch&#x27;s standing order confidence judge.")

        system_response = self.client.get(reverse("system_sessions"))

        self.assertEqual(system_response.status_code, 200)
        self.assertNotContains(system_response, "Visible")
        self.assertContains(system_response, "You are Hitch&#x27;s autonomous goal agent.")
        self.assertContains(system_response, "You are Hitch&#x27;s autonomous goal confidence judge.")
        self.assertContains(system_response, "You are Hitch&#x27;s standing order agent.")
        self.assertContains(system_response, "You are Hitch&#x27;s standing order confidence judge.")

    @patch("hitch.main.views.common.Codex")
    def test_untracked_system_session_non_thread_invalid_request_is_not_404(self, mock_codex: MagicMock) -> None:
        session_id = "00000000-0000-0000-0000-000000000001"
        client = _setup_codex(mock_codex)
        client._client.thread_resume.side_effect = InvalidRequestError(-32600, "model provider not found")

        with self.assertRaises(InvalidRequestError):
            self.client.get(reverse("system_session", kwargs={"session_id": session_id}))

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_accepted_candidate_thread_can_remain_visible(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        accepted = _session(
            "accepted-candidate",
            name="Accepted candidate",
            preview="You are Hitch's autonomous goal agent.\n\nAnalyze the repo.",
        )
        _setup_codex(mock_codex, threads=[accepted])
        mock_discover.return_value = []
        metadata = SessionMetadata.objects.create(
            thread_id="accepted-candidate",
            is_hidden_system_session=True,
        )
        CodexInstance.objects.create(
            pid=0,
            thread_id="accepted-candidate",
            cwd="/repo",
            prompt="Analyze the repo.",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
        )
        ProposedSession.objects.create(
            title="Accepted proposal",
            candidate_session=metadata,
            accepted_session=metadata,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
        )

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Accepted candidate")

        system_response = self.client.get(reverse("system_sessions"))
        self.assertNotContains(system_response, "Accepted candidate")



    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_index_keeps_pending_archive_rows_hidden(self, mock_codex: MagicMock, mock_discover: MagicMock) -> None:
        _setup_codex(mock_codex, threads=[_session("abc", name="Session")])
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertContains(response, ".session.pending-archive {")
        self.assertContains(response, "visibility: hidden;")


    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_visible_projects_filter_sessions(self, mock_codex: MagicMock, mock_discover: MagicMock) -> None:
        project = _make_project()
        other = _make_project(name="Other", repo_path="/other")
        sessions = [
            _session("matching", name="Matching", cwd="/repo"),
            _session("other", name="Other session", cwd="/other"),
            _session("no-project", name="No repo session", cwd="/elsewhere"),
        ]
        _setup_codex(mock_codex, threads=sessions)
        mock_discover.return_value = [Path("/repo"), Path("/other")]
        SessionMetadata.objects.create(thread_id="matching", cwd="/repo", project=project)
        SessionMetadata.objects.create(thread_id="other", cwd="/other", project=other)
        SessionMetadata.objects.create(thread_id="no-project", cwd="/elsewhere")

        response = self.client.post(
            reverse("update_visible_session_projects"),
            data={
                "visible_project": [str(other.pk)],
                "show_no_project_sessions": "true",
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            _cookie_value(response, _VISIBLE_SESSION_PROJECTS_COOKIE),
            f"[{other.pk}]",
        )
        self.assertEqual(
            _cookie_value(response, _SHOW_NO_PROJECT_SESSIONS_COOKIE),
            "true",
        )

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Other session")
        self.assertContains(response, "No repo session")
        self.assertNotContains(response, "Matching")

    @patch(
        "hitch.main.views.settings._visible_session_project_ids_cookie_fits",
        return_value=False,
    )
    def test_visible_projects_rejects_oversized_guest_cookie(self, mock_cookie_fits: MagicMock) -> None:
        project = _make_project()

        response = self.client.post(
            reverse("update_visible_session_projects"),
            data={"visible_project": [str(project.pk)]},
        )

        self.assertContains(
            response,
            "visible project selection is too large",
            status_code=400,
        )
        mock_cookie_fits.assert_called_once_with((project.pk,))
        self.assertNotIn(_VISIBLE_SESSION_PROJECTS_COOKIE, response.cookies)

    def test_settings_approve_all_updates_live_global_sessions(self) -> None:
        global_instance = CodexInstance.objects.create(
            pid=1,
            thread_id="global",
            cwd="/repo",
            prompt="hi",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
            approval_mode="prompt_user",
            approval_mode_live_editable=True,
        )
        global_pending = ApprovalRequest.objects.create(
            instance=global_instance,
            method="item/commandExecution/requestApproval",
            params={"item": {"command": "cargo bench"}},
            decision=ApprovalRequest.DECISION_PENDING,
        )
        SessionMetadata.objects.create(
            thread_id="override",
            cwd="/repo",
            approval_mode="prompt_user",
        )
        override_instance = CodexInstance.objects.create(
            pid=2,
            thread_id="override",
            cwd="/repo",
            prompt="hi",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
            approval_mode="prompt_user",
            approval_mode_live_editable=True,
        )
        override_pending = ApprovalRequest.objects.create(
            instance=override_instance,
            method="item/commandExecution/requestApproval",
            params={"item": {"command": "cargo test"}},
            decision=ApprovalRequest.DECISION_PENDING,
        )
        system_instance = CodexInstance.objects.create(
            pid=3,
            thread_id="system",
            cwd="/repo",
            prompt="qa",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            approval_mode="auto_review",
        )
        system_pending = ApprovalRequest.objects.create(
            instance=system_instance,
            method="item/commandExecution/requestApproval",
            params={"item": {"command": "cargo test"}},
            decision=ApprovalRequest.DECISION_PENDING,
        )

        response = self.client.post(
            reverse("update_settings"),
            data={"approval_mode": "approve_all"},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(_cookie_value(response, _APPROVAL_COOKIE), "approve_all")
        global_instance.refresh_from_db()
        self.assertEqual(global_instance.approval_mode, "approve_all")
        global_pending.refresh_from_db()
        self.assertEqual(global_pending.decision, ApprovalRequest.DECISION_ACCEPT)
        self.assertIsNotNone(global_pending.decided_at)
        override_instance.refresh_from_db()
        self.assertEqual(override_instance.approval_mode, "prompt_user")
        override_pending.refresh_from_db()
        self.assertEqual(override_pending.decision, ApprovalRequest.DECISION_PENDING)
        system_instance.refresh_from_db()
        self.assertEqual(system_instance.approval_mode, "auto_review")
        system_pending.refresh_from_db()
        self.assertEqual(system_pending.decision, ApprovalRequest.DECISION_PENDING)

    def test_settings_deny_all_declines_live_global_pending_approvals(self) -> None:
        global_instance = CodexInstance.objects.create(
            pid=1,
            thread_id="global",
            cwd="/repo",
            prompt="hi",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
            approval_mode="prompt_user",
            approval_mode_live_editable=True,
        )
        global_pending = ApprovalRequest.objects.create(
            instance=global_instance,
            method="item/commandExecution/requestApproval",
            params={"item": {"command": "cargo bench"}},
            decision=ApprovalRequest.DECISION_PENDING,
        )

        response = self.client.post(
            reverse("update_settings"),
            data={"approval_mode": "deny_all"},
        )

        self.assertEqual(response.status_code, 302)
        global_instance.refresh_from_db()
        self.assertEqual(global_instance.approval_mode, "deny_all")
        global_pending.refresh_from_db()
        self.assertEqual(global_pending.decision, ApprovalRequest.DECISION_DECLINE)
        self.assertIsNotNone(global_pending.decided_at)

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_session_list_omits_token_usage(self, mock_codex: MagicMock, mock_discover: MagicMock) -> None:
        rollout_path = _make_rollout(
            self,
            [
                _token_count_line(
                    input_tokens=400_000,
                    cached_input_tokens=25_000,
                    output_tokens=562_654,
                    total_tokens=987_654,
                )
            ],
        )
        active = _session("active", name="Active session", path=str(rollout_path))
        _setup_codex(mock_codex, threads=[active])
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Active session")
        self.assertNotContains(response, 'aria-label="Token usage"')
        self.assertNotContains(response, "987,654")
        self.assertEqual(ArchivedSessionTokenUsage.objects.count(), 0)

        rollout_path.write_text(
            _token_count_line(
                input_tokens=500_000,
                cached_input_tokens=30_000,
                output_tokens=704_567,
                total_tokens=1_234_567,
            ),
            encoding="utf-8",
        )

        response = self.client.get(reverse("index"))

        self.assertNotContains(response, "1,234,567")
        self.assertNotContains(response, "987,654")
        self.assertEqual(ArchivedSessionTokenUsage.objects.count(), 0)

    @patch("hitch.main.views.common.Codex")
    def test_usage_page_uses_cached_usage_and_refreshes_rollout_async(self, mock_codex: MagicMock) -> None:
        rollout_path = _make_rollout(
            self,
            [
                _token_count_line(
                    input_tokens=100_000,
                    cached_input_tokens=10_000,
                    output_tokens=23_456,
                    total_tokens=123_456,
                )
            ],
            archived=True,
        )
        os.utime(rollout_path, ns=(1_000_000_000, 1_000_000_000))
        _seed_usage_metadata(
            "archived",
            path=str(rollout_path),
        )
        _cache_token_usage(
            "archived",
            input_tokens=100_000,
            cached_input_tokens=10_000,
            output_tokens=23_456,
            total_tokens=123_456,
            path=rollout_path,
        )
        client = _setup_codex(mock_codex)

        with (
            patch("hitch.main.sessions.token_usage._start_usage_token_refresh_thread"),
            patch("hitch.main.caches._start_models_refresh_thread"),
            patch("hitch.main.caches._start_rate_limits_refresh_thread"),
        ):
            response = self.client.get(reverse("usage"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "90K")
        self.assertContains(response, "23K")
        self.assertContains(response, "10K")
        self.assertContains(response, "Refreshing session token usage...")
        self.assertNotContains(response, "113,456")
        self.assertNotContains(response, "123,456")
        cache = ArchivedSessionTokenUsage.objects.get(thread_id="archived")
        self.assertEqual(cache.total_tokens, 123_456)
        self.assertEqual(cache.rollout_mtime_ns, 1_000_000_000)
        client.thread_list.assert_not_called()

        rollout_path.write_text(
            _token_count_line(
                input_tokens=900_000,
                cached_input_tokens=90_000,
                output_tokens=99_999,
                total_tokens=999_999,
            ),
            encoding="utf-8",
        )
        os.utime(rollout_path, ns=(2_000_000_000, 2_000_000_000))
        SessionMetadata.objects.filter(thread_id="archived").update(usage_last_checked_at=datetime.now(UTC))

        with (
            patch("hitch.main.runtime.rollout.latest_token_usage") as latest_usage,
            patch("hitch.main.sessions.token_usage._start_usage_token_refresh_thread") as start_refresh,
            patch("hitch.main.caches._start_models_refresh_thread"),
            patch("hitch.main.caches._start_rate_limits_refresh_thread"),
            patch("hitch.main.sessions.token_usage._rollout_file_state_from_value") as rollout_state,
            self.captureOnCommitCallbacks(execute=True),
        ):
            response = self.client.get(reverse("usage"))

        latest_usage.assert_not_called()
        rollout_state.assert_not_called()
        start_refresh.assert_not_called()
        self.assertNotContains(response, "Refreshing session token usage...")
        self.assertContains(response, "90K")
        self.assertContains(response, "23K")
        self.assertContains(response, "10K")
        self.assertNotContains(response, "810K")
        self.assertNotContains(response, "909,999")
        self.assertNotContains(response, "999,999")

        SessionMetadata.objects.filter(thread_id="archived").update(
            usage_last_checked_at=(
                datetime.now(UTC) - token_usage._USAGE_TOKEN_REFRESH_CHECK_INTERVAL - timedelta(seconds=1)
            )
        )
        with (
            patch("hitch.main.runtime.rollout.latest_token_usage") as latest_usage,
            patch("hitch.main.sessions.token_usage._start_usage_token_refresh_thread") as start_refresh,
            patch("hitch.main.caches._start_models_refresh_thread"),
            patch("hitch.main.caches._start_rate_limits_refresh_thread"),
            patch("hitch.main.sessions.token_usage._rollout_file_state_from_value") as rollout_state,
            self.captureOnCommitCallbacks(execute=True),
        ):
            response = self.client.get(reverse("usage"))

        latest_usage.assert_not_called()
        rollout_state.assert_not_called()
        start_refresh.assert_called_once()
        refresh_items = start_refresh.call_args.args[0]
        self.assertEqual(len(refresh_items), 1)
        self.assertEqual(refresh_items[0].thread_id, "archived")
        self.assertEqual(refresh_items[0].codex_path, str(rollout_path))
        self.assertContains(response, "Refreshing session token usage...")
        lifetime_usage = cast(dict[str, Any], response.context["lifetime_usage"])
        self.assertEqual(lifetime_usage["total"]["input"], "90K")
        self.assertEqual(lifetime_usage["total"]["output"], "23K")
        self.assertEqual(lifetime_usage["total"]["cached"], "10K")
        self.assertContains(response, "90K")
        self.assertContains(response, "23K")
        self.assertContains(response, "10K")
        self.assertNotContains(response, "810K")
        self.assertNotContains(response, "909,999")
        self.assertNotContains(response, "999,999")

        token_usage._refresh_usage_token_cache_best_effort(
            [token_usage._UsageTokenRefreshItem("archived", str(rollout_path))]
        )

        self.assertNotContains(response, "909,999")
        self.assertNotContains(response, "999,999")
        cache.refresh_from_db()
        self.assertEqual(cache.total_tokens, 999_999)
        self.assertEqual(cache.rollout_mtime_ns, 2_000_000_000)

    @patch("hitch.main.caches.Codex")
    def test_usage_page_refreshes_rate_limits_after_first_render(self, mock_codex: MagicMock) -> None:
        session_index.mark_synced(archived=False, complete=True)
        session_index.mark_synced(archived=True, complete=True)
        client = _setup_codex(mock_codex)
        client._client.request.side_effect = None
        client._client.request.return_value = SimpleNamespace(
            rate_limits=SimpleNamespace(
                primary=SimpleNamespace(
                    used_percent=73,
                    resets_at="2026-05-30T12:00:00Z",
                    window_duration_mins=300,
                ),
                secondary=None,
                limit_name="Codex",
                plan_type=SimpleNamespace(value="Pro"),
            )
        )

        with (
            patch("hitch.main.caches._RATE_LIMITS_CACHE_VALUE", None),
            patch("hitch.main.caches._RATE_LIMITS_CACHE_HAS_VALUE", False),
            patch("hitch.main.caches._RATE_LIMITS_CACHE_FETCHED_AT", None),
            patch("hitch.main.caches._RATE_LIMITS_REFRESH_ATTEMPTED_AT", None),
            patch("hitch.main.caches._RATE_LIMITS_REFRESH_IN_FLIGHT", False),
            patch("hitch.main.caches._start_models_refresh_thread"),
            patch("hitch.main.caches.threading.Thread") as refresh_thread,
            patch(
                "hitch.main.caches.transaction.on_commit",
                side_effect=lambda callback: callback(),
            ),
        ):
            response = self.client.get(reverse("usage"))

            self.assertEqual(response.status_code, 200)
            self.assertNotContains(response, "Codex rate limits")
            self.assertContains(response, "Refreshing quota usage...")
            self.assertNotContains(response, "Usage unavailable.")
            client._client.request.assert_not_called()
            refresh_thread.assert_called_once_with(
                target=caches._refresh_rate_limits_cache_best_effort,
                kwargs={"enable_memories": False},
                name="rate-limits-refresh",
                daemon=True,
            )
            refresh_thread.return_value.start.assert_called_once_with()

            caches._refresh_rate_limits_cache_best_effort(enable_memories=False)

            refreshed_response = self.client.get(reverse("usage"))

            self.assertContains(refreshed_response, "Codex rate limits")
            self.assertContains(refreshed_response, "Plan: Pro")
            self.assertContains(refreshed_response, "27% remaining")
            self.assertContains(refreshed_response, "5-hour window")
            self.assertNotContains(refreshed_response, "Refreshing quota usage...")
            client._client.request.assert_called_once()
            refresh_thread.assert_called_once()

    @patch("hitch.main.views.common.Codex")
    def test_usage_page_hides_totals_until_active_and_archived_indexes_complete(self, mock_codex: MagicMock) -> None:
        rollout_path = _make_rollout(
            self,
            [
                _token_count_line(
                    input_tokens=400,
                    cached_input_tokens=50,
                    output_tokens=600,
                    total_tokens=1_000,
                )
            ],
        )
        session_index.mark_synced(archived=False, complete=True)
        SessionMetadata.objects.create(
            thread_id="active-only",
            codex_path=str(rollout_path),
            codex_updated_at=datetime(2025, 1, 5, tzinfo=UTC),
            usage_last_checked_at=datetime.now(UTC),
        )
        _cache_token_usage(
            "active-only",
            input_tokens=400,
            cached_input_tokens=50,
            output_tokens=600,
            total_tokens=1_000,
            path=rollout_path,
        )
        client = _setup_codex(mock_codex)

        with (
            patch("hitch.main.views.common._start_usage_session_index_refresh_thread") as start_index_refresh,
            patch("hitch.main.sessions.token_usage._start_usage_token_refresh_thread") as start_tokens,
            patch("hitch.main.caches._start_models_refresh_thread"),
            patch("hitch.main.caches._start_rate_limits_refresh_thread"),
            self.captureOnCommitCallbacks(execute=True),
        ):
            response = self.client.get(reverse("usage"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "All sessions usage unavailable.")
        self.assertIsNone(response.context["lifetime_usage"])
        client.thread_list.assert_not_called()
        start_tokens.assert_not_called()
        start_index_refresh.assert_called_once_with(
            enable_memories=False,
            include_active=False,
            include_archived=True,
        )

    def test_token_usage_snapshot_drops_stale_cache_when_rollout_has_no_usage(
        self,
    ) -> None:
        rollout_path = _make_rollout(self, ["{}"])
        os.utime(rollout_path, ns=(2_000_000_000, 2_000_000_000))
        ArchivedSessionTokenUsage.objects.create(
            thread_id="active",
            rollout_path=str(rollout_path),
            rollout_mtime_ns=1_000_000_000,
            input_tokens=100,
            cached_input_tokens=10,
            output_tokens=20,
            total_tokens=120,
            daily_usage={"2025-01-05": {"input": 90, "output": 20, "cached": 10}},
        )
        thread = _session("active", name="Active session", path=str(rollout_path))

        self.assertIsNone(token_usage._token_usage_snapshot_for(thread))

    def test_zero_usage_cache_short_circuits_rollout_reparse(self) -> None:
        # A rollout with no token_count events caches an all-zero row with an
        # empty daily map; that row must satisfy the cache gate -- demanding a
        # non-empty daily history meant every later read re-parsed the whole
        # file forever while still serving the same zeros.
        rollout_path = _make_rollout(self, ["{}"], archived=True)
        ArchivedSessionTokenUsage.objects.create(
            thread_id="archived",
            rollout_path=str(rollout_path),
            rollout_mtime_ns=rollout_path.stat().st_mtime_ns,
            usage_logic_version=token_usage._TOKEN_USAGE_LOGIC_VERSION,
        )
        thread = _session("archived", path=str(rollout_path))

        with patch(
            "hitch.main.sessions.token_usage._parse_token_usage_and_daily",
            side_effect=AssertionError("zero-usage cache row must short-circuit"),
        ):
            snapshot = token_usage._token_usage_snapshot_for(thread)

        assert snapshot is not None
        self.assertEqual(snapshot["usage"]["total_tokens"], 0)
        self.assertEqual(snapshot["daily_usage"], {})

    def test_usage_token_cache_state_accepts_archived_rollout_path_aliases(self) -> None:
        filename = "rollout-2025-01-05T12-00-00-thread.jsonl"
        active_path = f"/codex/sessions/2025/01/05/{filename}"
        metadata = token_usage._UsageTokenRefreshCandidate(
            thread_id="thread",
            codex_path=active_path,
            usage_last_checked_at=datetime.now(UTC),
        )

        for archived_path in (
            f"/codex/archived_sessions/{filename}",
            f"/codex/archived_sessions/2025/01/05/{filename}",
        ):
            with self.subTest(archived_path=archived_path):
                cache = ArchivedSessionTokenUsage(
                    thread_id="thread",
                    rollout_path=archived_path,
                    usage_logic_version=token_usage._TOKEN_USAGE_LOGIC_VERSION,
                    daily_usage={"2025-01-05": {"input": 1}},
                )

                with patch("hitch.main.sessions.token_usage._rollout_file_state_from_value") as rollout_state:
                    cache_state = token_usage._usage_token_cache_state(metadata, cache)

                rollout_state.assert_not_called()
                self.assertTrue(cache_state.cache_usable)
                self.assertFalse(cache_state.refresh_pending)

    def test_token_usage_snapshot_survives_compaction_reset(self) -> None:
        # A session that exhausts its context window records a token_count
        # whose total_token_usage is reset to zero (plus the window size). The
        # headline cumulative figure and the per-day chart are derived from two
        # different rollout reads (latest totals vs daily history),
        # so both must account for the pre-reset spend or the usage page shows
        # a session that suddenly "lost" most of its tokens and a chart that
        # disagrees with its own total.
        rollout_path = _make_rollout(
            self,
            [
                _token_count_line(
                    input_tokens=100_000,
                    cached_input_tokens=80_000,
                    output_tokens=20_000,
                    total_tokens=120_000,
                    context_tokens=120_000,
                    model_context_window=200_000,
                    timestamp="2025-01-05T12:00:00Z",
                ),
                _token_count_line(
                    input_tokens=0,
                    cached_input_tokens=0,
                    output_tokens=0,
                    total_tokens=200_000,
                    context_tokens=200_000,
                    model_context_window=200_000,
                    timestamp="2025-01-05T13:00:00Z",
                ),
                _token_count_line(
                    input_tokens=50_000,
                    cached_input_tokens=10_000,
                    output_tokens=5_000,
                    total_tokens=55_000,
                    context_tokens=55_000,
                    model_context_window=200_000,
                    timestamp="2025-01-06T12:00:00Z",
                ),
            ],
            archived=True,
        )
        thread = _session("archived", path=str(rollout_path))

        snapshot = token_usage._token_usage_snapshot_for(thread)
        assert snapshot is not None
        usage = snapshot["usage"]
        self.assertEqual(usage["input_tokens"], 150_000)
        self.assertEqual(usage["cached_input_tokens"], 90_000)
        self.assertEqual(usage["output_tokens"], 25_000)

        # Headline non-cached/cached/output must equal the sum of the per-day
        # buckets shown in the chart.
        daily = snapshot["daily_usage"]
        self.assertEqual(
            sum(bucket["input"] for bucket in daily.values()),
            token_usage._non_cached_input_tokens(usage),
        )
        self.assertEqual(
            sum(bucket["cached"] for bucket in daily.values()),
            usage["cached_input_tokens"],
        )
        self.assertEqual(
            sum(bucket["output"] for bucket in daily.values()),
            usage["output_tokens"],
        )

    def test_token_usage_snapshot_stamps_pre_read_mtime_on_concurrent_append(
        self,
    ) -> None:
        # The cache must be stamped with the rollout mtime captured BEFORE the
        # parse, not a fresh stat taken after it. A turn appended while the
        # snapshot is computed would otherwise leave the cache holding
        # pre-append numbers but stamped with the post-append mtime, so the
        # stale value reads back as "current" and never refreshes once the
        # session goes idle.
        from hitch.main.runtime import rollout

        rollout_path = _make_rollout(
            self,
            [
                _token_count_line(
                    input_tokens=100,
                    cached_input_tokens=10,
                    output_tokens=20,
                    total_tokens=120,
                )
            ],
        )
        pre_mtime = 1_000_000_000
        post_mtime = 2_000_000_000
        os.utime(rollout_path, ns=(pre_mtime, pre_mtime))
        thread = _session("racing", path=str(rollout_path))

        original_load = rollout._load_rollout_lines
        appended = {"done": False}

        def load_then_append(path: Path) -> Any:
            lines = original_load(path)
            if not appended["done"]:
                appended["done"] = True
                with open(path, "a", encoding="utf-8") as handle:
                    handle.write(
                        "\n"
                        + _token_count_line(
                            input_tokens=500,
                            cached_input_tokens=50,
                            output_tokens=200,
                            total_tokens=700,
                        )
                    )
                os.utime(path, ns=(post_mtime, post_mtime))
            return lines

        with patch.object(rollout, "_load_rollout_lines", side_effect=load_then_append):
            snapshot = token_usage._token_usage_snapshot_for(thread)

        assert snapshot is not None
        # The snapshot reflects the content actually parsed (pre-append).
        self.assertEqual(snapshot["usage"]["input_tokens"], 100)

        cache = ArchivedSessionTokenUsage.objects.get(thread_id="racing")
        self.assertEqual(cache.rollout_mtime_ns, pre_mtime)

        # The next read sees the mismatch and re-parses the appended file rather
        # than serving the stale cached numbers.
        refreshed = token_usage._token_usage_snapshot_for(thread)
        assert refreshed is not None
        self.assertEqual(refreshed["usage"]["input_tokens"], 500)


    @patch("hitch.main.views.common.Codex")
    def test_profile_shows_selected_project_token_usage(self, mock_codex: MagicMock) -> None:
        project = _make_project()
        other_project = _make_project(name="Other", repo_path="/other")
        _seed_cookies(self.client, **{_SELECTED_PROJECT_COOKIE: str(project.pk)})
        session_path = _make_rollout(
            self,
            [
                _token_count_line(
                    input_tokens=400,
                    cached_input_tokens=50,
                    output_tokens=250,
                    total_tokens=700,
                )
            ],
        )
        system_path = _make_rollout(
            self,
            [
                _token_count_line(
                    input_tokens=200,
                    cached_input_tokens=20,
                    output_tokens=80,
                    total_tokens=300,
                )
            ],
        )
        other_path = _make_rollout(
            self,
            [
                _token_count_line(
                    input_tokens=3_000,
                    cached_input_tokens=200,
                    output_tokens=2_000,
                    total_tokens=5_000,
                )
            ],
        )
        _seed_usage_metadata("session", path=session_path, project=project)
        _seed_usage_metadata(
            "system",
            path=system_path,
            project=project,
            thread_source=ThreadSource.subagent.value,
        )
        _seed_usage_metadata("other", path=other_path, project=other_project)
        _cache_token_usage(
            "session",
            input_tokens=400,
            cached_input_tokens=50,
            output_tokens=250,
            total_tokens=700,
            path=session_path,
        )
        _cache_token_usage(
            "system",
            input_tokens=200,
            cached_input_tokens=20,
            output_tokens=80,
            total_tokens=300,
            path=system_path,
        )
        _cache_token_usage(
            "other",
            input_tokens=3_000,
            cached_input_tokens=200,
            output_tokens=2_000,
            total_tokens=5_000,
            path=other_path,
        )
        client = _setup_codex(mock_codex)

        response = self.client.get(reverse("profile"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Active project")
        self.assertContains(response, "Hitch")
        self.assertContains(response, "System sessions")
        self.assertContains(response, 'class="project-usage-card"', count=2)
        self.assertContains(response, "total tokens", count=2)
        lifetime_usage = cast(dict[str, Any], response.context["lifetime_usage"])
        project_usage = lifetime_usage["selected_project"]
        self.assertEqual(project_usage["total"]["total"], "930")
        self.assertEqual(project_usage["total"]["input"], "530")
        self.assertEqual(project_usage["total"]["output"], "330")
        self.assertEqual(project_usage["total"]["cached"], "70")
        self.assertEqual(project_usage["system"]["total"], "280")
        self.assertEqual(project_usage["system"]["input"], "180")
        self.assertEqual(project_usage["system"]["output"], "80")
        self.assertEqual(project_usage["system"]["cached"], "20")
        client.thread_list.assert_not_called()

    def test_lifetime_token_chart_formats_segments(self) -> None:
        self.assertEqual(token_usage._format_lifetime_token_chart({}), [])
        self.assertEqual(token_usage._format_lifetime_token_chart_axis({}), [])
        self.assertEqual(token_usage._chart_segment_percent(0, 100), 0)
        self.assertEqual(token_usage._chart_segment_percent(5, 0), 0)
        self.assertEqual(token_usage._chart_segment_percent(1, 1_000), 0)
        self.assertEqual(
            token_usage._format_lifetime_token_chart(
                {
                    "2025-01-06": {"input": 50, "output": 50, "cached": 0},
                    "2025-01-05": {"input": 100, "output": 50, "cached": 50},
                }
            ),
            [
                {
                    "date": "2025-01-05",
                    "input": "100",
                    "output": "50",
                    "cached": "50",
                    "total": "200",
                    "input_percent": 50,
                    "output_percent": 25,
                    "cached_percent": 25,
                },
                {
                    "date": "2025-01-06",
                    "input": "50",
                    "output": "50",
                    "cached": "0",
                    "total": "100",
                    "input_percent": 25,
                    "output_percent": 25,
                    "cached_percent": 0,
                },
            ],
        )
        self.assertEqual(
            token_usage._format_lifetime_token_chart_axis(
                {
                    "2025-01-06": {"input": 50, "output": 50, "cached": 0},
                    "2025-01-05": {"input": 100, "output": 50, "cached": 50},
                }
            ),
            ["200", "100", "0"],
        )

    def test_usage_sweep_refreshes_terminal_rollout_that_reappears(self) -> None:
        rollout_path = _make_rollout(self, [])
        rollout_path.unlink()
        metadata = _seed_usage_metadata("restored", path=rollout_path)
        metadata.usage_last_checked_at = (
            datetime.now(UTC) - token_usage._USAGE_TOKEN_REFRESH_CHECK_INTERVAL - timedelta(seconds=1)
        )
        metadata.save(update_fields=["usage_last_checked_at"])
        cache = ArchivedSessionTokenUsage.objects.create(
            thread_id="restored",
            rollout_path=str(rollout_path),
            rollout_mtime_ns=0,
            usage_logic_version=token_usage._TOKEN_USAGE_LOGIC_VERSION,
        )
        candidates = token_usage._usage_token_refresh_candidates([metadata])

        self.assertTrue(token_usage._usage_token_cache_state(metadata, cache).refresh_pending)
        rollout_path.write_text(
            _token_count_line(
                input_tokens=400,
                cached_input_tokens=50,
                output_tokens=600,
                total_tokens=1_000,
            ),
            encoding="utf-8",
        )

        token_usage._refresh_usage_token_cache_best_effort(candidates)

        metadata.refresh_from_db()
        cache.refresh_from_db()
        self.assertGreater(cache.rollout_mtime_ns, 0)
        self.assertEqual(cache.total_tokens, 1_000)
        self.assertFalse(token_usage._usage_token_cache_state(metadata, cache).refresh_pending)

    @patch("hitch.main.sessions.token_usage.Codex")
    def test_usage_page_schedules_missing_metadata_path_refresh(self, mock_codex: MagicMock) -> None:
        rollout_path = _make_rollout(
            self,
            [
                _token_count_line(
                    input_tokens=400,
                    cached_input_tokens=50,
                    output_tokens=600,
                    total_tokens=1_000,
                )
            ],
        )
        _seed_usage_metadata("local-session")
        client = _setup_codex(mock_codex)

        with (
            patch("hitch.main.sessions.token_usage._start_usage_token_refresh_thread") as start_refresh,
            patch("hitch.main.caches._start_models_refresh_thread"),
            patch("hitch.main.caches._start_rate_limits_refresh_thread"),
            self.captureOnCommitCallbacks(execute=True),
        ):
            response = self.client.get(reverse("usage"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Refreshing session token usage...")
        metadata = SessionMetadata.objects.get(thread_id="local-session")
        self.assertEqual(metadata.codex_path, "")
        client._client.thread_resume.assert_not_called()
        client.thread_list.assert_not_called()
        start_refresh.assert_called_once()
        refresh_items = start_refresh.call_args.args[0]
        self.assertEqual(len(refresh_items), 1)
        self.assertEqual(refresh_items[0].thread_id, "local-session")
        self.assertEqual(refresh_items[0].codex_path, "")

        client._client.thread_resume.return_value = SimpleNamespace(
            thread=_session("local-session", path=str(rollout_path), cwd="/repo")
        )
        token_usage._refresh_usage_token_cache_best_effort(refresh_items)

        metadata.refresh_from_db()
        self.assertEqual(metadata.codex_path, str(rollout_path))
        cache = ArchivedSessionTokenUsage.objects.get(thread_id="local-session")
        self.assertEqual(cache.total_tokens, 1_000)

    @patch("hitch.main.sessions.token_usage.Codex")
    def test_usage_refresh_caches_zero_when_missing_path_cannot_be_repaired(self, mock_codex: MagicMock) -> None:
        _seed_usage_metadata("missing-path")
        client = _setup_codex(mock_codex)
        client._client.thread_resume.side_effect = CodexError("resume failed")

        token_usage._refresh_usage_token_cache_best_effort([token_usage._UsageTokenRefreshItem("missing-path", "")])

        metadata = SessionMetadata.objects.get(thread_id="missing-path")
        cache = ArchivedSessionTokenUsage.objects.get(thread_id="missing-path")
        self.assertEqual(cache.rollout_path, "")
        self.assertEqual(cache.total_tokens, 0)
        self.assertEqual(cache.daily_usage, {})
        self.assertIsNotNone(metadata.usage_last_checked_at)
        self.assertFalse(token_usage._usage_token_cache_state(metadata, cache).refresh_pending)
        self.assertFalse(token_usage._usage_token_refresh_needed(metadata, cache))

    @patch("hitch.main.sessions.token_usage.Codex")
    def test_usage_refresh_keeps_existing_terminal_missing_path_cache(self, mock_codex: MagicMock) -> None:
        missing_path = "/nonexistent/rollout.jsonl"
        _seed_usage_metadata("missing-path", path=missing_path)
        ArchivedSessionTokenUsage.objects.create(
            thread_id="missing-path",
            rollout_path=missing_path,
            rollout_mtime_ns=0,
            input_tokens=400,
            cached_input_tokens=50,
            output_tokens=600,
            total_tokens=1_000,
            daily_usage={"2025-01-05": {"input": 350, "output": 600, "cached": 50}},
            usage_logic_version=token_usage._TOKEN_USAGE_LOGIC_VERSION,
        )
        client = _setup_codex(mock_codex)
        client._client.thread_resume.side_effect = CodexError("resume failed")

        token_usage._refresh_usage_token_cache_best_effort(
            [token_usage._UsageTokenRefreshItem("missing-path", missing_path)]
        )

        metadata = SessionMetadata.objects.get(thread_id="missing-path")
        cache = ArchivedSessionTokenUsage.objects.get(thread_id="missing-path")
        self.assertEqual(cache.rollout_path, missing_path)
        self.assertEqual(cache.rollout_mtime_ns, 0)
        self.assertEqual(cache.total_tokens, 1_000)
        self.assertFalse(token_usage._usage_token_cache_state(metadata, cache).refresh_pending)
        self.assertFalse(token_usage._usage_token_refresh_needed(metadata, cache))

    @patch("hitch.main.sessions.token_usage.Codex")
    def test_usage_refresh_missing_metadata_path_handles_unexpected_resume_error(self, mock_codex: MagicMock) -> None:
        client = _setup_codex(mock_codex)
        client._client.thread_resume.side_effect = RuntimeError("boom")

        refreshed_path = token_usage._refresh_missing_usage_metadata_path(client, "missing-path", projects=[])

        self.assertIsNone(refreshed_path)

    def test_usage_refresh_keeps_pathless_old_cache_repair_pending(self) -> None:
        _seed_usage_metadata("missing-path")
        SessionMetadata.objects.filter(thread_id="missing-path").update(usage_last_checked_at=datetime.now(UTC))
        cache = ArchivedSessionTokenUsage.objects.create(
            thread_id="missing-path",
            rollout_path="/old/rollout.jsonl",
            rollout_mtime_ns=1_000_000_000,
            input_tokens=400,
            cached_input_tokens=50,
            output_tokens=600,
            total_tokens=1_000,
            daily_usage={"2025-01-05": {"input": 350, "output": 600, "cached": 50}},
            usage_logic_version=token_usage._TOKEN_USAGE_LOGIC_VERSION,
        )

        metadata = SessionMetadata.objects.get(thread_id="missing-path")
        cache_state = token_usage._usage_token_cache_state(metadata, cache)

        self.assertTrue(cache_state.cache_usable)
        self.assertTrue(cache_state.refresh_pending)
        self.assertTrue(token_usage._usage_token_refresh_needed(metadata, cache))

    def test_usage_refresh_zeros_stale_cache_when_rollout_has_no_usage(self) -> None:
        rollout_path = _make_rollout(self, ["{}"])
        os.utime(rollout_path, ns=(2_000_000_000, 2_000_000_000))
        _seed_usage_metadata("stale", path=rollout_path)
        ArchivedSessionTokenUsage.objects.create(
            thread_id="stale",
            rollout_path=str(rollout_path),
            rollout_mtime_ns=1_000_000_000,
            input_tokens=400,
            cached_input_tokens=50,
            output_tokens=600,
            total_tokens=1_000,
            daily_usage={"2025-01-05": {"input": 350, "output": 600, "cached": 50}},
            usage_logic_version=token_usage._TOKEN_USAGE_LOGIC_VERSION,
        )

        token_usage._refresh_usage_token_cache_best_effort(
            [token_usage._UsageTokenRefreshItem("stale", str(rollout_path))]
        )

        cache = ArchivedSessionTokenUsage.objects.get(thread_id="stale")
        self.assertEqual(cache.total_tokens, 0)
        self.assertEqual(cache.rollout_mtime_ns, 2_000_000_000)
        self.assertEqual(cache.daily_usage, {})

    @patch("hitch.main.sessions.token_usage.Codex")
    def test_usage_refresh_preserves_cache_when_rollout_path_missing(self, mock_codex: MagicMock) -> None:
        _seed_usage_metadata("missing", path="/nonexistent/rollout.jsonl")
        cache = ArchivedSessionTokenUsage.objects.create(
            thread_id="missing",
            rollout_path="/old/rollout.jsonl",
            rollout_mtime_ns=1_000_000_000,
            input_tokens=400,
            cached_input_tokens=50,
            output_tokens=600,
            total_tokens=1_000,
            daily_usage={"2025-01-05": {"input": 350, "output": 600, "cached": 50}},
            usage_logic_version=token_usage._TOKEN_USAGE_LOGIC_VERSION,
        )
        client = _setup_codex(mock_codex)
        client._client.thread_resume.side_effect = CodexError("resume failed")

        token_usage._refresh_usage_token_cache_best_effort(
            [token_usage._UsageTokenRefreshItem("missing", "/nonexistent/rollout.jsonl")]
        )

        cache.refresh_from_db()
        self.assertEqual(cache.total_tokens, 1_000)
        self.assertEqual(cache.rollout_path, "/nonexistent/rollout.jsonl")
        self.assertEqual(cache.rollout_mtime_ns, 0)
        metadata = SessionMetadata.objects.get(thread_id="missing")
        self.assertIsNotNone(metadata.usage_last_checked_at)
        cache_state = token_usage._usage_token_cache_state(metadata, cache)
        self.assertTrue(cache_state.cache_usable)
        self.assertFalse(cache_state.refresh_pending)
        self.assertFalse(token_usage._usage_token_refresh_needed(metadata, cache))

    def test_usage_refresh_marks_checked_rows_in_chunks(self) -> None:
        for index in range(5):
            _seed_usage_metadata(
                f"checked-{index}",
                mark_index_complete=False,
            )

        with (
            patch("hitch.main.sessions.token_usage._USAGE_TOKEN_REFRESH_CHECKED_UPDATE_BATCH_SIZE", 2),
            CaptureQueriesContext(connection) as queries,
        ):
            token_usage._mark_usage_token_refresh_checked_many(
                [
                    "checked-0",
                    "",
                    "checked-1",
                    "checked-1",
                    "checked-2",
                    "checked-3",
                    "checked-4",
                ]
            )

        update_queries = [
            query for query in queries.captured_queries if 'UPDATE "main_sessionmetadata"' in query["sql"]
        ]
        self.assertEqual(len(update_queries), 3)
        self.assertEqual(
            SessionMetadata.objects.filter(
                thread_id__startswith="checked-",
                usage_last_checked_at__isnull=False,
            ).count(),
            5,
        )

    def test_usage_refresh_thread_start_failure_clears_in_flight(self) -> None:
        token_usage._USAGE_TOKEN_REFRESH_IN_FLIGHT = False
        self.addCleanup(setattr, token_usage, "_USAGE_TOKEN_REFRESH_IN_FLIGHT", False)
        thread = MagicMock()
        thread.start.side_effect = RuntimeError("thread limit")

        with (
            self.assertLogs("hitch.main.sessions.token_usage", level="ERROR"),
            patch("hitch.main.sessions.token_usage.threading.Thread", return_value=thread),
        ):
            token_usage._start_usage_token_refresh_thread([token_usage._UsageTokenRefreshItem("thread", "")])

        self.assertFalse(token_usage._USAGE_TOKEN_REFRESH_IN_FLIGHT)

    def test_usage_refresh_drains_all_candidate_batches(self) -> None:
        for index in range(5):
            rollout_path = _make_rollout(
                self,
                [
                    _token_count_line(
                        input_tokens=100 + index,
                        cached_input_tokens=10,
                        output_tokens=20,
                        total_tokens=120 + index,
                    )
                ],
            )
            _seed_usage_metadata(
                f"batch-{index}",
                path=rollout_path,
                mark_index_complete=False,
            )
        rows = SessionMetadata.objects.order_by("thread_id")
        candidates = token_usage._usage_token_refresh_candidates(rows)

        with patch("hitch.main.sessions.token_usage._USAGE_TOKEN_REFRESH_BATCH_SIZE", 2):
            token_usage._refresh_usage_token_cache_best_effort(candidates)

        caches = ArchivedSessionTokenUsage.objects.order_by("thread_id")
        self.assertEqual(
            [cache.total_tokens for cache in caches],
            [120, 121, 122, 123, 124],
        )
        self.assertEqual(
            SessionMetadata.objects.filter(
                thread_id__startswith="batch-",
                usage_last_checked_at__isnull=False,
            ).count(),
            5,
        )

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_repo_dropdown_selects_saved_repo(self, mock_codex: MagicMock, mock_discover: MagicMock) -> None:
        _seed_cookies(
            self.client,
            **{_LAST_SELECTED_REPO_COOKIE: "/home/user/proj-b"},
        )
        _setup_codex(mock_codex)
        mock_discover.return_value = [Path("/home/user/proj-a"), Path("/home/user/proj-b")]

        response = self.client.get(reverse("new_session"))

        self.assertContains(response, 'value="/home/user/proj-b" selected')
        self.assertNotContains(response, 'value="/home/user/proj-a" selected')


class ProjectViewTests(TestCase):
    @patch("hitch.main.repos.discover_repos")
    def test_new_project_form_hides_repos_that_already_have_projects(self, mock_discover: MagicMock) -> None:
        _make_project(name="Existing")
        mock_discover.return_value = [Path("/repo"), Path("/other")]

        response = self.client.get(reverse("new_project"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, '<option value="/repo">/repo</option>', html=True)
        self.assertContains(response, '<option value="/other">/other</option>', html=True)

    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.repos.discover_repos")
    def test_creates_project_selects_it_and_associates_existing_sessions(
        self, mock_discover: MagicMock, mock_codex: MagicMock
    ) -> None:
        other = _make_project(name="Other", repo_path="/other")
        _seed_cookies(
            self.client,
            **{_VISIBLE_SESSION_PROJECTS_COOKIE: f"[{other.pk}]"},
        )
        mock_discover.return_value = [Path("/repo")]
        _setup_codex(
            mock_codex,
            threads=[
                _session("match", name="Match", cwd="/repo"),
                _session("miss", name="Miss", cwd="/other"),
            ],
        )

        response = self.client.post(
            reverse("new_project"),
            data={"name": "Hitch", "repo_path": "/repo"},
        )

        self.assertEqual(response.status_code, 302)
        project = Project.objects.get(repo_path="/repo")
        self.assertEqual(project.name, "Hitch")
        self.assertEqual(project.repo_path, "/repo")
        self.assertTrue(project.auto_pull_enabled)
        self.assertEqual(_cookie_value(response, "hitch_selected_project_id"), str(project.pk))
        self.assertEqual(
            _cookie_value(response, _VISIBLE_SESSION_PROJECTS_COOKIE),
            f"[{other.pk},{project.pk}]",
        )
        self.assertEqual(SessionMetadata.objects.get(thread_id="match").project, project)
        self.assertFalse(SessionMetadata.objects.filter(thread_id="miss").exists())

    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.views.common.git_common_dir")
    @patch("hitch.main.repos.discover_repos")
    def test_rejects_project_with_existing_git_common_dir(
        self,
        mock_discover: MagicMock,
        mock_common_dir: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        _make_project(
            name="Source",
            git_common_dir="/repo/.git",
        )
        mock_discover.return_value = [Path("/repo-worktree")]
        mock_common_dir.return_value = Path("/repo/.git")
        _setup_codex(mock_codex)

        response = self.client.post(
            reverse("new_project"),
            data={"name": "Worktree", "repo_path": "/repo-worktree"},
        )

        self.assertContains(
            response,
            "project already exists for repository",
            status_code=400,
        )
        self.assertEqual(Project.objects.count(), 1)

    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.repos.discover_repos")
    def test_project_creation_preserves_manually_cleared_sessions(
        self, mock_discover: MagicMock, mock_codex: MagicMock
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        _setup_codex(
            mock_codex,
            threads=[
                _session("cleared", name="Cleared", cwd="/repo"),
                _session("ordinary", name="Ordinary", cwd="/repo"),
            ],
        )
        SessionMetadata.objects.create(
            thread_id="cleared",
            cwd="/repo",
            project=None,
            project_cleared=True,
        )
        SessionMetadata.objects.create(thread_id="ordinary", cwd="/repo", project=None)

        response = self.client.post(
            reverse("new_project"),
            data={"name": "Hitch", "repo_path": "/repo"},
        )

        self.assertEqual(response.status_code, 302)
        project = Project.objects.get()
        self.assertIsNone(SessionMetadata.objects.get(thread_id="cleared").project)
        self.assertTrue(SessionMetadata.objects.get(thread_id="cleared").project_cleared)
        self.assertEqual(SessionMetadata.objects.get(thread_id="ordinary").project, project)

    def test_edit_project_updates_name_and_auto_pr_mode(self) -> None:
        project = _make_project()

        response = self.client.post(
            reverse("edit_project"),
            data={
                "project": str(project.pk),
                "name": "Renamed",
                "extra_system_prompt": "  Prefer project fixtures.  ",
                "auto_pr_mode": Project.AUTO_PR_ON,
                "auto_pull": "true",
            },
        )

        self.assertEqual(response.status_code, 302)
        project.refresh_from_db()
        self.assertEqual(project.name, "Renamed")
        self.assertEqual(project.extra_system_prompt, "Prefer project fixtures.")
        self.assertEqual(project.auto_pr_mode, Project.AUTO_PR_ON)
        self.assertTrue(project.auto_pull_enabled)

    def test_edit_project_rejects_invalid_posts(self) -> None:
        project = _make_project()

        for data, message in (
            (
                {
                    "project": "",
                    "name": "Renamed",
                    "auto_pr_mode": Project.AUTO_PR_ON,
                },
                "project is required",
            ),
            (
                {
                    "project": str(project.pk),
                    "name": "",
                    "auto_pr_mode": Project.AUTO_PR_ON,
                },
                "project name is required",
            ),
            (
                {
                    "project": str(project.pk),
                    "name": "Renamed",
                    "extra_system_prompt": "x" * (settings_cookies._EXTRA_SYSTEM_PROMPT_MAX_LEN + 1),
                    "auto_pr_mode": Project.AUTO_PR_ON,
                },
                "extra system prompt is too long",
            ),
            (
                {
                    "project": str(project.pk),
                    "name": "Renamed",
                    "auto_pr_mode": "maybe",
                },
                "invalid project auto-PR setting",
            ),
            (
                {
                    "project": str(project.pk),
                    "name": "Renamed",
                    "auto_pr_mode": Project.AUTO_PR_ON,
                    "auto_pull": "maybe",
                },
                "invalid project auto-pull setting",
            ),
        ):
            with self.subTest(message=message):
                response = self.client.post(reverse("edit_project"), data=data)
                self.assertContains(response, message, status_code=400)

    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.repos.discover_repos")
    def test_rejects_invalid_project_posts(self, mock_discover: MagicMock, mock_codex: MagicMock) -> None:
        mock_discover.return_value = [Path("/repo")]
        _setup_codex(mock_codex)

        for data, message in (
            ({"name": "", "repo_path": "/repo"}, "project name is required"),
            ({"name": "Hitch", "repo_path": "/etc"}, "repository must be a discovered repository"),
        ):
            with self.subTest(message=message):
                response = self.client.post(reverse("new_project"), data=data)
                self.assertContains(response, message, status_code=400)


class PrStageRefreshSchedulingTests(TestCase):
    @override
    def tearDown(self) -> None:
        # The threaded path adds to a module-level in-flight set; keep tests
        # isolated by clearing it.
        with session_stage_refresh._PR_STAGE_REFRESH_INFLIGHT_LOCK:
            session_stage_refresh._PR_STAGE_REFRESH_INFLIGHT.clear()

    @patch("hitch.main.sessions.session_stage_refresh._refresh_session_pr_stage")
    @patch("hitch.main.sessions.session_stage_refresh.threading.Thread")
    def test_schedule_spawns_one_thread_per_session_off_request(
        self, mock_thread: MagicMock, _mock_refresh: MagicMock
    ) -> None:
        with self.settings(TESTING=False):
            session_stage_refresh._schedule_pr_stage_refresh("sess-x")
            # A concurrent render for the same session does not spawn a duplicate.
            session_stage_refresh._schedule_pr_stage_refresh("sess-x")
        mock_thread.assert_called_once()
        mock_thread.return_value.start.assert_called_once()



class ArchiveUndoToastTests(TestCase):
    """Browser-level coverage for the per-row archive grace period and Undo.

    Regression guard: rapidly archiving several rows must keep every row
    independently undoable until its own 5s timer fires, rather than letting a
    later archive strand the earlier one with no working Undo.
    """

    def _seed_two_sessions(self) -> None:
        now = datetime.now(UTC)
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=now,
            is_complete=True,
        )
        for index in (1, 2):
            SessionMetadata.objects.create(
                thread_id=f"sess-{index}",
                cwd="/repo",
                codex_display_title=f"Session {index}",
                codex_name=f"Session {index}",
                codex_created_at=now,
                codex_updated_at=now - timedelta(minutes=index),
                codex_last_synced_at=now,
            )

    @patch("hitch.main.repos.discover_repos", return_value=[])
    @patch("hitch.main.views.common.Codex")
    def test_rapid_archive_keeps_every_row_undoable(self, mock_codex: MagicMock, _mock_discover: MagicMock) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = CodexError("thread list unavailable")
        self._seed_two_sessions()

        response = self.client.get(reverse("index"))
        self.assertEqual(response.status_code, 200)
        page_html = response.content.decode()

        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            self.skipTest(f"playwright unavailable: {exc}")

        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
            except PlaywrightError as exc:
                self.skipTest(f"playwright browser unavailable: {exc}")
            try:
                page = browser.new_page()
                page.set_content(page_html, wait_until="load")
                # Archive POSTs always succeed in this test; the 5s finalize
                # timers never fire within the test window.
                page.evaluate("() => { window.fetch = () => Promise.resolve({ ok: true }); }")
                self.assertEqual(
                    page.evaluate("() => document.querySelectorAll('[data-session-archive-form]').length"),
                    2,
                )
                # Archive both rows in quick succession.
                page.evaluate(
                    """
                    () => {
                        for (const form of document.querySelectorAll(
                            "[data-session-archive-form]")) {
                            form.requestSubmit();
                        }
                    }
                    """
                )
                page.wait_for_function("document.querySelectorAll('[data-session-row].pending-archive').length === 2")
                self.assertFalse(page.evaluate("() => document.querySelector('[data-archive-toast]').hidden"))

                undo = "() => document.querySelector('[data-archive-undo]').click()"
                # First Undo restores the most recently archived row; the toast
                # stays up because the other row's grace period is still open.
                page.evaluate(undo)
                page.wait_for_function("document.querySelectorAll('[data-session-row].pending-archive').length === 1")
                self.assertFalse(page.evaluate("() => document.querySelector('[data-archive-toast]').hidden"))
                # Second Undo restores the earlier row -- the case the single-slot
                # implementation dropped on the floor.
                page.evaluate(undo)
                page.wait_for_function("document.querySelectorAll('[data-session-row].pending-archive').length === 0")
                self.assertTrue(page.evaluate("() => document.querySelector('[data-archive-toast]').hidden"))
            finally:
                browser.close()

    @patch("hitch.main.repos.discover_repos", return_value=[])
    @patch("hitch.main.views.common.Codex")
    def test_undo_order_follows_archive_order_despite_post_race(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = CodexError("thread list unavailable")
        self._seed_two_sessions()

        response = self.client.get(reverse("index"))
        self.assertEqual(response.status_code, 200)
        page_html = response.content.decode()

        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            self.skipTest(f"playwright unavailable: {exc}")

        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
            except PlaywrightError as exc:
                self.skipTest(f"playwright browser unavailable: {exc}")
            try:
                page = browser.new_page()
                page.set_content(page_html, wait_until="load")
                # Hold every POST open so the test controls completion order:
                # archive POSTs are keyed by session id, undo POSTs queued FIFO.
                page.evaluate(
                    """
                    () => {
                        window.__archive = {};
                        window.__undo = [];
                        window.fetch = (url, opts) => new Promise((resolve) => {
                            const ok = () => resolve({ ok: true });
                            if (opts && opts.body && opts.body.includes("archived=true")) {
                                const m = String(url).match(/sessions\\/([^/]+)\\/archive/);
                                window.__archive[m[1]] = ok;
                            } else {
                                window.__undo.push(ok);
                            }
                        });
                    }
                    """
                )
                # User archives sess-1 first, then sess-2.
                for session_id in ("sess-1", "sess-2"):
                    page.evaluate(
                        "(id) => document.querySelector("
                        "`[data-session-archive-url*='${id}'] "
                        "[data-session-archive-form]`).requestSubmit()",
                        session_id,
                    )
                # The first-archived row's POST resolves LAST -- the race that
                # ordering by POST completion would get wrong.
                page.evaluate("() => window.__archive['sess-2']()")
                page.evaluate("() => window.__archive['sess-1']()")
                page.wait_for_function("document.querySelectorAll('[data-session-row].pending-archive').length === 2")

                undo = "() => document.querySelector('[data-archive-undo]').click()"
                resolve_undo = "() => window.__undo.shift()()"
                pending = (
                    "(id) => { const el = document.querySelector("
                    "`[data-session-archive-url*='${id}']`);"
                    " return !!el && el.classList.contains('pending-archive'); }"
                )
                restored = (
                    "(id) => { const el = document.querySelector("
                    "`[data-session-archive-url*='${id}']`);"
                    " return !!el && !el.classList.contains('pending-archive'); }"
                )
                # First Undo must restore the most recently archived row (sess-2),
                # not whichever POST happened to finish last.
                page.evaluate(undo)
                page.evaluate(resolve_undo)
                page.wait_for_function(restored, arg="sess-2")
                self.assertTrue(page.evaluate(pending, "sess-1"))
                # Second Undo restores the earlier row (sess-1).
                page.evaluate(undo)
                page.evaluate(resolve_undo)
                page.wait_for_function(restored, arg="sess-1")
            finally:
                browser.close()


class UnarchiveFailureTests(TestCase):
    """Browser-level coverage for a failed unarchive from the row menu.

    Regression guard: a rejected unarchive POST must surface feedback and not
    leave an unhandled promise rejection or a silently inconsistent row.
    """

    @patch("hitch.main.repos.discover_repos", return_value=[])
    @patch("hitch.main.views.common.Codex")
    def test_failed_unarchive_reports_error_and_keeps_row_archived(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        _seed_cookies(self.client, **{_SHOW_ARCHIVED_COOKIE: "true"})
        archived = _session(
            "arch-1",
            name="Archived one",
            path="/tmp/archived_sessions/arch.jsonl",
        )
        _setup_codex(mock_codex, threads=[], archived_threads=[archived])

        response = self.client.get(reverse("index"))
        self.assertEqual(response.status_code, 200)
        page_html = response.content.decode()

        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            self.skipTest(f"playwright unavailable: {exc}")

        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
            except PlaywrightError as exc:
                self.skipTest(f"playwright browser unavailable: {exc}")
            try:
                page = browser.new_page()
                page.set_content(page_html, wait_until="load")
                # The unarchive POST fails (non-OK response).
                page.evaluate("() => { window.fetch = () => Promise.resolve({ ok: false }); }")
                # Submitting an archived row's form takes the unarchive branch.
                page.evaluate(
                    "() => document.querySelector("
                    "\"[data-session-archive-url*='arch-1'] "
                    '[data-session-archive-form]").requestSubmit()'
                )
                page.wait_for_function(
                    "() => { const t = document.querySelector("
                    "'[data-archive-error-toast]');"
                    " return t && !t.hidden && document.querySelector("
                    "'[data-archive-error-text]').textContent"
                    ".includes('Couldn'); }"
                )
                # No successful-archive notice, and the row stays archived.
                self.assertTrue(page.evaluate("() => document.querySelector('[data-archive-toast]').hidden"))
                self.assertEqual(
                    page.evaluate(
                        "() => document.querySelector(\"[data-session-archive-url*='arch-1']\").dataset.sessionArchived"
                    ),
                    "true",
                )
            finally:
                browser.close()
class UsageTileAccessibilityTests(TestCase):
    """Only lifetime-stat tiles that actually have a chart may be interactive.

    Regression guard: a chartless tile must not render as an expandable button
    (role/tabindex/aria-expanded), since toggling it reveals nothing.
    """

    def _render(self, *, sessions_chart: bool, system_chart: bool) -> str:
        from django.template.loader import render_to_string

        chart = [
            {
                "date": "2025-01-02",
                "total": "10",
                "input": "5",
                "output": "3",
                "cached": "2",
                "input_percent": 50,
                "output_percent": 30,
                "cached_percent": 20,
            }
        ]
        lifetime_usage = {
            "total": {},
            "sessions": {
                "input": "5",
                "output": "3",
                "cached": "2",
                "chart": chart if sessions_chart else None,
                "chart_axis": [],
            },
            "system": {
                "input": "1",
                "output": "1",
                "cached": "0",
                "chart": chart if system_chart else None,
                "chart_axis": [],
            },
        }
        return render_to_string("_usage_sections.html", {"lifetime_usage": lifetime_usage})

    def test_both_tiles_interactive_when_both_charted(self) -> None:
        html = self._render(sessions_chart=True, system_chart=True)
        self.assertEqual(html.count('class="lifetime-stat" role="button"'), 2)
        self.assertNotIn('class="lifetime-stat">', html)


class SharedCsrfHelperTests(TestCase):
    """The shared window.hitch.csrfToken helper (_js_utils.html).

    Regression guard for the J5 dedup: the index page wires in the shared
    helper, which resolves the CSRF token (cookie first, hidden-form-input
    fallback). In the sandboxed test document `document.cookie` is unreadable,
    so this exercises the input-fallback branch -- index.html's original
    behavior, which must be preserved.
    """

    @patch("hitch.main.repos.discover_repos", return_value=[])
    @patch("hitch.main.views.common.Codex")
    def test_csrf_helper_falls_back_to_form_input(self, mock_codex: MagicMock, _mock_discover: MagicMock) -> None:
        _setup_codex(mock_codex, threads=[])
        html = self.client.get(reverse("index")).content.decode()
        self.assertIn("window.hitch.csrfToken", html)

        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            self.skipTest(f"playwright unavailable: {exc}")

        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
            except PlaywrightError as exc:
                self.skipTest(f"playwright browser unavailable: {exc}")
            try:
                page = browser.new_page()
                page.set_content(html, wait_until="load")
                result = page.evaluate(
                    """
                    () => {
                        const input = document.querySelector(
                            "input[name=csrfmiddlewaretoken]");
                        return {
                            token: window.hitch.csrfToken(),
                            input: input ? input.value : null,
                        };
                    }
                    """
                )
                self.assertTrue(result["input"])
                self.assertEqual(result["token"], result["input"])
            finally:
                browser.close()


class SharedPostFormHelperTests(TestCase):
    """The shared window.hitch.postForm helper (_js_utils.html).

    Regression guard for the J6 dedup: every form POST goes through one helper
    that attaches the CSRF token and form-encoded content type, and adds the
    X-Requested-With header only when opted in (some endpoints answer 204 vs
    redirect based on it, so it must not be sent unconditionally).
    """

    @patch("hitch.main.repos.discover_repos", return_value=[])
    @patch("hitch.main.views.common.Codex")
    def test_postform_headers_and_xhr_opt_in(self, mock_codex: MagicMock, _mock_discover: MagicMock) -> None:
        _setup_codex(mock_codex, threads=[])
        html = self.client.get(reverse("index")).content.decode()

        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            self.skipTest(f"playwright unavailable: {exc}")

        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
            except PlaywrightError as exc:
                self.skipTest(f"playwright browser unavailable: {exc}")
            try:
                page = browser.new_page()
                page.set_content(html, wait_until="load")
                result = page.evaluate(
                    """
                    async () => {
                        const calls = [];
                        window.fetch = (url, opts) => {
                            calls.push({ url, opts });
                            return Promise.resolve({ ok: true });
                        };
                        await window.hitch.postForm(
                            "/with-xhr", new URLSearchParams({ a: "1" }), { xhr: true });
                        await window.hitch.postForm("/no-xhr", "b=2");
                        return calls.map((c) => ({
                            url: c.url,
                            method: c.opts.method,
                            credentials: c.opts.credentials,
                            contentType: c.opts.headers["Content-Type"],
                            hasCsrf: "X-CSRFToken" in c.opts.headers,
                            xhr: c.opts.headers["X-Requested-With"] || null,
                        }));
                    }
                    """
                )
                self.assertEqual(len(result), 2)
                with_xhr, no_xhr = result
                for call in result:
                    self.assertEqual(call["method"], "POST")
                    self.assertEqual(call["credentials"], "same-origin")
                    self.assertEqual(call["contentType"], "application/x-www-form-urlencoded")
                    self.assertTrue(call["hasCsrf"])
                self.assertEqual(with_xhr["xhr"], "XMLHttpRequest")
                self.assertIsNone(no_xhr["xhr"])
            finally:
                browser.close()


class SharedTimeHelperTests(TestCase):
    """The shared window.hitch.relativeFromNow helper (_js_utils.html).

    Regression guard for the J4 dedup: index.html and _usage_scripts.html both
    rely on one relative-time formatter instead of their own copies.
    """

    @patch("hitch.main.repos.discover_repos", return_value=[])
    @patch("hitch.main.views.common.Codex")
    def test_relative_from_now(self, mock_codex: MagicMock, _mock_discover: MagicMock) -> None:
        _setup_codex(mock_codex, threads=[])
        html = self.client.get(reverse("index")).content.decode()
        self.assertIn("window.hitch.relativeFromNow", html)

        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            self.skipTest(f"playwright unavailable: {exc}")

        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
            except PlaywrightError as exc:
                self.skipTest(f"playwright browser unavailable: {exc}")
            try:
                page = browser.new_page()
                page.set_content(html, wait_until="load")
                results = page.evaluate(
                    """
                    () => ({
                        past: window.hitch.relativeFromNow(
                            new Date(Date.now() - 5 * 60 * 1000)),
                        future: window.hitch.relativeFromNow(
                            new Date(Date.now() + 2 * 3600 * 1000)),
                        recent: window.hitch.relativeFromNow(
                            new Date(Date.now() - 10 * 1000)),
                    })
                    """
                )
                self.assertIn("minute", results["past"])
                self.assertIn("hour", results["future"])
                self.assertEqual(results["recent"], "just now")
            finally:
                browser.close()
