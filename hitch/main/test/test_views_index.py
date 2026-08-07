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

from django.db import OperationalError, connection
from django.test import (
    TestCase,
)
from django.test.utils import CaptureQueriesContext
from django.urls import reverse
from django.utils import timezone
from openai_codex.errors import CodexError as CodexError
from openai_codex.errors import InvalidRequestError
from openai_codex.generated.v2_all import (
    GetAccountRateLimitsResponse,
    SortDirection,
    ThreadSortKey,
    ThreadSource,
)

from hitch.main import demo
from hitch.main.models import (
    ApprovalRequest,
    ArchivedSessionTokenUsage,
    CodexInstance,
    Project,
    ProposedSession,
    SessionIndexSyncState,
    SessionMetadata,
    SystemAgentRun,
    SystemWorkflow,
    UserInputRequest,
)
from hitch.main.runtime.rollout_state import _RolloutFileState
from hitch.main.sessions import (
    session_index,
    session_settings,
    session_stage,
    session_stage_refresh,
    settings_cookies,
    system_agent_summary,
    token_usage,
)
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
    _PR_PROMPT,
    _QA_PROMPT,
    _SELECTED_PROJECT_COOKIE,
    _SHOW_ARCHIVED_COOKIE,
    _SHOW_NO_PROJECT_SESSIONS_COOKIE,
    _USE_WORKTREES_COOKIE,
    _VISIBLE_SESSION_PROJECTS_COOKIE,
    _basic_session_rollout_lines,
    _cache_token_usage,
    _due_pr_monitor_state,
    _make_rollout,
    _merged_pr_monitor_observation,
    _seed_usage_metadata,
    _session,
    _token_count_line,
)
from hitch.main.views import common as common_views
from hitch.main.views import session_list
from hitch.main.views import settings as settings_views
from hitch.main.workflows import agent_io, gh_cli, pr_stage_refresh_state, system_agents


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
    def test_index_binds_primary_nav_before_collapsing_fallback(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        _setup_codex(mock_codex, threads=[])

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        body = response.content.decode()
        nav_start = body.index('<nav class="primary-nav"')
        nav_end = body.index("</nav>", nav_start)
        initializer_start = body.index("window.closePrimaryNavMenu = closeNavMenu;")
        self.assertGreater(initializer_start, nav_end)
        self.assertLess(initializer_start, body.index("<main>"))
        # The "js" class is bootstrapped exactly once; the nav initializer no
        # longer re-adds it (it runs after _js_class_bootstrap.html).
        self.assertEqual(body.count('classList.add("js")'), 1)
        self.assertContains(response, ".primary-nav.primary-nav-js .primary-nav-panel")
        self.assertNotContains(response, ":is(.js .primary-nav")

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_cached_session_list_does_not_call_thread_list(
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
            thread_id="cached",
            cwd="/repo",
            codex_display_title="Cached session",
            codex_name="Cached session",
            codex_created_at=now,
            codex_updated_at=now,
            codex_last_synced_at=now,
        )

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        # The live session list must not be served from the browser bfcache so a
        # Back navigation reflects sessions archived/renamed elsewhere.
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")
        self.assertContains(response, "Cached session")
        mock_codex.assert_not_called()
        client.thread_list.assert_not_called()

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_cached_session_list_shows_derived_stage_badge(
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
                            {
                                "url": "https://github.com/cberner/hitch/pull/94",
                                "state": "closed",
                                "merged": False,
                            }
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
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=now,
            is_complete=True,
        )
        metadata = SessionMetadata.objects.create(
            thread_id="closed-pr",
            cwd="/repo",
            codex_display_title="Closed PR",
            codex_preview="Open a PR",
            codex_path=str(rollout_path),
            codex_created_at=now,
            codex_updated_at=now,
            codex_last_synced_at=now,
        )

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Closed PR")
        self.assertContains(
            response, '<span class="stage-badge" data-tone="done">Done: Closed</span>'
        )
        metadata.refresh_from_db()
        self.assertEqual(metadata.derived_stage, "done_closed")
        mock_codex.assert_not_called()
        client.thread_list.assert_not_called()

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_cached_session_list_shows_pr_number_in_cached_pr_badge(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = CodexError("thread list unavailable")
        mock_discover.return_value = []
        now = datetime.now(UTC)
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
            codex_path=str(rollout_path),
            codex_created_at=now,
            codex_updated_at=now,
            codex_last_synced_at=now,
            derived_stage="pr",
            derived_stage_source_mtime_ns=rollout_path.stat().st_mtime_ns,
        )

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Cached PR")
        self.assertContains(
            response,
            '<span class="stage-badge" data-tone="active">PR #94</span>',
        )
        mock_codex.assert_not_called()
        client.thread_list.assert_not_called()

    @patch("hitch.main.workflows.pr_qa._gh_pr_view")
    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_cached_session_list_refreshes_cached_pr_stage_to_done_merged(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_gh_pr_view: MagicMock,
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = CodexError("thread list unavailable")
        mock_discover.return_value = []
        now = datetime.now(UTC)
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
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=now,
            is_complete=True,
        )
        metadata = SessionMetadata.objects.create(
            thread_id="cached-pr-merged",
            cwd=str(rollout_path.parent),
            codex_display_title="Cached PR merged",
            codex_preview="Open a PR",
            codex_path=str(rollout_path),
            codex_created_at=now,
            codex_updated_at=now,
            codex_last_synced_at=now,
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

        # First load serves the cached (open) PR badge with the refreshing
        # highlight and refreshes off-request, persisting the terminal stage.
        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Cached PR merged")
        self.assertContains(
            response,
            '<span class="stage-badge" data-tone="active" data-refreshing="true">PR #94</span>',
        )
        metadata.refresh_from_db()
        self.assertEqual(metadata.derived_stage, "done_merged")
        self.assertIsNotNone(metadata.derived_stage_pr_refresh_attempted_at)
        mock_gh_pr_view.assert_called_once()
        mock_codex.assert_not_called()
        client.thread_list.assert_not_called()

        # The next load reads the refreshed terminal stage from cache, no gh.
        response = self.client.get(reverse("index"))
        self.assertContains(
            response, '<span class="stage-badge" data-tone="done">Done: Merged</span>'
        )
        mock_gh_pr_view.assert_called_once()

    @patch("hitch.main.workflows.pr_qa._pr_monitor_observation_from_gh")
    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_cached_session_list_refreshes_due_pr_monitor_backoff_to_done_merged(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_observe: MagicMock,
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = CodexError("thread list unavailable")
        mock_discover.return_value = []
        now = datetime.now(UTC)
        pr_url = "https://github.com/cberner/hitch/pull/60"
        repo = "cberner/hitch"
        pr_number = 60
        rollout_path = _make_rollout(
            self,
            _basic_session_rollout_lines("Open a PR", "Opened."),
        )
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=now,
            is_complete=True,
        )
        SessionMetadata.objects.create(
            thread_id="monitor-pr-merged-list",
            cwd=str(rollout_path.parent),
            codex_display_title="Monitor PR merged list",
            codex_preview="Open a PR",
            codex_path=str(rollout_path),
            codex_created_at=now,
            codex_updated_at=now,
            codex_last_synced_at=now,
        )
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="monitor-pr-merged-list",
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

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Monitor PR merged list")
        self.assertContains(
            response,
            '<span class="stage-badge" data-tone="active" data-refreshing="true">PR #60</span>',
        )
        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(workflow.step, system_agents.STEP_PR_CLOSED)
        self.assertTrue(workflow.state["pr_handoff"]["merged"])
        mock_observe.assert_called_once()
        mock_codex.assert_not_called()
        client.thread_list.assert_not_called()

        response = self.client.get(reverse("index"))
        self.assertContains(
            response, '<span class="stage-badge" data-tone="done">Done: Merged</span>'
        )
        mock_observe.assert_called_once()

    @patch("hitch.main.sessions.session_stage_refresh._schedule_pr_stage_refresh")
    @patch("hitch.main.workflows.pr_qa.pr_snapshot_stage_refresh_due", return_value=True)
    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_session_list_skips_caching_stale_pr_stage_for_budget_deferred_row(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        _mock_due: MagicMock,
        mock_schedule: MagicMock,
    ) -> None:
        # Two PR rows are due for a gh refresh but the per-render budget allows
        # only one. The deferred row's snapshot is known-stale, so its derived
        # terminal stage must not be persisted to the mtime-keyed cache: the
        # cached fast path only rechecks PR stages, so a stale Done badge would
        # otherwise stick without ever scheduling another refresh.
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = CodexError("thread list unavailable")
        mock_discover.return_value = []
        now = datetime.now(UTC)
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=now,
            is_complete=True,
        )
        pr_url = "https://github.com/cberner/hitch/pull/94"
        rows = []
        for index in range(2):
            rollout_path = _make_rollout(
                self,
                [
                    _rollout_line(
                        "event_msg",
                        {
                            "type": "user_message",
                            "message": system_agents.PR_SLASH_PROMPT,
                        },
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
            rows.append(
                SessionMetadata.objects.create(
                    thread_id=f"pr-row-{index}",
                    cwd=str(rollout_path.parent),
                    codex_display_title=f"PR row {index}",
                    codex_preview="Open a PR",
                    codex_path=str(rollout_path),
                    codex_created_at=now,
                    codex_updated_at=now - timedelta(minutes=index),
                    codex_last_synced_at=now,
                )
            )

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        # Budget is 1: exactly one row schedules an off-request refresh.
        self.assertEqual(mock_schedule.call_count, 1)
        # Neither row caches its stale terminal stage while a refresh is due.
        for metadata in rows:
            metadata.refresh_from_db()
            self.assertEqual(metadata.derived_stage, "")
            self.assertEqual(metadata.derived_stage_source_mtime_ns, 0)
        mock_codex.assert_not_called()

    @patch("hitch.main.workflows.pr_qa._gh_pr_view")
    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_cached_session_list_refreshes_uncached_pr_snapshot_to_done_merged(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_gh_pr_view: MagicMock,
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = CodexError("thread list unavailable")
        mock_discover.return_value = []
        now = datetime.now(UTC)
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
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=now,
            is_complete=True,
        )
        metadata = SessionMetadata.objects.create(
            thread_id="uncached-pr-merged",
            cwd=str(rollout_path.parent),
            codex_display_title="Uncached PR merged",
            codex_preview="Open a PR",
            codex_path=str(rollout_path),
            codex_created_at=now,
            codex_updated_at=now,
            codex_last_synced_at=now,
        )
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="uncached-pr-merged",
            cwd=str(rollout_path.parent),
            status=SystemWorkflow.STATUS_MAX_ITERATIONS_REACHED,
            step=system_agents.STEP_MAX_ITERATIONS_REACHED,
            state={
                "pr_handoff": {
                    "url": pr_url,
                    "repository_full_name": "cberner/hitch",
                    "pr_number": 94,
                    "state": "open",
                },
                "hitch_pr_handoff": {
                    "url": pr_url,
                    "repository_full_name": "cberner/hitch",
                    "pr_number": 94,
                },
            },
        )
        SystemWorkflow.objects.filter(pk=workflow.pk).update(
            updated_at=now - timedelta(minutes=5)
        )
        mock_gh_pr_view.return_value = {
            "url": pr_url,
            "repository_full_name": "cberner/hitch",
            "pr_number": 94,
            "state": "closed",
            "merged": True,
            "merged_at": "2026-06-02T08:26:51Z",
        }

        # First load serves the open PR badge with the refreshing highlight and
        # refreshes the snapshot off-request; the stale workflow is stripped by
        # the main-lifecycle check so the refresh lands on the stage cache only.
        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Uncached PR merged")
        self.assertContains(
            response,
            '<span class="stage-badge" data-tone="active" data-refreshing="true">PR #94</span>',
        )
        metadata.refresh_from_db()
        self.assertEqual(metadata.derived_stage, "done_merged")
        self.assertIsNotNone(metadata.derived_stage_pr_refresh_attempted_at)
        workflow.refresh_from_db()
        self.assertEqual(workflow.step, system_agents.STEP_MAX_ITERATIONS_REACHED)
        mock_gh_pr_view.assert_called_once()
        mock_codex.assert_not_called()
        client.thread_list.assert_not_called()

        # The next load reads the refreshed terminal stage from cache, no gh.
        response = self.client.get(reverse("index"))
        self.assertContains(
            response, '<span class="stage-badge" data-tone="done">Done: Merged</span>'
        )
        mock_gh_pr_view.assert_called_once()

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_cached_session_list_omits_stale_pr_number_for_new_workflow(
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
        SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="new-pr-workflow",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_PR_PROMPT_RUNNING,
            state={},
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
    def test_cached_session_list_ignores_locked_stage_cache_update(
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
                    {"type": "user_message", "message": "Implement the feature"},
                )
            ],
        )
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=now,
            is_complete=True,
        )
        metadata = SessionMetadata.objects.create(
            thread_id="cached",
            cwd="/repo",
            codex_display_title="Cached session",
            codex_preview="Implement the feature",
            codex_path=str(rollout_path),
            codex_created_at=now,
            codex_updated_at=now,
            codex_last_synced_at=now,
        )

        with patch(
            "hitch.main.workflows.pr_stage._update_cached_stage",
            side_effect=OperationalError("database is locked"),
        ) as update_stage:
            response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Cached session")
        update_stage.assert_called_once()
        metadata.refresh_from_db()
        self.assertEqual(metadata.derived_stage, "")
        mock_codex.assert_not_called()
        client.thread_list.assert_not_called()

    @patch("hitch.main.runtime.codex_pool.worker_is_alive", return_value=True)
    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_cached_session_list_active_instance_overrides_terminal_stage(
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
                            {
                                "url": "https://github.com/cberner/hitch/pull/94",
                                "state": "closed",
                                "merged": False,
                            }
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
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=now,
            is_complete=True,
        )
        SessionMetadata.objects.create(
            thread_id="active-after-pr",
            cwd="/repo",
            codex_display_title="Active after PR",
            codex_preview="Follow up",
            codex_path=str(rollout_path),
            codex_created_at=now,
            codex_updated_at=now,
            codex_last_synced_at=now,
            derived_stage="done_closed",
            derived_stage_source_mtime_ns=rollout_path.stat().st_mtime_ns,
        )
        CodexInstance.objects.create(
            pid=os.getpid(),
            thread_id="active-after-pr",
            cwd="/repo",
            prompt="Follow up",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_RUNNING,
        )

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Active after PR")
        self.assertContains(
            response,
            '<span class="stage-badge" data-tone="active" data-executing="true">Implementation</span>',
        )
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
    def test_cached_session_list_flags_pending_spec_critic_input(
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
                    {"type": "user_message", "message": "Build this feature."},
                ),
            ],
        )
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=now,
            is_complete=True,
        )
        metadata = SessionMetadata.objects.create(
            thread_id="needs-input",
            cwd="/repo",
            codex_display_title="Needs input",
            codex_preview="Build this feature.",
            codex_path=str(rollout_path),
            codex_created_at=now,
            codex_updated_at=now,
            codex_last_synced_at=now,
            derived_stage="implementation",
            derived_stage_source_mtime_ns=rollout_path.stat().st_mtime_ns,
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.SPEC_CRITIC_WORKFLOW_KIND,
            main_thread_id="needs-input",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_SPEC_CRITIC_CLARIFYING,
        )
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="spec-hidden",
            cwd="/repo",
            prompt="Clarify",
            events_path="/tmp/spec-events.jsonl",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=agent_io.SPEC_RISK_AGENT_KIND,
            display_author=system_agents.SPEC_CRITIC_DISPLAY_AUTHOR,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=agent_io.SPEC_RISK_AGENT_KIND,
            thread_id="spec-hidden",
            instance=instance,
            status=SystemAgentRun.STATUS_COMPLETED,
        )
        UserInputRequest.objects.create(
            instance=instance,
            method=system_agents.SPEC_CRITIC_CLARIFICATION_METHOD,
            params={"questions": []},
        )

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Needs input")
        self.assertContains(
            response,
            '<span class="stage-badge" data-tone="warning">Awaiting Input</span>',
        )
        metadata.refresh_from_db()
        self.assertEqual(metadata.derived_stage, "implementation")
        mock_codex.assert_not_called()
        client.thread_list.assert_not_called()

    @patch("hitch.main.runtime.codex_pool.worker_is_alive", return_value=True)
    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_active_instance_stage_not_cached_after_worker_stops(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        _worker_is_alive: MagicMock,
    ) -> None:
        # An active worker forces the Implementation stage. That stage is not a
        # function of the rollout file, so it must not be written to the
        # mtime-keyed stage cache: once the worker stops without rewriting the
        # rollout (interrupted/aborted/no-op turn), the index must recompute the
        # real terminal stage rather than resurrect the stale active badge.
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = CodexError("thread list unavailable")
        mock_discover.return_value = []
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
                            {
                                "url": "https://github.com/cberner/hitch/pull/94",
                                "state": "closed",
                                "merged": False,
                            }
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
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=now,
            is_complete=True,
        )
        metadata = SessionMetadata.objects.create(
            thread_id="active-then-idle",
            cwd="/repo",
            codex_display_title="Active then idle",
            codex_preview="Follow up",
            codex_path=str(rollout_path),
            codex_created_at=now,
            codex_updated_at=now,
            codex_last_synced_at=now,
        )
        instance = CodexInstance.objects.create(
            pid=os.getpid(),
            thread_id="active-then-idle",
            cwd="/repo",
            prompt="Follow up",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_RUNNING,
        )
        mtime_before = rollout_path.stat().st_mtime_ns

        active_response = self.client.get(reverse("index"))

        self.assertEqual(active_response.status_code, 200)
        self.assertContains(
            active_response,
            '<span class="stage-badge" data-tone="active" data-executing="true">Implementation</span>',
        )

        # Worker finishes without touching the rollout, so its mtime is
        # unchanged and the terminal "Done: Closed" stage is the truth.
        instance.delete()
        self.assertEqual(rollout_path.stat().st_mtime_ns, mtime_before)

        idle_response = self.client.get(reverse("index"))

        self.assertEqual(idle_response.status_code, 200)
        self.assertContains(
            idle_response,
            '<span class="stage-badge" data-tone="done">Done: Closed</span>',
        )
        self.assertNotContains(
            idle_response,
            '<span class="stage-badge" data-tone="active" data-executing="true">Implementation</span>',
        )
        metadata.refresh_from_db()
        self.assertEqual(metadata.derived_stage, "done_closed")
        mock_codex.assert_not_called()
        client.thread_list.assert_not_called()

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_cached_session_list_ignores_stage_cache_without_rollout(
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
        metadata = SessionMetadata.objects.create(
            thread_id="cached-active",
            cwd="/repo",
            codex_display_title="Cached active session",
            codex_preview="Implement the change",
            codex_created_at=now,
            codex_updated_at=now,
            codex_last_synced_at=now,
            derived_stage="new",
            derived_stage_source_mtime_ns=0,
        )

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Cached active session")
        self.assertContains(
            response,
            '<span class="stage-badge" data-tone="idle">Implementation</span>',
        )
        self.assertNotContains(
            response, '<span class="stage-badge" data-tone="default">New</span>'
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

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_cached_session_list_ignores_stale_completed_pr_workflow(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = CodexError("thread list unavailable")
        mock_discover.return_value = []
        now = datetime.now(UTC)
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
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=now,
            is_complete=True,
        )
        SessionMetadata.objects.create(
            thread_id="stale-workflow",
            cwd="/repo",
            codex_display_title="Stale workflow",
            codex_preview="Next",
            codex_path=str(rollout_path),
            codex_created_at=now,
            codex_updated_at=now,
            codex_last_synced_at=now,
        )
        SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="stale-workflow",
            cwd="/repo",
            status=SystemWorkflow.STATUS_COMPLETED,
            step=system_agents.STEP_PR_READY,
            state={
                "pr_handoff": {"url": pr_url, "state": "closed", "merged": False}
            },
        )

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Stale workflow")
        self.assertContains(
            response,
            '<span class="stage-badge" data-tone="idle">Implementation</span>',
        )
        self.assertNotContains(response, "Done: Closed")
        mock_codex.assert_not_called()
        client.thread_list.assert_not_called()

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_cached_session_list_uses_terminal_cache_after_stale_pr_workflow(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = CodexError("thread list unavailable")
        mock_discover.return_value = []
        now = datetime.now(UTC)
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
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=now,
            is_complete=True,
        )
        SessionMetadata.objects.create(
            thread_id="terminal-cache-stale-workflow",
            cwd="/repo",
            codex_display_title="Terminal cache stale workflow",
            codex_preview="Open a PR",
            codex_path=str(rollout_path),
            codex_created_at=now,
            codex_updated_at=now,
            codex_last_synced_at=now,
            derived_stage="done_merged",
            derived_stage_source_mtime_ns=rollout_path.stat().st_mtime_ns,
        )
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="terminal-cache-stale-workflow",
            cwd="/repo",
            status=SystemWorkflow.STATUS_COMPLETED,
            step=system_agents.STEP_PR_READY,
            state={"pr_handoff": {"url": pr_url, "state": "open"}},
        )
        SystemWorkflow.objects.filter(pk=workflow.pk).update(
            updated_at=now - timedelta(minutes=5)
        )

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Terminal cache stale workflow")
        self.assertContains(
            response, '<span class="stage-badge" data-tone="done">Done: Merged</span>'
        )
        self.assertNotContains(response, "PR #98")
        mock_codex.assert_not_called()
        client.thread_list.assert_not_called()

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_cached_session_list_running_pr_workflow_without_handoff_ignores_old_pr(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = CodexError("thread list unavailable")
        mock_discover.return_value = []
        now = datetime.now(UTC)
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
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=now,
            is_complete=True,
        )
        SessionMetadata.objects.create(
            thread_id="running-workflow-no-handoff",
            cwd="/repo",
            codex_display_title="Running workflow no handoff",
            codex_preview="Open a PR",
            codex_path=str(rollout_path),
            codex_created_at=now,
            codex_updated_at=now,
            codex_last_synced_at=now,
        )
        SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="running-workflow-no-handoff",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_QA_RUNNING,
            state={},
        )

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Running workflow no handoff")
        self.assertContains(
            response,
            '<span class="stage-badge" data-tone="active">QA</span>',
        )
        self.assertNotContains(response, "Done: Merged")
        mock_codex.assert_not_called()
        client.thread_list.assert_not_called()

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_cached_session_list_running_pr_workflow_uses_terminal_handoff_pr(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = CodexError("thread list unavailable")
        mock_discover.return_value = []
        now = datetime.now(UTC)
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
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=now,
            is_complete=True,
        )
        SessionMetadata.objects.create(
            thread_id="running-workflow-terminal-pr",
            cwd="/repo",
            codex_display_title="Running workflow terminal PR",
            codex_preview="Open a PR",
            codex_path=str(rollout_path),
            codex_created_at=now,
            codex_updated_at=now,
            codex_last_synced_at=now,
        )
        SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="running-workflow-terminal-pr",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_QA_RUNNING,
            state={"pr_handoff": {"url": pr_url, "state": "open"}},
        )

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Running workflow terminal PR")
        self.assertContains(
            response,
            '<span class="stage-badge" data-tone="done">Done: Merged</span>',
        )
        self.assertNotContains(
            response,
            '<span class="stage-badge" data-tone="active">QA</span>',
        )
        mock_codex.assert_not_called()
        client.thread_list.assert_not_called()

    @patch("hitch.main.workflows.pr_qa._gh_pr_view")
    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_cached_session_list_refreshes_ready_pr_to_done_merged(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_gh_pr_view: MagicMock,
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
        SessionMetadata.objects.create(
            thread_id="ready-pr-merged-list",
            cwd=str(rollout_path.parent),
            codex_display_title="Ready PR merged list",
            codex_preview="Fix database locks",
            codex_path=str(rollout_path),
            codex_created_at=now,
            codex_updated_at=now,
            codex_last_synced_at=now,
        )
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="ready-pr-merged-list",
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

        # First load serves the open PR badge with the refreshing highlight and
        # refreshes off-request, persisting the terminal stage on the workflow.
        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Ready PR merged list")
        self.assertContains(
            response,
            '<span class="stage-badge" data-tone="active" data-refreshing="true">PR #344</span>',
        )
        workflow.refresh_from_db()
        self.assertEqual(workflow.step, system_agents.STEP_PR_CLOSED)
        self.assertTrue(workflow.state["pr_handoff"]["merged"])
        mock_gh_pr_view.assert_called_once()
        mock_codex.assert_not_called()
        client.thread_list.assert_not_called()

        # The next load derives the terminal stage from the closed workflow, no
        # gh call (the same PR is debounced).
        response = self.client.get(reverse("index"))
        self.assertContains(
            response, '<span class="stage-badge" data-tone="done">Done: Merged</span>'
        )
        mock_gh_pr_view.assert_called_once()

    @patch("hitch.main.workflows.pr_qa._gh_pr_view")
    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_cached_session_list_caps_ready_pr_refreshes(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_gh_pr_view: MagicMock,
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
        for index in range(2):
            pr_number = 400 + index
            pr_url = f"https://github.com/cberner/hitch/pull/{pr_number}"
            rollout_path = _make_rollout(
                self,
                [
                    _rollout_line(
                        "event_msg",
                        {
                            "type": "user_message",
                            "message": f"Fix database locks {index}",
                        },
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
            SessionMetadata.objects.create(
                thread_id=f"ready-pr-refresh-cap-{index}",
                cwd=str(rollout_path.parent),
                codex_display_title=f"Ready PR refresh cap {index}",
                codex_preview="Fix database locks",
                codex_path=str(rollout_path),
                codex_created_at=now - timedelta(seconds=index),
                codex_updated_at=now - timedelta(seconds=index),
                codex_last_synced_at=now,
            )
            SystemWorkflow.objects.create(
                kind=SystemWorkflow.KIND_PR_QA,
                main_thread_id=f"ready-pr-refresh-cap-{index}",
                cwd=str(rollout_path.parent),
                status=SystemWorkflow.STATUS_COMPLETED,
                step=system_agents.STEP_PR_READY,
                state={
                    "pr_handoff": {
                        "url": pr_url,
                        "repository_full_name": "cberner/hitch",
                        "pr_number": pr_number,
                        "state": "open",
                    },
                },
            )

        def merged_pr_for_selector(
            _workflow: SystemWorkflow,
            *,
            selector: str | None = None,
            source_tool: str,
            timeout_seconds: int,
        ) -> dict[str, object]:
            self.assertEqual(source_tool, "gh_pr_stage_refresh")
            self.assertEqual(
                timeout_seconds, system_agents._PR_STAGE_REFRESH_TIMEOUT_SECONDS
            )
            self.assertIsNotNone(selector)
            pr_number = int(str(selector).rsplit("/", 1)[1])
            return {
                "url": str(selector),
                "repository_full_name": "cberner/hitch",
                "pr_number": pr_number,
                "state": "closed",
                "merged": True,
                "merged_at": "2026-06-02T08:26:51Z",
            }

        mock_gh_pr_view.side_effect = merged_pr_for_selector

        # Both PR stages are due, but a single render schedules at most one
        # off-request refresh, so only one gh call happens and exactly one
        # workflow advances to its terminal stage this render.
        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Ready PR refresh cap 0")
        self.assertContains(response, "Ready PR refresh cap 1")
        self.assertContains(response, 'data-refreshing="true"')
        steps = list(
            SystemWorkflow.objects.order_by("main_thread_id").values_list(
                "step", flat=True
            )
        )
        self.assertEqual(steps.count(system_agents.STEP_PR_CLOSED), 1)
        self.assertEqual(steps.count(system_agents.STEP_PR_READY), 1)
        mock_gh_pr_view.assert_called_once()
        mock_codex.assert_not_called()
        client.thread_list.assert_not_called()

        # A second render refreshes the remaining due PR (one more gh call).
        self.client.get(reverse("index"))
        steps = list(
            SystemWorkflow.objects.values_list("step", flat=True)
        )
        self.assertEqual(steps.count(system_agents.STEP_PR_CLOSED), 2)
        self.assertEqual(mock_gh_pr_view.call_count, 2)

    @patch("hitch.main.workflows.system_agents.logger")
    @patch("hitch.main.workflows.pr_qa._gh_pr_view")
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
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="ready-pr-refresh-backoff",
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
        workflow.refresh_from_db()
        self.assertIn(pr_stage_refresh_state._PR_STAGE_REFRESH_STATE_KEY, workflow.state)
        self.assertEqual(workflow.step, system_agents.STEP_PR_READY)
        mock_gh_pr_view.assert_called_once()
        mock_logger.exception.assert_called_once()
        mock_codex.assert_not_called()
        client.thread_list.assert_not_called()

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_cached_session_list_prefers_newer_main_pr_over_stale_workflow(
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
                        "content": [{"type": "output_text", "text": "Opened."}],
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
            thread_id="newer-main-pr",
            cwd="/repo",
            codex_display_title="Newer main PR",
            codex_preview="Open a PR",
            codex_path=str(rollout_path),
            codex_created_at=now,
            codex_updated_at=now,
            codex_last_synced_at=now,
        )
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="newer-main-pr",
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

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Newer main PR")
        self.assertContains(
            response,
            '<span class="stage-badge" data-tone="active">PR #94</span>',
        )
        self.assertNotContains(response, "Done: Closed")
        mock_codex.assert_not_called()
        client.thread_list.assert_not_called()

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_cached_session_list_prefers_newer_main_pr_state_over_workflow(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = CodexError("thread list unavailable")
        mock_discover.return_value = []
        now = datetime.now(UTC)
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
                        "content": [{"type": "output_text", "text": "Reopened."}],
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
            thread_id="newer-main-pr-state",
            cwd="/repo",
            codex_display_title="Newer main PR state",
            codex_preview="Open a PR",
            codex_path=str(rollout_path),
            codex_created_at=now,
            codex_updated_at=now,
            codex_last_synced_at=now,
        )
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="newer-main-pr-state",
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

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Newer main PR state")
        self.assertContains(
            response,
            '<span class="stage-badge" data-tone="active">PR #94</span>',
        )
        self.assertNotContains(response, "Done: Closed")
        mock_codex.assert_not_called()
        client.thread_list.assert_not_called()

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_cached_session_list_ignores_workflow_only_stale_pr_handoff(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = CodexError("thread list unavailable")
        mock_discover.return_value = []
        now = datetime.now(UTC)
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
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=now,
            is_complete=True,
        )
        SessionMetadata.objects.create(
            thread_id="workflow-only-stale",
            cwd="/repo",
            codex_display_title="Workflow-only stale",
            codex_preview="Next",
            codex_path=str(rollout_path),
            codex_created_at=now,
            codex_updated_at=now,
            codex_last_synced_at=now,
        )
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="workflow-only-stale",
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

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Workflow-only stale")
        self.assertContains(
            response,
            '<span class="stage-badge" data-tone="idle">Implementation</span>',
        )
        self.assertNotContains(response, "Done: Closed")
        mock_codex.assert_not_called()
        client.thread_list.assert_not_called()

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_cached_session_list_keeps_server_created_pr_handoff(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = CodexError("thread list unavailable")
        mock_discover.return_value = []
        now = datetime.now(UTC)
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
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=now,
            is_complete=True,
        )
        SessionMetadata.objects.create(
            thread_id="server-created-pr-list",
            cwd="/repo",
            codex_display_title="Server-created PR",
            codex_preview="Open a PR",
            codex_path=str(rollout_path),
            codex_created_at=now,
            codex_updated_at=now,
            codex_last_synced_at=now,
        )
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="server-created-pr-list",
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

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Server-created PR")
        self.assertContains(
            response,
            '<span class="stage-badge" data-tone="active">PR #100</span>',
        )
        mock_codex.assert_not_called()
        client.thread_list.assert_not_called()

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_cached_hidden_system_flag_drives_main_and_system_lists(
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
            thread_id="visible",
            cwd="/repo",
            codex_display_title="Visible session",
            codex_created_at=now,
            codex_updated_at=now,
            codex_last_synced_at=now,
        )
        SessionMetadata.objects.create(
            thread_id="legacy-system",
            cwd="/repo",
            codex_display_title="Legacy system",
            codex_created_at=now,
            codex_updated_at=now,
            codex_last_synced_at=now,
            is_hidden_system_session=True,
        )

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Visible session")
        self.assertNotContains(response, "Legacy system")
        client.thread_list.assert_not_called()

        system_response = self.client.get(reverse("system_sessions"))

        self.assertEqual(system_response.status_code, 200)
        self.assertNotContains(system_response, "Visible session")
        self.assertContains(system_response, "Legacy system")
        self.assertContains(system_response, "Hitch system")
        self.assertContains(system_response, "untracked")

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_system_sessions_demo_upsert_keeps_main_session_visible(
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
        workflow = SystemWorkflow.objects.create(
            kind=demo.DEMO_WORKFLOW_KIND,
            main_thread_id="demo-thread",
            cwd="/repo",
            status=SystemWorkflow.STATUS_FAILED,
        )
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="demo-thread",
            cwd="/repo",
            prompt="Start an interactive web demo",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=demo.DEMO_AGENT_KIND,
            display_author=demo.DEMO_DISPLAY_AUTHOR,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=demo.DEMO_AGENT_KIND,
            thread_id="demo-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_FAILED,
        )

        system_response = self.client.get(reverse("system_sessions"))

        self.assertEqual(system_response.status_code, 200)
        self.assertContains(system_response, demo.DEMO_DISPLAY_AUTHOR)
        metadata = SessionMetadata.objects.get(thread_id="demo-thread")
        self.assertFalse(metadata.is_hidden_system_session)

        index_response = self.client.get(reverse("index"))

        self.assertEqual(index_response.status_code, 200)
        self.assertContains(index_response, demo.DEMO_DISPLAY_AUTHOR)
        client.thread_list.assert_not_called()

    def test_update_cached_name_preserves_activity_timestamp(self) -> None:
        old_updated_at = datetime.fromtimestamp(1000, UTC)
        SessionMetadata.objects.create(
            thread_id="old-session",
            cwd="/repo",
            codex_display_title="Old session",
            codex_name="Old session",
            codex_created_at=old_updated_at,
            codex_updated_at=old_updated_at,
            codex_last_synced_at=old_updated_at,
        )

        session_index.update_cached_name("old-session", "Renamed session")

        metadata = SessionMetadata.objects.get(thread_id="old-session")
        self.assertEqual(metadata.codex_name, "Renamed session")
        self.assertEqual(metadata.codex_display_title, "Renamed session")
        self.assertEqual(metadata.codex_updated_at, old_updated_at)
        self.assertIsNotNone(metadata.codex_last_synced_at)
        assert metadata.codex_last_synced_at is not None
        self.assertGreater(metadata.codex_last_synced_at, old_updated_at)

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_cached_session_order_uses_local_qa_activity_when_hidden_row_is_stale(
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
            thread_id="main-thread",
            cwd="/repo",
            codex_display_title="Main session",
            codex_created_at=datetime.fromtimestamp(900, UTC),
            codex_updated_at=datetime.fromtimestamp(900, UTC),
            codex_last_synced_at=now,
        )
        SessionMetadata.objects.create(
            thread_id="other-thread",
            cwd="/repo",
            codex_display_title="Other session",
            codex_created_at=datetime.fromtimestamp(1500, UTC),
            codex_updated_at=datetime.fromtimestamp(1500, UTC),
            codex_last_synced_at=now,
        )
        SessionMetadata.objects.create(
            thread_id="qa-thread",
            cwd="/repo",
            codex_display_title="Hidden QA",
            codex_created_at=datetime.fromtimestamp(1000, UTC),
            codex_updated_at=datetime.fromtimestamp(1000, UTC),
            codex_last_synced_at=datetime.fromtimestamp(1000, UTC),
        )
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
        )
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="qa-thread",
            cwd="/repo",
            prompt="qa",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            thread_id="qa-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_COMPLETED,
        )
        run_updated_at = datetime.fromtimestamp(2000, UTC)
        SystemAgentRun.objects.filter(pk=run.pk).update(updated_at=run_updated_at)
        SystemWorkflow.objects.filter(pk=workflow.pk).update(updated_at=run_updated_at)

        response = self.client.get(reverse("index"))
        body = response.content.decode()
        sessions_context = cast(list[dict[str, Any]], response.context["sessions"])

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [session["id"] for session in sessions_context],
            ["main-thread", "other-thread"],
        )
        self.assertEqual(sessions_context[0]["updated_at"], 2000)
        self.assertLess(body.index("Main session"), body.index("Other session"))
        self.assertNotContains(response, "Hidden QA")
        client.thread_list.assert_not_called()

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_cached_session_order_promotes_qa_activity_from_beyond_page_size(
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
        for index in range(session_list._SESSION_PAGE_SIZE):
            updated_at = datetime.fromtimestamp(5000 - index, UTC)
            SessionMetadata.objects.create(
                thread_id=f"ordinary-{index}",
                cwd="/repo",
                codex_display_title=f"Ordinary {index}",
                codex_created_at=updated_at,
                codex_updated_at=updated_at,
                codex_last_synced_at=now,
            )
        SessionMetadata.objects.create(
            thread_id="main-thread",
            cwd="/repo",
            codex_display_title="Main session",
            codex_created_at=datetime.fromtimestamp(1, UTC),
            codex_updated_at=datetime.fromtimestamp(1, UTC),
            codex_last_synced_at=now,
        )
        SessionMetadata.objects.create(
            thread_id="qa-thread",
            cwd="/repo",
            codex_display_title="Hidden QA",
            codex_created_at=datetime.fromtimestamp(1000, UTC),
            codex_updated_at=datetime.fromtimestamp(1000, UTC),
            codex_last_synced_at=datetime.fromtimestamp(1000, UTC),
        )
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
        )
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="qa-thread",
            cwd="/repo",
            prompt="qa",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            thread_id="qa-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_COMPLETED,
        )
        run_updated_at = datetime.fromtimestamp(10_000, UTC)
        SystemAgentRun.objects.filter(pk=run.pk).update(updated_at=run_updated_at)
        SystemWorkflow.objects.filter(pk=workflow.pk).update(updated_at=run_updated_at)

        response = self.client.get(reverse("index"))
        sessions_context = cast(list[dict[str, Any]], response.context["sessions"])
        session_ids = [session["id"] for session in sessions_context]

        self.assertEqual(response.status_code, 200)
        self.assertEqual(session_ids[0], "main-thread")
        self.assertIn("ordinary-48", session_ids)
        self.assertNotIn("ordinary-49", session_ids)
        self.assertNotContains(response, "Hidden QA")
        client.thread_list.assert_not_called()

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_index_cursor_keeps_later_pages_stable_when_rows_move(
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
        for i in range(51):
            SessionMetadata.objects.create(
                thread_id=f"thread-{i}",
                cwd="/repo",
                codex_display_title=f"Session {i}",
                codex_name=f"Session {i}",
                codex_created_at=datetime.fromtimestamp(1000 - i, UTC),
                codex_updated_at=datetime.fromtimestamp(1000 - i, UTC),
                codex_last_synced_at=now,
            )

        response = self.client.get(reverse("index"))
        load_more_url = self._assert_index_cursor_url(response)
        SessionMetadata.objects.create(
            thread_id="new-front",
            cwd="/repo",
            codex_display_title="New front session",
            codex_created_at=datetime.fromtimestamp(2000, UTC),
            codex_updated_at=datetime.fromtimestamp(2000, UTC),
            codex_last_synced_at=now,
        )

        response = self.client.get(load_more_url)

        self.assertNotContains(response, "Session 49")
        self.assertContains(response, "Session 50")
        self.assertNotContains(response, "New front session")
        client.thread_list.assert_not_called()

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_system_sessions_pages_before_helper_lookups_and_keeps_cursor_stable(
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
        for i in range(51):
            SessionMetadata.objects.create(
                thread_id=f"system-{i:02d}",
                cwd="/repo",
                codex_display_title=f"System {i:02d}",
                codex_name=f"System {i:02d}",
                codex_created_at=datetime.fromtimestamp(1000 - i, UTC),
                codex_updated_at=datetime.fromtimestamp(1000 - i, UTC),
                codex_last_synced_at=now,
                is_hidden_system_session=True,
            )

        with (
            patch(
                "hitch.main.sessions.system_agent_summary._system_agent_runs_by_thread_id",
                return_value={},
            ) as runs_by_thread_id,
            patch(
                "hitch.main.sessions.system_agent_summary._system_agent_instances_by_thread_id",
                return_value={},
            ) as instances_by_thread_id,
        ):
            response = self.client.get(reverse("system_sessions"))
            load_more_url = self._assert_index_cursor_url(response)

            self.assertEqual(response.status_code, 200)
            self.assertContains(response, "System 00")
            self.assertContains(response, "System 49")
            self.assertNotContains(response, "System 50")
            expected_first_page_ids = [f"system-{i:02d}" for i in range(50)]
            self.assertEqual(
                list(runs_by_thread_id.call_args.args[0]), expected_first_page_ids
            )
            self.assertEqual(
                list(instances_by_thread_id.call_args.args[0]),
                expected_first_page_ids,
            )

            SessionMetadata.objects.create(
                thread_id="new-front-system",
                cwd="/repo",
                codex_display_title="New front system",
                codex_created_at=datetime.fromtimestamp(2000, UTC),
                codex_updated_at=datetime.fromtimestamp(2000, UTC),
                codex_last_synced_at=now,
                is_hidden_system_session=True,
            )
            response = self.client.get(load_more_url)

            self.assertContains(response, "System 50")
            self.assertNotContains(response, "System 49")
            self.assertNotContains(response, "New front system")
            self.assertEqual(
                list(runs_by_thread_id.call_args.args[0]), ["system-50"]
            )
            self.assertEqual(
                list(instances_by_thread_id.call_args.args[0]), ["system-50"]
            )
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
    def test_system_sessions_accepts_cold_index_second_precision_cursor(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        mock_discover.return_value = []
        threads = [
            SimpleNamespace(
                id=f"system-{i:02d}",
                name=f"System {i:02d}",
                preview="",
                cwd="/repo",
                path=None,
                updated_at=1000 + ((50 - i) / 1_000_000),
                thread_source=ThreadSource.subagent,
            )
            for i in range(51)
        ]
        client = _setup_codex(mock_codex, threads=threads)

        response = self.client.get(reverse("system_sessions"))
        load_more_url = self._assert_index_cursor_url(response)

        self.assertContains(response, "System 50")
        self.assertContains(response, "System 01")
        self.assertNotContains(response, "System 00")

        response = self.client.get(load_more_url)

        self.assertContains(response, "System 00")
        self.assertNotContains(response, "System 01")
        self.assertNotContains(response, "System 50")
        self.assertEqual(client.thread_list.call_count, 1)

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
                cursor = "idx:" + base64.urlsafe_b64encode(
                    cursor_payload.encode()
                ).decode()

                response = self.client.get(
                    reverse("system_sessions"), {"cursor": cursor}
                )

                self.assertEqual(response.status_code, 200)
                self.assertContains(response, "System")
        client.thread_list.assert_not_called()

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_system_sessions_excludes_accepted_visible_system_thread(
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
        accepted = SessionMetadata.objects.create(
            thread_id="accepted-system",
            cwd="/repo",
            codex_display_title="Accepted visible system",
            codex_name="Accepted visible system",
            codex_created_at=now,
            codex_updated_at=now,
            codex_last_synced_at=now,
            is_hidden_system_session=True,
        )
        SessionMetadata.objects.create(
            thread_id="other-system",
            cwd="/repo",
            codex_display_title="Other system",
            codex_name="Other system",
            codex_created_at=datetime.fromtimestamp(1000, UTC),
            codex_updated_at=datetime.fromtimestamp(1000, UTC),
            codex_last_synced_at=now,
            is_hidden_system_session=True,
        )
        ProposedSession.objects.create(
            title="Accepted proposal",
            candidate_session=accepted,
            accepted_session=accepted,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
        )

        index_response = self.client.get(reverse("index"))
        system_response = self.client.get(reverse("system_sessions"))

        self.assertContains(index_response, "Accepted visible system")
        self.assertContains(system_response, "Other system")
        self.assertNotContains(system_response, "Accepted visible system")
        client.thread_list.assert_not_called()

    @patch("hitch.main.views.common.Codex")
    def test_full_refresh_invalidates_absent_active_rows(
        self, mock_codex: MagicMock
    ) -> None:
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
            session_index.indexed_sessions()
            .filter(thread_id="stale-active", codex_archived=False)
            .exists()
        )
        self.assertTrue(
            session_index.indexed_sessions().filter(thread_id="fresh-active").exists()
        )

    @patch("hitch.main.views.common.Codex")
    def test_refresh_marks_legacy_autonomous_goal_prompt_hidden(
        self, mock_codex: MagicMock
    ) -> None:
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
        client = _setup_codex(mock_codex, threads=[candidate])

        session_index.refresh_from_codex(
            client,
            projects=[],
            include_active=True,
            max_pages=None,
            use_state_db_only=True,
        )

        metadata = SessionMetadata.objects.get(thread_id="legacy-candidate")
        self.assertTrue(metadata.is_hidden_system_session)

    @patch("hitch.main.views.common.Codex")
    def test_state_db_refresh_does_not_invalidate_absent_active_rows(
        self, mock_codex: MagicMock
    ) -> None:
        now = datetime.now(UTC)
        SessionMetadata.objects.create(
            thread_id="cached-active",
            cwd="/repo",
            codex_display_title="Cached active",
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
            use_state_db_only=True,
        )

        self.assertTrue(
            session_index.indexed_sessions().filter(thread_id="cached-active").exists()
        )
        self.assertTrue(
            session_index.indexed_sessions().filter(thread_id="fresh-active").exists()
        )

    @patch("hitch.main.views.common.Codex")
    def test_background_session_index_refresh_uses_state_db_only(
        self, mock_codex: MagicMock
    ) -> None:
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
            all(
                mock_call.kwargs["use_state_db_only"] is True
                for mock_call in client.thread_list.call_args_list
            )
        )
        active_state = SessionIndexSyncState.objects.get(
            source=SessionIndexSyncState.SOURCE_ACTIVE
        )
        archived_state = SessionIndexSyncState.objects.get(
            source=SessionIndexSyncState.SOURCE_ARCHIVED
        )
        self.assertFalse(active_state.is_complete)
        self.assertFalse(archived_state.is_complete)

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_complete_empty_session_index_serves_cached_empty_state(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = CodexError("thread list unavailable")
        mock_discover.return_value = []
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=datetime.now(UTC),
            is_complete=True,
        )

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No sessions found.")
        client.thread_list.assert_not_called()

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_stale_complete_session_index_serves_cache_and_schedules_refresh(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        now = datetime.now(UTC)
        cached = _session("cached", name="Cached session", updated_at=2000)
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = CodexError("thread list unavailable")
        mock_discover.return_value = []
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=now - timedelta(minutes=5),
            is_complete=True,
        )
        session_index.upsert_thread(cached, projects=[])

        with (
            patch("hitch.main.caches._start_models_refresh_thread"),
            patch(
                "hitch.main.views.common._start_usage_session_index_refresh_thread"
            ) as start_index_refresh,
            self.captureOnCommitCallbacks(execute=True),
        ):
            response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Cached session")
        mock_codex.assert_not_called()
        client.thread_list.assert_not_called()
        start_index_refresh.assert_called_once_with(
            enable_memories=False,
            include_active=True,
            include_archived=False,
        )

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_complete_session_index_with_pending_cursor_serves_cache(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        now = datetime.now(UTC)
        cached = _session("cached", name="Cached session", updated_at=2000)
        client = _setup_codex(mock_codex)
        mock_discover.return_value = []
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=now,
            is_complete=True,
            next_cursor="page-2",
        )
        session_index.upsert_thread(cached, projects=[])

        with (
            patch("hitch.main.caches._start_models_refresh_thread"),
            patch(
                "hitch.main.views.common._start_usage_session_index_refresh_thread"
            ) as start_index_refresh,
            self.captureOnCommitCallbacks(execute=True),
        ):
            response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Cached session")
        mock_codex.assert_not_called()
        client.thread_list.assert_not_called()
        start_index_refresh.assert_called_once_with(
            enable_memories=False,
            include_active=True,
            include_archived=False,
        )

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_stale_complete_session_index_does_not_start_codex(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        now = datetime.now(UTC)
        mock_codex.side_effect = CodexError("codex unavailable")
        mock_discover.return_value = []
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=now - timedelta(minutes=5),
            is_complete=True,
        )
        SessionMetadata.objects.create(
            thread_id="cached",
            cwd="/repo",
            codex_display_title="Cached session",
            codex_name="Cached session",
            codex_created_at=now,
            codex_updated_at=now,
            codex_last_synced_at=now,
        )

        with (
            patch("hitch.main.caches._start_models_refresh_thread"),
            patch(
                "hitch.main.views.common._start_usage_session_index_refresh_thread"
            ) as start_index_refresh,
            self.captureOnCommitCallbacks(execute=True),
        ):
            response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Cached session")
        mock_codex.assert_not_called()
        start_index_refresh.assert_called_once_with(
            enable_memories=False,
            include_active=True,
            include_archived=False,
        )

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_stale_complete_session_index_keeps_index_cursor_page(
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
            patch(
                "hitch.main.views.common._start_usage_session_index_refresh_thread"
            ) as start_index_refresh,
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
        state = SessionIndexSyncState.objects.get(
            source=SessionIndexSyncState.SOURCE_ACTIVE
        )
        self.assertTrue(state.is_complete)
        self.assertEqual(state.next_cursor, "")
        start_index_refresh.assert_called_once_with(
            enable_memories=False,
            include_active=True,
            include_archived=False,
        )

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_stale_complete_pending_refresh_keeps_usage_totals_available(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        now = datetime.now(UTC)
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
        metadata = _seed_usage_metadata("usage-thread", path=rollout_path)
        SessionMetadata.objects.filter(pk=metadata.pk).update(
            usage_last_checked_at=now
        )
        SessionIndexSyncState.objects.filter(
            source=SessionIndexSyncState.SOURCE_ACTIVE
        ).update(last_synced_at=now - timedelta(minutes=5), next_cursor="page-2")
        _cache_token_usage(
            "usage-thread",
            input_tokens=400,
            cached_input_tokens=50,
            output_tokens=600,
            total_tokens=1_000,
            path=rollout_path,
        )
        _setup_codex(mock_codex)
        mock_discover.return_value = []

        with (
            patch("hitch.main.caches._rate_limits_for_usage_context", return_value=None),
            patch(
                "hitch.main.views.common._start_usage_session_index_refresh_thread"
            ) as start_index_refresh,
            patch("hitch.main.sessions.token_usage._start_usage_token_refresh_thread"),
            patch("hitch.main.caches._start_models_refresh_thread"),
            self.captureOnCommitCallbacks(execute=True),
        ):
            usage_response = self.client.get(reverse("usage"))

        self.assertEqual(usage_response.status_code, 200)
        self.assertNotContains(usage_response, "All sessions usage unavailable.")
        lifetime_usage = cast(dict[str, Any], usage_response.context["lifetime_usage"])
        self.assertEqual(lifetime_usage["total"]["input"], "350")
        self.assertEqual(lifetime_usage["total"]["output"], "600")
        self.assertEqual(lifetime_usage["total"]["cached"], "50")
        mock_codex.assert_not_called()
        start_index_refresh.assert_called_once_with(
            enable_memories=False,
            include_active=True,
            include_archived=False,
        )

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
        state = SessionIndexSyncState.objects.get(
            source=SessionIndexSyncState.SOURCE_ACTIVE
        )
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
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="visible",
            cwd="/repo",
        )
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="qa-thread",
            cwd="/repo",
            prompt="QA prompt",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            display_author="QA agent",
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            thread_id="qa-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_COMPLETED,
        )
        SessionMetadata.objects.create(
            thread_id="qa-thread",
            cwd="/repo",
            project=project,
        )

        response = self.client.get(reverse("system_sessions"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "QA agent")
        self.assertContains(response, "completed")
        metadata = SessionMetadata.objects.get(thread_id="qa-thread")
        self.assertEqual(metadata.project, project)
        self.assertIsNotNone(metadata.codex_updated_at)

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_archived_toggle_refreshes_missing_archived_cache(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _seed_cookies(self.client, **{_SHOW_ARCHIVED_COOKIE: "true"})
        now = datetime.now(UTC)
        active = _session("active", name="Active session")
        archived = _session(
            "archived",
            name="Archived session",
            path="/tmp/archived_sessions/archived.jsonl",
        )
        client = _setup_codex(mock_codex, threads=[active], archived_threads=[archived])
        mock_discover.return_value = []
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=now,
            is_complete=True,
        )
        SessionMetadata.objects.create(
            thread_id="active",
            cwd="/repo",
            codex_display_title="Active session",
            codex_created_at=now,
            codex_updated_at=now,
            codex_last_synced_at=now,
        )

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Active session")
        self.assertContains(response, "Archived session")
        client.thread_list.assert_any_call(
            limit=100,
            sort_key=ThreadSortKey.updated_at,
            sort_direction=SortDirection.desc,
            archived=True,
            use_state_db_only=True,
        )

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_incomplete_session_index_uses_codex_cursor_pagination(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        cached_page = [
            _session(f"thread-{i}", name=f"Cached {i}", updated_at=1000 - i)
            for i in range(50)
        ]
        client = _setup_codex(mock_codex)
        client.thread_list.return_value = SimpleNamespace(
            data=cached_page,
            next_cursor="page-2",
        )
        client.thread_list.side_effect = None
        mock_discover.return_value = []
        now = datetime.now(UTC)
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=now,
            is_complete=False,
            next_cursor="page-2",
        )
        for thread in cached_page:
            SessionMetadata.objects.create(
                thread_id=thread.id,
                cwd="/repo",
                codex_display_title=thread.name,
                codex_name=thread.name,
                codex_created_at=now,
                codex_updated_at=now,
                codex_last_synced_at=now,
            )

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Cached 0")
        self.assertContains(response, 'href="/?cursor=page-2"')
        mock_codex.assert_called_once()

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_codex_cursor_request_uses_codex_even_when_index_complete(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        cursor_page = [_session("cursor-thread", name="Cursor session")]
        client = _setup_codex(mock_codex)
        client.thread_list.return_value = SimpleNamespace(data=cursor_page)
        client.thread_list.side_effect = None
        mock_discover.return_value = []
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=datetime.now(UTC),
            is_complete=True,
        )

        response = self.client.get(f"{reverse('index')}?cursor=page-2")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Cursor session")
        mock_codex.assert_called_once()
        client.thread_list.assert_called_once_with(
            limit=100,
            sort_key=ThreadSortKey.updated_at,
            sort_direction=SortDirection.desc,
            cursor="page-2",
            use_state_db_only=True,
        )

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_empty_session_index_self_primes_from_index_view(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        first_page = [
            _session(f"thread-{i}", name=f"Session {i}", updated_at=1000 - i)
            for i in range(50)
        ]
        second_page = [
            _session(f"thread-{i}", name=f"Session {i}", updated_at=1000 - i)
            for i in range(50, 60)
        ]
        client = _setup_codex(mock_codex)

        def thread_list(*, cursor: str | None = None, **_: Any) -> SimpleNamespace:
            if cursor == "page-2":
                return SimpleNamespace(data=second_page)
            return SimpleNamespace(data=first_page, next_cursor="page-2")

        client.thread_list.side_effect = thread_list
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Session 0")
        self._assert_index_cursor_url(response)
        self.assertEqual(SessionMetadata.objects.exclude(codex_updated_at__isnull=True).count(), 60)
        self.assertTrue(SessionIndexSyncState.objects.get(source="active").is_complete)

        client.thread_list.reset_mock(side_effect=True)
        client.thread_list.side_effect = CodexError("thread list unavailable")

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Session 0")
        client.thread_list.assert_not_called()

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_renders_empty_state_and_new_session_button(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _setup_codex(mock_codex)
        mock_discover.return_value = []
        response = self.client.get(reverse("index"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "HITCH")
        self.assertContains(response, "No sessions found.")
        self.assertContains(response, "New session")
        self.assertContains(response, f'href="{reverse("new_session")}"')
        self.assertContains(response, "Create project")

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_hides_project_banner_when_project_exists(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _make_project()
        _setup_codex(mock_codex)
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "Create a project to group sessions")

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_hides_system_agent_threads(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        visible = _session("visible", preview="Visible")
        hidden = _session("qa-thread", preview="Hidden QA")
        _setup_codex(mock_codex, threads=[visible, hidden])
        mock_discover.return_value = []
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="visible",
            cwd="/repo",
        )
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="qa-thread",
            cwd="/repo",
            prompt="qa",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind="pr_qa",
            thread_id="qa-thread",
            instance=instance,
        )

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Visible")
        self.assertNotContains(response, "Hidden QA")

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
    def test_hides_orphan_hitch_system_prompt_threads(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        visible = _session("visible", name="Visible")
        candidate = _session(
            "orphan-candidate",
            name="You are Hitch's autonomous goal agent.",
            preview="You are Hitch's autonomous goal agent.\n\nAnalyze the repo.",
            thread_source=ThreadSource.subagent,
        )
        judge = _session(
            "orphan-judge",
            name="You are Hitch's autonomous goal confidence judge.",
            preview="You are Hitch's autonomous goal confidence judge.\n\nJudge it.",
            thread_source=ThreadSource.subagent,
        )
        candidate.turns = []
        judge.turns = []
        client = _setup_codex(mock_codex, threads=[visible, candidate, judge])
        client._client.thread_resume.return_value = SimpleNamespace(thread=candidate)
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Visible")
        self.assertNotContains(response, "You are Hitch&#x27;s autonomous goal agent.")
        self.assertNotContains(
            response, "You are Hitch&#x27;s autonomous goal confidence judge."
        )

        system_response = self.client.get(reverse("system_sessions"))

        self.assertEqual(system_response.status_code, 200)
        self.assertNotContains(system_response, "Visible")
        self.assertContains(system_response, "You are Hitch&#x27;s autonomous goal agent.")
        self.assertContains(
            system_response, "You are Hitch&#x27;s autonomous goal confidence judge."
        )
        self.assertContains(
            system_response,
            reverse("system_session", kwargs={"session_id": "orphan-candidate"}),
        )
        self.assertContains(system_response, "Hitch system", count=2)
        self.assertContains(system_response, "untracked", count=2)

        detail_response = self.client.get(
            reverse("system_session", kwargs={"session_id": "orphan-candidate"})
        )

        self.assertEqual(detail_response.status_code, 200)
        self.assertContains(detail_response, '<body class="read-only"')
        self.assertContains(detail_response, "You are Hitch&#x27;s autonomous goal agent.")

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
        self.assertNotContains(
            response, "You are Hitch&#x27;s autonomous goal confidence judge."
        )
        self.assertNotContains(response, "You are Hitch&#x27;s standing order agent.")
        self.assertNotContains(
            response, "You are Hitch&#x27;s standing order confidence judge."
        )

        system_response = self.client.get(reverse("system_sessions"))

        self.assertEqual(system_response.status_code, 200)
        self.assertNotContains(system_response, "Visible")
        self.assertContains(system_response, "You are Hitch&#x27;s autonomous goal agent.")
        self.assertContains(
            system_response, "You are Hitch&#x27;s autonomous goal confidence judge."
        )
        self.assertContains(system_response, "You are Hitch&#x27;s standing order agent.")
        self.assertContains(
            system_response, "You are Hitch&#x27;s standing order confidence judge."
        )

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_user_prompt_with_hitch_system_text_remains_visible(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        user_thread = _session(
            "user-prefixed",
            name="You are Hitch's autonomous goal agent. Please help",
            preview="You are Hitch's autonomous goal agent.\n\nPlease explain this.",
        )
        user_thread.turns = []
        client = _setup_codex(mock_codex, threads=[user_thread])
        client._client.thread_resume.return_value = SimpleNamespace(thread=user_thread)
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response, "You are Hitch&#x27;s autonomous goal agent. Please help"
        )

        system_detail_response = self.client.get(
            reverse("system_session", kwargs={"session_id": "user-prefixed"})
        )

        self.assertEqual(system_detail_response.status_code, 404)

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_user_prompt_with_legacy_autonomous_goal_title_remains_visible(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        user_thread = _session(
            "user-exact-title",
            name=system_agents.AUTONOMOUS_GOAL_AGENT_PROMPT_TITLE,
            preview=(
                f"{system_agents.AUTONOMOUS_GOAL_AGENT_PROMPT_TITLE}\n\n"
                "Please explain this."
            ),
        )
        _setup_codex(mock_codex, threads=[user_thread])
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "You are Hitch&#x27;s autonomous goal agent.")

        system_response = self.client.get(reverse("system_sessions"))

        self.assertEqual(system_response.status_code, 200)
        self.assertNotContains(system_response, "You are Hitch&#x27;s autonomous goal agent.")

    @patch("hitch.main.views.common.Codex")
    def test_untracked_system_session_resume_error_is_not_404(
        self, mock_codex: MagicMock
    ) -> None:
        session_id = "00000000-0000-0000-0000-000000000001"
        client = _setup_codex(mock_codex)
        client._client.thread_resume.side_effect = CodexError("app server down")

        with self.assertRaises(CodexError):
            self.client.get(
                reverse("system_session", kwargs={"session_id": session_id})
            )

    @patch("hitch.main.views.common.Codex")
    def test_untracked_system_session_missing_thread_is_404(
        self, mock_codex: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client._client.thread_resume.side_effect = InvalidRequestError(
            -32600, "thread orphan-system not found"
        )

        response = self.client.get(
            reverse("system_session", kwargs={"session_id": "orphan-system"})
        )

        self.assertEqual(response.status_code, 404)

    @patch("hitch.main.views.common.Codex")
    def test_untracked_system_session_invalid_session_id_is_404(
        self, mock_codex: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client._client.thread_resume.side_effect = InvalidRequestError(
            -32600,
            "invalid session id: invalid character: expected an optional prefix",
        )

        response = self.client.get(
            reverse("system_session", kwargs={"session_id": "orphan-system"})
        )

        self.assertEqual(response.status_code, 404)

    @patch("hitch.main.views.common.Codex")
    def test_untracked_system_session_non_thread_invalid_request_is_not_404(
        self, mock_codex: MagicMock
    ) -> None:
        session_id = "00000000-0000-0000-0000-000000000001"
        client = _setup_codex(mock_codex)
        client._client.thread_resume.side_effect = InvalidRequestError(
            -32600, "model provider not found"
        )

        with self.assertRaises(InvalidRequestError):
            self.client.get(
                reverse("system_session", kwargs={"session_id": session_id})
            )

    @patch("hitch.main.workflows.system_agents.accepted_visible_system_thread_ids")
    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_session_list_reuses_accepted_visible_thread_ids_across_pages(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_accepted_visible: MagicMock,
    ) -> None:
        hidden = _session(
            "hidden-subagent",
            name="Hidden subagent",
            thread_source=ThreadSource.subagent,
        )
        visible = _session("visible", name="Visible")
        client = _setup_codex(mock_codex)
        mock_discover.return_value = []
        mock_accepted_visible.return_value = set()

        def thread_list(*, cursor: str | None = None, **_: Any) -> SimpleNamespace:
            if cursor == "page-2":
                return SimpleNamespace(data=[visible])
            return SimpleNamespace(data=[hidden], next_cursor="page-2")

        client.thread_list.side_effect = thread_list

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Visible")
        self.assertNotContains(response, "Hidden subagent")
        mock_accepted_visible.assert_called_once_with()

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
        metadata = SessionMetadata.objects.create(thread_id="accepted-candidate")
        ProposedSession.objects.create(
            title="Accepted proposal",
            candidate_session=metadata,
            accepted_session=metadata,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
        )

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Accepted candidate")

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_session_list_self_primes_initial_page_and_links_next_offset(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        first_page = [
            _session(f"thread-{i}", name=f"Session {i}", updated_at=1000 - i)
            for i in range(50)
        ]
        second_page = [_session("thread-50", name="Session 50", updated_at=900)]
        client = _setup_codex(mock_codex)

        def thread_list(*, cursor: str | None = None, **_: Any) -> SimpleNamespace:
            if cursor == "page-2":
                return SimpleNamespace(data=second_page)
            return SimpleNamespace(data=first_page, next_cursor="page-2")

        client.thread_list.side_effect = thread_list
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Session 0")
        self._assert_index_cursor_url(response)
        self.assertEqual(client.thread_list.call_count, 2)
        self.assertTrue(SessionIndexSyncState.objects.get(source="active").is_complete)

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_historical_qa_run_does_not_disable_cursor_pagination(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        first_page = [
            _session(f"thread-{i}", name=f"Session {i}", updated_at=1000 - i)
            for i in range(50)
        ]
        second_page = [_session("thread-50", name="Session 50", updated_at=900)]
        client = _setup_codex(mock_codex)

        def thread_list(*, cursor: str | None = None, **_: Any) -> SimpleNamespace:
            if cursor == "page-2":
                return SimpleNamespace(data=second_page)
            return SimpleNamespace(data=first_page, next_cursor="page-2")

        client.thread_list.side_effect = thread_list
        mock_discover.return_value = []
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="old-main",
            cwd="/repo",
        )
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="old-qa",
            cwd="/repo",
            prompt="qa",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind="pr_qa",
            thread_id="old-qa",
            instance=instance,
        )

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Session 0")
        self._assert_index_cursor_url(response)
        self.assertEqual(client.thread_list.call_count, 2)
        self.assertTrue(SessionIndexSyncState.objects.get(source="active").is_complete)

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_active_session_pagination_stops_before_refetching_seen_cursor(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)

        def thread_list(*, cursor: str | None = None, **_: Any) -> SimpleNamespace:
            if cursor == "b":
                return SimpleNamespace(
                    data=[_session("session-b", name="Session B")], next_cursor="a"
                )
            return SimpleNamespace(
                data=[_session("session-a", name="Session A")], next_cursor="b"
            )

        client.thread_list.side_effect = thread_list
        mock_discover.return_value = []

        response = self.client.get(reverse("index"), {"cursor": "a"})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Session A")
        self.assertContains(response, "Session B")
        self.assertEqual(client.thread_list.call_count, 2)

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_load_more_resumes_partially_consumed_codex_page(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        page_one_visible = [
            _session(f"p1-{i}", name=f"Page 1 visible {i}", updated_at=300 - i)
            for i in range(30)
        ]
        page_one_hidden = [
            _session(f"hidden-{i}", name=f"Hidden {i}", updated_at=200 - i)
            for i in range(20)
        ]
        page_two = [
            _session(f"p2-{i}", name=f"Page 2 visible {i}", updated_at=100 - i)
            for i in range(50)
        ]
        client = _setup_codex(mock_codex)

        def thread_list(*, cursor: str | None = None, **_: Any) -> SimpleNamespace:
            if cursor == "c2":
                return SimpleNamespace(data=page_two, next_cursor="c3")
            if cursor == "c3":
                return SimpleNamespace(data=[])
            return SimpleNamespace(
                data=[*page_one_visible, *page_one_hidden], next_cursor="c2"
            )

        client.thread_list.side_effect = thread_list
        mock_discover.return_value = []
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_AUTONOMOUS_GOAL_RUN,
            main_thread_id="visible",
            cwd="/repo",
        )
        for i in range(20):
            instance = CodexInstance.objects.create(
                pid=i + 1,
                thread_id=f"hidden-{i}",
                cwd="/repo",
                prompt="qa",
                events_path="/dev/null",
                status=CodexInstance.STATUS_COMPLETED,
                purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
                workflow_id=workflow.pk,
            )
            SystemAgentRun.objects.create(
                workflow=workflow,
                agent_kind="autonomous_goal_run",
                thread_id=f"hidden-{i}",
                instance=instance,
            )

        response = self.client.get(reverse("index"))

        self.assertContains(response, "Page 2 visible 19")
        self.assertNotContains(response, "Page 2 visible 20")
        load_more_url = self._assert_index_cursor_url(response)

        response = self.client.get(load_more_url)

        self.assertContains(response, "Page 2 visible 20")
        self.assertContains(response, "Page 2 visible 49")
        self.assertNotContains(response, "Page 2 visible 19")

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_paginates_sessions_before_hiding_system_agent_threads(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        hidden = _session("qa-thread", preview="Hidden QA", updated_at=2000)
        visible = _session(
            "visible-next-page", preview="Visible next page", updated_at=1000
        )
        client = _setup_codex(mock_codex)

        def thread_list(
            *, archived: bool | None = None, cursor: str | None = None, **_: Any
        ) -> SimpleNamespace:
            if archived:
                return SimpleNamespace(data=[])
            if cursor == "page-2":
                return SimpleNamespace(data=[visible])
            return SimpleNamespace(data=[hidden], next_cursor="page-2")

        client.thread_list.side_effect = thread_list
        mock_discover.return_value = []
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="visible-next-page",
            cwd="/repo",
        )
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="qa-thread",
            cwd="/repo",
            prompt="qa",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind="pr_qa",
            thread_id="qa-thread",
            instance=instance,
        )

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Visible next page")
        self.assertNotContains(response, "Hidden QA")

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_qa_activity_can_promote_main_session_from_later_codex_page(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        ordinary = [
            _session(f"ordinary-{i}", name=f"Ordinary {i}", updated_at=1000 - i)
            for i in range(50)
        ]
        hidden_qa = _session("qa-thread", name="Hidden QA", updated_at=5000)
        main = _session("main-thread", name="Main session", updated_at=1)
        client = _setup_codex(mock_codex)

        def thread_list(*, cursor: str | None = None, **_: Any) -> SimpleNamespace:
            if cursor == "page-2":
                return SimpleNamespace(data=[main])
            return SimpleNamespace(data=[*ordinary, hidden_qa], next_cursor="page-2")

        client.thread_list.side_effect = thread_list
        mock_discover.return_value = []
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
        )
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="qa-thread",
            cwd="/repo",
            prompt="qa",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind="pr_qa",
            thread_id="qa-thread",
            instance=instance,
        )

        response = self.client.get(reverse("index"))

        self.assertContains(response, "Main session")
        self.assertContains(response, "Ordinary 48")
        self.assertNotContains(response, "Ordinary 49")
        self.assertNotContains(response, "Hidden QA")
        load_more_url = self._assert_index_cursor_url(response)
        self.assertNotContains(response, "materialized_order=1")

        response = self.client.get(load_more_url)

        self.assertNotContains(response, "Main session")
        self.assertContains(response, "Ordinary 49")

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_qa_activity_can_promote_main_session_from_later_in_fetch_page(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        ordinary = [
            _session(f"ordinary-{i}", name=f"Ordinary {i}", updated_at=1000 - i)
            for i in range(50)
        ]
        hidden_qa = _session("qa-thread", name="Hidden QA", updated_at=5000)
        main = _session("main-thread", name="Main session", updated_at=1)
        _setup_codex(mock_codex, threads=[hidden_qa, *ordinary, main])
        mock_discover.return_value = []
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
        )
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="qa-thread",
            cwd="/repo",
            prompt="qa",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind="pr_qa",
            thread_id="qa-thread",
            instance=instance,
        )

        response = self.client.get(reverse("index"))

        self.assertContains(response, "Main session")
        self.assertContains(response, "Ordinary 48")
        self.assertNotContains(response, "Ordinary 49")
        self.assertNotContains(response, "Hidden QA")
        self._assert_index_cursor_url(response)
        self.assertNotContains(response, "materialized_order=1")

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_mid_pagination_qa_activity_keeps_cursor_order(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        first_page = [
            _session(f"first-{i}", name=f"First {i}", updated_at=3000 - i)
            for i in range(50)
        ]
        second_page = [
            _session(f"second-{i}", name=f"Second {i}", updated_at=2000 - i)
            for i in range(50)
        ]
        hidden_qa = _session("qa-thread", name="Hidden QA", updated_at=5000)
        main = _session("main-thread", name="Main session", updated_at=1)
        client = _setup_codex(mock_codex)

        def thread_list(*, cursor: str | None = None, **_: Any) -> SimpleNamespace:
            if cursor == "page-2":
                return SimpleNamespace(data=[*second_page, hidden_qa], next_cursor="page-3")
            if cursor == "page-3":
                return SimpleNamespace(data=[main])
            return SimpleNamespace(data=first_page, next_cursor="page-2")

        client.thread_list.side_effect = thread_list
        mock_discover.return_value = []
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
        )
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="qa-thread",
            cwd="/repo",
            prompt="qa",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind="pr_qa",
            thread_id="qa-thread",
            instance=instance,
        )

        response = self.client.get(reverse("index"), {"cursor": "page-2"})

        self.assertContains(response, "Second 0")
        self.assertContains(response, "Second 49")
        self.assertNotContains(response, "First 0")
        self.assertNotContains(response, "materialized_order=1")

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_qa_activity_materializes_incomplete_final_page(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        ordinary = _session("ordinary", name="Ordinary session", updated_at=1000)
        hidden_qa = _session("qa-thread", name="Hidden QA", updated_at=5000)
        main = _session("main-thread", name="Main session", updated_at=1)
        client = _setup_codex(mock_codex)

        def thread_list(*, cursor: str | None = None, **_: Any) -> SimpleNamespace:
            if cursor == "page-2":
                return SimpleNamespace(data=[main])
            return SimpleNamespace(data=[ordinary, hidden_qa], next_cursor="page-2")

        client.thread_list.side_effect = thread_list
        mock_discover.return_value = []
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
        )
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="qa-thread",
            cwd="/repo",
            prompt="qa",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind="pr_qa",
            thread_id="qa-thread",
            instance=instance,
        )

        response = self.client.get(reverse("index"))
        body = response.content.decode()

        self.assertContains(response, "Main session")
        self.assertContains(response, "Ordinary session")
        self.assertLess(body.index("Main session"), body.index("Ordinary session"))
        self.assertNotContains(response, "Hidden QA")
        self.assertNotContains(response, "Load more")

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_qa_activity_promotes_main_session_when_archived_sessions_are_shown(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _seed_cookies(self.client, **{_SHOW_ARCHIVED_COOKIE: "true"})
        ordinary = [
            _session(f"ordinary-{i}", name=f"Ordinary {i}", updated_at=1000 - i)
            for i in range(50)
        ]
        hidden_qa = _session("qa-thread", name="Hidden QA", updated_at=5000)
        main = _session("main-thread", name="Main session", updated_at=1)
        client = _setup_codex(mock_codex)

        def thread_list(
            *, archived: bool | None = None, cursor: str | None = None, **_: Any
        ) -> SimpleNamespace:
            if archived:
                return SimpleNamespace(data=[])
            if cursor == "page-2":
                return SimpleNamespace(data=[main])
            return SimpleNamespace(data=[*ordinary, hidden_qa], next_cursor="page-2")

        client.thread_list.side_effect = thread_list
        mock_discover.return_value = []
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
        )
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="qa-thread",
            cwd="/repo",
            prompt="qa",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind="pr_qa",
            thread_id="qa-thread",
            instance=instance,
        )

        response = self.client.get(reverse("index"))

        self.assertContains(response, "Main session")
        self.assertContains(response, "Ordinary 48")
        self.assertNotContains(response, "Ordinary 49")
        self.assertNotContains(response, "Hidden QA")
        self._assert_index_cursor_url(response)
        self.assertNotContains(response, "materialized_order=1")

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_archived_merge_promotes_main_session_from_later_in_fetch_page(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _seed_cookies(self.client, **{_SHOW_ARCHIVED_COOKIE: "true"})
        ordinary = [
            _session(f"ordinary-{i}", name=f"Ordinary {i}", updated_at=1000 - i)
            for i in range(50)
        ]
        hidden_qa = _session("qa-thread", name="Hidden QA", updated_at=5000)
        main = _session("main-thread", name="Main session", updated_at=1)
        _setup_codex(mock_codex, threads=[hidden_qa, *ordinary, main])
        mock_discover.return_value = []
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
        )
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="qa-thread",
            cwd="/repo",
            prompt="qa",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind="pr_qa",
            thread_id="qa-thread",
            instance=instance,
        )

        response = self.client.get(reverse("index"))

        self.assertContains(response, "Main session")
        self.assertContains(response, "Ordinary 48")
        self.assertNotContains(response, "Ordinary 49")
        self.assertNotContains(response, "Hidden QA")
        self._assert_index_cursor_url(response)
        self.assertNotContains(response, "materialized_order=1")

    @patch("hitch.main.views.common.Codex")
    def test_system_sessions_lists_hidden_threads_as_read_only_links(
        self, mock_codex: MagicMock
    ) -> None:
        visible = _session("visible", preview="Visible")
        hidden = _session("qa-thread", preview="Hidden QA")
        _setup_codex(mock_codex, threads=[visible, hidden])
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="visible",
            cwd="/repo",
        )
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="qa-thread",
            cwd="/repo",
            prompt="qa",
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
            thread_id="qa-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_COMPLETED,
        )

        response = self.client.get(reverse("system_sessions"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "System sessions")
        self.assertContains(response, "data-session-list-menu")
        self.assertContains(response, 'class="session-list-menu-fallback"')
        self.assertContains(response, f'href="{reverse("index")}" role="menuitem"')
        self.assertContains(response, ">Sessions<")
        self.assertContains(response, "Hidden QA")
        self.assertContains(
            response, reverse("system_session", kwargs={"session_id": "qa-thread"})
        )
        self.assertContains(response, "QA agent")
        self.assertContains(response, "completed")
        self.assertNotContains(response, "Visible")
        self.assertNotContains(response, 'aria-label="Session actions"')
        self.assertNotContains(response, "data-session-archive-url")
        self.assertNotContains(
            response, '<dialog class="new-session" data-new-session-dialog', html=False
        )

    def test_system_session_helpers_defer_large_payload_fields(self) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="visible",
            cwd="/repo",
        )
        run_instance = CodexInstance.objects.create(
            pid=1,
            thread_id="qa-thread",
            cwd="/repo",
            prompt="prompt " * 2000,
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            display_author=system_agents.QA_DISPLAY_AUTHOR,
            developer_instructions="developer " * 2000,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            thread_id="qa-thread",
            instance=run_instance,
            status=SystemAgentRun.STATUS_COMPLETED,
            input={"prompt": "input " * 2000},
            output={"result": "output " * 2000},
            raw_output="raw " * 2000,
            error="error " * 2000,
        )
        CodexInstance.objects.create(
            pid=2,
            thread_id="instance-only-thread",
            cwd="/repo",
            prompt="prompt " * 2000,
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            display_author=system_agents.AUTONOMOUS_GOAL_DISPLAY_AUTHOR,
            developer_instructions="developer " * 2000,
        )

        with CaptureQueriesContext(connection) as captured:
            runs_by_thread_id = (
                system_agent_summary._system_agent_runs_by_thread_id(["qa-thread"])
            )
            instances_by_thread_id = (
                system_agent_summary._system_agent_instances_by_thread_id(
                    ["instance-only-thread"]
                )
            )
            run = runs_by_thread_id["qa-thread"]
            instance = instances_by_thread_id["instance-only-thread"]
            self.assertEqual(
                system_agent_summary._system_agent_run_label(run),
                system_agents.QA_DISPLAY_AUTHOR,
            )
            self.assertEqual(
                system_agent_summary._system_agent_status(run),
                SystemAgentRun.STATUS_COMPLETED,
            )
            self.assertEqual(
                system_agent_summary._system_agent_run_label(None, instance),
                system_agents.AUTONOMOUS_GOAL_DISPLAY_AUTHOR,
            )
            self.assertEqual(
                system_agent_summary._system_agent_status(None, instance),
                CodexInstance.STATUS_RUNNING,
            )

        self.assertEqual(len(captured), 2)
        self.assertNotIn("main_systemworkflow", captured[0]["sql"])
        self.assertTrue(
            {"input", "output", "raw_output", "error"}.issubset(
                run.get_deferred_fields()
            )
        )
        self.assertTrue(
            {"prompt", "developer_instructions"}.issubset(
                run.instance.get_deferred_fields()
            )
        )
        self.assertTrue(
            {"prompt", "developer_instructions"}.issubset(
                instance.get_deferred_fields()
            )
        )

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_index_links_to_new_session_page_instead_of_dialog(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _setup_codex(mock_codex)
        mock_discover.return_value = ["/repo"]

        response = self.client.get(reverse("index"))

        self.assertContains(response, f'href="{reverse("new_session")}"')
        self.assertNotContains(response, "data-new-session-dialog")
        self.assertNotContains(response, "keyboard-adjusted")

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_lists_sessions_sorted_descending(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        sessions = [
            _session("older", name="Older session", updated_at=1000),
            _session("newer", name="Newer session", updated_at=2000),
            _session("middle", name="Middle session", updated_at=1500),
        ]
        _setup_codex(mock_codex, threads=sessions)
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))
        body = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response, reverse("session", kwargs={"session_id": "newer"})
        )
        self.assertContains(
            response,
            'data-session-archive-url="'
            + reverse("set_session_archived", kwargs={"session_id": "newer"})
            + '"',
        )
        self.assertContains(
            response,
            'data-session-name-url="'
            + reverse("set_session_name", kwargs={"session_id": "newer"})
            + '"',
        )
        self.assertContains(response, 'data-session-archived="false"')
        self.assertContains(response, 'aria-label="Session actions"')
        self.assertContains(response, 'data-session-rename-open')
        self.assertContains(response, 'data-archived-visibility-form')
        self.assertContains(response, 'data-visible-projects-open')
        self.assertContains(response, "Visible projects")
        self.assertContains(
            response,
            '<dialog class="new-session" data-visible-projects-dialog',
            html=False,
        )
        self.assertContains(response, 'name="name" value="Newer session" maxlength="200"')
        self.assertContains(response, 'name="next" value="index"')
        self.assertContains(response, 'data-session-archive-label>Archive</button>')
        self.assertContains(response, "data-archive-undo")
        self.assertContains(response, "data-archived-visibility-fallback")
        self.assertLess(body.index("Newer session"), body.index("Middle session"))
        self.assertLess(body.index("Middle session"), body.index("Older session"))

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_index_keeps_pending_archive_rows_hidden(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _setup_codex(mock_codex, threads=[_session("abc", name="Session")])
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertContains(response, ".session.pending-archive {")
        self.assertContains(response, "visibility: hidden;")

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_session_updated_at_includes_hidden_qa_agent_activity(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        active = _session("active", name="Active session", updated_at=1000)
        other = _session("other", name="Other session", updated_at=1500)
        qa_thread = _session("qa-thread", name="Hidden QA", updated_at=2000)
        _setup_codex(mock_codex, threads=[active, other, qa_thread])
        mock_discover.return_value = []
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="active",
            cwd="/repo",
        )
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="qa-thread",
            cwd="/repo",
            prompt="qa",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            display_author=system_agents.QA_DISPLAY_AUTHOR,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            thread_id="qa-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )

        response = self.client.get(reverse("index"))
        body = response.content.decode()
        sessions_context = cast(list[dict[str, Any]], response.context["sessions"])

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [session["id"] for session in sessions_context], ["active", "other"]
        )
        self.assertEqual(sessions_context[0]["updated_at"], 2000)
        self.assertContains(response, 'data-updated-at="2000"')
        self.assertLess(body.index("Active session"), body.index("Other session"))
        self.assertNotContains(response, "Hidden QA")

    def test_qa_activity_lookup_is_scoped_to_current_sessions(self) -> None:
        current_workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="active",
            cwd="/repo",
        )
        current_instance = CodexInstance.objects.create(
            pid=1,
            thread_id="active-qa-thread",
            cwd="/repo",
            prompt="qa",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=current_workflow.pk,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
        )
        current_run = SystemAgentRun.objects.create(
            workflow=current_workflow,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            thread_id="active-qa-thread",
            instance=current_instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )
        old_workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="old-session",
            cwd="/repo",
        )
        old_instance = CodexInstance.objects.create(
            pid=2,
            thread_id="old-qa-thread",
            cwd="/repo",
            prompt="qa",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=old_workflow.pk,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
        )
        old_run = SystemAgentRun.objects.create(
            workflow=old_workflow,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            thread_id="old-qa-thread",
            instance=old_instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )
        old_time = datetime.fromtimestamp(1900, UTC)
        current_time = datetime.fromtimestamp(2000, UTC)
        newer_old_time = datetime.fromtimestamp(3000, UTC)
        SystemWorkflow.objects.filter(pk=current_workflow.pk).update(updated_at=old_time)
        SystemAgentRun.objects.filter(pk=current_run.pk).update(
            updated_at=current_time
        )
        SystemWorkflow.objects.filter(pk=old_workflow.pk).update(
            updated_at=newer_old_time
        )
        SystemAgentRun.objects.filter(pk=old_run.pk).update(updated_at=newer_old_time)

        updated_at_by_main_thread = (
            system_agent_summary._qa_activity_updated_at_by_main_thread_id(
                [_session("active", updated_at=1000)],
                system_agents.hidden_thread_ids(),
            )
        )

        self.assertEqual(updated_at_by_main_thread, {"active": 2000})

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_selected_project_filters_sessions(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        project = _make_project()
        other = _make_project(name="Other", repo_path="/other")
        sessions = [
            _session("matching", name="Matching", cwd="/repo"),
            _session("other", name="Other session", cwd="/other"),
        ]
        _setup_codex(mock_codex, threads=sessions)
        mock_discover.return_value = [Path("/repo"), Path("/other")]
        SessionMetadata.objects.create(
            thread_id="matching", cwd="/repo", project=project
        )
        SessionMetadata.objects.create(thread_id="other", cwd="/other", project=other)
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Matching")
        self.assertContains(response, "Hitch sessions")
        self.assertNotContains(response, "Other session")

    @patch("hitch.main.workflows.system_agents.hidden_thread_ids")
    @patch("hitch.main.views.common.Codex")
    def test_warm_index_filters_system_sessions_without_hidden_id_scan(
        self, mock_codex: MagicMock, mock_hidden_thread_ids: MagicMock
    ) -> None:
        project = _make_project()
        now = timezone.now()
        SessionMetadata.objects.create(
            thread_id="visible",
            cwd="/repo",
            project=project,
            codex_display_title="Visible session",
            codex_updated_at=now,
        )
        SessionMetadata.objects.create(
            thread_id="hidden-system",
            cwd="/repo",
            project=project,
            codex_display_title="Hidden system session",
            codex_updated_at=now + timedelta(seconds=1),
        )
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="visible",
            cwd="/repo",
            status=SystemWorkflow.STATUS_COMPLETED,
        )
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="hidden-system",
            cwd="/repo",
            prompt="qa",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            thread_id="hidden-system",
            instance=instance,
            status=SystemAgentRun.STATUS_COMPLETED,
        )
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=now,
            is_complete=True,
        )
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Visible session")
        self.assertNotContains(response, "Hidden system session")
        mock_hidden_thread_ids.assert_not_called()
        mock_codex.assert_not_called()

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_visible_projects_filter_sessions(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        project = _make_project()
        other = _make_project(name="Other", repo_path="/other")
        sessions = [
            _session("matching", name="Matching", cwd="/repo"),
            _session("other", name="Other session", cwd="/other"),
            _session("no-project", name="No repo session", cwd="/elsewhere"),
        ]
        _setup_codex(mock_codex, threads=sessions)
        mock_discover.return_value = [Path("/repo"), Path("/other")]
        SessionMetadata.objects.create(
            thread_id="matching", cwd="/repo", project=project
        )
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
    def test_visible_projects_rejects_oversized_guest_cookie(
        self, mock_cookie_fits: MagicMock
    ) -> None:
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

    def test_settings_selected_project_stays_visible_with_explicit_filter(self) -> None:
        project = _make_project()
        other = _make_project(name="Other", repo_path="/other")
        _seed_cookies(
            self.client,
            **{_VISIBLE_SESSION_PROJECTS_COOKIE: f"[{other.pk}]"},
        )

        response = self.client.post(
            reverse("update_settings"),
            data={"selected_project": str(project.pk)},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            _cookie_value(response, _SELECTED_PROJECT_COOKIE),
            str(project.pk),
        )
        self.assertEqual(
            _cookie_value(response, _VISIBLE_SESSION_PROJECTS_COOKIE),
            f"[{other.pk},{project.pk}]",
        )

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

    def test_settings_global_update_ignores_stale_session_override(self) -> None:
        SessionMetadata.objects.create(
            thread_id="stale-override",
            cwd="/repo",
            approval_mode="phantom",
        )
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="stale-override",
            cwd="/repo",
            prompt="hi",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
            approval_mode="prompt_user",
            approval_mode_live_editable=True,
        )
        pending = ApprovalRequest.objects.create(
            instance=instance,
            method="item/commandExecution/requestApproval",
            params={"item": {"command": "cargo bench"}},
            decision=ApprovalRequest.DECISION_PENDING,
        )

        response = self.client.post(
            reverse("update_settings"),
            data={"approval_mode": "approve_all"},
        )

        self.assertEqual(response.status_code, 302)
        instance.refresh_from_db()
        self.assertEqual(instance.approval_mode, "approve_all")
        pending.refresh_from_db()
        self.assertEqual(pending.decision, ApprovalRequest.DECISION_ACCEPT)

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
    def test_no_project_metadata_prevents_cwd_project_inference(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        project = _make_project()
        _setup_codex(mock_codex, threads=[_session("cleared", name="Cleared", cwd="/repo")])
        mock_discover.return_value = [Path("/repo")]
        SessionMetadata.objects.create(
            thread_id="cleared", cwd="/repo", project=None, project_cleared=True
        )
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "Cleared")

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_session_list_omits_token_usage(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
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
    def test_usage_page_uses_cached_usage_and_refreshes_rollout_async(
        self, mock_codex: MagicMock
    ) -> None:
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
        SessionMetadata.objects.filter(thread_id="archived").update(
            usage_last_checked_at=datetime.now(UTC)
        )

        with (
            patch("hitch.main.runtime.rollout.latest_token_usage") as latest_usage,
            patch("hitch.main.runtime.rollout.token_usage_history") as usage_history,
            patch("hitch.main.sessions.token_usage._start_usage_token_refresh_thread") as start_refresh,
            patch("hitch.main.caches._start_models_refresh_thread"),
            patch("hitch.main.caches._start_rate_limits_refresh_thread"),
            self.captureOnCommitCallbacks(execute=True),
        ):
            response = self.client.get(reverse("usage"))

        latest_usage.assert_not_called()
        usage_history.assert_not_called()
        start_refresh.assert_called_once()
        refresh_items = start_refresh.call_args.args[0]
        self.assertEqual(len(refresh_items), 1)
        self.assertEqual(refresh_items[0].thread_id, "archived")
        self.assertEqual(refresh_items[0].codex_path, str(rollout_path))
        self.assertContains(response, "Refreshing session token usage...")
        lifetime_usage = cast(dict[str, Any], response.context["lifetime_usage"])
        self.assertEqual(lifetime_usage["total"]["input"], "0")
        self.assertEqual(lifetime_usage["total"]["output"], "0")
        self.assertEqual(lifetime_usage["total"]["cached"], "0")
        self.assertNotContains(response, "90K")
        self.assertNotContains(response, "23K")
        self.assertNotContains(response, "10K")
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
    def test_usage_page_fetches_rate_limits_before_first_render(
        self, mock_codex: MagicMock
    ) -> None:
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
            patch("hitch.main.caches._RATE_LIMITS_REFRESH_IN_FLIGHT", False),
            patch("hitch.main.caches._start_models_refresh_thread"),
            patch(
                "hitch.main.caches._start_rate_limits_refresh_thread"
            ) as start_rate_limits,
            self.captureOnCommitCallbacks(execute=True),
        ):
            response = self.client.get(reverse("usage"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Codex rate limits")
        self.assertContains(response, "Plan: Pro")
        self.assertContains(response, "27% remaining")
        self.assertContains(response, "5-hour window")
        self.assertNotContains(response, "Usage unavailable.")
        client._client.request.assert_called_once_with(
            "account/rateLimits/read",
            None,
            response_model=GetAccountRateLimitsResponse,
        )
        start_rate_limits.assert_not_called()

    def test_lifetime_usage_skips_stale_file_backed_cache(self) -> None:
        rollout_path = _make_rollout(
            self,
            [
                _token_count_line(
                    input_tokens=900,
                    cached_input_tokens=90,
                    output_tokens=100,
                    total_tokens=1_000,
                )
            ],
        )
        os.utime(rollout_path, ns=(2_000_000_000, 2_000_000_000))
        metadata = _seed_usage_metadata("active", path=rollout_path)
        metadata.usage_last_checked_at = datetime.now(UTC)
        metadata.save(update_fields=["usage_last_checked_at"])
        ArchivedSessionTokenUsage.objects.create(
            thread_id="active",
            rollout_path=str(rollout_path),
            rollout_mtime_ns=1_000_000_000,
            input_tokens=400,
            cached_input_tokens=50,
            output_tokens=600,
            total_tokens=1_000,
            daily_usage={"2025-01-05": {"input": 350, "output": 600, "cached": 50}},
        )

        lifetime_usage = token_usage._lifetime_token_usage_for_metadata([metadata])

        self.assertTrue(lifetime_usage["refresh_pending"])
        self.assertEqual(lifetime_usage["refresh_pending_count"], 1)
        self.assertEqual(lifetime_usage["total"]["input"], "0")
        self.assertEqual(lifetime_usage["total"]["output"], "0")
        self.assertEqual(lifetime_usage["total"]["cached"], "0")
        self.assertEqual(lifetime_usage["total"]["chart"], [])

    @patch("hitch.main.views.common.Codex")
    def test_usage_page_schedules_initial_active_and_archived_index_refresh(
        self, mock_codex: MagicMock
    ) -> None:
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
            archived=True,
        )
        os.utime(rollout_path, ns=(1_000_000_000, 1_000_000_000))
        _cache_token_usage(
            "archived",
            input_tokens=400,
            cached_input_tokens=50,
            output_tokens=600,
            total_tokens=1_000,
            path=rollout_path,
        )
        client = _setup_codex(
            mock_codex,
            archived_threads=[
                _session("archived", name="Archived", path=str(rollout_path))
            ],
        )

        with (
            patch(
                "hitch.main.views.common._start_usage_session_index_refresh_thread"
            ) as start_index_refresh,
            patch("hitch.main.sessions.token_usage._start_usage_token_refresh_thread"),
            patch("hitch.main.caches._start_models_refresh_thread"),
            patch("hitch.main.caches._start_rate_limits_refresh_thread"),
            self.captureOnCommitCallbacks(execute=True),
        ):
            response = self.client.get(reverse("usage"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "All sessions usage unavailable.")
        self.assertFalse(SessionMetadata.objects.filter(thread_id="archived").exists())
        client.thread_list.assert_not_called()
        start_index_refresh.assert_called_once_with(
            enable_memories=False,
            include_active=True,
            include_archived=True,
        )

    @patch("hitch.main.views.common.Codex")
    def test_usage_page_throttles_recent_incomplete_index_refresh(
        self, mock_codex: MagicMock
    ) -> None:
        session_index.mark_synced(archived=False, complete=False)
        session_index.mark_synced(archived=True, complete=False)
        client = _setup_codex(mock_codex)

        with (
            patch(
                "hitch.main.views.common._start_usage_session_index_refresh_thread"
            ) as start_index_refresh,
            patch("hitch.main.sessions.token_usage._start_usage_token_refresh_thread"),
            patch("hitch.main.caches._start_models_refresh_thread"),
            patch("hitch.main.caches._start_rate_limits_refresh_thread"),
            self.captureOnCommitCallbacks(execute=True),
        ):
            response = self.client.get(reverse("usage"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "All sessions usage unavailable.")
        self.assertIsNone(response.context["lifetime_usage"])
        client.thread_list.assert_not_called()
        start_index_refresh.assert_not_called()

    @patch("hitch.main.views.common.Codex")
    def test_usage_page_renders_zero_usage_when_complete_index_is_empty(
        self, mock_codex: MagicMock
    ) -> None:
        session_index.mark_synced(archived=False, complete=True)
        session_index.mark_synced(archived=True, complete=True)
        client = _setup_codex(mock_codex)

        with (
            patch(
                "hitch.main.views.common._start_usage_session_index_refresh_thread"
            ) as start_index_refresh,
            patch("hitch.main.sessions.token_usage._start_usage_token_refresh_thread") as start_tokens,
            patch("hitch.main.caches._start_models_refresh_thread"),
            patch("hitch.main.caches._start_rate_limits_refresh_thread"),
        ):
            response = self.client.get(reverse("usage"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "All sessions usage unavailable.")
        lifetime_usage = cast(dict[str, Any], response.context["lifetime_usage"])
        self.assertEqual(lifetime_usage["total"]["input"], "0")
        self.assertEqual(lifetime_usage["total"]["output"], "0")
        self.assertEqual(lifetime_usage["total"]["cached"], "0")
        client.thread_list.assert_not_called()
        start_tokens.assert_not_called()
        start_index_refresh.assert_not_called()

    @patch("hitch.main.views.common.Codex")
    def test_usage_page_schedules_stale_index_refresh_and_renders_cached_usage(
        self, mock_codex: MagicMock
    ) -> None:
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
        _seed_usage_metadata("stale", path=rollout_path)
        _cache_token_usage(
            "stale",
            input_tokens=400,
            cached_input_tokens=50,
            output_tokens=600,
            total_tokens=1_000,
            path=rollout_path,
        )
        SessionMetadata.objects.filter(thread_id="stale").update(
            usage_last_checked_at=datetime.now(UTC)
        )
        SessionIndexSyncState.objects.update(
            last_synced_at=datetime(2025, 1, 1, tzinfo=UTC)
        )
        client = _setup_codex(mock_codex, threads=[], archived_threads=[])

        with (
            patch(
                "hitch.main.views.common._start_usage_session_index_refresh_thread"
            ) as start_index_refresh,
            patch("hitch.main.sessions.token_usage._start_usage_token_refresh_thread"),
            patch("hitch.main.caches._start_models_refresh_thread"),
            patch("hitch.main.caches._start_rate_limits_refresh_thread"),
            self.captureOnCommitCallbacks(execute=True),
        ):
            response = self.client.get(reverse("usage"))

        self.assertEqual(response.status_code, 200)
        metadata = SessionMetadata.objects.get(thread_id="stale")
        self.assertIsNotNone(metadata.codex_updated_at)
        lifetime_usage = cast(dict[str, Any], response.context["lifetime_usage"])
        self.assertEqual(lifetime_usage["total"]["input"], "350")
        self.assertEqual(lifetime_usage["total"]["output"], "600")
        self.assertEqual(lifetime_usage["total"]["cached"], "50")
        client.thread_list.assert_not_called()
        start_index_refresh.assert_called_once_with(
            enable_memories=False,
            include_active=True,
            include_archived=True,
        )

    @patch("hitch.main.views.common.Codex")
    def test_usage_page_hides_totals_until_active_and_archived_indexes_complete(
        self, mock_codex: MagicMock
    ) -> None:
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
            patch(
                "hitch.main.views.common._start_usage_session_index_refresh_thread"
            ) as start_index_refresh,
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

    def test_token_usage_snapshot_recomputes_stale_logic_version_cache(self) -> None:
        # A cache row written by an older counting-logic version must be
        # recomputed even when its (rollout_path, mtime) still match the file.
        # Archived rollouts are immutable, so without a logic-version stamp a
        # row written before a counting fix (e.g. the compaction-reset fix)
        # would be served verbatim forever and the corrected numbers would
        # never reach historical sessions.
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
                ),
            ],
            archived=True,
        )
        mtime_ns = rollout_path.stat().st_mtime_ns
        # Stale row: matches path+mtime and has daily usage, but carries the
        # wrong (pre-fix) numbers and the legacy logic version 0.
        ArchivedSessionTokenUsage.objects.create(
            thread_id="archived",
            rollout_path=str(rollout_path),
            rollout_mtime_ns=mtime_ns,
            input_tokens=1,
            cached_input_tokens=0,
            output_tokens=1,
            total_tokens=2,
            context_tokens=1,
            model_context_window=200_000,
            daily_usage={"2025-01-05": {"input": 1, "output": 1, "cached": 0}},
            usage_logic_version=0,
        )
        thread = _session("archived", path=str(rollout_path))

        snapshot = token_usage._token_usage_snapshot_for(thread)
        assert snapshot is not None
        usage = snapshot["usage"]
        # Recomputed from the rollout, not served from the stale row.
        self.assertEqual(usage["input_tokens"], 100_000)
        self.assertEqual(usage["cached_input_tokens"], 80_000)
        self.assertEqual(usage["output_tokens"], 20_000)
        self.assertEqual(usage["total_tokens"], 120_000)
        cache = ArchivedSessionTokenUsage.objects.get(thread_id="archived")
        self.assertEqual(cache.total_tokens, 120_000)
        self.assertEqual(cache.usage_logic_version, token_usage._TOKEN_USAGE_LOGIC_VERSION)

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

    def test_cached_token_usage_matches_rollout_state_requires_current_version(
        self,
    ) -> None:
        # The match check gates on path, mtime, AND logic version. A row with a
        # matching path+mtime but a stale logic version must not be treated as a
        # match, so a counting-logic bump forces a recompute even when the
        # rollout file is byte-for-byte unchanged.
        rollout_state = _RolloutFileState(
            path=Path("/codex/archived/rollout.jsonl"), mtime_ns=1_234
        )
        current = ArchivedSessionTokenUsage(
            thread_id="t",
            rollout_path=str(rollout_state.path),
            rollout_mtime_ns=rollout_state.mtime_ns,
            usage_logic_version=token_usage._TOKEN_USAGE_LOGIC_VERSION,
        )
        self.assertTrue(
            token_usage._cached_token_usage_matches_rollout_state(current, rollout_state)
        )
        legacy = ArchivedSessionTokenUsage(
            thread_id="t",
            rollout_path=str(rollout_state.path),
            rollout_mtime_ns=rollout_state.mtime_ns,
            usage_logic_version=token_usage._TOKEN_USAGE_LOGIC_VERSION - 1,
        )
        self.assertFalse(
            token_usage._cached_token_usage_matches_rollout_state(legacy, rollout_state)
        )

    def test_stale_logic_version_cache_is_not_current_without_rollout_path(
        self,
    ) -> None:
        # When the rollout file can't be located, a current-version row is still
        # trusted (we can't re-derive it), but a stale-version row must not be:
        # otherwise pathless cached-only sessions keep serving pre-fix counts
        # forever, since archived rollouts never change to trigger a refresh.
        current = ArchivedSessionTokenUsage(
            thread_id="t",
            rollout_path="",
            usage_logic_version=token_usage._TOKEN_USAGE_LOGIC_VERSION,
        )
        self.assertTrue(
            token_usage._cached_token_usage_is_current_for_state(current, None)
        )
        legacy = ArchivedSessionTokenUsage(
            thread_id="t",
            rollout_path="",
            usage_logic_version=token_usage._TOKEN_USAGE_LOGIC_VERSION - 1,
        )
        self.assertFalse(
            token_usage._cached_token_usage_is_current_for_state(legacy, None)
        )

    def test_usage_token_cache_state_rejects_stale_version_pathless_rows(self) -> None:
        # The lifetime-aggregation usability check must also reject stale-version
        # rows on its no-path branches, so a legacy version-0 row does not keep
        # contributing pre-fix counts to the summed totals while a refresh is
        # pending.
        metadata = token_usage._UsageTokenRefreshCandidate(
            thread_id="t", codex_path="", usage_last_checked_at=None
        )
        current = ArchivedSessionTokenUsage(
            thread_id="t",
            rollout_path="",
            usage_logic_version=token_usage._TOKEN_USAGE_LOGIC_VERSION,
        )
        self.assertTrue(token_usage._usage_token_cache_state(metadata, current).cache_usable)
        legacy = ArchivedSessionTokenUsage(
            thread_id="t",
            rollout_path="",
            usage_logic_version=token_usage._TOKEN_USAGE_LOGIC_VERSION - 1,
        )
        self.assertFalse(token_usage._usage_token_cache_state(metadata, legacy).cache_usable)

    def test_token_usage_snapshot_survives_compaction_reset(self) -> None:
        # A session that exhausts its context window records a token_count
        # whose total_token_usage is reset to zero (plus the window size). The
        # headline cumulative figure and the per-day chart are derived from two
        # different rollout reads (latest_token_usage vs token_usage_history),
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

        with patch.object(
            rollout, "_load_rollout_lines", side_effect=load_then_append
        ):
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

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_hides_archived_sessions_by_default(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        active = _session("active", name="Active session")
        archived = _session(
            "archived",
            name="Archived session",
            path="/home/user/.codex/archived_sessions/archived.jsonl",
        )
        _setup_codex(mock_codex, threads=[active], archived_threads=[archived])
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertContains(response, "Active session")
        self.assertNotContains(response, "Archived session")
        client = mock_codex.return_value.__enter__.return_value
        client.thread_list.assert_called_once_with(
            limit=100,
            sort_key=ThreadSortKey.updated_at,
            sort_direction=SortDirection.desc,
            use_state_db_only=True,
        )

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_active_session_not_hidden_when_codex_home_traverses_archived_sessions_dir(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        # Regression: a user whose Codex storage path happens to traverse an
        # unrelated parent directory named ``archived_sessions`` -- e.g. an
        # org-wide ``/data/archived_sessions/<user>/.codex`` layout, or a
        # personal HOME under ``/Users/archived_sessions`` -- previously had
        # every active session silently flipped to archived (and therefore
        # hidden) because ``_thread_is_archived`` scanned the FULL path for
        # the ``archived_sessions`` component instead of only the
        # rollout file's immediate ancestry.
        active = _session(
            "active",
            name="Active session",
            path=(
                "/data/archived_sessions/projects/me/.codex/sessions/"
                "2026/05/15/rollout-active.jsonl"
            ),
        )
        _setup_codex(mock_codex, threads=[active])
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertContains(response, "Active session")
        self.assertTrue(
            SessionMetadata.objects.filter(
                thread_id="active", codex_archived=False
            ).exists()
        )

    def test_upsert_thread_uses_codex_sdk_archived_flag(self) -> None:
        # Regression: ``session_index._thread_is_archived`` consulted only
        # the rollout file path, never the Codex SDK ``archived`` boolean
        # (unlike its views.py twin). The archive flag flips independently
        # of -- and before -- the rollout file is moved into
        # ``archived_sessions/``, so ``thread_list(archived=True)`` can
        # return a thread whose path still lives in the active-storage
        # tree. The path heuristic then cached it as
        # ``codex_archived=False`` and surfaced the just-archived session
        # in the active list, where users could not unarchive or hide it.
        freshly_archived = SimpleNamespace(
            id="freshly-archived",
            name="Freshly archived",
            preview="",
            cwd="/repo",
            path="/codex/sessions/2026/05/27/rollout-fresh.jsonl",
            created_at=1736078400,
            updated_at=1736078400,
            thread_source=None,
            archived=True,
        )

        metadata = session_index.upsert_thread(freshly_archived, projects=[])

        assert metadata is not None
        self.assertTrue(metadata.codex_archived)

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_archived_and_active_sessions_are_globally_paginated(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _seed_cookies(self.client, **{_SHOW_ARCHIVED_COOKIE: "true"})
        active = [
            _session(f"active-{i}", name=f"Active {i}", updated_at=100 - i)
            for i in range(50)
        ]
        archived_page_1 = [
            _session(
                f"archived-1-{i}",
                name=f"Archived page 1 {i}",
                path=f"/tmp/archived_sessions/one-{i}.jsonl",
                updated_at=200 - i,
            )
            for i in range(50)
        ]
        archived_page_2 = [
            _session(
                f"archived-2-{i}",
                name=f"Archived page 2 {i}",
                path=f"/tmp/archived_sessions/two-{i}.jsonl",
                updated_at=150 - i,
            )
            for i in range(50)
        ]
        client = _setup_codex(mock_codex)

        def thread_list(
            *, archived: bool | None = None, cursor: str | None = None, **_: Any
        ) -> SimpleNamespace:
            if archived and cursor == "archived-2":
                return SimpleNamespace(data=archived_page_2)
            if archived:
                return SimpleNamespace(data=archived_page_1, next_cursor="archived-2")
            return SimpleNamespace(data=active)

        client.thread_list.side_effect = thread_list
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertContains(response, "Archived page 1 49")
        self.assertNotContains(response, "Active 0")
        load_more_url = self._assert_index_cursor_url(response)

        response = self.client.get(load_more_url)

        self.assertContains(response, "Archived page 2 49")
        self.assertNotContains(response, "Active 0")
        load_more_url = self._assert_index_cursor_url(response)

        response = self.client.get(load_more_url)

        self.assertContains(response, "Active 0")
        self.assertNotContains(response, "Archived page 1 0")

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_archived_session_pagination_exhausts_cursor_cycles(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _seed_cookies(self.client, **{_SHOW_ARCHIVED_COOKIE: "true"})
        client = _setup_codex(mock_codex)

        def thread_list(
            *, archived: bool | None = None, cursor: str | None = None, **_: Any
        ) -> SimpleNamespace:
            if archived and cursor == "a":
                return SimpleNamespace(data=[], next_cursor="b")
            if archived and cursor == "b":
                return SimpleNamespace(data=[], next_cursor="a")
            if archived:
                return SimpleNamespace(data=[], next_cursor="a")
            return SimpleNamespace(data=[])

        client.thread_list.side_effect = thread_list
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No sessions found.")
        self.assertEqual(client.thread_list.call_count, 8)

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_archived_session_pagination_hydrates_page_metadata_once(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _seed_cookies(self.client, **{_SHOW_ARCHIVED_COOKIE: "true"})
        threads = [
            _session(f"active-{i}", name=f"Active {i}", updated_at=100 - i)
            for i in range(3)
        ]
        _setup_codex(mock_codex, threads=threads, archived_threads=[])
        mock_discover.return_value = []

        with patch(
            "hitch.main.views.session_list._metadata_by_thread_id",
            wraps=session_list._metadata_by_thread_id,
        ) as metadata_by_thread:
            response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Active 0")
        self.assertContains(response, "Active 2")
        self.assertLessEqual(metadata_by_thread.call_count, 2)

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_shows_archived_sessions_when_setting_enabled(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _seed_cookies(self.client, **{_SHOW_ARCHIVED_COOKIE: "true"})
        active = _session("active", name="Active session", updated_at=1000)
        archived = _session(
            "archived",
            name="Archived session",
            path="/home/user/.codex/archived_sessions/archived.jsonl",
            updated_at=2000,
        )
        _setup_codex(mock_codex, threads=[active], archived_threads=[archived])
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))
        body = response.content.decode()

        self.assertContains(response, "Active session")
        self.assertContains(response, "Archived session")
        self.assertContains(response, '<span class="archive-badge">Archived</span>')
        self.assertContains(response, 'data-session-archived="true"')
        self.assertContains(response, 'data-session-archive-label>Unarchive</button>')
        self.assertContains(response, 'name="archived" value="false"')
        self.assertLess(body.index("Archived session"), body.index("Active session"))
        client = mock_codex.return_value.__enter__.return_value
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

    @patch("hitch.main.views.common.Codex")
    def test_usage_page_sums_lifetime_token_usage_without_cached_double_count(
        self, mock_codex: MagicMock
    ) -> None:
        active_path = _make_rollout(
            self,
            [
                _token_count_line(
                    input_tokens=400,
                    cached_input_tokens=50,
                    output_tokens=600,
                    reasoning_output_tokens=40,
                    total_tokens=1_040,
                )
            ],
        )
        archived_path = _make_rollout(
            self,
            [
                _token_count_line(
                    input_tokens=1_000,
                    cached_input_tokens=200,
                    output_tokens=1_500,
                    reasoning_output_tokens=300,
                    total_tokens=3_000,
                )
            ],
            archived=True,
        )
        system_path = _make_rollout(
            self,
            [
                _token_count_line(
                    input_tokens=300,
                    cached_input_tokens=100,
                    output_tokens=700,
                    reasoning_output_tokens=80,
                    total_tokens=1_100,
                )
            ],
        )
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="active",
            cwd="/repo",
        )
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="system",
            cwd="/repo",
            prompt="qa",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            thread_id="system",
            instance=instance,
            status=SystemAgentRun.STATUS_COMPLETED,
        )
        _seed_usage_metadata("active", path=active_path)
        _seed_usage_metadata("system", path=system_path)
        _seed_usage_metadata("archived", path=archived_path)
        _cache_token_usage(
            "active",
            input_tokens=400,
            cached_input_tokens=50,
            output_tokens=600,
            total_tokens=1_040,
            path=active_path,
        )
        _cache_token_usage(
            "system",
            input_tokens=300,
            cached_input_tokens=100,
            output_tokens=700,
            total_tokens=1_100,
            path=system_path,
        )
        _cache_token_usage(
            "archived",
            input_tokens=1_000,
            cached_input_tokens=200,
            output_tokens=1_500,
            total_tokens=3_000,
            path=archived_path,
        )
        client = _setup_codex(mock_codex)

        response = self.client.get(reverse("usage"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "All sessions")
        self.assertContains(response, "Sessions")
        self.assertContains(response, "HITCH system")
        self.assertContains(response, 'class="usage-title-button"')
        self.assertContains(response, 'aria-controls="lifetime-total-chart"')
        self.assertContains(response, "data-lifetime-total-toggle")
        self.assertContains(
            response,
            '<div class="lifetime-stat" role="button" tabindex="0" aria-expanded="false">',
            count=2,
        )
        self.assertContains(response, "All sessions token usage by date")
        self.assertContains(response, "Sessions token usage by date")
        self.assertContains(response, "HITCH system token usage by date")
        self.assertContains(
            response,
            '<span class="lifetime-chart-label">01-05</span>',
            count=3,
        )
        self.assertContains(
            response,
            '<div class="lifetime-chart-axis" aria-hidden="true">',
            count=3,
        )
        self.assertContains(
            response,
            '<span class="lifetime-chart-axis-value">4.5K</span>',
        )
        self.assertContains(
            response,
            '<span class="lifetime-chart-axis-value">2.3K</span>',
        )
        self.assertContains(
            response,
            '<span class="lifetime-chart-axis-value">3.5K</span>',
        )
        self.assertContains(
            response,
            '<span class="lifetime-chart-axis-value">1.8K</span>',
        )
        self.assertContains(
            response,
            '<span class="lifetime-chart-axis-value">1K</span>',
        )
        self.assertContains(
            response,
            '<span class="lifetime-chart-axis-value">500</span>',
        )
        self.assertContains(response, "1.2K")
        self.assertContains(response, "2.1K")
        self.assertContains(response, "250")
        self.assertContains(response, "200")
        self.assertContains(response, "700")
        self.assertContains(response, "100")
        self.assertNotContains(response, "Lifetime")
        self.assertNotContains(response, "4,040")
        self.assertNotContains(response, "3,250")
        self.assertNotContains(response, "1,400")
        cache = ArchivedSessionTokenUsage.objects.get(thread_id="archived")
        self.assertEqual(cache.total_tokens, 3_000)
        client.thread_list.assert_not_called()

    @patch("hitch.main.views.common.Codex")
    def test_profile_shows_selected_project_token_usage(
        self, mock_codex: MagicMock
    ) -> None:
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

    @patch("hitch.main.views.common.Codex")
    def test_usage_page_buckets_orphan_hitch_system_prompt_threads(
        self, mock_codex: MagicMock
    ) -> None:
        session_path = _make_rollout(
            self,
            [
                _token_count_line(
                    input_tokens=100,
                    cached_input_tokens=10,
                    output_tokens=200,
                    total_tokens=300,
                )
            ],
        )
        orphan_system_path = _make_rollout(
            self,
            [
                _token_count_line(
                    input_tokens=300,
                    cached_input_tokens=20,
                    output_tokens=400,
                    total_tokens=700,
                )
            ],
        )
        _seed_usage_metadata("session", path=session_path)
        _seed_usage_metadata(
            "orphan-system",
            path=orphan_system_path,
            thread_source=ThreadSource.subagent.value,
        )
        _cache_token_usage(
            "session",
            input_tokens=100,
            cached_input_tokens=10,
            output_tokens=200,
            total_tokens=300,
            path=session_path,
        )
        _cache_token_usage(
            "orphan-system",
            input_tokens=300,
            cached_input_tokens=20,
            output_tokens=400,
            total_tokens=700,
            path=orphan_system_path,
        )
        client = _setup_codex(mock_codex)

        response = self.client.get(reverse("usage"))

        self.assertEqual(response.status_code, 200)
        lifetime_usage = cast(dict[str, Any], response.context["lifetime_usage"])
        self.assertEqual(lifetime_usage["sessions"]["input"], "90")
        self.assertEqual(lifetime_usage["sessions"]["output"], "200")
        self.assertEqual(lifetime_usage["sessions"]["cached"], "10")
        self.assertEqual(lifetime_usage["system"]["input"], "280")
        self.assertEqual(lifetime_usage["system"]["output"], "400")
        self.assertEqual(lifetime_usage["system"]["cached"], "20")
        self.assertEqual(lifetime_usage["total"]["input"], "370")
        self.assertEqual(lifetime_usage["total"]["output"], "600")
        self.assertEqual(lifetime_usage["total"]["cached"], "30")
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

    @patch("hitch.main.views.common.Codex")
    def test_usage_page_backfills_legacy_empty_daily_usage_cache(
        self, mock_codex: MagicMock
    ) -> None:
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
            archived=True,
        )
        os.utime(rollout_path, ns=(1_000_000_000, 1_000_000_000))
        ArchivedSessionTokenUsage.objects.create(
            thread_id="archived",
            rollout_path=str(rollout_path),
            rollout_mtime_ns=1_000_000_000,
            input_tokens=400,
            cached_input_tokens=50,
            output_tokens=600,
            total_tokens=1_000,
        )
        _seed_usage_metadata("archived", path=rollout_path)
        client = _setup_codex(mock_codex)

        response = self.client.get(reverse("usage"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(
            response, '<span class="lifetime-chart-label">01-05</span>'
        )
        cache = ArchivedSessionTokenUsage.objects.get(thread_id="archived")
        self.assertEqual(cache.daily_usage, {})
        client.thread_list.assert_not_called()

        token_usage._refresh_usage_token_cache_best_effort(
            [token_usage._UsageTokenRefreshItem("archived", str(rollout_path))]
        )

        cache.refresh_from_db()
        self.assertEqual(
            cache.daily_usage,
            {"2025-01-05": {"input": 350, "output": 600, "cached": 50}},
        )

    @patch("hitch.main.views.common.Codex")
    def test_usage_page_reuses_cached_active_session_usage(
        self, mock_codex: MagicMock
    ) -> None:
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
        _seed_usage_metadata("active", path=rollout_path)
        _cache_token_usage(
            "active",
            input_tokens=400,
            cached_input_tokens=50,
            output_tokens=600,
            total_tokens=1_000,
            path=rollout_path,
        )
        SessionMetadata.objects.filter(thread_id="active").update(
            usage_last_checked_at=datetime.now(UTC)
        )
        client = _setup_codex(mock_codex)

        with (
            patch("hitch.main.runtime.rollout.latest_token_usage") as latest_usage,
            patch("hitch.main.runtime.rollout.token_usage_history") as usage_history,
        ):
            response = self.client.get(reverse("usage"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "Refreshing session token usage...")
        cache = ArchivedSessionTokenUsage.objects.get(thread_id="active")
        self.assertEqual(cache.total_tokens, 1_000)
        self.assertEqual(
            cache.daily_usage,
            {"2025-01-05": {"input": 350, "output": 600, "cached": 50}},
        )
        latest_usage.assert_not_called()
        usage_history.assert_not_called()
        client.thread_list.assert_not_called()

        with (
            patch("hitch.main.runtime.rollout.latest_token_usage") as latest_usage,
            patch("hitch.main.runtime.rollout.token_usage_history") as usage_history,
        ):
            response = self.client.get(reverse("usage"))

        self.assertEqual(response.status_code, 200)
        latest_usage.assert_not_called()
        usage_history.assert_not_called()
        client.thread_list.assert_not_called()

    @patch("hitch.main.views.common.Codex")
    def test_usage_page_schedules_recent_invalid_path_for_repair(
        self, mock_codex: MagicMock
    ) -> None:
        _seed_usage_metadata("missing", path="/nonexistent/rollout.jsonl")
        SessionMetadata.objects.filter(thread_id="missing").update(
            usage_last_checked_at=datetime.now(UTC)
        )
        _cache_token_usage(
            "missing",
            input_tokens=400,
            cached_input_tokens=50,
            output_tokens=600,
            total_tokens=1_000,
        )
        _setup_codex(mock_codex)

        with (
            patch("hitch.main.sessions.token_usage._start_usage_token_refresh_thread") as start_refresh,
            patch("hitch.main.caches._start_models_refresh_thread"),
            patch("hitch.main.caches._start_rate_limits_refresh_thread"),
            patch(
                "hitch.main.views.common._rollout_path_from_value",
                side_effect=AssertionError("usage render touched rollout path"),
            ) as rollout_path_from_value,
            self.captureOnCommitCallbacks(execute=True),
        ):
            response = self.client.get(reverse("usage"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Refreshing session token usage...")
        rollout_path_from_value.assert_not_called()
        start_refresh.assert_called_once()
        refresh_items = start_refresh.call_args.args[0]
        self.assertEqual(len(refresh_items), 1)
        self.assertEqual(refresh_items[0].thread_id, "missing")
        self.assertEqual(refresh_items[0].codex_path, "/nonexistent/rollout.jsonl")

    @patch("hitch.main.views.common.Codex")
    def test_usage_page_treats_checked_missing_path_zero_cache_as_terminal(
        self, mock_codex: MagicMock
    ) -> None:
        missing_path = "/nonexistent/rollout.jsonl"
        _seed_usage_metadata("missing", path=missing_path)
        SessionMetadata.objects.filter(thread_id="missing").update(
            usage_last_checked_at=datetime.now(UTC) - timedelta(minutes=5)
        )
        ArchivedSessionTokenUsage.objects.create(
            thread_id="missing",
            rollout_path=missing_path,
            rollout_mtime_ns=0,
            input_tokens=0,
            cached_input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            context_tokens=0,
            model_context_window=0,
            daily_usage={},
            usage_logic_version=token_usage._TOKEN_USAGE_LOGIC_VERSION,
        )
        client = _setup_codex(mock_codex)

        with (
            patch(
                "hitch.main.sessions.token_usage._start_usage_token_refresh_thread"
            ) as start_refresh,
            patch("hitch.main.caches._start_models_refresh_thread"),
            patch("hitch.main.caches._start_rate_limits_refresh_thread"),
            self.captureOnCommitCallbacks(execute=True),
        ):
            response = self.client.get(reverse("usage"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "Refreshing session token usage...")
        lifetime_usage = cast(dict[str, Any], response.context["lifetime_usage"])
        self.assertFalse(lifetime_usage["refresh_pending"])
        self.assertEqual(lifetime_usage["refresh_pending_count"], 0)
        start_refresh.assert_not_called()
        client.thread_list.assert_not_called()

    @patch("hitch.main.views.common.Codex")
    def test_usage_page_uses_indexed_usage_when_session_list_fails(
        self, mock_codex: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = CodexError("thread list unavailable")
        _seed_usage_metadata("indexed")
        _cache_token_usage(
            "indexed",
            input_tokens=400,
            cached_input_tokens=50,
            output_tokens=600,
            total_tokens=1_000,
        )

        response = self.client.get(reverse("usage"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "All sessions")
        self.assertContains(response, "350")
        self.assertContains(response, "600")
        self.assertContains(response, "50")
        self.assertNotContains(response, "All sessions usage unavailable.")
        client.thread_list.assert_not_called()

    @patch("hitch.main.views.common.Codex")
    def test_usage_page_marks_usage_unavailable_until_initial_index_refresh_finishes(
        self, mock_codex: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = CodexError("thread list unavailable")

        with (
            patch(
                "hitch.main.views.common._start_usage_session_index_refresh_thread"
            ) as start_index_refresh,
            patch("hitch.main.caches._start_models_refresh_thread"),
            patch("hitch.main.caches._start_rate_limits_refresh_thread"),
            self.captureOnCommitCallbacks(execute=True),
        ):
            response = self.client.get(reverse("usage"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "All sessions usage unavailable.")
        self.assertNotContains(response, "Refreshing session token usage...")
        self.assertIsNone(response.context["lifetime_usage"])
        client.thread_list.assert_not_called()
        start_index_refresh.assert_called_once_with(
            enable_memories=False,
            include_active=True,
            include_archived=True,
        )

    @patch("hitch.main.sessions.token_usage.Codex")
    def test_usage_page_schedules_missing_metadata_path_refresh(
        self, mock_codex: MagicMock
    ) -> None:
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
    def test_usage_refresh_caches_zero_when_missing_path_cannot_be_repaired(
        self, mock_codex: MagicMock
    ) -> None:
        _seed_usage_metadata("missing-path")
        client = _setup_codex(mock_codex)
        client._client.thread_resume.side_effect = CodexError("resume failed")

        token_usage._refresh_usage_token_cache_best_effort(
            [token_usage._UsageTokenRefreshItem("missing-path", "")]
        )

        metadata = SessionMetadata.objects.get(thread_id="missing-path")
        cache = ArchivedSessionTokenUsage.objects.get(thread_id="missing-path")
        self.assertEqual(cache.rollout_path, "")
        self.assertEqual(cache.total_tokens, 0)
        self.assertEqual(cache.daily_usage, {})
        self.assertIsNotNone(metadata.usage_last_checked_at)
        self.assertFalse(
            token_usage._usage_token_cache_state(metadata, cache).refresh_pending
        )
        self.assertFalse(token_usage._usage_token_refresh_needed(metadata, cache))

    @patch("hitch.main.sessions.token_usage.Codex")
    def test_usage_refresh_caches_zero_when_repaired_path_is_blank(
        self, mock_codex: MagicMock
    ) -> None:
        missing_path = "/nonexistent/rollout.jsonl"
        _seed_usage_metadata("missing-path", path=missing_path)
        client = _setup_codex(mock_codex)
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=_session("missing-path", path="", cwd="/repo")
        )

        token_usage._refresh_usage_token_cache_best_effort(
            [token_usage._UsageTokenRefreshItem("missing-path", missing_path)]
        )

        metadata = SessionMetadata.objects.get(thread_id="missing-path")
        cache = ArchivedSessionTokenUsage.objects.get(thread_id="missing-path")
        self.assertEqual(metadata.codex_path, "")
        self.assertEqual(cache.rollout_path, "")
        self.assertEqual(cache.total_tokens, 0)
        self.assertFalse(
            token_usage._usage_token_cache_state(metadata, cache).refresh_pending
        )
        self.assertFalse(token_usage._usage_token_refresh_needed(metadata, cache))

    @patch("hitch.main.sessions.token_usage.Codex")
    def test_usage_refresh_stamps_disappeared_file_cache_after_failed_repair(
        self, mock_codex: MagicMock
    ) -> None:
        missing_path = "/nonexistent/rollout.jsonl"
        _seed_usage_metadata("missing-path", path=missing_path)
        SessionMetadata.objects.filter(thread_id="missing-path").update(
            usage_last_checked_at=datetime.now(UTC)
        )
        cache = ArchivedSessionTokenUsage.objects.create(
            thread_id="missing-path",
            rollout_path=missing_path,
            rollout_mtime_ns=123,
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
        client = _setup_codex(mock_codex)
        client._client.thread_resume.side_effect = CodexError("resume failed")

        token_usage._refresh_usage_token_cache_best_effort(
            [token_usage._UsageTokenRefreshItem("missing-path", missing_path)]
        )

        metadata.refresh_from_db()
        cache.refresh_from_db()
        self.assertEqual(cache.rollout_path, missing_path)
        self.assertEqual(cache.rollout_mtime_ns, 0)
        self.assertEqual(cache.total_tokens, 1_000)
        self.assertFalse(
            token_usage._usage_token_cache_state(metadata, cache).refresh_pending
        )
        self.assertFalse(token_usage._usage_token_refresh_needed(metadata, cache))

    @patch("hitch.main.sessions.token_usage.Codex")
    def test_usage_refresh_keeps_existing_terminal_missing_path_cache(
        self, mock_codex: MagicMock
    ) -> None:
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
        self.assertFalse(
            token_usage._usage_token_cache_state(metadata, cache).refresh_pending
        )
        self.assertFalse(token_usage._usage_token_refresh_needed(metadata, cache))

    @patch("hitch.main.sessions.token_usage.Codex")
    def test_usage_refresh_missing_metadata_path_handles_unexpected_resume_error(
        self, mock_codex: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client._client.thread_resume.side_effect = RuntimeError("boom")

        refreshed_path = token_usage._refresh_missing_usage_metadata_path(
            client, "missing-path", projects=[]
        )

        self.assertIsNone(refreshed_path)

    def test_usage_refresh_keeps_pathless_old_cache_repair_pending(self) -> None:
        _seed_usage_metadata("missing-path")
        SessionMetadata.objects.filter(thread_id="missing-path").update(
            usage_last_checked_at=datetime.now(UTC)
        )
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

    @patch("hitch.main.sessions.token_usage.Codex")
    def test_usage_refresh_caches_zero_for_unrepairable_invalid_path(
        self, mock_codex: MagicMock
    ) -> None:
        missing_path = "/nonexistent/rollout.jsonl"
        _seed_usage_metadata("missing-path", path=missing_path)
        client = _setup_codex(mock_codex)
        client._client.thread_resume.side_effect = CodexError("resume failed")

        token_usage._refresh_usage_token_cache_best_effort(
            [token_usage._UsageTokenRefreshItem("missing-path", missing_path)]
        )

        metadata = SessionMetadata.objects.get(thread_id="missing-path")
        cache = ArchivedSessionTokenUsage.objects.get(thread_id="missing-path")
        self.assertEqual(cache.rollout_path, missing_path)
        self.assertEqual(cache.total_tokens, 0)
        self.assertEqual(
            cache.usage_logic_version, token_usage._TOKEN_USAGE_LOGIC_VERSION
        )
        self.assertIsNotNone(metadata.usage_last_checked_at)
        self.assertFalse(
            token_usage._usage_token_cache_state(metadata, cache).refresh_pending
        )
        self.assertFalse(token_usage._usage_token_refresh_needed(metadata, cache))

    @patch("hitch.main.sessions.token_usage.Codex")
    def test_usage_refresh_rekeys_mismatched_zero_cache_when_path_cannot_be_repaired(
        self, mock_codex: MagicMock
    ) -> None:
        missing_path = "/new/missing/rollout.jsonl"
        _seed_usage_metadata("missing-path", path=missing_path)
        ArchivedSessionTokenUsage.objects.create(
            thread_id="missing-path",
            rollout_path="/old/missing/rollout.jsonl",
            rollout_mtime_ns=0,
            input_tokens=0,
            cached_input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            context_tokens=0,
            model_context_window=0,
            daily_usage={},
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
        self.assertEqual(cache.total_tokens, 0)
        self.assertFalse(
            token_usage._usage_token_cache_state(metadata, cache).refresh_pending
        )
        self.assertFalse(token_usage._usage_token_refresh_needed(metadata, cache))

    @patch("hitch.main.sessions.token_usage.Codex")
    def test_usage_refresh_replaces_unusable_cache_when_path_cannot_be_repaired(
        self, mock_codex: MagicMock
    ) -> None:
        missing_path = "/nonexistent/rollout.jsonl"
        _seed_usage_metadata("stale-cache", path=missing_path)
        ArchivedSessionTokenUsage.objects.create(
            thread_id="stale-cache",
            rollout_path=missing_path,
            rollout_mtime_ns=1_000_000_000,
            input_tokens=400,
            cached_input_tokens=50,
            output_tokens=600,
            total_tokens=1_000,
            daily_usage={"2025-01-05": {"input": 350, "output": 600, "cached": 50}},
            usage_logic_version=token_usage._TOKEN_USAGE_LOGIC_VERSION - 1,
        )
        client = _setup_codex(mock_codex)
        client._client.thread_resume.side_effect = CodexError("resume failed")

        token_usage._refresh_usage_token_cache_best_effort(
            [token_usage._UsageTokenRefreshItem("stale-cache", missing_path)]
        )

        metadata = SessionMetadata.objects.get(thread_id="stale-cache")
        cache = ArchivedSessionTokenUsage.objects.get(thread_id="stale-cache")
        self.assertEqual(cache.rollout_path, missing_path)
        self.assertEqual(cache.total_tokens, 0)
        self.assertEqual(
            cache.usage_logic_version, token_usage._TOKEN_USAGE_LOGIC_VERSION
        )
        self.assertFalse(
            token_usage._usage_token_cache_state(metadata, cache).refresh_pending
        )
        self.assertFalse(token_usage._usage_token_refresh_needed(metadata, cache))

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

    @patch("hitch.main.views.common.Codex")
    def test_usage_refresh_preserves_cache_when_rollout_path_missing(
        self, mock_codex: MagicMock
    ) -> None:
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
            query
            for query in queries.captured_queries
            if 'UPDATE "main_sessionmetadata"' in query["sql"]
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
            token_usage._start_usage_token_refresh_thread(
                [token_usage._UsageTokenRefreshItem("thread", "")]
            )

        self.assertFalse(token_usage._USAGE_TOKEN_REFRESH_IN_FLIGHT)

    def test_usage_refresh_thread_is_non_daemon_and_materializes_work(self) -> None:
        token_usage._USAGE_TOKEN_REFRESH_IN_FLIGHT = False
        self.addCleanup(setattr, token_usage, "_USAGE_TOKEN_REFRESH_IN_FLIGHT", False)
        thread = MagicMock()
        items = [
            token_usage._UsageTokenRefreshItem("thread-a", ""),
            token_usage._UsageTokenRefreshItem("thread-b", ""),
        ]

        with patch(
            "hitch.main.sessions.token_usage.threading.Thread", return_value=thread
        ) as thread_cls:
            token_usage._start_usage_token_refresh_thread(iter(items))

        thread_cls.assert_called_once()
        self.assertEqual(thread_cls.call_args.kwargs["args"], (tuple(items),))
        self.assertEqual(thread_cls.call_args.kwargs["name"], "usage-token-refresh")
        self.assertFalse(thread_cls.call_args.kwargs["daemon"])
        thread.start.assert_called_once()

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

    def test_usage_refresh_queue_rotates_checked_missing_cache_rows(self) -> None:
        for index in range(30):
            _seed_usage_metadata(f"session-{index:02d}", mark_index_complete=False)
        rows = list(SessionMetadata.objects.order_by("thread_id"))
        first_batch = token_usage._usage_token_refresh_items(rows, {})
        first_batch_ids = [item.thread_id for item in first_batch]

        self.assertEqual(len(first_batch_ids), 25)
        self.assertEqual(first_batch_ids[0], "session-00")
        self.assertEqual(first_batch_ids[-1], "session-24")

        SessionMetadata.objects.filter(thread_id__in=first_batch_ids).update(
            usage_last_checked_at=datetime(2025, 1, 6, tzinfo=UTC)
        )
        rows = list(SessionMetadata.objects.order_by("thread_id"))
        second_batch_ids = [
            item.thread_id for item in token_usage._usage_token_refresh_items(rows, {})
        ]

        self.assertEqual(
            second_batch_ids[:5],
            ["session-25", "session-26", "session-27", "session-28", "session-29"],
        )

    def test_usage_refresh_queue_includes_stale_file_backed_rows_with_many_path_repairs(
        self,
    ) -> None:
        for index in range(30):
            _seed_usage_metadata(f"missing-{index:02d}", mark_index_complete=False)
        stale_thread_ids = []
        for index in range(3):
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
            os.utime(rollout_path, ns=(2_000_000_000, 2_000_000_000))
            thread_id = f"stale-{index}"
            stale_thread_ids.append(thread_id)
            _seed_usage_metadata(
                thread_id, path=rollout_path, mark_index_complete=False
            )
            ArchivedSessionTokenUsage.objects.create(
                thread_id=thread_id,
                rollout_path=str(rollout_path),
                rollout_mtime_ns=1_000_000_000,
                input_tokens=100,
                cached_input_tokens=10,
                output_tokens=20,
                total_tokens=120,
            )
        rows = list(SessionMetadata.objects.order_by("thread_id"))
        caches = token_usage._token_usage_caches_by_thread_ids(
            row.thread_id for row in rows
        )

        batch_ids = [
            item.thread_id for item in token_usage._usage_token_refresh_items(rows, caches)
        ]

        self.assertEqual(len(batch_ids), 25)
        for thread_id in stale_thread_ids:
            self.assertIn(thread_id, batch_ids)

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_new_session_page_populates_project_and_bare_repo_selectors(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        project_a = _make_project(
            name="Project A",
            repo_path="/home/user/proj-a",
            auto_pr_mode=Project.AUTO_PR_ON,
        )
        _make_project(name="Project B", repo_path="/home/user/proj-b")
        _setup_codex(mock_codex)
        mock_discover.return_value = [Path("/home/user/proj-a"), Path("/home/user/proj-b")]

        response = self.client.get(reverse("new_session"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'name="project"')
        self.assertContains(response, "Project A")
        self.assertContains(response, "Project B")
        self.assertContains(response, "&lt;bare repo&gt;")
        self.assertContains(response, f'value="{project_a.pk}" selected')
        self.assertContains(response, 'data-auto-pr-default="true"')
        self.assertContains(response, "data-new-session-auto-pr checked")
        self.assertContains(response, "data-new-session-repo-field hidden")
        self.assertContains(response, "/home/user/proj-a")
        self.assertContains(response, "/home/user/proj-b")
        self.assertContains(response, 'name="cwd"')
        self.assertContains(response, 'enctype="multipart/form-data"')
        self.assertContains(response, 'name="input_images"')
        self.assertContains(response, "data-new-session-image-input")
        self.assertContains(response, 'accept="image/png,image/jpeg,image/gif,image/webp"')
        self.assertContains(response, "function clearNewSessionImages()")
        self.assertContains(response, "function requireNewSessionPromptOrImages()")
        self.assertContains(response, "!commandPrompt && !newSessionHasImages()")

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_repo_dropdown_selects_saved_repo(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _seed_cookies(
            self.client,
            **{_LAST_SELECTED_REPO_COOKIE: "/home/user/proj-b"},
        )
        _setup_codex(mock_codex)
        mock_discover.return_value = [Path("/home/user/proj-a"), Path("/home/user/proj-b")]

        response = self.client.get(reverse("new_session"))

        self.assertContains(response, 'value="/home/user/proj-b" selected')
        self.assertNotContains(response, 'value="/home/user/proj-a" selected')

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_project_dropdown_selects_saved_repo_project(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _make_project(name="Project A", repo_path="/home/user/proj-a")
        project_b = _make_project(name="Project B", repo_path="/home/user/proj-b")
        _seed_cookies(
            self.client,
            **{_LAST_SELECTED_REPO_COOKIE: "/home/user/proj-b"},
        )
        _setup_codex(mock_codex)
        mock_discover.return_value = [Path("/home/user/proj-a"), Path("/home/user/proj-b")]

        response = self.client.get(reverse("new_session"))

        self.assertContains(response, f'value="{project_b.pk}" selected')
        self.assertContains(response, "data-new-session-repo-field hidden")

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_project_dropdown_keeps_saved_unprojected_repo_on_bare_option(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _make_project(name="Project A", repo_path="/home/user/proj-a")
        _seed_cookies(
            self.client,
            **{_LAST_SELECTED_REPO_COOKIE: "/home/user/bare"},
        )
        _setup_codex(mock_codex)
        mock_discover.return_value = [Path("/home/user/proj-a"), Path("/home/user/bare")]

        response = self.client.get(reverse("new_session"))

        self.assertContains(
            response, f'value="{session_settings._BARE_REPO_PROJECT_VALUE}" selected'
        )
        self.assertNotContains(response, "data-new-session-repo-field hidden")
        self.assertContains(response, 'value="/home/user/bare" selected')

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_project_dropdown_ignores_stale_saved_repo(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        project = _make_project(name="Project A", repo_path="/home/user/proj-a")
        _seed_cookies(
            self.client,
            **{_LAST_SELECTED_REPO_COOKIE: "/home/user/missing"},
        )
        _setup_codex(mock_codex)
        mock_discover.return_value = [Path("/home/user/proj-a")]

        response = self.client.get(reverse("new_session"))

        self.assertContains(response, f'value="{project.pk}" selected')
        self.assertContains(response, "data-new-session-repo-field hidden")

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_new_session_page_supports_super_enter_submit(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _setup_codex(mock_codex)
        mock_discover.return_value = [Path("/home/user/proj")]

        response = self.client.get(reverse("new_session"))

        self.assertContains(response, "data-new-session-submit")
        self.assertContains(response, "event.metaKey")
        self.assertContains(response, 'event.key === "Enter"')
        self.assertContains(response, 'event.getModifierState("Meta")')
        self.assertContains(response, 'event.getModifierState("OS")')
        self.assertContains(response, "requestSubmit(newSessionForm, newSessionSubmit)")

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_new_session_page_exposes_plan_slash_command(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _setup_codex(mock_codex)
        mock_discover.return_value = [Path("/home/user/proj")]

        response = self.client.get(reverse("new_session"))

        self.assertContains(response, 'class="slash-trigger"')
        self.assertContains(response, 'name="plan_mode"')
        self.assertContains(response, "Plan mode")
        self.assertNotContains(response, "data-slash-pr")
        self.assertContains(response, _PR_PROMPT)
        self.assertContains(response, _QA_PROMPT)
        self.assertContains(response, "parseNewSessionPlanCommand")
        self.assertContains(response, "parseNewSessionPrCommand")
        self.assertContains(response, "parseNewSessionQaCommand")
        self.assertContains(response, 'toLowerCase() !== "/plan"')
        self.assertContains(response, 'toLowerCase() !== "/qa"')
        self.assertNotContains(response, 'name="coding_agent"')
        self.assertContains(response, "Enter a prompt or attach an image.")

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_new_session_page_exposes_worktree_override(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _setup_codex(mock_codex)
        mock_discover.return_value = [Path("/home/user/proj")]
        _seed_cookies(self.client, **{_USE_WORKTREES_COOKIE: "true"})

        response = self.client.get(reverse("new_session"))

        self.assertContains(response, "Use worktree")
        self.assertContains(response, 'name="use_worktrees" value="false"')
        self.assertContains(
            response,
            'name="use_worktrees" value="true" data-new-session-use-worktree checked',
        )

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_title_rendering(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        """Per-row title display: user-set name wins, otherwise the preview's
        first line clipped to 80 chars, otherwise the bare id."""
        long_text = "x" * 200
        sessions = [
            _session("long-preview", preview=long_text),
            _session("multiline", preview="first line\nsecond line\nthird line"),
            _session("named", name="Short title", preview="ignored long preview " * 20),
            _session("bare-id"),
        ]
        _setup_codex(mock_codex, threads=sessions)
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        # Long preview is clipped, untruncated form must not leak.
        self.assertContains(response, "x" * 80 + "...")
        self.assertNotContains(response, "x" * 120)
        # Multiline preview collapses to first line.
        self.assertContains(response, "first line")
        self.assertNotContains(response, "second line")
        # Named row uses the name, not the preview.
        self.assertContains(response, "Short title")
        self.assertNotContains(response, "ignored long preview")
        # No name + no preview → fall back to the id.
        self.assertContains(response, ">bare-id<")

class ProjectViewTests(TestCase):
    def test_projects_default_to_follow_global_auto_pr(self) -> None:
        project = _make_project()

        self.assertEqual(project.auto_pr_mode, Project.AUTO_PR_FOLLOW_GLOBAL)

    @patch("hitch.main.repos.discover_repos")
    def test_new_project_form_lists_discovered_repos(self, mock_discover: MagicMock) -> None:
        mock_discover.return_value = [Path("/repo")]

        response = self.client.get(reverse("new_project"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Create project")
        self.assertContains(response, 'name="repo_path"')
        self.assertContains(response, "/repo")

    @patch("hitch.main.repos.git_common_dir", return_value=None)
    @patch("hitch.main.views.common.git_common_dir", return_value=None)
    def test_creatable_project_filter_loads_projects_once(
        self, _common_dir: MagicMock, _repo_common_dir: MagicMock
    ) -> None:
        _make_project(name="One", repo_path="/repo-one")
        _make_project(name="Two", repo_path="/repo-two")

        with self.assertNumQueries(1):
            creatable = settings_views._creatable_project_repos(
                ["/repo-one", "/new-one", "/new-two"]
            )

        self.assertEqual(creatable, ["/new-one", "/new-two"])

    @patch("hitch.main.repos.discover_repos")
    def test_new_project_form_hides_repos_that_already_have_projects(
        self, mock_discover: MagicMock
    ) -> None:
        _make_project(name="Existing")
        mock_discover.return_value = [Path("/repo"), Path("/other")]

        response = self.client.get(reverse("new_project"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, '<option value="/repo">/repo</option>', html=True)
        self.assertContains(response, '<option value="/other">/other</option>', html=True)

    @patch("hitch.main.views.common.git_common_dir")
    @patch("hitch.main.repos.discover_repos")
    def test_new_project_form_hides_repos_with_existing_git_common_dir(
        self, mock_discover: MagicMock, mock_common_dir: MagicMock
    ) -> None:
        _make_project(
            name="Existing",
            git_common_dir="/repo/.git",
        )
        mock_discover.return_value = [Path("/repo-worktree")]
        mock_common_dir.return_value = Path("/repo/.git")

        response = self.client.get(reverse("new_project"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "/repo-worktree")
        self.assertContains(response, "No git repositories without projects")

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
                    "extra_system_prompt": "x"
                    * (settings_cookies._EXTRA_SYSTEM_PROMPT_MAX_LEN + 1),
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
    def test_rejects_invalid_project_posts(
        self, mock_discover: MagicMock, mock_codex: MagicMock
    ) -> None:
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
    def test_schedule_runs_inline_under_testing(
        self, mock_refresh: MagicMock
    ) -> None:
        session_stage_refresh._schedule_pr_stage_refresh("sess-1")
        mock_refresh.assert_called_once_with("sess-1")

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

    @patch("hitch.main.sessions.session_stage_refresh._schedule_pr_stage_refresh")
    @patch(
        "hitch.main.sessions.session_stage_refresh.pr_qa.pr_snapshot_stage_refresh_due",
        return_value=True,
    )
    @patch("hitch.main.sessions.session_stage_refresh._pr_snapshot_for_rollout_path")
    def test_cached_pr_row_drops_refreshing_when_budget_exhausted(
        self,
        mock_snapshot: MagicMock,
        _mock_due: MagicMock,
        mock_schedule: MagicMock,
    ) -> None:
        # A cached PR row whose refresh is due must only render data-refreshing
        # when a refresh was actually scheduled; otherwise rows beyond the
        # per-render budget keep _stage_refresh_script reloading forever.
        mock_snapshot.return_value = {"url": "https://github.com/cberner/hitch/pull/94"}
        rollout_state = _RolloutFileState(path=Path("/tmp/rollout.jsonl"), mtime_ns=1)
        session = {"cwd": "/repo", "stage_pr_refresh_attempted_at": None}

        _stage, _snap, remaining, refreshing = (
            session_stage_refresh._stage_from_cached_session_row(
                "sess-budget",
                session,
                rollout_state=rollout_state,
                cached_stage=session_stage.PR,
                pr_stage_refreshes_remaining=1,
            )
        )
        self.assertTrue(refreshing)
        self.assertEqual(remaining, 0)
        mock_schedule.assert_called_once_with("sess-budget")

        mock_schedule.reset_mock()

        _stage, _snap, remaining, refreshing = (
            session_stage_refresh._stage_from_cached_session_row(
                "sess-exhausted",
                session,
                rollout_state=rollout_state,
                cached_stage=session_stage.PR,
                pr_stage_refreshes_remaining=0,
            )
        )
        self.assertFalse(refreshing)
        self.assertEqual(remaining, 0)
        mock_schedule.assert_not_called()


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
    def test_rapid_archive_keeps_every_row_undoable(
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
                # Archive POSTs always succeed in this test; the 5s finalize
                # timers never fire within the test window.
                page.evaluate(
                    "() => { window.fetch = () => Promise.resolve({ ok: true }); }"
                )
                self.assertEqual(
                    page.evaluate(
                        "() => document.querySelectorAll("
                        "'[data-session-archive-form]').length"
                    ),
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
                page.wait_for_function(
                    "document.querySelectorAll("
                    "'[data-session-row].pending-archive').length === 2"
                )
                self.assertFalse(
                    page.evaluate(
                        "() => document.querySelector('[data-archive-toast]').hidden"
                    )
                )

                undo = "() => document.querySelector('[data-archive-undo]').click()"
                # First Undo restores the most recently archived row; the toast
                # stays up because the other row's grace period is still open.
                page.evaluate(undo)
                page.wait_for_function(
                    "document.querySelectorAll("
                    "'[data-session-row].pending-archive').length === 1"
                )
                self.assertFalse(
                    page.evaluate(
                        "() => document.querySelector('[data-archive-toast]').hidden"
                    )
                )
                # Second Undo restores the earlier row -- the case the single-slot
                # implementation dropped on the floor.
                page.evaluate(undo)
                page.wait_for_function(
                    "document.querySelectorAll("
                    "'[data-session-row].pending-archive').length === 0"
                )
                self.assertTrue(
                    page.evaluate(
                        "() => document.querySelector('[data-archive-toast]').hidden"
                    )
                )
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
                page.wait_for_function(
                    "document.querySelectorAll("
                    "'[data-session-row].pending-archive').length === 2"
                )

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
                page.evaluate(
                    "() => { window.fetch = () => Promise.resolve({ ok: false }); }"
                )
                # Submitting an archived row's form takes the unarchive branch.
                page.evaluate(
                    "() => document.querySelector("
                    "\"[data-session-archive-url*='arch-1'] "
                    "[data-session-archive-form]\").requestSubmit()"
                )
                page.wait_for_function(
                    "() => { const t = document.querySelector("
                    "'[data-archive-error-toast]');"
                    " return t && !t.hidden && document.querySelector("
                    "'[data-archive-error-text]').textContent"
                    ".includes('Couldn'); }"
                )
                # No successful-archive notice, and the row stays archived.
                self.assertTrue(
                    page.evaluate(
                        "() => document.querySelector('[data-archive-toast]').hidden"
                    )
                )
                self.assertEqual(
                    page.evaluate(
                        "() => document.querySelector("
                        "\"[data-session-archive-url*='arch-1']\")"
                        ".dataset.sessionArchived"
                    ),
                    "true",
                )
            finally:
                browser.close()


class ThreadListSortTests(TestCase):
    """`_thread_list_page` must sort SDK threads with heterogeneous timestamps.

    Regression guard: threads can carry a datetime, an epoch int/float, or no
    `updated_at` at all. Sorting that mix directly (or a datetime against the
    default 0) raises TypeError and 500s the session list, so the sort key is
    normalized through `updated_at_seconds`.
    """

    def test_sorts_mixed_and_absent_updated_at_without_crashing(self) -> None:
        threads = [
            SimpleNamespace(id="missing"),  # no updated_at attribute at all
            SimpleNamespace(id="epoch", updated_at=1_700_000_000),
            SimpleNamespace(id="dt", updated_at=datetime(2025, 1, 2, tzinfo=UTC)),
        ]
        codex = SimpleNamespace(
            thread_list=lambda **kwargs: SimpleNamespace(
                data=list(threads), next_cursor=""
            )
        )

        page = session_list._thread_list_page(
            cast(Any, codex), archived=False, cursor=""
        )

        # Newest first: the 2025 datetime, then the 2023 epoch, then the
        # timestampless thread (treated as oldest).
        self.assertEqual([thread.id for thread in page.threads], ["dt", "epoch", "missing"])


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
        return render_to_string(
            "_usage_sections.html", {"lifetime_usage": lifetime_usage}
        )

    def test_only_charted_tiles_are_interactive(self) -> None:
        html = self._render(sessions_chart=True, system_chart=False)
        # The charted Sessions tile is an expandable button; the chartless
        # system tile is a plain div with no button affordances.
        self.assertEqual(html.count('class="lifetime-stat" role="button"'), 1)
        self.assertEqual(html.count('class="lifetime-stat">'), 1)

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
    def test_csrf_helper_falls_back_to_form_input(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
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
    def test_postform_headers_and_xhr_opt_in(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
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
                    self.assertEqual(
                        call["contentType"], "application/x-www-form-urlencoded"
                    )
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
    def test_relative_from_now(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
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

    @patch("hitch.main.repos.discover_repos", return_value=[])
    @patch("hitch.main.views.common.Codex")
    def test_index_refreshes_relative_timestamps(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        _setup_codex(mock_codex, threads=[])
        html = self.client.get(reverse("index")).content.decode()
        script_marker = """<script>
        (function () {
            const absolute"""
        timestamp = 1_700_000_000
        instrumentation = f"""
        <time data-updated-at="{timestamp}"></time>
        <script>
            window.__testNow = {timestamp * 1000 + 10_000};
            Date.now = () => window.__testNow;
            window.__testIntervals = [];
            window.setInterval = (callback, delay) => {{
                window.__testIntervals.push({{ callback, delay }});
                return window.__testIntervals.length;
            }};
        </script>
        {script_marker}"""
        html = html.replace(script_marker, instrumentation, 1)

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
                        const el = document.querySelector(
                            "time[data-updated-at='1700000000']");
                        const before = el.textContent;
                        window.__testNow += 5 * 60 * 1000;
                        const timer = window.__testIntervals.find(
                            (candidate) => candidate.delay === 30_000);
                        timer.callback();
                        return { before, after: el.textContent, delay: timer.delay };
                    }
                    """
                )
                self.assertIn("just now", result["before"])
                self.assertIn("minute", result["after"])
                self.assertEqual(result["delay"], 30_000)
            finally:
                browser.close()
