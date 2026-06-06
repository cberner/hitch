"""View-layer tests: index, new_session, send_message, set_session_name,
session_stream.

Shared helpers configure the Codex mock and seed signed cookies so each
test stays focused on the behavior under examination.
"""

import base64
import html
import json
import os
import re
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast, override
from unittest.mock import MagicMock, call, patch

from django.contrib.auth import get_user_model
from django.core import signing
from django.core.exceptions import SuspiciousOperation
from django.core.files.uploadedfile import SimpleUploadedFile
from django.db import IntegrityError, OperationalError, connection
from django.db.migrations.executor import MigrationExecutor
from django.http import HttpResponse
from django.test import (
    Client,
    RequestFactory,
    TestCase,
    TransactionTestCase,
    override_settings,
)
from django.test.utils import CaptureQueriesContext
from django.urls import reverse
from django.utils import timezone
from openai_codex import Codex
from openai_codex.errors import AppServerError, InvalidRequestError, MethodNotFoundError
from openai_codex.generated.v2_all import (
    GetAccountRateLimitsResponse,
    SortDirection,
    ThreadSortKey,
    ThreadSource,
)

from hitch.main import (
    codex_pool,
    coding_agents,
    demo,
    session_index,
    session_stage,
    streaming,
    system_agents,
    views,
)
from hitch.main import (
    rollout as rollout_module,
)
from hitch.main.models import (
    ApprovalRequest,
    ArchivedSessionTokenUsage,
    AutonomousGoal,
    CodexInstance,
    Project,
    ProposedSession,
    SessionDemo,
    SessionIndexSyncState,
    SessionMetadata,
    SystemAgentRun,
    SystemWorkflow,
    UserInputRequest,
    UserSettings,
)
from hitch.main.worktrees import (
    ManagedWorktree,
    WorktreeCleanupError,
    WorktreeCreationError,
)

_PNG_BYTES = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
_JPEG_BYTES = b"\xff\xd8\xff\xe0JFIF"
_GIF_BYTES = b"GIF89a\x01\x00\x01\x00"
_WEBP_BYTES = b"RIFF\x0c\x00\x00\x00WEBPVP8 "

_SHOW_ARCHIVED_COOKIE = "hitch_show_archived_sessions"
_MODEL_COOKIE = "hitch_model"
_EXTRA_SYSTEM_PROMPT_COOKIE = "hitch_extra_system_prompt"
_USE_WORKTREES_COOKIE = "hitch_use_worktrees"
_AUTO_PR_COOKIE = "hitch_auto_pr"
_AUTO_QA_COOKIE = "hitch_auto_qa"
_QA_PANEL_COOKIE = "hitch_qa_panel"
_SPEC_CRITIC_COOKIE = "hitch_spec_critic"
_WEB_SEARCH_COOKIE = "hitch_web_search_mode"
_LAST_SELECTED_REPO_COOKIE = "hitch_last_selected_repo"
_SELECTED_PROJECT_COOKIE = "hitch_selected_project_id"
_VISIBLE_SESSION_PROJECTS_COOKIE = "hitch_visible_session_project_ids"
_SHOW_NO_PROJECT_SESSIONS_COOKIE = "hitch_show_no_project_sessions"
_CODING_AGENT_COOKIE = "hitch_coding_agent"
_ENABLE_MEMORIES_COOKIE = "hitch_enable_memories"
_PR_PROMPT = (
    "Rebase on the default branch, clean it up, and then open a PR"
)
_QA_PROMPT = "Run the QA agent on the current diff and fix anything it finds"


class _FailingUploadWriter:
    def __init__(self, fd: int) -> None:
        self.fd = fd

    def __enter__(self) -> "_FailingUploadWriter":
        return self

    def __exit__(self, *_args: object) -> None:
        os.close(self.fd)

    def write(self, _chunk: bytes) -> None:
        raise OSError("disk full")


class _UnreadableUpload(SimpleUploadedFile):
    @override
    def read(self, *_args: object, **_kwargs: object) -> bytes:
        raise OSError("/tmp/private/screen.png")


def _rollout_line(
    line_type: str,
    payload: dict[str, object],
    *,
    timestamp: str = "2025-01-05T12:00:00Z",
) -> str:
    return json.dumps({"timestamp": timestamp, "type": line_type, "payload": payload})


def _token_count_line(
    *,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    reasoning_output_tokens: int = 0,
    context_tokens: int = 0,
    model_context_window: int = 0,
    timestamp: str = "2025-01-05T12:00:00Z",
) -> str:
    return _rollout_line(
        "event_msg",
        {
            "type": "token_count",
            "info": {
                "total_token_usage": {
                    "input_tokens": input_tokens,
                    "cached_input_tokens": cached_input_tokens,
                    "output_tokens": output_tokens,
                    "reasoning_output_tokens": reasoning_output_tokens,
                    "total_tokens": total_tokens,
                },
                "last_token_usage": {
                    "total_tokens": context_tokens,
                },
                "model_context_window": model_context_window,
            },
        },
        timestamp=timestamp,
    )


def _make_rollout(
    testcase: TestCase, lines: list[str], *, archived: bool = False
) -> Path:
    temp_dir = tempfile.TemporaryDirectory()
    testcase.addCleanup(temp_dir.cleanup)
    parent = Path(temp_dir.name)
    if archived:
        parent = parent / "archived_sessions"
        parent.mkdir()
    path = parent / "rollout.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_codex_home_rollout(
    codex_home: str | Path, thread_id: str, lines: list[str]
) -> Path:
    rollout_dir = Path(codex_home) / "sessions" / "2025" / "01" / "05"
    rollout_dir.mkdir(parents=True)
    path = rollout_dir / f"rollout-2025-01-05T12-00-00-{thread_id}.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _basic_session_rollout_lines(user_message: str, assistant_text: str) -> list[str]:
    return [
        _rollout_line(
            "event_msg",
            {"type": "user_message", "message": user_message},
        ),
        _rollout_line(
            "response_item",
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": assistant_text}],
                "phase": "final_answer",
            },
        ),
    ]


def _due_pr_monitor_state(
    *, pr_url: str, repo: str, pr_number: int, now: datetime
) -> dict[str, object]:
    return {
        system_agents._PR_HANDOFF_STATE_KEY: {
            "url": pr_url,
            "repository_full_name": repo,
            "pr_number": pr_number,
            "state": "open",
            "merged": False,
        },
        system_agents._PR_PENDING_CHECKS_STATE_KEY: 1,
        system_agents._PR_MONITOR_BACKOFF_STATE_KEY: {
            "reason": "pending_gates",
            "scheduled_at": int(now.timestamp()) - 301,
            "next_attempt_at": int(now.timestamp()) - 1,
            "delay_seconds": 300,
        },
    }


def _merged_pr_monitor_observation(
    *, pr_url: str, repo: str, pr_number: int
) -> dict[str, object]:
    return {
        "status": "terminal",
        "summary": "PR was merged.",
        "feedback": "",
        "pr": {
            "url": pr_url,
            "repository_full_name": repo,
            "pr_number": pr_number,
            "state": "closed",
            "merged": True,
            "merged_at": "2026-06-05T05:20:00Z",
        },
        "blockers": [],
    }


def _seed_usage_metadata(
    thread_id: str,
    *,
    path: str | Path = "",
    thread_source: str = "",
    mark_index_complete: bool = True,
) -> SessionMetadata:
    if mark_index_complete:
        session_index.mark_synced(archived=False, complete=True)
        session_index.mark_synced(archived=True, complete=True)
    return SessionMetadata.objects.create(
        thread_id=thread_id,
        codex_path=str(path),
        codex_thread_source=thread_source,
        codex_updated_at=datetime(2025, 1, 5, tzinfo=UTC),
    )


def _cache_token_usage(
    thread_id: str,
    *,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    path: str | Path = "",
    daily_usage: dict[str, dict[str, int]] | None = None,
    usage_logic_version: int = views._TOKEN_USAGE_LOGIC_VERSION,
) -> ArchivedSessionTokenUsage:
    rollout_path = str(path) if path else ""
    rollout_mtime_ns = Path(path).stat().st_mtime_ns if path else 0
    if daily_usage is None:
        daily_usage = {
            "2025-01-05": {
                "input": max(input_tokens - cached_input_tokens, 0),
                "output": output_tokens,
                "cached": cached_input_tokens,
            }
        }
    return ArchivedSessionTokenUsage.objects.create(
        thread_id=thread_id,
        rollout_path=rollout_path,
        rollout_mtime_ns=rollout_mtime_ns,
        usage_logic_version=usage_logic_version,
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        daily_usage=daily_usage,
    )


def _setup_codex(
    mock_codex: MagicMock,
    *,
    threads: list[Any] | None = None,
    archived_threads: list[Any] | None = None,
    models: list[Any] | None = None,
) -> MagicMock:
    """Configure the Codex mock with ``thread_list`` and ``models``.

    The index view reads both active and, when enabled, archived thread
    lists. Also stubs ``_client.request`` to raise
    MethodNotFound so the rate-limits fetch falls through its
    unsupported-endpoint branch — tests that care set their own value."""
    ctx: MagicMock = mock_codex.return_value.__enter__.return_value

    def thread_list(*, archived: bool | None = None, **_: Any) -> SimpleNamespace:
        data = archived_threads if archived else threads
        return SimpleNamespace(data=data or [])

    ctx.thread_list.side_effect = thread_list
    ctx.models.return_value.data = models or []
    ctx._client.request.side_effect = MethodNotFoundError(
        -32601, "method not found", None
    )
    return ctx


def _run_borrowed_with(
    client: Any,
) -> Callable[..., object]:
    def side_effect(
        _factory: object, operation: Callable[[Any], object], **_kwargs: object
    ) -> object:
        return operation(client)

    return side_effect


def _make_model(model_id: str, *, is_default: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        id=model_id,
        display_name=model_id,
        is_default=is_default,
        default_reasoning_effort=SimpleNamespace(value="medium"),
        supported_reasoning_efforts=[
            SimpleNamespace(reasoning_effort=SimpleNamespace(value=v), description=v)
            for v in ("low", "medium", "high")
        ],
    )


def _sign(name: str, value: str) -> str:
    return signing.get_cookie_signer(salt=name).sign(value)


def _encode_extra_system_prompt(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode()).decode("ascii")


def _seed_cookies(client: Client, **values: str) -> None:
    for name, value in values.items():
        client.cookies[name] = _sign(name, value)


def _cookie_value(response: object, name: str) -> str:
    raw = response.cookies[name].value  # type: ignore[attr-defined]
    return signing.get_cookie_signer(salt=name).unsign(raw)


def _session(
    session_id: str = "sess",
    *,
    name: str | None = None,
    preview: str = "",
    cwd: str = "/repo",
    path: str | None = None,
    updated_at: int = 1,
    thread_source: ThreadSource | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=session_id,
        name=name,
        preview=preview,
        cwd=cwd,
        path=path,
        updated_at=updated_at,
        thread_source=thread_source,
    )


class SessionDetailFastPathTests(TestCase):
    @patch("hitch.main.views._start_models_refresh_thread")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views._start_models_refresh_thread")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views._start_models_refresh_thread")
    @patch("hitch.main.views.Codex")
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
                recovered = views._session_detail_metadata("abc-def")

        self.assertEqual(recovered, metadata)
        metadata.refresh_from_db()
        self.assertEqual(metadata.codex_path, "")

    @patch("hitch.main.views._start_models_refresh_thread")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views._start_models_refresh_thread")
    @patch("hitch.main.views.Codex")
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
            "hitch.main.rollout._load_rollout_lines",
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

    @patch("hitch.main.views._start_models_refresh_thread")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views._start_models_refresh_thread")
    @patch("hitch.main.views.Codex")
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
                "hitch.main.views.rollout.session_detail_data",
                side_effect=ValueError("unexpected rollout shape"),
            ),
            patch("hitch.main.views._models_for_plan_mode_fallback", return_value=[]),
            patch("hitch.main.views._entries_for", return_value=sdk_entries),
        ):
            response = self.client.get(
                reverse("session", kwargs={"session_id": "schema-drift"})
            )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "SDK answer")
        codex._client.thread_resume.assert_called_once_with("schema-drift")

    @patch("hitch.main.views._start_models_refresh_thread")
    @patch("hitch.main.views.Codex")
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
            "hitch.main.rollout._load_rollout_lines",
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

    @patch("hitch.main.views._start_models_refresh_thread")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views._start_models_refresh_thread")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views._start_models_refresh_thread")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.system_agents._gh_pr_view")
    @patch("hitch.main.views._start_models_refresh_thread")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.system_agents._pr_monitor_observation_from_gh")
    @patch("hitch.main.views._start_models_refresh_thread")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.system_agents._gh_pr_view")
    @patch("hitch.main.views._start_models_refresh_thread")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views._start_models_refresh_thread")
    @patch("hitch.main.views.Codex")
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

        real_session_detail_data = views._session_detail_data_for_metadata_resume

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
            "hitch.main.views._session_detail_data_for_metadata_resume",
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

    @patch("hitch.main.views._start_models_refresh_thread")
    @patch("hitch.main.views.Codex")
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

        with patch("hitch.main.codex_pool.worker_is_alive", return_value=True):
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

    @patch("hitch.main.views._schedule_pr_stage_refresh")
    @patch("hitch.main.views._start_models_refresh_thread")
    @patch("hitch.main.views.Codex")
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

        with patch("hitch.main.codex_pool.worker_is_alive", return_value=True):
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

    @patch("hitch.main.views._start_models_refresh_thread")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views._start_models_refresh_thread")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views._start_models_refresh_thread")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views._start_models_refresh_thread")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views._start_models_refresh_thread")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.Codex")
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

        with patch("hitch.main.codex_pool.worker_is_alive", return_value=True):
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

        self.assertTrue(views._entries_await_plan_approval(entries))

    def test_agent_answer_clears_pending_plan(self) -> None:
        entries = [
            {"kind": "user", "text": "Plan it"},
            {"kind": "plan", "text": "# Plan"},
            {"kind": "user", "text": "Implement the plan"},
            {"kind": "agent", "text": "Done"},
        ]

        self.assertFalse(views._entries_await_plan_approval(entries))

    def test_only_latest_pending_plan_is_actionable(self) -> None:
        entries = [
            {"kind": "user", "text": "Plan it"},
            {"kind": "plan", "text": "# Old Plan"},
            {"kind": "user", "text": "Revise"},
            {"kind": "plan", "text": "# Current Plan"},
        ]

        views._mark_pending_plan_actions(entries)

        self.assertFalse(entries[1]["show_plan_actions"])
        self.assertTrue(entries[3]["show_plan_actions"])

    def test_agent_answer_clears_plan_actions(self) -> None:
        entries = [
            {"kind": "user", "text": "Plan it"},
            {"kind": "plan", "text": "# Plan"},
            {"kind": "user", "text": "Implement the plan"},
            {"kind": "agent", "text": "Done"},
        ]

        views._mark_pending_plan_actions(entries)

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

        self.assertEqual(views._user_message_text(item), "[image]")

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

        trimmed = views._trim_in_progress_turn(entries, active)

        self.assertEqual(
            trimmed,
            [
                {"kind": "user", "text": "before"},
                {"kind": "agent", "text": "before reply"},
            ],
        )
        self.assertEqual(views._pending_user_prompt(active), "initial prompt")


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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_cached_session_list_does_not_call_thread_list(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_cached_session_list_shows_derived_stage_badge(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_cached_session_list_shows_pr_number_in_cached_pr_badge(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")
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

    @patch("hitch.main.system_agents._gh_pr_view")
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_cached_session_list_refreshes_cached_pr_stage_to_done_merged(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_gh_pr_view: MagicMock,
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")
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

    @patch("hitch.main.system_agents._pr_monitor_observation_from_gh")
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_cached_session_list_refreshes_due_pr_monitor_backoff_to_done_merged(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_observe: MagicMock,
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")
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

    @patch("hitch.main.views._schedule_pr_stage_refresh")
    @patch("hitch.main.system_agents.pr_snapshot_stage_refresh_due", return_value=True)
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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
        client.thread_list.side_effect = AppServerError("thread list unavailable")
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

    @patch("hitch.main.system_agents._gh_pr_view")
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_cached_session_list_refreshes_uncached_pr_snapshot_to_done_merged(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_gh_pr_view: MagicMock,
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_cached_session_list_omits_stale_pr_number_for_new_workflow(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_cached_session_list_ignores_locked_stage_cache_update(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")
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
            "hitch.main.views._update_cached_stage",
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

    @patch("hitch.main.views.codex_pool.worker_is_alive", return_value=True)
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_cached_session_list_active_instance_overrides_terminal_stage(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        _worker_is_alive: MagicMock,
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_cached_session_list_flags_pending_spec_critic_input(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")
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
            agent_kind=system_agents.SPEC_RISK_AGENT_KIND,
            display_author=system_agents.SPEC_CRITIC_DISPLAY_AUTHOR,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.SPEC_RISK_AGENT_KIND,
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
            '<span class="stage-badge" data-tone="warning">Waiting for User</span>',
        )
        metadata.refresh_from_db()
        self.assertEqual(metadata.derived_stage, "implementation")
        mock_codex.assert_not_called()
        client.thread_list.assert_not_called()

    @patch("hitch.main.views.codex_pool.worker_is_alive", return_value=True)
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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
        client.thread_list.side_effect = AppServerError("thread list unavailable")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_cached_session_list_ignores_stage_cache_without_rollout(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_cached_session_list_survives_malformed_stage_rollout(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_cached_session_list_ignores_stale_completed_pr_workflow(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_cached_session_list_uses_terminal_cache_after_stale_pr_workflow(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_cached_session_list_running_pr_workflow_without_handoff_ignores_old_pr(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_cached_session_list_running_pr_workflow_uses_terminal_handoff_pr(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")
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

    @patch("hitch.main.system_agents._gh_pr_view")
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_cached_session_list_refreshes_ready_pr_to_done_merged(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_gh_pr_view: MagicMock,
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")
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

    @patch("hitch.main.system_agents._gh_pr_view")
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_cached_session_list_caps_ready_pr_refreshes(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_gh_pr_view: MagicMock,
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")
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

    @patch("hitch.main.system_agents.logger")
    @patch("hitch.main.system_agents._gh_pr_view")
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_cached_session_list_backs_off_failed_ready_pr_refresh(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_gh_pr_view: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")
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
        mock_gh_pr_view.side_effect = system_agents._GhPrOpenError("gh unavailable")

        first_response = self.client.get(reverse("index"))
        second_response = self.client.get(reverse("index"))

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(second_response.status_code, 200)
        workflow.refresh_from_db()
        self.assertIn(system_agents._PR_STAGE_REFRESH_STATE_KEY, workflow.state)
        self.assertEqual(workflow.step, system_agents.STEP_PR_READY)
        mock_gh_pr_view.assert_called_once()
        mock_logger.exception.assert_called_once()
        mock_codex.assert_not_called()
        client.thread_list.assert_not_called()

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_cached_session_list_prefers_newer_main_pr_over_stale_workflow(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_cached_session_list_prefers_newer_main_pr_state_over_workflow(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_cached_session_list_ignores_workflow_only_stale_pr_handoff(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_cached_session_list_keeps_server_created_pr_handoff(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_cached_hidden_system_flag_drives_main_and_system_lists(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_system_sessions_demo_upsert_keeps_main_session_visible(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_cached_session_order_uses_local_qa_activity_when_hidden_row_is_stale(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_cached_session_order_promotes_qa_activity_from_beyond_page_size(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")
        mock_discover.return_value = []
        now = datetime.now(UTC)
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=now,
            is_complete=True,
        )
        for index in range(views._SESSION_PAGE_SIZE):
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_index_cursor_keeps_later_pages_stable_when_rows_move(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_system_sessions_pages_before_helper_lookups_and_keeps_cursor_stable(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")
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
                "hitch.main.views._system_agent_runs_by_thread_id",
                return_value={},
            ) as runs_by_thread_id,
            patch(
                "hitch.main.views._system_agent_instances_by_thread_id",
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_system_sessions_cursor_keeps_same_second_rows_stable(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_system_sessions_ignores_invalid_index_cursor_timestamps(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_system_sessions_excludes_accepted_visible_system_thread(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")
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

    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.Codex")
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

        views._refresh_usage_session_index_best_effort(
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_complete_empty_session_index_serves_cached_empty_state(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_stale_complete_session_index_serves_cache_and_schedules_refresh(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        now = datetime.now(UTC)
        cached = _session("cached", name="Cached session", updated_at=2000)
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")
        mock_discover.return_value = []
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=now - timedelta(minutes=5),
            is_complete=True,
        )
        session_index.upsert_thread(cached, projects=[])

        with (
            patch("hitch.main.views._start_models_refresh_thread"),
            patch(
                "hitch.main.views._start_usage_session_index_refresh_thread"
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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
            patch("hitch.main.views._start_models_refresh_thread"),
            patch(
                "hitch.main.views._start_usage_session_index_refresh_thread"
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_stale_complete_session_index_does_not_start_codex(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        now = datetime.now(UTC)
        mock_codex.side_effect = AppServerError("codex unavailable")
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
            patch("hitch.main.views._start_models_refresh_thread"),
            patch(
                "hitch.main.views._start_usage_session_index_refresh_thread"
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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
            patch("hitch.main.views._start_models_refresh_thread"),
            patch(
                "hitch.main.views._start_usage_session_index_refresh_thread"
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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
            patch("hitch.main.views._rate_limits_for_usage_context", return_value=None),
            patch(
                "hitch.main.views._start_usage_session_index_refresh_thread"
            ) as start_index_refresh,
            patch("hitch.main.views._start_usage_token_refresh_thread"),
            patch("hitch.main.views._start_models_refresh_thread"),
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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
            patch("hitch.main.views._start_models_refresh_thread"),
            patch("hitch.main.views._start_usage_session_index_refresh_thread"),
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_system_sessions_backfill_missing_cached_metadata(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _setup_codex(mock_codex)
        mock_discover.return_value = []
        now = datetime.now(UTC)
        project = Project.objects.create(name="Repo", repo_path="/repo")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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
        client.thread_list.side_effect = AppServerError("thread list unavailable")

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Session 0")
        client.thread_list.assert_not_called()

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_hides_project_banner_when_project_exists(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        Project.objects.create(name="Hitch", repo_path="/repo")
        _setup_codex(mock_codex)
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "Create a project to group sessions")

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.Codex")
    def test_untracked_system_session_resume_error_is_not_404(
        self, mock_codex: MagicMock
    ) -> None:
        session_id = "00000000-0000-0000-0000-000000000001"
        client = _setup_codex(mock_codex)
        client._client.thread_resume.side_effect = AppServerError("app server down")

        with self.assertRaises(AppServerError):
            self.client.get(
                reverse("system_session", kwargs={"session_id": session_id})
            )

    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.system_agents.accepted_visible_system_thread_ids")
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.Codex")
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
            base_instructions="base " * 2000,
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
            base_instructions="base " * 2000,
        )

        with CaptureQueriesContext(connection) as captured:
            runs_by_thread_id = views._system_agent_runs_by_thread_id(["qa-thread"])
            instances_by_thread_id = views._system_agent_instances_by_thread_id(
                ["instance-only-thread"]
            )
            run = runs_by_thread_id["qa-thread"]
            instance = instances_by_thread_id["instance-only-thread"]
            self.assertEqual(
                views._system_agent_run_label(run), system_agents.QA_DISPLAY_AUTHOR
            )
            self.assertEqual(
                views._system_agent_status(run), SystemAgentRun.STATUS_COMPLETED
            )
            self.assertEqual(
                views._system_agent_run_label(None, instance),
                system_agents.AUTONOMOUS_GOAL_DISPLAY_AUTHOR,
            )
            self.assertEqual(
                views._system_agent_status(None, instance),
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
            {"prompt", "developer_instructions", "base_instructions"}.issubset(
                run.instance.get_deferred_fields()
            )
        )
        self.assertTrue(
            {"prompt", "developer_instructions", "base_instructions"}.issubset(
                instance.get_deferred_fields()
            )
        )

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_index_links_to_new_session_page_instead_of_dialog(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _setup_codex(mock_codex)
        mock_discover.return_value = ["/repo"]

        response = self.client.get(reverse("index"))

        self.assertContains(response, f'href="{reverse("new_session")}"')
        self.assertNotContains(response, "data-new-session-dialog")
        self.assertNotContains(response, "keyboard-adjusted")

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_index_keeps_pending_archive_rows_hidden(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _setup_codex(mock_codex, threads=[_session("abc", name="Session")])
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertContains(response, ".session.pending-archive {")
        self.assertContains(response, "visibility: hidden;")

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

        updated_at_by_main_thread = views._qa_activity_updated_at_by_main_thread_id(
            [_session("active", updated_at=1000)],
            system_agents.hidden_thread_ids(),
        )

        self.assertEqual(updated_at_by_main_thread, {"active": 2000})

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_selected_project_filters_sessions(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        other = Project.objects.create(name="Other", repo_path="/other")
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

    @patch("hitch.main.views.system_agents.hidden_thread_ids")
    @patch("hitch.main.views.Codex")
    def test_warm_index_filters_system_sessions_without_hidden_id_scan(
        self, mock_codex: MagicMock, mock_hidden_thread_ids: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_visible_projects_filter_sessions(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        other = Project.objects.create(name="Other", repo_path="/other")
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
        "hitch.main.views._visible_session_project_ids_cookie_fits",
        return_value=False,
    )
    def test_visible_projects_rejects_oversized_guest_cookie(
        self, mock_cookie_fits: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")

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
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        other = Project.objects.create(name="Other", repo_path="/other")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_no_project_metadata_prevents_cwd_project_inference(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _setup_codex(mock_codex, threads=[_session("cleared", name="Cleared", cwd="/repo")])
        mock_discover.return_value = [Path("/repo")]
        SessionMetadata.objects.create(
            thread_id="cleared", cwd="/repo", project=None, project_cleared=True
        )
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "Cleared")

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.Codex")
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
            patch("hitch.main.views._start_usage_token_refresh_thread"),
            patch("hitch.main.views._start_models_refresh_thread"),
            patch("hitch.main.views._start_rate_limits_refresh_thread"),
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
            patch("hitch.main.views.rollout.latest_token_usage") as latest_usage,
            patch("hitch.main.views.rollout.token_usage_history") as usage_history,
            patch("hitch.main.views._start_usage_token_refresh_thread") as start_refresh,
            patch("hitch.main.views._start_models_refresh_thread"),
            patch("hitch.main.views._start_rate_limits_refresh_thread"),
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

        views._refresh_usage_token_cache_best_effort(
            [views._UsageTokenRefreshItem("archived", str(rollout_path))]
        )

        self.assertNotContains(response, "909,999")
        self.assertNotContains(response, "999,999")
        cache.refresh_from_db()
        self.assertEqual(cache.total_tokens, 999_999)
        self.assertEqual(cache.rollout_mtime_ns, 2_000_000_000)

    @patch("hitch.main.views.Codex")
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
            patch("hitch.main.views._RATE_LIMITS_CACHE_VALUE", None),
            patch("hitch.main.views._RATE_LIMITS_CACHE_HAS_VALUE", False),
            patch("hitch.main.views._RATE_LIMITS_CACHE_FETCHED_AT", None),
            patch("hitch.main.views._RATE_LIMITS_REFRESH_IN_FLIGHT", False),
            patch("hitch.main.views._start_models_refresh_thread"),
            patch(
                "hitch.main.views._start_rate_limits_refresh_thread"
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

        lifetime_usage = views._lifetime_token_usage_for_metadata([metadata])

        self.assertTrue(lifetime_usage["refresh_pending"])
        self.assertEqual(lifetime_usage["refresh_pending_count"], 1)
        self.assertEqual(lifetime_usage["total"]["input"], "0")
        self.assertEqual(lifetime_usage["total"]["output"], "0")
        self.assertEqual(lifetime_usage["total"]["cached"], "0")
        self.assertEqual(lifetime_usage["total"]["chart"], [])

    @patch("hitch.main.views.Codex")
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
                "hitch.main.views._start_usage_session_index_refresh_thread"
            ) as start_index_refresh,
            patch("hitch.main.views._start_usage_token_refresh_thread"),
            patch("hitch.main.views._start_models_refresh_thread"),
            patch("hitch.main.views._start_rate_limits_refresh_thread"),
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

    @patch("hitch.main.views.Codex")
    def test_usage_page_throttles_recent_incomplete_index_refresh(
        self, mock_codex: MagicMock
    ) -> None:
        session_index.mark_synced(archived=False, complete=False)
        session_index.mark_synced(archived=True, complete=False)
        client = _setup_codex(mock_codex)

        with (
            patch(
                "hitch.main.views._start_usage_session_index_refresh_thread"
            ) as start_index_refresh,
            patch("hitch.main.views._start_usage_token_refresh_thread"),
            patch("hitch.main.views._start_models_refresh_thread"),
            patch("hitch.main.views._start_rate_limits_refresh_thread"),
            self.captureOnCommitCallbacks(execute=True),
        ):
            response = self.client.get(reverse("usage"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "All sessions usage unavailable.")
        self.assertIsNone(response.context["lifetime_usage"])
        client.thread_list.assert_not_called()
        start_index_refresh.assert_not_called()

    @patch("hitch.main.views.Codex")
    def test_usage_page_renders_zero_usage_when_complete_index_is_empty(
        self, mock_codex: MagicMock
    ) -> None:
        session_index.mark_synced(archived=False, complete=True)
        session_index.mark_synced(archived=True, complete=True)
        client = _setup_codex(mock_codex)

        with (
            patch(
                "hitch.main.views._start_usage_session_index_refresh_thread"
            ) as start_index_refresh,
            patch("hitch.main.views._start_usage_token_refresh_thread") as start_tokens,
            patch("hitch.main.views._start_models_refresh_thread"),
            patch("hitch.main.views._start_rate_limits_refresh_thread"),
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

    @patch("hitch.main.views.Codex")
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
                "hitch.main.views._start_usage_session_index_refresh_thread"
            ) as start_index_refresh,
            patch("hitch.main.views._start_usage_token_refresh_thread"),
            patch("hitch.main.views._start_models_refresh_thread"),
            patch("hitch.main.views._start_rate_limits_refresh_thread"),
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

    @patch("hitch.main.views.Codex")
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
                "hitch.main.views._start_usage_session_index_refresh_thread"
            ) as start_index_refresh,
            patch("hitch.main.views._start_usage_token_refresh_thread") as start_tokens,
            patch("hitch.main.views._start_models_refresh_thread"),
            patch("hitch.main.views._start_rate_limits_refresh_thread"),
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

        self.assertIsNone(views._token_usage_snapshot_for(thread))

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

        snapshot = views._token_usage_snapshot_for(thread)
        assert snapshot is not None
        usage = snapshot["usage"]
        # Recomputed from the rollout, not served from the stale row.
        self.assertEqual(usage["input_tokens"], 100_000)
        self.assertEqual(usage["cached_input_tokens"], 80_000)
        self.assertEqual(usage["output_tokens"], 20_000)
        self.assertEqual(usage["total_tokens"], 120_000)
        cache = ArchivedSessionTokenUsage.objects.get(thread_id="archived")
        self.assertEqual(cache.total_tokens, 120_000)
        self.assertEqual(cache.usage_logic_version, views._TOKEN_USAGE_LOGIC_VERSION)

    def test_cached_token_usage_matches_rollout_state_requires_current_version(
        self,
    ) -> None:
        # The match check gates on path, mtime, AND logic version. A row with a
        # matching path+mtime but a stale logic version must not be treated as a
        # match, so a counting-logic bump forces a recompute even when the
        # rollout file is byte-for-byte unchanged.
        rollout_state = views._RolloutFileState(
            path=Path("/codex/archived/rollout.jsonl"), mtime_ns=1_234
        )
        current = ArchivedSessionTokenUsage(
            thread_id="t",
            rollout_path=str(rollout_state.path),
            rollout_mtime_ns=rollout_state.mtime_ns,
            usage_logic_version=views._TOKEN_USAGE_LOGIC_VERSION,
        )
        self.assertTrue(
            views._cached_token_usage_matches_rollout_state(current, rollout_state)
        )
        legacy = ArchivedSessionTokenUsage(
            thread_id="t",
            rollout_path=str(rollout_state.path),
            rollout_mtime_ns=rollout_state.mtime_ns,
            usage_logic_version=views._TOKEN_USAGE_LOGIC_VERSION - 1,
        )
        self.assertFalse(
            views._cached_token_usage_matches_rollout_state(legacy, rollout_state)
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
            usage_logic_version=views._TOKEN_USAGE_LOGIC_VERSION,
        )
        self.assertTrue(
            views._cached_token_usage_is_current_for_state(current, None)
        )
        legacy = ArchivedSessionTokenUsage(
            thread_id="t",
            rollout_path="",
            usage_logic_version=views._TOKEN_USAGE_LOGIC_VERSION - 1,
        )
        self.assertFalse(
            views._cached_token_usage_is_current_for_state(legacy, None)
        )

    def test_usage_token_cache_state_rejects_stale_version_pathless_rows(self) -> None:
        # The lifetime-aggregation usability check must also reject stale-version
        # rows on its no-path branches, so a legacy version-0 row does not keep
        # contributing pre-fix counts to the summed totals while a refresh is
        # pending.
        metadata = views._UsageTokenRefreshCandidate(
            thread_id="t", codex_path="", usage_last_checked_at=None
        )
        current = ArchivedSessionTokenUsage(
            thread_id="t",
            rollout_path="",
            usage_logic_version=views._TOKEN_USAGE_LOGIC_VERSION,
        )
        self.assertTrue(views._usage_token_cache_state(metadata, current).cache_usable)
        legacy = ArchivedSessionTokenUsage(
            thread_id="t",
            rollout_path="",
            usage_logic_version=views._TOKEN_USAGE_LOGIC_VERSION - 1,
        )
        self.assertFalse(views._usage_token_cache_state(metadata, legacy).cache_usable)

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

        snapshot = views._token_usage_snapshot_for(thread)
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
            views._non_cached_input_tokens(usage),
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
        from hitch.main import rollout

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
            snapshot = views._token_usage_snapshot_for(thread)

        assert snapshot is not None
        # The snapshot reflects the content actually parsed (pre-append).
        self.assertEqual(snapshot["usage"]["input_tokens"], 100)

        cache = ArchivedSessionTokenUsage.objects.get(thread_id="racing")
        self.assertEqual(cache.rollout_mtime_ns, pre_mtime)

        # The next read sees the mismatch and re-parses the appended file rather
        # than serving the stale cached numbers.
        refreshed = views._token_usage_snapshot_for(thread)
        assert refreshed is not None
        self.assertEqual(refreshed["usage"]["input_tokens"], 500)

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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
            "hitch.main.views._metadata_by_thread_id",
            wraps=views._metadata_by_thread_id,
        ) as metadata_by_thread:
            response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Active 0")
        self.assertContains(response, "Active 2")
        self.assertLessEqual(metadata_by_thread.call_count, 2)

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.Codex")
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

    def test_lifetime_human_token_formatter(self) -> None:
        self.assertEqual(views._format_human_token_count(-1), "0")
        self.assertEqual(views._format_human_token_count(999), "999")
        self.assertEqual(views._format_human_token_count(1_500_000), "1.5M")
        self.assertEqual(views._format_human_token_count(10_500_000), "11M")
        self.assertEqual(views._format_human_token_count(1_000_000_000), "1B")

    def test_lifetime_token_chart_formats_segments(self) -> None:
        self.assertEqual(views._format_lifetime_token_chart({}), [])
        self.assertEqual(views._format_lifetime_token_chart_axis({}), [])
        self.assertEqual(views._chart_segment_percent(0, 100), 0)
        self.assertEqual(views._chart_segment_percent(5, 0), 0)
        self.assertEqual(views._chart_segment_percent(1, 1_000), 0)
        self.assertEqual(
            views._format_lifetime_token_chart(
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
            views._format_lifetime_token_chart_axis(
                {
                    "2025-01-06": {"input": 50, "output": 50, "cached": 0},
                    "2025-01-05": {"input": 100, "output": 50, "cached": 50},
                }
            ),
            ["200", "100", "0"],
        )

    @patch("hitch.main.views.Codex")
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

        views._refresh_usage_token_cache_best_effort(
            [views._UsageTokenRefreshItem("archived", str(rollout_path))]
        )

        cache.refresh_from_db()
        self.assertEqual(
            cache.daily_usage,
            {"2025-01-05": {"input": 350, "output": 600, "cached": 50}},
        )

    @patch("hitch.main.views.Codex")
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
            patch("hitch.main.views.rollout.latest_token_usage") as latest_usage,
            patch("hitch.main.views.rollout.token_usage_history") as usage_history,
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
            patch("hitch.main.views.rollout.latest_token_usage") as latest_usage,
            patch("hitch.main.views.rollout.token_usage_history") as usage_history,
        ):
            response = self.client.get(reverse("usage"))

        self.assertEqual(response.status_code, 200)
        latest_usage.assert_not_called()
        usage_history.assert_not_called()
        client.thread_list.assert_not_called()

    @patch("hitch.main.views.Codex")
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
            patch("hitch.main.views._start_usage_token_refresh_thread") as start_refresh,
            patch("hitch.main.views._start_models_refresh_thread"),
            patch("hitch.main.views._start_rate_limits_refresh_thread"),
            patch(
                "hitch.main.views._rollout_path_from_value",
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

    @patch("hitch.main.views.Codex")
    def test_usage_page_uses_indexed_usage_when_session_list_fails(
        self, mock_codex: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")
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

    @patch("hitch.main.views.Codex")
    def test_usage_page_marks_usage_unavailable_until_initial_index_refresh_finishes(
        self, mock_codex: MagicMock
    ) -> None:
        client = _setup_codex(mock_codex)
        client.thread_list.side_effect = AppServerError("thread list unavailable")

        with (
            patch(
                "hitch.main.views._start_usage_session_index_refresh_thread"
            ) as start_index_refresh,
            patch("hitch.main.views._start_models_refresh_thread"),
            patch("hitch.main.views._start_rate_limits_refresh_thread"),
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

    @patch("hitch.main.views.Codex")
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
            patch("hitch.main.views._start_usage_token_refresh_thread") as start_refresh,
            patch("hitch.main.views._start_models_refresh_thread"),
            patch("hitch.main.views._start_rate_limits_refresh_thread"),
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
        views._refresh_usage_token_cache_best_effort(refresh_items)

        metadata.refresh_from_db()
        self.assertEqual(metadata.codex_path, str(rollout_path))
        cache = ArchivedSessionTokenUsage.objects.get(thread_id="local-session")
        self.assertEqual(cache.total_tokens, 1_000)

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
        )

        views._refresh_usage_token_cache_best_effort(
            [views._UsageTokenRefreshItem("stale", str(rollout_path))]
        )

        cache = ArchivedSessionTokenUsage.objects.get(thread_id="stale")
        self.assertEqual(cache.total_tokens, 0)
        self.assertEqual(cache.rollout_mtime_ns, 2_000_000_000)
        self.assertEqual(cache.daily_usage, {})

    @patch("hitch.main.views.Codex")
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
        )
        client = _setup_codex(mock_codex)
        client._client.thread_resume.side_effect = AppServerError("resume failed")

        views._refresh_usage_token_cache_best_effort(
            [views._UsageTokenRefreshItem("missing", "/nonexistent/rollout.jsonl")]
        )

        cache.refresh_from_db()
        self.assertEqual(cache.total_tokens, 1_000)
        metadata = SessionMetadata.objects.get(thread_id="missing")
        self.assertIsNotNone(metadata.usage_last_checked_at)

    def test_usage_refresh_marks_checked_rows_in_chunks(self) -> None:
        for index in range(5):
            _seed_usage_metadata(
                f"checked-{index}",
                mark_index_complete=False,
            )

        with (
            patch("hitch.main.views._USAGE_TOKEN_REFRESH_CHECKED_UPDATE_BATCH_SIZE", 2),
            CaptureQueriesContext(connection) as queries,
        ):
            views._mark_usage_token_refresh_checked_many(
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
        views._USAGE_TOKEN_REFRESH_IN_FLIGHT = False
        self.addCleanup(setattr, views, "_USAGE_TOKEN_REFRESH_IN_FLIGHT", False)
        thread = MagicMock()
        thread.start.side_effect = RuntimeError("thread limit")

        with (
            self.assertLogs("hitch.main.views", level="ERROR"),
            patch("hitch.main.views.threading.Thread", return_value=thread),
        ):
            views._start_usage_token_refresh_thread(
                [views._UsageTokenRefreshItem("thread", "")]
            )

        self.assertFalse(views._USAGE_TOKEN_REFRESH_IN_FLIGHT)

    def test_usage_refresh_thread_is_non_daemon_and_materializes_work(self) -> None:
        views._USAGE_TOKEN_REFRESH_IN_FLIGHT = False
        self.addCleanup(setattr, views, "_USAGE_TOKEN_REFRESH_IN_FLIGHT", False)
        thread = MagicMock()
        items = [
            views._UsageTokenRefreshItem("thread-a", ""),
            views._UsageTokenRefreshItem("thread-b", ""),
        ]

        with patch(
            "hitch.main.views.threading.Thread", return_value=thread
        ) as thread_cls:
            views._start_usage_token_refresh_thread(iter(items))

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
        candidates = views._usage_token_refresh_candidates(rows)

        with patch("hitch.main.views._USAGE_TOKEN_REFRESH_BATCH_SIZE", 2):
            views._refresh_usage_token_cache_best_effort(candidates)

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
        first_batch = views._usage_token_refresh_items(rows, {})
        first_batch_ids = [item.thread_id for item in first_batch]

        self.assertEqual(len(first_batch_ids), 25)
        self.assertEqual(first_batch_ids[0], "session-00")
        self.assertEqual(first_batch_ids[-1], "session-24")

        SessionMetadata.objects.filter(thread_id__in=first_batch_ids).update(
            usage_last_checked_at=datetime(2025, 1, 6, tzinfo=UTC)
        )
        rows = list(SessionMetadata.objects.order_by("thread_id"))
        second_batch_ids = [
            item.thread_id for item in views._usage_token_refresh_items(rows, {})
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
        caches = views._token_usage_caches_by_thread_ids(
            row.thread_id for row in rows
        )

        batch_ids = [
            item.thread_id for item in views._usage_token_refresh_items(rows, caches)
        ]

        self.assertEqual(len(batch_ids), 25)
        for thread_id in stale_thread_ids:
            self.assertIn(thread_id, batch_ids)

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_new_session_page_populates_project_and_bare_repo_selectors(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        project_a = Project.objects.create(
            name="Project A",
            repo_path="/home/user/proj-a",
            auto_pr_mode=Project.AUTO_PR_ON,
        )
        Project.objects.create(name="Project B", repo_path="/home/user/proj-b")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_project_dropdown_selects_saved_repo_project(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        Project.objects.create(name="Project A", repo_path="/home/user/proj-a")
        project_b = Project.objects.create(name="Project B", repo_path="/home/user/proj-b")
        _seed_cookies(
            self.client,
            **{_LAST_SELECTED_REPO_COOKIE: "/home/user/proj-b"},
        )
        _setup_codex(mock_codex)
        mock_discover.return_value = [Path("/home/user/proj-a"), Path("/home/user/proj-b")]

        response = self.client.get(reverse("new_session"))

        self.assertContains(response, f'value="{project_b.pk}" selected')
        self.assertContains(response, "data-new-session-repo-field hidden")

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_project_dropdown_keeps_saved_unprojected_repo_on_bare_option(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        Project.objects.create(name="Project A", repo_path="/home/user/proj-a")
        _seed_cookies(
            self.client,
            **{_LAST_SELECTED_REPO_COOKIE: "/home/user/bare"},
        )
        _setup_codex(mock_codex)
        mock_discover.return_value = [Path("/home/user/proj-a"), Path("/home/user/bare")]

        response = self.client.get(reverse("new_session"))

        self.assertContains(
            response, f'value="{views._BARE_REPO_PROJECT_VALUE}" selected'
        )
        self.assertNotContains(response, "data-new-session-repo-field hidden")
        self.assertContains(response, 'value="/home/user/bare" selected')

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_project_dropdown_ignores_stale_saved_repo(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        project = Project.objects.create(name="Project A", repo_path="/home/user/proj-a")
        _seed_cookies(
            self.client,
            **{_LAST_SELECTED_REPO_COOKIE: "/home/user/missing"},
        )
        _setup_codex(mock_codex)
        mock_discover.return_value = [Path("/home/user/proj-a")]

        response = self.client.get(reverse("new_session"))

        self.assertContains(response, f'value="{project.pk}" selected')
        self.assertContains(response, "data-new-session-repo-field hidden")

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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
        self.assertContains(response, "Enter a prompt or attach an image.")

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_new_session_page_exposes_coding_agent_override(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _setup_codex(mock_codex)
        mock_discover.return_value = [Path("/home/user/proj")]
        _seed_cookies(self.client, **{_CODING_AGENT_COOKIE: "hitch"})

        response = self.client.get(reverse("new_session"))

        self.assertContains(response, "data-new-session-coding-agent")
        self.assertContains(response, "Use settings (HITCH)")
        self.assertContains(response, 'name="coding_agent"')
        self.assertContains(response, 'value="codex"')
        self.assertContains(response, 'value="hitch"')

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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
        project = Project.objects.create(name="Hitch", repo_path="/repo")

        self.assertEqual(project.auto_pr_mode, Project.AUTO_PR_FOLLOW_GLOBAL)

    @patch("hitch.main.views.discover_repos")
    def test_new_project_form_lists_discovered_repos(self, mock_discover: MagicMock) -> None:
        mock_discover.return_value = [Path("/repo")]

        response = self.client.get(reverse("new_project"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Create project")
        self.assertContains(response, 'name="repo_path"')
        self.assertContains(response, "/repo")

    @patch("hitch.main.views.discover_repos")
    def test_new_project_form_hides_repos_that_already_have_projects(
        self, mock_discover: MagicMock
    ) -> None:
        Project.objects.create(name="Existing", repo_path="/repo")
        mock_discover.return_value = [Path("/repo"), Path("/other")]

        response = self.client.get(reverse("new_project"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, '<option value="/repo">/repo</option>', html=True)
        self.assertContains(response, '<option value="/other">/other</option>', html=True)

    @patch("hitch.main.views.git_common_dir")
    @patch("hitch.main.views.discover_repos")
    def test_new_project_form_hides_repos_with_existing_git_common_dir(
        self, mock_discover: MagicMock, mock_common_dir: MagicMock
    ) -> None:
        Project.objects.create(
            name="Existing",
            repo_path="/repo",
            git_common_dir="/repo/.git",
        )
        mock_discover.return_value = [Path("/repo-worktree")]
        mock_common_dir.return_value = Path("/repo/.git")

        response = self.client.get(reverse("new_project"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "/repo-worktree")
        self.assertContains(response, "No git repositories without projects")

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.discover_repos")
    def test_creates_project_selects_it_and_associates_existing_sessions(
        self, mock_discover: MagicMock, mock_codex: MagicMock
    ) -> None:
        other = Project.objects.create(name="Other", repo_path="/other")
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
        self.assertEqual(_cookie_value(response, "hitch_selected_project_id"), str(project.pk))
        self.assertEqual(
            _cookie_value(response, _VISIBLE_SESSION_PROJECTS_COOKIE),
            f"[{other.pk},{project.pk}]",
        )
        self.assertEqual(SessionMetadata.objects.get(thread_id="match").project, project)
        self.assertFalse(SessionMetadata.objects.filter(thread_id="miss").exists())

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.git_common_dir")
    @patch("hitch.main.views.discover_repos")
    def test_rejects_project_with_existing_git_common_dir(
        self,
        mock_discover: MagicMock,
        mock_common_dir: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        Project.objects.create(
            name="Source",
            repo_path="/repo",
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

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.discover_repos")
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
        project = Project.objects.create(name="Hitch", repo_path="/repo")

        response = self.client.post(
            reverse("edit_project"),
            data={
                "project": str(project.pk),
                "name": "Renamed",
                "extra_system_prompt": "  Prefer project fixtures.  ",
                "auto_pr_mode": Project.AUTO_PR_ON,
            },
        )

        self.assertEqual(response.status_code, 302)
        project.refresh_from_db()
        self.assertEqual(project.name, "Renamed")
        self.assertEqual(project.extra_system_prompt, "Prefer project fixtures.")
        self.assertEqual(project.auto_pr_mode, Project.AUTO_PR_ON)

    def test_edit_project_rejects_invalid_posts(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")

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
                    * (views._EXTRA_SYSTEM_PROMPT_MAX_LEN + 1),
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
        ):
            with self.subTest(message=message):
                response = self.client.post(reverse("edit_project"), data=data)
                self.assertContains(response, message, status_code=400)

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.discover_repos")
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


class NewSessionViewTests(TestCase):
    REPO = "/home/user/proj"

    @override
    def setUp(self) -> None:
        super().setUp()
        self._clear_models_cache()
        self.addCleanup(self._clear_models_cache)

    @staticmethod
    def _clear_models_cache() -> None:
        with views._MODELS_REFRESH_LOCK:
            views._MODELS_CACHE_VALUE = {}
            views._MODELS_CACHE_FETCHED_AT = {}
            views._MODELS_REFRESH_IN_FLIGHT = set()

    def _assert_new_session_spawn(
        self,
        mock_spawn: MagicMock,
        *,
        cwd: str = REPO,
        prompt: str = "do thing",
        **overrides: Any,
    ) -> None:
        expected = {
            "cwd": cwd,
            "prompt": prompt,
            "developer_instructions": None,
            "model": None,
            "reasoning_effort": None,
            "sandbox_policy": None,
            "approval_mode": "auto_review",
        }
        expected.update(overrides)
        mock_spawn.assert_called_once_with(**expected)

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_spawns_worker_and_redirects(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        # No models from Codex → reconcile is a no-op; spawn sees None/None.
        _setup_codex(mock_codex, models=[])

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "Refactor the login flow", "cwd": self.REPO},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            response.headers["Location"],
            reverse("session", kwargs={"session_id": "thread-xyz"}),
        )
        mock_spawn.assert_called_once_with(
            cwd=self.REPO,
            prompt="Refactor the login flow",
            developer_instructions=None,
            model=None,
            reasoning_effort=None,
            sandbox_policy=None,
            approval_mode="auto_review",
        )

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_post_uses_warm_model_cache(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        views._store_models_cache(
            enable_memories=False,
            models_data=[_make_model("gpt-5.4", is_default=True)],
        )
        _seed_cookies(self.client, **{_MODEL_COOKIE: "stale-model"})

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "Refactor the login flow", "cwd": self.REPO},
        )

        self.assertEqual(response.status_code, 302)
        mock_codex.assert_not_called()
        self._assert_new_session_spawn(
            mock_spawn,
            prompt="Refactor the login flow",
            model="gpt-5.4",
            reasoning_effort="medium",
        )
        self.assertEqual(_cookie_value(response, _MODEL_COOKIE), "gpt-5.4")

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_post_refreshes_empty_model_cache(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        views._store_models_cache(enable_memories=False, models_data=[])
        _setup_codex(mock_codex, models=[_make_model("gpt-5.4", is_default=True)])
        _seed_cookies(self.client, **{_MODEL_COOKIE: "removed-model"})

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "Refactor the login flow", "cwd": self.REPO},
        )

        self.assertEqual(response.status_code, 302)
        mock_codex.assert_called_once()
        self._assert_new_session_spawn(
            mock_spawn,
            prompt="Refactor the login flow",
            model="gpt-5.4",
            reasoning_effort="medium",
        )
        self.assertEqual(_cookie_value(response, _MODEL_COOKIE), "gpt-5.4")

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_post_refreshes_expired_model_cache(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        views._store_models_cache(
            enable_memories=False,
            models_data=[_make_model("removed-model", is_default=True)],
        )
        with views._MODELS_REFRESH_LOCK:
            views._MODELS_CACHE_FETCHED_AT[False] = (
                timezone.now()
                - views._MODELS_CACHE_TTL
                - timedelta(seconds=1)
            )
        _setup_codex(mock_codex, models=[_make_model("gpt-5.4", is_default=True)])
        _seed_cookies(self.client, **{_MODEL_COOKIE: "removed-model"})

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "Refactor the login flow", "cwd": self.REPO},
        )

        self.assertEqual(response.status_code, 302)
        mock_codex.assert_called_once()
        self._assert_new_session_spawn(
            mock_spawn,
            prompt="Refactor the login flow",
            model="gpt-5.4",
            reasoning_effort="medium",
        )
        self.assertEqual(_cookie_value(response, _MODEL_COOKIE), "gpt-5.4")

    @patch("hitch.main.views.system_agents.spec_critic_should_run", return_value=True)
    @patch("hitch.main.views.system_agents.start_spec_critic_workflow")
    @patch("hitch.main.views.codex_pool.create_session_thread", return_value="thread-spec")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_spec_critic_new_session_creates_visible_thread_before_implementation(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_create_thread: MagicMock,
        mock_start_spec_critic: MagicMock,
        mock_spec_critic_should_run: MagicMock,
    ) -> None:
        _seed_cookies(
            self.client,
            **{_SPEC_CRITIC_COOKIE: "true", _WEB_SEARCH_COOKIE: "live"},
        )
        mock_discover.return_value = [Path(self.REPO)]
        _setup_codex(mock_codex, models=[])

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "Improve onboarding", "cwd": self.REPO},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            response.headers["Location"],
            reverse("session", kwargs={"session_id": "thread-spec"}),
        )
        mock_spawn.assert_not_called()
        # The should-run classifier runs inside the workflow on a background
        # thread now, so the request path must not call it synchronously.
        mock_spec_critic_should_run.assert_not_called()
        mock_create_thread.assert_called_once_with(
            cwd=self.REPO,
            name="Improve onboarding",
            developer_instructions=None,
            model=None,
            enable_memories=False,
            web_search_mode="live",
        )
        mock_start_spec_critic.assert_called_once_with(
            main_thread_id="thread-spec",
            cwd=self.REPO,
            prompt="Improve onboarding",
            sandbox_policy=None,
            approval_mode="auto_review",
            model=None,
            reasoning_effort=None,
            developer_instructions=None,
            enable_memories=False,
            web_search_mode="live",
            initial_user_message_index=0,
            auto_pr_enabled=False,
            auto_qa_enabled=False,
        )
        metadata = SessionMetadata.objects.get(thread_id="thread-spec")
        self.assertEqual(metadata.cwd, self.REPO)

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_spawns_worker_with_uploaded_image(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        _setup_codex(mock_codex, models=[])

        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_EVENTS_DIR=Path(raw)),
        ):
            response = self.client.post(
                reverse("new_session"),
                data={
                    "prompt": "Use the screenshot",
                    "cwd": self.REPO,
                    "input_images": SimpleUploadedFile(
                        "screen.png", _PNG_BYTES, content_type="image/png"
                    ),
                },
            )

            self.assertEqual(response.status_code, 302)
            image_paths = mock_spawn.call_args.kwargs["input_image_paths"]
            self.assertEqual(len(image_paths), 1)
            saved_image = Path(image_paths[0])
            self.assertEqual(saved_image.suffix, ".png")
            self.assertTrue(saved_image.is_file())
            self.assertEqual(saved_image.read_bytes(), _PNG_BYTES)
            self.assertEqual(saved_image.stat().st_mode & 0o777, 0o600)
            self.assertEqual(saved_image.parent.stat().st_mode & 0o777, 0o700)
            self.assertEqual(saved_image.parent.parent.stat().st_mode & 0o777, 0o700)

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_spawns_worker_with_image_only_prompt(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        _setup_codex(mock_codex, models=[])

        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_EVENTS_DIR=Path(raw)),
        ):
            response = self.client.post(
                reverse("new_session"),
                data={
                    "prompt": "",
                    "cwd": self.REPO,
                    "input_images": SimpleUploadedFile(
                        "screen.png", _PNG_BYTES, content_type="image/png"
                    ),
                },
            )

            self.assertEqual(response.status_code, 302)
            image_paths = mock_spawn.call_args.kwargs["input_image_paths"]
            self.assertEqual(len(image_paths), 1)
            self.assertEqual(Path(image_paths[0]).read_bytes(), _PNG_BYTES)
        mock_spawn.assert_called_once()
        self.assertEqual(mock_spawn.call_args.kwargs["prompt"], "")

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_spawns_worker_with_multiple_uploaded_image_formats(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        _setup_codex(mock_codex, models=[])

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
                reverse("new_session"),
                data={
                    "prompt": "Use these screenshots",
                    "cwd": self.REPO,
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

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_cleans_uploaded_images_when_spawn_handoff_fails(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.side_effect = RuntimeError("launch failed")
        _setup_codex(mock_codex, models=[])

        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_EVENTS_DIR=Path(raw)),
        ):
            with self.assertRaises(RuntimeError):
                self.client.post(
                    reverse("new_session"),
                    data={
                        "prompt": "Use the screenshot",
                        "cwd": self.REPO,
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

    def test_input_image_upload_handler_rejects_limits_during_parse(self) -> None:
        with patch("hitch.main.views._INPUT_IMAGE_MAX_BYTES", 8):
            handler = views._InputImageLimitUploadHandler()
            with self.assertRaisesMessage(
                SuspiciousOperation,
                "image attachment is too large",
            ):
                handler.new_file(
                    "input_images",
                    "screen.png",
                    "image/png",
                    9,
                )

            handler = views._InputImageLimitUploadHandler()
            handler.new_file("input_images", "screen.png", "image/png", None)
            with self.assertRaisesMessage(
                SuspiciousOperation,
                "image attachment is too large",
            ):
                handler.receive_data_chunk(b"123456789", 0)

        handler = views._InputImageLimitUploadHandler()
        with self.assertRaisesMessage(
            SuspiciousOperation,
            "at most 4 image attachments are allowed",
        ):
            for index in range(5):
                handler.new_file(
                    "input_images",
                    f"screen-{index}.png",
                    "image/png",
                    1,
                )

    def test_input_image_request_size_cap_runs_before_parse(self) -> None:
        request = RequestFactory().post(reverse("new_session"), data={})
        request.META["CONTENT_LENGTH"] = str(views._INPUT_IMAGE_MAX_REQUEST_BYTES + 1)

        self.assertEqual(
            views._input_image_request_size_error(request),
            "image attachments are too large",
        )

    def test_input_image_upload_limiter_handles_cased_multipart_before_csrf(
        self,
    ) -> None:
        def view(request: Any) -> Any:
            self.assertIsInstance(
                request.upload_handlers[0],
                views._InputImageLimitUploadHandler,
            )
            return HttpResponse("ok")

        request = RequestFactory().generic("POST", reverse("new_session"), data=b"")
        request.content_type = "Multipart/form-data"
        request.META["CONTENT_TYPE"] = "Multipart/form-data; boundary=BOUNDARY"
        cast(Any, request)._dont_enforce_csrf_checks = True

        response = views._limit_input_image_uploads(view)(request)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(getattr(views.new_session, "csrf_exempt", False))
        self.assertTrue(getattr(views.send_message, "csrf_exempt", False))

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_upload_limited_new_session_still_enforces_csrf(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        _setup_codex(mock_codex, models=[])
        client = Client(enforce_csrf_checks=True)
        url = reverse("new_session")

        denied = client.post(
            url,
            data={
                "prompt": "Use this screenshot",
                "cwd": self.REPO,
                "input_images": SimpleUploadedFile(
                    "screen.png", _PNG_BYTES, content_type="image/png"
                ),
            },
        )

        self.assertEqual(denied.status_code, 403)
        mock_spawn.assert_not_called()

        client.get(reverse("index"))
        token = client.cookies["csrftoken"].value
        allowed = client.post(
            url,
            data={
                "csrfmiddlewaretoken": token,
                "prompt": "Use this screenshot",
                "cwd": self.REPO,
                "input_images": SimpleUploadedFile(
                    "screen.png", _PNG_BYTES, content_type="image/png"
                ),
            },
        )

        self.assertEqual(allowed.status_code, 302)
        mock_spawn.assert_called_once()

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_rejects_invalid_image_uploads_before_spawn(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        _setup_codex(mock_codex, models=[])
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
                    mock_spawn.reset_mock()
                    response = self.client.post(
                        reverse("new_session"),
                        data={
                            "prompt": "Use the screenshot",
                            "cwd": self.REPO,
                            "input_images": upload,
                        },
                    )

                    self.assertContains(response, message, status_code=400)
                    mock_spawn.assert_not_called()
                    self.assertFalse((Path(raw) / "attachments").exists())

            with patch("hitch.main.views._INPUT_IMAGE_MAX_BYTES", len(_PNG_BYTES) - 1):
                response = self.client.post(
                    reverse("new_session"),
                    data={
                        "prompt": "Use the screenshot",
                        "cwd": self.REPO,
                        "input_images": SimpleUploadedFile(
                            "screen.png", _PNG_BYTES, content_type="image/png"
                        ),
                    },
                )

            self.assertContains(response, "image attachment is too large", status_code=400)
            mock_spawn.assert_not_called()
            self.assertFalse((Path(raw) / "attachments").exists())

            with patch(
                "hitch.main.views.os.fdopen",
                side_effect=lambda fd, *_args: _FailingUploadWriter(fd),
            ), self.assertLogs("hitch.main.views", level="ERROR"):
                response = self.client.post(
                    reverse("new_session"),
                    data={
                        "prompt": "Use the screenshot",
                        "cwd": self.REPO,
                        "input_images": SimpleUploadedFile(
                            "screen.png", _PNG_BYTES, content_type="image/png"
                        ),
                    },
                )

            self.assertContains(
                response,
                "failed to save image attachment",
                status_code=400,
            )
            self.assertNotContains(response, "disk full", status_code=400)
            mock_spawn.assert_not_called()
            attachments = Path(raw) / "attachments"
            self.assertTrue(attachments.exists())
            self.assertEqual([path for path in attachments.rglob("*") if path.is_file()], [])

            real_fdopen = os.fdopen
            fdopen_calls = 0

            def fail_second_file(fd: int, *args: Any, **kwargs: Any) -> Any:
                nonlocal fdopen_calls
                fdopen_calls += 1
                if fdopen_calls == 2:
                    return _FailingUploadWriter(fd)
                return real_fdopen(fd, *args, **kwargs)

            with patch(
                "hitch.main.views.os.fdopen",
                side_effect=fail_second_file,
            ), self.assertLogs("hitch.main.views", level="ERROR"):
                response = self.client.post(
                    reverse("new_session"),
                    data={
                        "prompt": "Use the screenshots",
                        "cwd": self.REPO,
                        "input_images": [
                            SimpleUploadedFile(
                                "first.png", _PNG_BYTES, content_type="image/png"
                            ),
                            SimpleUploadedFile(
                                "second.png", _PNG_BYTES, content_type="image/png"
                            ),
                        ],
                    },
                )

            self.assertContains(
                response,
                "failed to save image attachment",
                status_code=400,
            )
            self.assertEqual([path for path in attachments.rglob("*") if path.is_file()], [])

    def test_image_upload_read_failure_returns_generic_error(self) -> None:
        with self.assertLogs("hitch.main.views", level="ERROR"):
            _extension, error = views._uploaded_input_image_extension(
                _UnreadableUpload("screen.png", _PNG_BYTES, content_type="image/png")
            )

        self.assertEqual(error, "failed to read image attachment")
        assert error is not None
        self.assertNotIn("/tmp/private", error)

    @patch("hitch.main.views.system_agents.start_pr_qa_workflow")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.codex_pool.create_session_thread")
    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_rejects_workflow_image_uploads_before_side_effects(
        self,
        mock_discover: MagicMock,
        mock_codex: MagicMock,
        mock_create_thread: MagicMock,
        mock_spawn: MagicMock,
        mock_start_workflow: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        _setup_codex(mock_codex, models=[])

        for prompt in ("/pr", "/qa"):
            with self.subTest(prompt=prompt):
                response = self.client.post(
                    reverse("new_session"),
                    data={
                        "prompt": prompt,
                        "cwd": self.REPO,
                        "input_images": SimpleUploadedFile(
                            "screen.png", _PNG_BYTES, content_type="image/png"
                        ),
                    },
                )

                self.assertContains(
                    response,
                    "image attachments are not supported for PR workflow requests",
                    status_code=400,
                )
                mock_create_thread.assert_not_called()
                mock_spawn.assert_not_called()
                mock_start_workflow.assert_not_called()

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_accepts_and_associates_proposed_work(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        _setup_codex(mock_codex, models=[])
        project = Project.objects.create(name="Hitch", repo_path=self.REPO)
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            title="Add parser coverage",
        )
        prompt = "Go ahead and implement this proposed session."
        mock_discover.return_value = [Path(self.REPO)]

        def spawn_after_concurrent_reject_attempt(**_kwargs: Any) -> SimpleNamespace:
            rejected = ProposedSession.objects.filter(
                pk=proposal.pk,
                outcome_status=ProposedSession.OUTCOME_UNSET,
            ).update(
                outcome_status=ProposedSession.OUTCOME_REJECTED,
                outcome_notes="Resolved from another tab.",
                updated_at=timezone.now(),
            )
            self.assertEqual(rejected, 0)
            return SimpleNamespace(thread_id="thread-xyz")

        mock_spawn.side_effect = spawn_after_concurrent_reject_attempt

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": prompt,
                "cwd": self.REPO,
                "proposed_session": str(proposal.pk),
            },
        )

        self.assertEqual(response.status_code, 302)
        metadata = SessionMetadata.objects.get(thread_id="thread-xyz")
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_ACCEPTED)
        self.assertEqual(proposal.accepted_session, metadata)
        self.assertNotIn(
            ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY,
            proposal.outcome_metadata,
        )
        self._assert_new_session_spawn(
            mock_spawn,
            prompt=prompt,
            thread_name="Add parser coverage",
        )

    @patch("hitch.main.views._save_posted_input_images")
    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_accept_losing_start_claim_redirects_without_spawn(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
        mock_save_images: MagicMock,
    ) -> None:
        _setup_codex(mock_codex, models=[])
        project = Project.objects.create(name="Hitch", repo_path=self.REPO)
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            title="Add parser coverage",
        )
        mock_discover.return_value = [Path(self.REPO)]

        def reject_after_lookup(_request: Any) -> tuple[list[str], str | None]:
            rejected = ProposedSession.objects.filter(
                pk=proposal.pk,
                outcome_status=ProposedSession.OUTCOME_UNSET,
            ).update(
                outcome_status=ProposedSession.OUTCOME_REJECTED,
                outcome_notes="Resolved from another tab.",
                updated_at=timezone.now(),
            )
            self.assertEqual(rejected, 1)
            return [], None

        mock_save_images.side_effect = reject_after_lookup

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": "Go ahead and implement this proposed session.",
                "cwd": self.REPO,
                "proposed_session": str(proposal.pk),
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], reverse("inbox"))
        mock_spawn.assert_not_called()
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_REJECTED)
        self.assertIsNone(proposal.accepted_session)

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_spawn_failure_resets_proposal_start_claim(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        _setup_codex(mock_codex, models=[])
        project = Project.objects.create(name="Hitch", repo_path=self.REPO)
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            title="Add parser coverage",
        )
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.side_effect = RuntimeError("worker failed")

        with self.assertRaises(RuntimeError):
            self.client.post(
                reverse("new_session"),
                data={
                    "prompt": "Go ahead and implement this proposed session.",
                    "cwd": self.REPO,
                    "proposed_session": str(proposal.pk),
                },
            )

        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_UNSET)
        self.assertIsNone(proposal.accepted_session)
        self.assertNotIn(
            ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY,
            proposal.outcome_metadata,
        )

    def test_new_session_finish_ignores_replaced_start_claim(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path=self.REPO)
        claim_key = ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY
        old_claim = "2026-06-05T14:00:00+00:00"
        new_claim = "2026-06-05T14:45:00+00:00"
        proposal = ProposedSession.objects.create(
            project=project,
            title="Add parser coverage",
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            outcome_metadata={
                "accepted_by": "user",
                "accepted_thread_id": "",
                claim_key: old_claim,
            },
        )
        ProposedSession.objects.filter(pk=proposal.pk).update(
            outcome_metadata={
                "accepted_by": "user",
                "accepted_thread_id": "",
                claim_key: new_claim,
            }
        )
        metadata = SessionMetadata.objects.create(
            thread_id="late-thread",
            cwd=self.REPO,
            project=project,
        )

        views._finish_new_session_proposal_start_claim(proposal, metadata)

        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_ACCEPTED)
        self.assertIsNone(proposal.accepted_session)
        self.assertEqual(proposal.outcome_metadata[claim_key], new_claim)

    def test_new_session_reset_ignores_replaced_start_claim(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path=self.REPO)
        claim_key = ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY
        old_claim = "2026-06-05T14:00:00+00:00"
        new_claim = "2026-06-05T14:45:00+00:00"
        proposal = ProposedSession.objects.create(
            project=project,
            title="Add parser coverage",
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            outcome_metadata={
                "accepted_by": "user",
                "accepted_thread_id": "",
                claim_key: old_claim,
            },
        )
        ProposedSession.objects.filter(pk=proposal.pk).update(
            outcome_metadata={
                "accepted_by": "user",
                "accepted_thread_id": "",
                claim_key: new_claim,
            }
        )

        views._reset_new_session_proposal_start_claim(proposal)

        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_ACCEPTED)
        self.assertIsNone(proposal.accepted_session)
        self.assertEqual(proposal.outcome_metadata[claim_key], new_claim)

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_accept_preserves_proposal_auto_merge_settings(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        _setup_codex(mock_codex, models=[])
        project = Project.objects.create(name="Hitch", repo_path=self.REPO)
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            auto_qa_enabled=True,
            auto_merge_to_local_branch=True,
            auto_merge_branch="release",
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            title="Add parser coverage",
            outcome_metadata={
                "auto_pr_enabled": False,
                "auto_qa_enabled": True,
                "auto_merge_to_local_branch": True,
                "auto_merge_branch": "release",
            },
        )
        AutonomousGoal.objects.filter(pk=goal.pk).update(
            auto_qa_enabled=False,
            auto_merge_to_local_branch=False,
            auto_merge_branch="",
        )
        prompt = "Go ahead and implement this proposed session."
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": prompt,
                "cwd": self.REPO,
                "proposed_session": str(proposal.pk),
                "auto_qa": "false",
            },
        )

        self.assertEqual(response.status_code, 302)
        metadata = SessionMetadata.objects.get(thread_id="thread-xyz")
        self.assertTrue(metadata.auto_qa_enabled)
        self.assertTrue(metadata.auto_merge_to_local_branch)
        self.assertEqual(metadata.auto_merge_branch, "release")
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_ACCEPTED)
        self.assertEqual(proposal.accepted_session, metadata)
        self.assertNotIn(
            ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY,
            proposal.outcome_metadata,
        )
        self._assert_new_session_spawn(
            mock_spawn,
            prompt=prompt,
            thread_name="Add parser coverage",
            auto_qa_enabled=True,
            auto_merge_to_local_branch=True,
            auto_merge_branch="release",
        )

    @patch("hitch.main.views.system_agents.spec_critic_should_run")
    @patch("hitch.main.views.system_agents.start_spec_critic_workflow")
    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_project_proposal_start_skips_preflight_and_repo_discovery(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
        mock_start_spec_critic: MagicMock,
        mock_spec_critic_should_run: MagicMock,
    ) -> None:
        _seed_cookies(self.client, **{_SPEC_CRITIC_COOKIE: "true"})
        _setup_codex(mock_codex, models=[])
        project = Project.objects.create(name="Hitch", repo_path=self.REPO)
        proposal = ProposedSession.objects.create(
            project=project,
            title="Add parser coverage",
        )
        prompt = "Go ahead and implement this proposed session."
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": prompt,
                "project": str(project.pk),
                "proposed_session": str(proposal.pk),
            },
        )

        self.assertEqual(response.status_code, 302)
        self._assert_new_session_spawn(
            mock_spawn,
            prompt=prompt,
            thread_name="Add parser coverage",
        )
        mock_discover.assert_not_called()
        mock_spec_critic_should_run.assert_not_called()
        mock_start_spec_critic.assert_not_called()

    @patch("hitch.main.views.system_agents.spec_critic_should_run", return_value=True)
    @patch("hitch.main.views.discover_managed_worktrees")
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    def test_new_session_accepts_candidate_worktree_and_starts_turn(
        self,
        mock_turn: MagicMock,
        mock_new_session: MagicMock,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_managed_worktrees: MagicMock,
        mock_spec_critic_should_run: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        mock_managed_worktrees.return_value = [Path("/repo-worktree")]
        codex = _setup_codex(mock_codex, models=[])
        project = Project.objects.create(name="Hitch", repo_path=self.REPO)
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            auto_qa_enabled=True,
            auto_merge_to_local_branch=True,
            auto_merge_branch="release",
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo-worktree",
            project=project,
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            candidate_session=candidate,
            title="Add parser coverage",
            outcome_metadata={
                "auto_pr_enabled": False,
                "auto_qa_enabled": True,
                "auto_merge_to_local_branch": True,
                "auto_merge_branch": "release",
            },
        )
        AutonomousGoal.objects.filter(pk=goal.pk).update(
            auto_qa_enabled=False,
            auto_merge_to_local_branch=False,
            auto_merge_branch="",
        )

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": "Go ahead and implement this proposed session.",
                "cwd": self.REPO,
                "proposed_session": str(proposal.pk),
                "auto_qa": "false",
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            response.headers["Location"],
            reverse("session", kwargs={"session_id": "candidate-thread"}),
        )
        mock_turn.assert_called_once_with(
            thread_id="candidate-thread",
            cwd="/repo-worktree",
            prompt=(
                "First, rebase or otherwise update this worktree onto the current "
                "project base branch before continuing. Resolve any conflicts, then "
                "continue with the user's instructions.\n\n"
                "Go ahead and implement this proposed session."
            ),
            developer_instructions=None,
            model=None,
            reasoning_effort=None,
            sandbox_policy=None,
            approval_mode="auto_review",
            auto_qa_enabled=True,
            stored_model=None,
            stored_reasoning_effort=None,
            user_message_index=0,
            auto_merge_to_local_branch=True,
            auto_merge_branch="release",
        )
        proposal.refresh_from_db()
        candidate.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_ACCEPTED)
        self.assertEqual(proposal.accepted_session, candidate)
        self.assertFalse(candidate.is_hidden_system_session)
        self.assertEqual(candidate.codex_name, "Add parser coverage")
        self.assertEqual(candidate.codex_display_title, "Add parser coverage")
        self.assertTrue(candidate.auto_qa_enabled)
        self.assertTrue(candidate.auto_merge_to_local_branch)
        self.assertEqual(candidate.auto_merge_branch, "release")
        codex._client.thread_set_name.assert_called_once_with(
            "candidate-thread", "Add parser coverage"
        )
        mock_new_session.assert_not_called()
        mock_spec_critic_should_run.assert_not_called()

    @patch("hitch.main.views.discover_managed_worktrees")
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    def test_candidate_worktree_uses_local_instance_for_next_user_message_index(
        self,
        mock_turn: MagicMock,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_managed_worktrees: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        mock_managed_worktrees.return_value = [Path("/repo-worktree")]
        codex = _setup_codex(mock_codex, models=[])
        codex._client.thread_resume.side_effect = AssertionError(
            "thread_resume should not be needed"
        )
        project = Project.objects.create(name="Hitch", repo_path=self.REPO)
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            auto_qa_enabled=True,
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo-worktree",
            project=project,
        )
        CodexInstance.objects.create(
            thread_id="candidate-thread",
            cwd="/repo-worktree",
            prompt="Find useful test coverage increments.",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_COMPLETED,
            pid=123,
            user_message_index=0,
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            candidate_session=candidate,
            title="Add parser coverage",
            outcome_metadata={"auto_qa_enabled": True},
        )

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": "Go ahead and implement this proposed session.",
                "cwd": self.REPO,
                "proposed_session": str(proposal.pk),
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(mock_turn.call_args.kwargs["user_message_index"], 1)
        codex._client.thread_resume.assert_not_called()

    @patch("hitch.main.views.discover_managed_worktrees")
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.run_borrowed_op_with_retry")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    def test_candidate_worktree_resumes_thread_when_latest_local_index_failed(
        self,
        mock_turn: MagicMock,
        mock_run_borrowed: MagicMock,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_managed_worktrees: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        mock_managed_worktrees.return_value = [Path("/repo-worktree")]
        codex = _setup_codex(mock_codex, models=[])
        rollout_path = _make_rollout(
            self,
            [
                _rollout_line(
                    "event_msg",
                    {
                        "type": "user_message",
                        "message": "Find useful test coverage increments.",
                    },
                ),
                _rollout_line(
                    "event_msg",
                    {"type": "user_message", "message": "Failed retry."},
                ),
            ],
        )
        codex._client.thread_resume.return_value = SimpleNamespace(
            thread=_session("candidate-thread", path=str(rollout_path))
        )
        mock_run_borrowed.side_effect = _run_borrowed_with(codex)
        project = Project.objects.create(name="Hitch", repo_path=self.REPO)
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            auto_qa_enabled=True,
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo-worktree",
            project=project,
        )
        CodexInstance.objects.create(
            thread_id="candidate-thread",
            cwd="/repo-worktree",
            prompt="Find useful test coverage increments.",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_COMPLETED,
            pid=123,
            user_message_index=0,
        )
        CodexInstance.objects.create(
            thread_id="candidate-thread",
            cwd="/repo-worktree",
            prompt="Failed retry.",
            events_path="/tmp/events-failed.jsonl",
            status=CodexInstance.STATUS_FAILED,
            pid=124,
            user_message_index=1,
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            candidate_session=candidate,
            title="Add parser coverage",
            outcome_metadata={"auto_qa_enabled": True},
        )

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": "Go ahead and implement this proposed session.",
                "cwd": self.REPO,
                "proposed_session": str(proposal.pk),
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(mock_turn.call_args.kwargs["user_message_index"], 2)
        codex._client.thread_resume.assert_called_once_with("candidate-thread")
        mock_run_borrowed.assert_called_once()
        self.assertIs(mock_run_borrowed.call_args.args[0], mock_codex)
        self.assertEqual(
            mock_run_borrowed.call_args.kwargs,
            {"enable_memories": False},
        )

    @patch("hitch.main.views._auto_merge_to_local_branch_for_proposal")
    @patch("hitch.main.views.discover_managed_worktrees")
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    def test_candidate_accept_losing_race_aborts_to_inbox(
        self,
        mock_turn: MagicMock,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_managed_worktrees: MagicMock,
        mock_auto_merge: MagicMock,
    ) -> None:
        # Stale-tab race: new_session fetched the still-unset proposal and began
        # continuing its candidate worktree, but an inbox reject commits before
        # the accept transition runs. The accept must lose, and the caller must
        # abort to the inbox rather than unhide the candidate (whose worktree the
        # reject path may have cleaned up) and redirect to it as a live session.
        mock_discover.return_value = [Path(self.REPO)]
        mock_managed_worktrees.return_value = [Path("/repo-worktree")]
        _setup_codex(mock_codex, models=[])
        project = Project.objects.create(name="Hitch", repo_path=self.REPO)
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo-worktree",
            project=project,
            is_hidden_system_session=True,
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            candidate_session=candidate,
            title="Add parser coverage",
        )

        def reject_concurrently(*_args: Any, **_kwargs: Any) -> tuple[bool, str]:
            ProposedSession.objects.filter(pk=proposal.pk).update(
                outcome_status=ProposedSession.OUTCOME_REJECTED,
                outcome_notes="Resolved from another tab.",
            )
            return False, ""

        mock_auto_merge.side_effect = reject_concurrently

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": "Go ahead and implement this proposed session.",
                "cwd": self.REPO,
                "proposed_session": str(proposal.pk),
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], reverse("inbox"))
        proposal.refresh_from_db()
        candidate.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_REJECTED)
        self.assertIsNone(proposal.accepted_session)
        # The losing accept must not have adopted the candidate as a live session.
        self.assertTrue(candidate.is_hidden_system_session)
        mock_turn.assert_not_called()

    @patch("hitch.main.views.discover_managed_worktrees")
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    def test_candidate_accept_claims_before_spawning_turn(
        self,
        mock_turn: MagicMock,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_managed_worktrees: MagicMock,
    ) -> None:
        # Starting the candidate turn is an external side effect. Hitch must win
        # the proposal acceptance before this point, otherwise a concurrent
        # inbox reject can clean up the candidate worktree after a worker has
        # already been spawned against it.
        mock_discover.return_value = [Path(self.REPO)]
        mock_managed_worktrees.return_value = [Path("/repo-worktree")]
        _setup_codex(mock_codex, models=[])
        project = Project.objects.create(name="Hitch", repo_path=self.REPO)
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo-worktree",
            project=project,
            is_hidden_system_session=True,
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            candidate_session=candidate,
            title="Add parser coverage",
        )

        def reject_from_stale_inbox(*_args: Any, **_kwargs: Any) -> None:
            applied = ProposedSession.objects.filter(
                pk=proposal.pk,
                outcome_status=ProposedSession.OUTCOME_UNSET,
            ).update(
                outcome_status=ProposedSession.OUTCOME_REJECTED,
                outcome_notes="Resolved from another tab.",
            )
            self.assertEqual(applied, 0)

        mock_turn.side_effect = reject_from_stale_inbox

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": "Go ahead and implement this proposed session.",
                "cwd": self.REPO,
                "proposed_session": str(proposal.pk),
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            response.headers["Location"],
            reverse("session", kwargs={"session_id": "candidate-thread"}),
        )
        mock_turn.assert_called_once()
        proposal.refresh_from_db()
        candidate.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_ACCEPTED)
        self.assertEqual(proposal.accepted_session, candidate)
        self.assertFalse(candidate.is_hidden_system_session)

    @patch("hitch.main.views.system_agents.spec_critic_should_run", return_value=True)
    @patch("hitch.main.views.system_agents.start_pr_qa_workflow")
    @patch("hitch.main.views.discover_managed_worktrees")
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    def test_new_session_candidate_worktree_slash_commands_start_qa_workflow(
        self,
        mock_turn: MagicMock,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_managed_worktrees: MagicMock,
        mock_start_workflow: MagicMock,
        mock_spec_critic_should_run: MagicMock,
    ) -> None:
        _seed_cookies(self.client, **{_SPEC_CRITIC_COOKIE: "true"})
        mock_discover.return_value = [Path(self.REPO)]
        mock_managed_worktrees.return_value = [Path("/repo-worktree")]
        codex = _setup_codex(mock_codex, models=[])
        codex._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(turns=[])
        )
        cases = [
            ("/pr", {}),
            ("/qa", {"open_pr_on_lgtm": False}),
        ]

        for index, (prompt, expected) in enumerate(cases):
            with self.subTest(prompt=prompt):
                project = Project.objects.create(
                    name=f"Hitch {index}", repo_path=f"{self.REPO}-{index}"
                )
                goal = AutonomousGoal.objects.create(
                    project=project,
                    title="Improve tests",
                    goal="Find useful test coverage increments.",
                )
                candidate = SessionMetadata.objects.create(
                    thread_id=f"candidate-thread-{index}",
                    cwd="/repo-worktree",
                    project=project,
                )
                proposal = ProposedSession.objects.create(
                    autonomous_goal=goal,
                    candidate_session=candidate,
                    title="Add parser coverage",
                )
                mock_discover.return_value = [Path(project.repo_path)]
                mock_start_workflow.reset_mock()
                mock_turn.reset_mock()
                mock_spec_critic_should_run.reset_mock()

                response = self.client.post(
                    reverse("new_session"),
                    data={
                        "prompt": prompt,
                        "cwd": project.repo_path,
                        "proposed_session": str(proposal.pk),
                    },
                )

                self.assertEqual(response.status_code, 302)
                self.assertEqual(
                    response.headers["Location"],
                    reverse(
                        "session",
                        kwargs={"session_id": f"candidate-thread-{index}"},
                    ),
                )
                workflow_kwargs: dict[str, Any] = {
                    "main_thread_id": f"candidate-thread-{index}",
                    "cwd": "/repo-worktree",
                    "sandbox_policy": None,
                    "approval_mode": "auto_review",
                    "model": None,
                    "reasoning_effort": None,
                    "developer_instructions": None,
                    "enable_memories": False,
                    "initial_user_message_index": 0,
                }
                workflow_kwargs.update(expected)
                mock_start_workflow.assert_called_once_with(**workflow_kwargs)
                mock_turn.assert_not_called()
                mock_spec_critic_should_run.assert_not_called()
                proposal.refresh_from_db()
                candidate.refresh_from_db()
                self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_ACCEPTED)
                self.assertEqual(proposal.accepted_session, candidate)
                self.assertFalse(candidate.auto_pr_enabled)
                self.assertFalse(candidate.auto_qa_enabled)

    @patch("hitch.main.views.system_agents.start_pr_qa_workflow")
    @patch("hitch.main.views.discover_managed_worktrees")
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    def test_candidate_worktree_qa_start_failure_resets_accept_claim(
        self,
        mock_turn: MagicMock,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_managed_worktrees: MagicMock,
        mock_start_workflow: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        mock_managed_worktrees.return_value = [Path("/repo-worktree")]
        codex = _setup_codex(mock_codex, models=[])
        codex._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(turns=[])
        )
        mock_start_workflow.side_effect = RuntimeError("workflow failed")
        project = Project.objects.create(name="Hitch", repo_path=self.REPO)
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo-worktree",
            project=project,
            is_hidden_system_session=True,
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            candidate_session=candidate,
            title="Add parser coverage",
        )

        with self.assertRaises(RuntimeError):
            self.client.post(
                reverse("new_session"),
                data={
                    "prompt": "/qa",
                    "cwd": self.REPO,
                    "proposed_session": str(proposal.pk),
                },
            )

        mock_start_workflow.assert_called_once()
        mock_turn.assert_not_called()
        proposal.refresh_from_db()
        candidate.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_UNSET)
        self.assertIsNone(proposal.accepted_session)
        self.assertTrue(candidate.is_hidden_system_session)

    @patch("hitch.main.views.system_agents.start_pr_qa_workflow")
    @patch("hitch.main.views.discover_managed_worktrees")
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    def test_candidate_worktree_slash_command_preserves_goal_auto_review(
        self,
        mock_turn: MagicMock,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_managed_worktrees: MagicMock,
        mock_start_workflow: MagicMock,
    ) -> None:
        # Accepting an autonomous-goal proposal with a /qa (or /pr) prompt must
        # persist the goal-derived auto-review and auto-merge configuration onto
        # the session, so subsequent turns keep honoring it rather than silently
        # reverting to manual review.
        mock_discover.return_value = [Path(self.REPO)]
        mock_managed_worktrees.return_value = [Path("/repo-worktree")]
        codex = _setup_codex(mock_codex, models=[])
        codex._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(turns=[])
        )
        project = Project.objects.create(name="Hitch", repo_path=self.REPO)
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            auto_qa_enabled=True,
            auto_merge_to_local_branch=True,
            auto_merge_branch="release",
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo-worktree",
            project=project,
            is_hidden_system_session=True,
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            candidate_session=candidate,
            title="Add parser coverage",
            outcome_metadata={
                "auto_pr_enabled": False,
                "auto_qa_enabled": True,
                "auto_merge_to_local_branch": True,
                "auto_merge_branch": "release",
            },
        )

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": "/qa",
                "cwd": self.REPO,
                "proposed_session": str(proposal.pk),
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            response.headers["Location"],
            reverse("session", kwargs={"session_id": "candidate-thread"}),
        )
        mock_start_workflow.assert_called_once_with(
            main_thread_id="candidate-thread",
            cwd="/repo-worktree",
            sandbox_policy=None,
            approval_mode="auto_review",
            model=None,
            reasoning_effort=None,
            developer_instructions=None,
            enable_memories=False,
            initial_user_message_index=0,
            open_pr_on_lgtm=False,
            auto_merge_branch="release",
        )
        mock_turn.assert_not_called()
        proposal.refresh_from_db()
        candidate.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_ACCEPTED)
        self.assertEqual(proposal.accepted_session, candidate)
        self.assertFalse(candidate.is_hidden_system_session)
        # The goal enabled auto-QA and auto-merge; the accepted session must
        # retain those so future turns continue to review and merge.
        self.assertFalse(candidate.auto_pr_enabled)
        self.assertTrue(candidate.auto_qa_enabled)
        self.assertTrue(candidate.auto_merge_to_local_branch)
        self.assertEqual(candidate.auto_merge_branch, "release")

    @patch("hitch.main.views.system_agents.spec_critic_should_run")
    @patch("hitch.main.views.system_agents.start_spec_critic_workflow")
    @patch("hitch.main.views.discover_managed_worktrees")
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    def test_new_session_candidate_worktree_skips_spec_critic_preflight(
        self,
        mock_turn: MagicMock,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_managed_worktrees: MagicMock,
        mock_start_spec_critic: MagicMock,
        mock_spec_critic_should_run: MagicMock,
    ) -> None:
        _seed_cookies(self.client, **{_SPEC_CRITIC_COOKIE: "true"})
        mock_discover.return_value = [Path(self.REPO)]
        mock_managed_worktrees.return_value = [Path("/repo-worktree")]
        codex = _setup_codex(mock_codex, models=[])
        project = Project.objects.create(name="Hitch", repo_path=self.REPO)
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo-worktree",
            project=project,
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            candidate_session=candidate,
            title="Add parser coverage",
        )

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": "Go ahead and implement this proposed session.",
                "cwd": self.REPO,
                "proposed_session": str(proposal.pk),
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            response.headers["Location"],
            reverse("session", kwargs={"session_id": "candidate-thread"}),
        )
        mock_turn.assert_called_once_with(
            thread_id="candidate-thread",
            cwd="/repo-worktree",
            prompt=(
                "First, rebase or otherwise update this worktree onto the current "
                "project base branch before continuing. Resolve any conflicts, then "
                "continue with the user's instructions.\n\n"
                "Go ahead and implement this proposed session."
            ),
            developer_instructions=None,
            model=None,
            reasoning_effort=None,
            sandbox_policy=None,
            approval_mode="auto_review",
        )
        mock_spec_critic_should_run.assert_not_called()
        mock_start_spec_critic.assert_not_called()
        codex._client.thread_resume.assert_not_called()
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_ACCEPTED)
        self.assertEqual(proposal.accepted_session, candidate)

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos", return_value=[Path(REPO)])
    def test_new_session_rejects_invalid_proposed_session_matrix(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path=self.REPO)
        other_project = Project.objects.create(name="Other", repo_path="/home/user/other")
        goal = AutonomousGoal.objects.create(
            project=other_project,
            title="Improve docs",
            goal="Find useful docs increments.",
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            title="Add docs coverage",
        )
        resolved_order = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        resolved = ProposedSession.objects.create(
            autonomous_goal=resolved_order,
            title="Add parser coverage",
            outcome_status=ProposedSession.OUTCOME_REJECTED,
        )
        mock_discover.return_value = [Path(self.REPO), Path(other_project.repo_path)]
        _setup_codex(mock_codex, models=[])
        cases = [
            (
                "non-numeric id",
                {"cwd": self.REPO, "proposed_session": "not-a-number"},
                b"proposed session is required",
            ),
            (
                "zero id",
                {"cwd": self.REPO, "proposed_session": "0"},
                b"proposed session is required",
            ),
            (
                "missing id",
                {"cwd": self.REPO, "proposed_session": "999"},
                b"proposed session is required",
            ),
            (
                "posted project mismatch",
                {"project": str(project.pk), "proposed_session": str(proposal.pk)},
                b"proposed session does not match project",
            ),
            (
                "implicit cwd mismatch",
                {"cwd": self.REPO, "proposed_session": str(proposal.pk)},
                b"proposed session does not match project",
            ),
            (
                "bare repo mismatch",
                {
                    "project": views._BARE_REPO_PROJECT_VALUE,
                    "cwd": self.REPO,
                    "proposed_session": str(proposal.pk),
                },
                b"proposed session does not match project",
            ),
            (
                "resolved proposal",
                {"project": str(project.pk), "proposed_session": str(resolved.pk)},
                b"proposed session is required",
            ),
        ]

        for label, data, message in cases:
            with self.subTest(label=label):
                response = self.client.post(
                    reverse("new_session"),
                    data={
                        "prompt": "Go ahead and implement this proposed session.",
                        **data,
                    },
                )

                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.content, message)
        mock_spawn.assert_not_called()

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_remembers_selected_repo_in_cookie(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        other_repo = "/home/user/other"
        mock_discover.return_value = [Path(self.REPO), Path(other_repo)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        _setup_codex(mock_codex, models=[])

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "do thing", "cwd": other_repo},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(_cookie_value(response, _LAST_SELECTED_REPO_COOKIE), other_repo)

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_remembers_selected_repo_in_account_settings(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        user_model = get_user_model()
        user = user_model.objects.create_user("dev@example.com", password="StrongPass123!")
        self.client.force_login(user)
        other_repo = "/home/user/other"
        mock_discover.return_value = [Path(self.REPO), Path(other_repo)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        _setup_codex(mock_codex, models=[])

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "do thing", "cwd": other_repo},
        )

        self.assertEqual(response.status_code, 302)
        settings = UserSettings.objects.get(user=user)
        self.assertEqual(settings.last_selected_repo, other_repo)
        self.assertEqual(_cookie_value(response, _LAST_SELECTED_REPO_COOKIE), other_repo)

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_auto_pr_precedence_matrix(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        _setup_codex(mock_codex, models=[])
        cases: list[dict[str, Any]] = [
            {
                "name": "posted override enables bare repo",
                "post_auto_pr": "true",
                "expected": True,
            },
            {
                "name": "posted override disables global setting",
                "global_auto_pr": "true",
                "post_auto_pr": "false",
                "expected": False,
            },
            {
                "name": "project on sets default",
                "project_auto_pr_mode": Project.AUTO_PR_ON,
                "expected": True,
            },
            {
                "name": "project off overrides global setting",
                "global_auto_pr": "true",
                "project_auto_pr_mode": Project.AUTO_PR_OFF,
                "expected": False,
            },
            {
                "name": "posted override disables project setting",
                "project_auto_pr_mode": Project.AUTO_PR_ON,
                "post_auto_pr": "false",
                "expected": False,
            },
        ]

        for index, case in enumerate(cases):
            with self.subTest(case["name"]):
                client = Client()
                thread_id = f"thread-{index}"
                repo = (
                    f"{self.REPO}-{index}"
                    if "project_auto_pr_mode" in case
                    else self.REPO
                )
                data = {"prompt": "do thing"}
                project_auto_pr_mode = case.get("project_auto_pr_mode")
                if project_auto_pr_mode is None:
                    data["cwd"] = repo
                else:
                    project = Project.objects.create(
                        name=f"Hitch {index}",
                        repo_path=repo,
                        auto_pr_mode=project_auto_pr_mode,
                    )
                    data["project"] = str(project.pk)
                if "post_auto_pr" in case:
                    data["auto_pr"] = case["post_auto_pr"]
                if "global_auto_pr" in case:
                    _seed_cookies(client, **{_AUTO_PR_COOKIE: case["global_auto_pr"]})
                mock_discover.return_value = [Path(repo)]
                mock_spawn.return_value = SimpleNamespace(thread_id=thread_id)
                mock_spawn.reset_mock()

                response = client.post(reverse("new_session"), data=data)

                self.assertEqual(response.status_code, 302)
                metadata = SessionMetadata.objects.get(thread_id=thread_id)
                self.assertEqual(metadata.auto_pr_enabled, case["expected"])
                expected_spawn: dict[str, Any] = {
                    "cwd": repo,
                    "prompt": "do thing",
                    "developer_instructions": None,
                    "model": None,
                    "reasoning_effort": None,
                    "sandbox_policy": None,
                    "approval_mode": "auto_review",
                }
                if case["expected"]:
                    expected_spawn["auto_pr_enabled"] = True
                mock_spawn.assert_called_once_with(**expected_spawn)

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_auto_qa_precedence_matrix(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        _setup_codex(mock_codex, models=[])
        cases: list[dict[str, Any]] = [
            {
                "name": "posted override enables bare repo",
                "post_auto_qa": "true",
                "expected_auto_pr": False,
                "expected_auto_qa": True,
            },
            {
                "name": "posted override disables global setting",
                "global_auto_qa": "true",
                "post_auto_qa": "false",
                "expected_auto_pr": False,
                "expected_auto_qa": False,
            },
            {
                "name": "global setting enables bare repo",
                "global_auto_qa": "true",
                "expected_auto_pr": False,
                "expected_auto_qa": True,
            },
            {
                "name": "auto-PR takes precedence",
                "post_auto_pr": "true",
                "post_auto_qa": "true",
                "expected_auto_pr": True,
                "expected_auto_qa": False,
            },
        ]

        for index, case in enumerate(cases):
            with self.subTest(case["name"]):
                client = Client()
                repo = f"{self.REPO}-auto-qa-{index}"
                thread_id = f"auto-qa-thread-{index}"
                data = {"prompt": "do thing", "cwd": repo}
                if "post_auto_pr" in case:
                    data["auto_pr"] = case["post_auto_pr"]
                if "post_auto_qa" in case:
                    data["auto_qa"] = case["post_auto_qa"]
                cookies: dict[str, str] = {}
                if "global_auto_qa" in case:
                    cookies[_AUTO_QA_COOKIE] = case["global_auto_qa"]
                if cookies:
                    _seed_cookies(client, **cookies)
                mock_discover.return_value = [Path(repo)]
                mock_spawn.return_value = SimpleNamespace(thread_id=thread_id)
                mock_spawn.reset_mock()

                response = client.post(reverse("new_session"), data=data)

                self.assertEqual(response.status_code, 302)
                metadata = SessionMetadata.objects.get(thread_id=thread_id)
                self.assertEqual(
                    metadata.auto_pr_enabled, case["expected_auto_pr"]
                )
                self.assertEqual(
                    metadata.auto_qa_enabled, case["expected_auto_qa"]
                )
                kwargs = mock_spawn.call_args.kwargs
                self.assertEqual(
                    kwargs.get("auto_pr_enabled", False),
                    case["expected_auto_pr"],
                )
                self.assertEqual(
                    kwargs.get("auto_qa_enabled", False),
                    case["expected_auto_qa"],
                )

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_auto_qa_forwards_qa_panel_to_spawn(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        _setup_codex(mock_codex, models=[])
        repo = f"{self.REPO}-auto-qa-panel"
        _seed_cookies(self.client, **{_QA_PANEL_COOKIE: "true"})
        mock_discover.return_value = [Path(repo)]
        mock_spawn.return_value = SimpleNamespace(thread_id="auto-qa-panel-thread")

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "do thing", "cwd": repo, "auto_qa": "true"},
        )

        self.assertEqual(response.status_code, 302)
        kwargs = mock_spawn.call_args.kwargs
        self.assertTrue(kwargs["auto_qa_enabled"])
        self.assertTrue(kwargs["qa_panel_enabled"])

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_project_routing_matrix(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        _setup_codex(mock_codex, models=[])
        cases: list[tuple[str, str, str, str | None, bool]] = [
            ("selected project", "selected", f"{self.REPO}/selected", "selected", False),
            ("posted repository", "cwd", "/home/user/other", "posted", False),
            ("posted project", "project", "/home/user/posted", "posted", False),
            ("bare repo override", "bare", "/home/user/bare", None, True),
            ("unprojected repo", "unprojected", "/home/user/unprojected", None, False),
        ]

        for index, (label, selector, repo, expected_project, cleared) in enumerate(cases):
            with self.subTest(label=label):
                client = Client()
                selected_repo = (
                    repo
                    if selector == "selected"
                    else f"{self.REPO}/selected-{index}"
                )
                selected_project = Project.objects.create(
                    name=f"Selected {index}", repo_path=selected_repo
                )
                posted_project = None
                if selector in {"cwd", "project", "bare"}:
                    posted_project = Project.objects.create(
                        name=f"Posted {index}", repo_path=repo
                    )
                data = {"prompt": "do thing"}
                if selector == "project":
                    assert posted_project is not None
                    data["project"] = str(posted_project.pk)
                else:
                    data["cwd"] = repo
                    if selector == "bare":
                        data["project"] = views._BARE_REPO_PROJECT_VALUE
                    _seed_cookies(
                        client, hitch_selected_project_id=str(selected_project.pk)
                    )
                discovered = [Path(selected_project.repo_path), Path(repo)]
                if posted_project is not None:
                    discovered.append(Path(posted_project.repo_path))
                mock_discover.return_value = discovered
                thread_id = f"thread-project-{index}"
                mock_spawn.return_value = SimpleNamespace(thread_id=thread_id)
                mock_spawn.reset_mock()

                response = client.post(reverse("new_session"), data=data)

                self.assertEqual(response.status_code, 302)
                self._assert_new_session_spawn(mock_spawn, cwd=repo)
                metadata = SessionMetadata.objects.get(thread_id=thread_id)
                self.assertEqual(metadata.cwd, repo)
                if expected_project == "selected":
                    self.assertEqual(metadata.project, selected_project)
                elif expected_project == "posted":
                    self.assertIsNotNone(posted_project)
                    self.assertEqual(metadata.project, posted_project)
                else:
                    self.assertIsNone(metadata.project)
                self.assertEqual(metadata.project_cleared, cleared)

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_forwards_cookie_settings_to_spawn_matrix(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        codex = _setup_codex(mock_codex, models=[])
        cases: list[tuple[str, dict[str, str], dict[str, Any]]] = [
            (
                "extra prompt",
                {
                    _EXTRA_SYSTEM_PROMPT_COOKIE: _encode_extra_system_prompt(
                        "  Always run focused tests.  "
                    )
                },
                {"developer_instructions": "Always run focused tests."},
            ),
            (
                "model effort sandbox",
                {
                    "hitch_model": "gpt-5",
                    "hitch_reasoning_effort": "high",
                    "hitch_sandbox_policy": "workspaceWrite",
                },
                {
                    "model": "gpt-5",
                    "reasoning_effort": "high",
                    "sandbox_policy": "workspaceWrite",
                },
            ),
            ("memories", {_ENABLE_MEMORIES_COOKIE: "true"}, {"enable_memories": True}),
            (
                "web search",
                {_WEB_SEARCH_COOKIE: "live"},
                {"web_search_mode": "live"},
            ),
            (
                "deny all approval",
                {"hitch_approval_mode": "deny_all"},
                {"approval_mode": "deny_all"},
            ),
            (
                "prompt user approval",
                {"hitch_approval_mode": "prompt_user"},
                {"approval_mode": "prompt_user"},
            ),
            (
                "hitch coding agent",
                {_CODING_AGENT_COOKIE: "hitch"},
                {"base_instructions": coding_agents.HITCH_BASE_INSTRUCTIONS},
            ),
        ]

        for index, (label, cookies, expected) in enumerate(cases):
            with self.subTest(label=label):
                self._clear_models_cache()
                client = Client()
                codex.models.return_value.data = (
                    [_make_model("gpt-5", is_default=True)]
                    if "hitch_model" in cookies
                    else []
                )
                mock_spawn.return_value = SimpleNamespace(thread_id=f"thread-{index}")
                mock_spawn.reset_mock()
                _seed_cookies(client, **cookies)

                response = client.post(
                    reverse("new_session"),
                    data={"prompt": "do thing", "cwd": self.REPO},
                )

                self.assertEqual(response.status_code, 302)
                self._assert_new_session_spawn(mock_spawn, **expected)

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_merges_global_and_project_developer_prompts(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        project = Project.objects.create(
            name="Hitch",
            repo_path=self.REPO,
            extra_system_prompt="Use project fixtures.",
        )
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-project-prompt")
        _setup_codex(mock_codex, models=[])
        _seed_cookies(
            self.client,
            **{
                _EXTRA_SYSTEM_PROMPT_COOKIE: _encode_extra_system_prompt(
                    "Always run focused tests."
                )
            },
        )

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "do thing", "project": str(project.pk)},
        )

        self.assertEqual(response.status_code, 302)
        self._assert_new_session_spawn(
            mock_spawn,
            developer_instructions=(
                "Always run focused tests.\n\nUse project fixtures."
            ),
        )

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_ignores_project_prompt_for_bare_repo_override(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        project = Project.objects.create(
            name="Hitch",
            repo_path=self.REPO,
            extra_system_prompt="Use project fixtures.",
        )
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-bare")
        _setup_codex(mock_codex, models=[])
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": "do thing",
                "cwd": self.REPO,
                "project": views._BARE_REPO_PROJECT_VALUE,
            },
        )

        self.assertEqual(response.status_code, 302)
        self._assert_new_session_spawn(mock_spawn)

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_web_search_override(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        _setup_codex(mock_codex, models=[])
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-web")
        _seed_cookies(self.client, **{_WEB_SEARCH_COOKIE: "cached"})

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": "do thing",
                "cwd": self.REPO,
                "web_search_mode": "disabled",
            },
        )

        self.assertEqual(response.status_code, 302)
        self._assert_new_session_spawn(mock_spawn, web_search_mode="disabled")

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_coding_agent_override_matrix(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        _setup_codex(mock_codex, models=[])
        cases: list[tuple[str, dict[str, str], dict[str, str], dict[str, Any]]] = [
            (
                "hitch override from codex setting",
                {_CODING_AGENT_COOKIE: "codex"},
                {"coding_agent": "hitch"},
                {"base_instructions": coding_agents.HITCH_BASE_INSTRUCTIONS},
            ),
            (
                "codex override from hitch setting",
                {_CODING_AGENT_COOKIE: "hitch"},
                {"coding_agent": "codex"},
                {},
            ),
        ]

        for index, (label, cookies, data, expected) in enumerate(cases):
            with self.subTest(label=label):
                client = Client()
                _seed_cookies(client, **cookies)
                mock_spawn.return_value = SimpleNamespace(thread_id=f"thread-{index}")
                mock_spawn.reset_mock()

                response = client.post(
                    reverse("new_session"),
                    data={"prompt": "do thing", "cwd": self.REPO, **data},
                )

                self.assertEqual(response.status_code, 302)
                self._assert_new_session_spawn(mock_spawn, **expected)

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_rejects_invalid_coding_agent_override(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        _setup_codex(mock_codex, models=[])

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": "do thing",
                "cwd": self.REPO,
                "coding_agent": "not-a-real-agent",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertContains(response, "invalid coding agent", status_code=400)

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_rejects_invalid_web_search_override(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        _setup_codex(mock_codex, models=[])

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": "do thing",
                "cwd": self.REPO,
                "web_search_mode": "maybe",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertContains(response, "invalid web search setting", status_code=400)
        mock_spawn.assert_not_called()

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_rejects_invalid_worktree_override(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        _setup_codex(mock_codex, models=[])

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": "do thing",
                "cwd": self.REPO,
                "use_worktrees": "maybe",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertContains(response, "invalid worktree setting", status_code=400)
        mock_spawn.assert_not_called()

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_plan_mode_matrix(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        _setup_codex(mock_codex, models=[_make_model("gpt-default", is_default=True)])
        cases: list[tuple[str, dict[str, str], dict[str, str], dict[str, Any]]] = [
            (
                "checkbox",
                {"prompt": "make a migration plan", "cwd": self.REPO, "plan_mode": "true"},
                {},
                {"model": "gpt-default", "reasoning_effort": "medium"},
            ),
            (
                "slash command",
                {"prompt": "/plan make a migration plan", "cwd": self.REPO},
                {_MODEL_COOKIE: "gpt-default"},
                {"model": "gpt-default"},
            ),
        ]

        for index, (label, data, cookies, expected) in enumerate(cases):
            with self.subTest(label=label):
                client = Client()
                mock_spawn.return_value = SimpleNamespace(thread_id=f"thread-plan-{index}")
                mock_spawn.reset_mock()
                if cookies:
                    _seed_cookies(client, **cookies)

                response = client.post(reverse("new_session"), data=data)

                self.assertEqual(response.status_code, 302)
                self._assert_new_session_spawn(
                    mock_spawn,
                    prompt="make a migration plan",
                    plan_mode=True,
                    **expected,
                )

    @patch("hitch.main.views.codex_pool.spawn_new_session")
    def test_plan_slash_command_without_prompt_is_rejected(
        self, mock_spawn: MagicMock
    ) -> None:
        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "/plan", "cwd": self.REPO},
        )

        self.assertEqual(response.status_code, 400)
        self.assertContains(response, "prompt is required", status_code=400)
        mock_spawn.assert_not_called()

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_plan_slash_command_allows_image_only_prompt(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-plan-image")
        _setup_codex(mock_codex, models=[_make_model("gpt-default", is_default=True)])

        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_EVENTS_DIR=Path(raw)),
        ):
            response = self.client.post(
                reverse("new_session"),
                data={
                    "prompt": "/plan",
                    "cwd": self.REPO,
                    "input_images": SimpleUploadedFile(
                        "screen.png", _PNG_BYTES, content_type="image/png"
                    ),
                },
            )

        self.assertEqual(response.status_code, 302)
        image_paths = mock_spawn.call_args.kwargs["input_image_paths"]
        self.assertEqual(len(image_paths), 1)
        self._assert_new_session_spawn(
            mock_spawn,
            prompt="",
            model="gpt-default",
            reasoning_effort="medium",
            plan_mode=True,
            input_image_paths=image_paths,
        )

    @patch("hitch.main.views.system_agents.start_pr_qa_workflow")
    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.create_session_thread")
    @patch("hitch.main.views.create_worktree_for_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_qa_workflow_slash_commands(
        self,
        mock_discover: MagicMock,
        mock_create_worktree: MagicMock,
        mock_create_thread: MagicMock,
        mock_codex: MagicMock,
        mock_start_workflow: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        codex = _setup_codex(
            mock_codex, models=[_make_model("gpt-5.4", is_default=True)]
        )
        cases: list[
            tuple[
                str,
                dict[str, str],
                dict[str, str],
                dict[str, Any],
                dict[str, Any],
            ]
        ] = [
            (
                "pr",
                {"prompt": "/PR", "cwd": self.REPO, "plan_mode": "true"},
                {
                    _MODEL_COOKIE: "gpt-5.4",
                    "hitch_reasoning_effort": "high",
                    _WEB_SEARCH_COOKIE: "live",
                    _EXTRA_SYSTEM_PROMPT_COOKIE: _encode_extra_system_prompt(
                        "Use repo conventions."
                    ),
                },
                {
                    "name": _PR_PROMPT,
                    "developer_instructions": "Use repo conventions.",
                    "model": "gpt-5.4",
                    "web_search_mode": "live",
                },
                {
                    "model": "gpt-5.4",
                    "reasoning_effort": "high",
                    "developer_instructions": "Use repo conventions.",
                    "web_search_mode": "live",
                },
            ),
            (
                "qa",
                {
                    "prompt": "/QA",
                    "cwd": self.REPO,
                    "plan_mode": "true",
                    "web_search_mode": "disabled",
                },
                {},
                {
                    "name": _QA_PROMPT,
                    "developer_instructions": None,
                    "model": "gpt-5.4",
                    "web_search_mode": "disabled",
                },
                {
                    "model": "gpt-5.4",
                    "reasoning_effort": "medium",
                    "developer_instructions": None,
                    "web_search_mode": "disabled",
                    "open_pr_on_lgtm": False,
                },
            ),
            (
                "qa hitch coding agent override",
                {"prompt": "/QA", "cwd": self.REPO, "coding_agent": "hitch"},
                {},
                {
                    "name": _QA_PROMPT,
                    "developer_instructions": None,
                    "model": "gpt-5.4",
                    "base_instructions": coding_agents.HITCH_BASE_INSTRUCTIONS,
                },
                {
                    "model": "gpt-5.4",
                    "reasoning_effort": "medium",
                    "developer_instructions": None,
                    "base_instructions": coding_agents.HITCH_BASE_INSTRUCTIONS,
                    "open_pr_on_lgtm": False,
                },
            ),
            (
                "pr uses selected repo when worktrees enabled",
                {"prompt": "/pr", "cwd": self.REPO},
                {_USE_WORKTREES_COOKIE: "true"},
                {"name": _PR_PROMPT, "developer_instructions": None, "model": None},
                {
                    "model": None,
                    "reasoning_effort": None,
                    "developer_instructions": None,
                },
            ),
        ]

        for index, (
            label,
            data,
            cookies,
            thread_kwargs,
            workflow_kwargs,
        ) in enumerate(cases):
            with self.subTest(label=label):
                self._clear_models_cache()
                client = Client()
                codex.models.return_value.data = (
                    []
                    if cookies.get(_USE_WORKTREES_COOKIE) == "true"
                    else [_make_model("gpt-5.4", is_default=True)]
                )
                mock_create_thread.return_value = f"thread-{index}"
                mock_create_thread.reset_mock()
                mock_create_worktree.reset_mock()
                mock_start_workflow.reset_mock()
                if cookies:
                    _seed_cookies(client, **cookies)

                response = client.post(reverse("new_session"), data=data)

                self.assertEqual(response.status_code, 302)
                mock_create_worktree.assert_not_called()
                mock_create_thread.assert_called_once_with(
                    cwd=self.REPO,
                    enable_memories=False,
                    **thread_kwargs,
                )
                mock_start_workflow.assert_called_once_with(
                    main_thread_id=f"thread-{index}",
                    cwd=self.REPO,
                    sandbox_policy=None,
                    approval_mode="auto_review",
                    enable_memories=False,
                    initial_user_message_index=0,
                    **workflow_kwargs,
                )

    @patch("hitch.main.views.system_agents.start_pr_qa_workflow")
    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.create_session_thread")
    @patch("hitch.main.views.discover_repos")
    def test_oneoff_qa_slash_does_not_persist_global_auto_review(
        self,
        mock_discover: MagicMock,
        mock_create_thread: MagicMock,
        mock_codex: MagicMock,
        mock_start_workflow: MagicMock,
    ) -> None:
        # A bare /qa is a one-off review, not a proposal acceptance. Even with
        # global auto-QA enabled, the resulting session must NOT carry auto-QA
        # forward, or every later follow-up in the review thread would be
        # auto-reviewed unexpectedly.
        mock_discover.return_value = [Path(self.REPO)]
        _setup_codex(mock_codex, models=[_make_model("gpt-5.4", is_default=True)])
        mock_create_thread.return_value = "oneoff-qa-thread"
        client = Client()
        _seed_cookies(client, **{_AUTO_QA_COOKIE: "true"})

        response = client.post(
            reverse("new_session"),
            data={"prompt": "/qa", "cwd": self.REPO},
        )

        self.assertEqual(response.status_code, 302)
        mock_start_workflow.assert_called_once()
        metadata = SessionMetadata.objects.get(thread_id="oneoff-qa-thread")
        self.assertFalse(metadata.auto_qa_enabled)
        self.assertFalse(metadata.auto_pr_enabled)
        self.assertFalse(metadata.auto_merge_to_local_branch)
        self.assertEqual(metadata.auto_merge_branch, "")

    @patch("hitch.main.views.system_agents.start_pr_qa_workflow")
    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.create_session_thread")
    @patch("hitch.main.views.discover_repos")
    def test_coding_agent_proposal_qa_slash_does_not_persist_global_auto_review(
        self,
        mock_discover: MagicMock,
        mock_create_thread: MagicMock,
        mock_codex: MagicMock,
        mock_start_workflow: MagicMock,
    ) -> None:
        # A coding-agent proposal (no autonomous goal) leaves the inbox's
        # auto-review inputs empty, so the proposal did not request auto-review.
        # Accepting it via /qa with global auto-QA enabled must not persist
        # auto-QA on the session: only proposal-requested settings carry forward.
        mock_discover.return_value = [Path(self.REPO)]
        _setup_codex(mock_codex, models=[_make_model("gpt-5.4", is_default=True)])
        mock_create_thread.return_value = "coding-proposal-thread"
        project = Project.objects.create(name="Hitch", repo_path=self.REPO)
        proposal = ProposedSession.objects.create(
            project=project,
            title="Tidy up logging",
        )
        self.assertIsNone(proposal.autonomous_goal)
        client = Client()
        _seed_cookies(client, **{_AUTO_QA_COOKIE: "true"})

        response = client.post(
            reverse("new_session"),
            data={
                "prompt": "/qa",
                "cwd": self.REPO,
                "proposed_session": str(proposal.pk),
            },
        )

        self.assertEqual(response.status_code, 302)
        mock_start_workflow.assert_called_once()
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_ACCEPTED)
        metadata = SessionMetadata.objects.get(thread_id="coding-proposal-thread")
        self.assertFalse(metadata.auto_qa_enabled)
        self.assertFalse(metadata.auto_pr_enabled)
        self.assertFalse(metadata.auto_merge_to_local_branch)
        self.assertEqual(metadata.auto_merge_branch, "")

    @patch("hitch.main.views.system_agents.start_pr_qa_workflow")
    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.create_session_thread")
    @patch("hitch.main.views.discover_repos")
    def test_proposal_qa_thread_create_failure_resets_start_claim(
        self,
        mock_discover: MagicMock,
        mock_create_thread: MagicMock,
        mock_codex: MagicMock,
        mock_start_workflow: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        _setup_codex(mock_codex, models=[])
        project = Project.objects.create(name="Hitch", repo_path=self.REPO)
        proposal = ProposedSession.objects.create(
            project=project,
            title="Tidy up logging",
        )
        mock_create_thread.side_effect = RuntimeError("thread failed")

        with self.assertRaises(RuntimeError):
            self.client.post(
                reverse("new_session"),
                data={
                    "prompt": "/qa",
                    "cwd": self.REPO,
                    "proposed_session": str(proposal.pk),
                },
            )

        mock_start_workflow.assert_not_called()
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_UNSET)
        self.assertIsNone(proposal.accepted_session)
        self.assertNotIn(
            ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY,
            proposal.outcome_metadata,
        )

    @patch("hitch.main.views.system_agents.start_pr_qa_workflow")
    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.create_session_thread")
    @patch("hitch.main.views.discover_repos")
    def test_proposal_qa_workflow_start_failure_resets_start_claim(
        self,
        mock_discover: MagicMock,
        mock_create_thread: MagicMock,
        mock_codex: MagicMock,
        mock_start_workflow: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        _setup_codex(mock_codex, models=[])
        project = Project.objects.create(name="Hitch", repo_path=self.REPO)
        proposal = ProposedSession.objects.create(
            project=project,
            title="Tidy up logging",
        )
        mock_create_thread.return_value = "proposal-qa-thread"
        mock_start_workflow.side_effect = RuntimeError("workflow failed")

        with self.assertRaises(RuntimeError):
            self.client.post(
                reverse("new_session"),
                data={
                    "prompt": "/qa",
                    "cwd": self.REPO,
                    "proposed_session": str(proposal.pk),
                },
            )

        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_UNSET)
        self.assertIsNone(proposal.accepted_session)
        self.assertNotIn(
            ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY,
            proposal.outcome_metadata,
        )

    @patch("hitch.main.views.system_agents.start_pr_qa_workflow")
    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.create_session_thread")
    @patch("hitch.main.views.discover_repos")
    def test_pr_new_session_project_assignment_matrix(
        self,
        mock_discover: MagicMock,
        mock_create_thread: MagicMock,
        mock_codex: MagicMock,
        mock_start_workflow: MagicMock,
    ) -> None:
        repo_b = "/home/user/other"
        project_a = Project.objects.create(name="Project A", repo_path=self.REPO)
        project_b = Project.objects.create(name="Project B", repo_path=repo_b)
        _setup_codex(mock_codex, models=[])
        cases = [
            (
                "posted repository",
                {"prompt": "/pr", "cwd": repo_b},
                {Path(self.REPO), Path(repo_b)},
                project_b,
                False,
                {project_a.pk},
            ),
            (
                "bare repo",
                {
                    "prompt": "/pr",
                    "project": views._BARE_REPO_PROJECT_VALUE,
                    "cwd": self.REPO,
                },
                {Path(self.REPO)},
                None,
                True,
                set(),
            ),
        ]

        for index, (
            label,
            data,
            discovered,
            expected_project,
            project_cleared,
            selected_projects,
        ) in enumerate(cases):
            with self.subTest(label=label):
                client = Client()
                for project_id in selected_projects:
                    _seed_cookies(client, hitch_selected_project_id=str(project_id))
                thread_id = f"thread-project-{index}"
                mock_discover.return_value = list(discovered)
                mock_create_thread.return_value = thread_id
                mock_create_thread.reset_mock()
                mock_start_workflow.reset_mock()

                response = client.post(reverse("new_session"), data=data)

                self.assertEqual(response.status_code, 302)
                metadata = SessionMetadata.objects.get(thread_id=thread_id)
                self.assertEqual(metadata.cwd, data["cwd"])
                self.assertEqual(metadata.project, expected_project)
                self.assertEqual(metadata.project_cleared, project_cleared)
                mock_start_workflow.assert_called_once()

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_plan_mode_returns_bad_request_when_model_cannot_be_resolved(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        _setup_codex(mock_codex, models=[])

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": "make a migration plan",
                "cwd": self.REPO,
                "plan_mode": "true",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertContains(response, "plan mode requires a model", status_code=400)
        mock_spawn.assert_not_called()

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_reconciles_stale_model_before_spawning(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        """A long-lived tab can POST with a session that names a model the
        running Codex no longer offers; reconcile catches it so
        ``thread_start(model=...)`` doesn't get a stale id."""
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        _setup_codex(mock_codex, models=[_make_model("gpt-5", is_default=True)])
        _seed_cookies(
            self.client, hitch_model="ancient-model", hitch_reasoning_effort="low"
        )

        self.client.post(
            reverse("new_session"),
            data={"prompt": "do thing", "cwd": self.REPO},
        )

        mock_spawn.assert_called_once_with(
            cwd=self.REPO,
            prompt="do thing",
            developer_instructions=None,
            model="gpt-5",
            reasoning_effort="medium",
            sandbox_policy=None,
            approval_mode="auto_review",
        )

    @patch("hitch.main.views.create_worktree_for_session")
    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_uses_managed_worktree_when_setting_enabled(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
        mock_create_worktree: MagicMock,
    ) -> None:
        worktree = Path("/home/user/.hitch/worktrees/proj/20260516120000-abcdef12")
        mock_discover.return_value = [Path(self.REPO)]
        mock_create_worktree.return_value = ManagedWorktree(
            path=worktree,
            branch="hitch/proj/20260516120000-abcdef12",
            source_repo=Path(self.REPO),
        )
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        _setup_codex(mock_codex, models=[])
        _seed_cookies(self.client, **{_USE_WORKTREES_COOKIE: "true"})

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "do thing", "cwd": self.REPO},
        )

        self.assertEqual(response.status_code, 302)
        mock_create_worktree.assert_called_once_with(self.REPO)
        mock_spawn.assert_called_once_with(
            cwd=str(worktree),
            prompt="do thing",
            developer_instructions=None,
            model=None,
            reasoning_effort=None,
            sandbox_policy=None,
            approval_mode="auto_review",
        )

    @patch("hitch.main.views.create_worktree_for_session")
    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_worktree_override_precedence_matrix(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
        mock_create_worktree: MagicMock,
    ) -> None:
        worktree = Path("/home/user/.hitch/worktrees/proj/20260516120000-abcdef12")
        mock_create_worktree.return_value = ManagedWorktree(
            path=worktree,
            branch="hitch/proj/20260516120000-abcdef12",
            source_repo=Path(self.REPO),
        )
        _setup_codex(mock_codex, models=[])
        cases: list[tuple[str, dict[str, str], str, str, bool]] = [
            (
                "posted override enables global default off",
                {},
                "true",
                str(worktree),
                True,
            ),
            (
                "posted override disables global setting",
                {_USE_WORKTREES_COOKIE: "true"},
                "false",
                self.REPO,
                False,
            ),
        ]

        for index, (
            label,
            cookies,
            post_use_worktrees,
            expected_cwd,
            expected_create,
        ) in enumerate(cases):
            with self.subTest(label):
                client = Client()
                if cookies:
                    _seed_cookies(client, **cookies)
                mock_discover.return_value = [Path(self.REPO)]
                mock_spawn.return_value = SimpleNamespace(thread_id=f"thread-{index}")
                mock_spawn.reset_mock()
                mock_create_worktree.reset_mock()

                response = client.post(
                    reverse("new_session"),
                    data={
                        "prompt": "do thing",
                        "cwd": self.REPO,
                        "use_worktrees": post_use_worktrees,
                    },
                )

                self.assertEqual(response.status_code, 302)
                if expected_create:
                    mock_create_worktree.assert_called_once_with(self.REPO)
                else:
                    mock_create_worktree.assert_not_called()
                self._assert_new_session_spawn(mock_spawn, cwd=expected_cwd)

    @patch("hitch.main.views.cleanup_worktree")
    @patch("hitch.main.views.create_worktree_for_session")
    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_cleans_up_managed_worktree_when_spawn_fails(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
        mock_create_worktree: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        worktree = ManagedWorktree(
            path=Path("/home/user/.hitch/worktrees/proj/20260516120000-abcdef12"),
            branch="hitch/proj/20260516120000-abcdef12",
            source_repo=Path(self.REPO),
        )
        mock_discover.return_value = [Path(self.REPO)]
        mock_create_worktree.return_value = worktree
        mock_spawn.side_effect = RuntimeError("spawn failed")
        _setup_codex(mock_codex, models=[])
        _seed_cookies(self.client, **{_USE_WORKTREES_COOKIE: "true"})

        with self.assertRaisesRegex(RuntimeError, "spawn failed"):
            self.client.post(
                reverse("new_session"),
                data={"prompt": "do thing", "cwd": self.REPO},
            )

        mock_cleanup.assert_called_once_with(worktree)

    @patch("hitch.main.views.cleanup_worktree")
    @patch("hitch.main.views.create_worktree_for_session")
    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_cleans_up_managed_worktree_when_upload_validation_fails(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
        mock_create_worktree: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        worktree = ManagedWorktree(
            path=Path("/home/user/.hitch/worktrees/proj/20260516120000-abcdef12"),
            branch="hitch/proj/20260516120000-abcdef12",
            source_repo=Path(self.REPO),
        )
        mock_discover.return_value = [Path(self.REPO)]
        mock_create_worktree.return_value = worktree
        _setup_codex(mock_codex, models=[])
        _seed_cookies(self.client, **{_USE_WORKTREES_COOKIE: "true"})

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": "use this screenshot",
                "cwd": self.REPO,
                "input_images": SimpleUploadedFile(
                    "screen.png", b"not an image", content_type="image/png"
                ),
            },
        )

        self.assertContains(
            response,
            "image attachment must be PNG, JPEG, GIF, or WebP",
            status_code=400,
        )
        mock_cleanup.assert_called_once_with(worktree)
        mock_spawn.assert_not_called()

    @patch("hitch.main.views.cleanup_worktree")
    @patch("hitch.main.views.create_worktree_for_session")
    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_preserves_spawn_error_when_managed_worktree_cleanup_fails(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
        mock_create_worktree: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        worktree = ManagedWorktree(
            path=Path("/home/user/.hitch/worktrees/proj/20260516120000-abcdef12"),
            branch="hitch/proj/20260516120000-abcdef12",
            source_repo=Path(self.REPO),
        )
        mock_discover.return_value = [Path(self.REPO)]
        mock_create_worktree.return_value = worktree
        mock_spawn.side_effect = RuntimeError("spawn failed")
        mock_cleanup.side_effect = WorktreeCleanupError("cleanup failed")
        _setup_codex(mock_codex, models=[])
        _seed_cookies(self.client, **{_USE_WORKTREES_COOKIE: "true"})

        with (
            self.assertLogs("hitch.main.views", level="ERROR") as logs,
            self.assertRaisesRegex(RuntimeError, "spawn failed"),
        ):
            self.client.post(
                reverse("new_session"),
                data={"prompt": "do thing", "cwd": self.REPO},
            )

        mock_cleanup.assert_called_once_with(worktree)
        self.assertIn("failed to clean up managed worktree", logs.output[0])

    @patch("hitch.main.views.create_worktree_for_session")
    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_reports_worktree_creation_failure(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
        mock_create_worktree: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        mock_create_worktree.side_effect = WorktreeCreationError("boom")
        _setup_codex(mock_codex, models=[])
        _seed_cookies(self.client, **{_USE_WORKTREES_COOKIE: "true"})

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "do thing", "cwd": self.REPO},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.content, b"boom")
        mock_spawn.assert_not_called()

    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_rejects_invalid_input(
        self, mock_discover: MagicMock, mock_spawn: MagicMock
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]

        cases: list[tuple[dict[str, str], str]] = [
            ({"prompt": "", "cwd": self.REPO}, "empty prompt"),
            ({"prompt": "hello", "cwd": ""}, "missing cwd"),
            ({"prompt": "hello", "cwd": "/etc"}, "cwd outside allowed list"),
        ]
        for data, label in cases:
            with self.subTest(label=label):
                mock_spawn.reset_mock()
                response = self.client.post(reverse("new_session"), data=data)
                self.assertEqual(response.status_code, 400)
                mock_spawn.assert_not_called()

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_renders_get(self, mock_codex: MagicMock, mock_discover: MagicMock) -> None:
        _setup_codex(mock_codex)
        mock_discover.return_value = []

        response = self.client.get(reverse("new_session"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Start a new session")
        self.assertContains(response, 'class="new-session-form"')
        self.assertContains(response, 'class="new-session-close"')
        self.assertContains(response, 'aria-label="Cancel new session"')
        self.assertContains(response, ">Cancel</a>", count=1)


class SendMessageViewTests(TestCase):
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

    @patch("hitch.main.views.codex_pool.steer_instance")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.codex_pool.steer_instance")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_steers_latest_active_when_form_has_no_instance(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_steer: MagicMock,
    ) -> None:
        instance = CodexInstance.objects.create(
            pid=123,
            thread_id="abc",
            cwd="/repo",
            prompt="already running",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_RUNNING,
        )

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "also lint"},
        )

        self.assertEqual(response.status_code, 302)
        mock_steer.assert_called_once_with(
            instance.pk,
            expected_thread_id="abc",
            prompt="also lint",
        )
        mock_spawn.assert_not_called()
        mock_codex.assert_not_called()

    @patch("hitch.main.views.codex_pool.steer_instance")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_steers_active_workflow_user_turn(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_steer: MagicMock,
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="abc",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_USER_STEERING_RUNNING,
        )
        instance = CodexInstance.objects.create(
            pid=123,
            thread_id="abc",
            cwd="/repo",
            prompt="user steering",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_USER,
            workflow_id=workflow.pk,
        )

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "also lint"},
        )

        self.assertEqual(response.status_code, 302)
        mock_steer.assert_called_once_with(
            instance.pk,
            expected_thread_id="abc",
            prompt="also lint",
        )
        mock_spawn.assert_not_called()
        mock_codex.assert_not_called()

    @patch("hitch.main.views.codex_pool.steer_instance")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_blocks_active_workflow_system_feedback_worker_steering(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_steer: MagicMock,
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="abc",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_FEEDBACK_RUNNING,
        )
        instance = CodexInstance.objects.create(
            pid=123,
            thread_id="abc",
            cwd="/repo",
            prompt="fix QA feedback",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_SYSTEM_FEEDBACK,
            workflow_id=workflow.pk,
        )

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "also lint", "active_instance": str(instance.pk)},
        )

        self.assertEqual(response.status_code, 400)
        self.assertContains(
            response, "PR workflow is running for this session", status_code=400
        )
        mock_steer.assert_not_called()
        mock_spawn.assert_not_called()
        mock_codex.assert_not_called()

    @patch("hitch.main.views.codex_pool.steer_instance")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_steers_active_instance_with_uploaded_image(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_steer: MagicMock,
    ) -> None:
        instance = CodexInstance.objects.create(
            pid=123,
            thread_id="abc",
            cwd="/repo",
            prompt="already running",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_RUNNING,
        )
        mock_steer.return_value = instance

        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_EVENTS_DIR=Path(raw)),
        ):
            response = self.client.post(
                reverse("send_message", kwargs={"session_id": "abc"}),
                data={
                    "prompt": "use this image",
                    "input_images": SimpleUploadedFile(
                        "screen.png", _PNG_BYTES, content_type="image/png"
                    ),
                },
            )

            self.assertEqual(response.status_code, 302)
            image_paths = mock_steer.call_args.kwargs["input_image_paths"]
            self.assertEqual(len(image_paths), 1)
            self.assertEqual(Path(image_paths[0]).read_bytes(), _PNG_BYTES)

        mock_steer.assert_called_once()
        self.assertEqual(mock_steer.call_args.args[0], instance.pk)
        self.assertEqual(mock_steer.call_args.kwargs["expected_thread_id"], "abc")
        self.assertEqual(mock_steer.call_args.kwargs["prompt"], "use this image")
        mock_spawn.assert_not_called()
        mock_codex.assert_not_called()

    @patch("hitch.main.views.codex_pool.steer_instance")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.steer_instance")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_failed_steer_falls_back_to_spawn_matrix(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_steer: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        mock_steer.return_value = None
        resolved_plan_path = str(self._make_resolved_plan_rollout())
        cases: list[
            tuple[str, dict[str, str], str | None, int | str, str, dict[str, Any]]
        ] = [
            (
                "posted stale instance",
                {"prompt": "follow up", "active_instance": "42"},
                None,
                42,
                "follow up",
                {},
            ),
            (
                "posted stale recomputes default plan mode",
                {
                    "prompt": "follow up",
                    "active_instance": "42",
                    "plan_mode": "true",
                    "default_plan_mode": "true",
                },
                resolved_plan_path,
                42,
                "follow up",
                {},
            ),
            (
                "posted stale keeps explicit plan mode",
                {
                    "prompt": "make another plan",
                    "active_instance": "42",
                    "plan_mode": "true",
                    "plan_mode_explicit": "true",
                },
                resolved_plan_path,
                42,
                "make another plan",
                {"model": "gpt-5", "plan_mode": True},
            ),
            (
                "latest active instance",
                {"prompt": "also lint"},
                None,
                "latest",
                "also lint",
                {},
            ),
        ]

        for label, data, rollout_path, steered_instance, prompt, expected in cases:
            with self.subTest(label=label):
                CodexInstance.objects.all().delete()
                SessionMetadata.objects.all().delete()
                if steered_instance == "latest":
                    instance = CodexInstance.objects.create(
                        pid=0,
                        thread_id="abc",
                        cwd="/repo",
                        prompt="launching",
                        events_path="/tmp/events.jsonl",
                        status=CodexInstance.STATUS_STARTING,
                    )
                    expected_instance = instance.pk
                else:
                    assert isinstance(steered_instance, int)
                    CodexInstance.objects.create(
                        pid=123,
                        thread_id="abc",
                        cwd="/repo",
                        prompt="newer work",
                        events_path="/tmp/events.jsonl",
                        status=CodexInstance.STATUS_RUNNING,
                    )
                    expected_instance = steered_instance
                self._patch_codex(mock_codex, path=rollout_path)
                mock_steer.reset_mock()
                mock_spawn.reset_mock()

                response = self.client.post(
                    reverse("send_message", kwargs={"session_id": "abc"}),
                    data=data,
                )

                self.assertEqual(response.status_code, 302)
                mock_steer.assert_called_once_with(
                    expected_instance,
                    expected_thread_id="abc",
                    prompt=prompt,
                )
                self._assert_follow_up_spawn(mock_spawn, prompt=prompt, **expected)

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.steer_instance", return_value=None)
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_failed_steer_falls_back_to_spawn_with_uploaded_image(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_steer: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex)
        mock_discover.return_value = [Path("/repo")]

        def fail_steer_and_delete_owned_copy(*_args: Any, **kwargs: Any) -> None:
            for image_path in kwargs.get("input_image_paths", []):
                Path(image_path).unlink()
            return None

        mock_steer.side_effect = fail_steer_and_delete_owned_copy

        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_EVENTS_DIR=Path(raw)),
        ):
            response = self.client.post(
                reverse("send_message", kwargs={"session_id": "abc"}),
                data={
                    "prompt": "use this screenshot",
                    "active_instance": "42",
                    "input_images": SimpleUploadedFile(
                        "screen.png", _PNG_BYTES, content_type="image/png"
                    ),
                },
            )

            self.assertEqual(response.status_code, 302)
            steered_paths = mock_steer.call_args.kwargs["input_image_paths"]
            spawned_paths = mock_spawn.call_args.kwargs["input_image_paths"]
            self.assertNotEqual(spawned_paths, steered_paths)
            self.assertEqual(len(spawned_paths), 1)
            self.assertEqual(Path(spawned_paths[0]).read_bytes(), _PNG_BYTES)
            self.assertFalse(Path(steered_paths[0]).exists())

        mock_steer.assert_called_once_with(
            42,
            expected_thread_id="abc",
            prompt="use this screenshot",
            input_image_paths=steered_paths,
        )
        self._assert_follow_up_spawn(
            mock_spawn,
            prompt="use this screenshot",
            input_image_paths=spawned_paths,
        )

    @patch("hitch.main.views.codex_pool.steer_instance")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_image_steer_attachment_cap_returns_bad_request_without_fallback(
        self,
        mock_codex: MagicMock,
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_spawns_turn_and_redirects(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex)
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "  follow-up question  "},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            response.headers["Location"],
            reverse("session", kwargs={"session_id": "abc"}),
        )
        # Whitespace is trimmed before forwarding.
        mock_spawn.assert_called_once_with(
            thread_id="abc",
            cwd="/repo",
            prompt="follow-up question",
            sandbox_policy=None,
            approval_mode="auto_review",
        )

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_first_follow_up_uses_project_developer_prompt(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        project = Project.objects.create(
            name="Hitch",
            repo_path="/repo",
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_idle_follow_up_resumes_from_disk_without_app_server(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        # An idle follow-up reads cwd/entries/plan-state from SessionMetadata +
        # the rollout file instead of a live thread_resume (which the detached
        # worker repeats moments later), so no app-server is opened here.
        rollout_path = self._make_rollout(
            [
                _rollout_line(
                    "event_msg", {"type": "user_message", "message": "Hi"}
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
            ]
        )
        SessionMetadata.objects.create(
            thread_id="abc", cwd="/repo", codex_path=str(rollout_path)
        )
        CodexInstance.objects.create(
            pid=1,
            thread_id="abc",
            cwd="/repo",
            prompt="prior turn",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_COMPLETED,
            model="gpt-5.4",
        )
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "follow-up"},
        )

        self.assertEqual(response.status_code, 302)
        self._assert_follow_up_spawn(mock_spawn)
        # The disk path never constructs a live app-server.
        mock_codex.assert_not_called()

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
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
            with views._MODELS_REFRESH_LOCK:
                views._MODELS_CACHE_VALUE = {}
                views._MODELS_CACHE_FETCHED_AT = {}
                views._MODELS_REFRESH_IN_FLIGHT = set()

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

    @patch("hitch.main.views.system_agents.spec_critic_should_run", return_value=True)
    @patch("hitch.main.views.system_agents.start_spec_critic_workflow")
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_spec_critic_resume_intercepts_ambiguous_implementation_prompt(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
        mock_start_spec_critic: MagicMock,
        mock_spec_critic_should_run: MagicMock,
    ) -> None:
        _seed_cookies(
            self.client,
            **{_SPEC_CRITIC_COOKIE: "true", _WEB_SEARCH_COOKIE: "cached"},
        )
        self._patch_codex(mock_codex, model="gpt-5.4", reasoning_effort="high")
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "Improve onboarding"},
        )

        self.assertEqual(response.status_code, 302)
        mock_spawn.assert_not_called()
        mock_spec_critic_should_run.assert_not_called()
        mock_start_spec_critic.assert_called_once_with(
            main_thread_id="abc",
            cwd="/repo",
            prompt="Improve onboarding",
            sandbox_policy=None,
            approval_mode="auto_review",
            model="gpt-5.4",
            reasoning_effort="high",
            developer_instructions=None,
            enable_memories=False,
            web_search_mode="cached",
            initial_user_message_index=0,
            auto_pr_enabled=False,
            auto_qa_enabled=False,
        )

    @patch("hitch.main.views.system_agents.spec_critic_should_run", return_value=True)
    @patch("hitch.main.views.system_agents.start_spec_critic_workflow")
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_spec_critic_resume_preserves_auto_merge_settings(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
        mock_start_spec_critic: MagicMock,
        mock_spec_critic_should_run: MagicMock,
    ) -> None:
        _seed_cookies(self.client, **{_SPEC_CRITIC_COOKIE: "true"})
        self._patch_codex(mock_codex)
        mock_discover.return_value = [Path("/repo")]
        SessionMetadata.objects.create(
            thread_id="abc",
            cwd="/repo",
            auto_qa_enabled=True,
            auto_merge_to_local_branch=True,
            auto_merge_branch="main",
        )

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "Improve onboarding"},
        )

        self.assertEqual(response.status_code, 302)
        mock_spawn.assert_not_called()
        mock_spec_critic_should_run.assert_not_called()
        kwargs = mock_start_spec_critic.call_args.kwargs
        self.assertFalse(kwargs["auto_pr_enabled"])
        self.assertTrue(kwargs["auto_qa_enabled"])
        self.assertTrue(kwargs["auto_merge_to_local_branch"])
        self.assertEqual(kwargs["auto_merge_branch"], "main")

    @patch("hitch.main.views.system_agents.spec_critic_should_run")
    @patch("hitch.main.views.system_agents.start_spec_critic_workflow")
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_spec_critic_resume_defers_classification_to_background(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
        mock_start_spec_critic: MagicMock,
        mock_spec_critic_should_run: MagicMock,
    ) -> None:
        # The view no longer branches on the classifier: it always hands off to
        # the workflow, which classifies on a background thread and either runs
        # the critique or the original prompt. The request must not block on the
        # classifier or spawn the turn synchronously.
        _seed_cookies(self.client, **{_SPEC_CRITIC_COOKIE: "true"})
        self._patch_codex(mock_codex)
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={
                "prompt": (
                    'Change the settings checkbox label from "Auto-PR" '
                    'to "Open PR automatically".'
                )
            },
        )

        self.assertEqual(response.status_code, 302)
        mock_spec_critic_should_run.assert_not_called()
        mock_spawn.assert_not_called()
        mock_start_spec_critic.assert_called_once()

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_spawns_turn_with_uploaded_image(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex)
        mock_discover.return_value = [Path("/repo")]

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

            self.assertEqual(response.status_code, 302)
            image_paths = mock_spawn.call_args.kwargs["input_image_paths"]
            self.assertEqual(len(image_paths), 1)
            self.assertEqual(Path(image_paths[0]).read_bytes(), _PNG_BYTES)

        mock_spawn.assert_called_once()
        self.assertEqual(mock_spawn.call_args.kwargs["prompt"], "use this screenshot")

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_spawns_turn_with_image_only_prompt(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex)
        mock_discover.return_value = [Path("/repo")]

        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_EVENTS_DIR=Path(raw)),
        ):
            response = self.client.post(
                reverse("send_message", kwargs={"session_id": "abc"}),
                data={
                    "prompt": "",
                    "input_images": SimpleUploadedFile(
                        "screen.png", _PNG_BYTES, content_type="image/png"
                    ),
                },
            )

            self.assertEqual(response.status_code, 302)
            image_paths = mock_spawn.call_args.kwargs["input_image_paths"]
            self.assertEqual(len(image_paths), 1)
            self.assertEqual(Path(image_paths[0]).read_bytes(), _PNG_BYTES)

        mock_spawn.assert_called_once()
        self.assertEqual(mock_spawn.call_args.kwargs["prompt"], "")

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_managed_worktrees", return_value=[])
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_send_message_cleans_uploaded_images_when_resume_validation_fails(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
        _mock_managed_worktrees: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex, cwd="/repo")
        mock_discover.return_value = [Path("/other")]

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
                "thread cwd is not an allowed repository",
                status_code=400,
            )
            mock_spawn.assert_not_called()
            attachments = Path(raw) / "attachments"
            self.assertEqual(
                [path for path in attachments.rglob("*") if path.is_file()],
                [],
            )

    @patch("hitch.main.views.codex_pool.steer_instance")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
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

            with patch("hitch.main.views._INPUT_IMAGE_MAX_BYTES", len(_PNG_BYTES) - 1):
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

    @patch("hitch.main.views.system_agents.start_pr_qa_workflow")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_send_message_rejects_workflow_image_uploads_before_side_effects(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_start_workflow: MagicMock,
    ) -> None:
        for prompt in ("/pr", "/qa", "/fix-pr"):
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
                    "image attachments are not supported for PR workflow requests",
                    status_code=400,
                )
                mock_start_workflow.assert_not_called()
                mock_spawn.assert_not_called()
                mock_codex.assert_not_called()

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_unwraps_pydantic_rootmodel_cwd(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        # The SDK's Thread.cwd is an AbsolutePathBuf (pydantic RootModel[str]),
        # not a bare str, so the view has to unwrap ``.root``.
        self._patch_codex(mock_codex, cwd=SimpleNamespace(root="/repo"))
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "hi"},
        )

        self.assertEqual(response.status_code, 302)
        mock_spawn.assert_called_once_with(
            thread_id="abc",
            cwd="/repo",
            prompt="hi",
            sandbox_policy=None,
            approval_mode="auto_review",
        )

    @patch("hitch.main.views.discover_managed_worktrees")
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_allows_follow_up_turns_in_managed_worktrees(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
        mock_managed_worktrees: MagicMock,
    ) -> None:
        worktree = "/home/user/.hitch/worktrees/proj/20260516120000-abcdef12"
        self._patch_codex(mock_codex, cwd=worktree)
        mock_discover.return_value = [Path("/repo")]
        mock_managed_worktrees.return_value = [Path(worktree)]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "hi"},
        )

        self.assertEqual(response.status_code, 302)
        mock_spawn.assert_called_once_with(
            thread_id="abc",
            cwd=worktree,
            prompt="hi",
            sandbox_policy=None,
            approval_mode="auto_review",
        )

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_follow_up_clears_previous_web_search_when_setting_is_default(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex)
        mock_discover.return_value = [Path("/repo")]
        CodexInstance.objects.create(
            pid=999,
            thread_id="abc",
            cwd="/repo",
            prompt="first",
            web_search_mode="live",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
        )

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
            approval_mode="auto_review",
            web_search_mode="",
        )

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_hitch_coding_agent_forwards_base_instructions_to_follow_up(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex)
        mock_discover.return_value = [Path("/repo")]
        _seed_cookies(self.client, **{_CODING_AGENT_COOKIE: "hitch"})

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "follow-up"},
        )

        self.assertEqual(response.status_code, 302)
        base_instructions = mock_spawn.call_args.kwargs["base_instructions"]
        self.assertIn("You are running inside HITCH", base_instructions)

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_codex_coding_agent_clears_previous_hitch_base_instructions(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex)
        mock_discover.return_value = [Path("/repo")]
        _seed_cookies(self.client, **{_CODING_AGENT_COOKIE: "codex"})
        CodexInstance.objects.create(
            pid=999,
            thread_id="abc",
            cwd="/repo",
            prompt="first",
            base_instructions=coding_agents.HITCH_BASE_INSTRUCTIONS,
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
        )

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "follow-up"},
        )

        self.assertEqual(response.status_code, 302)
        base_instructions = mock_spawn.call_args.kwargs["base_instructions"]
        self.assertEqual(
            base_instructions, coding_agents.default_codex_base_instructions()
        )
        self.assertNotIn("You are running inside HITCH", base_instructions)

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_codex_coding_agent_clears_unknown_previous_base_instructions(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex)
        mock_discover.return_value = [Path("/repo")]
        _seed_cookies(self.client, **{_CODING_AGENT_COOKIE: "codex"})

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "follow-up"},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            mock_spawn.call_args.kwargs["base_instructions"],
            coding_agents.default_codex_base_instructions(),
        )

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_auto_pr_session_marks_follow_up_turn(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        self._patch_codex(
            mock_codex,
            model="gpt-5.4",
            reasoning_effort="high",
        )
        mock_discover.return_value = [Path("/repo")]
        SessionMetadata.objects.create(
            thread_id="abc",
            cwd="/repo",
            auto_pr_enabled=True,
        )

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
            approval_mode="auto_review",
            auto_pr_enabled=True,
            user_message_index=0,
            stored_model="gpt-5.4",
            stored_reasoning_effort="high",
        )

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_auto_qa_session_marks_follow_up_turn(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        self._patch_codex(
            mock_codex,
            model="gpt-5.4",
            reasoning_effort="high",
        )
        mock_discover.return_value = [Path("/repo")]
        SessionMetadata.objects.create(
            thread_id="abc",
            cwd="/repo",
            auto_qa_enabled=True,
        )

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
            approval_mode="auto_review",
            auto_qa_enabled=True,
            user_message_index=0,
            stored_model="gpt-5.4",
            stored_reasoning_effort="high",
        )

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_auto_merge_session_marks_follow_up_turn(
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
            auto_qa_enabled=True,
            auto_merge_to_local_branch=True,
            auto_merge_branch="main",
        )

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "follow-up"},
        )

        self.assertEqual(response.status_code, 302)
        self.assertTrue(mock_spawn.call_args.kwargs["auto_qa_enabled"])
        self.assertTrue(mock_spawn.call_args.kwargs["auto_merge_to_local_branch"])
        self.assertEqual(mock_spawn.call_args.kwargs["auto_merge_branch"], "main")

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_auto_qa_session_forwards_qa_panel_to_follow_up_turn(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        self._patch_codex(
            mock_codex,
            model="gpt-5.4",
            reasoning_effort="high",
        )
        _seed_cookies(self.client, **{_QA_PANEL_COOKIE: "true"})
        mock_discover.return_value = [Path("/repo")]
        SessionMetadata.objects.create(
            thread_id="abc",
            cwd="/repo",
            auto_qa_enabled=True,
        )

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
            approval_mode="auto_review",
            auto_qa_enabled=True,
            user_message_index=0,
            stored_model="gpt-5.4",
            stored_reasoning_effort="high",
            qa_panel_enabled=True,
        )

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_plan_routing_to_spawn_matrix(
        self,
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
                    "prompt": "Implement the plan.",
                    "auto_pr_enabled": True,
                    "user_message_index": 1,
                    "stored_model": "gpt-5.4",
                    "stored_reasoning_effort": None,
                    "model": "gpt-5.4",
                    "collaboration_mode": "default",
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_plan_mode_state_uses_stored_fallback_when_rollout_unreadable(
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
            path="/nonexistent/rollout.jsonl",
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_plan_slash_follow_up_allows_image_only_prompt(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        self._patch_codex(mock_codex, model="gpt-5.4")

        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_EVENTS_DIR=Path(raw)),
        ):
            response = self.client.post(
                reverse("send_message", kwargs={"session_id": "abc"}),
                data={
                    "prompt": "/plan",
                    "input_images": SimpleUploadedFile(
                        "screen.png", _PNG_BYTES, content_type="image/png"
                    ),
                },
            )

        self.assertEqual(response.status_code, 302)
        image_paths = mock_spawn.call_args.kwargs["input_image_paths"]
        self.assertEqual(len(image_paths), 1)
        self._assert_follow_up_spawn(
            mock_spawn,
            prompt="",
            model="gpt-5.4",
            plan_mode=True,
            input_image_paths=image_paths,
        )

    @patch("hitch.main.views.system_agents.start_pr_qa_workflow")
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_pr_qa_activation_routes_to_workflow_matrix(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_start_workflow: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        cases: list[
            tuple[str, dict[str, str], str | None, str | None, dict[str, Any]]
        ] = [
            ("pr slash", {"prompt": "/pr"}, None, None, {}),
            (
                "qa slash ignores posted plan mode",
                {"prompt": "/qa", "plan_mode": "true"},
                None,
                "high",
                {"open_pr_on_lgtm": False},
            ),
            (
                "qa menu",
                {"prompt": _QA_PROMPT},
                None,
                None,
                {"open_pr_on_lgtm": False},
            ),
            (
                "pr slash after pending plan",
                {"prompt": "/pr", "plan_mode": "true"},
                "pending",
                None,
                {"initial_user_message_index": 1},
            ),
            (
                "pr menu after pending plan",
                {"prompt": _PR_PROMPT},
                "pending",
                None,
                {"initial_user_message_index": 1},
            ),
        ]

        for label, data, rollout, reasoning_effort, expected in cases:
            with self.subTest(label=label):
                rollout_path = None
                if rollout == "pending":
                    rollout_path = str(self._make_pending_plan_rollout())
                else:
                    self.assertIsNone(rollout)
                self._patch_codex(
                    mock_codex,
                    model="gpt-5.4",
                    reasoning_effort=reasoning_effort,
                    path=rollout_path,
                )
                mock_start_workflow.reset_mock()

                response = self.client.post(
                    reverse("send_message", kwargs={"session_id": "abc"}),
                    data=data,
                )

                self.assertEqual(response.status_code, 302)
                workflow_kwargs: dict[str, Any] = {
                    "main_thread_id": "abc",
                    "cwd": "/repo",
                    "sandbox_policy": None,
                    "approval_mode": "auto_review",
                    "model": "gpt-5.4",
                    "reasoning_effort": reasoning_effort,
                    "developer_instructions": None,
                    "enable_memories": False,
                    "initial_user_message_index": 0,
                }
                workflow_kwargs.update(expected)
                mock_start_workflow.assert_called_once_with(**workflow_kwargs)

    @patch("hitch.main.views.system_agents.start_pr_qa_workflow")
    @patch("hitch.main.views.system_agents.start_pr_monitor_workflow")
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_fix_pr_slash_starts_monitor_for_opened_pr(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_start_monitor: MagicMock,
        mock_start_workflow: MagicMock,
    ) -> None:
        pr_url = "https://github.com/cberner/hitch/pull/169"
        rollout_path = self._make_rollout(
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
            ]
        )
        self._patch_codex(
            mock_codex,
            model="gpt-5.4",
            reasoning_effort="high",
            path=str(rollout_path),
        )
        mock_discover.return_value = [Path("/repo")]
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="abc",
            cwd="/repo",
            status=SystemWorkflow.STATUS_COMPLETED,
            step=system_agents.STEP_PR_READY,
            state={
                "pr_handoff": {
                    "url": "https://github.com/cberner/hitch/pull/168",
                    "state": "closed",
                    "merged": False,
                }
            },
        )
        SystemWorkflow.objects.filter(pk=workflow.pk).update(
            updated_at=datetime.now(UTC) - timedelta(minutes=5)
        )

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "/fix-pr"},
        )

        self.assertEqual(response.status_code, 302)
        mock_start_workflow.assert_not_called()
        mock_start_monitor.assert_called_once_with(
            main_thread_id="abc",
            cwd="/repo",
            pr_url=pr_url,
            sandbox_policy=None,
            approval_mode="auto_review",
            model="gpt-5.4",
            reasoning_effort="high",
            developer_instructions=None,
            enable_memories=False,
            initial_user_message_index=1,
        )

    @patch("hitch.main.views.system_agents.start_pr_monitor_workflow")
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_fix_pr_slash_requires_opened_pr(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_start_monitor: MagicMock,
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
        mock_start_monitor.assert_not_called()

    @patch("hitch.main.views.system_agents.start_pr_monitor_workflow")
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_fix_pr_slash_rejects_lifecycle_superseded_pr_url(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_start_monitor: MagicMock,
    ) -> None:
        pr_url = "https://github.com/cberner/hitch/pull/169"
        rollout_path = self._make_rollout(
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
            ]
        )
        self._patch_codex(mock_codex, model="gpt-5.4", path=str(rollout_path))
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
        mock_start_monitor.assert_not_called()

    @patch("hitch.main.views.system_agents.start_pr_qa_workflow")
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_slash_commands_forward_session_auto_merge_branch_to_workflow(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_start_workflow: MagicMock,
    ) -> None:
        # SessionMetadata says "after QA approves, merge into ``release``
        # locally instead of opening a PR". The auto-review code path in
        # ``system_agents`` already honors this for auto_qa/auto_pr workers;
        # the manual /qa and /pr slash activations must too, or the user's
        # configured local merge silently disappears every time they trigger
        # QA from the composer.
        mock_discover.return_value = [Path("/repo")]
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        SessionMetadata.objects.create(
            thread_id="abc",
            cwd="/repo",
            project=project,
            auto_qa_enabled=True,
            auto_merge_to_local_branch=True,
            auto_merge_branch="release",
        )

        for prompt in ("/qa", "/pr"):
            with self.subTest(prompt=prompt):
                self._patch_codex(mock_codex, model="gpt-5.4")
                mock_start_workflow.reset_mock()

                response = self.client.post(
                    reverse("send_message", kwargs={"session_id": "abc"}),
                    data={"prompt": prompt},
                )

                self.assertEqual(response.status_code, 302)
                self.assertEqual(
                    mock_start_workflow.call_args.kwargs["auto_merge_branch"],
                    "release",
                )

    @patch("hitch.main.views.system_agents.start_pr_qa_workflow")
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_qa_slash_command_forwards_hitch_base_instructions_to_workflow(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_start_workflow: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex, model="gpt-5.4")
        mock_discover.return_value = [Path("/repo")]
        _seed_cookies(self.client, **{_CODING_AGENT_COOKIE: "hitch"})

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "/qa"},
        )

        self.assertEqual(response.status_code, 302)
        base_instructions = mock_start_workflow.call_args.kwargs["base_instructions"]
        self.assertIn("You are running inside HITCH", base_instructions)

    @patch("hitch.main.views.system_agents.start_pr_qa_workflow")
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_qa_slash_command_forwards_parallel_panel_setting(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_start_workflow: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex, model="gpt-5.4")
        mock_discover.return_value = [Path("/repo")]
        _seed_cookies(self.client, **{_QA_PANEL_COOKIE: "true"})

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "/qa"},
        )

        self.assertEqual(response.status_code, 302)
        self.assertTrue(mock_start_workflow.call_args.kwargs["qa_panel_enabled"])

    @patch("hitch.main.views.system_agents.start_pr_qa_workflow")
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_pr_slash_command_inherits_session_web_search_when_setting_is_default(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_start_workflow: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex, model="gpt-5.4")
        mock_discover.return_value = [Path("/repo")]
        CodexInstance.objects.create(
            pid=999,
            thread_id="abc",
            cwd="/repo",
            prompt="first",
            web_search_mode="live",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
        )

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "/pr"},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(mock_start_workflow.call_args.kwargs["web_search_mode"], "live")

    @patch("hitch.main.views.system_agents.start_pr_qa_workflow")
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_qa_slash_command_forwards_web_search_setting(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_start_workflow: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex, model="gpt-5.4")
        mock_discover.return_value = [Path("/repo")]
        _seed_cookies(self.client, **{_WEB_SEARCH_COOKIE: "cached"})

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "/qa"},
        )

        self.assertEqual(response.status_code, 302)
        kwargs = mock_start_workflow.call_args.kwargs
        self.assertEqual(kwargs["web_search_mode"], "cached")
        self.assertFalse(kwargs["open_pr_on_lgtm"])

    @patch("hitch.main.views.system_agents.start_pr_qa_workflow")
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_qa_slash_command_clears_hitch_base_instructions_for_codex(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_start_workflow: MagicMock,
    ) -> None:
        self._patch_codex(mock_codex, model="gpt-5.4")
        mock_discover.return_value = [Path("/repo")]
        _seed_cookies(self.client, **{_CODING_AGENT_COOKIE: "codex"})
        CodexInstance.objects.create(
            pid=999,
            thread_id="abc",
            cwd="/repo",
            prompt="first",
            base_instructions=coding_agents.HITCH_BASE_INSTRUCTIONS,
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
        )

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "/qa"},
        )

        self.assertEqual(response.status_code, 302)
        base_instructions = mock_start_workflow.call_args.kwargs["base_instructions"]
        self.assertEqual(
            base_instructions, coding_agents.default_codex_base_instructions()
        )
        self.assertNotIn("You are running inside HITCH", base_instructions)

    @patch("hitch.main.views.system_agents.start_user_steering_turn")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_running_qa_workflow_routes_normal_follow_up_to_user_steering(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_start_steering: MagicMock,
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="abc",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step="qa_running",
        )
        mock_start_steering.return_value = SimpleNamespace()

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "please also do this"},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            response.headers["Location"],
            reverse("session", kwargs={"session_id": "abc"}),
        )
        mock_start_steering.assert_called_once_with(
            workflow,
            prompt="please also do this",
        )
        mock_codex.assert_not_called()
        mock_spawn.assert_not_called()

    @patch("hitch.main.views.system_agents.start_user_steering_turn")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_running_pr_workflow_non_qa_step_blocks_normal_follow_up(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_start_steering: MagicMock,
    ) -> None:
        SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="abc",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_FEEDBACK_RUNNING,
        )

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "please also do this"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertContains(
            response, "PR workflow is running for this session", status_code=400
        )
        mock_start_steering.assert_not_called()
        mock_codex.assert_not_called()
        mock_spawn.assert_not_called()

    @patch("hitch.main.views.system_agents.start_pr_qa_workflow")
    @patch("hitch.main.views.Codex")
    def test_duplicate_pr_command_during_running_workflow_redirects(
        self, mock_codex: MagicMock, mock_start_workflow: MagicMock
    ) -> None:
        SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="abc",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step="qa_running",
        )

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "/pr"},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            response.headers["Location"],
            reverse("session", kwargs={"session_id": "abc"}),
        )
        mock_codex.assert_not_called()
        mock_start_workflow.assert_not_called()

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
    def test_rejects_get(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
    ) -> None:
        response = self.client.get(
            reverse("send_message", kwargs={"session_id": "abc"})
        )

        self.assertEqual(response.status_code, 405)
        mock_codex.assert_not_called()
        mock_spawn.assert_not_called()


class SetSessionNameViewTests(TestCase):
    @patch("hitch.main.views.Codex")
    def test_updates_name_and_response_shape(self, mock_codex: MagicMock) -> None:
        client = mock_codex.return_value.__enter__.return_value
        cases: list[tuple[str, dict[str, str], bool, int, str | None]] = [
            (
                "session redirect trims whitespace",
                {"name": "  New title  "},
                False,
                302,
                reverse("session", kwargs={"session_id": "abc"}),
            ),
            (
                "index redirect",
                {"name": "New title", "next": "index"},
                False,
                302,
                reverse("index"),
            ),
            (
                "ajax",
                {"name": "New title"},
                True,
                204,
                None,
            ),
        ]
        for label, data, ajax, status, location in cases:
            with self.subTest(label=label):
                client._client.thread_set_name.reset_mock()
                url = reverse("set_session_name", kwargs={"session_id": "abc"})
                if ajax:
                    response = self.client.post(
                        url,
                        data=data,
                        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
                    )
                else:
                    response = self.client.post(url, data=data)

                self.assertEqual(response.status_code, status)
                if location is None:
                    self.assertNotIn("Location", response.headers)
                else:
                    self.assertEqual(response.headers["Location"], location)
                client._client.thread_set_name.assert_called_once_with(
                    "abc", "New title"
                )

    @patch("hitch.main.views.Codex")
    def test_rejects_invalid_requests(self, mock_codex: MagicMock) -> None:
        # The form caps input client-side; the view enforces the same bounds
        # so a hand-crafted POST can't bypass them.
        cases: list[tuple[str, dict[str, str], str, int]] = [
            ("post", {"name": ""}, "empty", 400),
            ("post", {"name": "   "}, "whitespace only", 400),
            ("post", {"name": "x" * 201}, "over length cap", 400),
            ("get", {}, "method", 405),
        ]
        for method, data, label, status in cases:
            with self.subTest(label=label):
                url = reverse("set_session_name", kwargs={"session_id": "abc"})
                if method == "post":
                    response = self.client.post(url, data=data)
                else:
                    response = self.client.get(url)
                self.assertEqual(response.status_code, status)
        mock_codex.assert_not_called()


class StartSessionDemoViewTests(TestCase):
    @patch("hitch.main.views.demo.request_demo_start")
    @patch("hitch.main.views.system_agents.active_workflow_for_thread")
    def test_rejects_start_while_system_workflow_is_active(
        self, mock_active_workflow: MagicMock, mock_request_demo: MagicMock
    ) -> None:
        mock_active_workflow.return_value = SimpleNamespace()

        response = self.client.post(
            reverse("start_session_demo", kwargs={"session_id": "abc"})
        )

        self.assertEqual(response.status_code, 400)
        self.assertContains(
            response,
            "PR workflow is running for this session",
            status_code=400,
        )
        mock_request_demo.assert_not_called()

    @patch("hitch.main.views.demo.request_demo_start")
    @patch("hitch.main.views.system_agents.active_workflow_for_thread", return_value=None)
    def test_rejects_start_while_user_turn_is_active(
        self, _mock_active_workflow: MagicMock, mock_request_demo: MagicMock
    ) -> None:
        CodexInstance.objects.create(
            thread_id="abc",
            cwd="/repo",
            prompt="user turn",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_RUNNING,
            pid=123,
        )

        response = self.client.post(
            reverse("start_session_demo", kwargs={"session_id": "abc"})
        )

        self.assertEqual(response.status_code, 400)
        self.assertContains(
            response,
            "Codex is already working for this session",
            status_code=400,
        )
        mock_request_demo.assert_not_called()

    @override_settings(HITCH_DEMO_RUNTIME="docker")
    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.demo.request_demo_start")
    @patch("hitch.main.views.system_agents.active_workflow_for_thread", return_value=None)
    def test_rejects_unsupported_runtime_before_spawning_agent(
        self,
        _mock_active_workflow: MagicMock,
        mock_request_demo: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        response = self.client.post(
            reverse("start_session_demo", kwargs={"session_id": "abc"})
        )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.content, b"only podman demo runtime is supported")
        mock_request_demo.assert_not_called()
        mock_codex.assert_not_called()

    @patch("hitch.main.views.demo.cleanup_unregistered_demo_containers")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.system_agents.active_workflow_for_thread", return_value=None)
    @patch("hitch.main.views.discover_managed_worktrees", return_value=[])
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.run_borrowed_op_with_retry")
    def test_requests_demo_agent_turn(
        self,
        mock_run_borrowed: MagicMock,
        mock_discover: MagicMock,
        _mock_managed: MagicMock,
        _mock_workflow: MagicMock,
        mock_spawn: MagicMock,
        _mock_cleanup: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        client = SimpleNamespace(
            _client=SimpleNamespace(thread_resume=MagicMock())
        )
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(cwd="/repo", turns=[])
        )
        mock_run_borrowed.side_effect = _run_borrowed_with(client)
        spawned_instances: list[CodexInstance] = []

        def spawn_side_effect(**_kwargs: object) -> CodexInstance:
            instance = CodexInstance.objects.create(
                thread_id="abc",
                cwd="/repo",
                prompt="demo",
                events_path="/tmp/events.jsonl",
                status=CodexInstance.STATUS_RUNNING,
                pid=123,
                agent_kind=demo.DEMO_AGENT_KIND,
            )
            spawned_instances.append(instance)
            return instance

        mock_spawn.side_effect = spawn_side_effect
        _seed_cookies(self.client, **{_WEB_SEARCH_COOKIE: "live"})

        response = self.client.post(
            reverse("start_session_demo", kwargs={"session_id": "abc"})
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], reverse("session", kwargs={"session_id": "abc"}))
        mock_spawn.assert_called_once()
        kwargs = mock_spawn.call_args.kwargs
        self.assertEqual(kwargs["thread_id"], "abc")
        self.assertEqual(kwargs["cwd"], "/repo")
        self.assertEqual(kwargs["purpose"], CodexInstance.PURPOSE_SYSTEM_AGENT)
        self.assertEqual(kwargs["agent_kind"], demo.DEMO_AGENT_KIND)
        self.assertEqual(kwargs["web_search_mode"], "live")
        self.assertNotIn("output_schema", kwargs)
        self.assertIsNone(kwargs["user_message_index"])
        self.assertIn("Start an interactive web demo", kwargs["prompt"])
        self.assertIn("Registration token:", kwargs["prompt"])
        self.assertIn("io.hitch.managed=demo", kwargs["prompt"])
        self.assertIn("http://testserver/sessions/abc/demo/", kwargs["prompt"])
        client._client.thread_resume.assert_called_once_with("abc")
        mock_run_borrowed.assert_called_once()
        self.assertIs(mock_run_borrowed.call_args.args[0], Codex)
        self.assertEqual(
            mock_run_borrowed.call_args.kwargs,
            {"enable_memories": False},
        )
        session_demo = SessionDemo.objects.get(thread_id="abc")
        self.assertTrue(session_demo.registration_token)
        self.assertEqual(spawned_instances[0].agent_kind, demo.DEMO_AGENT_KIND)
        workflow = SystemWorkflow.objects.get(
            kind=demo.DEMO_WORKFLOW_KIND,
            main_thread_id="abc",
        )
        run = SystemAgentRun.objects.get(workflow=workflow)
        self.assertEqual(run.agent_kind, demo.DEMO_AGENT_KIND)
        self.assertEqual(run.thread_id, "abc")
        self.assertEqual(run.instance, spawned_instances[0])

    @patch("hitch.main.views.demo.cleanup_demo_for_session")
    @patch("hitch.main.views.codex_pool.spawn_turn", side_effect=RuntimeError("spawn failed"))
    @patch("hitch.main.views.system_agents.active_workflow_for_thread", return_value=None)
    @patch("hitch.main.views.discover_managed_worktrees", return_value=[])
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_cleans_up_demo_when_worker_dispatch_fails(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        _mock_managed: MagicMock,
        _mock_workflow: MagicMock,
        _mock_spawn: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(cwd="/repo", turns=[])
        )
        with self.assertRaisesRegex(RuntimeError, "spawn failed"):
            self.client.post(reverse("start_session_demo", kwargs={"session_id": "abc"}))

        mock_cleanup.assert_called_once_with("abc")

    @patch("hitch.main.views.demo.cleanup_demo_for_session")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.system_agents.active_workflow_for_thread", return_value=None)
    @patch("hitch.main.views.discover_managed_worktrees", return_value=[])
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_cleans_up_demo_when_workflow_state_save_fails(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        _mock_managed: MagicMock,
        _mock_workflow: MagicMock,
        mock_spawn: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(cwd="/repo", turns=[])
        )
        original_save = SystemWorkflow.save

        def save_side_effect(
            workflow: SystemWorkflow, *args: Any, **kwargs: Any
        ) -> None:
            if kwargs.get("update_fields") == ["state", "updated_at"]:
                raise RuntimeError("state save failed")
            original_save(workflow, *args, **kwargs)

        with patch.object(SystemWorkflow, "save", autospec=True) as mock_save:
            mock_save.side_effect = save_side_effect
            with self.assertRaisesRegex(RuntimeError, "state save failed"):
                self.client.post(reverse("start_session_demo", kwargs={"session_id": "abc"}))

        workflow = SystemWorkflow.objects.get(
            kind=demo.DEMO_WORKFLOW_KIND,
            main_thread_id="abc",
        )
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_FAILED)
        mock_spawn.assert_not_called()
        mock_cleanup.assert_called_once_with("abc")

    @patch("hitch.main.views.demo.cleanup_demo_for_session")
    @patch(
        "hitch.main.views.demo.start_demo_prompt_for",
        side_effect=RuntimeError("prompt failed"),
    )
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.system_agents.active_workflow_for_thread", return_value=None)
    @patch("hitch.main.views.discover_managed_worktrees", return_value=[])
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_cleans_up_demo_when_prompt_construction_fails(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        _mock_managed: MagicMock,
        _mock_workflow: MagicMock,
        mock_spawn: MagicMock,
        _mock_prompt: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(cwd="/repo", turns=[])
        )

        with self.assertRaisesRegex(RuntimeError, "prompt failed"):
            self.client.post(reverse("start_session_demo", kwargs={"session_id": "abc"}))

        workflow = SystemWorkflow.objects.get(
            kind=demo.DEMO_WORKFLOW_KIND,
            main_thread_id="abc",
        )
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_FAILED)
        mock_spawn.assert_not_called()
        mock_cleanup.assert_called_once_with("abc")

    @patch("hitch.main.views.demo.cleanup_unregistered_demo_containers")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.system_agents.active_workflow_for_thread", return_value=None)
    @patch("hitch.main.views.discover_managed_worktrees", return_value=[])
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_requests_demo_agent_turn_tolerates_existing_system_run(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        _mock_managed: MagicMock,
        _mock_workflow: MagicMock,
        mock_spawn: MagicMock,
        _mock_cleanup: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(cwd="/repo", turns=[])
        )

        def spawn_side_effect(**kwargs: object) -> CodexInstance:
            workflow_id = cast(int, kwargs["workflow_id"])
            instance = CodexInstance.objects.create(
                thread_id="abc",
                cwd="/repo",
                prompt="demo",
                events_path="/tmp/events.jsonl",
                status=CodexInstance.STATUS_COMPLETED,
                pid=123,
                purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
                workflow_id=workflow_id,
                agent_kind=demo.DEMO_AGENT_KIND,
            )
            workflow = SystemWorkflow.objects.get(pk=workflow_id)
            SystemAgentRun.objects.create(
                workflow=workflow,
                agent_kind=demo.DEMO_AGENT_KIND,
                thread_id="abc",
                instance=instance,
                status=SystemAgentRun.STATUS_COMPLETED,
            )
            return instance

        mock_spawn.side_effect = spawn_side_effect

        response = self.client.post(
            reverse("start_session_demo", kwargs={"session_id": "abc"})
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(SystemAgentRun.objects.count(), 1)
        self.assertEqual(
            SystemAgentRun.objects.get().status,
            SystemAgentRun.STATUS_COMPLETED,
        )

    @patch("hitch.main.views.demo.cleanup_demo_for_session")
    @patch("hitch.main.views.demo.cleanup_unregistered_demo_containers")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.system_agents.active_workflow_for_thread", return_value=None)
    @patch("hitch.main.views.discover_managed_worktrees", return_value=[])
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_duplicate_running_demo_workflow_rejects_without_mutating_owner(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        _mock_managed: MagicMock,
        _mock_workflow: MagicMock,
        mock_spawn: MagicMock,
        _mock_sweep: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(cwd="/repo", turns=[])
        )
        stale_workflow = SystemWorkflow.objects.create(
            kind=demo.DEMO_WORKFLOW_KIND,
            main_thread_id="abc",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
        )

        response = self.client.post(
            reverse("start_session_demo", kwargs={"session_id": "abc"})
        )

        self.assertEqual(response.status_code, 400)
        self.assertContains(
            response,
            "demo setup workflow is already running",
            status_code=400,
        )
        mock_codex.assert_not_called()
        mock_spawn.assert_not_called()
        mock_cleanup.assert_not_called()
        self.assertFalse(SessionDemo.objects.filter(thread_id="abc").exists())
        stale_workflow.refresh_from_db()
        self.assertEqual(stale_workflow.status, SystemWorkflow.STATUS_RUNNING)

    @patch("hitch.main.views.demo.request_demo_start")
    @patch("hitch.main.views.SystemWorkflow.objects.create")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.system_agents.active_workflow_for_thread", return_value=None)
    @patch("hitch.main.views.discover_managed_worktrees", return_value=[])
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_demo_workflow_integrity_error_rejects_before_mutating_demo_state(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        _mock_managed: MagicMock,
        _mock_workflow: MagicMock,
        mock_spawn: MagicMock,
        mock_create_workflow: MagicMock,
        mock_request_demo: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(cwd="/repo", turns=[])
        )
        mock_create_workflow.side_effect = IntegrityError(
            "uniq_running_system_workflow"
        )

        response = self.client.post(
            reverse("start_session_demo", kwargs={"session_id": "abc"})
        )

        self.assertEqual(response.status_code, 400)
        self.assertContains(
            response,
            "demo setup workflow is already running",
            status_code=400,
        )
        mock_request_demo.assert_not_called()
        mock_spawn.assert_not_called()
        self.assertFalse(SessionDemo.objects.filter(thread_id="abc").exists())

    @patch("hitch.main.views.Codex")
    def test_system_sessions_lists_demo_run_without_hiding_user_session(
        self, mock_codex: MagicMock
    ) -> None:
        thread = _session("thread-1", preview="User feature")
        _setup_codex(mock_codex, threads=[thread])
        workflow = SystemWorkflow.objects.create(
            kind=demo.DEMO_WORKFLOW_KIND,
            main_thread_id="thread-1",
            cwd="/repo",
            status=SystemWorkflow.STATUS_FAILED,
        )
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="thread-1",
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
            thread_id="thread-1",
            instance=instance,
            status=SystemAgentRun.STATUS_FAILED,
        )

        index_response = self.client.get(reverse("index"))
        system_response = self.client.get(reverse("system_sessions"))

        self.assertContains(index_response, "User feature")
        self.assertContains(system_response, "User feature")
        self.assertContains(system_response, "Demo agent")
        self.assertContains(
            system_response,
            reverse("system_session", kwargs={"session_id": "thread-1"}),
        )

    @patch("hitch.main.views.demo.request_demo_start")
    @patch("hitch.main.views.system_agents.active_workflow_for_thread", return_value=None)
    @patch("hitch.main.views.discover_managed_worktrees", return_value=[])
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_rejects_missing_cwd_before_starting_container(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        _mock_managed: MagicMock,
        _mock_workflow: MagicMock,
        mock_request_demo: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(cwd="", turns=[])
        )

        response = self.client.post(reverse("start_session_demo", kwargs={"session_id": "abc"}))

        self.assertEqual(response.status_code, 400)
        self.assertContains(response, "thread has no cwd", status_code=400)
        mock_request_demo.assert_not_called()

    @patch("hitch.main.views.demo.request_demo_start", side_effect=demo.DemoError("no podman"))
    @patch("hitch.main.views.system_agents.active_workflow_for_thread", return_value=None)
    @patch("hitch.main.views.discover_managed_worktrees", return_value=[])
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_reports_demo_start_failure(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        _mock_managed: MagicMock,
        _mock_workflow: MagicMock,
        _mock_request_demo: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(cwd="/repo", turns=[])
        )
        original_save = SystemWorkflow.save

        def save_side_effect(
            workflow: SystemWorkflow, *args: Any, **kwargs: Any
        ) -> None:
            if kwargs.get("update_fields") == ["status", "updated_at"]:
                raise AssertionError("status failure should use queryset update")
            original_save(workflow, *args, **kwargs)

        with patch.object(SystemWorkflow, "save", autospec=True) as mock_save:
            mock_save.side_effect = save_side_effect
            response = self.client.post(
                reverse("start_session_demo", kwargs={"session_id": "abc"})
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.content, b"no podman")
        workflow = SystemWorkflow.objects.get(
            kind=demo.DEMO_WORKFLOW_KIND,
            main_thread_id="abc",
        )
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_FAILED)

    @patch("hitch.main.views.demo.request_demo_start", side_effect=RuntimeError("boom"))
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.system_agents.active_workflow_for_thread", return_value=None)
    @patch("hitch.main.views.discover_managed_worktrees", return_value=[])
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_fails_workflow_when_demo_start_raises_unexpected_error(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        _mock_managed: MagicMock,
        _mock_workflow: MagicMock,
        mock_spawn: MagicMock,
        _mock_request_demo: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(cwd="/repo", turns=[])
        )

        with self.assertRaisesRegex(RuntimeError, "boom"):
            self.client.post(reverse("start_session_demo", kwargs={"session_id": "abc"}))

        workflow = SystemWorkflow.objects.get(
            kind=demo.DEMO_WORKFLOW_KIND,
            main_thread_id="abc",
        )
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_FAILED)
        mock_spawn.assert_not_called()

    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.system_agents.active_workflow_for_thread", return_value=None)
    @patch("hitch.main.views.discover_managed_worktrees", return_value=[])
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_rejects_pending_demo_before_spawning_agent(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        _mock_managed: MagicMock,
        _mock_workflow: MagicMock,
        mock_spawn: MagicMock,
    ) -> None:
        SessionDemo.objects.create(
            thread_id="abc",
            host="127.0.0.1",
            port=3000,
            status=SessionDemo.STATUS_REQUESTED,
            registration_token="token",
        )
        mock_discover.return_value = [Path("/repo")]
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(cwd="/repo", turns=[])
        )

        response = self.client.post(reverse("start_session_demo", kwargs={"session_id": "abc"}))

        self.assertEqual(response.status_code, 400)
        self.assertContains(response, "demo setup is already running", status_code=400)
        mock_spawn.assert_not_called()

    @patch("hitch.main.views.demo.request_demo_start")
    @patch("hitch.main.views.system_agents.active_workflow_for_thread", return_value=None)
    @patch("hitch.main.views.discover_managed_worktrees", return_value=[])
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_rejects_unallowed_cwd_before_starting_container(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        _mock_managed: MagicMock,
        _mock_workflow: MagicMock,
        mock_request_demo: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(cwd="/elsewhere", turns=[])
        )

        response = self.client.post(
            reverse("start_session_demo", kwargs={"session_id": "abc"})
        )

        self.assertEqual(response.status_code, 400)
        mock_request_demo.assert_not_called()


class RegisterSessionDemoViewTests(TestCase):
    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    @patch("hitch.main.demo._verify_registered_container_labels")
    def test_registers_active_demo_from_json(
        self, _mock_verify: MagicMock, _cleanup: MagicMock
    ) -> None:
        session_demo = demo.request_demo_start("abc")

        response = self.client.post(
            reverse("session_demo_register", kwargs={"session_id": "abc"}),
            data=json.dumps(
                {
                    "token": session_demo.registration_token,
                    "status": "active",
                    "container_name": "hitch-demo-abc-1234",
                    "container_id": "container123",
                    "host": "127.0.0.1",
                    "port": 45678,
                }
            ),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.content)
        self.assertEqual(payload["status"], SessionDemo.STATUS_ACTIVE)
        session_demo.refresh_from_db()
        self.assertEqual(session_demo.container_name, "hitch-demo-abc-1234")
        self.assertEqual(session_demo.port, 45678)

    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    def test_rejects_invalid_registration_token(self, _cleanup: MagicMock) -> None:
        demo.request_demo_start("abc")

        response = self.client.post(
            reverse("session_demo_register", kwargs={"session_id": "abc"}),
            data=json.dumps(
                {
                    "token": "bad",
                    "status": "preparing",
                    "container_name": "hitch-demo-abc-1234",
                }
            ),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.content, b"invalid demo registration token")

    def test_rejects_invalid_json(self) -> None:
        response = self.client.post(
            reverse("session_demo_register", kwargs={"session_id": "abc"}),
            data=b"{",
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.content, b"invalid JSON")


class SetSessionArchivedViewTests(TestCase):
    @patch("hitch.main.views.Codex")
    def test_updates_archive_state_and_response_shape(
        self, mock_codex: MagicMock
    ) -> None:
        client = mock_codex.return_value.__enter__.return_value
        cases: list[
            tuple[str, dict[str, str], bool, int, str | None, str, bool]
        ] = [
            (
                "archive",
                {"archived": "true"},
                False,
                302,
                reverse("index"),
                "thread_archive",
                True,
            ),
            (
                "archive ajax",
                {"archived": "true"},
                True,
                204,
                None,
                "thread_archive",
                False,
            ),
            (
                "unarchive to session",
                {"archived": "false"},
                False,
                302,
                reverse("session", kwargs={"session_id": "abc"}),
                "thread_unarchive",
                True,
            ),
            (
                "unarchive to index",
                {"archived": "false", "next": "index"},
                False,
                302,
                reverse("index"),
                "thread_unarchive",
                False,
            ),
        ]
        for label, data, ajax, status, location, expected_call, seed_cache in cases:
            with self.subTest(label=label):
                ArchivedSessionTokenUsage.objects.all().delete()
                client.thread_archive.reset_mock()
                client.thread_unarchive.reset_mock()
                if seed_cache:
                    ArchivedSessionTokenUsage.objects.create(
                        thread_id="abc",
                        total_tokens=100,
                    )
                    ArchivedSessionTokenUsage.objects.create(
                        thread_id="abc-child",
                        total_tokens=200,
                    )

                url = reverse("set_session_archived", kwargs={"session_id": "abc"})
                if ajax:
                    response = self.client.post(
                        url,
                        data=data,
                        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
                    )
                else:
                    response = self.client.post(url, data=data)

                self.assertEqual(response.status_code, status)
                if location is None:
                    self.assertNotIn("Location", response.headers)
                else:
                    self.assertEqual(response.headers["Location"], location)
                if expected_call == "thread_archive":
                    client.thread_archive.assert_called_once_with("abc")
                    client.thread_unarchive.assert_not_called()
                else:
                    client.thread_unarchive.assert_called_once_with("abc")
                    client.thread_archive.assert_not_called()
                if seed_cache:
                    # The toggled session's cache is dropped because its
                    # rollout path moves when codex archives/unarchives it,
                    # but every other session's cache must survive — wiping
                    # the whole table forces /profile and /usage to re-parse
                    # every archived rollout file the next time they render.
                    self.assertFalse(
                        ArchivedSessionTokenUsage.objects.filter(
                            thread_id="abc"
                        ).exists()
                    )
                    self.assertTrue(
                        ArchivedSessionTokenUsage.objects.filter(
                            thread_id="abc-child"
                        ).exists()
                    )

    @patch("hitch.main.views.Codex")
    def test_archive_keeps_cached_usage_for_unrelated_sessions(
        self, mock_codex: MagicMock
    ) -> None:
        ArchivedSessionTokenUsage.objects.create(
            thread_id="abc", total_tokens=100
        )
        ArchivedSessionTokenUsage.objects.create(
            thread_id="other-1", total_tokens=200
        )
        ArchivedSessionTokenUsage.objects.create(
            thread_id="other-2", total_tokens=300
        )

        response = self.client.post(
            reverse("set_session_archived", kwargs={"session_id": "abc"}),
            data={"archived": "true"},
        )

        self.assertEqual(response.status_code, 302)
        self.assertFalse(
            ArchivedSessionTokenUsage.objects.filter(thread_id="abc").exists()
        )
        other_totals = dict(
            ArchivedSessionTokenUsage.objects.filter(
                thread_id__in=["other-1", "other-2"]
            ).values_list("thread_id", "total_tokens")
        )
        self.assertEqual(other_totals, {"other-1": 200, "other-2": 300})

    @patch("hitch.main.demo.subprocess.run")
    @patch("hitch.main.views.Codex")
    def test_archive_cleans_up_active_demo_container(
        self, mock_codex: MagicMock, mock_run: MagicMock
    ) -> None:
        mock_run.side_effect = [
            SimpleNamespace(
                stdout=(
                    '[{"Config":{"Labels":{'
                    '"io.hitch.managed":"demo",'
                    '"io.hitch.session":"abc",'
                    '"io.hitch.demo_token":"token",'
                    '"io.hitch.container_name":"hitch-demo-abc-abcd"'
                    "}}}]"
                ),
                stderr="",
                returncode=0,
            ),
            SimpleNamespace(stdout="", stderr="", returncode=0),
            SimpleNamespace(stdout="[]", stderr="", returncode=0),
        ]
        SessionDemo.objects.create(
            thread_id="abc",
            host="127.0.0.1",
            port=45678,
            container_id="container-1",
            container_name="hitch-demo-abc-abcd",
            runtime="podman",
            status=SessionDemo.STATUS_ACTIVE,
            registration_token="token",
        )

        response = self.client.post(
            reverse("set_session_archived", kwargs={"session_id": "abc"}),
            data={"archived": "true"},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(SessionDemo.objects.get(thread_id="abc").status, SessionDemo.STATUS_STOPPED)
        self.assertEqual(mock_run.call_args_list[1], call(
            ["podman", "rm", "-f", "container-1"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ))
        mock_codex.return_value.__enter__.return_value.thread_archive.assert_called_once_with("abc")

    @patch("hitch.main.demo.subprocess.run")
    @patch("hitch.main.views.Codex")
    def test_failed_archive_does_not_clean_up_active_demo(
        self, mock_codex: MagicMock, mock_run: MagicMock
    ) -> None:
        mock_codex.return_value.__enter__.return_value.thread_archive.side_effect = (
            RuntimeError("codex unavailable")
        )
        mock_run.side_effect = [
            SimpleNamespace(
                stdout=(
                    '[{"Config":{"Labels":{'
                    '"io.hitch.managed":"demo",'
                    '"io.hitch.session":"abc",'
                    '"io.hitch.demo_token":"token",'
                    '"io.hitch.container_name":"hitch-demo-abc-abcd"'
                    "}}}]"
                ),
                stderr="",
                returncode=0,
            ),
            SimpleNamespace(stdout="", stderr="", returncode=0),
            SimpleNamespace(stdout="[]", stderr="", returncode=0),
        ]
        SessionDemo.objects.create(
            thread_id="abc",
            host="127.0.0.1",
            port=45678,
            container_id="container-1",
            container_name="hitch-demo-abc-abcd",
            runtime="podman",
            status=SessionDemo.STATUS_ACTIVE,
            registration_token="token",
        )

        with self.assertRaises(RuntimeError):
            self.client.post(
                reverse("set_session_archived", kwargs={"session_id": "abc"}),
                data={"archived": "true"},
            )

        mock_run.assert_not_called()
        self.assertEqual(
            SessionDemo.objects.get(thread_id="abc").status,
            SessionDemo.STATUS_ACTIVE,
        )

    @patch("hitch.main.views.codex_pool.cleanup_input_images_for_thread")
    @patch("hitch.main.views.Codex")
    def test_archive_keeps_retained_input_images_for_unarchive(
        self, mock_codex: MagicMock, mock_cleanup_images: MagicMock
    ) -> None:
        response = self.client.post(
            reverse("set_session_archived", kwargs={"session_id": "abc"}),
            data={"archived": "true"},
        )

        self.assertEqual(response.status_code, 302)
        mock_cleanup_images.assert_not_called()
        mock_codex.return_value.__enter__.return_value.thread_archive.assert_called_once_with(
            "abc"
        )

    @patch("hitch.main.views.Codex")
    def test_rejects_invalid_archive_requests(self, mock_codex: MagicMock) -> None:
        cases: list[tuple[str, dict[str, str], int]] = [
            ("post", {}, 400),
            ("post", {"archived": ""}, 400),
            ("post", {"archived": "yes"}, 400),
            ("get", {}, 405),
        ]
        for method, data, status in cases:
            with self.subTest(method=method, data=data):
                url = reverse("set_session_archived", kwargs={"session_id": "abc"})
                if method == "post":
                    response = self.client.post(url, data=data)
                else:
                    response = self.client.get(url)
                self.assertEqual(response.status_code, status)
        mock_codex.assert_not_called()


class StopSessionViewTests(TestCase):
    @patch("hitch.main.views.codex_pool.interrupt_instance")
    @patch("hitch.main.views.codex_pool.interrupt_active")
    def test_targets_instance_from_form_value(
        self,
        mock_interrupt_active: MagicMock,
        mock_interrupt_instance: MagicMock,
    ) -> None:
        # The Stop button posts the active worker's pk so a stale tab
        # cannot accidentally abort a newer overlapping worker. The
        # view forwards the id (and the URL's session id, as a
        # cross-thread guard) to ``interrupt_instance``.
        response = self.client.post(
            reverse("stop_session", kwargs={"session_id": "abc"}),
            data={"instance": "42"},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            response.headers["Location"],
            reverse("session", kwargs={"session_id": "abc"}),
        )
        mock_interrupt_instance.assert_called_once_with(42, expected_thread_id="abc")
        mock_interrupt_active.assert_not_called()

    @patch("hitch.main.views.codex_pool.interrupt_instance")
    @patch("hitch.main.views.codex_pool.interrupt_active")
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

    @patch("hitch.main.views.codex_pool.interrupt_instance")
    @patch("hitch.main.views.codex_pool.interrupt_active")
    def test_stop_with_over_limit_images_still_interrupts_worker(
        self,
        mock_interrupt_active: MagicMock,
        mock_interrupt_instance: MagicMock,
    ) -> None:
        response = self.client.post(
            reverse("stop_session", kwargs={"session_id": "abc"}),
            data={
                "instance": "42",
                "input_images": [
                    SimpleUploadedFile(
                        f"screen-{index}.png", _PNG_BYTES, content_type="image/png"
                    )
                    for index in range(5)
                ],
            },
        )

        self.assertEqual(response.status_code, 302)
        mock_interrupt_instance.assert_called_once_with(42, expected_thread_id="abc")
        mock_interrupt_active.assert_not_called()

    @patch(
        "hitch.main.views.system_agents.stop_active_workflow",
        wraps=system_agents.stop_active_workflow,
    )
    @patch("hitch.main.views.codex_pool.interrupt_instance")
    @patch("hitch.main.views.codex_pool.interrupt_active")
    def test_falls_back_to_latest_active_without_instance(
        self,
        mock_interrupt_active: MagicMock,
        mock_interrupt_instance: MagicMock,
        mock_stop_workflow: MagicMock,
    ) -> None:
        # Older cached page (or a direct curl POST) won't carry the
        # instance field; fall back to "latest active worker for this
        # thread" so the stop click still has a chance to do something.
        # ``None`` models a double-click after the worker already finished;
        # the view should still redirect instead of surfacing an error.
        mock_interrupt_active.return_value = None
        response = self.client.post(
            reverse("stop_session", kwargs={"session_id": "abc"})
        )

        self.assertEqual(response.status_code, 302)
        mock_stop_workflow.assert_called_once_with("abc")
        mock_interrupt_active.assert_called_once_with("abc")
        mock_interrupt_instance.assert_not_called()

    @patch("hitch.main.views.codex_pool.interrupt_active")
    @patch("hitch.main.views.system_agents.stop_active_workflow", return_value=True)
    def test_stops_active_system_workflow_without_instance(
        self, mock_stop_workflow: MagicMock, mock_interrupt_active: MagicMock
    ) -> None:
        response = self.client.post(
            reverse("stop_session", kwargs={"session_id": "abc"})
        )

        self.assertEqual(response.status_code, 302)
        mock_stop_workflow.assert_called_once_with("abc")
        mock_interrupt_active.assert_not_called()

    @patch("hitch.main.views.codex_pool.interrupt_instance")
    @patch("hitch.main.views.codex_pool.interrupt_active")
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

    @patch("hitch.main.streaming._IDLE_MAX_STREAM_SECONDS", 0.001)
    @patch("hitch.main.streaming._IDLE_POLL_INTERVAL", 0.001)
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

    @patch("hitch.main.streaming._IDLE_MAX_STREAM_SECONDS", 0.001)
    @patch("hitch.main.streaming._IDLE_POLL_INTERVAL", 0.001)
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

    @patch("hitch.main.system_agents.codex_pool.spawn_turn")
    @patch("hitch.main.codex_pool.worker_is_alive", return_value=False)
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

    @patch("hitch.main.streaming._IDLE_MAX_STREAM_SECONDS", 0.001)
    @patch("hitch.main.streaming._IDLE_POLL_INTERVAL", 0.001)
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

    @patch("hitch.main.streaming._POLL_INTERVAL", 0.01)
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

    @patch("hitch.main.streaming._POLL_INTERVAL", 0.001)
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

    def test_normalizes_legacy_decision_values(self) -> None:
        """Tabs loaded before a deploy may still POST the old UI values.
        Normalize them at the boundary so a click doesn't poison the row
        with a value app-server treats as a declined request."""
        aliases = {
            "approved": "accept",
            "denied": "decline",
            "abort": "cancel",
        }
        for posted, stored in aliases.items():
            with self.subTest(posted=posted):
                approval = self._make_approval()
                response = self.client.post(
                    reverse("resolve_approval", kwargs={"approval_id": approval.pk}),
                    data={"decision": posted},
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.content, stored.encode())
                approval.refresh_from_db()
                self.assertEqual(approval.decision, stored)

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


class SessionViewApprovalContextTests(TestCase):
    """The session detail view exposes POST URL templates for live
    browser prompts. Pin them so a URL refactor can't quietly break the
    streaming approval or structured-input loops."""

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.worker_is_alive", return_value=True)
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


class AutonomousGoalViewTests(TestCase):
    @patch("hitch.main.views.system_agents.maybe_start_auto_proposal_workflows")
    @patch("hitch.main.views.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.Codex")
    def test_get_pages_do_not_start_auto_proposals(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_scheduler: MagicMock,
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            auto_proposal_enabled=True,
        )
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)

        for route in ("index", "inbox", "autonomous_goals"):
            with self.subTest(route=route):
                response = self.client.get(reverse(route))
                self.assertEqual(response.status_code, 200)

        mock_scheduler.assert_not_called()

    @patch("hitch.main.views.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.Codex")
    def test_page_lists_goals_and_inbox_count_for_selected_project(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        other_project = Project.objects.create(name="Other", repo_path="/other")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            ambition=AutonomousGoal.AMBITION_HIGH,
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            auto_qa_enabled=True,
            web_search_mode=AutonomousGoal.WEB_SEARCH_LIVE,
            auto_merge_to_local_branch=True,
            auto_merge_branch="main",
            proposal_budget=25000,
        )
        AutonomousGoal.objects.create(
            project=other_project,
            title="Other goal",
            goal="Should not render.",
        )
        AutonomousGoal.objects.create(
            project=project,
            title="Deleted goal",
            goal="Should not render.",
            deleted_at=timezone.now(),
        )
        ProposedSession.objects.create(
            autonomous_goal=goal,
            title="Add parser coverage",
        )
        ProposedSession.objects.create(
            project=other_project,
            title="Other proposal",
            summary="Should not count for selected project.",
        )

        response = self.client.get(reverse("autonomous_goals"))

        self.assertEqual(response.status_code, 200)
        body = response.content.decode()
        nav_start = body.index('<nav class="primary-nav"')
        nav_end = body.index("</nav>", nav_start)
        nav_html = body[nav_start:nav_end]
        self.assertIn(
            f'href="{reverse("autonomous_goals")}" aria-current="page"', nav_html
        )
        self.assertIn(f'href="{reverse("inbox")}"', nav_html)
        self.assertIn(
            'class="primary-nav-badge" aria-label="1 inbox message">1</span>',
            nav_html,
        )
        self.assertIn(">auto goals</a>", nav_html)
        self.assertContains(response, "--accent-soft")
        self.assertContains(response, "--shadow-lg")
        self.assertContains(response, "[hidden] { display: none !important; }")
        self.assertContains(response, "Improve tests")
        self.assertContains(response, "Ambition")
        self.assertContains(response, "Ambition: High")
        self.assertContains(response, "Autonomy")
        self.assertContains(response, "Autonomy: Draft patch")
        self.assertContains(response, "Auto-QA: On")
        self.assertContains(
            response, 'value="draft_patch" data-auto-qa-supported="true"'
        )
        self.assertContains(
            response, 'value="draft_pr" data-auto-qa-supported="false"'
        )
        self.assertContains(
            response,
            'value="draft_pr" data-auto-qa-supported="false" data-auto-qa-required="true"',
        )
        self.assertContains(response, "Web search: Live")
        self.assertContains(response, "Proposal budget: 25000 tokens")
        self.assertContains(response, "Auto-proposal: Off")
        self.assertContains(response, "Auto merge: main")
        self.assertContains(response, 'class="goal-menu" data-goal-menu')
        self.assertContains(response, 'role="menuitem">Run</button>')
        self.assertContains(
            response,
            f'action="{reverse("run_autonomous_goal", args=[goal.pk])}"',
        )
        self.assertContains(response, 'role="menuitem">Delete</button>')
        self.assertContains(
            response,
            f'action="{reverse("delete_autonomous_goal", args=[goal.pk])}"',
        )
        self.assertContains(
            response,
            f'data-edit-url="{reverse("edit_autonomous_goal", args=[goal.pk])}"',
        )
        self.assertContains(
            response, f'data-autonomy="{AutonomousGoal.AUTONOMY_DRAFT_PATCH}"'
        )
        self.assertContains(response, 'data-auto-qa="true"')
        self.assertContains(
            response, f'data-web-search-mode="{AutonomousGoal.WEB_SEARCH_LIVE}"'
        )
        self.assertContains(response, 'data-auto-proposal-enabled="false"')
        self.assertContains(response, 'data-proposal-budget="25000"')
        self.assertContains(response, 'data-auto-merge-to-local-branch="true"')
        self.assertContains(response, 'data-auto-merge-branch="main"')
        self.assertContains(response, 'data-autonomous-goal-edit')
        self.assertNotContains(response, "Add parser coverage")
        self.assertNotContains(response, 'name="proposed_session"')
        self.assertNotContains(response, "Other goal")
        self.assertNotContains(response, "Deleted goal")

    @patch("hitch.main.views.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.Codex")
    def test_page_shows_tappable_run_status_indicators(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        blocked_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve raptorq",
            goal="Investigate raptorq failures.",
        )
        running_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        blocked_no_log_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve docs",
            goal="Investigate documentation failures.",
        )
        blocked_workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=system_agents._autonomous_goal_main_thread_id(blocked_goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_BLOCKED,
            step=system_agents.STEP_BLOCKED,
            state={
                "autonomous_goal_id": blocked_goal.pk,
                "error": "raptorq decoder exhausted repair symbols",
            },
        )
        blocked_instance = CodexInstance.objects.create(
            pid=0,
            thread_id="blocked-agent-thread",
            cwd="/repo",
            prompt="run autonomous goal",
            events_path="/dev/null",
            status=CodexInstance.STATUS_FAILED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=blocked_workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        SystemAgentRun.objects.create(
            workflow=blocked_workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id=blocked_instance.thread_id,
            instance=blocked_instance,
            status=SystemAgentRun.STATUS_FAILED,
        )
        SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=system_agents._autonomous_goal_main_thread_id(running_goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
            state={"autonomous_goal_id": running_goal.pk},
        )
        SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=system_agents._autonomous_goal_main_thread_id(
                blocked_no_log_goal.pk
            ),
            cwd="/repo",
            status=SystemWorkflow.STATUS_BLOCKED,
            step=system_agents.STEP_BLOCKED,
            state={
                "autonomous_goal_id": blocked_no_log_goal.pk,
                "error": "blocked before the run log was created",
            },
        )

        response = self.client.get(reverse("autonomous_goals"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'data-run-status-dialog')
        self.assertContains(response, 'data-state="blocked"')
        self.assertContains(
            response, 'data-run-status-title="Autonomous goal is blocked"'
        )
        self.assertContains(response, "raptorq decoder exhausted repair symbols")
        self.assertContains(
            response,
            f'data-run-status-log-url="{reverse("autonomous_goal_run_log", args=[blocked_workflow.pk])}"',
        )
        self.assertContains(response, 'data-state="running"')
        self.assertContains(
            response, 'data-run-status-title="Autonomous goal is running"'
        )
        self.assertContains(response, "This autonomous goal run is still working.")
        self.assertContains(response, "blocked before the run log was created")
        self.assertContains(response, 'data-run-status-log-url=""', count=2)
        self.assertNotContains(response, 'data-run-status-log-url="None"')

    @patch("hitch.main.views.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.Codex")
    def test_edit_form_sync_preserves_auto_qa_choice_when_required(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            auto_qa_enabled=True,
        )

        response = self.client.get(reverse("autonomous_goals"))
        self.assertEqual(response.status_code, 200)
        body = response.content.decode()
        self.assertIn(
            'autoQa.dataset.autoQaUserChecked = autoQa.checked ? "true" : "false";',
            body,
        )
        self.assertIn(
            'autoQa.checked = autoQa.dataset.autoQaUserChecked === "true";',
            body,
        )
        self.assertIn("delete editGoalAutoQa.dataset.autoQaUserChecked;", body)
        self.assertIn("autoQa.disabled = required || !supported;", body)

    @patch("hitch.main.views.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.Codex")
    def test_draft_pr_goal_shows_auto_qa_required_on_reopen(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PR,
            auto_qa_enabled=False,
        )

        response = self.client.get(reverse("autonomous_goals"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Auto-QA: Required")
        self.assertContains(response, 'data-autonomy="draft_pr"')
        self.assertContains(response, 'data-auto-qa="false"')
        self.assertContains(
            response,
            'value="draft_pr" data-auto-qa-supported="false" data-auto-qa-required="true"',
        )

    @patch("hitch.main.views.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.Codex")
    def test_inbox_page_lists_proposals_for_selected_project(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        other_project = Project.objects.create(name="Other", repo_path="/other")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            ambition=AutonomousGoal.AMBITION_HIGH,
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo",
            project=project,
        )
        judge = SessionMetadata.objects.create(
            thread_id="judge-thread",
            cwd="/repo",
            project=project,
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            title="Add parser coverage",
            summary="This adds focused parser coverage.",
            prompt=(
                "Go ahead and implement this proposed session.\n\n"
                "Autonomous goal objective:\n"
                "Find useful test coverage increments.\n\n"
                "Implementation guidance:\n"
                "Add focused rollout parser tests before changing behavior."
            ),
            confidence=AutonomousGoal.CONFIDENCE_HIGH,
            relevant_files=["hitch/main/rollout.py"],
            candidate_session=candidate,
            judge_session=judge,
            outcome_metadata={
                "auto_pr_enabled": True,
                "auto_qa_enabled": False,
            },
        )
        ProposedSession.objects.create(
            project=other_project,
            title="Other proposal",
            summary="Should not render.",
        )

        response = self.client.get(reverse("inbox"))

        self.assertEqual(response.status_code, 200)
        body = response.content.decode()
        nav_start = body.index('<nav class="primary-nav"')
        nav_end = body.index("</nav>", nav_start)
        nav_html = body[nav_start:nav_end]
        self.assertIn(f'href="{reverse("inbox")}" aria-current="page"', nav_html)
        self.assertIn(
            'class="primary-nav-badge" aria-label="1 inbox message">1</span>',
            nav_html,
        )
        self.assertContains(response, 'data-visible-projects-open')
        self.assertContains(response, "Visible projects")
        main_start = body.index("<main>")
        self.assertLess(body.index('aria-label="Inbox actions"'), main_start)
        self.assertLess(body.index("data-visible-projects-open"), main_start)
        self.assertContains(
            response,
            '<dialog class="new-session" data-visible-projects-dialog',
            html=False,
        )
        self.assertContains(response, "Add parser coverage")
        self.assertContains(response, "This adds focused parser coverage.")
        self.assertContains(response, "hitch/main/rollout.py")
        self.assertContains(response, 'data-proposed-session-do')
        self.assertContains(response, f'data-proposed-session-id="{proposal.pk}"')
        self.assertContains(response, f'data-proposed-session-project="{project.pk}"')
        start_modal_title = (
            '<h2 id="do-session-title" tabindex="-1" autofocus>'
            "Continue proposed session</h2>"
        )
        self.assertContains(response, start_modal_title)
        self.assertContains(response, "if (doHeading) doHeading.focus();")
        self.assertNotContains(response, "doPrompt.focus()")
        self.assertContains(
            response,
            'if (doForm) doForm.addEventListener("submit", () => hideDialog(doDialog));',
        )
        self.assertContains(response, 'data-proposed-session-auto-pr="true"')
        self.assertContains(response, 'data-proposed-session-auto-qa="false"')
        self.assertContains(
            response,
            'data-proposed-session-prompt="Go ahead and implement this proposed session.',
        )
        self.assertContains(
            response, f'aria-label="Actions for {proposal.title}"'
        )
        self.assertContains(
            response,
            f'action="{reverse("update_proposed_session_outcome", args=[proposal.pk])}"',
        )
        proposal_header_start = body.index('<div class="proposal-header">')
        proposal_actions_start = body.index(
            '<div class="proposal-actions">', proposal_header_start
        )
        proposal_menu_start = body.index(
            '<div class="proposal-menu"', proposal_header_start
        )
        self.assertLess(proposal_menu_start, proposal_actions_start)
        self.assertContains(
            response, f'value="{ProposedSession.OUTCOME_DISMISSED}"'
        )
        self.assertContains(response, "Judge log")
        self.assertContains(response, 'name="proposed_session"')
        self.assertNotContains(response, "Other proposal")

    @patch("hitch.main.views.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.Codex")
    def test_inbox_recovers_stale_proposal_start_claim(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        stale_claimed_at = (
            datetime.now(UTC)
            - ProposedSession.ACCEPTED_SESSION_START_CLAIM_TTL
            - timedelta(seconds=1)
        )
        proposal = ProposedSession.objects.create(
            project=project,
            title="Add parser coverage",
            summary="This adds focused parser coverage.",
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            outcome_metadata={
                "accepted_by": "user",
                "accepted_thread_id": "",
                ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY: (
                    stale_claimed_at.isoformat()
                ),
            },
        )

        response = self.client.get(reverse("inbox"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Add parser coverage")
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_UNSET)
        self.assertIsNone(proposal.accepted_session)
        self.assertNotIn(
            ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY,
            proposal.outcome_metadata,
        )

    @patch("hitch.main.views.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.Codex")
    def test_inbox_keeps_active_proposal_start_claim_hidden(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        proposal = ProposedSession.objects.create(
            project=project,
            title="Add parser coverage",
            summary="This adds focused parser coverage.",
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            outcome_metadata={
                "accepted_by": "user",
                "accepted_thread_id": "",
                ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY: (
                    datetime.now(UTC).isoformat()
                ),
            },
        )

        response = self.client.get(reverse("inbox"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "Add parser coverage")
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_ACCEPTED)
        self.assertIsNone(proposal.accepted_session)
        self.assertIn(
            ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY,
            proposal.outcome_metadata,
        )

    @patch("hitch.main.views.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.Codex")
    def test_inbox_visible_projects_filter_messages(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        other_project = Project.objects.create(name="Other", repo_path="/other")
        _setup_codex(mock_codex)
        ProposedSession.objects.create(
            project=project,
            title="Matching proposal",
            summary="Should not render.",
        )
        ProposedSession.objects.create(
            project=other_project,
            title="Other proposal",
            summary="Should render.",
        )
        ProposedSession.objects.create(
            title="No repo notice",
            inbox_kind=ProposedSession.INBOX_KIND_NOTICE,
            summary="No project attached.",
        )

        response = self.client.post(
            reverse("update_visible_session_projects"),
            data={
                "visible_project": [str(other_project.pk)],
                "show_no_project_sessions": "true",
                "next": reverse("inbox"),
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], reverse("inbox"))
        self.assertEqual(
            _cookie_value(response, _VISIBLE_SESSION_PROJECTS_COOKIE),
            f"[{other_project.pk}]",
        )
        self.assertEqual(
            _cookie_value(response, _SHOW_NO_PROJECT_SESSIONS_COOKIE),
            "true",
        )

        response = self.client.get(reverse("inbox"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Visible projects")
        self.assertContains(response, "Other proposal")
        self.assertContains(response, "No repo notice")
        self.assertContains(response, "No repo -")
        self.assertNotContains(response, "Matching proposal")

    @patch("hitch.main.views.cleanup_managed_worktree_path")
    def test_reject_proposed_session_uses_visible_project_filter(
        self, mock_cleanup: MagicMock
    ) -> None:
        selected_project = Project.objects.create(name="Hitch", repo_path="/repo")
        visible_project = Project.objects.create(name="Other", repo_path="/other")
        _seed_cookies(
            self.client,
            **{
                _SELECTED_PROJECT_COOKIE: str(selected_project.pk),
                _VISIBLE_SESSION_PROJECTS_COOKIE: f"[{visible_project.pk}]",
            },
        )
        proposal = ProposedSession.objects.create(
            project=visible_project,
            title="Add docs coverage",
        )

        response = self.client.post(
            reverse("update_proposed_session_outcome", args=[proposal.pk]),
            {
                "outcome_status": ProposedSession.OUTCOME_REJECTED,
                "reason": "Not useful enough.",
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], reverse("inbox"))
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_REJECTED)
        self.assertEqual(proposal.outcome_notes, "Not useful enough.")
        mock_cleanup.assert_not_called()

    @patch("hitch.main.views.cleanup_managed_worktree_path")
    def test_update_outcome_rejects_proposal_hidden_by_visible_project_filter(
        self, mock_cleanup: MagicMock
    ) -> None:
        visible_project = Project.objects.create(name="Hitch", repo_path="/repo")
        hidden_project = Project.objects.create(name="Other", repo_path="/other")
        _seed_cookies(
            self.client,
            **{
                _SELECTED_PROJECT_COOKIE: str(visible_project.pk),
                _VISIBLE_SESSION_PROJECTS_COOKIE: f"[{visible_project.pk}]",
            },
        )
        proposal = ProposedSession.objects.create(
            project=hidden_project,
            title="Add docs coverage",
        )

        response = self.client.post(
            reverse("update_proposed_session_outcome", args=[proposal.pk]),
            {
                "outcome_status": ProposedSession.OUTCOME_REJECTED,
                "reason": "Not useful enough.",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.content, b"proposed session is required")
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_UNSET)
        mock_cleanup.assert_not_called()

    @patch("hitch.main.views.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.Codex")
    def test_new_session_page_prefills_proposed_session(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _setup_codex(mock_codex)
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            title="Add parser coverage",
            summary="This adds focused parser coverage.",
            prompt="Add focused rollout parser tests before changing behavior.",
        )

        response = self.client.get(
            f"{reverse('new_session')}?proposed_session={proposal.pk}"
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'name="proposed_session"')
        self.assertContains(response, f'value="{proposal.pk}"')
        self.assertContains(response, "Add focused rollout parser tests")
        self.assertContains(response, f'value="{project.pk}" selected')
        self.assertContains(response, f'href="{reverse("inbox")}"')

    @patch("hitch.main.views.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.Codex")
    def test_new_session_page_recovers_stale_proposal_start_claim(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _setup_codex(mock_codex)
        stale_claimed_at = (
            datetime.now(UTC)
            - ProposedSession.ACCEPTED_SESSION_START_CLAIM_TTL
            - timedelta(seconds=1)
        )
        proposal = ProposedSession.objects.create(
            project=project,
            title="Add parser coverage",
            prompt="Add focused rollout parser tests before changing behavior.",
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            outcome_metadata={
                "accepted_by": "user",
                "accepted_thread_id": "",
                ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY: (
                    stale_claimed_at.isoformat()
                ),
            },
        )

        response = self.client.get(
            f"{reverse('new_session')}?proposed_session={proposal.pk}"
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, f'value="{proposal.pk}"')
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_UNSET)
        self.assertIsNone(proposal.accepted_session)
        self.assertNotIn(
            ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY,
            proposal.outcome_metadata,
        )

    @patch("hitch.main.views.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.Codex")
    def test_new_session_page_prefills_prompt_and_project_from_query(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _setup_codex(mock_codex)
        prompt = (
            "Debug and fix the user's issue from session UID thread-1.\n\n"
            "User issue: "
        )

        response = self.client.get(
            reverse("new_session"), {"prompt": prompt, "project": str(project.pk)}
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, html.escape(prompt))
        self.assertContains(response, f'value="{project.pk}" selected')

    @patch("hitch.main.views.discover_repos", return_value=[Path("/other")])
    @patch("hitch.main.views.Codex")
    def test_new_session_page_rejects_unavailable_project_from_query(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _setup_codex(mock_codex)

        response = self.client.get(
            reverse("new_session"), {"prompt": "debug this", "project": str(project.pk)}
        )

        self.assertEqual(response.status_code, 404)

    @patch("hitch.main.views.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.Codex")
    def test_new_session_page_prefills_bare_repo_cwd_from_query(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _setup_codex(mock_codex)

        response = self.client.get(
            reverse("new_session"), {"prompt": "debug this", "cwd": "/repo"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "debug this")
        self.assertContains(response, 'value="__bare_repo__" selected')
        self.assertContains(response, '<option value="/repo" selected>')
        self.assertNotContains(response, f'value="{project.pk}" selected')

    @patch("hitch.main.views.discover_repos", return_value=[Path("/other")])
    @patch("hitch.main.views.Codex")
    def test_new_session_page_rejects_unavailable_bare_repo_cwd(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        _setup_codex(mock_codex)

        response = self.client.get(reverse("new_session"), {"cwd": "/repo"})

        self.assertEqual(response.status_code, 404)

    @patch("hitch.main.views.discover_repos", return_value=[Path("/other")])
    @patch("hitch.main.views.Codex")
    def test_new_session_page_rejects_proposed_session_for_unavailable_repo(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        proposal = ProposedSession.objects.create(
            project=project,
            title="Add parser coverage",
            prompt="Add focused rollout parser tests before changing behavior.",
        )

        response = self.client.get(
            f"{reverse('new_session')}?proposed_session={proposal.pk}"
        )

        self.assertEqual(response.status_code, 404)
        mock_codex.assert_not_called()

    @patch("hitch.main.views.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.Codex")
    def test_page_lists_no_proposal_notice_with_dismiss(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        ProposedSession.objects.create(
            autonomous_goal=goal,
            title="No proposal from Improve tests",
            inbox_kind=ProposedSession.INBOX_KIND_NOTICE,
            summary="No concrete test increment was worth proposing.",
        )

        response = self.client.get(reverse("inbox"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No proposal from Improve tests")
        self.assertContains(response, "From autonomous goal: Improve tests")
        self.assertContains(
            response, "No concrete test increment was worth proposing."
        )
        self.assertContains(response, "Dismiss")
        self.assertNotContains(response, 'data-proposed-session-id="')
        self.assertNotContains(response, 'data-reject-url="')

    @patch("hitch.main.views.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.Codex")
    def test_page_lists_agent_created_proposal(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        proposal = ProposedSession.objects.create(
            project=project,
            title="Add CLI proposal tests",
            summary="Cover the proposed session CLI.",
            prompt="Implement tests for the proposed session CLI.",
            relevant_files=["hitch/main/management/commands/propose_session.py"],
        )

        response = self.client.get(reverse("inbox"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Add CLI proposal tests")
        self.assertContains(response, "From coding agent")
        self.assertContains(response, 'data-proposed-session-do')
        self.assertContains(response, f'data-proposed-session-id="{proposal.pk}"')
        self.assertContains(response, "Implement tests for the proposed session CLI.")
        self.assertContains(response, f'data-proposed-session-project="{project.pk}"')

    @patch("hitch.main.views.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.Codex")
    def test_page_shows_create_form_inline_when_no_goals(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)

        response = self.client.get(reverse("autonomous_goals"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'data-autonomous-goal-create-form')
        self.assertContains(response, "data-autonomous-goal-auto-qa")
        self.assertContains(
            response,
            '<input type="checkbox" name="auto_qa" value="true" data-autonomous-goal-auto-qa disabled>',
            html=True,
        )
        self.assertContains(
            response, 'value="draft_patch" data-auto-qa-supported="true"'
        )
        self.assertContains(
            response, 'value="draft_pr" data-auto-qa-supported="false"'
        )
        self.assertContains(response, "Create autonomous goal")
        self.assertNotContains(response, "No autonomous goals yet.")
        self.assertNotContains(
            response,
            '<button type="button" role="menuitem" data-create-autonomous-goal-open>',
        )
        self.assertNotContains(
            response,
            '<dialog class="new-session" data-create-autonomous-goal-dialog',
        )

    @patch("hitch.main.views.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.Codex")
    def test_page_moves_create_form_to_header_dialog_when_goals_exist(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )

        response = self.client.get(reverse("autonomous_goals"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'class="page-menu" data-page-menu')
        self.assertContains(
            response,
            '<button type="button" role="menuitem" data-create-autonomous-goal-open>',
        )
        self.assertContains(response, 'role="menuitem">Run all</button>')
        self.assertContains(
            response,
            '<dialog class="new-session" data-create-autonomous-goal-dialog',
        )
        self.assertNotContains(response, '<p class="section-label">Create</p>')

    def test_create_autonomous_goal_for_selected_project(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))

        response = self.client.post(
            reverse("create_autonomous_goal"),
            {
                "title": "Improve tests",
                "goal": "Find useful test coverage increments.",
                "ambition": AutonomousGoal.AMBITION_YOLO,
                "autonomy": AutonomousGoal.AUTONOMY_DRAFT_PR,
                "auto_qa": "true",
                "auto_proposal": "true",
                "stacked_diff_depth": "3",
                "proposal_budget": "25000",
                "confidence_threshold": AutonomousGoal.CONFIDENCE_VERY_HIGH,
                "web_search_mode": AutonomousGoal.WEB_SEARCH_LIVE,
            },
        )

        self.assertEqual(response.status_code, 302)
        goal = AutonomousGoal.objects.get()
        self.assertEqual(goal.project, project)
        self.assertEqual(goal.title, "Improve tests")
        self.assertEqual(goal.ambition, AutonomousGoal.AMBITION_YOLO)
        self.assertEqual(goal.autonomy, AutonomousGoal.AUTONOMY_DRAFT_PR)
        self.assertFalse(goal.auto_qa_enabled)
        self.assertEqual(goal.stacked_diff_depth, 3)
        self.assertEqual(goal.proposal_budget, 25000)
        self.assertEqual(goal.web_search_mode, AutonomousGoal.WEB_SEARCH_LIVE)
        self.assertTrue(goal.auto_proposal_enabled)
        self.assertEqual(
            goal.confidence_threshold,
            AutonomousGoal.CONFIDENCE_VERY_HIGH,
        )

    def test_create_autonomous_goal_stores_auto_merge_branch(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))

        with patch("hitch.main.views.local_branch_names", return_value=["main"]):
            response = self.client.post(
                reverse("create_autonomous_goal"),
                {
                    "title": "Improve tests",
                    "goal": "Find useful test coverage increments.",
                    "ambition": AutonomousGoal.AMBITION_HIGH,
                    "autonomy": AutonomousGoal.AUTONOMY_DRAFT_PATCH,
                    "auto_qa": "true",
                    "confidence_threshold": AutonomousGoal.CONFIDENCE_VERY_HIGH,
                    "auto_merge_to_local_branch": "true",
                    "auto_merge_branch": "main",
                },
            )

        self.assertEqual(response.status_code, 302)
        goal = AutonomousGoal.objects.get()
        self.assertTrue(goal.auto_qa_enabled)
        self.assertTrue(goal.auto_merge_to_local_branch)
        self.assertEqual(goal.auto_merge_branch, "main")

    def test_edit_autonomous_goal_updates_selected_project_goal(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            ambition=AutonomousGoal.AMBITION_INCREMENTAL,
            autonomy=AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
            auto_proposal_enabled=True,
            confidence_threshold=AutonomousGoal.CONFIDENCE_HIGH,
            web_search_mode=AutonomousGoal.WEB_SEARCH_CACHED,
            proposal_budget=10000,
        )

        response = self.client.post(
            reverse("edit_autonomous_goal", args=[goal.pk]),
            {
                "title": "Improve docs",
                "goal": "Find useful docs increments.",
                "ambition": AutonomousGoal.AMBITION_HIGH,
                "autonomy": AutonomousGoal.AUTONOMY_DRAFT_PATCH,
                "auto_qa": "true",
                "auto_proposal": "false",
                "stacked_diff_depth": "4",
                "proposal_budget": "30000",
                "confidence_threshold": AutonomousGoal.CONFIDENCE_VERY_HIGH,
                "web_search_mode": AutonomousGoal.WEB_SEARCH_DISABLED,
            },
        )

        self.assertEqual(response.status_code, 302)
        goal.refresh_from_db()
        self.assertEqual(goal.title, "Improve docs")
        self.assertEqual(goal.goal, "Find useful docs increments.")
        self.assertEqual(goal.ambition, AutonomousGoal.AMBITION_HIGH)
        self.assertEqual(goal.autonomy, AutonomousGoal.AUTONOMY_DRAFT_PATCH)
        self.assertTrue(goal.auto_qa_enabled)
        self.assertEqual(goal.stacked_diff_depth, 4)
        self.assertEqual(goal.proposal_budget, 30000)
        self.assertEqual(goal.web_search_mode, AutonomousGoal.WEB_SEARCH_DISABLED)
        self.assertFalse(goal.auto_proposal_enabled)
        self.assertEqual(
            goal.confidence_threshold,
            AutonomousGoal.CONFIDENCE_VERY_HIGH,
        )

    def test_edit_autonomous_goal_clears_proposal_budget_when_blank(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            ambition=AutonomousGoal.AMBITION_INCREMENTAL,
            autonomy=AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
            proposal_budget=10000,
            confidence_threshold=AutonomousGoal.CONFIDENCE_HIGH,
        )

        response = self.client.post(
            reverse("edit_autonomous_goal", args=[goal.pk]),
            {
                "title": "Improve tests",
                "goal": "Find useful test coverage increments.",
                "ambition": AutonomousGoal.AMBITION_INCREMENTAL,
                "autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
                "proposal_budget": "",
                "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
            },
        )

        self.assertEqual(response.status_code, 302)
        goal.refresh_from_db()
        self.assertIsNone(goal.proposal_budget)

    def test_edit_autonomous_goal_can_reset_web_search_to_codex_default(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            ambition=AutonomousGoal.AMBITION_INCREMENTAL,
            autonomy=AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
            confidence_threshold=AutonomousGoal.CONFIDENCE_HIGH,
            web_search_mode=AutonomousGoal.WEB_SEARCH_LIVE,
        )

        response = self.client.post(
            reverse("edit_autonomous_goal", args=[goal.pk]),
            {
                "title": "Improve tests",
                "goal": "Find useful test coverage increments.",
                "ambition": AutonomousGoal.AMBITION_INCREMENTAL,
                "autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
                "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
                "web_search_mode": AutonomousGoal.WEB_SEARCH_DEFAULT,
            },
        )

        self.assertEqual(response.status_code, 302)
        goal.refresh_from_db()
        self.assertEqual(goal.web_search_mode, AutonomousGoal.WEB_SEARCH_DEFAULT)

    def test_edit_autonomous_goal_updates_auto_merge_branch(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
        )

        with patch("hitch.main.views.local_branch_names", return_value=["release"]):
            response = self.client.post(
                reverse("edit_autonomous_goal", args=[goal.pk]),
                {
                    "title": "Improve tests",
                    "goal": "Find useful test coverage increments.",
                    "ambition": AutonomousGoal.AMBITION_HIGH,
                    "autonomy": AutonomousGoal.AUTONOMY_DRAFT_PATCH,
                    "auto_qa": "true",
                    "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
                    "auto_merge_to_local_branch": "true",
                    "auto_merge_branch": "release",
                },
            )

        self.assertEqual(response.status_code, 302)
        goal.refresh_from_db()
        self.assertTrue(goal.auto_qa_enabled)
        self.assertTrue(goal.auto_merge_to_local_branch)
        self.assertEqual(goal.auto_merge_branch, "release")

    def test_edit_autonomous_goal_clears_auto_merge_when_unchecked(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            auto_qa_enabled=True,
            auto_merge_to_local_branch=True,
            auto_merge_branch="release",
        )

        response = self.client.post(
            reverse("edit_autonomous_goal", args=[goal.pk]),
            {
                "title": "Improve tests",
                "goal": "Find useful test coverage increments.",
                "ambition": AutonomousGoal.AMBITION_HIGH,
                "autonomy": AutonomousGoal.AUTONOMY_DRAFT_PATCH,
                "auto_qa": "true",
                "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
            },
        )

        self.assertEqual(response.status_code, 302)
        goal.refresh_from_db()
        self.assertTrue(goal.auto_qa_enabled)
        self.assertFalse(goal.auto_merge_to_local_branch)
        self.assertEqual(goal.auto_merge_branch, "")

    def test_edit_autonomous_goal_preserves_autonomy_when_omitted(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            ambition=AutonomousGoal.AMBITION_INCREMENTAL,
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PR,
            auto_proposal_enabled=True,
            confidence_threshold=AutonomousGoal.CONFIDENCE_HIGH,
            web_search_mode=AutonomousGoal.WEB_SEARCH_CACHED,
        )

        response = self.client.post(
            reverse("edit_autonomous_goal", args=[goal.pk]),
            {
                "title": "Improve docs",
                "goal": "Find useful docs increments.",
                "ambition": AutonomousGoal.AMBITION_HIGH,
                "confidence_threshold": AutonomousGoal.CONFIDENCE_VERY_HIGH,
            },
        )

        self.assertEqual(response.status_code, 302)
        goal.refresh_from_db()
        self.assertEqual(goal.autonomy, AutonomousGoal.AUTONOMY_DRAFT_PR)
        self.assertEqual(goal.web_search_mode, AutonomousGoal.WEB_SEARCH_CACHED)
        self.assertTrue(goal.auto_proposal_enabled)

    def test_edit_autonomous_goal_clears_auto_proposal_no_proposal_sha(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            ambition=AutonomousGoal.AMBITION_INCREMENTAL,
            autonomy=AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
            confidence_threshold=AutonomousGoal.CONFIDENCE_HIGH,
            auto_proposal_last_no_proposal_sha="a" * 40,
        )

        response = self.client.post(
            reverse("edit_autonomous_goal", args=[goal.pk]),
            {
                "title": "Improve tests",
                "goal": "Find useful test coverage increments.",
                "ambition": AutonomousGoal.AMBITION_INCREMENTAL,
                "autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
                "auto_proposal": "true",
                "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
            },
        )

        self.assertEqual(response.status_code, 302)
        goal.refresh_from_db()
        self.assertTrue(goal.auto_proposal_enabled)
        self.assertEqual(goal.auto_proposal_last_no_proposal_sha, "")

    def test_edit_autonomous_goal_preserves_auto_qa_when_omitted(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            ambition=AutonomousGoal.AMBITION_INCREMENTAL,
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            auto_qa_enabled=True,
            confidence_threshold=AutonomousGoal.CONFIDENCE_HIGH,
        )

        response = self.client.post(
            reverse("edit_autonomous_goal", args=[goal.pk]),
            {
                "title": "Improve docs",
                "goal": "Find useful docs increments.",
                "ambition": AutonomousGoal.AMBITION_HIGH,
                "confidence_threshold": AutonomousGoal.CONFIDENCE_VERY_HIGH,
            },
        )

        self.assertEqual(response.status_code, 302)
        goal.refresh_from_db()
        self.assertEqual(goal.autonomy, AutonomousGoal.AUTONOMY_DRAFT_PATCH)
        self.assertTrue(goal.auto_qa_enabled)

    def test_edit_autonomous_goal_disables_auto_qa_when_false_is_explicit(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            ambition=AutonomousGoal.AMBITION_INCREMENTAL,
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            auto_qa_enabled=True,
            confidence_threshold=AutonomousGoal.CONFIDENCE_HIGH,
        )

        response = self.client.post(
            reverse("edit_autonomous_goal", args=[goal.pk]),
            {
                "title": "Improve docs",
                "goal": "Find useful docs increments.",
                "ambition": AutonomousGoal.AMBITION_HIGH,
                "autonomy": AutonomousGoal.AUTONOMY_DRAFT_PATCH,
                "auto_qa": "false",
                "confidence_threshold": AutonomousGoal.CONFIDENCE_VERY_HIGH,
            },
        )

        self.assertEqual(response.status_code, 302)
        goal.refresh_from_db()
        self.assertFalse(goal.auto_qa_enabled)

    def test_edit_autonomous_goal_is_scoped_to_selected_project(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        other_project = Project.objects.create(name="Other", repo_path="/other")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=other_project,
            title="Other goal",
            goal="Should not change.",
        )

        response = self.client.post(
            reverse("edit_autonomous_goal", args=[goal.pk]),
            {
                "title": "Changed",
                "goal": "Changed.",
                "ambition": AutonomousGoal.AMBITION_HIGH,
                "confidence_threshold": AutonomousGoal.CONFIDENCE_VERY_HIGH,
            },
        )

        self.assertEqual(response.status_code, 404)
        goal.refresh_from_db()
        self.assertEqual(goal.title, "Other goal")
        self.assertEqual(goal.goal, "Should not change.")

    def test_edit_autonomous_goal_rejects_invalid_posts(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )

        for data, message in (
            (
                {
                    "title": "",
                    "goal": "Find useful docs increments.",
                    "ambition": AutonomousGoal.AMBITION_HIGH,
                    "autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
                    "auto_proposal": "false",
                    "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
                },
                "title is required",
            ),
            (
                {
                    "title": "Improve docs",
                    "goal": "",
                    "ambition": AutonomousGoal.AMBITION_HIGH,
                    "autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
                    "auto_proposal": "false",
                    "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
                },
                "goal is required",
            ),
            (
                {
                    "title": "Improve docs",
                    "goal": "Find useful docs increments.",
                    "ambition": "huge",
                    "autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
                    "auto_proposal": "false",
                    "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
                },
                "ambition is invalid",
            ),
            (
                {
                    "title": "Improve docs",
                    "goal": "Find useful docs increments.",
                    "ambition": AutonomousGoal.AMBITION_HIGH,
                    "autonomy": "self_driving",
                    "auto_proposal": "false",
                    "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
                },
                "autonomy is invalid",
            ),
            (
                {
                    "title": "Improve docs",
                    "goal": "Find useful docs increments.",
                    "ambition": AutonomousGoal.AMBITION_HIGH,
                    "autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
                    "auto_qa": "yes",
                    "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
                },
                "auto-QA setting is invalid",
            ),
            (
                {
                    "title": "Improve docs",
                    "goal": "Find useful docs increments.",
                    "ambition": AutonomousGoal.AMBITION_HIGH,
                    "autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
                    "auto_proposal": "maybe",
                    "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
                },
                "auto-proposal is invalid",
            ),
            (
                {
                    "title": "Improve docs",
                    "goal": "Find useful docs increments.",
                    "ambition": AutonomousGoal.AMBITION_HIGH,
                    "autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
                    "stacked_diff_depth": "0",
                    "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
                },
                "stacked diff depth is invalid",
            ),
            (
                {
                    "title": "Improve docs",
                    "goal": "Find useful docs increments.",
                    "ambition": AutonomousGoal.AMBITION_HIGH,
                    "autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
                    "stacked_diff_depth": "several",
                    "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
                },
                "stacked diff depth is invalid",
            ),
            (
                {
                    "title": "Improve docs",
                    "goal": "Find useful docs increments.",
                    "ambition": AutonomousGoal.AMBITION_HIGH,
                    "autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
                    "proposal_budget": "0",
                    "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
                },
                "proposal budget is invalid",
            ),
            (
                {
                    "title": "Improve docs",
                    "goal": "Find useful docs increments.",
                    "ambition": AutonomousGoal.AMBITION_HIGH,
                    "autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
                    "proposal_budget": "many",
                    "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
                },
                "proposal budget is invalid",
            ),
            (
                {
                    "title": "Improve docs",
                    "goal": "Find useful docs increments.",
                    "ambition": AutonomousGoal.AMBITION_HIGH,
                    "autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
                    "stacked_diff_depth": "2",
                    "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
                },
                "stacked diff depth requires draft patch or draft PR",
            ),
            (
                {
                    "title": "Improve docs",
                    "goal": "Find useful docs increments.",
                    "ambition": AutonomousGoal.AMBITION_HIGH,
                    "autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
                    "auto_proposal": "false",
                    "confidence_threshold": "absolute",
                },
                "confidence threshold is invalid",
            ),
            (
                {
                    "title": "Improve docs",
                    "goal": "Find useful docs increments.",
                    "ambition": AutonomousGoal.AMBITION_HIGH,
                    "autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
                    "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
                    "web_search_mode": "maybe",
                },
                "web search setting is invalid",
            ),
            (
                {
                    "title": "Improve docs",
                    "goal": "Find useful docs increments.",
                    "ambition": AutonomousGoal.AMBITION_HIGH,
                    "autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
                    "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
                    "auto_merge_to_local_branch": "true",
                    "auto_merge_branch": "main",
                },
                "auto merge requires auto-QA",
            ),
            (
                {
                    "title": "Improve docs",
                    "goal": "Find useful docs increments.",
                    "ambition": AutonomousGoal.AMBITION_HIGH,
                    "autonomy": AutonomousGoal.AUTONOMY_DRAFT_PATCH,
                    "auto_qa": "true",
                    "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
                    "auto_merge_to_local_branch": "true",
                    "auto_merge_branch": "missing",
                },
                "auto merge branch is invalid",
            ),
        ):
            with self.subTest(message=message):
                response = self.client.post(
                    reverse("edit_autonomous_goal", args=[goal.pk]),
                    data,
                )

                self.assertContains(response, message, status_code=400)

        goal.refresh_from_db()
        self.assertEqual(goal.title, "Improve tests")
        self.assertEqual(goal.goal, "Find useful test coverage increments.")

    def test_delete_autonomous_goal_soft_deletes_selected_project_goal(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        other_project = Project.objects.create(name="Other", repo_path="/other")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            auto_proposal_enabled=True,
        )
        other_goal = AutonomousGoal.objects.create(
            project=other_project,
            title="Other goal",
            goal="Should stay.",
        )

        response = self.client.post(reverse("delete_autonomous_goal", args=[goal.pk]))

        self.assertEqual(response.status_code, 302)
        goal.refresh_from_db()
        other_goal.refresh_from_db()
        self.assertIsNotNone(goal.deleted_at)
        self.assertFalse(goal.auto_proposal_enabled)
        self.assertIsNone(other_goal.deleted_at)

    def test_delete_autonomous_goal_is_scoped_to_selected_project(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        other_project = Project.objects.create(name="Other", repo_path="/other")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=other_project,
            title="Other goal",
            goal="Should not delete.",
        )

        response = self.client.post(reverse("delete_autonomous_goal", args=[goal.pk]))

        self.assertEqual(response.status_code, 404)
        goal.refresh_from_db()
        self.assertIsNone(goal.deleted_at)

    def test_delete_autonomous_goal_preserves_accepted_proposal(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo",
            project=project,
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            candidate_session=candidate,
            accepted_session=candidate,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            title="Add parser coverage",
        )

        response = self.client.post(reverse("delete_autonomous_goal", args=[goal.pk]))

        self.assertEqual(response.status_code, 302)
        goal.refresh_from_db()
        self.assertIsNotNone(goal.deleted_at)
        proposal.refresh_from_db()
        self.assertEqual(proposal.autonomous_goal_id, goal.pk)
        self.assertEqual(
            system_agents.accepted_visible_system_thread_ids(),
            {"candidate-thread"},
        )

    @patch("hitch.main.views.cleanup_managed_worktree_path")
    def test_delete_autonomous_goal_dismisses_unresolved_proposal(
        self, mock_cleanup: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo-worktree",
            project=project,
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            candidate_session=candidate,
            title="Add parser coverage",
        )

        response = self.client.post(reverse("delete_autonomous_goal", args=[goal.pk]))

        self.assertEqual(response.status_code, 302)
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_DISMISSED)
        self.assertEqual(
            proposal.outcome_notes,
            system_agents.AUTONOMOUS_GOAL_DELETED_ERROR,
        )
        mock_cleanup.assert_called_once_with("/repo-worktree")

    @patch("hitch.main.views.cleanup_managed_worktree_path")
    def test_delete_autonomous_goal_cleans_hidden_stacked_proposal(
        self, mock_cleanup: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo-worktree",
            project=project,
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            candidate_session=candidate,
            outcome_status=ProposedSession.OUTCOME_DISMISSED,
            outcome_metadata={"stacked_diff_hidden_until_complete": True},
            title="Add parser coverage",
        )

        response = self.client.post(reverse("delete_autonomous_goal", args=[goal.pk]))

        self.assertEqual(response.status_code, 302)
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_DISMISSED)
        self.assertEqual(
            proposal.outcome_notes,
            system_agents.AUTONOMOUS_GOAL_DELETED_ERROR,
        )
        self.assertFalse(
            proposal.outcome_metadata["stacked_diff_hidden_until_complete"]
        )
        mock_cleanup.assert_called_once_with("/repo-worktree")

    @patch("hitch.main.views.cleanup_managed_worktree_path")
    def test_delete_autonomous_goal_keeps_accepted_proposal_worktree(
        self, mock_cleanup: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo-worktree",
            project=project,
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            candidate_session=candidate,
            accepted_session=candidate,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            title="Add parser coverage",
        )

        response = self.client.post(reverse("delete_autonomous_goal", args=[goal.pk]))

        self.assertEqual(response.status_code, 302)
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_ACCEPTED)
        mock_cleanup.assert_not_called()

    @patch("hitch.main.system_agents.codex_pool.interrupt_instance")
    def test_delete_autonomous_goal_reconciles_terminal_running_workflow(
        self, mock_interrupt: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=system_agents._autonomous_goal_main_thread_id(goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
            state={"autonomous_goal_id": goal.pk},
        )
        instance = CodexInstance.objects.create(
            pid=0,
            thread_id="goal-thread",
            cwd="/repo",
            prompt="run autonomous goal",
            events_path="/dev/null",
            status=CodexInstance.STATUS_FAILED,
            error="worker process exited before callback",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id=instance.thread_id,
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )

        response = self.client.post(reverse("delete_autonomous_goal", args=[goal.pk]))

        self.assertEqual(response.status_code, 302)
        mock_interrupt.assert_not_called()
        goal.refresh_from_db()
        run.refresh_from_db()
        workflow.refresh_from_db()
        self.assertIsNotNone(goal.deleted_at)
        self.assertEqual(run.status, SystemAgentRun.STATUS_FAILED)
        self.assertIn("worker process exited before callback", run.error)
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
        proposal = ProposedSession.objects.get(source_workflow=workflow)
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_DISMISSED)
        self.assertEqual(
            proposal.outcome_notes,
            system_agents.AUTONOMOUS_GOAL_DELETED_ERROR,
        )

    @patch("hitch.main.system_agents.cleanup_managed_worktree_path")
    @patch("hitch.main.system_agents.codex_pool.interrupt_instance")
    def test_delete_autonomous_goal_stops_running_workflow(
        self, mock_interrupt: MagicMock, mock_cleanup: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=system_agents._autonomous_goal_main_thread_id(goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
            state={"autonomous_goal_id": goal.pk, "session_cwd": "/repo-worktree"},
        )
        instance = CodexInstance.objects.create(
            pid=0,
            thread_id="goal-thread",
            cwd="/repo",
            prompt="run autonomous goal",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id=instance.thread_id,
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )
        mock_interrupt.return_value = instance

        response = self.client.post(reverse("delete_autonomous_goal", args=[goal.pk]))

        self.assertEqual(response.status_code, 302)
        mock_interrupt.assert_called_once_with(
            instance.pk, expected_thread_id=instance.thread_id
        )
        mock_cleanup.assert_not_called()
        run.refresh_from_db()
        workflow.refresh_from_db()
        self.assertEqual(run.status, SystemAgentRun.STATUS_FAILED)
        self.assertEqual(run.error, "Autonomous goal deleted by user")
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
        self.assertEqual(workflow.step, system_agents.STEP_BLOCKED)
        self.assertEqual(workflow.state["error"], "Autonomous goal deleted by user")
        goal.refresh_from_db()
        self.assertIsNotNone(goal.deleted_at)

    @patch("hitch.main.system_agents.cleanup_managed_worktree_path")
    @patch("hitch.main.system_agents.codex_pool.interrupt_instance")
    def test_delete_autonomous_goal_cleans_worktree_when_interrupt_is_terminal(
        self, mock_interrupt: MagicMock, mock_cleanup: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=system_agents._autonomous_goal_main_thread_id(goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
            state={"autonomous_goal_id": goal.pk, "session_cwd": "/repo-worktree"},
        )
        instance = CodexInstance.objects.create(
            pid=0,
            thread_id="goal-thread",
            cwd="/repo",
            prompt="run autonomous goal",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id=instance.thread_id,
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )
        terminal_instance = instance
        terminal_instance.status = CodexInstance.STATUS_FAILED
        mock_interrupt.return_value = terminal_instance

        response = self.client.post(reverse("delete_autonomous_goal", args=[goal.pk]))

        self.assertEqual(response.status_code, 302)
        run.refresh_from_db()
        self.assertEqual(run.status, SystemAgentRun.STATUS_FAILED)
        mock_cleanup.assert_called_once_with("/repo-worktree")

    @patch("hitch.main.system_agents.cleanup_managed_worktree_path")
    @patch(
        "hitch.main.system_agents.codex_pool.interrupt_instance",
        return_value=None,
    )
    def test_delete_autonomous_goal_keeps_goal_when_running_workflow_cannot_stop(
        self, mock_interrupt: MagicMock, mock_cleanup: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=system_agents._autonomous_goal_main_thread_id(goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
            state={"autonomous_goal_id": goal.pk, "session_cwd": "/repo-worktree"},
        )
        instance = CodexInstance.objects.create(
            pid=0,
            thread_id="goal-thread",
            cwd="/repo",
            prompt="run autonomous goal",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id=instance.thread_id,
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )

        response = self.client.post(reverse("delete_autonomous_goal", args=[goal.pk]))

        self.assertContains(
            response,
            "autonomous goal run could not be stopped",
            status_code=400,
        )
        mock_interrupt.assert_called_once_with(
            instance.pk, expected_thread_id=instance.thread_id
        )
        mock_cleanup.assert_not_called()
        run.refresh_from_db()
        workflow.refresh_from_db()
        self.assertEqual(run.status, SystemAgentRun.STATUS_RUNNING)
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_RUNNING)
        goal.refresh_from_db()
        self.assertIsNone(goal.deleted_at)

    @patch("hitch.main.views.system_agents.start_autonomous_goal_workflow")
    def test_run_single_starts_selected_project_goal(
        self, mock_start: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        other_project = Project.objects.create(name="Other", repo_path="/other")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        AutonomousGoal.objects.create(
            project=other_project,
            title="Other goal",
            goal="Should not run.",
        )

        response = self.client.post(reverse("run_autonomous_goal", args=[goal.pk]))

        self.assertEqual(response.status_code, 302)
        self.assertEqual(mock_start.call_count, 1)
        self.assertEqual(mock_start.call_args.kwargs["autonomous_goal"], goal)
        self.assertTrue(mock_start.call_args.kwargs["use_worktrees"])

    @patch("hitch.main.views.system_agents.start_autonomous_goal_workflow")
    def test_run_single_always_uses_worktrees(
        self, mock_start: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(
            self.client,
            hitch_selected_project_id=str(project.pk),
            **{_USE_WORKTREES_COOKIE: "true"},
        )
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )

        response = self.client.post(reverse("run_autonomous_goal", args=[goal.pk]))

        self.assertEqual(response.status_code, 302)
        self.assertTrue(mock_start.call_args.kwargs["use_worktrees"])

    @patch("hitch.main.views.system_agents.start_autonomous_goal_workflow")
    def test_run_single_is_scoped_to_selected_project(
        self, mock_start: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        other_project = Project.objects.create(name="Other", repo_path="/other")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=other_project,
            title="Other goal",
            goal="Should not run.",
        )

        response = self.client.post(reverse("run_autonomous_goal", args=[goal.pk]))

        self.assertEqual(response.status_code, 404)
        mock_start.assert_not_called()

    @patch("hitch.main.views.system_agents.start_autonomous_goal_workflow")
    def test_run_all_starts_each_selected_project_goal(
        self, mock_start: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        other_project = Project.objects.create(name="Other", repo_path="/other")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        first = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        second = AutonomousGoal.objects.create(
            project=project,
            title="Improve docs",
            goal="Find useful docs increments.",
        )
        AutonomousGoal.objects.create(
            project=project,
            title="Deleted goal",
            goal="Should not run.",
            deleted_at=timezone.now(),
        )
        AutonomousGoal.objects.create(
            project=other_project,
            title="Other goal",
            goal="Should not run.",
        )

        response = self.client.post(reverse("run_autonomous_goals"))

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            [call.kwargs["autonomous_goal"] for call in mock_start.call_args_list],
            [first, second],
        )
        self.assertEqual(
            [call.kwargs["use_worktrees"] for call in mock_start.call_args_list],
            [True, True],
        )

    @patch("hitch.main.views.cleanup_managed_worktree_path")
    def test_reject_proposed_session_requires_reason(
        self, mock_cleanup: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            title="Add parser coverage",
        )

        response = self.client.post(
            reverse("update_proposed_session_outcome", args=[proposal.pk]),
            {"outcome_status": ProposedSession.OUTCOME_REJECTED},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.content, b"reason is required")
        mock_cleanup.assert_not_called()

    @patch("hitch.main.views.Codex")
    def test_accept_proposed_session_links_candidate_session(
        self, mock_codex: MagicMock
    ) -> None:
        codex = _setup_codex(mock_codex, models=[])
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo",
            project=project,
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            candidate_session=candidate,
            title="Add parser coverage",
        )

        response = self.client.post(
            reverse("update_proposed_session_outcome", args=[proposal.pk]),
            {"outcome_status": ProposedSession.OUTCOME_ACCEPTED},
        )

        self.assertEqual(response.status_code, 302)
        proposal.refresh_from_db()
        candidate.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_ACCEPTED)
        self.assertEqual(proposal.accepted_session, candidate)
        self.assertEqual(candidate.codex_name, "Add parser coverage")
        self.assertEqual(candidate.codex_display_title, "Add parser coverage")
        codex._client.thread_set_name.assert_called_once_with(
            "candidate-thread", "Add parser coverage"
        )

    def test_dismiss_notice_updates_outcome(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        notice = ProposedSession.objects.create(
            autonomous_goal=goal,
            title="No proposal from Improve tests",
            inbox_kind=ProposedSession.INBOX_KIND_NOTICE,
            summary="No concrete test increment was worth proposing.",
        )

        response = self.client.post(
            reverse("update_proposed_session_outcome", args=[notice.pk]),
            {"outcome_status": ProposedSession.OUTCOME_DISMISSED},
        )

        self.assertEqual(response.status_code, 302)
        notice.refresh_from_db()
        self.assertEqual(notice.outcome_status, ProposedSession.OUTCOME_DISMISSED)

    def test_notice_rejects_non_dismissed_outcome(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        notice = ProposedSession.objects.create(
            autonomous_goal=goal,
            title="No proposal from Improve tests",
            inbox_kind=ProposedSession.INBOX_KIND_NOTICE,
        )

        response = self.client.post(
            reverse("update_proposed_session_outcome", args=[notice.pk]),
            {"outcome_status": ProposedSession.OUTCOME_ACCEPTED},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.content, b"outcome status is invalid")

    @patch("hitch.main.views.cleanup_managed_worktree_path")
    def test_dismiss_proposed_session_uses_distinct_outcome(
        self, mock_cleanup: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            title="Add parser coverage",
            candidate_session=SessionMetadata.objects.create(
                thread_id="candidate-thread",
                cwd="/repo-worktree",
                project=project,
            ),
        )

        response = self.client.post(
            reverse("update_proposed_session_outcome", args=[proposal.pk]),
            {"outcome_status": ProposedSession.OUTCOME_DISMISSED},
        )

        self.assertEqual(response.status_code, 302)
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_DISMISSED)
        self.assertNotEqual(proposal.outcome_status, ProposedSession.OUTCOME_REJECTED)
        self.assertEqual(proposal.outcome_notes, "")
        mock_cleanup.assert_called_once_with("/repo-worktree")

    @patch("hitch.main.views.cleanup_managed_worktree_path")
    def test_reject_proposed_session_cleans_candidate_worktree(
        self, mock_cleanup: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo-worktree",
            project=project,
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            title="Add parser coverage",
            candidate_session=candidate,
        )

        response = self.client.post(
            reverse("update_proposed_session_outcome", args=[proposal.pk]),
            {
                "outcome_status": ProposedSession.OUTCOME_REJECTED,
                "reason": "Not useful enough.",
            },
        )

        self.assertEqual(response.status_code, 302)
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_REJECTED)
        self.assertEqual(proposal.outcome_notes, "Not useful enough.")
        mock_cleanup.assert_called_once_with("/repo-worktree")

    @patch("hitch.main.views.cleanup_managed_worktree_path")
    def test_update_outcome_rejects_already_resolved_proposal(
        self, mock_cleanup: MagicMock
    ) -> None:
        # A proposal accepted into its candidate session (accepted_session ==
        # candidate_session) un-hides that otherwise-hidden system thread, so the
        # user can see and work in it. A stale inbox tab can still post a
        # dismiss/reject for the same proposal; re-deciding it must be refused so
        # the recorded outcome is not corrupted and the live session stays
        # visible.
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo-worktree",
            project=project,
            is_hidden_system_session=True,
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            title="Add parser coverage",
            candidate_session=candidate,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            accepted_session=candidate,
        )
        self.assertIn(
            "candidate-thread",
            system_agents.accepted_visible_system_thread_ids(),
        )

        for outcome in (
            ProposedSession.OUTCOME_DISMISSED,
            ProposedSession.OUTCOME_REJECTED,
        ):
            with self.subTest(outcome=outcome):
                response = self.client.post(
                    reverse("update_proposed_session_outcome", args=[proposal.pk]),
                    {"outcome_status": outcome, "reason": "Changed my mind."},
                )

                self.assertEqual(response.status_code, 400)
                self.assertEqual(
                    response.content, b"proposed session has already been resolved"
                )
                proposal.refresh_from_db()
                self.assertEqual(
                    proposal.outcome_status, ProposedSession.OUTCOME_ACCEPTED
                )
                self.assertEqual(proposal.accepted_session, candidate)
        # The accepted session stayed visible and its worktree was never removed.
        self.assertIn(
            "candidate-thread",
            system_agents.accepted_visible_system_thread_ids(),
        )
        mock_cleanup.assert_not_called()

    def test_update_outcome_rejects_unset_target_status(self) -> None:
        # OUTCOME_UNSET is the pending inbox state, not a decision; the endpoint
        # must not let a request re-open a proposal by posting it.
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            title="Add parser coverage",
        )

        response = self.client.post(
            reverse("update_proposed_session_outcome", args=[proposal.pk]),
            {"outcome_status": ProposedSession.OUTCOME_UNSET},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.content, b"outcome status is invalid")
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_UNSET)

    def test_accept_helper_does_not_overwrite_resolved_proposal(self) -> None:
        # The accept path (new-session "Do it") and the inbox outcome endpoint
        # race on the same proposal. If the inbox endpoint already rejected it
        # -- which also cleans up the candidate worktree -- the accept helper
        # must leave that decision intact rather than flip it to accepted, which
        # would leave accepted_session pointing at a removed worktree. Exactly
        # one transition wins across both endpoints.
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        proposal = ProposedSession.objects.create(
            project=project,
            title="Add parser coverage",
            outcome_status=ProposedSession.OUTCOME_REJECTED,
            outcome_notes="Not useful enough.",
        )
        started = SessionMetadata.objects.create(
            thread_id="started-thread",
            cwd="/repo",
            project=project,
        )

        views._accept_proposed_session_for_session(proposal, started)

        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_REJECTED)
        self.assertIsNone(proposal.accepted_session)
        self.assertEqual(proposal.outcome_notes, "Not useful enough.")


class ResetStaleStageCacheMigrationTests(TransactionTestCase):
    """The 0048 data migration heals transient stage rows persisted by the
    pre-fix write path, which the mtime-keyed read guard would otherwise serve
    indefinitely after the active owner exits without rewriting the rollout."""

    migrate_from = [("main", "0047_sessionmetadata_derived_stage")]
    migrate_to = [("main", "0048_reset_stale_stage_cache")]

    def _migrate(self, targets: list[tuple[str, str]]) -> MigrationExecutor:
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(targets)
        return executor

    def test_reset_clears_persisted_stage_cache(self) -> None:
        leaf = MigrationExecutor(connection).loader.graph.leaf_nodes("main")
        self.addCleanup(self._migrate, leaf)

        old_apps = self._migrate(self.migrate_from).loader.project_state(
            self.migrate_from
        ).apps
        SessionMetadata = old_apps.get_model("main", "SessionMetadata")
        SessionMetadata.objects.create(
            thread_id="stale-transient",
            derived_stage="implementation",
            derived_stage_source_mtime_ns=123,
        )
        SessionMetadata.objects.create(
            thread_id="already-empty",
            derived_stage="",
            derived_stage_source_mtime_ns=0,
        )

        new_apps = self._migrate(self.migrate_to).loader.project_state(
            self.migrate_to
        ).apps
        SessionMetadata = new_apps.get_model("main", "SessionMetadata")
        stale = SessionMetadata.objects.get(thread_id="stale-transient")
        self.assertEqual(stale.derived_stage, "")
        self.assertEqual(stale.derived_stage_source_mtime_ns, 0)
        empty = SessionMetadata.objects.get(thread_id="already-empty")
        self.assertEqual(empty.derived_stage, "")
        self.assertEqual(empty.derived_stage_source_mtime_ns, 0)


class PrStageRefreshSchedulingTests(TestCase):
    @override
    def tearDown(self) -> None:
        # The threaded path adds to a module-level in-flight set; keep tests
        # isolated by clearing it.
        with views._PR_STAGE_REFRESH_INFLIGHT_LOCK:
            views._PR_STAGE_REFRESH_INFLIGHT.clear()

    @patch("hitch.main.views._refresh_session_pr_stage")
    def test_schedule_runs_inline_under_testing(
        self, mock_refresh: MagicMock
    ) -> None:
        views._schedule_pr_stage_refresh("sess-1")
        mock_refresh.assert_called_once_with("sess-1")

    @patch("hitch.main.views._refresh_session_pr_stage")
    @patch("hitch.main.views.threading.Thread")
    def test_schedule_spawns_one_thread_per_session_off_request(
        self, mock_thread: MagicMock, _mock_refresh: MagicMock
    ) -> None:
        with self.settings(TESTING=False):
            views._schedule_pr_stage_refresh("sess-x")
            # A concurrent render for the same session does not spawn a duplicate.
            views._schedule_pr_stage_refresh("sess-x")
        mock_thread.assert_called_once()
        mock_thread.return_value.start.assert_called_once()

    @patch("hitch.main.views._schedule_pr_stage_refresh")
    @patch("hitch.main.views.system_agents.pr_snapshot_stage_refresh_due", return_value=True)
    @patch("hitch.main.views._pr_snapshot_for_rollout_path")
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
        rollout_state = views._RolloutFileState(path=Path("/tmp/rollout.jsonl"), mtime_ns=1)
        session = {"cwd": "/repo", "stage_pr_refresh_attempted_at": None}

        _stage, _snap, remaining, refreshing = views._stage_from_cached_session_row(
            "sess-budget",
            session,
            rollout_state=rollout_state,
            cached_stage=session_stage.PR,
            pr_stage_refreshes_remaining=1,
        )
        self.assertTrue(refreshing)
        self.assertEqual(remaining, 0)
        mock_schedule.assert_called_once_with("sess-budget")

        mock_schedule.reset_mock()

        _stage, _snap, remaining, refreshing = views._stage_from_cached_session_row(
            "sess-exhausted",
            session,
            rollout_state=rollout_state,
            cached_stage=session_stage.PR,
            pr_stage_refreshes_remaining=0,
        )
        self.assertFalse(refreshing)
        self.assertEqual(remaining, 0)
        mock_schedule.assert_not_called()
