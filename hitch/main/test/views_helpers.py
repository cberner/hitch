"""Shared fixtures for the view-layer test modules (test_views_*)."""


import os
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, override

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import (
    TestCase,
)
from openai_codex.generated.v2_all import (
    ThreadSource,
)

from hitch.main.models import (
    ArchivedSessionTokenUsage,
    Project,
    SessionMetadata,
)
from hitch.main.sessions import (
    session_index,
    token_usage,
)
from hitch.main.test.support import (
    _rollout_line,
)
from hitch.main.workflows import system_agents

_PNG_BYTES = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
_JPEG_BYTES = b"\xff\xd8\xff\xe0JFIF"
_GIF_BYTES = b"GIF89a\x01\x00\x01\x00"
_WEBP_BYTES = b"RIFF\x0c\x00\x00\x00WEBPVP8 "

_SHOW_ARCHIVED_COOKIE = "hitch_show_archived_sessions"
_MODEL_COOKIE = "hitch_model"
_SANDBOX_COOKIE = "hitch_sandbox_policy"
_APPROVAL_COOKIE = "hitch_approval_mode"
_EXTRA_SYSTEM_PROMPT_COOKIE = "hitch_extra_system_prompt"
_USE_WORKTREES_COOKIE = "hitch_use_worktrees"
_AUTO_PR_COOKIE = "hitch_auto_pr"
_AUTO_QA_COOKIE = "hitch_auto_qa"
_SPEC_CRITIC_COOKIE = "hitch_spec_critic"
_WEB_SEARCH_COOKIE = "hitch_web_search_mode"
_LAST_SELECTED_REPO_COOKIE = "hitch_last_selected_repo"
_SELECTED_PROJECT_COOKIE = "hitch_selected_project_id"
_VISIBLE_SESSION_PROJECTS_COOKIE = "hitch_visible_session_project_ids"
_SHOW_NO_PROJECT_SESSIONS_COOKIE = "hitch_show_no_project_sessions"
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
    project: Project | None = None,
    thread_source: str = "",
    mark_index_complete: bool = True,
) -> SessionMetadata:
    if mark_index_complete:
        session_index.mark_synced(archived=False, complete=True)
        session_index.mark_synced(archived=True, complete=True)
    return SessionMetadata.objects.create(
        thread_id=thread_id,
        codex_path=str(path),
        project=project,
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
    usage_logic_version: int = token_usage._TOKEN_USAGE_LOGIC_VERSION,
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


def _run_borrowed_with(
    client: Any,
) -> Callable[..., object]:
    def side_effect(
        _factory: object, operation: Callable[[Any], object], **_kwargs: object
    ) -> object:
        return operation(client)

    return side_effect


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
