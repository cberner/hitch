"""Shared debounce helpers for GitHub-backed PR stage refreshes."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from django.utils import timezone

from hitch.main.runtime import rate_limit
from hitch.main.runtime.sdk_values import is_nonbool_int, string_from_any
from hitch.main.workflows.pr_handoff import _pr_handoff_is_terminal

_SECONDS_PER_MINUTE = 60
_PR_STAGE_REFRESH_MIN_SECONDS = 5 * _SECONDS_PER_MINUTE


def _pr_handoff_selector(handoff: dict[str, Any]) -> str:
    url = string_from_any(handoff.get("url"))
    if url:
        return url
    number = handoff.get("pr_number")
    if is_nonbool_int(number):
        return str(number)
    return ""


def _pr_stage_rate_limit_key(handoff: Mapping[str, Any]) -> str:
    """Stable key identifying a PR for the central refresh debounce.

    Keying on PR *identity* -- not the tool invocation or session that triggered the
    refresh -- is what makes the floor global: the list view, the detail view,
    both background schedulers, and every session pointing at the same PR share
    one window.
    """
    url = string_from_any(handoff.get("url"))
    if url:
        return f"gh:pr-view:{url}"
    repo = string_from_any(handoff.get("repository_full_name"))
    number = handoff.get("pr_number")
    if is_nonbool_int(number):
        return f"gh:pr-view:{repo}#{number}" if repo else f"gh:pr-view:#{number}"
    return ""


def _pr_stage_refresh_globally_due(handoff: Mapping[str, Any]) -> bool:
    """Whether the central per-PR debounce window is open for this handoff.

    Layered on top of the per-record / per-session windows so renders and
    background workers do not flag a PR as refreshing -- and therefore schedule
    a worker and trigger a page reload -- when another path refreshed the same
    PR within the global window. ``refreshed_pr_*`` still claim atomically; this
    read-only check just keeps the UI from looping on a window that will deny.
    """
    key = _pr_stage_rate_limit_key(handoff)
    return not key or rate_limit.due(key)


def _should_refresh_pr_snapshot_for_stage(
    cwd: str,
    handoff: dict[str, Any],
    *,
    attempted_at: datetime | None,
    force: bool,
) -> bool:
    if _pr_handoff_is_terminal(handoff):
        return False
    if not _pr_handoff_selector(handoff):
        return False
    if not Path(cwd).is_dir():
        return False
    if force:
        return True
    if attempted_at is None:
        return True
    attempted_seconds = int(attempted_at.timestamp())
    return int(timezone.now().timestamp()) - attempted_seconds >= (
        _PR_STAGE_REFRESH_MIN_SECONDS
    )
