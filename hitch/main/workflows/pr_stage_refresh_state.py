"""PR-stage-refresh debounce and cache-state helpers.

Extracted verbatim from ``system_agents`` so the central per-PR refresh
debounce window, the per-workflow attempt timestamps, and the hitch PR
handoff marker cache live in one focused module. Pure code movement: no
behaviour changes.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from django.utils import timezone

from hitch.main.models import SystemWorkflow
from hitch.main.runtime import rate_limit
from hitch.main.runtime.sdk_values import is_nonbool_int, string_from_any
from hitch.main.workflows.pr_handoff import _compact_pr_handoff, _pr_handoff_is_terminal

_SECONDS_PER_MINUTE = 60
_PR_HITCH_HANDOFF_STATE_KEY = "hitch_pr_handoff"
_PR_STAGE_REFRESH_MIN_SECONDS = 5 * _SECONDS_PER_MINUTE
_PR_STAGE_REFRESH_STATE_KEY = "pr_stage_refresh"


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

    Keying on PR *identity* -- not the workflow or session that triggered the
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

    Layered on top of the per-workflow / per-session windows so renders and
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


def _mark_pr_stage_refresh_attempt(workflow: SystemWorkflow) -> None:
    workflow.state = {
        **workflow.state,
        _PR_STAGE_REFRESH_STATE_KEY: {
            "attempted_at": int(timezone.now().timestamp()),
        },
    }


def _pr_stage_refresh_attempted_at(workflow: SystemWorkflow) -> int:
    value = workflow.state.get(_PR_STAGE_REFRESH_STATE_KEY)
    if not isinstance(value, dict):
        return 0
    attempted_at = value.get("attempted_at")
    if is_nonbool_int(attempted_at):
        return attempted_at
    return 0


def hitch_pr_handoff_for_workflow(workflow: SystemWorkflow | None) -> dict[str, Any]:
    if workflow is None or workflow.kind != SystemWorkflow.KIND_PR_QA:
        return {}
    return _hitch_pr_handoff_marker(workflow.state.get(_PR_HITCH_HANDOFF_STATE_KEY))


def _mark_hitch_pr_handoff(workflow: SystemWorkflow, handoff: dict[str, Any]) -> None:
    marker = _hitch_pr_handoff_marker(handoff)
    if marker:
        workflow.state = {**workflow.state, _PR_HITCH_HANDOFF_STATE_KEY: marker}


def _hitch_pr_handoff_marker(value: Any) -> dict[str, Any]:
    handoff = _compact_pr_handoff(value)
    marker: dict[str, Any] = {}
    for key in ("url", "repository_full_name", "pr_number"):
        if key in handoff:
            marker[key] = handoff[key]
    if "url" in marker or (
        "repository_full_name" in marker and "pr_number" in marker
    ):
        return marker
    return {}
