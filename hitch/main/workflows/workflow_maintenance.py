"""Background maintenance for durable Hitch system workflows."""

from __future__ import annotations

import logging
import threading
import time

from django.db import close_old_connections
from django.utils import timezone

from hitch.main.runtime import codex_pool, disk_cleanup, server_lifecycle
from hitch.main.workflows import system_agents

logger = logging.getLogger(__name__)

_WORKFLOW_MAINTENANCE_INTERVAL_SECONDS = 60
_DISK_USAGE_CLEANUP_INTERVAL_SECONDS = 10 * 60
_STALE_BLOCKED_ARCHIVE_INTERVAL_SECONDS = 60 * 60
# Cap PR-stage refreshes per tick: each due session can spend up to the gh-pr-
# view timeout, and this tick also owns reconcile_dead and PR-monitor backoff
# polling, so an unbounded sweep over dozens of stale sessions would delay the
# next reconcile by minutes and revive the stale-running-badge problem. The
# leftover rows converge on later ticks (and on demand from the request path).
_PR_STAGE_REFRESH_LIMIT_PER_TICK = 5
# Same rationale for the PR-monitor backoff sweep: each due monitor shells out to
# gh, so cap how many one tick polls and let the rest converge on later ticks.
_PR_MONITOR_BACKOFF_LIMIT_PER_TICK = 5
_SCHEDULER_ENV = "HITCH_WORKFLOW_MAINTENANCE_SCHEDULER"

_scheduler = server_lifecycle.SchedulerHandle(
    thread_name="hitch-workflow-maintenance"
)


def start_workflow_maintenance_scheduler() -> bool:
    """Start the in-process workflow maintenance scheduler when enabled."""
    if not _workflow_maintenance_scheduler_enabled():
        return False
    return _scheduler.start(_workflow_maintenance_scheduler_loop)


def _workflow_maintenance_scheduler_enabled() -> bool:
    return server_lifecycle.background_work_enabled(
        env_var=_SCHEDULER_ENV,
        include_wsgi_server_commands=True,
    )


def _workflow_maintenance_scheduler_loop() -> None:
    stop = threading.Event()
    start = time.monotonic()
    next_stale_blocked_archive_at = start + _STALE_BLOCKED_ARCHIVE_INTERVAL_SECONDS
    next_disk_cleanup_at = start + _DISK_USAGE_CLEANUP_INTERVAL_SECONDS
    while True:
        _run_workflow_maintenance_scheduler_tick()
        # Archive stale blocked PR-QA workflows so they stop surfacing a stale
        # "Blocked" badge in the inbox. Disk cleanup no longer depends on this:
        # blocked workflows are a failure state and no longer pin their worktrees.
        next_stale_blocked_archive_at = _run_due_stale_blocked_archive(
            next_due_at=next_stale_blocked_archive_at
        )
        next_disk_cleanup_at = _run_due_disk_usage_cleanup(
            next_due_at=next_disk_cleanup_at
        )
        stop.wait(_WORKFLOW_MAINTENANCE_INTERVAL_SECONDS)


def _run_workflow_maintenance_scheduler_tick() -> None:
    close_old_connections()
    try:
        codex_pool.reconcile_dead()
        refreshed = system_agents.refresh_due_pr_monitor_backoffs(
            limit=_PR_MONITOR_BACKOFF_LIMIT_PER_TICK
        )
        if refreshed:
            logger.info("refreshed %s PR monitor backoff workflow(s)", refreshed)
        # Converge GitHub-backed PR stages in the background. This scheduler
        # runs under production server commands (gunicorn et al.), whereas the
        # auto-proposal scheduler does not, so without this the per-session
        # `gh pr view` stage refresh only ever fires from the session-list
        # request path (capped at one row per render) -- dominating dashboard
        # latency once a session's 5-minute refresh window elapses.
        pr_stages = system_agents.refresh_unarchived_session_pr_stages(
            limit=_PR_STAGE_REFRESH_LIMIT_PER_TICK
        )
        if pr_stages:
            logger.info("refreshed %s session PR stage(s)", pr_stages)
    except Exception:
        logger.exception("failed to run workflow maintenance scheduler tick")
    finally:
        close_old_connections()


def _run_due_disk_usage_cleanup(
    *,
    next_due_at: float,
    now: float | None = None,
) -> float:
    current = time.monotonic() if now is None else now
    if current < next_due_at:
        return next_due_at

    close_old_connections()
    try:
        disk_cleanup.run_finished_session_disk_cleanup()
    except Exception:
        logger.exception("failed to run scheduled Hitch disk cleanup")
    finally:
        close_old_connections()
    return current + _DISK_USAGE_CLEANUP_INTERVAL_SECONDS


def _run_due_stale_blocked_archive(
    *,
    next_due_at: float,
    now: float | None = None,
) -> float:
    """Archive stale blocked PR-QA workflows when this hourly tick comes due.

    This clears the stale "Blocked" badge from the inbox. It no longer affects
    disk reclamation: ``disk_cleanup`` only treats RUNNING workflows as pinning a
    worktree, so a blocked workflow's worktree is already eligible for cleanup.
    """
    current = time.monotonic() if now is None else now
    if current < next_due_at:
        return next_due_at

    close_old_connections()
    try:
        cutoff = timezone.now() - system_agents.STALE_BLOCKED_AGE
        archived_ids = system_agents.archive_stale_blocked_workflows(
            older_than=cutoff, apply=True
        )
        if archived_ids:
            logger.info(
                "archived %s stale blocked PR-QA workflow(s): %s",
                len(archived_ids),
                ", ".join(str(workflow_id) for workflow_id in archived_ids),
            )
    except Exception:
        logger.exception("failed to run scheduled stale blocked workflow archive")
    finally:
        close_old_connections()
    return current + _STALE_BLOCKED_ARCHIVE_INTERVAL_SECONDS
