"""Background maintenance for durable Hitch system workflows."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time

from django.conf import settings
from django.db import close_old_connections
from django.utils import timezone

from hitch.main import codex_pool, disk_cleanup, system_agents

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
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_SERVER_COMMANDS = frozenset({"gunicorn", "uvicorn", "daphne", "uwsgi"})

_scheduler_lock = threading.Lock()
_scheduler_started = False


def start_workflow_maintenance_scheduler() -> bool:
    """Start the in-process workflow maintenance scheduler when enabled."""
    global _scheduler_started
    if not _workflow_maintenance_scheduler_enabled():
        return False
    with _scheduler_lock:
        if _scheduler_started:
            return False
        _scheduler_started = True
        threading.Thread(
            target=_workflow_maintenance_scheduler_loop,
            name="hitch-workflow-maintenance",
            daemon=True,
        ).start()
        return True


def _workflow_maintenance_scheduler_enabled() -> bool:
    configured = os.environ.get(_SCHEDULER_ENV)
    if configured is not None:
        normalized = configured.strip().lower()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False

    if getattr(settings, "TESTING", False):
        return False
    argv = sys.argv[1:]
    if argv and argv[0] == "runserver":
        return os.environ.get("RUN_MAIN") == "true" or "--noreload" in argv
    return _running_from_server_command()


def _running_from_server_command() -> bool:
    return os.path.basename(sys.argv[0]) in _SERVER_COMMANDS


def _workflow_maintenance_scheduler_loop() -> None:
    stop = threading.Event()
    start = time.monotonic()
    next_stale_blocked_archive_at = start + _STALE_BLOCKED_ARCHIVE_INTERVAL_SECONDS
    next_disk_cleanup_at = start + _DISK_USAGE_CLEANUP_INTERVAL_SECONDS
    while True:
        _run_workflow_maintenance_scheduler_tick()
        # Archive stale blocked workflows before disk cleanup on the same loop
        # iteration: archiving moves them out of RUNNING/BLOCKED, which unpins
        # their worktrees so the disk-cleanup tick that follows can reclaim them.
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

    Runs ahead of the disk-cleanup tick on the same loop iteration: archiving a
    long-blocked workflow drops it out of the RUNNING/BLOCKED set that
    ``disk_cleanup`` treats as pinning a worktree, so its worktree becomes
    eligible for reclamation in the very next tick rather than the next hour.
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
