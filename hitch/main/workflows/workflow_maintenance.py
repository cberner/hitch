"""Background maintenance for durable Hitch system workflows."""

from __future__ import annotations

import logging
import threading
import time

from django.db import close_old_connections

from hitch.main.runtime import disk_cleanup, reconciliation, retention, server_lifecycle

logger = logging.getLogger(__name__)

_WORKFLOW_MAINTENANCE_INTERVAL_SECONDS = 60
_DISK_USAGE_CLEANUP_INTERVAL_SECONDS = 10 * 60
# Row/file retention runs daily, but the first sweep happens shortly after
# startup: this server restarts often enough that "a day after boot" could
# mean never, and a backlogged first sweep is already bounded per pass.
_ROW_RETENTION_INTERVAL_SECONDS = 24 * 60 * 60
_ROW_RETENTION_STARTUP_DELAY_SECONDS = 15 * 60
_SCHEDULER_ENV = "HITCH_WORKFLOW_MAINTENANCE_SCHEDULER"

_scheduler = server_lifecycle.SchedulerHandle(
    thread_name="hitch-workflow-maintenance",
    tick_interval_seconds=_WORKFLOW_MAINTENANCE_INTERVAL_SECONDS,
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
    # One-shot at startup: app-servers left behind by the previous server
    # process (pooled servers checked out across a restart, workers killed
    # before their cgroup reap) each hold a CODEX_HOME state-DB connection
    # and surface as "database is locked" on the first turns after boot.
    # Runs before this process warms its own pool, so only ownerless
    # servers can match.
    try:
        reconciliation.reap_orphaned_app_servers()
    except Exception:
        logger.exception("failed to reap orphaned codex app-servers at startup")
    finally:
        close_old_connections()
    stop = threading.Event()
    start = time.monotonic()
    next_disk_cleanup_at = start + _DISK_USAGE_CLEANUP_INTERVAL_SECONDS
    next_row_retention_at = start + _ROW_RETENTION_STARTUP_DELAY_SECONDS
    while True:
        _run_workflow_maintenance_scheduler_tick()
        next_disk_cleanup_at = _run_due_disk_usage_cleanup(
            next_due_at=next_disk_cleanup_at
        )
        next_row_retention_at = _run_due_row_retention(
            next_due_at=next_row_retention_at
        )
        stop.wait(_WORKFLOW_MAINTENANCE_INTERVAL_SECONDS)


def _run_workflow_maintenance_scheduler_tick() -> None:
    _scheduler.run_tick(_workflow_maintenance_tick)


def _workflow_maintenance_tick() -> None:
    reconciliation.reconcile_dead()


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


def _run_due_row_retention(
    *,
    next_due_at: float,
    now: float | None = None,
) -> float:
    """Reap stale rate-limit debounce rows when the daily tick is due."""
    current = time.monotonic() if now is None else now
    if current < next_due_at:
        return next_due_at

    close_old_connections()
    try:
        result = retention.run_retention_sweep()
        if result.throttles_deleted:
            logger.info(
                "retention removed %s throttle row(s)", result.throttles_deleted
            )
    except Exception:
        logger.exception("failed to run scheduled retention sweep")
    finally:
        close_old_connections()
    return current + _ROW_RETENTION_INTERVAL_SECONDS
