"""Periodic runner for auto-proposal autonomous goals."""

from __future__ import annotations

import contextlib
import logging
import threading
from typing import NamedTuple

from openai_codex import Codex

from hitch.main.models import Project
from hitch.main.runtime import app_server_pool, codex_pool, reconciliation, server_lifecycle
from hitch.main.sessions import session_index
from hitch.main.workflows import autonomous_goals, pr_qa

logger = logging.getLogger(__name__)

_AUTO_PROPOSAL_SCHEDULER_INTERVAL_SECONDS = 60
# Cap GitHub-backed PR-stage refreshes per tick: each due session shells out to
# `gh pr view` (seconds each), so an unbounded sweep would let one tick run for
# minutes and stall the rest of the scheduler. Leftover rows converge on later
# 60s ticks, matching the workflow-maintenance scheduler's PR-stage cap.
_PR_STAGE_REFRESH_LIMIT_PER_TICK = 5
# Pages of the active session list refreshed per tick. The scheduler resumes
# from its own cursor each tick, so the whole list is still covered
# incrementally -- this only bounds the per-tick work so a busy instance with
# many active sessions does not rescan all of them every minute.
_SESSION_STATE_REFRESH_MAX_PAGES = 5
_SCHEDULER_ENV = "HITCH_AUTO_PROPOSAL_SCHEDULER"

_scheduler = server_lifecycle.SchedulerHandle(
    thread_name="hitch-auto-proposals",
    tick_interval_seconds=_AUTO_PROPOSAL_SCHEDULER_INTERVAL_SECONDS,
)


class SessionStateRefreshResult(NamedTuple):
    synced: int
    failed: bool
    pr_stages_refreshed: int
    # Cursor to resume the incremental active-index scan from on the next tick;
    # empty means the list was fully traversed (restart from the front).
    active_next_cursor: str = ""


class _SchedulerCodex:
    """Holds one long-lived Codex app-server reused across scheduler ticks.

    Owned by the single scheduler thread, so it needs no locking. Reusing one
    app-server means the Codex state DB under CODEX_HOME is initialized once for
    the process lifetime instead of on every 60s tick -- those per-tick inits
    were what raced user-initiated app-server startups and surfaced as
    "database is locked". ``reset`` drops a dead/failed app-server so the next
    ``get`` reconnects.
    """

    def __init__(self) -> None:
        self._codex: Codex | None = None

    def get(self) -> Codex:
        if self._codex is None:
            self._codex = app_server_pool.start_codex(codex_pool.app_server_config())
        return self._codex

    def reset(self) -> None:
        codex, self._codex = self._codex, None
        if codex is not None:
            with contextlib.suppress(Exception):
                codex.close()


def start_auto_proposal_scheduler() -> bool:
    """Start the in-process auto-proposal scheduler when enabled."""
    if not _auto_proposal_scheduler_enabled():
        return False
    return _scheduler.start(_auto_proposal_scheduler_loop)


def _auto_proposal_scheduler_enabled() -> bool:
    return server_lifecycle.background_work_enabled(env_var=_SCHEDULER_ENV)


def _auto_proposal_scheduler_loop() -> None:
    stop = threading.Event()
    # Reused across ticks so the periodic state refresh doesn't spawn (and
    # re-init the Codex state DB for) a fresh app-server every 60 seconds.
    scheduler_codex = _SchedulerCodex()
    # Resume point for the incremental active-index scan; carried across ticks
    # so the scan walks the whole list a bounded window at a time rather than
    # rescanning everything each tick.
    active_cursor = ""
    try:
        while True:
            active_cursor = _run_auto_proposal_scheduler_tick(
                scheduler_codex, start_cursor=active_cursor
            )
            stop.wait(_AUTO_PROPOSAL_SCHEDULER_INTERVAL_SECONDS)
    finally:
        scheduler_codex.reset()


def _run_auto_proposal_scheduler_tick(
    scheduler_codex: _SchedulerCodex | None = None,
    *,
    start_cursor: str = "",
) -> str:
    """Run one scheduler tick. Returns the active-index cursor for the next tick."""

    # Captured by the closure so a failure in a later phase (e.g. the
    # workflow starts) keeps the cursor the refresh already advanced --
    # run_tick returns None on error, and resetting the scan to the front on
    # every such tick would rescan only the first window forever.
    next_cursor = ""

    def _tick() -> None:
        nonlocal next_cursor
        reconciliation.reconcile_dead()
        next_cursor = _refresh_unarchived_session_state_best_effort(
            scheduler_codex, start_cursor=start_cursor
        )
        started = autonomous_goals.maybe_start_auto_proposal_workflows()
        if started:
            logger.info("started %s auto-proposal workflow(s)", started)

    _scheduler.run_tick(_tick)
    return next_cursor


def _refresh_unarchived_session_state_best_effort(
    scheduler_codex: _SchedulerCodex | None = None,
    *,
    start_cursor: str = "",
) -> str:
    """Refresh a window of active session state. Returns the next-tick cursor.

    A failed refresh (or a reused app-server that died) resets the cursor to the
    front so the next tick starts a clean pass rather than resuming from a stale
    position against a freshly reconnected app-server.
    """
    try:
        codex = scheduler_codex.get() if scheduler_codex is not None else None
        result = refresh_unarchived_session_state(codex, start_cursor=start_cursor)
        # A failed metadata refresh may mean the reused app-server died; drop it
        # so the next tick reconnects rather than reusing a dead transport.
        if scheduler_codex is not None and result.failed:
            scheduler_codex.reset()
            return ""
        return result.active_next_cursor
    except Exception:
        if scheduler_codex is not None:
            scheduler_codex.reset()
        logger.exception("failed to refresh unarchived session state")
        return ""


def refresh_unarchived_session_state(
    codex: Codex | None = None,
    *,
    start_cursor: str = "",
    max_pages: int = _SESSION_STATE_REFRESH_MAX_PAGES,
) -> SessionStateRefreshResult:
    """Refresh active Codex session metadata and GitHub-derived PR stages.

    ``codex`` lets the scheduler pass a long-lived app-server it reuses across
    ticks; when omitted a short-lived one is opened just for this call. The
    active-index scan is bounded to ``max_pages`` per call and resumes from
    ``start_cursor``, so successive ticks cover the whole list incrementally
    instead of rescanning it every tick.
    """
    codex_synced = 0
    codex_failed = False
    active_next_cursor = ""
    try:
        projects = list(Project.objects.all())
        if codex is not None:
            window = session_index.refresh_active_window(
                codex,
                projects=projects,
                start_cursor=start_cursor,
                max_pages=max_pages,
            )
        else:
            config = codex_pool.app_server_config()
            with app_server_pool.open_codex(lambda: Codex(config=config)) as opened:
                window = session_index.refresh_active_window(
                    opened,
                    projects=projects,
                    start_cursor=start_cursor,
                    max_pages=max_pages,
                )
        codex_synced = window.synced
        codex_failed = window.failed
        active_next_cursor = window.next_cursor
    except Exception:
        codex_failed = True
        logger.exception("failed to refresh active Codex session metadata")
    pr_stages_refreshed = pr_qa.refresh_unarchived_session_pr_stages(
        limit=_PR_STAGE_REFRESH_LIMIT_PER_TICK
    )
    return SessionStateRefreshResult(
        synced=codex_synced,
        failed=codex_failed,
        pr_stages_refreshed=pr_stages_refreshed,
        active_next_cursor=active_next_cursor,
    )
