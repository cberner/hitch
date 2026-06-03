"""Periodic runner for auto-proposal autonomous goals."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading
from typing import NamedTuple

from django.conf import settings
from django.db import close_old_connections
from openai_codex import Codex

from hitch.main import codex_pool, session_index, system_agents
from hitch.main.models import Project

logger = logging.getLogger(__name__)

_AUTO_PROPOSAL_SCHEDULER_INTERVAL_SECONDS = 60
# Cap GitHub-backed PR-stage refreshes per tick: each due session shells out to
# `gh pr view` (seconds each), so an unbounded sweep would let one tick run for
# minutes and stall the rest of the scheduler. Leftover rows converge on later
# 60s ticks, matching the workflow-maintenance scheduler's PR-stage cap.
_PR_STAGE_REFRESH_LIMIT_PER_TICK = 5
_SCHEDULER_ENV = "HITCH_AUTO_PROPOSAL_SCHEDULER"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})

_scheduler_lock = threading.Lock()
_scheduler_started = False


class SessionStateRefreshResult(NamedTuple):
    synced: int
    failed: bool
    pr_stages_refreshed: int


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
            self._codex = codex_pool.start_codex(codex_pool.app_server_config())
        return self._codex

    def reset(self) -> None:
        codex, self._codex = self._codex, None
        if codex is not None:
            with contextlib.suppress(Exception):
                codex.close()


def start_auto_proposal_scheduler() -> bool:
    """Start the in-process auto-proposal scheduler when enabled."""
    global _scheduler_started
    if not _auto_proposal_scheduler_enabled():
        return False
    with _scheduler_lock:
        if _scheduler_started:
            return False
        _scheduler_started = True
        threading.Thread(
            target=_auto_proposal_scheduler_loop,
            name="hitch-auto-proposals",
            daemon=True,
        ).start()
        return True


def _auto_proposal_scheduler_enabled() -> bool:
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
    if not argv or argv[0] != "runserver":
        return False
    return os.environ.get("RUN_MAIN") == "true" or "--noreload" in argv


def _auto_proposal_scheduler_loop() -> None:
    stop = threading.Event()
    # Reused across ticks so the periodic state refresh doesn't spawn (and
    # re-init the Codex state DB for) a fresh app-server every 60 seconds.
    scheduler_codex = _SchedulerCodex()
    try:
        while True:
            _run_auto_proposal_scheduler_tick(scheduler_codex)
            stop.wait(_AUTO_PROPOSAL_SCHEDULER_INTERVAL_SECONDS)
    finally:
        scheduler_codex.reset()


def _run_auto_proposal_scheduler_tick(
    scheduler_codex: _SchedulerCodex | None = None,
) -> None:
    close_old_connections()
    try:
        codex_pool.reconcile_dead()
        _refresh_unarchived_session_state_best_effort(scheduler_codex)
        started = system_agents.maybe_start_auto_proposal_workflows()
        if started:
            logger.info("started %s auto-proposal workflow(s)", started)
    except Exception:
        logger.exception("failed to run auto-proposal scheduler tick")
    finally:
        close_old_connections()


def _refresh_unarchived_session_state_best_effort(
    scheduler_codex: _SchedulerCodex | None = None,
) -> SessionStateRefreshResult | None:
    try:
        codex = scheduler_codex.get() if scheduler_codex is not None else None
        result = refresh_unarchived_session_state(codex)
        # A failed metadata refresh may mean the reused app-server died; drop it
        # so the next tick reconnects rather than reusing a dead transport.
        if scheduler_codex is not None and result.failed:
            scheduler_codex.reset()
        return result
    except Exception:
        if scheduler_codex is not None:
            scheduler_codex.reset()
        logger.exception("failed to refresh unarchived session state")
        return None


def refresh_unarchived_session_state(
    codex: Codex | None = None,
) -> SessionStateRefreshResult:
    """Refresh active Codex session metadata and GitHub-derived PR stages.

    ``codex`` lets the scheduler pass a long-lived app-server it reuses across
    ticks; when omitted a short-lived one is opened just for this call.
    """
    codex_synced = 0
    codex_failed = False
    try:
        projects = list(Project.objects.all())
        if codex is not None:
            codex_result = session_index.refresh_from_codex(
                codex,
                projects=projects,
                include_active=True,
                include_archived=False,
                max_pages=None,
            )
        else:
            config = codex_pool.app_server_config()
            with codex_pool.open_codex(lambda: Codex(config=config)) as opened:
                codex_result = session_index.refresh_from_codex(
                    opened,
                    projects=projects,
                    include_active=True,
                    include_archived=False,
                    max_pages=None,
                )
        codex_synced = codex_result.synced
        codex_failed = codex_result.failed
    except Exception:
        codex_failed = True
        logger.exception("failed to refresh active Codex session metadata")
    pr_stages_refreshed = system_agents.refresh_unarchived_session_pr_stages(
        limit=_PR_STAGE_REFRESH_LIMIT_PER_TICK
    )
    return SessionStateRefreshResult(
        synced=codex_synced,
        failed=codex_failed,
        pr_stages_refreshed=pr_stages_refreshed,
    )
