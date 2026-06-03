"""Periodic runner for auto-proposal autonomous goals."""

from __future__ import annotations

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
_SCHEDULER_ENV = "HITCH_AUTO_PROPOSAL_SCHEDULER"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})

_scheduler_lock = threading.Lock()
_scheduler_started = False


class SessionStateRefreshResult(NamedTuple):
    synced: int
    failed: bool
    pr_stages_refreshed: int


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
    while True:
        _run_auto_proposal_scheduler_tick()
        stop.wait(_AUTO_PROPOSAL_SCHEDULER_INTERVAL_SECONDS)


def _run_auto_proposal_scheduler_tick() -> None:
    close_old_connections()
    try:
        codex_pool.reconcile_dead()
        _refresh_unarchived_session_state_best_effort()
        started = system_agents.maybe_start_auto_proposal_workflows()
        if started:
            logger.info("started %s auto-proposal workflow(s)", started)
    except Exception:
        logger.exception("failed to run auto-proposal scheduler tick")
    finally:
        close_old_connections()


def _refresh_unarchived_session_state_best_effort() -> SessionStateRefreshResult | None:
    try:
        return refresh_unarchived_session_state()
    except Exception:
        logger.exception("failed to refresh unarchived session state")
        return None


def refresh_unarchived_session_state() -> SessionStateRefreshResult:
    """Refresh active Codex session metadata and GitHub-derived PR stages."""
    codex_synced = 0
    codex_failed = False
    try:
        config = codex_pool.app_server_config()
        projects = list(Project.objects.all())
        with codex_pool.open_codex(lambda: Codex(config=config)) as codex:
            codex_result = session_index.refresh_from_codex(
                codex,
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
    pr_stages_refreshed = system_agents.refresh_unarchived_session_pr_stages()
    return SessionStateRefreshResult(
        synced=codex_synced,
        failed=codex_failed,
        pr_stages_refreshed=pr_stages_refreshed,
    )
