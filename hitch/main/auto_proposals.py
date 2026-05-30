"""Periodic runner for auto-proposal autonomous goals."""

from __future__ import annotations

import logging
import os
import sys
import threading

from django.conf import settings
from django.db import close_old_connections

from hitch.main import codex_pool, system_agents

logger = logging.getLogger(__name__)

_AUTO_PROPOSAL_SCHEDULER_INTERVAL_SECONDS = 5 * 60
_SCHEDULER_ENV = "HITCH_AUTO_PROPOSAL_SCHEDULER"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})

_scheduler_lock = threading.Lock()
_scheduler_started = False


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
        system_agents.maybe_advance_pr_monitors()
        started = system_agents.maybe_start_auto_proposal_workflows()
        if started:
            logger.info("started %s auto-proposal workflow(s)", started)
    except Exception:
        logger.exception("failed to run auto-proposal scheduler tick")
    finally:
        close_old_connections()
