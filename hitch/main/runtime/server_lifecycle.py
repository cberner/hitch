"""Shared ownership rules for in-process background server work."""

from __future__ import annotations

import logging
import os
import sys
import threading
from collections.abc import Callable
from datetime import datetime
from typing import NamedTuple, TypeVar

from django.conf import settings
from django.db import close_old_connections
from django.utils import timezone

logger = logging.getLogger(__name__)

_TickResultT = TypeVar("_TickResultT")

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_SERVER_PROCESS_COMMANDS = frozenset({"gunicorn", "uvicorn", "daphne", "uwsgi"})


def background_work_enabled(
    *,
    env_var: str | None = None,
    include_wsgi_server_commands: bool = False,
) -> bool:
    """Whether this process owns in-process background work.

    The owner must be a single long-lived serving process. Django's autoreloader
    parent imports the app before it forks the ``RUN_MAIN=true`` child, so
    plain ``runserver`` without either marker is a watcher, not an owner.
    """
    if env_var is not None:
        configured = _configured_bool(env_var)
        if configured is not None:
            return configured

    if getattr(settings, "TESTING", False):
        return False

    return is_single_serving_process(
        include_wsgi_server_commands=include_wsgi_server_commands
    )


def is_single_serving_process(*, include_wsgi_server_commands: bool = False) -> bool:
    argv = sys.argv
    args = argv[1:]
    if args and args[0] == "runserver":
        return os.environ.get("RUN_MAIN") == "true" or "--noreload" in args
    if include_wsgi_server_commands and argv:
        return os.path.basename(argv[0]) in _SERVER_PROCESS_COMMANDS
    return False


def _configured_bool(env_var: str) -> bool | None:
    configured = os.environ.get(env_var)
    if configured is None:
        return None
    normalized = configured.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return None

class SchedulerStatus(NamedTuple):
    """One scheduler's liveness snapshot for the health dashboard."""

    name: str
    started: bool
    started_at: datetime | None
    tick_interval_seconds: float
    tick_count: int
    last_tick_at: datetime | None
    # Whether the most recent tick raised; last_error/last_error_at keep the
    # most recent failure on record even after the scheduler recovers.
    last_tick_errored: bool
    last_error: str
    last_error_at: datetime | None


# Every SchedulerHandle is a module-level singleton created at import, and all
# scheduler modules are imported from apps.ready(), so by the time any request
# renders /health this registry holds the complete set.
_HANDLES: list[SchedulerHandle] = []


def scheduler_statuses() -> list[SchedulerStatus]:
    """Liveness snapshots for every registered scheduler, by name."""
    return sorted(
        (handle.status() for handle in _HANDLES), key=lambda status: status.name
    )


class SchedulerHandle:
    """Once-only starter for an in-process background scheduler thread.

    Owns the started flag and its lock so every scheduler (workflow
    maintenance, auto proposals, codex pool keepalive) shares one start
    protocol instead of each module hand-rolling the same guard. Loops route
    their per-tick work through :meth:`run_tick`, which owns the DB-connection
    bracketing and exception logging and records the heartbeat the health
    dashboard reads -- a silently-dead scheduler thread is otherwise
    indistinguishable from a healthy quiet one.
    """

    def __init__(self, *, thread_name: str, tick_interval_seconds: float = 60) -> None:
        self._lock = threading.Lock()
        self._started = False
        self._thread_name = thread_name
        self._tick_interval_seconds = tick_interval_seconds
        self._started_at: datetime | None = None
        self._tick_count = 0
        self._last_tick_at: datetime | None = None
        self._last_tick_errored = False
        self._last_error = ""
        self._last_error_at: datetime | None = None
        _HANDLES.append(self)

    def start(self, target: Callable[[], None]) -> bool:
        """Start ``target`` on a daemon thread once; False if already started."""
        with self._lock:
            if self._started:
                return False
            thread = threading.Thread(
                target=target,
                name=self._thread_name,
                daemon=True,
            )
            started_at = timezone.now()
            try:
                thread.start()
            except Exception:
                # A failed OS thread start must remain retryable. Marking the
                # handle started here would suppress every later start attempt
                # while health incorrectly reported a live scheduler.
                self._started = False
                self._started_at = None
                raise
            self._started = True
            self._started_at = started_at
            return True

    def run_tick(
        self, tick: Callable[[], _TickResultT]
    ) -> _TickResultT | None:
        """Run one tick with shared bracketing, logging, and heartbeat.

        Returns ``tick``'s result, or ``None`` when it raised (the exception
        is logged and recorded for the health page, never propagated -- one
        bad tick must not kill the scheduler thread).
        """
        close_old_connections()
        result: _TickResultT | None = None
        errored = False
        try:
            result = tick()
        except Exception as exc:
            errored = True
            with self._lock:
                self._last_error = repr(exc)
                self._last_error_at = timezone.now()
            logger.exception("scheduler %s tick failed", self._thread_name)
        finally:
            with self._lock:
                self._tick_count += 1
                self._last_tick_at = timezone.now()
                self._last_tick_errored = errored
            close_old_connections()
        return result

    def status(self) -> SchedulerStatus:
        with self._lock:
            return SchedulerStatus(
                name=self._thread_name,
                started=self._started,
                started_at=self._started_at,
                tick_interval_seconds=self._tick_interval_seconds,
                tick_count=self._tick_count,
                last_tick_at=self._last_tick_at,
                last_tick_errored=self._last_tick_errored,
                last_error=self._last_error,
                last_error_at=self._last_error_at,
            )

    def reset_for_tests(self) -> None:
        with self._lock:
            self._started = False
            self._started_at = None
            self._tick_count = 0
            self._last_tick_at = None
            self._last_tick_errored = False
            self._last_error = ""
            self._last_error_at = None
