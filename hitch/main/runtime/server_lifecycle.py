"""Shared ownership rules for in-process background server work."""

from __future__ import annotations

import logging
import os
import sys
import threading
from collections.abc import Callable
from typing import TypeVar

from django.conf import settings
from django.db import close_old_connections

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

    Environment overrides may disable work in a serving process, but may never
    turn a management command into an owner. Migrations import app configs
    before the schema is current, so starting background database work there
    can race or break the migration itself.
    """
    if getattr(settings, "TESTING", False):
        return False

    if not is_single_serving_process(
        include_wsgi_server_commands=include_wsgi_server_commands
    ):
        return False

    if env_var is not None:
        configured = _configured_bool(env_var)
        if configured is not None:
            return configured

    return True


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

class SchedulerHandle:
    """Start a scheduler once and wrap ticks in DB cleanup and exception logging."""

    def __init__(self, *, thread_name: str) -> None:
        self._lock = threading.Lock()
        self._started = False
        self._thread_name = thread_name

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
            # Only mark successful starts so a failed OS thread start is retryable.
            thread.start()
            self._started = True
            return True

    def run_tick(self, tick: Callable[[], _TickResultT]) -> _TickResultT | None:
        """Run one tick, logging failures without killing the scheduler thread."""
        close_old_connections()
        try:
            return tick()
        except Exception:
            logger.exception("scheduler %s tick failed", self._thread_name)
            return None
        finally:
            close_old_connections()

    def reset_for_tests(self) -> None:
        with self._lock:
            self._started = False
