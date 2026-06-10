"""Shared ownership rules for in-process background server work."""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Callable

from django.conf import settings

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

class SchedulerHandle:
    """Once-only starter for an in-process background scheduler thread.

    Owns the started flag and its lock so every scheduler (workflow
    maintenance, auto proposals, codex pool keepalive) shares one start
    protocol instead of each module hand-rolling the same guard.
    """

    def __init__(self, *, thread_name: str) -> None:
        self._lock = threading.Lock()
        self._started = False
        self._thread_name = thread_name

    def start(self, target: Callable[[], None]) -> bool:
        """Start ``target`` on a daemon thread once; False if already started."""
        with self._lock:
            if self._started:
                return False
            self._started = True
            threading.Thread(
                target=target,
                name=self._thread_name,
                daemon=True,
            ).start()
            return True

    def reset_for_tests(self) -> None:
        with self._lock:
            self._started = False
