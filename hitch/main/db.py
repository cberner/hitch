"""SQLite concurrency helpers.

The database runs in WAL mode with ``transaction_mode=IMMEDIATE`` and a 60s
``busy_timeout`` (see ``hitch.settings.common``). A single global write lock
serializes all writers; ``busy_timeout`` makes a contended writer wait for the
lock rather than fail immediately, but a writer can still raise
``OperationalError: database is locked`` if the lock stays held past the
timeout under sustained contention.

For writes that are safe to skip because a later poll/render/reconcile tick
repeats them -- derived-stage cache refreshes, background reconciliation
sweeps, SSE heartbeat housekeeping -- swallowing a transient lock keeps a
single contended tick from surfacing a 500 to the user or tearing down a live
SSE stream. Writes that must land (status transitions, approval/input
decisions) are intentionally *not* routed through here: they rely on
``busy_timeout`` to wait out the contention instead.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

from django.db import OperationalError

logger = logging.getLogger(__name__)

T = TypeVar("T")


def is_database_locked_error(exc: BaseException) -> bool:
    """True when ``exc`` is SQLite reporting a busy/locked database.

    Matches both ``database is locked`` (file-level busy) and ``database table
    is locked`` (a table held by a cursor in the same connection).
    """
    message = " ".join(str(arg) for arg in exc.args).lower()
    return "database is locked" in message or "database table is locked" in message


def run_ignoring_database_locks(
    operation: Callable[[], T], *, description: str
) -> T | None:
    """Run ``operation``; swallow a transient "database is locked" error.

    Returns the operation's result, or ``None`` when it was skipped because the
    database was locked. Only use for writes that are safe to skip because a
    later tick repeats them -- a dropped write here is silently lost.
    """
    try:
        return operation()
    except OperationalError as exc:
        if not is_database_locked_error(exc):
            raise
        logger.warning("skipping %s because database is locked", description)
        return None
