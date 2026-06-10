"""Central, process-global debounce for external pings (GitHub, Codex/OpenAI).

Hitch pings external services from several independent code paths -- synchronous
web renders, the auto-proposal scheduler, the workflow-maintenance scheduler, and
detached ``codex_worker`` subprocesses. Module-level caches are per-process, so
they cannot coordinate across gunicorn workers or detached subprocesses.

This module backs the debounce with a small DB table (``RefreshThrottle``) keyed
by an opaque string ("the same thing"), giving a single global floor on how often
any external resource is hit. ``claim`` is the primitive: it atomically records an
attempt and reports whether the caller won the right to hit the resource now, so
at most one caller per ``min_interval`` per key shells out across the whole app.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from django.db import transaction
from django.utils import timezone

from hitch.main.models import RefreshThrottle
from hitch.main.runtime.db import run_ignoring_database_locks

# Default floor on how often any one resource may be pinged across the whole app.
DEFAULT_MIN_INTERVAL = timedelta(minutes=2)


def claim(
    key: str,
    *,
    min_interval: timedelta = DEFAULT_MIN_INTERVAL,
    now: datetime | None = None,
) -> bool:
    """Atomically record an attempt against ``key`` and report whether the caller
    won the right to hit the resource now.

    Returns ``True`` for at most one caller per ``min_interval`` per key across
    the whole app; concurrent or too-soon callers get ``False`` and should serve
    cached state instead. A transient SQLite lock is treated as "not claimed" so
    write contention never triggers an extra external ping.
    """
    moment = now if now is not None else timezone.now()
    threshold = moment - min_interval

    def _claim() -> bool:
        with transaction.atomic():
            # A single UPDATE...WHERE serializes against concurrent writers: the
            # winner flips ``attempted_at`` past the threshold, so a racing
            # claimer matches zero rows and falls through to ``get_or_create``,
            # which returns ``created=False`` for the existing (now-fresh) row.
            won = RefreshThrottle.objects.filter(
                key=key, attempted_at__lte=threshold
            ).update(attempted_at=moment)
            if won:
                return True
            _, created = RefreshThrottle.objects.get_or_create(
                key=key, defaults={"attempted_at": moment}
            )
            return created

    return bool(run_ignoring_database_locks(_claim, description=f"rate-limit claim {key}"))


def due(
    key: str,
    *,
    min_interval: timedelta = DEFAULT_MIN_INTERVAL,
    now: datetime | None = None,
) -> bool:
    """Read-only check: would :func:`claim` succeed right now? Records nothing.

    Useful for deciding whether to surface a "refreshing" indicator without
    committing to a refresh attempt.
    """
    moment = now if now is not None else timezone.now()
    attempted_at = (
        RefreshThrottle.objects.filter(key=key)
        .values_list("attempted_at", flat=True)
        .first()
    )
    if attempted_at is None:
        return True
    return moment - attempted_at >= min_interval


def mark(key: str, *, now: datetime | None = None) -> None:
    """Record an attempt against ``key`` without gating (for always-hit callers)."""
    moment = now if now is not None else timezone.now()
    run_ignoring_database_locks(
        lambda: RefreshThrottle.objects.update_or_create(
            key=key, defaults={"attempted_at": moment}
        ),
        description=f"rate-limit mark {key}",
    )
