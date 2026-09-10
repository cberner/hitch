"""Central, process-global debounce for external pings (GitHub, Codex/OpenAI).

Quota refreshes and worker reconciliation run in web and detached worker
processes. Module-level caches cannot coordinate those processes.

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
