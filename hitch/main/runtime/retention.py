"""Age-based cleanup of stale rate-limit debounce rows.

A deliberately minimal, conservative backstop: ``RefreshThrottle`` rows
accumulate per debounced quota or reconciliation resource. An old key is safe
to drop without affecting session history. The runtime maintenance scheduler
runs this sweep daily.

Reaping terminal ``SystemWorkflow`` / ``CodexInstance`` rows and their event
files is intentionally out of scope -- those are read back by the
historical system-session logs and session resume in ways
that make age alone an unsafe deletion signal -- and is left to a separate,
more carefully scoped change. Disk-pressure cleanup remains the backstop for
worktrees.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import NamedTuple

from django.utils import timezone

from hitch.main.models import RefreshThrottle

logger = logging.getLogger(__name__)

# How long an untouched throttle row is kept.
RETENTION_AGE = timedelta(days=30)

# Rows handled per DELETE, and the cap on batches per sweep, so a first run
# against a large backlog cannot hold the throttle write lock past the
# per-sweep budget; the remainder converges on later daily sweeps.
_BATCH_SIZE = 500
_MAX_BATCHES_PER_SWEEP = 10


class RetentionResult(NamedTuple):
    throttles_deleted: int


def run_retention_sweep(*, now: datetime | None = None) -> RetentionResult:
    """Delete stale throttle rows older than the cutoff."""
    cutoff = (now or timezone.now()) - RETENTION_AGE
    return RetentionResult(throttles_deleted=_delete_expired_refresh_throttles(cutoff))


def _delete_expired_refresh_throttles(cutoff: datetime) -> int:
    """Drop debounce rows whose key has not been pinged since ``cutoff``.

    Staleness is judged by ``attempted_at`` (indexed), not the auto-now
    ``updated_at``: ``rate_limit.claim`` refreshes live rows with a
    ``QuerySet.update`` that does not advance ``updated_at``, so that field can
    look ancient on an actively-claimed key. Old per-PR keys can expire;
    live quota and reconciliation keys are re-touched and survive. The delete
    rechecks staleness so a concurrent claim cannot lose its refreshed row.
    """
    deleted = 0
    for _ in range(_MAX_BATCHES_PER_SWEEP):
        expired_ids = list(
            RefreshThrottle.objects.filter(attempted_at__lt=cutoff).values_list(
                "pk", flat=True
            )[:_BATCH_SIZE]
        )
        if not expired_ids:
            break
        removed, _by_model = RefreshThrottle.objects.filter(
            pk__in=expired_ids, attempted_at__lt=cutoff
        ).delete()
        deleted += removed
        if len(expired_ids) < _BATCH_SIZE:
            break
    return deleted
