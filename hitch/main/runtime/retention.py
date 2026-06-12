"""Age-based cleanup of stale rate-limit debounce rows.

A deliberately minimal, conservative backstop: ``RefreshThrottle`` rows
accumulate one per distinct debounced resource (mostly per-PR URLs) and
nothing else prunes them, yet an old key is safe to drop without reasoning
about what any UI reader or session resume needs. This runs as a daily sweep
on the workflow-maintenance scheduler.

Reaping terminal ``SystemWorkflow`` / ``CodexInstance`` rows and their event
files is intentionally out of scope -- those are read back by the
autonomous-goals run display, PR-stage rendering, and session resume in ways
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
    look ancient on an actively-claimed key. Stale keys are mostly URLs of
    long-merged PRs; live keys (the account rate-limit endpoint, active PRs)
    are re-touched constantly and survive. Batched and capped, and the
    staleness predicate is re-asserted on the delete so a key a concurrent
    ``claim`` refreshed between the id select and the delete is not erased.
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
