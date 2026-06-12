from datetime import timedelta
from unittest.mock import patch

from django.test import SimpleTestCase, TestCase
from django.utils import timezone

from hitch.main.models import RefreshThrottle
from hitch.main.runtime import retention
from hitch.main.runtime.retention import RetentionResult
from hitch.main.workflows import workflow_maintenance


class StaleThrottleSweepTests(TestCase):
    def test_drops_stale_rows_by_attempted_at(self) -> None:
        # Staleness keys off attempted_at: rate_limit.claim refreshes live rows
        # via QuerySet.update, which never advances auto-now updated_at.
        stale = RefreshThrottle.objects.create(
            key="https://github.com/cberner/hitch/pull/1",
            attempted_at=timezone.now() - timedelta(days=31),
        )
        live = RefreshThrottle.objects.create(
            key="codex-account-rate-limits",
            attempted_at=timezone.now(),
        )
        # An ancient updated_at must not make a freshly-claimed key eligible.
        RefreshThrottle.objects.filter(pk=live.pk).update(
            updated_at=timezone.now() - timedelta(days=90)
        )

        result = retention.run_retention_sweep()

        self.assertEqual(result.throttles_deleted, 1)
        self.assertFalse(RefreshThrottle.objects.filter(pk=stale.pk).exists())
        self.assertTrue(RefreshThrottle.objects.filter(pk=live.pk).exists())

    def test_no_stale_rows_is_a_noop(self) -> None:
        RefreshThrottle.objects.create(
            key="fresh", attempted_at=timezone.now()
        )
        self.assertEqual(retention.run_retention_sweep().throttles_deleted, 0)


class RetentionTickTests(SimpleTestCase):
    # The sweep itself is covered above against the real DB; here we only assert
    # the scheduler gate, so mock it out and stay a SimpleTestCase -- that keeps
    # the close_old_connections() calls in _run_due_row_retention away from a
    # TestCase's wrapping atomic transaction.
    def test_runs_only_when_due_and_reschedules(self) -> None:
        with patch.object(
            retention,
            "run_retention_sweep",
            return_value=RetentionResult(throttles_deleted=3),
        ) as sweep:
            unchanged = workflow_maintenance._run_due_row_retention(
                next_due_at=100.0, now=50.0
            )
            self.assertEqual(unchanged, 100.0)
            sweep.assert_not_called()

            rescheduled = workflow_maintenance._run_due_row_retention(
                next_due_at=100.0, now=150.0
            )
            self.assertEqual(
                rescheduled,
                150.0 + workflow_maintenance._ROW_RETENTION_INTERVAL_SECONDS,
            )
            sweep.assert_called_once_with()
