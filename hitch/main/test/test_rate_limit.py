"""Tests for the central, process-global refresh debounce."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import override

from django.test import TestCase
from django.utils import timezone

from hitch.main.models import RefreshThrottle
from hitch.main.runtime import rate_limit


class RateLimitTests(TestCase):
    @override
    def setUp(self) -> None:
        self.now = datetime(2026, 6, 3, 12, 0, 0, tzinfo=timezone.get_current_timezone())

    def test_second_claim_within_window_loses(self) -> None:
        self.assertTrue(rate_limit.claim("k", now=self.now))
        self.assertFalse(
            rate_limit.claim("k", now=self.now + timedelta(seconds=119))
        )
        # The losing claim must not advance the recorded attempt.
        self.assertEqual(RefreshThrottle.objects.get(key="k").attempted_at, self.now)

    def test_claim_succeeds_again_after_window(self) -> None:
        self.assertTrue(rate_limit.claim("k", now=self.now))
        later = self.now + timedelta(minutes=2)
        self.assertTrue(rate_limit.claim("k", now=later))
        self.assertEqual(RefreshThrottle.objects.get(key="k").attempted_at, later)

    def test_only_one_concurrent_claimer_wins_for_same_key(self) -> None:
        # Simulates two independent code paths (e.g. list + detail render, or two
        # sessions on the same PR) racing for the same resource: exactly one wins.
        results = [
            rate_limit.claim("shared", now=self.now),
            rate_limit.claim("shared", now=self.now),
        ]
        self.assertEqual(results.count(True), 1)

    def test_due_does_not_record_an_attempt(self) -> None:
        self.assertTrue(rate_limit.due("k", now=self.now))
        self.assertFalse(RefreshThrottle.objects.filter(key="k").exists())
        # A claim is still available because `due` recorded nothing.
        self.assertTrue(rate_limit.claim("k", now=self.now))
