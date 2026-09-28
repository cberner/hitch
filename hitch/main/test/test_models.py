from __future__ import annotations

from django.test import SimpleTestCase

from hitch.main.models import CodexInstance


class CodexInstanceActiveTests(SimpleTestCase):
    """Pin the single source of truth for "this worker is live"."""

    def test_active_statuses_are_starting_and_running(self) -> None:
        self.assertEqual(
            CodexInstance.ACTIVE_STATUSES,
            (CodexInstance.STATUS_STARTING, CodexInstance.STATUS_RUNNING),
        )

    def test_is_active_covers_every_status(self) -> None:
        expected = {
            CodexInstance.STATUS_STARTING: True,
            CodexInstance.STATUS_RUNNING: True,
            CodexInstance.STATUS_COMPLETED: False,
            CodexInstance.STATUS_FAILED: False,
        }
        # Guard against a status being added without deciding its activeness.
        self.assertEqual(
            {status for status, _ in CodexInstance.STATUS_CHOICES},
            set(expected),
        )
        for status, is_active in expected.items():
            self.assertEqual(
                CodexInstance(pid=0, status=status).is_active,
                is_active,
                msg=status,
            )
