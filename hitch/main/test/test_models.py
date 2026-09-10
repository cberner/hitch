from __future__ import annotations

from django.test import SimpleTestCase

from hitch.main.models import CodexInstance, SystemWorkflow


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


class SystemWorkflowActiveTests(SimpleTestCase):
    """Pin the single source of truth for "this workflow still pins its worktree"."""

    def test_only_running_is_active(self) -> None:
        self.assertEqual(SystemWorkflow.ACTIVE_STATUSES, (SystemWorkflow.STATUS_RUNNING,))

    def test_is_active_covers_every_status(self) -> None:
        expected = {
            SystemWorkflow.STATUS_RUNNING: True,
            SystemWorkflow.STATUS_BLOCKED: False,
            SystemWorkflow.STATUS_COMPLETED: False,
            SystemWorkflow.STATUS_FAILED: False,
            SystemWorkflow.STATUS_MAX_ITERATIONS_REACHED: False,
        }
        # A new workflow status must explicitly opt in or out of "active".
        self.assertEqual(
            {status for status, _ in SystemWorkflow.STATUS_CHOICES},
            set(expected),
        )
        for status, is_active in expected.items():
            self.assertEqual(
                SystemWorkflow(status=status).is_active,
                is_active,
                msg=status,
            )
