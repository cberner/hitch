from unittest.mock import MagicMock, patch

from django.test import TestCase

from hitch.main.sessions import lifecycle


class SessionLifecycleLockTests(TestCase):
    def test_nonblocking_claim_reports_busy_lock(self) -> None:
        with lifecycle.hold("thread-1") as acquired:
            self.assertTrue(acquired)
            with lifecycle.hold("thread-1", blocking=False) as competing:
                self.assertFalse(competing)

        with lifecycle.hold("thread-1", blocking=False) as reacquired:
            self.assertTrue(reacquired)

    @patch("hitch.main.sessions.lifecycle.os.close")
    @patch("hitch.main.sessions.lifecycle.fcntl.flock")
    @patch("hitch.main.sessions.lifecycle.os.open", return_value=123)
    def test_acquire_closes_descriptor_when_flock_fails(
        self, _open: MagicMock, flock: MagicMock, close: MagicMock
    ) -> None:
        flock.side_effect = OSError("flock failed")

        with self.assertRaisesRegex(OSError, "flock failed"):
            lifecycle._acquire("thread-1", blocking=True)

        close.assert_called_once_with(123)
