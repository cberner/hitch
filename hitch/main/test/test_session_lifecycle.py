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
