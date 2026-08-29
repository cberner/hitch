from django.db import OperationalError
from django.test import SimpleTestCase

from hitch.main.runtime.db import is_database_locked_error, run_ignoring_database_locks


class IsDatabaseLockedErrorTests(SimpleTestCase):
    def test_matches_regardless_of_case(self) -> None:
        self.assertTrue(
            is_database_locked_error(OperationalError("Database Is Locked"))
        )


class RunIgnoringDatabaseLocksTests(SimpleTestCase):
    def test_reraises_other_operational_errors(self) -> None:
        def boom() -> int:
            raise OperationalError("no such table")

        with self.assertRaises(OperationalError):
            run_ignoring_database_locks(boom, description="probe")
