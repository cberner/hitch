from django.db import OperationalError
from django.test import SimpleTestCase

from hitch.main.db import is_database_locked_error, run_ignoring_database_locks


class IsDatabaseLockedErrorTests(SimpleTestCase):
    def test_matches_file_and_table_lock_messages(self) -> None:
        self.assertTrue(
            is_database_locked_error(OperationalError("database is locked"))
        )
        self.assertTrue(
            is_database_locked_error(OperationalError("database table is locked"))
        )

    def test_matches_regardless_of_case(self) -> None:
        self.assertTrue(
            is_database_locked_error(OperationalError("Database Is Locked"))
        )

    def test_ignores_unrelated_operational_errors(self) -> None:
        self.assertFalse(
            is_database_locked_error(OperationalError("no such table: main_foo"))
        )


class RunIgnoringDatabaseLocksTests(SimpleTestCase):
    def test_returns_operation_result_on_success(self) -> None:
        self.assertEqual(
            run_ignoring_database_locks(lambda: 42, description="probe"), 42
        )

    def test_swallows_locked_error_and_returns_none(self) -> None:
        def locked() -> int:
            raise OperationalError("database is locked")

        with self.assertLogs("hitch.main.db", level="WARNING") as logs:
            result = run_ignoring_database_locks(locked, description="probe write")

        self.assertIsNone(result)
        self.assertIn("probe write", "".join(logs.output))
        self.assertIn("database is locked", "".join(logs.output))

    def test_reraises_other_operational_errors(self) -> None:
        def boom() -> int:
            raise OperationalError("no such table")

        with self.assertRaises(OperationalError):
            run_ignoring_database_locks(boom, description="probe")

    def test_reraises_non_operational_errors(self) -> None:
        def boom() -> int:
            raise ValueError("nope")

        with self.assertRaises(ValueError):
            run_ignoring_database_locks(boom, description="probe")
