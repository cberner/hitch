from collections.abc import Callable
from typing import Any, cast
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase
from openai_codex import TransportClosedError

from hitch.main import codex_pool

_LOCKED = "app-server closed stdout. stderr_tail=... (code: 5) database is locked"


class _FakeCodex:
    """Minimal stand-in matching ``Codex``'s ``__enter__`` (returns self) and
    ``__exit__`` (closes) so ``open_codex`` can be exercised without a real
    app-server subprocess."""

    def __init__(self) -> None:
        self.closed = False

    def __enter__(self) -> "_FakeCodex":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.closed = True


class StartCodexWithRetryTests(SimpleTestCase):
    def test_returns_immediately_on_clean_start(self) -> None:
        sentinel = object()
        factory = MagicMock(return_value=sentinel)
        with patch("hitch.main.codex_pool.time.sleep") as mock_sleep:
            result = codex_pool._start_codex_with_retry(factory)

        self.assertIs(result, sentinel)
        self.assertEqual(factory.call_count, 1)
        mock_sleep.assert_not_called()

    def test_retries_locked_state_db_then_succeeds(self) -> None:
        sentinel = object()
        factory = MagicMock(
            side_effect=[
                TransportClosedError(_LOCKED),
                TransportClosedError(_LOCKED),
                sentinel,
            ]
        )
        with patch("hitch.main.codex_pool.time.sleep") as mock_sleep:
            result = codex_pool._start_codex_with_retry(factory)

        self.assertIs(result, sentinel)
        self.assertEqual(factory.call_count, 3)
        # Backoff slept once per failed attempt, never after the success.
        self.assertEqual(mock_sleep.call_count, 2)

    def test_non_locked_transport_error_is_not_retried(self) -> None:
        boom = TransportClosedError("app-server closed stdout. stderr_tail=segfault")
        factory = MagicMock(side_effect=boom)
        with (
            patch("hitch.main.codex_pool.time.sleep"),
            self.assertRaises(TransportClosedError),
        ):
            codex_pool._start_codex_with_retry(factory)

        self.assertEqual(factory.call_count, 1)

    def test_reraises_after_exhausting_attempts(self) -> None:
        factory = MagicMock(side_effect=TransportClosedError(_LOCKED))
        with (
            patch("hitch.main.codex_pool.time.sleep"),
            self.assertRaises(TransportClosedError),
        ):
            codex_pool._start_codex_with_retry(factory)

        self.assertEqual(factory.call_count, codex_pool._APPSERVER_START_MAX_ATTEMPTS)


class OpenCodexTests(SimpleTestCase):
    def test_yields_entered_codex_and_closes_on_exit(self) -> None:
        codex = _FakeCodex()
        factory = cast("Callable[[], Any]", lambda: codex)
        with codex_pool.open_codex(factory) as opened:
            self.assertIs(opened, codex)
            self.assertFalse(codex.closed)
        self.assertTrue(codex.closed)


class RunCodexOpWithRetryTests(SimpleTestCase):
    def test_returns_operation_result_on_clean_run(self) -> None:
        codex = _FakeCodex()
        factory = MagicMock(return_value=codex)
        operation = MagicMock(return_value="ok")
        with patch("hitch.main.codex_pool.time.sleep") as mock_sleep:
            result = codex_pool.run_codex_op_with_retry(factory, operation)

        self.assertEqual(result, "ok")
        self.assertEqual(factory.call_count, 1)
        operation.assert_called_once_with(codex)
        # open_codex closes the session even on the success path.
        self.assertTrue(codex.closed)
        mock_sleep.assert_not_called()

    def test_retries_locked_operation_with_fresh_server(self) -> None:
        servers: list[_FakeCodex] = []

        def factory() -> _FakeCodex:
            server = _FakeCodex()
            servers.append(server)
            return server

        attempts = {"n": 0}

        def operation(_codex: object) -> str:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise TransportClosedError(_LOCKED)
            return "ok"

        with patch("hitch.main.codex_pool.time.sleep") as mock_sleep:
            result = codex_pool.run_codex_op_with_retry(
                cast("Callable[[], Any]", factory), operation
            )

        self.assertEqual(result, "ok")
        # Each attempt opened (and closed) a fresh app-server, never reusing the
        # one that exited on the locked state DB.
        self.assertEqual(len(servers), 3)
        self.assertTrue(all(server.closed for server in servers))
        self.assertEqual(mock_sleep.call_count, 2)

    def test_non_locked_transport_error_is_not_retried(self) -> None:
        codex = _FakeCodex()
        factory = MagicMock(return_value=codex)
        operation = MagicMock(
            side_effect=TransportClosedError("app-server closed stdout. crash")
        )
        with (
            patch("hitch.main.codex_pool.time.sleep"),
            self.assertRaises(TransportClosedError),
        ):
            codex_pool.run_codex_op_with_retry(factory, operation)

        self.assertEqual(operation.call_count, 1)

    def test_locked_construction_retries_stay_bounded(self) -> None:
        # A lock during *construction* (factory raising) must be retried by this
        # single loop, not by a nested startup loop -- so factory runs at most
        # _APPSERVER_START_MAX_ATTEMPTS times, never that count squared.
        factory = MagicMock(side_effect=TransportClosedError(_LOCKED))
        operation = MagicMock()
        with (
            patch("hitch.main.codex_pool.time.sleep"),
            self.assertRaises(TransportClosedError),
        ):
            codex_pool.run_codex_op_with_retry(factory, operation)

        self.assertEqual(factory.call_count, codex_pool._APPSERVER_START_MAX_ATTEMPTS)
        operation.assert_not_called()

    def test_non_transport_error_propagates_immediately(self) -> None:
        codex = _FakeCodex()
        factory = MagicMock(return_value=codex)
        operation = MagicMock(side_effect=ValueError("boom"))
        with (
            patch("hitch.main.codex_pool.time.sleep"),
            self.assertRaises(ValueError),
        ):
            codex_pool.run_codex_op_with_retry(factory, operation)

        self.assertEqual(operation.call_count, 1)

    def test_reraises_after_exhausting_attempts(self) -> None:
        factory = MagicMock(side_effect=lambda: _FakeCodex())
        operation = MagicMock(side_effect=TransportClosedError(_LOCKED))
        with (
            patch("hitch.main.codex_pool.time.sleep"),
            self.assertRaises(TransportClosedError),
        ):
            codex_pool.run_codex_op_with_retry(factory, operation)

        self.assertEqual(
            operation.call_count, codex_pool._APPSERVER_START_MAX_ATTEMPTS
        )
