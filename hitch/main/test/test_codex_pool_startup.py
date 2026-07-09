from collections.abc import Callable
from typing import Any, cast
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase, override_settings
from openai_codex import TransportClosedError
from pydantic import ValidationError

from hitch.main.runtime import app_server_pool, reconciliation

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
        with patch("hitch.main.runtime.app_server_pool.time.sleep") as mock_sleep:
            result = app_server_pool._start_codex_with_retry(factory)

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
        with patch("hitch.main.runtime.app_server_pool.time.sleep") as mock_sleep:
            result = app_server_pool._start_codex_with_retry(factory)

        self.assertIs(result, sentinel)
        self.assertEqual(factory.call_count, 3)
        # Backoff slept once per failed attempt, never after the success.
        self.assertEqual(mock_sleep.call_count, 2)

    def test_non_locked_transport_error_is_not_retried(self) -> None:
        boom = TransportClosedError("app-server closed stdout. stderr_tail=segfault")
        factory = MagicMock(side_effect=boom)
        with (
            patch("hitch.main.runtime.app_server_pool.time.sleep"),
            self.assertRaises(TransportClosedError),
        ):
            app_server_pool._start_codex_with_retry(factory)

        self.assertEqual(factory.call_count, 1)

    def test_reraises_after_exhausting_attempts(self) -> None:
        factory = MagicMock(side_effect=TransportClosedError(_LOCKED))
        with (
            patch("hitch.main.runtime.app_server_pool.time.sleep"),
            self.assertRaises(TransportClosedError),
        ):
            app_server_pool._start_codex_with_retry(factory)

        self.assertEqual(factory.call_count, app_server_pool._APPSERVER_START_MAX_ATTEMPTS)


class OpenCodexTests(SimpleTestCase):
    def test_yields_entered_codex_and_closes_on_exit(self) -> None:
        codex = _FakeCodex()
        factory = cast("Callable[[], Any]", lambda: codex)
        with app_server_pool.open_codex(factory) as opened:
            self.assertIs(opened, codex)
            self.assertFalse(codex.closed)
        self.assertTrue(codex.closed)


class RunCodexOpWithRetryTests(SimpleTestCase):
    def test_returns_operation_result_on_clean_run(self) -> None:
        codex = _FakeCodex()
        factory = MagicMock(return_value=codex)
        operation = MagicMock(return_value="ok")
        with patch("hitch.main.runtime.app_server_pool.time.sleep") as mock_sleep:
            result = app_server_pool.run_codex_op_with_retry(factory, operation)

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

        with patch("hitch.main.runtime.app_server_pool.time.sleep") as mock_sleep:
            result = app_server_pool.run_codex_op_with_retry(
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
            patch("hitch.main.runtime.app_server_pool.time.sleep"),
            self.assertRaises(TransportClosedError),
        ):
            app_server_pool.run_codex_op_with_retry(factory, operation)

        self.assertEqual(operation.call_count, 1)

    def test_locked_construction_retries_stay_bounded(self) -> None:
        # A lock during *construction* (factory raising) must be retried by this
        # single loop, not by a nested startup loop -- so factory runs at most
        # _APPSERVER_START_MAX_ATTEMPTS times, never that count squared.
        factory = MagicMock(side_effect=TransportClosedError(_LOCKED))
        operation = MagicMock()
        with (
            patch("hitch.main.runtime.app_server_pool.time.sleep"),
            self.assertRaises(TransportClosedError),
        ):
            app_server_pool.run_codex_op_with_retry(factory, operation)

        self.assertEqual(factory.call_count, app_server_pool._APPSERVER_START_MAX_ATTEMPTS)
        operation.assert_not_called()

    def test_non_transport_error_propagates_immediately(self) -> None:
        codex = _FakeCodex()
        factory = MagicMock(return_value=codex)
        operation = MagicMock(side_effect=ValueError("boom"))
        with (
            patch("hitch.main.runtime.app_server_pool.time.sleep"),
            self.assertRaises(ValueError),
        ):
            app_server_pool.run_codex_op_with_retry(factory, operation)

        self.assertEqual(operation.call_count, 1)

    def test_reraises_after_exhausting_attempts(self) -> None:
        factory = MagicMock(side_effect=lambda: _FakeCodex())
        operation = MagicMock(side_effect=TransportClosedError(_LOCKED))
        with (
            patch("hitch.main.runtime.app_server_pool.time.sleep"),
            self.assertRaises(TransportClosedError),
        ):
            app_server_pool.run_codex_op_with_retry(factory, operation)

        self.assertEqual(
            operation.call_count, app_server_pool._APPSERVER_START_MAX_ATTEMPTS
        )


class _ResumableCodex(_FakeCodex):
    """``_FakeCodex`` plus a ``thread_resume`` that can fail a set number of
    times before returning a sentinel thread, used to drive
    ``open_codex_resumed``'s open+configure+resume retry loop."""

    def __init__(self, fail_times: int = 0, error: Exception | None = None) -> None:
        super().__init__()
        self._fail_times = fail_times
        self._error = error or TransportClosedError(_LOCKED)
        self.resume_calls = 0

    def thread_resume(self, _thread_id: str, **_kwargs: object) -> object:
        self.resume_calls += 1
        if self.resume_calls <= self._fail_times:
            raise self._error
        return ("thread", _thread_id)


def _thread_resume_validation_error(*, loc: tuple[str, ...]) -> ValidationError:
    return ValidationError.from_exception_data(
        "ThreadResumeResponse",
        [
            {
                "type": "enum",
                "loc": loc,
                "input": "max",
                "ctx": {"expected": "'none'"},
            }
        ],
    )


class _RawResumeClient:
    def __init__(self) -> None:
        self.raw_requests: list[tuple[str, object]] = []

    def _request_raw(self, method: str, params: object) -> object:
        self.raw_requests.append((method, params))
        return {"thread": {"id": "thread-raw"}}


class _ValidationFailingCodex(_FakeCodex):
    def __init__(self, error: ValidationError) -> None:
        super().__init__()
        self._error = error
        self._client = _RawResumeClient()
        self.resume_calls = 0

    def thread_resume(self, _thread_id: str, **_kwargs: object) -> object:
        self.resume_calls += 1
        raise self._error


class OpenCodexResumedTests(SimpleTestCase):
    def test_yields_resumed_thread_and_closes_on_exit(self) -> None:
        codex = _ResumableCodex()
        configured: list[object] = []
        with app_server_pool.open_codex_resumed(
            cast("Callable[[], Any]", lambda: codex),
            thread_id="t1",
            configure=configured.append,
        ) as (opened, thread):
            self.assertIs(opened, codex)
            self.assertEqual(thread, ("thread", "t1"))
            self.assertEqual(configured, [codex])
            self.assertFalse(codex.closed)
        self.assertTrue(codex.closed)

    def test_raw_resume_fallback_for_sdk_metadata_validation(self) -> None:
        codex = _ValidationFailingCodex(
            _thread_resume_validation_error(loc=("reasoningEffort",))
        )
        with app_server_pool.open_codex_resumed(
            cast("Callable[[], Any]", lambda: codex),
            thread_id="t1",
            resume_kwargs={
                "base_instructions": "Base.",
                "developer_instructions": "Dev.",
            },
        ) as (_opened, thread):
            self.assertEqual(thread.id, "thread-raw")

        self.assertEqual(codex.resume_calls, 1)
        self.assertEqual(
            codex._client.raw_requests,
            [
                (
                    "thread/resume",
                    {
                        "baseInstructions": "Base.",
                        "developerInstructions": "Dev.",
                        "threadId": "t1",
                    },
                )
            ],
        )
        self.assertTrue(codex.closed)

    def test_resume_validation_inside_thread_is_not_masked(self) -> None:
        codex = _ValidationFailingCodex(
            _thread_resume_validation_error(loc=("thread", "reasoningEffort"))
        )
        with (
            self.assertRaises(ValidationError),
            app_server_pool.open_codex_resumed(
                cast("Callable[[], Any]", lambda: codex),
                thread_id="t1",
            ),
        ):
            pass

        self.assertEqual(codex._client.raw_requests, [])
        self.assertTrue(codex.closed)

    def test_retries_locked_resume_with_fresh_server(self) -> None:
        servers: list[_ResumableCodex] = []

        def factory() -> _ResumableCodex:
            # The first server loses the migration race and exits on resume; the
            # next attempt opens a fresh server whose resume succeeds.
            server = _ResumableCodex(fail_times=1 if not servers else 0)
            servers.append(server)
            return server

        configured: list[object] = []
        with (
            patch("hitch.main.runtime.app_server_pool.time.sleep") as mock_sleep,
            app_server_pool.open_codex_resumed(
                cast("Callable[[], Any]", factory),
                thread_id="t1",
                configure=configured.append,
            ) as (opened, thread),
        ):
            self.assertEqual(thread, ("thread", "t1"))

        # First server failed its resume and was closed; second succeeded.
        self.assertEqual(len(servers), 2)
        self.assertTrue(servers[0].closed)
        self.assertIs(opened, servers[1])
        # configure runs once per attempt against that attempt's server.
        self.assertEqual(configured, servers)
        self.assertEqual(mock_sleep.call_count, 1)

    def test_non_locked_resume_error_is_not_retried(self) -> None:
        codex = _ResumableCodex(fail_times=1, error=TransportClosedError("crash"))
        with (
            patch("hitch.main.runtime.app_server_pool.time.sleep"),
            self.assertRaises(TransportClosedError),
            app_server_pool.open_codex_resumed(
                cast("Callable[[], Any]", lambda: codex), thread_id="t1"
            ),
        ):
            pass
        self.assertEqual(codex.resume_calls, 1)
        self.assertTrue(codex.closed)

    def test_locked_worker_construction_uses_worker_retry_budget(self) -> None:
        factory = MagicMock(side_effect=TransportClosedError(_LOCKED))
        with (
            patch("hitch.main.runtime.app_server_pool.time.sleep"),
            self.assertRaises(TransportClosedError),
            app_server_pool.open_codex_resumed(
                cast("Callable[[], Any]", factory), thread_id="t1"
            ),
        ):
            pass

        self.assertEqual(
            factory.call_count, app_server_pool._APPSERVER_WORKER_START_MAX_ATTEMPTS
        )

    def test_mixed_worker_construction_and_resume_locks_share_budget(self) -> None:
        factory_calls = 0
        servers: list[_ResumableCodex] = []

        def factory() -> _ResumableCodex:
            nonlocal factory_calls
            factory_calls += 1
            if factory_calls % 2:
                raise TransportClosedError(_LOCKED)
            server = _ResumableCodex(fail_times=1)
            servers.append(server)
            return server

        with (
            patch("hitch.main.runtime.app_server_pool.time.sleep"),
            self.assertRaises(TransportClosedError),
            app_server_pool.open_codex_resumed(
                cast("Callable[[], Any]", factory), thread_id="t1"
            ),
        ):
            pass

        self.assertEqual(
            factory_calls, app_server_pool._APPSERVER_WORKER_START_MAX_ATTEMPTS
        )
        self.assertEqual(
            len(servers), app_server_pool._APPSERVER_WORKER_START_MAX_ATTEMPTS // 2
        )
        self.assertTrue(all(server.closed for server in servers))

    def test_caller_exception_closes_server(self) -> None:
        codex = _ResumableCodex()
        with (
            self.assertRaises(ValueError),
            app_server_pool.open_codex_resumed(
                cast("Callable[[], Any]", lambda: codex), thread_id="t1"
            ),
        ):
            raise ValueError("boom")
        self.assertTrue(codex.closed)


class ReconcileDeadIfDueTests(SimpleTestCase):
    @override_settings(TESTING=False)
    def test_runs_sweep_when_claim_wins(self) -> None:
        with (
            patch("hitch.main.runtime.reconciliation.reconcile_dead", return_value=3) as sweep,
            patch("hitch.main.runtime.reconciliation.rate_limit.claim", return_value=True) as claim,
        ):
            self.assertEqual(reconciliation.reconcile_dead_if_due(), 3)
        sweep.assert_called_once_with()
        claim.assert_called_once()

    @override_settings(TESTING=False)
    def test_skips_sweep_when_claim_loses(self) -> None:
        with (
            patch("hitch.main.runtime.reconciliation.reconcile_dead") as sweep,
            patch("hitch.main.runtime.reconciliation.rate_limit.claim", return_value=False),
        ):
            self.assertEqual(reconciliation.reconcile_dead_if_due(), 0)
        sweep.assert_not_called()

    @override_settings(TESTING=True)
    def test_always_sweeps_under_testing(self) -> None:
        with (
            patch("hitch.main.runtime.reconciliation.reconcile_dead", return_value=1) as sweep,
            patch("hitch.main.runtime.reconciliation.rate_limit.claim") as claim,
        ):
            self.assertEqual(reconciliation.reconcile_dead_if_due(), 1)
        sweep.assert_called_once_with()
        claim.assert_not_called()
