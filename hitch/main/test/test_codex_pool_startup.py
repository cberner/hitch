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
    def test_raw_resume_fallback_for_sdk_metadata_validation(self) -> None:
        codex = _ValidationFailingCodex(
            _thread_resume_validation_error(loc=("reasoningEffort",))
        )
        with app_server_pool.open_codex_resumed(
            cast("Callable[[], Any]", lambda: codex),
            thread_id="t1",
            resume_kwargs={
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
