"""Coverage for the detached Codex worker plumbing: spawning, launching the
subprocess, the worker management command, and the bookkeeping that keeps
the CodexInstance row in sync with the OS process.
"""

import dataclasses
import json
import os
import tempfile
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.core.management import call_command
from django.test import TestCase, override_settings
from openai_codex.generated.v2_all import (
    ReasoningEffort,
    Turn,
    TurnCompletedNotification,
    TurnError,
    TurnStatus,
)
from pydantic import BaseModel

from hitch.main import codex_pool
from hitch.main.management.commands.codex_worker import _serialize_event
from hitch.main.models import CodexInstance


def _events_dir() -> tempfile.TemporaryDirectory[str]:
    return tempfile.TemporaryDirectory()


def _stub_codex_thread_start(mock_codex: MagicMock, thread_id: str = "t") -> MagicMock:
    codex: MagicMock = mock_codex.return_value.__enter__.return_value
    codex.thread_start.return_value = SimpleNamespace(id=thread_id)
    return codex


def _completed_event(turn_id: str, status: TurnStatus, error_message: str | None = None) -> SimpleNamespace:
    """Build a turn/completed event whose payload is a real
    TurnCompletedNotification — the worker's status logic narrows on the
    SDK type, not on duck-typed shapes, so the test must use the real model.
    """
    return SimpleNamespace(
        method="turn/completed",
        payload=TurnCompletedNotification(
            thread_id="thread-1",
            turn=Turn(
                id=turn_id,
                items=[],
                status=status,
                error=TurnError(message=error_message) if error_message else None,
            ),
        ),
    )


def _stub_thread_resume(events: list[SimpleNamespace], turn_id: str = "turn-1") -> object:
    """Return an object shaped like ``thread_resume(...).turn(...).stream()``."""
    return SimpleNamespace(
        turn=lambda _input: SimpleNamespace(id=turn_id, stream=lambda: iter(events)),
    )


class SpawnNewSessionTests(TestCase):
    @patch("hitch.main.codex_pool._launch_worker_process")
    @patch("hitch.main.codex_pool.Codex")
    def test_creates_thread_then_spawns_worker(
        self, mock_codex: MagicMock, mock_launch: MagicMock
    ) -> None:
        codex = _stub_codex_thread_start(mock_codex, "thread-abc")
        mock_launch.return_value = SimpleNamespace(pid=4242)

        with (
            _events_dir() as events_dir,
            override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
        ):
            instance = codex_pool.spawn_new_session(cwd="/repo", prompt="hi")

        self.assertEqual(instance.thread_id, "thread-abc")
        self.assertEqual(instance.cwd, "/repo")
        self.assertEqual(instance.prompt, "hi")
        self.assertEqual(instance.pid, 4242)
        self.assertEqual(instance.status, CodexInstance.STATUS_STARTING)
        self.assertTrue(instance.events_path.endswith(f"{instance.pk}.jsonl"))
        # ``model=None`` means "fall back to whatever Codex's config picks".
        codex.thread_start.assert_called_once_with(cwd="/repo", model=None)
        # ``thread/start`` defers writing the rollout file to disk, so the
        # cross-process ``thread/resume`` the worker and the session view both
        # rely on would fail with "no rollout found" without an explicit
        # metadata write to materialise the rollout. ``thread/set-name`` is the
        # cheapest such write; it must happen inside the same Codex context as
        # ``thread/start`` so the in-memory thread is still loaded.
        codex._client.thread_set_name.assert_called_once_with("thread-abc", "hi")
        # Worker subprocess only receives the row id and (when set) the
        # reasoning effort; the prompt is read from the row to avoid argparse
        # misinterpreting prompts that begin with '-'.
        mock_launch.assert_called_once_with(instance_id=instance.pk, reasoning_effort=None)

    @patch("hitch.main.codex_pool._launch_worker_process")
    @patch("hitch.main.codex_pool.Codex")
    def test_initial_thread_name_derivation(
        self, mock_codex: MagicMock, mock_launch: MagicMock
    ) -> None:
        """The initial name is the first line of the prompt, trimmed of
        whitespace, clipped to 200 chars; whitespace-only prompts fall back
        to a static placeholder rather than being passed through verbatim
        (Codex rejects whitespace-only names)."""
        codex = _stub_codex_thread_start(mock_codex)
        mock_launch.return_value = SimpleNamespace(pid=1)

        cases = [
            ("  Refactor the parser \nthen write tests\n", "Refactor the parser"),
            ("a" * 500, "a" * 200),
            ("   \n\t  ", "New session"),
        ]
        for prompt, expected in cases:
            with self.subTest(prompt=prompt[:30]):
                codex._client.thread_set_name.reset_mock()
                with (
                    _events_dir() as events_dir,
                    override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
                ):
                    codex_pool.spawn_new_session(cwd="/repo", prompt=prompt)
                codex._client.thread_set_name.assert_called_once_with("t", expected)

    @patch("hitch.main.codex_pool._launch_worker_process")
    @patch("hitch.main.codex_pool.Codex")
    def test_forwards_model_and_effort(
        self, mock_codex: MagicMock, mock_launch: MagicMock
    ) -> None:
        """The settings dialog's model selector flows into
        ``thread_start(model=...)`` and the effort flows into the worker as
        a CLI arg. Pin both wirings so a refactor can't quietly drop them.
        """
        codex = _stub_codex_thread_start(mock_codex)
        mock_launch.return_value = SimpleNamespace(pid=1)

        with (
            _events_dir() as events_dir,
            override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
        ):
            instance = codex_pool.spawn_new_session(
                cwd="/repo", prompt="hi", model="gpt-5", reasoning_effort="high"
            )

        codex.thread_start.assert_called_once_with(cwd="/repo", model="gpt-5")
        mock_launch.assert_called_once_with(
            instance_id=instance.pk, reasoning_effort="high"
        )


class SpawnFailureTests(TestCase):
    @patch("hitch.main.codex_pool._launch_worker_process")
    def test_marks_instance_failed_when_launch_raises(
        self, mock_launch: MagicMock
    ) -> None:
        mock_launch.side_effect = OSError("boom")

        with (
            _events_dir() as events_dir,
            override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
            self.assertRaises(OSError),
        ):
            codex_pool.spawn_turn(thread_id="t", cwd="/repo", prompt="hi")

        # The exception propagates to the caller, but the row is left in a
        # terminal state so it isn't treated as forever-pending.
        instance = CodexInstance.objects.latest("started_at")
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertEqual(instance.pid, 0)
        self.assertIn("boom", instance.error)
        self.assertIsNotNone(instance.ended_at)


class SpawnTurnTests(TestCase):
    @patch("hitch.main.codex_pool._launch_worker_process")
    def test_resumes_existing_thread_without_calling_codex(
        self, mock_launch: MagicMock
    ) -> None:
        """``spawn_turn`` operates entirely on a known thread id — it must
        never reach out to the Codex app-server, which keeps it cheap on the
        request-serving path."""
        mock_launch.return_value = SimpleNamespace(pid=1234)

        with (
            _events_dir() as events_dir,
            override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
        ):
            instance = codex_pool.spawn_turn(
                thread_id="thread-xyz", cwd="/repo", prompt="follow-up"
            )

        self.assertEqual(instance.thread_id, "thread-xyz")
        self.assertEqual(instance.prompt, "follow-up")
        self.assertEqual(instance.pid, 1234)
        mock_launch.assert_called_once_with(instance_id=instance.pk, reasoning_effort=None)


class LaunchWorkerProcessTests(TestCase):
    @patch("hitch.main.codex_pool.subprocess.Popen")
    def test_launches_manage_command_in_new_session(self, mock_popen: MagicMock) -> None:
        mock_popen.return_value = SimpleNamespace(pid=999)

        codex_pool._launch_worker_process(instance_id=7)

        args, kwargs = mock_popen.call_args
        argv = args[0]
        self.assertEqual(argv[1].endswith("manage.py"), True)
        self.assertEqual(argv[2], "codex_worker")
        self.assertIn("--instance-id", argv)
        self.assertEqual(argv[argv.index("--instance-id") + 1], "7")
        # The prompt is *not* passed on the command line — it lives on the
        # CodexInstance row so a leading '-' can't be reinterpreted as an
        # argparse option.
        self.assertNotIn("--prompt", argv)
        # The worker must outlive the Django process, so it gets its own
        # session and all stdio is redirected to /dev/null.
        self.assertTrue(kwargs["start_new_session"])
        from subprocess import DEVNULL

        self.assertEqual(kwargs["stdin"], DEVNULL)
        self.assertEqual(kwargs["stdout"], DEVNULL)
        self.assertEqual(kwargs["stderr"], DEVNULL)
        self.assertTrue(kwargs["close_fds"])
        self.assertEqual(
            kwargs["env"]["DJANGO_SETTINGS_MODULE"],
            "hitch.settings.dev",
        )
        # No effort passed → no --reasoning-effort on the CLI; Codex's own
        # default takes over inside the worker.
        self.assertNotIn("--reasoning-effort", argv)

    @patch("hitch.main.codex_pool.subprocess.Popen")
    def test_forwards_reasoning_effort_as_cli_arg(self, mock_popen: MagicMock) -> None:
        mock_popen.return_value = SimpleNamespace(pid=999)

        codex_pool._launch_worker_process(instance_id=7, reasoning_effort="high")

        argv = mock_popen.call_args.args[0]
        self.assertIn("--reasoning-effort", argv)
        self.assertEqual(argv[argv.index("--reasoning-effort") + 1], "high")


class IsAliveTests(TestCase):
    def test_known_pid_states(self) -> None:
        self.assertTrue(codex_pool.is_alive(os.getpid()))
        self.assertFalse(codex_pool.is_alive(0))
        self.assertFalse(codex_pool.is_alive(-1))
        # 2**22 is well above the default pid_max on Linux/macOS.
        self.assertFalse(codex_pool.is_alive(2**22))

    @patch("hitch.main.codex_pool.os.kill")
    def test_permission_error_means_alive(self, mock_kill: MagicMock) -> None:
        # ``os.kill(pid, 0)`` raises PermissionError when the pid exists but
        # is owned by another user; the process is still alive in that case.
        mock_kill.side_effect = PermissionError
        self.assertTrue(codex_pool.is_alive(1234))

    @patch("hitch.main.codex_pool.os.kill")
    def test_other_os_error_is_treated_as_dead(self, mock_kill: MagicMock) -> None:
        mock_kill.side_effect = OSError
        self.assertFalse(codex_pool.is_alive(1234))


class CodexInstanceModelTests(TestCase):
    def test_str_includes_pid_thread_and_status(self) -> None:
        instance = CodexInstance(
            pid=42,
            thread_id="abc",
            cwd="/r",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
        )
        rendered = str(instance)
        self.assertIn("pid=42", rendered)
        self.assertIn("thread_id=abc", rendered)
        self.assertIn("status=running", rendered)


class ReconcileAndLookupTests(TestCase):
    def _make(self, *, pid: int = 1, thread_id: str = "t", status: str | None = None) -> CodexInstance:
        return CodexInstance.objects.create(
            pid=pid,
            thread_id=thread_id,
            cwd="/r",
            events_path="/dev/null",
            status=status or CodexInstance.STATUS_COMPLETED,
        )

    @patch("hitch.main.codex_pool.is_alive")
    def test_reconcile_marks_only_dead_pending_rows_failed(
        self, mock_alive: MagicMock
    ) -> None:
        dead_running = self._make(pid=10, status=CodexInstance.STATUS_RUNNING)
        live_running = self._make(pid=11, status=CodexInstance.STATUS_RUNNING)
        completed = self._make(pid=12, status=CodexInstance.STATUS_COMPLETED)
        mock_alive.side_effect = lambda pid: pid == 11

        n = codex_pool.reconcile_dead()

        self.assertEqual(n, 1)
        dead_running.refresh_from_db()
        live_running.refresh_from_db()
        completed.refresh_from_db()
        self.assertEqual(dead_running.status, CodexInstance.STATUS_FAILED)
        self.assertIsNotNone(dead_running.ended_at)
        self.assertEqual(live_running.status, CodexInstance.STATUS_RUNNING)
        self.assertIsNone(live_running.ended_at)
        self.assertEqual(completed.status, CodexInstance.STATUS_COMPLETED)
        self.assertIn("exited", dead_running.error)

    def test_list_and_latest_for_thread(self) -> None:
        first = self._make(thread_id="t1")
        second = self._make(thread_id="t1")
        self._make(thread_id="t2")

        self.assertEqual(
            [r.pk for r in codex_pool.list_for_thread("t1")],
            [second.pk, first.pk],
        )
        self.assertEqual(codex_pool.latest_for_thread("t1"), second)
        self.assertIsNone(codex_pool.latest_for_thread("nothing"))


class EventsDirTests(TestCase):
    def test_uses_setting_when_configured(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_EVENTS_DIR=Path(raw)),
        ):
            self.assertEqual(codex_pool.events_dir(), Path(raw))

    def test_falls_back_to_home_dir(self) -> None:
        with override_settings(CODEX_EVENTS_DIR=None):
            self.assertEqual(
                codex_pool.events_dir(),
                Path.home() / ".hitch" / "codex_events",
            )


class _FakePayload(BaseModel):
    method_kind: str = "demo"
    detail: str


class SerializeEventTests(TestCase):
    def test_serializes_payload_shapes(self) -> None:
        @dataclasses.dataclass
        class Params:
            params: dict[str, str]

        # pydantic, dataclass, plain dict all flatten to a JSON payload field.
        cases = [
            (_FakePayload(detail="hello"), {"method_kind": "demo", "detail": "hello"}),
            (Params(params={"k": "v"}), {"params": {"k": "v"}}),
            ({"k": 1}, {"k": 1}),
        ]
        for payload, expected in cases:
            with self.subTest(payload=type(payload).__name__):
                parsed = json.loads(_serialize_event("m", payload))
                self.assertEqual(parsed["method"], "m")
                self.assertEqual(parsed["payload"], expected)


class CodexWorkerCommandTests(TestCase):
    def _make_instance(self, events_dir: Path, *, prompt: str = "hi") -> CodexInstance:
        return CodexInstance.objects.create(
            pid=12345,
            thread_id="thread-1",
            cwd="/repo",
            prompt=prompt,
            events_path=str(events_dir / "events.jsonl"),
            status=CodexInstance.STATUS_STARTING,
        )

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_streams_notifications_and_marks_completed(self, mock_codex: MagicMock) -> None:
        events = [
            SimpleNamespace(
                method="item/agentMessage/delta",
                payload=_FakePayload(detail="chunk-1"),
            ),
            _completed_event("turn-1", TurnStatus.completed),
        ]
        codex_ctx = mock_codex.return_value.__enter__.return_value
        codex_ctx.thread_resume.return_value = _stub_thread_resume(events)

        with tempfile.TemporaryDirectory() as raw:
            instance = self._make_instance(Path(raw))
            call_command("codex_worker", "--instance-id", str(instance.pk))

            with open(instance.events_path, encoding="utf-8") as fh:
                lines = [json.loads(line) for line in fh]

        codex_ctx.thread_resume.assert_called_once_with("thread-1")
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["method"], "item/agentMessage/delta")
        self.assertEqual(lines[0]["payload"]["detail"], "chunk-1")
        self.assertEqual(lines[1]["method"], "turn/completed")

        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_COMPLETED)
        self.assertIsNotNone(instance.ended_at)
        self.assertEqual(instance.error, "")

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_reads_prompt_from_instance_row(self, mock_codex: MagicMock) -> None:
        """The prompt isn't a CLI arg — verify it round-trips via the row,
        including a leading-dash value that argparse would reject."""
        captured: dict[str, object] = {}

        def _capture_turn(input_obj: object) -> object:
            captured["input"] = input_obj
            return SimpleNamespace(
                id="turn-1",
                stream=lambda: iter([_completed_event("turn-1", TurnStatus.completed)]),
            )

        codex_ctx = mock_codex.return_value.__enter__.return_value
        codex_ctx.thread_resume.return_value = SimpleNamespace(turn=_capture_turn)

        with tempfile.TemporaryDirectory() as raw:
            instance = self._make_instance(Path(raw), prompt="- a markdown bullet")
            call_command("codex_worker", "--instance-id", str(instance.pk))

        self.assertEqual(getattr(captured["input"], "text", None), "- a markdown bullet")

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_marks_failed_for_non_completed_outcomes(self, mock_codex: MagicMock) -> None:
        """Failed / interrupted turn statuses and a stream that ends without
        a turn/completed event all leave the row in STATUS_FAILED with a
        human-readable error blurb."""
        codex_ctx = mock_codex.return_value.__enter__.return_value

        cases = [
            (
                [_completed_event("turn-1", TurnStatus.failed, error_message="model said no")],
                "model said no",
            ),
            ([_completed_event("turn-1", TurnStatus.interrupted)], "interrupted"),
            ([], "turn/completed"),
        ]
        for events, expected_in_error in cases:
            with self.subTest(case=expected_in_error):
                codex_ctx.thread_resume.reset_mock()
                codex_ctx.thread_resume.return_value = _stub_thread_resume(events)
                with tempfile.TemporaryDirectory() as raw:
                    instance = self._make_instance(Path(raw))
                    call_command("codex_worker", "--instance-id", str(instance.pk))
                    instance.refresh_from_db()
                self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
                self.assertIn(expected_in_error, instance.error)
                self.assertIsNotNone(instance.ended_at)

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_records_failure_when_codex_raises(self, mock_codex: MagicMock) -> None:
        mock_codex.return_value.__enter__.side_effect = RuntimeError("boom")

        with tempfile.TemporaryDirectory() as raw:
            instance = self._make_instance(Path(raw))
            with self.assertRaises(RuntimeError):
                call_command(
                    "codex_worker",
                    "--instance-id",
                    str(instance.pk),
                    stderr=StringIO(),
                )

        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertIn("boom", instance.error)
        self.assertIsNotNone(instance.ended_at)

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_reasoning_effort_cli_arg_round_trip(self, mock_codex: MagicMock) -> None:
        """The effort rides into the worker as a CLI arg (no DB or cookie
        lookup in the subprocess). A known value reaches ``turn(effort=)``;
        an unknown one (e.g. stale enum after SDK upgrade) is silently
        dropped so Codex's own default takes over."""
        captured: dict[str, object] = {}

        def _capture_turn(input_obj: object, **kwargs: object) -> object:
            captured.update(kwargs)
            return SimpleNamespace(
                id="turn-1",
                stream=lambda: iter([_completed_event("turn-1", TurnStatus.completed)]),
            )

        codex_ctx = mock_codex.return_value.__enter__.return_value
        codex_ctx.thread_resume.return_value = SimpleNamespace(turn=_capture_turn)

        cases = [("high", ReasoningEffort.high), ("ludicrous", None)]
        for cli_value, expected in cases:
            with self.subTest(cli_value=cli_value):
                captured.clear()
                with tempfile.TemporaryDirectory() as raw:
                    instance = self._make_instance(Path(raw))
                    call_command(
                        "codex_worker",
                        "--instance-id",
                        str(instance.pk),
                        "--reasoning-effort",
                        cli_value,
                    )
                self.assertEqual(captured.get("effort"), expected)

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_marks_running_before_streaming(self, mock_codex: MagicMock) -> None:
        """The row flips to ``running`` before the first event so a slow
        codex initialization is visible to observers (UI / reconciliation)."""
        observed_status: dict[str, str] = {}

        def _capture_thread_resume(*_args: object, **_kwargs: object) -> object:
            observed_status["value"] = CodexInstance.objects.get(pk=instance.pk).status
            return _stub_thread_resume([_completed_event("turn-1", TurnStatus.completed)])

        codex_ctx = mock_codex.return_value.__enter__.return_value
        codex_ctx.thread_resume.side_effect = _capture_thread_resume

        with tempfile.TemporaryDirectory() as raw:
            instance = self._make_instance(Path(raw))
            call_command("codex_worker", "--instance-id", str(instance.pk))

        self.assertEqual(observed_status["value"], CodexInstance.STATUS_RUNNING)
