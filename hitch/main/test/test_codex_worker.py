import json
import tempfile
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.core.management import call_command
from django.test import TestCase
from openai_codex.generated.v2_all import (
    Turn,
    TurnCompletedNotification,
    TurnError,
    TurnStatus,
)
from pydantic import BaseModel

from hitch.main.management.commands.codex_worker import _serialize_event
from hitch.main.models import CodexInstance


class _FakePayload(BaseModel):
    method_kind: str = "demo"
    detail: str


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


class SerializeEventTests(TestCase):
    def test_serializes_pydantic_payload(self) -> None:
        payload = _FakePayload(detail="hello")
        line = _serialize_event("item/agentMessage/delta", payload)
        parsed = json.loads(line)
        self.assertEqual(parsed["method"], "item/agentMessage/delta")
        self.assertEqual(parsed["payload"]["detail"], "hello")

    def test_serializes_dataclass_payload(self) -> None:
        import dataclasses

        @dataclasses.dataclass
        class Params:
            params: dict[str, str]

        line = _serialize_event("unknown/method", Params(params={"k": "v"}))
        parsed = json.loads(line)
        self.assertEqual(parsed["payload"], {"params": {"k": "v"}})

    def test_serializes_plain_dict_payload(self) -> None:
        line = _serialize_event("m", {"k": 1})
        self.assertEqual(json.loads(line)["payload"], {"k": 1})


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
    def test_marks_failed_when_turn_status_is_failed(self, mock_codex: MagicMock) -> None:
        events = [_completed_event("turn-1", TurnStatus.failed, error_message="model said no")]
        codex_ctx = mock_codex.return_value.__enter__.return_value
        codex_ctx.thread_resume.return_value = _stub_thread_resume(events)

        with tempfile.TemporaryDirectory() as raw:
            instance = self._make_instance(Path(raw))
            call_command("codex_worker", "--instance-id", str(instance.pk))

        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertEqual(instance.error, "model said no")
        self.assertIsNotNone(instance.ended_at)

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_marks_failed_when_turn_status_is_interrupted(self, mock_codex: MagicMock) -> None:
        events = [_completed_event("turn-1", TurnStatus.interrupted)]
        codex_ctx = mock_codex.return_value.__enter__.return_value
        codex_ctx.thread_resume.return_value = _stub_thread_resume(events)

        with tempfile.TemporaryDirectory() as raw:
            instance = self._make_instance(Path(raw))
            call_command("codex_worker", "--instance-id", str(instance.pk))

        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertIn("interrupted", instance.error)

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_marks_failed_when_no_completion_event(self, mock_codex: MagicMock) -> None:
        # A stream that ended (TurnHandle.stream is closed) without a
        # turn/completed event is also an unsuccessful outcome.
        codex_ctx = mock_codex.return_value.__enter__.return_value
        codex_ctx.thread_resume.return_value = _stub_thread_resume([])

        with tempfile.TemporaryDirectory() as raw:
            instance = self._make_instance(Path(raw))
            call_command("codex_worker", "--instance-id", str(instance.pk))

        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertIn("turn/completed", instance.error)

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
