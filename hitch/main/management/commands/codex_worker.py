"""Detached worker that runs one Codex turn and streams events to disk.

Invoked by ``codex_pool._launch_worker_process`` as a fresh-session Django
``manage.py`` subprocess. The lifecycle is:

  1. Mark the CodexInstance as ``running`` and open its JSONL events file
     for line-buffered append.
  2. Start a Codex app-server, resume the thread by id, and submit ``prompt``
     as a single turn.
  3. Stream every notification produced by that turn into the events file.
  4. On success mark the row ``completed``; on any exception mark ``failed``
     and write the exception message to ``error``.

The events file plus the status transitions on the row are the only output —
stdout/stderr are redirected to /dev/null by the parent.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import shutil
from typing import IO, Any, override

from django.core.management.base import BaseCommand, CommandParser
from django.utils import timezone
from openai_codex import AppServerConfig, Codex, TextInput
from openai_codex.generated.v2_all import (
    ReasoningEffort,
    Turn,
    TurnCompletedNotification,
    TurnStatus,
)
from pydantic import BaseModel

from hitch.main.models import CodexInstance


class Command(BaseCommand):
    help = "Run one Codex turn for an existing CodexInstance and stream events to disk."

    @override
    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--instance-id", type=int, required=True)
        # The settings dialog stores reasoning effort in a cookie; the spawner
        # forwards it here so the detached worker doesn't have to reach back
        # into the parent process or any shared store.
        parser.add_argument("--reasoning-effort", type=str, default=None)

    @override
    def handle(self, *args: Any, **options: Any) -> None:
        instance_id: int = options["instance_id"]
        reasoning_effort: str | None = options.get("reasoning_effort")
        instance = CodexInstance.objects.get(pk=instance_id)

        instance.status = CodexInstance.STATUS_RUNNING
        instance.save(update_fields=["status"])

        try:
            with open(instance.events_path, "a", buffering=1, encoding="utf-8") as events_file:
                final_turn = _run_turn(
                    thread_id=instance.thread_id,
                    prompt=instance.prompt,
                    events_file=events_file,
                    reasoning_effort=reasoning_effort,
                )
        except Exception as exc:  # noqa: BLE001 - record any failure, then re-raise
            instance.status = CodexInstance.STATUS_FAILED
            instance.ended_at = timezone.now()
            instance.error = repr(exc)
            instance.save(update_fields=["status", "ended_at", "error"])
            raise

        instance.ended_at = timezone.now()
        # The TurnCompletedNotification carries the actual outcome — including
        # ``interrupted`` and ``failed`` terminal states the SDK does not
        # raise on — so map it through rather than blanket-marking completed.
        if final_turn is None:
            instance.status = CodexInstance.STATUS_FAILED
            instance.error = "stream ended without a turn/completed notification"
        elif final_turn.status == TurnStatus.completed:
            instance.status = CodexInstance.STATUS_COMPLETED
        else:
            instance.status = CodexInstance.STATUS_FAILED
            error = final_turn.error
            instance.error = (
                error.message
                if error is not None and error.message
                else f"turn ended with status {final_turn.status.value}"
            )
        instance.save(update_fields=["status", "ended_at", "error"])


def _run_turn(
    *,
    thread_id: str,
    prompt: str,
    events_file: IO[str],
    reasoning_effort: str | None = None,
) -> Turn | None:
    config = AppServerConfig(codex_bin=shutil.which("codex"))
    turn_kwargs: dict[str, Any] = {}
    if reasoning_effort:
        # Unknown strings are ignored rather than crashing the worker — Codex
        # will fall back to the model's default effort in that case, which is
        # preferable to losing the whole turn over a stale enum value.
        with contextlib.suppress(ValueError):
            turn_kwargs["effort"] = ReasoningEffort(reasoning_effort)
    final_turn: Turn | None = None
    with Codex(config=config) as codex:
        thread = codex.thread_resume(thread_id)
        turn = thread.turn(TextInput(prompt), **turn_kwargs)
        for event in turn.stream():
            events_file.write(_serialize_event(event.method, event.payload) + "\n")
            payload = event.payload
            if isinstance(payload, TurnCompletedNotification) and payload.turn.id == turn.id:
                final_turn = payload.turn
    return final_turn


def _serialize_event(method: str, payload: Any) -> str:
    if isinstance(payload, BaseModel):
        payload_data: Any = payload.model_dump(mode="json", by_alias=True)
    elif dataclasses.is_dataclass(payload) and not isinstance(payload, type):
        payload_data = dataclasses.asdict(payload)
    else:
        payload_data = payload
    return json.dumps({"method": method, "payload": payload_data})
