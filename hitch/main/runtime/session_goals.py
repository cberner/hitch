"""Native goal state and controls for ordinary coding sessions."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import signal
import sqlite3
import tempfile
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast, override

from django.conf import settings
from openai_codex import Input, TurnHandle
from openai_codex._inputs import _normalize_run_input, _to_wire_input
from openai_codex.client import CodexClient
from openai_codex.errors import CodexError
from openai_codex.generated.v2_all import (
    ThreadGoalClearedNotification,
    ThreadGoalUpdatedNotification,
    TurnCompletedNotification,
    TurnInterruptResponse,
    TurnStartedNotification,
    TurnStatus,
    TurnSteerResponse,
)
from openai_codex.models import Notification

from hitch.main.models import CodexInstance
from hitch.main.runtime import codex_pool

_DATABASE = "goals_1.sqlite"
_STATUSES = {"usage_limited": "usageLimited", "budget_limited": "budgetLimited"}


def sqlite_home(thread_id: str) -> Path:
    key = hashlib.sha256(thread_id.encode()).hexdigest()
    return Path(settings.CODEX_SQLITE_HOME_BASE) / "session-goals" / key


def acquire_home(thread_id: str) -> codex_pool.WorkerSqliteHome:
    home = sqlite_home(thread_id)
    fd = os.open(home.with_suffix(".lock"), os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        raise ValueError("Another worker is using this session's goal database. Try again shortly.") from exc
    return codex_pool.WorkerSqliteHome(home=home, _lock_fd=fd, overflow=False)


def compact_cleared_home(thread_id: str) -> None:
    """Called while leased, after Codex closes. The empty home records Clear."""
    home = sqlite_home(thread_id)
    if _stored_goal(home / _DATABASE, thread_id) is None:
        for path in home.glob("*.sqlite*"):
            path.unlink()


def _stored_goal(database: Path, thread_id: str) -> dict[str, Any] | None:
    if not database.is_file():
        return None
    with contextlib.closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        row = db.execute("SELECT * FROM thread_goals WHERE thread_id = ?", (thread_id,)).fetchone()
    if row is None:
        return None
    return {
        "threadId": thread_id, "objective": row["objective"],
        "status": _STATUSES.get(row["status"], row["status"]),
        "tokenBudget": row["token_budget"], "tokensUsed": row["tokens_used"],
        "timeUsedSeconds": row["time_used_seconds"],
        "createdAt": row["created_at_ms"] // 1000, "updatedAt": row["updated_at_ms"] // 1000,
    }


def current_goal(thread_id: str) -> dict[str, Any] | None:
    return _stored_goal(sqlite_home(thread_id) / _DATABASE, thread_id)


def prepare_home(thread_id: str) -> Path:
    """Called under the session lifecycle lock, with no active worker."""
    home = sqlite_home(thread_id)
    home.mkdir(parents=True, exist_ok=True)
    return home


def preserve_worker_goal(thread_id: str, source_home: str) -> None:
    """Keep agent-created goals before a temporary worker home is released."""
    home = sqlite_home(thread_id)
    source = Path(source_home) / _DATABASE
    if not home.is_dir() and _stored_goal(source, thread_id):
        home.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".goal-", dir=home.parent) as temporary:
            staging = Path(temporary)
            with (
                contextlib.closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as src,
                contextlib.closing(sqlite3.connect(staging / _DATABASE)) as dst,
            ):
                src.backup(dst)
            staging.rename(home)


def apply_goal(client: CodexClient, thread_id: str, change: dict[str, Any]) -> dict[str, Any] | None:
    action = change["action"]
    if action == "clear":
        client._request_raw("thread/goal/clear", {"threadId": thread_id})
        return None
    params: dict[str, Any] = {}
    if action == "budget":
        params["tokenBudget"] = change["tokenBudget"]
    elif action == "resume":
        params["status"] = "active"
    elif action == "pause":
        params["status"] = "paused"
    response = cast(dict[str, Any], client._request_raw("thread/goal/set", {"threadId": thread_id, **params}))
    return cast(dict[str, Any], response["goal"])


class GoalTurn(TurnHandle):
    """Follow native continuation turns while retaining their physical IDs."""

    def __init__(self, client: CodexClient, thread_id: str) -> None:
        self._client, self.thread_id = client, thread_id
        self.state = client.register_goal_operation(thread_id)
        try:
            apply_goal(client, thread_id, {"action": "resume"})
            turn_id = self.state.wait_for_start(30)
            if turn_id is None:
                raise ValueError("Codex did not start the goal. Try resuming it again.")
            self.id = turn_id
        except BaseException:
            client.cancel_goal_operation(self.state)
            client.unregister_goal_operation(self.state)
            raise

    @property
    def active_id(self) -> str:
        return self.state.current_turn() or self.id

    @override
    def interrupt(self) -> TurnInterruptResponse:
        # Clear leaves the physical turn running even though pause now fails.
        with contextlib.suppress(CodexError):
            self._client.pause_goal(self.thread_id)
        return self._client.turn_interrupt(self.thread_id, self.active_id)

    @override
    def steer(self, input: Input | str) -> TurnSteerResponse:
        turn_id = self.state.active_turn()
        if turn_id is None:
            raise ValueError("The goal has stopped.")
        return self._client.turn_steer(
            self.thread_id, turn_id, _to_wire_input(_normalize_run_input(input)),
        )

    @override
    def stream(self) -> Iterator[Notification]:
        active = True
        running = False
        completed = False
        try:
            while True:
                event = self._client.next_goal_notification(self.state)
                payload = event.payload
                if isinstance(payload, ThreadGoalUpdatedNotification):
                    active = payload.goal.status.value == "active"
                elif isinstance(payload, ThreadGoalClearedNotification):
                    active = False
                else:
                    if isinstance(payload, TurnStartedNotification):
                        running = True
                    elif isinstance(payload, TurnCompletedNotification):
                        running, completed = False, True
                        if payload.turn.status in {TurnStatus.interrupted, TurnStatus.failed}:
                            active = False
                    yield event
                if completed and not running and not active:
                    break
        finally:
            self.state.finish()
            self._client.unregister_goal_operation(self.state)


def request_live_change(instance: CodexInstance, change: dict[str, Any]) -> dict[str, Any] | None:
    if instance.status != CodexInstance.STATUS_RUNNING or not codex_pool._pid_is_instance_worker(instance):
        raise ValueError("The worker is starting or stopping. Try again shortly.")
    request_id = uuid.uuid4().hex
    path = codex_pool.control_path_for(instance)
    deadline = time.time() + 3
    codex_pool._append_control_request(instance, {
        "op": "goal", "id": request_id, "expiresAt": deadline, "change": change,
    })
    with contextlib.suppress(OSError):
        os.kill(instance.pid, signal.SIGUSR1)
    offset = 0
    while time.time() < deadline:
        with path.open(encoding="utf-8") as fh:
            fh.seek(offset)
            for line in fh:
                if not line.endswith("\n"):
                    break
                offset += len(line.encode())
                reply = json.loads(line)
                if reply.get("op") == "goal_ack" and reply.get("id") == request_id:
                    if reply.get("error"):
                        raise ValueError(reply["error"])
                    return cast(dict[str, Any] | None, reply.get("goal"))
        time.sleep(0.05)
    raise ValueError("Could not confirm the goal change. Refresh the session before trying again.")
