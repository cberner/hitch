"""Helpers for reading Hitch's per-worker Codex event logs."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from hitch.main.models import CodexInstance
from hitch.main.runtime.sdk_values import is_nonbool_int

logger = logging.getLogger(__name__)

GOAL_CLEARED_METHOD = "thread/goal/cleared"
GOAL_UPDATED_METHOD = "thread/goal/updated"
GOAL_METHODS = frozenset({GOAL_CLEARED_METHOD, GOAL_UPDATED_METHOD})
TASK_PLAN_UPDATED_METHOD = "turn/plan/updated"
TURN_DIFF_UPDATED_METHOD = "turn/diff/updated"

_TURN_DIFF_EVENT_LINE_RE = re.compile(rb'^\s*\{\s*"method"\s*:\s*"turn/diff/updated"(?:\s*[,}])')
_T = TypeVar("_T")


def append_event(path: str | Path, method: str, payload: dict[str, Any]) -> None:
    """Append a Hitch worker event frame to ``path``."""
    event = {"method": method, "payload": payload}
    with Path(path).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event) + "\n")


def prune_diff_events(events_path: str | Path) -> int:
    """Atomically remove obsolete full-diff notifications from a finished log."""
    path = Path(events_path)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.compact")
    original_stat = path.stat()
    original_size = original_stat.st_size
    removed = False
    try:
        with path.open("rb") as source, temporary_path.open("wb") as destination:
            os.fchmod(destination.fileno(), original_stat.st_mode & 0o7777)
            for line in source:
                if _TURN_DIFF_EVENT_LINE_RE.match(line):
                    removed = True
                    continue
                destination.write(line)
            destination.flush()
            os.fsync(destination.fileno())
        if not removed:
            return 0
        os.replace(temporary_path, path)
        return max(0, original_size - path.stat().st_size)
    finally:
        temporary_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class _GoalEvent:
    order: tuple[int, int, int]
    objective: str | None
    tokens_used: int | None


@dataclass(frozen=True)
class TaskPlanStep:
    step: str
    status: str


@dataclass(frozen=True)
class TaskPlanSnapshot:
    explanation: str
    steps: tuple[TaskPlanStep, ...]
    order: tuple[int, int, int]


@dataclass(frozen=True)
class _TaskPlanEvent:
    order: tuple[int, int, int]
    snapshot: TaskPlanSnapshot


def latest_goal_for_thread(thread_id: str) -> str:
    paths = (
        CodexInstance.objects.filter(thread_id=thread_id)
        .order_by("pk")
        .values_list("events_path", flat=True)
    )
    return latest_goal_from_event_paths(paths, thread_id=thread_id) or ""


def latest_goal_from_event_paths(paths: Iterable[str | Path], *, thread_id: str) -> str | None:
    current = _latest_goal_event_from_event_paths(paths, thread_id=thread_id)
    return current.objective if current is not None else None


def latest_goal_tokens_for_instance(instance: CodexInstance | None) -> int | None:
    if instance is None or not instance.events_path:
        return None
    return latest_goal_tokens_from_event_paths(
        [instance.events_path],
        thread_id=instance.thread_id,
    )


def latest_goal_tokens_from_event_paths(paths: Iterable[str | Path], *, thread_id: str) -> int | None:
    current = _latest_goal_event_from_event_paths(paths, thread_id=thread_id)
    return current.tokens_used if current is not None else None


def _latest_goal_event_from_event_paths(paths: Iterable[str | Path], *, thread_id: str) -> _GoalEvent | None:
    current: _GoalEvent | None = None
    for event in _parsed_events_from_paths(
        paths,
        thread_id=thread_id,
        parser=_goal_event_from_event,
    ):
        if current is None or event.order > current.order:
            current = event
    return current


def latest_task_plan_for_instance(instance: CodexInstance | None) -> TaskPlanSnapshot | None:
    if instance is None or not instance.events_path:
        return None
    return latest_task_plan_from_event_paths(
        [instance.events_path],
        thread_id=instance.thread_id,
    )


def latest_task_plan_for_thread(thread_id: str) -> TaskPlanSnapshot | None:
    latest = (
        CodexInstance.objects.filter(thread_id=thread_id)
        .order_by("-started_at", "-pk")
        .first()
    )
    return latest_task_plan_for_instance(latest)


def latest_task_plan_from_event_paths(paths: Iterable[str | Path], *, thread_id: str) -> TaskPlanSnapshot | None:
    current: _TaskPlanEvent | None = None
    for event in _parsed_events_from_paths(
        paths,
        thread_id=thread_id,
        parser=_task_plan_event_from_event,
    ):
        if current is None or event.order > current.order:
            current = event
    return current.snapshot if current is not None else None


def _parsed_events_from_paths(
    paths: Iterable[str | Path],
    *,
    thread_id: str,
    parser: Callable[[dict[str, Any], str, int], _T | None],
) -> Iterator[_T]:
    fallback_order = 0
    for raw_path in paths:
        if not raw_path:
            continue
        path = Path(raw_path)
        try:
            with path.open("r", encoding="utf-8") as fh:
                for raw in fh:
                    fallback_order += 1
                    parsed = _event_from_line(raw)
                    if parsed is None:
                        continue
                    event = parser(parsed, thread_id, fallback_order)
                    if event is not None:
                        yield event
        except FileNotFoundError:
            continue
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("failed to read Codex events %s: %s", path, exc)


def _event_from_line(raw: str) -> dict[str, Any] | None:
    if not (raw := raw.strip()):
        return None
    try:
        event = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return event if isinstance(event, dict) else None


def _goal_event_from_event(event: dict[str, Any], thread_id: str, fallback_order: int) -> _GoalEvent | None:
    method = event.get("method")
    if method not in GOAL_METHODS:
        return None
    payload = event.get("payload")
    if not isinstance(payload, dict) or _payload_thread_id(payload) != thread_id:
        return None
    order = _event_order(event, fallback_order)
    if method == GOAL_CLEARED_METHOD:
        return _GoalEvent(order=order, objective=None, tokens_used=None)
    goal = payload.get("goal")
    if not isinstance(goal, dict):
        return None
    objective = goal.get("objective")
    if not isinstance(objective, str):
        return None
    return _GoalEvent(
        order=order,
        objective=objective.strip() or None,
        tokens_used=_goal_tokens_used(goal),
    )


def _goal_tokens_used(goal: dict[str, Any]) -> int | None:
    for key in ("tokensUsed", "tokens_used"):
        value = goal.get(key)
        if is_nonbool_int(value):
            return max(0, value)
    return None


def _task_plan_event_from_event(
    event: dict[str, Any], thread_id: str, fallback_order: int
) -> _TaskPlanEvent | None:
    if event.get("method") != TASK_PLAN_UPDATED_METHOD:
        return None
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return None
    payload_thread_id = _payload_thread_id(payload)
    if payload_thread_id is not None and payload_thread_id != thread_id:
        return None
    explanation = payload.get("explanation")
    plan = payload.get("plan")
    if "plan" in payload and not isinstance(plan, list):
        return None
    if not isinstance(explanation, str) and not isinstance(plan, list):
        return None
    plan = plan if isinstance(plan, list) else []
    snapshot = _task_plan_snapshot(
        explanation if isinstance(explanation, str) else "",
        plan,
        _event_order(event, fallback_order),
    )
    if not snapshot.explanation and not snapshot.steps and plan:
        return None
    return _TaskPlanEvent(order=snapshot.order, snapshot=snapshot)


def _task_plan_snapshot(
    explanation: str,
    plan: list[Any],
    order: tuple[int, int, int],
) -> TaskPlanSnapshot:
    steps = tuple(step for raw_step in plan if (step := _task_plan_step(raw_step)) is not None)
    return TaskPlanSnapshot(explanation=explanation.strip(), steps=steps, order=order)


def _task_plan_step(raw_step: Any) -> TaskPlanStep | None:
    if not isinstance(raw_step, dict):
        return None
    step = raw_step.get("step")
    if not isinstance(step, str) or not (step := step.strip()):
        return None
    return TaskPlanStep(
        step=step,
        status=_normalize_task_plan_status(raw_step.get("status")),
    )


def _normalize_task_plan_status(status: Any) -> str:
    if status == "completed":
        return "completed"
    if status in {"inProgress", "in_progress"}:
        return "inProgress"
    return "pending"


def _event_order(event: dict[str, Any], fallback_order: int) -> tuple[int, int, int]:
    event_seq = _int_field(event, "eventSeq")
    recorded_at = _int_field(event, "recordedAt")
    if recorded_at is None:
        recorded_at = _int_field(event, "recorded_at")
    if recorded_at is not None:
        return (recorded_at, event_seq or 0, fallback_order)
    if event_seq is not None:
        return (0, event_seq, fallback_order)
    return (0, 0, fallback_order)


def _int_field(event: dict[str, Any], key: str) -> int | None:
    raw = event.get(key)
    return raw if is_nonbool_int(raw) else None


def _payload_thread_id(payload: dict[str, Any]) -> str | None:
    thread_id = payload.get("threadId")
    if isinstance(thread_id, str):
        return thread_id
    thread_id = payload.get("thread_id")
    return thread_id if isinstance(thread_id, str) else None
