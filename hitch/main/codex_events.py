"""Helpers for reading Hitch's per-worker Codex event logs."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hitch.main.models import CodexInstance

logger = logging.getLogger(__name__)

GOAL_CLEARED_METHOD = "thread/goal/cleared"
GOAL_UPDATED_METHOD = "thread/goal/updated"
GOAL_METHODS = frozenset({GOAL_CLEARED_METHOD, GOAL_UPDATED_METHOD})
TASK_PLAN_UPDATED_METHOD = "turn/plan/updated"


@dataclass(frozen=True)
class _GoalEvent:
    order: tuple[int, int, int]
    objective: str | None


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
    """Return the latest known goal objective for ``thread_id``, or ``""``.

    Goal notifications are emitted on the SDK stream rather than exposed on
    the SDK ``Thread`` model. Hitch already persists each worker's raw stream
    to ``CodexInstance.events_path`` for SSE replay, so the session view can
    recover the latest objective from those append-only logs.
    """
    paths = CodexInstance.objects.filter(thread_id=thread_id).order_by("pk").values_list(
        "events_path", flat=True
    )
    return latest_goal_from_event_paths(paths, thread_id=thread_id) or ""


def latest_goal_from_event_paths(
    paths: Iterable[str | Path], *, thread_id: str
) -> str | None:
    """Return the final goal state after applying goal events in ``paths``.

    Workers for the same thread can overlap, so prefer the per-notification
    ``recordedAt`` timestamp assigned by the worker's SDK reader thread rather
    than assuming event files are chronologically ordered by worker creation
    time.
    """
    current: _GoalEvent | None = None
    fallback_order = 0
    for raw_path in paths:
        if not raw_path:
            continue
        path = Path(raw_path)
        try:
            with path.open("r", encoding="utf-8") as fh:
                for raw in fh:
                    fallback_order += 1
                    event = _goal_event_from_line(
                        raw,
                        thread_id=thread_id,
                        fallback_order=fallback_order,
                    )
                    if event is None:
                        continue
                    if current is None or event.order > current.order:
                        current = event
        except FileNotFoundError:
            continue
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("failed to read Codex events %s: %s", path, exc)
            continue
    return current.objective if current is not None else None


def latest_task_plan_for_instance(instance: CodexInstance | None) -> TaskPlanSnapshot | None:
    """Return the latest visible task-plan snapshot for an active worker."""
    if instance is None or not instance.events_path:
        return None
    return latest_task_plan_from_event_paths(
        [instance.events_path],
        thread_id=instance.thread_id,
    )


def latest_task_plan_from_event_paths(
    paths: Iterable[str | Path], *, thread_id: str
) -> TaskPlanSnapshot | None:
    """Return the final ``turn/plan/updated`` state after applying event logs."""
    current: _TaskPlanEvent | None = None
    fallback_order = 0
    for raw_path in paths:
        if not raw_path:
            continue
        path = Path(raw_path)
        try:
            with path.open("r", encoding="utf-8") as fh:
                for raw in fh:
                    fallback_order += 1
                    event = _task_plan_event_from_line(
                        raw,
                        thread_id=thread_id,
                        fallback_order=fallback_order,
                    )
                    if event is None:
                        continue
                    if current is None or event.order > current.order:
                        current = event
        except FileNotFoundError:
            continue
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("failed to read Codex events %s: %s", path, exc)
            continue
    return current.snapshot if current is not None else None


def _goal_event_from_line(
    raw: str, *, thread_id: str, fallback_order: int
) -> _GoalEvent | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        event = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(event, dict):
        return None
    return _goal_event_from_event(
        event,
        thread_id=thread_id,
        fallback_order=fallback_order,
    )


def _goal_event_from_event(
    event: dict[str, Any], *, thread_id: str, fallback_order: int
) -> _GoalEvent | None:
    method = event.get("method")
    if method not in GOAL_METHODS:
        return None
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return None
    if _payload_thread_id(payload) != thread_id:
        return None
    order = _event_order(event, fallback_order)
    if method == GOAL_CLEARED_METHOD:
        return _GoalEvent(order=order, objective=None)
    goal = payload.get("goal")
    if not isinstance(goal, dict):
        return None
    objective = goal.get("objective")
    if not isinstance(objective, str):
        return None
    objective = objective.strip()
    return _GoalEvent(order=order, objective=objective or None)


def _task_plan_event_from_line(
    raw: str, *, thread_id: str, fallback_order: int
) -> _TaskPlanEvent | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        event = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(event, dict):
        return None
    return _task_plan_event_from_event(
        event,
        thread_id=thread_id,
        fallback_order=fallback_order,
    )


def _task_plan_event_from_event(
    event: dict[str, Any], *, thread_id: str, fallback_order: int
) -> _TaskPlanEvent | None:
    if event.get("method") != TASK_PLAN_UPDATED_METHOD:
        return None
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return None
    payload_thread_id = _payload_thread_id(payload)
    if payload_thread_id is not None and payload_thread_id != thread_id:
        return None
    order = _event_order(event, fallback_order)
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
        order,
    )
    if not snapshot.explanation and not snapshot.steps and plan:
        return None
    return _TaskPlanEvent(order=order, snapshot=snapshot)


def _task_plan_snapshot(
    explanation: str,
    plan: list[Any],
    order: tuple[int, int, int],
) -> TaskPlanSnapshot:
    steps = tuple(
        step
        for raw_step in plan
        if (step := _task_plan_step(raw_step)) is not None
    )
    explanation = explanation.strip()
    return TaskPlanSnapshot(explanation=explanation, steps=steps, order=order)


def _task_plan_step(raw_step: Any) -> TaskPlanStep | None:
    if not isinstance(raw_step, dict):
        return None
    step = raw_step.get("step")
    if not isinstance(step, str):
        return None
    step = step.strip()
    if not step:
        return None
    return TaskPlanStep(
        step=step,
        status=_normalize_task_plan_status(raw_step.get("status")),
    )


def _normalize_task_plan_status(status: Any) -> str:
    if not isinstance(status, str):
        return "pending"
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
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    return None


def _payload_thread_id(payload: dict[str, Any]) -> str | None:
    thread_id = payload.get("threadId")
    if isinstance(thread_id, str):
        return thread_id
    thread_id = payload.get("thread_id")
    if isinstance(thread_id, str):
        return thread_id
    return None
