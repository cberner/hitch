"""Helpers for reading Hitch's per-worker Codex event logs."""

from __future__ import annotations

import json
import logging
import re
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
ITEM_COMPLETED_METHOD = "item/completed"

_GITHUB_PR_URL_RE = re.compile(
    r"https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/pull/([0-9]+)"
)
_PR_INFO_TOOLS = frozenset(
    {
        "create_pull_request",
        "create_pr",
        "get_pr_info",
        "fetch_pr",
        "merge_pull_request",
    }
)
_PR_COMMENT_TOOLS = frozenset({"fetch_pr_comments"})
_PR_THREAD_TOOLS = frozenset({"list_pull_request_review_threads"})
_PR_REVIEW_TOOLS = frozenset({"list_pull_request_reviews"})
_PR_REACTION_TOOLS = frozenset({"get_pr_reactions"})
_CI_STATUS_TOOLS = frozenset(
    {
        "get_commit_combined_status",
        "fetch_commit_workflow_runs",
        "fetch_workflow_run_jobs",
    }
)
_SUCCESS_CONCLUSIONS = frozenset({"success", "neutral", "skipped"})
_FAILURE_CONCLUSIONS = frozenset(
    {"action_required", "cancelled", "failure", "startup_failure", "timed_out"}
)
_PR_TEXT_MAX_CHARS = 500
_PR_DETAIL_LIMIT = 5


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


@dataclass(frozen=True)
class _PrSnapshotUpdate:
    order: tuple[int, int, int]
    values: dict[str, Any]


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
    current = _latest_goal_event_from_event_paths(paths, thread_id=thread_id)
    return current.objective if current is not None else None


def latest_goal_tokens_for_instance(instance: CodexInstance | None) -> int | None:
    if instance is None or not instance.events_path:
        return None
    return latest_goal_tokens_from_event_paths(
        [instance.events_path],
        thread_id=instance.thread_id,
    )


def latest_goal_tokens_from_event_paths(
    paths: Iterable[str | Path], *, thread_id: str
) -> int | None:
    current = _latest_goal_event_from_event_paths(paths, thread_id=thread_id)
    return current.tokens_used if current is not None else None


def _latest_goal_event_from_event_paths(
    paths: Iterable[str | Path], *, thread_id: str
) -> _GoalEvent | None:
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
    return current


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


def latest_pr_snapshot_for_instance(instance: CodexInstance | None) -> dict[str, Any] | None:
    """Return the latest GitHub PR state observed by a worker, if any."""
    if instance is None or not instance.events_path:
        return None
    return latest_pr_snapshot_from_event_paths(
        [instance.events_path],
        thread_id=instance.thread_id,
    )


def latest_pr_snapshot_from_event_paths(
    paths: Iterable[str | Path], *, thread_id: str
) -> dict[str, Any] | None:
    """Recover a compact PR handoff snapshot from completed GitHub MCP calls.

    The PR workflow's visible turn already checks GitHub via MCP tools. Persisting
    the latest structured results lets Hitch continue follow-up from durable state
    instead of asking the next agent to rediscover which PR/branch it was handling.
    """
    updates: list[_PrSnapshotUpdate] = []
    fallback_order = 0
    for raw_path in paths:
        if not raw_path:
            continue
        path = Path(raw_path)
        try:
            with path.open("r", encoding="utf-8") as fh:
                for raw in fh:
                    fallback_order += 1
                    update = _pr_snapshot_update_from_line(
                        raw,
                        thread_id=thread_id,
                        fallback_order=fallback_order,
                    )
                    if update is not None:
                        updates.append(update)
        except FileNotFoundError:
            continue
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("failed to read Codex events %s: %s", path, exc)
            continue
    if not updates:
        return None

    snapshot: dict[str, Any] = {}
    for update in sorted(updates, key=lambda item: item.order):
        _merge_pr_snapshot_update(snapshot, update)
    return _finalize_pr_snapshot(snapshot)


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
        return _GoalEvent(order=order, objective=None, tokens_used=None)
    goal = payload.get("goal")
    if not isinstance(goal, dict):
        return None
    objective = goal.get("objective")
    if not isinstance(objective, str):
        return None
    objective = objective.strip()
    return _GoalEvent(
        order=order,
        objective=objective or None,
        tokens_used=_goal_tokens_used(goal),
    )


def _goal_tokens_used(goal: dict[str, Any]) -> int | None:
    for key in ("tokensUsed", "tokens_used"):
        value = goal.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return max(0, value)
    return None


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


def _pr_snapshot_update_from_line(
    raw: str, *, thread_id: str, fallback_order: int
) -> _PrSnapshotUpdate | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        event = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(event, dict):
        return None
    return _pr_snapshot_update_from_event(
        event,
        thread_id=thread_id,
        fallback_order=fallback_order,
    )


def _pr_snapshot_update_from_event(
    event: dict[str, Any], *, thread_id: str, fallback_order: int
) -> _PrSnapshotUpdate | None:
    if event.get("method") != ITEM_COMPLETED_METHOD:
        return None
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return None
    payload_thread_id = _payload_thread_id(payload)
    if payload_thread_id is not None and payload_thread_id != thread_id:
        return None
    item = payload.get("item")
    if not isinstance(item, dict) or item.get("type") != "mcpToolCall":
        return None
    tool = _normalized_github_tool(item)
    if not tool:
        return None

    values: dict[str, Any] = {"last_observed_at": _event_observed_at(event)}
    _copy_pr_identity_from_args(values, item.get("arguments"))
    result_values = _mcp_result_values(item.get("result"))
    if tool in _PR_INFO_TOOLS:
        for result_value in result_values:
            _copy_pr_info_fields(values, result_value)
        if "source_tool" not in values:
            values["source_tool"] = tool
    elif tool in _PR_COMMENT_TOOLS:
        for result_value in result_values:
            _copy_comment_fields(values, result_value)
    elif tool in _PR_THREAD_TOOLS:
        for result_value in result_values:
            _copy_review_thread_fields(values, result_value)
    elif tool in _PR_REVIEW_TOOLS:
        for result_value in result_values:
            _copy_review_fields(values, result_value)
    elif tool in _PR_REACTION_TOOLS:
        for result_value in result_values:
            _copy_reaction_fields(values, result_value)
    elif tool in _CI_STATUS_TOOLS:
        _copy_ci_fields(values, tool, item.get("arguments"), result_values)
    else:
        return None

    if len(values) <= 1:
        return None
    return _PrSnapshotUpdate(
        order=_event_order(event, fallback_order),
        values=values,
    )


def _normalized_github_tool(item: dict[str, Any]) -> str:
    raw_tool = item.get("tool")
    tool = raw_tool if isinstance(raw_tool, str) else ""
    raw_server = item.get("server")
    server = raw_server if isinstance(raw_server, str) else ""
    normalized = tool.lower().replace("-", "_")
    if normalized.startswith("github_"):
        return normalized.removeprefix("github_").lstrip("_")
    if "github" in server.lower():
        return normalized.lstrip("_")
    return ""


def _copy_pr_identity_from_args(target: dict[str, Any], raw_args: Any) -> None:
    if not isinstance(raw_args, dict):
        return
    repo = _string_from_any(
        raw_args.get("repository_full_name") or raw_args.get("repo_full_name")
    )
    if repo:
        target["repository_full_name"] = repo
    number = _positive_int(raw_args.get("pr_number") or raw_args.get("pull_number"))
    if number is not None:
        target["pr_number"] = number
    commit_sha = _string_from_any(raw_args.get("commit_sha"))
    if commit_sha:
        target["latest_commit_sha"] = commit_sha


def _mcp_result_values(raw_result: Any) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    if isinstance(raw_result, dict):
        for key in ("structuredContent", "structured_content"):
            structured = raw_result.get(key)
            if isinstance(structured, dict):
                values.append(structured)
        content = raw_result.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                parsed = _json_dict_from_text(item.get("text"))
                if parsed is not None:
                    values.append(parsed)
        if not values:
            values.append(raw_result)
    else:
        parsed = _json_dict_from_text(raw_result)
        if parsed is not None:
            values.append(parsed)
    return values


def _json_dict_from_text(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _copy_pr_info_fields(target: dict[str, Any], source: dict[str, Any]) -> None:
    url = _string_from_any(source.get("display_url") or source.get("url"))
    if url:
        target["url"] = url
        _copy_identity_from_pr_url(target, url)
    number = _positive_int(source.get("number") or source.get("pr_number"))
    if number is not None:
        target["pr_number"] = number
    for source_key, target_key in (
        ("state", "state"),
        ("title", "title"),
        ("base", "base"),
        ("base_sha", "base_sha"),
        ("head", "head"),
        ("head_sha", "head_sha"),
        ("merge_commit_sha", "merge_commit_sha"),
        ("created_at", "created_at"),
        ("updated_at", "updated_at"),
        ("closed_at", "closed_at"),
        ("merged_at", "merged_at"),
    ):
        text = _string_from_any(source.get(source_key))
        if text:
            target[target_key] = _compact_text(text)
    for source_key, target_key in (
        ("merged", "merged"),
        ("mergeable", "mergeable"),
        ("draft", "draft"),
    ):
        value = source.get(source_key)
        if isinstance(value, bool):
            target[target_key] = value
    repo = _string_from_any(
        source.get("repository_full_name") or source.get("repo_full_name")
    )
    if repo:
        target["repository_full_name"] = repo


def _copy_comment_fields(target: dict[str, Any], source: dict[str, Any]) -> None:
    url = _string_from_any(source.get("display_url") or source.get("url"))
    if url:
        target["url"] = url
        _copy_identity_from_pr_url(target, url)
    comments = source.get("comments")
    if isinstance(comments, list):
        target["comment_count"] = len(comments)
        target["latest_comments"] = _compact_items(comments)


def _copy_review_thread_fields(target: dict[str, Any], source: dict[str, Any]) -> None:
    threads = source.get("review_threads")
    if not isinstance(threads, list):
        return
    unresolved = [
        thread
        for thread in threads
        if isinstance(thread, dict)
        and thread.get("is_resolved") is not True
        and thread.get("is_outdated") is not True
    ]
    target["review_thread_count"] = len(threads)
    target["unresolved_thread_count"] = len(unresolved)
    target["unresolved_threads"] = _compact_items(unresolved)


def _copy_review_fields(target: dict[str, Any], source: dict[str, Any]) -> None:
    reviews = source.get("reviews")
    if not isinstance(reviews, list):
        return
    states = [
        state.upper()
        for review in reviews
        if isinstance(review, dict)
        and isinstance((state := review.get("state")), str)
        and state
    ]
    target["review_count"] = len(reviews)
    if "CHANGES_REQUESTED" in states:
        target["review_signal"] = "changes_requested"
    elif "APPROVED" in states:
        target["review_signal"] = "approved"
    elif states:
        target["review_signal"] = "commented"
    else:
        # ``""`` is the explicit review-signal clear; see _merge_pr_snapshot_update.
        target["review_signal"] = ""


def _copy_reaction_fields(target: dict[str, Any], source: dict[str, Any]) -> None:
    reactions = source.get("reactions")
    if not isinstance(reactions, list):
        return
    contents = [
        content
        for reaction in reactions
        if isinstance(reaction, dict)
        and isinstance((content := reaction.get("content")), str)
    ]
    target["reaction_count"] = len(reactions)
    if "+1" in contents and target.get("review_signal") != "changes_requested":
        target["review_signal"] = "thumbs_up"


def _copy_ci_fields(
    target: dict[str, Any],
    tool: str,
    raw_args: Any,
    result_values: list[dict[str, Any]],
) -> None:
    _copy_pr_identity_from_args(target, raw_args)
    for source in result_values:
        if tool == "get_commit_combined_status":
            status = _ci_status_from_statuses(source.get("statuses"))
            if status:
                target["ci_status"] = status
        elif tool == "fetch_commit_workflow_runs":
            status = _ci_status_from_runs(source.get("workflow_runs"))
            if status:
                target["ci_status"] = status
        elif tool == "fetch_workflow_run_jobs":
            status, failing_jobs, pending_jobs = _ci_status_from_jobs(source.get("jobs"))
            if status:
                # Empty ``failing_jobs``/``pending_jobs`` are explicit "observed
                # and found none" overwrites that ``_merge_pr_snapshot_update``
                # propagates so a clean re-observation clears any stale list
                # left behind by an earlier failing or pending observation in
                # the same turn. Without this, a flaky test that fails on the
                # first ``fetch_workflow_run_jobs`` call and passes on a later
                # one leaves ``failing_jobs`` pointing at the resolved job --
                # the same shape b90ceed and 48b0840 fixed at the merge layer.
                target["ci_status"] = status
                target["failing_jobs"] = failing_jobs
                target["pending_jobs"] = pending_jobs


def _ci_status_from_statuses(raw_statuses: Any) -> str:
    if not isinstance(raw_statuses, list):
        return ""
    if not raw_statuses:
        return "unknown"
    states = [
        state.lower()
        for status in raw_statuses
        if isinstance(status, dict)
        and isinstance((state := status.get("state")), str)
    ]
    if any(state in {"failure", "error"} for state in states):
        return "failure"
    if any(state in {"pending", "queued", "in_progress"} for state in states):
        return "pending"
    if states and all(state == "success" for state in states):
        return "success"
    return "unknown"


def _ci_status_from_runs(raw_runs: Any) -> str:
    # Match the failure-over-pending precedence used by ``_ci_status_from_statuses``
    # and ``_ci_status_from_jobs``: a confirmed failure on one workflow run must
    # surface even while a sibling run is still in progress, so the PR follow-up
    # agent and the snapshot UI both see the break instead of treating the PR as
    # healthy-but-waiting until the slowest workflow finishes.
    if not isinstance(raw_runs, list):
        return ""
    if not raw_runs:
        return "unknown"
    has_failure = False
    has_pending = False
    saw_completed = False
    for run in raw_runs:
        if not isinstance(run, dict):
            continue
        status = _string_from_any(run.get("status")).lower()
        conclusion = _string_from_any(run.get("conclusion")).lower()
        if status != "completed":
            has_pending = True
            continue
        saw_completed = True
        if conclusion in _FAILURE_CONCLUSIONS or (
            conclusion and conclusion not in _SUCCESS_CONCLUSIONS
        ):
            has_failure = True
    if has_failure:
        return "failure"
    if has_pending:
        return "pending"
    return "success" if saw_completed else "unknown"


def _ci_status_from_jobs(raw_jobs: Any) -> tuple[str, list[str], list[str]]:
    # Mirror the "must have observed at least one completed job" guard the
    # ``_ci_status_from_runs`` precedence fix added: a non-empty ``jobs``
    # list whose every entry is filtered out (e.g. every item is not a dict
    # because the upstream payload was malformed) leaves both ``failing``
    # and ``pending`` empty, so the prior unconditional ``"success"`` would
    # falsely mark the PR CI gate as green for the follow-up agent and the
    # snapshot UI even though no job actually succeeded.
    if not isinstance(raw_jobs, list):
        return "", [], []
    if not raw_jobs:
        return "unknown", [], []
    failing: list[str] = []
    pending: list[str] = []
    saw_completed = False
    for job in raw_jobs:
        if not isinstance(job, dict):
            continue
        name = _string_from_any(job.get("name")) or "unnamed job"
        status = _string_from_any(job.get("status")).lower()
        conclusion = _string_from_any(job.get("conclusion")).lower()
        if status != "completed":
            pending.append(name)
            continue
        saw_completed = True
        if conclusion in _FAILURE_CONCLUSIONS or (
            conclusion and conclusion not in _SUCCESS_CONCLUSIONS
        ):
            failing.append(name)
    if failing:
        return "failure", failing[:_PR_DETAIL_LIMIT], pending[:_PR_DETAIL_LIMIT]
    if pending:
        return "pending", [], pending[:_PR_DETAIL_LIMIT]
    return ("success", [], []) if saw_completed else ("unknown", [], [])


def _merge_pr_snapshot_update(
    snapshot: dict[str, Any], update: _PrSnapshotUpdate
) -> None:
    values = dict(update.values)
    if _pr_snapshot_identity_changed(snapshot, values):
        snapshot.clear()
    for key, value in values.items():
        # ``None`` is an "absent" sentinel; empty list/dict is an explicit
        # "observed and found none" clear. ``""`` is "absent" for every
        # string field except ``review_signal``, which uses it as an
        # explicit clear so a clean reviews re-observation drops a stale
        # ``changes_requested`` left over from an earlier observation. The
        # clear is recorded as ``""`` in the snapshot (not popped) so it
        # propagates through ``system_agents._merge_pr_handoff_dicts``
        # cross-worker. Reaction-derived ``thumbs_up`` is held back from
        # the clear since the reviews tool does not speak for it.
        if value is None:
            continue
        if value == "":
            if key == "review_signal" and snapshot.get(key) != "thumbs_up":
                snapshot[key] = ""
            continue
        if (
            key == "review_signal"
            and value == "thumbs_up"
            and snapshot.get("review_signal") == "changes_requested"
        ):
            continue
        snapshot[key] = value


def _pr_snapshot_identity_changed(
    current: dict[str, Any], update: dict[str, Any]
) -> bool:
    if not current:
        return False
    for key in ("repository_full_name", "url"):
        current_value = current.get(key)
        update_value = update.get(key)
        if (
            isinstance(current_value, str)
            and current_value
            and isinstance(update_value, str)
            and update_value
            and current_value != update_value
        ):
            return True
    current_number = current.get("pr_number")
    update_number = update.get("pr_number")
    return (
        isinstance(current_number, int)
        and not isinstance(current_number, bool)
        and isinstance(update_number, int)
        and not isinstance(update_number, bool)
        and current_number != update_number
    )


def _finalize_pr_snapshot(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    if not snapshot:
        return None
    url = _string_from_any(snapshot.get("url"))
    if url:
        _copy_identity_from_pr_url(snapshot, url)
    if not url and not (
        isinstance(snapshot.get("repository_full_name"), str)
        and isinstance(snapshot.get("pr_number"), int)
    ):
        return None
    return snapshot


def _copy_identity_from_pr_url(target: dict[str, Any], url: str) -> None:
    match = _GITHUB_PR_URL_RE.search(url)
    if match is None:
        return
    owner, repo, number = match.groups()
    target["repository_full_name"] = f"{owner}/{repo}"
    target["pr_number"] = int(number)


def _compact_items(items: list[Any]) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        compact: dict[str, Any] = {}
        for key in (
            "id",
            "path",
            "line",
            "original_line",
            "state",
            "submitted_at",
            "created_at",
            "updated_at",
            "url",
        ):
            value = item.get(key)
            if isinstance(value, str) and value:
                compact[key] = _compact_text(value)
            elif isinstance(value, int) and not isinstance(value, bool):
                compact[key] = value
        body = _string_from_any(item.get("body"))
        if body:
            compact["body"] = _compact_text(" ".join(body.split()))
        if compact:
            compacted.append(compact)
        if len(compacted) >= _PR_DETAIL_LIMIT:
            break
    return compacted


def _compact_text(value: str) -> str:
    text = value.strip()
    if len(text) <= _PR_TEXT_MAX_CHARS:
        return text
    return f"{text[: _PR_TEXT_MAX_CHARS - 3].rstrip()}..."


def _string_from_any(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.isdecimal():
        parsed = int(value)
        return parsed if parsed > 0 else None
    return None


def _event_observed_at(event: dict[str, Any]) -> int:
    recorded_at = _int_field(event, "recordedAt")
    if recorded_at is None:
        recorded_at = _int_field(event, "recorded_at")
    if recorded_at is not None:
        return recorded_at
    event_seq = _int_field(event, "eventSeq")
    return event_seq or 0


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
