"""Helpers for reading Hitch's per-worker Codex event logs."""

from __future__ import annotations

import json
import logging
import re
import shlex
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
    r"(?=$|[?#\s'\"),\]}])"
)
_GH_REPO_SELECTOR_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_GH_JQ_OBJECT_FIELD_RE = re.compile(
    r"(?P<shorthand>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<key>[\"']?[A-Za-z_][A-Za-z0-9_]*[\"']?)"
    r"\s*:\s*(?P<source>\.?[A-Za-z_][A-Za-z0-9_]*)"
)
_SHELL_ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*")
_SHELL_COMMAND_SEPARATORS = frozenset({";", "|", "&", "\n"})
_SHELL_CONTROL_OPERATORS = _SHELL_COMMAND_SEPARATORS | {"&&", "||"}
_GH_JQ_IDENTITY_FIELDS = frozenset({"display_url", "number", "pr_number", "url"})
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


@dataclass(frozen=True)
class _GhPrCommand:
    source_tool: str
    repository_full_name: str
    pr_number: int | None = None
    url: str = ""
    jq_expression: str = ""
    template_expression: str = ""
    requires_identity_output: bool = False


@dataclass(frozen=True)
class PrObservationTurn:
    """A completed session turn with the PR-observation items it produced."""

    is_pr_prompt: bool
    is_completed: bool
    items: tuple[dict[str, Any], ...]
    has_lifecycle_activity: bool = False


@dataclass(frozen=True)
class PrObservationResult:
    """The current PR epoch after replaying completed session turns."""

    snapshot: dict[str, Any] | None
    superseded_by_lifecycle: bool = False


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
    """Recover a compact PR handoff snapshot from completed GitHub calls.

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

    return _pr_snapshot_from_updates(sorted(updates, key=lambda item: item.order))


def pr_snapshot_from_completed_mcp_items(
    items: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    """Recover the latest GitHub PR state from completed MCP tool-call items."""
    updates = _pr_snapshot_updates_from_completed_items(items)
    return _pr_snapshot_from_updates(updates)


def _pr_snapshot_updates_from_completed_items(
    items: Iterable[dict[str, Any]],
) -> list[_PrSnapshotUpdate]:
    updates: list[_PrSnapshotUpdate] = []
    for fallback_order, item in enumerate(items, start=1):
        update = _pr_snapshot_update_from_completed_item(
            item,
            order=(0, 0, fallback_order),
            observed_at="",
        )
        if update is not None:
            updates.append(update)
    return updates


def _pr_snapshot_from_updates(
    updates: Iterable[_PrSnapshotUpdate],
) -> dict[str, Any] | None:
    snapshot: dict[str, Any] = {}
    for update in updates:
        _merge_pr_snapshot_update(snapshot, update)
    return _finalize_pr_snapshot(snapshot)


def pr_snapshot_from_observation_turns(
    turns: Iterable[PrObservationTurn],
) -> dict[str, Any] | None:
    return pr_observation_result_from_turns(turns).snapshot


def pr_observation_result_from_turns(
    turns: Iterable[PrObservationTurn],
) -> PrObservationResult:
    """Recover PR state with completed /pr turns acting as lifecycle boundaries.

    A completed ``/pr`` turn establishes the current PR epoch. Later normal
    turns may refresh that same PR, but unrelated or unsupported MCP calls do
    not keep the epoch alive; a completed normal lifecycle turn with no accepted
    PR observation clears it.
    """
    updates_since_clear: list[_PrSnapshotUpdate] = []
    current_snapshot: dict[str, Any] | None = None
    superseded_by_lifecycle = False
    for turn in turns:
        turn_updates = _pr_snapshot_updates_from_completed_items(turn.items)
        if not turn.is_pr_prompt:
            if not _pr_snapshot_has_identity(current_snapshot):
                continue
            # Accept updates sequentially against a working copy so an earlier
            # update in the turn that records the PR's run ids (a commit-
            # correlated ``fetch_commit_workflow_runs``) lets a later
            # ``fetch_workflow_run_jobs`` update in the same turn correlate by
            # run id.
            working = dict(current_snapshot) if current_snapshot else {}
            accepted_updates = []
            for update in turn_updates:
                if _pr_update_belongs_to_current_pr(working, update):
                    accepted_updates.append(update)
                    _merge_pr_snapshot_update(working, update)
            if (
                not accepted_updates
                and turn.is_completed
                and turn.has_lifecycle_activity
            ):
                updates_since_clear = []
                current_snapshot = None
                superseded_by_lifecycle = True
                continue
            updates_since_clear.extend(accepted_updates)
            current_snapshot = _pr_snapshot_from_updates(updates_since_clear)
            continue
        if not turn.is_completed:
            continue
        turn_snapshot = _pr_snapshot_from_updates(turn_updates)
        if not _pr_snapshot_has_identity(turn_snapshot):
            updates_since_clear = []
            current_snapshot = None
            superseded_by_lifecycle = True
            continue
        updates_since_clear.extend(turn_updates)
        current_snapshot = _pr_snapshot_from_updates(updates_since_clear)
        superseded_by_lifecycle = False
    return PrObservationResult(
        snapshot=current_snapshot,
        superseded_by_lifecycle=superseded_by_lifecycle,
    )


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
    if not isinstance(item, dict):
        return None
    if item.get("type") == "commandExecution":
        return _pr_snapshot_update_from_completed_command_item(
            item,
            order=_event_order(event, fallback_order),
            observed_at=_event_observed_at(event),
        )
    if item.get("type") != "mcpToolCall":
        return None
    return _pr_snapshot_update_from_completed_item(
        item,
        order=_event_order(event, fallback_order),
        observed_at=_event_observed_at(event),
    )


def _pr_snapshot_update_from_completed_item(
    item: dict[str, Any], *, order: tuple[int, int, int], observed_at: Any
) -> _PrSnapshotUpdate | None:
    if item.get("type") != "mcpToolCall":
        return None
    tool = _normalized_github_tool(item)
    if not tool:
        return None

    values: dict[str, Any] = {"last_observed_at": observed_at}
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
        order=order,
        values=values,
    )


def _pr_snapshot_update_from_completed_command_item(
    item: dict[str, Any], *, order: tuple[int, int, int], observed_at: Any
) -> _PrSnapshotUpdate | None:
    if not _command_succeeded(item):
        return None
    command = _gh_pr_command(item)
    if command is None:
        return None
    values: dict[str, Any] = {
        "last_observed_at": observed_at,
        "source_tool": command.source_tool,
    }
    if command.repository_full_name:
        values["repository_full_name"] = command.repository_full_name
    if command.pr_number is not None:
        values["pr_number"] = command.pr_number
    if command.url:
        values["url"] = command.url
    result_values = _gh_pr_result_values(item.get("aggregatedOutput"), command=command)
    if not result_values:
        return None
    for result_value in result_values:
        next_values = dict(values)
        _copy_pr_info_fields(next_values, result_value)
        if _pr_snapshot_identity_changed(values, next_values):
            return None
        values = next_values
    if not _pr_snapshot_has_identity(_finalize_pr_snapshot(dict(values))):
        return None
    return _PrSnapshotUpdate(order=order, values=values)


def _normalized_github_tool(item: dict[str, Any]) -> str:
    raw_tool = item.get("tool")
    tool = raw_tool if isinstance(raw_tool, str) else ""
    raw_server = item.get("server")
    server = raw_server if isinstance(raw_server, str) else ""
    normalized = tool.lower().replace("-", "_")
    if normalized.startswith("mcp__codex_apps__github_"):
        return normalized.removeprefix("mcp__codex_apps__github_").lstrip("_")
    if normalized.startswith("github_"):
        return normalized.removeprefix("github_").lstrip("_")
    if "github" in server.lower():
        return normalized.lstrip("_")
    return ""


def _command_succeeded(item: dict[str, Any]) -> bool:
    if item.get("status") != "completed":
        return False
    exit_code = item.get("exitCode")
    return isinstance(exit_code, int) and not isinstance(exit_code, bool) and exit_code == 0


def _gh_pr_command(item: dict[str, Any]) -> _GhPrCommand | None:
    command = item.get("command")
    if isinstance(command, str):
        return _gh_pr_command_from_text(command)
    return None


def _gh_pr_command_from_text(command: str) -> _GhPrCommand | None:
    for tokens in _command_token_sequences(command):
        parsed = _gh_pr_command_from_tokens(tokens)
        if parsed is not None:
            return parsed
    return None


def _command_token_sequences(command: str) -> Iterable[list[str]]:
    try:
        tokens = _shell_tokens(command)
    except ValueError:
        return
    if not tokens:
        return
    yield tokens
    shell_script = _shell_script_arg(tokens)
    if shell_script:
        try:
            yield _shell_tokens(shell_script)
        except ValueError:
            return


def _shell_tokens(command: str) -> list[str]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|\n")
    lexer.whitespace_split = True
    lexer.commenters = ""
    lexer.whitespace = " \t\r"
    return list(lexer)


def _shell_script_arg(tokens: list[str]) -> str:
    idx = _shell_script_command_index(tokens)
    if idx >= len(tokens) or Path(tokens[idx]).name not in {"bash", "sh"}:
        return ""
    for token_idx, token in enumerate(tokens[idx:-1], start=idx):
        if token in {"-c", "-lc"}:
            return tokens[token_idx + 1]
    return ""


def _shell_script_command_index(tokens: list[str]) -> int:
    env_command = _env_command_index_and_assignments(tokens)
    return env_command[0] if env_command is not None else 0


def _env_command_index_and_assignments(
    tokens: list[str], idx: int = 0
) -> tuple[int, list[str]] | None:
    if idx >= len(tokens) or Path(tokens[idx]).name != "env":
        return None
    idx += 1
    assignments: list[str] = []
    while idx < len(tokens):
        token = tokens[idx]
        if token == "--":
            return idx + 1, assignments
        if _SHELL_ASSIGNMENT_RE.fullmatch(token):
            assignments.append(token)
            idx += 1
            continue
        if token in {"-0", "-i", "--ignore-environment", "--null"}:
            idx += 1
            continue
        if token in {"-C", "-S", "-u", "--chdir", "--split-string", "--unset"}:
            idx += 2
            continue
        if token.startswith(("--chdir=", "--split-string=", "--unset=")):
            idx += 1
            continue
        return idx, assignments
    return idx, assignments


def _gh_pr_command_from_tokens(tokens: list[str]) -> _GhPrCommand | None:
    for gh_idx, token in enumerate(tokens):
        if Path(token).name != "gh":
            continue
        if not _is_shell_command_position(tokens, gh_idx):
            continue
        if gh_idx + 1 >= len(tokens) or tokens[gh_idx + 1] != "pr":
            continue
        repository_full_name = _gh_repo_from_command_prefix(tokens, gh_idx)
        parsed = _gh_pr_subcommand_from_tokens(
            tokens[gh_idx + 2 :],
            repository_full_name=repository_full_name,
            requires_identity_output=_has_later_untrusted_output_command(
                tokens, gh_idx
            ),
        )
        if parsed is not None:
            return parsed
    return None


def _is_shell_command_position(tokens: list[str], idx: int) -> bool:
    return _gh_command_assignment_tokens(tokens, idx) is not None


def _gh_command_assignment_tokens(tokens: list[str], idx: int) -> list[str] | None:
    assignments = _shell_command_assignment_tokens(_shell_command_prefix(tokens, idx))
    if assignments is not None:
        return assignments
    prefix = _conditional_shell_command_prefix(tokens, idx)
    if prefix is None or _has_later_shell_command(tokens, idx):
        return None
    return _shell_command_assignment_tokens(prefix)


def _shell_command_prefix(tokens: list[str], idx: int) -> list[str]:
    start = idx
    while start > 0 and tokens[start - 1] not in _SHELL_COMMAND_SEPARATORS:
        start -= 1
    return tokens[start:idx]


def _conditional_shell_command_prefix(tokens: list[str], idx: int) -> list[str] | None:
    prefix = _shell_command_prefix(tokens, idx)
    try:
        and_idx = len(prefix) - 1 - list(reversed(prefix)).index("&&")
    except ValueError:
        return None
    return prefix[and_idx + 1 :]


def _has_later_shell_command(tokens: list[str], idx: int) -> bool:
    return any(
        token in _SHELL_CONTROL_OPERATORS
        and not _is_redirection_ampersand(tokens, token_idx)
        for token_idx, token in enumerate(tokens[idx + 1 :], start=idx + 1)
    )


def _has_later_untrusted_output_command(tokens: list[str], idx: int) -> bool:
    for token_idx, token in enumerate(tokens[idx + 1 :], start=idx + 1):
        if token not in _SHELL_CONTROL_OPERATORS:
            continue
        if _is_redirection_ampersand(tokens, token_idx):
            continue
        if token in {"&&", "||"} and not _is_output_fallback_command(
            tokens, token_idx + 1
        ):
            continue
        return True
    return False


def _is_output_fallback_command(tokens: list[str], idx: int) -> bool:
    if idx >= len(tokens):
        return False
    fallback_command = Path(tokens[idx]).name
    return fallback_command not in {":", "exit", "false", "return", "true"}


def _is_redirection_ampersand(tokens: list[str], idx: int) -> bool:
    if tokens[idx] != "&" or idx == 0 or idx + 1 >= len(tokens):
        return False
    return tokens[idx - 1].endswith(">") and (
        tokens[idx + 1] == "-" or tokens[idx + 1].isdecimal()
    )


def _shell_command_assignment_tokens(prefix: list[str]) -> list[str] | None:
    assignments: list[str] = []
    idx = 0
    while idx < len(prefix) and _SHELL_ASSIGNMENT_RE.fullmatch(prefix[idx]):
        assignments.append(prefix[idx])
        idx += 1
    if idx == len(prefix):
        return assignments
    env_command = _env_command_index_and_assignments(prefix, idx)
    if env_command is None:
        return None
    command_idx, env_assignments = env_command
    if command_idx != len(prefix):
        return None
    assignments.extend(env_assignments)
    return assignments


def _gh_repo_from_command_prefix(tokens: list[str], idx: int) -> str:
    assignments = _gh_command_assignment_tokens(tokens, idx)
    for assignment in assignments or []:
        name, value = assignment.split("=", 1)
        if name == "GH_REPO" and value:
            return _normalize_gh_repo_selector(value)
    return ""


def _gh_pr_subcommand_from_tokens(
    tokens: list[str],
    *,
    repository_full_name: str = "",
    requires_identity_output: bool = False,
) -> _GhPrCommand | None:
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token in {"create", "new", "view"}:
            args = tokens[idx + 1 :]
            parsed = _parse_gh_pr_args(args)
            repository_full_name = parsed.get("repository_full_name") or repository_full_name
            number = _positive_int(parsed.get("pr_number"))
            source_tool = "gh_pr_view" if token == "view" else "gh_pr_create"
            return _GhPrCommand(
                source_tool=source_tool,
                repository_full_name=repository_full_name,
                pr_number=number,
                url=parsed.get("url") or "",
                jq_expression=parsed.get("jq_expression") or "",
                template_expression=parsed.get("template_expression") or "",
                requires_identity_output=requires_identity_output,
            )
        if token in {"-R", "--repo"} and idx + 1 < len(tokens):
            repository_full_name = _normalize_gh_repo_selector(tokens[idx + 1])
            idx += 2
            continue
        if token.startswith("--repo="):
            repository_full_name = _normalize_gh_repo_selector(token.split("=", 1)[1])
            idx += 1
            continue
        if token.startswith("-R") and token != "-R":
            repository_full_name = _normalize_gh_repo_selector(token[2:])
            idx += 1
            continue
        return None
    return None


def _parse_gh_pr_args(tokens: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    positional: list[str] = []
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token in _SHELL_CONTROL_OPERATORS:
            break
        if token in {"-R", "--repo"} and idx + 1 < len(tokens):
            values["repository_full_name"] = _normalize_gh_repo_selector(tokens[idx + 1])
            idx += 2
            continue
        if token.startswith("--repo="):
            values["repository_full_name"] = _normalize_gh_repo_selector(
                token.split("=", 1)[1]
            )
            idx += 1
            continue
        if token.startswith("-R") and token != "-R":
            values["repository_full_name"] = _normalize_gh_repo_selector(token[2:])
            idx += 1
            continue
        if token in {"-q", "--jq"} and idx + 1 < len(tokens):
            values["jq_expression"] = tokens[idx + 1]
            idx += 2
            continue
        if token.startswith("--jq="):
            values["jq_expression"] = token.split("=", 1)[1]
            idx += 1
            continue
        if token.startswith("-q") and token != "-q":
            values["jq_expression"] = token[2:]
            idx += 1
            continue
        if token in {"-T", "--template"} and idx + 1 < len(tokens):
            values["template_expression"] = tokens[idx + 1]
            idx += 2
            continue
        if token.startswith("--template="):
            values["template_expression"] = token.split("=", 1)[1]
            idx += 1
            continue
        if token.startswith("-T") and token != "-T":
            values["template_expression"] = token[2:]
            idx += 1
            continue
        if token.startswith("-"):
            if (
                "=" not in token
                and _gh_pr_option_takes_value(token)
                and idx + 1 < len(tokens)
            ):
                idx += 2
            else:
                idx += 1
            continue
        positional.append(token)
        idx += 1
    for value in positional:
        match = _GITHUB_PR_URL_RE.search(value)
        if match is not None:
            values["url"] = match.group(0)
            owner, repo, number = match.groups()
            values["repository_full_name"] = f"{owner}/{repo}"
            values["pr_number"] = number
            return values
        if "pr_number" not in values and value.isdecimal():
            values["pr_number"] = value
    return values


def _gh_pr_option_takes_value(token: str) -> bool:
    name = token.split("=", 1)[0]
    return name in {
        "-R",
        "--repo",
        "--json",
        "--jq",
        "--template",
        "--head",
        "--base",
        "--title",
        "--body",
        "--body-file",
        "--state",
        "--author",
        "--assignee",
        "--label",
        "--milestone",
        "--project",
        "--recover",
        "--reviewer",
        "--search",
        "--app",
        "--limit",
        "-a",
        "-B",
        "-H",
        "-l",
        "-m",
        "-p",
        "-r",
        "-t",
        "-b",
        "-F",
        "-q",
        "-T",
        "-L",
    }


def _gh_pr_result_values(
    raw_output: Any, *, command: _GhPrCommand
) -> list[dict[str, Any]]:
    output = _string_from_any(raw_output)
    if not output:
        return []
    parsed = _json_value_from_text(output)
    if isinstance(parsed, dict):
        if command.requires_identity_output or command.template_expression:
            return []
        normalized = _normalize_gh_pr_result(parsed)
        if command.jq_expression and (
            not _gh_jq_preserves_pr_field_object(command.jq_expression)
            or not _gh_pr_result_has_identity_fields(normalized)
        ):
            return []
        return [normalized] if _gh_pr_result_has_pr_fields(normalized) else []
    parsed_number = _positive_int(parsed)
    if (
        command.source_tool == "gh_pr_view"
        and parsed_number is not None
        and not command.requires_identity_output
        and _gh_jq_selects_pr_number(command.jq_expression)
    ):
        return [{"number": parsed_number}]
    if isinstance(parsed, list):
        if command.requires_identity_output or command.jq_expression:
            return []
        items = [
            normalized
            for item in parsed
            if isinstance(item, dict)
            if _gh_pr_result_has_pr_fields(
                normalized := _normalize_gh_pr_result(item)
            )
        ]
        return items if len(items) == 1 else []
    if command.requires_identity_output:
        return []
    if command.jq_expression and not _gh_jq_selects_pr_url(command.jq_expression):
        return []
    if command.template_expression and not _gh_template_selects_pr_url(
        command.template_expression
    ):
        return []
    urls = [{"url": match.group(0)} for match in _GITHUB_PR_URL_RE.finditer(output)]
    return urls if len(urls) == 1 else []


def _gh_jq_selects_pr_number(expression: str) -> bool:
    return expression.strip() in {".number", "number"}


def _gh_jq_selects_pr_url(expression: str) -> bool:
    return expression.strip() in {".url", "url"}


def _gh_jq_preserves_pr_field_object(expression: str) -> bool:
    stripped = expression.strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        return False
    body = stripped[1:-1].strip()
    if not body:
        return False
    has_identity = False
    for field in body.split(","):
        match = _GH_JQ_OBJECT_FIELD_RE.fullmatch(field.strip())
        if match is None:
            return False
        key = (match.group("shorthand") or match.group("key") or "").strip("\"'")
        source = (match.group("source") or key).removeprefix(".")
        if source != key:
            return False
        if key in _GH_JQ_IDENTITY_FIELDS:
            has_identity = True
    return has_identity


def _gh_template_selects_pr_url(expression: str) -> bool:
    return re.sub(r"\s+", "", expression) == "{{.url}}"


def _gh_pr_result_has_pr_fields(value: dict[str, Any]) -> bool:
    return any(
        key in value
        for key in {
            "base",
            "baseRefName",
            "baseRefOid",
            "base_sha",
            "closed",
            "closedAt",
            "closed_at",
            "ci_status",
            "display_url",
            "draft",
            "failing_jobs",
            "head",
            "headRefName",
            "headRefOid",
            "head_sha",
            "isDraft",
            "merge_commit_sha",
            "mergeable",
            "merged",
            "mergedAt",
            "merged_at",
            "number",
            "pending_jobs",
            "pr_number",
            "repo_full_name",
            "repository_full_name",
            "reviewDecision",
            "review_signal",
            "state",
            "statusCheckRollup",
            "title",
            "url",
        }
    )


def _gh_pr_result_has_identity_fields(value: dict[str, Any]) -> bool:
    if _positive_int(value.get("number") or value.get("pr_number")) is not None:
        return True
    url = _string_from_any(value.get("display_url") or value.get("url"))
    return _GITHUB_PR_URL_RE.search(url) is not None


def _json_value_from_text(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _normalize_gh_pr_result(source: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(source)
    for source_key, target_key in (
        ("baseRefName", "base"),
        ("headRefName", "head"),
        ("baseRefOid", "base_sha"),
        ("headRefOid", "head_sha"),
    ):
        if source_key in source and target_key not in normalized:
            normalized[target_key] = source[source_key]
    if "isDraft" in source and "draft" not in normalized:
        normalized["draft"] = source["isDraft"]
    closed_at = _string_from_any(source.get("closedAt"))
    if closed_at:
        normalized["closed_at"] = closed_at
        normalized.setdefault("state", "closed")
    if source.get("closed") is True:
        normalized.setdefault("state", "closed")
    merged_at = _string_from_any(source.get("mergedAt"))
    if merged_at:
        normalized["merged_at"] = merged_at
        normalized["merged"] = True
        normalized.setdefault("state", "closed")
    mergeable = _string_from_any(source.get("mergeable")).lower()
    if mergeable == "mergeable":
        normalized["mergeable"] = True
    elif mergeable == "conflicting":
        normalized["mergeable"] = False
    elif mergeable == "unknown":
        normalized["mergeable"] = ""
    review_decision = _string_from_any(source.get("reviewDecision")).upper()
    if review_decision == "APPROVED":
        normalized["review_signal"] = "approved"
    elif review_decision == "CHANGES_REQUESTED":
        normalized["review_signal"] = "changes_requested"
    elif review_decision == "REVIEW_REQUIRED":
        normalized["review_signal"] = ""
    status, failing_jobs, pending_jobs = _ci_status_from_status_check_rollup(
        source.get("statusCheckRollup")
    )
    if status:
        normalized["ci_status"] = status
        normalized["failing_jobs"] = failing_jobs
        normalized["pending_jobs"] = pending_jobs
    state = _string_from_any(source.get("state"))
    if state:
        normalized["state"] = "closed" if state.lower() == "merged" else state.lower()
        if state.lower() == "merged" and "merged" not in normalized:
            normalized["merged"] = True
    return normalized


def _normalize_gh_repo_selector(value: Any) -> str:
    repo = _string_from_any(value).strip().removeprefix("https://")
    repo = repo.removeprefix("http://").strip("/")
    if repo.endswith(".git"):
        repo = repo[:-4]
    parts = [part for part in repo.split("/") if part]
    if len(parts) >= 3:
        candidate = f"{parts[-2]}/{parts[-1]}"
        return candidate if _GH_REPO_SELECTOR_RE.fullmatch(candidate) else ""
    if len(parts) == 2:
        candidate = f"{parts[0]}/{parts[1]}"
        return candidate if _GH_REPO_SELECTOR_RE.fullmatch(candidate) else ""
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
    raw_result = _unwrap_mcp_result_envelope(raw_result)
    if raw_result is None:
        return []
    if isinstance(raw_result, str):
        parsed = _json_dict_from_text(raw_result)
        return _mcp_result_values(parsed) if parsed is not None else []
    if not isinstance(raw_result, dict):
        return []

    values: list[dict[str, Any]] = []
    for key in ("structuredContent", "structured_content"):
        structured = raw_result.get(key)
        if isinstance(structured, dict):
            values.extend(_mcp_result_values(structured))
    content = raw_result.get("content")
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            values.extend(_mcp_result_values(item.get("text")))
    if values:
        return values
    return [raw_result]


def _unwrap_mcp_result_envelope(raw_result: Any) -> Any:
    if not isinstance(raw_result, dict):
        return raw_result
    if "Ok" in raw_result:
        return raw_result.get("Ok")
    if "Err" in raw_result:
        return None
    return raw_result


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
        ("ci_status", "ci_status"),
    ):
        text = _string_from_any(source.get(source_key))
        if text:
            target[target_key] = _compact_text(text)
    if "review_signal" in source:
        target["review_signal"] = _compact_text(
            _string_from_any(source.get("review_signal"))
        )
    for source_key in ("failing_jobs", "pending_jobs"):
        value = source.get(source_key)
        if isinstance(value, list):
            target[source_key] = [
                _compact_text(text)
                for item in value
                if (text := _string_from_any(item))
            ][:_PR_DETAIL_LIMIT]
    for source_key, target_key in (("merged", "merged"), ("draft", "draft")):
        value = source.get(source_key)
        if isinstance(value, bool):
            target[target_key] = value
    if "mergeable" in source:
        value = source.get("mergeable")
        if isinstance(value, bool):
            target["mergeable"] = value
        elif value == "":
            target["mergeable"] = ""
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
    if tool == "fetch_workflow_run_jobs" and isinstance(raw_args, dict):
        # ``fetch_workflow_run_jobs`` names only a ``run_id``; record it as the
        # correlation key so a later turn that drills into a run already seen
        # for the current PR (its id captured from ``fetch_commit_workflow_runs``
        # below) is attributed to that PR instead of being treated as either an
        # unrelated run or unrelated work that supersedes the PR epoch.
        observed_run_id = _positive_int(raw_args.get("run_id"))
        if observed_run_id is not None:
            target["observed_run_id"] = observed_run_id
            target["workflow_run_ids"] = _merge_run_ids(
                target.get("workflow_run_ids"), [observed_run_id]
            )
    for source in result_values:
        if tool == "get_commit_combined_status":
            status = _ci_status_from_statuses(source.get("statuses"))
            if status:
                # Combined-status covers GitHub's commit Statuses API (external
                # CI integrations), not workflow runs / check runs -- so a
                # success here proves nothing about whether a previously-
                # observed ``fetch_workflow_run_jobs`` failure has recovered.
                # Update ``ci_status`` only; never clear the per-job lists
                # from this branch. ``system_agents._ci_gate`` already
                # short-circuits on non-empty ``failing_jobs`` ahead of
                # reading ``ci_status``, so leaving a real workflow-run
                # failure in the list keeps the actionable BLOCKED verdict
                # the follow-up agent needs.
                target["ci_status"] = status
        elif tool == "fetch_commit_workflow_runs":
            # Workflow-runs observations carry ``commit_sha`` (so they correlate
            # to the current PR by commit) and enumerate the run ids that
            # ``fetch_workflow_run_jobs`` later drills into; record them so those
            # follow-up job observations correlate by run id.
            run_ids = _workflow_run_ids_from_runs(source.get("workflow_runs"))
            if run_ids:
                target["workflow_run_ids"] = _merge_run_ids(
                    target.get("workflow_run_ids"), run_ids
                )
            status = _ci_status_from_runs(source.get("workflow_runs"))
            if status:
                # Workflow-runs observations DO speak for the same check-run
                # universe that ``fetch_workflow_run_jobs`` enumerates, so a
                # definitive ``success`` across all runs supersedes a prior
                # per-job snapshot on the same commit. ``pending`` -- with no
                # failed completed runs -- means a previously-observed
                # failing workflow is being re-run on the same commit, so
                # the earlier ``failing_jobs`` list is now obsolete and
                # leaving it intact would have ``_ci_gate`` keep BLOCKING
                # on "Failing CI jobs were observed" instead of waiting for
                # the rerun via the "CI is still running" pending gate; clear
                # the failing list on pending too. ``pending_jobs`` stays put
                # because workflow-runs observations don't enumerate jobs,
                # so the prior per-job pending list remains the most specific
                # signal we have to carry into the next agent's feedback.
                # ``unknown`` / ``failure`` keep both lists as-is: ``failure``
                # already has the per-job list as its best detail, and
                # ``unknown`` proves nothing.
                target["ci_status"] = status
                if status == "success":
                    target["failing_jobs"] = []
                    target["pending_jobs"] = []
                elif status == "pending":
                    target["failing_jobs"] = []
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


def _ci_status_from_status_check_rollup(raw_checks: Any) -> tuple[str, list[str], list[str]]:
    if not isinstance(raw_checks, list):
        return "", [], []
    if not raw_checks:
        return "unknown", [], []
    failing: list[str] = []
    pending: list[str] = []
    saw_completed = False
    for check in raw_checks:
        if not isinstance(check, dict):
            continue
        name = (
            _string_from_any(check.get("name"))
            or _string_from_any(check.get("context"))
            or "unnamed check"
        )
        status = _string_from_any(check.get("status") or check.get("state")).lower()
        conclusion = _string_from_any(check.get("conclusion")).lower()
        if status in {
            "expected",
            "pending",
            "queued",
            "in_progress",
            "waiting",
            "requested",
        }:
            pending.append(name)
            continue
        if status in {"failure", "error"} or conclusion in _FAILURE_CONCLUSIONS:
            failing.append(name)
            continue
        if conclusion and conclusion not in _SUCCESS_CONCLUSIONS:
            failing.append(name)
            continue
        if status in {"completed", "success"} or conclusion in _SUCCESS_CONCLUSIONS:
            saw_completed = True
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
    elif _pr_snapshot_head_changed(snapshot, values):
        # A new push advances the head: workflow run ids captured for the old
        # head identify a superseded commit's runs, so drop them. Otherwise a
        # later job drill into one of those stale run ids would still correlate
        # and overwrite the new head's CI status with results from the old
        # commit, leaving the PR follow-up blocked on superseded CI.
        for key in ("workflow_run_ids", "ci_status", "failing_jobs", "pending_jobs"):
            snapshot.pop(key, None)
    for key, value in values.items():
        # ``None`` is an "absent" sentinel; empty list/dict is an explicit
        # "observed and found none" clear. ``""`` is "absent" for every
        # string field except explicit gate clears that need to propagate
        # through ``system_agents._merge_pr_handoff_dicts`` cross-worker.
        # Reaction-derived ``thumbs_up`` is held back from the review clear
        # since the reviews tool does not speak for it.
        if value is None:
            continue
        if value == "":
            if key == "mergeable" or (
                key == "review_signal" and snapshot.get(key) != "thumbs_up"
            ):
                snapshot[key] = ""
            continue
        if (
            key == "review_signal"
            and value == "thumbs_up"
            and snapshot.get("review_signal") == "changes_requested"
        ):
            continue
        if key == "workflow_run_ids" and isinstance(value, list):
            # Run ids accumulate across observations (an empty list never
            # clears them) so a run seen once for this PR keeps correlating
            # later job observations.
            snapshot[key] = _merge_run_ids(snapshot.get(key), value)
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


def _pr_head_commit_values(snapshot: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in ("head_sha", "latest_commit_sha"):
        value = snapshot.get(key)
        if isinstance(value, str) and value:
            values.add(value)
    return values


def _pr_snapshot_head_changed(current: dict[str, Any], update: dict[str, Any]) -> bool:
    if not current:
        return False
    current_values = _pr_head_commit_values(current)
    update_values = _pr_head_commit_values(update)
    return bool(current_values and update_values and current_values != update_values)


def _finalize_pr_snapshot(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    if not snapshot:
        return None
    url = _string_from_any(snapshot.get("url"))
    if url:
        _copy_identity_from_pr_url(snapshot, url)
    if not url and not (
        isinstance(snapshot.get("repository_full_name"), str)
        and isinstance(snapshot.get("pr_number"), int)
        and not isinstance(snapshot.get("pr_number"), bool)
    ):
        return None
    return snapshot


def _pr_snapshot_has_identity(snapshot: dict[str, Any] | None) -> bool:
    if not snapshot:
        return False
    if isinstance(snapshot.get("url"), str) and snapshot["url"]:
        return True
    return (
        isinstance(snapshot.get("repository_full_name"), str)
        and bool(snapshot["repository_full_name"])
        and isinstance(snapshot.get("pr_number"), int)
        and not isinstance(snapshot.get("pr_number"), bool)
    )


def _pr_snapshot_matches_current_pr(
    current: dict[str, Any] | None, update: dict[str, Any] | None
) -> bool:
    if current is None or update is None:
        return False
    if not _pr_snapshot_has_identity(current) or not _pr_snapshot_has_identity(update):
        return False
    return not _pr_snapshot_identity_changed(current, update)


def _pr_update_belongs_to_current_pr(
    current: dict[str, Any] | None, update: _PrSnapshotUpdate
) -> bool:
    if not _pr_snapshot_has_identity(current):
        return False
    update_snapshot = _finalize_pr_snapshot(dict(update.values))
    if _pr_snapshot_has_identity(update_snapshot):
        return _pr_snapshot_matches_current_pr(current, update_snapshot)
    update_repo = _string_from_any(update.values.get("repository_full_name"))
    current_repo = (
        _string_from_any(current.get("repository_full_name")) if current else ""
    )
    if update_repo and current_repo and update_repo != current_repo:
        return False
    update_commit = _string_from_any(update.values.get("latest_commit_sha"))
    if update_commit:
        return update_commit in _pr_commit_shas(current)
    # No PR identity and no commit SHA. A ``fetch_workflow_run_jobs`` observation
    # is in this shape -- it names only a ``run_id``. Correlate it to the current
    # PR by that run id: accept it only when the run was already seen among the
    # PR's own workflow runs (recorded from a commit-correlated
    # ``fetch_commit_workflow_runs``). This keeps a routine "drill into the
    # failing run's jobs" follow-up from superseding the PR epoch, without
    # attributing a job check for an unrelated repo/PR's ``run_id`` to it.
    update_run_id = update.values.get("observed_run_id")
    if isinstance(update_run_id, int) and not isinstance(update_run_id, bool):
        return update_run_id in _pr_workflow_run_ids(current)
    return False


def _pr_workflow_run_ids(snapshot: dict[str, Any] | None) -> set[int]:
    if not snapshot:
        return set()
    raw = snapshot.get("workflow_run_ids")
    if not isinstance(raw, list):
        return set()
    return {
        value
        for value in raw
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
    }


def _workflow_run_ids_from_runs(raw_runs: Any) -> list[int]:
    if not isinstance(raw_runs, list):
        return []
    ids: list[int] = []
    for run in raw_runs:
        if not isinstance(run, dict):
            continue
        run_id = _positive_int(
            run.get("id") or run.get("run_id") or run.get("databaseId")
        )
        if run_id is not None and run_id not in ids:
            ids.append(run_id)
    return ids


def _merge_run_ids(existing: Any, new: list[int]) -> list[int]:
    merged: list[int] = []
    for source in (existing, new):
        if not isinstance(source, list):
            continue
        for value in source:
            if (
                isinstance(value, int)
                and not isinstance(value, bool)
                and value > 0
                and value not in merged
            ):
                merged.append(value)
    return merged


def _pr_commit_shas(snapshot: dict[str, Any] | None) -> set[str]:
    if not snapshot:
        return set()
    return {
        commit
        for key in ("latest_commit_sha", "head_sha", "merge_commit_sha")
        if (commit := _string_from_any(snapshot.get(key)))
    }


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
