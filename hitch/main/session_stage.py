"""Derive a display stage for a Hitch session from durable session state."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from hitch.main import rollout, system_agents
from hitch.main.models import CodexInstance, SystemWorkflow


@dataclass(frozen=True)
class SessionStage:
    key: str
    label: str
    tone: str = "default"

    def as_context(self) -> dict[str, str]:
        return {"key": self.key, "label": self.label, "tone": self.tone}


NEW = SessionStage("new", "New")
PLAN = SessionStage("plan", "Plan", "active")
IMPLEMENTATION = SessionStage("implementation", "Implementation", "active")
QA = SessionStage("qa", "QA", "active")
PR = SessionStage("pr", "PR", "active")
BLOCKED = SessionStage("blocked", "Blocked", "warning")
DONE_MERGED = SessionStage("done_merged", "Done: Merged", "done")
DONE_CLOSED = SessionStage("done_closed", "Done: Closed", "done")
_STAGES_BY_KEY = {
    stage.key: stage
    for stage in (NEW, PLAN, IMPLEMENTATION, QA, PR, BLOCKED, DONE_MERGED, DONE_CLOSED)
}

_STAGE_BY_WORKFLOW_STEP = {
    system_agents.STEP_LOCAL_BRANCH_MERGED: DONE_MERGED,
    system_agents.STEP_QA_RUNNING: QA,
    system_agents.STEP_QA_APPROVED: QA,
    system_agents.STEP_FEEDBACK_RUNNING: IMPLEMENTATION,
    system_agents.STEP_USER_STEERING_RUNNING: IMPLEMENTATION,
    system_agents.STEP_PR_FEEDBACK_RUNNING: IMPLEMENTATION,
    system_agents.STEP_PR_PROMPT_SPAWNED: PR,
    system_agents.STEP_PR_PROMPT_RUNNING: PR,
    system_agents.STEP_PR_MONITORING: PR,
    system_agents.STEP_PR_READY: PR,
    system_agents.STEP_PR_CLOSED: PR,
    system_agents.STEP_BLOCKED: BLOCKED,
}
_BLOCKED_WORKFLOW_STATUSES = (
    SystemWorkflow.STATUS_BLOCKED,
    SystemWorkflow.STATUS_FAILED,
    SystemWorkflow.STATUS_MAX_ITERATIONS_REACHED,
)


def derive_stage(
    *,
    entries: Iterable[Mapping[str, Any]] = (),
    active_instance: CodexInstance | None = None,
    workflow: SystemWorkflow | None = None,
    pr_snapshot: Mapping[str, Any] | None = None,
    workflow_pr_snapshot: Mapping[str, Any] | None = None,
) -> SessionStage:
    """Return the current stage after callers select the current lifecycle owner."""
    selected_pr = _select_pr_snapshot(
        log_pr_snapshot=pr_snapshot,
        workflow_pr_snapshot=workflow_pr_snapshot,
    )

    if workflow is not None and workflow.status == SystemWorkflow.STATUS_RUNNING:
        if running_stage := _running_workflow_stage(workflow, selected_pr):
            return running_stage
        return IMPLEMENTATION

    if active_instance is not None:
        return PLAN if active_instance.plan_mode else IMPLEMENTATION

    if terminal_stage := _terminal_pr_stage(selected_pr):
        return terminal_stage

    if workflow_stage := _workflow_stage(workflow, workflow_pr_snapshot):
        return workflow_stage

    entries_list = list(entries)
    if _has_pr_identity(selected_pr):
        return PR
    if _entries_are_waiting_on_plan(entries_list):
        return PLAN
    if _has_session_activity(entries_list):
        return IMPLEMENTATION
    return NEW


def merge_pr_snapshots(
    *,
    log_pr_snapshot: Mapping[str, Any] | None,
    workflow_pr_snapshot: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return dict(
        _select_pr_snapshot(
            log_pr_snapshot=log_pr_snapshot,
            workflow_pr_snapshot=workflow_pr_snapshot,
        )
        or {}
    )


def stage_for_key(key: str) -> SessionStage | None:
    return _STAGES_BY_KEY.get(key)


def _running_workflow_stage(
    workflow: SystemWorkflow, pr_snapshot: Mapping[str, Any] | None
) -> SessionStage | None:
    if workflow.kind != SystemWorkflow.KIND_PR_QA:
        return None
    if terminal_stage := _terminal_pr_stage(pr_snapshot):
        return terminal_stage
    return _stage_for_workflow_step(workflow.step)


def _workflow_stage(
    workflow: SystemWorkflow | None, pr_snapshot: Mapping[str, Any] | None
) -> SessionStage | None:
    if workflow is None or workflow.kind != SystemWorkflow.KIND_PR_QA:
        return None
    if workflow.status in _BLOCKED_WORKFLOW_STATUSES:
        return BLOCKED
    if terminal_stage := _terminal_pr_stage(pr_snapshot):
        return terminal_stage
    if step_stage := _stage_for_workflow_step(workflow.step):
        return step_stage
    if _has_pr_identity(pr_snapshot):
        return PR
    return None


def _stage_for_workflow_step(step: str) -> SessionStage | None:
    return _STAGE_BY_WORKFLOW_STEP.get(step)


def _select_pr_snapshot(
    *,
    log_pr_snapshot: Mapping[str, Any] | None,
    workflow_pr_snapshot: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if not log_pr_snapshot:
        return workflow_pr_snapshot
    if not workflow_pr_snapshot:
        return log_pr_snapshot
    if _same_pr_identity(log_pr_snapshot, workflow_pr_snapshot):
        if _terminal_pr_stage(workflow_pr_snapshot):
            return {**log_pr_snapshot, **workflow_pr_snapshot}
        return {**workflow_pr_snapshot, **log_pr_snapshot}
    if _has_pr_identity(workflow_pr_snapshot):
        return workflow_pr_snapshot
    return log_pr_snapshot


def _terminal_pr_stage(snapshot: Mapping[str, Any] | None) -> SessionStage | None:
    if not snapshot:
        return None
    if snapshot.get("merged") is True or _string(snapshot.get("merged_at")):
        return DONE_MERGED
    if _string(snapshot.get("state")).lower() == "closed":
        return DONE_CLOSED
    return None


def _has_pr_identity(snapshot: Mapping[str, Any] | None) -> bool:
    if not snapshot:
        return False
    return bool(_string(snapshot.get("url"))) or (
        bool(_string(snapshot.get("repository_full_name")))
        and isinstance(snapshot.get("pr_number"), int)
        and not isinstance(snapshot.get("pr_number"), bool)
    )


def _same_pr_identity(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_url = _string(left.get("url"))
    right_url = _string(right.get("url"))
    if left_url and right_url:
        return left_url == right_url
    left_repo = _string(left.get("repository_full_name"))
    right_repo = _string(right.get("repository_full_name"))
    left_number = left.get("pr_number")
    right_number = right.get("pr_number")
    return (
        bool(left_repo)
        and left_repo == right_repo
        and isinstance(left_number, int)
        and not isinstance(left_number, bool)
        and left_number == right_number
    )


def _entries_are_waiting_on_plan(entries: list[Mapping[str, Any]]) -> bool:
    return rollout.entries_await_plan_approval([dict(entry) for entry in entries])


def _has_session_activity(entries: Iterable[Mapping[str, Any]]) -> bool:
    for entry in entries:
        kind = entry.get("kind")
        if kind in {"user", "agent", "plan"}:
            return True
        if kind == "intermediate" and entry.get("items"):
            return True
        if kind == "tool_call":
            return True
    return False


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""
