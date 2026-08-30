"""Derive a display stage for a Hitch session from durable session state."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from hitch.main.models import CodexInstance
from hitch.main.runtime import rollout
from hitch.main.sessions import agent_tasks


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
AWAITING_INPUT = SessionStage("awaiting_input", "Awaiting Input", "warning")
BLOCKED = SessionStage("blocked", "Blocked", "warning")
DONE_MERGED = SessionStage("done_merged", "Done: Merged", "done")
DONE_CLOSED = SessionStage("done_closed", "Done: Closed", "done")
_STAGES_BY_KEY = {
    stage.key: stage
    for stage in (
        NEW,
        PLAN,
        IMPLEMENTATION,
        QA,
        PR,
        AWAITING_INPUT,
        BLOCKED,
        DONE_MERGED,
        DONE_CLOSED,
    )
}
_LEGACY_STAGES_BY_KEY = {
    "waiting_for_user": AWAITING_INPUT,
}

def derive_stage(
    *,
    entries: Iterable[Mapping[str, Any]] = (),
    active_instance: CodexInstance | None = None,
    awaiting_user_input: bool = False,
    pr_snapshot: Mapping[str, Any] | None = None,
) -> SessionStage:
    """Return the current stage from ordinary turns and durable PR state."""
    if awaiting_user_input:
        return AWAITING_INPUT

    if active_instance is not None:
        task_stage = agent_tasks.stage_for_agent_kind(active_instance.agent_kind)
        if task_stage == "qa":
            return QA
        if task_stage == "pr":
            return PR
        return PLAN if active_instance.plan_mode else IMPLEMENTATION

    if terminal_stage := _terminal_pr_stage(pr_snapshot):
        return terminal_stage

    entries_list = list(entries)
    if _has_pr_identity(pr_snapshot):
        return PR
    if _latest_agent_task_stage(entries_list) == "qa":
        return QA
    if _entries_are_waiting_on_plan(entries_list):
        return PLAN
    if _has_session_activity(entries_list):
        return IMPLEMENTATION
    return NEW


def stage_for_key(key: str) -> SessionStage | None:
    return _STAGES_BY_KEY.get(key) or _LEGACY_STAGES_BY_KEY.get(key)


def _latest_agent_task_stage(entries: list[Mapping[str, Any]]) -> str:
    for entry in reversed(entries):
        if entry.get("kind") != "user":
            continue
        text = entry.get("text")
        return agent_tasks.stage_for_agent_prompt(text if isinstance(text, str) else "")
    return ""


def _terminal_pr_stage(snapshot: Mapping[str, Any] | None) -> SessionStage | None:
    if not snapshot:
        return None
    state = _string(snapshot.get("state")).lower()
    if (
        snapshot.get("merged") is True
        or _string(snapshot.get("merged_at"))
        or state == "merged"
    ):
        return DONE_MERGED
    if state == "closed":
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
