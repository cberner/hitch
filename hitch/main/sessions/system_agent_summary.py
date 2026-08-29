from collections.abc import Iterable
from typing import Any

from hitch.main.models import (
    CodexInstance,
    SystemAgentRun,
)
from hitch.main.runtime.sdk_values import updated_at_seconds


def _system_agent_runs_by_thread_id(
    thread_ids: Iterable[str],
) -> dict[str, SystemAgentRun]:
    ids = [thread_id for thread_id in thread_ids if thread_id]
    if not ids:
        return {}
    runs = (
        SystemAgentRun.objects.filter(thread_id__in=ids)
        .exclude(thread_id="")
        .select_related("instance")
        .only(
            "id",
            "workflow",
            "agent_kind",
            "thread_id",
            "instance",
            "status",
            "created_at",
            "instance__id",
            "instance__thread_id",
            "instance__display_author",
            "instance__agent_kind",
            "instance__status",
            "instance__started_at",
        )
        .order_by("thread_id", "-created_at", "-pk")
    )
    by_thread_id: dict[str, SystemAgentRun] = {}
    for run in runs:
        by_thread_id.setdefault(run.thread_id, run)
    return by_thread_id


def _system_agent_instances_by_thread_id(
    thread_ids: Iterable[str],
) -> dict[str, CodexInstance]:
    ids = [thread_id for thread_id in thread_ids if thread_id]
    if not ids:
        return {}
    instances = (
        CodexInstance.objects.filter(
            thread_id__in=ids,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
        )
        .exclude(thread_id="")
        .only(
            "id",
            "thread_id",
            "display_author",
            "agent_kind",
            "status",
            "started_at",
        )
        .order_by("thread_id", "-started_at", "-pk")
    )
    by_thread_id: dict[str, CodexInstance] = {}
    for instance in instances:
        by_thread_id.setdefault(instance.thread_id, instance)
    return by_thread_id


def _updated_at_sort_key(updated_at: Any) -> float:
    seconds = updated_at_seconds(updated_at)
    return seconds if seconds is not None else 0.0


def _system_agent_run_for_thread(thread_id: str, *, run_id: int | None = None) -> SystemAgentRun | None:
    if not thread_id:
        return None
    if run_id is not None:
        return SystemAgentRun.objects.filter(pk=run_id, thread_id=thread_id).select_related(
            "instance", "workflow"
        ).first()
    return (
        SystemAgentRun.objects.filter(thread_id=thread_id)
        .select_related("instance", "workflow")
        .order_by("-created_at", "-pk")
    ).first()


def _system_agent_instance_for_thread(thread_id: str) -> CodexInstance | None:
    if not thread_id:
        return None
    return (
        CodexInstance.objects.filter(
            thread_id=thread_id,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
        ).order_by("-started_at", "-pk")
    ).first()


def _system_agent_kind(run: SystemAgentRun | None, instance: CodexInstance | None = None) -> str:
    if run is not None:
        return run.agent_kind
    if instance is not None:
        return instance.agent_kind
    return ""


def _system_agent_run_label(run: SystemAgentRun | None, instance: CodexInstance | None = None) -> str:
    source_instance = run.instance if run is not None else instance
    display_author = source_instance.display_author.strip() if source_instance else ""
    if display_author:
        return display_author
    agent_kind = _system_agent_kind(run, instance)
    return agent_kind.replace("_", " ") if agent_kind else "system agent"


def _system_agent_status(run: SystemAgentRun | None, instance: CodexInstance | None = None) -> str:
    if run is not None:
        return run.status
    return instance.status if instance is not None else ""


def _system_agent_run_detail_title(run: SystemAgentRun | None, instance: CodexInstance | None = None) -> str:
    label = _system_agent_run_label(run, instance)
    return f"{label} log" if label else "System session"
