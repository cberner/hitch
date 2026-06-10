from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlencode

from django.urls import reverse

from hitch.main import demo
from hitch.main.models import (
    CodexInstance,
    SystemAgentRun,
    SystemWorkflow,
)
from hitch.main.sdk_values import (
    latest_updated_at,
    updated_at_seconds,
)


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
        .exclude(agent_kind=demo.DEMO_AGENT_KIND)
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


def _qa_activity_updated_at_by_main_thread_id(
    threads: Iterable[Any], hidden_thread_ids: set[str]
) -> dict[str, Any]:
    current_thread_ids = {
        thread_id
        for thread in threads
        if isinstance((thread_id := getattr(thread, "id", None)), str)
    }
    current_main_thread_ids = current_thread_ids - hidden_thread_ids
    if not current_main_thread_ids:
        return {}

    hidden_updated_at_by_thread_id: dict[str, Any] = {}
    for thread in threads:
        thread_id = getattr(thread, "id", None)
        if isinstance(thread_id, str) and thread_id in hidden_thread_ids:
            hidden_updated_at_by_thread_id[thread_id] = getattr(
                thread, "updated_at", None
            )

    runs = (
        SystemAgentRun.objects.filter(
            workflow__kind=SystemWorkflow.KIND_PR_QA,
            workflow__main_thread_id__in=current_main_thread_ids,
        )
        .exclude(thread_id="")
        .select_related("workflow")
    )
    updated_at_by_main_thread: dict[str, Any] = {}
    for run in runs:
        main_thread_id = run.workflow.main_thread_id
        if not main_thread_id:
            continue
        run_updated_at = hidden_updated_at_by_thread_id.get(run.thread_id)
        if updated_at_seconds(run_updated_at) is None:
            run_updated_at = latest_updated_at(run.updated_at, run.workflow.updated_at)
        updated_at_by_main_thread[main_thread_id] = latest_updated_at(
            updated_at_by_main_thread.get(main_thread_id),
            run_updated_at,
        )
    return updated_at_by_main_thread


def _session_updated_at(
    thread: Any, qa_updated_at_by_main_thread: Mapping[str, Any]
) -> Any:
    return latest_updated_at(
        getattr(thread, "updated_at", None),
        qa_updated_at_by_main_thread.get(getattr(thread, "id", "")),
    )


def _updated_at_sort_key(updated_at: Any) -> float:
    seconds = updated_at_seconds(updated_at)
    return seconds if seconds is not None else 0.0


def _demo_system_thread_ids() -> set[str]:
    return set(
        SystemAgentRun.objects.filter(agent_kind=demo.DEMO_AGENT_KIND)
        .exclude(thread_id="")
        .values_list("thread_id", flat=True)
        .distinct()
    )


def _demo_system_session_url(session_id: str) -> str:
    if not session_id:
        return ""
    run = (
        SystemAgentRun.objects.filter(
            thread_id=session_id,
            agent_kind=demo.DEMO_AGENT_KIND,
        )
        .order_by("-created_at", "-pk")
        .first()
    )
    if run is None:
        return ""
    path = reverse("system_session", kwargs={"session_id": session_id})
    return f"{path}?{urlencode({'run_id': run.pk})}"


def _system_agent_run_for_thread(
    thread_id: str, *, run_id: int | None = None
) -> SystemAgentRun | None:
    if not thread_id:
        return None
    if run_id is not None:
        return (
            SystemAgentRun.objects.filter(pk=run_id, thread_id=thread_id)
            .select_related("instance", "workflow")
            .first()
        )
    return (
        SystemAgentRun.objects.filter(thread_id=thread_id)
        .select_related("instance", "workflow")
        .order_by("-created_at", "-pk")
        .first()
    )


def _system_agent_instance_for_thread(thread_id: str) -> CodexInstance | None:
    if not thread_id:
        return None
    return (
        CodexInstance.objects.filter(
            thread_id=thread_id,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
        )
        .exclude(agent_kind=demo.DEMO_AGENT_KIND)
        .order_by("-started_at", "-pk")
        .first()
    )


def _system_agent_kind(
    run: SystemAgentRun | None, instance: CodexInstance | None = None
) -> str:
    if run is not None:
        return run.agent_kind
    if instance is not None:
        return instance.agent_kind
    return ""


def _system_agent_run_label(
    run: SystemAgentRun | None, instance: CodexInstance | None = None
) -> str:
    source_instance = run.instance if run is not None else instance
    display_author = source_instance.display_author.strip() if source_instance else ""
    if display_author:
        return display_author
    agent_kind = _system_agent_kind(run, instance)
    return agent_kind.replace("_", " ") if agent_kind else "system agent"


def _system_agent_status(
    run: SystemAgentRun | None, instance: CodexInstance | None = None
) -> str:
    if run is not None:
        return run.status
    return instance.status if instance is not None else ""


def _system_agent_run_detail_title(
    run: SystemAgentRun | None, instance: CodexInstance | None = None
) -> str:
    label = _system_agent_run_label(run, instance)
    return f"{label} log" if label else "System session"
