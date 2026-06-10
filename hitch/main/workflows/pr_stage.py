"""PR-stage cache helpers extracted from ``views``.

This module holds the leaf helpers that read and persist a session's cached
derived stage and look up its stage-relevant workflows. It must NOT import
``views`` -- the view layer imports this module and calls these helpers
module-qualified (``pr_stage._foo(...)``) so there is exactly one binding per
symbol, which keeps test ``mock.patch`` of these names intercepting both the
view call sites and the internal sibling calls between the helpers below.

The PR-stage *refresh* worker chain (``_schedule_pr_stage_refresh`` and
friends) stays in ``views`` because it depends on view-only rollout and
workflow-lifecycle helpers; moving it would require an import cycle.
"""

from collections.abc import Iterable

from django.db.models import Q
from django.utils import timezone

from hitch.main.models import SessionMetadata, SystemWorkflow
from hitch.main.runtime.db import run_ignoring_database_locks
from hitch.main.sessions import session_stage


def _latest_stage_workflows_by_thread_id(
    thread_ids: Iterable[str],
) -> dict[str, SystemWorkflow]:
    ids = [thread_id for thread_id in dict.fromkeys(thread_ids) if thread_id]
    if not ids:
        return {}
    workflows = (
        SystemWorkflow.objects.filter(
            main_thread_id__in=ids,
        )
        .filter(
            Q(kind=SystemWorkflow.KIND_PR_QA)
            | Q(status=SystemWorkflow.STATUS_RUNNING)
        )
        .order_by("main_thread_id", "-updated_at", "-pk")
    )
    by_thread_id: dict[str, SystemWorkflow] = {}
    for workflow in workflows:
        by_thread_id.setdefault(workflow.main_thread_id, workflow)
    return by_thread_id


def _update_cached_stage(
    session_id: str, stage: session_stage.SessionStage, source_mtime_ns: int
) -> None:
    SessionMetadata.objects.filter(thread_id=session_id).exclude(
        derived_stage=stage.key,
        derived_stage_source_mtime_ns=source_mtime_ns,
    ).update(
        derived_stage=stage.key,
        derived_stage_source_mtime_ns=source_mtime_ns,
    )


def _update_cached_stage_best_effort(
    session_id: str, stage: session_stage.SessionStage, source_mtime_ns: int
) -> None:
    run_ignoring_database_locks(
        lambda: _update_cached_stage(session_id, stage, source_mtime_ns),
        description="session stage cache update",
    )


def _mark_cached_pr_stage_refresh_attempt(session_id: str) -> None:
    run_ignoring_database_locks(
        lambda: SessionMetadata.objects.filter(thread_id=session_id).update(
            derived_stage_pr_refresh_attempted_at=timezone.now()
        ),
        description="PR stage refresh backoff",
    )


def _latest_pr_workflow_for_thread(session_id: str) -> SystemWorkflow | None:
    return (
        SystemWorkflow.objects.filter(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id=session_id,
        )
        .order_by("-updated_at", "-pk")
        .first()
    )
