"""Cached session-stage persistence helpers."""

from hitch.main.models import SessionMetadata
from hitch.main.runtime.db import run_ignoring_database_locks
from hitch.main.sessions import session_stage


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
