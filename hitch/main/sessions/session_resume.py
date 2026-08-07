"""Reading a session's persisted metadata and resuming archived sessions.

Owns reading a session's persisted Codex metadata plus the archived-session
resume/unarchive/re-archive helpers used when a turn touches an inactive or
archived thread.
"""

from __future__ import annotations

import glob
import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from django.utils import timezone
from openai_codex import Codex
from openai_codex.errors import InvalidRequestError

from hitch.main.models import (
    ArchivedSessionTokenUsage,
    CodexInstance,
    SessionMetadata,
    SystemWorkflow,
)
from hitch.main.runtime import app_server_pool, codex_pool, rollout
from hitch.main.runtime.rollout_state import (
    _ARCHIVED_SESSIONS_DIR,
    _rollout_path_from_value,
    _rollout_path_is_archived,
)
from hitch.main.runtime.sdk_values import updated_at_seconds
from hitch.main.sessions import session_index
from hitch.main.sessions.entry_render import collapse_flat_entries
from hitch.main.sessions.settings_cookies import SettingsValues

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _MetadataThread:
    id: str
    cwd: str
    path: str
    name: str
    preview: str
    created_at: float | None
    updated_at: float | None
    archived: bool
    thread_source: str
    turns: tuple[Any, ...] = ()


@dataclass(frozen=True)
class _MetadataResume:
    thread: _MetadataThread
    entries: tuple[dict[str, Any], ...]
    model: str = ""
    reasoning_effort: str = ""
    model_config: rollout.SessionModelConfig | None = None
    rollout_data: rollout.SessionDetailData | None = None


_ROLLOUT_FILENAME_RE = re.compile(
    r"^rollout-\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-(?P<thread_id>.+)\.jsonl$"
)


def _metadata_indicates_archived(metadata: SessionMetadata | None) -> bool:
    return metadata is not None and metadata.codex_archived


def _metadata_rollout_path_indicates_archived(
    metadata: SessionMetadata | None,
) -> bool:
    if metadata is None:
        return False
    rollout_path = _rollout_path_from_value(metadata.codex_path)
    return rollout_path is not None and _rollout_path_is_archived(rollout_path)


def _thread_resume_archived_error(exc: InvalidRequestError) -> bool:
    message = str(exc).lower()
    return " is archived" in message and "unarchive" in message


def _record_session_unarchived(session_id: str, *, thread: Any | None = None) -> None:
    session_index.update_cached_archived(
        session_id, archived=False, thread=thread
    )
    SessionMetadata.objects.filter(thread_id=session_id).update(codex_path="")
    ArchivedSessionTokenUsage.objects.filter(thread_id=session_id).delete()


def _record_session_rearchived_after_rejected_turn(session_id: str) -> None:
    session_index.update_cached_archived(session_id, archived=True)
    SessionMetadata.objects.filter(thread_id=session_id).update(codex_path="")
    ArchivedSessionTokenUsage.objects.filter(thread_id=session_id).delete()


def _unarchive_session_for_turn(
    session_id: str, settings: SettingsValues, *, codex: Codex | None = None
) -> None:
    if codex is None:
        with app_server_pool.borrow_codex(
            Codex, enable_memories=settings.enable_memories
        ) as borrowed:
            borrowed.thread_unarchive(session_id)
    else:
        codex.thread_unarchive(session_id)


def _archive_session_for_turn(session_id: str, settings: SettingsValues) -> None:
    with app_server_pool.borrow_codex(
        Codex, enable_memories=settings.enable_memories
    ) as codex:
        codex.thread_archive(session_id)


def _restore_archived_session_for_rejected_turn(
    session_id: str, settings: SettingsValues
) -> None:
    try:
        _archive_session_for_turn(session_id, settings)
        _record_session_rearchived_after_rejected_turn(session_id)
    except Exception:
        logger.warning(
            "failed to restore archived session after rejected follow-up: %s",
            session_id,
            exc_info=True,
        )


def _session_detail_metadata(session_id: str) -> SessionMetadata | None:
    metadata = (
        SessionMetadata.objects.select_related("project")
        .filter(thread_id=session_id)
        .first()
    )
    if metadata is None or metadata.codex_path:
        return metadata
    rollout_path = _stored_rollout_path_for_thread(session_id)
    if rollout_path is None:
        return metadata
    metadata.codex_path = str(rollout_path)
    metadata.codex_archived = metadata.codex_archived or _rollout_path_is_archived(
        rollout_path
    )
    SessionMetadata.objects.filter(pk=metadata.pk, codex_path="").update(
        codex_path=metadata.codex_path,
        codex_archived=metadata.codex_archived,
        codex_last_synced_at=timezone.now(),
    )
    return metadata


def _stored_rollout_path_for_thread(
    session_id: str, *, archived: bool | None = None
) -> Path | None:
    if not session_id:
        return None
    codex_home = codex_pool.codex_home_dir()
    pattern = f"rollout-*-{glob.escape(session_id)}.jsonl"
    base_names = (
        (_ARCHIVED_SESSIONS_DIR, "sessions")
        if archived
        else ("sessions", _ARCHIVED_SESSIONS_DIR)
    )
    for base_name in base_names:
        base = codex_home / base_name
        if not base.is_dir():
            continue
        try:
            for path in sorted(base.rglob(pattern), reverse=True):
                if path.is_file() and _rollout_filename_matches_thread_id(
                    path, session_id
                ):
                    return path
        except OSError:
            logger.warning("failed to search Codex rollout directory: %s", base)
    return None


def _rollout_filename_matches_thread_id(path: Path, session_id: str) -> bool:
    match = _ROLLOUT_FILENAME_RE.fullmatch(path.name)
    return match is not None and match.group("thread_id") == session_id


def _metadata_resume_for_inactive_session(
    session_id: str,
    metadata: SessionMetadata | None,
    *,
    active_instance: CodexInstance | None,
    active_system_workflow: SystemWorkflow | None,
    require_system_agent_thread: bool,
) -> _MetadataResume | None:
    if metadata is None or require_system_agent_thread:
        return None
    archived = (
        _metadata_indicates_archived(metadata)
        or _metadata_rollout_path_indicates_archived(metadata)
    )
    # Archived Codex threads cannot be resumed. Their rollout remains the
    # authoritative detail source even if stale workflow state is still active.
    if not archived and (
        active_instance is not None or active_system_workflow is not None
    ):
        return None
    rollout_path = _rollout_path_from_value(metadata.codex_path)
    if rollout_path is None:
        return None
    rollout_data = _session_detail_data_for_metadata_resume(rollout_path)
    if rollout_data is None:
        return None
    thread = _metadata_thread(metadata, rollout_path=rollout_path)
    entries = tuple(collapse_flat_entries(list(rollout_data.flat_entries)))
    if not archived and not _entries_include_transcript(entries):
        return None
    model_config = rollout_data.latest_model_config or _stored_model_config_for_session(
        session_id
    )
    return _MetadataResume(
        thread=thread,
        entries=entries,
        model=model_config.model if model_config is not None else "",
        reasoning_effort=(
            model_config.reasoning_effort if model_config is not None else ""
        ),
        model_config=model_config,
        rollout_data=rollout_data,
    )


def _pending_resume_for_active_session(
    session_id: str,
    metadata: SessionMetadata | None,
    *,
    active_instance: CodexInstance | None,
    active_system_workflow: SystemWorkflow | None,
) -> _MetadataResume | None:
    if active_instance is None and active_system_workflow is None:
        return None
    cwd = ""
    if active_instance is not None and active_instance.cwd:
        cwd = active_instance.cwd
    elif active_system_workflow is not None and active_system_workflow.cwd:
        cwd = active_system_workflow.cwd
    elif metadata is not None:
        cwd = metadata.cwd
    if not cwd:
        return None
    updated_at = (
        updated_at_seconds(metadata.codex_updated_at)
        if metadata is not None and metadata.codex_updated_at is not None
        else None
    )
    if updated_at is None and active_instance is not None:
        updated_at = active_instance.started_at.timestamp()
    preview = ""
    if metadata is not None and metadata.codex_preview:
        preview = metadata.codex_preview
    elif active_instance is not None:
        preview = active_instance.prompt
    thread = _MetadataThread(
        id=session_id,
        cwd=cwd,
        path=metadata.codex_path if metadata is not None else "",
        name=metadata.codex_name if metadata is not None else "",
        preview=preview,
        created_at=(
            updated_at_seconds(metadata.codex_created_at)
            if metadata is not None
            else None
        ),
        updated_at=updated_at,
        archived=False,
        thread_source=metadata.codex_thread_source if metadata is not None else "",
    )
    model_config = (
        rollout.SessionModelConfig(
            model=active_instance.model,
            reasoning_effort=active_instance.reasoning_effort,
        )
        if active_instance is not None
        else None
    )
    return _MetadataResume(
        thread=thread,
        entries=(),
        model=model_config.model if model_config is not None else "",
        reasoning_effort=(
            model_config.reasoning_effort if model_config is not None else ""
        ),
        model_config=model_config,
    )


def _metadata_thread(
    metadata: SessionMetadata, *, rollout_path: Path | None = None
) -> _MetadataThread:
    path = str(rollout_path) if rollout_path is not None else metadata.codex_path
    return _MetadataThread(
        id=metadata.thread_id,
        cwd=metadata.cwd,
        path=path,
        name=metadata.codex_name,
        preview=metadata.codex_preview,
        created_at=updated_at_seconds(metadata.codex_created_at),
        updated_at=updated_at_seconds(metadata.codex_updated_at),
        archived=metadata.codex_archived
        or (rollout_path is not None and _rollout_path_is_archived(rollout_path)),
        thread_source=metadata.codex_thread_source,
    )


def _session_detail_data_for_metadata_resume(
    rollout_path: Path,
) -> rollout.SessionDetailData | None:
    try:
        return rollout.session_detail_data(rollout_path)
    except Exception:
        logger.exception(
            "failed to parse rollout %s for metadata resume; falling back to SDK turns",
            rollout_path,
        )
        return None


def _entries_include_transcript(entries: Iterable[Mapping[str, Any]]) -> bool:
    return any(entry.get("kind") in {"user", "agent"} for entry in entries)


def _stored_model_config_for_session(
    session_id: str,
) -> rollout.SessionModelConfig | None:
    """Return the latest worker row's atomic model configuration."""
    row = (
        CodexInstance.objects.filter(thread_id=session_id)
        .order_by("-started_at", "-pk")
        .values_list("model", "reasoning_effort")
        .first()
    )
    if row is None:
        return None
    model, reasoning_effort = row
    return rollout.SessionModelConfig(
        model=model,
        reasoning_effort=reasoning_effort,
    )
