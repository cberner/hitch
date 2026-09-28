"""Display and pagination helpers for indexed sessions."""

from typing import Any

from django.db.models import QuerySet

from hitch.main.models import SessionMetadata
from hitch.main.runtime.sdk_values import latest_updated_at, updated_at_seconds
from hitch.main.sessions.session_cursor import _index_cursor_for_sort_key


def _updated_at_sort_key(updated_at: Any) -> float:
    seconds = updated_at_seconds(updated_at)
    return seconds if seconds is not None else 0.0


def _sorted_visible_index_rows(
    rows: QuerySet[SessionMetadata],
) -> list[dict[str, Any]]:
    return _sort_session_rows(
        [
            {
                "id": row["thread_id"],
                "updated_at": latest_updated_at(row["codex_updated_at"]),
            }
            for row in rows.values("thread_id", "codex_updated_at")
        ]
    )


def _session_row_for_metadata(metadata: SessionMetadata) -> dict[str, Any]:
    return {
        "id": metadata.thread_id,
        "cwd": metadata.cwd,
        "updated_at": latest_updated_at(metadata.codex_updated_at),
        "display_title": metadata.codex_display_title or metadata.thread_id,
        "name_value": metadata.codex_name,
        "is_archived": metadata.codex_archived,
        "project": metadata.project,
        "codex_path": metadata.codex_path,
        "has_activity": bool(metadata.codex_preview),
        "stage_main_updated_at": metadata.codex_updated_at,
        "stage_cache_key": metadata.derived_stage,
        "stage_cache_mtime_ns": metadata.derived_stage_source_mtime_ns,
    }


def _sort_session_rows(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        sessions,
        key=_session_index_sort_key,
        reverse=True,
    )


def _session_index_sort_key(session: dict[str, Any]) -> tuple[float, str]:
    return (_updated_at_sort_key(session["updated_at"]), str(session["id"]))


def _index_cursor_for_session(session: dict[str, Any]) -> str:
    return _index_cursor_for_sort_key(_session_index_sort_key(session))


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        return 0
    return max(parsed, 0)
