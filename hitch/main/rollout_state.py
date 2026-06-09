"""Read the on-disk state of a Codex rollout (transcript) file.

A rollout file is identified by its path plus modification time; that pair is
enough to tell whether a cached, derived-from-transcript value (token usage, PR
stage, ...) is still current without re-parsing the file. When a live session's
rollout has been archived out from under us, the lookup also resolves the
archived copy under ``archived_sessions/`` so cached values keep matching.
"""

from __future__ import annotations

from pathlib import Path
from stat import S_ISREG
from typing import Any, NamedTuple

_ARCHIVED_SESSIONS_DIR = "archived_sessions"

# Codex's archived rollouts live at most four levels below the
# ``archived_sessions/`` directory (``archived_sessions/YYYY/MM/DD/rollout-*.jsonl``);
# five gives a small cushion for future structural changes without re-opening
# the false-positive case where a user's CODEX_HOME unrelatedly traverses an
# ``archived_sessions`` parent.
_ARCHIVED_SESSIONS_ANCESTOR_DEPTH = 5


class _RolloutFileState(NamedTuple):
    path: Path
    mtime_ns: int


def _rollout_path_for(thread: Any) -> Path | None:
    return _rollout_path_from_value(getattr(thread, "path", None))


def _rollout_path_from_value(path: object) -> Path | None:
    rollout_state = _rollout_file_state_from_value(path)
    return rollout_state.path if rollout_state is not None else None


def _rollout_file_state_from_value(path: object) -> _RolloutFileState | None:
    if not isinstance(path, str) or not path:
        return None
    rollout_path = Path(path)
    rollout_state = _rollout_file_state_for_path(rollout_path)
    if rollout_state is not None:
        return rollout_state
    return _archived_rollout_file_state_for_missing_session_path(rollout_path)


def _rollout_file_state_for_path(rollout_path: Path) -> _RolloutFileState | None:
    try:
        stat_result = rollout_path.stat()
    except OSError:
        return None
    if not S_ISREG(stat_result.st_mode):
        return None
    return _RolloutFileState(path=rollout_path, mtime_ns=stat_result.st_mtime_ns)


def _archived_rollout_file_state_for_missing_session_path(
    rollout_path: Path,
) -> _RolloutFileState | None:
    if rollout_path.suffix != ".jsonl" or not rollout_path.name.startswith("rollout-"):
        return None
    sessions_dir = next(
        (parent for parent in rollout_path.parents if parent.name == "sessions"),
        None,
    )
    if sessions_dir is None:
        return None
    archived_dir = sessions_dir.parent / _ARCHIVED_SESSIONS_DIR
    candidates = [archived_dir / rollout_path.name]
    try:
        archived_relative_path = archived_dir / rollout_path.relative_to(sessions_dir)
    except ValueError:
        archived_relative_path = None
    if archived_relative_path is not None and archived_relative_path not in candidates:
        candidates.append(archived_relative_path)
    for candidate in candidates:
        rollout_state = _rollout_file_state_for_path(candidate)
        if rollout_state is not None:
            return rollout_state
    return None


def _rollout_mtime_ns(rollout_path: Path | None) -> int:
    if rollout_path is None:
        return 0
    rollout_state = _rollout_file_state_for_path(rollout_path)
    return rollout_state.mtime_ns if rollout_state is not None else 0


def _thread_is_archived(thread: Any) -> bool:
    """Return whether Codex resumed this thread from archived rollout storage."""
    archived = getattr(thread, "archived", None)
    if isinstance(archived, bool):
        return archived
    path = getattr(thread, "path", None)
    if not isinstance(path, str) or not path:
        return False
    return _rollout_path_is_archived(Path(path))


def _rollout_path_is_archived(rollout_path: Path) -> bool:
    # Walk only the rollout file's immediate ancestry. Scanning the full path
    # for ``archived_sessions`` would false-positive every active session
    # whose ``CODEX_HOME`` happens to traverse an unrelated directory of
    # that name (e.g. ``/data/archived_sessions/<user>/.codex/sessions/...``).
    return any(
        parent.name == _ARCHIVED_SESSIONS_DIR
        for parent in list(rollout_path.parents)[:_ARCHIVED_SESSIONS_ANCESTOR_DEPTH]
    )
