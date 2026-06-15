"""Token-usage parsing, caching, refresh scheduling, and lifetime formatting.

Token counts live only in the on-disk Codex rollout file; this module reads
them, caches the result per session (``ArchivedSessionTokenUsage``), schedules
best-effort background refreshes, and formats both per-session and lifetime
roll-ups for the UI.
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
from collections.abc import Iterable, Iterator, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, NamedTuple

from django.db import close_old_connections, transaction
from django.utils import timezone
from openai_codex import AppServerError, Codex
from openai_codex.errors import InvalidRequestError

from hitch.main import formatting
from hitch.main.models import (
    TOKEN_USAGE_LOGIC_VERSION,
    ArchivedSessionTokenUsage,
    CodexInstance,
    Project,
    SessionMetadata,
)
from hitch.main.runtime import app_server_pool, rollout
from hitch.main.runtime.rollout_state import (
    _rollout_file_state_from_value,
    _rollout_path_for,
    _RolloutFileState,
    _thread_is_archived,
)
from hitch.main.runtime.sdk_values import updated_at_seconds
from hitch.main.sessions import session_index
from hitch.main.workflows import system_agents

logger = logging.getLogger(__name__)

_USAGE_TOKEN_REFRESH_LOCK = threading.Lock()
_USAGE_TOKEN_REFRESH_IN_FLIGHT = False
_USAGE_TOKEN_REFRESH_BATCH_SIZE = 25
_USAGE_TOKEN_REFRESH_CHECKED_UPDATE_BATCH_SIZE = 500
_USAGE_TOKEN_REFRESH_CHECK_INTERVAL = timedelta(seconds=30)


@dataclass(frozen=True)
class _UsageTokenRefreshCandidate:
    thread_id: str
    codex_path: str
    usage_last_checked_at: datetime | None


@dataclass(frozen=True)
class _UsageTokenRefreshItem:
    thread_id: str
    path: str


@dataclass(frozen=True)
class _UsageTokenRefreshThread:
    id: str
    path: str


type _UsageTokenRefreshSource = SessionMetadata | _UsageTokenRefreshCandidate
type _UsageTokenRefreshWork = _UsageTokenRefreshCandidate | _UsageTokenRefreshItem


class _UsageTokenCacheState(NamedTuple):
    refresh_pending: bool
    cache_usable: bool


_TOKEN_USAGE_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "total_tokens",
    "context_tokens",
    "model_context_window",
)
# Canonical definition lives in ``models`` so the Claude worker can stamp rows
# with the same version without importing this (heavy) module. A cached
# ArchivedSessionTokenUsage row stamped below this is treated as stale and
# recomputed even when the (immutable) rollout file is byte-for-byte unchanged,
# so counting-logic fixes reach already-cached archived sessions.
_TOKEN_USAGE_LOGIC_VERSION = TOKEN_USAGE_LOGIC_VERSION
_MISSING_TOKEN_USAGE_CACHE = object()


def _token_usage_for(thread: Any) -> dict[str, str] | None:
    """Return formatted input/cached/output token counts, or None.

    Token usage is only persisted by Codex in the on-disk rollout file (as
    ``TokenCount`` event_msg entries); the SDK ``Thread`` does not carry it.
    Archived sessions use ``ArchivedSessionTokenUsage`` as a cache once the
    value has been parsed. Live sessions always parse the current rollout so
    the page reflects the latest turn.
    """
    usage = _token_usage_numbers_for(thread)
    if usage is None:
        return None
    return _format_session_token_usage(usage)


def _claude_token_usage_for(session_id: str) -> dict[str, str] | None:
    """Return formatted token counts for a Claude thread, or None.

    Claude sessions have no rollout file: the worker writes the counts directly
    into the ArchivedSessionTokenUsage cache (``rollout_path=""``) at turn
    completion, so the value is read straight from there rather than parsed.
    """
    cache = ArchivedSessionTokenUsage.objects.filter(thread_id=session_id).first()
    if (
        cache is None
        or cache.rollout_path != ""
        or not _cached_token_usage_logic_is_current(cache)
    ):
        return None
    return _format_session_token_usage(_token_usage_from_cache(cache))


def _format_session_token_usage(usage: Mapping[str, int]) -> dict[str, str]:
    formatted = {
        "input": _format_token_count(_non_cached_input_tokens(usage)),
        "cached": _format_token_count(usage["cached_input_tokens"]),
        "output": _format_token_count(usage["output_tokens"]),
    }
    context_tokens = usage.get("context_tokens", 0)
    context_window = usage.get("model_context_window", 0)
    if context_tokens > 0 and context_window > 0:
        percent = round((context_tokens / context_window) * 100)
        percent = min(100, max(0, percent))
        formatted.update(
            {
                "context": f"{percent}%",
                "context_title": f"{context_tokens:,} of {context_window:,} tokens in current context",
            }
        )
    return formatted


def _token_usage_numbers_for(thread: Any) -> dict[str, int] | None:
    if not _thread_is_archived(thread):
        return _latest_token_usage_numbers_for(thread)
    snapshot = _token_usage_snapshot_for(thread)
    return snapshot["usage"] if snapshot is not None else None


def _token_usage_snapshot_for(
    thread: Any,
    cached_usage: ArchivedSessionTokenUsage | object = _MISSING_TOKEN_USAGE_CACHE,
) -> dict[str, Any] | None:
    thread_id = getattr(thread, "id", None)
    if not isinstance(thread_id, str) or not thread_id:
        usage, daily_usage = _parse_token_usage_and_daily(_rollout_path_for(thread))
        if usage is None:
            return None
        return {"usage": usage, "daily_usage": daily_usage}
    # Capture the rollout's mtime once, before parsing, and stamp the cache with
    # it. Re-stat'ing after the read would let a concurrent append mark the
    # cache "current" while it holds pre-append numbers, so the stale value
    # would never be refreshed for a session that then goes idle. Stamping the
    # pre-read mtime instead means any append during parsing surfaces as a
    # mismatch on the next read and triggers a re-parse.
    rollout_state = _rollout_file_state_from_value(getattr(thread, "path", None))
    cached = (
        ArchivedSessionTokenUsage.objects.filter(thread_id=thread_id).first()
        if cached_usage is _MISSING_TOKEN_USAGE_CACHE
        else cached_usage
    )
    cached = cached if isinstance(cached, ArchivedSessionTokenUsage) else None
    rollout_path = rollout_state.path if rollout_state is not None else None
    if (
        cached is not None
        and _cached_token_usage_is_current_for_state(cached, rollout_state)
        and _cached_token_usage_has_daily_usage(cached, rollout_path)
    ):
        return {
            "usage": _token_usage_from_cache(cached),
            "daily_usage": _daily_token_usage_from_cache(cached),
        }
    usage, daily_usage = _parse_token_usage_and_daily(rollout_path)
    if usage is None:
        if cached is None or not _cached_token_usage_is_current_for_state(
            cached, rollout_state
        ):
            return None
        return {
            "usage": _token_usage_from_cache(cached),
            "daily_usage": _daily_token_usage_from_cache(cached),
        }
    cached, _created = ArchivedSessionTokenUsage.objects.update_or_create(
        thread_id=thread_id,
        defaults={
            **_token_usage_cache_defaults(
                rollout_path,
                rollout_state.mtime_ns if rollout_state is not None else 0,
                usage,
            ),
            "daily_usage": daily_usage,
        },
    )
    return {"usage": _token_usage_from_cache(cached), "daily_usage": daily_usage}


def _latest_token_usage_numbers_for(thread: Any) -> dict[str, int] | None:
    rollout_path = _rollout_path_for(thread)
    if rollout_path is None:
        return None
    usage = rollout.latest_token_usage(rollout_path)
    if usage is None:
        return None
    return {key: usage.get(key, 0) for key in _TOKEN_USAGE_KEYS}


def _parse_token_usage_and_daily(
    rollout_path: Path | None,
) -> tuple[dict[str, int] | None, dict[str, dict[str, int]]]:
    """Parse the headline usage and per-day breakdown from one rollout read.

    Both figures come from a single in-memory snapshot so they cannot disagree
    about the file's contents the way two independent reads can when an append
    lands between them.
    """
    if rollout_path is None:
        return None, {}
    raw_usage, history = rollout.token_usage_snapshot(rollout_path)
    if raw_usage is None:
        return None, {}
    usage = {key: raw_usage.get(key, 0) for key in _TOKEN_USAGE_KEYS}
    return usage, _daily_token_usage_from_history(history)


def _cached_token_usage_is_current_for_state(
    cache: ArchivedSessionTokenUsage, rollout_state: _RolloutFileState | None
) -> bool:
    # A row produced by superseded counting logic is never current, even when
    # the path is missing/unreadable: returning True there for a stale-version
    # row would keep serving its pre-fix counts indefinitely, since archived
    # rollouts are immutable and the read path short-circuits before re-parsing.
    if not _cached_token_usage_logic_is_current(cache):
        return False
    # ``rollout_state is None`` means the path is missing/unreadable; there is
    # nothing to compare against, so treat the cache as current rather than
    # discarding the only numbers we have.
    if rollout_state is None:
        return True
    return _cached_token_usage_matches_rollout_state(cache, rollout_state)


def _cached_token_usage_logic_is_current(cache: ArchivedSessionTokenUsage) -> bool:
    return cache.usage_logic_version >= _TOKEN_USAGE_LOGIC_VERSION


def _cached_token_usage_matches_rollout_state(
    cache: ArchivedSessionTokenUsage, rollout_state: _RolloutFileState
) -> bool:
    return (
        cache.rollout_path == str(rollout_state.path)
        and cache.rollout_mtime_ns == rollout_state.mtime_ns
        and _cached_token_usage_logic_is_current(cache)
    )


def _cached_token_usage_has_daily_usage(
    cache: ArchivedSessionTokenUsage, rollout_path: Path | None
) -> bool:
    if rollout_path is None or _daily_token_usage_from_cache(cache):
        return True
    # An all-zero row (a rollout with no token_count events) legitimately has
    # no per-day history; demanding a non-empty daily map would skip the cache
    # and re-parse such rollouts on every single read, forever.
    return not _cached_token_usage_has_counts(cache)


def _cached_token_usage_has_counts(cache: ArchivedSessionTokenUsage) -> bool:
    return any(
        value > 0
        for value in (
            cache.input_tokens,
            cache.cached_input_tokens,
            cache.output_tokens,
            cache.total_tokens,
            cache.context_tokens,
        )
    )


def _token_usage_cache_defaults(
    rollout_path: Path | None, rollout_mtime_ns: int, usage: dict[str, int]
) -> dict[str, str | int]:
    # ``rollout_mtime_ns`` must be captured before the rollout was parsed, never
    # re-stat'd here: a fresh stat could record an mtime newer than the parsed
    # content and mask a concurrent append as a cache hit.
    return {
        "rollout_path": str(rollout_path) if rollout_path is not None else "",
        "rollout_mtime_ns": rollout_mtime_ns,
        "usage_logic_version": _TOKEN_USAGE_LOGIC_VERSION,
        "input_tokens": usage["input_tokens"],
        "cached_input_tokens": usage["cached_input_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
        "context_tokens": usage["context_tokens"],
        "model_context_window": usage["model_context_window"],
    }


def _token_usage_from_cache(cache: ArchivedSessionTokenUsage) -> dict[str, int]:
    return {
        "input_tokens": cache.input_tokens,
        "cached_input_tokens": cache.cached_input_tokens,
        "output_tokens": cache.output_tokens,
        "total_tokens": cache.total_tokens,
        "context_tokens": cache.context_tokens,
        "model_context_window": cache.model_context_window,
    }


def _daily_token_usage_from_cache(cache: ArchivedSessionTokenUsage) -> dict[str, dict[str, int]]:
    if not isinstance(cache.daily_usage, dict):
        return {}
    daily: dict[str, dict[str, int]] = {}
    for date_key, values in cache.daily_usage.items():
        if not isinstance(date_key, str) or not isinstance(values, dict):
            continue
        daily[date_key] = {
            "input": _coerce_usage_int(values.get("input")),
            "output": _coerce_usage_int(values.get("output")),
            "cached": _coerce_usage_int(values.get("cached")),
        }
    return daily


def _coerce_usage_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    return value if isinstance(value, int) and value > 0 else 0


def _daily_token_usage_from_history(
    history: list[dict[str, int]],
) -> dict[str, dict[str, int]]:
    usage_by_date: dict[str, dict[str, int]] = {}
    previous = _empty_raw_token_usage()
    for event in history:
        date_key = datetime.fromtimestamp(event["timestamp"], UTC).date().isoformat()
        bucket = usage_by_date.setdefault(date_key, _empty_lifetime_token_usage())
        input_delta = max(event["input_tokens"] - previous["input_tokens"], 0)
        cached_delta = max(
            event["cached_input_tokens"] - previous["cached_input_tokens"], 0
        )
        output_delta = max(event["output_tokens"] - previous["output_tokens"], 0)
        bucket["input"] += max(input_delta - cached_delta, 0)
        bucket["output"] += output_delta
        bucket["cached"] += cached_delta
        previous = {
            "input_tokens": event["input_tokens"],
            "cached_input_tokens": event["cached_input_tokens"],
            "output_tokens": event["output_tokens"],
        }
    return usage_by_date


def _format_token_count(value: int) -> str:
    return f"{value:,}"


# Codex reports cached input as part of input_tokens and total_tokens; keep
# cache as a breakdown rather than adding it back into displayed totals.
def _non_cached_input_tokens(usage: Mapping[str, int]) -> int:
    return max(usage.get("input_tokens", 0) - usage.get("cached_input_tokens", 0), 0)


def _lifetime_token_usage_for_metadata(
    metadata_rows: list[SessionMetadata],
    *,
    selected_project_id: int | None = None,
) -> dict[str, Any]:
    accepted_visible_thread_ids = system_agents.accepted_visible_system_thread_ids()
    hidden_thread_ids = system_agents.hidden_thread_ids(
        accepted_visible_thread_ids=accepted_visible_thread_ids
    )
    hidden_thread_ids.update(
        metadata.thread_id
        for metadata in metadata_rows
        if metadata.codex_thread_source == "subagent"
        and metadata.thread_id not in accepted_visible_thread_ids
    )
    cached_usage_by_thread_id = _token_usage_caches_by_thread_ids(
        metadata.thread_id for metadata in metadata_rows
    )
    # Only empty-``codex_path`` rows are ambiguous between Claude and a fresh
    # Codex thread, so resolve the backend for those alone.
    claude_thread_ids = _claude_thread_ids(
        metadata.thread_id for metadata in metadata_rows if not metadata.codex_path
    )
    total_usage = _empty_lifetime_token_usage()
    session_usage = _empty_lifetime_token_usage()
    system_usage = _empty_lifetime_token_usage()
    selected_project_usage = _empty_lifetime_token_usage()
    selected_project_system_usage = _empty_lifetime_token_usage()
    total_by_date: dict[str, dict[str, int]] = {}
    session_by_date: dict[str, dict[str, int]] = {}
    system_by_date: dict[str, dict[str, int]] = {}
    refresh_pending_count = 0
    for metadata in metadata_rows:
        cache = cached_usage_by_thread_id.get(metadata.thread_id)
        cache_state = _usage_token_cache_state(
            metadata, cache, is_claude=metadata.thread_id in claude_thread_ids
        )
        if cache_state.refresh_pending:
            refresh_pending_count += 1
        if cache is None or not cache_state.cache_usable:
            continue
        daily_usage = _daily_token_usage_from_cache(cache)
        usage = _token_usage_from_cache(cache)
        is_system = metadata.thread_id in hidden_thread_ids
        total_usage["input"] += _non_cached_input_tokens(usage)
        total_usage["output"] += usage.get("output_tokens", 0)
        total_usage["cached"] += usage.get("cached_input_tokens", 0)
        bucket = system_usage if is_system else session_usage
        bucket["input"] += _non_cached_input_tokens(usage)
        bucket["output"] += usage.get("output_tokens", 0)
        bucket["cached"] += usage.get("cached_input_tokens", 0)
        if (
            selected_project_id is not None
            and metadata.project_id == selected_project_id
        ):
            selected_project_usage["input"] += _non_cached_input_tokens(usage)
            selected_project_usage["output"] += usage.get("output_tokens", 0)
            selected_project_usage["cached"] += usage.get("cached_input_tokens", 0)
            if is_system:
                selected_project_system_usage["input"] += _non_cached_input_tokens(
                    usage
                )
                selected_project_system_usage["output"] += usage.get("output_tokens", 0)
                selected_project_system_usage["cached"] += usage.get(
                    "cached_input_tokens", 0
                )
        _merge_daily_token_usage(total_by_date, daily_usage)
        _merge_daily_token_usage(
            system_by_date if is_system else session_by_date,
            daily_usage,
        )
    lifetime_usage = _formatted_lifetime_token_usage(
        total_usage=total_usage,
        session_usage=session_usage,
        system_usage=system_usage,
        total_by_date=total_by_date,
        session_by_date=session_by_date,
        system_by_date=system_by_date,
    )
    if selected_project_id is not None:
        lifetime_usage["selected_project"] = {
            "total": _format_lifetime_token_usage(selected_project_usage),
            "system": _format_lifetime_token_usage(selected_project_system_usage),
        }
    lifetime_usage["refresh_pending"] = refresh_pending_count > 0
    lifetime_usage["refresh_pending_count"] = refresh_pending_count
    return lifetime_usage


def _schedule_usage_token_refresh(metadata_rows: list[SessionMetadata]) -> None:
    # Refresh filtering checks rollout files, so let the worker filter candidates.
    candidates = _usage_token_refresh_candidates(metadata_rows)
    if not candidates:
        return
    transaction.on_commit(lambda: _start_usage_token_refresh_thread(candidates))


def _usage_token_refresh_candidates(
    metadata_rows: Iterable[SessionMetadata],
) -> list[_UsageTokenRefreshCandidate]:
    return [
        _UsageTokenRefreshCandidate(
            thread_id=metadata.thread_id,
            codex_path=metadata.codex_path,
            usage_last_checked_at=metadata.usage_last_checked_at,
        )
        for metadata in metadata_rows
        if metadata.thread_id
    ]


def _usage_token_cache_state(
    metadata: _UsageTokenRefreshSource,
    cache: ArchivedSessionTokenUsage | None,
    *,
    is_claude: bool = False,
) -> _UsageTokenCacheState:
    if not metadata.thread_id:
        return _UsageTokenCacheState(refresh_pending=False, cache_usable=False)
    if not metadata.codex_path:
        # A Claude thread has no rollout path; its cache row (``rollout_path ==
        # ""``) is the authoritative accumulated usage and cannot be repaired
        # from a file. Treat a usable one as current -- not a path-repair
        # candidate -- so ``/usage`` and ``/profile`` stop reporting it as
        # refresh-pending and the refresh worker stops probing the Codex
        # app-server with a local Claude UUID. A known Claude row stays
        # non-pending even before its first cache: there is nothing to repair
        # (the worker, not this path, writes the cache), whereas a freshly
        # created Codex row with an empty path is still awaiting path repair.
        cache_usable = _claude_usage_cache_is_authoritative(cache)
        return _UsageTokenCacheState(
            refresh_pending=(False if is_claude else not cache_usable),
            cache_usable=cache_usable,
        )
    if cache is None:
        return _UsageTokenCacheState(refresh_pending=True, cache_usable=False)
    rollout_state = _rollout_file_state_from_value(metadata.codex_path)
    if rollout_state is None:
        return _UsageTokenCacheState(
            refresh_pending=True,
            cache_usable=(
                cache.rollout_path == metadata.codex_path
                and _cached_token_usage_logic_is_current(cache)
            ),
        )
    cache_is_current = _cached_token_usage_matches_rollout_state(cache, rollout_state)
    if _cached_token_usage_has_counts(cache) and not _daily_token_usage_from_cache(cache):
        return _UsageTokenCacheState(
            refresh_pending=True,
            cache_usable=cache_is_current,
        )
    return _UsageTokenCacheState(
        refresh_pending=(
            not cache_is_current
            or _usage_token_refresh_check_is_stale(metadata.usage_last_checked_at)
        ),
        cache_usable=cache_is_current,
    )


def _usage_token_refresh_check_is_stale(checked_at: datetime | None) -> bool:
    if checked_at is None:
        return True
    return checked_at <= timezone.now() - _USAGE_TOKEN_REFRESH_CHECK_INTERVAL


def _usage_token_refresh_items(
    metadata_rows: Iterable[_UsageTokenRefreshSource],
    cached_usage_by_thread_id: Mapping[str, ArchivedSessionTokenUsage],
    claude_thread_ids: AbstractSet[str] = frozenset(),
) -> list[_UsageTokenRefreshItem]:
    path_repair_candidates: list[_UsageTokenRefreshSource] = []
    file_backed_candidates: list[_UsageTokenRefreshSource] = []
    for metadata in metadata_rows:
        if not metadata.thread_id:
            continue
        cache = cached_usage_by_thread_id.get(metadata.thread_id)
        if not _usage_token_refresh_needed(
            metadata, cache, is_claude=metadata.thread_id in claude_thread_ids
        ):
            continue
        if _usage_token_refresh_needs_path_repair(metadata):
            path_repair_candidates.append(metadata)
        else:
            file_backed_candidates.append(metadata)
    path_repair_candidates.sort(key=_usage_token_refresh_sort_key)
    file_backed_candidates.sort(key=_usage_token_refresh_sort_key)
    path_repair_limit = (
        _USAGE_TOKEN_REFRESH_BATCH_SIZE
        if not file_backed_candidates
        else _USAGE_TOKEN_REFRESH_BATCH_SIZE // 2
    )
    selected = path_repair_candidates[:path_repair_limit]
    selected.extend(
        file_backed_candidates[: _USAGE_TOKEN_REFRESH_BATCH_SIZE - len(selected)]
    )
    if len(selected) < _USAGE_TOKEN_REFRESH_BATCH_SIZE:
        extra_path_repair_count = _USAGE_TOKEN_REFRESH_BATCH_SIZE - len(selected)
        selected.extend(
            path_repair_candidates[
                path_repair_limit : path_repair_limit + extra_path_repair_count
            ]
        )
    return [
        _UsageTokenRefreshItem(thread_id=metadata.thread_id, path=metadata.codex_path)
        for metadata in selected
    ]


def _usage_token_refresh_sort_key(
    metadata: _UsageTokenRefreshSource,
) -> tuple[float, str]:
    last_checked_at = updated_at_seconds(metadata.usage_last_checked_at)
    return (
        last_checked_at if last_checked_at is not None else 0.0,
        metadata.thread_id,
    )


def _usage_token_refresh_needs_path_repair(
    metadata: _UsageTokenRefreshSource,
) -> bool:
    return not metadata.codex_path or _rollout_file_state_from_value(
        metadata.codex_path
    ) is None


def _claude_usage_cache_is_authoritative(
    cache: ArchivedSessionTokenUsage | None,
) -> bool:
    """Whether a cache row is a usable Claude-written row (no rollout to repair)."""
    return (
        cache is not None
        and cache.rollout_path == ""
        and _cached_token_usage_logic_is_current(cache)
    )


def _claude_thread_ids(thread_ids: Iterable[str]) -> set[str]:
    """Of ``thread_ids``, those whose latest worker backend is Claude.

    A Claude thread has no Codex rollout: its usage cache is written directly by
    the worker at turn completion and is unrecoverable from the Codex app-server.
    Callers use this to keep an uncached Claude row (empty ``codex_path``) -- which
    is otherwise indistinguishable from a freshly-created Codex row -- out of Codex
    path repair, where it would stay refresh-pending and re-probe forever.
    """
    ids = {thread_id for thread_id in thread_ids if thread_id}
    if not ids:
        return set()
    latest_backend: dict[str, str] = {}
    for thread_id, backend in (
        CodexInstance.objects.filter(thread_id__in=ids)
        .order_by("thread_id", "-started_at", "-pk")
        .values_list("thread_id", "backend")
    ):
        latest_backend.setdefault(thread_id, backend)
    return {
        thread_id
        for thread_id, backend in latest_backend.items()
        if backend == CodexInstance.BACKEND_CLAUDE
    }


def _usage_token_refresh_needed(
    metadata: _UsageTokenRefreshSource,
    cache: ArchivedSessionTokenUsage | None,
    *,
    is_claude: bool = False,
) -> bool:
    if not metadata.codex_path:
        # A known Claude row never needs a refresh: it has no rollout to repair
        # and the worker writes its cache directly, so scheduling Codex path
        # repair (the empty-path branch below) would only re-probe forever.
        if is_claude:
            return False
        # A Claude row (rollout_path == "") is authoritative and unrepairable, so
        # it never needs a refresh; only a genuinely empty/uncached row does.
        return not _claude_usage_cache_is_authoritative(cache)
    rollout_state = _rollout_file_state_from_value(metadata.codex_path)
    if rollout_state is None:
        return True
    if cache is None:
        return True
    if not _cached_token_usage_matches_rollout_state(cache, rollout_state):
        return True
    return _cached_token_usage_has_counts(cache) and not _cached_token_usage_has_daily_usage(
        cache, rollout_state.path
    )


def _start_usage_token_refresh_thread(items: Iterable[_UsageTokenRefreshWork]) -> None:
    global _USAGE_TOKEN_REFRESH_IN_FLIGHT
    with _USAGE_TOKEN_REFRESH_LOCK:
        if _USAGE_TOKEN_REFRESH_IN_FLIGHT:
            return
        work_items = tuple(items)
        if not work_items:
            return
        _USAGE_TOKEN_REFRESH_IN_FLIGHT = True
    try:
        # Django's threaded dev server runs request handlers as daemon threads,
        # so make the refresh worker explicitly non-daemon.
        threading.Thread(
            target=_refresh_usage_token_cache_best_effort,
            args=(work_items,),
            name="usage-token-refresh",
            daemon=False,
        ).start()
    except Exception:
        with _USAGE_TOKEN_REFRESH_LOCK:
            _USAGE_TOKEN_REFRESH_IN_FLIGHT = False
        logger.exception("failed to start usage token refresh thread")


def _refresh_usage_token_cache_best_effort(
    items: Iterable[_UsageTokenRefreshWork],
) -> None:
    global _USAGE_TOKEN_REFRESH_IN_FLIGHT
    try:
        close_old_connections()
        with contextlib.ExitStack() as stack:
            codex: Codex | None = None
            projects: list[Project] | None = None
            for batch in _usage_token_refresh_work_batches(items):
                cached_usage_by_thread_id = _token_usage_caches_by_thread_ids(
                    item.thread_id for item in batch
                )
                for item in batch:
                    try:
                        path = item.path
                        rollout_state = _rollout_file_state_from_value(path)
                        if rollout_state is None:
                            if codex is None:
                                codex = stack.enter_context(
                                    app_server_pool.borrow_codex(
                                        Codex, enable_memories=False
                                    )
                                )
                            if projects is None:
                                projects = list(Project.objects.all())
                            path = _refresh_missing_usage_metadata_path(
                                codex, item.thread_id, projects=projects
                            )
                            rollout_state = _rollout_file_state_from_value(path)
                        if rollout_state is None:
                            continue
                        rollout_path = rollout_state.path
                        thread = _UsageTokenRefreshThread(id=item.thread_id, path=path)
                        snapshot = _token_usage_snapshot_for(
                            thread,
                            cached_usage=cached_usage_by_thread_id.get(
                                item.thread_id, _MISSING_TOKEN_USAGE_CACHE
                            ),
                        )
                        if snapshot is None and _rollout_file_parses_as_jsonl(
                            rollout_path
                        ):
                            _write_zero_token_usage_cache(
                                item.thread_id, rollout_path, rollout_state.mtime_ns
                            )
                    except Exception:
                        logger.exception(
                            "failed to refresh token usage for %s", item.thread_id
                        )
                    finally:
                        _mark_usage_token_refresh_checked(item.thread_id)
    finally:
        close_old_connections()
        with _USAGE_TOKEN_REFRESH_LOCK:
            _USAGE_TOKEN_REFRESH_IN_FLIGHT = False


def _usage_token_refresh_work_batches(
    items: Iterable[_UsageTokenRefreshWork],
) -> Iterator[list[_UsageTokenRefreshItem]]:
    refresh_items: list[_UsageTokenRefreshItem] = []
    candidates: list[_UsageTokenRefreshCandidate] = []
    for item in items:
        if isinstance(item, _UsageTokenRefreshItem):
            refresh_items.append(item)
        else:
            candidates.append(item)
    if refresh_items:
        yield refresh_items
    remaining_candidates = candidates
    # Resolve backends once for the empty-path candidates so an uncached Claude
    # row is never scheduled for (futile) Codex path repair below.
    claude_thread_ids = _claude_thread_ids(
        candidate.thread_id for candidate in candidates if not candidate.codex_path
    )
    while remaining_candidates:
        cached_usage_by_thread_id = _token_usage_caches_by_thread_ids(
            candidate.thread_id for candidate in remaining_candidates
        )
        selected_items = _usage_token_refresh_items(
            remaining_candidates, cached_usage_by_thread_id, claude_thread_ids
        )
        selected_thread_ids = {item.thread_id for item in selected_items}
        checked_thread_ids = {
            candidate.thread_id
            for candidate in remaining_candidates
            if candidate.thread_id not in selected_thread_ids
            and not _usage_token_refresh_needed(
                candidate,
                cached_usage_by_thread_id.get(candidate.thread_id),
                is_claude=candidate.thread_id in claude_thread_ids,
            )
        }
        _mark_usage_token_refresh_checked_many(checked_thread_ids)
        remaining_candidates = [
            candidate
            for candidate in remaining_candidates
            if candidate.thread_id not in selected_thread_ids
            and candidate.thread_id not in checked_thread_ids
        ]
        if selected_items:
            yield selected_items
            continue
        if remaining_candidates:
            _mark_usage_token_refresh_checked_many(
                candidate.thread_id for candidate in remaining_candidates
            )
        return


def _refresh_missing_usage_metadata_path(
    codex: Codex, thread_id: str, *, projects: list[Project]
) -> str:
    try:
        resumed = codex._client.thread_resume(thread_id)
    except (AppServerError, InvalidRequestError):
        logger.warning("failed to refresh usage metadata for %s", thread_id)
        return ""
    except Exception:
        logger.exception("failed to refresh usage metadata for %s", thread_id)
        return ""
    thread = getattr(resumed, "thread", None)
    metadata = session_index.upsert_thread(thread, projects=projects)
    if metadata is not None:
        return metadata.codex_path
    path = getattr(thread, "path", None)
    return path if isinstance(path, str) else ""


def _mark_usage_token_refresh_checked(thread_id: str) -> None:
    SessionMetadata.objects.filter(thread_id=thread_id).update(
        usage_last_checked_at=timezone.now()
    )


def _mark_usage_token_refresh_checked_many(thread_ids: Iterable[str]) -> None:
    ids: list[str] = []
    seen: set[str] = set()
    for thread_id in thread_ids:
        if not thread_id or thread_id in seen:
            continue
        seen.add(thread_id)
        ids.append(thread_id)
    if not ids:
        return
    checked_at = timezone.now()
    for start in range(0, len(ids), _USAGE_TOKEN_REFRESH_CHECKED_UPDATE_BATCH_SIZE):
        SessionMetadata.objects.filter(
            thread_id__in=ids[
                start : start + _USAGE_TOKEN_REFRESH_CHECKED_UPDATE_BATCH_SIZE
            ]
        ).update(usage_last_checked_at=checked_at)


def _rollout_file_parses_as_jsonl(rollout_path: Path) -> bool:
    try:
        text = rollout_path.read_text()
    except (OSError, UnicodeDecodeError):
        return False
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            json.loads(raw)
        except json.JSONDecodeError:
            return False
    return True


def _write_zero_token_usage_cache(
    thread_id: str, rollout_path: Path, rollout_mtime_ns: int
) -> None:
    ArchivedSessionTokenUsage.objects.update_or_create(
        thread_id=thread_id,
        defaults={
            **_token_usage_cache_defaults(
                rollout_path, rollout_mtime_ns, dict.fromkeys(_TOKEN_USAGE_KEYS, 0)
            ),
            "daily_usage": {},
        },
    )


def _formatted_lifetime_token_usage(
    *,
    total_usage: Mapping[str, int],
    session_usage: Mapping[str, int],
    system_usage: Mapping[str, int],
    total_by_date: Mapping[str, Mapping[str, int]],
    session_by_date: Mapping[str, Mapping[str, int]],
    system_by_date: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    return {
        "total": {
            **_format_lifetime_token_usage(total_usage),
            "chart": _format_lifetime_token_chart(total_by_date),
            "chart_axis": _format_lifetime_token_chart_axis(total_by_date),
        },
        "sessions": {
            **_format_lifetime_token_usage(session_usage),
            "chart": _format_lifetime_token_chart(session_by_date),
            "chart_axis": _format_lifetime_token_chart_axis(session_by_date),
        },
        "system": {
            **_format_lifetime_token_usage(system_usage),
            "chart": _format_lifetime_token_chart(system_by_date),
            "chart_axis": _format_lifetime_token_chart_axis(system_by_date),
        },
    }


def _token_usage_caches_by_thread_ids(
    thread_ids: Iterable[str],
) -> dict[str, ArchivedSessionTokenUsage]:
    ids: list[str] = []
    seen: set[str] = set()
    for thread_id in thread_ids:
        if not thread_id or thread_id in seen:
            continue
        seen.add(thread_id)
        ids.append(thread_id)
    if not ids:
        return {}
    return ArchivedSessionTokenUsage.objects.in_bulk(ids, field_name="thread_id")


def _empty_lifetime_token_usage() -> dict[str, int]:
    return {"input": 0, "output": 0, "cached": 0}


def _format_lifetime_token_usage(usage: Mapping[str, int]) -> dict[str, str]:
    return {
        "total": formatting.format_token_count(_lifetime_token_usage_total(usage)),
        "input": formatting.format_token_count(usage["input"]),
        "output": formatting.format_token_count(usage["output"]),
        "cached": formatting.format_token_count(usage["cached"]),
    }


def _lifetime_token_usage_total(usage: Mapping[str, int]) -> int:
    return usage["input"] + usage["output"] + usage["cached"]


def _merge_daily_token_usage(
    usage_by_date: dict[str, dict[str, int]],
    daily_usage: Mapping[str, Mapping[str, int]],
) -> None:
    for date_key, values in daily_usage.items():
        bucket = usage_by_date.setdefault(date_key, _empty_lifetime_token_usage())
        bucket["input"] += values.get("input", 0)
        bucket["output"] += values.get("output", 0)
        bucket["cached"] += values.get("cached", 0)


def _empty_raw_token_usage() -> dict[str, int]:
    return {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}


def _format_lifetime_token_chart(
    usage_by_date: Mapping[str, Mapping[str, int]],
) -> list[dict[str, str | int]]:
    max_total = _lifetime_token_chart_max_total(usage_by_date)
    chart: list[dict[str, str | int]] = []
    for date_key in sorted(usage_by_date):
        values = usage_by_date[date_key]
        total = values["input"] + values["output"] + values["cached"]
        chart.append(
            {
                "date": date_key,
                "input": formatting.format_token_count(values["input"]),
                "output": formatting.format_token_count(values["output"]),
                "cached": formatting.format_token_count(values["cached"]),
                "total": formatting.format_token_count(total),
                "input_percent": _chart_segment_percent(values["input"], max_total),
                "output_percent": _chart_segment_percent(values["output"], max_total),
                "cached_percent": _chart_segment_percent(values["cached"], max_total),
            }
        )
    return chart


def _format_lifetime_token_chart_axis(
    usage_by_date: Mapping[str, Mapping[str, int]],
) -> list[str]:
    if not usage_by_date:
        return []
    max_total = _lifetime_token_chart_max_total(usage_by_date)
    if max_total <= 0:
        return ["0"]
    midpoint = (max_total + 1) // 2
    ticks = [max_total]
    if 0 < midpoint < max_total:
        ticks.append(midpoint)
    ticks.append(0)
    return [formatting.format_token_count(value) for value in ticks]


def _lifetime_token_chart_max_total(
    usage_by_date: Mapping[str, Mapping[str, int]],
) -> int:
    return max(
        (
            values["input"] + values["output"] + values["cached"]
            for values in usage_by_date.values()
        ),
        default=0,
    )


def _chart_segment_percent(value: int, max_total: int) -> int:
    if value <= 0 or max_total <= 0:
        return 0
    return round((value / max_total) * 100)
