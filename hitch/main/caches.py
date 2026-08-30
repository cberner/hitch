"""Background-refresh cache layer for model lists and account rate limits.

This module holds the per-process caches (with their locks and in-flight
state) plus the helpers that schedule and perform best-effort background
refreshes against Codex. It must NOT import ``views`` -- the view layer
imports this module and calls these helpers module-qualified
(``caches._foo(...)``) so there is exactly one binding per symbol, which
keeps test ``mock.patch`` of these names intercepting both the view call
sites and the internal sibling calls between the helpers below.
"""

import logging
import threading
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, NamedTuple

from django.db import close_old_connections, transaction
from django.utils import timezone
from openai_codex import Codex, CodexError
from openai_codex.generated.v2_all import (
    GetAccountRateLimitsResponse,
    RateLimitSnapshot,
)

from hitch.main.runtime import app_server_pool, rate_limit

logger = logging.getLogger(__name__)

_MINUTES_PER_HOUR = 60
_MINUTES_PER_DAY = 24 * _MINUTES_PER_HOUR

_RATE_LIMITS_REFRESH_LOCK = threading.Lock()
_RATE_LIMITS_REFRESH_IN_FLIGHT = False
_RATE_LIMITS_CACHE_VALUE: dict[str, Any] | None = None
_RATE_LIMITS_CACHE_HAS_VALUE = False
_RATE_LIMITS_CACHE_FETCHED_AT: datetime | None = None
_RATE_LIMITS_REFRESH_ATTEMPTED_AT: datetime | None = None
# The account rate-limit endpoint is a real OpenAI ping; honour the central
# debounce floor rather than re-hitting it every render.
_RATE_LIMITS_CACHE_TTL = rate_limit.DEFAULT_MIN_INTERVAL
_RATE_LIMITS_RATE_LIMIT_KEY = "codex:account-rate-limits"
_MODELS_REFRESH_LOCK = threading.Lock()
_MODELS_REFRESH_IN_FLIGHT: set[bool] = set()
_MODELS_CACHE_VALUE: dict[bool, list[Any]] = {}
_MODELS_CACHE_FETCHED_AT: dict[bool, datetime] = {}
_MODELS_CACHE_TTL = timedelta(minutes=5)


class _RateLimitsUsageState(NamedTuple):
    rate_limits: dict[str, Any] | None
    refresh_pending: bool


def _cached_models_data(*, enable_memories: bool) -> list[Any]:
    with _MODELS_REFRESH_LOCK:
        return list(_MODELS_CACHE_VALUE.get(enable_memories, []))


def _store_models_cache(*, enable_memories: bool, models_data: list[Any]) -> None:
    with _MODELS_REFRESH_LOCK:
        _MODELS_CACHE_VALUE[enable_memories] = list(models_data)
        _MODELS_CACHE_FETCHED_AT[enable_memories] = timezone.now()


def _models_cache_has_value(*, enable_memories: bool) -> bool:
    with _MODELS_REFRESH_LOCK:
        return enable_memories in _MODELS_CACHE_FETCHED_AT


def _string_value(value: Any) -> str:
    if value is None:
        return ""
    return getattr(value, "value", str(value))


def _raw_reasoning_effort_option(raw: Any) -> SimpleNamespace | None:
    if isinstance(raw, dict):
        effort = _string_value(raw.get("reasoningEffort") or raw.get("reasoning_effort"))
        description = _string_value(raw.get("description"))
    else:
        effort = _string_value(raw)
        description = ""
    if not effort:
        return None
    return SimpleNamespace(
        reasoning_effort=SimpleNamespace(value=effort),
        description=description,
    )


def _raw_model(raw: Any) -> SimpleNamespace | None:
    if not isinstance(raw, dict):
        return None
    model_id = _string_value(raw.get("id"))
    if not model_id:
        return None
    display_name = _string_value(
        raw.get("displayName") or raw.get("display_name") or model_id
    )
    default_effort = _string_value(
        raw.get("defaultReasoningEffort") or raw.get("default_reasoning_effort")
    )
    supported_efforts = [
        option
        for option in (
            _raw_reasoning_effort_option(item)
            for item in (
                raw.get("supportedReasoningEfforts")
                or raw.get("supported_reasoning_efforts")
                or []
            )
        )
        if option is not None
    ]
    return SimpleNamespace(
        id=model_id,
        display_name=display_name,
        is_default=bool(raw.get("isDefault") or raw.get("is_default")),
        default_reasoning_effort=SimpleNamespace(value=default_effort),
        supported_reasoning_efforts=supported_efforts,
    )


def _models_data_from_raw_response(raw: Any) -> list[Any]:
    if not isinstance(raw, dict):
        raise ValueError("model/list response must be an object")
    data = raw.get("data")
    if not isinstance(data, list):
        raise ValueError("model/list response data must be a list")
    return [model for model in (_raw_model(item) for item in data) if model is not None]


def _models_data_from_codex(codex: Any) -> list[Any]:
    raw_request = getattr(getattr(codex, "_client", None), "_request_raw", None)
    if callable(raw_request):
        try:
            return _models_data_from_raw_response(
                raw_request("model/list", {"includeHidden": False})
            )
        except Exception:
            pass
    return list(codex.models().data)


def _fetch_models_data(*, enable_memories: bool, codex_cls: Any = None) -> list[Any]:
    with app_server_pool.borrow_codex(
        codex_cls or Codex, enable_memories=enable_memories
    ) as codex:
        models_data = _models_data_from_codex(codex)
    _store_models_cache(enable_memories=enable_memories, models_data=models_data)
    return models_data


def _cached_models_for_session_detail(*, enable_memories: bool) -> list[Any]:
    models_data = _cached_models_data(enable_memories=enable_memories)
    _schedule_models_refresh(enable_memories=enable_memories)
    return models_data


def _schedule_models_refresh(*, enable_memories: bool) -> None:
    if not _models_refresh_needed(enable_memories=enable_memories):
        return
    transaction.on_commit(
        lambda: _start_models_refresh_thread(enable_memories=enable_memories)
    )


def _models_refresh_needed(*, enable_memories: bool) -> bool:
    with _MODELS_REFRESH_LOCK:
        fetched_at = _MODELS_CACHE_FETCHED_AT.get(enable_memories)
        if fetched_at is None:
            return True
        return timezone.now() - _MODELS_CACHE_TTL >= fetched_at


def _start_models_refresh_thread(*, enable_memories: bool) -> None:
    with _MODELS_REFRESH_LOCK:
        if enable_memories in _MODELS_REFRESH_IN_FLIGHT:
            return
        fetched_at = _MODELS_CACHE_FETCHED_AT.get(enable_memories)
        if fetched_at is not None and timezone.now() - _MODELS_CACHE_TTL < fetched_at:
            return
        _MODELS_REFRESH_IN_FLIGHT.add(enable_memories)
    try:
        threading.Thread(
            target=_refresh_models_cache_best_effort,
            kwargs={"enable_memories": enable_memories},
            name="models-refresh",
            daemon=True,
        ).start()
    except Exception:
        with _MODELS_REFRESH_LOCK:
            _MODELS_REFRESH_IN_FLIGHT.discard(enable_memories)
        logger.exception("failed to start models refresh thread")


def _refresh_models_cache_best_effort(*, enable_memories: bool) -> None:
    try:
        close_old_connections()
        _fetch_models_data(enable_memories=enable_memories)
    except Exception:
        logger.exception("failed to refresh models cache")
    finally:
        close_old_connections()
        with _MODELS_REFRESH_LOCK:
            _MODELS_REFRESH_IN_FLIGHT.discard(enable_memories)


def _rate_limits_for_usage_context(*, enable_memories: bool) -> _RateLimitsUsageState:
    _schedule_rate_limits_refresh(enable_memories=enable_memories)
    with _RATE_LIMITS_REFRESH_LOCK:
        return _RateLimitsUsageState(
            rate_limits=(
                _RATE_LIMITS_CACHE_VALUE if _RATE_LIMITS_CACHE_HAS_VALUE else None
            ),
            refresh_pending=(
                not _RATE_LIMITS_CACHE_HAS_VALUE and _RATE_LIMITS_REFRESH_IN_FLIGHT
            ),
        )


def _schedule_rate_limits_refresh(*, enable_memories: bool) -> None:
    if not _rate_limits_refresh_needed():
        return
    transaction.on_commit(
        lambda: _start_rate_limits_refresh_thread(enable_memories=enable_memories)
    )


def _rate_limits_last_attempt_at_locked() -> datetime | None:
    if _RATE_LIMITS_CACHE_FETCHED_AT is None:
        return _RATE_LIMITS_REFRESH_ATTEMPTED_AT
    if _RATE_LIMITS_REFRESH_ATTEMPTED_AT is None:
        return _RATE_LIMITS_CACHE_FETCHED_AT
    return max(_RATE_LIMITS_CACHE_FETCHED_AT, _RATE_LIMITS_REFRESH_ATTEMPTED_AT)


def _rate_limits_refresh_needed() -> bool:
    with _RATE_LIMITS_REFRESH_LOCK:
        if _RATE_LIMITS_REFRESH_IN_FLIGHT:
            return False
        last_attempt_at = _rate_limits_last_attempt_at_locked()
        if last_attempt_at is None:
            return True
        return timezone.now() - _RATE_LIMITS_CACHE_TTL >= last_attempt_at


def _start_rate_limits_refresh_thread(*, enable_memories: bool) -> None:
    global _RATE_LIMITS_REFRESH_IN_FLIGHT
    global _RATE_LIMITS_REFRESH_ATTEMPTED_AT
    with _RATE_LIMITS_REFRESH_LOCK:
        if _RATE_LIMITS_REFRESH_IN_FLIGHT:
            return
        last_attempt_at = _rate_limits_last_attempt_at_locked()
        if last_attempt_at is not None and (
            timezone.now() - _RATE_LIMITS_CACHE_TTL < last_attempt_at
        ):
            return
        _RATE_LIMITS_REFRESH_IN_FLIGHT = True
    try:
        threading.Thread(
            target=_refresh_rate_limits_cache_best_effort,
            kwargs={"enable_memories": enable_memories},
            name="rate-limits-refresh",
            daemon=True,
        ).start()
    except Exception:
        with _RATE_LIMITS_REFRESH_LOCK:
            _RATE_LIMITS_REFRESH_IN_FLIGHT = False
            _RATE_LIMITS_REFRESH_ATTEMPTED_AT = timezone.now()
        logger.exception("failed to start rate limits refresh thread")


def _refresh_rate_limits_cache_best_effort(*, enable_memories: bool) -> None:
    global _RATE_LIMITS_CACHE_FETCHED_AT
    global _RATE_LIMITS_CACHE_HAS_VALUE
    global _RATE_LIMITS_CACHE_VALUE
    global _RATE_LIMITS_REFRESH_ATTEMPTED_AT
    global _RATE_LIMITS_REFRESH_IN_FLIGHT
    rate_limits: dict[str, Any] | None = None
    claimed = False
    fetched = False
    try:
        # Hit OpenAI for the account rate limits only when the central, app-wide
        # debounce floor allows; otherwise serve the last cached value. This is
        # the cross-process guard the per-process TTL cannot provide.
        claimed = rate_limit.claim(_RATE_LIMITS_RATE_LIMIT_KEY)
        if claimed:
            close_old_connections()
            with app_server_pool.borrow_codex(
                Codex, enable_memories=enable_memories
            ) as codex:
                rate_limits = _fetch_rate_limits(codex)
            fetched = True
    except Exception:
        logger.exception("failed to refresh rate limits cache")
    finally:
        close_old_connections()
        with _RATE_LIMITS_REFRESH_LOCK:
            attempted_at = timezone.now()
            if fetched and (rate_limits is not None or not _RATE_LIMITS_CACHE_HAS_VALUE):
                _RATE_LIMITS_CACHE_VALUE = rate_limits
                _RATE_LIMITS_CACHE_HAS_VALUE = True
            # Preserve a usable snapshot through failed refreshes. Record local
            # backoff only after winning the shared claim; a cold process that
            # loses the claim must remain eligible to observe when the shared
            # throttle becomes due.
            if fetched or _RATE_LIMITS_CACHE_HAS_VALUE:
                _RATE_LIMITS_CACHE_FETCHED_AT = attempted_at
            if claimed:
                _RATE_LIMITS_REFRESH_ATTEMPTED_AT = attempted_at
            _RATE_LIMITS_REFRESH_IN_FLIGHT = False


def _fetch_rate_limits(codex: Codex) -> dict[str, Any] | None:
    """Fetch the account/rateLimits/read snapshot, or None if unavailable.

    The endpoint is meaningful only when Codex is talking to a real OpenAI
    account; local-dev (no auth, custom provider via ollama) and older
    Codex builds will fail with MethodNotFound or an auth error. The
    usage page must still render in those modes, so any failure here
    swallows into None and the page shows its empty state.
    """
    try:
        response = codex._client.request(
            "account/rateLimits/read",
            None,
            response_model=GetAccountRateLimitsResponse,
        )
    except CodexError:
        return None
    except Exception:
        logger.exception("failed to fetch account rate limits; showing usage empty state")
        return None
    return _format_rate_limit_snapshot(response.rate_limits)


def _format_rate_limit_snapshot(snapshot: RateLimitSnapshot) -> dict[str, Any] | None:
    """Project a RateLimitSnapshot into a template-friendly dict.

    Returns None when neither the primary nor secondary window is set; the
    template hides the section entirely in that case rather than render an
    empty header. ``used_percent`` from Codex describes consumption, so we
    expose ``remaining_percent`` (the more intuitive framing for a user
    looking at their remaining budget) alongside it.
    """
    windows: list[dict[str, Any]] = []
    for label, window in (("Primary", snapshot.primary), ("Secondary", snapshot.secondary)):
        if window is None:
            continue
        used = max(0, min(100, window.used_percent))
        windows.append(
            {
                "label": label,
                "used_percent": used,
                "remaining_percent": 100 - used,
                "resets_at": window.resets_at,
                "window_duration_label": _format_window_duration(window.window_duration_mins),
            }
        )
    if not windows:
        return None
    plan_type = snapshot.plan_type.value if snapshot.plan_type is not None else None
    return {
        "windows": windows,
        "limit_name": snapshot.limit_name,
        "plan_type": plan_type,
    }


def _format_window_duration(window_duration_mins: int | None) -> str | None:
    if window_duration_mins is None or window_duration_mins <= 0:
        return None
    if window_duration_mins % _MINUTES_PER_DAY == 0:
        days = window_duration_mins // _MINUTES_PER_DAY
        return f"{days}-day"
    if window_duration_mins % _MINUTES_PER_HOUR == 0:
        hours = window_duration_mins // _MINUTES_PER_HOUR
        return f"{hours}-hour"
    return f"{window_duration_mins}-min"
