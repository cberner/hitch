"""Token-usage capture for the Claude Code backend.

Codex persists per-turn token counts in its on-disk rollout file, which the
session views parse and cache in :class:`ArchivedSessionTokenUsage`. Claude has
no such rollout, so the worker captures ``ResultMessage.usage`` directly and
writes the *same* cache row (with ``rollout_path=""``). The per-session display
and the lifetime aggregation already treat a ``rollout_path==""`` row as the
authoritative numbers for a Claude thread, so no separate read path is needed.

These are pure-ish helpers (one DB upsert) kept out of ``views`` so the detached
worker can import them without pulling in the request stack.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from django.db import transaction

from hitch.main import claude_options
from hitch.main.models import TOKEN_USAGE_LOGIC_VERSION, ArchivedSessionTokenUsage


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value if value > 0 else 0
    if isinstance(value, float) and value > 0:
        return int(value)
    return 0


def normalize_turn_usage(
    raw_usage: Any, model: str | None
) -> dict[str, int] | None:
    """Translate one Anthropic ``ResultMessage.usage`` into the Codex count shape.

    Anthropic reports ``input_tokens`` *excluding* cache, plus separate
    ``cache_read_input_tokens`` / ``cache_creation_input_tokens``. Codex folds
    cached input *into* ``input_tokens`` and exposes the cached portion
    separately, so the display's ``input - cached`` subtraction recovers the
    non-cached input. Mirror that folding here.

    Returns ``None`` when the turn reported no usable counts (e.g. a failed turn
    that never reached the API), so the caller can skip writing a zeroed row.
    """
    if not isinstance(raw_usage, dict):
        return None
    non_cached_input = _coerce_int(raw_usage.get("input_tokens"))
    cache_read = _coerce_int(raw_usage.get("cache_read_input_tokens"))
    cache_creation = _coerce_int(raw_usage.get("cache_creation_input_tokens"))
    output_tokens = _coerce_int(raw_usage.get("output_tokens"))
    cached = cache_read + cache_creation
    input_tokens = non_cached_input + cached
    if input_tokens <= 0 and output_tokens <= 0:
        return None
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached,
        "output_tokens": output_tokens,
        # The latest request's full input plus its output is the context-window
        # occupancy after the turn -- the figure the "% of context" gauge wants.
        "context_tokens": input_tokens + output_tokens,
        "model_context_window": claude_options.context_window_for(model),
    }


def record_turn_usage(thread_id: str, raw_usage: Any, model: str | None) -> None:
    """Accumulate one Claude turn's usage onto the thread's cache row.

    Per-turn input/cached/output counts are *added* to the running session
    totals, while the context-window occupancy reflects only the latest turn.
    Idempotency is not required: each turn is recorded exactly once on
    completion. A pre-existing row from superseded logic (or a Codex rollout) is
    overwritten rather than added to, so stale counts never compound.
    """
    turn = normalize_turn_usage(raw_usage, model)
    if turn is None or not thread_id:
        return
    today = datetime.now(UTC).date().isoformat()
    non_cached = max(turn["input_tokens"] - turn["cached_input_tokens"], 0)
    with transaction.atomic():
        cache = (
            ArchivedSessionTokenUsage.objects.select_for_update()
            .filter(thread_id=thread_id)
            .first()
        )
        accumulate = (
            cache is not None
            and cache.rollout_path == ""
            and cache.usage_logic_version >= TOKEN_USAGE_LOGIC_VERSION
        )
        if cache is None:
            cache = ArchivedSessionTokenUsage(thread_id=thread_id)
        if accumulate:
            cache.input_tokens += turn["input_tokens"]
            cache.cached_input_tokens += turn["cached_input_tokens"]
            cache.output_tokens += turn["output_tokens"]
            cache.total_tokens += turn["input_tokens"] + turn["output_tokens"]
            daily = _normalized_daily(cache.daily_usage)
        else:
            cache.input_tokens = turn["input_tokens"]
            cache.cached_input_tokens = turn["cached_input_tokens"]
            cache.output_tokens = turn["output_tokens"]
            cache.total_tokens = turn["input_tokens"] + turn["output_tokens"]
            daily = {}
        cache.context_tokens = turn["context_tokens"]
        cache.model_context_window = turn["model_context_window"]
        cache.rollout_path = ""
        cache.rollout_mtime_ns = 0
        cache.usage_logic_version = TOKEN_USAGE_LOGIC_VERSION
        bucket = daily.setdefault(today, {"input": 0, "output": 0, "cached": 0})
        bucket["input"] += non_cached
        bucket["output"] += turn["output_tokens"]
        bucket["cached"] += turn["cached_input_tokens"]
        cache.daily_usage = daily
        cache.save()


def _normalized_daily(value: Any) -> dict[str, dict[str, int]]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, dict[str, int]] = {}
    for date_key, values in value.items():
        if not isinstance(date_key, str) or not isinstance(values, dict):
            continue
        out[date_key] = {
            "input": _coerce_int(values.get("input")),
            "output": _coerce_int(values.get("output")),
            "cached": _coerce_int(values.get("cached")),
        }
    return out
