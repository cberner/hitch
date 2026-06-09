"""Pure data transforms for the PR-handoff snapshot dict.

The PR-QA workflow persists a compact snapshot of a pull request's state. This
module owns the dependency-free transforms over that snapshot: compacting raw
GitHub observations down to the safe field set, merging an update onto the
persisted snapshot (resetting gate observations when the head moves), and the
identity/head-change and terminal-state predicates those rely on.
"""

from __future__ import annotations

from typing import Any

_PR_GATE_OBSERVATION_FIELDS = frozenset(
    {
        "mergeable",
        "review_thread_count",
        "unresolved_thread_count",
        "unresolved_threads",
        "review_count",
        "review_signal",
        "reaction_count",
        "ci_status",
        "failing_jobs",
        "pending_jobs",
        "draft",
    }
)
_PR_HANDOFF_FIELDS = (
    "url",
    "repository_full_name",
    "pr_number",
    "state",
    "merged",
    "mergeable",
    "draft",
    "title",
    "base",
    "base_sha",
    "head",
    "head_sha",
    "merge_commit_sha",
    "created_at",
    "updated_at",
    "closed_at",
    "merged_at",
    "last_observed_at",
    "latest_commit_sha",
    "source_tool",
    "review_thread_count",
    "unresolved_thread_count",
    "unresolved_threads",
    "comment_count",
    "latest_comments",
    "review_count",
    "review_signal",
    "reaction_count",
    "ci_status",
    "failing_jobs",
    "pending_jobs",
)
_PR_HANDOFF_BOOLEAN_FIELDS = frozenset({"merged", "mergeable", "draft"})
_PR_HANDOFF_INTEGER_FIELDS = frozenset(
    {
        "pr_number",
        "last_observed_at",
        "review_thread_count",
        "unresolved_thread_count",
        "comment_count",
        "review_count",
        "reaction_count",
    }
)
_PR_HANDOFF_LIST_FIELDS = frozenset(
    {"unresolved_threads", "latest_comments", "failing_jobs", "pending_jobs"}
)
_PR_SAFE_LIST_ITEM_FIELDS = (
    "path",
    "line",
    "start_line",
    "url",
    "html_url",
    "id",
    "database_id",
    "name",
    "status",
    "conclusion",
)


def _merge_pr_handoff_dicts(
    current: dict[str, Any], update: dict[str, Any]
) -> dict[str, Any]:
    if _pr_handoff_identity_changed(current, update):
        current = {}
    merged = dict(current)
    if _pr_handoff_head_changed(current, update):
        canonical_head_sha = _canonical_update_head_sha(update)
        for key in _PR_GATE_OBSERVATION_FIELDS:
            merged.pop(key, None)
        merged.pop("head_sha", None)
        merged.pop("latest_commit_sha", None)
        if canonical_head_sha:
            update = {
                **update,
                "head_sha": canonical_head_sha,
                "latest_commit_sha": canonical_head_sha,
            }
    for key, value in update.items():
        # ``None`` and ``""`` are "absent" for every key except
        # ``review_signal``, which uses ``""`` as the explicit reviews-clear
        # sentinel (see ``codex_events._copy_review_fields``). Empty
        # list/dict updates are "observed and found none" overwrites.
        # Reaction-derived ``thumbs_up`` is held back from review-only clears,
        # but a reaction observation may explicitly clear it.
        if value is None:
            continue
        if value == "":
            if key == "review_signal" and (
                merged.get(key) != "thumbs_up" or "reaction_count" in update
            ):
                merged.pop(key, None)
            continue
        merged[key] = value
    return merged


def _pr_handoff_identity_changed(
    current: dict[str, Any], update: dict[str, Any]
) -> bool:
    if not current:
        return False
    current_number = current.get("pr_number")
    update_number = update.get("pr_number")
    if (
        isinstance(current_number, int)
        and not isinstance(current_number, bool)
        and isinstance(update_number, int)
        and not isinstance(update_number, bool)
    ):
        return current_number != update_number
    current_url = current.get("url")
    update_url = update.get("url")
    return (
        isinstance(current_url, str)
        and isinstance(update_url, str)
        and bool(current_url)
        and bool(update_url)
        and current_url != update_url
    )


def _pr_handoff_head_changed(current: dict[str, Any], update: dict[str, Any]) -> bool:
    if not current:
        return False
    current_values = _pr_head_sha_values(current)
    update_values = _pr_head_sha_values(update)
    return bool(current_values and update_values and current_values != update_values)


def _pr_head_sha_values(handoff: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in ("head_sha", "latest_commit_sha"):
        value = handoff.get(key)
        if isinstance(value, str) and value:
            values.add(value)
    return values


def _canonical_update_head_sha(update: dict[str, Any]) -> str:
    latest = update.get("latest_commit_sha")
    if isinstance(latest, str) and latest:
        return latest
    head = update.get("head_sha")
    return head if isinstance(head, str) else ""


def _compact_pr_handoff(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    compact: dict[str, Any] = {}
    for key in _PR_HANDOFF_FIELDS:
        raw = value.get(key)
        if (
            (key in _PR_HANDOFF_BOOLEAN_FIELDS and isinstance(raw, bool))
            or (
                key in _PR_HANDOFF_INTEGER_FIELDS
                and isinstance(raw, int)
                and not isinstance(raw, bool)
            )
        ):
            compact[key] = raw
        elif isinstance(raw, str):
            stripped = raw.strip()
            if stripped:
                compact[key] = stripped
            elif key == "review_signal":
                # ``""`` is the explicit reviews-clear sentinel; preserve
                # it so ``_merge_pr_handoff_dicts`` can drop a stale verdict.
                compact[key] = ""
        elif key in _PR_HANDOFF_LIST_FIELDS and isinstance(raw, list):
            compact[key] = _compact_pr_list(raw)
    return compact


def _compact_pr_list(items: list[Any]) -> list[Any]:
    compacted: list[Any] = []
    for item in items[:5]:
        if isinstance(item, str):
            text = item.strip()
            if text:
                compacted.append(text[:500])
        elif isinstance(item, dict):
            compact_item: dict[str, Any] = {}
            for key in _PR_SAFE_LIST_ITEM_FIELDS:
                value = item.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    compact_item[key] = value
                elif isinstance(value, str) and value.strip():
                    compact_item[key] = value.strip()[:500]
            if compact_item:
                compacted.append(compact_item)
    return compacted


def _pr_handoff_is_terminal(handoff: dict[str, Any]) -> bool:
    state = handoff.get("state")
    return handoff.get("merged") is True or (
        isinstance(state, str) and state.lower() in {"closed", "merged"}
    )
