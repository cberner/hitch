"""Pure extraction of a GitHub PR's observable state into handoff fields.

``gh`` returns a PR's review, reaction, comment, and CI-status-check state as
nested JSON. This module owns the dependency-free transforms that compact that
JSON down to the safe PR-handoff observation fields, plus the review-signal and
CI-status normalizers those rely on.
"""

from __future__ import annotations

from typing import Any

from hitch.main.pr_handoff import _compact_pr_list
from hitch.main.sdk_values import string_from_any

_CI_PASSING_STATUSES = frozenset(
    {"neutral", "pass", "passed", "skipped", "success", "successful"}
)
_CI_PENDING_STATUSES = frozenset(
    {
        "completed",
        "expected",
        "in_progress",
        "pending",
        "queued",
        "requested",
        "running",
        "waiting",
    }
)
_CI_BLOCKING_STATUSES = frozenset(
    {
        "action_required",
        "cancelled",
        "error",
        "failed",
        "failure",
        "startup_failure",
        "stale",
        "timed_out",
    }
)


def _copy_gh_review_fields(target: dict[str, Any], payload: dict[str, Any]) -> None:
    reviews = payload.get("latestReviews")
    if not isinstance(reviews, list):
        reviews = payload.get("reviews")
    if not isinstance(reviews, list):
        reviews = []
    states = [
        state.upper()
        for review in reviews
        if isinstance(review, dict)
        and isinstance((state := review.get("state")), str)
        and state
    ]
    review_decision = string_from_any(payload.get("reviewDecision")).upper()
    target["review_count"] = len(reviews)
    if review_decision == "CHANGES_REQUESTED":
        target["review_signal"] = "changes_requested"
    elif review_decision == "APPROVED":
        target["review_signal"] = "approved"
    elif review_decision:
        target["review_signal"] = "commented" if states else ""
    elif "CHANGES_REQUESTED" in states:
        target["review_signal"] = "changes_requested"
    elif "APPROVED" in states:
        target["review_signal"] = "approved"
    elif states:
        target["review_signal"] = "commented"
    else:
        target["review_signal"] = ""


def _copy_gh_reaction_fields(target: dict[str, Any], payload: dict[str, Any]) -> None:
    groups = payload.get("reactionGroups")
    if not isinstance(groups, list):
        return
    total = 0
    thumbs_up = 0
    for group in groups:
        if not isinstance(group, dict):
            continue
        count = _reaction_group_count(group)
        total += count
        content = string_from_any(group.get("content")).lower()
        if content in {"thumbs_up", "+1", "thumbsup"}:
            thumbs_up += count
    target["reaction_count"] = total
    current_signal = _normalize_review_signal(target.get("review_signal"))
    review_decision = string_from_any(payload.get("reviewDecision")).upper()
    review_required = bool(
        review_decision and review_decision not in {"APPROVED", "CHANGES_REQUESTED"}
    )
    if (
        thumbs_up > 0
        and current_signal not in {"changes_requested", "approved"}
        and not review_required
    ):
        target["review_signal"] = "thumbs_up"
    elif thumbs_up == 0 and current_signal == "thumbs_up":
        target["review_signal"] = ""


def _reaction_group_count(group: dict[str, Any]) -> int:
    users = group.get("users")
    if isinstance(users, dict):
        count = users.get("totalCount")
        if isinstance(count, int) and not isinstance(count, bool) and count > 0:
            return count
    count = group.get("totalCount") or group.get("count")
    if isinstance(count, int) and not isinstance(count, bool) and count > 0:
        return count
    return 0


def _copy_gh_comment_fields(target: dict[str, Any], payload: dict[str, Any]) -> None:
    comments = payload.get("comments")
    if not isinstance(comments, list):
        return
    target["comment_count"] = len(comments)
    target["latest_comments"] = _compact_pr_list(
        [_safe_gh_comment_identifier(comment) for comment in comments[-5:]]
    )


def _safe_gh_comment_identifier(comment: Any) -> dict[str, Any]:
    if not isinstance(comment, dict):
        return {}
    item: dict[str, Any] = {}
    comment_id = comment.get("databaseId") or comment.get("id")
    if isinstance(comment_id, int) and not isinstance(comment_id, bool):
        item["database_id"] = comment_id
    elif isinstance(comment_id, str):
        item["id"] = comment_id
    url = string_from_any(comment.get("url"))
    if url:
        item["url"] = url
    return item


def _copy_gh_status_check_fields(
    target: dict[str, Any], raw_checks: Any, *, complete: bool = True
) -> None:
    status, failing, pending = _ci_status_from_gh_status_checks(raw_checks)
    if not complete and status != "failure":
        status = "pending"
    if not status:
        return
    target["ci_status"] = status
    target["failing_jobs"] = failing
    target["pending_jobs"] = pending


def _ci_status_from_gh_status_checks(
    raw_checks: Any,
) -> tuple[str, list[dict[str, str]], list[dict[str, str]]]:
    if raw_checks is None:
        return "pending", [], []
    if not isinstance(raw_checks, list):
        return "", [], []
    if not raw_checks:
        return "pending", [], []
    failing: list[dict[str, str]] = []
    pending: list[dict[str, str]] = []
    saw_success = False
    for raw_check in raw_checks:
        if not isinstance(raw_check, dict):
            continue
        status = _gh_check_status(raw_check)
        check = _compact_gh_check(raw_check)
        if status == "failure":
            failing.append(check)
            continue
        if status == "pending":
            pending.append(check)
            continue
        if status == "success":
            saw_success = True
    if failing:
        return "failure", failing[:5], pending[:5]
    if pending:
        return "pending", [], pending[:5]
    if saw_success:
        return "success", [], []
    return "pending", [], []


def _gh_check_status(check: dict[str, Any]) -> str:
    state = string_from_any(check.get("state")).lower()
    if state:
        normalized = _normalize_ci_status(state)
        if normalized:
            return normalized
    status = string_from_any(check.get("status")).lower()
    conclusion = string_from_any(check.get("conclusion")).lower()
    if conclusion:
        normalized = _normalize_ci_status(conclusion)
        if normalized:
            return normalized
    if status and status != "completed":
        return "pending"
    if status == "completed":
        return "pending"
    return ""


def _compact_gh_check(check: dict[str, Any]) -> dict[str, str]:
    item: dict[str, str] = {}
    for source_key, target_key in (
        ("name", "name"),
        ("context", "name"),
        ("workflowName", "name"),
        ("status", "status"),
        ("state", "status"),
        ("conclusion", "conclusion"),
        ("detailsUrl", "url"),
        ("link", "url"),
        ("targetUrl", "url"),
    ):
        value = string_from_any(check.get(source_key))
        if value and target_key not in item:
            item[target_key] = value
    if "name" not in item:
        item["name"] = "unnamed check"
    return item


def _normalize_review_signal(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    signal = value.strip().lower().replace("-", "_").replace(" ", "_")
    if signal in {"approved", "approval", "approve", "lgtm"}:
        return "approved"
    if signal in {"+1", "thumbs_up", "thumbsup", "thumbs"}:
        return "thumbs_up"
    if signal in {"changes_requested", "change_requested", "request_changes"}:
        return "changes_requested"
    if signal in {"comment", "commented", "comments", "reviewed"}:
        return "commented"
    if signal in {"none", "no_review", "no_reviews"}:
        return ""
    return signal


def _normalize_ci_status(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    status = value.strip().lower().replace("-", "_").replace(" ", "_")
    if status in _CI_PASSING_STATUSES:
        return "success"
    if status in _CI_BLOCKING_STATUSES:
        return "failure"
    if status in _CI_PENDING_STATUSES:
        return "pending"
    return ""
