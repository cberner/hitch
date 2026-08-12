"""Normalize review histories into the currently effective reviews."""

from __future__ import annotations

from typing import Any

from hitch.main.runtime.sdk_values import string_from_any

_DECISIVE_REVIEW_STATES = frozenset({"APPROVED", "CHANGES_REQUESTED"})


def latest_effective_reviews_by_author(reviews: list[Any]) -> list[Any]:
    """Keep each identified reviewer's latest effective review.

    Connector and ``gh`` payloads use different author and timestamp field
    shapes, so both PR-observation paths share this normalization. Reviews
    without an identity remain independent because they cannot be superseded
    safely.
    """
    latest_by_author: dict[str, tuple[dict[str, Any], int]] = {}
    anonymous: list[Any] = []
    for index, review in enumerate(reviews):
        if not isinstance(review, dict):
            anonymous.append(review)
            continue
        login = _review_author_login(review)
        if not login:
            anonymous.append(review)
            continue
        key = login.casefold()
        current = latest_by_author.get(key)
        if current is None or _review_supersedes(
            review, index=index, current=current[0], current_index=current[1]
        ):
            latest_by_author[key] = (review, index)
    return [*anonymous, *(review for review, _index in latest_by_author.values())]


def _review_author_login(review: dict[str, Any]) -> str:
    author = review.get("author")
    if isinstance(author, dict):
        login = string_from_any(author.get("login"))
        if login:
            return login
    user = review.get("user")
    if isinstance(user, dict):
        login = string_from_any(user.get("login"))
        if login:
            return login
    return string_from_any(review.get("author_login"))


def _review_supersedes(
    review: dict[str, Any],
    *,
    index: int,
    current: dict[str, Any],
    current_index: int,
) -> bool:
    state = string_from_any(review.get("state")).upper()
    current_state = string_from_any(current.get("state")).upper()
    # Dismissals clear change requests but do not erase approvals.
    changes_request_dismissal_pair = {state, current_state} == {
        "CHANGES_REQUESTED",
        "DISMISSED",
    }
    if not changes_request_dismissal_pair:
        if (
            state in _DECISIVE_REVIEW_STATES
            and current_state not in _DECISIVE_REVIEW_STATES
        ):
            return True
        if (
            state not in _DECISIVE_REVIEW_STATES
            and current_state in _DECISIVE_REVIEW_STATES
        ):
            return False
    submitted_at = string_from_any(
        review.get("submitted_at") or review.get("submittedAt")
    )
    current_submitted_at = string_from_any(
        current.get("submitted_at") or current.get("submittedAt")
    )
    if submitted_at and current_submitted_at:
        return (submitted_at, index) > (current_submitted_at, current_index)
    return index > current_index
