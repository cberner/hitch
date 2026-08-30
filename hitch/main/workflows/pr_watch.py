"""Agent-invoked GitHub PR watching.

The visible coding agent owns the follow-up loop. This module only performs a
bounded, read-only watch and returns the GitHub evidence needed for the agent to
decide what to fix and whether another watch is useful.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from django.utils import timezone

from hitch.main.runtime.sdk_values import positive_int, string_from_any
from hitch.main.workflows.gh_cli import (
    _GH_PR_VIEW_TIMEOUT_SECONDS,
    _gh_pr_review_threads,
    _gh_pr_status_checks,
    _gh_pr_view_payload,
    _GhPrOpenError,
    _run_git_cli,
)
from hitch.main.workflows.gh_observations import (
    _copy_gh_comment_fields,
    _copy_gh_reaction_fields,
    _copy_gh_review_fields,
    _copy_gh_review_thread_fields,
    _copy_gh_status_check_fields,
    _evaluate_pr_gates,
    _gh_watch_blockers,
    _gh_watch_feedback,
    _gh_watch_summary,
    _github_pr_url_from_text,
    _pr_gates_all_passed,
    _pr_gates_have_actionable_blockers,
    _pr_handoff_from_github_url,
)
from hitch.main.workflows.pr_handoff import (
    _compact_pr_handoff,
    _merge_pr_handoff_dicts,
    _pr_handoff_identity_changed,
    _pr_handoff_is_terminal,
)

PR_WATCH_RESULT_STATE_KEY = "last_pr_watch"
PR_WATCH_RESULT_TURN_INDEX_STATE_KEY = "last_pr_watch_turn_index"

_PR_WATCH_POLL_SECONDS = 2 * 60
_PR_WATCH_TIMEOUT_SECONDS = 30 * 60
_PR_WATCH_CANCEL_POLL_SECONDS = 1
_GH_PR_VIEW_FIELDS = (
    "url",
    "number",
    "state",
    "isDraft",
    "title",
    "baseRefName",
    "headRefName",
    "headRefOid",
    "mergeable",
    "mergeCommit",
    "createdAt",
    "updatedAt",
    "closedAt",
    "mergedAt",
)
_GH_PR_WATCH_FIELDS = (
    *_GH_PR_VIEW_FIELDS,
    "comments",
    "latestReviews",
    "reactionGroups",
    "reviewDecision",
    "reviews",
)
_PUBLISHED_PR_VALIDATION_FIELDS = (
    *_GH_PR_VIEW_FIELDS,
    "headRepository",
    "headRepositoryOwner",
)
_GITHUB_REMOTE_RE = re.compile(
    r"^(?:(?:https?|git|ssh)://(?:[^/@]+@)?github\.com/|"
    r"(?:[^/@]+@)?github\.com:)"
    r"(?P<repository>[^/\s]+/[^/\s]+?)(?:\.git)?/?$",
    re.IGNORECASE,
)
_SSH_REMOTE_RE = re.compile(
    r"^(?:ssh://(?:[^/@\s]+@)?(?P<url_host>[^/:\s]+)(?::\d+)?/|"
    r"(?:[^/@:\s]+@)?(?P<scp_host>[^/:\s]+):)"
    r"(?P<repository>[^/\s]+/[^/\s]+?)(?:\.git)?/?$",
    re.IGNORECASE,
)
_GITHUB_SSH_HOSTS = frozenset({"github.com", "ssh.github.com"})
_SSH_CONFIG_TIMEOUT_SECONDS = 5


class PrWatchError(RuntimeError):
    pass


class _PrWatchDeadlineError(RuntimeError):
    pass


@dataclass(frozen=True)
class _WatchTarget:
    cwd: str


def _not_cancelled() -> bool:
    return False


def watch_pr(
    *,
    cwd: str,
    url: str,
    previous_feedback_fingerprint: str = "",
    poll_seconds: float = _PR_WATCH_POLL_SECONDS,
    timeout_seconds: float = _PR_WATCH_TIMEOUT_SECONDS,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    cancel_requested: Callable[[], bool] = _not_cancelled,
) -> dict[str, Any]:
    """Watch until the PR needs attention, is ready/terminal, or times out."""
    normalized_url = validate_pr_watch_target(cwd=cwd, url=url)
    if poll_seconds < 0 or timeout_seconds < 0:
        raise ValueError("watch timing values must be non-negative")

    deadline = monotonic() + timeout_seconds
    last_observation: dict[str, Any] = {}
    while True:
        _raise_if_cancelled(cancel_requested)
        if monotonic() >= deadline:
            return _result_from_observation("timed_out", last_observation)
        try:
            observation = observe_pr(
                cwd=cwd,
                url=normalized_url,
                command_timeout_seconds=_command_timeout_provider(
                    deadline=deadline,
                    monotonic=monotonic,
                    cancel_requested=cancel_requested,
                ),
            )
        except _PrWatchDeadlineError:
            return _result_from_observation("timed_out", last_observation)
        except _GhPrOpenError as exc:
            if monotonic() >= deadline:
                return _result_from_observation("timed_out", last_observation)
            raise PrWatchError(str(exc)) from exc
        last_observation = observation
        result = _watch_result(
            observation,
            previous_feedback_fingerprint=previous_feedback_fingerprint,
        )
        if result is not None:
            return result

        remaining = deadline - monotonic()
        if remaining <= 0:
            return _result_from_observation("timed_out", observation)
        _interruptible_sleep(
            min(poll_seconds, remaining),
            monotonic=monotonic,
            sleep=sleep,
            cancel_requested=cancel_requested,
        )


def validate_pr_watch_target(*, cwd: str, url: str) -> str:
    """Validate the local checkout and return the canonical GitHub PR URL."""
    normalized_url = _normalized_pr_url(url)
    if not Path(cwd).is_dir():
        raise PrWatchError(f"repository cwd is missing: {cwd}")
    return normalized_url


def validate_published_pr_checkout(*, cwd: str, url: str) -> None:
    """Require a newly published PR to identify the active checkout HEAD."""
    normalized_url = validate_pr_watch_target(cwd=cwd, url=url)
    target = _WatchTarget(cwd=cwd)
    try:
        payload = _gh_pr_view_payload(
            target,
            selector=normalized_url,
            fields=_PUBLISHED_PR_VALIDATION_FIELDS,
            timeout_seconds=_GH_PR_VIEW_TIMEOUT_SECONDS,
        )
    except _GhPrOpenError as exc:
        raise PrWatchError(str(exc)) from exc
    if payload is None:
        raise PrWatchError("`gh pr view` did not return PR data")
    observed = pr_handoff_from_gh_view(
        payload,
        source_tool="hitch_watch_pr_validation",
    )
    requested = _compact_pr_handoff(
        _pr_handoff_from_github_url(
            normalized_url,
            source_tool="hitch_watch_pr_validation",
        )
    )
    if not pr_identity_matches(requested, observed):
        raise PrWatchError("GitHub returned a different pull request than requested")
    if string_from_any(observed.get("state")) != "open":
        raise PrWatchError("published pull request must be open")

    head_repository = _github_pr_head_repository(payload)
    head = string_from_any(observed.get("head"))
    head_sha = string_from_any(observed.get("head_sha"))
    if not head_repository or not head or not head_sha:
        raise PrWatchError(
            "GitHub did not return the published PR head repository and commit"
        )

    try:
        remote_result = _run_git_cli(target, ["remote", "-v"])
    except _GhPrOpenError as exc:
        raise PrWatchError(str(exc)) from exc
    if remote_result.returncode != 0:
        raise PrWatchError("could not inspect publishing checkout remotes")
    repositories = _github_remote_repositories(remote_result.stdout)
    if head_repository not in repositories:
        raise PrWatchError(
            f"published PR head repository {head_repository} does not match the "
            "publishing checkout"
        )

    branch = _publishing_checkout_value(target, ["branch", "--show-current"])
    if branch != head:
        raise PrWatchError(
            f"published PR head branch {head} does not match publishing checkout "
            f"branch {branch}"
        )
    checkout_sha = _publishing_checkout_value(target, ["rev-parse", "HEAD"])
    if checkout_sha.lower() != head_sha.lower():
        raise PrWatchError(
            "published PR head commit does not match publishing checkout HEAD"
        )


def _publishing_checkout_value(target: _WatchTarget, args: list[str]) -> str:
    try:
        result = _run_git_cli(target, args)
    except _GhPrOpenError as exc:
        raise PrWatchError(str(exc)) from exc
    value = result.stdout.strip()
    if result.returncode != 0 or not value:
        raise PrWatchError(
            f"could not inspect publishing checkout with `git {' '.join(args)}`"
        )
    return value


def _github_remote_repositories(output: str) -> set[str]:
    repositories: set[str] = set()
    remote_urls: set[str] = set()
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 2:
            remote_urls.add(fields[1])
    for remote_url in remote_urls:
        match = _GITHUB_REMOTE_RE.fullmatch(remote_url)
        if match is not None:
            repositories.add(match.group("repository").lower())
            continue
        match = _SSH_REMOTE_RE.fullmatch(remote_url)
        if match is None:
            continue
        host = match.group("url_host") or match.group("scp_host")
        if _resolved_ssh_hostname(host) in _GITHUB_SSH_HOSTS:
            repositories.add(match.group("repository").lower())
    return repositories


def _resolved_ssh_hostname(host: str) -> str:
    try:
        result = subprocess.run(
            ["ssh", "-G", "--", host],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=_SSH_CONFIG_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    for line in result.stdout.splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) == 2 and fields[0].lower() == "hostname":
            return fields[1].strip().rstrip(".").lower()
    return ""


def _github_pr_head_repository(payload: dict[str, Any]) -> str:
    repository = payload.get("headRepository")
    owner = payload.get("headRepositoryOwner")
    if not isinstance(repository, dict) or not isinstance(owner, dict):
        return ""
    name_with_owner = string_from_any(repository.get("nameWithOwner"))
    if "/" in name_with_owner:
        return name_with_owner.lower()
    name = string_from_any(repository.get("name"))
    login = string_from_any(owner.get("login"))
    return f"{login}/{name}".lower() if login and name else ""


def observe_pr(
    *,
    cwd: str,
    url: str,
    command_timeout_seconds: Callable[[], float] | None = None,
) -> dict[str, Any]:
    persisted = _compact_pr_handoff(
        _pr_handoff_from_github_url(url, source_tool="hitch_watch_pr")
    )
    return observation_from_gh(
        _WatchTarget(cwd=cwd),
        persisted=persisted,
        command_timeout_seconds=command_timeout_seconds,
    )


def observation_from_gh(
    target: Any,
    *,
    persisted: dict[str, Any],
    command_timeout_seconds: Callable[[], float] | None = None,
) -> dict[str, Any]:
    selector = string_from_any(persisted.get("url"))
    payload = _gh_pr_view_payload(
        target,
        selector=selector or None,
        fields=_GH_PR_WATCH_FIELDS,
        timeout_seconds=(
            command_timeout_seconds()
            if command_timeout_seconds is not None
            else _GH_PR_VIEW_TIMEOUT_SECONDS
        ),
    )
    if payload is None:
        raise _GhPrOpenError("`gh pr view` did not return PR data")
    pr = pr_handoff_from_gh_view(payload, source_tool="hitch_watch_pr")
    if persisted and not _pr_handoff_identity_changed(persisted, pr):
        pr = _merge_pr_handoff_dicts(persisted, pr)

    _copy_gh_review_fields(pr, payload)
    _copy_gh_reaction_fields(pr, payload)
    _copy_gh_comment_fields(pr, payload)
    review_threads, review_threads_complete = _gh_pr_review_threads(
        target,
        pr,
        command_timeout_seconds=command_timeout_seconds,
    )
    _copy_gh_review_thread_fields(
        pr, review_threads, complete=review_threads_complete
    )
    status_checks, status_checks_complete = _gh_pr_status_checks(
        target,
        pr,
        command_timeout_seconds=command_timeout_seconds,
    )
    _copy_gh_status_check_fields(
        pr, status_checks, complete=status_checks_complete
    )

    compact_pr = _compact_pr_handoff(pr)
    gates = _evaluate_pr_gates(compact_pr)
    return {
        "summary": _gh_watch_summary(gates, compact_pr),
        "feedback": _gh_watch_feedback(payload, review_threads, compact_pr),
        "pr": compact_pr,
        "gates": gates,
        "blockers": _gh_watch_blockers(gates),
    }


def pr_handoff_from_gh_view(payload: Any, *, source_tool: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise _GhPrOpenError("`gh pr view` returned a non-object payload")

    url = string_from_any(payload.get("url"))
    handoff = (
        _pr_handoff_from_github_url(url, source_tool=source_tool) if url else {}
    )
    number = positive_int(payload.get("number"))
    if number is not None:
        handoff["pr_number"] = number
    state = string_from_any(payload.get("state")).lower()
    if state:
        handoff["state"] = state
    merged_at = string_from_any(payload.get("mergedAt"))
    handoff["merged"] = bool(merged_at) or state == "merged"
    draft = payload.get("isDraft")
    if isinstance(draft, bool):
        handoff["draft"] = draft
    mergeable = _gh_mergeable_value(payload.get("mergeable"))
    if mergeable is not None:
        handoff["mergeable"] = mergeable

    _copy_gh_string(payload, handoff, "title", "title")
    _copy_gh_string(payload, handoff, "baseRefName", "base")
    _copy_gh_string(payload, handoff, "headRefName", "head")
    head_sha = string_from_any(payload.get("headRefOid"))
    if head_sha:
        handoff["head_sha"] = head_sha
        handoff["latest_commit_sha"] = head_sha
    _copy_gh_string(payload, handoff, "createdAt", "created_at")
    _copy_gh_string(payload, handoff, "updatedAt", "updated_at")
    _copy_gh_string(payload, handoff, "closedAt", "closed_at")
    if merged_at:
        handoff["merged_at"] = merged_at
    merge_commit = payload.get("mergeCommit")
    if isinstance(merge_commit, dict):
        merge_commit_sha = string_from_any(merge_commit.get("oid"))
        if merge_commit_sha:
            handoff["merge_commit_sha"] = merge_commit_sha
    handoff["source_tool"] = source_tool
    handoff["last_observed_at"] = int(timezone.now().timestamp())
    return _compact_pr_handoff(handoff)


def feedback_fingerprint(observation: dict[str, Any]) -> str:
    feedback = string_from_any(observation.get("feedback"))
    if not feedback:
        return ""
    return hashlib.sha256(feedback.encode()).hexdigest()


def pr_identity_matches(expected: dict[str, Any], observed: dict[str, Any]) -> bool:
    expected_repo = string_from_any(expected.get("repository_full_name")).lower()
    observed_repo = string_from_any(observed.get("repository_full_name")).lower()
    expected_number = positive_int(expected.get("pr_number"))
    observed_number = positive_int(observed.get("pr_number"))
    if expected_repo and observed_repo and expected_number and observed_number:
        return (expected_repo, expected_number) == (observed_repo, observed_number)
    expected_url = string_from_any(expected.get("url"))
    observed_url = string_from_any(observed.get("url"))
    return bool(expected_url and observed_url and expected_url == observed_url)


def _command_timeout_provider(
    *,
    deadline: float,
    monotonic: Callable[[], float],
    cancel_requested: Callable[[], bool],
) -> Callable[[], float]:
    def _remaining_timeout() -> float:
        _raise_if_cancelled(cancel_requested)
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise _PrWatchDeadlineError
        return min(float(_GH_PR_VIEW_TIMEOUT_SECONDS), remaining)

    return _remaining_timeout


def _interruptible_sleep(
    duration: float,
    *,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
    cancel_requested: Callable[[], bool],
) -> None:
    wake_at = monotonic() + max(duration, 0)
    while True:
        _raise_if_cancelled(cancel_requested)
        remaining = wake_at - monotonic()
        if remaining <= 0:
            return
        sleep(min(float(_PR_WATCH_CANCEL_POLL_SECONDS), remaining))


def _raise_if_cancelled(cancel_requested: Callable[[], bool]) -> None:
    if cancel_requested():
        raise PrWatchError("PR watch cancelled")


def _watch_result(
    observation: dict[str, Any], *, previous_feedback_fingerprint: str
) -> dict[str, Any] | None:
    pr = _compact_pr_handoff(observation.get("pr"))
    if _pr_handoff_is_terminal(pr):
        return _result_from_observation("terminal", observation)
    gates = observation.get("gates")
    safe_gates = gates if isinstance(gates, list) else []
    if _pr_gates_have_actionable_blockers(safe_gates):
        return _result_from_observation("action_required", observation)
    current_fingerprint = feedback_fingerprint(observation)
    if current_fingerprint and current_fingerprint != previous_feedback_fingerprint:
        return _result_from_observation("attention", observation)
    if _pr_gates_all_passed(safe_gates):
        return _result_from_observation("ready", observation)
    return None


def _result_from_observation(
    status: str, observation: dict[str, Any]
) -> dict[str, Any]:
    return {
        "status": status,
        "summary": string_from_any(observation.get("summary"))
        or "Hitch checked the PR.",
        "feedback": string_from_any(observation.get("feedback")),
        "feedback_fingerprint": feedback_fingerprint(observation),
        "pr": _compact_pr_handoff(observation.get("pr")),
        "gates": observation.get("gates")
        if isinstance(observation.get("gates"), list)
        else [],
        "blockers": observation.get("blockers")
        if isinstance(observation.get("blockers"), list)
        else [],
    }


def _normalized_pr_url(url: str) -> str:
    normalized = url.strip().rstrip("/")
    parsed = _github_pr_url_from_text(normalized)
    if not parsed or parsed != normalized:
        raise PrWatchError("url must be a GitHub pull request URL")
    return parsed


def _copy_gh_string(
    source: dict[str, Any], target: dict[str, Any], source_key: str, target_key: str
) -> None:
    value = string_from_any(source.get(source_key))
    if value:
        target[target_key] = value


def _gh_mergeable_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in {"mergeable", "clean", "has_hooks", "unstable"}:
        return True
    if normalized in {"conflicting", "dirty", "blocked"}:
        return False
    return None
