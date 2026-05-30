"""Drive GitHub PR operations through the ``gh`` CLI and ``git`` directly.

Hitch used to rely on the coding agent's GitHub MCP connector to open PRs and
to observe PR state (comments, CI, reviews, reactions). That coupled the PR
lifecycle to Codex specifically. This module lets Hitch own those operations
itself so the same lifecycle works for coding agents that do not expose a
GitHub MCP connector.

The snapshot dicts returned here intentionally match the PR-handoff field shape
consumed by ``system_agents`` gate evaluation (``mergeable``, ``review_signal``,
``ci_status``, ``unresolved_threads``, ``failing_jobs``, ``pending_jobs``, ...)
so the existing gate logic is reused unchanged.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from typing import Any

logger = logging.getLogger(__name__)

_GH_TIMEOUT_SECONDS = 60
_GIT_TIMEOUT_SECONDS = 120

# Fields requested from ``gh pr view --json``. Each maps onto one or more
# PR-handoff snapshot fields below.
_PR_VIEW_FIELDS = ",".join(
    (
        "number",
        "url",
        "state",
        "isDraft",
        "mergedAt",
        "mergeCommit",
        "mergeable",
        "title",
        "baseRefName",
        "headRefName",
        "headRefOid",
        "createdAt",
        "updatedAt",
        "closedAt",
        "reviewDecision",
        "reviews",
        "latestReviews",
        "comments",
        "reactionGroups",
        "statusCheckRollup",
    )
)

# ``gh pr view --json`` does not expose ``reviewThreads`` (it is a GraphQL-only
# field), so unresolved review threads are fetched separately via ``gh api
# graphql``.
_REVIEW_THREADS_QUERY = (
    "query($owner:String!,$name:String!,$number:Int!){"
    "repository(owner:$owner,name:$name){pullRequest(number:$number){"
    "reviewThreads(first:100){nodes{id isResolved isOutdated path line "
    "comments(first:5){nodes{body url}}}}}}}"
)

_GITHUB_PR_URL_RE = re.compile(
    r"https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/pull/([0-9]+)"
)
_SUCCESS_CONCLUSIONS = frozenset({"SUCCESS", "NEUTRAL", "SKIPPED"})
_PR_TEXT_MAX_CHARS = 500
_PR_DETAIL_LIMIT = 5
_NO_PR_MARKERS = (
    "no pull requests found",
    "no open pull requests found",
    "no pull request found",
    "could not resolve to a pullrequest",
)


class GithubCliError(RuntimeError):
    """Raised when a ``gh`` or ``git`` invocation fails."""


def _run(
    argv: list[str], *, cwd: str, timeout: int, check: bool = True
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            argv,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GithubCliError(f"{argv[0]} command failed: {exc}") from exc
    if check and result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        raise GithubCliError(
            message or f"{' '.join(argv[:2])} exited with status {result.returncode}"
        )
    return result


def current_branch(cwd: str) -> str:
    """Return the checked-out branch for ``cwd``, raising on a detached HEAD."""
    result = _run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=cwd,
        timeout=_GIT_TIMEOUT_SECONDS,
    )
    branch = result.stdout.strip()
    if not branch or branch == "HEAD":
        raise GithubCliError("worktree is not on a named branch")
    return branch


def push_branch(cwd: str, branch: str) -> None:
    """Push ``branch`` to ``origin``, setting upstream tracking.

    The coding agent is asked to rebase before Hitch pushes, so an existing
    PR branch has rewritten history and a plain push would be rejected as a
    non-fast-forward. ``--force-with-lease`` makes the rebased update succeed
    while still refusing to clobber unexpected remote work.
    """
    _run(
        ["git", "push", "--force-with-lease", "-u", "origin", branch],
        cwd=cwd,
        timeout=_GIT_TIMEOUT_SECONDS,
    )


def open_or_update_pr(
    cwd: str,
    *,
    branch: str,
    base: str | None = None,
    title: str | None = None,
    body: str | None = None,
    draft: bool = False,
) -> dict[str, Any]:
    """Push ``branch`` and ensure a PR exists for it, returning its snapshot.

    Hitch owns the push and PR creation. Merge conflicts and other branch
    problems surface as ``GithubCliError`` from the underlying ``git``/``gh``
    call rather than being resolved here.
    """
    push_branch(cwd, branch)
    if _existing_pr_number(cwd, branch) is None:
        argv = ["gh", "pr", "create", "--head", branch]
        if base:
            argv += ["--base", base]
        if title:
            argv += ["--title", title, "--body", body or ""]
        else:
            argv += ["--fill"]
        if draft:
            argv += ["--draft"]
        _run(argv, cwd=cwd, timeout=_GH_TIMEOUT_SECONDS)
    snapshot = fetch_pr_snapshot(cwd, branch=branch)
    if snapshot is None:
        raise GithubCliError("opened a PR but could not read its state back")
    return snapshot


def _existing_pr_number(cwd: str, branch: str) -> int | None:
    result = _run(
        [
            "gh",
            "pr",
            "list",
            "--head",
            branch,
            "--state",
            "open",
            "--json",
            "number",
            "--limit",
            "1",
        ],
        cwd=cwd,
        timeout=_GH_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode != 0:
        return None
    parsed = _json_or_none(result.stdout)
    if isinstance(parsed, list) and parsed:
        number = parsed[0].get("number") if isinstance(parsed[0], dict) else None
        if isinstance(number, int) and not isinstance(number, bool):
            return number
    return None


def fetch_pr_snapshot(
    cwd: str, *, pr_number: int | None = None, branch: str | None = None
) -> dict[str, Any] | None:
    """Return a handoff-shaped PR snapshot, or ``None`` if no PR exists.

    Looks up the PR by ``pr_number`` when available, otherwise by ``branch``.
    """
    ref = str(pr_number) if pr_number else (branch or "")
    argv = ["gh", "pr", "view"]
    if ref:
        argv.append(ref)
    argv += ["--json", _PR_VIEW_FIELDS]
    result = _run(argv, cwd=cwd, timeout=_GH_TIMEOUT_SECONDS, check=False)
    if result.returncode != 0:
        stderr = (result.stderr or "").lower()
        if any(marker in stderr for marker in _NO_PR_MARKERS):
            return None
        raise GithubCliError(
            (result.stderr or result.stdout or "gh pr view failed").strip()
        )
    data = _json_or_none(result.stdout)
    if not isinstance(data, dict):
        raise GithubCliError("gh pr view returned unexpected output")
    snapshot = _snapshot_from_pr_view(data)
    threads = _fetch_review_threads(
        cwd,
        repo_full_name=snapshot.get("repository_full_name"),
        pr_number=snapshot.get("pr_number"),
    )
    # ``None`` means the thread fetch failed/was unobservable; leave the thread
    # fields unset so the review gate stays pending rather than treating the PR
    # as having zero unresolved threads.
    if threads is not None:
        _apply_review_threads(snapshot, threads)
    return snapshot


def _fetch_review_threads(
    cwd: str, *, repo_full_name: Any, pr_number: Any
) -> list[Any] | None:
    """Return the PR's review threads via ``gh api graphql``.

    ``gh pr view --json`` cannot return review threads, so they are fetched
    separately. Returns ``None`` when the threads could not be observed (bad
    identity, CLI error, or unexpected payload) so the caller can keep the
    unresolved-thread state unknown instead of falsely clearing it; returns a
    (possibly empty) list on a successful query.
    """
    if not isinstance(repo_full_name, str) or "/" not in repo_full_name:
        return None
    if not isinstance(pr_number, int) or isinstance(pr_number, bool):
        return None
    owner, _, name = repo_full_name.partition("/")
    result = _run(
        [
            "gh",
            "api",
            "graphql",
            "-f",
            f"query={_REVIEW_THREADS_QUERY}",
            "-f",
            f"owner={owner}",
            "-f",
            f"name={name}",
            "-F",
            f"number={pr_number}",
        ],
        cwd=cwd,
        timeout=_GH_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode != 0:
        logger.warning("failed to fetch review threads: %s", result.stderr.strip())
        return None
    parsed = _json_or_none(result.stdout)
    try:
        nodes = parsed["data"]["repository"]["pullRequest"]["reviewThreads"]["nodes"]
    except (TypeError, KeyError):
        return None
    return nodes if isinstance(nodes, list) else None


def _snapshot_from_pr_view(data: dict[str, Any]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    url = _str(data.get("url"))
    if url:
        snapshot["url"] = url
        match = _GITHUB_PR_URL_RE.search(url)
        if match is not None:
            owner, repo, number = match.groups()
            snapshot["repository_full_name"] = f"{owner}/{repo}"
            snapshot["pr_number"] = int(number)
    number = data.get("number")
    if isinstance(number, int) and not isinstance(number, bool):
        snapshot["pr_number"] = number

    state = _str(data.get("state")).lower()
    if state:
        snapshot["state"] = state
    # ``gh pr view --json`` has no ``merged`` field; derive it from the
    # supported ``state``/``mergedAt`` fields instead.
    snapshot["merged"] = state == "merged" or bool(_str(data.get("mergedAt")))
    snapshot["draft"] = bool(data.get("isDraft"))

    mergeable = _str(data.get("mergeable")).upper()
    if mergeable == "MERGEABLE":
        snapshot["mergeable"] = True
    elif mergeable == "CONFLICTING":
        snapshot["mergeable"] = False

    for source_key, target_key in (
        ("title", "title"),
        ("baseRefName", "base"),
        ("headRefName", "head"),
        ("headRefOid", "head_sha"),
        ("createdAt", "created_at"),
        ("updatedAt", "updated_at"),
        ("closedAt", "closed_at"),
        ("mergedAt", "merged_at"),
    ):
        value = _str(data.get(source_key))
        if value:
            snapshot[target_key] = value
    head_sha = _str(data.get("headRefOid"))
    if head_sha:
        snapshot["latest_commit_sha"] = head_sha
    merge_commit = data.get("mergeCommit")
    if isinstance(merge_commit, dict):
        oid = _str(merge_commit.get("oid"))
        if oid:
            snapshot["merge_commit_sha"] = oid

    _copy_review_fields(snapshot, data)
    _copy_comment_fields(snapshot, data)
    _copy_ci_fields(snapshot, data)
    return snapshot


def _copy_review_fields(snapshot: dict[str, Any], data: dict[str, Any]) -> None:
    reviews = data.get("reviews")
    reviews = reviews if isinstance(reviews, list) else []
    snapshot["review_count"] = len(reviews)

    reactions = data.get("reactionGroups")
    reactions = reactions if isinstance(reactions, list) else []
    reaction_count = 0
    has_thumbs_up = False
    for reaction in reactions:
        if not isinstance(reaction, dict):
            continue
        users = reaction.get("users")
        count = users.get("totalCount") if isinstance(users, dict) else None
        if isinstance(count, int) and not isinstance(count, bool):
            reaction_count += count
            if _str(reaction.get("content")).upper() == "THUMBS_UP" and count > 0:
                has_thumbs_up = True
    snapshot["reaction_count"] = reaction_count

    decision = _str(data.get("reviewDecision")).upper()
    if decision == "CHANGES_REQUESTED":
        snapshot["review_signal"] = "changes_requested"
    elif decision == "APPROVED":
        snapshot["review_signal"] = "approved"
    elif has_thumbs_up:
        snapshot["review_signal"] = "thumbs_up"
    elif reviews:
        snapshot["review_signal"] = "commented"
    else:
        # Explicit reviews-clear sentinel consumed by _merge_pr_handoff_dicts.
        snapshot["review_signal"] = ""


def _apply_review_threads(snapshot: dict[str, Any], threads: list[Any]) -> None:
    # A successful (possibly empty) thread fetch always records an integer
    # unresolved count so the review gate's "approved and unresolved_thread_count
    # == 0" pass condition can be satisfied.
    snapshot["review_thread_count"] = len(threads)
    unresolved = [
        thread
        for thread in threads
        if isinstance(thread, dict)
        and thread.get("isResolved") is not True
        and thread.get("isOutdated") is not True
    ]
    snapshot["unresolved_thread_count"] = len(unresolved)
    snapshot["unresolved_threads"] = _compact_threads(unresolved)
    # The unresolved-thread identifiers (path/line/id) tell the fix agent where
    # to look but not what was asked. Surface the inline review-comment text
    # itself through the untrusted comment channel so the follow-up turn sees
    # the actual requested change.
    thread_comments = _review_thread_comments(unresolved)
    if thread_comments:
        existing = snapshot.get("latest_comments")
        existing = existing if isinstance(existing, list) else []
        snapshot["latest_comments"] = (thread_comments + existing)[:_PR_DETAIL_LIMIT]


def _review_thread_comments(threads: list[Any]) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for thread in threads:
        if not isinstance(thread, dict):
            continue
        path = _str(thread.get("path"))
        for comment in _as_node_list(thread.get("comments"))[:1]:
            if not isinstance(comment, dict):
                continue
            body = _str(comment.get("body"))
            if not body:
                continue
            prefix = f"[review thread {path}] " if path else "[review thread] "
            item: dict[str, Any] = {
                "body": _compact_text(f"{prefix}{' '.join(body.split())}")
            }
            url = _str(comment.get("url"))
            if url:
                item["url"] = url[:_PR_TEXT_MAX_CHARS]
            collected.append(item)
    return collected


def _copy_comment_fields(snapshot: dict[str, Any], data: dict[str, Any]) -> None:
    comments = _as_node_list(data.get("comments"))
    snapshot["comment_count"] = len(comments)
    snapshot["latest_comments"] = _compact_comments(comments[-_PR_DETAIL_LIMIT:])


def _copy_ci_fields(snapshot: dict[str, Any], data: dict[str, Any]) -> None:
    checks = _as_node_list(data.get("statusCheckRollup"))
    failing: list[str] = []
    pending: list[str] = []
    saw_completed = False
    for check in checks:
        if not isinstance(check, dict):
            continue
        name = (
            _str(check.get("name"))
            or _str(check.get("context"))
            or "unnamed check"
        )
        status = _str(check.get("status")).upper()
        conclusion = _str(check.get("conclusion")).upper()
        state = _str(check.get("state")).upper()
        if state:
            # StatusContext entries report a single ``state`` field.
            if state == "SUCCESS":
                saw_completed = True
            elif state in {"PENDING", "EXPECTED"}:
                pending.append(name)
            else:
                failing.append(name)
            continue
        # CheckRun entries report ``status``; only a COMPLETED run counts as a
        # finished check. A missing/unknown status is treated as still pending
        # rather than silently counted as a passed check.
        if status != "COMPLETED":
            pending.append(name)
            continue
        saw_completed = True
        if conclusion and conclusion not in _SUCCESS_CONCLUSIONS:
            failing.append(name)

    if failing:
        snapshot["ci_status"] = "failure"
    elif pending:
        snapshot["ci_status"] = "pending"
    else:
        # No failing or pending checks: either everything passed or the PR has
        # no CI configured. Either way nothing is blocking on CI.
        snapshot["ci_status"] = "success" if (saw_completed or not checks) else "pending"
    snapshot["failing_jobs"] = [{"name": name} for name in failing[:_PR_DETAIL_LIMIT]]
    snapshot["pending_jobs"] = [{"name": name} for name in pending[:_PR_DETAIL_LIMIT]]


def _compact_threads(threads: list[Any]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for thread in threads[:_PR_DETAIL_LIMIT]:
        if not isinstance(thread, dict):
            continue
        item: dict[str, Any] = {}
        path = _str(thread.get("path"))
        if path:
            item["path"] = path[:_PR_TEXT_MAX_CHARS]
        line = thread.get("line")
        if isinstance(line, int) and not isinstance(line, bool):
            item["line"] = line
        identifier = _str(thread.get("id"))
        if identifier:
            item["id"] = identifier[:_PR_TEXT_MAX_CHARS]
        if item:
            compact.append(item)
    return compact


def _compact_comments(comments: list[Any]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        item: dict[str, Any] = {}
        body = _str(comment.get("body"))
        if body:
            item["body"] = _compact_text(" ".join(body.split()))
        url = _str(comment.get("url"))
        if url:
            item["url"] = url[:_PR_TEXT_MAX_CHARS]
        created = _str(comment.get("createdAt"))
        if created:
            item["created_at"] = created
        if item:
            compact.append(item)
    return compact


def _as_node_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        nodes = value.get("nodes")
        if isinstance(nodes, list):
            return nodes
    return []


def _compact_text(value: str) -> str:
    text = value.strip()
    if len(text) <= _PR_TEXT_MAX_CHARS:
        return text
    return f"{text[: _PR_TEXT_MAX_CHARS - 3].rstrip()}..."


def _str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _json_or_none(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
