"""Subprocess execution layer for the ``gh`` and ``git`` CLIs.

Isolates the raw subprocess runners and low-level gh/git helpers used by PR
watching from handoff parsing and durable session-state tracking. These helpers
depend only on stdlib, Django, and the leaf siblings ``pr_handoff``,
``gh_observations``, and ``sdk_values``.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, Iterable
from typing import Any, Protocol

from hitch.main.git_support import hermetic_git_env
from hitch.main.runtime.sdk_values import string_from_any
from hitch.main.workflows.gh_observations import _review_threads_page, _status_checks_page


class _CwdContext(Protocol):
    @property
    def cwd(self) -> Any: ...

_GH_CLI_TIMEOUT_SECONDS = 120
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
_GH_REVIEW_THREAD_PAGE_LIMIT = 5
_GH_STATUS_CHECK_PAGE_LIMIT = 10
# PR watching uses gh pr view plus paginated
# reviewThreads/statusCheckRollup GraphQL reads. Without a bound, one watch can
# inherit the 120s general CLI timeout for every request. These reads normally
# return in a couple seconds, so cap each call.
_GH_PR_VIEW_TIMEOUT_SECONDS = 20
_GH_REVIEW_THREADS_QUERY = """
query($owner: String!, $repo: String!, $number: Int!, $after: String) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $number) {
      reviewThreads(first: 100, after: $after) {
        pageInfo {
          hasNextPage
          endCursor
        }
        nodes {
          id
          isOutdated
          isResolved
          line
          path
          startLine
          comments(last: 20) {
            nodes {
              author {
                login
              }
              body
              databaseId
              id
              line
              path
              url
            }
          }
        }
      }
    }
  }
}
""".strip()
_GH_STATUS_CHECKS_QUERY = """
query($owner: String!, $repo: String!, $number: Int!, $after: String) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $number) {
      statusCheckRollup {
        contexts(first: 100, after: $after) {
          pageInfo {
            hasNextPage
            endCursor
          }
          nodes {
            __typename
            ... on CheckRun {
              conclusion
              detailsUrl
              name
              status
            }
            ... on StatusContext {
              context
              state
              targetUrl
            }
          }
        }
      }
    }
  }
}
""".strip()


class _GhPrOpenError(RuntimeError):
    pass


def _gh_pr_view_payload(
    workflow: _CwdContext,
    *,
    selector: str | None,
    fields: Iterable[str],
    optional: bool = False,
    timeout_seconds: float = _GH_CLI_TIMEOUT_SECONDS,
) -> dict[str, Any] | None:
    args = ["pr", "view"]
    if selector:
        args.append(selector)
    args.extend(["--json", ",".join(fields)])
    viewed = _run_gh_cli(workflow, args, timeout_seconds=timeout_seconds)
    if viewed.returncode != 0:
        if optional:
            return None
        raise _GhPrOpenError(f"`gh pr view` failed: {_gh_error(viewed)}")
    try:
        payload = json.loads(viewed.stdout)
    except json.JSONDecodeError as exc:
        raise _GhPrOpenError(f"`gh pr view` returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise _GhPrOpenError("`gh pr view` returned a non-object payload")
    return payload


def _run_gh_cli(
    workflow: _CwdContext,
    args: list[str],
    *,
    timeout_seconds: float = _GH_CLI_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "GH_PROMPT_DISABLED": "1"}
    command = ["gh", *args]
    try:
        return subprocess.run(
            command,
            cwd=workflow.cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise _GhPrOpenError(f"`{' '.join(command)}` timed out") from exc
    except OSError as exc:
        raise _GhPrOpenError(f"`{' '.join(command)}` could not run: {exc}") from exc


def _run_git_cli(
    workflow: _CwdContext, args: list[str]
) -> subprocess.CompletedProcess[str]:
    command = ["git", *args]
    try:
        return subprocess.run(
            command,
            cwd=workflow.cwd,
            capture_output=True,
            text=True,
            timeout=_GH_CLI_TIMEOUT_SECONDS,
            check=False,
            # Inherited repo-discovery overrides (GIT_DIR & co.) would point
            # the command at a different repo. The hermetic environment also
            # prevents an unexpected credential prompt from stalling it.
            env=hermetic_git_env(),
        )
    except subprocess.TimeoutExpired as exc:
        raise _GhPrOpenError(f"`{' '.join(command)}` timed out") from exc
    except OSError as exc:
        raise _GhPrOpenError(f"`{' '.join(command)}` could not run: {exc}") from exc


def _gh_error(result: subprocess.CompletedProcess[str]) -> str:
    detail = (result.stderr or result.stdout or "").strip()
    if not detail:
        return f"exit status {result.returncode}"
    return " ".join(detail.split())[:500]


def _gh_pr_review_threads(
    workflow: _CwdContext,
    handoff: dict[str, Any],
    *,
    command_timeout_seconds: Callable[[], float] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    repo = string_from_any(handoff.get("repository_full_name"))
    number = handoff.get("pr_number")
    if "/" not in repo or not isinstance(number, int) or isinstance(number, bool):
        return [], True
    owner, repo_name = repo.split("/", 1)
    threads: list[dict[str, Any]] = []
    after = ""
    for _page in range(_GH_REVIEW_THREAD_PAGE_LIMIT):
        args = [
            "api",
            "graphql",
            "-f",
            f"query={_GH_REVIEW_THREADS_QUERY}",
            "-F",
            f"owner={owner}",
            "-F",
            f"repo={repo_name}",
            "-F",
            f"number={number}",
        ]
        if after:
            args.extend(["-F", f"after={after}"])
        result = _run_gh_cli(
            workflow,
            args,
            timeout_seconds=(
                command_timeout_seconds()
                if command_timeout_seconds is not None
                else _GH_PR_VIEW_TIMEOUT_SECONDS
            ),
        )
        if result.returncode != 0:
            raise _GhPrOpenError(f"`gh api graphql` failed: {_gh_error(result)}")
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise _GhPrOpenError(
                f"`gh api graphql` returned invalid JSON: {exc}"
            ) from exc
        page = _review_threads_page(payload)
        threads.extend(page["nodes"])
        if not page["has_next_page"] or not page["end_cursor"]:
            return threads, True
        after = page["end_cursor"]
    return threads, False


def _gh_pr_status_checks(
    workflow: _CwdContext,
    handoff: dict[str, Any],
    *,
    command_timeout_seconds: Callable[[], float] | None = None,
) -> tuple[Any, bool]:
    repo = string_from_any(handoff.get("repository_full_name"))
    number = handoff.get("pr_number")
    if "/" not in repo or not isinstance(number, int) or isinstance(number, bool):
        return None, True
    owner, repo_name = repo.split("/", 1)
    checks: list[dict[str, Any]] = []
    after = ""
    for _page in range(_GH_STATUS_CHECK_PAGE_LIMIT):
        args = [
            "api",
            "graphql",
            "-f",
            f"query={_GH_STATUS_CHECKS_QUERY}",
            "-F",
            f"owner={owner}",
            "-F",
            f"repo={repo_name}",
            "-F",
            f"number={number}",
        ]
        if after:
            args.extend(["-F", f"after={after}"])
        result = _run_gh_cli(
            workflow,
            args,
            timeout_seconds=(
                command_timeout_seconds()
                if command_timeout_seconds is not None
                else _GH_PR_VIEW_TIMEOUT_SECONDS
            ),
        )
        if result.returncode != 0:
            raise _GhPrOpenError(f"`gh api graphql` failed: {_gh_error(result)}")
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise _GhPrOpenError(
                f"`gh api graphql` returned invalid JSON: {exc}"
            ) from exc
        page = _status_checks_page(payload)
        if page["nodes"] is None:
            return None, True
        checks.extend(page["nodes"])
        if not page["has_next_page"] or not page["end_cursor"]:
            return checks, True
        after = page["end_cursor"]
    return checks, False
