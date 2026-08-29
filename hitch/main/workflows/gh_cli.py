"""Subprocess execution layer for the ``gh`` and ``git`` CLIs.

Isolates the raw subprocess runners and the low-level gh/git helpers used by
the PR workflow from the handoff-parsing and state-key coupled orchestration in
``system_agents``. These helpers depend only on stdlib, Django, and the leaf
siblings ``pr_handoff``, ``gh_observations``, and ``sdk_values`` -- never on
``system_agents`` -- so importing them here introduces no cycle.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, Iterable
from typing import Any, Protocol

from hitch.main.git_support import hermetic_git_env
from hitch.main.models import SystemWorkflow
from hitch.main.runtime.sdk_values import string_from_any
from hitch.main.workflows.gh_observations import _review_threads_page, _status_checks_page
from hitch.main.workflows.pr_handoff import _compact_pr_handoff, _pr_handoff_is_terminal


class _CwdContext(Protocol):
    cwd: Any

_GH_PR_CREATE_TIMEOUT_SECONDS = 120
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
# inherit the 120s create timeout for every request. These reads normally return in a
# couple seconds, so cap each call.
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


class _PrWorkflowNoCommitsError(RuntimeError):
    """The PR branch has no commits beyond the base, so no PR is warranted.

    The PR cleanup turn can legitimately produce no delta (it rebased its work
    away or the diff was already clean). That is a successful no-op, not a
    failure, so it must complete the workflow rather than block it.
    """


def _push_current_branch_with_git_cli(
    workflow: SystemWorkflow,
    *,
    active_pr_handoff: dict[str, Any] | None = None,
) -> None:
    branch = _current_git_branch(workflow)
    _ensure_not_default_git_branch(workflow, branch)
    refspec = f"HEAD:refs/heads/{branch}"
    push_args = ["push", "-u", "origin", refspec]
    pushed = _run_git_cli(workflow, push_args)
    if pushed.returncode == 0:
        return
    expected_head_sha = _force_push_expected_head_sha(
        branch, pushed, active_pr_handoff=active_pr_handoff
    )
    if expected_head_sha:
        lease = f"--force-with-lease=refs/heads/{branch}:{expected_head_sha}"
        force_push_args = ["push", lease, "-u", "origin", refspec]
        force_pushed = _run_git_cli(
            workflow,
            force_push_args,
        )
        if force_pushed.returncode == 0:
            return
        raise _GhPrOpenError(
            f"`git {' '.join(force_push_args)}` failed after "
            f"`git {' '.join(push_args)}` was rejected: {_gh_error(force_pushed)}"
        )

    raise _GhPrOpenError(
        f"`git {' '.join(push_args)}` failed: {_gh_error(pushed)}"
    )


def _force_push_expected_head_sha(
    branch: str,
    failed_push: subprocess.CompletedProcess[str],
    *,
    active_pr_handoff: dict[str, Any] | None = None,
) -> str:
    if not _git_push_rejected_non_fast_forward(failed_push):
        return ""
    handoff = _compact_pr_handoff(active_pr_handoff)
    if _pr_handoff_is_terminal(handoff):
        return ""
    if string_from_any(handoff.get("head")) != branch:
        return ""
    return string_from_any(handoff.get("head_sha"))


def _git_push_rejected_non_fast_forward(
    result: subprocess.CompletedProcess[str],
) -> bool:
    detail = f"{result.stderr}\n{result.stdout}".lower()
    rejection_markers = (
        "non-fast-forward",
        "fetch first",
        "tip of your current branch is behind",
        "updates were rejected because the tip",
    )
    return any(marker in detail for marker in rejection_markers)


def _current_git_branch(workflow: SystemWorkflow) -> str:
    result = _run_git_cli(workflow, ["symbolic-ref", "--quiet", "--short", "HEAD"])
    if result.returncode != 0:
        raise _GhPrOpenError(
            f"`git symbolic-ref --short HEAD` failed: {_gh_error(result)}"
        )
    branch = result.stdout.strip()
    if not branch:
        raise _GhPrOpenError("current checkout is detached; cannot push a PR branch")
    return branch


def _ensure_not_default_git_branch(workflow: SystemWorkflow, branch: str) -> None:
    default_branch = _origin_default_git_branch(workflow)
    if default_branch:
        if branch == default_branch:
            raise _GhPrOpenError(f"refusing to push default branch {branch!r}")
        return
    if branch in {"main", "master", "trunk", "develop"}:
        raise _GhPrOpenError(f"refusing to push likely default branch {branch!r}")


def _origin_default_git_branch(workflow: SystemWorkflow) -> str:
    result = _run_git_cli(
        workflow, ["symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD"]
    )
    if result.returncode != 0:
        return ""
    remote_ref = result.stdout.strip()
    if not remote_ref:
        return ""
    prefix = "origin/"
    return (
        remote_ref.removeprefix(prefix)
        if remote_ref.startswith(prefix)
        else remote_ref
    )


def _gh_pr_view_payload(
    workflow: _CwdContext,
    *,
    selector: str | None,
    fields: Iterable[str],
    optional: bool = False,
    timeout_seconds: float = _GH_PR_CREATE_TIMEOUT_SECONDS,
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
    timeout_seconds: float = _GH_PR_CREATE_TIMEOUT_SECONDS,
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
    workflow: SystemWorkflow, args: list[str]
) -> subprocess.CompletedProcess[str]:
    command = ["git", *args]
    try:
        return subprocess.run(
            command,
            cwd=workflow.cwd,
            capture_output=True,
            text=True,
            timeout=_GH_PR_CREATE_TIMEOUT_SECONDS,
            check=False,
            # Inherited repo-discovery overrides (GIT_DIR & co.) would point
            # the push at a different repo; the hermetic env also disables
            # credential prompts so an unauthenticated push fails fast
            # instead of stalling out the full timeout.
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
