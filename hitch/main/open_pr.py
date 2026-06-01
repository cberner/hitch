"""Open a GitHub pull request for a worktree via the ``gh`` CLI.

Hitch owns PR creation for the auto-PR workflow: the work agent only prepares
and pushes the branch, and Hitch opens the PR once its turn is done. Keeping the
``gh`` invocation here means the agent never needs the create-PR tool and the
resulting PR identity flows straight into the follow-up monitor handoff.
"""

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hitch.main import repos

_GIT_TIMEOUT_SECONDS = 30
_GH_TIMEOUT_SECONDS = 120
_PR_URL_RE = re.compile(
    r"https://github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)"
    r"/pull/(?P<number>[0-9]+)"
)
# Refuse to push these when ``origin/HEAD`` is unset and the default cannot be
# resolved authoritatively, so the guard fails closed rather than pushing a
# default branch directly.
_COMMON_DEFAULT_BRANCHES = frozenset({"main", "master", "trunk", "develop"})


class OpenPrError(Exception):
    """Raised when Hitch cannot open a pull request via ``gh``."""


@dataclass(frozen=True)
class OpenedPr:
    url: str
    repository_full_name: str
    pr_number: int
    head_sha: str

    def as_handoff(self) -> dict[str, Any]:
        handoff: dict[str, Any] = {
            "url": self.url,
            "repository_full_name": self.repository_full_name,
            "pr_number": self.pr_number,
            "state": "open",
        }
        if self.head_sha:
            handoff["head_sha"] = self.head_sha
            handoff["latest_commit_sha"] = self.head_sha
        return handoff


def open_pull_request(cwd: str | Path) -> OpenedPr:
    """Push the worktree branch and open (or reuse) its PR via ``gh``.

    Returns the opened PR's identity so the caller can hand it to the PR
    follow-up monitor. Raises :class:`OpenPrError` for any git/``gh`` failure.
    """
    path = Path(cwd).expanduser()
    branch = _current_branch(path)
    if not branch:
        raise OpenPrError("could not determine the current git branch")
    # Never push the default branch: a ``/pr`` workflow run against the repo cwd
    # (rather than a managed worktree) could otherwise update origin's default
    # branch directly, bypassing the review-branch + approval path. Resolve the
    # default authoritatively (origin/HEAD only) so a single-branch checkout's
    # feature branch is not misread as the default; when it cannot be resolved,
    # fail closed on the common default names, while still allowing a known
    # non-default integration branch like ``develop`` to open a PR.
    default_branch = repos.symbolic_default_branch_name(path)
    if branch == default_branch or (
        not default_branch and branch in _COMMON_DEFAULT_BRANCHES
    ):
        raise OpenPrError(
            f"refusing to open a PR from the default branch {branch!r}; "
            "create a feature branch first"
        )
    # The reviewed diff is whatever the QA agent approved; refuse to open a PR
    # against a dirty worktree so we never monitor a branch missing that work.
    if not _worktree_is_clean(path):
        raise OpenPrError(
            "refusing to open a PR with uncommitted changes in the worktree; "
            "commit or discard them first"
        )
    head_sha = _head_sha(path)
    _push_branch(path, branch)
    url = _existing_pr_url(path, branch) or _create_pr(path, branch)
    return _opened_pr(url, head_sha)


def _opened_pr(url: str, head_sha: str) -> OpenedPr:
    match = _PR_URL_RE.search(url)
    if match is None:
        raise OpenPrError(f"gh returned an unrecognized PR URL: {url!r}")
    owner = match.group("owner")
    repo = match.group("repo")
    return OpenedPr(
        url=match.group(0),
        repository_full_name=f"{owner}/{repo}",
        pr_number=int(match.group("number")),
        head_sha=head_sha,
    )


def _current_branch(cwd: Path) -> str:
    result = _run(cwd, ["git", "rev-parse", "--abbrev-ref", "HEAD"])
    branch = result.stdout.strip()
    return "" if branch == "HEAD" else branch


def _worktree_is_clean(cwd: Path) -> bool:
    result = _run(
        cwd, ["git", "-c", "core.fsmonitor=false", "status", "--porcelain"]
    )
    return result.stdout.strip() == ""


def _head_sha(cwd: Path) -> str:
    return _run(cwd, ["git", "rev-parse", "HEAD"], check=False).stdout.strip()


def _push_branch(cwd: Path, branch: str) -> None:
    # The prep turn rebases/cleans up history, so updating an already-pushed PR
    # branch is a non-fast-forward; --force-with-lease updates it while still
    # refusing to clobber commits that landed on the remote since our last fetch.
    result = _run(
        cwd,
        ["git", "push", "--force-with-lease", "-u", "origin", f"{branch}:{branch}"],
        check=False,
    )
    if result.returncode != 0:
        raise OpenPrError(
            result.stderr.strip()
            or f"git push of {branch!r} failed with status {result.returncode}"
        )


def _existing_pr_url(cwd: Path, branch: str) -> str:
    """Return the URL of an open PR for ``branch``, or "" when there is none.

    ``gh pr view <branch>`` can resolve a historical closed/merged PR for the
    branch, so the state is checked explicitly: only an ``OPEN`` PR is reused;
    otherwise the caller creates a fresh PR for the just-pushed branch.
    """
    result = _run(
        cwd,
        ["gh", "pr", "view", branch, "--json", "url,state"],
        check=False,
    )
    if result.returncode != 0:
        return ""
    try:
        data = json.loads(result.stdout)
    except (json.JSONDecodeError, ValueError):
        return ""
    if not isinstance(data, dict) or str(data.get("state", "")).upper() != "OPEN":
        return ""
    url = data.get("url")
    return url if isinstance(url, str) else ""


def _create_pr(cwd: Path, branch: str) -> str:
    result = _run(
        cwd,
        ["gh", "pr", "create", "--fill", "--head", branch],
        check=False,
    )
    if result.returncode != 0:
        raise OpenPrError(
            result.stderr.strip()
            or f"gh pr create failed with status {result.returncode}"
        )
    match = _PR_URL_RE.search(result.stdout)
    if match is None:
        raise OpenPrError("gh pr create did not report a PR URL")
    return match.group(0)


def _run(
    cwd: Path, command: list[str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    timeout = _GH_TIMEOUT_SECONDS if command and command[0] == "gh" else _GIT_TIMEOUT_SECONDS
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            check=False,
            env=_command_env(),
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OpenPrError(f"{command[0]} failed to run: {exc}") from exc
    if check and result.returncode != 0:
        raise OpenPrError(
            result.stderr.strip()
            or f"{' '.join(command)} failed with status {result.returncode}"
        )
    return result


def _command_env() -> dict[str, str]:
    # Inherit the process environment so ``gh`` finds its auth/config, but never
    # let git or gh block on an interactive prompt inside the worker.
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env
