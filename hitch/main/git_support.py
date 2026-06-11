"""Shared low-level helpers for invoking git subprocesses.

Each domain module (``diffs``, ``repos``, ``worktrees``, ``local_merges``) keeps
its own thin wrapper that owns the policy decisions -- the timeout, the
environment, and how a failure is surfaced. This module owns the one mechanism
they all share: building the ``git -C <cwd>`` command line and running it under a
timeout. Centralizing the spawn/timeout handling keeps that error-prone surface
in a single tested place instead of four subtly diverging copies.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Literal, overload


class GitCommandError(Exception):
    """Raised when a git subprocess cannot be spawned or times out."""


# Environment variables that override git's repository discovery. If the server
# was launched from a git hook (or any tool that exports these), inheriting them
# would silently redirect every ``git -C <cwd>`` call at a different repository
# than the one named on the command line.
_GIT_REPO_OVERRIDE_ENV_VARS = (
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CEILING_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_DIR",
    "GIT_INDEX_FILE",
    "GIT_NAMESPACE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_PREFIX",
    "GIT_WORK_TREE",
)


def hermetic_git_env(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """The process environment minus git's repo-discovery overrides.

    Also disables credential prompts so an unauthenticated remote operation
    fails fast instead of stalling on a hidden terminal prompt.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in _GIT_REPO_OVERRIDE_ENV_VARS
    }
    env["GIT_TERMINAL_PROMPT"] = "0"
    if extra_env:
        env.update(extra_env)
    return env


@overload
def run_git(
    cwd: str | Path,
    args: list[str],
    *,
    timeout: float,
    hooks_path: Path | None = ...,
    env: dict[str, str] | None = ...,
    input_text: str | None = ...,
    text: Literal[False] = ...,
) -> subprocess.CompletedProcess[bytes]: ...


@overload
def run_git(
    cwd: str | Path,
    args: list[str],
    *,
    timeout: float,
    hooks_path: Path | None = ...,
    env: dict[str, str] | None = ...,
    input_text: str | None = ...,
    text: Literal[True],
) -> subprocess.CompletedProcess[str]: ...


def run_git(
    cwd: str | Path,
    args: list[str],
    *,
    timeout: float,
    hooks_path: Path | None = None,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
    text: bool = False,
) -> subprocess.CompletedProcess[Any]:
    """Run ``git -C <cwd> <args>`` and return the completed process.

    Only the spawn-failure and timeout paths are handled here (raised as
    ``GitCommandError``); callers interpret the return code and output according
    to their own contract. ``env=None`` runs under :func:`hermetic_git_env`.
    """
    command = ["git"]
    if hooks_path is not None:
        command.extend(["-c", f"core.hooksPath={hooks_path}"])
    command.extend(["-C", str(cwd), *args])
    try:
        return subprocess.run(
            command,
            input=input_text,
            capture_output=True,
            check=False,
            timeout=timeout,
            env=env if env is not None else hermetic_git_env(),
            text=text,
            # Text mode must round-trip arbitrary bytes: ``git diff --binary``
            # emits non-UTF-8 *text* file content verbatim (git only treats
            # NUL-bearing files as binary), and strict decoding would raise
            # UnicodeDecodeError out of subprocess.run -- bypassing every
            # caller's GitCommandError handling. surrogateescape decodes those
            # bytes losslessly and re-encodes them identically when the same
            # patch is piped back into ``git apply``.
            errors="surrogateescape" if text else None,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GitCommandError(str(exc)) from exc


def resolved_path(path: Path) -> Path:
    """Resolve ``path`` to an absolute path, falling back to the input on error."""
    try:
        return path.resolve()
    except OSError:
        return path
