"""Shared low-level helpers for invoking git subprocesses.

Each domain module (``diffs``, ``repos``, ``worktrees``, ``local_merges``) keeps
its own thin wrapper that owns the policy decisions -- the timeout, the
environment, and how a failure is surfaced. This module owns the one mechanism
they all share: building the ``git -C <cwd>`` command line and running it under a
timeout. Centralizing the spawn/timeout handling keeps that error-prone surface
in a single tested place instead of four subtly diverging copies.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Literal, overload


class GitCommandError(Exception):
    """Raised when a git subprocess cannot be spawned or times out."""


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
    to their own contract.
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
            env=env,
            text=text,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GitCommandError(str(exc)) from exc


def resolved_path(path: Path) -> Path:
    """Resolve ``path`` to an absolute path, falling back to the input on error."""
    try:
        return path.resolve()
    except OSError:
        return path
