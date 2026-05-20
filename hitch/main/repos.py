"""Discover and identify local git repositories."""

import subprocess
from pathlib import Path

_GIT_TIMEOUT_SECONDS = 10


def discover_repos(home: Path | None = None, *, max_depth: int = 2) -> list[Path]:
    """Return git repo roots within ``max_depth`` directory levels of ``home``.

    A directory is considered a repo root when it contains a ``.git`` entry,
    which matches both standard checkouts (directory) and worktrees/submodules
    (file). Once a repo is detected its subtree is not descended into.
    """
    base = (home if home is not None else Path.home()).expanduser()
    if not base.is_dir():
        return []

    found: dict[Path, Path] = {}
    _walk(base, depth=0, max_depth=max_depth, found=found)
    return sorted(found.values())


def _walk(directory: Path, *, depth: int, max_depth: int, found: dict[Path, Path]) -> None:
    if depth >= max_depth:
        return
    try:
        entries = list(directory.iterdir())
    except (OSError, PermissionError):
        return
    for entry in entries:
        if not entry.is_dir():
            continue
        if (entry / ".git").exists():
            try:
                key = entry.resolve()
            except OSError:
                key = entry
            found.setdefault(key, entry)
            continue
        _walk(entry, depth=depth + 1, max_depth=max_depth, found=found)


def git_common_dir(cwd: str | Path) -> Path | None:
    """Return the resolved git common dir for ``cwd``, or None if unavailable."""
    path = Path(cwd).expanduser()
    output = _git_output(path, ["rev-parse", "--git-common-dir"])
    if not output:
        return None
    common = Path(output.strip())
    if not common.is_absolute():
        common = path / common
    return _resolved_path(common)


def same_repo_or_worktree(cwd: str | Path, repo_path: str | Path, repo_common_dir: str = "") -> bool:
    """Return whether ``cwd`` is ``repo_path`` or a worktree of it."""
    cwd_path = _resolved_path(Path(cwd).expanduser())
    repo = _resolved_path(Path(repo_path).expanduser())
    if cwd_path == repo:
        return True
    expected_common = Path(repo_common_dir) if repo_common_dir else git_common_dir(repo)
    actual_common = git_common_dir(cwd_path)
    if expected_common is None or actual_common is None:
        return False
    return _resolved_path(expected_common) == _resolved_path(actual_common)


def _git_output(cwd: Path, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.decode("utf-8", errors="replace")


def _resolved_path(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path
