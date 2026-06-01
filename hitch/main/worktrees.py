"""Create and discover Hitch-managed git worktrees."""

from __future__ import annotations

import re
import subprocess
import tempfile
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from django.conf import settings

_GIT_TIMEOUT_SECONDS = 10
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_REPO_SLUG_LEN = 48


class WorktreeCreationError(Exception):
    """Raised when a managed worktree cannot be created."""


class WorktreeCleanupError(Exception):
    """Raised when a managed worktree cannot be removed."""


@dataclass(frozen=True)
class ManagedWorktree:
    path: Path
    branch: str
    source_repo: Path


def create_worktree_for_session(
    source_cwd: str, *, base_ref: str = "HEAD", disable_hooks: bool = False
) -> ManagedWorktree:
    """Create a new branch and worktree for a Codex session."""
    repo = _repo_root(Path(source_cwd))
    if repo is None:
        raise WorktreeCreationError("source cwd is not a git repository")

    repo_slug = _safe_slug(repo.name) or "repo"
    suffix = f"{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    branch = f"hitch/{repo_slug}/{suffix}"
    path = Path(settings.HITCH_WORKTREES_DIR).expanduser() / repo_slug / suffix
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise WorktreeCreationError(str(exc)) from exc
    if _has_commit(repo, base_ref):
        _add_worktree(
            repo,
            ["worktree", "add", "-b", branch, str(path), base_ref],
            disable_hooks=disable_hooks,
        )
    elif base_ref == "HEAD":
        _add_worktree(
            repo,
            ["worktree", "add", "--orphan", "-b", branch, str(path)],
            disable_hooks=disable_hooks,
        )
    else:
        raise WorktreeCreationError(f"base ref {base_ref!r} does not exist")
    return ManagedWorktree(path=path, branch=branch, source_repo=repo)


def cleanup_worktree(worktree: ManagedWorktree) -> None:
    """Remove a just-created managed worktree and its branch."""
    _git(
        worktree.source_repo,
        ["worktree", "remove", "--force", str(worktree.path)],
        error_cls=WorktreeCleanupError,
    )
    if _branch_exists(worktree.source_repo, worktree.branch):
        _git(
            worktree.source_repo,
            ["branch", "-D", worktree.branch],
            error_cls=WorktreeCleanupError,
        )


def cleanup_managed_worktree_path(cwd: str) -> bool:
    """Remove a Hitch-managed worktree by path if ``cwd`` points at one."""
    path = Path(cwd).expanduser()
    if not _is_managed_worktree_path(path):
        return False
    branch = _managed_branch_for_path(path)
    common_dir = (
        _git(
            path,
            ["rev-parse", "--path-format=absolute", "--git-common-dir"],
            error_cls=WorktreeCleanupError,
        )
        or ""
    ).strip()
    if not branch or not common_dir:
        raise WorktreeCleanupError("managed worktree metadata is incomplete")
    source_repo = _source_repo_from_common_dir(Path(common_dir))
    cleanup_worktree(ManagedWorktree(path=path, branch=branch, source_repo=source_repo))
    return True


def discover_managed_worktrees() -> list[Path]:
    """Return Hitch-managed worktree roots."""
    base = Path(settings.HITCH_WORKTREES_DIR).expanduser()
    if not base.is_dir():
        return []
    roots: dict[Path, Path] = {}
    for repo_dir in _child_dirs(base):
        for worktree in _child_dirs(repo_dir):
            if not (worktree / ".git").exists():
                continue
            key = _resolved_path(worktree)
            roots.setdefault(key, worktree)
    return sorted(roots.values())


def is_managed_worktree_path(cwd: str | Path) -> bool:
    """Return whether ``cwd`` points at a Hitch-managed worktree root."""
    path = Path(cwd).expanduser()
    if not _is_managed_worktree_path(path):
        return False
    try:
        _managed_branch_for_path(path)
    except WorktreeCleanupError:
        return False
    return True


def _child_dirs(path: Path) -> Iterator[Path]:
    try:
        for child in path.iterdir():
            try:
                if child.is_dir():
                    yield child
            except OSError:
                continue
    except OSError:
        return


def _resolved_path(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def _is_managed_worktree_path(path: Path) -> bool:
    resolved_path = _resolved_path(path)
    resolved_base = _resolved_path(Path(settings.HITCH_WORKTREES_DIR).expanduser())
    try:
        resolved_path.relative_to(resolved_base)
    except ValueError:
        return False
    return (path / ".git").exists()


def _managed_branch_for_path(path: Path) -> str:
    resolved_path = _resolved_path(path)
    resolved_base = _resolved_path(Path(settings.HITCH_WORKTREES_DIR).expanduser())
    try:
        relative = resolved_path.relative_to(resolved_base)
    except ValueError as exc:
        raise WorktreeCleanupError("path is not under managed worktree root") from exc
    if len(relative.parts) != 2:
        raise WorktreeCleanupError("managed worktree path is not branch-addressable")
    repo_slug, suffix = relative.parts
    return f"hitch/{repo_slug}/{suffix}"


def _source_repo_from_common_dir(common_dir: Path) -> Path:
    if common_dir.name == ".git":
        return common_dir.parent
    return common_dir


def _has_commit(repo: Path, ref: str) -> bool:
    return (
        _git(repo, ["rev-parse", "--verify", ref], raise_on_error=False) is not None
    )


def _add_worktree(repo: Path, args: list[str], *, disable_hooks: bool) -> None:
    if disable_hooks:
        with tempfile.TemporaryDirectory(prefix="hitch-hooks-") as raw_hooks:
            _git(repo, args, hooks_path=Path(raw_hooks))
        return
    _git(repo, args)


def _branch_exists(repo: Path, branch: str) -> bool:
    return (
        _git(
            repo,
            ["show-ref", "--verify", f"refs/heads/{branch}"],
            raise_on_error=False,
        )
        is not None
    )


def _repo_root(cwd: Path) -> Path | None:
    output = _git(cwd, ["rev-parse", "--show-toplevel"], raise_on_error=False)
    if not output:
        return None
    root = output.strip()
    return Path(root) if root else None


def _git(
    cwd: Path,
    args: list[str],
    *,
    raise_on_error: bool = True,
    error_cls: type[Exception] = WorktreeCreationError,
    hooks_path: Path | None = None,
) -> str | None:
    command = ["git"]
    if hooks_path is not None:
        command.extend(["-c", f"core.hooksPath={hooks_path}"])
    command.extend(["-C", str(cwd), *args])
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        if raise_on_error:
            raise error_cls(str(exc)) from exc
        return None
    if result.returncode != 0:
        if raise_on_error:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            raise error_cls(stderr or "git worktree command failed")
        return None
    return result.stdout.decode("utf-8", errors="replace")


def _safe_slug(value: str) -> str:
    slug = _SLUG_RE.sub("-", value.strip())
    slug = re.sub(r"\.{2,}", "-", slug).strip(".-")
    slug = slug[:_MAX_REPO_SLUG_LEN].strip(".-")
    while slug.endswith(".lock"):
        slug = slug[: -len(".lock")].strip(".-")
    return slug
