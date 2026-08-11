"""Create and discover Hitch-managed git worktrees."""

from __future__ import annotations

import re
import shutil
import tempfile
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from django.conf import settings

from .git_support import GitCommandError, hermetic_git_env, resolved_path, run_git

_resolved_path = resolved_path

_GIT_TIMEOUT_SECONDS = 10
# ``worktree add`` checks out the whole tree and ``worktree remove`` deletes
# it; both scale with repo size, so the metadata-command timeout above would
# SIGKILL git mid-checkout on any large repo and leak the partial worktree.
_GIT_CHECKOUT_TIMEOUT_SECONDS = 300
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_REPO_SLUG_LEN = 48
_SNAPSHOT_COMMIT_ENV = {
    "GIT_AUTHOR_NAME": "Hitch",
    "GIT_AUTHOR_EMAIL": "hitch@localhost",
    "GIT_COMMITTER_NAME": "Hitch",
    "GIT_COMMITTER_EMAIL": "hitch@localhost",
}


class WorktreeCreationError(Exception):
    """Raised when a managed worktree cannot be created."""


class WorktreeCleanupError(Exception):
    """Raised when a managed worktree cannot be removed."""


@dataclass(frozen=True)
class ManagedWorktree:
    path: Path
    branch: str
    source_repo: Path


def snapshot_worktree_to_commit(
    source_cwd: str | Path,
    *,
    message: str = "Snapshot Hitch stacked diff proposal",
) -> str:
    """Create an internal commit for a worktree tree without mutating it."""
    repo = _repo_root(Path(source_cwd))
    if repo is None:
        raise WorktreeCreationError("source cwd is not a git repository")
    parent_sha = (
        _git(repo, ["rev-parse", "--verify", "HEAD^{commit}"], raise_on_error=False)
        or ""
    ).strip()
    with tempfile.TemporaryDirectory(prefix="hitch-worktree-index-") as raw_tmp:
        extra_env = {"GIT_INDEX_FILE": str(Path(raw_tmp) / "index")}
        if parent_sha:
            _git(repo, ["read-tree", parent_sha], extra_env=extra_env)
        else:
            _git(repo, ["read-tree", "--empty"], extra_env=extra_env)
        _git(repo, ["add", "-A", "--"], extra_env=extra_env)
        tree_output = _git(repo, ["write-tree"], extra_env=extra_env)
        if tree_output is None:
            raise WorktreeCreationError("failed to write worktree snapshot tree")
        tree_sha = tree_output.strip()
    args = ["commit-tree", tree_sha, "-m", message]
    if parent_sha:
        args[2:2] = ["-p", parent_sha]
    commit_output = _git(repo, args, extra_env=_SNAPSHOT_COMMIT_ENV)
    if commit_output is None:
        raise WorktreeCreationError("failed to create worktree snapshot commit")
    return commit_output.strip()


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
        _checked_out_add_worktree(
            repo,
            ["worktree", "add", "-b", branch, str(path), base_ref],
            branch=branch,
            path=path,
            disable_hooks=disable_hooks,
        )
    elif base_ref == "HEAD":
        _checked_out_add_worktree(
            repo,
            ["worktree", "add", "--orphan", "-b", branch, str(path)],
            branch=branch,
            path=path,
            disable_hooks=disable_hooks,
        )
    else:
        raise WorktreeCreationError(f"base ref {base_ref!r} does not exist")
    return ManagedWorktree(path=path, branch=branch, source_repo=repo)


def cleanup_worktree(worktree: ManagedWorktree) -> None:
    """Remove a just-created managed worktree and its branch."""
    path_existed = worktree.path.exists()
    try:
        try:
            _git(
                worktree.source_repo,
                ["worktree", "remove", "--force", str(worktree.path)],
                error_cls=WorktreeCleanupError,
                timeout=_GIT_CHECKOUT_TIMEOUT_SECONDS,
            )
        except WorktreeCleanupError:
            # git refuses (locked / already-partial worktree) or the repo cannot
            # express the removal; reap the directory directly and prune the stale
            # .git/worktrees/<name> admin entry so retries don't accumulate state.
            shutil.rmtree(worktree.path, ignore_errors=True)
            _git(worktree.source_repo, ["worktree", "prune"], raise_on_error=False)
        if _branch_exists(worktree.source_repo, worktree.branch):
            _git(
                worktree.source_repo,
                ["branch", "-D", worktree.branch],
                error_cls=WorktreeCleanupError,
            )
    finally:
        _invalidate_disk_usage_if_removed(worktree.path, path_existed=path_existed)


def cleanup_managed_worktree_path(cwd: str) -> bool:
    """Remove a Hitch-managed worktree by path if ``cwd`` points at one."""
    path = Path(cwd).expanduser()
    if not _is_under_managed_worktree_base(path):
        return False
    if not (path / ".git").exists():
        # The worktree was deleted (or half-deleted) out-of-band, so the
        # source repo can no longer be located through the gitlink. Free the
        # directory rather than leaving cleanup wedged forever -- but only a
        # branch-addressable leaf (<base>/<slug>/<suffix>): a stale cwd
        # naming the managed base or a slug directory must never rmtree the
        # sibling worktrees that live beneath it.
        if not path.is_dir() or not _is_managed_worktree_leaf(path):
            return False
        path_existed = path.exists()
        shutil.rmtree(path, ignore_errors=True)
        _invalidate_disk_usage_if_removed(path, path_existed=path_existed)
        return True
    branch = _managed_branch_for_path(path)
    common_dir = (
        _git(
            path,
            ["rev-parse", "--path-format=absolute", "--git-common-dir"],
            raise_on_error=False,
        )
        or ""
    ).strip()
    if not branch:
        raise WorktreeCleanupError("managed worktree metadata is incomplete")
    if not common_dir:
        # The source repository is gone or unreadable; its branch and admin
        # entry died with it, so reclaiming the directory is all that's left.
        path_existed = path.exists()
        shutil.rmtree(path, ignore_errors=True)
        _invalidate_disk_usage_if_removed(path, path_existed=path_existed)
        return True
    source_repo = _source_repo_from_common_dir(Path(common_dir))
    cleanup_worktree(ManagedWorktree(path=path, branch=branch, source_repo=source_repo))
    return True


def _invalidate_disk_usage_if_removed(path: Path, *, path_existed: bool) -> None:
    if not path_existed or path.exists():
        return
    # disk_cleanup imports these worktree helpers, so keep this dependency lazy.
    from hitch.main.runtime.disk_cleanup import invalidate_hitch_home_disk_usage

    invalidate_hitch_home_disk_usage()


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


def _is_managed_worktree_path(path: Path) -> bool:
    return _is_under_managed_worktree_base(path) and (path / ".git").exists()


def _is_under_managed_worktree_base(path: Path) -> bool:
    resolved_path = _resolved_path(path)
    resolved_base = _resolved_path(Path(settings.HITCH_WORKTREES_DIR).expanduser())
    try:
        resolved_path.relative_to(resolved_base)
    except ValueError:
        return False
    return True


def _is_managed_worktree_leaf(path: Path) -> bool:
    """Whether ``path`` is a branch-addressable ``<base>/<slug>/<suffix>`` leaf."""
    resolved_path = _resolved_path(path)
    resolved_base = _resolved_path(Path(settings.HITCH_WORKTREES_DIR).expanduser())
    try:
        relative = resolved_path.relative_to(resolved_base)
    except ValueError:
        return False
    return len(relative.parts) == 2


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


def _checked_out_add_worktree(
    repo: Path, args: list[str], *, branch: str, path: Path, disable_hooks: bool
) -> None:
    try:
        _add_worktree(repo, args, disable_hooks=disable_hooks)
    except WorktreeCreationError:
        # ``worktree add -b`` creates the branch and the .git/worktrees entry
        # before populating the tree, so a failure (notably a timeout killing
        # git mid-checkout) strands all three with no ManagedWorktree handle
        # for the caller to clean up. Reap them here so a retry starts fresh.
        shutil.rmtree(path, ignore_errors=True)
        _git(repo, ["worktree", "prune"], raise_on_error=False)
        if _branch_exists(repo, branch):
            _git(repo, ["branch", "-D", branch], raise_on_error=False)
        raise


def _add_worktree(repo: Path, args: list[str], *, disable_hooks: bool) -> None:
    if disable_hooks:
        with tempfile.TemporaryDirectory(prefix="hitch-hooks-") as raw_hooks:
            _git(
                repo,
                args,
                hooks_path=Path(raw_hooks),
                timeout=_GIT_CHECKOUT_TIMEOUT_SECONDS,
            )
        return
    _git(repo, args, timeout=_GIT_CHECKOUT_TIMEOUT_SECONDS)


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
    extra_env: dict[str, str] | None = None,
    timeout: float = _GIT_TIMEOUT_SECONDS,
) -> str | None:
    try:
        result = run_git(
            cwd,
            args,
            timeout=timeout,
            hooks_path=hooks_path,
            env=hermetic_git_env(extra_env),
        )
    except GitCommandError as exc:
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
