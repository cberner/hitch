"""Discover, identify, and update local git repositories."""

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .git_support import GitCommandError, run_git
from .git_support import resolved_path as _resolved_path

_GIT_TIMEOUT_SECONDS = 10
_GIT_PULL_TIMEOUT_SECONDS = 120
_DEFAULT_BRANCH_NAMES = ("main", "master", "trunk", "develop")
_DEFAULT_BRANCH_HEAD_REFS = ("refs/remotes/origin/HEAD",)


class AutoPullError(Exception):
    """Raised when Hitch cannot fast-forward the default branch from origin."""


@dataclass(frozen=True)
class AutoPullResult:
    branch: str
    before_sha: str
    after_sha: str
    changed: bool


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


def repo_root(cwd: str | Path) -> Path | None:
    """Return the resolved git worktree root for ``cwd``, or None if unavailable."""
    repo = _repo_root(Path(cwd).expanduser())
    return _resolved_path(repo) if repo is not None else None


def default_branch_commit_hash(cwd: str | Path) -> str | None:
    """Return the current commit hash for the repository's default branch."""
    repo = _repo_root(Path(cwd).expanduser())
    if repo is None:
        return None
    ref = _default_branch_ref(repo)
    if ref is None:
        return None
    return _commit_hash_for_ref(repo, ref)


def commit_hash_for_ref(cwd: str | Path, ref: str) -> str | None:
    """Return the commit hash for ``ref`` in the repository containing ``cwd``."""
    repo = _repo_root(Path(cwd).expanduser())
    if repo is None:
        return None
    return _commit_hash_for_ref(repo, ref)


def default_branch_name(cwd: str | Path) -> str | None:
    """Return the repository default branch name when it can be resolved."""
    repo = _repo_root(Path(cwd).expanduser())
    if repo is None:
        return None
    ref = _default_branch_ref(repo)
    if ref is None:
        return None
    return _branch_name_from_ref(ref) or None


def pull_default_branch_from_origin(cwd: str | Path) -> AutoPullResult:
    """Fast-forward the default branch in ``cwd`` from ``origin``.

    This intentionally runs in the repository checkout named by ``cwd`` and
    refuses to pull when that checkout is not currently on the default branch.
    Otherwise ``git pull origin <default>`` would update whichever branch the
    user had checked out.
    """
    repo = _repo_root(Path(cwd).expanduser())
    if repo is None:
        raise AutoPullError("project repository is unavailable")
    branch = _origin_default_branch_name(repo)
    if not branch:
        raise AutoPullError("project default branch is unavailable")
    current_branch = _current_branch_name(repo)
    if current_branch != branch:
        checkout = current_branch or "detached HEAD"
        raise AutoPullError(
            f"project repository is on {checkout}, not default branch {branch}"
        )
    if not _worktree_is_clean(repo):
        raise AutoPullError("project repository has uncommitted changes")
    before_sha = _commit_hash_for_ref(repo, "HEAD") or ""
    with tempfile.TemporaryDirectory(prefix="hitch-hooks-") as raw_hooks:
        result = _run_git_for_auto_pull(
            repo,
            [
                "pull",
                "--ff-only",
                "--no-recurse-submodules",
                "--no-tags",
                "origin",
                branch,
            ],
            hooks_path=Path(raw_hooks),
        )
    if result.returncode != 0:
        raise AutoPullError(_git_failure_message(result))
    after_sha = _commit_hash_for_ref(repo, "HEAD") or ""
    origin_sha = _commit_hash_for_ref(repo, f"refs/remotes/origin/{branch}") or ""
    if after_sha != origin_sha:
        raise AutoPullError(f"project repository is ahead of origin/{branch}")
    return AutoPullResult(
        branch=branch,
        before_sha=before_sha,
        after_sha=after_sha,
        changed=bool(before_sha and after_sha and before_sha != after_sha),
    )


def default_branch_checkout_commit_hash(cwd: str | Path) -> str | None:
    """Return the default branch SHA only when the checkout has that clean tree."""
    repo = _repo_root(Path(cwd).expanduser())
    if repo is None:
        return None
    default_ref = _default_branch_ref(repo)
    if default_ref is None:
        return None
    default_branch = _branch_name_from_ref(default_ref)
    current_branch = _current_branch_name(repo)
    if not default_branch or current_branch != default_branch:
        return None
    default_sha = _commit_hash_for_ref(repo, default_ref)
    if not default_sha:
        return None
    head_sha = _commit_hash_for_ref(repo, "HEAD")
    if head_sha != default_sha:
        return None
    if not _worktree_is_clean(repo):
        return None
    return default_sha


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
        result = run_git(cwd, args, timeout=_GIT_TIMEOUT_SECONDS)
    except GitCommandError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.decode("utf-8", errors="replace")


def _run_git_for_auto_pull(
    cwd: Path, args: list[str], *, hooks_path: Path | None = None
) -> subprocess.CompletedProcess[bytes]:
    try:
        return run_git(
            cwd,
            args,
            timeout=_GIT_PULL_TIMEOUT_SECONDS,
            hooks_path=hooks_path,
        )
    except GitCommandError as exc:
        raise AutoPullError(str(exc)) from exc


def _git_failure_message(result: object) -> str:
    stdout = getattr(result, "stdout", b"")
    stderr = getattr(result, "stderr", b"")
    output = b"\n".join(part for part in (stderr, stdout) if isinstance(part, bytes))
    message = output.decode("utf-8", errors="replace").strip()
    if not message:
        return "git pull failed"
    return message.splitlines()[-1]


def _commit_hash_for_ref(repo: Path, ref: str) -> str | None:
    output = _git_output(repo, ["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"])
    if not output:
        return None
    sha = output.strip().splitlines()[0] if output.strip() else ""
    return sha or None


def _default_branch_ref(repo: Path) -> str | None:
    explicit_ref = _explicit_default_branch_ref(repo)
    if explicit_ref is not None:
        return explicit_ref
    local_refs = _local_branch_refs(repo)
    if len(local_refs) == 1:
        return local_refs[0]
    named_refs = _named_default_branch_refs(repo)
    if len(named_refs) == 1:
        return next(iter(named_refs.values()))
    if named_refs:
        return None
    return None


def _origin_default_branch_name(repo: Path) -> str | None:
    ref = _explicit_default_branch_ref(repo)
    if ref is None or not ref.startswith("refs/remotes/origin/"):
        return None
    branch = ref.removeprefix("refs/remotes/origin/")
    return branch or None


def _explicit_default_branch_ref(repo: Path) -> str | None:
    for ref in _DEFAULT_BRANCH_HEAD_REFS:
        resolved_ref = _resolve_symbolic_ref(repo, ref)
        if resolved_ref == ref:
            continue
        if _commit_hash_for_ref(repo, resolved_ref):
            return resolved_ref
    return None


def _resolve_symbolic_ref(repo: Path, ref: str) -> str:
    output = _git_output(repo, ["symbolic-ref", "--quiet", ref])
    resolved = output.strip() if output else ""
    return resolved or ref


def _named_default_branch_refs(repo: Path) -> dict[str, str]:
    refs_by_name: dict[str, str] = {}
    for name in _DEFAULT_BRANCH_NAMES:
        local_ref = f"refs/heads/{name}"
        remote_ref = f"refs/remotes/origin/{name}"
        if _commit_hash_for_ref(repo, local_ref):
            refs_by_name[name] = local_ref
        elif _commit_hash_for_ref(repo, remote_ref):
            refs_by_name[name] = remote_ref
    return refs_by_name


def _local_branch_refs(repo: Path) -> list[str]:
    output = _git_output(repo, ["for-each-ref", "--format=%(refname)", "refs/heads"])
    if not output:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def _current_branch_name(repo: Path) -> str | None:
    output = _git_output(repo, ["symbolic-ref", "--quiet", "--short", "HEAD"])
    branch = output.strip() if output else ""
    return branch or None


def _branch_name_from_ref(ref: str) -> str:
    if ref.startswith("refs/remotes/"):
        parts = ref.split("/")
        return "/".join(parts[3:])
    if ref.startswith("refs/heads/"):
        return ref.removeprefix("refs/heads/")
    return ref.rsplit("/", maxsplit=1)[-1]


def _worktree_is_clean(cwd: Path) -> bool:
    output = _git_output(cwd, ["-c", "core.fsmonitor=false", "status", "--porcelain"])
    return output == ""


def _repo_root(cwd: Path) -> Path | None:
    output = _git_output(cwd, ["rev-parse", "--show-toplevel"])
    if not output:
        return None
    root = output.strip()
    return Path(root) if root else None
