"""Local git helpers for applying QA-approved session diffs."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from hashlib import sha1
from os import environ, readlink
from pathlib import Path

_GIT_TIMEOUT_SECONDS = 30
_AUTO_MERGE_COMMIT_MESSAGE = "Apply QA-approved Hitch session diff"
_MAX_REVIEW_PATCH_CHARS = 500_000
_INCOMPLETE_REVIEWED_DIFF_LINES = frozenset(
    (
        "[diff truncated]",
        "+Binary file not shown",
        "+File preview truncated",
        "+Symlink not shown",
    )
)
_INCOMPLETE_REVIEWED_DIFF_PREFIXES = ("Binary files ",)
_INCOMPLETE_REVIEWED_DIFF_SUFFIXES = (
    " untracked files omitted from diff preview",
)
_NON_REVIEWABLE_REVIEW_PATCH_LINES = frozenset(
    (
        "GIT binary patch",
        "new file mode 120000",
        "old mode 120000",
        "new mode 120000",
        "deleted file mode 120000",
    )
)
_SUBMODULE_REVIEW_PATCH_LINES = frozenset(
    (
        "new file mode 160000",
        "old mode 160000",
        "new mode 160000",
        "deleted file mode 160000",
    )
)


class LocalBranchMergeError(Exception):
    """Raised when Hitch cannot merge a QA-approved diff into a local branch."""


@dataclass(frozen=True)
class LocalBranchMergeResult:
    branch: str
    commit_sha: str
    target_worktree: str
    changed: bool


@dataclass(frozen=True)
class AutoMergeReviewPatch:
    patch: str
    target_sha: str
    base_sha: str = ""


@dataclass(frozen=True)
class CachedIndexEntry:
    mode: str
    blob_sha: str
    skip_worktree: bool


def local_branch_names(repo_path: str | Path) -> list[str]:
    """Return local branch names for a repo, or an empty list when unavailable."""
    try:
        output = _git(
            Path(repo_path),
            ["for-each-ref", "--format=%(refname:short)", "refs/heads"],
        )
    except LocalBranchMergeError:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def _auto_merge_source_base(
    source_repo: Path, target_ref: str, hooks_path: Path
) -> str:
    ref = _auto_merge_source_base_ref(source_repo, target_ref)
    result = _run_git(
        source_repo,
        ["rev-parse", "--verify", f"{ref}^{{commit}}"],
        hooks_path=hooks_path,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _record_auto_merge_source_base(
    source_repo: Path, target_ref: str, *, hooks_path: Path
) -> None:
    # The source-session tree is the authority for follow-up deltas; the target
    # branch may also contain unrelated commits that must not become deletions.
    tree_sha = _source_worktree_tree(source_repo, hooks_path=hooks_path)
    commit_sha = _git(
        source_repo,
        ["commit-tree", tree_sha, "-m", "Record Hitch auto-merge source base"],
        hooks_path=hooks_path,
    ).strip()
    _git(
        source_repo,
        ["update-ref", _auto_merge_source_base_ref(source_repo, target_ref), commit_sha],
        hooks_path=hooks_path,
    )


def _auto_merge_source_base_ref(source_repo: Path, target_ref: str) -> str:
    worktree_head = _git(source_repo, ["rev-parse", "--git-path", "HEAD"]).strip()
    key = sha1(f"{target_ref}\0{worktree_head}".encode()).hexdigest()
    return f"refs/hitch/auto-merge-source-bases/{key}"


def build_auto_merge_review_patch(
    source_cwd: str | Path, branch: str
) -> AutoMergeReviewPatch:
    """Return the exact reviewable patch that would be applied to the target."""
    branch = branch.strip()
    if not branch:
        raise LocalBranchMergeError("target branch is required")
    source_repo = _repo_root(Path(source_cwd))
    target_ref = f"refs/heads/{branch}"
    target_sha = _target_branch_sha(source_repo, target_ref)
    merge_base_sha = _git(source_repo, ["merge-base", "HEAD", target_ref]).strip()
    if not merge_base_sha:
        raise LocalBranchMergeError(
            "source worktree and target branch have no merge base"
        )
    with tempfile.TemporaryDirectory(prefix="hitch-hooks-") as raw_hooks:
        hooks_path = Path(raw_hooks)
        base_sha = (
            _auto_merge_source_base(source_repo, target_ref, hooks_path)
            or merge_base_sha
        )
        source_tree_sha = _source_worktree_tree(source_repo, hooks_path=hooks_path)
        session_delta_patch = _normalize_patch_text(
            _git(
                source_repo,
                [
                    "diff",
                    "--binary",
                    "--full-index",
                    "--find-renames",
                    "--no-color",
                    "--no-ext-diff",
                    base_sha,
                    source_tree_sha,
                    "--",
                ],
                hooks_path=hooks_path,
            )
        )
        _validate_reviewable_patch(session_delta_patch)
        review_tree_sha = _target_tree_with_session_delta(
            source_repo,
            target_sha,
            session_delta_patch,
            hooks_path=hooks_path,
        )
        patch = _normalize_patch_text(
            _git(
                source_repo,
                [
                    "diff",
                    "--binary",
                    "--full-index",
                    "--find-renames",
                    "--no-color",
                    "--no-ext-diff",
                    target_sha,
                    review_tree_sha,
                    "--",
                ],
                hooks_path=hooks_path,
            )
        )
    _validate_reviewable_patch(patch)
    return AutoMergeReviewPatch(
        patch=patch, target_sha=target_sha, base_sha=base_sha
    )


def merge_worktree_diff_to_branch(
    source_cwd: str | Path,
    branch: str,
    reviewed_patch: str,
    reviewed_target_sha: str,
) -> LocalBranchMergeResult:
    """Apply the QA-reviewed patch to ``branch`` and commit it."""
    branch = branch.strip()
    if not branch:
        raise LocalBranchMergeError("target branch is required")

    source_repo = _repo_root(Path(source_cwd))
    target_ref = f"refs/heads/{branch}"
    target_sha = _target_branch_sha(source_repo, target_ref)
    if target_sha != reviewed_target_sha:
        raise LocalBranchMergeError(
            "target branch changed since QA review; rerun QA before auto merge"
        )
    patch = _validated_reviewed_patch(reviewed_patch)

    with tempfile.TemporaryDirectory(prefix="hitch-local-merge-") as raw_tmp:
        target = Path(raw_tmp) / "target"
        hooks = Path(raw_tmp) / "hooks"
        hooks.mkdir()
        checked_out_path = _checked_out_worktree_for_branch(source_repo, branch)
        if checked_out_path is not None and _same_path(checked_out_path, source_repo):
            raise LocalBranchMergeError(
                "target branch is checked out in the source worktree; "
                "auto merge requires a separate worktree"
            )
        if checked_out_path is not None:
            _ensure_clean_worktree(checked_out_path, branch, hooks_path=hooks)
        _git(
            source_repo,
            ["worktree", "add", "--detach", str(target), target_sha],
            hooks_path=hooks,
        )
        try:
            commit_sha, changed = _commit_patch_in_isolated_worktree(
                target,
                target_sha,
                patch,
                hooks_path=hooks,
            )
            if changed:
                _fast_forward_target_branch(
                    source_repo,
                    branch,
                    target_sha,
                    commit_sha,
                    checked_out_path=checked_out_path,
                    hooks_path=hooks,
                )
            _record_auto_merge_source_base(
                source_repo, target_ref, hooks_path=hooks
            )
            return LocalBranchMergeResult(
                branch=branch,
                commit_sha=commit_sha,
                target_worktree=str(checked_out_path or ""),
                changed=changed,
            )
        finally:
            _remove_worktree(source_repo, target, hooks_path=hooks)


def _commit_patch_in_isolated_worktree(
    target: Path, target_sha: str, patch: str, *, hooks_path: Path
) -> tuple[str, bool]:
    if patch.strip():
        _apply_patch_to_index(target, patch, hooks_path=hooks_path)
    tree_sha = _git(target, ["write-tree"], hooks_path=hooks_path).strip()
    target_tree_sha = _git(
        target, ["rev-parse", f"{target_sha}^{{tree}}"], hooks_path=hooks_path
    ).strip()
    if tree_sha == target_tree_sha:
        return target_sha, False
    commit_sha = _git(
        target,
        [
            "commit-tree",
            tree_sha,
            "-p",
            target_sha,
            "-m",
            _AUTO_MERGE_COMMIT_MESSAGE,
        ],
        hooks_path=hooks_path,
    ).strip()
    return commit_sha, True


def _target_tree_with_session_delta(
    repo: Path,
    target_sha: str,
    session_delta_patch: str,
    *,
    hooks_path: Path,
) -> str:
    target_tree_sha = _git(
        repo, ["rev-parse", f"{target_sha}^{{tree}}"], hooks_path=hooks_path
    ).strip()
    if not session_delta_patch.strip():
        return target_tree_sha
    with tempfile.TemporaryDirectory(prefix="hitch-review-target-") as raw_tmp:
        target = Path(raw_tmp) / "target"
        _git(
            repo,
            ["worktree", "add", "--detach", str(target), target_sha],
            hooks_path=hooks_path,
        )
        try:
            try:
                _apply_patch_to_index(
                    target,
                    session_delta_patch,
                    hooks_path=hooks_path,
                )
            except LocalBranchMergeError as exc:
                if _patch_already_applied(
                    target,
                    session_delta_patch,
                    hooks_path=hooks_path,
                ):
                    return target_tree_sha
                raise LocalBranchMergeError(
                    "session diff does not apply cleanly to target branch; "
                    "rebase the session worktree before auto merge"
                ) from exc
            return _git(target, ["write-tree"], hooks_path=hooks_path).strip()
        finally:
            _remove_worktree(repo, target, hooks_path=hooks_path)


def _apply_patch_to_index(target: Path, patch: str, *, hooks_path: Path) -> None:
    applied = _run_git(
        target,
        ["apply", "--index", "--binary", "-"],
        input_text=patch,
        hooks_path=hooks_path,
        check=False,
    )
    if applied.returncode == 0:
        return
    three_way = _run_git(
        target,
        ["apply", "--index", "--3way", "--binary", "-"],
        input_text=patch,
        hooks_path=hooks_path,
        check=False,
    )
    if three_way.returncode == 0:
        return
    stderr = three_way.stderr.strip() or applied.stderr.strip()
    raise LocalBranchMergeError(stderr or "git apply failed")


def _patch_already_applied(target: Path, patch: str, *, hooks_path: Path) -> bool:
    return (
        _run_git(
            target,
            ["apply", "--check", "--reverse", "--index", "--binary", "-"],
            input_text=patch,
            hooks_path=hooks_path,
            check=False,
        ).returncode
        == 0
    )


def _source_worktree_tree(
    source_repo: Path, *, hooks_path: Path | None = None
) -> str:
    with tempfile.TemporaryDirectory(prefix="hitch-source-index-") as raw_tmp:
        index_path = str(Path(raw_tmp) / "index")
        extra_env = {"GIT_INDEX_FILE": index_path}
        _git(
            source_repo,
            ["read-tree", "--empty"],
            extra_env=extra_env,
            hooks_path=hooks_path,
        )
        for relpath in _worktree_index_paths(source_repo, hooks_path=hooks_path):
            entry = _worktree_index_entry(source_repo, relpath, hooks_path=hooks_path)
            if entry is None:
                continue
            mode, blob_sha = entry
            _git(
                source_repo,
                [
                    "update-index",
                    "--add",
                    "--cacheinfo",
                    f"{mode},{blob_sha},{relpath}",
                ],
                extra_env=extra_env,
                hooks_path=hooks_path,
            )
        return _git(
            source_repo,
            ["write-tree"],
            extra_env=extra_env,
            hooks_path=hooks_path,
        ).strip()


def _worktree_index_paths(source_repo: Path, *, hooks_path: Path | None) -> list[str]:
    output = _git(
        source_repo,
        ["ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        hooks_path=hooks_path,
    )
    return list(dict.fromkeys(path for path in output.split("\0") if path))


def _worktree_index_entry(
    source_repo: Path, relpath: str, *, hooks_path: Path | None
) -> tuple[str, str] | None:
    if "\n" in relpath or "\r" in relpath:
        raise LocalBranchMergeError("auto merge does not support paths with newlines")
    cached_entry = _cached_index_entry(source_repo, relpath, hooks_path=hooks_path)
    if cached_entry is not None and cached_entry.mode == "160000":
        return _submodule_index_entry(
            source_repo,
            relpath,
            cached_sha=cached_entry.blob_sha,
            hooks_path=hooks_path,
        )
    path = source_repo / relpath
    if not path.exists() and not path.is_symlink():
        if cached_entry is not None and cached_entry.skip_worktree:
            return cached_entry.mode, cached_entry.blob_sha
        return None
    if path.is_dir():
        return None
    if path.is_symlink():
        blob_sha = _git(
            source_repo,
            ["hash-object", "-w", "--stdin"],
            input_text=readlink(path),
            hooks_path=hooks_path,
        ).strip()
        return "120000", blob_sha
    if not path.is_file():
        raise LocalBranchMergeError("auto merge does not support submodule changes")
    mode = "100755" if path.stat().st_mode & 0o111 else "100644"
    blob_sha = _git(
        source_repo,
        ["hash-object", "-w", "--no-filters", "--", relpath],
        hooks_path=hooks_path,
    ).strip()
    return mode, blob_sha


def _cached_index_entry(
    source_repo: Path, relpath: str, *, hooks_path: Path | None
) -> CachedIndexEntry | None:
    output = _git(
        source_repo,
        ["ls-files", "-s", "-t", "--", relpath],
        hooks_path=hooks_path,
    ).strip()
    if not output:
        return None
    first = output.splitlines()[0]
    skip_worktree = first.startswith("S ")
    meta = first.split("\t", 1)[0]
    parts = meta.split()
    if len(parts) < 3:
        return None
    return CachedIndexEntry(parts[1], parts[2], skip_worktree)


def _submodule_index_entry(
    source_repo: Path,
    relpath: str,
    *,
    cached_sha: str,
    hooks_path: Path | None,
) -> tuple[str, str]:
    path = source_repo / relpath
    if (path / ".git").exists():
        current_sha = _submodule_head(path, hooks_path=hooks_path)
        if current_sha and current_sha != cached_sha:
            raise LocalBranchMergeError(
                "auto merge does not support submodule changes"
            )
    return "160000", cached_sha


def _submodule_head(path: Path, *, hooks_path: Path | None) -> str:
    try:
        return _git(
            path, ["rev-parse", "--verify", "HEAD"], hooks_path=hooks_path
        ).strip()
    except LocalBranchMergeError:
        return ""


def _validated_reviewed_patch(reviewed_patch: str) -> str:
    if not reviewed_patch.strip():
        return ""
    if _contains_incomplete_preview_marker(reviewed_patch):
        raise LocalBranchMergeError(
            "reviewed diff is incomplete; auto merge requires a complete text diff"
        )
    patch = _normalize_patch_text(reviewed_patch)
    _validate_reviewable_patch(patch)
    return patch


def _contains_incomplete_preview_marker(reviewed_patch: str) -> bool:
    diff_has_index = False
    in_hunk = False
    for line in reviewed_patch.splitlines():
        if line.startswith("diff --git "):
            diff_has_index = False
            in_hunk = False
            continue
        if line.startswith("index "):
            diff_has_index = True
            continue
        if line.startswith("@@ "):
            in_hunk = True
            continue
        if line.startswith(_INCOMPLETE_REVIEWED_DIFF_PREFIXES):
            return True
        if line in _INCOMPLETE_REVIEWED_DIFF_LINES:
            if in_hunk and diff_has_index:
                continue
            return True
        if line.startswith("+") and line.endswith(_INCOMPLETE_REVIEWED_DIFF_SUFFIXES):
            if in_hunk and diff_has_index:
                continue
            return True
    return False


def _validate_reviewable_patch(patch: str) -> None:
    if len(patch) > _MAX_REVIEW_PATCH_CHARS:
        raise LocalBranchMergeError(
            "auto merge diff is too large to review; reduce the change size"
        )
    for line in patch.splitlines():
        if (
            line in _NON_REVIEWABLE_REVIEW_PATCH_LINES
            or line.startswith("index ")
            and line.endswith(" 120000")
        ):
            raise LocalBranchMergeError(
                "auto merge diff contains binary or symlink changes"
            )
        if (
            line in _SUBMODULE_REVIEW_PATCH_LINES
            or line.startswith("index ")
            and line.endswith(" 160000")
        ):
            raise LocalBranchMergeError("auto merge does not support submodule changes")


def _normalize_patch_text(patch: str) -> str:
    if not patch.strip():
        return ""
    return patch if patch.endswith("\n") else f"{patch}\n"


def _fast_forward_target_branch(
    repo: Path,
    branch: str,
    old_sha: str,
    commit_sha: str,
    *,
    checked_out_path: Path | None,
    hooks_path: Path,
) -> None:
    if checked_out_path is not None:
        _git(
            checked_out_path,
            ["merge", "--ff-only", commit_sha],
            hooks_path=hooks_path,
        )
        return
    _git(
        repo,
        ["update-ref", f"refs/heads/{branch}", commit_sha, old_sha],
        hooks_path=hooks_path,
    )


def _repo_root(cwd: Path) -> Path:
    output = _git(cwd, ["rev-parse", "--show-toplevel"])
    root = output.strip()
    if not root:
        raise LocalBranchMergeError("source cwd is not a git repository")
    return Path(root)


def _target_branch_sha(repo: Path, target_ref: str) -> str:
    return _git(repo, ["rev-parse", "--verify", target_ref]).strip()


def _checked_out_worktree_for_branch(repo: Path, branch: str) -> Path | None:
    output = _git(repo, ["worktree", "list", "--porcelain"])
    target_ref = f"refs/heads/{branch}"
    for record in output.strip().split("\n\n"):
        path = ""
        ref = ""
        for line in record.splitlines():
            if line.startswith("worktree "):
                path = line[len("worktree ") :].strip()
            elif line.startswith("branch "):
                ref = line[len("branch ") :].strip()
        if path and ref == target_ref:
            return Path(path)
    return None


def _ensure_clean_worktree(target: Path, branch: str, *, hooks_path: Path) -> None:
    status = _git(target, ["status", "--porcelain"], hooks_path=hooks_path).strip()
    if status:
        raise LocalBranchMergeError(
            f"target branch {branch!r} has uncommitted changes in {target}"
        )


def _remove_worktree(repo: Path, target: Path, *, hooks_path: Path) -> None:
    _run_git(
        repo,
        ["worktree", "remove", "--force", str(target)],
        hooks_path=hooks_path,
        check=False,
    )
    shutil.rmtree(target, ignore_errors=True)


def _same_path(left: Path, right: Path) -> bool:
    return _resolved_path(left) == _resolved_path(right)


def _resolved_path(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def _git(
    cwd: Path,
    args: list[str],
    *,
    input_text: str | None = None,
    extra_env: dict[str, str] | None = None,
    hooks_path: Path | None = None,
) -> str:
    return _run_git(
        cwd,
        args,
        input_text=input_text,
        extra_env=extra_env,
        hooks_path=hooks_path,
    ).stdout


def _run_git(
    cwd: Path,
    args: list[str],
    *,
    input_text: str | None = None,
    extra_env: dict[str, str] | None = None,
    hooks_path: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = ["git"]
    if hooks_path is not None:
        command.extend(["-c", f"core.hooksPath={hooks_path}"])
    command.extend(["-C", str(cwd), *args])
    try:
        result = subprocess.run(
            command,
            input=input_text,
            capture_output=True,
            check=False,
            env=_git_env(extra_env),
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LocalBranchMergeError(str(exc)) from exc
    if check and result.returncode != 0:
        stderr = result.stderr.strip()
        raise LocalBranchMergeError(
            stderr or f"git {' '.join(args)} failed with status {result.returncode}"
        )
    return result


def _git_env(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    env = {
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": environ.get("HOME", ""),
        "LANG": environ.get("LANG", "C.UTF-8"),
        "LC_ALL": environ.get("LC_ALL", "C.UTF-8"),
        "PATH": environ.get("PATH", ""),
    }
    if extra_env is not None:
        env.update(extra_env)
    return env
