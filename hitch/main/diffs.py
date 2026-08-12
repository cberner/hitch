"""Build a highlighted worktree diff for the session page."""

import difflib
import html
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, cast

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexer import Lexer
from pygments.lexers import TextLexer, find_lexer_class_for_filename

from .git_support import GitCommandError, run_git

DiffLineKind = Literal["add", "remove", "context", "hunk", "meta"]

_GIT_TIMEOUT_SECONDS = 3
_MAX_DIFF_CHARS = 500_000
# Highlighting and table markup can expand raw diff lines several-fold.
_MAX_DIFF_PREVIEW_LINES = 1_000
_MAX_UNTRACKED_FILES = 25
_MAX_UNTRACKED_FILE_BYTES = 200_000
_HUNK_RE = re.compile(r"^@@ -(?P<old>\d+)(?:,\d+)? \+(?P<new>\d+)(?:,\d+)? @@")
_FORMATTER = HtmlFormatter(nowrap=True)
_BRANCH_DIFF_DEFAULT_REF = "refs/remotes/origin/HEAD"
_BRANCH_DIFF_FALLBACK_REFS = (
    "refs/remotes/origin/main",
    "refs/remotes/origin/master",
)
_DIFF_ARGS = [
    "diff",
    "--no-color",
    "--no-ext-diff",
    "--no-textconv",
    "--ignore-submodules=none",
    "--submodule=short",
    "--find-renames",
    "--src-prefix=a/",
    "--dst-prefix=b/",
]
_GIT_PATH_ESCAPES = {
    "a": b"\a",
    "b": b"\b",
    "f": b"\f",
    "n": b"\n",
    "r": b"\r",
    "t": b"\t",
    "v": b"\v",
    "\\": b"\\",
    '"': b'"',
}


@dataclass(frozen=True)
class DiffLine:
    kind: DiffLineKind
    old_lineno: int | None
    new_lineno: int | None
    html: str


@dataclass(frozen=True)
class DiffFile:
    path: str
    old_path: str | None
    status: str
    additions: int
    deletions: int
    lines: list[DiffLine]


@dataclass(frozen=True)
class DiffView:
    files: list[DiffFile]
    additions: int = 0
    deletions: int = 0
    truncated: bool = False
    error: str = ""

    @property
    def has_changes(self) -> bool:
        return bool(self.files)

    @property
    def file_count(self) -> int:
        return len(self.files)


class IncompleteDiffError(Exception):
    """Raised when a complete reviewer diff cannot be constructed."""


@dataclass(frozen=True)
class _GitlinkDiffMetadata:
    has_same_commit_dirty_gitlink: bool
    renamed_destinations: frozenset[str]


def build_worktree_diff(cwd: str | None) -> DiffView:
    """Return the current git session diff for ``cwd``.

    The viewer is informational, so git failures degrade to an empty/error
    state instead of blocking the session page render.
    """
    text = _worktree_diff_text(cwd)
    truncated = len(text) > _MAX_DIFF_CHARS
    if truncated:
        text = text[:_MAX_DIFF_CHARS]
    lines = _split_diff_lines(text)
    if len(lines) > _MAX_DIFF_PREVIEW_LINES:
        lines = lines[:_MAX_DIFF_PREVIEW_LINES]
        truncated = True
    text = "\n".join(lines)
    return _parse_unified_diff(text, truncated=truncated)


def build_worktree_diff_text(cwd: str | None) -> str:
    """Return a complete reviewer diff, or raise when it cannot be represented."""
    text = _strict_worktree_diff_text(cwd)
    for line in text.split("\n"):
        if line.startswith("Binary files ") and line.endswith(" differ"):
            raise IncompleteDiffError(
                "tracked binary changes cannot be represented in the reviewer diff"
            )
    if len(text) > _MAX_DIFF_CHARS:
        raise IncompleteDiffError("worktree diff exceeds the reviewer size limit")
    return text


def _has_dirty_gitlink_section(
    text: str, renamed_gitlink_destinations: frozenset[str]
) -> bool:
    section_is_gitlink = False
    section_rename_destination = ""
    section_has_dirty_marker = False
    for line in text.split("\n"):
        if line.startswith("diff --git "):
            if (
                section_is_gitlink
                or section_rename_destination in renamed_gitlink_destinations
            ) and section_has_dirty_marker:
                return True
            section_is_gitlink = False
            section_rename_destination = ""
            section_has_dirty_marker = False
            continue
        if (
            line in {
                "new file mode 160000",
                "deleted file mode 160000",
                "old mode 160000",
                "new mode 160000",
            }
            or line.startswith("index ")
            and line.endswith(" 160000")
        ):
            section_is_gitlink = True
        elif line.startswith("rename to "):
            section_rename_destination = _decode_git_path(
                line.removeprefix("rename to ")
            )
        elif line.startswith(
            ("+Subproject commit ", "-Subproject commit ")
        ) and line.endswith("-dirty"):
            section_has_dirty_marker = True
    return (
        section_is_gitlink
        or section_rename_destination in renamed_gitlink_destinations
    ) and section_has_dirty_marker


def _worktree_diff_text(cwd: str | None) -> str:
    if not cwd:
        return ""
    repo = _repo_root(Path(cwd))
    if repo is None:
        return ""

    return "\n".join(
        part
        for part in (
            _tracked_diff(repo),
            _untracked_diff(repo),
        )
        if part
    )


def _tracked_diff(repo: Path) -> str:
    diff_base = _branch_diff_base_ref(repo)
    if diff_base is not None:
        raw_diff = _git_output(repo, [*_DIFF_ARGS, diff_base, "--"])
        if raw_diff is not None:
            return raw_diff
    raw_diff = _git_output(repo, [*_DIFF_ARGS, "HEAD", "--"])
    if raw_diff is not None:
        return raw_diff
    return _git_output(repo, [*_DIFF_ARGS, "--"]) or ""


def _strict_worktree_diff_text(cwd: str | None) -> str:
    if not cwd:
        raise IncompleteDiffError("reviewed checkout path is unavailable")
    checkout = Path(cwd)
    if not checkout.is_dir():
        raise IncompleteDiffError("reviewed checkout path is unavailable")
    root_result = _strict_git_result(checkout, ["rev-parse", "--show-toplevel"])
    try:
        root_text = root_result.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IncompleteDiffError("reviewed repository path is not UTF-8") from exc
    root = root_text.strip()
    if not root:
        raise IncompleteDiffError("git could not identify the reviewed repository")
    repo = Path(root)
    parts = (_strict_tracked_diff(repo), _strict_untracked_diff(repo))
    return "\n".join(part for part in parts if part)


def _strict_tracked_diff(repo: Path) -> str:
    diff_base = _strict_branch_diff_base_ref(repo)
    result = _strict_git_result(
        repo, ["-c", "core.quotePath=true", *_DIFF_ARGS, diff_base, "--"]
    )
    if b"\0" in result.stdout:
        raise IncompleteDiffError(
            "tracked NUL-bearing changes cannot be represented in the reviewer diff"
        )
    try:
        text = result.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IncompleteDiffError(
            "tracked non-UTF-8 changes cannot be represented in the reviewer diff"
        ) from exc
    gitlinks = _strict_gitlink_diff_metadata(repo, diff_base)
    if gitlinks.has_same_commit_dirty_gitlink or _has_dirty_gitlink_section(
        text, gitlinks.renamed_destinations
    ):
        raise IncompleteDiffError(
            "dirty submodule changes cannot be represented in the reviewer diff"
        )
    return text


def _strict_gitlink_diff_metadata(
    repo: Path, diff_base: str
) -> _GitlinkDiffMetadata:
    # Raw output identifies same-commit dirtiness by equal OIDs and gives
    # renamed gitlinks unambiguous NUL-delimited destination paths. Formatted
    # ``diff --git`` headers alone cannot safely split paths containing `` b/``.
    result = _strict_git_result(
        repo,
        [
            "diff",
            "--raw",
            "-z",
            "--no-ext-diff",
            "--no-textconv",
            "--ignore-submodules=none",
            "--find-renames",
            diff_base,
            "--",
        ],
    )
    fields = result.stdout.split(b"\0")
    if fields and fields[-1] == b"":
        fields.pop()
    index = 0
    has_same_commit_dirty = False
    renamed_destinations: set[str] = set()
    while index < len(fields):
        metadata = fields[index].split()
        index += 1
        if len(metadata) != 5 or not metadata[0].startswith(b":"):
            raise IncompleteDiffError("git returned malformed raw reviewer diff metadata")
        old_mode = metadata[0][1:]
        new_mode, old_oid, new_oid, status = metadata[1:]
        if index >= len(fields):
            raise IncompleteDiffError("git returned malformed raw reviewer diff metadata")
        index += 1  # source path
        destination_path: bytes | None = None
        if status[:1] in {b"R", b"C"}:
            if index >= len(fields):
                raise IncompleteDiffError(
                    "git returned malformed raw reviewer diff metadata"
                )
            destination_path = fields[index]
            index += 1
        if (
            old_mode == b"160000"
            and new_mode == b"160000"
            and old_oid == new_oid
            and status == b"M"
        ):
            has_same_commit_dirty = True
        if (
            old_mode == b"160000"
            and new_mode == b"160000"
            and status.startswith(b"R")
            and destination_path is not None
        ):
            try:
                renamed_destinations.add(destination_path.decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise IncompleteDiffError(
                    "renamed non-UTF-8 gitlink path cannot be represented in the reviewer diff"
                ) from exc
    return _GitlinkDiffMetadata(
        has_same_commit_dirty_gitlink=has_same_commit_dirty,
        renamed_destinations=frozenset(renamed_destinations),
    )


def _strict_branch_diff_base_ref(repo: Path) -> str:
    head_exists = _strict_ref_exists(repo, "HEAD")
    if not head_exists:
        return _strict_empty_tree_hash(repo)

    saw_base_ref = False
    saw_no_common_ancestor = False
    closest_merge_base = ""
    closest_distance: int | None = None
    for index, base_ref in enumerate(
        (_BRANCH_DIFF_DEFAULT_REF, *_BRANCH_DIFF_FALLBACK_REFS)
    ):
        if not _strict_ref_exists(repo, base_ref):
            continue
        saw_base_ref = True
        result = _strict_git_result(
            repo,
            ["merge-base", "HEAD", base_ref],
            allow_statuses={0, 1},
        )
        if result.returncode == 1:
            saw_no_common_ancestor = True
            continue
        merge_base = _strict_ascii_output(result, "merge base")
        if not merge_base:
            raise IncompleteDiffError("git returned an empty merge base")
        if index == 0:
            return merge_base
        distance_result = _strict_git_result(
            repo, ["rev-list", "--count", f"{merge_base}..HEAD"]
        )
        distance_text = _strict_ascii_output(distance_result, "commit distance")
        try:
            distance = int(distance_text)
        except ValueError as exc:
            raise IncompleteDiffError("git returned an invalid commit distance") from exc
        if closest_distance is None or distance < closest_distance:
            closest_merge_base = merge_base
            closest_distance = distance
    if closest_merge_base:
        return closest_merge_base
    if saw_base_ref and saw_no_common_ancestor:
        shallow_result = _strict_git_result(repo, ["rev-parse", "--is-shallow-repository"])
        shallow = _strict_ascii_output(shallow_result, "shallow repository state")
        if shallow == "true":
            raise IncompleteDiffError(
                "git cannot determine the reviewer baseline in this shallow repository"
            )
        if shallow != "false":
            raise IncompleteDiffError("git returned an invalid shallow repository state")
        return _strict_empty_tree_hash(repo)
    return "HEAD"


def _strict_ref_exists(repo: Path, ref: str) -> bool:
    result = _strict_git_result(
        repo,
        ["rev-parse", "--verify", "--quiet", ref],
        allow_statuses={0, 1},
    )
    return result.returncode == 0


def _strict_empty_tree_hash(repo: Path) -> str:
    result = _strict_git_result(repo, ["hash-object", "-t", "tree", "/dev/null"])
    value = _strict_ascii_output(result, "empty tree hash")
    if not value:
        raise IncompleteDiffError("git returned an empty tree hash")
    return value


def _branch_diff_base_ref(repo: Path) -> str | None:
    # origin/HEAD is authoritative: if it shares history with HEAD, diff against
    # that merge-base outright. Otherwise fall back to the closest merge-base
    # among the well-known remote default branches.
    fallback_ref = None
    saw_no_common_ancestor = False
    closest_merge_base = None
    closest_distance = None
    for index, base_ref in enumerate(
        (_BRANCH_DIFF_DEFAULT_REF, *_BRANCH_DIFF_FALLBACK_REFS)
    ):
        if not _ref_exists(repo, base_ref):
            continue
        if fallback_ref is None:
            fallback_ref = base_ref
        # Allow status 1 (no common ancestor) so it stays distinguishable from
        # an execution failure (timeout / lock), which returns None.
        merge_base = _git_output(
            repo, ["merge-base", "HEAD", base_ref], allow_statuses={0, 1}
        )
        if merge_base is None:
            continue
        merge_base = merge_base.strip()
        if not merge_base:
            saw_no_common_ancestor = True
            continue
        if index == 0:
            return merge_base
        distance = _commit_distance_from_head(repo, merge_base)
        if distance is None:
            continue
        if closest_distance is None or distance < closest_distance:
            closest_merge_base = merge_base
            closest_distance = distance
    if closest_merge_base is not None:
        return closest_merge_base
    # git found no common ancestor between HEAD and a base ref. In a complete
    # repository that means a genuinely disjoint history (orphan branch / origin
    # re-pointed at an unrelated repo): diff against the empty tree so the
    # branch's content shows as additions rather than the unrelated ref's files
    # as spurious deletions. In a shallow clone the shared ancestor may simply
    # be unfetched, so keep diffing against the ref directly; likewise fall back
    # to the ref when merge-base could not be computed at all.
    if saw_no_common_ancestor and not _is_shallow_repo(repo):
        empty_tree = _empty_tree_hash(repo)
        if empty_tree is not None:
            return empty_tree
    return fallback_ref


def _ref_exists(repo: Path, ref: str) -> bool:
    output = _git_output(repo, ["rev-parse", "--verify", "--quiet", ref])
    return bool(output and output.strip())


def _is_shallow_repo(repo: Path) -> bool:
    output = _git_output(repo, ["rev-parse", "--is-shallow-repository"])
    return output is not None and output.strip() == "true"


def _empty_tree_hash(repo: Path) -> str | None:
    # Compute the empty-tree object id for the repo's object format (sha1 vs
    # sha256) instead of hard-coding the sha1 value, which is not a valid object
    # name in a sha256 repository.
    output = _git_output(repo, ["hash-object", "-t", "tree", "/dev/null"])
    if output is None:
        return None
    value = output.strip()
    return value or None


def _commit_distance_from_head(repo: Path, commit: str) -> int | None:
    output = _git_output(repo, ["rev-list", "--count", f"{commit}..HEAD"])
    if output is None:
        return None
    try:
        return int(output.strip())
    except ValueError:
        return None


def _repo_root(cwd: Path) -> Path | None:
    if not cwd.exists():
        return None
    output = _git_output(cwd, ["rev-parse", "--show-toplevel"])
    if output is None:
        return None
    root = output.strip()
    return Path(root) if root else None


def _git_output(cwd: Path, args: list[str], *, allow_statuses: set[int] | None = None) -> str | None:
    statuses = allow_statuses or {0}
    try:
        result = run_git(cwd, args, timeout=_GIT_TIMEOUT_SECONDS)
    except GitCommandError:
        return None
    if result.returncode not in statuses:
        return None
    return result.stdout.decode("utf-8", errors="replace")


def _strict_git_result(
    cwd: Path,
    args: list[str],
    *,
    allow_statuses: set[int] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    statuses = allow_statuses or {0}
    try:
        result = run_git(cwd, args, timeout=_GIT_TIMEOUT_SECONDS)
    except GitCommandError as exc:
        raise IncompleteDiffError(f"git could not run {args[0]}") from exc
    if result.returncode not in statuses:
        raise IncompleteDiffError(
            f"git {args[0]} failed while building the reviewer diff"
        )
    return result


def _strict_ascii_output(
    result: subprocess.CompletedProcess[bytes], description: str
) -> str:
    stdout = result.stdout
    try:
        value = stdout.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise IncompleteDiffError(f"git returned an invalid {description}") from exc
    return value


def _untracked_diff(repo: Path) -> str:
    raw = _git_output(repo, ["ls-files", "--others", "--exclude-standard", "-z"])
    if not raw:
        return ""
    pieces: list[str] = []
    relpaths = [p for p in raw.split("\0") if p]
    for relpath in relpaths[:_MAX_UNTRACKED_FILES]:
        piece = _synthetic_new_file_diff(repo, relpath)
        if piece:
            pieces.append(piece)
    if len(relpaths) > _MAX_UNTRACKED_FILES:
        pieces.append(
            "diff --git a/.hitch-diff-limit b/.hitch-diff-limit\n"
            "--- a/.hitch-diff-limit\n"
            "+++ b/.hitch-diff-limit\n"
            "@@ -1 +1 @@\n"
            f"+{len(relpaths) - _MAX_UNTRACKED_FILES} untracked files omitted from diff preview"
        )
    return "\n".join(pieces)


def _strict_untracked_diff(repo: Path) -> str:
    result = _strict_git_result(
        repo, ["ls-files", "--others", "--exclude-standard", "-z"]
    )
    raw_paths = [path for path in result.stdout.split(b"\0") if path]
    if len(raw_paths) > _MAX_UNTRACKED_FILES:
        raise IncompleteDiffError(
            f"worktree has more than {_MAX_UNTRACKED_FILES} untracked files"
        )
    relpaths: list[str] = []
    for raw_path in raw_paths:
        try:
            relpath = raw_path.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise IncompleteDiffError(
                "untracked non-UTF-8 path cannot be represented in the reviewer diff"
            ) from exc
        if "\n" in relpath or "\r" in relpath or "\t" in relpath:
            raise IncompleteDiffError(
                f"untracked path cannot be represented in the reviewer diff: {relpath!r}"
            )
        relpaths.append(relpath)
    if not relpaths:
        return ""
    file_mode_enabled = _strict_core_file_mode(repo)
    return "\n".join(
        _strict_synthetic_new_file_diff(repo, relpath, file_mode_enabled)
        for relpath in relpaths
    )


def _strict_core_file_mode(repo: Path) -> bool:
    result = _strict_git_result(
        repo,
        ["config", "--type=bool", "--get", "core.fileMode"],
        allow_statuses={0, 1},
    )
    if result.returncode == 1:
        return True
    value = _strict_ascii_output(result, "core.fileMode value")
    if value not in {"true", "false"}:
        raise IncompleteDiffError("git returned an invalid core.fileMode value")
    return value == "true"


def _synthetic_new_file_diff(repo: Path, relpath: str) -> str:
    if "\n" in relpath or "\r" in relpath:
        return ""
    path = repo / relpath
    if path.is_symlink():
        return _synthetic_notice_diff(relpath, "Symlink not shown")
    if not path.is_file():
        return ""
    try:
        with path.open("rb") as fh:
            data = fh.read(_MAX_UNTRACKED_FILE_BYTES + 1)
    except OSError:
        return ""
    if b"\0" in data:
        return (
            f"diff --git a/{relpath} b/{relpath}\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            f"+++ b/{relpath}\n"
            "@@ -0,0 +1 @@\n"
            "+Binary file not shown"
        )
    clipped = len(data) > _MAX_UNTRACKED_FILE_BYTES
    text = data[:_MAX_UNTRACKED_FILE_BYTES].decode("utf-8", errors="replace")
    # Split on ``\n`` only (see ``_split_diff_lines``): ``str.splitlines`` would
    # break on embedded form feeds / Unicode separators and synthesise extra
    # ``+`` rows plus a wrong ``@@ -0,0 +1,N @@`` line count for files that
    # contain them.
    lines = _split_diff_lines(text)
    # The split drops line terminators, so capture the trailing-newline status
    # here to emit the git-style ``\ No newline at end of file`` marker that
    # ``_parse_unified_diff`` surfaces as a meta row and that reviewer agents
    # look for. Only ``\n`` counts as a git line terminator, so a file ending in
    # a bare ``\r`` (classic-Mac) correctly reads as having no trailing newline.
    # When the file is clipped the marker would describe the synthetic ``File
    # preview truncated`` line rather than the real file, so suppress it.
    file_ends_with_newline = text.endswith("\n")
    if clipped:
        lines.append("File preview truncated")
    body = list(
        difflib.unified_diff(
            [],
            lines,
            fromfile="/dev/null",
            tofile=f"b/{relpath}",
            lineterm="",
        )
    )
    if lines and not clipped and not file_ends_with_newline:
        body.append("\\ No newline at end of file")
    return "\n".join([f"diff --git a/{relpath} b/{relpath}", "new file mode 100644", *body])


def _strict_synthetic_new_file_diff(
    repo: Path, relpath: str, file_mode_enabled: bool
) -> str:
    path = repo / relpath
    if path.is_symlink():
        raise IncompleteDiffError(
            f"untracked symbolic link cannot be represented in the reviewer diff: {relpath}"
        )
    try:
        with path.open("rb") as fh:
            file_stat = os.fstat(fh.fileno())
            data = fh.read(_MAX_UNTRACKED_FILE_BYTES + 1)
    except OSError as exc:
        raise IncompleteDiffError(
            f"untracked file could not be read for the reviewer diff: {relpath}"
        ) from exc
    if not stat.S_ISREG(file_stat.st_mode):
        raise IncompleteDiffError(
            f"untracked path is not a regular file: {relpath}"
        )
    if len(data) > _MAX_UNTRACKED_FILE_BYTES:
        raise IncompleteDiffError(
            f"untracked file exceeds the reviewer size limit: {relpath}"
        )
    if b"\0" in data:
        raise IncompleteDiffError(
            f"untracked binary file cannot be represented in the reviewer diff: {relpath}"
        )
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IncompleteDiffError(
            f"untracked non-UTF-8 file cannot be represented in the reviewer diff: {relpath}"
        ) from exc
    lines = _split_file_lines(text)
    body = list(
        difflib.unified_diff(
            [],
            lines,
            fromfile="/dev/null",
            tofile=f"b/{relpath}",
            lineterm="",
        )
    )
    if lines and not text.endswith("\n"):
        body.append("\\ No newline at end of file")
    executable = file_mode_enabled and bool(file_stat.st_mode & stat.S_IXUSR)
    mode = "100755" if executable else "100644"
    return "\n".join(
        [f"diff --git a/{relpath} b/{relpath}", f"new file mode {mode}", *body]
    ) + "\n"


def _split_file_lines(text: str) -> list[str]:
    """Split file content on LF while preserving every preceding CR byte."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _synthetic_notice_diff(relpath: str, message: str) -> str:
    return (
        f"diff --git a/{relpath} b/{relpath}\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        f"+++ b/{relpath}\n"
        "@@ -0,0 +1 @@\n"
        f"+{message}"
    )


def _split_diff_lines(text: str) -> list[str]:
    """Split git diff / file text into lines on ``\\n`` only.

    git frames both its diff output and file content on ``\\n``; every other
    byte -- including form feed (``\\f``), vertical tab (``\\v``), the ASCII
    file/group/record separators (``\\x1c``-``\\x1e``), and the Unicode line
    (``\\u2028``)/paragraph (``\\u2029``)/NEL (``\\x85``) separators -- is
    ordinary line *content*. ``str.splitlines`` breaks on all of those, which
    would tear a single diff line in two: the tail is reparsed as its own row
    (silently truncating the real content, emitting a phantom meta line, and --
    when the tail happens to start with ``+``/``-``/`` `` -- inventing a
    phantom add/remove that drifts every following line number and the
    add/delete totals). A trailing ``\\r`` is dropped so a CRLF diff renders
    identically to an LF one.
    """
    lines = text.split("\n")
    if lines and lines[-1] == "":
        # ``text`` is newline-terminated; the split's trailing empty element is
        # not a real line.
        lines.pop()
    return [line[:-1] if line.endswith("\r") else line for line in lines]


def _parse_unified_diff(text: str, *, truncated: bool = False) -> DiffView:
    files: list[DiffFile] = []
    current: _MutableDiffFile | None = None
    old_lineno: int | None = None
    new_lineno: int | None = None

    for raw_line in _split_diff_lines(text):
        if raw_line.startswith("diff --git "):
            if current is not None:
                files.append(current.freeze())
            current = _MutableDiffFile(path=_path_from_diff_git(raw_line))
            old_lineno = None
            new_lineno = None
            continue
        if current is None:
            continue
        before_first_hunk = old_lineno is None and new_lineno is None
        if before_first_hunk and raw_line.startswith("--- "):
            current.old_path = _clean_diff_path(raw_line[4:])
            current.add_meta(raw_line)
            continue
        if before_first_hunk and raw_line.startswith("+++ "):
            new_path = _clean_diff_path(raw_line[4:])
            if new_path:
                current.path = new_path
            current.add_meta(raw_line)
            continue
        if raw_line.startswith("new file mode"):
            current.status = "Added"
            current.add_meta(raw_line)
            continue
        if raw_line.startswith("deleted file mode"):
            current.status = "Deleted"
            current.add_meta(raw_line)
            continue
        if raw_line.startswith("rename from "):
            current.status = "Renamed"
            # git's ``rename from``/``rename to`` headers carry the raw file
            # paths -- the ``--src-prefix``/``--dst-prefix`` tags only apply to
            # the ``diff --git`` and ``---``/``+++`` lines. Decode any C-quoted
            # escapes but do NOT strip a leading ``a/``/``b/`` segment, which
            # could be a real directory name.
            current.old_path = _decode_git_path(
                raw_line.removeprefix("rename from ").strip()
            )
            current.add_meta(raw_line)
            continue
        if raw_line.startswith("rename to "):
            current.status = "Renamed"
            new_path = _decode_git_path(raw_line.removeprefix("rename to ").strip())
            if new_path:
                current.path = new_path
            current.add_meta(raw_line)
            continue

        match = _HUNK_RE.match(raw_line)
        if match:
            old_lineno = int(match.group("old"))
            new_lineno = int(match.group("new"))
            current.add_line("hunk", None, None, html.escape(raw_line))
            continue

        if raw_line.startswith("\\"):
            current.add_meta(raw_line)
            continue
        if old_lineno is None or new_lineno is None:
            current.add_meta(raw_line)
            continue

        if not raw_line:
            # ``build_worktree_diff`` joins ``git diff`` output (which is
            # always newline-terminated) to the synthetic untracked-file
            # diff with ``"\n".join``; ``splitlines()`` yields a blank
            # string at that boundary. Unified-diff hunk content always
            # carries a ``+``/``-``/`` ``/``\`` prefix, so a truly empty
            # line is never legitimate content -- accepting it as context
            # would render a phantom blank row and bump line counters past
            # EOF, misnumbering everything that follows in the same file.
            continue
        prefix = raw_line[:1]
        if prefix == "+":
            current.add_line("add", None, new_lineno, _highlight_code(current.path, raw_line[1:]))
            current.additions += 1
            new_lineno += 1
        elif prefix == "-":
            current.add_line(
                "remove",
                old_lineno,
                None,
                _highlight_code(current.old_path or current.path, raw_line[1:]),
            )
            current.deletions += 1
            old_lineno += 1
        elif prefix == " ":
            current.add_line("context", old_lineno, new_lineno, _highlight_code(current.path, raw_line[1:]))
            old_lineno += 1
            new_lineno += 1
        else:
            current.add_meta(raw_line)

    if current is not None:
        files.append(current.freeze())
    return DiffView(
        files=files,
        additions=sum(f.additions for f in files),
        deletions=sum(f.deletions for f in files),
        truncated=truncated,
    )


@dataclass
class _MutableDiffFile:
    path: str
    old_path: str | None = None
    status: str = "Modified"
    additions: int = 0
    deletions: int = 0
    lines: list[DiffLine] | None = None

    def add_line(self, kind: DiffLineKind, old_lineno: int | None, new_lineno: int | None, html_text: str) -> None:
        if self.lines is None:
            self.lines = []
        self.lines.append(DiffLine(kind=kind, old_lineno=old_lineno, new_lineno=new_lineno, html=html_text))

    def add_meta(self, text: str) -> None:
        self.add_line("meta", None, None, html.escape(text))

    def freeze(self) -> DiffFile:
        status = (
            "Renamed"
            if self.old_path and self.old_path != self.path and self.status == "Modified"
            else self.status
        )
        return DiffFile(
            path=self.path,
            old_path=self.old_path,
            status=status,
            additions=self.additions,
            deletions=self.deletions,
            lines=self.lines or [],
        )


def _path_from_diff_git(line: str) -> str:
    paths = _split_diff_git_paths(line)
    for raw_path in reversed(paths):
        path = _clean_diff_path(raw_path)
        if path:
            return path
    return "changed file"


def _split_diff_git_paths(line: str) -> tuple[str, ...]:
    rest = line.removeprefix("diff --git ")
    if not rest.lstrip().startswith('"'):
        paths = _split_unquoted_diff_git_paths(rest)
        if paths:
            return paths
    first, rest = _consume_git_path(rest)
    second, rest = _consume_git_path(rest)
    if first and second:
        return (first, second)
    return tuple(path for path in (first, second) if path)


def _split_unquoted_diff_git_paths(text: str) -> tuple[str, ...]:
    candidates: list[tuple[str, str]] = []
    start = 0
    while True:
        index = text.find(" b/", start)
        if index == -1:
            break
        old_path = text[:index].strip()
        new_path = text[index + 1 :].strip()
        if old_path.startswith("a/") and new_path.startswith("b/"):
            candidates.append((old_path, new_path))
        start = index + 1

    for old_path, new_path in candidates:
        if _clean_diff_path(old_path) == _clean_diff_path(new_path):
            return old_path, new_path
    return candidates[0] if candidates else ()


def _consume_git_path(text: str) -> tuple[str, str]:
    text = text.lstrip()
    if not text:
        return "", ""
    if text[0] != '"':
        head, _separator, tail = text.partition(" ")
        return head, tail

    escaped = False
    for i, char in enumerate(text[1:], start=1):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            return text[: i + 1], text[i + 1 :]
    return text, ""


def _clean_diff_path(value: str) -> str | None:
    path = _decode_git_path(value.strip())
    if path == "/dev/null":
        return None
    if path.startswith(("a/", "b/")):
        path = path[2:]
    return path


def _decode_git_path(value: str) -> str:
    if len(value) < 2 or value[0] != '"' or value[-1] != '"':
        return value
    raw = value[1:-1]
    decoded = bytearray()
    index = 0
    while index < len(raw):
        char = raw[index]
        if char != "\\":
            decoded.extend(char.encode("utf-8"))
            index += 1
            continue

        index += 1
        if index >= len(raw):
            decoded.append(ord("\\"))
            break

        escape = raw[index]
        if escape in "01234567":
            digits = escape
            index += 1
            while index < len(raw) and len(digits) < 3 and raw[index] in "01234567":
                digits += raw[index]
                index += 1
            byte = int(digits, 8)
            if byte <= 0xFF:
                decoded.append(byte)
            else:
                decoded.extend(f"\\{digits}".encode("ascii"))
            continue

        decoded.extend(_GIT_PATH_ESCAPES.get(escape, escape.encode("utf-8")))
        index += 1
    return decoded.decode("utf-8", errors="replace")


def _highlight_code(filename: str, code: str) -> str:
    if not code:
        return "&nbsp;"
    lexer = _lexer_class_for_filename(filename)(stripnl=False, ensurenl=False)
    rendered = highlight(code, lexer, _FORMATTER).rstrip("\n")
    return rendered or "&nbsp;"


@lru_cache(maxsize=512)
def _lexer_class_for_filename(filename: str) -> type[Lexer]:
    return cast(type[Lexer], find_lexer_class_for_filename(filename) or TextLexer)
