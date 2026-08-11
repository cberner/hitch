"""Build a highlighted worktree diff for the session page."""

import difflib
import html
import re
from dataclasses import dataclass
from functools import lru_cache
from os import fstat
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


class IncompleteDiffError(RuntimeError):
    """Raised when a system-agent review would receive an incomplete patch."""


class DiffTooLargeError(IncompleteDiffError):
    """Raised when the complete patch exceeds the system-agent review limit."""


@dataclass(frozen=True)
class _DiffText:
    text: str
    incomplete_reason: str = ""


@dataclass(frozen=True)
class _GitDiffOutput:
    text: str | None
    has_non_utf8: bool = False


@dataclass(frozen=True)
class _RepositoryObservation:
    root: Path | None
    incomplete_reason: str = ""


@dataclass(frozen=True)
class _BranchBaseObservation:
    ref: str | None
    incomplete_reason: str = ""


@dataclass(frozen=True)
class _UntrackedPathsObservation:
    paths: tuple[str, ...]
    incomplete_reason: str = ""


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
    """Return the raw current git diff text for system-agent prompts."""
    result = _worktree_diff(cwd)
    if result.incomplete_reason:
        raise IncompleteDiffError(result.incomplete_reason)
    text = result.text
    if len(text) > _MAX_DIFF_CHARS:
        raise DiffTooLargeError(
            "worktree diff exceeds the QA review limit; split the change into "
            "a smaller patch before retrying"
        )
    return text


def _worktree_diff_text(cwd: str | None) -> str:
    return _worktree_diff(cwd).text


def _worktree_diff(cwd: str | None) -> _DiffText:
    if not cwd:
        return _DiffText("", "workflow checkout path is missing")
    repository = _repo_root_observation(Path(cwd))
    if repository.root is None:
        return _DiffText("", repository.incomplete_reason)
    repo = repository.root

    tracked = _tracked_diff(repo)
    untracked = _untracked_diff(repo)
    incomplete_reason = tracked.incomplete_reason or _tracked_diff_incomplete_reason(
        tracked.text
    )
    return _DiffText(
        text="\n".join(part for part in (tracked.text, untracked.text) if part),
        incomplete_reason=incomplete_reason or untracked.incomplete_reason,
    )


def _tracked_diff_incomplete_reason(text: str) -> str:
    if "\0" in text:
        return "tracked diff contains NUL bytes that QA cannot review as text"
    for line in _split_diff_lines(text):
        if line.startswith("Binary files ") and line.endswith(" differ"):
            return "worktree diff contains a binary change that QA cannot review"
        if line.startswith("+Subproject commit ") and line.endswith("-dirty"):
            return (
                "worktree diff contains dirty submodule changes that QA cannot "
                "review from the parent patch"
            )
    return ""


def _tracked_diff(repo: Path) -> _DiffText:
    branch_base = _branch_diff_base(repo)
    diff_base = branch_base.ref
    incomplete_reason = branch_base.incomplete_reason
    if diff_base is not None:
        result = _git_diff_output(repo, [*_DIFF_ARGS, diff_base, "--"])
        completed = _completed_git_diff(result, incomplete_reason)
        if completed is not None:
            return completed
        incomplete_reason = incomplete_reason or (
            f"git diff against branch base {diff_base!r} failed; QA cannot "
            "verify the complete branch change"
        )
    else:
        head_state = _ref_state(repo, "HEAD")
        if head_state is None:
            incomplete_reason = incomplete_reason or (
                "git could not determine whether HEAD exists; QA cannot "
                "select a reliable diff baseline"
            )
    if diff_base is None and head_state is False:
        empty_tree = _empty_tree_hash(repo)
        if empty_tree is None:
            return _DiffText(
                "",
                "git could not construct an empty-tree diff for the unborn "
                "repository; QA cannot verify the complete tracked change",
            )
        result = _git_diff_output(repo, [*_DIFF_ARGS, empty_tree, "--"])
        completed = _completed_git_diff(result, incomplete_reason)
        if completed is not None:
            return completed
        incomplete_reason = (
            "git diff against the empty tree failed; QA cannot verify the "
            "complete tracked change"
        )
    result = _git_diff_output(repo, [*_DIFF_ARGS, "HEAD", "--"])
    completed = _completed_git_diff(result, incomplete_reason)
    if completed is not None:
        return completed
    if not incomplete_reason:
        incomplete_reason = (
            "git diff against HEAD failed; QA cannot verify the complete "
            "tracked change"
        )
    result = _git_diff_output(repo, [*_DIFF_ARGS, "--"])
    return _DiffText(result.text or "", incomplete_reason)


def _completed_git_diff(
    result: _GitDiffOutput, inherited_reason: str = ""
) -> _DiffText | None:
    """Keep preview text while preserving any reason QA cannot trust it."""
    if result.text is None:
        return None
    reason = inherited_reason
    if not reason and result.has_non_utf8:
        reason = (
            "tracked diff contains non-UTF-8 bytes that QA cannot review "
            "unambiguously"
        )
    return _DiffText(result.text, reason)


def _branch_diff_base(repo: Path) -> _BranchBaseObservation:
    # origin/HEAD is authoritative: if it shares history with HEAD, diff against
    # that merge-base outright. Otherwise fall back to the closest merge-base
    # among the well-known remote default branches.
    fallback_ref = None
    saw_no_common_ancestor = False
    closest_merge_base = None
    closest_distance = None
    incomplete_reason = ""
    for index, base_ref in enumerate(
        (_BRANCH_DIFF_DEFAULT_REF, *_BRANCH_DIFF_FALLBACK_REFS)
    ):
        ref_state = _ref_state(repo, base_ref)
        if ref_state is None:
            incomplete_reason = incomplete_reason or (
                f"git could not inspect candidate branch-base ref {base_ref!r}; "
                "QA cannot select a reliable diff baseline"
            )
            continue
        if not ref_state:
            continue
        if fallback_ref is None:
            fallback_ref = base_ref
        # Allow status 1 (no common ancestor) so it stays distinguishable from
        # an execution failure (timeout / lock), which returns None.
        merge_base = _git_output(
            repo, ["merge-base", "HEAD", base_ref], allow_statuses={0, 1}
        )
        if merge_base is None:
            incomplete_reason = incomplete_reason or (
                f"git merge-base failed for {base_ref!r}; QA cannot verify "
                "the branch baseline"
            )
            continue
        merge_base = merge_base.strip()
        if not merge_base:
            saw_no_common_ancestor = True
            continue
        if index == 0:
            return _BranchBaseObservation(merge_base, incomplete_reason)
        distance = _commit_distance_from_head(repo, merge_base)
        if distance is None:
            incomplete_reason = incomplete_reason or (
                f"git could not measure branch-base distance for {base_ref!r}; "
                "QA cannot verify the selected baseline"
            )
            continue
        if closest_distance is None or distance < closest_distance:
            closest_merge_base = merge_base
            closest_distance = distance
    if closest_merge_base is not None:
        return _BranchBaseObservation(closest_merge_base, incomplete_reason)
    # git found no common ancestor between HEAD and a base ref. In a complete
    # repository that means a genuinely disjoint history (orphan branch / origin
    # re-pointed at an unrelated repo): diff against the empty tree so the
    # branch's content shows as additions rather than the unrelated ref's files
    # as spurious deletions. In a shallow clone the shared ancestor may simply
    # be unfetched, so keep diffing against the ref directly; likewise fall back
    # to the ref when merge-base could not be computed at all.
    if saw_no_common_ancestor:
        shallow = _is_shallow_repo(repo)
        if shallow is None:
            incomplete_reason = incomplete_reason or (
                "git could not determine whether the repository is shallow; "
                "QA cannot verify the selected branch baseline"
            )
        elif not shallow:
            empty_tree = _empty_tree_hash(repo)
            if empty_tree is not None:
                return _BranchBaseObservation(empty_tree, incomplete_reason)
            incomplete_reason = incomplete_reason or (
                "git could not construct the empty-tree branch baseline"
            )
    return _BranchBaseObservation(fallback_ref, incomplete_reason)


def _ref_state(repo: Path, ref: str) -> bool | None:
    try:
        result = run_git(
            repo,
            ["rev-parse", "--verify", "--quiet", ref],
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except GitCommandError:
        return None
    if result.returncode == 1:
        return False
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def _is_shallow_repo(repo: Path) -> bool | None:
    output = _git_output(repo, ["rev-parse", "--is-shallow-repository"])
    if output is None:
        return None
    value = output.strip()
    if value == "true":
        return True
    if value == "false":
        return False
    return None


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


def _repo_root_observation(cwd: Path) -> _RepositoryObservation:
    if not cwd.exists():
        return _RepositoryObservation(
            None, f"workflow checkout {str(cwd)!r} does not exist"
        )
    try:
        result = run_git(
            cwd, ["rev-parse", "--show-toplevel"], timeout=_GIT_TIMEOUT_SECONDS
        )
    except GitCommandError:
        return _RepositoryObservation(
            None, "git repository discovery failed; QA cannot inspect the checkout"
        )
    if result.returncode != 0:
        return _RepositoryObservation(
            None, "workflow checkout is not a readable Git repository"
        )
    try:
        root = result.stdout.decode("utf-8").strip()
    except UnicodeDecodeError:
        return _RepositoryObservation(
            None, "Git repository path contains non-UTF-8 bytes"
        )
    if not root:
        return _RepositoryObservation(None, "Git repository root is empty")
    return _RepositoryObservation(Path(root))


def _git_output(
    cwd: Path,
    args: list[str],
    *,
    allow_statuses: set[int] | None = None,
) -> str | None:
    statuses = allow_statuses or {0}
    try:
        result = run_git(cwd, args, timeout=_GIT_TIMEOUT_SECONDS)
    except GitCommandError:
        return None
    if result.returncode not in statuses:
        return None
    return result.stdout.decode("utf-8", errors="replace")


def _git_diff_output(cwd: Path, args: list[str]) -> _GitDiffOutput:
    try:
        result = run_git(cwd, args, timeout=_GIT_TIMEOUT_SECONDS)
    except GitCommandError:
        return _GitDiffOutput(None)
    if result.returncode != 0:
        return _GitDiffOutput(None)
    try:
        return _GitDiffOutput(result.stdout.decode("utf-8"))
    except UnicodeDecodeError:
        return _GitDiffOutput(
            result.stdout.decode("utf-8", errors="replace"), has_non_utf8=True
        )


def _core_file_mode(repo: Path) -> bool | None:
    try:
        result = run_git(
            repo,
            ["config", "--type=bool", "--get", "core.fileMode"],
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except GitCommandError:
        return None
    if result.returncode == 1:
        return True
    if result.returncode != 0:
        return None
    value = result.stdout.decode("ascii", errors="replace").strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def _untracked_diff(repo: Path) -> _DiffText:
    untracked_paths = _untracked_paths(repo)
    relpaths = list(untracked_paths.paths)
    if not relpaths:
        return _DiffText("", untracked_paths.incomplete_reason)
    core_file_mode = _core_file_mode(repo)
    if core_file_mode is None:
        incomplete_reason = untracked_paths.incomplete_reason or (
            "git could not read core.fileMode; QA cannot verify untracked "
            "file modes"
        )
        core_file_mode = False
    else:
        incomplete_reason = untracked_paths.incomplete_reason
    pieces: list[str] = []
    for relpath in relpaths[:_MAX_UNTRACKED_FILES]:
        result = _synthetic_new_file_diff(
            repo, relpath, core_file_mode=core_file_mode
        )
        if result.text:
            pieces.append(result.text)
        if not incomplete_reason and result.incomplete_reason:
            incomplete_reason = result.incomplete_reason
    if len(relpaths) > _MAX_UNTRACKED_FILES:
        omitted = len(relpaths) - _MAX_UNTRACKED_FILES
        pieces.append(
            "diff --git a/.hitch-diff-limit b/.hitch-diff-limit\n"
            "--- a/.hitch-diff-limit\n"
            "+++ b/.hitch-diff-limit\n"
            "@@ -1 +1 @@\n"
            f"+{omitted} untracked files omitted from diff preview"
        )
        if not incomplete_reason:
            incomplete_reason = (
                f"worktree diff omits {omitted} untracked files; reduce the "
                "change size before retrying QA"
            )
    return _DiffText("\n".join(pieces), incomplete_reason)


def _untracked_paths(repo: Path) -> _UntrackedPathsObservation:
    try:
        result = run_git(
            repo,
            ["ls-files", "--others", "--exclude-standard", "-z"],
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except GitCommandError:
        return _UntrackedPathsObservation(
            (), "git ls-files failed; QA cannot verify all untracked files"
        )
    if result.returncode != 0:
        return _UntrackedPathsObservation(
            (), "git ls-files failed; QA cannot verify all untracked files"
        )
    paths: list[str] = []
    incomplete_reason = ""
    for raw_path in (value for value in result.stdout.split(b"\0") if value):
        try:
            paths.append(raw_path.decode("utf-8"))
        except UnicodeDecodeError:
            incomplete_reason = incomplete_reason or (
                "git reported a non-UTF-8 untracked filename; QA cannot "
                "identify every file unambiguously"
            )
    return _UntrackedPathsObservation(tuple(paths), incomplete_reason)


def _synthetic_new_file_diff(
    repo: Path, relpath: str, *, core_file_mode: bool
) -> _DiffText:
    if "\n" in relpath or "\r" in relpath:
        return _DiffText(
            "", "worktree diff contains an unreviewable untracked filename"
        )
    path = repo / relpath
    if path.is_symlink():
        return _DiffText(
            _synthetic_notice_diff(relpath, "Symlink not shown"),
            f"worktree diff omits untracked symlink {relpath!r}",
        )
    if not path.is_file():
        return _DiffText(
            "", f"worktree diff could not read untracked path {relpath!r}"
        )
    try:
        with path.open("rb") as fh:
            data = fh.read(_MAX_UNTRACKED_FILE_BYTES + 1)
            executable = core_file_mode and bool(fstat(fh.fileno()).st_mode & 0o100)
    except OSError:
        return _DiffText(
            "", f"worktree diff could not read untracked file {relpath!r}"
        )
    if b"\0" in data:
        mode = "100755" if executable else "100644"
        return _DiffText(
            (
                f"diff --git a/{relpath} b/{relpath}\n"
                f"new file mode {mode}\n"
                "--- /dev/null\n"
                f"+++ b/{relpath}\n"
                "@@ -0,0 +1 @@\n"
                "+Binary file not shown"
            ),
            f"worktree diff omits binary untracked file {relpath!r}",
        )
    clipped = len(data) > _MAX_UNTRACKED_FILE_BYTES
    raw_text = data[:_MAX_UNTRACKED_FILE_BYTES]
    try:
        text = raw_text.decode("utf-8")
        has_non_utf8 = False
    except UnicodeDecodeError:
        text = raw_text.decode("utf-8", errors="replace")
        has_non_utf8 = True
    # File bytes and Git's diff framing have different CRLF semantics. Keep a
    # file's trailing ``\r`` as hunk content; ``_split_diff_lines`` removes it
    # only when parsing Git's own CRLF-framed command output for the UI.
    lines = _split_file_lines(text)
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
    mode = "100755" if executable else "100644"
    return _DiffText(
        "\n".join(
            [f"diff --git a/{relpath} b/{relpath}", f"new file mode {mode}", *body]
        ),
        (
            (
                f"worktree diff truncates untracked file {relpath!r}; reduce "
                "the file size before retrying QA"
                if clipped
                else f"untracked file {relpath!r} contains non-UTF-8 bytes "
                "that QA cannot review unambiguously"
            )
            if clipped or has_non_utf8
            else ""
        ),
    )


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
    """Split Git diff framing into lines on ``\\n`` only.

    Git frames its diff output on ``\\n``; every other
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


def _split_file_lines(text: str) -> list[str]:
    """Split file content on LF without normalizing a preceding CR byte."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


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
