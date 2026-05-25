"""Build a highlighted worktree diff for the session page."""

import difflib
import html
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import TextLexer, get_lexer_for_filename
from pygments.util import ClassNotFound

DiffLineKind = Literal["add", "remove", "context", "hunk", "meta"]

_GIT_TIMEOUT_SECONDS = 3
_MAX_DIFF_CHARS = 500_000
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


def build_worktree_diff(cwd: str | None) -> DiffView:
    """Return the current git session diff for ``cwd``.

    The viewer is informational, so git failures degrade to an empty/error
    state instead of blocking the session page render.
    """
    if not cwd:
        return DiffView(files=[])
    repo = _repo_root(Path(cwd))
    if repo is None:
        return DiffView(files=[])

    diff_base = _branch_diff_base_ref(repo)
    raw_diff = None
    if diff_base is not None:
        raw_diff = _git_output(repo, [*_DIFF_ARGS, diff_base, "--"])
    if raw_diff is None:
        raw_diff = _git_output(repo, [*_DIFF_ARGS, "HEAD", "--"])
    if raw_diff is None:
        raw_diff = _git_output(repo, [*_DIFF_ARGS, "--"]) or ""

    parts = [raw_diff] if raw_diff else []
    untracked_diff = _untracked_diff(repo)
    if untracked_diff:
        parts.append(untracked_diff)
    text = "\n".join(part for part in parts if part)
    truncated = len(text) > _MAX_DIFF_CHARS
    if truncated:
        text = text[:_MAX_DIFF_CHARS]
    return _parse_unified_diff(text, truncated=truncated)


def build_worktree_diff_text(cwd: str | None) -> str:
    """Return the raw current git diff text for system-agent prompts."""
    if not cwd:
        return ""
    repo = _repo_root(Path(cwd))
    if repo is None:
        return ""

    diff_base = _branch_diff_base_ref(repo)
    raw_diff = None
    if diff_base is not None:
        raw_diff = _git_output(repo, [*_DIFF_ARGS, diff_base, "--"])
    if raw_diff is None:
        raw_diff = _git_output(repo, [*_DIFF_ARGS, "HEAD", "--"])
    if raw_diff is None:
        raw_diff = _git_output(repo, [*_DIFF_ARGS, "--"]) or ""

    parts = [raw_diff] if raw_diff else []
    untracked_diff = _untracked_diff(repo)
    if untracked_diff:
        parts.append(untracked_diff)
    text = "\n".join(part for part in parts if part)
    if len(text) > _MAX_DIFF_CHARS:
        text = text[:_MAX_DIFF_CHARS] + "\n\n[diff truncated]"
    return text


def _branch_diff_base_ref(repo: Path) -> str | None:
    default_base = _merge_base_or_ref(repo, _BRANCH_DIFF_DEFAULT_REF)
    if default_base is not None:
        return default_base

    fallback_ref = None
    closest_merge_base = None
    closest_distance = None
    for base_ref in _BRANCH_DIFF_FALLBACK_REFS:
        output = _git_output(repo, ["rev-parse", "--verify", "--quiet", base_ref])
        if not output or not output.strip():
            continue
        if fallback_ref is None:
            fallback_ref = base_ref
        merge_base = _git_output(repo, ["merge-base", "HEAD", base_ref])
        if not merge_base or not merge_base.strip():
            continue
        distance = _commit_distance_from_head(repo, merge_base.strip())
        if distance is None:
            continue
        if closest_distance is None or distance < closest_distance:
            closest_merge_base = merge_base.strip()
            closest_distance = distance
    return closest_merge_base or fallback_ref


def _merge_base_or_ref(repo: Path, base_ref: str) -> str | None:
    output = _git_output(repo, ["rev-parse", "--verify", "--quiet", base_ref])
    if not output or not output.strip():
        return None
    merge_base = _git_output(repo, ["merge-base", "HEAD", base_ref])
    return merge_base.strip() if merge_base and merge_base.strip() else base_ref


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
        result = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode not in statuses:
        return None
    return result.stdout.decode("utf-8", errors="replace")


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
    lines = text.splitlines()
    if clipped:
        lines.append("File preview truncated")
    body = difflib.unified_diff(
        [],
        lines,
        fromfile="/dev/null",
        tofile=f"b/{relpath}",
        lineterm="",
    )
    return "\n".join([f"diff --git a/{relpath} b/{relpath}", "new file mode 100644", *body])


def _synthetic_notice_diff(relpath: str, message: str) -> str:
    return (
        f"diff --git a/{relpath} b/{relpath}\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        f"+++ b/{relpath}\n"
        "@@ -0,0 +1 @@\n"
        f"+{message}"
    )


def _parse_unified_diff(text: str, *, truncated: bool = False) -> DiffView:
    files: list[DiffFile] = []
    current: _MutableDiffFile | None = None
    old_lineno: int | None = None
    new_lineno: int | None = None

    for raw_line in text.splitlines():
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
    try:
        lexer = get_lexer_for_filename(filename, stripnl=False, ensurenl=False)
    except ClassNotFound:
        lexer = TextLexer(stripnl=False, ensurenl=False)
    rendered = highlight(code, lexer, _FORMATTER).rstrip("\n")
    return rendered or "&nbsp;"
