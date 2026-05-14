"""Discover local git repositories under the user's home directory."""

from pathlib import Path


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
