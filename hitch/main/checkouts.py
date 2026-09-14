"""Canonical identity and leases for Hitch-managed checkouts."""

import contextlib
import os
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from django.conf import settings

from hitch.main.runtime.file_locks import lock_fd

_worktree_locks = threading.local()


@dataclass(frozen=True)
class CheckoutIdentity:
    path: Path
    relative_path: Path
    root: Path


def identity(cwd: str | Path) -> CheckoutIdentity | None:
    """Resolve aliases, including missing paths; unmanaged paths have no identity.

    Resolution errors propagate so writers cannot proceed without their lease.
    """
    if not cwd:
        return None
    path = Path(cwd).expanduser().resolve()
    base = Path(settings.HITCH_WORKTREES_DIR).expanduser().resolve()
    if not path.is_relative_to(base):
        return None
    relative = path.relative_to(base)
    # A managed checkout is <base>/<repo>/<checkout>; descendants share its lease.
    root = base.joinpath(*relative.parts[:2])
    return CheckoutIdentity(path, relative, root)


def managed_key(cwd: str | Path) -> str | None:
    """Return a canonical key for protection reads, excluding unresolvable paths."""
    try:
        checkout = identity(cwd)
    except (OSError, ValueError):
        return None
    return str(checkout.root) if checkout is not None else None


@contextlib.contextmanager
def hold_many(cwds: Iterable[str], *, blocking: bool = True) -> Iterator[bool]:
    """Acquire distinct checkout leases in canonical order after the session lease."""
    paths = {checkout.root for cwd in cwds if (checkout := identity(cwd)) is not None}
    with contextlib.ExitStack() as stack:
        for path in sorted(paths):
            if not stack.enter_context(hold(str(path), blocking=blocking)):
                yield False
                return
        yield True


@contextlib.contextmanager
def hold(cwd: str, *, blocking: bool = True, require_exists: bool = False) -> Iterator[bool]:
    """Coordinate managed-checkout removal with startup, including shared checkouts."""
    checkout = identity(cwd)
    if checkout is None:
        yield True
        return
    path = checkout.root
    held = getattr(_worktree_locks, "held", None)
    if held is None:
        held = _worktree_locks.held = set()
    # Session lifecycle callers may create a worker while holding this lease.
    if path in held:
        if require_exists and not checkout.path.is_dir():
            raise FileNotFoundError(f"session worktree no longer exists: {checkout.path}")
        yield True
        return
    # Lock the existing directory inode so disk-full recovery needs no lock files.
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    except FileNotFoundError:
        if require_exists:
            raise
        yield True
        return
    lease = lock_fd(fd, blocking=blocking)
    if lease is None:
        yield False
        return
    try:
        if require_exists and not checkout.path.is_dir():
            raise FileNotFoundError(f"session worktree no longer exists: {checkout.path}")
        held.add(path)
        try:
            yield True
        finally:
            held.remove(path)
    finally:
        lease.release()

