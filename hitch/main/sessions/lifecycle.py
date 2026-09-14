"""Serialize archive operations with workflow startup for one Codex thread."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
import tempfile
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from django.conf import settings

from hitch.main.models import CodexInstance, SessionMetadata

_worktree_locks = threading.local()
_session_locks = threading.local()


@dataclass
class _Lease:
    fd: int

    def release(self) -> None:
        if self.fd < 0:
            return
        with contextlib.suppress(OSError):
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            os.close(self.fd)
        self.fd = -1


def _lock_dir() -> Path:
    database_name = str(settings.DATABASES["default"]["NAME"])
    deployment = hashlib.sha256(database_name.encode()).hexdigest()[:16]
    if getattr(settings, "TESTING", False):
        return Path(tempfile.gettempdir()) / "hitch-session-lifecycle" / (
            f"{deployment}-{os.getpid()}"
        )
    return Path(settings.HITCH_HOME_DIR) / "session_lifecycle" / deployment


def _acquire(thread_id: str, *, blocking: bool) -> _Lease | None:
    lock_dir = _lock_dir()
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_name = hashlib.sha256(thread_id.encode()).hexdigest()
    fd = os.open(
        lock_dir / f"{lock_name}.lock",
        os.O_CREAT | os.O_RDWR | os.O_CLOEXEC,
        0o600,
    )
    return _lock_fd(fd, blocking=blocking)


def _lock_fd(fd: int, *, blocking: bool) -> _Lease | None:
    operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
    try:
        fcntl.flock(fd, operation)
    except BlockingIOError:
        os.close(fd)
        return None
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(fd)
        raise
    return _Lease(fd)


@contextlib.contextmanager
def hold(
    thread_id: str, *, blocking: bool = True, observed_cwd: str = "", allow_nested: bool = False,
) -> Iterator[bool]:
    """Hold the thread lock, yielding false when a nonblocking claim loses."""
    held = getattr(_session_locks, "held", None)
    if held is None:
        held = _session_locks.held = set()
    if allow_nested and thread_id in held:
        yield True
        return
    lease = _acquire(thread_id, blocking=blocking)
    if lease is None:
        yield False
        return
    try:
        held.add(thread_id)
        cwd = SessionMetadata.objects.filter(thread_id=thread_id).values_list("cwd", flat=True).first()
        if not cwd:
            cwd = (
                CodexInstance.objects.filter(thread_id=thread_id).order_by("-pk")
                .values_list("cwd", flat=True).first()
            )
        paths = sorted({str(Path(value).expanduser().resolve()) for value in (cwd, observed_cwd) if value})
        with contextlib.ExitStack() as stack:
            for path in paths:
                if not stack.enter_context(hold_worktree(path, blocking=blocking)):
                    yield False
                    return
            yield True
    finally:
        held.remove(thread_id)
        lease.release()


@contextlib.contextmanager
def hold_worktree(cwd: str, *, blocking: bool = True, require_exists: bool = False) -> Iterator[bool]:
    """Coordinate managed-checkout removal with startup, including shared checkouts."""
    path = Path(cwd).expanduser().resolve()
    base = Path(settings.HITCH_WORKTREES_DIR).expanduser().resolve()
    if not cwd or not path.is_relative_to(base):
        yield True
        return
    held = getattr(_worktree_locks, "held", None)
    if held is None:
        held = _worktree_locks.held = set()
    # Session lifecycle callers may create a worker while holding this lease.
    if path in held:
        if require_exists and not path.is_dir():
            raise FileNotFoundError(f"session worktree no longer exists: {path}")
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
    lease = _lock_fd(fd, blocking=blocking)
    if lease is None:
        yield False
        return
    try:
        if require_exists and not path.is_dir():
            raise FileNotFoundError(f"session worktree no longer exists: {path}")
        held.add(path)
        try:
            yield True
        finally:
            held.remove(path)
    finally:
        lease.release()


def archive_has_active_work(thread_id: str) -> bool:
    """Return whether archiving would strand a visible turn."""
    return CodexInstance.objects.filter(
        thread_id=thread_id,
        status__in=CodexInstance.ACTIVE_STATUSES,
    ).exists()
