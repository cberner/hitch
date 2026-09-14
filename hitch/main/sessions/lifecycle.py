"""Serialize archive operations with workflow startup for one Codex thread."""

from __future__ import annotations

import contextlib
import hashlib
import os
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path

from django.conf import settings

from hitch.main import checkouts
from hitch.main.models import CodexInstance, SessionMetadata
from hitch.main.runtime.file_locks import FileLease, lock_fd

_session_locks = threading.local()


def _lock_dir() -> Path:
    database_name = str(settings.DATABASES["default"]["NAME"])
    deployment = hashlib.sha256(database_name.encode()).hexdigest()[:16]
    if getattr(settings, "TESTING", False):
        return Path(tempfile.gettempdir()) / "hitch-session-lifecycle" / (
            f"{deployment}-{os.getpid()}"
        )
    return Path(settings.HITCH_HOME_DIR) / "session_lifecycle" / deployment


def _acquire(thread_id: str, *, blocking: bool) -> FileLease | None:
    lock_dir = _lock_dir()
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_name = hashlib.sha256(thread_id.encode()).hexdigest()
    fd = os.open(
        lock_dir / f"{lock_name}.lock",
        os.O_CREAT | os.O_RDWR | os.O_CLOEXEC,
        0o600,
    )
    return lock_fd(fd, blocking=blocking)


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
        with checkouts.hold_many((cwd or "", observed_cwd), blocking=blocking) as acquired:
            yield acquired
    finally:
        held.remove(thread_id)
        lease.release()


def archive_has_active_work(thread_id: str) -> bool:
    """Return whether archiving would strand a visible turn."""
    return CodexInstance.objects.filter(
        thread_id=thread_id,
        status__in=CodexInstance.ACTIVE_STATUSES,
    ).exists()
