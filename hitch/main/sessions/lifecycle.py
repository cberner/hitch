"""Serialize archive operations with workflow startup for one Codex thread."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from django.conf import settings

from hitch.main.models import CodexInstance, SessionMetadata, SystemWorkflow

_WORKFLOW_KINDS = (SystemWorkflow.KIND_PR_QA,)


class WorkflowStartBlockedError(RuntimeError):
    pass


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
def hold(thread_id: str, *, blocking: bool = True) -> Iterator[bool]:
    """Hold the thread lock, yielding false when a nonblocking claim loses."""
    lease = _acquire(thread_id, blocking=blocking)
    if lease is None:
        yield False
        return
    try:
        yield True
    finally:
        lease.release()


@contextlib.contextmanager
def hold_for_workflow_start(
    thread_id: str, *, lifecycle_lock_held: bool = False
) -> Iterator[None]:
    """Hold the lifecycle lock unless the caller already owns this transition."""
    if lifecycle_lock_held:
        yield
        return
    with hold(thread_id):
        yield


def archive_has_active_work(thread_id: str) -> bool:
    """Return whether archiving would strand a visible turn or workflow."""
    if CodexInstance.objects.filter(
        thread_id=thread_id,
        status__in=CodexInstance.ACTIVE_STATUSES,
    ).exists():
        return True
    return SystemWorkflow.objects.filter(
        kind__in=_WORKFLOW_KINDS,
        main_thread_id=thread_id,
        status=SystemWorkflow.STATUS_RUNNING,
    ).exists()


def ensure_workflow_start_allowed(thread_id: str, *, kind: str) -> None:
    """Reject a workflow start that conflicts with durable session state."""
    archived = SessionMetadata.objects.filter(
        thread_id=thread_id,
        codex_archived=True,
    ).exists()
    active_turn = CodexInstance.objects.filter(
        thread_id=thread_id,
        status__in=CodexInstance.ACTIVE_STATUSES,
    ).exists()
    other_workflow = (
        SystemWorkflow.objects.filter(
            kind__in=_WORKFLOW_KINDS,
            main_thread_id=thread_id,
            status=SystemWorkflow.STATUS_RUNNING,
        )
        .exclude(kind=kind)
        .exists()
    )
    if archived or active_turn or other_workflow:
        raise WorkflowStartBlockedError(
            f"session {thread_id} is archived or already running work"
        )
