"""Host- and process-level probes for the Hitch health dashboard.

These read ``/proc`` (and the systemd worker-unit cgroups surfaced there) to
surface the operational leaks that DB rows alone cannot show:

* leaked worker grandchildren — a worker's Python died but a sandbox grandchild
  (e.g. a runaway ``cargo bench``) survives in the now-orphaned worker cgroup,
  holding gigabytes for hours (the highest-priority incident class);
* CPU saturation — load average and the count of concurrently active worker
  units against the core count;
* runserver socket/fd accumulation — open fds and CLOSE_WAIT sockets that leak
  when a streaming client aborts.

Everything is Linux/``/proc``-specific and best-effort: on a host without the
files it needs, each probe returns ``None`` / an empty result so the dashboard
renders "unavailable" rather than failing. All filesystem roots are injectable
so the probes are testable against fixture trees.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from django.utils import timezone

from hitch.main.models import CodexInstance

# Worker units are named ``hitch-codex-worker-<deployment>-<instance id>.service``.
# Legacy units omitted the deployment discriminator and legacy scope units used
# ``.scope``; support all forms so the health page still sees old workers.
_SCOPE_RE = re.compile(
    r"hitch-codex-worker-(?:[a-f0-9]{12}-)?(?P<instance_id>\d+)\.(?:service|scope)"
)
_SOCKET_INODE_RE = re.compile(r"socket:\[(\d+)\]")
# Process names expected inside a healthy worker unit. Anything else is a
# sandbox grandchild (the build/test/bench runners that leak).
_KNOWN_SCOPE_COMMS = frozenset({"python", "python3", "node", "codex"})
# A turn that reached a terminal row only moments ago may still have its
# app-server shutting down; don't flag it as leaked inside this grace window.
_LEAK_GRACE = timedelta(seconds=90)
# /proc/net/tcp connection state for CLOSE_WAIT.
_TCP_CLOSE_WAIT = "08"


@dataclass(frozen=True)
class ScopeProcess:
    pid: int
    comm: str
    cmdline: str
    rss_bytes: int
    foreign: bool


@dataclass(frozen=True)
class LeakedScope:
    instance_id: int
    scope_unit: str
    db_status: str
    processes: list[ScopeProcess]

    @property
    def total_rss_bytes(self) -> int:
        return sum(proc.rss_bytes for proc in self.processes)

    @property
    def foreign_commands(self) -> list[str]:
        return [proc.cmdline or proc.comm for proc in self.processes if proc.foreign]


@dataclass(frozen=True)
class WorkerScopeProbe:
    active_count: int
    leaked: list[LeakedScope]

    @property
    def total_leaked_rss_bytes(self) -> int:
        return sum(scope.total_rss_bytes for scope in self.leaked)


@dataclass(frozen=True)
class _ProcInfo:
    pid: int
    scope_unit: str
    scope_instance_id: int
    comm: str
    cmdline: str
    rss_bytes: int
    is_codex_worker: bool


def _read_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None


def _read_text(path: Path) -> str | None:
    data = _read_bytes(path)
    if data is None:
        return None
    return data.decode("utf-8", errors="replace")


def _rss_bytes_from_status(status: str | None) -> int:
    if not status:
        return 0
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                # VmRSS is reported in kB.
                return int(parts[1]) * 1024
            return 0
    return 0


def _scan_worker_scope_procs(proc_root: Path) -> list[_ProcInfo]:
    """Single /proc pass yielding one record per process inside a worker unit."""
    if not proc_root.exists():
        return []
    infos: list[_ProcInfo] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        cgroup = _read_bytes(entry / "cgroup")
        if cgroup is None:
            continue
        match = _SCOPE_RE.search(cgroup.decode("utf-8", errors="replace"))
        if match is None:
            continue
        cmdline_bytes = _read_bytes(entry / "cmdline") or b""
        cmdline = cmdline_bytes.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
        comm = (_read_text(entry / "comm") or "").strip()
        infos.append(
            _ProcInfo(
                pid=int(entry.name),
                scope_unit=match.group(0),
                scope_instance_id=int(match.group("instance_id")),
                comm=comm,
                cmdline=cmdline,
                rss_bytes=_rss_bytes_from_status(_read_text(entry / "status")),
                is_codex_worker=b"codex_worker" in cmdline_bytes.split(b"\0"),
            )
        )
    return infos


def probe_worker_scopes(
    *, proc_root: Path = Path("/proc"), now: datetime | None = None
) -> WorkerScopeProbe:
    """Active worker-unit count and leaked orphaned-grandchild units.

    A unit is *leaked* when its instance row is already terminal (and past the
    shutdown grace window) yet processes remain in its cgroup, and none of them
    is a live ``codex_worker`` (a live worker means the cgroup belongs to an
    active launch, not an orphaned grandchild leak).
    """
    infos = _scan_worker_scope_procs(proc_root)
    by_scope: dict[str, list[_ProcInfo]] = {}
    for info in infos:
        by_scope.setdefault(info.scope_unit, []).append(info)

    active_count = len({info.scope_unit for info in infos if info.is_codex_worker})

    instance_ids = {info.scope_instance_id for info in infos}
    terminal: dict[int, tuple[str, datetime | None]] = {
        row["id"]: (row["status"], row["ended_at"])
        for row in CodexInstance.objects.filter(
            id__in=instance_ids,
            status__in=(CodexInstance.STATUS_COMPLETED, CodexInstance.STATUS_FAILED),
        ).values("id", "status", "ended_at")
    }

    moment = now or timezone.now()
    leaked: list[LeakedScope] = []
    for scope_unit, procs in by_scope.items():
        instance_id = procs[0].scope_instance_id
        status_ended = terminal.get(instance_id)
        if status_ended is None:
            continue
        status, ended_at = status_ended
        if ended_at is not None and moment - ended_at < _LEAK_GRACE:
            continue
        if any(proc.is_codex_worker for proc in procs):
            continue
        scope_processes = [
            ScopeProcess(
                pid=proc.pid,
                comm=proc.comm,
                cmdline=proc.cmdline,
                rss_bytes=proc.rss_bytes,
                foreign=proc.comm not in _KNOWN_SCOPE_COMMS,
            )
            for proc in sorted(procs, key=lambda p: p.pid)
        ]
        leaked.append(
            LeakedScope(
                instance_id=instance_id,
                scope_unit=scope_unit,
                db_status=status,
                processes=scope_processes,
            )
        )
    leaked.sort(key=lambda scope: scope.instance_id)
    return WorkerScopeProbe(active_count=active_count, leaked=leaked)


def load_average() -> tuple[float, float, float] | None:
    try:
        return os.getloadavg()
    except (OSError, AttributeError):
        return None


def cpu_count() -> int | None:
    return os.cpu_count()


def runserver_fd_count(*, proc_root: Path = Path("/proc")) -> int | None:
    try:
        return sum(1 for _ in (proc_root / "self" / "fd").iterdir())
    except OSError:
        return None


def _our_socket_inodes(proc_root: Path) -> set[str] | None:
    fd_dir = proc_root / "self" / "fd"
    try:
        entries = list(fd_dir.iterdir())
    except OSError:
        return None
    inodes: set[str] = set()
    for entry in entries:
        try:
            target = os.readlink(entry)
        except OSError:
            continue
        match = _SOCKET_INODE_RE.fullmatch(target)
        if match is not None:
            inodes.add(match.group(1))
    return inodes


def runserver_close_wait_count(*, proc_root: Path = Path("/proc")) -> int | None:
    """CLOSE_WAIT sockets owned by the runserver process (leaked fds)."""
    inodes = _our_socket_inodes(proc_root)
    if inodes is None:
        return None
    count = 0
    for name in ("tcp", "tcp6"):
        text = _read_text(proc_root / "net" / name)
        if not text:
            continue
        for line in text.splitlines()[1:]:
            fields = line.split()
            if len(fields) < 10:
                continue
            if fields[3] != _TCP_CLOSE_WAIT:
                continue
            if fields[9] in inodes:
                count += 1
    return count
