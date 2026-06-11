"""systemd-scoped worker isolation: launch, cgroup limits, scope reaping.

Workers optionally run in transient systemd scopes so memory limits apply
and leaked grandchildren (sandbox processes that re-parent into their own
sessions) can be reaped through the cgroup rather than a process group.
Spawn coordination stays in ``codex_pool``; this module owns the
systemd-run launch path, the slice/limit property assembly, and the
scope/cgroup probes the interrupt and reconcile paths use.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, BinaryIO

from django.conf import settings

from hitch.main.models import CodexInstance
from hitch.main.runtime import codex_pool

logger = logging.getLogger(__name__)

_WORKER_UNIT_RE = re.compile(r"hitch-codex-worker-(\d+)\.(?:service|scope)")

def _systemd_scope_is_missing(systemctl: str, scope_unit: str) -> bool:
    try:
        result = subprocess.run(
            [systemctl, "--user", "show", scope_unit, "--property=LoadState", "--value"],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            # Bound the user-manager round trip: this probe runs on the
            # reconcile and force-kill paths, and a wedged dbus would
            # otherwise hang the scheduler tick (or a Stop click) forever.
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if result.returncode != 0:
        return False
    return result.stdout.decode("utf-8", errors="replace").strip() in {"", "not-found"}

def _scope_has_live_worker(scope_unit: str, *, proc_root: Path = Path("/proc")) -> bool:
    """Whether any live ``codex_worker`` process currently runs in ``scope_unit``.

    Worker unit names are not deployment-unique, so once our dead worker's unit
    is collected another Hitch checkout under the same user can create a unit
    with the same name. We only reap a unit whose own worker is already gone, so
    a *live* ``codex_worker`` in it means the name was reused by a different
    launch -- signaling it would kill that launch's worker and grandchildren.
    Linux-only; without ``/proc`` it reports ``False`` (the reap is then
    best-effort, as before).
    """
    if not proc_root.exists():
        return False
    target = scope_unit.encode()
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if b"codex_worker" not in cmdline.split(b"\0"):
            continue
        try:
            cgroup = (entry / "cgroup").read_bytes()
        except OSError:
            continue
        if target in cgroup:
            return True
    return False

def _worker_unit_from_pid_cgroup(
    pid: int, instance_id: int, *, proc_root: Path = Path("/proc")
) -> str | None:
    """Return the worker unit containing ``pid`` when /proc exposes it."""
    try:
        cgroup = (proc_root / str(pid) / "cgroup").read_bytes()
    except OSError:
        return None
    decoded = cgroup.decode("utf-8", errors="replace")
    for match in _WORKER_UNIT_RE.finditer(decoded):
        if match.group(1) == str(instance_id):
            return match.group(0)
    return None

def _reap_scope_cgroup(instance: CodexInstance) -> None:
    """Best-effort kill of a dead systemd worker's cgroup to clear leaked
    grandchildren.

    A worker reaped here exited without reporting completion -- wedged,
    OOM-killed, or SIGKILL'd. The codex exec sandbox runs each command in its own
    pgid/session, so a grandchild it spawned (e.g. a runaway ``cargo bench``) is
    reparented out of the worker's process group but stays in the worker's
    systemd cgroup, holding memory until the unit's last process exits. A unit only
    dies when that last process exits, so without this the grandchild can hold
    gigabytes for hours after the worker is gone. ``systemctl kill
    --kill-whom=all`` reaches every process in the cgroup; an already
    empty/collected unit is a no-op. Direct launches have no systemd cgroup
    to sweep and their already-dead pid must never be re-signaled, so they are
    skipped.

    Unit names are not deployment-unique, so a collected unit's name can be
    reused by another checkout: skip the reap if a live worker now holds the
    unit (it can't be ours -- ours is already dead) rather than killing an
    unrelated launch's worker.
    """
    if not instance.systemd_scope_unit:
        return
    if _scope_has_live_worker(instance.systemd_scope_unit):
        return
    try:
        codex_pool._force_kill_instance(instance)
    except ProcessLookupError:
        return
    except OSError:
        codex_pool.logger.warning(
            "failed to reap systemd cgroup %s for instance %s",
            instance.systemd_scope_unit,
            instance.pk,
        )

def _systemd_run_for_isolation(requested_isolation: str) -> str | None:
    systemd_run = shutil.which("systemd-run")
    if systemd_run is None:
        if requested_isolation == codex_pool._WORKER_ISOLATION_SYSTEMD:
            raise RuntimeError("systemd-run is required for Codex worker isolation")
        return None
    if requested_isolation == codex_pool._WORKER_ISOLATION_AUTO and not _systemd_user_manager_available():
        return None
    return systemd_run

def _systemd_user_manager_available() -> bool:
    systemctl = shutil.which("systemctl")
    if systemctl is None:
        return False
    try:
        result = subprocess.run(
            [systemctl, "--user", "show-environment"],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=0.5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0

def _launch_systemd_worker(
    *,
    systemd_run: str,
    scope_unit: str,
    worker_argv: list[str],
    env: dict[str, str],
    stderr: Any = subprocess.DEVNULL,
    stderr_capture: BinaryIO | None = None,
) -> codex_pool.WorkerLaunch:
    codex_pool._ensure_systemd_worker_slice()
    if stderr_capture is not None:
        stderr_offset = stderr_capture.tell()
        proc = codex_pool._popen_detached(
            codex_pool._systemd_scope_argv(
                systemd_run=systemd_run,
                scope_unit=scope_unit,
                worker_argv=worker_argv,
                env=env,
                stderr_log_path=_stderr_log_path(stderr_capture),
            ),
            env=env,
            stderr=stderr,
        )
        client_exited = codex_pool._check_systemd_run_start_result(
            proc,
            scope_unit,
            stderr_capture,
            stderr_offset=stderr_offset,
        )
        return codex_pool.WorkerLaunch(
            pid=0,
            proc=None if client_exited else proc,
            scope_unit=scope_unit,
        )
    with tempfile.TemporaryFile() as stderr_file:
        proc = codex_pool._popen_detached(
            codex_pool._systemd_scope_argv(
                systemd_run=systemd_run,
                scope_unit=scope_unit,
                worker_argv=worker_argv,
                env=env,
            ),
            env=env,
            stderr=stderr_file,
        )
        client_exited = codex_pool._check_systemd_run_start_result(proc, scope_unit, stderr_file)
    return codex_pool.WorkerLaunch(
        pid=0,
        proc=None if client_exited else proc,
        scope_unit=scope_unit,
    )

def _stderr_log_path(stderr_capture: BinaryIO) -> str | None:
    name = getattr(stderr_capture, "name", None)
    return name if isinstance(name, str) else None

def _worker_slice() -> str:
    return str(getattr(settings, "CODEX_WORKER_SLICE", "") or "").strip()

def _is_finite_limit(value: str) -> bool:
    """Whether a cgroup limit value actually bounds the unit.

    systemd treats an empty value as "unset" and ``infinity`` as "no limit", so
    neither is a real ceiling the swap cap can make "true".
    """
    return bool(value) and value.strip().lower() != "infinity"

def _limit_property(name: str, value: str, *, declarative: bool) -> str | None:
    """Render a single cgroup limit property, or ``None`` to omit it.

    A configured value is rendered as-is. A cleared value is omitted by default
    — fine for a fresh transient scope — but in ``declarative`` mode it renders
    as ``infinity`` instead. The slice is configured with the *stateful*
    ``systemctl set-property --runtime``, which only changes the properties it
    is handed, so omitting a cleared cap would leave a previously-applied value
    lingering on the runtime unit; emitting ``infinity`` resets it to unlimited.
    """
    if value:
        return f"{name}={value}"
    if declarative:
        return f"{name}=infinity"
    return None

def _memory_cgroup_properties(
    high_setting: str, max_setting: str, swap_setting: str, *, declarative: bool = False
) -> list[str]:
    """Build systemd memory-cgroup properties for the named settings.

    ``MemoryAccounting=yes`` must accompany any ``MemoryHigh``/``MemoryMax``:
    hosts with ``DefaultMemoryAccounting=no`` (or a legacy cgroup v1 hierarchy)
    silently ignore the limits unless accounting is explicitly enabled on the
    unit, so the cap would not actually bound the worker.

    ``MemorySwapMax`` rides along with the *hard* ``MemoryMax`` because cgroup
    v2 counts only RAM toward ``MemoryMax``: without a swap cap a runaway worker
    is reclaimed to swap instead of OOM-killed, so the hard cap never fires and
    the turn thrashes the host indefinitely rather than failing. It is gated on
    a *finite* ``MemoryMax`` rather than any limit: ``MemoryHigh`` is a soft
    throttle that usage may exceed (graceful degradation, no OOM) and
    ``MemoryMax=infinity`` is no limit at all, so neither gives the swap cap a
    hard ceiling to make "true" — capping swap there would silently deny swap
    to a config that deliberately has no OOM ceiling.

    ``declarative`` makes cleared caps render as ``infinity`` rather than being
    omitted, so a stateful ``set-property`` target (the slice) fully resets to
    the configured state instead of inheriting stale runtime values. The
    per-worker unit and the aggregate slice share this builder so their
    accounting/limit/swap handling cannot drift apart.
    """
    high = str(getattr(settings, high_setting, "") or "").strip()
    hard = str(getattr(settings, max_setting, "") or "").strip()
    swap = str(getattr(settings, swap_setting, "") or "").strip()
    # Swap is only a real cap below a finite hard ceiling; otherwise it is
    # unset (and, in declarative mode, reset to unlimited rather than left at a
    # stale value).
    swap_value = swap if _is_finite_limit(hard) else ""
    properties: list[str] = []
    if _is_finite_limit(high) or _is_finite_limit(hard):
        properties.append("MemoryAccounting=yes")
    for prop in (
        _limit_property("MemoryHigh", high, declarative=declarative),
        _limit_property("MemoryMax", hard, declarative=declarative),
        _limit_property("MemorySwapMax", swap_value, declarative=declarative),
    ):
        if prop is not None:
            properties.append(prop)
    return properties

def _systemd_worker_slice_properties() -> list[str]:
    # The slice is configured with the stateful ``set-property --runtime``, so
    # build it declaratively: a cleared cap resets to ``infinity`` rather than
    # leaving a stale runtime value (and misleading the hierarchy warning).
    return _memory_cgroup_properties(
        "CODEX_WORKER_SLICE_MEMORY_HIGH",
        "CODEX_WORKER_SLICE_MEMORY_MAX",
        "CODEX_WORKER_SLICE_MEMORY_SWAP_MAX",
        declarative=True,
    )

_CPU_WEIGHT_MIN = 1

_CPU_WEIGHT_MAX = 10000
