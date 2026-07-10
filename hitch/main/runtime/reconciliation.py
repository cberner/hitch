"""Dead/orphaned worker reconciliation and app-server process hygiene.

The sweeps that keep the database and the process table agreeing: mark
rows FAILED when their worker pid is gone (with post-terminal routing to
the system agents), kill leaked workers whose row is already terminal,
and find/nuke app-server processes belonging to this deployment.
Worker spawning and liveness primitives stay in ``codex_pool``.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
from collections.abc import Iterable
from datetime import timedelta
from pathlib import Path

from django.conf import settings
from django.utils import timezone

from hitch.main.models import CodexInstance
from hitch.main.runtime import codex_pool, rate_limit, systemd_isolation, worker_errors

logger = logging.getLogger(__name__)


def _host_proc_scan_disabled_for_test_run(
    *, proc_root: Path, manage_py: str | None
) -> bool:
    """Keep Django tests from reaping real deployment workers.

    Use ``sys.argv`` rather than ``settings.TESTING`` because tests can override
    that setting while still running against a test DB. Explicit proc fixtures
    pass a ``proc_root`` or ``manage_py`` and remain enabled.
    """
    return "test" in sys.argv and manage_py is None and proc_root == Path("/proc")


def reconcile_dead() -> int:
    """Mark workers as failed whose PID is no longer alive.

    A worker that crashed before writing its terminal status leaves a row
    stuck in ``starting``/``running``. We sweep those rows and mark them
    failed so the UI doesn't show a permanently-pending turn.
    """
    codex_pool._reap_finished_workers()
    pending = CodexInstance.objects.filter(
        status__in=CodexInstance.ACTIVE_STATUSES
    )
    updated = _mark_dead_instances_failed(pending)
    _reconcile_terminal_workflow_instances()
    reconcile_orphaned_workers()
    codex_pool.retry_failed_input_image_cleanups()
    _prune_reaped_workers()
    return updated

_ORPHAN_REAP_GRACE = timedelta(seconds=60)

def _iter_running_worker_pids(
    *,
    proc_root: Path = Path("/proc"),
    manage_py: str | None = None,
) -> Iterable[tuple[int, int]]:
    """Yield ``(pid, instance_id)`` for this deployment's live ``codex_worker``
    processes.

    Matches the same ``codex_worker --instance-id <pk>`` marker
    ``_pid_is_our_worker`` uses, and *additionally* requires this deployment's
    ``manage.py`` path on the command line. Without that, a second Hitch checkout
    running under the same Unix user -- whose worker command lines carry the same
    generic ``codex_worker`` marker but whose instance ids belong to a different
    database -- would be scanned here and could be reaped as "not in our expected
    set". Linux-only (the deployment target); on a host without ``/proc`` it
    yields nothing, so the orphan reap is simply a no-op there rather than
    guessing.
    """
    if _host_proc_scan_disabled_for_test_run(proc_root=proc_root, manage_py=manage_py):
        return
    if not proc_root.exists():
        return
    marker = (manage_py or codex_pool._our_manage_py()).encode()
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        parts = cmdline.split(b"\0")
        if b"codex_worker" not in parts or marker not in parts:
            continue
        try:
            idx = parts.index(b"--instance-id")
        except ValueError:
            continue
        if idx + 1 >= len(parts):
            continue
        try:
            instance_id = int(parts[idx + 1])
            pid = int(entry.name)
        except ValueError:
            continue
        yield pid, instance_id

def _is_tracked_worker(pid: int) -> bool:
    """Whether ``pid`` is a worker this process spawned and still supervises.

    ``reconcile_dead`` calls ``_reap_finished_workers`` first, which drops every
    tracked worker that has *exited*, so a pid still tracked here is alive and
    mid-shutdown -- its tracker (``_wait_for_tracked_worker``) reaps it when it
    exits -- rather than leaked.
    """
    with codex_pool._TRACKED_WORKER_PROCS_LOCK:
        return pid in codex_pool._TRACKED_WORKER_PROCS

def reconcile_orphaned_workers() -> int:
    """Kill leaked worker processes whose instance is no longer expected to run.

    ``reconcile_dead`` reconciles the DB->process direction (rows whose worker pid
    is gone). This reconciles the reverse: a live ``codex_worker`` process whose
    CodexInstance has reached a terminal status (or no longer exists) has leaked.
    Until it exits it keeps a connection open to the shared CODEX_HOME state DB
    and contends on its single writer lock, which surfaces as "database is locked"
    for every other app-server start. Force-killing the worker's process group
    takes its app-server child down with it (the app-server inherits the worker's
    session, so ``killpg``/``systemctl kill --kill-whom=all`` reaches it).

    A still-running worker is spared only while a turn could genuinely be in
    progress or just finishing:

    * its instance is still ``starting``/``running`` (spanning *all* purposes --
      user, system-agent, and workflow turns -- so a system-session worker is
      never reaped);
    * its instance reached a terminal status within ``_ORPHAN_REAP_GRACE``:
      ``codex_worker`` commits the terminal status *before* running
      ``_notify_system_agents`` (which can spawn follow-up turns) and input-image
      cleanup, so a terminal-but-live worker inside that window is finishing
      hooks, not leaked. *Past* the grace a still-live terminal worker is wedged
      and is reaped even if this process spawned it -- the tracked check only
      guards the thin race before ``ended_at`` is recorded, so a same-process
      hung worker holding the lock is never exempted indefinitely.

    We only ever kill on a positive DB answer: if the rows cannot be read,
    nothing is killed.
    """
    running = list(_iter_running_worker_pids())
    if not running:
        return 0
    instance_ids = {instance_id for _, instance_id in running}
    try:
        rows = {
            pk: (status, ended_at)
            for pk, status, ended_at in CodexInstance.objects.filter(
                pk__in=instance_ids
            ).values_list("pk", "status", "ended_at")
        }
    except Exception:
        # Never kill on incomplete information: if the DB read fails (e.g. it is
        # momentarily locked) we cannot tell which workers are still expected.
        codex_pool.logger.exception("could not read worker rows; skipping orphan reap")
        return 0
    now = timezone.now()
    killed = 0
    for pid, instance_id in running:
        row = rows.get(instance_id)
        if row is not None:
            status, ended_at = row
            if status in CodexInstance.ACTIVE_STATUSES:
                continue
            # Terminal: spare it only while it may still be running its
            # post-terminal hooks. Within the grace window it is finishing them;
            # *past* the grace a still-live terminal worker is wedged and IS
            # reaped -- even one this process spawned -- so the same-process
            # process actually holding the state-DB lock gets killed rather than
            # exempted forever by the tracked check.
            if ended_at is not None:
                if ended_at > now - _ORPHAN_REAP_GRACE:
                    continue
            elif _is_tracked_worker(pid):
                # No ``ended_at`` recorded yet but we still supervise it: it is
                # mid terminal-commit, not leaked. (Real terminal rows set
                # ``ended_at`` before committing, so this is a thin race guard.)
                continue
        if _kill_orphaned_worker(pid, instance_id):
            killed += 1
            # The worker was killed before its own post-terminal cleanup, so
            # cancel a failed turn's dangling prompts and surface a completed
            # turn whose auto-PR/QA follow-up was dropped.
            _finalize_reaped_instance(instance_id)
    if killed:
        codex_pool.logger.warning("reaped %s orphaned codex worker process(es)", killed)
    return killed

def _kill_orphaned_worker(pid: int, instance_id: int) -> bool:
    """Force-kill a leaked worker (and its app-server child); report success."""
    instance = None
    try:
        instance = CodexInstance.objects.filter(pk=instance_id).first()
    except Exception:
        codex_pool.logger.exception("could not load instance %s for orphan reap", instance_id)
    scope_unit = instance.systemd_scope_unit if instance is not None else None
    # The systemd unit may be absent even for a systemd worker: the row can be
    # gone (e.g. a reset/cleaned DB) or exist with an empty
    # ``systemd_scope_unit`` (the parent died after ``systemd-run`` returned but
    # before saving it). Systemd workers are not launched as our direct session
    # leaders; if the scanned pid is ours under the relaxed check but fails the
    # session-leader check it was launched under systemd isolation, and the
    # killpg path below would skip it -- leaving its app-server (and the Codex DB
    # lock) alive. Reap it through its derived unit instead.
    if (
        not scope_unit
        and codex_pool._pid_is_our_worker(pid, instance_id, require_session_leader=False)
        and not codex_pool._pid_is_our_worker(pid, instance_id)
    ):
        scope_unit = (
            systemd_isolation._worker_unit_from_pid_cgroup(pid, instance_id)
            or codex_pool._scope_unit_for_instance(instance_id)
        )
    if scope_unit:
        # Legacy unit names (``hitch-codex-worker-<id>``) were not
        # deployment-unique, so ``systemctl kill <unit>`` could hit another
        # checkout's reused unit if our systemd worker exited since the scan.
        # Reverify the scanned pid is still our deployment's worker for this
        # instance (systemd workers are not session leaders) before signaling.
        if not codex_pool._pid_is_our_worker(pid, instance_id, require_session_leader=False):
            return False
        # Carry the effective systemd unit on the target even when the row had it
        # empty (a derived unit), so _force_kill_instance signals the unit
        # rather than falling back to killpg.
        target = instance or CodexInstance(pid=pid)
        target.systemd_scope_unit = scope_unit
        try:
            codex_pool._force_kill_instance(target)
            return True
        except ProcessLookupError:
            return False
        except OSError:
            codex_pool.logger.warning(
                "failed to kill orphaned systemd worker for instance %s", instance_id
            )
            return False
    # Re-verify identity right before signaling: the pid could have been recycled
    # since the /proc scan, and we must never SIGKILL an unrelated process group.
    if not codex_pool._pid_is_our_worker(pid, instance_id):
        return False
    try:
        os.killpg(pid, signal.SIGKILL)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        codex_pool.logger.warning(
            "failed to kill orphaned worker pid %s (instance %s)", pid, instance_id
        )
        return False

_APP_SERVER_DEPLOYMENT_ENV = "HITCH_CODEX_DEPLOYMENT"

def _app_server_deployment_id() -> str:
    """Stable per-deployment id stamped on the app-servers Hitch spawns.

    Uses this checkout's ``BASE_DIR`` -- the same deployment identity the worker
    reaper derives from ``manage.py`` (see ``_our_manage_py``) -- so two
    checkouts running under one Unix user and sharing a resolved ``codex``
    binary never match each other's app-servers.
    """
    return str(settings.BASE_DIR)

def _proc_is_our_app_server(pid_dir: Path, deployment_id: str) -> bool:
    """Whether ``pid_dir`` is a ``codex app-server`` process this deployment spawned.

    Two independent signals, both fixed at ``exec`` and therefore stable across
    reparenting (so a leaked/orphaned app-server whose worker died is still
    matched):

    * cmdline is the SDK's stdio app-server invocation
      (``codex ... app-server --listen stdio://``) -- so an interactive
      ``codex`` TUI a developer runs by hand never matches;
    * environ carries our ``HITCH_CODEX_DEPLOYMENT`` marker
      (stamped in ``app_server_config``) -- so another checkout's app-servers
      are excluded even when the resolved ``codex`` binary is shared. Pinning
      ``argv[0]`` to the binary path alone could not tell two such checkouts
      apart, which is why the deployment marker is required.

    Re-reading ``/proc`` here (rather than caching the scan result) also lets
    callers reuse this as the pre-signal identity recheck that guards against
    pid recycling.
    """
    try:
        cmdline = (pid_dir / "cmdline").read_bytes()
    except OSError:
        return False
    parts = cmdline.split(b"\0")
    if b"app-server" not in parts or b"stdio://" not in parts:
        return False
    try:
        environ = (pid_dir / "environ").read_bytes()
    except OSError:
        return False
    marker = f"{_APP_SERVER_DEPLOYMENT_ENV}={deployment_id}".encode()
    return marker in environ.split(b"\0")

def _proc_ppid(pid_dir: Path) -> int | None:
    """Parent pid from ``/proc/<pid>/stat`` field 4, or ``None`` if unreadable.

    ``stat`` field 2 (``comm``) is wrapped in parentheses and may itself contain
    spaces or a literal ``)``, so the positional fields are read from after the
    final ``)`` -- the parsing the kernel docs prescribe. There ``state`` is the
    first field and ``ppid`` the second.
    """
    try:
        stat = (pid_dir / "stat").read_text()
    except OSError:
        return None
    rparen = stat.rfind(")")
    if rparen == -1:
        return None
    fields = stat[rparen + 1 :].split()
    if len(fields) < 2:
        return None
    try:
        return int(fields[1])
    except ValueError:
        return None

def _matched_app_server_pids(
    *, proc_root: Path, deployment_id: str
) -> dict[int, Path]:
    """Map every matching ``codex app-server`` pid to its ``/proc`` entry.

    Includes both halves of each logical app-server: the ``codex`` CLI is a node
    wrapper that re-execs a native child, and both inherit the SDK app-server
    argv and our ``HITCH_CODEX_DEPLOYMENT`` marker, so each logical app-server
    matches twice. Callers decide how to treat the pair -- the nuke sweep
    signals every entry, while the health count collapses each wrapper/child
    pair to one. Linux-only; without ``/proc`` the map is empty.
    """
    matched: dict[int, Path] = {}
    if not proc_root.exists():
        return matched
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        if not _proc_is_our_app_server(entry, deployment_id):
            continue
        try:
            matched[int(entry.name)] = entry
        except ValueError:
            continue
    return matched

def _iter_codex_app_server_pids(
    *, proc_root: Path = Path("/proc"), deployment_id: str | None = None
) -> Iterable[int]:
    """Yield the pid of *every* ``codex app-server`` process this deployment started.

    Discovery is process-based (a /proc scan), not DB-based, on purpose: a
    leaked app-server -- one whose worker died without reaping it, or whose
    CodexInstance row is already terminal or gone -- is exactly what this is
    meant to find, and it has no live DB row to locate it by. Scoping to this
    deployment is handled by ``_proc_is_our_app_server``. Linux-only; on a host
    without ``/proc`` it yields nothing.

    Both halves of each logical app-server (the node wrapper and its native
    re-exec child) are yielded, deliberately not deduped: the nuke sweep that
    drives this must SIGKILL each one. SIGKILL is not delivered to a process's
    children, and the native child -- not the wrapper -- is what actually runs
    the app-server and holds the CODEX_HOME state-DB connection, so killing only
    the wrapper would orphan the lock-holder. ``count_running_codex_app_servers``
    is the surface that collapses the pair to one logical app-server.
    """
    if deployment_id is None:
        deployment_id = _app_server_deployment_id()
    yield from _matched_app_server_pids(
        proc_root=proc_root, deployment_id=deployment_id
    )

def nuke_codex_app_servers(*, proc_root: Path = Path("/proc")) -> int:
    """SIGKILL every ``codex app-server`` process this deployment started.

    A manual escape hatch (surfaced on the profile page) for when app-servers
    have leaked: each one holds a connection to the shared CODEX_HOME state DB
    and contends on its single writer lock, so a pile of orphans surfaces as
    "database is locked" on every new turn. ``reconcile_orphaned_workers``
    handles the common case, but it only reaps app-servers reachable from a
    known worker row; this sweeps live processes directly so it also kills the
    truly orphaned ones.

    Each app-server is signaled individually with ``os.kill`` rather than
    ``killpg``: the SDK spawns it sharing its parent's process group (the
    detached worker, or the Django process itself for synchronous opens), so a
    group kill could take down the parent. Both halves of each logical
    app-server -- the node wrapper and its native child -- are signaled directly
    for the same reason SIGKILL cannot be left to cascade: it is not delivered
    to children, and the native child is the lock-holder, so it must be killed
    in its own right. Returns the number of processes signaled (roughly twice
    the logical app-server count when wrapper/child pairs are present).
    """
    deployment_id = _app_server_deployment_id()
    killed = 0
    for pid in list(
        _iter_codex_app_server_pids(proc_root=proc_root, deployment_id=deployment_id)
    ):
        # Re-verify identity immediately before signaling: the pid could have
        # been recycled since the scan, and we must never SIGKILL an unrelated
        # process. Re-reading /proc (not just trusting ProcessLookupError, which
        # a recycled pid would never raise) closes that race the same way
        # ``_kill_orphaned_worker`` does.
        if not _proc_is_our_app_server(proc_root / str(pid), deployment_id):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            # Exited between the recheck and the signal -- nothing to kill.
            continue
        except OSError:
            codex_pool.logger.warning("failed to SIGKILL codex app-server pid %s", pid)
            continue
        killed += 1
    if killed:
        codex_pool.logger.warning("nuked %s codex app-server process(es)", killed)
    return killed

def reap_orphaned_app_servers(*, proc_root: Path = Path("/proc")) -> int:
    """SIGKILL this deployment's app-servers that no live owner is driving.

    Every Codex app-server this deployment spawns is owned by a live process:
    a detached worker (an ACTIVE ``CodexInstance`` whose pid still belongs to
    that worker), or a web/scheduler process of this checkout. When an owner
    dies without closing its app-server -- a web-server restart with pooled
    servers checked out, a worker SIGKILLed mid-turn before its cgroup reap --
    the app-server lives on, holding a CODEX_HOME state-DB connection and
    contending on its single writer lock, which surfaces as "database is
    locked" on the first turns after a restart. The manual nuke
    (``nuke_codex_app_servers``) kills every deployment-matched server
    including owned ones, so it cannot run unattended; this sweep kills only
    the ownerless ones and runs automatically when the maintenance scheduler
    starts.

    A matched server is kept when walking its parent chain (through the node
    wrapper half of the wrapper/native pair) reaches a live owner: a verified
    ACTIVE worker pid, this process, or any live process whose working
    directory is this checkout (a sibling web process of whatever server
    flavor). Pid identity is re-verified immediately before signaling, the
    same discipline as the nuke.
    """
    deployment_id = _app_server_deployment_id()
    matched = _matched_app_server_pids(
        proc_root=proc_root, deployment_id=deployment_id
    )
    if not matched:
        return 0
    owner_pids = {os.getpid()}
    for instance in CodexInstance.objects.filter(
        status__in=CodexInstance.ACTIVE_STATUSES
    ):
        if instance.pid > 0 and codex_pool.worker_is_alive(instance):
            owner_pids.add(instance.pid)
    killed = 0
    for pid in list(matched):
        if _app_server_has_live_owner(
            pid, matched=matched, owner_pids=owner_pids, proc_root=proc_root
        ):
            continue
        if not _proc_is_our_app_server(proc_root / str(pid), deployment_id):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except OSError:
            codex_pool.logger.warning(
                "failed to SIGKILL orphaned codex app-server pid %s", pid
            )
            continue
        killed += 1
    if killed:
        codex_pool.logger.warning(
            "reaped %s orphaned codex app-server process(es)", killed
        )
    return killed


def _app_server_has_live_owner(
    pid: int,
    *,
    matched: dict[int, Path],
    owner_pids: set[int],
    proc_root: Path,
) -> bool:
    """Whether ``pid``'s parent chain reaches a live owner of this deployment.

    The chain is walked through other matched app-server processes (the node
    wrapper parents its native child) and ends at the first non-matched
    parent, which must be an owner. A vanished parent entry means the chain
    was cut by an exited process -- exactly the orphan case.
    """
    current = pid
    for _ in range(4):
        ppid = _proc_ppid(proc_root / str(current))
        if ppid is None or ppid <= 1:
            return False
        if ppid in owner_pids:
            return True
        if _proc_cwd_is_this_checkout(proc_root / str(ppid)):
            return True
        if ppid in matched:
            current = ppid
            continue
        return False
    return False


def _proc_cwd_is_this_checkout(pid_dir: Path) -> bool:
    """Whether the process's working directory is this deployment's checkout.

    Identifies sibling server processes of any flavor (runserver, gunicorn
    workers) without relying on their argv: they all run from ``BASE_DIR``.
    Workers do not (they run in session cwds) but are owned via their ACTIVE
    rows instead.
    """
    try:
        return (pid_dir / "cwd").resolve() == Path(settings.BASE_DIR).resolve()
    except OSError:
        return False


def count_running_codex_app_servers(*, proc_root: Path = Path("/proc")) -> int:
    """Number of *logical* ``codex app-server`` processes this deployment is running.

    Read-only counterpart to ``nuke_codex_app_servers``: a health surface for
    spotting leaked app-servers (each holds a CODEX_HOME state-DB connection)
    without killing anything. The ``codex`` CLI is a node wrapper that re-execs a
    native child, so each logical app-server matches the /proc scan twice; drop
    any matched pid whose parent is itself matched (the native child) so the
    figure reflects logical app-servers, not doubled pids. A pid whose parent is
    unknown or unmatched -- including a native child orphaned by a dead wrapper
    -- counts, so a leaked app-server is never undercounted.
    """
    matched = _matched_app_server_pids(
        proc_root=proc_root, deployment_id=_app_server_deployment_id()
    )
    return sum(
        1
        for pid, entry in matched.items()
        if (ppid := _proc_ppid(entry)) is None or ppid not in matched
    )

def _reaped_turn_lost_auto_review(instance: CodexInstance) -> bool:
    """Whether a reaped COMPLETED turn's auto-PR/QA follow-up was lost.

    The worker fires the auto-review workflow from ``_notify_system_agents``
    *after* committing the terminal status, claiming ``auto_pr_triggered_at`` /
    ``auto_qa_triggered_at`` as it does. A reaped COMPLETED user turn with the
    automation enabled but neither field set was killed before it could fire --
    and unlike workflow-owned rows there is no later reconcile that recovers it.

    Excludes turns where the automation would have been *intentionally* declined
    (visible-approval mode, or a pending proposed plan): there the null
    timestamps are by design, so the turn is a real success, not a lost
    follow-up, and must not be rewritten as failed.
    """
    if not (
        instance.status == CodexInstance.STATUS_COMPLETED
        and instance.purpose == CodexInstance.PURPOSE_USER
        and instance.workflow_id is None
        and not instance.plan_mode
        and (instance.auto_pr_enabled or instance.auto_qa_enabled)
        and instance.auto_pr_triggered_at is None
        and instance.auto_qa_triggered_at is None
    ):
        return False
    try:
        from hitch.main.workflows import system_agents

        if system_agents.auto_review_intentionally_skipped(instance):
            return False
    except Exception:
        # If we cannot determine intent, prefer leaving a completed turn intact
        # over rewriting a successful result as a false failure.
        codex_pool.logger.exception(
            "could not check auto-review intent for reaped instance %s", instance.pk
        )
        return False
    return True

def _finalize_reaped_instance(instance_id: int) -> None:
    """Clean up after force-killing a reaped worker so its turn isn't left in a
    silently-broken state.

    Things the killed worker never got to handle itself:

    * finish routing: a terminal demo/system-agent (or workflow-owned user) turn
      relies on ``_notify_system_agents_if_needed`` to route its post-terminal
      hooks, the same idempotent callback ``_mark_dead_instances_failed`` runs for
      rows that died after saving terminal status; without it the
      ``SessionDemo``/``SystemAgentRun``/workflow follow-up is stranded;
    * a ``FAILED`` turn's dangling prompts: ``codex_worker`` cancels its pending
      approval/input prompts before exiting, and reaped terminal rows never pass
      through ``_mark_dead_instances_failed`` (which does the same), so otherwise
      the UI keeps actionable cards no worker can answer;
    * a ``COMPLETED`` plain user turn whose auto-PR/QA never fired
      (``_reaped_turn_lost_auto_review``; not covered by the routing above):
      surface it as a failed turn with a retry hint so the user sees the dropped
      follow-up instead of a silent success.
    """
    try:
        instance = CodexInstance.objects.filter(pk=instance_id).first()
    except Exception:
        codex_pool.logger.exception("could not load reaped instance %s for finalize", instance_id)
        return
    if instance is None or instance.status not in (
        CodexInstance.STATUS_COMPLETED,
        CodexInstance.STATUS_FAILED,
    ):
        return
    if instance.status == CodexInstance.STATUS_FAILED:
        codex_pool._resolve_dangling_requests(instance.pk)
    # Idempotent finish routing for demo/system-agent/workflow-owned rows (a
    # no-op for a plain user turn, which the lost-auto-review check below covers).
    _notify_system_agents_if_needed(instance)
    codex_pool.cleanup_requested_input_images_for(instance)
    if _reaped_turn_lost_auto_review(instance):
        automation = "auto-PR" if instance.auto_pr_enabled else "auto-QA"
        CodexInstance.objects.filter(
            pk=instance_id, status=CodexInstance.STATUS_COMPLETED
        ).update(
            status=CodexInstance.STATUS_FAILED,
            error=(
                f"This turn finished, but its {automation} workflow could not "
                "start because the worker had to be terminated while holding the "
                "Codex database lock. Send the message again to retry."
            ),
        )

_RECONCILE_DEAD_MIN_INTERVAL = timedelta(seconds=2)

def reconcile_dead_if_due() -> int:
    """Debounced ``reconcile_dead`` for the request and SSE paths.

    Every major GET view and every SSE (re)connect ran the full
    ``reconcile_dead`` sweep, so N concurrent browser tabs produced N concurrent
    full-table sweeps all contending for SQLite's single write lock. Gating the
    sweep through ``rate_limit.claim`` collapses that to at most one sweep per
    ``_RECONCILE_DEAD_MIN_INTERVAL`` across the whole app; skipped callers rely
    on the next due request and the 60s workflow-maintenance scheduler (which
    still calls ``reconcile_dead`` directly) to clear dead workers. Tests run the
    sweep unconditionally so existing per-request reconcile assertions hold.
    """
    if getattr(settings, "TESTING", False):
        return reconcile_dead()
    if rate_limit.claim(
        "reconcile_dead", min_interval=_RECONCILE_DEAD_MIN_INTERVAL
    ):
        return reconcile_dead()
    return 0

def reconcile_dead_for_thread(thread_id: str) -> int:
    """Mark dead active workers for one user-visible thread.

    Session detail and SSE routing need this exact thread's active-worker state
    to be fresh even when the global sweep is debounced. Keep the scope narrow so
    opening a stale session repairs it without doing a full active-worker scan.
    """
    codex_pool._reap_finished_workers()
    pending = CodexInstance.objects.filter(
        thread_id=thread_id,
        status__in=CodexInstance.ACTIVE_STATUSES,
    )
    updated = _mark_dead_instances_failed(pending)
    _reconcile_terminal_workflow_instances(main_thread_id=thread_id)
    if updated:
        _reconcile_orphaned_workers_if_due()
    _prune_reaped_workers()
    return updated

def reconcile_dead_for_workflow(
    workflow_id: int, *, main_thread_id: str | None = None
) -> int:
    """Mark dead workers for one workflow without sweeping every session."""
    codex_pool._reap_finished_workers()
    pending = CodexInstance.objects.filter(
        workflow_id=workflow_id,
        status__in=CodexInstance.ACTIVE_STATUSES,
    )
    updated = _mark_dead_instances_failed(pending)
    _reconcile_terminal_workflow_instances(
        main_thread_id=main_thread_id,
        workflow_id=workflow_id,
    )
    # A hidden workflow stream may be the only thing reconciling (maintenance
    # scheduler disabled, or between its 60s ticks), so reap leaked workers here
    # too -- otherwise a wedged workflow worker keeps the Codex state-DB lock.
    # The reap is a global /proc scan, so debounce it: many concurrent workflow
    # streams collapse to one sweep per interval (the 60s reap grace makes this
    # coarse gate harmless).
    _reconcile_orphaned_workers_if_due()
    _prune_reaped_workers()
    return updated

def _reconcile_orphaned_workers_if_due() -> int:
    """Debounced global orphan reap for scoped callers (workflow streams)."""
    if getattr(settings, "TESTING", False):
        return reconcile_orphaned_workers()
    if rate_limit.claim(
        "reconcile_orphaned_workers", min_interval=_RECONCILE_DEAD_MIN_INTERVAL
    ):
        return reconcile_orphaned_workers()
    return 0

def _mark_dead_instances_failed(pending: Iterable[CodexInstance]) -> int:
    updated = 0
    now = timezone.now()
    for instance in pending:
        if codex_pool.worker_is_alive(instance):
            continue
        error = worker_errors._dead_worker_error(instance)
        # Conditional UPDATE keyed on the active statuses so a worker that
        # reached a terminal state in the gap between the queryset read and this
        # write is preserved rather than retroactively rewritten as failed (and
        # falsely routed to the system agents). Mirrors ``_mark_failed``.
        rows = CodexInstance.objects.filter(
            pk=instance.pk,
            status__in=CodexInstance.ACTIVE_STATUSES,
        ).update(
            status=CodexInstance.STATUS_FAILED,
            error=error,
            ended_at=now,
        )
        if rows == 0:
            # The worker reached a terminal state in the gap. Preserve its
            # status (don't count it as a kill), but still run finish routing:
            # demo system-agent rows are excluded from
            # reconcile_terminal_workflow_instances() and rely on this callback,
            # so a worker that died after saving its status but before notifying
            # would otherwise strand the SessionDemo/SystemAgentRun. Routing is
            # idempotent, so a worker that already notified is a no-op.
            try:
                instance.refresh_from_db()
            except CodexInstance.DoesNotExist:
                continue
            if instance.status in (
                CodexInstance.STATUS_COMPLETED,
                CodexInstance.STATUS_FAILED,
            ):
                _notify_system_agents_if_needed(instance)
                codex_pool.cleanup_requested_input_images_for(instance)
            continue
        codex_pool._resolve_dangling_requests(instance.pk)
        instance.refresh_from_db()
        # A dead worker cannot run its own completion hook. Preserve the turn's
        # terminal activity in the cached index before routing the failure.
        codex_pool._record_session_activity(instance)
        # The worker died without reporting completion, so its systemd cgroup may
        # still hold grandchildren the codex sandbox reparented into their own
        # session (a process-group signal would miss them). Reap the cgroup so a
        # leaked ``cargo bench`` can't hold memory long after the worker is gone.
        systemd_isolation._reap_scope_cgroup(instance)
        _notify_system_agents_if_needed(instance)
        codex_pool.cleanup_requested_input_images_for(instance)
        updated += 1
    return updated

def _prune_reaped_workers() -> None:
    with codex_pool._TRACKED_WORKER_PROCS_LOCK:
        reaped_workers = set(codex_pool._REAPED_WORKERS)
    if not reaped_workers:
        return
    active_workers = set(
        CodexInstance.objects.filter(
            pk__in=[instance_id for _, instance_id in reaped_workers],
            status__in=CodexInstance.ACTIVE_STATUSES,
        ).values_list("pid", "pk")
    )
    with codex_pool._TRACKED_WORKER_PROCS_LOCK:
        codex_pool._REAPED_WORKERS.intersection_update(active_workers)

def _notify_system_agents_if_needed(instance: CodexInstance) -> None:
    system_agents_handled = False
    if instance.purpose in (
        CodexInstance.PURPOSE_SYSTEM_AGENT,
        CodexInstance.PURPOSE_SYSTEM_FEEDBACK,
    ) or (
        instance.purpose == CodexInstance.PURPOSE_USER
        and instance.workflow_id is not None
    ):
        try:
            from hitch.main.workflows import system_agents

            system_agents_handled = system_agents.on_codex_instance_finished(instance)
        except Exception:
            codex_pool.logger.exception(
                "failed to notify system workflow for reconciled instance %s",
                instance.pk,
            )
    try:
        from hitch.main import demo

        if (
            system_agents_handled
            and instance.purpose == CodexInstance.PURPOSE_SYSTEM_AGENT
            and instance.agent_kind == demo.DEMO_AGENT_KIND
        ):
            return
        demo.on_codex_instance_finished(instance)
    except Exception:
        codex_pool.logger.exception(
            "failed to notify demo workflow for reconciled instance %s",
            instance.pk,
        )

def _reconcile_terminal_workflow_instances(
    *, main_thread_id: str | None = None, workflow_id: int | None = None
) -> None:
    try:
        from hitch.main.workflows import system_agents

        system_agents.reconcile_terminal_workflow_instances(
            main_thread_id=main_thread_id,
            workflow_id=workflow_id,
        )
    except Exception:
        codex_pool.logger.exception("failed to reconcile terminal workflow instances")
