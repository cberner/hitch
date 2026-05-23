"""Pool of detached Codex worker subprocesses.

Each worker runs a single turn for one Codex thread and then exits. Workers
are spawned with ``start_new_session=True`` (a fresh process group) so they
survive a Django restart: the parent's stdin/stdout/stderr are redirected to
``/dev/null`` and the child no longer inherits the parent's controlling
terminal. The CodexInstance row + JSONL events file on disk are the durable
post-spawn links back to a worker; a sibling control JSONL file carries
mid-turn requests such as steer payloads into the detached process.

The worker is the ``codex_worker`` management command in this app; running it
as a Django command lets it use the same ORM/settings as the parent without
re-implementing Django bootstrap.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from django.conf import settings
from django.db import transaction
from django.utils import timezone
from openai_codex import AppServerConfig, Codex
from openai_codex.generated.v2_all import ThreadSource

from hitch.main.models import CodexInstance

logger = logging.getLogger(__name__)

_TRACKED_WORKER_PROCS: dict[int, tuple[int, subprocess.Popen[bytes]]] = {}
_REAPED_WORKERS: set[tuple[int, int]] = set()
_TRACKED_WORKER_PROCS_LOCK = threading.Lock()


def spawn_new_session(
    *,
    cwd: str,
    prompt: str,
    thread_name: str | None = None,
    base_instructions: str | None = None,
    developer_instructions: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    sandbox_policy: str | None = None,
    approval_mode: str | None = None,
    enable_memories: bool = False,
    plan_mode: bool = False,
    thread_source: ThreadSource | None = None,
    purpose: str = CodexInstance.PURPOSE_USER,
    workflow_id: int | None = None,
    agent_kind: str = "",
    display_author: str = "",
    output_schema: dict[str, Any] | None = None,
    user_message_index: int | None = 0,
    auto_pr_enabled: bool = False,
) -> CodexInstance:
    """Create a fresh Codex thread and detach a worker to run the initial prompt.

    ``developer_instructions`` maps to the Codex SDK's per-thread
    ``developerInstructions`` field; the remaining overrides come from the
    settings cookies the request handler reads. ``None`` means "let Codex
    apply its own default." The thread is created synchronously (so the caller
    has an id to redirect to immediately); the prompt itself is run by the
    detached worker.
    """
    config = app_server_config(enable_memories=enable_memories)
    with Codex(config=config) as codex:
        start_kwargs: dict[str, Any] = {
            "cwd": cwd,
            "developer_instructions": developer_instructions,
            "model": model,
        }
        if base_instructions:
            start_kwargs["base_instructions"] = base_instructions
        if thread_source is not None:
            start_kwargs["thread_source"] = thread_source
        thread = codex.thread_start(**start_kwargs)
        thread_id = thread.id
        # ``thread/start`` only creates the thread in the app-server's
        # in-memory map; the rollout file on disk is not written until
        # something triggers a metadata persist. Without this step, the
        # worker subprocess and the session view both fail with "no rollout
        # found for thread id" the moment we exit the Codex context here
        # (which tears down the app-server holding the in-memory thread).
        # ``thread/set-name`` is the cheapest write that goes through
        # ``live_thread_for_persistence``, so it blocks until the rollout
        # file exists on disk. The first line of the prompt mirrors the
        # title the session list would otherwise compute from ``preview``
        # once the first turn streams in, so this is usually invisible in
        # the UI. Callers can pass ``thread_name`` when the prompt starts
        # with generic instructions and a better task title is known.
        name_source = (
            thread_name
            if thread_name is not None and thread_name.strip()
            else prompt
        )
        codex._client.thread_set_name(thread_id, _initial_thread_name(name_source))
    return _spawn_worker(
        thread_id=thread_id,
        cwd=cwd,
        prompt=prompt,
        base_instructions=base_instructions,
        developer_instructions=developer_instructions,
        model=model if plan_mode else None,
        stored_model=model,
        reasoning_effort=reasoning_effort,
        sandbox_policy=sandbox_policy,
        approval_mode=approval_mode,
        enable_memories=enable_memories,
        plan_mode=plan_mode,
        purpose=purpose,
        workflow_id=workflow_id,
        agent_kind=agent_kind,
        display_author=display_author,
        output_schema=output_schema,
        user_message_index=user_message_index,
        auto_pr_enabled=auto_pr_enabled,
    )


def create_session_thread(
    *,
    cwd: str,
    name: str,
    base_instructions: str | None = None,
    developer_instructions: str | None = None,
    model: str | None = None,
    enable_memories: bool = False,
) -> str:
    """Create and persist a visible Codex thread without starting a turn."""
    config = app_server_config(enable_memories=enable_memories)
    with Codex(config=config) as codex:
        start_kwargs: dict[str, Any] = {
            "cwd": cwd,
            "developer_instructions": developer_instructions,
            "model": model,
        }
        if base_instructions:
            start_kwargs["base_instructions"] = base_instructions
        thread = codex.thread_start(**start_kwargs)
        codex._client.thread_set_name(thread.id, _initial_thread_name(name))
        return thread.id


# Upper bound for the auto-derived thread name. Matches the
# ``_NAME_MAX_LEN`` cap that ``set_session_name`` enforces on user-supplied
# names so the two write paths stay consistent.
_INITIAL_THREAD_NAME_MAX_LEN = 200


def _initial_thread_name(prompt: str) -> str:
    """Return a non-empty thread name derived from ``prompt``.

    Codex rejects whitespace-only names, so a prompt that strips to empty
    falls back to a static placeholder rather than failing the wire call.
    The view layer already rejects empty prompts up front, so this branch
    only protects against future callers passing a degenerate string.
    """
    first_line = prompt.split("\n", 1)[0].strip()[:_INITIAL_THREAD_NAME_MAX_LEN].rstrip()
    return first_line or "New session"


def spawn_turn(
    *,
    thread_id: str,
    cwd: str,
    prompt: str,
    model: str | None = None,
    stored_model: str | None = None,
    reasoning_effort: str | None = None,
    stored_reasoning_effort: str | None = None,
    sandbox_policy: str | None = None,
    approval_mode: str | None = None,
    enable_memories: bool = False,
    collaboration_mode: str | None = None,
    plan_mode: bool = False,
    base_instructions: str | None = None,
    developer_instructions: str | None = None,
    purpose: str = CodexInstance.PURPOSE_USER,
    workflow_id: int | None = None,
    agent_kind: str = "",
    display_author: str = "",
    output_schema: dict[str, Any] | None = None,
    user_message_index: int | None = None,
    auto_pr_enabled: bool = False,
) -> CodexInstance:
    """Detach a worker that resumes an existing thread to run one prompt."""
    if base_instructions is None:
        previous = latest_for_thread(thread_id)
        base_instructions = previous.base_instructions if previous is not None else None
    if developer_instructions is None:
        previous = latest_for_thread(thread_id)
        developer_instructions = (
            previous.developer_instructions if previous is not None else None
        )
    return _spawn_worker(
        thread_id=thread_id,
        cwd=cwd,
        prompt=prompt,
        base_instructions=base_instructions or None,
        developer_instructions=developer_instructions or None,
        model=model,
        stored_model=stored_model,
        reasoning_effort=reasoning_effort,
        stored_reasoning_effort=stored_reasoning_effort,
        sandbox_policy=sandbox_policy,
        approval_mode=approval_mode,
        enable_memories=enable_memories,
        collaboration_mode=collaboration_mode,
        plan_mode=plan_mode,
        purpose=purpose,
        workflow_id=workflow_id,
        agent_kind=agent_kind,
        display_author=display_author,
        output_schema=output_schema,
        user_message_index=user_message_index,
        auto_pr_enabled=auto_pr_enabled,
    )


def is_alive(pid: int) -> bool:
    """Return whether ``pid`` is currently a live process on this host.

    PIDs can be recycled, so the answer is only meaningful when combined with
    a recent CodexInstance.started_at. A false reading triggers a status
    reconciliation; a true reading is treated as best-effort.
    """
    if pid <= 0:
        return False
    with _TRACKED_WORKER_PROCS_LOCK:
        has_tracked_proc = pid in _TRACKED_WORKER_PROCS
        has_reaped_worker = any(reaped_pid == pid for reaped_pid, _ in _REAPED_WORKERS)
        if has_reaped_worker and not has_tracked_proc:
            return False
    if _reap_tracked_worker(pid):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but is owned by another user.
        return True
    except OSError:
        return False
    state = _linux_proc_state(pid)
    return state not in ("", "X", "Z", "x")


def worker_is_alive(instance: CodexInstance) -> bool:
    """Return whether ``instance`` still has a live worker process."""
    if instance.pid <= 0:
        return False
    with _TRACKED_WORKER_PROCS_LOCK:
        if (instance.pid, instance.pk) in _REAPED_WORKERS:
            return False
        tracked = _TRACKED_WORKER_PROCS.get(instance.pid)
    if tracked is not None:
        tracked_instance_id, proc = tracked
        if tracked_instance_id == instance.pk:
            return not _reap_tracked_worker_process(
                instance.pid, tracked_instance_id, proc
            )
    return is_alive(instance.pid)


def _track_worker_process(instance_id: int, proc: subprocess.Popen[bytes]) -> None:
    """Keep a child handle so exited detached workers can be reaped.

    ``start_new_session=True`` detaches the worker from the terminal, not
    from this parent process. While the Django process that spawned it stays
    alive, an exited worker remains our child until we wait on it.
    """
    with _TRACKED_WORKER_PROCS_LOCK:
        _REAPED_WORKERS.discard((proc.pid, instance_id))
        _TRACKED_WORKER_PROCS[proc.pid] = (instance_id, proc)
    threading.Thread(
        target=_wait_for_tracked_worker,
        args=(proc.pid, instance_id, proc),
        name=f"codex-worker-reaper-{proc.pid}",
        daemon=True,
    ).start()


def _wait_for_tracked_worker(
    pid: int, instance_id: int, proc: subprocess.Popen[bytes]
) -> None:
    try:
        proc.wait()
    except OSError:
        logger.debug("dropping unreapable worker process handle for pid %s", pid)
    finally:
        with _TRACKED_WORKER_PROCS_LOCK:
            if _TRACKED_WORKER_PROCS.get(pid) == (instance_id, proc):
                del _TRACKED_WORKER_PROCS[pid]
                _REAPED_WORKERS.add((pid, instance_id))


def _reap_tracked_worker(pid: int) -> bool:
    with _TRACKED_WORKER_PROCS_LOCK:
        tracked = _TRACKED_WORKER_PROCS.get(pid)
    if tracked is None:
        return False
    instance_id, proc = tracked
    return _reap_tracked_worker_process(pid, instance_id, proc)


def _reap_finished_workers() -> None:
    with _TRACKED_WORKER_PROCS_LOCK:
        tracked = list(_TRACKED_WORKER_PROCS.items())
    for pid, (instance_id, proc) in tracked:
        _reap_tracked_worker_process(pid, instance_id, proc)


def _reap_tracked_worker_process(
    pid: int, instance_id: int, proc: subprocess.Popen[bytes]
) -> bool:
    try:
        proc.wait(timeout=0)
    except subprocess.TimeoutExpired:
        return False
    except OSError:
        logger.debug("dropping unreapable worker process handle for pid %s", pid)
    with _TRACKED_WORKER_PROCS_LOCK:
        if _TRACKED_WORKER_PROCS.get(pid) == (instance_id, proc):
            del _TRACKED_WORKER_PROCS[pid]
            _REAPED_WORKERS.add((pid, instance_id))
    return True


def _linux_proc_state(pid: int) -> str | None:
    """Return Linux's one-letter process state, or None when unavailable.

    ``""`` means /proc exists but the specific pid disappeared between
    ``kill(pid, 0)`` and the stat read.
    """
    proc_root = Path("/proc")
    if not proc_root.exists():
        return None
    try:
        stat = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except OSError:
        return None
    end = stat.rfind(")")
    if end < 0 or end + 2 >= len(stat):
        return None
    return stat[end + 2]


def list_for_thread(thread_id: str) -> list[CodexInstance]:
    """Return all workers ever spawned for a thread, newest first."""
    return list(CodexInstance.objects.filter(thread_id=thread_id).order_by("-started_at"))


def latest_for_thread(thread_id: str) -> CodexInstance | None:
    return (
        CodexInstance.objects.filter(thread_id=thread_id).order_by("-started_at").first()
    )


def latest_active_for_thread(thread_id: str) -> CodexInstance | None:
    """Return the most recent active CodexInstance for ``thread_id``, or None.

    ``send_message`` can queue another turn before the previous one ends, so
    a fast-failing newer worker would otherwise mask an older still-running
    one if we filtered only by ``started_at``. Filter on active status first
    so a session is treated as live whenever *any* worker for it is still
    starting or running.
    """
    return (
        CodexInstance.objects.filter(
            thread_id=thread_id,
            status__in=(CodexInstance.STATUS_STARTING, CodexInstance.STATUS_RUNNING),
        )
        .order_by("-started_at")
        .first()
    )


def latest_id_for_thread(thread_id: str) -> int | None:
    """Return the highest ``CodexInstance.pk`` for ``thread_id``, or ``None``.

    Used by the idle SSE stream as a baseline so it can detect *any* new
    worker for the session — including one that started and completed
    between two polls of ``latest_active_for_thread``. Without this, a
    short-lived out-of-band turn would leave the open session page stale
    until the next manual refresh.
    """
    return (
        CodexInstance.objects.filter(thread_id=thread_id)
        .order_by("-pk")
        .values_list("pk", flat=True)
        .first()
    )


def interrupt_active(thread_id: str) -> CodexInstance | None:
    """Stop the most recent active worker for ``thread_id``.

    Fallback entry point for callers that only know a thread id (the
    session page itself prefers ``interrupt_instance`` so each click
    targets the exact worker the page is streaming, not "whichever
    worker is latest at click time").
    """
    instance = latest_active_for_thread(thread_id)
    if instance is None:
        return None
    return _interrupt_instance(instance)


def interrupt_instance(
    instance_id: int, *, expected_thread_id: str
) -> CodexInstance | None:
    """Stop a specific worker, identified by its primary key.

    The session page renders the active worker's id into the Stop
    button so each click targets that exact worker rather than
    "latest active for this thread". This matters because
    ``send_message`` can stack overlapping turns on the same thread:
    a stale tab whose page was rendered before a newer turn started
    would otherwise abort the newer worker the user can't even see.

    ``expected_thread_id`` cross-checks the form value against the URL
    so a tampered/stale post can't be used to stop a worker that
    belongs to a different thread.

    Returns None when the instance is unknown, belongs to a different
    thread, has already reached a terminal status, is still launching
    (pid unset), or could not be signaled.
    """
    try:
        instance = CodexInstance.objects.get(pk=instance_id)
    except CodexInstance.DoesNotExist:
        return None
    if instance.thread_id != expected_thread_id:
        return None
    if instance.status not in (
        CodexInstance.STATUS_STARTING,
        CodexInstance.STATUS_RUNNING,
    ):
        return None
    return _interrupt_instance(instance)


def steer_active(thread_id: str, *, prompt: str) -> CodexInstance | None:
    """Inject ``prompt`` into the most recent active worker for ``thread_id``."""
    instance = latest_active_for_thread(thread_id)
    if instance is None:
        return None
    return _steer_instance(instance, prompt=prompt)


def steer_instance(
    instance_id: int, *, expected_thread_id: str, prompt: str
) -> CodexInstance | None:
    """Steer a specific active worker, identified by its primary key.

    Mirrors ``interrupt_instance``'s stale-tab protection: the posted worker
    id must still belong to the URL's thread and must still be active. The
    worker reads the payload from its control JSONL file and calls
    ``TurnHandle.steer(...)`` on the in-process handle, so the SDK supplies
    the currently running turn id as the expected turn id.
    """
    try:
        instance = CodexInstance.objects.get(pk=instance_id)
    except CodexInstance.DoesNotExist:
        return None
    if instance.thread_id != expected_thread_id:
        return None
    if instance.status not in (
        CodexInstance.STATUS_STARTING,
        CodexInstance.STATUS_RUNNING,
    ):
        return None
    return _steer_instance(instance, prompt=prompt)


def control_path_for(instance: CodexInstance) -> Path:
    """Return the per-worker control JSONL path next to its events log."""
    events_path = Path(instance.events_path)
    return events_path.with_name(f"{events_path.stem}.control.jsonl")


def _steer_instance(instance: CodexInstance, *, prompt: str) -> CodexInstance | None:
    """Queue one steer request for ``instance`` and nudge its worker.

    The payload is appended before the signal so the worker never wakes up to
    an empty control channel. If the worker is still in ``starting`` we skip
    SIGUSR1: the handler may not be installed yet, and the worker drains the
    file once the ``TurnHandle`` exists.
    """
    if instance.pid <= 0:
        return None
    if not prompt.strip():
        return None
    if not _pid_is_our_worker(instance.pid, instance.pk):
        _mark_failed(instance, "worker process unavailable for steer")
        return None

    try:
        _append_control_request(
            instance,
            {
                "op": "steer",
                "input": prompt,
            },
        )
    except OSError:
        return None
    if instance.status != CodexInstance.STATUS_RUNNING:
        instance.refresh_from_db()
        if instance.status not in (
            CodexInstance.STATUS_STARTING,
            CodexInstance.STATUS_RUNNING,
        ):
            return None
        return instance
    try:
        os.kill(instance.pid, signal.SIGUSR1)
    except ProcessLookupError:
        _mark_failed(instance, "worker process exited before steer")
        return None
    except OSError:
        return None
    instance.refresh_from_db()
    if instance.status not in (
        CodexInstance.STATUS_STARTING,
        CodexInstance.STATUS_RUNNING,
    ):
        return None
    return instance


def _append_control_request(instance: CodexInstance, payload: dict[str, str]) -> None:
    path = control_path_for(instance)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, separators=(",", ":")) + "\n"
    with path.open("ab") as fh:
        fh.write(line.encode("utf-8"))


def _interrupt_instance(instance: CodexInstance) -> CodexInstance | None:
    """Stop one worker, escalating SIGTERM → SIGKILL on a second click.

    The first Stop request sends SIGTERM to the worker (not its group).
    The worker's signal handler defers to a flag the stream loop checks
    between events; on observing it, the loop calls the SDK's
    ``turn.interrupt()`` — a graceful cancellation that lets the
    app-server emit remaining events (notably a ``turn/completed`` with
    status ``interrupted``) before the worker writes its own terminal
    row status. The row's ``interrupt_requested_at`` is set so we can
    tell, on a subsequent click, that polite cancellation was already
    attempted.

    A second click on the still-active row escalates: we send SIGKILL
    to the process group, taking down both the worker and its codex
    app-server child, and write a terminal status ourselves since the
    worker no longer has the chance to.

    PID safety (``_pid_is_our_worker``): every signaling path is gated
    on the cmdline-verified identity check so a recycled pid never
    receives a signal meant for our worker.

    Returns the refreshed instance, or None when the worker is still
    launching (pid unset), a non-ESRCH signal failure left it running,
    or it raced us to a terminal status.
    """
    if instance.pid <= 0:
        # Launch race: the parent has created the DB row but not yet
        # written ``pid`` (Popen hasn't returned, or the row is fresh
        # from spawn_new_session). The codex_worker subprocess's first
        # action is to reset ``status`` to RUNNING, so flipping to
        # FAILED here would be silently undone and the turn would
        # continue running despite the user's stop click. Treat as
        # not-yet-interruptible; the user can retry after a moment.
        return None

    if not _pid_is_our_worker(instance.pid, instance.pk):
        # PID gone, recycled, or owned by an unrelated process. No safe
        # target for either SIGTERM or SIGKILL, but the leftover row
        # still has to be flipped to failed so the UI exits streaming
        # mode rather than waiting on ``reconcile_dead``.
        return _mark_failed(instance, "interrupted by user")

    if instance.interrupt_requested_at is None:
        # First click: polite interrupt. Signal only the worker (not
        # the group) — the worker's handler turns this into an SDK
        # ``turn.interrupt()`` and lets the app-server emit its
        # remaining events. Status will be updated by the worker
        # itself when the stream completes; if we wrote a terminal
        # status now the worker's later save would silently overwrite
        # it, so we leave the row alone except for the timestamp that
        # marks "polite stop already issued" for the escalation path.
        try:
            os.kill(instance.pid, signal.SIGTERM)
        except ProcessLookupError:
            # Worker exited between the identity check and the signal;
            # treat as if it had finished on its own and let the row
            # be reconciled to failed below.
            return _mark_failed(instance, "interrupted by user")
        except OSError:
            # EPERM (or any other non-ESRCH failure): the worker is
            # still alive but we could not signal it. Don't lie that
            # we stopped the turn.
            return None
        CodexInstance.objects.filter(pk=instance.pk).update(
            interrupt_requested_at=timezone.now()
        )
        instance.refresh_from_db()
        return instance

    # Second click on a still-active row: the polite interrupt didn't
    # take. Escalate to SIGKILL on the whole process group so the
    # worker AND its in-process codex app-server child both die
    # immediately, then write a terminal status ourselves — the
    # worker no longer gets to run its end-of-turn save.
    try:
        os.killpg(instance.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        return None
    return _mark_failed(instance, "forcibly stopped by user")


def _mark_failed(instance: CodexInstance, error: str) -> CodexInstance | None:
    """Conditionally flip a still-active row to FAILED with ``error``.

    Atomic UPDATE keyed on the active statuses so a worker that
    legitimately reached a terminal state in the gap between the row
    read and this call is preserved — preventing a stop click from
    retroactively rewriting a completed turn as failed.
    """
    updated = CodexInstance.objects.filter(
        pk=instance.pk,
        status__in=(CodexInstance.STATUS_STARTING, CodexInstance.STATUS_RUNNING),
    ).update(
        status=CodexInstance.STATUS_FAILED,
        ended_at=timezone.now(),
        error=error,
    )
    if updated == 0:
        return None
    instance.refresh_from_db()
    return instance


def _pid_is_our_worker(pid: int, instance_id: int) -> bool:
    """Return whether ``pid`` is still *our* worker for ``instance_id``.

    Two-layer identity check:

    1. ``os.getsid(pid) == pid`` — workers are spawned with
       ``start_new_session=True`` so the worker pid is by definition
       its own session id. A recycled pid owned by a non-session-leader
       fails this cheap check.

    2. ``/proc/<pid>/cmdline`` must contain ``codex_worker
       --instance-id <pk>`` — distinguishes our worker from any other
       session leader (e.g. a shell, another tool's daemon) that
       happens to inherit the recycled pid. Without this layer the
       getsid check alone could match any session leader and the
       signal would terminate an unrelated process group.

    On systems without /proc (non-Linux), the cmdline layer is
    unavailable; we fall back to trusting the getsid check. The
    deployment target is Linux, so this is a best-effort branch for
    local-dev on macOS where pid recycling within milliseconds is
    rare in practice.
    """
    try:
        if os.getsid(pid) != pid:
            return False
    except OSError:
        return False
    proc_root = Path("/proc")
    try:
        cmdline = (proc_root / str(pid) / "cmdline").read_bytes()
    except FileNotFoundError:
        # FileNotFoundError is ambiguous: it can mean ``/proc`` itself
        # doesn't exist (non-Linux dev — fall back to trusting the
        # session-leader check above) OR ``/proc/<pid>`` vanished
        # between getsid() and now (Linux race: worker exited and the
        # pid is at risk of being recycled — we MUST reject to avoid
        # signaling whoever inherits the pid next). The presence of
        # ``/proc`` itself disambiguates: on Linux we always have
        # ``/proc`` even when a specific pid entry disappears.
        return not proc_root.exists()
    except OSError:
        return False
    parts = cmdline.split(b"\0")
    if b"codex_worker" not in parts:
        return False
    try:
        idx = parts.index(b"--instance-id")
    except ValueError:
        return False
    return idx + 1 < len(parts) and parts[idx + 1] == str(instance_id).encode()


def reconcile_dead() -> int:
    """Mark workers as failed whose PID is no longer alive.

    A worker that crashed before writing its terminal status leaves a row
    stuck in ``starting``/``running``. We sweep those rows and mark them
    failed so the UI doesn't show a permanently-pending turn.
    """
    _reap_finished_workers()
    pending = CodexInstance.objects.filter(
        status__in=(CodexInstance.STATUS_STARTING, CodexInstance.STATUS_RUNNING)
    )
    updated = 0
    now = timezone.now()
    for instance in pending:
        if worker_is_alive(instance):
            continue
        instance.status = CodexInstance.STATUS_FAILED
        if not instance.error:
            instance.error = "worker process exited before reporting completion"
        instance.ended_at = now
        instance.save(update_fields=["status", "error", "ended_at"])
        _notify_system_agents_if_needed(instance)
        updated += 1
    _prune_reaped_workers()
    return updated


def _prune_reaped_workers() -> None:
    with _TRACKED_WORKER_PROCS_LOCK:
        reaped_workers = set(_REAPED_WORKERS)
    if not reaped_workers:
        return
    active_workers = set(
        CodexInstance.objects.filter(
            pk__in=[instance_id for _, instance_id in reaped_workers],
            status__in=(CodexInstance.STATUS_STARTING, CodexInstance.STATUS_RUNNING),
        ).values_list("pid", "pk")
    )
    with _TRACKED_WORKER_PROCS_LOCK:
        _REAPED_WORKERS.intersection_update(active_workers)


def _notify_system_agents_if_needed(instance: CodexInstance) -> None:
    if instance.purpose in (
        CodexInstance.PURPOSE_SYSTEM_AGENT,
        CodexInstance.PURPOSE_SYSTEM_FEEDBACK,
    ):
        try:
            from hitch.main import system_agents

            system_agents.on_codex_instance_finished(instance)
        except Exception:
            logger.exception(
                "failed to notify system workflow for reconciled instance %s",
                instance.pk,
            )
    try:
        from hitch.main import demo

        demo.on_codex_instance_finished(instance)
    except Exception:
        logger.exception(
            "failed to notify demo workflow for reconciled instance %s",
            instance.pk,
        )


def events_dir() -> Path:
    """Filesystem directory holding per-worker JSONL event logs."""
    configured = getattr(settings, "CODEX_EVENTS_DIR", None)
    if configured is not None:
        return Path(configured)
    return Path.home() / ".hitch" / "codex_events"


def _codex_bin() -> str | None:
    """Resolve the ``codex`` binary path for AppServerConfig.

    Returning None lets AppServerConfig fall back to the pinned runtime
    package; we prefer an explicit PATH lookup so dev environments with a
    locally-built codex pick that up.
    """
    return shutil.which("codex")


def app_server_config(*, enable_memories: bool = False) -> AppServerConfig:
    memory_value = "true" if enable_memories else "false"
    overrides = (f"features.memories={memory_value}",)
    return AppServerConfig(codex_bin=_codex_bin(), config_overrides=overrides)


def _spawn_worker(
    *,
    thread_id: str,
    cwd: str,
    prompt: str,
    base_instructions: str | None = None,
    developer_instructions: str | None = None,
    model: str | None = None,
    stored_model: str | None = None,
    reasoning_effort: str | None = None,
    stored_reasoning_effort: str | None = None,
    sandbox_policy: str | None = None,
    approval_mode: str | None = None,
    enable_memories: bool = False,
    collaboration_mode: str | None = None,
    plan_mode: bool = False,
    purpose: str = CodexInstance.PURPOSE_USER,
    workflow_id: int | None = None,
    agent_kind: str = "",
    display_author: str = "",
    output_schema: dict[str, Any] | None = None,
    user_message_index: int | None = None,
    auto_pr_enabled: bool = False,
) -> CodexInstance:
    target_dir = events_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    with transaction.atomic():
        instance = CodexInstance.objects.create(
            thread_id=thread_id,
            cwd=cwd,
            prompt=prompt,
            base_instructions=base_instructions or "",
            developer_instructions=developer_instructions or "",
            enable_memories=enable_memories,
            model=(stored_model if stored_model is not None else model) or "",
            reasoning_effort=(
                stored_reasoning_effort
                if stored_reasoning_effort is not None
                else reasoning_effort
            )
            or "",
            sandbox_policy=sandbox_policy or "",
            approval_mode=approval_mode or "",
            plan_mode=plan_mode,
            auto_pr_enabled=auto_pr_enabled,
            events_path="",
            status=CodexInstance.STATUS_STARTING,
            pid=0,
            purpose=purpose,
            workflow_id=workflow_id,
            agent_kind=agent_kind,
            display_author=display_author,
            output_schema=output_schema,
            user_message_index=user_message_index,
        )
        instance.events_path = str(target_dir / f"{instance.pk}.jsonl")
        instance.save(update_fields=["events_path"])

    try:
        launch_kwargs: dict[str, Any] = {
            "instance_id": instance.pk,
            "reasoning_effort": reasoning_effort,
            "sandbox_policy": sandbox_policy,
            "approval_mode": approval_mode,
        }
        if enable_memories:
            launch_kwargs["enable_memories"] = True
        if model or plan_mode:
            launch_kwargs["model"] = model
            launch_kwargs["plan_mode"] = plan_mode
        if collaboration_mode:
            launch_kwargs["collaboration_mode"] = collaboration_mode
        proc = _launch_worker_process(**launch_kwargs)
    except Exception as exc:
        # Without this, a Popen failure (e.g. ENOMEM, E2BIG, missing python)
        # would leave the row stuck in ``starting`` with pid=0 and no
        # subprocess will ever update it.
        instance.status = CodexInstance.STATUS_FAILED
        instance.ended_at = timezone.now()
        instance.error = f"failed to launch worker process: {exc!r}"
        instance.save(update_fields=["status", "ended_at", "error"])
        raise
    instance.pid = proc.pid
    instance.save(update_fields=["pid"])
    if isinstance(proc, subprocess.Popen):
        _track_worker_process(instance.pk, proc)
    return instance


def _launch_worker_process(
    *,
    instance_id: int,
    model: str | None = None,
    reasoning_effort: str | None = None,
    sandbox_policy: str | None = None,
    approval_mode: str | None = None,
    enable_memories: bool = False,
    collaboration_mode: str | None = None,
    plan_mode: bool = False,
) -> subprocess.Popen[bytes]:
    manage_py = str(Path(settings.BASE_DIR) / "manage.py")
    env = os.environ.copy()
    # Django needs an explicit settings module since hitch ships per-env
    # settings files; inherit whatever the parent process is running.
    if settings.SETTINGS_MODULE:
        env["DJANGO_SETTINGS_MODULE"] = settings.SETTINGS_MODULE

    argv = [
        sys.executable,
        manage_py,
        "codex_worker",
        "--instance-id",
        str(instance_id),
    ]
    if reasoning_effort:
        # Passed as a CLI arg rather than read from a request-side store so
        # the worker stays self-contained: the parent dies, the worker
        # already has every input it needs to finish the turn.
        argv.extend(["--reasoning-effort", reasoning_effort])
    if model:
        argv.extend(["--model", model])
    if sandbox_policy:
        argv.extend(["--sandbox-policy", sandbox_policy])
    if approval_mode:
        argv.extend(["--approval-mode", approval_mode])
    if enable_memories:
        argv.append("--enable-memories")
    if collaboration_mode:
        argv.extend(["--collaboration-mode", collaboration_mode])
    if plan_mode:
        argv.append("--plan-mode")

    return subprocess.Popen(
        argv,
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        close_fds=True,
    )
