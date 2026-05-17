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
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from django.conf import settings
from django.db import transaction
from django.utils import timezone
from openai_codex import AppServerConfig, Codex

from hitch.main.models import CodexInstance


def spawn_new_session(
    *,
    cwd: str,
    prompt: str,
    developer_instructions: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    sandbox_policy: str | None = None,
    approval_mode: str | None = None,
    plan_mode: bool = False,
) -> CodexInstance:
    """Create a fresh Codex thread and detach a worker to run the initial prompt.

    ``developer_instructions`` maps to the Codex SDK's per-thread
    ``developerInstructions`` field; the remaining overrides come from the
    settings cookies the request handler reads. ``None`` means "let Codex
    apply its own default." The thread is created synchronously (so the caller
    has an id to redirect to immediately); the prompt itself is run by the
    detached worker.
    """
    config = AppServerConfig(codex_bin=_codex_bin())
    with Codex(config=config) as codex:
        thread = codex.thread_start(
            cwd=cwd,
            developer_instructions=developer_instructions,
            model=model,
        )
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
        # once the first turn streams in, so this is invisible in the UI.
        codex._client.thread_set_name(thread_id, _initial_thread_name(prompt))
    return _spawn_worker(
        thread_id=thread_id,
        cwd=cwd,
        prompt=prompt,
        developer_instructions=developer_instructions,
        model=model if plan_mode else None,
        reasoning_effort=reasoning_effort,
        sandbox_policy=sandbox_policy,
        approval_mode=approval_mode,
        plan_mode=plan_mode,
    )


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
    reasoning_effort: str | None = None,
    sandbox_policy: str | None = None,
    approval_mode: str | None = None,
    plan_mode: bool = False,
) -> CodexInstance:
    """Detach a worker that resumes an existing thread to run one prompt."""
    previous = latest_for_thread(thread_id)
    developer_instructions = (
        previous.developer_instructions if previous is not None else None
    )
    return _spawn_worker(
        thread_id=thread_id,
        cwd=cwd,
        prompt=prompt,
        developer_instructions=developer_instructions or None,
        model=model,
        reasoning_effort=reasoning_effort,
        sandbox_policy=sandbox_policy,
        approval_mode=approval_mode,
        plan_mode=plan_mode,
    )


def is_alive(pid: int) -> bool:
    """Return whether ``pid`` is currently a live process on this host.

    PIDs can be recycled, so the answer is only meaningful when combined with
    a recent CodexInstance.started_at. A false reading triggers a status
    reconciliation; a true reading is treated as best-effort.
    """
    if pid <= 0:
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
    return True


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
    pending = CodexInstance.objects.filter(
        status__in=(CodexInstance.STATUS_STARTING, CodexInstance.STATUS_RUNNING)
    )
    updated = 0
    now = timezone.now()
    for instance in pending:
        if is_alive(instance.pid):
            continue
        instance.status = CodexInstance.STATUS_FAILED
        if not instance.error:
            instance.error = "worker process exited before reporting completion"
        instance.ended_at = now
        instance.save(update_fields=["status", "error", "ended_at"])
        updated += 1
    return updated


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


def _spawn_worker(
    *,
    thread_id: str,
    cwd: str,
    prompt: str,
    developer_instructions: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    sandbox_policy: str | None = None,
    approval_mode: str | None = None,
    plan_mode: bool = False,
) -> CodexInstance:
    target_dir = events_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    with transaction.atomic():
        instance = CodexInstance.objects.create(
            thread_id=thread_id,
            cwd=cwd,
            prompt=prompt,
            developer_instructions=developer_instructions or "",
            events_path="",
            status=CodexInstance.STATUS_STARTING,
            pid=0,
        )
        instance.events_path = str(target_dir / f"{instance.pk}.jsonl")
        instance.save(update_fields=["events_path"])

    try:
        if model or plan_mode:
            proc = _launch_worker_process(
                instance_id=instance.pk,
                model=model,
                reasoning_effort=reasoning_effort,
                sandbox_policy=sandbox_policy,
                approval_mode=approval_mode,
                plan_mode=plan_mode,
            )
        else:
            proc = _launch_worker_process(
                instance_id=instance.pk,
                reasoning_effort=reasoning_effort,
                sandbox_policy=sandbox_policy,
                approval_mode=approval_mode,
            )
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
    return instance


def _launch_worker_process(
    *,
    instance_id: int,
    model: str | None = None,
    reasoning_effort: str | None = None,
    sandbox_policy: str | None = None,
    approval_mode: str | None = None,
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
