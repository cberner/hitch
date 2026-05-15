"""Pool of detached Codex worker subprocesses.

Each worker runs a single turn for one Codex thread and then exits. Workers
are spawned with ``start_new_session=True`` (a fresh process group) so they
survive a Django restart: the parent's stdin/stdout/stderr are redirected to
``/dev/null`` and the child no longer inherits the parent's controlling
terminal. The CodexInstance row + JSONL events file on disk are the only
post-spawn link back to a worker.

The worker is the ``codex_worker`` management command in this app; running it
as a Django command lets it use the same ORM/settings as the parent without
re-implementing Django bootstrap.
"""

from __future__ import annotations

import os
import shutil
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
    model: str | None = None,
    reasoning_effort: str | None = None,
    sandbox_policy: str | None = None,
    approval_mode: str | None = None,
) -> CodexInstance:
    """Create a fresh Codex thread and detach a worker to run the initial prompt.

    ``model``, ``reasoning_effort``, ``sandbox_policy`` and ``approval_mode``
    come from the settings cookies the request handler reads; ``None`` means
    "let Codex apply its own default." The thread is created synchronously
    (so the caller has an id to redirect to immediately); the prompt itself
    is run by the detached worker.
    """
    config = AppServerConfig(codex_bin=_codex_bin())
    with Codex(config=config) as codex:
        thread = codex.thread_start(cwd=cwd, model=model)
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
        reasoning_effort=reasoning_effort,
        sandbox_policy=sandbox_policy,
        approval_mode=approval_mode,
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
    reasoning_effort: str | None = None,
    sandbox_policy: str | None = None,
    approval_mode: str | None = None,
) -> CodexInstance:
    """Detach a worker that resumes an existing thread to run one prompt."""
    return _spawn_worker(
        thread_id=thread_id,
        cwd=cwd,
        prompt=prompt,
        reasoning_effort=reasoning_effort,
        sandbox_policy=sandbox_policy,
        approval_mode=approval_mode,
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
    reasoning_effort: str | None = None,
    sandbox_policy: str | None = None,
    approval_mode: str | None = None,
) -> CodexInstance:
    target_dir = events_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    with transaction.atomic():
        instance = CodexInstance.objects.create(
            thread_id=thread_id,
            cwd=cwd,
            prompt=prompt,
            events_path="",
            status=CodexInstance.STATUS_STARTING,
            pid=0,
        )
        instance.events_path = str(target_dir / f"{instance.pk}.jsonl")
        instance.save(update_fields=["events_path"])

    try:
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
    reasoning_effort: str | None = None,
    sandbox_policy: str | None = None,
    approval_mode: str | None = None,
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
    if sandbox_policy:
        argv.extend(["--sandbox-policy", sandbox_policy])
    if approval_mode:
        argv.extend(["--approval-mode", approval_mode])

    return subprocess.Popen(
        argv,
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        close_fds=True,
    )
