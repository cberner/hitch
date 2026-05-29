"""Pool of detached Codex worker subprocesses.

Each worker runs a single turn for one Codex thread and then exits. Workers
are launched outside the Django process tree, preferably inside a per-worker
systemd user scope with its own memory cgroup. The CodexInstance row + JSONL
events file on disk are the durable post-spawn links back to a worker; a
sibling control JSONL file carries mid-turn requests such as steer payloads
into the detached process.

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
import tempfile
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

from django.conf import settings
from django.db import transaction
from django.utils import timezone
from openai_codex import AppServerConfig, Codex
from openai_codex.generated.v2_all import ThreadSource, WebSearchMode

from hitch.main.codex_tools import registered_dynamic_tool_specs
from hitch.main.models import CodexInstance

logger = logging.getLogger(__name__)

_TRACKED_WORKER_PROCS: dict[int, tuple[int, subprocess.Popen[bytes]]] = {}
_REAPED_WORKERS: set[tuple[int, int]] = set()
_TRACKED_WORKER_PROCS_LOCK = threading.Lock()
_VALID_WEB_SEARCH_MODES = frozenset(mode.value for mode in WebSearchMode)
_MAX_INPUT_ATTACHMENT_PATHS_PER_INSTANCE = 16
_MAX_INPUT_ATTACHMENT_PATHS_PER_THREAD = 64
# How long a freshly spawned row may sit with pid=0 before reconcile_dead
# treats it as orphaned. ``_spawn_worker`` commits the row before
# ``subprocess.Popen`` returns, so a transient pid=0 window is a normal part
# of the launch handshake; the grace is generous enough to absorb a slow Popen
# on a loaded host but bounded so a parent that crashed mid-spawn cannot leave
# the row pending forever.
_PID_ASSIGNMENT_GRACE = timedelta(minutes=2)
_WORKER_ISOLATION_AUTO = "auto"
_WORKER_ISOLATION_DIRECT = "direct"
_WORKER_ISOLATION_SYSTEMD = "systemd"


@dataclass(frozen=True)
class WorkerLaunch:
    """Result of launching a detached worker or its systemd-run wrapper."""

    pid: int
    proc: subprocess.Popen[bytes] | None = None
    scope_unit: str = ""


def _normalized_web_search_mode(web_search_mode: str | None) -> str | None:
    if web_search_mode is None:
        return None
    value = web_search_mode.strip()
    if not value:
        return None
    if value not in _VALID_WEB_SEARCH_MODES:
        raise ValueError(f"invalid web_search_mode: {web_search_mode!r}")
    return value


class InputAttachmentLimitExceededError(Exception):
    """Raised when one active worker already owns too many input images."""


def spawn_new_session(
    *,
    cwd: str,
    prompt: str,
    input_image_paths: list[str] | None = None,
    thread_name: str | None = None,
    base_instructions: str | None = None,
    developer_instructions: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    sandbox_policy: str | None = None,
    approval_mode: str | None = None,
    web_search_mode: str | None = None,
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
    auto_qa_enabled: bool = False,
    qa_panel_enabled: bool = False,
    auto_merge_to_local_branch: bool = False,
    auto_merge_branch: str = "",
) -> CodexInstance:
    """Create a fresh Codex thread and detach a worker to run the initial prompt.

    ``developer_instructions`` maps to the Codex SDK's per-thread
    ``developerInstructions`` field; the remaining overrides come from the
    settings cookies the request handler reads. ``None`` means "let Codex
    apply its own default." The thread is created synchronously (so the caller
    has an id to redirect to immediately); the prompt itself is run by the
    detached worker.
    """
    config = app_server_config(
        enable_memories=enable_memories,
        web_search_mode=web_search_mode,
    )
    with Codex(config=config) as codex:
        start_kwargs: dict[str, Any] = {
            "cwd": cwd,
            "developerInstructions": developer_instructions,
            "model": model,
        }
        if base_instructions:
            start_kwargs["baseInstructions"] = base_instructions
        if thread_source is not None:
            start_kwargs["threadSource"] = thread_source.value
        if purpose == CodexInstance.PURPOSE_USER:
            start_kwargs["dynamicTools"] = registered_dynamic_tool_specs()
        response = codex._client.thread_start(start_kwargs)
        thread_id = response.thread.id
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
        input_image_paths=input_image_paths,
        base_instructions=base_instructions,
        developer_instructions=developer_instructions,
        model=model if plan_mode else None,
        stored_model=model,
        reasoning_effort=reasoning_effort,
        sandbox_policy=sandbox_policy,
        approval_mode=approval_mode,
        web_search_mode=web_search_mode,
        enable_memories=enable_memories,
        plan_mode=plan_mode,
        purpose=purpose,
        workflow_id=workflow_id,
        agent_kind=agent_kind,
        display_author=display_author,
        output_schema=output_schema,
        user_message_index=user_message_index,
        auto_pr_enabled=auto_pr_enabled,
        auto_qa_enabled=auto_qa_enabled,
        qa_panel_enabled=qa_panel_enabled,
        auto_merge_to_local_branch=auto_merge_to_local_branch,
        auto_merge_branch=auto_merge_branch,
    )


def create_session_thread(
    *,
    cwd: str,
    name: str,
    base_instructions: str | None = None,
    developer_instructions: str | None = None,
    model: str | None = None,
    enable_memories: bool = False,
    web_search_mode: str | None = None,
) -> str:
    """Create and persist a visible Codex thread without starting a turn."""
    config = app_server_config(
        enable_memories=enable_memories,
        web_search_mode=web_search_mode,
    )
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
    Image-only sessions can intentionally pass an empty prompt, so fall back
    to a stable title rather than sending Codex a whitespace-only name.
    """
    first_line = prompt.split("\n", 1)[0].strip()[:_INITIAL_THREAD_NAME_MAX_LEN].rstrip()
    return first_line or "New session"


def spawn_turn(
    *,
    thread_id: str,
    cwd: str,
    prompt: str,
    input_image_paths: list[str] | None = None,
    model: str | None = None,
    stored_model: str | None = None,
    reasoning_effort: str | None = None,
    stored_reasoning_effort: str | None = None,
    sandbox_policy: str | None = None,
    approval_mode: str | None = None,
    web_search_mode: str | None = None,
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
    auto_qa_enabled: bool = False,
    qa_panel_enabled: bool = False,
    auto_merge_to_local_branch: bool = False,
    auto_merge_branch: str = "",
) -> CodexInstance:
    """Detach a worker that resumes an existing thread to run one prompt.

    Per-turn settings are owned by the caller. Only thread-scoped instruction
    text is copied from prior rows; omitted tool/config values mean Codex
    default for this turn, not "inherit the last worker row."
    """
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
        input_image_paths=input_image_paths,
        base_instructions=base_instructions or None,
        developer_instructions=developer_instructions or None,
        model=model,
        stored_model=stored_model,
        reasoning_effort=reasoning_effort,
        stored_reasoning_effort=stored_reasoning_effort,
        sandbox_policy=sandbox_policy,
        approval_mode=approval_mode,
        web_search_mode=web_search_mode,
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
        auto_qa_enabled=auto_qa_enabled,
        qa_panel_enabled=qa_panel_enabled,
        auto_merge_to_local_branch=auto_merge_to_local_branch,
        auto_merge_branch=auto_merge_branch,
    )


def is_alive(pid: int) -> bool:
    """Return whether ``pid`` is currently a live process on this host.

    PIDs can be recycled, so the answer is only meaningful when combined with
    a recent CodexInstance.started_at. A false reading triggers a status
    reconciliation; a true reading is treated as best-effort.
    """
    if pid <= 0:
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
    """Return whether ``instance`` still has its original live worker process.

    Generic pid existence is not enough here: a crashed worker's pid can be
    recycled to an unrelated process while the CodexInstance row is still
    marked running.

    A row with ``pid <= 0`` and a recent ``started_at`` is in the spawn
    handshake window: ``_spawn_worker`` commits the row before
    ``subprocess.Popen`` returns the real pid. Treating that as "dead" lets a
    concurrent ``reconcile_dead`` overwrite a still-launching worker with a
    terminal status and (for system-agent purposes) route the row through its
    workflow's failure handler before the worker has even started.
    """
    if instance.pid <= 0:
        started_at = instance.started_at
        if started_at is None:
            return False
        return started_at >= timezone.now() - _PID_ASSIGNMENT_GRACE
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
    return _pid_is_instance_worker(instance)


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


def steer_active(
    thread_id: str, *, prompt: str, input_image_paths: list[str] | None = None
) -> CodexInstance | None:
    """Inject ``prompt`` into the most recent active worker for ``thread_id``."""
    instance = latest_active_for_thread(thread_id)
    if instance is None:
        return None
    kwargs: dict[str, Any] = {"prompt": prompt}
    if input_image_paths:
        kwargs["input_image_paths"] = input_image_paths
    return _steer_instance(instance, **kwargs)


def steer_instance(
    instance_id: int,
    *,
    expected_thread_id: str,
    prompt: str,
    input_image_paths: list[str] | None = None,
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
    kwargs: dict[str, Any] = {"prompt": prompt}
    if input_image_paths:
        kwargs["input_image_paths"] = input_image_paths
    return _steer_instance(instance, **kwargs)


def control_path_for(instance: CodexInstance) -> Path:
    """Return the per-worker control JSONL path next to its events log."""
    events_path = Path(instance.events_path)
    return events_path.with_name(f"{events_path.stem}.control.jsonl")


def _steer_instance(
    instance: CodexInstance,
    *,
    prompt: str,
    input_image_paths: list[str] | None = None,
) -> CodexInstance | None:
    """Queue one steer request for ``instance`` and nudge its worker.

    The payload is appended before the signal so the worker never wakes up to
    an empty control channel. If the worker is still in ``starting`` we skip
    SIGUSR1: the handler may not be installed yet, and the worker drains the
    file once the ``TurnHandle`` exists.
    """
    image_paths = _normalized_input_image_paths(input_image_paths)
    if instance.pid <= 0:
        return None
    if not prompt.strip() and not image_paths:
        return None
    identity_required = not (
        instance.systemd_scope_unit and instance.status != CodexInstance.STATUS_RUNNING
    )
    if identity_required and not _pid_is_instance_worker(instance):
        _mark_failed(instance, "worker process unavailable for steer")
        return None

    images_tracked = False
    if not _track_steer_input_attachments(instance, image_paths):
        return None
    images_tracked = bool(image_paths)
    try:
        payload: dict[str, Any] = {
            "op": "steer",
            "input": prompt,
        }
        if image_paths:
            payload["inputImagePaths"] = image_paths
        _append_control_request(instance, payload)
    except OSError:
        if images_tracked:
            _remove_input_attachment_paths(instance, image_paths)
        return None
    if instance.status != CodexInstance.STATUS_RUNNING:
        instance.refresh_from_db()
        if instance.status not in (
            CodexInstance.STATUS_STARTING,
            CodexInstance.STATUS_RUNNING,
        ):
            if images_tracked:
                _remove_input_attachment_paths(instance, image_paths)
            return None
        return instance
    try:
        os.kill(instance.pid, signal.SIGUSR1)
    except ProcessLookupError:
        if images_tracked:
            _remove_input_attachment_paths(instance, image_paths)
        _mark_failed(instance, "worker process exited before steer")
        return None
    except OSError:
        return instance
    instance.refresh_from_db()
    if instance.status not in (
        CodexInstance.STATUS_STARTING,
        CodexInstance.STATUS_RUNNING,
    ):
        if images_tracked:
            _remove_input_attachment_paths(instance, image_paths)
        return None
    return instance


def _append_control_request(instance: CodexInstance, payload: dict[str, Any]) -> None:
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
    if instance.systemd_scope_unit and instance.status == CodexInstance.STATUS_STARTING:
        # The stored pid is still the systemd-run wrapper until the worker
        # records its real pid. Do not treat the wrapper as interruptible.
        return None

    if not _pid_is_instance_worker(instance):
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
        _force_kill_instance(instance)
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


def _force_kill_instance(instance: CodexInstance) -> None:
    if instance.systemd_scope_unit:
        systemctl = shutil.which("systemctl")
        if systemctl is None:
            raise OSError("systemctl is required to kill scoped Codex workers")
        result = subprocess.run(
            [
                systemctl,
                "--user",
                "kill",
                "--kill-whom=all",
                "--signal=SIGKILL",
                instance.systemd_scope_unit,
            ],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if result.returncode == 0:
            return
        if _systemd_scope_is_missing(systemctl, instance.systemd_scope_unit):
            raise ProcessLookupError
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        message = "systemctl failed to kill scoped Codex worker"
        if detail:
            message = f"{message}: {detail}"
        raise OSError(message)
    os.killpg(instance.pid, signal.SIGKILL)


def _systemd_scope_is_missing(systemctl: str, scope_unit: str) -> bool:
    try:
        result = subprocess.run(
            [systemctl, "--user", "show", scope_unit, "--property=LoadState", "--value"],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    if result.returncode != 0:
        return False
    return result.stdout.decode("utf-8", errors="replace").strip() in {"", "not-found"}


def _pid_is_instance_worker(instance: CodexInstance) -> bool:
    if instance.systemd_scope_unit:
        return _pid_is_our_worker(
            instance.pid,
            instance.pk,
            require_session_leader=False,
        )
    return _pid_is_our_worker(instance.pid, instance.pk)


def _pid_is_our_worker(
    pid: int, instance_id: int, *, require_session_leader: bool = True
) -> bool:
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
    if require_session_leader:
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
        return require_session_leader and not proc_root.exists()
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
    updated = _mark_dead_instances_failed(pending)
    _reconcile_terminal_workflow_instances()
    retry_failed_input_image_cleanups()
    _prune_reaped_workers()
    return updated


def reconcile_dead_for_workflow(
    workflow_id: int, *, main_thread_id: str | None = None
) -> int:
    """Mark dead workers for one workflow without sweeping every session."""
    _reap_finished_workers()
    pending = CodexInstance.objects.filter(
        workflow_id=workflow_id,
        status__in=(CodexInstance.STATUS_STARTING, CodexInstance.STATUS_RUNNING),
    )
    updated = _mark_dead_instances_failed(pending)
    _reconcile_terminal_workflow_instances(
        main_thread_id=main_thread_id,
        workflow_id=workflow_id,
    )
    _prune_reaped_workers()
    return updated


def _mark_dead_instances_failed(pending: Iterable[CodexInstance]) -> int:
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
        cleanup_requested_input_images_for(instance)
        updated += 1
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
    system_agents_handled = False
    if instance.purpose in (
        CodexInstance.PURPOSE_SYSTEM_AGENT,
        CodexInstance.PURPOSE_SYSTEM_FEEDBACK,
    ) or (
        instance.purpose == CodexInstance.PURPOSE_USER
        and instance.workflow_id is not None
    ):
        try:
            from hitch.main import system_agents

            system_agents_handled = system_agents.on_codex_instance_finished(instance)
        except Exception:
            logger.exception(
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
        logger.exception(
            "failed to notify demo workflow for reconciled instance %s",
            instance.pk,
        )


def _reconcile_terminal_workflow_instances(
    *, main_thread_id: str | None = None, workflow_id: int | None = None
) -> None:
    try:
        from hitch.main import system_agents

        system_agents.reconcile_terminal_workflow_instances(
            main_thread_id=main_thread_id,
            workflow_id=workflow_id,
        )
    except Exception:
        logger.exception("failed to reconcile terminal workflow instances")


def events_dir() -> Path:
    """Filesystem directory holding per-worker JSONL event logs."""
    configured = getattr(settings, "CODEX_EVENTS_DIR", None)
    if configured is not None:
        return Path(configured)
    return Path.home() / ".hitch" / "codex_events"


def input_attachments_dir() -> Path:
    """Filesystem directory holding uploaded local-image inputs."""
    return events_dir() / "attachments"


def cleanup_input_images_for(instance: CodexInstance) -> None:
    """Delete image files for an explicit session-retention boundary.

    Uploaded images are submitted to Codex as local paths, and the Codex
    thread can reference those paths on later resumes. Terminal turn status is
    therefore not a cleanup boundary; callers should invoke this only when the
    session/thread no longer needs those local-image references.
    """
    current = CodexInstance.objects.filter(pk=instance.pk).first()
    if current is None:
        return
    image_paths = _merged_input_image_paths(
        current.input_attachment_paths,
        current.input_image_paths,
    )
    if not image_paths:
        return
    root = input_attachments_dir().resolve(strict=False)
    remaining_paths: list[str] = []
    for raw_path in image_paths:
        path = Path(raw_path).resolve(strict=False)
        if not _path_within(path, root):
            logger.warning(
                "refusing to clean up input image outside attachment dir: %s",
                raw_path,
            )
            remaining_paths.append(raw_path)
            continue
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("failed to clean up input image attachment %s", path)
            remaining_paths.append(raw_path)
        else:
            _prune_empty_attachment_dirs(path.parent, root)
    CodexInstance.objects.filter(pk=instance.pk).update(
        input_image_paths=[],
        input_attachment_paths=remaining_paths,
        input_attachment_cleanup_requested=bool(remaining_paths),
    )
    instance.input_image_paths = []
    instance.input_attachment_paths = remaining_paths
    instance.input_attachment_cleanup_requested = bool(remaining_paths)


def cleanup_input_images_for_thread(thread_id: str) -> None:
    """Delete retained input images for every turn in a thread."""
    CodexInstance.objects.filter(
        thread_id=thread_id,
        status__in=(CodexInstance.STATUS_STARTING, CodexInstance.STATUS_RUNNING),
    ).exclude(input_attachment_paths=[]).update(
        input_attachment_cleanup_requested=True
    )
    terminal_instances = CodexInstance.objects.filter(
        thread_id=thread_id,
        status__in=(CodexInstance.STATUS_COMPLETED, CodexInstance.STATUS_FAILED),
    )
    for instance in terminal_instances:
        cleanup_input_images_for(instance)


def cleanup_requested_input_images_for(instance: CodexInstance) -> None:
    current = CodexInstance.objects.filter(pk=instance.pk).first()
    if current is None or not current.input_attachment_cleanup_requested:
        return
    cleanup_input_images_for(current)


def discard_input_attachment_paths(
    instance: CodexInstance, input_image_paths: list[str]
) -> None:
    """Delete undelivered steer images and release their attachment ledger paths."""
    image_paths = _normalized_input_image_paths(input_image_paths)
    if not image_paths:
        return
    root = input_attachments_dir().resolve(strict=False)
    removed_paths: list[str] = []
    failed_paths: list[str] = []
    for raw_path in image_paths:
        path = Path(raw_path).resolve(strict=False)
        if not _path_within(path, root):
            logger.warning(
                "refusing to discard input image outside attachment dir: %s",
                raw_path,
            )
            removed_paths.append(raw_path)
            continue
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("failed to discard input image attachment %s", path)
            failed_paths.append(raw_path)
        else:
            _prune_empty_attachment_dirs(path.parent, root)
            removed_paths.append(raw_path)
    if removed_paths:
        _remove_input_attachment_paths(instance, removed_paths)
    if failed_paths:
        CodexInstance.objects.filter(pk=instance.pk).update(
            input_attachment_cleanup_requested=True
        )
        instance.input_attachment_cleanup_requested = True


def retry_failed_input_image_cleanups() -> int:
    """Retry explicit attachment cleanups that previously failed.

    Rows with ``input_image_paths`` still set are retained because Codex may
    need those paths to resume the thread. ``cleanup_input_images_for`` clears
    that field when a real cleanup is requested, leaving failed unlinks in
    ``input_attachment_paths`` for this retry path.
    """
    retried = 0
    instances = CodexInstance.objects.filter(
        status__in=(
            CodexInstance.STATUS_COMPLETED,
            CodexInstance.STATUS_FAILED,
        ),
        input_attachment_cleanup_requested=True,
    ).exclude(input_attachment_paths=[])
    for instance in instances:
        cleanup_input_images_for(instance)
        retried += 1
    return retried


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _prune_empty_attachment_dirs(path: Path, root: Path) -> None:
    while path != root and _path_within(path, root):
        try:
            path.rmdir()
        except OSError:
            return
        path = path.parent


def _codex_bin() -> str | None:
    """Resolve the ``codex`` binary path for AppServerConfig.

    Returning None lets AppServerConfig fall back to the pinned runtime
    package; we prefer an explicit PATH lookup so dev environments with a
    locally-built codex pick that up.
    """
    return shutil.which("codex")


def app_server_config(
    *, enable_memories: bool = False, web_search_mode: str | None = None
) -> AppServerConfig:
    memory_value = "true" if enable_memories else "false"
    overrides = [f"features.memories={memory_value}"]
    web_search_mode = _normalized_web_search_mode(web_search_mode)
    if web_search_mode:
        overrides.append(f"web_search={json.dumps(web_search_mode)}")
    return AppServerConfig(codex_bin=_codex_bin(), config_overrides=tuple(overrides))


def _normalized_input_image_paths(input_image_paths: Any) -> list[str]:
    if not isinstance(input_image_paths, list):
        return []
    return [
        path for path in input_image_paths if isinstance(path, str) and path.strip()
    ]


def _merged_input_image_paths(*path_lists: Any) -> list[str]:
    merged: list[str] = []
    for path_list in path_lists:
        for path in _normalized_input_image_paths(path_list):
            if path not in merged:
                merged.append(path)
    return merged


def _input_attachment_count_for_thread(
    thread_id: str, *, exclude_instance_id: int | None = None
) -> int:
    query = CodexInstance.objects.filter(thread_id=thread_id)
    if exclude_instance_id is not None:
        query = query.exclude(pk=exclude_instance_id)
    paths: set[str] = set()
    for path_list in query.values_list("input_attachment_paths", flat=True):
        paths.update(_normalized_input_image_paths(path_list))
    return len(paths)


def _add_input_attachment_paths(
    instance: CodexInstance, input_image_paths: list[str]
) -> None:
    image_paths = _normalized_input_image_paths(input_image_paths)
    if not image_paths:
        return
    with transaction.atomic():
        current = CodexInstance.objects.select_for_update().get(pk=instance.pk)
        merged = _merged_input_image_paths(current.input_attachment_paths, image_paths)
        if len(merged) > _MAX_INPUT_ATTACHMENT_PATHS_PER_INSTANCE:
            raise InputAttachmentLimitExceededError(
                "too many image attachments are queued for this turn"
            )
        thread_total = _input_attachment_count_for_thread(
            current.thread_id,
            exclude_instance_id=current.pk,
        ) + len(merged)
        if thread_total > _MAX_INPUT_ATTACHMENT_PATHS_PER_THREAD:
            raise InputAttachmentLimitExceededError(
                "too many image attachments are retained for this session"
            )
        if merged != _normalized_input_image_paths(current.input_attachment_paths):
            current.input_attachment_paths = merged
            current.save(update_fields=["input_attachment_paths"])
    instance.input_attachment_paths = merged


def _remove_input_attachment_paths(
    instance: CodexInstance, input_image_paths: list[str]
) -> None:
    image_paths = set(_normalized_input_image_paths(input_image_paths))
    if not image_paths:
        return
    with transaction.atomic():
        current = CodexInstance.objects.select_for_update().get(pk=instance.pk)
        remaining = [
            path
            for path in _normalized_input_image_paths(current.input_attachment_paths)
            if path not in image_paths
        ]
        current.input_attachment_paths = remaining
        current.save(update_fields=["input_attachment_paths"])
    instance.input_attachment_paths = remaining


def _track_steer_input_attachments(
    instance: CodexInstance, input_image_paths: list[str]
) -> bool:
    image_paths = _normalized_input_image_paths(input_image_paths)
    if not image_paths:
        return True
    _add_input_attachment_paths(instance, image_paths)
    instance.refresh_from_db(fields=["status"])
    if instance.status in (CodexInstance.STATUS_STARTING, CodexInstance.STATUS_RUNNING):
        return True
    _remove_input_attachment_paths(instance, image_paths)
    return False


def _spawn_worker(
    *,
    thread_id: str,
    cwd: str,
    prompt: str,
    input_image_paths: list[str] | None = None,
    base_instructions: str | None = None,
    developer_instructions: str | None = None,
    model: str | None = None,
    stored_model: str | None = None,
    reasoning_effort: str | None = None,
    stored_reasoning_effort: str | None = None,
    sandbox_policy: str | None = None,
    approval_mode: str | None = None,
    web_search_mode: str | None = None,
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
    auto_qa_enabled: bool = False,
    qa_panel_enabled: bool = False,
    auto_merge_to_local_branch: bool = False,
    auto_merge_branch: str = "",
) -> CodexInstance:
    web_search_mode = _normalized_web_search_mode(web_search_mode)
    target_dir = events_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    normalized_input_image_paths = _normalized_input_image_paths(input_image_paths)
    if (
        normalized_input_image_paths
        and _input_attachment_count_for_thread(thread_id)
        + len(set(normalized_input_image_paths))
        > _MAX_INPUT_ATTACHMENT_PATHS_PER_THREAD
    ):
        raise InputAttachmentLimitExceededError(
            "too many image attachments are retained for this session"
        )

    with transaction.atomic():
        instance = CodexInstance.objects.create(
            thread_id=thread_id,
            cwd=cwd,
            prompt=prompt,
            input_image_paths=normalized_input_image_paths,
            input_attachment_paths=normalized_input_image_paths,
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
            web_search_mode=web_search_mode or "",
            plan_mode=plan_mode,
            auto_pr_enabled=auto_pr_enabled,
            auto_qa_enabled=auto_qa_enabled,
            qa_panel_enabled=qa_panel_enabled,
            auto_merge_to_local_branch=auto_merge_to_local_branch,
            auto_merge_branch=auto_merge_branch,
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
        if web_search_mode:
            launch_kwargs["web_search_mode"] = web_search_mode
        if enable_memories:
            launch_kwargs["enable_memories"] = True
        if model or plan_mode:
            launch_kwargs["model"] = model
            launch_kwargs["plan_mode"] = plan_mode
        if collaboration_mode:
            launch_kwargs["collaboration_mode"] = collaboration_mode
        launch = _launch_worker_process(**launch_kwargs)
    except Exception as exc:
        # Without this, a Popen failure (e.g. ENOMEM, E2BIG, missing python)
        # would leave the row stuck in ``starting`` with pid=0 and no
        # subprocess will ever update it.
        instance.status = CodexInstance.STATUS_FAILED
        instance.ended_at = timezone.now()
        instance.error = f"failed to launch worker process: {exc!r}"
        instance.save(update_fields=["status", "ended_at", "error"])
        cleanup_input_images_for(instance)
        raise
    update_fields: list[str] = []
    launch_pid = getattr(launch, "pid", 0)
    scope_unit = getattr(launch, "scope_unit", "")
    if launch_pid > 0:
        instance.pid = launch_pid
        update_fields.append("pid")
    if scope_unit:
        instance.systemd_scope_unit = scope_unit
        update_fields.append("systemd_scope_unit")
    if update_fields:
        instance.save(update_fields=update_fields)
    launch_proc = getattr(launch, "proc", None)
    if isinstance(launch_proc, subprocess.Popen):
        _track_worker_process(instance.pk, cast(subprocess.Popen[bytes], launch_proc))
    elif isinstance(launch, subprocess.Popen):
        _track_worker_process(instance.pk, cast(subprocess.Popen[bytes], launch))
    return instance


def _launch_worker_process(
    *,
    instance_id: int,
    model: str | None = None,
    reasoning_effort: str | None = None,
    sandbox_policy: str | None = None,
    approval_mode: str | None = None,
    web_search_mode: str | None = None,
    enable_memories: bool = False,
    collaboration_mode: str | None = None,
    plan_mode: bool = False,
) -> WorkerLaunch:
    web_search_mode = _normalized_web_search_mode(web_search_mode)
    env = os.environ.copy()
    # Django needs an explicit settings module since hitch ships per-env
    # settings files; inherit whatever the parent process is running.
    if settings.SETTINGS_MODULE:
        env["DJANGO_SETTINGS_MODULE"] = settings.SETTINGS_MODULE

    argv = _worker_argv(
        instance_id=instance_id,
        model=model,
        reasoning_effort=reasoning_effort,
        sandbox_policy=sandbox_policy,
        approval_mode=approval_mode,
        web_search_mode=web_search_mode,
        enable_memories=enable_memories,
        collaboration_mode=collaboration_mode,
        plan_mode=plan_mode,
    )

    requested_isolation = _worker_isolation()
    if requested_isolation == _WORKER_ISOLATION_DIRECT:
        proc = _popen_detached(argv, env=env)
        return WorkerLaunch(pid=proc.pid, proc=proc)
    if requested_isolation not in (
        _WORKER_ISOLATION_AUTO,
        _WORKER_ISOLATION_SYSTEMD,
    ):
        raise ValueError(f"invalid CODEX_WORKER_ISOLATION: {requested_isolation!r}")
    systemd_run = _systemd_run_for_isolation(requested_isolation)
    if systemd_run is None:
        proc = _popen_detached(argv, env=env)
        return WorkerLaunch(pid=proc.pid, proc=proc)
    return _launch_systemd_worker(
        systemd_run=systemd_run,
        scope_unit=_scope_unit_for_instance(instance_id),
        worker_argv=argv,
        env=env,
    )


def _systemd_run_for_isolation(requested_isolation: str) -> str | None:
    systemd_run = shutil.which("systemd-run")
    if systemd_run is None:
        if requested_isolation == _WORKER_ISOLATION_SYSTEMD:
            raise RuntimeError("systemd-run is required for Codex worker isolation")
        return None
    if requested_isolation == _WORKER_ISOLATION_AUTO and not _systemd_user_manager_available():
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
) -> WorkerLaunch:
    _ensure_systemd_worker_slice()
    with tempfile.TemporaryFile() as stderr_file:
        proc = _popen_detached(
            _systemd_scope_argv(
                systemd_run=systemd_run,
                scope_unit=scope_unit,
                worker_argv=worker_argv,
            ),
            env=env,
            stderr=stderr_file,
        )
        _raise_for_immediate_systemd_run_failure(proc, scope_unit, stderr_file)
    return WorkerLaunch(pid=proc.pid, proc=proc, scope_unit=scope_unit)


def _worker_slice() -> str:
    return str(getattr(settings, "CODEX_WORKER_SLICE", "") or "").strip()


def _memory_cgroup_properties(high_setting: str, max_setting: str) -> list[str]:
    """Build systemd memory-cgroup properties for the named settings.

    ``MemoryAccounting=yes`` must accompany any ``MemoryHigh``/``MemoryMax``:
    hosts with ``DefaultMemoryAccounting=no`` (or a legacy cgroup v1 hierarchy)
    silently ignore the limits unless accounting is explicitly enabled on the
    unit, so the cap would not actually bound the worker. The per-worker scope
    and the aggregate slice share this builder so their accounting/limit
    handling cannot drift apart.
    """
    high = str(getattr(settings, high_setting, "") or "").strip()
    hard = str(getattr(settings, max_setting, "") or "").strip()
    properties: list[str] = []
    if high or hard:
        properties.append("MemoryAccounting=yes")
    if high:
        properties.append(f"MemoryHigh={high}")
    if hard:
        properties.append(f"MemoryMax={hard}")
    return properties


def _systemd_worker_slice_properties() -> list[str]:
    return _memory_cgroup_properties(
        "CODEX_WORKER_SLICE_MEMORY_HIGH", "CODEX_WORKER_SLICE_MEMORY_MAX"
    )


def _ensure_systemd_worker_slice() -> None:
    slice_unit = _worker_slice()
    if not slice_unit:
        return
    properties = _systemd_worker_slice_properties()
    if not properties:
        return
    systemctl = shutil.which("systemctl")
    if systemctl is None:
        raise RuntimeError("systemctl is required to configure Codex worker slice")
    try:
        result = subprocess.run(
            [
                systemctl,
                "--user",
                "set-property",
                "--runtime",
                slice_unit,
                *properties,
            ],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            f"failed to configure Codex worker slice {slice_unit}"
        ) from exc
    if result.returncode == 0:
        return
    detail = result.stderr.decode("utf-8", errors="replace").strip()
    message = f"failed to configure Codex worker slice {slice_unit}"
    if detail:
        message = f"{message}: {detail}"
    else:
        message = f"{message}: exited with status {result.returncode}"
    raise RuntimeError(message)


def _worker_argv(
    *,
    instance_id: int,
    model: str | None = None,
    reasoning_effort: str | None = None,
    sandbox_policy: str | None = None,
    approval_mode: str | None = None,
    web_search_mode: str | None = None,
    enable_memories: bool = False,
    collaboration_mode: str | None = None,
    plan_mode: bool = False,
) -> list[str]:
    manage_py = str(Path(settings.BASE_DIR) / "manage.py")
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
    if web_search_mode:
        argv.extend(["--web-search-mode", web_search_mode])
    if enable_memories:
        argv.append("--enable-memories")
    if collaboration_mode:
        argv.extend(["--collaboration-mode", collaboration_mode])
    if plan_mode:
        argv.append("--plan-mode")
    return argv


def _worker_isolation() -> str:
    return str(
        getattr(settings, "CODEX_WORKER_ISOLATION", _WORKER_ISOLATION_AUTO)
    ).strip().lower()


def _scope_unit_for_instance(instance_id: int) -> str:
    return f"hitch-codex-worker-{instance_id}.scope"


def _systemd_scope_argv(
    *,
    systemd_run: str,
    scope_unit: str,
    worker_argv: list[str],
) -> list[str]:
    argv = [
        systemd_run,
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        f"--unit={scope_unit.removesuffix('.scope')}",
    ]
    worker_slice = _worker_slice()
    if worker_slice:
        argv.append(f"--slice={worker_slice}")
    for property_value in _memory_cgroup_properties(
        "CODEX_WORKER_MEMORY_HIGH", "CODEX_WORKER_MEMORY_MAX"
    ):
        argv.append(f"--property={property_value}")
    argv.append("--")
    argv.extend(worker_argv)
    return argv


def _raise_for_immediate_systemd_run_failure(
    proc: subprocess.Popen[bytes], scope_unit: str, stderr_file: Any
) -> None:
    try:
        returncode = proc.wait(timeout=0.25)
    except subprocess.TimeoutExpired:
        return
    if returncode == 0:
        return
    stderr_file.seek(0)
    stderr = stderr_file.read()
    if isinstance(stderr, bytes):
        detail = stderr.decode("utf-8", errors="replace").strip()
    else:
        detail = str(stderr).strip()
    message = f"systemd-run failed to launch Codex worker scope {scope_unit}"
    if detail:
        message = f"{message}: {detail}"
    else:
        message = f"{message}: exited with status {returncode}"
    raise RuntimeError(message)


def _popen_detached(
    argv: list[str], *, env: dict[str, str], stderr: Any = subprocess.DEVNULL
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        argv,
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=stderr,
        env=env,
        close_fds=True,
    )
