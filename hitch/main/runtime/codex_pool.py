"""Pool of detached Codex worker subprocesses.

Each worker runs a single turn for one Codex thread and then exits. Workers
are launched outside the Django process tree, preferably as a per-worker
systemd user service with its own memory cgroup. The CodexInstance row + JSONL
events file on disk are the durable post-spawn links back to a worker; a
sibling control JSONL file carries mid-turn requests such as steer payloads
into the detached process.

The worker is the ``codex_worker`` management command in this app; running it
as a Django command lets it use the same ORM/settings as the parent without
re-implementing Django bootstrap.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, BinaryIO, TypeVar, cast

from django.conf import settings
from django.db import transaction
from django.utils import timezone
from openai_codex import Codex, CodexConfig
from openai_codex.generated.v2_all import (
    ApprovalsReviewer,
    AskForApprovalValue,
    SandboxMode,
    ThreadSource,
    WebSearchMode,
)

from hitch.main.models import ApprovalRequest, CodexInstance, UserInputRequest
from hitch.main.runtime.codex_tools import registered_dynamic_tool_specs
from hitch.main.sessions import session_index

logger = logging.getLogger(__name__)

T = TypeVar("T")

_TRACKED_WORKER_PROCS: dict[int, tuple[int, subprocess.Popen[bytes]]] = {}
_REAPED_WORKERS: set[tuple[int, int]] = set()
_TRACKED_WORKER_PROCS_LOCK = threading.Lock()
# Latches once the per-process swap-cap hierarchy check has run; see
# ``_ensure_systemd_worker_slice``.
_swap_hierarchy_warned = False
_VALID_WEB_SEARCH_MODES = frozenset(mode.value for mode in WebSearchMode)
_THREAD_START_SANDBOX_MODES = {
    "readOnly": SandboxMode.read_only.value,
    "workspaceWrite": SandboxMode.workspace_write.value,
    "dangerFullAccess": SandboxMode.danger_full_access.value,
}
_THREAD_START_APPROVAL_SETTINGS = {
    "auto_review": (
        AskForApprovalValue.on_request.value,
        ApprovalsReviewer.auto_review.value,
    ),
    "deny_all": (AskForApprovalValue.never.value, None),
}
_MAX_INPUT_ATTACHMENT_PATHS_PER_INSTANCE = 16
_MAX_INPUT_ATTACHMENT_PATHS_PER_THREAD = 64
# How long a freshly spawned row may sit with pid=0 before reconcile_dead
# treats it as orphaned. ``_spawn_worker`` commits the row before
# ``subprocess.Popen`` returns, so a transient pid=0 window is a normal part
# of the launch handshake; the grace is generous enough to absorb a slow Popen
# on a loaded host but bounded so a parent that crashed mid-spawn cannot leave
# the row pending forever.
_PID_ASSIGNMENT_GRACE = timedelta(minutes=2)
# How long a steer waits for the worker's delivery ack. The signal-driven
# drain normally acks within a poll interval; the long tail is a turn that
# completes mid-steer, where the row turning terminal resolves the wait well
# before this cap. The cap only bites on a wedged-but-alive worker.
_STEER_ACK_TIMEOUT_SECONDS = 8.0
_STEER_ACK_POLL_SECONDS = 0.1
_WORKER_ISOLATION_AUTO = "auto"
_WORKER_ISOLATION_DIRECT = "direct"
_WORKER_ISOLATION_SYSTEMD = "systemd"
_CODEX_THREAD_PATH_ATTR = "_hitch_codex_thread_path"


@dataclass(frozen=True)
class WorkerLaunch:
    """Result of launching a detached worker or its systemd unit."""

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
    start_kwargs: dict[str, Any] = {
        "cwd": cwd,
        "developerInstructions": developer_instructions,
        "model": model,
    }
    if reasoning_effort:
        start_kwargs["config"] = {"model_reasoning_effort": reasoning_effort}
    if sandbox := _THREAD_START_SANDBOX_MODES.get(sandbox_policy or ""):
        start_kwargs["sandbox"] = sandbox
    if approval := _THREAD_START_APPROVAL_SETTINGS.get(approval_mode or ""):
        start_kwargs["approvalPolicy"] = approval[0]
        if approval[1] is not None:
            start_kwargs["approvalsReviewer"] = approval[1]
    if thread_source is not None:
        start_kwargs["threadSource"] = thread_source.value
    dynamic_tools = registered_dynamic_tool_specs(
        purpose=purpose, agent_kind=agent_kind
    )
    if dynamic_tools:
        start_kwargs["dynamicTools"] = dynamic_tools
    name_source = (
        thread_name if thread_name is not None and thread_name.strip() else prompt
    )

    def _create_and_persist(codex: Codex) -> tuple[str, str | None]:
        response = codex._client.thread_start(start_kwargs)
        thread = response.thread
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
        codex._client.thread_set_name(thread.id, _initial_thread_name(name_source))
        return thread.id, _thread_path_value(thread)

    # ``thread_set_name`` triggers the CODEX_HOME state-DB persist, whose
    # one-time migration path has no SQLITE_BUSY retry; a lock there kills the
    # app-server mid-operation as a ``TransportClosedError`` that ``open_codex``
    # (construction-only retry) never sees. Retrying the whole open+create here
    # is safe even though ``thread_start`` is not idempotent: that failure means
    # the app-server exited *before* the thread was persisted to disk, so the
    # discarded in-memory thread leaves nothing behind to duplicate.
    thread_id, thread_path = app_server_pool.run_codex_op_with_retry(
        lambda: Codex(config=config), _create_and_persist
    )
    instance = _spawn_worker(
        thread_id=thread_id,
        cwd=cwd,
        prompt=prompt,
        input_image_paths=input_image_paths,
        developer_instructions=developer_instructions,
        model=model,
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
    )
    if thread_path:
        setattr(instance, _CODEX_THREAD_PATH_ATTR, thread_path)
    return instance


def create_session_thread(
    *,
    cwd: str,
    name: str,
    developer_instructions: str | None = None,
    model: str | None = None,
    enable_memories: bool = False,
    web_search_mode: str | None = None,
) -> str:
    """Create and persist a visible Codex thread without starting a turn."""
    thread_id, _thread_path = create_session_thread_with_path(
        cwd=cwd,
        name=name,
        developer_instructions=developer_instructions,
        model=model,
        enable_memories=enable_memories,
        web_search_mode=web_search_mode,
    )
    return thread_id


def create_session_thread_with_path(
    *,
    cwd: str,
    name: str,
    developer_instructions: str | None = None,
    model: str | None = None,
    enable_memories: bool = False,
    web_search_mode: str | None = None,
    purpose: str = CodexInstance.PURPOSE_USER,
    agent_kind: str = "",
    thread_source: ThreadSource | None = None,
) -> tuple[str, str]:
    """Create a persisted role-scoped thread without starting its first turn."""
    config = app_server_config(
        enable_memories=enable_memories,
        web_search_mode=web_search_mode,
    )
    start_kwargs: dict[str, Any] = {
        "cwd": cwd,
        "developerInstructions": developer_instructions,
        "model": model,
    }
    dynamic_tools = registered_dynamic_tool_specs(
        purpose=purpose,
        agent_kind=agent_kind,
    )
    if dynamic_tools:
        start_kwargs["dynamicTools"] = dynamic_tools
    if thread_source is not None:
        start_kwargs["threadSource"] = thread_source.value

    def _create_and_persist(codex: Codex) -> tuple[str, str]:
        response = codex._client.thread_start(start_kwargs)
        thread = response.thread
        codex._client.thread_set_name(thread.id, _initial_thread_name(name))
        return thread.id, _thread_path_value(thread)

    # See ``spawn_new_session``: retry the open+create when the ``thread_set_name``
    # persist races the CODEX_HOME state-DB migration. Safe to retry despite the
    # non-idempotent ``thread_start`` because a locked persist exits the
    # app-server before anything reaches disk.
    return app_server_pool.run_codex_op_with_retry(
        lambda: Codex(config=config), _create_and_persist
    )


# Upper bound for the auto-derived thread name. Matches the
# ``_NAME_MAX_LEN`` cap that ``set_session_name`` enforces on user-supplied
# names so the two write paths stay consistent.
_INITIAL_THREAD_NAME_MAX_LEN = session_index.SESSION_NAME_MAX_LEN


def _initial_thread_name(prompt: str) -> str:
    """Return a non-empty thread name derived from ``prompt``.

    Codex rejects whitespace-only names, so a prompt that strips to empty
    falls back to a static placeholder rather than failing the wire call.
    Image-only sessions can intentionally pass an empty prompt, so fall back
    to a stable title rather than sending Codex a whitespace-only name.
    """
    first_line = prompt.split("\n", 1)[0].strip()[:_INITIAL_THREAD_NAME_MAX_LEN].rstrip()
    return first_line or "New session"


def thread_path_for_instance(instance: object) -> str:
    value = getattr(instance, _CODEX_THREAD_PATH_ATTR, "")
    return value if isinstance(value, str) else ""


def _thread_path_value(thread: object) -> str:
    value = getattr(thread, "path", "")
    return value if isinstance(value, str) else ""


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
    developer_instructions: str | None = None,
    purpose: str = CodexInstance.PURPOSE_USER,
    workflow_id: int | None = None,
    agent_kind: str = "",
    display_author: str = "",
    output_schema: dict[str, Any] | None = None,
    user_message_index: int | None = None,
    auto_pr_enabled: bool = False,
    auto_qa_enabled: bool = False,
) -> CodexInstance:
    """Detach a worker that resumes an existing thread to run one prompt.

    Per-turn settings are owned by the caller. Only thread-scoped instruction
    text is copied from prior rows; omitted tool/config values mean Codex
    default for this turn, not "inherit the last worker row."
    """
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
    try:
        threading.Thread(
            target=_wait_for_tracked_worker,
            args=(proc.pid, instance_id, proc),
            name=f"codex-worker-reaper-{proc.pid}",
            daemon=True,
        ).start()
    except Exception:
        # The detached worker is already live, so failing its request now would
        # report a launch failure while the turn continues in the background.
        # Keep the Popen in the registry: reconcile_dead polls this same handle
        # and will reap it on the next sweep even without the waiter thread.
        logger.exception(
            "failed to start reaper thread for Codex worker pid %s; "
            "falling back to reconciliation",
            proc.pid,
        )


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
        # The comm field holds the raw executable basename and may contain
        # arbitrary non-UTF-8 bytes (pids are recycled, so this can be a
        # foreign process). Decode tolerantly: the state char we want sits
        # after the final ')', which is always ASCII.
        stat = (proc_root / str(pid) / "stat").read_bytes().decode(
            "utf-8", errors="replace"
        )
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
            status__in=CodexInstance.ACTIVE_STATUSES,
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
    instance_id: int,
    *,
    expected_thread_id: str,
    force: bool = False,
    error: str | None = None,
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

    ``force`` is reserved for internal workflow recovery after a bounded
    graceful-interrupt window. It skips the first-click SIGTERM path while
    retaining the same worker-identity checks and terminal-row update.

    Returns None when the instance is unknown, belongs to a different thread,
    has already reached a terminal status, is still launching (pid unset), or
    could not be signaled.
    """
    try:
        instance = CodexInstance.objects.get(pk=instance_id)
    except CodexInstance.DoesNotExist:
        return None
    if instance.thread_id != expected_thread_id:
        return None
    if not instance.is_active:
        return None
    return _interrupt_instance(instance, force=force, error=error)


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
    if not instance.is_active:
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
    steer_id = uuid.uuid4().hex
    try:
        payload: dict[str, Any] = {
            "op": "steer",
            "id": steer_id,
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
        if not instance.is_active:
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
    return _await_steer_ack(
        instance, steer_id, image_paths=image_paths, images_tracked=images_tracked
    )


def _await_steer_ack(
    instance: CodexInstance,
    steer_id: str,
    *,
    image_paths: list[str],
    images_tracked: bool,
) -> CodexInstance | None:
    """Wait for the worker to record the steer's delivery outcome.

    A turn can complete while the steer is in flight: the worker's final
    drain still reads the payload but the SDK rejects steering a finished
    turn, and anything appended after that drain is never read at all --
    previously both shapes were reported as a successful steer and the
    user's message silently vanished. The worker now acks every delivery
    attempt; a failed ack (or no ack by the time the row turns terminal)
    means the message was never delivered, so return None and let the
    caller preserve it as a follow-up turn. A worker that stays RUNNING
    without ever acking (wedged drain) falls back to the old optimistic
    answer at the timeout rather than risking a duplicate turn.
    """
    control_path = control_path_for(instance)
    deadline = time.monotonic() + _STEER_ACK_TIMEOUT_SECONDS
    while True:
        delivered = _read_steer_ack(control_path, steer_id)
        if delivered is True:
            return instance
        if delivered is False:
            # The worker already discarded the steer's duplicated attachments.
            return None
        instance.refresh_from_db()
        if not instance.is_active:
            if images_tracked:
                _remove_input_attachment_paths(instance, image_paths)
            return None
        if time.monotonic() >= deadline:
            return instance
        time.sleep(_STEER_ACK_POLL_SECONDS)


def _read_steer_ack(control_path: Path, steer_id: str) -> bool | None:
    try:
        data = control_path.read_bytes()
    except OSError:
        return None
    for raw in data.split(b"\n"):
        if not raw:
            continue
        try:
            record = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if (
            isinstance(record, dict)
            and record.get("op") == "steer_ack"
            and record.get("id") == steer_id
        ):
            return bool(record.get("delivered"))
    return None


def _append_control_request(instance: CodexInstance, payload: dict[str, Any]) -> None:
    path = control_path_for(instance)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, separators=(",", ":")) + "\n"
    with path.open("ab") as fh:
        fh.write(line.encode("utf-8"))


def _interrupt_instance(
    instance: CodexInstance,
    *,
    force: bool = False,
    error: str | None = None,
) -> CodexInstance | None:
    """Stop one worker, escalating SIGTERM → SIGKILL on a second click.

    The first Stop request sends SIGTERM to the worker (not its group).
    The worker's signal handler wakes a dedicated control thread, which calls
    the SDK's ``turn.interrupt()`` independently of the event stream. That is a
    graceful cancellation that lets the app-server emit remaining events
    (notably a ``turn/completed`` with status ``interrupted``) before the
    worker writes its own terminal row status. The row's
    ``interrupt_requested_at`` is set so we can tell, on a subsequent click,
    that polite cancellation was already attempted.

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
        # Legacy scope launches could briefly store the systemd-run client pid
        # until the worker recorded its real pid. Do not treat that client as
        # interruptible.
        return None

    if not _pid_is_instance_worker(instance):
        # PID gone, recycled, or owned by an unrelated process. No safe
        # target for either SIGTERM or SIGKILL, but the leftover row
        # still has to be flipped to failed so the UI exits streaming
        # mode rather than waiting on ``reconcile_dead``.
        return _mark_failed(instance, error or "interrupted by user")

    if not force and instance.interrupt_requested_at is None:
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
    return _mark_failed(instance, error or "forcibly stopped by user")


def _mark_failed(instance: CodexInstance, error: str) -> CodexInstance | None:
    """Conditionally flip a still-active row to FAILED with ``error``.

    Atomic UPDATE keyed on the active statuses so a worker that
    legitimately reached a terminal state in the gap between the row
    read and this call is preserved — preventing a stop click from
    retroactively rewriting a completed turn as failed.
    """
    updated = CodexInstance.objects.filter(
        pk=instance.pk,
        status__in=CodexInstance.ACTIVE_STATUSES,
    ).update(
        status=CodexInstance.STATUS_FAILED,
        ended_at=timezone.now(),
        error=error,
    )
    if updated == 0:
        return None
    _resolve_dangling_requests(instance.pk)
    instance.refresh_from_db()
    _record_session_activity(instance)
    return instance


def _record_session_activity(
    instance: CodexInstance, *, updated_at: datetime | None = None
) -> None:
    """Best-effort recency update for a turn lifecycle transition."""
    try:
        session_index.record_turn_activity(
            instance.thread_id,
            updated_at=updated_at or instance.ended_at or timezone.now(),
        )
    except Exception:
        logger.exception(
            "failed to record session activity for worker %s", instance.pk
        )


def resolve_dangling_requests_for_instance(instance_pk: int) -> None:
    """Close out approval/input rows a dead worker left pending.

    Only call this once the owning instance has been flipped to FAILED — the
    worker is then confirmed gone and can never consume an answer. Without this
    the browser keeps rendering an actionable approval/input card for a turn
    that is over: a click would flip the row and return 200 while the decision
    is silently dropped, and a page reload re-renders the same stale card.
    Cancelling the rows makes a late click resolve to 409 and a reload show no
    pending prompt.
    """
    now = timezone.now()
    ApprovalRequest.objects.filter(instance_id=instance_pk, decision="").update(
        decision=ApprovalRequest.DECISION_CANCEL,
        decided_at=now,
    )
    UserInputRequest.objects.filter(instance_id=instance_pk, response__isnull=True).update(
        response={"answers": {}},
        responded_at=now,
    )


def _resolve_dangling_requests(instance_pk: int) -> None:
    resolve_dangling_requests_for_instance(instance_pk)


def _force_kill_instance(instance: CodexInstance) -> None:
    if instance.systemd_scope_unit:
        systemctl = shutil.which("systemctl")
        if systemctl is None:
            raise OSError("systemctl is required to kill systemd Codex workers")
        try:
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
                # Bound the user-manager round trip: this runs on the Stop
                # request path, and a wedged dbus would otherwise hang the
                # request thread forever (and stack one thread per retry click).
                timeout=5,
            )
        except subprocess.TimeoutExpired as exc:
            raise OSError(
                "systemctl timed out killing systemd Codex worker"
            ) from exc
        if result.returncode == 0:
            return
        if systemd_isolation._systemd_scope_is_missing(systemctl, instance.systemd_scope_unit):
            raise ProcessLookupError
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        message = "systemctl failed to kill systemd Codex worker"
        if detail:
            message = f"{message}: {detail}"
        raise OSError(message)
    os.killpg(instance.pid, signal.SIGKILL)


def _pid_is_instance_worker(instance: CodexInstance) -> bool:
    if instance.systemd_scope_unit:
        return _pid_is_our_worker(
            instance.pid,
            instance.pk,
            require_session_leader=False,
        )
    return _pid_is_our_worker(instance.pid, instance.pk)


def _our_manage_py() -> str:
    """This deployment's ``manage.py`` path, matching ``_worker_argv``.

    Built with ``os.path`` rather than ``pathlib.Path`` so the value is
    unaffected by code (and tests) that patch ``codex_pool.Path`` to feed a fake
    ``/proc`` cmdline through ``_pid_is_our_worker``.
    """
    return os.path.join(str(settings.BASE_DIR), "manage.py")


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
    # Require this deployment's manage.py too: a recycled pid could belong to a
    # second Hitch checkout's codex_worker carrying the same generic marker and
    # even the same instance id, and signaling it would kill the other
    # deployment's process group.
    if _our_manage_py().encode() not in parts:
        return False
    try:
        idx = parts.index(b"--instance-id")
    except ValueError:
        return False
    return idx + 1 < len(parts) and parts[idx + 1] == str(instance_id).encode()


# Grace after a worker commits its terminal status before the orphan reaper may
# kill it. ``codex_worker`` runs ``_notify_system_agents`` (which can spawn
# follow-up turns) and input-image cleanup *after* the terminal commit, so a
# still-live terminal worker inside this window is finishing those hooks rather
# than leaked. Generous so even a slow hook (e.g. spawning a follow-up workflow)
# completes; a genuinely leaked worker is still reaped one grace later.


# Floor on how often the request/SSE-path debounce lets the global sweep run.
# Short enough that a crashed worker still clears within a couple seconds, long
# enough that a burst of concurrent page loads / SSE reconnects collapses to one
# sweep instead of one per request.


def events_dir() -> Path:
    """Filesystem directory holding per-worker JSONL event logs."""
    configured = getattr(settings, "CODEX_EVENTS_DIR", None)
    if configured is not None:
        return Path(configured)
    return Path.home() / ".hitch" / "codex_events"


def hitch_home_dir() -> Path:
    """Filesystem directory holding shared Hitch runtime state."""
    configured = getattr(settings, "HITCH_HOME_DIR", None)
    if configured is not None:
        return Path(configured)
    return Path.home() / ".hitch"


def worker_logs_dir() -> Path:
    """Filesystem directory holding detached worker stderr logs."""
    configured = getattr(settings, "CODEX_WORKER_LOG_DIR", None)
    if configured is not None:
        return Path(configured)
    return hitch_home_dir() / "worker_logs"


def worker_log_path(instance_id: int) -> Path:
    """Return the durable stderr log path for a Codex worker."""
    return worker_logs_dir() / f"{instance_id}.log"


def worker_log_io_enabled() -> bool:
    """Whether this process should touch worker diagnostic logs.

    These logs are an optional visibility side channel; the CodexInstance row is
    the lifecycle source of truth. Tests opt in with an explicit temp log dir so
    assertions cannot depend on stale host files under ``~/.hitch``.
    """
    return not (
        getattr(settings, "TESTING", False)
        and getattr(settings, "CODEX_WORKER_LOG_DIR", None) is None
    )


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
    candidate_paths = set(image_paths)
    retained_paths: list[str] = []
    for raw_path in image_paths:
        path = Path(raw_path).resolve(strict=False)
        if not _path_within(path, root):
            logger.warning(
                "refusing to clean up input image outside attachment dir: %s",
                raw_path,
            )
            retained_paths.append(raw_path)
            continue
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("failed to clean up input image attachment %s", path)
            retained_paths.append(raw_path)
        else:
            _prune_empty_attachment_dirs(path.parent, root)
    # Recompute the ledger from the locked row (unlinks happen above, outside the
    # lock) so a steer image added concurrently isn't clobbered. Mirrors the
    # locked read-modify-write in ``_remove_input_attachment_paths``.
    with transaction.atomic():
        current = CodexInstance.objects.select_for_update().get(pk=instance.pk)
        new_attachment = [
            stored
            for stored in _normalized_input_image_paths(current.input_attachment_paths)
            if stored not in candidate_paths
        ]
        for retained in retained_paths:
            if retained not in new_attachment:
                new_attachment.append(retained)
        new_image = [
            stored
            for stored in _normalized_input_image_paths(current.input_image_paths)
            if stored not in candidate_paths
        ]
        current.input_image_paths = new_image
        current.input_attachment_paths = new_attachment
        current.input_attachment_cleanup_requested = bool(new_attachment)
        current.save(
            update_fields=[
                "input_image_paths",
                "input_attachment_paths",
                "input_attachment_cleanup_requested",
            ]
        )
    instance.input_image_paths = new_image
    instance.input_attachment_paths = new_attachment
    instance.input_attachment_cleanup_requested = bool(new_attachment)


def cleanup_input_images_for_thread(thread_id: str) -> None:
    """Delete retained input images for every turn in a thread."""
    CodexInstance.objects.filter(
        thread_id=thread_id,
        status__in=CodexInstance.ACTIVE_STATUSES,
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


# Codex's native env var for relocating its SQLite databases away from
# ``$CODEX_HOME`` (state/src/lib.rs SQLITE_HOME_ENV). The SDK merges
# ``CodexConfig.env`` onto the inherited environment, so setting it here is
# authoritative for the app-server we spawn.
_CODEX_SQLITE_HOME_ENV = "CODEX_SQLITE_HOME"


def _sqlite_home_base() -> Path:
    return Path(settings.CODEX_SQLITE_HOME_BASE)


def _worker_sqlite_pool_size() -> int:
    return max(1, int(getattr(settings, "CODEX_WORKER_SQLITE_POOL_SIZE", 1)))


def web_sqlite_home() -> Path:
    """Shared ``sqlite_home`` for in-process (request + scheduler) app-servers.

    Keeping the request pool, keepalive, and scheduler on one home means the
    ``use_state_db_only`` thread listing they drive reads a single, populated
    state-DB index rather than a per-process one that would start empty.
    """
    home = _sqlite_home_base() / "web"
    home.mkdir(parents=True, exist_ok=True)
    return home


def worker_sqlite_slot_home(slot: int) -> Path:
    """Path of the ``slot``-th pooled worker ``sqlite_home`` (not created)."""
    return _sqlite_home_base() / f"worker-{slot}"


@dataclass
class WorkerSqliteHome:
    """An exclusively-leased (or private overflow) worker ``sqlite_home``.

    While held, no other worker uses ``home``: the ``flock`` on ``_lock_fd`` is
    kept for the worker's whole lifetime and the OS releases it even if the
    worker crashes, so a turn has sole ownership of its home's SQLite writer
    lock. ``release`` drops the lease; an *overflow* home -- allocated only when
    every pooled slot was already leased -- is a private directory and is removed
    on release rather than reused.
    """

    home: Path
    _lock_fd: int | None
    overflow: bool

    def release(self) -> None:
        if self._lock_fd is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            with contextlib.suppress(OSError):
                os.close(self._lock_fd)
            self._lock_fd = None
        if self.overflow:
            shutil.rmtree(self.home, ignore_errors=True)


def acquire_worker_sqlite_home(instance_id: int) -> WorkerSqliteHome:
    """Lease an exclusive ``sqlite_home`` for a detached worker's turn.

    Scans the bounded pool of homes and takes the first whose lock file it can
    ``flock(LOCK_EX | LOCK_NB)``, so concurrent workers never share a home (and
    so never contend on each other's SQLite writer lock, openai/codex#20213)
    while reuse of the low-numbered slots keeps each home's one-time backfill of
    the shared ``CODEX_HOME`` rollouts amortized. When every slot is already
    leased -- more concurrent turns than pool slots -- a private per-instance
    overflow home is returned instead: still unshared, at the cost of its own
    one-time backfill. The caller must ``release`` the returned lease.
    """
    base = _sqlite_home_base()
    base.mkdir(parents=True, exist_ok=True)
    for slot in range(_worker_sqlite_pool_size()):
        lock_path = base / f"worker-{slot}.lock"
        # PEP 446 makes this fd non-inheritable, so the app-server subprocess the
        # worker spawns never holds the lease open past the worker's own exit.
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            continue
        try:
            home = worker_sqlite_slot_home(slot)
            home.mkdir(parents=True, exist_ok=True)
        except OSError:
            # Without this, the held flock (and its fd) would outlive the
            # caller's CODEX_HOME fallback and block the slot for the whole
            # turn.
            os.close(fd)
            raise
        return WorkerSqliteHome(home=home, _lock_fd=fd, overflow=False)
    home = base / f"worker-overflow-{instance_id}"
    home.mkdir(parents=True, exist_ok=True)
    return WorkerSqliteHome(home=home, _lock_fd=None, overflow=True)


def _default_sqlite_home() -> Path | None:
    # Under tests we leave ``sqlite_home`` unset so app-servers (which are mocked
    # away) never create state directories, and the env assertion in
    # test_codex_subprocess stays exact. Production callers that do not pass an
    # explicit home are the in-process request/scheduler paths -> web home.
    if getattr(settings, "TESTING", False):
        return None
    return web_sqlite_home()


def codex_home_dir() -> Path:
    """Codex's home directory: ``$CODEX_HOME`` if set, else ``~/.codex``.

    Used as the deterministic fallback when a worker's home lease fails: passing
    this as an explicit ``sqlite_home`` forces Codex's DBs back to ``$CODEX_HOME``
    (its pre-isolation default), overriding any ``CODEX_SQLITE_HOME`` the
    deployment exported -- which a bare omission would otherwise leave in effect.
    """
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")


def app_server_config(
    *,
    enable_memories: bool = False,
    web_search_mode: str | None = None,
    sqlite_home: str | os.PathLike[str] | None = None,
    additional_config_overrides: tuple[str, ...] = (),
) -> CodexConfig:
    memory_value = "true" if enable_memories else "false"
    overrides = [f"features.memories={memory_value}"]
    web_search_mode = _normalized_web_search_mode(web_search_mode)
    if web_search_mode:
        overrides.append(f"web_search={json.dumps(web_search_mode)}")
    overrides.extend(additional_config_overrides)
    # Stamp every app-server we spawn with this deployment's id (merged onto the
    # inherited environment by the SDK). The profile "nuke" sweep scopes its
    # SIGKILLs to this marker so a second checkout sharing the resolved codex
    # binary -- whose app-server command lines are otherwise identical -- is
    # never swept.
    env = {reconciliation._APP_SERVER_DEPLOYMENT_ENV: reconciliation._app_server_deployment_id()}
    resolved_home = sqlite_home if sqlite_home is not None else _default_sqlite_home()
    if resolved_home is not None:
        env[_CODEX_SQLITE_HOME_ENV] = os.fspath(resolved_home)
    return CodexConfig(
        config_overrides=tuple(overrides),
        env=env,
    )


# Codex's diagnostic log DB filename (state/src/lib.rs LOGS_DB_FILENAME is
# ``logs_2.sqlite``). Matched by glob so a Codex bump to ``logs_<n>.sqlite``
# still gets pruned; the state DB (``state_*.sqlite``) is deliberately excluded.
_CODEX_LOGS_DB_GLOB = "logs_*.sqlite"
_CODEX_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm")


def prune_worker_logs_db(
    sqlite_home: str | os.PathLike[str], *, max_bytes: int | None = None
) -> int:
    """Delete a worker home's Codex log DB once it exceeds ``max_bytes``.

    Called after a worker's turn, when that worker's app-server is already
    closed, so the file handle is released. Best-effort: a home may still be
    shared by another in-flight worker, but the log DB is purely diagnostic and
    Codex recreates it (``create_if_missing``) on the next open, so a racing
    unlink only drops diagnostic rows -- it never risks the state DB, which is
    left untouched. Returns the number of bytes freed.
    """
    if max_bytes is None:
        max_bytes = int(getattr(settings, "CODEX_WORKER_LOGS_DB_MAX_BYTES", 0))
    home = Path(sqlite_home)
    freed = 0
    for db_path in sorted(home.glob(_CODEX_LOGS_DB_GLOB)):
        group = [db_path] + [
            Path(f"{db_path}{suffix}") for suffix in _CODEX_SQLITE_SIDECAR_SUFFIXES
        ]
        sizes: dict[Path, int] = {}
        for path in group:
            try:
                sizes[path] = path.stat().st_size
            except OSError:
                continue
        if sum(sizes.values()) <= max_bytes:
            continue
        for path in group:
            with contextlib.suppress(OSError):
                path.unlink()
                freed += sizes.get(path, 0)
        logger.info(
            "pruned oversized Codex log DB %s (%d bytes)",
            db_path,
            sum(sizes.values()),
        )
    return freed


# Every ``Codex(config=config)`` spawns a ``codex app-server`` subprocess that
# initializes the Codex Rust runtime's own SQLite state database under
# ``$CODEX_HOME`` -- separate from Hitch's Django DB. That state DB hardcodes a
# 5s busy_timeout, and its one-time init/migration/backfill path has no
# SQLITE_BUSY retry (openai/codex#20213). Concurrent startups (request handlers,
# detached workers, the scheduler) can race that migration on a fresh CODEX_HOME
# or after a codex upgrade: the loser exits with "database is locked", surfaced
# as a TransportClosedError whose stderr tail carries that message.
#
# We do NOT serialize startups. Once the schema is current, most app-server opens
# don't need exclusive state-DB work; serializing every startup behind one
# machine-wide lock made unrelated turns queue behind each other. So we retry a
# locked init instead. Request-path opens stay bounded so a page render does not
# hang for minutes, while detached worker starts get a longer budget: a worker
# sitting behind Codex's own long state/log maintenance is much less harmful than
# failing the whole turn or system-agent workflow.


# Capped per-attempt backoff; request-path callers spend about 26s in Hitch
# backoff before giving up, on top of Codex's own 5s SQLite busy timeout per
# failed attempt. Detached workers use a larger attempt count.


# The Django web process used to open (and so re-init the CODEX_HOME state DB
# for) a fresh app-server on every short metadata call -- session lists, model
# lookups, archive/resume. Those bursty per-request inits were the dominant
# source of the state-DB init races ``open_codex`` only retries after the fact.
# ``borrow_codex`` instead keeps a small pool of long-lived app-servers warm and
# hands one out per call, so the state DB is initialized once per pooled server
# (the same property the auto-proposal scheduler gets from its single reused
# app-server) and steady-state borrows do not even spawn a subprocess. Bursts
# past the cap still construct a private server (retrying a locked init) rather
# than blocking or failing, so this only ever reduces init churn.


# (enable_memories, normalized web_search_mode). In practice every web call uses
# the default web_search_mode, so the live key space is just enable_memories.




# How often the keepalive exercises a warm pooled server. Short relative to any
# plausible app-server idle timeout so the server stays warm, long enough not to
# add meaningful load.


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
    if instance.is_active:
        return True
    _remove_input_attachment_paths(instance, image_paths)
    return False


def _spawn_worker(
    *,
    thread_id: str,
    cwd: str,
    prompt: str,
    input_image_paths: list[str] | None = None,
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
            approval_mode_live_editable=(
                approval_mode in CodexInstance.LIVE_EDITABLE_APPROVAL_MODES
            ),
            web_search_mode=web_search_mode or "",
            plan_mode=plan_mode,
            auto_pr_enabled=auto_pr_enabled,
            auto_qa_enabled=auto_qa_enabled,
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

    # A submitted prompt is session activity even if the detached worker is
    # killed before its own completion hook can run.
    _record_session_activity(instance, updated_at=instance.started_at)

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
        if model:
            launch_kwargs["model"] = model
        if plan_mode:
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
        _record_session_activity(instance)
        cleanup_input_images_for(instance)
        raise
    launch_pid = getattr(launch, "pid", 0)
    scope_unit = getattr(launch, "scope_unit", "")
    if scope_unit:
        # The worker never touches systemd_scope_unit, so the parent owns it
        # outright; force-kill escalation needs the systemd unit name.
        instance.systemd_scope_unit = scope_unit
        instance.save(update_fields=["systemd_scope_unit"])
    if launch_pid > 0:
        # The worker's first action is to overwrite pid with its own real pid.
        # Under direct isolation launch_pid is already the worker. Systemd
        # isolation returns pid=0 and relies on the worker's own first write,
        # so a webserver restart never leaves the row aimed at a systemd-run
        # client process.
        claimed = CodexInstance.objects.filter(pk=instance.pk, pid=0).update(
            pid=launch_pid
        )
        if claimed:
            instance.pid = launch_pid
        else:
            instance.refresh_from_db(fields=["pid"])
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
    worker_log = _open_worker_log_file(instance_id)
    try:
        stderr: Any = worker_log if worker_log is not None else subprocess.DEVNULL
        if requested_isolation == _WORKER_ISOLATION_DIRECT:
            proc = _popen_detached(argv, env=env, stderr=stderr)
            return WorkerLaunch(pid=proc.pid, proc=proc)
        if requested_isolation not in (
            _WORKER_ISOLATION_AUTO,
            _WORKER_ISOLATION_SYSTEMD,
        ):
            raise ValueError(f"invalid CODEX_WORKER_ISOLATION: {requested_isolation!r}")
        systemd_run = systemd_isolation._systemd_run_for_isolation(requested_isolation)
        if systemd_run is None:
            proc = _popen_detached(argv, env=env, stderr=stderr)
            return WorkerLaunch(pid=proc.pid, proc=proc)
        return systemd_isolation._launch_systemd_worker(
            systemd_run=systemd_run,
            scope_unit=_scope_unit_for_instance(instance_id),
            worker_argv=argv,
            env=env,
            stderr=stderr,
            stderr_capture=worker_log,
        )
    finally:
        if worker_log is not None:
            worker_log.close()


def _open_worker_log_file(instance_id: int) -> BinaryIO | None:
    if not worker_log_io_enabled():
        return None
    try:
        log_path = worker_log_path(instance_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("a+b", buffering=0)
    except OSError:
        logger.exception("failed to open Codex worker log for instance %s", instance_id)
        return None
    _write_worker_log_marker(log_file, f"launcher pid={os.getpid()} starting worker")
    return log_file


def _write_worker_log_marker(log_file: BinaryIO, message: str) -> None:
    line = f"{timezone.now().isoformat()} {message}\n".encode("utf-8", errors="replace")
    with contextlib.suppress(Exception):
        log_file.write(line)


# cgroup-v2 cpu.weight bounds (CPUWeight=1..10000, default 100).


_CPU_WEIGHT_DEFAULT = 100


def _parent_slice() -> str:
    return str(getattr(settings, "CODEX_PARENT_SLICE", "") or "").strip()


def _systemd_parent_slice_properties() -> list[str]:
    """Build the CPU-weight property for Hitch's parent slice.

    The weight biases the runserver subtree against the worker pool when CPU is
    contested (see ``CODEX_PARENT_SLICE`` in settings for why this must live on
    the parent slice, not the workers slice). Configured declaratively like the
    memory caps: a cleared value resets to the cgroup-v2 default of 100 rather
    than leaving a stale ``--runtime`` weight lingering. An out-of-range or
    non-integer value is dropped with a warning so a typo is loud but the slice
    keeps whatever weight it already has rather than failing every worker spawn.
    """
    raw = str(getattr(settings, "CODEX_PARENT_SLICE_CPU_WEIGHT", "") or "").strip()
    if not raw:
        return [f"CPUWeight={_CPU_WEIGHT_DEFAULT}"]
    try:
        weight = int(raw)
    except ValueError:
        logger.warning(
            "ignoring CODEX_PARENT_SLICE_CPU_WEIGHT=%r: not an integer", raw
        )
        return []
    if not systemd_isolation._CPU_WEIGHT_MIN <= weight <= systemd_isolation._CPU_WEIGHT_MAX:
        logger.warning(
            "ignoring CODEX_PARENT_SLICE_CPU_WEIGHT=%r: outside the cgroup-v2 "
            "range %d-%d",
            raw,
            systemd_isolation._CPU_WEIGHT_MIN,
            systemd_isolation._CPU_WEIGHT_MAX,
        )
        return []
    return [f"CPUWeight={weight}"]


# Powers-of-1024 suffixes systemd accepts for absolute memory sizes. Used only
# for the best-effort swap-cap comparison below; percentages and "infinity"
# deliberately fall outside this map so we skip numeric comparison for them.
_MEMORY_SUFFIX_FACTORS = {
    "K": 1024,
    "M": 1024**2,
    "G": 1024**3,
    "T": 1024**4,
    "P": 1024**5,
    "E": 1024**6,
}


def _parse_memory_bytes(value: str) -> int | None:
    """Best-effort parse of a systemd memory value to an integer byte count.

    Returns ``None`` for anything we cannot compare numerically — empty,
    ``infinity``, percentages, decimals, or unrecognized suffixes — so callers
    fall back to the unambiguous ``== "0"`` check rather than guessing.
    """
    text = value.strip()
    if not text:
        return None
    if text[-1].isdigit():
        try:
            return int(text)
        except ValueError:
            return None
    factor = _MEMORY_SUFFIX_FACTORS.get(text[-1].upper())
    if factor is None:
        return None
    try:
        return int(text[:-1]) * factor
    except ValueError:
        return None


def _effective_swap_cap(max_setting: str, swap_setting: str) -> str | None:
    """Return the ``MemorySwapMax`` value a unit will actually enforce, if any.

    Mirrors :func:`_memory_cgroup_properties`: the swap cap is only emitted
    (hence enforced) alongside a finite hard ``MemoryMax``, so a unit with no
    real hard cap reports ``None`` — it leaves swap unlimited regardless of the
    setting.
    """
    hard = str(getattr(settings, max_setting, "") or "").strip()
    swap = str(getattr(settings, swap_setting, "") or "").strip()
    if swap and systemd_isolation._is_finite_limit(hard):
        return swap
    return None


def _warn_on_swap_cap_hierarchy() -> None:
    """Warn when the parent slice's swap cap overrides the per-worker setting.

    cgroup v2 swap limits are hierarchical, so a worker can never use more swap
    than its enclosing slice allows. Two silent surprises follow, both fixed by
    raising (or clearing) ``CODEX_WORKER_SLICE_MEMORY_SWAP_MAX``:

    * An operator raises only ``CODEX_WORKER_MEMORY_SWAP_MAX`` to grant a
      cushion, leaving the slice at the fail-fast ``0`` default — the stricter
      slice cap nullifies the cushion.
    * An operator clears ``CODEX_WORKER_MEMORY_SWAP_MAX`` to opt a worker out of
      the per-scope cap — the slice still enforces its own cap, so the worker
      gets no swap despite the cleared setting.
    """
    slice_swap = _effective_swap_cap(
        "CODEX_WORKER_SLICE_MEMORY_MAX",
        "CODEX_WORKER_SLICE_MEMORY_SWAP_MAX",
    )
    # No enforced, finite slice cap means the slice imposes no swap restriction
    # the worker could be surprised by.
    if slice_swap is None or not systemd_isolation._is_finite_limit(slice_swap):
        return

    worker_swap = _effective_swap_cap(
        "CODEX_WORKER_MEMORY_MAX",
        "CODEX_WORKER_MEMORY_SWAP_MAX",
    )
    if worker_swap is not None and systemd_isolation._is_finite_limit(worker_swap):
        # The worker imposes its own finite swap cap.
        if worker_swap == "0":
            return  # Worker denies swap too; consistent with the slice cap.
        worker_bytes = _parse_memory_bytes(worker_swap)
        slice_bytes = _parse_memory_bytes(slice_swap)
        if worker_bytes is not None and slice_bytes is not None:
            clipped = slice_bytes < worker_bytes
        else:
            # Can't compare magnitudes (percentages, decimals); only a hard-zero
            # slice is unambiguously stricter than a non-zero cushion.
            clipped = slice_swap == "0"
        if clipped:
            logger.warning(
                "CODEX_WORKER_MEMORY_SWAP_MAX=%r is nullified by the stricter "
                "parent slice CODEX_WORKER_SLICE_MEMORY_SWAP_MAX=%r: cgroup swap "
                "limits are hierarchical, so the per-worker swap cushion is "
                "ineffective until the slice cap is raised to at least match it.",
                worker_swap,
                slice_swap,
            )
        return

    # The worker has a finite hard cap but leaves swap unlimited (the setting is
    # cleared or ``infinity``) — an opt-out the slice silently overrides.
    worker_hard = str(getattr(settings, "CODEX_WORKER_MEMORY_MAX", "") or "").strip()
    worker_swap_raw = str(
        getattr(settings, "CODEX_WORKER_MEMORY_SWAP_MAX", "") or ""
    ).strip()
    if systemd_isolation._is_finite_limit(worker_hard) and not systemd_isolation._is_finite_limit(worker_swap_raw):
        logger.warning(
            "CODEX_WORKER_MEMORY_SWAP_MAX=%r leaves the worker's swap uncapped, "
            "but the enclosing slice still enforces "
            "CODEX_WORKER_SLICE_MEMORY_SWAP_MAX=%r: cgroup swap limits are "
            "hierarchical, so the worker is still limited to the slice's swap "
            "budget until that cap is cleared or raised.",
            worker_swap_raw,
            slice_swap,
        )


def _apply_slice_properties(slice_unit: str, properties: list[str]) -> None:
    """Apply declarative cgroup properties to a slice via ``set-property``.

    A no-op when the slice name or property list is empty so callers can
    configure optional slices unconditionally. Idempotent: ``--runtime``
    set-property only changes the properties it is handed, and the configured
    state is rendered declaratively so re-applying converges on the same values.
    """
    if not slice_unit or not properties:
        return
    systemctl = shutil.which("systemctl")
    if systemctl is None:
        raise RuntimeError(
            f"systemctl is required to configure Codex slice {slice_unit}"
        )
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
        raise RuntimeError(f"failed to configure Codex slice {slice_unit}") from exc
    if result.returncode == 0:
        return
    detail = result.stderr.decode("utf-8", errors="replace").strip()
    message = f"failed to configure Codex slice {slice_unit}"
    if detail:
        message = f"{message}: {detail}"
    else:
        message = f"{message}: exited with status {result.returncode}"
    raise RuntimeError(message)


def _ensure_systemd_worker_slice() -> None:
    global _swap_hierarchy_warned
    slice_unit = systemd_isolation._worker_slice()
    if not slice_unit:
        # Without a worker slice, `_systemd_scope_argv` omits `--slice`, so
        # workers land in systemd-run's default slice rather than under the
        # parent slice. Biasing the parent would not reach them (and a failure
        # configuring an unrelated slice would needlessly abort every launch),
        # so skip all slice configuration when worker placement is disabled.
        return
    # Running inside a slice means its parent swap cap applies; check the
    # hierarchy once per process so a misconfigured cushion doesn't go
    # unnoticed without spamming the log on every worker launch.
    if not _swap_hierarchy_warned:
        _swap_hierarchy_warned = True
        _warn_on_swap_cap_hierarchy()
    _apply_slice_properties(slice_unit, systemd_isolation._systemd_worker_slice_properties())
    # Bias the parent slice so the user-facing runserver wins CPU contests
    # against the worker pool. The CPU weight belongs on the parent (sibling to
    # the runserver's slice), not on the leaf workers slice, which the workers
    # nest under. Empty CODEX_PARENT_SLICE disables just this bias.
    _apply_slice_properties(_parent_slice(), _systemd_parent_slice_properties())


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
    manage_py = _our_manage_py()
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
    return f"hitch-codex-worker-{_deployment_unit_suffix()}-{instance_id}.service"


def _deployment_unit_suffix() -> str:
    """Stable, systemd-safe discriminator for this checkout's worker units."""
    base_dir = os.path.realpath(os.fspath(settings.BASE_DIR))
    return hashlib.sha256(base_dir.encode("utf-8")).hexdigest()[:12]


_SYSTEMD_ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_SYSTEMD_ENV_DENYLIST = frozenset(
    {
        "INVOCATION_ID",
        "JOURNAL_STREAM",
        "LISTEN_FDS",
        "LISTEN_FDNAMES",
        "LISTEN_PID",
        "MAINPID",
        "MANAGERPID",
        "NOTIFY_SOCKET",
        "WATCHDOG_PID",
        "WATCHDOG_USEC",
    }
)


def _systemd_env_args(env: dict[str, str]) -> list[str]:
    """Pass the launcher's valid environment names to a transient service.

    Transient services run in the user manager's clean environment rather than
    inheriting the caller like ``--scope`` does, so the worker must explicitly
    receive settings such as ``DJANGO_SETTINGS_MODULE``, ``CODEX_HOME``, and
    deployment-specific Hitch paths. ``--setenv=NAME`` copies the value from
    the ``systemd-run`` client's environment without placing secrets directly
    on the command line.
    """
    return [
        f"--setenv={name}"
        for name in sorted(env)
        if name not in _SYSTEMD_ENV_DENYLIST and _SYSTEMD_ENV_NAME_RE.fullmatch(name)
    ]


def _systemd_scope_argv(
    *,
    systemd_run: str,
    scope_unit: str,
    worker_argv: list[str],
    env: dict[str, str] | None = None,
    stderr_log_path: str | None = None,
) -> list[str]:
    unit_name = scope_unit.removesuffix(".service").removesuffix(".scope")
    argv = [
        systemd_run,
        "--user",
        "--quiet",
        "--collect",
        "--service-type=exec",
        f"--unit={unit_name}",
    ]
    worker_slice = systemd_isolation._worker_slice()
    if worker_slice:
        argv.append(f"--slice={worker_slice}")
    if env is not None:
        argv.extend(_systemd_env_args(env))
    argv.append("--property=StandardOutput=null")
    if stderr_log_path:
        argv.append(f"--property=StandardError=append:{stderr_log_path}")
    else:
        argv.append("--property=StandardError=null")
    # Do not let one kernel OOM victim make systemd terminate every surviving
    # process in the unit. When a child command is selected, Codex can observe
    # its failure and recover (for example, by reducing build parallelism).
    argv.append("--property=OOMPolicy=continue")
    argv.extend(
        f"--property={property_value}"
        for property_value in systemd_isolation._memory_cgroup_properties(
            "CODEX_WORKER_MEMORY_HIGH",
            "CODEX_WORKER_MEMORY_MAX",
            "CODEX_WORKER_MEMORY_SWAP_MAX",
        )
    )
    argv.append("--")
    argv.extend(worker_argv)
    return argv


def _check_systemd_run_start_result(
    proc: subprocess.Popen[bytes],
    scope_unit: str,
    stderr_file: Any,
    *,
    stderr_offset: int = 0,
) -> bool:
    """Return whether the systemd-run client exited cleanly.

    A still-running client can be waiting on a slow user manager or transient
    service start job. The worker row remains ``pid=0`` during that handshake;
    the worker writes its real pid once it starts, and ``reconcile_dead`` fails
    the row later if that never happens.
    """
    try:
        returncode = proc.wait(timeout=0.25)
    except subprocess.TimeoutExpired:
        return False
    if returncode == 0:
        return True
    stderr_file.seek(stderr_offset)
    stderr = stderr_file.read()
    if isinstance(stderr, bytes):
        detail = stderr.decode("utf-8", errors="replace").strip()
    else:
        detail = str(stderr).strip()
    message = f"systemd-run failed to launch Codex worker unit {scope_unit}"
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


# Imported last: the pool and error-detail submodules reach back into this
# module for the path/config helpers, so they need its namespace to be
# fully initialized.
from hitch.main.runtime import app_server_pool, reconciliation, systemd_isolation  # noqa: E402
