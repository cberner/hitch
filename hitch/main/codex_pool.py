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

import atexit
import contextlib
import fcntl
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, BinaryIO, TypeVar, cast

from django.conf import settings
from django.db import transaction
from django.utils import timezone
from openai_codex import AppServerConfig, Codex, TransportClosedError
from openai_codex.generated.v2_all import ThreadSource, WebSearchMode

from hitch.main import rate_limit, server_lifecycle
from hitch.main.codex_tools import registered_dynamic_tool_specs
from hitch.main.db import is_database_locked_error
from hitch.main.models import ApprovalRequest, CodexInstance, UserInputRequest

logger = logging.getLogger(__name__)

T = TypeVar("T")

_TRACKED_WORKER_PROCS: dict[int, tuple[int, subprocess.Popen[bytes]]] = {}
_REAPED_WORKERS: set[tuple[int, int]] = set()
_TRACKED_WORKER_PROCS_LOCK = threading.Lock()
# Latches once the per-process swap-cap hierarchy check has run; see
# ``_ensure_systemd_worker_slice``.
_swap_hierarchy_warned = False
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
_CODEX_THREAD_PATH_ATTR = "_hitch_codex_thread_path"
_WORKER_UNIT_RE = re.compile(r"hitch-codex-worker-(\d+)\.(?:service|scope)")


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
    thread_id, thread_path = run_codex_op_with_retry(
        lambda: Codex(config=config), _create_and_persist
    )
    instance = _spawn_worker(
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
        auto_merge_to_local_branch=auto_merge_to_local_branch,
        auto_merge_branch=auto_merge_branch,
    )
    if thread_path:
        setattr(instance, _CODEX_THREAD_PATH_ATTR, thread_path)
    return instance


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
    # Use the low-level client (like spawn_new_session) so this visible session
    # can register Hitch dynamic tools. These threads are real user sessions the
    # user drives directly, so they need the same tools (e.g.
    # hitch.propose_session) an ordinary spawn_new_session session gets.
    start_kwargs: dict[str, Any] = {
        "cwd": cwd,
        "developerInstructions": developer_instructions,
        "model": model,
        "dynamicTools": registered_dynamic_tool_specs(),
    }
    if base_instructions:
        start_kwargs["baseInstructions"] = base_instructions

    def _create_and_persist(codex: Codex) -> str:
        response = codex._client.thread_start(start_kwargs)
        thread = response.thread
        codex._client.thread_set_name(thread.id, _initial_thread_name(name))
        return thread.id

    # See ``spawn_new_session``: retry the open+create when the ``thread_set_name``
    # persist races the CODEX_HOME state-DB migration. Safe to retry despite the
    # non-idempotent ``thread_start`` because a locked persist exits the
    # app-server before anything reaches disk.
    return run_codex_op_with_retry(lambda: Codex(config=config), _create_and_persist)


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
    if not instance.is_active:
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
    instance.refresh_from_db()
    if not instance.is_active:
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
        # Legacy scope launches could briefly store the systemd-run client pid
        # until the worker recorded its real pid. Do not treat that client as
        # interruptible.
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
    return instance


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
        message = "systemctl failed to kill systemd Codex worker"
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
        _force_kill_instance(instance)
    except ProcessLookupError:
        return
    except OSError:
        logger.warning(
            "failed to reap systemd cgroup %s for instance %s",
            instance.systemd_scope_unit,
            instance.pk,
        )


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


def reconcile_dead() -> int:
    """Mark workers as failed whose PID is no longer alive.

    A worker that crashed before writing its terminal status leaves a row
    stuck in ``starting``/``running``. We sweep those rows and mark them
    failed so the UI doesn't show a permanently-pending turn.
    """
    _reap_finished_workers()
    pending = CodexInstance.objects.filter(
        status__in=CodexInstance.ACTIVE_STATUSES
    )
    updated = _mark_dead_instances_failed(pending)
    _reconcile_terminal_workflow_instances()
    reconcile_orphaned_workers()
    retry_failed_input_image_cleanups()
    _prune_reaped_workers()
    return updated


# Grace after a worker commits its terminal status before the orphan reaper may
# kill it. ``codex_worker`` runs ``_notify_system_agents`` (which can spawn
# follow-up turns) and input-image cleanup *after* the terminal commit, so a
# still-live terminal worker inside this window is finishing those hooks rather
# than leaked. Generous so even a slow hook (e.g. spawning a follow-up workflow)
# completes; a genuinely leaked worker is still reaped one grace later.
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
    if not proc_root.exists():
        return
    marker = (manage_py or _our_manage_py()).encode()
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
    with _TRACKED_WORKER_PROCS_LOCK:
        return pid in _TRACKED_WORKER_PROCS


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
        logger.exception("could not read worker rows; skipping orphan reap")
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
        logger.warning("reaped %s orphaned codex worker process(es)", killed)
    return killed


def _kill_orphaned_worker(pid: int, instance_id: int) -> bool:
    """Force-kill a leaked worker (and its app-server child); report success."""
    instance = None
    try:
        instance = CodexInstance.objects.filter(pk=instance_id).first()
    except Exception:
        logger.exception("could not load instance %s for orphan reap", instance_id)
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
        and _pid_is_our_worker(pid, instance_id, require_session_leader=False)
        and not _pid_is_our_worker(pid, instance_id)
    ):
        scope_unit = (
            _worker_unit_from_pid_cgroup(pid, instance_id)
            or _scope_unit_for_instance(instance_id)
        )
    if scope_unit:
        # The unit name (``hitch-codex-worker-<id>``) is not
        # deployment-unique, so ``systemctl kill <unit>`` could hit another
        # checkout's reused unit if our systemd worker exited since the scan.
        # Reverify the scanned pid is still our deployment's worker for this
        # instance (systemd workers are not session leaders) before signaling.
        if not _pid_is_our_worker(pid, instance_id, require_session_leader=False):
            return False
        # Carry the effective systemd unit on the target even when the row had it
        # empty (a derived unit), so _force_kill_instance signals the unit
        # rather than falling back to killpg.
        target = instance or CodexInstance(pid=pid)
        target.systemd_scope_unit = scope_unit
        try:
            _force_kill_instance(target)
            return True
        except ProcessLookupError:
            return False
        except OSError:
            logger.warning(
                "failed to kill orphaned systemd worker for instance %s", instance_id
            )
            return False
    # Re-verify identity right before signaling: the pid could have been recycled
    # since the /proc scan, and we must never SIGKILL an unrelated process group.
    if not _pid_is_our_worker(pid, instance_id):
        return False
    try:
        os.killpg(pid, signal.SIGKILL)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        logger.warning(
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
            logger.warning("failed to SIGKILL codex app-server pid %s", pid)
            continue
        killed += 1
    if killed:
        logger.warning("nuked %s codex app-server process(es)", killed)
    return killed


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
        from hitch.main import system_agents

        if system_agents.auto_review_intentionally_skipped(instance):
            return False
    except Exception:
        # If we cannot determine intent, prefer leaving a completed turn intact
        # over rewriting a successful result as a false failure.
        logger.exception(
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
        logger.exception("could not load reaped instance %s for finalize", instance_id)
        return
    if instance is None or instance.status not in (
        CodexInstance.STATUS_COMPLETED,
        CodexInstance.STATUS_FAILED,
    ):
        return
    if instance.status == CodexInstance.STATUS_FAILED:
        _resolve_dangling_requests(instance.pk)
    # Idempotent finish routing for demo/system-agent/workflow-owned rows (a
    # no-op for a plain user turn, which the lost-auto-review check below covers).
    _notify_system_agents_if_needed(instance)
    cleanup_requested_input_images_for(instance)
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


# Floor on how often the request/SSE-path debounce lets the global sweep run.
# Short enough that a crashed worker still clears within a couple seconds, long
# enough that a burst of concurrent page loads / SSE reconnects collapses to one
# sweep instead of one per request.
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
    _reap_finished_workers()
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
    _reap_finished_workers()
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
        if worker_is_alive(instance):
            continue
        error = _dead_worker_error(instance)
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
                cleanup_requested_input_images_for(instance)
            continue
        _resolve_dangling_requests(instance.pk)
        instance.refresh_from_db()
        # The worker died without reporting completion, so its systemd cgroup may
        # still hold grandchildren the codex sandbox reparented into their own
        # session (a process-group signal would miss them). Reap the cgroup so a
        # leaked ``cargo bench`` can't hold memory long after the worker is gone.
        _reap_scope_cgroup(instance)
        _notify_system_agents_if_needed(instance)
        cleanup_requested_input_images_for(instance)
        updated += 1
    return updated


def _dead_worker_error(instance: CodexInstance) -> str:
    if instance.error:
        return instance.error
    detail = _dead_worker_last_event_detail(instance.events_path)
    log_detail = _dead_worker_log_detail(instance.pk)
    if detail:
        message = f"worker process exited before reporting completion; last event: {detail}"
        if log_detail:
            message = f"{message}; worker log: {log_detail}"
        return message
    if log_detail:
        return (
            "worker process exited before reporting completion; "
            f"worker log: {log_detail}"
        )
    return "worker process exited before reporting completion"


def _dead_worker_last_event_detail(events_path: str) -> str:
    if not events_path:
        return ""
    try:
        with Path(events_path).open(encoding="utf-8") as events_file:
            recent_lines = deque(events_file, maxlen=200)
    except OSError:
        return ""

    auto_approval_started_detail = ""
    completed_auto_approval_keys: set[str] = set()
    failed_command_detail = ""
    fallback_detail = ""
    for line in reversed(recent_lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        method = event.get("method")
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        if method == "item/autoApprovalReview/completed":
            completed_auto_approval_keys.add(_auto_approval_event_key(payload))
            detail = _auto_approval_event_detail(payload)
            if detail:
                return detail
        if (
            method == "item/autoApprovalReview/started"
            and _auto_approval_event_key(payload) not in completed_auto_approval_keys
        ):
            detail = _auto_approval_event_detail(payload)
            if detail and not auto_approval_started_detail:
                auto_approval_started_detail = detail
        if method == "error":
            detail = _error_event_detail(payload)
            if detail:
                return detail
        if method == "item/completed" and not failed_command_detail:
            failed_command_detail = _failed_command_event_detail(payload)
        if not fallback_detail:
            fallback_detail = _last_visible_event_detail(method, payload)
    return failed_command_detail or auto_approval_started_detail or fallback_detail


def _auto_approval_event_detail(payload: dict[str, Any]) -> str:
    review = payload.get("review")
    if not isinstance(review, dict):
        return ""
    status = str(review.get("status") or "").strip()
    if not status or status == "approved":
        return ""
    action = payload.get("action")
    command = ""
    if isinstance(action, dict):
        command = _compact_error_detail(str(action.get("command") or ""), limit=160)
    rationale = _compact_error_detail(str(review.get("rationale") or ""), limit=240)
    status_text = _approval_status_text(status)
    if command and rationale:
        return f"auto-approval review {status_text} for `{command}`: {rationale}"
    if command:
        return f"auto-approval review {status_text} for `{command}`"
    if rationale:
        return f"auto-approval review {status_text}: {rationale}"
    return f"auto-approval review {status_text}"


def _auto_approval_event_key(payload: dict[str, Any]) -> str:
    action = payload.get("action")
    if not isinstance(action, dict):
        return ""
    try:
        return json.dumps(action, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(action)


def _error_event_detail(payload: dict[str, Any]) -> str:
    error = payload.get("error")
    if not isinstance(error, dict):
        return ""
    message = _compact_error_detail(str(error.get("message") or ""), limit=160)
    details = _compact_error_detail(
        str(error.get("additionalDetails") or ""), limit=240
    )
    if message and details:
        return f"{message}: {details}"
    return message or details


def _failed_command_event_detail(payload: dict[str, Any]) -> str:
    item = payload.get("item")
    if not isinstance(item, dict):
        return ""
    if item.get("type") != "commandExecution" or item.get("status") != "failed":
        return ""
    command = _compact_error_detail(str(item.get("command") or ""), limit=200)
    if command:
        return f"command failed: `{command}`"
    return "command failed"


def _last_visible_event_detail(method: object, payload: dict[str, Any]) -> str:
    if method not in ("item/started", "item/completed"):
        return ""
    item = payload.get("item")
    if not isinstance(item, dict):
        return ""
    item_type = item.get("type")
    if item_type == "commandExecution":
        command = _compact_error_detail(str(item.get("command") or ""), limit=200)
        if not command:
            return ""
        status = str(item.get("status") or "").strip()
        if method == "item/started" or status == "inProgress":
            return f"command started: `{command}`"
        if status == "completed":
            return f"command completed: `{command}`"
        return f"command {status}: `{command}`" if status else f"command: `{command}`"
    if item_type == "agentMessage":
        text = _compact_error_detail(str(item.get("text") or ""), limit=240)
        if text:
            return f"agent last said: {text}"
    return ""


def _dead_worker_log_detail(instance_id: int) -> str:
    if not worker_log_io_enabled():
        return ""
    try:
        with worker_log_path(instance_id).open(encoding="utf-8", errors="replace") as log_file:
            recent_lines = deque(log_file, maxlen=20)
    except OSError:
        return ""
    tail = [line.strip() for line in recent_lines if line.strip()]
    if not tail:
        return ""
    return _compact_error_detail(" | ".join(tail[-3:]), limit=500)


def _approval_status_text(status: str) -> str:
    if status == "inProgress":
        return "in progress"
    if status == "timedOut":
        return "timed out"
    return status


def _compact_error_detail(value: str, *, limit: int) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."


def _prune_reaped_workers() -> None:
    with _TRACKED_WORKER_PROCS_LOCK:
        reaped_workers = set(_REAPED_WORKERS)
    if not reaped_workers:
        return
    active_workers = set(
        CodexInstance.objects.filter(
            pk__in=[instance_id for _, instance_id in reaped_workers],
            status__in=CodexInstance.ACTIVE_STATUSES,
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


def _codex_bin() -> str | None:
    """Resolve the ``codex`` binary path for AppServerConfig.

    Returning None lets AppServerConfig fall back to the pinned runtime
    package; we prefer an explicit PATH lookup so dev environments with a
    locally-built codex pick that up.
    """
    return shutil.which("codex")


# Codex's native env var for relocating its SQLite databases away from
# ``$CODEX_HOME`` (state/src/lib.rs SQLITE_HOME_ENV). The SDK merges
# ``AppServerConfig.env`` onto the inherited environment, so setting it here is
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
        home = worker_sqlite_slot_home(slot)
        home.mkdir(parents=True, exist_ok=True)
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
) -> AppServerConfig:
    memory_value = "true" if enable_memories else "false"
    overrides = [f"features.memories={memory_value}"]
    web_search_mode = _normalized_web_search_mode(web_search_mode)
    if web_search_mode:
        overrides.append(f"web_search={json.dumps(web_search_mode)}")
    # Stamp every app-server we spawn with this deployment's id (merged onto the
    # inherited environment by the SDK). The profile "nuke" sweep scopes its
    # SIGKILLs to this marker so a second checkout sharing the resolved codex
    # binary -- whose app-server command lines are otherwise identical -- is
    # never swept.
    env = {_APP_SERVER_DEPLOYMENT_ENV: _app_server_deployment_id()}
    resolved_home = sqlite_home if sqlite_home is not None else _default_sqlite_home()
    if resolved_home is not None:
        env[_CODEX_SQLITE_HOME_ENV] = os.fspath(resolved_home)
    return AppServerConfig(
        codex_bin=_codex_bin(),
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
_APPSERVER_START_MAX_ATTEMPTS = 10
_APPSERVER_WORKER_START_MAX_ATTEMPTS = 24
_APPSERVER_START_BACKOFF_BASE_SECONDS = 0.2
# Capped per-attempt backoff; request-path callers spend about 26s in Hitch
# backoff before giving up, on top of Codex's own 5s SQLite busy timeout per
# failed attempt. Detached workers use a larger attempt count.
_APPSERVER_START_BACKOFF_MAX_SECONDS = 5.0


def _start_codex_with_retry(factory: Callable[[], Codex]) -> Codex:
    """Call ``factory`` to construct a ``Codex``, retrying a locked init.

    ``factory`` is a zero-arg closure (typically ``lambda: Codex(config=...)``)
    so the call site keeps referencing its own module-local ``Codex`` symbol --
    important both for clarity and so tests that patch ``<module>.Codex`` still
    intercept construction. Only a ``TransportClosedError`` carrying the state
    DB's "database is locked" message is retried (the transient state-DB
    migration race); any other startup failure propagates immediately.
    """
    last_error: TransportClosedError | None = None
    for attempt in range(_APPSERVER_START_MAX_ATTEMPTS):
        try:
            return factory()
        except TransportClosedError as exc:
            if not is_database_locked_error(exc):
                raise
            last_error = exc
            logger.warning(
                "Codex app-server state DB locked on start (attempt %s/%s)",
                attempt + 1,
                _APPSERVER_START_MAX_ATTEMPTS,
            )
        if attempt + 1 < _APPSERVER_START_MAX_ATTEMPTS:
            backoff = min(
                _APPSERVER_START_BACKOFF_BASE_SECONDS * (2**attempt),
                _APPSERVER_START_BACKOFF_MAX_SECONDS,
            )
            time.sleep(backoff)
    assert last_error is not None
    raise last_error


def run_codex_op_with_retry(
    factory: Callable[[], Codex],
    operation: Callable[[Codex], T],
) -> T:
    """Open a fresh app-server, run ``operation`` against it, and retry the whole
    open+operation when a contended CODEX_HOME state DB surfaces.

    ``_start_codex_with_retry`` only guards *construction*. But the Codex
    runtime's state-DB migration/backfill path (no SQLITE_BUSY retry,
    openai/codex#20213) is also reached lazily by operations like
    ``thread_resume`` -- resuming a thread persisted by another worker migrates
    that thread's rows -- and a lock there *exits the app-server mid-operation*,
    surfacing as a ``TransportClosedError`` the construction-only retry never
    sees. Because the server is gone, recovery means reconstructing it, so this
    is a single retry loop spanning both construction and the operation: each
    attempt builds a fresh app-server and then runs ``operation`` against it. A
    locked ``TransportClosedError`` from *either* phase is retried by this one
    loop, bounded at ``_APPSERVER_START_MAX_ATTEMPTS`` rather than nesting
    ``_start_codex_with_retry``'s loop inside this one. ``operation`` must
    therefore be idempotent (it may run more than once) -- safe for reads like
    ``thread_resume``/``thread_list``, not for turn starts. Non-locked errors
    (including ``Http404`` the operation may raise) propagate immediately.
    """
    last_error: TransportClosedError | None = None
    for attempt in range(_APPSERVER_START_MAX_ATTEMPTS):
        try:
            codex = factory()
            with codex as entered:
                return operation(entered)
        except TransportClosedError as exc:
            if not is_database_locked_error(exc):
                raise
            last_error = exc
            logger.warning(
                "Codex app-server state DB locked during open+operation "
                "(attempt %s/%s)",
                attempt + 1,
                _APPSERVER_START_MAX_ATTEMPTS,
            )
        if attempt + 1 < _APPSERVER_START_MAX_ATTEMPTS:
            backoff = min(
                _APPSERVER_START_BACKOFF_BASE_SECONDS * (2**attempt),
                _APPSERVER_START_BACKOFF_MAX_SECONDS,
            )
            time.sleep(backoff)
    assert last_error is not None
    raise last_error


def run_borrowed_op_with_retry(
    codex_factory: Callable[..., Codex],
    operation: Callable[[Codex], T],
    *,
    enable_memories: bool = False,
    web_search_mode: str | None = None,
) -> T:
    """Run ``operation`` against a *warm* pooled app-server when one exists,
    falling back to a retrying cold open only when the pool is empty.

    ``run_codex_op_with_retry`` always cold-opens a fresh app-server, so every
    call pays the CODEX_HOME init write that contends on the state-DB writer lock
    -- the failure mode behind "failed to initialize sqlite state runtime ...
    database is locked" on request paths like the session-detail resume. This
    instead borrows an already-initialized server from the shared pool first, so
    the steady-state request does *no* init write at all. ``operation`` must be
    idempotent (a warm server that dies on a locked op is dropped and the call
    falls back to the cold path, so it may run more than once) -- safe for reads
    like ``thread_resume``/``thread_list``, not for turn starts.
    """
    config = app_server_config(
        enable_memories=enable_memories, web_search_mode=web_search_mode
    )
    if _shared_pool_enabled():
        key = _pool_key(enable_memories, web_search_mode)
        warm = _SHARED_POOL.checkout_warm_only(key)
        if warm is not None:
            healthy = True
            try:
                return operation(warm)
            except TransportClosedError as exc:
                healthy = False
                if not is_database_locked_error(exc):
                    raise
                # The warm server exited on a locked op; drop it and fall through
                # to a fresh cold open (which retries the locked init itself).
                logger.warning(
                    "warm app-server state DB locked during borrowed op; "
                    "falling back to a fresh open"
                )
            except BaseException:
                healthy = False
                raise
            finally:
                _SHARED_POOL.release(key, warm, healthy=healthy)
    return run_codex_op_with_retry(lambda: codex_factory(config=config), operation)


def start_codex(config: AppServerConfig) -> Codex:
    """Construct a long-lived Codex app-server with ``_start_codex_with_retry``.

    For callers that own and reuse one app-server across many operations (e.g.
    the background scheduler) rather than opening a fresh one per use; the
    caller is responsible for ``close()``. Reusing a single app-server keeps its
    state DB initialized once instead of racing a new init on every operation.
    """
    return _start_codex_with_retry(lambda: Codex(config=config))


@contextlib.contextmanager
def open_codex(factory: Callable[[], Codex]) -> Generator[Codex]:
    """Open a Codex app-server, tolerating a contended state-DB init.

    Replaces ``with Codex(config=config) as codex`` with
    ``with open_codex(lambda: Codex(config=config)) as codex``: a locked
    state-DB init is retried (see ``_start_codex_with_retry``), then the
    constructed session's own ``__enter__``/``__exit__`` run as usual so it is
    closed on exit.
    """
    codex = _start_codex_with_retry(factory)
    with codex as entered:
        yield entered


@contextlib.contextmanager
def open_codex_resumed(
    factory: Callable[[], Codex],
    *,
    thread_id: str,
    resume_kwargs: dict[str, Any] | None = None,
    configure: Callable[[Codex], None] | None = None,
) -> Generator[tuple[Codex, Any]]:
    """Open an app-server, run ``configure`` then ``thread_resume``, retrying a
    locked CODEX_HOME state DB across the whole sequence, and yield the live
    ``(codex, thread)`` so the caller can start a (non-idempotent) turn.

    ``open_codex`` only retries *construction*. But the state-DB
    migration/backfill path (no SQLITE_BUSY retry, openai/codex#20213) is also
    reached lazily by ``thread_resume`` -- resuming a thread another worker
    persisted migrates its rows -- and a lock there exits the app-server
    mid-resume as a ``TransportClosedError`` the construction-only retry never
    sees. This retries construction+configure+resume together. ``thread_resume``
    is idempotent, so re-running it is safe. ``configure`` runs once per attempt
    against the attempt's fresh server (e.g. to install approval/notification
    handlers); a retried attempt discards its server, so any helper threads
    ``configure`` starts must tolerate that server closing under them (the
    worker's goal forwarder exits cleanly when its source transport closes).
    """
    resume_kwargs = resume_kwargs or {}
    last_error: TransportClosedError | None = None
    for attempt in range(_APPSERVER_WORKER_START_MAX_ATTEMPTS):
        codex = None
        entered = None
        try:
            codex = factory()
            entered = codex.__enter__()
            if configure is not None:
                configure(entered)
            thread = entered.thread_resume(thread_id, **resume_kwargs)
        except TransportClosedError as exc:
            if entered is not None and codex is not None:
                codex.__exit__(type(exc), exc, exc.__traceback__)
            if not is_database_locked_error(exc):
                raise
            last_error = exc
            logger.warning(
                "Codex app-server state DB locked during worker open+resume "
                "(attempt %s/%s)",
                attempt + 1,
                _APPSERVER_WORKER_START_MAX_ATTEMPTS,
            )
            if attempt + 1 < _APPSERVER_WORKER_START_MAX_ATTEMPTS:
                time.sleep(
                    min(
                        _APPSERVER_START_BACKOFF_BASE_SECONDS * (2**attempt),
                        _APPSERVER_START_BACKOFF_MAX_SECONDS,
                    )
            )
            continue
        except BaseException as exc:
            if entered is not None and codex is not None:
                codex.__exit__(type(exc), exc, exc.__traceback__)
            raise
        try:
            yield entered, thread
        except BaseException as exc:
            codex.__exit__(type(exc), exc, exc.__traceback__)
            raise
        codex.__exit__(None, None, None)
        return
    assert last_error is not None
    raise last_error


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
_SHARED_POOL_MAX = 4

# (enable_memories, normalized web_search_mode). In practice every web call uses
# the default web_search_mode, so the live key space is just enable_memories.
_ConfigKey = tuple[bool, str | None]


def _pool_key(enable_memories: bool, web_search_mode: str | None) -> _ConfigKey:
    return (enable_memories, _normalized_web_search_mode(web_search_mode))


def _close_quietly(codex: Codex) -> None:
    with contextlib.suppress(Exception):
        codex.close()


def _codex_is_alive(codex: Codex) -> bool:
    """Best-effort check that the app-server subprocess is still running.

    A pooled server is long-lived, so an idle one may have exited (crash, OOM
    kill) since it was returned -- or been returned looking healthy by a borrow
    whose helper swallowed the ``TransportClosedError``. Handing such a dead
    server back would surface an error to a single request that the old
    open-per-call path never hit, so the pool drops it on checkout and reuses a
    live one instead. Reaches into the SDK client's process handle the way our
    call sites already reach into ``codex._client``; ``poll()`` is a cheap,
    non-blocking liveness probe (no app-server round-trip).
    """
    proc = getattr(getattr(codex, "_client", None), "_proc", None)
    if proc is None:
        return False
    try:
        return proc.poll() is None
    except Exception:
        return False


class _SharedCodexPool:
    """Bounded pool of long-lived app-servers with exclusive checkout.

    Only one borrower drives a given ``Codex`` at a time, so reuse never relies
    on the SDK being safe for concurrent stdin writes from multiple request
    threads. Checkout skips (and closes) idle servers whose subprocess has died
    so a stale transport never reaches a borrower; a borrow that dies mid-use
    drops its instance -- mirroring ``_SchedulerCodex.reset`` -- so the next
    borrow reconnects. The cap bounds total warm servers; a full pool evicts an
    idle server from *another* config key rather than refusing to keep this
    key's, so no key is starved of a warm server. Checkouts past the cap
    construct (and on return close) a private server rather than blocking.
    """

    def __init__(self, max_size: int = _SHARED_POOL_MAX) -> None:
        self._lock = threading.Lock()
        self._idle: dict[_ConfigKey, deque[Codex]] = {}
        self._in_use = 0
        self._max = max_size
        # Config keys borrowed recently, in LRU order (oldest first), so the
        # keepalive knows which keys to keep warm -- a memories-enabled session
        # uses a different key than the default and would otherwise cold-open on
        # its first render after idle. A dict is used as an ordered set.
        self._seen_keys: dict[_ConfigKey, None] = {}

    def _note_key(self, key: _ConfigKey) -> None:
        """Record ``key`` as most-recently used. Caller holds ``self._lock``."""
        self._seen_keys.pop(key, None)
        self._seen_keys[key] = None

    def _total_warm(self) -> int:
        return self._in_use + sum(len(idle) for idle in self._idle.values())

    def warm_target_keys(self) -> list[_ConfigKey]:
        """Keys the keepalive should keep warm, capped at pool capacity.

        Warming more keys than the pool can hold (``_max``) would have each tick
        cold-open the keys that the previous tick evicted -- reintroducing the
        init churn the keepalive exists to avoid. So return the default key
        (always kept warm) plus the most-recently-used other keys, up to ``_max``
        total.
        """
        default = _pool_key(enable_memories=False, web_search_mode=None)
        with self._lock:
            recent = [k for k in reversed(self._seen_keys) if k != default]
        return [default, *recent[: max(self._max - 1, 0)]]

    def _pop_idle_other_key(self, key: _ConfigKey) -> Codex | None:
        """Pop the oldest idle server belonging to a different config key."""
        for other_key, servers in self._idle.items():
            if other_key != key and servers:
                return servers.pop()
        return None

    def checkout(self, key: _ConfigKey, factory: Callable[[], Codex]) -> Codex:
        dead: list[Codex] = []
        with self._lock:
            self._note_key(key)
            idle = self._idle.get(key)
            reused: Codex | None = None
            while idle:
                candidate = idle.pop()
                if _codex_is_alive(candidate):
                    reused = candidate
                    break
                dead.append(candidate)
            self._in_use += 1
        for stale in dead:
            _close_quietly(stale)
        if reused is not None:
            return reused
        # Construct outside the structure lock: a cold start spawns a
        # subprocess and may retry a locked state-DB init.
        try:
            return _start_codex_with_retry(factory)
        except BaseException:
            with self._lock:
                self._in_use -= 1
            raise

    def checkout_warm_only(self, key: _ConfigKey) -> Codex | None:
        """Check out a live idle server without ever constructing one.

        Returns ``None`` when no warm server is available rather than cold-opening
        (and so re-initializing the CODEX_HOME state DB). Lets callers prefer a
        warm server on the request path and fall back to their own retrying cold
        open only when the pool is empty -- the construction write is exactly what
        contends on the state-DB writer lock, so skipping it when a warm server
        exists avoids the lock entirely.
        """
        dead: list[Codex] = []
        reused: Codex | None = None
        with self._lock:
            self._note_key(key)
            idle = self._idle.get(key)
            while idle:
                candidate = idle.pop()
                if _codex_is_alive(candidate):
                    reused = candidate
                    break
                dead.append(candidate)
            if reused is not None:
                self._in_use += 1
        for stale in dead:
            _close_quietly(stale)
        return reused

    def release(self, key: _ConfigKey, codex: Codex, *, healthy: bool) -> None:
        to_close: Codex | None = None
        with self._lock:
            self._in_use -= 1
            if healthy:
                if self._total_warm() < self._max:
                    self._idle.setdefault(key, deque()).appendleft(codex)
                    return
                # Pool full: keep this key's server by evicting an idle server
                # from another key. If only this key is idle we are already at
                # our share, so close the returning server instead.
                to_close = self._pop_idle_other_key(key)
                if to_close is not None:
                    self._idle.setdefault(key, deque()).appendleft(codex)
            if to_close is None:
                to_close = codex
        _close_quietly(to_close)

    def close_all(self) -> None:
        with self._lock:
            idle, self._idle = self._idle, {}
        for servers in idle.values():
            for codex in servers:
                _close_quietly(codex)


_SHARED_POOL = _SharedCodexPool()
atexit.register(_SHARED_POOL.close_all)


def _shared_pool_enabled() -> bool:
    # Under tests each case patches ``Codex`` fresh and expects construction per
    # call, so caching pooled instances across cases would leak mocks. The pool
    # itself is covered directly by test_codex_pool_shared.
    return not getattr(settings, "TESTING", False)


# How often the keepalive exercises a warm pooled server. Short relative to any
# plausible app-server idle timeout so the server stays warm, long enough not to
# add meaningful load.
_KEEPALIVE_INTERVAL_SECONDS = 30
_keepalive = server_lifecycle.SchedulerHandle(
    thread_name="hitch-codex-pool-keepalive"
)


def _codex_pool_keepalive_enabled() -> bool:
    """Whether this process serves requests (and so uses the shared pool).

    Independent of the background schedulers: the keepalive must run wherever the
    request-path pool is enabled, even on a server that disabled the maintenance
    scheduler (e.g. ``HITCH_WORKFLOW_MAINTENANCE_SCHEDULER=0`` because maintenance
    runs elsewhere). Mirrors the schedulers' "real server process" gate so it
    never starts under management commands, migrations, or tests.
    """
    return server_lifecycle.background_work_enabled(
        include_wsgi_server_commands=True
    )


def start_codex_pool_keepalive() -> bool:
    """Start a daemon that keeps one warm pooled app-server present and healthy.

    The shared pool only fills when a request borrows, and an idle pooled server
    can die (laptop sleep, OOM, a Codex-side idle exit) with nothing noticing
    until the next checkout cold-opens a replacement. After an idle stretch that
    makes the first request -- and the session-detail resume -- cold-open and
    race the per-turn worker on the CODEX_HOME init write, which is the
    "database is locked" users hit first thing in the morning. This periodically
    borrows each used config key and runs one cheap read, so an initialized
    server is already warm when the user returns and a dead one is rebuilt
    *before* they hit it rather than on their request.
    """
    if not _codex_pool_keepalive_enabled():
        return False
    return _keepalive.start(_codex_pool_keepalive_loop)


def _codex_pool_keepalive_loop() -> None:
    stop = threading.Event()
    while True:
        _codex_pool_keepalive_tick()
        stop.wait(_KEEPALIVE_INTERVAL_SECONDS)


def _codex_pool_keepalive_tick() -> None:
    """Borrow and exercise one server per used config key with a cheap read.

    Borrowing reconstructs a dead pooled server (checkout drops one whose
    subprocess has exited) and the read both keeps the app-server from idling out
    and surfaces a wedged-but-alive server -- a failed probe makes ``borrow_codex``
    drop it, so the next tick rebuilds a healthy one. Every key ever borrowed is
    kept warm, not just the default: a memories-enabled session uses a different
    pool key and would otherwise cold-open (and race the state-DB init lock) on
    its first detail render after idle. Best-effort: a failure on one key is
    logged and retried next tick rather than killing the daemon.
    """
    for enable_memories, web_search_mode in _SHARED_POOL.warm_target_keys():
        try:
            with borrow_codex(
                Codex,
                enable_memories=enable_memories,
                web_search_mode=web_search_mode,
            ) as codex:
                codex.thread_list(limit=1, use_state_db_only=True)
        except Exception:
            logger.warning(
                "codex pool keepalive probe failed for key "
                "(enable_memories=%s, web_search_mode=%s)",
                enable_memories,
                web_search_mode,
                exc_info=True,
            )


@contextlib.contextmanager
def borrow_codex(
    codex_factory: Callable[..., Codex],
    *,
    enable_memories: bool = False,
    web_search_mode: str | None = None,
) -> Generator[Codex]:
    """Borrow a warm, already-initialized app-server from the shared pool.

    ``codex_factory`` is the caller's ``Codex`` symbol (constructed as
    ``codex_factory(config=...)``); keeping construction at the call site lets
    callers tune the config and keeps test patches on the caller's module
    effective. Steady-state borrows reuse an idle long-lived server with no
    subprocess spawn; cold construction still goes through
    ``_start_codex_with_retry`` so a genuine state-DB init race is retried. The
    pool owns the server's lifecycle, so -- unlike ``open_codex`` -- the yielded
    server is not entered/closed per borrow.
    """
    config = app_server_config(
        enable_memories=enable_memories, web_search_mode=web_search_mode
    )

    def factory() -> Codex:
        return codex_factory(config=config)

    if not _shared_pool_enabled():
        with open_codex(factory) as codex:
            yield codex
        return

    key = _pool_key(enable_memories, web_search_mode)
    codex = _SHARED_POOL.checkout(key, factory)
    healthy = True
    try:
        yield codex
    except BaseException:
        # A failed borrow may have killed the transport; drop rather than reuse.
        healthy = False
        raise
    finally:
        _SHARED_POOL.release(key, codex, healthy=healthy)


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
            approval_mode_live_editable=(
                approval_mode in CodexInstance.LIVE_EDITABLE_APPROVAL_MODES
            ),
            web_search_mode=web_search_mode or "",
            plan_mode=plan_mode,
            auto_pr_enabled=auto_pr_enabled,
            auto_qa_enabled=auto_qa_enabled,
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
        systemd_run = _systemd_run_for_isolation(requested_isolation)
        if systemd_run is None:
            proc = _popen_detached(argv, env=env, stderr=stderr)
            return WorkerLaunch(pid=proc.pid, proc=proc)
        return _launch_systemd_worker(
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
    stderr: Any = subprocess.DEVNULL,
    stderr_capture: BinaryIO | None = None,
) -> WorkerLaunch:
    _ensure_systemd_worker_slice()
    if stderr_capture is not None:
        stderr_offset = stderr_capture.tell()
        proc = _popen_detached(
            _systemd_scope_argv(
                systemd_run=systemd_run,
                scope_unit=scope_unit,
                worker_argv=worker_argv,
                env=env,
                stderr_log_path=_stderr_log_path(stderr_capture),
            ),
            env=env,
            stderr=stderr,
        )
        client_exited = _check_systemd_run_start_result(
            proc,
            scope_unit,
            stderr_capture,
            stderr_offset=stderr_offset,
        )
        return WorkerLaunch(
            pid=0,
            proc=None if client_exited else proc,
            scope_unit=scope_unit,
        )
    with tempfile.TemporaryFile() as stderr_file:
        proc = _popen_detached(
            _systemd_scope_argv(
                systemd_run=systemd_run,
                scope_unit=scope_unit,
                worker_argv=worker_argv,
                env=env,
            ),
            env=env,
            stderr=stderr_file,
        )
        client_exited = _check_systemd_run_start_result(proc, scope_unit, stderr_file)
    return WorkerLaunch(
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


# cgroup-v2 cpu.weight bounds (CPUWeight=1..10000, default 100).
_CPU_WEIGHT_MIN = 1
_CPU_WEIGHT_MAX = 10000
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
    if not _CPU_WEIGHT_MIN <= weight <= _CPU_WEIGHT_MAX:
        logger.warning(
            "ignoring CODEX_PARENT_SLICE_CPU_WEIGHT=%r: outside the cgroup-v2 "
            "range %d-%d",
            raw,
            _CPU_WEIGHT_MIN,
            _CPU_WEIGHT_MAX,
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
    if swap and _is_finite_limit(hard):
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
    if slice_swap is None or not _is_finite_limit(slice_swap):
        return

    worker_swap = _effective_swap_cap(
        "CODEX_WORKER_MEMORY_MAX",
        "CODEX_WORKER_MEMORY_SWAP_MAX",
    )
    if worker_swap is not None and _is_finite_limit(worker_swap):
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
    if _is_finite_limit(worker_hard) and not _is_finite_limit(worker_swap_raw):
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
    slice_unit = _worker_slice()
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
    _apply_slice_properties(slice_unit, _systemd_worker_slice_properties())
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
    return f"hitch-codex-worker-{instance_id}.service"


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
    worker_slice = _worker_slice()
    if worker_slice:
        argv.append(f"--slice={worker_slice}")
    if env is not None:
        argv.extend(_systemd_env_args(env))
    argv.append("--property=StandardOutput=null")
    if stderr_log_path:
        argv.append(f"--property=StandardError=append:{stderr_log_path}")
    else:
        argv.append("--property=StandardError=null")
    for property_value in _memory_cgroup_properties(
        "CODEX_WORKER_MEMORY_HIGH",
        "CODEX_WORKER_MEMORY_MAX",
        "CODEX_WORKER_MEMORY_SWAP_MAX",
    ):
        argv.append(f"--property={property_value}")
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
