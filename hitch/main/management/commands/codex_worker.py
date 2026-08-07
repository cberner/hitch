"""Detached worker that runs one Codex turn and streams events to disk.

Invoked by ``codex_pool._launch_worker_process`` as a fresh-session Django
``manage.py`` subprocess. The lifecycle is:

  1. Mark the CodexInstance as ``running`` and open its JSONL events file
     for line-buffered append.
  2. Start a Codex app-server, resume the thread by id, and submit ``prompt``
     as a single turn.
  3. Stream every notification produced by that turn into the events file.
  4. On success mark the row ``completed``; on any exception mark ``failed``
     and write the exception message to ``error``.

The events file plus the status transitions on the row are the primary output;
the parent also redirects stderr to a durable per-worker log for crash
forensics.

Interactive browser prompts route through ``_make_approval_handler``: when
the SDK reader thread receives an approval or ``request_user_input`` request,
the handler creates a durable pending row, emits a synthetic event into the
events file (so the SSE stream surfaces it), and blocks on a short-poll loop
until the Django view records the browser response. The cap on that wait is
intentionally generous (``_APPROVAL_WAIT_SECONDS``) so a user who steps away
from the laptop doesn't lose the turn; on timeout approvals decline and
structured input returns empty answers.
"""

from __future__ import annotations

import contextlib
import dataclasses
import itertools
import json
import logging
import os
import signal
import sys
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any, Protocol, cast, override

from django.conf import settings
from django.core.management.base import BaseCommand, CommandParser
from django.utils import timezone
from openai_codex import (
    ApprovalMode,
    Codex,
    Input,
    LocalImageInput,
    TextInput,
    Thread,
    TurnHandle,
)
from openai_codex.generated.v2_all import (
    ApprovalsReviewer,
    AskForApproval,
    AskForApprovalValue,
    CollaborationMode,
    DangerFullAccessSandboxPolicy,
    LocalImageUserInput,
    ModeKind,
    ReadOnlySandboxPolicy,
    ReasoningEffort,
    SandboxPolicy,
    TextUserInput,
    Turn,
    TurnCompletedNotification,
    TurnError,
    TurnStartParams,
    TurnStatus,
    UserInput,
    WorkspaceWriteSandboxPolicy,
)
from openai_codex.generated.v2_all import (
    Settings as CodexModeSettings,
)
from openai_codex.models import Notification
from pydantic import BaseModel

from hitch.main.models import ApprovalRequest, CodexInstance, UserInputRequest
from hitch.main.runtime import disk_cleanup
from hitch.main.runtime.app_server_pool import open_codex_resumed
from hitch.main.runtime.codex_events import GOAL_METHODS
from hitch.main.runtime.codex_pool import (
    WorkerSqliteHome,
    _record_session_activity,
    acquire_worker_sqlite_home,
    app_server_config,
    cleanup_requested_input_images_for,
    codex_home_dir,
    control_path_for,
    discard_input_attachment_paths,
    prune_worker_logs_db,
    resolve_dangling_requests_for_instance,
    worker_log_io_enabled,
)
from hitch.main.runtime.codex_tools import (
    ToolContext,
    handle_dynamic_tool_call,
    is_dynamic_tool_call,
)

logger = logging.getLogger(__name__)

# JSON-RPC method names the SDK invokes on the client transport when codex's
# auto-reviewer escalates an action. Custom user-reviewer worker modes also
# route through these methods, so the same handler covers both interactive
# and rubber-stamp policies.
_APPROVAL_METHODS = frozenset(
    {
        "item/commandExecution/requestApproval",
        "item/fileChange/requestApproval",
    }
)
_USER_INPUT_METHODS = frozenset(
    {
        "request_user_input",
        "requestUserInput",
        "item/tool/request_user_input",
        "item/tool/requestUserInput",
    }
)
# Send default-mode instructions explicitly so approving a plan replaces the
# prior Plan Mode developer block in model-visible history.
_DEFAULT_COLLABORATION_INSTRUCTIONS = (
    "# Collaboration Mode: Default\n\n"
    "You are now in Default mode. Any previous instructions for other modes "
    "(e.g. Plan mode) are no longer active.\n\n"
    "Your active mode changes only when new developer instructions with a "
    "different `<collaboration_mode>...</collaboration_mode>` change it; user "
    "requests or tool descriptions do not change mode by themselves. Known "
    "mode names are Plan and Default.\n\n"
    "## request_user_input availability\n\n"
    "Use the `request_user_input` tool only when it is listed in the available "
    "tools for this turn.\n\n"
    "In Default mode, strongly prefer making reasonable assumptions and "
    "executing the user's request rather than stopping to ask questions. If "
    "you absolutely must ask a question because the answer cannot be "
    "discovered from local context and a reasonable assumption would be risky, "
    "ask the user directly with a concise plain-text question. Never write a "
    "multiple choice question as a textual assistant message.\n"
)

# How long the worker waits on a single approval before defaulting to
# ``decline``. 30 minutes leaves plenty of slack for a user who stepped
# away without holding a worker forever; ``decline`` is the safer
# no-answer fallback because the agent keeps running but the
# specific action is refused.
_APPROVAL_WAIT_SECONDS = 60 * 30

# Cadence at which the approval handler polls the row for a recorded
# decision. Short enough that a click feels immediate; long enough not to
# pin a CPU per pending approval.
_APPROVAL_POLL_INTERVAL = 0.2

# Cadence for the steer control thread's fallback poll. SIGUSR1 sets its
# wakeup event for prompt delivery; the poll catches queued payloads from
# STARTING-state races where the request process intentionally skipped signal.
_STEER_CONTROL_POLL_INTERVAL = 0.2

# Custom approval mode names not in ``ApprovalMode``. The SDK enum exposes
# only ``auto_review`` and ``deny_all``; these are wired through by
# bypassing ``thread.turn(approval_mode=)`` and posting an on-request
# policy with ``ApprovalsReviewer.user`` so every escalation reaches the
# client transport.
_PROMPT_USER = "prompt_user"
_APPROVE_ALL = "approve_all"
_DENY_ALL = "deny_all"
_USER_REVIEWER_APPROVAL_MODES = frozenset({_PROMPT_USER, _APPROVE_ALL})
_PLAN_MODE_REASONING_EFFORT = ReasoningEffort.medium
_DEFAULT_COLLABORATION_MODE = "default"

# Set by the SIGTERM handler so the turn-control paths know to call
# ``turn.interrupt()``. Plain module-level bool is fine — CPython makes
# single-attribute reads/writes atomic, and the signal handler is intentionally
# minimal (it must not make a blocking JSON-RPC call itself).
_cancel_requested = False

# Set by the active turn so the SIGTERM handler can wake the interrupt
# forwarder without making a blocking JSON-RPC call inside the handler.
_interrupt_wakeup: threading.Event | None = None

# Set by the active turn so the SIGUSR1 handler can wake the control-file
# forwarder without doing blocking JSON-RPC work from inside the handler.
_steer_wakeup: threading.Event | None = None


def _worker_log(instance_id: int, message: str) -> None:
    if not _worker_stderr_logging_enabled():
        return
    with contextlib.suppress(Exception):
        print(
            f"{timezone.now().isoformat()} codex_worker[{os.getpid()}] "
            f"instance={instance_id} {message}",
            file=sys.stderr,
            flush=True,
        )


def _worker_stderr_logging_enabled() -> bool:
    return worker_log_io_enabled()


def _worker_log_exception(exc: BaseException) -> None:
    if not _worker_stderr_logging_enabled():
        return
    with contextlib.suppress(Exception):
        traceback.print_exception(
            type(exc),
            exc,
            exc.__traceback__,
            file=sys.stderr,
        )


def _on_sigterm(_signum: int, _frame: Any) -> None:
    """Mark the active turn for graceful cancellation.

    Defers the actual SDK ``turn.interrupt()`` call to normal Python threads so
    we don't issue a blocking JSON-RPC request from inside a signal handler.
    The Django stop endpoint sends SIGTERM here; a follow-up click escalates to
    SIGKILL, which has no Python-level handler and tears the worker down
    immediately.
    """
    global _cancel_requested
    _cancel_requested = True
    if _interrupt_wakeup is not None:
        _interrupt_wakeup.set()


def _on_sigusr1(_signum: int, _frame: Any) -> None:
    """Mark the active turn for a control-file drain."""
    if _steer_wakeup is not None:
        _steer_wakeup.set()

# Map the cookie/CLI policy strings onto factories for the discriminated
# SandboxPolicy variants the SDK expects. Lookup misses (unknown / stale
# value) are treated as "no override" by ``_build_sandbox_policy``.
_SANDBOX_POLICY_BUILDERS: dict[str, Callable[[], SandboxPolicy]] = {
    "readOnly": lambda: SandboxPolicy(root=ReadOnlySandboxPolicy(type="readOnly")),
    "workspaceWrite": lambda: SandboxPolicy(
        root=WorkspaceWriteSandboxPolicy(type="workspaceWrite")
    ),
    "dangerFullAccess": lambda: SandboxPolicy(
        root=DangerFullAccessSandboxPolicy(type="dangerFullAccess")
    ),
}


class Command(BaseCommand):
    help = "Run one Codex turn for an existing CodexInstance and stream events to disk."

    @override
    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--instance-id", type=int, required=True)
        # The settings dialog stores reasoning effort in a cookie; the spawner
        # forwards it here so the detached worker doesn't have to reach back
        # into the parent process or any shared store.
        parser.add_argument("--reasoning-effort", type=str, default=None)
        parser.add_argument("--model", type=str, default=None)
        parser.add_argument("--sandbox-policy", type=str, default=None)
        parser.add_argument("--approval-mode", type=str, default=None)
        parser.add_argument("--web-search-mode", type=str, default=None)
        parser.add_argument("--enable-memories", action="store_true")
        parser.add_argument("--collaboration-mode", type=str, default=None)
        parser.add_argument("--plan-mode", action="store_true")

    @override
    def handle(self, *args: Any, **options: Any) -> None:
        instance_id: int = options["instance_id"]
        reasoning_effort: str | None = options.get("reasoning_effort")
        model: str | None = options.get("model")
        sandbox_policy: str | None = options.get("sandbox_policy")
        approval_mode: str | None = options.get("approval_mode")
        web_search_mode: str | None = options.get("web_search_mode")
        enable_memories: bool = options.get("enable_memories", False)
        collaboration_mode: str | None = options.get("collaboration_mode")
        plan_mode: bool = options.get("plan_mode", False)
        instance = CodexInstance.objects.get(pk=instance_id)
        # Lease an exclusive Codex sqlite_home so this worker does not share a
        # SQLite writer lock with the web pool or any sibling worker (the lease
        # is released, and the OS drops the flock, when this process exits).
        # Skipped under tests, where the app-server is mocked and a real home
        # would only create stray state directories.
        sqlite_lease: WorkerSqliteHome | None = None
        sqlite_home: str | None = None
        if not getattr(settings, "TESTING", False):
            try:
                sqlite_lease = acquire_worker_sqlite_home(instance_id)
                sqlite_home = str(sqlite_lease.home)
            except OSError:
                # A broken state-dir filesystem should degrade to Codex's default
                # $CODEX_HOME, not the web home (which lives under the same base
                # that just failed). Pass it explicitly so it overrides any
                # CODEX_SQLITE_HOME the deployment exported.
                _worker_log(instance_id, "failed to lease sqlite_home; using CODEX_HOME")
                sqlite_home = str(codex_home_dir())

        # Install the signal handlers before flipping to RUNNING so a Stop or
        # Steer request that lands the instant we transition can still be
        # observed by the stream loop.
        signal.signal(signal.SIGTERM, _on_sigterm)
        signal.signal(signal.SIGUSR1, _on_sigusr1)

        _apply_worker_oom_score_adjust()
        # Conditional on the row still being active: an unconditional save
        # here would resurrect a row that reconcile_dead already marked FAILED
        # (slow systemd start past the pid-assignment grace) -- the failure
        # has been routed to system agents by then, so running the turn anyway
        # double-drives the workflow.
        claimed = CodexInstance.objects.filter(
            pk=instance.pk,
            status__in=CodexInstance.ACTIVE_STATUSES,
        ).update(pid=os.getpid(), status=CodexInstance.STATUS_RUNNING)
        if not claimed:
            _worker_log(
                instance_id,
                "row was already terminal before the turn started; exiting",
            )
            return
        instance.pid = os.getpid()
        instance.status = CodexInstance.STATUS_RUNNING
        _worker_log(
            instance_id,
            f"started thread_id={instance.thread_id} events_path={instance.events_path}",
        )

        try:
            with open(instance.events_path, "a", buffering=1, encoding="utf-8") as events_file:
                final_turn = _run_turn(
                    instance=instance,
                    prompt=instance.prompt,
                    events_file=events_file,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    sandbox_policy=sandbox_policy,
                    approval_mode=approval_mode,
                    web_search_mode=(
                        web_search_mode
                        if web_search_mode is not None
                        else instance.web_search_mode or None
                    ),
                    enable_memories=enable_memories or instance.enable_memories,
                    collaboration_mode=collaboration_mode,
                    plan_mode=plan_mode,
                    output_schema=instance.output_schema,
                    sqlite_home=sqlite_home,
                )
        except BaseException as exc:  # noqa: BLE001 - record any failure, then re-raise
            _worker_log(instance_id, f"failed with {type(exc).__name__}: {exc!r}")
            _worker_log_exception(exc)
            instance.status = CodexInstance.STATUS_FAILED
            instance.ended_at = timezone.now()
            instance.error = repr(exc)
            _commit_terminal_status(instance)
            if instance.status == CodexInstance.STATUS_FAILED:
                resolve_dangling_requests_for_instance(instance.pk)
            _notify_system_agents(instance)
            cleanup_requested_input_images_for(instance)
            disk_cleanup.run_finished_session_disk_cleanup()
            raise
        finally:
            # Worker turns write thread metadata to an isolated home the web
            # index never reads, so bump the session's recency directly to keep
            # the list ordered by real activity. Best-effort: a failed bump must
            # not fail an already-finished turn.
            _record_session_activity(instance)
            # _run_turn has closed the app-server (and its log-DB handle) by the
            # time it returns or raises, so pruning only unlinks a released file;
            # releasing then frees the leased slot (or removes an overflow home).
            if sqlite_lease is not None:
                with contextlib.suppress(Exception):
                    prune_worker_logs_db(sqlite_lease.home)
                sqlite_lease.release()

        instance.ended_at = timezone.now()
        instance.codex_error_info = None
        # The TurnCompletedNotification carries the actual outcome — including
        # ``interrupted`` and ``failed`` terminal states the SDK does not
        # raise on — so map it through rather than blanket-marking completed.
        if final_turn is None:
            instance.status = CodexInstance.STATUS_FAILED
            instance.error = "stream ended without a turn/completed notification"
        elif final_turn.status == TurnStatus.completed:
            instance.status = CodexInstance.STATUS_COMPLETED
        else:
            instance.status = CodexInstance.STATUS_FAILED
            error = final_turn.error
            instance.codex_error_info = _serialized_codex_error_info(error)
            instance.error = (
                error.message
                if error is not None and error.message
                else f"turn ended with status {final_turn.status.value}"
            )
        _worker_log(
            instance_id,
            f"finished status={instance.status} error={instance.error!r}",
        )
        _commit_terminal_status(instance)
        if instance.status == CodexInstance.STATUS_FAILED:
            resolve_dangling_requests_for_instance(instance.pk)
        _notify_system_agents(instance)
        cleanup_requested_input_images_for(instance)
        disk_cleanup.run_finished_session_disk_cleanup()


def _commit_terminal_status(instance: CodexInstance) -> None:
    """Persist the worker's terminal status without clobbering a parent write.

    The parent (a forced Stop in ``_force_kill_instance``/``_mark_failed`` or a
    ``reconcile_dead`` sweep) flips a row to a terminal state with a conditional
    ``UPDATE ... WHERE status IN (starting, running)`` so an already-terminal row
    is preserved. The worker's own end-of-turn save must use the same guard: an
    unconditional save that lands just after the parent forced the row to FAILED
    would resurrect it as COMPLETED, silently dropping the user's Stop and
    leaving the cancelled approval/input prompts inconsistent with the status.
    If the guarded update matches no row, the parent already wrote a terminal
    state; adopt it rather than overwriting it.
    """
    updated = CodexInstance.objects.filter(
        pk=instance.pk,
        status__in=CodexInstance.ACTIVE_STATUSES,
    ).update(
        status=instance.status,
        ended_at=instance.ended_at,
        error=instance.error,
        codex_error_info=instance.codex_error_info,
    )
    if updated == 0:
        instance.refresh_from_db()


def _serialized_codex_error_info(error: TurnError | None) -> Any:
    if error is None or error.codex_error_info is None:
        return None
    return error.codex_error_info.model_dump(mode="json", by_alias=True)


def _notify_system_agents(instance: CodexInstance) -> None:
    system_agents_handled = False
    try:
        from hitch.main.workflows import system_agents

        system_agents_handled = system_agents.on_codex_instance_finished(instance)
    except Exception:
        logger.exception("failed to route completed worker %s to system agents", instance.pk)
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
        logger.exception("failed to route completed worker %s to demo workflow", instance.pk)


def _apply_worker_oom_score_adjust(
    path: Path = Path("/proc/self/oom_score_adj"),
) -> None:
    """Prefer this worker and its descendants during global OOM selection."""
    raw_score = getattr(settings, "CODEX_WORKER_OOM_SCORE_ADJ", 0)
    try:
        score = int(raw_score)
    except (TypeError, ValueError):
        logger.warning("invalid CODEX_WORKER_OOM_SCORE_ADJ: %r", raw_score)
        return
    if score == 0:
        return
    score = max(-1000, min(1000, score))
    try:
        path.write_text(f"{score}\n", encoding="utf-8")
    except FileNotFoundError:
        return
    except OSError:
        logger.exception("failed to set worker oom_score_adj")


def _run_turn(
    *,
    instance: CodexInstance,
    prompt: str,
    events_file: IO[str],
    model: str | None = None,
    reasoning_effort: str | None = None,
    sandbox_policy: str | None = None,
    approval_mode: str | None = None,
    web_search_mode: str | None = None,
    enable_memories: bool = False,
    collaboration_mode: str | None = None,
    plan_mode: bool = False,
    output_schema: dict[str, Any] | None = None,
    sqlite_home: str | None = None,
) -> Turn | None:
    os.environ["HITCH_THREAD_ID"] = instance.thread_id
    os.environ["HITCH_CWD"] = instance.cwd
    project_dir = Path(settings.BASE_DIR)
    manage_py = Path(settings.BASE_DIR) / "manage.py"
    os.environ["HITCH_PROJECT_DIR"] = str(project_dir)
    os.environ["HITCH_MANAGE_PY"] = str(manage_py)
    os.environ["HITCH_MANAGE_COMMAND"] = "uv"
    os.environ["HITCH_PROPOSE_SESSION_COMMAND"] = "uv"
    config = app_server_config(
        enable_memories=enable_memories,
        web_search_mode=web_search_mode,
        sqlite_home=sqlite_home,
    )
    raw_effort = reasoning_effort.strip() if reasoning_effort else None
    effort: ReasoningEffort | None = None
    if raw_effort:
        with contextlib.suppress(ValueError):
            effort = ReasoningEffort(raw_effort)
    policy = _build_sandbox_policy(sandbox_policy)
    # Serialise writes between the SDK reader thread (which calls the
    # approval handler) and the main thread (which appends streamed turn
    # events). Without this lock both threads can interleave partial JSON
    # lines into the events file.
    write_lock = threading.Lock()

    def _write_event(
        method: str,
        payload: Any,
        *,
        recorded_at: int | None = None,
        event_seq: int | None = None,
    ) -> None:
        line = (
            _serialize_event(
                method,
                payload,
                recorded_at=recorded_at,
                event_seq=event_seq,
            )
            + "\n"
        )
        with write_lock:
            events_file.write(line)

    def _write_notification(event: Notification) -> None:
        recorded_at, event_seq = notification_order(event)
        _write_event(
            event.method,
            event.payload,
            recorded_at=recorded_at,
            event_seq=event_seq,
        )

    def _discard_notification(event: Notification) -> None:
        notification_order(event)

    final_turn: Turn | None = None
    interrupt_forwarder: _InterruptForwarder | None = None
    goal_forwarder: threading.Thread | None = None
    steer_forwarder: _SteerControlForwarder | None = None
    notification_order: NotificationOrdering = _fallback_notification_order
    control_path = control_path_for(instance)
    resume_kwargs: dict[str, Any] = {}
    if instance.developer_instructions:
        resume_kwargs["developer_instructions"] = instance.developer_instructions

    def _configure(codex: Codex) -> None:
        # Runs once per app-server open attempt (``open_codex_resumed`` retries
        # the whole open+configure+resume when the resume races the CODEX_HOME
        # state-DB migration). All three steps are safe to redo against a fresh
        # server: the first two only mutate the client, and a goal forwarder
        # left over from a discarded attempt exits cleanly once its transport
        # closes.
        nonlocal notification_order, goal_forwarder
        notification_order = _install_notification_sequencer(codex)
        # The Codex top-level class instantiates its own AppServerClient
        # without an ``approval_handler`` argument, so the only way to wire
        # an interactive callback is to swap the bound method on the client
        # after construction. The SDK's default handler unconditionally
        # rubber-stamps, so we always install our own — either
        # the interactive one that opens an ``ApprovalRequest`` row, or
        # (only under ``approve_all``) a rubber-stamp that uses the
        # current ``accept`` wire value.
        codex._client._approval_handler = _make_approval_handler(
            instance=instance,
            write_event=_write_event,
            approval_mode=approval_mode,
        )
        goal_forwarder = _start_goal_event_forwarder(
            codex._client,
            thread_id=instance.thread_id,
            write_notification=_write_notification,
            discard_notification=_discard_notification,
        )

    try:
        with open_codex_resumed(
            lambda: Codex(config=config),
            thread_id=instance.thread_id,
            resume_kwargs=resume_kwargs,
            configure=_configure,
        ) as (codex, thread):
            turn = _start_turn(
                codex,
                thread,
                prompt=prompt,
                input_image_paths=_instance_input_image_paths(instance),
                model=model,
                effort=effort,
                raw_effort=raw_effort,
                sandbox_policy=policy,
                approval_mode=approval_mode,
                collaboration_mode=collaboration_mode,
                plan_mode=plan_mode,
                output_schema=output_schema,
            )
            interrupt_forwarder = _start_interrupt_forwarder(turn)
            steer_forwarder = _start_steer_control_forwarder(
                turn,
                instance=instance,
                control_path=control_path,
            )
            try:
                # A Stop click that landed before the turn handle existed sets the
                # flag without us being able to call interrupt yet; act on it now
                # that the handle is ready.
                _forward_interrupt_if_requested(
                    turn,
                    sent=interrupt_forwarder.sent,
                    send_lock=interrupt_forwarder.send_lock,
                )
                for event in turn.stream():
                    _write_notification(event)
                    payload = event.payload
                    if (
                        isinstance(payload, TurnCompletedNotification)
                        and payload.turn.id == turn.id
                    ):
                        final_turn = payload.turn
                    if _cancel_requested:
                        # SDK-level interrupt is the graceful cancellation path:
                        # the app-server stops the model, emits the remaining
                        # events (including a turn/completed with status=interrupted),
                        # and the worker's normal status-update code at the end
                        # records that as a failed turn. SIGKILL is the next
                        # escalation if the user clicks Stop again.
                        _forward_interrupt_if_requested(
                            turn,
                            sent=interrupt_forwarder.sent,
                            send_lock=interrupt_forwarder.send_lock,
                        )
            finally:
                if steer_forwarder is not None:
                    _stop_steer_control_forwarder(steer_forwarder)
                    steer_forwarder = None
                if interrupt_forwarder is not None:
                    _stop_interrupt_forwarder(interrupt_forwarder)
                    interrupt_forwarder = None
    finally:
        if steer_forwarder is not None:
            _stop_steer_control_forwarder(steer_forwarder)
        if interrupt_forwarder is not None:
            _stop_interrupt_forwarder(interrupt_forwarder)
        if goal_forwarder is not None:
            goal_forwarder.join(timeout=0.5)
    return final_turn


def _try_interrupt(turn: TurnHandle) -> bool:
    """Send a single SDK-level interrupt; report whether it was sent.

    Returning True even on SDK errors prevents a re-attempt loop when
    the turn has already ended (or the app-server rejected the call) —
    a hard-kill via SIGKILL is the user's next lever, not retries.
    """
    with contextlib.suppress(Exception):
        turn.interrupt()
    return True


@dataclasses.dataclass(slots=True)
class _InterruptForwarder:
    thread: threading.Thread
    wakeup: threading.Event
    stop: threading.Event
    sent: threading.Event
    send_lock: threading.Lock


def _start_interrupt_forwarder(turn: TurnHandle) -> _InterruptForwarder:
    """Forward SIGTERM cancellation even while the turn stream is silent."""
    global _interrupt_wakeup
    wakeup = threading.Event()
    stop = threading.Event()
    sent = threading.Event()
    send_lock = threading.Lock()
    forwarder = _InterruptForwarder(
        thread=threading.Thread(
            target=_forward_interrupt_requests,
            kwargs={
                "turn": turn,
                "wakeup": wakeup,
                "stop": stop,
                "sent": sent,
                "send_lock": send_lock,
            },
            daemon=True,
        ),
        wakeup=wakeup,
        stop=stop,
        sent=sent,
        send_lock=send_lock,
    )
    _interrupt_wakeup = wakeup
    forwarder.thread.start()
    return forwarder


def _stop_interrupt_forwarder(forwarder: _InterruptForwarder) -> None:
    global _interrupt_wakeup
    forwarder.stop.set()
    forwarder.wakeup.set()
    forwarder.thread.join(timeout=0.5)
    if _interrupt_wakeup is forwarder.wakeup:
        _interrupt_wakeup = None


def _forward_interrupt_requests(
    *,
    turn: TurnHandle,
    wakeup: threading.Event,
    stop: threading.Event,
    sent: threading.Event,
    send_lock: threading.Lock,
) -> None:
    while not stop.is_set():
        if _forward_interrupt_if_requested(
            turn,
            sent=sent,
            send_lock=send_lock,
        ):
            return
        wakeup.wait()
        wakeup.clear()


def _forward_interrupt_if_requested(
    turn: TurnHandle,
    *,
    sent: threading.Event,
    send_lock: threading.Lock,
) -> bool:
    """Send at most one interrupt across the stream and watcher threads."""
    if not _cancel_requested or sent.is_set():
        return False
    with send_lock:
        if sent.is_set():
            return False
        # Claim the send before the blocking SDK call. If that call wedges, a
        # second Stop still escalates to SIGKILL instead of piling up callers.
        sent.set()
    _try_interrupt(turn)
    return True


@dataclasses.dataclass(slots=True)
class _SteerControlForwarder:
    thread: threading.Thread
    wakeup: threading.Event
    stop: threading.Event


def _start_steer_control_forwarder(
    turn: TurnHandle,
    *,
    instance: CodexInstance,
    control_path: Path,
) -> _SteerControlForwarder:
    """Start a side drain for per-turn steer payloads."""
    global _steer_wakeup
    wakeup = threading.Event()
    stop = threading.Event()
    forwarder = _SteerControlForwarder(
        thread=threading.Thread(
            target=_forward_steer_requests,
            kwargs={
                "turn": turn,
                "instance": instance,
                "control_path": control_path,
                "wakeup": wakeup,
                "stop": stop,
            },
            daemon=True,
        ),
        wakeup=wakeup,
        stop=stop,
    )
    _steer_wakeup = wakeup
    forwarder.thread.start()
    return forwarder


def _stop_steer_control_forwarder(forwarder: _SteerControlForwarder) -> None:
    """Stop the steer forwarder after waking it for one final drain."""
    global _steer_wakeup
    forwarder.stop.set()
    forwarder.wakeup.set()
    forwarder.thread.join(timeout=0.5)
    if _steer_wakeup is forwarder.wakeup:
        _steer_wakeup = None


def _forward_steer_requests(
    *,
    turn: TurnHandle,
    instance: CodexInstance,
    control_path: Path,
    wakeup: threading.Event,
    stop: threading.Event,
) -> None:
    """Forward complete JSONL steer requests into the active turn.

    The initial drain catches requests queued while the worker was still
    STARTING. The final drain catches requests appended after the last stream
    event but before the worker records a terminal row status.
    """
    control_offset = 0
    while not stop.is_set():
        control_offset = _drain_steer_requests(
            turn,
            instance=instance,
            control_path=control_path,
            control_offset=control_offset,
        )
        wakeup.wait(_STEER_CONTROL_POLL_INTERVAL)
        wakeup.clear()
    _drain_steer_requests(
        turn,
        instance=instance,
        control_path=control_path,
        control_offset=control_offset,
    )


def _drain_steer_requests(
    turn: TurnHandle,
    *,
    instance: CodexInstance,
    control_path: Path,
    control_offset: int,
) -> int:
    """Read new complete control-file lines and apply steer requests.

    The control file is append-only JSONL. Incomplete trailing bytes are left
    for the next drain so a concurrent writer cannot produce a corrupt request.
    """
    try:
        with control_path.open("rb") as fh:
            fh.seek(control_offset)
            chunk = fh.read()
    except FileNotFoundError:
        return control_offset
    if not chunk:
        return control_offset
    newline = chunk.rfind(b"\n")
    if newline < 0:
        return control_offset
    complete = chunk[: newline + 1]
    # Split only on "\n" to match the byte-offset accounting above; bytes.splitlines()
    # would also break on \r, \v, \f, \x1c-\x1e, \x85, ... and could tear a record.
    for raw in complete.split(b"\n"):
        if not raw:
            continue
        try:
            request = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(request, dict) or request.get("op") != "steer":
            continue
        text = request.get("input")
        input_image_paths = _control_input_image_paths(request.get("inputImagePaths"))
        if isinstance(text, str) and (text.strip() or input_image_paths):
            sent = _try_steer(turn, text, input_image_paths=input_image_paths)
            if not sent:
                discard_input_attachment_paths(instance, input_image_paths)
            _append_steer_ack(control_path, request.get("id"), delivered=sent)
    return control_offset + len(complete)


def _append_steer_ack(control_path: Path, steer_id: Any, *, delivered: bool) -> None:
    """Record a steer's delivery outcome for the requesting process.

    ``steer_instance`` waits on this ack: a failed delivery (the SDK refuses
    to steer a finished turn) tells it to fall back to spawning a follow-up
    turn instead of silently dropping the user's message.
    """
    if not isinstance(steer_id, str) or not steer_id:
        return
    line = (
        json.dumps(
            {"op": "steer_ack", "id": steer_id, "delivered": delivered},
            separators=(",", ":"),
        )
        + "\n"
    )
    with contextlib.suppress(OSError), control_path.open("ab") as fh:
        fh.write(line.encode("utf-8"))


def _instance_input_image_paths(instance: CodexInstance) -> list[str]:
    return _control_input_image_paths(instance.input_image_paths)


def _control_input_image_paths(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [path for path in value if isinstance(path, str) and path.strip()]


def _turn_input(prompt: str, input_image_paths: list[str] | None = None) -> Input:
    image_paths = _control_input_image_paths(input_image_paths)
    if not image_paths:
        return TextInput(prompt)
    input_items: list[Any] = []
    if prompt:
        input_items.append(TextInput(prompt))
    input_items.extend(LocalImageInput(path=path) for path in image_paths)
    return input_items


def _typed_turn_input(
    prompt: str, input_image_paths: list[str] | None = None
) -> list[UserInput]:
    input_items: list[UserInput] = []
    if prompt:
        input_items.append(UserInput(root=TextUserInput(type="text", text=prompt)))
    input_items.extend(
        UserInput(root=LocalImageUserInput(type="localImage", path=path))
        for path in _control_input_image_paths(input_image_paths)
    )
    return input_items


def _try_steer(
    turn: TurnHandle, text: str, input_image_paths: list[str] | None = None
) -> bool:
    """Send one SDK-level steer request; report whether it was attempted."""
    try:
        turn.steer(_turn_input(text, input_image_paths))
    except Exception:
        return False
    else:
        return True


def _start_turn(
    codex: Codex,
    thread: Thread,
    *,
    prompt: str,
    input_image_paths: list[str] | None,
    model: str | None,
    effort: ReasoningEffort | None,
    raw_effort: str | None,
    sandbox_policy: SandboxPolicy | None,
    approval_mode: str | None,
    collaboration_mode: str | None,
    plan_mode: bool,
    output_schema: dict[str, Any] | None,
) -> TurnHandle:
    """Start a turn under the requested approval policy.

    The custom user-reviewer modes are not ``ApprovalMode`` enum values, so
    we bypass ``Thread.turn(approval_mode=)`` and post the wire-level params
    directly: an on-request approval policy paired with the ``user`` reviewer
    routes every escalation back to the client transport. ``prompt_user``
    surfaces the normal browser prompt there; ``approve_all`` is answered by
    a special rubber-stamp handler. Leaving ``approvals_reviewer`` unset falls
    back to server-side routing (typically ``auto_review``), which can still
    decide without involving the browser. Known SDK values (``auto_review``,
    ``deny_all``) and the unset case go through the typed ``Thread.turn`` API.
    """
    if plan_mode:
        return _start_plan_turn(
            codex,
            thread,
            prompt=prompt,
            input_image_paths=input_image_paths,
            model=model,
            sandbox_policy=sandbox_policy,
            approval_mode=approval_mode,
            output_schema=output_schema,
        )
    if collaboration_mode == _DEFAULT_COLLABORATION_MODE:
        return _start_default_collaboration_turn(
            codex,
            thread,
            prompt=prompt,
            input_image_paths=input_image_paths,
            model=model,
            effort=effort,
            raw_effort=raw_effort,
            sandbox_policy=sandbox_policy,
            approval_mode=approval_mode,
            output_schema=output_schema,
        )
    if collaboration_mode:
        raise ValueError(f"unsupported collaboration mode: {collaboration_mode}")

    if raw_effort and effort is None:
        return _start_raw_turn(
            codex,
            thread,
            prompt=prompt,
            input_image_paths=input_image_paths,
            model=model,
            effort=raw_effort,
            sandbox_policy=sandbox_policy,
            approval_mode=approval_mode,
            output_schema=output_schema,
        )

    if approval_mode in _USER_REVIEWER_APPROVAL_MODES:
        typed_input = _typed_turn_input(prompt, input_image_paths)
        params = TurnStartParams(
            thread_id=thread.id,
            input=typed_input,
            approval_policy=AskForApproval(root=AskForApprovalValue.on_request),
            approvals_reviewer=ApprovalsReviewer.user,
            effort=effort,
            model=model,
            sandbox_policy=sandbox_policy,
            output_schema=output_schema,
        )
        # ``_client.turn_start`` requires the input again as its second
        # positional arg; the value in ``params.input`` is overwritten by
        # the normalized form of this argument in the JSON-RPC payload.
        wire_input = [item.model_dump(mode="json", by_alias=True) for item in typed_input]
        response = codex._client.turn_start(thread.id, wire_input, params=params)
        return TurnHandle(codex._client, thread.id, response.turn.id)

    turn_kwargs: dict[str, Any] = {}
    if effort is not None:
        turn_kwargs["effort"] = effort
    if model is not None:
        turn_kwargs["model"] = model
    if sandbox_policy is not None:
        turn_kwargs["sandbox_policy"] = sandbox_policy
    if output_schema is not None:
        turn_kwargs["output_schema"] = output_schema
    mode = _build_approval_mode(approval_mode)
    if mode is not None:
        turn_kwargs["approval_mode"] = mode
    return thread.turn(_turn_input(prompt, input_image_paths), **turn_kwargs)


def _start_raw_turn(
    codex: Codex,
    thread: Thread,
    *,
    prompt: str,
    input_image_paths: list[str] | None,
    model: str | None,
    effort: str,
    sandbox_policy: SandboxPolicy | None,
    approval_mode: str | None,
    output_schema: dict[str, Any] | None,
) -> TurnHandle:
    typed_input = _typed_turn_input(prompt, input_image_paths)
    wire_input = [item.model_dump(mode="json", by_alias=True) for item in typed_input]
    params: dict[str, Any] = {
        "threadId": thread.id,
        "input": wire_input,
        "effort": effort,
    }
    if model is not None:
        params["model"] = model
    if sandbox_policy is not None:
        params["sandboxPolicy"] = sandbox_policy.model_dump(mode="json", by_alias=True)
    if output_schema is not None:
        params["outputSchema"] = output_schema
    if approval_mode in _USER_REVIEWER_APPROVAL_MODES:
        params["approvalPolicy"] = AskForApprovalValue.on_request.value
        params["approvalsReviewer"] = ApprovalsReviewer.user.value
    else:
        mode = _build_approval_mode(approval_mode)
        if mode is not None:
            approval_policy, approvals_reviewer = _approval_mode_params(mode)
            params["approvalPolicy"] = approval_policy
            if approvals_reviewer is not None:
                params["approvalsReviewer"] = approvals_reviewer
    response = codex._client.turn_start(thread.id, wire_input, params=params)
    return TurnHandle(codex._client, thread.id, response.turn.id)


def _start_plan_turn(
    codex: Codex,
    thread: Thread,
    *,
    prompt: str,
    input_image_paths: list[str] | None,
    model: str | None,
    sandbox_policy: SandboxPolicy | None,
    approval_mode: str | None,
    output_schema: dict[str, Any] | None,
) -> TurnHandle:
    if not model:
        raise ValueError("plan mode requires a model")
    collaboration_mode = CollaborationMode(
        mode=ModeKind.plan,
        settings=CodexModeSettings(
            developer_instructions=None,
            model=model,
            reasoning_effort=_PLAN_MODE_REASONING_EFFORT,
        ),
    )
    return _start_collaboration_turn(
        codex,
        thread,
        prompt=prompt,
        input_image_paths=input_image_paths,
        collaboration_mode=collaboration_mode,
        sandbox_policy=sandbox_policy,
        approval_mode=approval_mode,
        output_schema=output_schema,
    )


def _start_default_collaboration_turn(
    codex: Codex,
    thread: Thread,
    *,
    prompt: str,
    input_image_paths: list[str] | None,
    model: str | None,
    effort: ReasoningEffort | None,
    raw_effort: str | None,
    sandbox_policy: SandboxPolicy | None,
    approval_mode: str | None,
    output_schema: dict[str, Any] | None,
) -> TurnHandle:
    if not model:
        raise ValueError("default collaboration mode requires a model")
    effort_value = raw_effort or (effort.value if effort is not None else None)
    collaboration_mode = {
        "mode": ModeKind.default.value,
        "settings": {
            "developer_instructions": _DEFAULT_COLLABORATION_INSTRUCTIONS,
            "model": model,
            "reasoning_effort": effort_value,
        },
    }
    return _start_collaboration_turn(
        codex,
        thread,
        prompt=prompt,
        input_image_paths=input_image_paths,
        collaboration_mode=collaboration_mode,
        sandbox_policy=sandbox_policy,
        approval_mode=approval_mode,
        output_schema=output_schema,
    )


def _start_collaboration_turn(
    codex: Codex,
    thread: Thread,
    *,
    prompt: str,
    input_image_paths: list[str] | None,
    collaboration_mode: CollaborationMode | dict[str, Any],
    sandbox_policy: SandboxPolicy | None,
    approval_mode: str | None,
    output_schema: dict[str, Any] | None,
) -> TurnHandle:
    typed_input = _typed_turn_input(prompt, input_image_paths)
    wire_input = [item.model_dump(mode="json", by_alias=True) for item in typed_input]
    collaboration_payload = (
        collaboration_mode
        if isinstance(collaboration_mode, dict)
        else collaboration_mode.model_dump(mode="json", by_alias=True)
    )
    params: dict[str, Any] = {
        "threadId": thread.id,
        "input": wire_input,
        "collaborationMode": collaboration_payload,
    }
    if sandbox_policy is not None:
        params["sandboxPolicy"] = sandbox_policy.model_dump(mode="json", by_alias=True)
    if output_schema is not None:
        params["outputSchema"] = output_schema
    if approval_mode in _USER_REVIEWER_APPROVAL_MODES:
        params["approvalPolicy"] = AskForApproval(
            root=AskForApprovalValue.on_request
        ).model_dump(mode="json", by_alias=True)
        params["approvalsReviewer"] = ApprovalsReviewer.user.value
    else:
        mode = _build_approval_mode(approval_mode)
        if mode is not None:
            approval_policy, approvals_reviewer = _approval_mode_params(mode)
            params["approvalPolicy"] = approval_policy
            if approvals_reviewer is not None:
                params["approvalsReviewer"] = approvals_reviewer
    response = codex._client.turn_start(thread.id, wire_input, params=params)
    return TurnHandle(codex._client, thread.id, response.turn.id)


def _approval_mode_params(
    mode: ApprovalMode,
) -> tuple[str, str | None]:
    if mode == ApprovalMode.auto_review:
        return AskForApprovalValue.on_request.value, ApprovalsReviewer.auto_review.value
    if mode == ApprovalMode.deny_all:
        return AskForApprovalValue.never.value, None
    raise AssertionError(f"Unhandled approval mode: {mode!r}")


def _build_sandbox_policy(value: str | None) -> SandboxPolicy | None:
    """Construct a SandboxPolicy from the CLI string, or None to skip.

    Unknown strings (stale cookie after an SDK upgrade, manual edit) return
    None so the turn runs under Codex's default policy rather than crashing.
    """
    if not value:
        return None
    builder = _SANDBOX_POLICY_BUILDERS.get(value)
    if builder is None:
        return None
    return builder()


def _build_approval_mode(value: str | None) -> ApprovalMode | None:
    """Map the CLI string onto ``ApprovalMode``, or None to leave it unset.

    Unknown strings (stale cookie after an SDK upgrade, manual edit) return
    None so the turn falls back to the SDK default rather than crashing the
    worker over a value that no longer maps to an enum member.
    """
    if not value:
        return None
    try:
        return ApprovalMode(value)
    except ValueError:
        return None


def _serialize_event(
    method: str,
    payload: Any,
    *,
    recorded_at: int | None = None,
    event_seq: int | None = None,
) -> str:
    if isinstance(payload, BaseModel):
        payload_data: Any = payload.model_dump(mode="json", by_alias=True)
    elif dataclasses.is_dataclass(payload) and not isinstance(payload, type):
        payload_data = dataclasses.asdict(payload)
    else:
        payload_data = payload
    payload_data = _redact_local_image_paths(payload_data)
    event: dict[str, Any] = {"method": method, "payload": payload_data}
    if recorded_at is not None:
        event["recordedAt"] = recorded_at
    if event_seq is not None:
        event["eventSeq"] = event_seq
    return json.dumps(event)


def _redact_local_image_paths(value: Any) -> Any:
    if isinstance(value, dict):
        is_local_image = value.get("type") == "localImage"
        return {
            key: (
                "[redacted]"
                if is_local_image and key == "path"
                else ["[redacted]" for _item in child]
                if key == "local_images" and isinstance(child, list)
                else _redact_local_image_paths(child)
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_redact_local_image_paths(child) for child in value]
    return value


WriteEvent = Callable[[str, Any], None]
WriteNotification = Callable[[Notification], None]
DiscardNotification = Callable[[Notification], None]
NotificationOrder = tuple[int, int]
NotificationOrdering = Callable[[Notification], NotificationOrder]


class _NotificationSource(Protocol):
    def next_notification(self) -> Notification: ...


def _fallback_notification_order(_event: Notification) -> NotificationOrder:
    return (time.time_ns() // 1_000, 0)


def _install_notification_sequencer(codex: Codex) -> NotificationOrdering:
    """Tag SDK notifications in reader-thread arrival order before routing.

    The SDK splits notifications into turn-specific and global queues. Hitch
    consumes those queues from separate threads, so write time cannot be used
    to recover the original arrival order later. Recording the timestamp and
    sequence at the router boundary preserves that order across the split.
    """
    lock = threading.Lock()
    counter = itertools.count(1)
    order_by_id: dict[int, NotificationOrder] = {}
    last_recorded_at = 0
    router = codex._client._router
    route_notification = router.route_notification

    def next_order() -> NotificationOrder:
        nonlocal last_recorded_at
        recorded_at = max(time.time_ns() // 1_000, last_recorded_at)
        last_recorded_at = recorded_at
        return (recorded_at, next(counter))

    def ordered_route(notification: Notification) -> None:
        with lock:
            order_by_id[id(notification)] = next_order()
        if _preserve_early_turn_completed(router, notification):
            return
        route_notification(notification)

    def notification_order(notification: Notification) -> NotificationOrder:
        with lock:
            order = order_by_id.pop(id(notification), None)
            if order is None:
                order = next_order()
        return order

    router.route_notification = ordered_route  # type: ignore[method-assign]
    return notification_order


def _preserve_early_turn_completed(router: Any, notification: Notification) -> bool:
    """Work around SDK router versions that drop early ``turn/completed``.

    Fast turns can finish after ``turn/start`` responds but before
    ``TurnHandle.stream()`` registers its queue. The pinned SDK preserves
    early in-turn notifications but discards ``turn/completed`` in that
    window, leaving the worker blocked forever waiting for a completion
    event that already arrived. Store it with the pending turn notifications
    so stream registration replays it normally.
    """
    if notification.method != "turn/completed":
        return False
    turn_id = router._notification_turn_id(notification)
    if turn_id is None:
        return False
    with router._lock:
        if router._turn_notifications.get(turn_id) is not None:
            return False
        router._pending_turn_notifications.setdefault(turn_id, deque()).append(notification)
    return True


def _start_goal_event_forwarder(
    source: _NotificationSource,
    *,
    thread_id: str,
    write_notification: WriteNotification,
    discard_notification: DiscardNotification,
) -> threading.Thread:
    """Forward global thread-goal notifications into the worker event log."""
    thread = threading.Thread(
        target=_forward_goal_notifications,
        kwargs={
            "source": source,
            "thread_id": thread_id,
            "write_notification": write_notification,
            "discard_notification": discard_notification,
        },
        daemon=True,
    )
    thread.start()
    return thread


def _forward_goal_notifications(
    *,
    source: _NotificationSource,
    thread_id: str,
    write_notification: WriteNotification,
    discard_notification: DiscardNotification,
) -> None:
    """Drain global notifications until the SDK transport closes.

    ``TurnHandle.stream()`` only yields notifications routed to the active
    turn. ``thread/goal/cleared`` has no turn id, and ``thread/goal/updated``
    may also be global, so those events need this side drain to reach the
    same SSE log as turn-scoped events.
    """
    while True:
        try:
            event = source.next_notification()
        except Exception:
            return
        if not isinstance(event, Notification):
            return
        if event.method not in GOAL_METHODS:
            discard_notification(event)
            continue
        if _notification_thread_id(event.payload) != thread_id:
            discard_notification(event)
            continue
        write_notification(event)


def _notification_thread_id(payload: Any) -> str | None:
    thread_id = getattr(payload, "thread_id", None)
    if isinstance(thread_id, str):
        return thread_id
    if isinstance(payload, dict):
        raw = payload.get("threadId") or payload.get("thread_id")
        if isinstance(raw, str):
            return raw
    return None


def _make_approval_handler(
    *,
    instance: CodexInstance,
    write_event: WriteEvent,
    approval_mode: str | None,
) -> Callable[[str, dict[str, Any] | None], dict[str, Any]]:
    """Return an approval-handler closure bound to a single CodexInstance.

    ``approve_all`` mode auto-answers command/file escalations with
    ``accept``. Any other mode creates an ``ApprovalRequest`` row, emits an
    ``approval/requested`` event so the SSE stream surfaces it, and blocks
    polling the row until the Django view records a decision via
    ``POST /approval/<id>/``.

    The approval mode can be changed from the session UI while this worker is
    already running, so command/file approval handling reads the current
    ``CodexInstance.approval_mode`` instead of relying solely on the mode
    captured at worker startup.

    The handler runs on the SDK's reader thread (the same thread that reads
    JSON-RPC frames off codex's stdout), so it must:

    * Be safe to call from a non-Django-request thread — Django's ORM is
      thread-safe but does not auto-cleanup connections; we close after each
      ORM call so the per-thread connection doesn't leak across the long
      polling wait.
    * Block synchronously: the SDK is waiting on the return value to write
      the JSON-RPC response back to codex.

    Methods outside ``_APPROVAL_METHODS`` fall through to an empty object,
    matching the SDK's previous default-handler behaviour for unknown
    server-to-client requests.
    """

    def _handler(
        method: str, params: dict[str, Any] | None
    ) -> dict[str, Any]:
        if is_dynamic_tool_call(method):
            return handle_dynamic_tool_call(
                params,
                ToolContext(cwd=instance.cwd, thread_id=instance.thread_id),
            )
        if _is_user_input_request_method(method):
            return _handle_user_input_request(
                instance=instance,
                write_event=write_event,
                method=method,
                params=params or {},
            )
        if method not in _APPROVAL_METHODS:
            return {}
        current_approval_mode = _current_approval_mode(
            instance=instance,
            fallback=approval_mode,
        )
        live_decision = _approval_decision_for_mode(current_approval_mode)
        if live_decision is not None:
            return {"decision": live_decision}
        request_id = _create_pending_approval(
            instance_id=instance.pk,
            method=method,
            params=params or {},
        )
        live_decision = _approval_decision_for_mode(
            _current_approval_mode(instance=instance, fallback=approval_mode)
        )
        if live_decision is not None:
            return {"decision": _record_live_approval_decision(request_id, live_decision)}
        # Surface the pending approval through the events file so the SSE
        # stream pushes it to the browser without a separate transport.
        # The ``id`` we emit is the row pk the POST endpoint expects.
        write_event(
            "approval/requested",
            {
                "id": request_id,
                "method": method,
                "params": params or {},
            },
        )
        decision = _wait_for_decision(request_id)
        write_event(
            "approval/resolved",
            {"id": request_id, "method": method, "decision": decision},
        )
        return {"decision": decision}

    return _handler


def _approval_decision_for_mode(approval_mode: str | None) -> str | None:
    if approval_mode == _APPROVE_ALL:
        return ApprovalRequest.DECISION_ACCEPT
    if approval_mode == _DENY_ALL:
        return ApprovalRequest.DECISION_DECLINE
    return None


def _record_live_approval_decision(
    request_id: int, decision: str
) -> str | dict[str, Any]:
    from django.db import connection
    from django.utils import timezone as tz

    try:
        updated = ApprovalRequest.objects.filter(pk=request_id, decision="").update(
            decision=decision,
            decided_at=tz.now(),
        )
        if updated:
            return decision
        stored_decision, payload = ApprovalRequest.objects.values_list(
            "decision", "decision_payload"
        ).get(pk=request_id)
        return _stored_approval_decision(stored_decision, payload)
    except ApprovalRequest.DoesNotExist:
        return decision
    finally:
        connection.close()


def _current_approval_mode(
    *, instance: CodexInstance, fallback: str | None
) -> str | None:
    """Return the live approval mode for ``instance``.

    Called from the SDK reader thread, so release the per-thread Django
    connection immediately after the short query.
    """
    from django.db import connection

    try:
        value = (
            CodexInstance.objects.values_list("approval_mode", flat=True)
            .filter(pk=instance.pk)
            .first()
        )
        return value or fallback
    finally:
        connection.close()


def _is_user_input_request_method(method: str) -> bool:
    return (
        method in _USER_INPUT_METHODS
        or method.endswith(("/requestUserInput", "/request_user_input"))
    )


def _handle_user_input_request(
    *,
    instance: CodexInstance,
    write_event: WriteEvent,
    method: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    request_id = _create_pending_user_input(
        instance_id=instance.pk,
        method=method,
        params=params,
    )
    write_event(
        "input/requested",
        {
            "id": request_id,
            "method": method,
            "params": params,
        },
    )
    response = _wait_for_user_input_response(request_id)
    write_event(
        "input/resolved",
        {"id": request_id, "method": method, "response": response},
    )
    return response


def _create_pending_approval(
    *, instance_id: int, method: str, params: dict[str, Any]
) -> int:
    """Insert a pending ApprovalRequest row and return its primary key.

    Wrapped so the reader thread releases its database connection right
    after the insert — the worker holds approvals open for up to
    ``_APPROVAL_WAIT_SECONDS`` and we don't want the per-thread Django
    connection to stay checked out for that whole window.
    """
    from django.db import connection

    try:
        approval = ApprovalRequest.objects.create(
            instance_id=instance_id, method=method, params=params
        )
        return approval.pk
    finally:
        connection.close()


def _create_pending_user_input(
    *, instance_id: int, method: str, params: dict[str, Any]
) -> int:
    from django.db import connection

    try:
        input_request = UserInputRequest.objects.create(
            instance_id=instance_id,
            method=method,
            params=params,
        )
        return input_request.pk
    finally:
        connection.close()


def _wait_for_user_input_response(request_id: int) -> dict[str, Any]:
    deadline = time.monotonic() + _APPROVAL_WAIT_SECONDS
    # Stop on cancellation too: while blocked here the main stream loop can't act
    # on a SIGTERM, so a Stop click would otherwise hang until SIGKILL. Falling
    # through records the empty-answer fallback (the conditional UPDATE preserves
    # a real answer submitted at the boundary) and lets the main loop interrupt.
    while time.monotonic() < deadline and not _cancel_requested:
        response = _user_input_response_value(request_id)
        if isinstance(response, dict):
            return response
        time.sleep(_APPROVAL_POLL_INTERVAL)

    return _record_default_user_input_response(request_id, {"answers": {}})


def _user_input_response_value(request_id: int) -> Any:
    from django.db import connection

    try:
        return UserInputRequest.objects.values_list("response", flat=True).get(
            pk=request_id
        )
    except UserInputRequest.DoesNotExist:
        return {"answers": {}}
    finally:
        connection.close()


def _record_default_user_input_response(
    request_id: int, response: dict[str, Any]
) -> dict[str, Any]:
    """Record the no-answer fallback and return the response codex must see.

    The conditional ``response__isnull=True`` UPDATE serialises against a
    user who submits at the deadline boundary: if it matches zero rows the
    row already carries the user's real answer, so round-trip that back to
    codex instead of clobbering it with the empty fallback. Mirrors the
    ``decision=""`` guard in ``_wait_for_decision``.
    """
    from django.db import connection

    try:
        updated = UserInputRequest.objects.filter(
            pk=request_id, response__isnull=True
        ).update(
            response=response,
            responded_at=timezone.now(),
        )
        if updated:
            return response
        stored = (
            UserInputRequest.objects.values_list("response", flat=True)
            .filter(pk=request_id)
            .first()
        )
        return stored if isinstance(stored, dict) else response
    finally:
        connection.close()


def _stored_approval_decision(decision: str, payload: Any) -> str | dict[str, Any]:
    if isinstance(payload, dict):
        return cast(dict[str, Any], payload)
    return ApprovalRequest.normalize_decision(decision)


def _wait_for_decision(request_id: int) -> str | dict[str, Any]:
    """Poll the row for a recorded decision; default to ``decline`` on timeout.

    Polling (rather than a Postgres ``LISTEN``/``NOTIFY``-style wakeup)
    matches the rest of the project: the streaming layer also short-polls
    the events file, and the parent process is sqlite. The timeout exists
    so a worker doesn't sit forever if the user closed the browser tab —
    ``decline`` is the safest no-answer fallback (refuse the action, but
    let the agent keep running so a follow-up turn can recover).

    On the timeout path the conditional ``decision=""`` UPDATE is what
    serialises against a user who clicks at the deadline boundary: if
    that UPDATE matches zero rows, the row already carries a real
    decision and we round-trip it back to codex instead of clobbering
    the user's pick with ``decline``.
    """
    from django.db import connection
    from django.utils import timezone as tz

    deadline = time.monotonic() + _APPROVAL_WAIT_SECONDS
    while True:
        try:
            decision, payload = ApprovalRequest.objects.values_list(
                "decision", "decision_payload"
            ).get(pk=request_id)
        except ApprovalRequest.DoesNotExist:
            # The row was deleted out from under the wait (e.g. an admin or
            # cascade delete). This handler runs on the SDK reader thread, so
            # letting the exception escape would tear down the entire turn;
            # decline just this action instead, like the timeout path.
            return ApprovalRequest.DECISION_DECLINE
        finally:
            connection.close()
        if decision:
            return _stored_approval_decision(decision, payload)
        # A Stop click (SIGTERM) can't be acted on by the main stream loop while
        # it is blocked here awaiting the JSON-RPC reply, so honour cancellation
        # in this wait: decline the pending action and return so codex unblocks
        # and the main loop can issue turn.interrupt(). Without this the first
        # Stop click is a silent no-op until a second click escalates to SIGKILL.
        if _cancel_requested or time.monotonic() >= deadline:
            try:
                updated = ApprovalRequest.objects.filter(
                    pk=request_id, decision=""
                ).update(
                    decision=ApprovalRequest.DECISION_DECLINE,
                    decided_at=tz.now(),
                )
                if updated:
                    return ApprovalRequest.DECISION_DECLINE
                # Zero rows matched → the user wrote a real decision in
                # the window between the last read and this UPDATE.
                # Honour it rather than overwriting with ``decline``.
                final_decision, final_payload = ApprovalRequest.objects.values_list(
                    "decision", "decision_payload"
                ).get(pk=request_id)
                return _stored_approval_decision(final_decision, final_payload)
            finally:
                connection.close()
        time.sleep(_APPROVAL_POLL_INTERVAL)
