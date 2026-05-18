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

The events file plus the status transitions on the row are the only output —
stdout/stderr are redirected to /dev/null by the parent.

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
import shutil
import signal
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any, Protocol, override

from django.core.management.base import BaseCommand, CommandParser
from django.utils import timezone
from openai_codex import (
    ApprovalMode,
    AppServerConfig,
    Codex,
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
    ModeKind,
    ReadOnlySandboxPolicy,
    ReasoningEffort,
    SandboxPolicy,
    TextUserInput,
    Turn,
    TurnCompletedNotification,
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

from hitch.main.codex_events import GOAL_METHODS
from hitch.main.codex_pool import control_path_for
from hitch.main.models import ApprovalRequest, CodexInstance, UserInputRequest

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
_USER_REVIEWER_APPROVAL_MODES = frozenset({_PROMPT_USER, _APPROVE_ALL})
_PLAN_MODE_REASONING_EFFORT = ReasoningEffort.medium
_DEFAULT_COLLABORATION_MODE = "default"

# Set by the SIGTERM handler so the stream loop knows to call
# ``turn.interrupt()`` between events. Plain module-level bool is fine —
# CPython makes single-attribute reads/writes atomic, and the signal
# handler is intentionally minimal (it must avoid blocking JSON-RPC
# calls that would race the main loop's read on the response pipe).
_cancel_requested = False

# Set by the active turn so the SIGUSR1 handler can wake the control-file
# forwarder without doing blocking JSON-RPC work from inside the handler.
_steer_wakeup: threading.Event | None = None


def _on_sigterm(_signum: int, _frame: Any) -> None:
    """Mark the active turn for graceful cancellation.

    Defers the actual SDK ``turn.interrupt()`` call to the main loop so
    we don't issue a blocking JSON-RPC request from inside a signal
    handler — that would contend with the loop's read on the same
    response pipe and could deadlock. The Django stop endpoint sends
    SIGTERM here; a follow-up click escalates to SIGKILL, which has
    no Python-level handler and tears the worker down immediately.
    """
    global _cancel_requested
    _cancel_requested = True


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
        parser.add_argument("--collaboration-mode", type=str, default=None)
        parser.add_argument("--plan-mode", action="store_true")

    @override
    def handle(self, *args: Any, **options: Any) -> None:
        instance_id: int = options["instance_id"]
        reasoning_effort: str | None = options.get("reasoning_effort")
        model: str | None = options.get("model")
        sandbox_policy: str | None = options.get("sandbox_policy")
        approval_mode: str | None = options.get("approval_mode")
        collaboration_mode: str | None = options.get("collaboration_mode")
        plan_mode: bool = options.get("plan_mode", False)
        instance = CodexInstance.objects.get(pk=instance_id)

        # Install the signal handlers before flipping to RUNNING so a Stop or
        # Steer request that lands the instant we transition can still be
        # observed by the stream loop.
        signal.signal(signal.SIGTERM, _on_sigterm)
        signal.signal(signal.SIGUSR1, _on_sigusr1)

        instance.status = CodexInstance.STATUS_RUNNING
        instance.save(update_fields=["status"])

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
                    collaboration_mode=collaboration_mode,
                    plan_mode=plan_mode,
                )
        except Exception as exc:  # noqa: BLE001 - record any failure, then re-raise
            instance.status = CodexInstance.STATUS_FAILED
            instance.ended_at = timezone.now()
            instance.error = repr(exc)
            instance.save(update_fields=["status", "ended_at", "error"])
            raise

        instance.ended_at = timezone.now()
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
            instance.error = (
                error.message
                if error is not None and error.message
                else f"turn ended with status {final_turn.status.value}"
            )
        instance.save(update_fields=["status", "ended_at", "error"])


def _run_turn(
    *,
    instance: CodexInstance,
    prompt: str,
    events_file: IO[str],
    model: str | None = None,
    reasoning_effort: str | None = None,
    sandbox_policy: str | None = None,
    approval_mode: str | None = None,
    collaboration_mode: str | None = None,
    plan_mode: bool = False,
) -> Turn | None:
    config = AppServerConfig(codex_bin=shutil.which("codex"))
    effort: ReasoningEffort | None = None
    if reasoning_effort:
        # Unknown strings are ignored rather than crashing the worker — Codex
        # will fall back to the model's default effort in that case, which is
        # preferable to losing the whole turn over a stale enum value.
        with contextlib.suppress(ValueError):
            effort = ReasoningEffort(reasoning_effort)
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
    interrupt_sent = False
    goal_forwarder: threading.Thread | None = None
    steer_forwarder: _SteerControlForwarder | None = None
    notification_order: NotificationOrdering = _fallback_notification_order
    control_path = control_path_for(instance)
    try:
        with Codex(config=config) as codex:
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
            resume_kwargs: dict[str, Any] = {}
            if instance.developer_instructions:
                resume_kwargs["developer_instructions"] = instance.developer_instructions
            thread = codex.thread_resume(instance.thread_id, **resume_kwargs)
            turn = _start_turn(
                codex,
                thread,
                prompt=prompt,
                model=model,
                effort=effort,
                sandbox_policy=policy,
                approval_mode=approval_mode,
                collaboration_mode=collaboration_mode,
                plan_mode=plan_mode,
            )
            steer_forwarder = _start_steer_control_forwarder(
                turn,
                control_path=control_path,
            )
            try:
                # A Stop click that landed before the turn handle existed sets the
                # flag without us being able to call interrupt yet; act on it now
                # that the handle is ready.
                if _cancel_requested and not interrupt_sent:
                    interrupt_sent = _try_interrupt(turn)
                for event in turn.stream():
                    _write_notification(event)
                    payload = event.payload
                    if (
                        isinstance(payload, TurnCompletedNotification)
                        and payload.turn.id == turn.id
                    ):
                        final_turn = payload.turn
                    if _cancel_requested and not interrupt_sent:
                        # SDK-level interrupt is the graceful cancellation path:
                        # the app-server stops the model, emits the remaining
                        # events (including a turn/completed with status=interrupted),
                        # and the worker's normal status-update code at the end
                        # records that as a failed turn. SIGKILL is the next
                        # escalation if the user clicks Stop again.
                        interrupt_sent = _try_interrupt(turn)
            finally:
                if steer_forwarder is not None:
                    _stop_steer_control_forwarder(steer_forwarder)
                    steer_forwarder = None
    finally:
        if steer_forwarder is not None:
            _stop_steer_control_forwarder(steer_forwarder)
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
class _SteerControlForwarder:
    thread: threading.Thread
    wakeup: threading.Event
    stop: threading.Event


def _start_steer_control_forwarder(
    turn: TurnHandle,
    *,
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
            control_path=control_path,
            control_offset=control_offset,
        )
        wakeup.wait(_STEER_CONTROL_POLL_INTERVAL)
        wakeup.clear()
    _drain_steer_requests(
        turn,
        control_path=control_path,
        control_offset=control_offset,
    )


def _drain_steer_requests(
    turn: TurnHandle,
    *,
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
    for raw in complete.splitlines():
        try:
            request = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(request, dict) or request.get("op") != "steer":
            continue
        text = request.get("input")
        if isinstance(text, str) and text.strip():
            _try_steer(turn, text)
    return control_offset + len(complete)


def _try_steer(turn: TurnHandle, text: str) -> bool:
    """Send one SDK-level steer request; report whether it was attempted."""
    try:
        turn.steer(TextInput(text))
    except Exception:
        return False
    else:
        return True


def _start_turn(
    codex: Codex,
    thread: Thread,
    *,
    prompt: str,
    model: str | None,
    effort: ReasoningEffort | None,
    sandbox_policy: SandboxPolicy | None,
    approval_mode: str | None,
    collaboration_mode: str | None,
    plan_mode: bool,
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
            model=model,
            sandbox_policy=sandbox_policy,
            approval_mode=approval_mode,
        )
    if collaboration_mode == _DEFAULT_COLLABORATION_MODE:
        return _start_default_collaboration_turn(
            codex,
            thread,
            prompt=prompt,
            model=model,
            effort=effort,
            sandbox_policy=sandbox_policy,
            approval_mode=approval_mode,
        )
    if collaboration_mode:
        raise ValueError(f"unsupported collaboration mode: {collaboration_mode}")

    if approval_mode in _USER_REVIEWER_APPROVAL_MODES:
        typed_input = [UserInput(root=TextUserInput(type="text", text=prompt))]
        params = TurnStartParams(
            thread_id=thread.id,
            input=typed_input,
            approval_policy=AskForApproval(root=AskForApprovalValue.on_request),
            approvals_reviewer=ApprovalsReviewer.user,
            effort=effort,
            model=model,
            sandbox_policy=sandbox_policy,
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
    mode = _build_approval_mode(approval_mode)
    if mode is not None:
        turn_kwargs["approval_mode"] = mode
    return thread.turn(TextInput(prompt), **turn_kwargs)


def _start_plan_turn(
    codex: Codex,
    thread: Thread,
    *,
    prompt: str,
    model: str | None,
    sandbox_policy: SandboxPolicy | None,
    approval_mode: str | None,
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
        collaboration_mode=collaboration_mode,
        sandbox_policy=sandbox_policy,
        approval_mode=approval_mode,
    )


def _start_default_collaboration_turn(
    codex: Codex,
    thread: Thread,
    *,
    prompt: str,
    model: str | None,
    effort: ReasoningEffort | None,
    sandbox_policy: SandboxPolicy | None,
    approval_mode: str | None,
) -> TurnHandle:
    if not model:
        raise ValueError("default collaboration mode requires a model")
    collaboration_mode = CollaborationMode(
        mode=ModeKind.default,
        settings=CodexModeSettings(
            developer_instructions=None,
            model=model,
            reasoning_effort=effort,
        ),
    )
    return _start_collaboration_turn(
        codex,
        thread,
        prompt=prompt,
        collaboration_mode=collaboration_mode,
        sandbox_policy=sandbox_policy,
        approval_mode=approval_mode,
    )


def _start_collaboration_turn(
    codex: Codex,
    thread: Thread,
    *,
    prompt: str,
    collaboration_mode: CollaborationMode,
    sandbox_policy: SandboxPolicy | None,
    approval_mode: str | None,
) -> TurnHandle:
    typed_input = [UserInput(root=TextUserInput(type="text", text=prompt))]
    wire_input = [item.model_dump(mode="json", by_alias=True) for item in typed_input]
    params: dict[str, Any] = {
        "threadId": thread.id,
        "input": wire_input,
        "collaborationMode": collaboration_mode.model_dump(mode="json", by_alias=True),
    }
    if sandbox_policy is not None:
        params["sandboxPolicy"] = sandbox_policy.model_dump(mode="json", by_alias=True)
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
    event: dict[str, Any] = {"method": method, "payload": payload_data}
    if recorded_at is not None:
        event["recordedAt"] = recorded_at
    if event_seq is not None:
        event["eventSeq"] = event_seq
    return json.dumps(event)


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

    Two flavors:

    * ``approve_all`` mode: every escalation is auto-answered ``accept``.
      This preserves the existing "approve everything" promise of that
      mode without going through the interactive UI loop.
    * Any other mode (including ``auto_review`` and ``prompt_user``): each
      escalation creates an ``ApprovalRequest`` row, emits an
      ``approval/requested`` event so the SSE stream surfaces it, and blocks
      polling the row until the Django view records a decision via
      ``POST /approval/<id>/``.

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

    if approval_mode == _APPROVE_ALL:

        def _approve_all_handler(
            method: str, _params: dict[str, Any] | None
        ) -> dict[str, Any]:
            if _is_user_input_request_method(method):
                return _handle_user_input_request(
                    instance=instance,
                    write_event=write_event,
                    method=method,
                    params=_params or {},
                )
            if method not in _APPROVAL_METHODS:
                return {}
            return {"decision": ApprovalRequest.DECISION_ACCEPT}

        return _approve_all_handler

    def _interactive_handler(
        method: str, params: dict[str, Any] | None
    ) -> dict[str, Any]:
        if _is_user_input_request_method(method):
            return _handle_user_input_request(
                instance=instance,
                write_event=write_event,
                method=method,
                params=params or {},
            )
        if method not in _APPROVAL_METHODS:
            return {}
        request_id = _create_pending_approval(
            instance_id=instance.pk,
            method=method,
            params=params or {},
        )
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

    return _interactive_handler


def _is_user_input_request_method(method: str) -> bool:
    return (
        method in _USER_INPUT_METHODS
        or method.endswith("/requestUserInput")
        or method.endswith("/request_user_input")
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
    while time.monotonic() < deadline:
        response = _user_input_response_value(request_id)
        if isinstance(response, dict):
            return response
        time.sleep(_APPROVAL_POLL_INTERVAL)

    fallback: dict[str, Any] = {"answers": {}}
    _record_default_user_input_response(request_id, fallback)
    return fallback


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
) -> None:
    from django.db import connection

    try:
        UserInputRequest.objects.filter(pk=request_id, response__isnull=True).update(
            response=response,
            responded_at=timezone.now(),
        )
    finally:
        connection.close()


def _wait_for_decision(request_id: int) -> str:
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
            decision: str = ApprovalRequest.objects.values_list(
                "decision", flat=True
            ).get(pk=request_id)
        finally:
            connection.close()
        if decision:
            return ApprovalRequest.normalize_decision(decision)
        if time.monotonic() >= deadline:
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
                final_decision: str = ApprovalRequest.objects.values_list(
                    "decision", flat=True
                ).get(pk=request_id)
                return ApprovalRequest.normalize_decision(final_decision)
            finally:
                connection.close()
        time.sleep(_APPROVAL_POLL_INTERVAL)
