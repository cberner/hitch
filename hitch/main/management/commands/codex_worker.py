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

Interactive approvals route through ``_make_approval_handler``: when the
SDK reader thread receives an ``item/{commandExecution,fileChange}/
requestApproval`` request, the handler creates an ``ApprovalRequest`` row,
emits a synthetic ``approval/requested`` event into the events file (so
the SSE stream surfaces it), and blocks on a short-poll loop until the
Django view records a decision via ``POST /approval/<id>/``. The cap on
that wait is intentionally generous (``_APPROVAL_WAIT_SECONDS``) so a
user who steps away from the laptop doesn't lose the turn; on timeout the
handler defaults to ``denied`` because that is the safest no-answer
fallback (refuse the action, but let the agent keep running).
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import shutil
import signal
import threading
import time
from collections.abc import Callable
from typing import IO, Any, override

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
    DangerFullAccessSandboxPolicy,
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
from pydantic import BaseModel

from hitch.main.models import ApprovalRequest, CodexInstance

# JSON-RPC method names the SDK invokes on the client transport when codex's
# auto-reviewer escalates an action. The ``approve_all`` worker mode also
# routes through these methods (the wire-level ``ApprovalsReviewer.user``
# pairing forces every escalation back to the client transport), so the same
# handler covers both interactive and rubber-stamp policies.
_APPROVAL_METHODS = frozenset(
    {
        "item/commandExecution/requestApproval",
        "item/fileChange/requestApproval",
    }
)

# How long the worker waits on a single approval before defaulting to
# ``denied``. 30 minutes leaves plenty of slack for a user who stepped
# away without holding a worker forever; ``denied`` is the safer
# no-answer fallback because the agent keeps running but the
# specific action is refused.
_APPROVAL_WAIT_SECONDS = 60 * 30

# Cadence at which the approval handler polls the row for a recorded
# decision. Short enough that a click feels immediate; long enough not to
# pin a CPU per pending approval.
_APPROVAL_POLL_INTERVAL = 0.2

# Custom approval mode name not in ``ApprovalMode``. The SDK enum exposes
# only ``auto_review`` and ``deny_all``; this one is wired through by
# bypassing ``thread.turn(approval_mode=)`` and posting an on-request +
# no-reviewer ``TurnStartParams`` so the client's default auto-approve
# handler answers every escalation unconditionally.
_APPROVE_ALL = "approve_all"

# Set by the SIGTERM handler so the stream loop knows to call
# ``turn.interrupt()`` between events. Plain module-level bool is fine —
# CPython makes single-attribute reads/writes atomic, and the signal
# handler is intentionally minimal (it must avoid blocking JSON-RPC
# calls that would race the main loop's read on the response pipe).
_cancel_requested = False


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
        parser.add_argument("--sandbox-policy", type=str, default=None)
        parser.add_argument("--approval-mode", type=str, default=None)

    @override
    def handle(self, *args: Any, **options: Any) -> None:
        instance_id: int = options["instance_id"]
        reasoning_effort: str | None = options.get("reasoning_effort")
        sandbox_policy: str | None = options.get("sandbox_policy")
        approval_mode: str | None = options.get("approval_mode")
        instance = CodexInstance.objects.get(pk=instance_id)

        # Install the cancel handler before flipping to RUNNING so a Stop
        # click that lands the instant we transition can still be observed
        # by the stream loop (it'll see the flag on the first iteration).
        signal.signal(signal.SIGTERM, _on_sigterm)

        instance.status = CodexInstance.STATUS_RUNNING
        instance.save(update_fields=["status"])

        try:
            with open(instance.events_path, "a", buffering=1, encoding="utf-8") as events_file:
                final_turn = _run_turn(
                    instance=instance,
                    prompt=instance.prompt,
                    events_file=events_file,
                    reasoning_effort=reasoning_effort,
                    sandbox_policy=sandbox_policy,
                    approval_mode=approval_mode,
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
    reasoning_effort: str | None = None,
    sandbox_policy: str | None = None,
    approval_mode: str | None = None,
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

    def _write_event(method: str, payload: Any) -> None:
        line = _serialize_event(method, payload) + "\n"
        with write_lock:
            events_file.write(line)

    final_turn: Turn | None = None
    interrupt_sent = False
    with Codex(config=config) as codex:
        # The Codex top-level class instantiates its own AppServerClient
        # without an ``approval_handler`` argument, so the only way to wire
        # an interactive callback is to swap the bound method on the client
        # after construction. The SDK's default handler unconditionally
        # rubber-stamps (and emits a wire string codex's ``ReviewDecision``
        # enum no longer accepts), so we always install our own — either
        # the interactive one that opens an ``ApprovalRequest`` row, or
        # (only under ``approve_all``) a rubber-stamp that uses the
        # correct ``approved`` wire value.
        codex._client._approval_handler = _make_approval_handler(
            instance=instance,
            write_event=_write_event,
            approval_mode=approval_mode,
        )
        resume_kwargs: dict[str, Any] = {}
        if instance.developer_instructions:
            resume_kwargs["developer_instructions"] = instance.developer_instructions
        thread = codex.thread_resume(instance.thread_id, **resume_kwargs)
        turn = _start_turn(
            codex,
            thread,
            prompt=prompt,
            effort=effort,
            sandbox_policy=policy,
            approval_mode=approval_mode,
        )
        # A Stop click that landed before the turn handle existed sets the
        # flag without us being able to call interrupt yet; act on it now
        # that the handle is ready.
        if _cancel_requested and not interrupt_sent:
            interrupt_sent = _try_interrupt(turn)
        for event in turn.stream():
            _write_event(event.method, event.payload)
            payload = event.payload
            if isinstance(payload, TurnCompletedNotification) and payload.turn.id == turn.id:
                final_turn = payload.turn
            if _cancel_requested and not interrupt_sent:
                # SDK-level interrupt is the graceful cancellation path:
                # the app-server stops the model, emits the remaining
                # events (including a turn/completed with status=interrupted),
                # and the worker's normal status-update code at the end
                # records that as a failed turn. SIGKILL is the next
                # escalation if the user clicks Stop again.
                interrupt_sent = _try_interrupt(turn)
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


def _start_turn(
    codex: Codex,
    thread: Thread,
    *,
    prompt: str,
    effort: ReasoningEffort | None,
    sandbox_policy: SandboxPolicy | None,
    approval_mode: str | None,
) -> TurnHandle:
    """Start a turn under the requested approval policy.

    ``approve_all`` is not an ``ApprovalMode`` enum value, so we bypass
    ``Thread.turn(approval_mode=)`` and post the wire-level params
    directly: an on-request approval policy paired with the ``user``
    reviewer routes every escalation back to the client transport,
    where ``AppServerClient._default_approval_handler`` rubber-stamps
    both ``commandExecution`` and ``fileChange`` requests. Leaving
    ``approvals_reviewer`` unset falls back to server-side routing
    (typically ``auto_review``), which can still decline — so the
    explicit ``ApprovalsReviewer.user`` is load-bearing for the
    "approve everything" promise. Known SDK values (``auto_review``,
    ``deny_all``) and the unset case go through the typed
    ``Thread.turn`` API.
    """
    if approval_mode == _APPROVE_ALL:
        typed_input = [UserInput(root=TextUserInput(type="text", text=prompt))]
        params = TurnStartParams(
            thread_id=thread.id,
            input=typed_input,
            approval_policy=AskForApproval(root=AskForApprovalValue.on_request),
            approvals_reviewer=ApprovalsReviewer.user,
            effort=effort,
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
    if sandbox_policy is not None:
        turn_kwargs["sandbox_policy"] = sandbox_policy
    mode = _build_approval_mode(approval_mode)
    if mode is not None:
        turn_kwargs["approval_mode"] = mode
    return thread.turn(TextInput(prompt), **turn_kwargs)


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


def _serialize_event(method: str, payload: Any) -> str:
    if isinstance(payload, BaseModel):
        payload_data: Any = payload.model_dump(mode="json", by_alias=True)
    elif dataclasses.is_dataclass(payload) and not isinstance(payload, type):
        payload_data = dataclasses.asdict(payload)
    else:
        payload_data = payload
    return json.dumps({"method": method, "payload": payload_data})


WriteEvent = Callable[[str, Any], None]


def _make_approval_handler(
    *,
    instance: CodexInstance,
    write_event: WriteEvent,
    approval_mode: str | None,
) -> Callable[[str, dict[str, Any] | None], dict[str, Any]]:
    """Return an approval-handler closure bound to a single CodexInstance.

    Two flavors:

    * ``approve_all`` mode: every escalation is auto-answered ``approved``.
      This preserves the existing "approve everything" promise of that
      mode without going through the interactive UI loop.
    * Any other mode (including ``auto_review``): each escalation creates
      an ``ApprovalRequest`` row, emits an ``approval/requested`` event
      so the SSE stream surfaces it, and blocks polling the row until
      the Django view records a decision via ``POST /approval/<id>/``.

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
            if method not in _APPROVAL_METHODS:
                return {}
            return {"decision": ApprovalRequest.DECISION_APPROVED}

        return _approve_all_handler

    def _interactive_handler(
        method: str, params: dict[str, Any] | None
    ) -> dict[str, Any]:
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


def _wait_for_decision(request_id: int) -> str:
    """Poll the row for a recorded decision; default to ``denied`` on timeout.

    Polling (rather than a Postgres ``LISTEN``/``NOTIFY``-style wakeup)
    matches the rest of the project: the streaming layer also short-polls
    the events file, and the parent process is sqlite. The timeout exists
    so a worker doesn't sit forever if the user closed the browser tab —
    ``denied`` is the safest no-answer fallback (refuse the action, but
    let the agent keep running so a follow-up turn can recover).

    On the timeout path the conditional ``decision=""`` UPDATE is what
    serialises against a user who clicks at the deadline boundary: if
    that UPDATE matches zero rows, the row already carries a real
    decision and we round-trip it back to codex instead of clobbering
    the user's pick with ``denied``.
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
            return decision
        if time.monotonic() >= deadline:
            try:
                updated = ApprovalRequest.objects.filter(
                    pk=request_id, decision=""
                ).update(
                    decision=ApprovalRequest.DECISION_DENIED,
                    decided_at=tz.now(),
                )
                if updated:
                    return ApprovalRequest.DECISION_DENIED
                # Zero rows matched → the user wrote a real decision in
                # the window between the last read and this UPDATE.
                # Honour it rather than overwriting with ``denied``.
                return ApprovalRequest.objects.values_list(
                    "decision", flat=True
                ).get(pk=request_id)
            finally:
                connection.close()
        time.sleep(_APPROVAL_POLL_INTERVAL)
