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
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import shutil
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

from hitch.main.models import CodexInstance

# Custom approval mode name not in ``ApprovalMode``. The SDK enum exposes
# only ``auto_review`` and ``deny_all``; this one is wired through by
# bypassing ``thread.turn(approval_mode=)`` and posting an on-request +
# no-reviewer ``TurnStartParams`` so the client's default auto-approve
# handler answers every escalation unconditionally.
_APPROVE_ALL = "approve_all"

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

        instance.status = CodexInstance.STATUS_RUNNING
        instance.save(update_fields=["status"])

        try:
            with open(instance.events_path, "a", buffering=1, encoding="utf-8") as events_file:
                final_turn = _run_turn(
                    thread_id=instance.thread_id,
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
    thread_id: str,
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
    final_turn: Turn | None = None
    with Codex(config=config) as codex:
        thread = codex.thread_resume(thread_id)
        turn = _start_turn(
            codex,
            thread,
            prompt=prompt,
            effort=effort,
            sandbox_policy=policy,
            approval_mode=approval_mode,
        )
        for event in turn.stream():
            events_file.write(_serialize_event(event.method, event.payload) + "\n")
            payload = event.payload
            if isinstance(payload, TurnCompletedNotification) and payload.turn.id == turn.id:
                final_turn = payload.turn
    return final_turn


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
