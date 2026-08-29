"""Per-kind workflow handler registry for the system-agent engine.

Each ``SystemWorkflow`` kind registers a :class:`WorkflowHandler` that
declares the steps its state machine may occupy and receives lifecycle
events for its workflows. The registry is the single dispatch point for
finished system-agent turns, and the step-transition writers in
``system_agents`` validate every transition against the owning handler's
declared step set so an illegal step can never be persisted silently.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import ClassVar, TypeVar

from django.db import transaction

from hitch.main.models import CodexInstance, SystemAgentRun, SystemWorkflow

_ClaimResultT = TypeVar("_ClaimResultT")

# Administrative steps any kind may occupy: ``_block_workflow`` parks a
# failed workflow on ``blocked`` and the stale-workflow archiver moves
# long-blocked rows to ``archived``.
ADMINISTRATIVE_STEPS = frozenset({"blocked", "archived"})


def claim_workflow_transition(
    workflow: SystemWorkflow,
    apply: Callable[[SystemWorkflow], _ClaimResultT],
    *,
    expect_step: str | None = None,
    guard: Callable[[SystemWorkflow], bool] | None = None,
    require_active: bool = True,
) -> _ClaimResultT | None:
    """Lock the workflow row, re-validate, apply one transition, sync the snapshot.

    This is the engine's concurrency contract for every step/state writer,
    made structural instead of conventional:

    * Validation runs against the *locked* row, never the caller's snapshot.
      An in-memory copy can be stale by the time the lock is granted -- a
      concurrent claim (e.g. user steering) may have moved the step or bumped
      a revision, and writing through the snapshot would silently revert it.
    * ``apply`` runs inside the transaction. Row saves that must be atomic
      with the transition (the finished run's terminal status, sibling-row
      locks) belong in it; it must not spawn workers, run subprocesses, or
      call out -- those side effects belong after this returns, keyed off the
      returned action, so the SQLite write lock is never held across them.
    * On success the caller's ``workflow`` snapshot is refreshed from the
      locked row, so post-commit code never acts on pre-claim state. Work
      that runs unlocked afterwards and then writes again must re-claim.

    Returns ``apply``'s result, or ``None`` when the workflow was not
    claimable (inactive unless ``require_active=False``, step mismatch, or
    ``guard`` returned False against the locked row).
    """
    with transaction.atomic():
        locked = SystemWorkflow.objects.select_for_update().get(pk=workflow.pk)
        if require_active and not locked.is_active:
            return None
        if expect_step is not None and locked.step != expect_step:
            return None
        if guard is not None and not guard(locked):
            return None
        result = apply(locked)
        workflow.status = locked.status
        workflow.step = locked.step
        workflow.state = locked.state
        workflow.iteration = locked.iteration
        return result

# State keys the shared engine reads/writes for every kind: the per-turn
# config snapshot the spawn helpers replay, plus failure bookkeeping.
SHARED_STATE_KEYS = frozenset(
    {
        "model",
        "reasoning_effort",
        "sandbox_policy",
        "approval_mode",
        "web_search_mode",
        "developer_instructions",
        "enable_memories",
        "next_user_message_index",
        "archived_from_blocked",
        "error",
        "deferred_failure_surface",
        "failure_owner",
        "failure_surfaced",
        "workflow_turn_owner_index",
        "workflow_turn_owner_step",
        "workflow_turn_death_retries",
    }
)


@dataclass(frozen=True)
class SpawnRecoverySpec:
    """How to recover one ``(kind, step)`` workflow stranded by a dead spawn.

    A workflow commits its next step and *then* spawns the worker for it. If
    the process dies in that gap no ``CodexInstance`` row is created, so the
    terminal reconcilers have nothing to route and the workflow sits in
    ``step`` forever. ``needs_recovery`` is the authoritative "no live or
    finish-routing worker owns this step" predicate (re-checked under the
    claim lock so a worker that appears mid-sweep is never double-driven);
    ``recover`` re-drives the spawn or -- when the turn's prompt is
    unrecoverable -- blocks the workflow.
    """

    kind: str
    step: str
    stale_timeout: timedelta
    needs_recovery: Callable[[SystemWorkflow], bool]
    recover: Callable[[SystemWorkflow], None]


class WorkflowHandler:
    """Lifecycle hooks and the legal step set for one workflow kind.

    Subclasses set ``kind`` and ``steps`` (or ``None`` to opt out of step
    validation) and implement ``on_agent_finished``. Kinds with several
    agent shapes (e.g. an Autonomous Goal's candidate and judge) register
    multiple handlers for the same kind and discriminate in
    ``matches_run``; registration order decides precedence.
    """

    kind: ClassVar[str]
    steps: ClassVar[frozenset[str] | None]
    # Top-level SystemWorkflow.state keys this kind reads/writes, or None to
    # opt out of state-key validation. The typed readers in workflow_state
    # reject undeclared keys so a typo'd key fails loudly instead of
    # silently reading a default.
    state_keys: ClassVar[frozenset[str] | None] = None

    def matches_run(self, run: SystemAgentRun, instance: CodexInstance) -> bool:
        return True

    def on_agent_finished(
        self,
        instance: CodexInstance,
        run: SystemAgentRun,
        workflow: SystemWorkflow,
    ) -> None:
        raise NotImplementedError

    def on_feedback_finished(
        self, instance: CodexInstance, workflow: SystemWorkflow
    ) -> None:
        """A PURPOSE_SYSTEM_FEEDBACK turn for this kind finished. Default: ignore."""

    def on_user_turn_finished(
        self, instance: CodexInstance, workflow: SystemWorkflow
    ) -> None:
        """A workflow-owned PURPOSE_USER turn finished. Default: ignore."""

    def spawn_recovery_specs(self) -> tuple[SpawnRecoverySpec, ...]:
        """How to recover this kind's workflows stranded by a dead spawn.

        Default: no recoverable steps; the stale-workflow archiver is the
        only backstop for the kind.
        """
        return ()


_HANDLERS: list[WorkflowHandler] = []


def register(handler_cls: type[WorkflowHandler]) -> type[WorkflowHandler]:
    _HANDLERS.append(handler_cls())
    return handler_cls


def handler_for(
    workflow: SystemWorkflow,
    *,
    run: SystemAgentRun,
    instance: CodexInstance,
) -> WorkflowHandler | None:
    for handler in _HANDLERS:
        if handler.kind == workflow.kind and handler.matches_run(run, instance):
            return handler
    return None


def primary_handler(kind: str) -> WorkflowHandler | None:
    """The kind's default handler: the last-registered one for the kind.

    Used for events that are not tied to a SystemAgentRun (feedback and
    workflow-owned user turns), where ``matches_run`` discrimination does
    not apply.
    """
    for handler in reversed(_HANDLERS):
        if handler.kind == kind:
            return handler
    return None


def legal_steps(kind: str) -> frozenset[str] | None:
    """Steps workflows of ``kind`` may be moved to, or None when unvalidated.

    The union of every registered handler's declared steps for the kind,
    plus the administrative steps shared by all kinds. Returns None for
    unregistered kinds and for kinds whose handlers opt out of validation.
    """
    declared: set[str] = set()
    known = False
    for handler in _HANDLERS:
        if handler.kind != kind:
            continue
        if handler.steps is None:
            return None
        declared.update(handler.steps)
        known = True
    if not known:
        return None
    return frozenset(declared) | ADMINISTRATIVE_STEPS


def spawn_recovery_spec(kind: str, step: str) -> SpawnRecoverySpec | None:
    """The registered recovery spec for ``(kind, step)``, if any."""
    for handler in _HANDLERS:
        if handler.kind != kind:
            continue
        for spec in handler.spawn_recovery_specs():
            if spec.step == step:
                return spec
    return None


def declared_state_keys(kind: str) -> frozenset[str] | None:
    """State keys workflows of ``kind`` may read, or None when unvalidated.

    The union of every registered handler's declared keys for the kind plus
    the engine-shared keys. Returns None for unregistered kinds and for
    kinds whose handlers opt out.
    """
    declared: set[str] = set()
    known = False
    for handler in _HANDLERS:
        if handler.kind != kind:
            continue
        if handler.state_keys is None:
            return None
        declared.update(handler.state_keys)
        known = True
    if not known:
        return None
    return frozenset(declared) | SHARED_STATE_KEYS
