"""Per-kind workflow handler registry for the system-agent engine.

Each ``SystemWorkflow`` kind registers a :class:`WorkflowHandler` that
declares the steps its state machine may occupy and receives lifecycle
events for its workflows. The registry is the single dispatch point for
finished system-agent turns, and the step-transition writers in
``system_agents`` validate every transition against the owning handler's
declared step set so an illegal step can never be persisted silently.
"""

from __future__ import annotations

from typing import ClassVar

from hitch.main.models import CodexInstance, SystemAgentRun, SystemWorkflow

# Administrative steps any kind may occupy: ``_block_workflow`` parks a
# failed workflow on ``blocked`` and the stale-workflow archiver moves
# long-blocked rows to ``archived``.
ADMINISTRATIVE_STEPS = frozenset({"blocked", "archived"})


class WorkflowHandler:
    """Lifecycle hooks and the legal step set for one workflow kind.

    Subclasses set ``kind`` and ``steps`` (or ``None`` to opt out of step
    validation) and implement ``on_agent_finished``. Kinds with several
    agent shapes (e.g. the PR-QA workflow's followup monitor) register
    multiple handlers for the same kind and discriminate in
    ``matches_run``; registration order decides precedence.
    """

    kind: ClassVar[str]
    steps: ClassVar[frozenset[str] | None]

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
