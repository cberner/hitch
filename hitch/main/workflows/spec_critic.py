"""The Spec Critic workflow: pre-implementation analysis of a user prompt.

State machine: classifying -> analyzing -> (clarifying ->) synthesizing ->
implementation_spawned. A classifier turn first decides whether critique is
warranted at all; the analysis fans out requirements/risk/test agents, an
optional clarification round asks the user durable questions, and synthesis
produces the brief the implementation turn runs with.

Shared spawn/transition/blocking helpers stay in ``system_agents`` and are
reached through the module object so test patches on that namespace keep
intercepting.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import timedelta
from typing import Any, override

from django.db import IntegrityError, close_old_connections, transaction
from django.utils import timezone
from openai_codex import ApprovalMode, Codex, TextInput
from openai_codex.generated.v2_all import (
    ReadOnlySandboxPolicy,
    SandboxPolicy,
    ThreadSource,
    Turn,
    TurnCompletedNotification,
    TurnStatus,
)

from hitch.main import caches
from hitch.main.models import (
    CodexInstance,
    SystemAgentRun,
    SystemWorkflow,
    UserInputRequest,
)
from hitch.main.runtime import app_server_pool, codex_pool
from hitch.main.workflows import engine, system_agents
from hitch.main.workflows.agent_io import (
    SPEC_REQUIREMENTS_AGENT_KIND,
    SPEC_RISK_AGENT_KIND,
    SPEC_SYNTHESIZER_AGENT_KIND,
    SPEC_TEST_AGENT_KIND,
    _parse_spec_critic_output,
)
from hitch.main.workflows.spec_critic_prompts import (
    _SPEC_CRITIC_ANALYSIS_AGENT_KINDS,
    _latest_agent_text_from_turn,
    _parse_spec_critic_classifier_output,
    _spec_critic_classifier_model_rank,
    _spec_critic_classifier_prompt,
    _spec_critic_should_run_heuristic,
    _spec_implementation_prompt,
    _spec_questions_from_outputs,
    _spec_questions_from_state,
    _spec_requirements_prompt,
    _spec_risk_prompt,
    _spec_safe_defaults_from_state,
    _spec_synthesis_prompt,
    _spec_test_prompt,
)
from hitch.main.workflows.workflow_state import (
    _state_bool,
    _state_dict,
    _state_int,
    _state_string,
)

logger = logging.getLogger(__name__)

# A classification/analysis claim older than this with no worker is stranded.
_SPEC_CRITIC_CLASSIFY_STALE_TIMEOUT = timedelta(minutes=5)


_SPEC_CRITIC_CLASSIFIER_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "should_run": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["should_run", "reason"],
}

def spec_critic_should_run(prompt: str, *, cwd: str | None = None) -> bool:
    """Return whether an ordinary implementation prompt needs preflight critique."""
    text = " ".join(prompt.strip().split())
    if not text:
        return False
    classified = _classify_spec_critic_prompt_with_codex(text, cwd=cwd)
    if classified is not None:
        return classified
    return _spec_critic_should_run_heuristic(text)

def _classify_spec_critic_prompt_with_codex(
    prompt: str, *, cwd: str | None
) -> bool | None:
    try:
        with app_server_pool.borrow_codex(Codex, enable_memories=False) as codex:
            model = _smallest_available_codex_model(
                caches._models_data_from_codex(codex)
            )
            thread = codex.thread_start(
                cwd=cwd or os.getcwd(),
                ephemeral=True,
                model=model,
                approval_mode=ApprovalMode.deny_all,
                thread_source=ThreadSource.subagent,
            )
            turn = thread.turn(
                TextInput(_spec_critic_classifier_prompt(prompt)),
                model=model,
                approval_mode=ApprovalMode.deny_all,
                sandbox_policy=SandboxPolicy(
                    root=ReadOnlySandboxPolicy(type="readOnly")
                ),
                output_schema=_SPEC_CRITIC_CLASSIFIER_OUTPUT_SCHEMA,
            )
            final_turn: Turn | None = None
            for event in turn.stream():
                payload = getattr(event, "payload", None)
                if isinstance(payload, TurnCompletedNotification):
                    final_turn = payload.turn
            if final_turn is None or final_turn.status != TurnStatus.completed:
                return None
            return _parse_spec_critic_classifier_output(
                _latest_agent_text_from_turn(final_turn)
            )
    except Exception:
        system_agents.logger.warning("failed to classify Spec Critic prompt with Codex", exc_info=True)
        return None

def _smallest_available_codex_model(models_data: list[Any]) -> str | None:
    visible_models = [
        model for model in models_data if not bool(getattr(model, "hidden", False))
    ]
    candidates = visible_models or models_data
    if not candidates:
        return None
    model = min(candidates, key=_spec_critic_classifier_model_rank)
    model_id = getattr(model, "id", None)
    return model_id if isinstance(model_id, str) and model_id.strip() else None

def start_spec_critic_workflow(
    *,
    main_thread_id: str,
    cwd: str,
    prompt: str,
    sandbox_policy: str | None,
    approval_mode: str | None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    base_instructions: str | None = None,
    developer_instructions: str | None = None,
    enable_memories: bool = False,
    web_search_mode: str | None = None,
    initial_user_message_index: int = 0,
    auto_pr_enabled: bool = False,
    auto_qa_enabled: bool = False,
    auto_merge_to_local_branch: bool = False,
    auto_merge_branch: str = "",
) -> SystemWorkflow:
    """Start the Spec Critic workflow for the visible implementation turn.

    The workflow opens in ``STEP_SPEC_CRITIC_CLASSIFYING`` and runs the
    should-run classifier on a background thread, so the request that triggered
    it returns immediately instead of blocking on an LLM call. The classifier
    then either advances to the analysis agents or skips straight to the user's
    original prompt.
    """
    auto_merge_branch = (
        auto_merge_branch.strip() if auto_merge_to_local_branch else ""
    )
    auto_merge_to_local_branch = bool(auto_qa_enabled and auto_merge_branch)
    if not auto_merge_to_local_branch:
        auto_merge_branch = ""
    try:
        with transaction.atomic():
            workflow = SystemWorkflow.objects.create(
                kind=system_agents.SPEC_CRITIC_WORKFLOW_KIND,
                main_thread_id=main_thread_id,
                cwd=cwd,
                status=SystemWorkflow.STATUS_RUNNING,
                step=system_agents.STEP_SPEC_CRITIC_CLASSIFYING,
                max_iterations=1,
                state={
                    "original_prompt": prompt,
                    "sandbox_policy": sandbox_policy or "",
                    "approval_mode": approval_mode or "",
                    "model": model or "",
                    "reasoning_effort": reasoning_effort or "",
                    "base_instructions": base_instructions or "",
                    "developer_instructions": developer_instructions or "",
                    "enable_memories": enable_memories,
                    "web_search_mode": web_search_mode or "",
                    "next_user_message_index": max(initial_user_message_index, 0),
                    "auto_pr_enabled": auto_pr_enabled,
                    "auto_qa_enabled": auto_qa_enabled,
                    "auto_merge_to_local_branch": auto_merge_to_local_branch,
                    "auto_merge_branch": auto_merge_branch,
                },
            )
    except IntegrityError:
        existing_workflow = SystemWorkflow.objects.filter(
            kind=system_agents.SPEC_CRITIC_WORKFLOW_KIND,
            main_thread_id=main_thread_id,
            status=SystemWorkflow.STATUS_RUNNING,
        ).first()
        if existing_workflow is None:
            raise
        return existing_workflow

    _start_spec_critic_classification(workflow)
    return workflow

def _begin_spec_critic_analysis(workflow: SystemWorkflow) -> None:
    try:
        _spawn_spec_critic_analysis_runs(workflow)
    except Exception as exc:
        _block_spec_critic_workflow(
            workflow, f"failed to start Spec Critic agents: {exc!r}"
        )

def _spec_critic_spawned_analysis_kinds(workflow: SystemWorkflow) -> set[str]:
    """Analysis agent kinds a prior spawn already launched.

    Counts a kind as launched if it has either a ``SystemAgentRun`` or a
    workflow-owned ``CodexInstance`` (the instance is persisted before its run
    row, so a handler that died in that gap leaves an instance with no run).
    """
    kinds = set(
        workflow.agent_runs.filter(
            agent_kind__in=_SPEC_CRITIC_ANALYSIS_AGENT_KINDS
        ).values_list("agent_kind", flat=True)
    )
    kinds |= set(
        CodexInstance.objects.filter(
            workflow_id=workflow.pk,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind__in=_SPEC_CRITIC_ANALYSIS_AGENT_KINDS,
        ).values_list("agent_kind", flat=True)
    )
    return kinds

def _spec_critic_analysis_needs_recovery(workflow: SystemWorkflow) -> bool:
    """True when an ANALYZING workflow can no longer reach a complete fan-in.

    The fan-in advances only once every analysis kind has a *completed* run, so
    a stranded ANALYZING step is one that is either missing a kind (the
    never-spawned orphan or a partial spawn) or has a kind whose run already
    failed (its finish handler died before blocking). A failed run can never
    reach COMPLETED and the terminal reconciler skips terminal runs, so without
    recovery the workflow would hang forever.
    """
    if _spec_critic_spawned_analysis_kinds(workflow) != set(
        _SPEC_CRITIC_ANALYSIS_AGENT_KINDS
    ):
        return True
    return workflow.agent_runs.filter(
        agent_kind__in=_SPEC_CRITIC_ANALYSIS_AGENT_KINDS,
        status=SystemAgentRun.STATUS_FAILED,
    ).exists()

def _recover_spec_critic_analysis(workflow: SystemWorkflow) -> None:
    """Recover an ANALYZING workflow whose analysis fan-out was stranded.

    Re-spawn the agents only for the never-launched orphan, where nothing
    exists to duplicate. A partial spawn cannot be safely completed from here:
    re-launching only the missing kinds races the original spawn loop (which
    holds no per-kind claim), and a kind whose run already failed can never
    reach COMPLETED, so the fan-in would never advance. Block instead and
    surface the failure so the user can retry the critique cleanly.
    """
    if not _spec_critic_spawned_analysis_kinds(workflow):
        _begin_spec_critic_analysis(workflow)
        return
    _block_spec_critic_workflow(
        workflow,
        "Spec Critic analysis did not start all of its agents. Please retry.",
    )

def _start_spec_critic_classification(workflow: SystemWorkflow) -> None:
    """Classify the prompt off the request path, then route the workflow."""
    try:
        threading.Thread(
            target=_run_spec_critic_classification,
            args=(workflow.pk,),
            name=f"spec-critic-classify-{workflow.pk}",
            daemon=True,
        ).start()
    except Exception:
        # If the classifier thread cannot even start, run the critique inline so
        # the request is never silently dropped.
        system_agents.logger.exception("failed to start Spec Critic classifier thread")
        _advance_spec_critic_to_analysis(workflow)

def _run_spec_critic_classification(workflow_id: int) -> None:
    close_old_connections()
    try:
        workflow = SystemWorkflow.objects.filter(
            pk=workflow_id,
            kind=system_agents.SPEC_CRITIC_WORKFLOW_KIND,
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_SPEC_CRITIC_CLASSIFYING,
        ).first()
        if workflow is None:
            return
        try:
            needs_critique = spec_critic_should_run(
                _state_string(workflow, "original_prompt"), cwd=workflow.cwd or None
            )
        except Exception:
            # spec_critic_should_run already falls back to a heuristic internally,
            # so reaching here is unexpected; skip the critique rather than trap
            # the user's turn behind a broken preflight.
            system_agents.logger.exception("Spec Critic prompt classification raised")
            needs_critique = False
        if needs_critique:
            _advance_spec_critic_to_analysis(workflow)
        else:
            _skip_spec_critic_and_implement(workflow)
    except Exception:
        system_agents.logger.exception(
            "Spec Critic classification routing failed for workflow %s", workflow_id
        )
    finally:
        close_old_connections()

def _advance_spec_critic_to_analysis(workflow: SystemWorkflow) -> None:
    def _advance(locked: SystemWorkflow) -> bool:
        locked.step = system_agents.STEP_SPEC_CRITIC_ANALYZING
        locked.save(update_fields=["step", "updated_at"])
        return True

    if engine.claim_workflow_transition(
        workflow, _advance, expect_step=system_agents.STEP_SPEC_CRITIC_CLASSIFYING
    ):
        _begin_spec_critic_analysis(workflow)

def _skip_spec_critic_and_implement(workflow: SystemWorkflow) -> None:
    """Run the user's original prompt directly when no critique is warranted."""

    def _skip(locked: SystemWorkflow) -> bool:
        # Claim the workflow before spawning so the turn cannot be double-started.
        # ``skipped_classification`` is recorded now (not on completion) so a
        # reconciler can tell a stranded IMPLEMENTATION_SPAWNED workflow apart
        # from the synthesis path and recover it with the original prompt.
        locked.step = system_agents.STEP_SPEC_CRITIC_IMPLEMENTATION_SPAWNED
        locked.state = {**locked.state, "skipped_classification": True}
        locked.save(update_fields=["step", "state", "updated_at"])
        return True

    if engine.claim_workflow_transition(
        workflow, _skip, expect_step=system_agents.STEP_SPEC_CRITIC_CLASSIFYING
    ):
        _finalize_spec_critic_skip(workflow)

def _finalize_spec_critic_skip(workflow: SystemWorkflow) -> None:
    """Spawn the original-prompt turn for a skipped workflow, then complete it.

    Idempotent: if the implementation turn already exists (e.g. a restart killed
    the thread between the spawn and the completion save) it only finalizes the
    workflow row rather than spawning a duplicate turn.
    """
    if not _spec_critic_implementation_turn_exists(workflow):
        try:
            _spawn_spec_critic_implementation_turn(workflow, None)
        except Exception as exc:
            _block_spec_critic_workflow(
                workflow,
                f"failed to start implementation after Spec Critic skip: {exc!r}",
            )
            return
    workflow.status = SystemWorkflow.STATUS_COMPLETED
    workflow.save(update_fields=["status", "updated_at"])

def _spec_critic_implementation_turn_exists(workflow: SystemWorkflow) -> bool:
    """Whether the skipped workflow's original-prompt turn was already spawned.

    The turn is the next user turn on the visible thread, so it is uniquely
    identified by the thread id and the recorded user-message index.
    """
    return CodexInstance.objects.filter(
        thread_id=workflow.main_thread_id,
        user_message_index=_state_int(workflow, "next_user_message_index"),
    ).exists()

@engine.register
class _SpecCriticHandler(engine.WorkflowHandler):
    kind = system_agents.SPEC_CRITIC_WORKFLOW_KIND
    # Top-level SystemWorkflow.state keys this machine reads and writes (the
    # engine-shared turn-config/failure keys live in engine.SHARED_STATE_KEYS).
    state_keys = frozenset(
        {
            "auto_merge_branch",
            "auto_merge_to_local_branch",
            "auto_pr_enabled",
            "auto_qa_enabled",
            "clarification_answers",
            "clarification_questions",
            "clarification_request_id",
            "clarification_safe_defaults",
            "clarification_source",
            "original_prompt",
            "skipped_classification",
            "synthesized_brief",
        }
    )

    @override
    def spawn_recovery_specs(self) -> tuple[engine.SpawnRecoverySpec, ...]:
        return (
            engine.SpawnRecoverySpec(
                kind=self.kind,
                step=system_agents.STEP_SPEC_CRITIC_CLASSIFYING,
                stale_timeout=_SPEC_CRITIC_CLASSIFY_STALE_TIMEOUT,
                needs_recovery=lambda w: True,
                recover=lambda w: _start_spec_critic_classification(w),
            ),
            engine.SpawnRecoverySpec(
                kind=self.kind,
                step=system_agents.STEP_SPEC_CRITIC_ANALYZING,
                stale_timeout=_SPEC_CRITIC_CLASSIFY_STALE_TIMEOUT,
                # Recover ANALYZING when it is missing a run for any analysis
                # kind: re-spawn the never-launched orphan, but block a partial
                # spawn (which cannot be safely completed -- see
                # _recover_spec_critic_analysis).
                needs_recovery=_spec_critic_analysis_needs_recovery,
                recover=_recover_spec_critic_analysis,
            ),
            engine.SpawnRecoverySpec(
                kind=self.kind,
                step=system_agents.STEP_SPEC_CRITIC_SYNTHESIZING,
                stale_timeout=_SPEC_CRITIC_CLASSIFY_STALE_TIMEOUT,
                needs_recovery=_spec_critic_synthesizing_needs_recovery,
                recover=_recover_spec_critic_synthesizing,
            ),
            engine.SpawnRecoverySpec(
                kind=self.kind,
                step=system_agents.STEP_SPEC_CRITIC_IMPLEMENTATION_SPAWNED,
                stale_timeout=_SPEC_CRITIC_CLASSIFY_STALE_TIMEOUT,
                # Only the skip path leaves this step RUNNING (the synthesis
                # path sets it together with COMPLETED), so finalizing with
                # the original prompt is correct.
                needs_recovery=lambda w: _state_bool(w, "skipped_classification"),
                recover=lambda w: _finalize_spec_critic_skip(w),
            ),
        )

    steps = frozenset(
        {
            system_agents.STEP_SPEC_CRITIC_CLASSIFYING,
            system_agents.STEP_SPEC_CRITIC_ANALYZING,
            system_agents.STEP_SPEC_CRITIC_CLARIFYING,
            system_agents.STEP_SPEC_CRITIC_SYNTHESIZING,
            system_agents.STEP_SPEC_CRITIC_IMPLEMENTATION_SPAWNED,
        }
    )

    @override
    def on_agent_finished(
        self,
        instance: CodexInstance,
        run: SystemAgentRun,
        workflow: SystemWorkflow,
    ) -> None:
        _handle_spec_critic_agent_finished(instance, run, workflow)

def _handle_spec_critic_agent_finished(
    instance: CodexInstance, run: SystemAgentRun, workflow: SystemWorkflow
) -> None:
    if not workflow.is_active:
        _finish_spec_critic_run(instance, run, block_workflow=False)
        return
    if instance.status != CodexInstance.STATUS_COMPLETED:
        system_agents._fail_run(
            run,
            f"Spec Critic agent {run.agent_kind} failed: {instance.error}",
            block_workflow=False,
        )
        _block_spec_critic_workflow(
            workflow, f"Spec Critic agent {run.agent_kind} failed: {instance.error}"
        )
        return
    if not _finish_spec_critic_run(instance, run, block_workflow=True):
        return
    if run.agent_kind in _SPEC_CRITIC_ANALYSIS_AGENT_KINDS:
        _maybe_advance_spec_critic_after_analysis(workflow)
        return
    if run.agent_kind == SPEC_SYNTHESIZER_AGENT_KIND:
        _complete_spec_critic_workflow(workflow, run)
        return
    _block_spec_critic_workflow(
        workflow, f"unsupported Spec Critic agent kind {run.agent_kind!r}"
    )

def _finish_spec_critic_run(
    instance: CodexInstance, run: SystemAgentRun, *, block_workflow: bool
) -> bool:
    if run.status in (SystemAgentRun.STATUS_COMPLETED, SystemAgentRun.STATUS_FAILED):
        return run.status == SystemAgentRun.STATUS_COMPLETED
    raw_output = system_agents._final_agent_text(instance.events_path)
    parsed = _parse_spec_critic_output(run.agent_kind, raw_output)
    if parsed is None:
        error = f"Spec Critic agent {run.agent_kind} output was not valid JSON"
        system_agents._fail_run(
            run,
            error,
            raw_output=raw_output,
            block_workflow=False,
        )
        if block_workflow:
            _block_spec_critic_workflow(run.workflow, error)
        return False
    run.status = SystemAgentRun.STATUS_COMPLETED
    run.output = parsed
    run.raw_output = raw_output
    run.save(update_fields=["status", "output", "raw_output", "updated_at"])
    return True

def _maybe_advance_spec_critic_after_analysis(workflow: SystemWorkflow) -> None:
    action, error = _claim_spec_critic_analysis_advance(workflow)
    if action == "block":
        _block_spec_critic_workflow(workflow, error)
        return
    if action != "synthesize":
        return
    try:
        _spawn_spec_critic_synthesizer_run(workflow)
    except Exception as exc:
        _block_spec_critic_workflow(
            workflow, f"failed to start Spec Critic synthesizer: {exc!r}"
        )

def _claim_spec_critic_analysis_advance(workflow: SystemWorkflow) -> tuple[str, str]:
    def _advance(locked: SystemWorkflow) -> tuple[str, str]:
        completed_kinds = set(
            locked.agent_runs.filter(
                agent_kind__in=_SPEC_CRITIC_ANALYSIS_AGENT_KINDS,
                status=SystemAgentRun.STATUS_COMPLETED,
            ).values_list("agent_kind", flat=True)
        )
        if completed_kinds != set(_SPEC_CRITIC_ANALYSIS_AGENT_KINDS):
            return "", ""
        required, safe_defaults = _spec_critic_clarification_plan(locked)
        if required:
            run = (
                locked.agent_runs.filter(
                    agent_kind=SPEC_RISK_AGENT_KIND,
                    status=SystemAgentRun.STATUS_COMPLETED,
                )
                .select_related("instance")
                .order_by("-created_at")
                .first()
            )
            if run is None:
                return "block", "Spec Critic could not create a clarification request"
            _create_spec_critic_clarification_request(
                locked, run, required, safe_defaults
            )
            return "clarify", ""
        locked.state = {
            **locked.state,
            "clarification_answers": safe_defaults,
            "clarification_source": "safe_defaults" if safe_defaults else "not_needed",
        }
        locked.step = system_agents.STEP_SPEC_CRITIC_SYNTHESIZING
        locked.save(update_fields=["step", "state", "updated_at"])
        return "synthesize", ""

    result = engine.claim_workflow_transition(
        workflow, _advance, expect_step=system_agents.STEP_SPEC_CRITIC_ANALYZING
    )
    return result if result is not None else ("", "")

def on_user_input_resolved(input_request: UserInputRequest) -> None:
    """Resume workflows that created their own durable clarification prompt."""
    if input_request.method != system_agents.SPEC_CRITIC_CLARIFICATION_METHOD:
        return
    run = (
        SystemAgentRun.objects.select_related("workflow")
        .filter(instance=input_request.instance)
        .first()
    )
    if run is None or run.workflow.kind != system_agents.SPEC_CRITIC_WORKFLOW_KIND:
        return
    workflow = run.workflow
    if (
        not workflow.is_active
        or workflow.step != system_agents.STEP_SPEC_CRITIC_CLARIFYING
    ):
        return
    _handle_spec_critic_clarification_response(workflow, input_request)

def _handle_spec_critic_clarification_response(
    workflow: SystemWorkflow, input_request: UserInputRequest
) -> None:
    action, error = _claim_spec_critic_clarification_response(
        workflow, input_request
    )
    if action == "block":
        _block_spec_critic_workflow(workflow, error)
        return
    if action != "synthesize":
        return
    try:
        _spawn_spec_critic_synthesizer_run(workflow)
    except Exception as exc:
        _block_spec_critic_workflow(
            workflow, f"failed to start Spec Critic synthesizer: {exc!r}"
        )

def _claim_spec_critic_clarification_response(
    workflow: SystemWorkflow, input_request: UserInputRequest
) -> tuple[str, str]:
    answers = system_agents._answers_from_input_request(input_request)

    def _resolve(locked: SystemWorkflow) -> tuple[str, str]:
        questions = _spec_questions_from_state(locked, only_pending=True)
        safe_defaults = _spec_safe_defaults_from_state(locked)
        recorded_answers = {
            **safe_defaults,
            **_state_dict(locked, "clarification_answers"),
        }
        merged_answers: dict[str, Any] = {}
        missing: list[dict[str, Any]] = []
        for question in questions:
            qid = question["id"]
            answer = answers.get(qid)
            if system_agents._answer_is_present(answer):
                merged_answers[qid] = answer
                continue
            if qid in safe_defaults:
                merged_answers[qid] = safe_defaults[qid]
                continue
            missing.append(question)
        recorded_answers = {**recorded_answers, **merged_answers}
        locked.state = {
            **locked.state,
            "clarification_answers": recorded_answers,
            "clarification_source": "user",
        }
        if missing:
            run = _spec_critic_clarification_run(locked)
            if run is None:
                locked.save(update_fields=["state", "updated_at"])
                return "block", "Spec Critic could not create a clarification request"
            _create_spec_critic_clarification_request(
                locked, run, missing, safe_defaults
            )
            return "clarify", ""
        locked.step = system_agents.STEP_SPEC_CRITIC_SYNTHESIZING
        locked.save(update_fields=["step", "state", "updated_at"])
        return "synthesize", ""

    result = engine.claim_workflow_transition(
        workflow, _resolve, expect_step=system_agents.STEP_SPEC_CRITIC_CLARIFYING
    )
    return result if result is not None else ("", "")

def _complete_spec_critic_workflow(
    workflow: SystemWorkflow, run: SystemAgentRun
) -> None:
    output = run.output if isinstance(run.output, dict) else {}
    brief = output.get("brief")
    if not isinstance(brief, str) or not brief.strip():
        _block_spec_critic_workflow(workflow, "Spec Critic synthesizer returned no brief")
        return
    # Idempotent like the skip path: synthesizing recovery re-drives this
    # completion after a death between the implementation spawn and the
    # workflow save, and must not start a duplicate implementation turn.
    if not _spec_critic_implementation_turn_exists(workflow):
        try:
            _spawn_spec_critic_implementation_turn(workflow, brief.strip())
        except Exception as exc:
            _block_spec_critic_workflow(
                workflow, f"failed to start implementation from Spec Critic brief: {exc!r}"
            )
            return
    workflow.state = {**workflow.state, "synthesized_brief": brief.strip()}
    system_agents._complete_workflow(workflow, system_agents.STEP_SPEC_CRITIC_IMPLEMENTATION_SPAWNED)

def _spec_critic_synthesizer_run(workflow: SystemWorkflow) -> SystemAgentRun | None:
    return (
        workflow.agent_runs.filter(agent_kind=SPEC_SYNTHESIZER_AGENT_KIND)
        .order_by("-created_at", "-pk")
        .first()
    )

def _spec_critic_synthesizing_needs_recovery(workflow: SystemWorkflow) -> bool:
    """No live or finish-routing synthesizer owns the SYNTHESIZING step.

    Two stranding shapes: the claim committed the step but died before
    spawning the synthesizer (no run exists), or the synthesizer's finish
    handler saved the terminal run and died before advancing the workflow
    (the terminal-instance reconciler skips runs that are already terminal,
    so nothing else can route it).
    """
    run = _spec_critic_synthesizer_run(workflow)
    if run is None:
        return True
    if run.status not in (
        SystemAgentRun.STATUS_COMPLETED,
        SystemAgentRun.STATUS_FAILED,
    ):
        # A live worker (or a dead one awaiting terminal-instance
        # reconciliation) owns the step.
        return False
    # Leave a freshly-claimed routing alone: the original finish handler may
    # still be advancing the workflow right now.
    fresh_claim = timezone.now() - system_agents._WORKFLOW_ROUTE_CLAIM_TIMEOUT
    return not CodexInstance.objects.filter(
        pk=run.instance_id,
        workflow_routing_started_at__gte=fresh_claim,
    ).exists()

def _recover_spec_critic_synthesizing(workflow: SystemWorkflow) -> None:
    run = _spec_critic_synthesizer_run(workflow)
    if run is None:
        try:
            _spawn_spec_critic_synthesizer_run(workflow)
        except Exception as exc:
            _block_spec_critic_workflow(
                workflow, f"failed to start Spec Critic synthesizer: {exc!r}"
            )
        return
    if run.status == SystemAgentRun.STATUS_COMPLETED:
        _complete_spec_critic_workflow(workflow, run)
        return
    _block_spec_critic_workflow(
        workflow,
        run.error or "Spec Critic synthesizer failed",
    )

def _spawn_spec_critic_analysis_runs(workflow: SystemWorkflow) -> list[SystemAgentRun]:
    prompts_and_schemas = (
        (
            SPEC_REQUIREMENTS_AGENT_KIND,
            _spec_requirements_prompt(workflow),
            system_agents._SPEC_REQUIREMENTS_OUTPUT_SCHEMA,
            {"focus": "requirements"},
        ),
        (
            SPEC_RISK_AGENT_KIND,
            _spec_risk_prompt(workflow),
            system_agents._SPEC_RISK_OUTPUT_SCHEMA,
            {"focus": "ambiguity_risk"},
        ),
        (
            SPEC_TEST_AGENT_KIND,
            _spec_test_prompt(workflow),
            system_agents._SPEC_TEST_OUTPUT_SCHEMA,
            {"focus": "acceptance_tests"},
        ),
    )
    runs: list[SystemAgentRun] = []
    for agent_kind, prompt, schema, run_input in prompts_and_schemas:
        instance = codex_pool.spawn_new_session(
            cwd=workflow.cwd,
            prompt=prompt,
            base_instructions=_state_string(workflow, "base_instructions") or None,
            developer_instructions=_state_string(workflow, "developer_instructions")
            or None,
            model=_state_string(workflow, "model") or None,
            reasoning_effort=_state_string(workflow, "reasoning_effort") or None,
            approval_mode=system_agents.SYSTEM_AGENT_APPROVAL_MODE,
            sandbox_policy="readOnly",
            enable_memories=_state_bool(workflow, "enable_memories"),
            web_search_mode=system_agents._workflow_web_search_mode(workflow),
            thread_source=ThreadSource.subagent,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=agent_kind,
            display_author=system_agents.SPEC_CRITIC_DISPLAY_AUTHOR,
            output_schema=schema,
        )
        run, _created = SystemAgentRun.objects.get_or_create(
            instance=instance,
            defaults={
                "workflow": workflow,
                "agent_kind": agent_kind,
                "thread_id": instance.thread_id,
                "status": SystemAgentRun.STATUS_RUNNING,
                "input": {
                    "cwd": workflow.cwd,
                    "prompt": _state_string(workflow, "original_prompt"),
                    **run_input,
                },
            },
        )
        runs.append(run)
    return runs

def _spawn_spec_critic_synthesizer_run(workflow: SystemWorkflow) -> SystemAgentRun:
    prompt = _spec_synthesis_prompt(workflow)
    instance = codex_pool.spawn_new_session(
        cwd=workflow.cwd,
        prompt=prompt,
        base_instructions=_state_string(workflow, "base_instructions") or None,
        developer_instructions=_state_string(workflow, "developer_instructions") or None,
        model=_state_string(workflow, "model") or None,
        reasoning_effort=_state_string(workflow, "reasoning_effort") or None,
        approval_mode=system_agents.SYSTEM_AGENT_APPROVAL_MODE,
        sandbox_policy="readOnly",
        enable_memories=_state_bool(workflow, "enable_memories"),
        web_search_mode=system_agents._workflow_web_search_mode(workflow),
        thread_source=ThreadSource.subagent,
        purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
        workflow_id=workflow.pk,
        agent_kind=SPEC_SYNTHESIZER_AGENT_KIND,
        display_author=system_agents.SPEC_CRITIC_DISPLAY_AUTHOR,
        output_schema=system_agents._SPEC_SYNTHESIS_OUTPUT_SCHEMA,
    )
    run, _created = SystemAgentRun.objects.get_or_create(
        instance=instance,
        defaults={
            "workflow": workflow,
            "agent_kind": SPEC_SYNTHESIZER_AGENT_KIND,
            "thread_id": instance.thread_id,
            "status": SystemAgentRun.STATUS_RUNNING,
            "input": {
                "cwd": workflow.cwd,
                "prompt": _state_string(workflow, "original_prompt"),
                "clarification_answers": _state_dict(
                    workflow, "clarification_answers"
                ),
            },
        },
    )
    return run

def _spawn_spec_critic_implementation_turn(
    workflow: SystemWorkflow, brief: str | None
) -> CodexInstance:
    # A None brief means the classifier decided no critique was needed, so run
    # the user's original request verbatim instead of a synthesized brief.
    prompt = (
        _spec_implementation_prompt(workflow, brief)
        if brief is not None
        else _state_string(workflow, "original_prompt")
    )
    auto_qa_enabled = _state_bool(workflow, "auto_qa_enabled")
    auto_merge_branch = _state_string(workflow, "auto_merge_branch")
    auto_merge_to_local_branch = bool(
        auto_qa_enabled
        and _state_bool(workflow, "auto_merge_to_local_branch")
        and auto_merge_branch
    )
    return codex_pool.spawn_turn(
        thread_id=workflow.main_thread_id,
        cwd=workflow.cwd,
        prompt=prompt,
        model=_state_string(workflow, "model") or None,
        stored_model=_state_string(workflow, "model") or None,
        reasoning_effort=_state_string(workflow, "reasoning_effort") or None,
        stored_reasoning_effort=_state_string(workflow, "reasoning_effort") or None,
        base_instructions=_state_string(workflow, "base_instructions") or None,
        developer_instructions=_state_string(workflow, "developer_instructions") or None,
        sandbox_policy=_state_string(workflow, "sandbox_policy") or None,
        approval_mode=_state_string(workflow, "approval_mode") or None,
        enable_memories=_state_bool(workflow, "enable_memories"),
        web_search_mode=system_agents._workflow_web_search_mode(workflow),
        user_message_index=_state_int(workflow, "next_user_message_index"),
        auto_pr_enabled=_state_bool(workflow, "auto_pr_enabled"),
        auto_qa_enabled=auto_qa_enabled,
        auto_merge_to_local_branch=auto_merge_to_local_branch,
        auto_merge_branch=auto_merge_branch if auto_merge_to_local_branch else "",
    )

def _block_spec_critic_workflow(workflow: SystemWorkflow, error: str) -> None:
    def _block(locked: SystemWorkflow) -> bool:
        locked.status = SystemWorkflow.STATUS_BLOCKED
        locked.step = system_agents.STEP_BLOCKED
        locked.state = {**locked.state, "error": error}
        locked.save(update_fields=["status", "step", "state", "updated_at"])
        return True

    engine.claim_workflow_transition(workflow, _block, require_active=False)
    _surface_spec_critic_failure(workflow, error)

def _surface_spec_critic_failure(workflow: SystemWorkflow, error: str) -> None:
    def _mark_surfaced(locked: SystemWorkflow) -> bool:
        locked.state = {**locked.state, "failure_surfaced": True}
        locked.save(update_fields=["state", "updated_at"])
        return True

    claimed = engine.claim_workflow_transition(
        workflow,
        _mark_surfaced,
        guard=lambda locked: locked.state.get("failure_surfaced") is not True,
        require_active=False,
    )
    if not claimed:
        return
    try:
        _spawn_spec_critic_failure_turn(workflow, error)
    except Exception:
        system_agents.logger.exception(
            "failed to surface Spec Critic workflow failure for workflow %s",
            workflow.pk,
        )

def _spawn_spec_critic_failure_turn(
    workflow: SystemWorkflow, error: str
) -> CodexInstance:
    original_prompt = _state_string(workflow, "original_prompt") or "(unknown request)"
    return system_agents._spawn_workflow_turn(
        workflow,
        prompt=(
            "Hitch Spec Critic could not complete pre-implementation analysis.\n\n"
            f"Original user request:\n{original_prompt}\n\n"
            f"Status: {error}\n\n"
            "Tell the user the implementation was not started because the "
            "Spec Critic preflight failed. Keep the explanation concise."
        ),
        purpose=CodexInstance.PURPOSE_SYSTEM_FEEDBACK,
        display_author=system_agents.SPEC_CRITIC_DISPLAY_AUTHOR,
        agent_kind=system_agents.SPEC_CRITIC_WORKFLOW_KIND,
    )

def _spec_critic_clarification_plan(
    workflow: SystemWorkflow,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    questions = _spec_questions_from_outputs(workflow)
    required: list[dict[str, Any]] = []
    safe_defaults: dict[str, str] = {}
    for question in questions:
        safe_default = question.get("safe_default")
        if (
            question.get("allow_safe_default") is True
            and isinstance(safe_default, str)
            and safe_default.strip()
        ):
            safe_defaults[question["id"]] = safe_default.strip()
            continue
        if question.get("required") is True:
            required.append(system_agents._question_for_user_input(question))
    return required, safe_defaults

def _spec_critic_clarification_run(
    workflow: SystemWorkflow,
) -> SystemAgentRun | None:
    return (
        workflow.agent_runs.filter(agent_kind=SPEC_RISK_AGENT_KIND)
        .select_related("instance")
        .order_by("-created_at")
        .first()
    )

def _create_spec_critic_clarification_request(
    workflow: SystemWorkflow,
    run: SystemAgentRun,
    questions: list[dict[str, Any]],
    safe_defaults: dict[str, str],
) -> UserInputRequest:
    recorded_answers = {
        **safe_defaults,
        **_state_dict(workflow, "clarification_answers"),
    }
    input_request = UserInputRequest.objects.create(
        instance=run.instance,
        method=system_agents.SPEC_CRITIC_CLARIFICATION_METHOD,
        params={"questions": questions},
    )
    workflow.state = {
        **workflow.state,
        "clarification_request_id": input_request.pk,
        "clarification_questions": questions,
        "clarification_safe_defaults": safe_defaults,
        "clarification_answers": recorded_answers,
    }
    system_agents._advance_workflow_step(workflow, system_agents.STEP_SPEC_CRITIC_CLARIFYING)
    return input_request

def _cancel_pending_spec_critic_input_requests(
    workflow: SystemWorkflow, reason: str
) -> None:
    instance_ids = list(workflow.agent_runs.values_list("instance_id", flat=True))
    if not instance_ids:
        return
    UserInputRequest.objects.filter(
        instance_id__in=instance_ids,
        method=system_agents.SPEC_CRITIC_CLARIFICATION_METHOD,
        response__isnull=True,
    ).update(
        response={"cancelled": True, "reason": reason},
        responded_at=timezone.now(),
    )
