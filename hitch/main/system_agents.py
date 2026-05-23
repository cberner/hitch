"""Reusable orchestration for Hitch-owned background Codex agents."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from django.db import IntegrityError, models, transaction
from django.utils import timezone
from openai_codex.generated.v2_all import ThreadSource

from hitch.main import codex_pool
from hitch.main.diffs import build_worktree_diff_text
from hitch.main.models import (
    CodexInstance,
    ProposedSession,
    SessionMetadata,
    StandingOrder,
    SystemAgentRun,
    SystemWorkflow,
)

logger = logging.getLogger(__name__)

PR_QA_AGENT_KIND = "pr_qa"
STANDING_ORDER_AGENT_KIND = SystemWorkflow.KIND_STANDING_ORDER_RUN
STANDING_ORDER_JUDGE_AGENT_KIND = "standing_order_judge"
QA_DISPLAY_AUTHOR = "QA agent"
STANDING_ORDER_DISPLAY_AUTHOR = "Standing order agent"
STANDING_ORDER_JUDGE_DISPLAY_AUTHOR = "Standing order judge"
PR_SLASH_DISPLAY_PROMPT = (
    "Rebase on master, clean it up, and then open a PR"
)
QA_SLASH_DISPLAY_PROMPT = (
    "Run the QA agent on the current diff and fix anything it finds"
)
PR_SLASH_PROMPT = (
    f"{PR_SLASH_DISPLAY_PROMPT}. After opening it, poll the PR every 2 minutes "
    "until you have CI status and at least one review signal: code review "
    "comments, a thumbs up emoji on the PR, or an explicit review approval. "
    "On each poll, check whether the PR has merge conflicts. Address CI "
    "failures, review comments, merge conflicts, and any other blocking issues; "
    "push fixes and keep looping until CI, review, and mergeability are all clean. "
    "Stop and report back if any single polling iteration has no results after "
    "30 minutes."
)
SYSTEM_AGENT_APPROVAL_MODE = "auto_review"
QA_WORKFLOW_MAX_ITERATIONS = 3
PR_QA_WORKFLOW_MAX_ITERATIONS = QA_WORKFLOW_MAX_ITERATIONS + 3
STEP_QA_RUNNING = "qa_running"
STEP_FEEDBACK_RUNNING = "feedback_running"
STEP_BLOCKED = "blocked"
STEP_MAX_ITERATIONS_REACHED = "max_iterations_reached"
STEP_QA_APPROVED = "qa_approved"
STEP_PR_PROMPT_SPAWNED = "pr_prompt_spawned"
STEP_STANDING_ORDER_CANDIDATE_RUNNING = "standing_order_candidate_running"
STEP_STANDING_ORDER_JUDGE_RUNNING = "standing_order_judge_running"
STEP_STANDING_ORDER_PROPOSED = "standing_order_proposed"
STEP_STANDING_ORDER_SKIPPED = "standing_order_skipped"

_QA_DESIGN_SYNTHESIS_STATE_KEY = "qa_design_synthesis_gate"
_QA_DESIGN_SYNTHESIS_MIN_CATEGORY_OVERLAP = 2
_QA_DESIGN_SYNTHESIS_RECENT_RUN_LIMIT = 50
_QA_DESIGN_SYNTHESIS_MATCH_LIMIT = 3
_QA_DESIGN_FEEDBACK_SUMMARY_CHARS = 360
_QA_DESIGN_URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s`<>()\[\]]+", re.IGNORECASE)
_QA_DESIGN_FILE_RE = re.compile(
    r"(?<![\w.:/-])(?:[\w.-]+/)+[\w.-]+\.[A-Za-z0-9]+(?=$|[^\w/.-])"
    r"|(?<![\w.:/-])[\w.-]+\.(?:"
    r"bash|c|cc|cfg|conf|cpp|cs|css|cxx|fish|go|h|hpp|html|ini|java|js|json|"
    r"jsx|kt|lock|md|php|py|pyi|rb|rs|rst|sh|sql|svg|svelte|swift|toml|ts|"
    r"tsx|txt|vue|xml|yaml|yml|zsh"
    r")(?=$|[^\w/.-])",
    re.IGNORECASE,
)
_QA_DESIGN_PATH_RE = re.compile(
    r"(?<![\w.:/-])(?:[\w.-]+/)+[\w.-]+(?:\.[A-Za-z0-9]+)?/?(?=$|[^\w/.-])",
    re.IGNORECASE,
)
_QA_DESIGN_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_QA_DESIGN_KEYWORDS_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    "state_lifecycle": (
        "active",
        "cancelled",
        "cleanup",
        "duplicate",
        "generation",
        "in-flight",
        "pending",
        "race",
        "retry",
        "stale",
        "state",
        "superseded",
        "terminal",
        "overwrite",
    ),
    "authority_boundary": (
        "approval",
        "bypass",
        "permission",
        "sandbox",
        "security",
        "token",
    ),
    "persistence_contract": (
        "database",
        "migration",
        "persist",
        "schema",
        "stored",
        "upgrade",
    ),
    "streaming_visibility": (
        "browser",
        "live",
        "reload",
        "render",
        "status",
        "stream",
        "ui",
    ),
}
_STANDING_ORDER_INLINE_HISTORY_CHARS = 10_000
_STANDING_ORDER_TITLE_MAX_LEN = 200
_CONFIDENCE_RANK = {
    StandingOrder.CONFIDENCE_MEDIUM: 1,
    StandingOrder.CONFIDENCE_HIGH: 2,
    StandingOrder.CONFIDENCE_VERY_HIGH: 3,
}

_QA_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["feedback", "lgtm"],
    "properties": {
        "feedback": {"type": "string"},
        "lgtm": {"type": "boolean"},
    },
}

_STANDING_ORDER_CANDIDATE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["proposal", "message"],
    "properties": {
        "proposal": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "required": [
                "title",
                "summary",
                "impact",
                "implementation_direction",
                "relevant_files",
            ],
            "properties": {
                "title": {"type": "string"},
                "summary": {"type": "string"},
                "impact": {"type": "string"},
                "implementation_direction": {"type": "string"},
                "relevant_files": {"type": "array", "items": {"type": "string"}},
            },
        },
        "message": {"type": "string"},
    },
}

_STANDING_ORDER_JUDGE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["confidence", "summary", "rationale"],
    "properties": {
        "confidence": {
            "type": "string",
            "enum": [
                StandingOrder.CONFIDENCE_MEDIUM,
                StandingOrder.CONFIDENCE_HIGH,
                StandingOrder.CONFIDENCE_VERY_HIGH,
            ],
        },
        "summary": {"type": "string"},
        "rationale": {"type": "string"},
    },
}

_CODEX_REVIEW_GUIDANCE = (
    "Apply the same review standards as Codex /review:\n"
    "- Flag only bugs or risks that meaningfully affect correctness, performance, "
    "security, or maintainability.\n"
    "- Each finding must be discrete, actionable, introduced by this diff, and "
    "something the author would likely fix.\n"
    "- Do not rely on unstated assumptions, speculative downstream breakage, or "
    "intentional behavior changes.\n"
    "- Ignore trivial style unless it obscures meaning or violates documented "
    "standards.\n"
    "- Do not stop at the first issue; keep reviewing until every qualifying "
    "finding is listed.\n"
    "- Prioritize findings as [P0], [P1], [P2], or [P3], using P0 only for "
    "universal release-blocking issues.\n"
    "- For each finding, include the shortest useful file/line reference that "
    "overlaps the diff and a one-paragraph explanation of why the issue matters.\n"
    "- If there are no qualifying findings, say that clearly rather than "
    "inventing nits.\n"
)


def start_pr_qa_workflow(
    *,
    main_thread_id: str,
    cwd: str,
    sandbox_policy: str | None,
    approval_mode: str | None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    base_instructions: str | None = None,
    developer_instructions: str | None = None,
    enable_memories: bool = False,
    initial_user_message_index: int = 0,
    open_pr_on_lgtm: bool = True,
) -> SystemWorkflow:
    """Start a QA workflow before optionally running the work-agent PR prompt."""
    try:
        with transaction.atomic():
            workflow = SystemWorkflow.objects.create(
                kind=SystemWorkflow.KIND_PR_QA,
                main_thread_id=main_thread_id,
                cwd=cwd,
                status=SystemWorkflow.STATUS_RUNNING,
                step=STEP_QA_RUNNING,
                max_iterations=(
                    PR_QA_WORKFLOW_MAX_ITERATIONS
                    if open_pr_on_lgtm
                    else QA_WORKFLOW_MAX_ITERATIONS
                ),
                state={
                    "pr_prompt": PR_SLASH_PROMPT,
                    "sandbox_policy": sandbox_policy or "",
                    "approval_mode": approval_mode or "",
                    "model": model or "",
                    "reasoning_effort": reasoning_effort or "",
                    "base_instructions": base_instructions or "",
                    "developer_instructions": developer_instructions or "",
                    "enable_memories": enable_memories,
                    "next_user_message_index": max(initial_user_message_index, 0),
                    "open_pr_on_lgtm": open_pr_on_lgtm,
                },
            )
    except IntegrityError:
        existing_workflow = SystemWorkflow.objects.filter(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id=main_thread_id,
            status=SystemWorkflow.STATUS_RUNNING,
        ).first()
        if existing_workflow is None:
            raise
        return existing_workflow

    try:
        _spawn_pr_qa_run(workflow)
    except Exception as exc:
        _block_workflow(workflow, f"failed to start QA agent: {exc!r}")
    return workflow


def start_standing_order_workflow(*, standing_order: StandingOrder) -> SystemWorkflow:
    standing_order = (
        StandingOrder.objects.select_related("project")
        .filter(pk=standing_order.pk)
        .get()
    )
    main_thread_id = _standing_order_main_thread_id(standing_order.pk)
    try:
        with transaction.atomic():
            workflow = SystemWorkflow.objects.create(
                kind=STANDING_ORDER_AGENT_KIND,
                main_thread_id=main_thread_id,
                cwd=standing_order.project.repo_path,
                status=SystemWorkflow.STATUS_RUNNING,
                step=STEP_STANDING_ORDER_CANDIDATE_RUNNING,
                state={"standing_order_id": standing_order.pk},
            )
    except IntegrityError:
        existing_workflow = SystemWorkflow.objects.filter(
            kind=STANDING_ORDER_AGENT_KIND,
            main_thread_id=main_thread_id,
            status=SystemWorkflow.STATUS_RUNNING,
        ).first()
        if existing_workflow is None:
            raise
        return existing_workflow

    try:
        _spawn_standing_order_candidate_run(workflow, standing_order)
    except Exception as exc:
        _block_workflow(
            workflow,
            f"failed to start standing order agent: {exc!r}",
            surface_to_thread=False,
        )
    return workflow


def hidden_thread_ids() -> set[str]:
    hidden_ids = set(
        SystemAgentRun.objects.exclude(thread_id="")
        .exclude(agent_kind="demo")
        .values_list("thread_id", flat=True)
        .distinct()
    )
    visible_ids = set(
        ProposedSession.objects.filter(
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            candidate_session__isnull=False,
            accepted_session=models.F("candidate_session"),
        ).values_list("candidate_session__thread_id", flat=True)
    )
    return hidden_ids - visible_ids


def active_workflow_for_thread(main_thread_id: str) -> SystemWorkflow | None:
    return (
        SystemWorkflow.objects.filter(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id=main_thread_id,
            status=SystemWorkflow.STATUS_RUNNING,
        )
        .order_by("-created_at")
        .first()
    )


def stop_active_workflow(main_thread_id: str) -> bool:
    workflow = active_workflow_for_thread(main_thread_id)
    if workflow is None:
        return False
    run = (
        workflow.agent_runs.filter(status=SystemAgentRun.STATUS_RUNNING)
        .select_related("instance")
        .order_by("-created_at")
        .first()
    )
    if run is None:
        return False
    interrupted = codex_pool.interrupt_instance(
        run.instance_id, expected_thread_id=run.thread_id
    )
    if interrupted is None:
        return False
    run.status = SystemAgentRun.STATUS_FAILED
    run.error = "QA workflow stopped by user"
    run.save(update_fields=["status", "error", "updated_at"])
    _block_workflow(workflow, "QA workflow stopped by user")
    return True


def on_codex_instance_finished(instance: CodexInstance) -> None:
    """Route a terminal worker to its owning system workflow, if any."""
    if instance.purpose == CodexInstance.PURPOSE_SYSTEM_AGENT:
        _handle_system_agent_finished(instance)
        return
    if instance.purpose == CodexInstance.PURPOSE_SYSTEM_FEEDBACK:
        _handle_system_feedback_finished(instance)
        return
    _maybe_start_auto_pr_workflow(instance)


def _maybe_start_auto_pr_workflow(instance: CodexInstance) -> None:
    if (
        instance.purpose != CodexInstance.PURPOSE_USER
        or instance.workflow_id is not None
        or not instance.auto_pr_enabled
        or instance.plan_mode
        or instance.status != CodexInstance.STATUS_COMPLETED
    ):
        return
    claimed = CodexInstance.objects.filter(
        pk=instance.pk,
        auto_pr_triggered_at__isnull=True,
    ).update(auto_pr_triggered_at=timezone.now())
    if not claimed:
        return
    try:
        start_pr_qa_workflow(
            main_thread_id=instance.thread_id,
            cwd=instance.cwd,
            sandbox_policy=instance.sandbox_policy or None,
            approval_mode=instance.approval_mode or SYSTEM_AGENT_APPROVAL_MODE,
            model=instance.model or None,
            reasoning_effort=instance.reasoning_effort or None,
            base_instructions=instance.base_instructions or None,
            developer_instructions=instance.developer_instructions or None,
            enable_memories=instance.enable_memories,
            initial_user_message_index=(instance.user_message_index or 0) + 1,
        )
    except Exception:
        CodexInstance.objects.filter(pk=instance.pk).update(auto_pr_triggered_at=None)
        raise


def _handle_system_agent_finished(instance: CodexInstance) -> None:
    run = _system_agent_run_for_instance(instance)
    if run is None:
        return
    if run.status in (SystemAgentRun.STATUS_COMPLETED, SystemAgentRun.STATUS_FAILED):
        return
    workflow = run.workflow
    if workflow.kind == STANDING_ORDER_AGENT_KIND:
        _handle_standing_order_agent_finished(instance, run, workflow)
        return
    if workflow.kind != SystemWorkflow.KIND_PR_QA:
        _fail_unsupported_system_agent_run(run, workflow)
        return
    if (
        workflow.status != SystemWorkflow.STATUS_RUNNING
        or workflow.step != STEP_QA_RUNNING
    ):
        return
    if instance.status != CodexInstance.STATUS_COMPLETED:
        _fail_run_and_block_workflow(run, f"QA worker failed: {instance.error}")
        return

    raw_output = _final_agent_text(instance.events_path)
    parsed = _parse_qa_output(raw_output)
    if parsed is None:
        _fail_run_and_block_workflow(run, "QA output was not valid JSON", raw_output)
        return

    run.status = SystemAgentRun.STATUS_COMPLETED
    run.output = parsed
    run.raw_output = raw_output
    run.save(update_fields=["status", "output", "raw_output", "updated_at"])

    feedback = parsed["feedback"].strip()
    lgtm = parsed["lgtm"]
    workflow.state = {**workflow.state, "last_feedback": feedback}
    if lgtm:
        if workflow.state.get("open_pr_on_lgtm", True) is not True:
            workflow.status = SystemWorkflow.STATUS_COMPLETED
            workflow.step = STEP_QA_APPROVED
            workflow.save(update_fields=["status", "step", "state", "updated_at"])
            return
        try:
            _spawn_pr_prompt(workflow)
        except Exception as exc:
            _block_workflow(workflow, f"failed to start PR prompt: {exc!r}")
            return
        workflow.status = SystemWorkflow.STATUS_COMPLETED
        workflow.step = STEP_PR_PROMPT_SPAWNED
        workflow.save(update_fields=["status", "step", "state", "updated_at"])
        return

    if workflow.iteration >= workflow.max_iterations:
        workflow.status = SystemWorkflow.STATUS_MAX_ITERATIONS_REACHED
        workflow.step = STEP_MAX_ITERATIONS_REACHED
        workflow.save(update_fields=["status", "step", "state", "updated_at"])
        _surface_workflow_failure(
            workflow,
            (
                "QA agent reached the maximum feedback loop count without "
                "approving the diff."
            ),
        )
        return

    synthesis_gate = _maybe_build_qa_design_synthesis_gate(
        workflow, feedback, current_run_id=run.pk
    )
    if synthesis_gate is not None:
        workflow.state = {
            **workflow.state,
            _QA_DESIGN_SYNTHESIS_STATE_KEY: synthesis_gate,
        }
    workflow.iteration += 1
    workflow.step = STEP_FEEDBACK_RUNNING
    workflow.save(update_fields=["iteration", "step", "state", "updated_at"])
    try:
        _spawn_qa_feedback_turn(
            workflow, feedback, synthesis_gate=synthesis_gate
        )
    except Exception as exc:
        _block_workflow(workflow, f"failed to start QA feedback turn: {exc!r}")


def _handle_system_feedback_finished(instance: CodexInstance) -> None:
    if instance.status != CodexInstance.STATUS_COMPLETED:
        workflow = _workflow_for_instance(instance)
        if workflow is not None:
            _block_workflow(workflow, f"QA feedback worker failed: {instance.error}")
        return
    workflow = _workflow_for_instance(instance)
    if workflow is None or workflow.kind != SystemWorkflow.KIND_PR_QA:
        return
    if workflow.status != SystemWorkflow.STATUS_RUNNING or workflow.step != STEP_FEEDBACK_RUNNING:
        return
    workflow.step = STEP_QA_RUNNING
    workflow.save(update_fields=["step", "updated_at"])
    try:
        _spawn_pr_qa_run(workflow)
    except Exception as exc:
        _block_workflow(workflow, f"failed to restart QA agent: {exc!r}")


def _fail_unsupported_system_agent_run(
    run: SystemAgentRun, workflow: SystemWorkflow
) -> None:
    error = f"system workflow kind {workflow.kind!r} is no longer supported"
    if workflow.status == SystemWorkflow.STATUS_RUNNING:
        _fail_run_and_block_workflow(run, error, surface_to_thread=False)
        return
    run.status = SystemAgentRun.STATUS_FAILED
    run.error = error
    run.save(update_fields=["status", "error", "updated_at"])


def _handle_standing_order_agent_finished(
    instance: CodexInstance, run: SystemAgentRun, workflow: SystemWorkflow
) -> None:
    if workflow.status != SystemWorkflow.STATUS_RUNNING:
        return
    if instance.status != CodexInstance.STATUS_COMPLETED:
        _fail_run_and_block_workflow(
            run,
            f"standing order worker failed: {instance.error}",
            surface_to_thread=False,
        )
        return

    standing_order = (
        StandingOrder.objects.select_related("project")
        .filter(pk=_state_int(workflow, "standing_order_id"))
        .first()
    )
    if standing_order is None:
        _fail_run_and_block_workflow(
            run,
            "standing order no longer exists",
            surface_to_thread=False,
        )
        return

    raw_output = _final_agent_text(instance.events_path)
    if workflow.step == STEP_STANDING_ORDER_CANDIDATE_RUNNING:
        candidate_output = _parse_standing_order_candidate_output(raw_output)
        if candidate_output is None:
            _fail_run_and_block_workflow(
                run,
                "standing order candidate output was not valid JSON",
                raw_output,
                surface_to_thread=False,
            )
            return
        run.status = SystemAgentRun.STATUS_COMPLETED
        run.output = candidate_output
        run.raw_output = raw_output
        run.save(update_fields=["status", "output", "raw_output", "updated_at"])
        if candidate_output["proposal"] is None:
            message = str(candidate_output["message"])
            ProposedSession.objects.create(
                project=standing_order.project,
                standing_order=standing_order,
                source_workflow=workflow,
                title=f"No proposal from {standing_order.title}"[
                    :_STANDING_ORDER_TITLE_MAX_LEN
                ],
                inbox_kind=ProposedSession.INBOX_KIND_NOTICE,
                summary=message,
                candidate_session=_session_metadata_from_state(
                    workflow, "candidate_session_id"
                ),
            )
            workflow.step = STEP_STANDING_ORDER_SKIPPED
            workflow.status = SystemWorkflow.STATUS_COMPLETED
            workflow.state = {**workflow.state, "candidate": candidate_output}
            workflow.save(update_fields=["status", "step", "state", "updated_at"])
            return
        candidate = candidate_output["proposal"]
        workflow.step = STEP_STANDING_ORDER_JUDGE_RUNNING
        workflow.state = {**workflow.state, "candidate": candidate}
        workflow.save(update_fields=["step", "state", "updated_at"])
        try:
            _spawn_standing_order_judge_run(workflow, standing_order, candidate)
        except Exception as exc:
            _block_workflow(
                workflow,
                f"failed to start standing order judge: {exc!r}",
                surface_to_thread=False,
            )
        return

    if workflow.step != STEP_STANDING_ORDER_JUDGE_RUNNING:
        return
    judgment = _parse_standing_order_judge_output(raw_output)
    if judgment is None:
        _fail_run_and_block_workflow(
            run,
            "standing order judge output was not valid JSON",
            raw_output,
            surface_to_thread=False,
        )
        return
    run.status = SystemAgentRun.STATUS_COMPLETED
    run.output = judgment
    run.raw_output = raw_output
    run.save(update_fields=["status", "output", "raw_output", "updated_at"])

    candidate = workflow.state.get("candidate")
    if not isinstance(candidate, dict):
        candidate = {}
    if _confidence_meets_threshold(
        judgment["confidence"], standing_order.confidence_threshold
    ):
        ProposedSession.objects.create(
            project=standing_order.project,
            standing_order=standing_order,
            source_workflow=workflow,
            title=str(candidate.get("title", standing_order.title))[
                :_STANDING_ORDER_TITLE_MAX_LEN
            ],
            summary=judgment["summary"],
            prompt=_standing_order_proposed_session_prompt(
                standing_order, candidate, judgment
            ),
            confidence=judgment["confidence"],
            relevant_files=_string_list(candidate.get("relevant_files")),
            candidate_session=_session_metadata_from_state(
                workflow, "candidate_session_id"
            ),
            judge_session=_session_metadata_from_state(workflow, "judge_session_id"),
        )
        workflow.step = STEP_STANDING_ORDER_PROPOSED
    else:
        workflow.step = STEP_STANDING_ORDER_SKIPPED
    workflow.status = SystemWorkflow.STATUS_COMPLETED
    workflow.state = {**workflow.state, "judgment": judgment}
    workflow.save(update_fields=["status", "step", "state", "updated_at"])


def _spawn_pr_qa_run(workflow: SystemWorkflow) -> SystemAgentRun:
    diff_text = build_worktree_diff_text(workflow.cwd)
    prompt = _qa_prompt(workflow.cwd, diff_text)
    instance = codex_pool.spawn_new_session(
        cwd=workflow.cwd,
        prompt=prompt,
        base_instructions=_state_string(workflow, "base_instructions") or None,
        developer_instructions=_state_string(workflow, "developer_instructions") or None,
        model=_state_string(workflow, "model") or None,
        reasoning_effort=_state_string(workflow, "reasoning_effort") or None,
        approval_mode=SYSTEM_AGENT_APPROVAL_MODE,
        sandbox_policy=_state_string(workflow, "sandbox_policy") or None,
        enable_memories=_state_bool(workflow, "enable_memories"),
        thread_source=ThreadSource.subagent,
        purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
        workflow_id=workflow.pk,
        agent_kind=PR_QA_AGENT_KIND,
        display_author=QA_DISPLAY_AUTHOR,
        output_schema=_QA_OUTPUT_SCHEMA,
    )
    run, _created = SystemAgentRun.objects.get_or_create(
        instance=instance,
        defaults={
            "workflow": workflow,
            "agent_kind": PR_QA_AGENT_KIND,
            "thread_id": instance.thread_id,
            "status": SystemAgentRun.STATUS_RUNNING,
            "input": {"cwd": workflow.cwd, "diff_chars": len(diff_text)},
        },
    )
    return run


def _spawn_standing_order_candidate_run(
    workflow: SystemWorkflow, standing_order: StandingOrder
) -> SystemAgentRun:
    prompt = _standing_order_candidate_prompt(workflow, standing_order)
    instance = codex_pool.spawn_new_session(
        cwd=workflow.cwd,
        prompt=prompt,
        approval_mode=SYSTEM_AGENT_APPROVAL_MODE,
        thread_source=ThreadSource.subagent,
        purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
        workflow_id=workflow.pk,
        agent_kind=STANDING_ORDER_AGENT_KIND,
        display_author=STANDING_ORDER_DISPLAY_AUTHOR,
        output_schema=_STANDING_ORDER_CANDIDATE_OUTPUT_SCHEMA,
    )
    metadata, _created = SessionMetadata.objects.update_or_create(
        thread_id=instance.thread_id,
        defaults={
            "cwd": workflow.cwd,
            "project": standing_order.project,
            "project_cleared": False,
            "auto_pr_enabled": False,
        },
    )
    workflow.state = {**workflow.state, "candidate_session_id": metadata.pk}
    workflow.save(update_fields=["state", "updated_at"])
    run, _created = SystemAgentRun.objects.get_or_create(
        instance=instance,
        defaults={
            "workflow": workflow,
            "agent_kind": STANDING_ORDER_AGENT_KIND,
            "thread_id": instance.thread_id,
            "status": SystemAgentRun.STATUS_RUNNING,
            "input": {
                "cwd": workflow.cwd,
                "standing_order_id": standing_order.pk,
            },
        },
    )
    return run


def _spawn_standing_order_judge_run(
    workflow: SystemWorkflow, standing_order: StandingOrder, candidate: dict[str, Any]
) -> SystemAgentRun:
    prompt, history_files = _standing_order_judge_prompt(
        workflow, standing_order, candidate
    )
    if history_files:
        workflow.state = {**workflow.state, "history_files": history_files}
        workflow.save(update_fields=["state", "updated_at"])
    instance = codex_pool.spawn_new_session(
        cwd=workflow.cwd,
        prompt=prompt,
        approval_mode=SYSTEM_AGENT_APPROVAL_MODE,
        thread_source=ThreadSource.subagent,
        purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
        workflow_id=workflow.pk,
        agent_kind=STANDING_ORDER_JUDGE_AGENT_KIND,
        display_author=STANDING_ORDER_JUDGE_DISPLAY_AUTHOR,
        output_schema=_STANDING_ORDER_JUDGE_OUTPUT_SCHEMA,
    )
    metadata, _created = SessionMetadata.objects.update_or_create(
        thread_id=instance.thread_id,
        defaults={
            "cwd": workflow.cwd,
            "project": standing_order.project,
            "project_cleared": False,
            "auto_pr_enabled": False,
        },
    )
    workflow.state = {**workflow.state, "judge_session_id": metadata.pk}
    workflow.save(update_fields=["state", "updated_at"])
    run, _created = SystemAgentRun.objects.get_or_create(
        instance=instance,
        defaults={
            "workflow": workflow,
            "agent_kind": STANDING_ORDER_JUDGE_AGENT_KIND,
            "thread_id": instance.thread_id,
            "status": SystemAgentRun.STATUS_RUNNING,
            "input": {
                "cwd": workflow.cwd,
                "standing_order_id": standing_order.pk,
                "candidate": candidate,
                "history_files": history_files,
            },
        },
    )
    return run


def _spawn_qa_feedback_turn(
    workflow: SystemWorkflow,
    feedback: str,
    *,
    synthesis_gate: dict[str, Any] | None = None,
) -> CodexInstance:
    return _spawn_workflow_turn(
        workflow,
        prompt=(
            _qa_design_synthesis_feedback_prompt(feedback, synthesis_gate)
            if synthesis_gate is not None
            else f"Feedback from Hitch QA agent:\n\n{feedback}"
        ),
        purpose=CodexInstance.PURPOSE_SYSTEM_FEEDBACK,
        display_author=QA_DISPLAY_AUTHOR,
    )


def _spawn_pr_prompt(workflow: SystemWorkflow) -> CodexInstance:
    return _spawn_workflow_turn(
        workflow,
        prompt=_state_string(workflow, "pr_prompt") or PR_SLASH_PROMPT,
    )


def _spawn_workflow_failure_turn(
    workflow: SystemWorkflow, error: str
) -> CodexInstance:
    return _spawn_workflow_turn(
        workflow,
        prompt=(
            "Hitch QA agent could not complete the PR workflow.\n\n"
            f"Status: {error}\n\n"
            "Tell the user the PR workflow needs attention before continuing."
        ),
        purpose=CodexInstance.PURPOSE_SYSTEM_FEEDBACK,
        display_author=QA_DISPLAY_AUTHOR,
    )


def _spawn_workflow_turn(
    workflow: SystemWorkflow,
    *,
    prompt: str,
    purpose: str = CodexInstance.PURPOSE_USER,
    display_author: str = "",
) -> CodexInstance:
    user_message_index = _state_int(workflow, "next_user_message_index")
    instance = codex_pool.spawn_turn(
        thread_id=workflow.main_thread_id,
        cwd=workflow.cwd,
        prompt=prompt,
        model=_state_string(workflow, "model") or None,
        reasoning_effort=_state_string(workflow, "reasoning_effort") or None,
        base_instructions=_state_string(workflow, "base_instructions") or None,
        developer_instructions=_state_string(workflow, "developer_instructions") or None,
        sandbox_policy=_state_string(workflow, "sandbox_policy") or None,
        approval_mode=_state_string(workflow, "approval_mode") or None,
        enable_memories=_state_bool(workflow, "enable_memories"),
        purpose=purpose,
        workflow_id=workflow.pk,
        agent_kind=PR_QA_AGENT_KIND if purpose != CodexInstance.PURPOSE_USER else "",
        display_author=display_author,
        user_message_index=user_message_index,
    )
    workflow.state = {
        **workflow.state,
        "next_user_message_index": user_message_index + 1,
    }
    workflow.save(update_fields=["state", "updated_at"])
    return instance


def _qa_prompt(cwd: str, diff_text: str) -> str:
    diff = diff_text or "(No current worktree diff was detected.)"
    return (
        "You are Hitch's QA agent for a PR workflow.\n\n"
        "Thoroughly review the current code diff before the PR agent runs its final "
        "cleanup/open-PR pass.\n\n"
        f"{_CODEX_REVIEW_GUIDANCE}\n"
        "Also do your own manual QA: if there is an interactive interface related "
        "to the diff, manually test it out and include concrete failures or gaps in "
        "your feedback. For browser QA, run `just qa-browser-setup` if Playwright "
        "or Chromium is missing, then use Playwright/Chromium to exercise the "
        "affected UI. If browser setup still fails, include that concrete setup "
        "failure in your feedback.\n\n"
        "Set lgtm to false when there are substantive findings, missing tests, or "
        "manual-QA failures the work agent should fix. Set lgtm to true only when "
        "the diff is ready for the PR agent to continue.\n\n"
        f"Repository cwd: {cwd}\n\n"
        "Current diff:\n"
        "```diff\n"
        f"{diff}\n"
        "```\n\n"
        "Return only JSON matching this shape: "
        '{"feedback": string, "lgtm": boolean}. Put the prioritized review '
        "findings, manual-QA results, or a clear no-findings statement in feedback."
    )


def _maybe_build_qa_design_synthesis_gate(
    workflow: SystemWorkflow, feedback: str, *, current_run_id: int
) -> dict[str, Any] | None:
    if workflow.state.get(_QA_DESIGN_SYNTHESIS_STATE_KEY):
        return None
    current_signal = _qa_design_feedback_signal(feedback)
    if not current_signal["categories"]:
        return None

    matches: list[dict[str, Any]] = []
    recurring_categories: set[str] = set()
    recurring_files: set[str] = set()
    recent_runs = (
        SystemAgentRun.objects.filter(
            workflow__kind=SystemWorkflow.KIND_PR_QA,
            workflow__cwd=workflow.cwd,
            agent_kind=PR_QA_AGENT_KIND,
            status=SystemAgentRun.STATUS_COMPLETED,
        )
        .exclude(pk=current_run_id)
        .select_related("workflow")
        .order_by("-created_at")[:_QA_DESIGN_SYNTHESIS_RECENT_RUN_LIMIT]
    )
    for prior_run in recent_runs:
        prior_feedback = _qa_feedback_from_run(prior_run)
        if not prior_feedback:
            continue
        prior_signal = _qa_design_feedback_signal(prior_feedback)
        category_overlap = current_signal["categories"] & prior_signal["categories"]
        file_overlap = current_signal["files"] & prior_signal["files"]
        same_workflow = prior_run.workflow_id == workflow.pk
        if not _qa_design_signals_match(
            category_overlap=category_overlap,
            file_overlap=file_overlap,
            same_workflow=same_workflow,
        ):
            continue
        recurring_categories.update(category_overlap)
        recurring_files.update(file_overlap)
        matches.append(
            {
                "workflow_id": prior_run.workflow_id,
                "run_id": prior_run.pk,
                "same_workflow": same_workflow,
                "categories": sorted(category_overlap),
                "files": sorted(file_overlap),
                "feedback": _summarize_qa_feedback(prior_feedback),
            }
        )
        if len(matches) >= _QA_DESIGN_SYNTHESIS_MATCH_LIMIT:
            break

    if not matches:
        return None
    if (
        workflow.iteration < 1
        and len(recurring_categories) < _QA_DESIGN_SYNTHESIS_MIN_CATEGORY_OVERLAP
    ):
        return None
    return {
        "triggered_at_iteration": workflow.iteration + 1,
        "current_categories": sorted(current_signal["categories"]),
        "current_files": sorted(current_signal["files"]),
        "recurring_categories": sorted(recurring_categories),
        "recurring_files": sorted(recurring_files),
        "matches": matches,
    }


def _qa_design_signals_match(
    *,
    category_overlap: set[str],
    file_overlap: set[str],
    same_workflow: bool,
) -> bool:
    if file_overlap and category_overlap:
        return True
    if len(category_overlap) >= _QA_DESIGN_SYNTHESIS_MIN_CATEGORY_OVERLAP:
        return True
    return same_workflow and bool(category_overlap)


def _qa_design_feedback_signal(feedback: str) -> dict[str, set[str]]:
    feedback_without_urls = _QA_DESIGN_URL_RE.sub(" ", feedback)
    feedback_without_paths = _QA_DESIGN_PATH_RE.sub(" ", feedback_without_urls)
    feedback_without_paths = _QA_DESIGN_FILE_RE.sub(" ", feedback_without_paths)
    normalized = feedback_without_paths.lower()
    tokens = set(_QA_DESIGN_TOKEN_RE.findall(normalized))
    categories = {
        category
        for category, keywords in _QA_DESIGN_KEYWORDS_BY_CATEGORY.items()
        if any(keyword in tokens for keyword in keywords)
    }
    files = {
        match.strip("`.,:;()[]")
        for match in _QA_DESIGN_FILE_RE.findall(feedback_without_urls)
    }
    return {"categories": categories, "files": files}


def _qa_feedback_from_run(run: SystemAgentRun) -> str:
    output = run.output
    if isinstance(output, dict):
        feedback = output.get("feedback")
        if output.get("lgtm") is False and isinstance(feedback, str):
            return feedback
    parsed = _parse_qa_output(run.raw_output)
    if parsed is None or parsed["lgtm"] is not False:
        return ""
    feedback = parsed["feedback"]
    return feedback if isinstance(feedback, str) else ""


def _summarize_qa_feedback(feedback: str) -> str:
    summary = " ".join(feedback.split())
    if len(summary) <= _QA_DESIGN_FEEDBACK_SUMMARY_CHARS:
        return summary
    return f"{summary[: _QA_DESIGN_FEEDBACK_SUMMARY_CHARS - 3].rstrip()}..."


def _qa_design_synthesis_feedback_prompt(
    feedback: str, synthesis_gate: dict[str, Any]
) -> str:
    categories = ", ".join(synthesis_gate.get("recurring_categories") or [])
    files = ", ".join(synthesis_gate.get("recurring_files") or [])
    evidence_lines = []
    for match in synthesis_gate.get("matches") or []:
        if not isinstance(match, dict):
            continue
        evidence_lines.append(
            "- "
            f"workflow {match.get('workflow_id')}, run {match.get('run_id')}: "
            f"{match.get('feedback', '')}"
        )
    evidence = "\n".join(evidence_lines) or "- No prior feedback summary available."
    return (
        "QA Design Synthesis Gate\n\n"
        "Hitch QA is seeing recurring design-level feedback, not just isolated "
        "defects. Before applying another tactical fix, pause and simplify the "
        "underlying design.\n\n"
        f"Recurring categories: {categories or 'unspecified'}\n"
        f"Recurring files: {files or 'none detected'}\n\n"
        "Prior related QA feedback:\n"
        f"{evidence}\n\n"
        "Current QA feedback:\n\n"
        f"{feedback}\n\n"
        "First identify the shared invariant, ownership boundary, or lifecycle "
        "rule that keeps breaking. Then implement the smallest coherent design "
        "change that makes that rule explicit and removes the need for another "
        "narrow fixup. Keep the diff focused, preserve existing behavior that is "
        "not implicated by the recurring feedback, and run the relevant tests."
    )


def _standing_order_candidate_prompt(
    workflow: SystemWorkflow, standing_order: StandingOrder
) -> str:
    ambition = _standing_order_ambition_guidance(standing_order)
    return (
        "You are Hitch's standing order agent.\n\n"
        "Thoroughly analyze the codebase and find one way to make "
        f"{ambition.candidate_progress} toward the standing order goal. "
        "Do not make code changes. "
        "Focus on a concrete session that a user could accept and continue from.\n\n"
        f"Repository cwd: {workflow.cwd}\n"
        f"Standing order title: {standing_order.title}\n\n"
        "Standing order goal:\n"
        f"{standing_order.goal}\n\n"
        "Return only JSON matching this shape: "
        '{"proposal": {"title": string, "summary": string, "impact": string, '
        '"implementation_direction": string, "relevant_files": [string]} | null, '
        '"message": string}. If you find a concrete proposal, put it in '
        '"proposal" and leave "message" empty. If you find nothing worth '
        'proposing, set "proposal" to null and put a concise user-facing '
        'explanation in "message". The title should be concise. The summary '
        "should explain the proposed session. Impact should describe the likely "
        "user-visible or engineering benefit. Implementation direction should "
        "be specific enough for the user to continue the work in this session. "
        f"{ambition.candidate_instruction}"
    )


def _standing_order_proposed_session_prompt(
    standing_order: StandingOrder, candidate: dict[str, Any], judgment: dict[str, str]
) -> str:
    parts = [
        "Go ahead and implement this proposed session.",
        "",
        f"Standing order: {standing_order.title}",
    ]
    if standing_order.goal:
        parts.extend(["", f"Standing order goal:\n{standing_order.goal}"])
    title = str(candidate.get("title", standing_order.title)).strip()
    if title:
        parts.extend(["", f"Proposed session: {title}"])
    summary = judgment.get("summary", "").strip()
    if summary:
        parts.extend(["", f"Summary:\n{summary}"])
    implementation_direction = candidate.get("implementation_direction")
    if isinstance(implementation_direction, str) and implementation_direction.strip():
        parts.extend(
            [
                "",
                f"Implementation guidance:\n{implementation_direction.strip()}",
            ]
        )
    files = _string_list(candidate.get("relevant_files"))
    if files:
        parts.extend(["", "Relevant files:", *[f"- {file}" for file in files]])
    return "\n".join(parts)


@dataclass(frozen=True)
class _StandingOrderAmbitionGuidance:
    candidate_progress: str
    candidate_instruction: str
    judge_progress: str
    judge_instruction: str


def _standing_order_ambition_guidance(
    standing_order: StandingOrder,
) -> _StandingOrderAmbitionGuidance:
    ambitions = {value for value, _label in StandingOrder.AMBITION_CHOICES}
    ambition = (
        standing_order.ambition
        if standing_order.ambition in ambitions
        else StandingOrder.AMBITION_INCREMENTAL
    )
    if ambition == StandingOrder.AMBITION_YOLO:
        return _StandingOrderAmbitionGuidance(
            candidate_progress="bold, high-leverage progress",
            candidate_instruction=(
                "For YOLO ambition, prefer a substantial session with clear "
                "upside over a cautious cleanup."
            ),
            judge_progress="bold, high-leverage progress",
            judge_instruction=(
                "For YOLO ambition, confidence should reflect whether this "
                "specific session is substantial and high-upside, not merely "
                "a small cleanup."
            ),
        )
    return _StandingOrderAmbitionGuidance(
        candidate_progress=f"{ambition} progress",
        candidate_instruction="",
        judge_progress=f"{ambition} progress",
        judge_instruction=(
            "Confidence should reflect whether this specific session is "
            "likely to advance the goal incrementally."
        ),
    )


def _standing_order_judge_prompt(
    workflow: SystemWorkflow, standing_order: StandingOrder, candidate: dict[str, Any]
) -> tuple[str, list[str]]:
    history_sections = _standing_order_history_sections(standing_order)
    inline_history, overflow_history = _split_standing_order_history(history_sections)
    history_files = _write_standing_order_history_files(workflow, overflow_history)
    history_file_text = (
        "\n".join(f"- {path}" for path in history_files) if history_files else "(none)"
    )
    ambition = _standing_order_ambition_guidance(standing_order)
    candidate_text = json.dumps(candidate, indent=2, sort_keys=True)
    candidate_session = _session_metadata_from_state(workflow, "candidate_session_id")
    candidate_thread_id = (
        candidate_session.thread_id if candidate_session is not None else "(unknown)"
    )
    return (
        "You are Hitch's standing order confidence judge.\n\n"
        "Judge whether the candidate session is likely to make meaningful "
        f"{ambition.judge_progress} toward the standing order goal. "
        "Use the standing order's "
        "accepted and rejected proposal history to calibrate your judgment. "
        "Do not reward broad or vague ideas; confidence should reflect whether "
        f"the proposal is concrete and well-scoped. {ambition.judge_instruction}\n\n"
        f"Repository cwd: {workflow.cwd}\n"
        f"Standing order title: {standing_order.title}\n"
        f"Confidence threshold: {standing_order.confidence_threshold}\n\n"
        "Standing order goal:\n"
        f"{standing_order.goal}\n\n"
        "Candidate session JSON:\n"
        f"Candidate session ID: {candidate_thread_id}\n"
        f"{candidate_text}\n\n"
        "Accepted/rejected proposal history included inline:\n"
        f"{inline_history or '(none)'}\n\n"
        "Additional history files:\n"
        f"{history_file_text}\n\n"
        "Return only JSON matching this shape: "
        '{"confidence": "medium" | "high" | "very_high", '
        '"summary": string, "rationale": string}. Summary is shown to the user '
        "in the inbox and should explain the expected impact."
    ), history_files


def _standing_order_history_sections(standing_order: StandingOrder) -> list[str]:
    proposals = (
        standing_order.proposed_sessions.filter(
            inbox_kind=ProposedSession.INBOX_KIND_PROPOSAL
        )
        .exclude(outcome_status=ProposedSession.OUTCOME_UNSET)
        .select_related("candidate_session")
        .order_by("-updated_at", "-id")[:50]
    )
    return [_format_proposed_session_context(proposal) for proposal in proposals]


def _format_proposed_session_context(proposal: ProposedSession) -> str:
    files = _string_list(proposal.relevant_files)
    candidate_id = (
        proposal.candidate_session.thread_id if proposal.candidate_session else "(none)"
    )
    notes_label = (
        "Reject reason"
        if proposal.outcome_status == ProposedSession.OUTCOME_REJECTED
        else "Outcome notes"
    )
    return (
        f"ProposedSession ID: {proposal.pk}\n"
        f"Candidate session ID: {candidate_id}\n"
        f"Title: {proposal.title}\n"
        f"Confidence: {proposal.confidence}\n"
        f"Summary: {proposal.summary or '(none)'}\n"
        f"Prompt: {proposal.prompt or '(none)'}\n"
        f"Relevant files: {', '.join(files) if files else '(none)'}\n"
        f"Outcome status: {proposal.outcome_status}\n"
        f"{notes_label}: {proposal.outcome_notes or '(none)'}"
    )


def _split_standing_order_history(sections: list[str]) -> tuple[str, list[str]]:
    inline_parts: list[str] = []
    overflow: list[str] = []
    used_chars = 0
    for section in sections:
        section_chars = len(section) + 2
        if used_chars + section_chars <= _STANDING_ORDER_INLINE_HISTORY_CHARS:
            inline_parts.append(section)
            used_chars += section_chars
        else:
            overflow.append(section)
    return "\n\n".join(inline_parts), overflow


def _write_standing_order_history_files(
    workflow: SystemWorkflow, sections: list[str]
) -> list[str]:
    if not sections:
        return []
    directory = codex_pool.events_dir() / "standing_order_history" / str(workflow.pk)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "proposal_history.txt"
    path.write_text("\n\n---\n\n".join(sections), encoding="utf-8")
    return [str(path)]


def _parse_qa_output(raw_output: str) -> dict[str, Any] | None:
    text = raw_output.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    feedback = parsed.get("feedback")
    lgtm = parsed.get("lgtm")
    if not isinstance(feedback, str) or not isinstance(lgtm, bool):
        return None
    return {"feedback": feedback, "lgtm": lgtm}


def _parse_standing_order_candidate_output(raw_output: str) -> dict[str, Any] | None:
    text = _strip_json_markdown_fence(raw_output)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    if "proposal" in parsed:
        proposal = parsed.get("proposal")
        message = parsed.get("message")
        if proposal is None:
            if not isinstance(message, str) or not message.strip():
                return None
            return {"proposal": None, "message": message.strip()}
        if not isinstance(proposal, dict):
            return None
        normalized = _parse_standing_order_candidate_proposal(proposal)
        if normalized is None:
            return None
        return {"proposal": normalized, "message": ""}
    normalized = _parse_standing_order_candidate_proposal(parsed)
    if normalized is None:
        return None
    return {"proposal": normalized, "message": ""}


def _parse_standing_order_candidate_proposal(
    parsed: dict[str, Any],
) -> dict[str, Any] | None:
    title = parsed.get("title")
    summary = parsed.get("summary")
    impact = parsed.get("impact")
    implementation_direction = parsed.get("implementation_direction")
    if not isinstance(title, str):
        return None
    if not isinstance(summary, str):
        return None
    if not isinstance(impact, str):
        return None
    if not isinstance(implementation_direction, str):
        return None
    title = title.strip()
    if not title:
        return None
    return {
        "title": title,
        "summary": summary.strip(),
        "impact": impact.strip(),
        "implementation_direction": implementation_direction.strip(),
        "relevant_files": _string_list(parsed.get("relevant_files")),
    }


def _parse_standing_order_judge_output(raw_output: str) -> dict[str, str] | None:
    text = _strip_json_markdown_fence(raw_output)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    confidence = parsed.get("confidence")
    summary = parsed.get("summary")
    rationale = parsed.get("rationale")
    if confidence not in _CONFIDENCE_RANK:
        return None
    if not isinstance(summary, str) or not isinstance(rationale, str):
        return None
    return {
        "confidence": confidence,
        "summary": summary.strip(),
        "rationale": rationale.strip(),
    }


def _strip_json_markdown_fence(raw_output: str) -> str:
    text = raw_output.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _final_agent_text(events_path: str) -> str:
    path = Path(events_path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    latest = ""
    deltas: dict[str, str] = {}
    for raw in lines:
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        method = event.get("method")
        payload = event.get("payload") or {}
        if method == "item/agentMessage/delta":
            item_id = payload.get("itemId")
            delta = payload.get("delta")
            if isinstance(item_id, str) and isinstance(delta, str):
                deltas[item_id] = deltas.get(item_id, "") + delta
                latest = deltas[item_id]
        elif method == "item/completed":
            item = payload.get("item") or {}
            if (
                item.get("type") == "agentMessage"
                and item.get("phase") != "commentary"
                and isinstance(item.get("text"), str)
            ):
                latest = item["text"]
    return latest


def _fail_run_and_block_workflow(
    run: SystemAgentRun,
    error: str,
    raw_output: str = "",
    *,
    surface_to_thread: bool = True,
) -> None:
    run.status = SystemAgentRun.STATUS_FAILED
    run.error = error
    run.raw_output = raw_output
    run.save(update_fields=["status", "error", "raw_output", "updated_at"])
    workflow = run.workflow
    _block_workflow(workflow, error, surface_to_thread=surface_to_thread)


def _block_workflow(
    workflow: SystemWorkflow, error: str, *, surface_to_thread: bool = True
) -> None:
    workflow.status = SystemWorkflow.STATUS_BLOCKED
    workflow.step = STEP_BLOCKED
    workflow.state = {**workflow.state, "error": error}
    workflow.save(update_fields=["status", "step", "state", "updated_at"])
    if surface_to_thread:
        _surface_workflow_failure(workflow, error)


def _surface_workflow_failure(workflow: SystemWorkflow, error: str) -> None:
    if workflow.state.get("failure_surfaced") is True:
        return
    workflow.state = {**workflow.state, "failure_surfaced": True}
    workflow.save(update_fields=["state", "updated_at"])
    try:
        _spawn_workflow_failure_turn(workflow, error)
    except Exception:
        logger.exception(
            "failed to surface system workflow failure for workflow %s", workflow.pk
        )


def _state_string(workflow: SystemWorkflow, key: str) -> str:
    value = workflow.state.get(key)
    return value if isinstance(value, str) else ""


def _state_int(workflow: SystemWorkflow, key: str) -> int:
    value = workflow.state.get(key)
    return value if isinstance(value, int) and value >= 0 else 0


def _state_bool(workflow: SystemWorkflow, key: str) -> bool:
    return workflow.state.get(key) is True


def _confidence_meets_threshold(confidence: str, threshold: str) -> bool:
    return _CONFIDENCE_RANK.get(confidence, 0) >= _CONFIDENCE_RANK.get(threshold, 0)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        item = item.strip()
        if item and item not in normalized:
            normalized.append(item)
    return normalized


def _session_metadata_from_state(
    workflow: SystemWorkflow, key: str
) -> SessionMetadata | None:
    session_id = _state_int(workflow, key)
    if session_id < 1:
        return None
    return SessionMetadata.objects.filter(pk=session_id).first()


def _standing_order_main_thread_id(standing_order_id: int) -> str:
    return f"standing-order:{standing_order_id}"


def _workflow_for_instance(instance: CodexInstance) -> SystemWorkflow | None:
    if instance.workflow_id is None:
        return None
    try:
        return SystemWorkflow.objects.get(pk=instance.workflow_id)
    except SystemWorkflow.DoesNotExist:
        return None


def _system_agent_run_for_instance(instance: CodexInstance) -> SystemAgentRun | None:
    try:
        return SystemAgentRun.objects.select_related("workflow").get(instance=instance)
    except SystemAgentRun.DoesNotExist:
        pass
    if instance.workflow_id is None or not instance.agent_kind:
        return None
    try:
        workflow = SystemWorkflow.objects.get(pk=instance.workflow_id)
    except SystemWorkflow.DoesNotExist:
        return None
    run, _created = SystemAgentRun.objects.get_or_create(
        instance=instance,
        defaults={
            "workflow": workflow,
            "agent_kind": instance.agent_kind,
            "thread_id": instance.thread_id,
            "status": SystemAgentRun.STATUS_RUNNING,
            "input": {"cwd": instance.cwd},
        },
    )
    return run
