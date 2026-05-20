"""Reusable orchestration for Hitch-owned background Codex agents."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from django.db import IntegrityError, transaction
from django.utils import timezone
from openai_codex.generated.v2_all import ThreadSource

from hitch.main import codex_pool
from hitch.main.diffs import build_worktree_diff_text
from hitch.main.models import CodexInstance, KeyResult, ProposedTask, SystemAgentRun, SystemWorkflow

logger = logging.getLogger(__name__)

PR_QA_AGENT_KIND = "pr_qa"
OKR_TASK_AGENT_KIND = SystemWorkflow.KIND_OKR_TASK_GENERATION
QA_DISPLAY_AUTHOR = "QA agent"
OKR_TASK_DISPLAY_AUTHOR = "Task planning agent"
PR_SLASH_DISPLAY_PROMPT = (
    "Do a thorough review of the diff. Rebase on master, clean it up, "
    "and then open a PR"
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
STEP_QA_RUNNING = "qa_running"
STEP_FEEDBACK_RUNNING = "feedback_running"
STEP_BLOCKED = "blocked"
STEP_MAX_ITERATIONS_REACHED = "max_iterations_reached"
STEP_PR_PROMPT_SPAWNED = "pr_prompt_spawned"
STEP_OKR_TASKS_RUNNING = "okr_tasks_running"
STEP_OKR_TASKS_SAVED = "okr_tasks_saved"

_OKR_TASK_INLINE_CONTEXT_CHARS = 14_000
_OKR_TASK_TITLE_MAX_LEN = 200

_QA_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["feedback", "lgtm"],
    "properties": {
        "feedback": {"type": "string"},
        "lgtm": {"type": "boolean"},
    },
}

_OKR_TASK_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["tasks"],
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "title",
                    "description",
                    "success_criteria",
                    "rationale",
                ],
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "success_criteria": {"type": "string"},
                    "rationale": {"type": "string"},
                },
            },
        }
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
    developer_instructions: str | None = None,
    enable_memories: bool = False,
    initial_user_message_index: int = 0,
) -> SystemWorkflow:
    """Start a PR workflow by running QA before the work-agent PR prompt."""
    try:
        with transaction.atomic():
            workflow = SystemWorkflow.objects.create(
                kind=SystemWorkflow.KIND_PR_QA,
                main_thread_id=main_thread_id,
                cwd=cwd,
                status=SystemWorkflow.STATUS_RUNNING,
                step=STEP_QA_RUNNING,
                state={
                    "pr_prompt": PR_SLASH_PROMPT,
                    "sandbox_policy": sandbox_policy or "",
                    "approval_mode": approval_mode or "",
                    "model": model or "",
                    "reasoning_effort": reasoning_effort or "",
                    "developer_instructions": developer_instructions or "",
                    "enable_memories": enable_memories,
                    "next_user_message_index": max(initial_user_message_index, 0),
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


def start_okr_task_generation_workflow(*, key_result: KeyResult) -> SystemWorkflow:
    key_result = (
        KeyResult.objects.select_related("objective__project")
        .filter(pk=key_result.pk)
        .get()
    )
    main_thread_id = _okr_task_main_thread_id(key_result.pk)
    try:
        with transaction.atomic():
            workflow = SystemWorkflow.objects.create(
                kind=OKR_TASK_AGENT_KIND,
                main_thread_id=main_thread_id,
                cwd=key_result.objective.project.repo_path,
                status=SystemWorkflow.STATUS_RUNNING,
                step=STEP_OKR_TASKS_RUNNING,
                state={"key_result_id": key_result.pk},
            )
    except IntegrityError:
        existing_workflow = SystemWorkflow.objects.filter(
            kind=OKR_TASK_AGENT_KIND,
            main_thread_id=main_thread_id,
            status=SystemWorkflow.STATUS_RUNNING,
        ).first()
        if existing_workflow is None:
            raise
        return existing_workflow

    try:
        _spawn_okr_task_generation_run(workflow, key_result)
    except Exception as exc:
        _block_workflow(
            workflow,
            f"failed to start task planning agent: {exc!r}",
            surface_to_thread=False,
        )
    return workflow


def hidden_thread_ids() -> set[str]:
    return set(
        SystemAgentRun.objects.exclude(thread_id="")
        .values_list("thread_id", flat=True)
        .distinct()
    )


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
    if not CodexInstance.objects.filter(
        pk=instance.pk,
        auto_pr_triggered_at__isnull=True,
    ).exists():
        return
    start_pr_qa_workflow(
        main_thread_id=instance.thread_id,
        cwd=instance.cwd,
        sandbox_policy=instance.sandbox_policy or None,
        approval_mode=instance.approval_mode or SYSTEM_AGENT_APPROVAL_MODE,
        model=instance.model or None,
        reasoning_effort=instance.reasoning_effort or None,
        developer_instructions=instance.developer_instructions or None,
        enable_memories=instance.enable_memories,
        initial_user_message_index=(instance.user_message_index or 0) + 1,
    )
    CodexInstance.objects.filter(
        pk=instance.pk,
        auto_pr_triggered_at__isnull=True,
    ).update(auto_pr_triggered_at=timezone.now())


def _handle_system_agent_finished(instance: CodexInstance) -> None:
    run = _system_agent_run_for_instance(instance)
    if run is None:
        return
    if run.status in (SystemAgentRun.STATUS_COMPLETED, SystemAgentRun.STATUS_FAILED):
        return
    workflow = run.workflow
    if workflow.kind == OKR_TASK_AGENT_KIND:
        _handle_okr_task_agent_finished(instance, run, workflow)
        return
    if workflow.kind != SystemWorkflow.KIND_PR_QA:
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

    workflow.iteration += 1
    workflow.step = STEP_FEEDBACK_RUNNING
    workflow.save(update_fields=["iteration", "step", "state", "updated_at"])
    try:
        _spawn_qa_feedback_turn(workflow, feedback)
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


def _handle_okr_task_agent_finished(
    instance: CodexInstance, run: SystemAgentRun, workflow: SystemWorkflow
) -> None:
    if (
        workflow.status != SystemWorkflow.STATUS_RUNNING
        or workflow.step != STEP_OKR_TASKS_RUNNING
    ):
        return
    if instance.status != CodexInstance.STATUS_COMPLETED:
        _fail_run_and_block_workflow(
            run,
            f"task planning worker failed: {instance.error}",
            surface_to_thread=False,
        )
        return

    raw_output = _final_agent_text(instance.events_path)
    parsed = _parse_okr_task_output(raw_output)
    if parsed is None:
        _fail_run_and_block_workflow(
            run,
            "task planning output was not valid JSON",
            raw_output,
            surface_to_thread=False,
        )
        return

    key_result = (
        KeyResult.objects.select_related("objective__project")
        .filter(pk=_state_int(workflow, "key_result_id"))
        .first()
    )
    if key_result is None:
        _fail_run_and_block_workflow(
            run,
            "task planning key result no longer exists",
            raw_output,
            surface_to_thread=False,
        )
        return

    with transaction.atomic():
        for idx, task in enumerate(parsed["tasks"]):
            ProposedTask.objects.create(
                key_result=key_result,
                source_workflow=workflow,
                title=task["title"][:_OKR_TASK_TITLE_MAX_LEN],
                description=task["description"],
                success_criteria=task["success_criteria"],
                rationale=task["rationale"],
                sort_order=idx,
            )
        run.status = SystemAgentRun.STATUS_COMPLETED
        run.output = parsed
        run.raw_output = raw_output
        run.save(update_fields=["status", "output", "raw_output", "updated_at"])
        workflow.status = SystemWorkflow.STATUS_COMPLETED
        workflow.step = STEP_OKR_TASKS_SAVED
        workflow.state = {**workflow.state, "saved_task_count": len(parsed["tasks"])}
        workflow.save(update_fields=["status", "step", "state", "updated_at"])


def _spawn_pr_qa_run(workflow: SystemWorkflow) -> SystemAgentRun:
    diff_text = build_worktree_diff_text(workflow.cwd)
    prompt = _qa_prompt(workflow.cwd, diff_text)
    instance = codex_pool.spawn_new_session(
        cwd=workflow.cwd,
        prompt=prompt,
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


def _spawn_okr_task_generation_run(
    workflow: SystemWorkflow, key_result: KeyResult
) -> SystemAgentRun:
    prompt, context_files = _okr_task_generation_prompt(workflow, key_result)
    if context_files:
        workflow.state = {**workflow.state, "context_files": context_files}
        workflow.save(update_fields=["state", "updated_at"])
    instance = codex_pool.spawn_new_session(
        cwd=workflow.cwd,
        prompt=prompt,
        approval_mode=SYSTEM_AGENT_APPROVAL_MODE,
        thread_source=ThreadSource.subagent,
        purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
        workflow_id=workflow.pk,
        agent_kind=OKR_TASK_AGENT_KIND,
        display_author=OKR_TASK_DISPLAY_AUTHOR,
        output_schema=_OKR_TASK_OUTPUT_SCHEMA,
    )
    run, _created = SystemAgentRun.objects.get_or_create(
        instance=instance,
        defaults={
            "workflow": workflow,
            "agent_kind": OKR_TASK_AGENT_KIND,
            "thread_id": instance.thread_id,
            "status": SystemAgentRun.STATUS_RUNNING,
            "input": {
                "cwd": workflow.cwd,
                "key_result_id": key_result.pk,
                "context_files": context_files,
            },
        },
    )
    return run


def _spawn_qa_feedback_turn(workflow: SystemWorkflow, feedback: str) -> CodexInstance:
    return _spawn_workflow_turn(
        workflow,
        prompt=f"Feedback from Hitch QA agent:\n\n{feedback}",
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
        "review/cleanup/open-PR pass.\n\n"
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


def _okr_task_generation_prompt(
    workflow: SystemWorkflow, key_result: KeyResult
) -> tuple[str, list[str]]:
    objective = key_result.objective
    project = objective.project
    sibling_key_results = list(
        objective.key_results.prefetch_related("proposed_tasks").order_by(
            "created_at", "id"
        )
    )
    prior_task_sections = _prior_task_sections(key_result, sibling_key_results)
    inline_prior, overflow_prior = _split_task_context(prior_task_sections)
    context_files = _write_okr_task_context_files(workflow, overflow_prior)
    context_file_text = (
        "\n".join(f"- {path}" for path in context_files) if context_files else "(none)"
    )
    sibling_text = "\n".join(
        _format_key_result_context(kr, is_target=(kr.pk == key_result.pk))
        for kr in sibling_key_results
    )
    return (
        "You are Hitch's task planning agent for an OKR workflow.\n\n"
        "Act like a senior software engineering manager doing practical planning "
        "for a general software project. Create a task list that accomplishes "
        "the target key result, supports the objective, and must not regress the "
        "other key results in the same objective.\n\n"
        "Split tasks into small, but logically consistent pieces. For example, "
        "for a blogging platform, one task might be the tagging system, another "
        "might be a basic comment implementation, with a follow-on task to add "
        "rich text to the comments. Avoid both vague umbrella tasks and tiny "
        "implementation chores that would not stand alone as meaningful work.\n\n"
        "Use past proposed tasks and their outcomes to tailor the list to the "
        "user's feedback. Repeat patterns from accepted or completed tasks when "
        "they fit; avoid or adjust patterns from rejected or superseded tasks; "
        "honor outcome notes over your own assumptions.\n\n"
        f"Project: {project.name}\n"
        f"Repository cwd: {project.repo_path}\n\n"
        "Objective:\n"
        f"Title: {objective.title}\n"
        f"Description: {objective.description or '(none)'}\n\n"
        "Key results in this objective:\n"
        f"{sibling_text or '(none)'}\n\n"
        "Past proposed tasks and outcomes included inline:\n"
        f"{inline_prior or '(none)'}\n\n"
        "Additional past proposed task context files:\n"
        f"{context_file_text}\n\n"
        "Return only JSON matching this shape: "
        '{"tasks": [{"title": string, "description": string, '
        '"success_criteria": string, "rationale": string}]}. '
        "Each title must be concise. Each description should explain the work "
        "without assuming a specific framework unless the OKR context does. "
        "Success criteria should be observable completion checks."
    ), context_files


def _format_key_result_context(key_result: KeyResult, *, is_target: bool) -> str:
    label = "Target KR" if is_target else "Sibling KR"
    return (
        f"- {label}: {key_result.title}\n"
        f"  Description: {key_result.description or '(none)'}\n"
        f"  Work instructions: {key_result.work_instructions or '(none)'}"
    )


def _prior_task_sections(
    target_key_result: KeyResult, key_results: list[KeyResult]
) -> list[tuple[bool, str]]:
    sections: list[tuple[bool, str]] = []
    for key_result in key_results:
        for task in key_result.proposed_tasks.all():
            important = (
                key_result.pk == target_key_result.pk
                or bool(task.outcome_status)
                or bool(task.outcome_notes.strip())
            )
            sections.append((important, _format_proposed_task_context(key_result, task)))
    sections.sort(key=lambda item: (not item[0],))
    return sections


def _format_proposed_task_context(key_result: KeyResult, task: ProposedTask) -> str:
    return (
        f"KR: {key_result.title}\n"
        f"Task: {task.title}\n"
        f"Description: {task.description or '(none)'}\n"
        f"Success criteria: {task.success_criteria or '(none)'}\n"
        f"Rationale: {task.rationale or '(none)'}\n"
        f"Outcome status: {task.outcome_status or '(not set)'}\n"
        f"Outcome notes: {task.outcome_notes or '(none)'}"
    )


def _split_task_context(sections: list[tuple[bool, str]]) -> tuple[str, list[str]]:
    inline_parts: list[str] = []
    overflow: list[str] = []
    used_chars = 0
    for _important, section in sections:
        section_chars = len(section) + 2
        if used_chars + section_chars <= _OKR_TASK_INLINE_CONTEXT_CHARS:
            inline_parts.append(section)
            used_chars += section_chars
        else:
            overflow.append(section)
    return "\n\n".join(inline_parts), overflow


def _write_okr_task_context_files(
    workflow: SystemWorkflow, sections: list[str]
) -> list[str]:
    if not sections:
        return []
    directory = codex_pool.events_dir() / "okr_task_context" / str(workflow.pk)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "prior_tasks.txt"
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


def _parse_okr_task_output(raw_output: str) -> dict[str, Any] | None:
    text = _strip_json_markdown_fence(raw_output)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    tasks = parsed.get("tasks")
    if not isinstance(tasks, list):
        return None
    normalized_tasks: list[dict[str, str]] = []
    for task in tasks:
        if not isinstance(task, dict):
            return None
        title = task.get("title")
        description = task.get("description")
        success_criteria = task.get("success_criteria")
        rationale = task.get("rationale")
        if not isinstance(title, str):
            return None
        if not isinstance(description, str):
            return None
        if not isinstance(success_criteria, str):
            return None
        if not isinstance(rationale, str):
            return None
        if not title.strip():
            return None
        normalized_tasks.append(
            {
                "title": title.strip(),
                "description": description.strip(),
                "success_criteria": success_criteria.strip(),
                "rationale": rationale.strip(),
            }
        )
    return {"tasks": normalized_tasks}


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


def _okr_task_main_thread_id(key_result_id: int) -> str:
    return f"okr-key-result:{key_result_id}"


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
