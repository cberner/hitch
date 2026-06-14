"""Derive a thread/session's PR, plan-mode, and auto-flag state.

This module owns the helpers that read a thread's rollout, workflow, and
metadata to derive its PR state (URLs, snapshots, observation epochs), its
Plan Mode state, and its auto-PR/auto-QA/auto-merge session settings. It is a
leaf module: it depends only on the sibling helpers and never imports views.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, NamedTuple

from hitch.main.models import CodexInstance, SessionMetadata, SystemWorkflow
from hitch.main.runtime import codex_events, codex_pool, rollout
from hitch.main.runtime.rollout_state import _rollout_path_for
from hitch.main.runtime.sdk_values import (
    is_nonbool_int,
    plain_sdk_value,
    sdk_model_dump_value,
    string_value,
    updated_at_seconds,
    value_for,
)
from hitch.main.sessions.entry_render import find_final_agent_idx, user_message_text
from hitch.main.workflows import pr_qa, pr_stage, pr_stage_refresh_state, system_agents

logger = logging.getLogger(__name__)

_PR_SLASH_PROMPT = system_agents.PR_SLASH_DISPLAY_PROMPT
_PR_SLASH_FINAL_PROMPT = system_agents.PR_SLASH_PROMPT
_PREVIOUS_DEFAULT_BRANCH_PR_SLASH_DISPLAY_PROMPT = (
    "Rebase on the repository's default branch, clean it up, and then open a PR"
)
_PREVIOUS_DEFAULT_BRANCH_PR_SLASH_FINAL_PROMPT = (
    "Rebase on the repository's default branch, polish it, get it ready, "
    "and commit the final changes. Do not push the branch or open a PR; "
    "Hitch will push and open it after this turn completes."
)
_PREVIOUS_PR_SLASH_DISPLAY_PROMPT = (
    "Rebase on master, clean it up, and then open a PR"
)
_PREVIOUS_REBASE_MASTER_PR_SLASH_FINAL_PROMPT = (
    "Rebase on master, polish it, get it ready, and commit the final changes. "
    "Do not push the branch or open a PR; Hitch will push and open it "
    "after this turn completes."
)
_PREVIOUS_HITCH_OWNED_PR_SLASH_FINAL_PROMPT = (
    "Polish it, get it ready, and commit the final changes. "
    "Do not push the branch or open a PR; Hitch will push and open it "
    "after this turn completes."
)
_PREVIOUS_HITCH_PR_SLASH_FINAL_PROMPT = (
    "Polish it, get it ready, commit the final changes, and push the branch. "
    "Do not open a PR; Hitch will open it after this turn completes."
)
_PREVIOUS_PR_SLASH_FINAL_PROMPT = (
    "Polish it, get it ready, and open or update the PR."
)
_LEGACY_PR_SLASH_PROMPT = (
    "Do a thorough review of the diff. Rebase on master, clean it up, "
    "and then open a PR"
)
_LEGACY_PR_SLASH_FINAL_PROMPT = (
    f"{_LEGACY_PR_SLASH_PROMPT}. After opening it, poll the PR every 2 minutes "
    "until you have CI status and at least one review signal: code review "
    "comments, a thumbs up emoji on the PR, or an explicit review approval. "
    "On each poll, check whether the PR has merge conflicts. Address CI "
    "failures, review comments, merge conflicts, and any other blocking issues; "
    "push fixes and keep looping until CI, review, and mergeability are all clean. "
    "Stop and report back if any single polling iteration has no results after "
    "30 minutes."
)
_PR_PROMPT_ALIASES = frozenset(
    {
        _PR_SLASH_PROMPT,
        _PR_SLASH_FINAL_PROMPT,
        _PREVIOUS_DEFAULT_BRANCH_PR_SLASH_DISPLAY_PROMPT,
        _PREVIOUS_DEFAULT_BRANCH_PR_SLASH_FINAL_PROMPT,
        _PREVIOUS_PR_SLASH_DISPLAY_PROMPT,
        _PREVIOUS_REBASE_MASTER_PR_SLASH_FINAL_PROMPT,
        _PREVIOUS_HITCH_OWNED_PR_SLASH_FINAL_PROMPT,
        _PREVIOUS_HITCH_PR_SLASH_FINAL_PROMPT,
        _PREVIOUS_PR_SLASH_FINAL_PROMPT,
        _LEGACY_PR_SLASH_PROMPT,
        _LEGACY_PR_SLASH_FINAL_PROMPT,
    }
)
_PR_WORKFLOW_PROMPT_PREFIXES = (
    "Hitch QA agent could not complete the PR workflow.",
    "Hitch PR workflow could not complete.",
    "Hitch PR monitor found follow-up work on the active PR.",
)
_GITHUB_PR_TOOL_RE = re.compile(
    r"(?i)(?:^|[/:\s._-])(?:github|mcp__codex_apps__github)(?:$|[/:\s._-]).*"
    r"(?:_?create[_\s-]?(?:pr|pull[_\s-]?request)|open[_\s-]?(?:pr|pull[_\s-]?request))"
)
_GITHUB_PR_URL_RE = re.compile(
    r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/pull/[0-9]+"
)
_GITHUB_PR_IDENTITY_RE = re.compile(
    r"https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/pull/([0-9]+)"
)
_ROLLOUT_COLLABORATION_MODE_NOT_PROVIDED = object()


class _ThreadPlanModeState(NamedTuple):
    active: bool
    awaiting_approval: bool


def _pr_observation_result_for_rollout_path(
    rollout_path: Path | None,
) -> codex_events.PrObservationResult:
    if rollout_path is None:
        return codex_events.PrObservationResult(snapshot=None)
    try:
        return rollout.latest_pr_observation_result(rollout_path)
    except Exception:
        logger.exception("failed to parse rollout %s for PR stage snapshot", rollout_path)
        return codex_events.PrObservationResult(snapshot=None)


def _workflow_after_main_lifecycle(
    workflow: SystemWorkflow | None,
    pr_observation: codex_events.PrObservationResult,
    *,
    main_updated_at: Any = None,
) -> SystemWorkflow | None:
    """Keep completed PR workflows only when main work has not superseded them."""
    if workflow is None or workflow.is_active:
        return workflow
    if pr_observation.superseded_by_lifecycle:
        if _workflow_pr_handoff_survives_lifecycle(
            workflow,
            main_updated_at=main_updated_at,
        ):
            return workflow
        return None
    main_updated_seconds = updated_at_seconds(main_updated_at)
    workflow_updated_seconds = updated_at_seconds(workflow.updated_at)
    main_is_newer = (
        main_updated_seconds is not None
        and workflow_updated_seconds is not None
        and main_updated_seconds > workflow_updated_seconds
    )
    if main_is_newer and _pr_snapshot_identity(pr_observation.snapshot) is not None:
        return None
    if pr_observation.snapshot is None and main_is_newer:
        return None
    return workflow


def _workflow_pr_handoff_survives_lifecycle(
    workflow: SystemWorkflow,
    *,
    main_updated_at: Any,
) -> bool:
    handoff = pr_qa.pr_handoff_for_workflow(workflow)
    handoff_identity = _pr_snapshot_identity(handoff)
    if handoff_identity is None:
        return False
    hitch_handoff = pr_stage_refresh_state.hitch_pr_handoff_for_workflow(workflow)
    if _pr_snapshot_identity(hitch_handoff) != handoff_identity:
        return False
    main_updated_seconds = updated_at_seconds(main_updated_at)
    workflow_updated_seconds = updated_at_seconds(workflow.updated_at)
    return (
        main_updated_seconds is None
        or workflow_updated_seconds is None
        or workflow_updated_seconds >= main_updated_seconds
    )


def _pr_snapshot_identity(snapshot: Mapping[str, Any] | None) -> tuple[str, int] | None:
    if not snapshot:
        return None
    url = string_value(snapshot.get("url"))
    if url:
        match = _GITHUB_PR_IDENTITY_RE.search(url)
        if match is not None:
            owner, repo, number = match.groups()
            return f"{owner}/{repo}", int(number)
    repo = string_value(snapshot.get("repository_full_name"))
    number = snapshot.get("pr_number")
    if repo and is_nonbool_int(number):
        return repo, number
    return None


def _thread_plan_mode_state(
    session_id: str,
    thread: Any,
    entries: list[dict[str, Any]],
    *,
    active_instance: CodexInstance | None = None,
    latest_collaboration_mode: str
    | None
    | object = _ROLLOUT_COLLABORATION_MODE_NOT_PROVIDED,
) -> _ThreadPlanModeState:
    """Return the Plan Mode state Codex recorded for this thread."""
    awaiting_approval = _entries_await_plan_approval(entries)
    latest_mode = (
        _latest_rollout_collaboration_mode(thread)
        if latest_collaboration_mode is _ROLLOUT_COLLABORATION_MODE_NOT_PROVIDED
        else latest_collaboration_mode
    )
    stored_plan_mode = (
        _latest_user_instance_ended_in_plan_mode(session_id)
        if latest_mode is None
        else False
    )
    active = (
        awaiting_approval
        or latest_mode == "plan"
        or stored_plan_mode
        or (active_instance is not None and active_instance.plan_mode)
    )
    return _ThreadPlanModeState(active=active, awaiting_approval=awaiting_approval)


def _latest_user_instance_ended_in_plan_mode(session_id: str) -> bool:
    latest = codex_pool.latest_for_thread(session_id)
    return bool(
        latest is not None
        and latest.purpose == CodexInstance.PURPOSE_USER
        and latest.workflow_id is None
        and latest.status == CodexInstance.STATUS_COMPLETED
        and latest.plan_mode
    )


def _latest_rollout_collaboration_mode(thread: Any) -> str | None:
    rollout_path = _rollout_path_for(thread)
    if rollout_path is None:
        return None
    return rollout.latest_collaboration_mode(rollout_path)


def _pr_url_for_thread(thread: Any) -> str | None:
    """Return the PR opened by the latest completed /pr turn, if any."""
    turns = getattr(thread, "turns", []) or []
    for turn in reversed(turns):
        items = [thread_item.root for thread_item in getattr(turn, "items", []) or []]
        if not _is_pr_creation_prompt_turn(items):
            continue
        final_idx = find_final_agent_idx(items)
        if final_idx == -1:
            continue
        # The model can emit the create_pull_request MCP call in the same
        # response that also carries the final-answer ``agentMessage``: the
        # tool runs after that response, so the completed ``mcpToolCall`` item
        # lands in the turn AFTER the final-answer item. ``items[:final_idx]``
        # would silently drop that result and the session page would render
        # no PR pill for the PR the user just opened. Iterate every item in
        # the turn after confirming a final-answer exists; the ``-1`` guard
        # above keeps incomplete turns out. Mirrors the fix applied to
        # ``rollout.latest_pr_url`` for the function_call_output-after-final
        # shape on the rollout path.
        urls: list[str] = []
        for item in items:
            if _github_pr_tool_call_used(item):
                urls.extend(_pr_urls_from_value(value_for(item, "result")))
        return urls[-1] if urls else None
    if turns:
        return None
    rollout_path = _rollout_path_for(thread)
    return rollout.latest_pr_url(rollout_path) if rollout_path is not None else None


def _current_pr_url_for_thread(
    thread: Any,
    *,
    pr_observation: codex_events.PrObservationResult,
    stage_pr_workflow: SystemWorkflow | None,
    latest_pr_url: str | None = None,
    latest_pr_url_loaded: bool = False,
) -> str | None:
    # A raw latest PR URL is only valid while the PR observation epoch is
    # current. Lifecycle-cleared sessions must not expose old PR actions.
    if not pr_observation.superseded_by_lifecycle:
        thread_url = latest_pr_url if latest_pr_url_loaded else _pr_url_for_thread(thread)
        if thread_url:
            return thread_url
    workflow_handoff = pr_qa.pr_handoff_for_workflow(stage_pr_workflow)
    workflow_url = string_value(workflow_handoff.get("url"))
    if workflow_url:
        return workflow_url
    snapshot = pr_observation.snapshot
    return string_value(snapshot.get("url") if snapshot else None) or None


def _fix_pr_url_for_thread(session_id: str, thread: Any) -> str | None:
    pr_observation = _pr_observation_result_for_thread(thread)
    stage_pr_workflow = _workflow_after_main_lifecycle(
        pr_stage._latest_pr_workflow_for_thread(session_id),
        pr_observation,
        main_updated_at=getattr(thread, "updated_at", None),
    )
    return _current_pr_url_for_thread(
        thread,
        pr_observation=pr_observation,
        stage_pr_workflow=stage_pr_workflow,
        latest_pr_url=None,
    )


def _pr_snapshot_for_thread(thread: Any) -> dict[str, Any] | None:
    return _pr_observation_result_for_thread(thread).snapshot


def _pr_observation_result_for_thread(thread: Any) -> codex_events.PrObservationResult:
    turns = getattr(thread, "turns", []) or []
    if not turns:
        return _pr_observation_result_for_rollout_path(_rollout_path_for(thread))
    observation_turns: list[codex_events.PrObservationTurn] = []
    for turn in getattr(thread, "turns", []) or []:
        items = [thread_item.root for thread_item in getattr(turn, "items", []) or []]
        mcp_items = tuple(_mcp_tool_items_for_items(items))
        is_pr_prompt = _turn_starts_pr_observation_epoch(items, mcp_items)
        is_pr_workflow_notice = _is_pr_workflow_notice_turn(items)
        final_idx = find_final_agent_idx(items)
        # Scan the whole turn rather than ``items[:final_idx]``: the create_
        # pull_request ``mcpToolCall`` (and any other GitHub MCP result) can
        # land AFTER the final-answer ``agentMessage`` when the model emits
        # the call and narrates it in the same response. Slicing here would
        # leave ``pr_observation.snapshot`` missing the PR identity even
        # though ``_pr_url_for_thread`` recovers the link, so the session
        # stage badge and ``derived_stage`` cache fall back to
        # ``IMPLEMENTATION`` and any ``closed``/``merged`` state is dropped.
        observation_turns.append(
            codex_events.PrObservationTurn(
                is_pr_prompt=is_pr_prompt,
                is_completed=final_idx != -1,
                items=mcp_items,
                has_lifecycle_activity=(
                    not is_pr_prompt
                    and not is_pr_workflow_notice
                    and final_idx != -1
                    and _turn_has_lifecycle_activity(items)
                ),
            )
        )
    return codex_events.pr_observation_result_from_turns(observation_turns)


def _mcp_tool_items_for_items(items: Iterable[Any]) -> Iterator[dict[str, Any]]:
    for item in items:
        if value_for(item, "type") != "mcpToolCall":
            continue
        yield {
            "type": "mcpToolCall",
            "server": string_value(value_for(item, "server")),
            "tool": string_value(value_for(item, "tool")),
            "arguments": plain_sdk_value(value_for(item, "arguments")) or {},
            "result": plain_sdk_value(value_for(item, "result")),
        }


def _is_pr_creation_prompt_turn(items: list[Any]) -> bool:
    for item in items:
        if value_for(item, "type") != "userMessage":
            continue
        if _is_pr_creation_prompt(user_message_text(item)):
            return True
    return False


def _is_pr_workflow_notice_turn(items: list[Any]) -> bool:
    for item in items:
        if value_for(item, "type") != "userMessage":
            continue
        if _is_pr_workflow_notice(user_message_text(item)):
            return True
    return False


def _turn_starts_pr_observation_epoch(
    items: list[Any], mcp_items: tuple[dict[str, Any], ...]
) -> bool:
    if _is_pr_creation_prompt_turn(items):
        return True
    if not _is_pr_workflow_notice_turn(items):
        return False
    return codex_events.pr_snapshot_from_completed_mcp_items(mcp_items) is not None


def _is_pr_creation_prompt(text: str) -> bool:
    return text.strip() in _PR_PROMPT_ALIASES


def _is_pr_workflow_notice(text: str) -> bool:
    text = text.strip()
    return any(
        text.startswith(prefix) for prefix in _PR_WORKFLOW_PROMPT_PREFIXES
    )


def _turn_has_lifecycle_activity(items: list[Any]) -> bool:
    return any(
        value_for(item, "type") in {"userMessage", "agentMessage"} for item in items
    )


def _github_pr_tool_call_used(item: Any) -> bool:
    if value_for(item, "type") != "mcpToolCall":
        return False
    server = string_value(value_for(item, "server"))
    tool = string_value(value_for(item, "tool"))
    detail = f"{server} / {tool}".strip()
    return _GITHUB_PR_TOOL_RE.search(detail) is not None


def _pr_urls_from_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return _GITHUB_PR_URL_RE.findall(value)
    if isinstance(value, dict):
        urls: list[str] = []
        for child in value.values():
            urls.extend(_pr_urls_from_value(child))
        return urls
    if isinstance(value, list | tuple):
        urls = []
        for child in value:
            urls.extend(_pr_urls_from_value(child))
        return urls
    text = string_value(value_for(value, "text"))
    if text:
        return _GITHUB_PR_URL_RE.findall(text)
    dumped = sdk_model_dump_value(value)
    if dumped is not value:
        return _pr_urls_from_value(dumped)
    urls = []
    for attr in ("url", "display_url", "displayUrl", "structured_content", "content"):
        urls.extend(_pr_urls_from_value(value_for(value, attr)))
    return urls


def _entries_await_plan_approval(entries: list[dict[str, Any]]) -> bool:
    return rollout.entries_await_plan_approval(entries)


def _pending_plan_entry(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    return rollout.pending_plan_entry(entries)


def _mark_pending_plan_actions(
    entries: list[dict[str, Any]], *, enabled: bool = True
) -> None:
    _clear_plan_actions(entries)
    if not enabled:
        return
    pending_plan = _pending_plan_entry(entries)
    if pending_plan is not None:
        pending_plan["show_plan_actions"] = True


def _clear_plan_actions(entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        if entry.get("kind") == "plan":
            entry["show_plan_actions"] = False
        elif entry.get("kind") == "intermediate":
            _clear_plan_actions(entry.get("items", []))


def _count_user_entries(entries: list[dict[str, Any]]) -> int:
    count = 0
    for entry in entries:
        if entry.get("kind") == "user":
            count += 1
        elif entry.get("kind") == "intermediate":
            count += _count_user_entries(entry.get("items", []))
    return count


def _auto_pr_enabled_for_session(session_id: str) -> bool:
    return SessionMetadata.objects.filter(
        thread_id=session_id, auto_pr_enabled=True
    ).exists()


def _auto_qa_enabled_for_session(session_id: str) -> bool:
    return SessionMetadata.objects.filter(
        thread_id=session_id, auto_qa_enabled=True
    ).exists()


def _auto_merge_to_local_branch_for_session(session_id: str) -> tuple[bool, str]:
    metadata = (
        SessionMetadata.objects.filter(thread_id=session_id)
        .only("auto_merge_to_local_branch", "auto_merge_branch")
        .first()
    )
    if metadata is None or not metadata.auto_merge_to_local_branch:
        return False, ""
    branch = metadata.auto_merge_branch.strip()
    return bool(branch), branch
