"""Parse codex rollout files directly to surface items the SDK drops.

`thread/read` rebuilds turns through the Limited-mode persistence policy in
`codex-rs/rollout/src/policy.rs`. That filter drops every `commandExecution`
thread item: the underlying `ExecCommandBegin` event is never persisted, and
`ExecCommandEnd` is only persisted under the deprecated Extended mode that
the app-server now ignores. The raw `FunctionCall` response items the model
emitted are still in the on-disk JSONL, but the SDK's `ThreadHistoryBuilder`
only consumes `ResponseItem::Message` (for hook prompts), so function calls
never reach the wire.

Parsing the rollout file at `Thread.path` ourselves recovers those entries.
`Thread.path` is marked UNSTABLE in the SDK, so callers must fall back to
the SDK-built turns when the path is missing or the file can't be read.
"""

import json
import logging
import re
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Function names codex assigns to shell-like tools (see
# `core/src/tools/handlers/shell_spec.rs` in codex-rs). FunctionCall response
# items with one of these names get rendered as `commandExecution` rows;
# anything else stays invisible because we cannot guess how to surface it.
_SHELL_TOOL_NAMES = frozenset(
    {
        "exec_command",
        "shell_command",
        "shell",
        "container.exec",
    }
)
_SHELL_FAILURE_PREFIXES = tuple(f"{name} failed for `" for name in sorted(_SHELL_TOOL_NAMES))
_GITHUB_PR_TOOL_RE = re.compile(
    r"(?i)(?:^|[/:\s._-])(?:github|mcp__codex_apps__github)(?:$|[/:\s._-]).*"
    r"(?:_?create[_\s-]?(?:pr|pull[_\s-]?request)|open[_\s-]?(?:pr|pull[_\s-]?request))"
)
_GITHUB_PR_URL_RE = re.compile(
    r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/pull/[0-9]+"
)
_PR_PROMPT_ALIASES = frozenset(
    {
        "/pr",
        "Rebase on master, clean it up, and then open a PR",
        "Polish it, get it ready, and open or update the PR.",
        (
            "Do a thorough review of the diff. Rebase on master, clean it up, "
            "and then open a PR"
        ),
        (
            "Do a thorough review of the diff. Rebase on master, clean it up, "
            "and then open a PR. After opening it, poll the PR every 2 minutes "
            "until you have CI status and at least one review signal: code review "
            "comments, a thumbs up emoji on the PR, or an explicit review approval. "
            "On each poll, check whether the PR has merge conflicts. Address CI "
            "failures, review comments, merge conflicts, and any other blocking issues; "
            "push fixes and keep looping until CI, review, and mergeability are all clean. "
            "Stop and report back if any single polling iteration has no results after "
            "30 minutes."
        ),
    }
)
_PLAN_APPROVAL_PROMPT = "Implement the plan."
_COLLABORATION_MODE_PLAN = "plan"
_COLLABORATION_MODE_DEFAULT = "default"


def iter_entries(rollout_path: Path) -> Iterator[dict[str, Any]]:
    """Yield session-view entries directly from a codex rollout JSONL file.

    Dict shape matches `views._render_entries` so the template renders it
    without changes. Entries are emitted in rollout-file order, which is the
    same chronological order codex used when writing them.
    """
    lines = _load_rollout_lines(rollout_path)
    if lines is None:
        return
    yield from _entries_from_lines(lines)


def entries_await_plan_approval(entries: list[dict[str, Any]]) -> bool:
    return pending_plan_entry(entries) is not None


def pending_plan_entry(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    for entry in reversed(entries):
        kind = entry.get("kind")
        if kind in {"intermediate", "approval_declined", "tool_call", "thinking", "user"}:
            continue
        if kind == "plan":
            return entry
        if kind == "agent":
            return None
    return None


def latest_token_usage(rollout_path: Path) -> dict[str, int] | None:
    """Return latest token counts and context window for a thread.

    Codex emits a `TokenCount` event_msg after each turn whose
    `info.total_token_usage` is the running session total and whose
    `info.last_token_usage` is the latest active context size. Only the
    most recent such event is kept; earlier ones are obsoleted by it.
    Returns None when the rollout is unreadable or contains no parseable
    token_count event (e.g. a session that has yet to receive a response).
    """
    lines = _load_rollout_lines(rollout_path)
    if lines is None:
        return None
    latest: dict[str, Any] | None = None
    latest_context: dict[str, Any] = {}
    latest_context_window: int = 0
    for entry in lines:
        if entry.get("type") != "event_msg":
            continue
        payload = entry.get("payload") or {}
        if payload.get("type") != "token_count":
            continue
        info = payload.get("info")
        if not isinstance(info, dict):
            continue
        total = info.get("total_token_usage")
        if isinstance(total, dict):
            latest = total
            last = info.get("last_token_usage")
            latest_context = last if isinstance(last, dict) else {}
            latest_context_window = _coerce_int(info.get("model_context_window"))
    if latest is None:
        return None
    return {
        "input_tokens": _coerce_int(latest.get("input_tokens")),
        "cached_input_tokens": _coerce_int(latest.get("cached_input_tokens")),
        "output_tokens": _coerce_int(latest.get("output_tokens")),
        "total_tokens": _coerce_int(latest.get("total_tokens")),
        "context_tokens": _coerce_int(latest_context.get("total_tokens")),
        "model_context_window": latest_context_window,
    }


def latest_collaboration_mode(rollout_path: Path) -> str | None:
    """Return the last collaboration mode Codex recorded in the rollout."""
    lines = _load_rollout_lines(rollout_path)
    if lines is None:
        return None
    modes = _collaboration_modes_by_turn(lines)
    if not modes:
        return None
    return modes[max(modes)]


def token_usage_history(rollout_path: Path) -> list[dict[str, int]]:
    """Return cumulative token counts for every timestamped token_count event."""
    lines = _load_rollout_lines(rollout_path)
    if lines is None:
        return []
    history: list[dict[str, int]] = []
    for entry in lines:
        timestamp = _iso_to_unix_seconds(entry.get("timestamp"))
        if timestamp is None or entry.get("type") != "event_msg":
            continue
        payload = entry.get("payload") or {}
        if payload.get("type") != "token_count":
            continue
        info = payload.get("info")
        if not isinstance(info, dict):
            continue
        total = info.get("total_token_usage")
        if not isinstance(total, dict):
            continue
        history.append(
            {
                "timestamp": timestamp,
                "input_tokens": _coerce_int(total.get("input_tokens")),
                "cached_input_tokens": _coerce_int(total.get("cached_input_tokens")),
                "output_tokens": _coerce_int(total.get("output_tokens")),
                "total_tokens": _coerce_int(total.get("total_tokens")),
            }
        )
    return history


def latest_pr_url(rollout_path: Path) -> str | None:
    """Return the last GitHub PR URL produced by a completed /pr turn."""
    lines = _load_rollout_lines(rollout_path)
    if lines is None:
        return None
    function_calls_by_id = _function_calls_by_id(lines)
    latest: str | None = None
    for _, turn_lines in _lines_by_turn(lines):
        if not _turn_is_pr_prompt(turn_lines):
            continue
        final_idx = _find_final_agent_line_idx(turn_lines)
        if final_idx == -1:
            continue
        urls: list[str] = []
        for entry in turn_lines[:final_idx]:
            urls.extend(
                _github_pr_urls_from_value(
                    _github_pr_tool_result_value(entry, function_calls_by_id)
                )
            )
        latest = urls[-1] if urls else None
    return latest


def _load_rollout_lines(rollout_path: Path) -> list[dict[str, Any]] | None:
    try:
        text = rollout_path.read_text()
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("failed to read rollout %s: %s", rollout_path, exc)
        return None
    lines: list[dict[str, Any]] = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("skipping malformed rollout line in %s", rollout_path)
            continue
        if isinstance(entry, dict):
            lines.append(entry)
        else:
            logger.debug("skipping malformed rollout line in %s", rollout_path)
    return lines


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


Entry = dict[str, Any]
EntryResult = Entry | list[Entry] | None
AgentDedupeKey = tuple[str, str]
MemoryCitation = dict[str, Any]
MemoryCitationsByKey = dict[int, dict[AgentDedupeKey, list[MemoryCitation]]]

_MEMORY_CITATION_OPEN_TAG = "<oai-mem-citation>"
_MEMORY_CITATION_CLOSE_TAG = "</oai-mem-citation>"


def _entries_from_lines(lines: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    # Index function_call_output by call_id so each command entry can show
    # whether the call actually succeeded without a second file pass.
    outputs: dict[str, dict[str, Any]] = {}
    for entry in lines:
        if entry.get("type") != "response_item":
            continue
        payload = entry.get("payload") or {}
        if payload.get("type") != "function_call_output":
            continue
        call_id = payload.get("call_id")
        if isinstance(call_id, str):
            outputs[call_id] = payload

    represented_agent_texts_by_turn = _represented_agent_texts_by_turn(lines)
    memory_citations_by_turn_text = _memory_citations_by_turn_text(lines)
    completed_plan_texts_by_turn = _completed_plan_texts_by_turn(lines)
    plan_mode_turns = _plan_mode_turns(lines)
    proposed_plan_texts_by_turn, event_plan_texts_by_turn = _proposed_plan_texts_by_turn(
        lines,
        plan_mode_turns,
        completed_plan_texts_by_turn,
    )
    for turn_idx, entry in _lines_with_turn_indices(lines):
        represented_agent_texts = represented_agent_texts_by_turn.get(turn_idx, Counter())
        memory_citations_by_text = memory_citations_by_turn_text.get(turn_idx, {})
        completed_plan_texts = completed_plan_texts_by_turn.get(turn_idx, set())
        proposed_plan_texts = proposed_plan_texts_by_turn.get(turn_idx, set())
        event_plan_texts = event_plan_texts_by_turn.get(turn_idx, set())
        result = _entry_for_rollout_line(
            entry,
            outputs,
            represented_agent_texts,
            memory_citations_by_text,
            completed_plan_texts,
            proposed_plan_texts,
            event_plan_texts,
        )
        if isinstance(result, list):
            yield from result
        elif result is not None:
            yield result


def _entry_for_rollout_line(
    line: dict[str, Any],
    outputs: dict[str, dict[str, Any]],
    represented_agent_texts: Counter[AgentDedupeKey],
    memory_citations_by_text: dict[AgentDedupeKey, list[MemoryCitation]],
    completed_plan_texts: set[str],
    proposed_plan_texts: set[str],
    event_plan_texts: set[str],
) -> EntryResult:
    line_type = line.get("type")
    payload = line.get("payload") or {}
    timestamp = _iso_to_unix_seconds(line.get("timestamp"))

    if line_type == "event_msg":
        return _entry_from_event(
            payload,
            timestamp,
            memory_citations_by_text,
            completed_plan_texts,
            proposed_plan_texts,
            event_plan_texts,
        )
    if line_type == "response_item":
        return _entry_from_response_item(
            payload,
            timestamp,
            outputs,
            represented_agent_texts,
            completed_plan_texts,
            proposed_plan_texts,
            event_plan_texts,
        )
    return None


def _entry_from_event(
    payload: dict[str, Any],
    timestamp: int | None,
    memory_citations_by_text: dict[AgentDedupeKey, list[MemoryCitation]],
    completed_plan_texts: set[str],
    proposed_plan_texts: set[str],
    event_plan_texts: set[str],
) -> dict[str, Any] | None:
    event_type = payload.get("type")
    if event_type == "user_message":
        return {
            "kind": "user",
            "text": _user_message_text(payload),
            "timestamp": timestamp,
        }
    if event_type == "agent_message":
        text = payload.get("message")
        if not isinstance(text, str) or not text:
            return None
        text, event_memory_citation = _strip_memory_citations(text)
        text = text.strip()
        if not text:
            return None
        plan_text = _proposed_plan_text(text)
        if plan_text is not None and plan_text in completed_plan_texts:
            return None
        if plan_text is not None and plan_text in event_plan_texts:
            return {"kind": "plan", "text": plan_text, "timestamp": timestamp}
        if plan_text is not None and proposed_plan_texts:
            return None
        # `phase` is preserved so the view layer can pick the turn's final
        # agent reply with the same `MessagePhase` semantics as the SDK
        # (final_answer wins, commentary never wins, unset is eligible).
        phase = payload.get("phase")
        entry: dict[str, Any] = {
            "kind": "agent",
            "text": text,
            "timestamp": timestamp,
            "phase": phase if isinstance(phase, str) else None,
        }
        citation = _pop_memory_citation(
            memory_citations_by_text, _agent_dedupe_key(text, payload)
        )
        if citation is not None or event_memory_citation is not None:
            citation = citation or event_memory_citation
            entry["memory_citation"] = citation
        return entry
    if event_type == "item_completed":
        return _plan_entry_from_completed_item(payload, timestamp)
    if event_type == "patch_apply_end":
        changes = payload.get("changes") or {}
        return _tool_call(
            "fileChange",
            "File change",
            _format_paths(list(changes.keys())),
            _non_completed_status(payload.get("status")),
            timestamp,
        )
    if event_type == "context_compacted":
        return _tool_call("contextCompaction", "Context compaction", "", None, timestamp)
    if event_type == "web_search_end":
        return _tool_call(
            "webSearch",
            "Web search",
            payload.get("query") or "",
            None,
            timestamp,
        )
    if event_type == "mcp_tool_call_end":
        invocation = payload.get("invocation") or {}
        detail = f"{invocation.get('server', '')} / {invocation.get('tool', '')}"
        return _tool_call(
            "mcpToolCall",
            "MCP tool call",
            detail,
            _mcp_end_status(payload.get("result")),
            timestamp,
        )
    if event_type == "agent_reasoning":
        text = payload.get("text") or ""
        if not text:
            return None
        return _tool_call(
            "reasoning",
            "Reasoning",
            text.split("\n", 1)[0],
            None,
            timestamp,
        )
    if event_type == "entered_review_mode":
        return _tool_call("enteredReviewMode", "Entered review mode", "", None, timestamp)
    if event_type == "exited_review_mode":
        return _tool_call("exitedReviewMode", "Exited review mode", "", None, timestamp)
    return None


def _entry_from_response_item(
    payload: dict[str, Any],
    timestamp: int | None,
    outputs: dict[str, dict[str, Any]],
    represented_agent_texts: Counter[AgentDedupeKey],
    completed_plan_texts: set[str],
    proposed_plan_texts: set[str],
    event_plan_texts: set[str],
) -> EntryResult:
    item_type = payload.get("type")
    if item_type == "message":
        return _agent_entry_from_response_message(
            payload,
            timestamp,
            represented_agent_texts,
            completed_plan_texts,
            proposed_plan_texts,
            event_plan_texts,
        )
    if item_type == "function_call":
        name = payload.get("name") or ""
        if name not in _SHELL_TOOL_NAMES:
            return None
        command = _extract_command(payload.get("arguments"))
        call_id = payload.get("call_id")
        output_payload = outputs.get(call_id) if isinstance(call_id, str) else None
        status = _function_call_status(output_payload)
        tool_entry = _tool_call(
            "commandExecution",
            "Command",
            command,
            status,
            timestamp,
        )
        approval_entry = _approval_declined_entry(command, output_payload, timestamp)
        if approval_entry is not None:
            return [tool_entry, approval_entry]
        return tool_entry
    if item_type == "local_shell_call":
        action = payload.get("action") or {}
        if action.get("type") != "exec":
            return None
        parts = action.get("command") or []
        command = " ".join(str(p) for p in parts)
        return _tool_call(
            "commandExecution",
            "Command",
            command,
            _non_completed_status(payload.get("status")),
            timestamp,
        )
    return None


def _lines_with_turn_indices(lines: list[dict[str, Any]]) -> Iterator[tuple[int, dict[str, Any]]]:
    turn_idx = 0
    started = False
    for entry in lines:
        if _is_user_message_line(entry):
            if started:
                turn_idx += 1
            started = True
        yield turn_idx, entry


def _is_user_message_line(entry: dict[str, Any]) -> bool:
    if entry.get("type") != "event_msg":
        return False
    payload = entry.get("payload") or {}
    return payload.get("type") == "user_message"


def _is_pr_prompt(text: str) -> bool:
    return text in _PR_PROMPT_ALIASES


def _function_calls_by_id(lines: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    calls: dict[str, dict[str, Any]] = {}
    for entry in lines:
        if entry.get("type") != "response_item":
            continue
        payload = entry.get("payload") or {}
        if payload.get("type") != "function_call":
            continue
        call_id = payload.get("call_id")
        if isinstance(call_id, str):
            calls[call_id] = payload
    return calls


def _turn_is_pr_prompt(turn_lines: list[dict[str, Any]]) -> bool:
    for entry in turn_lines:
        if not _is_user_message_line(entry):
            continue
        payload = entry.get("payload") or {}
        return _is_pr_prompt(_user_message_text(payload).strip())
    return False


def _find_final_agent_line_idx(turn_lines: list[dict[str, Any]]) -> int:
    for idx in range(len(turn_lines) - 1, -1, -1):
        if _is_agent_response_line(turn_lines[idx]) and _agent_response_phase(
            turn_lines[idx]
        ) == "final_answer":
            return idx
    for idx in range(len(turn_lines) - 1, -1, -1):
        if _is_agent_response_line(turn_lines[idx]):
            return idx
    return -1


def _github_pr_tool_result_value(
    entry: dict[str, Any], function_calls_by_id: dict[str, dict[str, Any]]
) -> Any:
    payload = entry.get("payload") or {}
    if entry.get("type") == "event_msg" and payload.get("type") == "mcp_tool_call_end":
        invocation = payload.get("invocation") or {}
        if not isinstance(invocation, dict):
            return None
        if _github_pr_tool_used(invocation.get("server"), invocation.get("tool")):
            return payload.get("result")
        return None
    if entry.get("type") == "response_item" and payload.get("type") == "function_call_output":
        call_id = payload.get("call_id")
        call = function_calls_by_id.get(call_id) if isinstance(call_id, str) else None
        if call is not None and _github_pr_tool_used(call.get("name")):
            return payload.get("output")
    return None


def _github_pr_tool_used(*parts: Any) -> bool:
    detail = " / ".join(part for part in parts if isinstance(part, str))
    return _GITHUB_PR_TOOL_RE.search(detail) is not None


def _github_pr_urls_from_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return _GITHUB_PR_URL_RE.findall(value)
    if isinstance(value, dict):
        urls: list[str] = []
        for child in value.values():
            urls.extend(_github_pr_urls_from_value(child))
        return urls
    if isinstance(value, list | tuple):
        urls = []
        for child in value:
            urls.extend(_github_pr_urls_from_value(child))
        return urls
    return []


def _is_agent_response_line(entry: dict[str, Any]) -> bool:
    payload = entry.get("payload") or {}
    if _agent_response_phase(entry) == "commentary":
        return False
    if entry.get("type") == "event_msg" and payload.get("type") == "agent_message":
        return True
    return (
        entry.get("type") == "response_item"
        and payload.get("type") == "message"
        and payload.get("role") == "assistant"
    )


def _agent_response_phase(entry: dict[str, Any]) -> str | None:
    payload = entry.get("payload") or {}
    phase = payload.get("phase")
    return phase if isinstance(phase, str) else None


def _plan_mode_turns(lines: list[dict[str, Any]]) -> set[int]:
    return {
        turn_idx
        for turn_idx, mode in _collaboration_modes_by_turn(lines).items()
        if mode == _COLLABORATION_MODE_PLAN
    }


def _collaboration_modes_by_turn(lines: list[dict[str, Any]]) -> dict[int, str]:
    modes: dict[int, str] = {}
    pending_mode: str | None = None
    turn_idx = -1
    current_turn_accepts_mode = False
    for entry in lines:
        mode = _collaboration_mode_from_line(entry)
        if mode is not None:
            if current_turn_accepts_mode:
                modes[turn_idx] = mode
            else:
                pending_mode = mode
            continue
        if _is_user_message_line(entry):
            turn_idx += 1
            current_turn_accepts_mode = True
            if pending_mode is not None:
                modes[turn_idx] = pending_mode
            pending_mode = None
            continue
        current_turn_accepts_mode = False
    return modes


def _collaboration_mode_from_line(entry: dict[str, Any]) -> str | None:
    if entry.get("type") == "turn_context":
        return _turn_context_collaboration_mode(entry)
    if _is_task_started_line(entry):
        return _task_started_collaboration_mode(entry)
    return None


def _turn_context_collaboration_mode(entry: dict[str, Any]) -> str | None:
    payload = entry.get("payload") or {}
    mode_data = payload.get("collaboration_mode") or payload.get("collaborationMode")
    if not isinstance(mode_data, dict):
        return None
    mode = mode_data.get("mode")
    return mode if isinstance(mode, str) else None


def _is_task_started_line(entry: dict[str, Any]) -> bool:
    if entry.get("type") != "event_msg":
        return False
    payload = entry.get("payload") or {}
    return payload.get("type") == "task_started"


def _task_started_collaboration_mode(entry: dict[str, Any]) -> str | None:
    payload = entry.get("payload") or {}
    mode = payload.get("collaboration_mode_kind")
    return mode if isinstance(mode, str) else None


def _proposed_plan_texts_by_turn(
    lines: list[dict[str, Any]],
    plan_mode_turns: set[int],
    completed_plan_texts_by_turn: dict[int, set[str]],
) -> tuple[dict[int, set[str]], dict[int, set[str]]]:
    by_turn: dict[int, set[str]] = {}
    event_by_turn: dict[int, set[str]] = {}
    modes_by_turn = _collaboration_modes_by_turn(lines)
    awaiting_plan_approval = False
    allow_plan_mode_followup = False
    for turn_idx, turn_lines in _lines_by_turn(lines):
        mode = modes_by_turn.get(turn_idx)
        is_plan_mode_turn = turn_idx in plan_mode_turns
        turn_started_awaiting_plan_approval = awaiting_plan_approval
        exits_plan_mode = _default_turn_exits_plan_mode(
            mode,
            turn_lines,
            turn_started_awaiting_plan_approval,
            allow_plan_mode_followup,
        )
        if exits_plan_mode:
            awaiting_plan_approval = False
            allow_plan_mode_followup = False
        event_turn_texts: set[str] = set()
        response_turn_texts: set[str] = set()
        for entry in turn_lines:
            source, plan_text = _proposed_plan_text_from_line(entry)
            if plan_text is None or not _should_render_proposed_plan(
                plan_text,
                is_plan_mode_turn,
                awaiting_plan_approval,
                allow_plan_mode_followup,
                exits_plan_mode,
            ):
                continue
            if source == "event":
                event_turn_texts.add(plan_text)
            elif source == "response":
                response_turn_texts.add(plan_text)
        turn_texts = response_turn_texts or event_turn_texts
        if response_turn_texts:
            event_turn_texts = set()
        if turn_texts:
            by_turn[turn_idx] = turn_texts
        if event_turn_texts:
            event_by_turn[turn_idx] = event_turn_texts
        if not exits_plan_mode and (
            completed_plan_texts_by_turn.get(turn_idx) or turn_texts
        ):
            awaiting_plan_approval = True
        elif not exits_plan_mode and _turn_has_agent_response(turn_lines):
            awaiting_plan_approval = False
        allow_plan_mode_followup = bool(
            not exits_plan_mode
            and is_plan_mode_turn
            and not turn_texts
            and _turn_has_agent_response(turn_lines)
        )
    return by_turn, event_by_turn


def _lines_by_turn(lines: list[dict[str, Any]]) -> Iterator[tuple[int, list[dict[str, Any]]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for turn_idx, entry in _lines_with_turn_indices(lines):
        grouped.setdefault(turn_idx, []).append(entry)
    yield from grouped.items()


def _proposed_plan_text_from_line(entry: dict[str, Any]) -> tuple[str | None, str | None]:
    payload = entry.get("payload") or {}
    if payload.get("phase") == "commentary":
        return None, None
    if entry.get("type") == "event_msg" and payload.get("type") == "agent_message":
        text = payload.get("message")
        if isinstance(text, str):
            return "event", proposed_plan_text_from_agent_text(text)
    if (
        entry.get("type") == "response_item"
        and payload.get("type") == "message"
        and payload.get("role") == "assistant"
    ):
        text = _response_message_text(payload.get("content"))
        return "response", proposed_plan_text_from_agent_text(text)
    return None, None


def _turn_has_agent_response(turn_lines: list[dict[str, Any]]) -> bool:
    for entry in turn_lines:
        source, _ = _proposed_plan_text_from_line(entry)
        if source is not None:
            return True
        payload = entry.get("payload") or {}
        if payload.get("phase") == "commentary":
            continue
        if entry.get("type") == "event_msg" and payload.get("type") == "agent_message":
            return True
    return False


def _turn_is_plan_approval(turn_lines: list[dict[str, Any]]) -> bool:
    for entry in turn_lines:
        if not _is_user_message_line(entry):
            continue
        payload = entry.get("payload") or {}
        if _user_message_text(payload).strip() == _PLAN_APPROVAL_PROMPT:
            return True
    return False


def _default_turn_exits_plan_mode(
    mode: str | None,
    turn_lines: list[dict[str, Any]],
    turn_started_awaiting_plan_approval: bool,
    allow_plan_mode_followup: bool,
) -> bool:
    if mode != _COLLABORATION_MODE_DEFAULT:
        return False
    if _turn_is_plan_approval(turn_lines):
        return True
    if turn_started_awaiting_plan_approval:
        return False
    # Older active Plan Mode follow-ups can appear as default turns. If that
    # one-turn fallback explicitly asks for the plan, keep the plan parser open.
    return not (allow_plan_mode_followup and _turn_requests_plan(turn_lines))


def _turn_requests_plan(turn_lines: list[dict[str, Any]]) -> bool:
    for entry in turn_lines:
        if not _is_user_message_line(entry):
            continue
        payload = entry.get("payload") or {}
        if re.search(r"\bplan\b", _user_message_text(payload), re.IGNORECASE):
            return True
    return False


def _represented_agent_texts_by_turn(
    lines: list[dict[str, Any]],
) -> dict[int, Counter[AgentDedupeKey]]:
    by_turn: dict[int, Counter[AgentDedupeKey]] = {}
    for turn_idx, entry in _lines_with_turn_indices(lines):
        if entry.get("type") != "event_msg":
            continue
        payload = entry.get("payload") or {}
        if payload.get("type") != "agent_message":
            continue
        text = payload.get("message")
        if not isinstance(text, str):
            continue
        text, _ = _strip_memory_citations(text)
        text = text.strip()
        if text:
            by_turn.setdefault(turn_idx, Counter())[_agent_dedupe_key(text, payload)] += 1
    return by_turn


def _completed_plan_texts_by_turn(lines: list[dict[str, Any]]) -> dict[int, set[str]]:
    by_turn: dict[int, set[str]] = {}
    for turn_idx, entry in _lines_with_turn_indices(lines):
        if entry.get("type") != "event_msg":
            continue
        payload = entry.get("payload") or {}
        if payload.get("type") != "item_completed":
            continue
        plan_text = _completed_plan_text(payload)
        if plan_text is not None:
            by_turn.setdefault(turn_idx, set()).add(plan_text)
    return by_turn


def _memory_citations_by_turn_text(lines: list[dict[str, Any]]) -> MemoryCitationsByKey:
    by_turn: MemoryCitationsByKey = {}
    for turn_idx, entry in _lines_with_turn_indices(lines):
        if entry.get("type") != "response_item":
            continue
        payload = entry.get("payload") or {}
        if payload.get("type") != "message" or payload.get("role") != "assistant":
            continue
        raw_text = _response_message_text(payload.get("content"))
        if not raw_text:
            continue
        stripped_text, citation = _strip_memory_citations(raw_text)
        if citation is None:
            continue
        text = _response_agent_text(stripped_text, set())
        if not text:
            continue
        key = _agent_dedupe_key(text, payload)
        by_turn.setdefault(turn_idx, {}).setdefault(key, []).append(citation)
    return by_turn


def _agent_entry_from_response_message(
    payload: dict[str, Any],
    timestamp: int | None,
    represented_agent_texts: Counter[AgentDedupeKey],
    completed_plan_texts: set[str],
    proposed_plan_texts: set[str],
    event_plan_texts: set[str],
) -> dict[str, Any] | None:
    if payload.get("role") != "assistant":
        return None
    raw_text = _response_message_text(payload.get("content"))
    if not raw_text:
        return None
    stripped_text, memory_citation = _strip_memory_citations(raw_text)
    plan_text = _proposed_plan_text(stripped_text.strip())
    if plan_text is not None and plan_text in event_plan_texts:
        return None
    if plan_text is not None and plan_text in proposed_plan_texts:
        if plan_text in completed_plan_texts:
            return None
        return {"kind": "plan", "text": plan_text, "timestamp": timestamp}
    text = _response_agent_text(stripped_text, completed_plan_texts)
    if not text:
        return None
    key = _agent_dedupe_key(text, payload)
    if represented_agent_texts[key] > 0:
        represented_agent_texts[key] -= 1
        return None
    phase = payload.get("phase")
    entry: dict[str, Any] = {
        "kind": "agent",
        "text": text,
        "timestamp": timestamp,
        "phase": phase if isinstance(phase, str) else None,
    }
    if memory_citation is not None:
        entry["memory_citation"] = memory_citation
    return entry


def _pop_memory_citation(
    memory_citations_by_text: dict[AgentDedupeKey, list[MemoryCitation]],
    key: AgentDedupeKey,
) -> MemoryCitation | None:
    citations = memory_citations_by_text.get(key)
    if not citations:
        return None
    citation = citations.pop(0)
    if not citations:
        del memory_citations_by_text[key]
    return citation


def _response_message_text(content: Any) -> str:
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
    return "".join(parts)


def _response_agent_text(text: str, completed_plan_texts: set[str]) -> str:
    text = text.strip()
    plan_text = _proposed_plan_text(text)
    if plan_text in completed_plan_texts:
        return ""
    return text


def _strip_memory_citations(text: str) -> tuple[str, MemoryCitation | None]:
    parts: list[str] = []
    citations: list[str] = []
    cursor = 0
    while True:
        start = text.find(_MEMORY_CITATION_OPEN_TAG, cursor)
        if start == -1:
            parts.append(text[cursor:])
            break
        parts.append(text[cursor:start])
        body_start = start + len(_MEMORY_CITATION_OPEN_TAG)
        end = text.find(_MEMORY_CITATION_CLOSE_TAG, body_start)
        if end == -1:
            citations.append(text[body_start:])
            cursor = len(text)
            break
        citations.append(text[body_start:end])
        cursor = end + len(_MEMORY_CITATION_CLOSE_TAG)
    return "".join(parts), _memory_citation_from_bodies(citations)


def _memory_citation_from_bodies(citations: list[str]) -> MemoryCitation | None:
    entries: list[dict[str, Any]] = []
    thread_ids: list[str] = []
    seen_thread_ids: set[str] = set()
    for citation in citations:
        entries_block = _extract_memory_block(
            citation, "<citation_entries>", "</citation_entries>"
        )
        if entries_block is not None:
            entries.extend(
                entry
                for line in entries_block.splitlines()
                if (entry := _parse_memory_citation_entry(line)) is not None
            )
        ids_block = _extract_ids_block(citation)
        if ids_block is None:
            continue
        for thread_id in (line.strip() for line in ids_block.splitlines()):
            if thread_id and thread_id not in seen_thread_ids:
                seen_thread_ids.add(thread_id)
                thread_ids.append(thread_id)

    if not entries and not thread_ids:
        return None
    # ``count`` drives the "Memories used: N" summary in the session view,
    # which expands to a popover listing BOTH ``entries`` and ``thread_ids``.
    # A single citation block can carry both kinds (file-line refs and
    # prior-session refs) at once, so the summary must sum across them or
    # the count will silently undershoot what the popover actually shows.
    return {
        "count": len(entries) + len(thread_ids),
        "entries": entries,
        "thread_ids": thread_ids,
    }


def _parse_memory_citation_entry(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None
    try:
        location, note = line.rsplit("|note=[", 1)
        if not note.endswith("]"):
            return None
        note = note[:-1].strip()
        path, line_range = location.rsplit(":", 1)
        line_start, line_end = line_range.split("-", 1)
        return {
            "path": path.strip(),
            "line_start": int(line_start.strip()),
            "line_end": int(line_end.strip()),
            "note": note,
        }
    except (ValueError, TypeError):
        return None


def _extract_ids_block(text: str) -> str | None:
    rollout_ids = _extract_memory_block(text, "<rollout_ids>", "</rollout_ids>")
    if rollout_ids is not None:
        return rollout_ids
    return _extract_memory_block(text, "<thread_ids>", "</thread_ids>")


def _extract_memory_block(text: str, open_tag: str, close_tag: str) -> str | None:
    try:
        _, rest = text.split(open_tag, 1)
        body, _ = rest.split(close_tag, 1)
    except ValueError:
        return None
    return body


def proposed_plan_text(text: str) -> str | None:
    open_tag = "<proposed_plan>"
    close_tag = "</proposed_plan>"
    if text.startswith(open_tag) and text.endswith(close_tag):
        return text[len(open_tag) : -len(close_tag)].strip()
    return None


def proposed_plan_text_from_agent_text(text: str) -> str | None:
    text, _ = _strip_memory_citations(text)
    return proposed_plan_text(text.strip())


def _proposed_plan_text(text: str) -> str | None:
    return proposed_plan_text(text)


def _should_render_proposed_plan(
    plan_text: str | None,
    is_plan_mode_turn: bool,
    awaiting_plan_approval: bool,
    allow_plan_mode_followup: bool,
    exits_plan_mode: bool,
) -> bool:
    if plan_text is None or exits_plan_mode:
        return False
    if is_plan_mode_turn:
        return True
    if awaiting_plan_approval:
        return True
    return allow_plan_mode_followup and looks_like_plan_text(plan_text)


def looks_like_plan_text(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines()]
    non_empty = [line for line in lines if line]
    if not non_empty:
        return False
    if _looks_like_literal_plan_example(non_empty):
        return False
    if _looks_like_simple_plan_heading(non_empty):
        return True
    if _looks_like_list_plan(non_empty):
        return True
    section_markers = {
        "summary",
        "key changes",
        "test plan",
        "validation",
        "implementation",
        "assumptions",
    }
    matched_sections = 0
    for line in non_empty:
        normalized = line.strip("#*_:- ").lower()
        if normalized in section_markers:
            matched_sections += 1
    return matched_sections >= 2


# Whole-word match keeps real plan headings whose words happen to embed a
# marker -- "Tagging strategy", "Stage rollout", "Vintage cleanup" -- from
# being downgraded to plain agent entries by the literal-example fallback.
_LITERAL_PLAN_EXAMPLE_MARKERS_RE = re.compile(
    r"\b(?:example|xml|tag|syntax|literal)\b"
)


def _looks_like_literal_plan_example(lines: list[str]) -> bool:
    heading = lines[0].strip("#*_:- ").lower()
    return _LITERAL_PLAN_EXAMPLE_MARKERS_RE.search(heading) is not None


def _looks_like_simple_plan_heading(lines: list[str]) -> bool:
    if len(lines) < 2:
        return False
    heading = lines[0].strip("#*_:- ").lower()
    if not heading:
        return False
    return heading == "plan" or heading.endswith(" plan")


def _looks_like_list_plan(lines: list[str]) -> bool:
    list_items = 0
    for line in lines:
        if re.match(r"(?:[-*]|\d+[.)])\s+\S", line):
            list_items += 1
    return list_items >= 2


def _plan_entry_from_completed_item(
    payload: dict[str, Any], timestamp: int | None
) -> dict[str, Any] | None:
    text = _completed_plan_text(payload)
    if text is None:
        return None
    return {
        "kind": "plan",
        "text": text,
        "timestamp": timestamp,
    }


def _completed_plan_text(payload: dict[str, Any]) -> str | None:
    item = payload.get("item")
    if not isinstance(item, dict):
        return None
    item_type = item.get("type")
    if not isinstance(item_type, str) or item_type.lower() != "plan":
        return None
    text = item.get("text")
    if not isinstance(text, str):
        return None
    text = text.strip()
    return text or None


def _agent_dedupe_key(text: str, payload: dict[str, Any]) -> AgentDedupeKey:
    return (text, "commentary" if payload.get("phase") == "commentary" else "answer")


def _tool_call(
    type_: str,
    label: str,
    detail: str,
    status: str | None,
    timestamp: int | None,
) -> dict[str, Any]:
    return {
        "kind": "tool_call",
        "type": type_,
        "label": label,
        "detail": detail,
        "status": status,
        "timestamp": timestamp,
    }


def _approval_declined_entry(
    command: str,
    output_payload: dict[str, Any] | None,
    timestamp: int | None,
) -> dict[str, Any] | None:
    reason = _approval_rejection_reason(output_payload)
    if reason is None or not command:
        return None
    return {
        "kind": "approval_declined",
        "detail": command,
        "rationale": reason,
        "timestamp": timestamp,
    }


def _extract_command(arguments: Any) -> str:
    if not isinstance(arguments, str) or not arguments:
        return ""
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError:
        return arguments
    if not isinstance(args, dict):
        return arguments
    # `exec_command` uses `cmd`; the legacy `shell`/`shell_command` tools use
    # `command`. Codex's `shell` tool spec and `container.exec` carry the
    # command as an argv-style array (see ``core/src/tools/handlers/shell_spec.rs``
    # in codex-rs) -- mirroring the payload ``local_shell_call`` uses -- so
    # join those parts into the same single-line detail the local-shell
    # renderer produces; otherwise every shell invocation routed through this
    # function-call path would surface as a ``Command:`` row with no detail.
    value = args.get("cmd") or args.get("command")
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(str(part) for part in value)
    return ""


def _function_call_status(output_payload: dict[str, Any] | None) -> str | None:
    if output_payload is None:
        return "inProgress"
    if _approval_rejection_reason(output_payload) is not None:
        return "declined"
    output = output_payload.get("output")
    # `exec_command` returns a JSON string matching `unified_exec_output_schema`
    # in shell_spec.rs; a non-zero `exit_code` surfaces as a failure badge.
    if isinstance(output, str):
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            exit_code = parsed.get("exit_code")
            if isinstance(exit_code, int) and exit_code != 0:
                return "failed"
    return None


def _approval_rejection_reason(output_payload: dict[str, Any] | None) -> str | None:
    if output_payload is None:
        return None
    output = output_payload.get("output")
    if not isinstance(output, str):
        return None
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict) and "exit_code" in parsed:
        return None
    wrapper_prefix = next(
        (prefix for prefix in _SHELL_FAILURE_PREFIXES if output.startswith(prefix)),
        None,
    )
    if wrapper_prefix is None:
        return None
    wrapper_end = output.find("`: CreateProcess", len(wrapper_prefix))
    if wrapper_end == -1:
        return None
    rejection_scope = output[wrapper_end + len("`: ") :]
    rejection_start = rejection_scope.find("Rejected(")
    if rejection_start == -1:
        rejection_start = rejection_scope.find("This action was rejected")
    if rejection_start == -1:
        return None
    rejection = rejection_scope[rejection_start:]
    marker = "Reason:"
    if marker not in rejection:
        return "Action was rejected by the approval policy."
    reason = rejection.split(marker, 1)[1].lstrip()
    for separator in ("\\\\n", "\\n", "\n"):
        if separator in reason:
            reason = reason.split(separator, 1)[0]
            break
    return reason.strip().strip('"')


def _non_completed_status(value: Any) -> str | None:
    if isinstance(value, str) and value and value != "completed":
        return value
    return None


def _user_message_text(payload: dict[str, Any]) -> str:
    """Render a user_message payload to its plain-text UI form.

    The rollout's `message` field already inlines `@mention` / `/skill` text,
    but `images` and `local_images` live alongside as separate arrays; append
    generic image markers so image-only or mixed multimodal prompts don't
    render blank without leaking server-side attachment paths.
    """
    parts: list[str] = []
    message = payload.get("message")
    if isinstance(message, str) and message:
        parts.append(message)
    images = payload.get("images")
    if isinstance(images, list):
        for _ in images:
            parts.append("[image]")
    local_images = payload.get("local_images")
    if isinstance(local_images, list):
        for _ in local_images:
            parts.append("[image]")
    return "\n".join(parts)


def _mcp_end_status(result: Any) -> str | None:
    """Map an `mcp_tool_call_end` result payload to a UI status string.

    Codex serialises `result: Result<CallToolResult, String>` as the
    externally-tagged enum `{"Ok": {...}}` or `{"Err": "..."}`. An `Err`
    means the call failed before producing a tool result; an `Ok` whose
    `is_error` is true means the tool reported a tool-level failure. The
    event itself carries no top-level status field, so reading `result` is
    the only way to surface a failure badge.
    """
    if not isinstance(result, dict):
        return None
    if "Err" in result:
        return "failed"
    ok = result.get("Ok")
    if isinstance(ok, dict) and ok.get("is_error") is True:
        return "failed"
    return None


def _format_paths(paths: list[Any]) -> str:
    text_paths = [str(p) for p in paths if p]
    if not text_paths:
        return ""
    if len(text_paths) == 1:
        return text_paths[0]
    return f"{text_paths[0]} (+{len(text_paths) - 1} more)"


def _iso_to_unix_seconds(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp())
