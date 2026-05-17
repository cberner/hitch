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


def iter_entries(rollout_path: Path) -> Iterator[dict[str, Any]]:
    """Yield session-view entries directly from a codex rollout JSONL file.

    Dict shape matches `views._render_entries` so the template renders it
    without changes. Entries are emitted in rollout-file order, which is the
    same chronological order codex used when writing them.
    """
    try:
        text = rollout_path.read_text()
    except OSError as exc:
        logger.warning("failed to read rollout %s: %s", rollout_path, exc)
        return
    yield from _entries_from_text(text, rollout_path)


def latest_token_usage(rollout_path: Path) -> dict[str, int] | None:
    """Return cumulative input/cached/output token counts for a thread.

    Codex emits a `TokenCount` event_msg after each turn whose
    `info.total_token_usage` is the running session total. Only the most
    recent such event is kept; earlier ones are obsoleted by it. Returns
    None when the rollout is unreadable or contains no parseable
    token_count event (e.g. a session that has yet to receive a response).
    """
    try:
        text = rollout_path.read_text()
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("failed to read rollout %s: %s", rollout_path, exc)
        return None
    latest: dict[str, Any] | None = None
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
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
    if latest is None:
        return None
    return {
        "input_tokens": _coerce_int(latest.get("input_tokens")),
        "cached_input_tokens": _coerce_int(latest.get("cached_input_tokens")),
        "output_tokens": _coerce_int(latest.get("output_tokens")),
    }


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


Entry = dict[str, Any]
EntryResult = Entry | list[Entry] | None
AgentDedupeKey = tuple[str, str]


def _entries_from_text(text: str, rollout_path: Path) -> Iterator[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            lines.append(json.loads(raw))
        except json.JSONDecodeError:
            logger.debug("skipping malformed rollout line in %s", rollout_path)

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
    completed_plan_texts_by_turn = _completed_plan_texts_by_turn(lines)
    for turn_idx, entry in _lines_with_turn_indices(lines):
        represented_agent_texts = represented_agent_texts_by_turn.get(turn_idx, Counter())
        completed_plan_texts = completed_plan_texts_by_turn.get(turn_idx, set())
        result = _entry_for_rollout_line(
            entry,
            outputs,
            represented_agent_texts,
            completed_plan_texts,
        )
        if isinstance(result, list):
            yield from result
        elif result is not None:
            yield result


def _entry_for_rollout_line(
    line: dict[str, Any],
    outputs: dict[str, dict[str, Any]],
    represented_agent_texts: Counter[AgentDedupeKey],
    completed_plan_texts: set[str],
) -> EntryResult:
    line_type = line.get("type")
    payload = line.get("payload") or {}
    timestamp = _iso_to_unix_seconds(line.get("timestamp"))

    if line_type == "event_msg":
        return _entry_from_event(payload, timestamp)
    if line_type == "response_item":
        return _entry_from_response_item(
            payload,
            timestamp,
            outputs,
            represented_agent_texts,
            completed_plan_texts,
        )
    return None


def _entry_from_event(payload: dict[str, Any], timestamp: int | None) -> dict[str, Any] | None:
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
        # `phase` is preserved so the view layer can pick the turn's final
        # agent reply with the same `MessagePhase` semantics as the SDK
        # (final_answer wins, commentary never wins, unset is eligible).
        phase = payload.get("phase")
        return {
            "kind": "agent",
            "text": text,
            "timestamp": timestamp,
            "phase": phase if isinstance(phase, str) else None,
        }
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
) -> EntryResult:
    item_type = payload.get("type")
    if item_type == "message":
        return _agent_entry_from_response_message(
            payload,
            timestamp,
            represented_agent_texts,
            completed_plan_texts,
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
        item = payload.get("item")
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        text = item.get("text")
        if not isinstance(item_type, str) or item_type.lower() != "plan":
            continue
        if isinstance(text, str) and text.strip():
            by_turn.setdefault(turn_idx, set()).add(text.strip())
    return by_turn


def _agent_entry_from_response_message(
    payload: dict[str, Any],
    timestamp: int | None,
    represented_agent_texts: Counter[AgentDedupeKey],
    completed_plan_texts: set[str],
) -> dict[str, Any] | None:
    if payload.get("role") != "assistant":
        return None
    text = _response_message_text(payload.get("content"))
    if not text:
        return None
    text = _response_agent_text(text, completed_plan_texts)
    if not text:
        return None
    key = _agent_dedupe_key(text, payload)
    if represented_agent_texts[key] > 0:
        represented_agent_texts[key] -= 1
        return None
    phase = payload.get("phase")
    return {
        "kind": "agent",
        "text": text,
        "timestamp": timestamp,
        "phase": phase if isinstance(phase, str) else None,
    }


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
        return plan_text
    return text


def _proposed_plan_text(text: str) -> str | None:
    open_tag = "<proposed_plan>"
    close_tag = "</proposed_plan>"
    if text.startswith(open_tag) and text.endswith(close_tag):
        return text[len(open_tag) : -len(close_tag)].strip()
    return None


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
    # `command`.
    value = args.get("cmd") or args.get("command")
    return value if isinstance(value, str) else ""


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
    but `images` and `local_images` live alongside as separate arrays; mirror
    the SDK's `_user_message_text` and append `[image]` / `[image: path]`
    markers so image-only or mixed multimodal prompts don't render blank.
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
        for path in local_images:
            parts.append(f"[image: {path}]")
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
