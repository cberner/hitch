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

    for entry in lines:
        result = _entry_for_rollout_line(entry, outputs)
        if result is not None:
            yield result


def _entry_for_rollout_line(
    line: dict[str, Any],
    outputs: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    line_type = line.get("type")
    payload = line.get("payload") or {}
    timestamp = _iso_to_unix_seconds(line.get("timestamp"))

    if line_type == "event_msg":
        return _entry_from_event(payload, timestamp)
    if line_type == "response_item":
        return _entry_from_response_item(payload, timestamp, outputs)
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
        text = payload.get("message") or ""
        if not text:
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
) -> dict[str, Any] | None:
    item_type = payload.get("type")
    if item_type == "function_call":
        name = payload.get("name") or ""
        if name not in _SHELL_TOOL_NAMES:
            return None
        command = _extract_command(payload.get("arguments"))
        call_id = payload.get("call_id")
        output_payload = outputs.get(call_id) if isinstance(call_id, str) else None
        return _tool_call(
            "commandExecution",
            "Command",
            command,
            _function_call_status(output_payload),
            timestamp,
        )
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
