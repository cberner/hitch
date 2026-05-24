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
    """Return latest token counts and context window for a thread.

    Codex emits a `TokenCount` event_msg after each turn whose
    `info.total_token_usage` is the running session total and whose
    `info.last_token_usage` is the latest active context size. Only the
    most recent such event is kept; earlier ones are obsoleted by it.
    Returns None when the rollout is unreadable or contains no parseable
    token_count event (e.g. a session that has yet to receive a response).
    """
    try:
        text = rollout_path.read_text()
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("failed to read rollout %s: %s", rollout_path, exc)
        return None
    latest: dict[str, Any] | None = None
    latest_context: dict[str, Any] = {}
    latest_context_window: int = 0
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


def token_usage_history(rollout_path: Path) -> list[dict[str, int]]:
    """Return cumulative token counts for every timestamped token_count event."""
    try:
        text = rollout_path.read_text()
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("failed to read rollout %s: %s", rollout_path, exc)
        return []
    history: list[dict[str, int]] = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
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


def _plan_mode_turns(lines: list[dict[str, Any]]) -> set[int]:
    plan_turns: set[int] = set()
    pending_plan_mode = False
    turn_idx = -1
    current_turn_accepts_mode = False
    for entry in lines:
        if entry.get("type") == "turn_context":
            is_plan = _turn_context_is_plan(entry)
            if current_turn_accepts_mode:
                _set_plan_turn_mode(plan_turns, turn_idx, is_plan)
            else:
                pending_plan_mode = is_plan
            continue
        if _is_task_started_line(entry):
            mode = _task_started_collaboration_mode(entry)
            if mode is not None:
                is_plan = mode == "plan"
                if current_turn_accepts_mode:
                    _set_plan_turn_mode(plan_turns, turn_idx, is_plan)
                else:
                    pending_plan_mode = is_plan
            continue
        if _is_user_message_line(entry):
            turn_idx += 1
            current_turn_accepts_mode = True
            if pending_plan_mode:
                plan_turns.add(turn_idx)
            pending_plan_mode = False
            continue
        current_turn_accepts_mode = False
    return plan_turns


def _set_plan_turn_mode(plan_turns: set[int], turn_idx: int, is_plan: bool) -> None:
    if is_plan:
        plan_turns.add(turn_idx)
    else:
        plan_turns.discard(turn_idx)


def _turn_context_is_plan(entry: dict[str, Any]) -> bool:
    payload = entry.get("payload") or {}
    mode_data = payload.get("collaboration_mode") or payload.get("collaborationMode")
    if not isinstance(mode_data, dict):
        return False
    return mode_data.get("mode") == "plan"


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
    awaiting_plan_approval = False
    for turn_idx, turn_lines in _lines_by_turn(lines):
        event_turn_texts: set[str] = set()
        response_turn_texts: set[str] = set()
        for entry in turn_lines:
            source, plan_text = _proposed_plan_text_from_line(entry)
            if plan_text is None or not _should_render_proposed_plan(
                plan_text,
                turn_idx in plan_mode_turns,
                awaiting_plan_approval,
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
        if completed_plan_texts_by_turn.get(turn_idx) or turn_texts:
            awaiting_plan_approval = True
        elif _turn_has_agent_response(turn_lines):
            awaiting_plan_approval = False
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
            text, _ = _strip_memory_citations(text)
            return "event", _proposed_plan_text(text.strip())
    if (
        entry.get("type") == "response_item"
        and payload.get("type") == "message"
        and payload.get("role") == "assistant"
    ):
        text, _ = _strip_memory_citations(_response_message_text(payload.get("content")))
        return "response", _proposed_plan_text(text.strip())
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
    return {
        "count": len(entries) if entries else len(thread_ids),
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


def _proposed_plan_text(text: str) -> str | None:
    open_tag = "<proposed_plan>"
    close_tag = "</proposed_plan>"
    if text.startswith(open_tag) and text.endswith(close_tag):
        return text[len(open_tag) : -len(close_tag)].strip()
    return None


def _should_render_proposed_plan(
    plan_text: str | None,
    is_plan_mode_turn: bool,
    awaiting_plan_approval: bool,
) -> bool:
    if plan_text is None:
        return False
    if is_plan_mode_turn:
        return True
    return awaiting_plan_approval


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
