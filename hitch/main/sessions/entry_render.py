"""Collapse a Codex thread's turns/items into the entries the thread template
renders.

Each turn's user message and final agent reply become top-level entries; all
intermediate agent commentary and tool calls fold into a single collapsible
"intermediate" entry so long sessions don't bury the answer. The same shaping is
applied to the flat per-item entries produced by the rollout parser
(``collapse_flat_entries``) so resumed/replayed threads render identically.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from hitch.main.formatting import looks_like_markdown, render_markdown
from hitch.main.sdk_values import int_value, sequence_value, string_value, value_for

_NON_MESSAGE_LABELS = {
    "commandExecution": "Command",
    "mcpToolCall": "MCP tool call",
    "dynamicToolCall": "Tool call",
    "fileChange": "File change",
    "webSearch": "Web search",
    "collabAgentToolCall": "Collab agent call",
    "imageView": "Image view",
    "imageGeneration": "Image generation",
    "reasoning": "Reasoning",
    "plan": "Plan",
    "hookPrompt": "Hook prompt",
    "enteredReviewMode": "Entered review mode",
    "exitedReviewMode": "Exited review mode",
    "contextCompaction": "Context compaction",
}


def collapse_flat_entries(flat: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Apply the same intermediate-collapsing as ``render_entries`` to the
    flat per-item entries produced by the rollout parser.

    Turn boundaries are detected via the user kind because the rollout file
    is a chronological log without per-item turn ids exposed at the entry
    level. This matches the SDK's `handle_user_message` fallback for streams
    that did not open turns explicitly.
    """
    turn: list[dict[str, Any]] = []
    for entry in flat:
        if entry["kind"] == "user" and turn:
            yield from _emit_collapsed_turn(turn)
            turn = []
        turn.append(entry)
    if turn:
        yield from _emit_collapsed_turn(turn)


def _emit_collapsed_turn(turn: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    final_idx = _final_agent_idx_in_flat(turn)
    intermediate: list[dict[str, Any]] = []
    for i, entry in enumerate(turn):
        if i == final_idx:
            if intermediate:
                yield _make_intermediate_entry(intermediate)
                intermediate = []
            yield _finalize_agent_entry(_strip_phase(entry))
        elif entry["kind"] == "user":
            # `collapse_flat_entries` splits on every user past the first, so
            # any user reaching this branch is the leading entry of the turn
            # and intermediate is empty.
            yield entry
        elif entry["kind"] == "agent":
            intermediate.append({**_strip_phase(entry), "kind": "thinking"})
        elif entry["kind"] in {"approval_declined", "plan"}:
            if intermediate:
                yield _make_intermediate_entry(intermediate)
                intermediate = []
            yield entry
        else:
            intermediate.append(entry)
    if intermediate:
        yield _make_intermediate_entry(intermediate)


def _final_agent_idx_in_flat(entries: list[dict[str, Any]]) -> int:
    for i in range(len(entries) - 1, -1, -1):
        entry = entries[i]
        if entry["kind"] == "agent" and entry.get("phase") == "final_answer":
            return i
    for i in range(len(entries) - 1, -1, -1):
        entry = entries[i]
        if entry["kind"] != "agent":
            continue
        if entry.get("phase") == "commentary":
            continue
        return i
    return -1


def _strip_phase(entry: dict[str, Any]) -> dict[str, Any]:
    """Return ``entry`` minus its ``phase`` key.

    Called on every agent entry just before it leaves the collapsing pass so
    the template never sees the internal phase marker. The rollout parser
    sets ``phase`` on every agent entry (to ``None`` when absent), so this
    function never has to consult the dict before stripping.
    """
    return {k: v for k, v in entry.items() if k != "phase"}


def _finalize_agent_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Annotate a turn's final agent entry with rendered-markdown HTML.

    Only the final agent message of a turn is ever fed through this
    function, so collapsed "thinking" messages stay plain-text. A failure
    to recognise the text as markdown leaves the entry untouched, and the
    template falls back to the plain-text body.
    """
    text = entry.get("text")
    if isinstance(text, str) and looks_like_markdown(text):
        entry["html"] = render_markdown(text)
    return entry


def render_entries(thread: Any) -> Iterator[dict[str, Any]]:
    """Walk every turn's items in order, surfacing the user message and the
    final agent reply as top-level entries and folding everything else
    (intermediate agent commentary plus every tool-call variant) into a
    single collapsible "intermediate" entry per run so long sessions don't
    bury the actual answer behind dozens of tool rows.

    The SDK marks final responses with MessagePhase.final_answer when known;
    for sessions where phase is unset (older data or an in-progress turn)
    the last agentMessage in the turn is treated as final. Each entry
    carries the turn's started_at timestamp; per-item timestamps are not
    exposed by the SDK.
    """
    for turn in getattr(thread, "turns", []) or []:
        timestamp = getattr(turn, "started_at", None)
        items = [thread_item.root for thread_item in turn.items]
        final_idx = find_final_agent_idx(items)
        intermediate: list[dict[str, Any]] = []

        for i, item in enumerate(items):
            if i == final_idx:
                if intermediate:
                    yield _make_intermediate_entry(intermediate)
                    intermediate = []
                agent_entry: dict[str, Any] = {
                    "kind": "agent",
                    "text": item.text,
                    "timestamp": timestamp,
                }
                memory_citation = _memory_citation_from_item(item)
                if memory_citation is not None:
                    agent_entry["memory_citation"] = memory_citation
                yield _finalize_agent_entry(agent_entry)
            elif item.type == "userMessage":
                if intermediate:
                    yield _make_intermediate_entry(intermediate)
                    intermediate = []
                yield {
                    "kind": "user",
                    "text": user_message_text(item),
                    "timestamp": timestamp,
                }
            elif item.type == "agentMessage":
                thinking_entry: dict[str, Any] = {
                    "kind": "thinking",
                    "text": item.text,
                    "timestamp": timestamp,
                }
                memory_citation = _memory_citation_from_item(item)
                if memory_citation is not None:
                    thinking_entry["memory_citation"] = memory_citation
                intermediate.append(thinking_entry)
            elif item.type == "plan":
                if intermediate:
                    yield _make_intermediate_entry(intermediate)
                    intermediate = []
                yield {
                    "kind": "plan",
                    "text": getattr(item, "text", "") or "",
                    "timestamp": timestamp,
                }
            else:
                intermediate.append(_make_tool_call_entry(item, timestamp))

        if intermediate:
            yield _make_intermediate_entry(intermediate)


def find_final_agent_idx(items: list[Any]) -> int:
    """Index of the agent message to display as this turn's final response, or
    -1 if there is no agent message that could be the final.

    An explicit MessagePhase.final_answer always wins (the last one if
    multiple). Otherwise the last agent message whose phase is not
    MessagePhase.commentary is treated as final — i.e. unset phases (older
    sessions / in-progress turns) are eligible to be the final, but explicit
    commentary never is.
    """
    for i in range(len(items) - 1, -1, -1):
        item = items[i]
        if item.type == "agentMessage" and _phase_value(item) == "final_answer":
            return i
    for i in range(len(items) - 1, -1, -1):
        item = items[i]
        if item.type != "agentMessage":
            continue
        if _phase_value(item) == "commentary":
            continue
        return i
    return -1


def _phase_value(item: Any) -> str | None:
    # The SDK normally deserializes `phase` into a MessagePhase enum instance
    # (with `.value`), but accept a raw wire string too so this stays robust
    # against thread data that bypasses pydantic deserialization.
    phase = getattr(item, "phase", None)
    if phase is None:
        return None
    if isinstance(phase, str):
        return phase
    value = getattr(phase, "value", None)
    return value if isinstance(value, str) else None


def _memory_citation_from_item(item: Any) -> dict[str, Any] | None:
    value = value_for(item, "memory_citation")
    if value is None:
        value = value_for(item, "memoryCitation")
    if value is None:
        return None

    entries: list[dict[str, Any]] = []
    for raw_entry in sequence_value(value_for(value, "entries")):
        path = string_value(value_for(raw_entry, "path"))
        line_start = int_value(value_for(raw_entry, "line_start"))
        if line_start == 0:
            line_start = int_value(value_for(raw_entry, "lineStart"))
        line_end = int_value(value_for(raw_entry, "line_end"))
        if line_end == 0:
            line_end = int_value(value_for(raw_entry, "lineEnd"))
        if not path or line_start == 0 or line_end == 0:
            continue
        entries.append(
            {
                "path": path,
                "line_start": line_start,
                "line_end": line_end,
                "note": string_value(value_for(raw_entry, "note")),
            }
        )

    thread_ids = [
        thread_id
        for raw_id in sequence_value(
            value_for(value, "thread_ids") or value_for(value, "threadIds")
        )
        if (thread_id := string_value(raw_id))
    ]
    # See _memory_citation_from_bodies in rollout.py: ``count`` covers both
    # citation kinds because the popover renders entries and thread_ids
    # together; counting only one half would silently underreport.
    count = len(entries) + len(thread_ids)
    if count == 0:
        return None
    return {"count": count, "entries": entries, "thread_ids": thread_ids}


def _make_tool_call_entry(item: Any, timestamp: Any) -> dict[str, Any]:
    item_type = item.type
    return {
        "kind": "tool_call",
        "type": item_type,
        "label": _NON_MESSAGE_LABELS.get(item_type, item_type),
        "detail": tool_call_detail(item, item_type),
        "status": tool_call_status(item),
        "timestamp": timestamp,
    }


def _make_intermediate_entry(items: list[dict[str, Any]]) -> dict[str, Any]:
    thinking_count = sum(1 for e in items if e["kind"] == "thinking")
    tool_call_count = sum(1 for e in items if e["kind"] == "tool_call")
    return {
        "kind": "intermediate",
        "thinking_count": thinking_count,
        "tool_call_count": tool_call_count,
        "items": items,
    }


def user_message_text(item: Any) -> str:
    parts: list[str] = []
    for input_item in item.content:
        inner = input_item.root
        match inner.type:
            case "text":
                if inner.text:
                    parts.append(inner.text)
            case "mention":
                parts.append(f"@{inner.name}")
            case "skill":
                parts.append(f"/{inner.name}")
            case "image":
                parts.append("[image]")
            case "localImage":
                parts.append("[image]")
    return "\n".join(parts)


def tool_call_detail(item: Any, item_type: str) -> str:
    """Return a short, human-readable description of a tool-call item.

    Returns an empty string for item types that do not carry useful inline
    detail; the label alone is enough to surface them in the UI.
    """
    match item_type:
        case "commandExecution":
            return getattr(item, "command", "") or ""
        case "mcpToolCall":
            return f"{item.server} / {item.tool}"
        case "dynamicToolCall":
            namespace = getattr(item, "namespace", None)
            return f"{namespace}::{item.tool}" if namespace else item.tool
        case "fileChange":
            paths = [str(change.path) for change in getattr(item, "changes", []) or []]
            if not paths:
                return ""
            if len(paths) == 1:
                return paths[0]
            return f"{paths[0]} (+{len(paths) - 1} more)"
        case "webSearch":
            return getattr(item, "query", "") or ""
        case "plan":
            text = getattr(item, "text", "") or ""
            return text.split("\n", 1)[0]
        case "imageView":
            return getattr(item, "path", "") or ""
        case "imageGeneration":
            return (
                getattr(item, "revised_prompt", None)
                or getattr(item, "saved_path", None)
                or ""
            )
        case "collabAgentToolCall":
            tool = getattr(item, "tool", None)
            tool_name = getattr(tool, "value", None) or (tool if isinstance(tool, str) else "")
            receivers = getattr(item, "receiver_thread_ids", None) or []
            if receivers:
                return f"{tool_name} → {receivers[0]}"
            return str(tool_name)
        case _:
            return ""


def tool_call_status(item: Any) -> str | None:
    """Return a non-success status string (e.g. ``failed``) or None.

    Hides ``completed`` so the common case stays uncluttered; surfaces
    in-progress, failed, and declined so unusual outcomes are visible.
    """
    status = getattr(item, "status", None)
    if status is None:
        return None
    value = getattr(status, "value", status)
    if not isinstance(value, str) or value == "completed":
        return None
    return value
