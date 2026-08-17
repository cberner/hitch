"""Shape a Codex thread's turns/items for the session transcript.

User, final-agent, and Thinking messages remain top-level entries. Consecutive
runs of command, reasoning, and web-search entries are grouped so the transcript
can show only the latest activity by default without hiding the agent's
narration. The same shaping is applied to rollout-parser entries and SDK
fallback entries.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from hitch.main.formatting import looks_like_markdown, render_markdown
from hitch.main.runtime.sdk_values import int_value, sequence_value, string_value, value_for

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

_COLLAPSIBLE_ACTIVITY_TYPES = {"commandExecution", "reasoning", "webSearch"}


def collapse_flat_entries(flat: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Apply the same activity grouping as ``render_entries`` to rollout entries.

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
    activity: list[dict[str, Any]] = []
    for i, entry in enumerate(turn):
        if i == final_idx:
            display_entry = _finalize_agent_entry(_strip_phase(entry))
        elif entry["kind"] == "user":
            # `collapse_flat_entries` splits on every user past the first, so
            # any user reaching this branch is the leading entry of the turn
            # and activity is empty.
            display_entry = entry
        elif entry["kind"] == "agent":
            display_entry = {**_strip_phase(entry), "kind": "thinking"}
        else:
            display_entry = entry

        if _is_collapsible_activity(display_entry):
            activity.append(display_entry)
            continue
        yield from _emit_activity_run(activity)
        activity = []
        yield display_entry
    yield from _emit_activity_run(activity)


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
    function, so Thinking messages stay plain-text. A failure
    to recognise the text as markdown leaves the entry untouched, and the
    template falls back to the plain-text body.
    """
    text = entry.get("text")
    if isinstance(text, str) and looks_like_markdown(text):
        entry["html"] = render_markdown(text)
    return entry


def render_entries(thread: Any) -> Iterator[dict[str, Any]]:
    """Walk every turn's items in order and group repetitive activity.

    User messages, final replies, Thinking messages, plans, and tool calls other
    than commands/reasoning/web searches remain top-level. Consecutive activity
    runs are grouped only when there is an earlier item to hide.

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
        activity: list[dict[str, Any]] = []

        for i, item in enumerate(items):
            if i == final_idx:
                agent_entry: dict[str, Any] = {
                    "kind": "agent",
                    "text": item.text,
                    "timestamp": timestamp,
                }
                memory_citation = _memory_citation_from_item(item)
                if memory_citation is not None:
                    agent_entry["memory_citation"] = memory_citation
                display_entry = _finalize_agent_entry(agent_entry)
            elif item.type == "userMessage":
                display_entry = {
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
                display_entry = thinking_entry
            elif item.type == "plan":
                display_entry = {
                    "kind": "plan",
                    "text": getattr(item, "text", "") or "",
                    "timestamp": timestamp,
                }
            else:
                display_entry = _make_tool_call_entry(item, timestamp)

            if _is_collapsible_activity(display_entry):
                activity.append(display_entry)
                continue
            yield from _emit_activity_run(activity)
            activity = []
            yield display_entry

        yield from _emit_activity_run(activity)


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


def _is_collapsible_activity(entry: dict[str, Any]) -> bool:
    return (
        entry.get("kind") == "tool_call"
        and entry.get("type") in _COLLAPSIBLE_ACTIVITY_TYPES
    )


def _emit_activity_run(
    items: list[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    if len(items) == 1:
        yield items[0]
    elif items:
        yield _make_intermediate_entry(items)


def _make_intermediate_entry(items: list[dict[str, Any]]) -> dict[str, Any]:
    reasoning_count = sum(1 for entry in items if entry["type"] == "reasoning")
    command_count = sum(
        1 for entry in items if entry["type"] == "commandExecution"
    )
    web_search_count = sum(1 for entry in items if entry["type"] == "webSearch")
    return {
        "kind": "intermediate",
        "summary": _activity_summary(
            reasoning_count, command_count, web_search_count
        ),
        "reasoning_count": reasoning_count,
        "command_count": command_count,
        "web_search_count": web_search_count,
        "item_count": len(items),
        "items": items,
        "earlier_items": items[:-1],
        "latest_item": items[-1],
    }


def _activity_summary(
    reasoning_count: int, command_count: int, web_search_count: int
) -> str:
    parts: list[str] = []
    if reasoning_count:
        suffix = "" if reasoning_count == 1 else "s"
        parts.append(f"{reasoning_count} reasoning message{suffix}")
    if command_count:
        suffix = "" if command_count == 1 else "s"
        parts.append(f"{command_count} command message{suffix}")
    if web_search_count:
        suffix = "" if web_search_count == 1 else "es"
        parts.append(f"{web_search_count} web search{suffix}")
    if len(parts) < 3:
        return " and ".join(parts)
    return f"{', '.join(parts[:-1])}, and {parts[-1]}"


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
        case "reasoning":
            for field in ("summary", "content"):
                for value in sequence_value(value_for(item, field)):
                    if text := string_value(value):
                        return text.split("\n", 1)[0]
            return ""
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
