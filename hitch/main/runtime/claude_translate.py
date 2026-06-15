"""Translate Claude Agent SDK messages into Hitch's Codex event schema.

The whole Hitch downstream -- the SSE layer, the ``session.html`` renderer, the
``codex_events`` parsers -- consumes per-worker JSONL lines shaped like Codex
app-server notifications (``{"method": ..., "payload": ...}``). To reuse all of
that for the Claude backend, ``claude_worker`` feeds each SDK message through
:class:`EventTranslator`, which emits the same ``item/started`` /
``item/completed`` / ``turn/plan/updated`` events Codex would.

The translator is intentionally pure and stateful-only-in-memory: it correlates
``ToolUseBlock`` (which opens an item) with the later ``ToolResultBlock`` (which
closes it) by tool-use id, but it does no I/O. ``claude_worker`` owns writing,
ordering, and the approval/turn-completion bookkeeping.
"""

from __future__ import annotations

import json
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

Event = tuple[str, dict[str, Any]]

ITEM_STARTED = "item/started"
ITEM_COMPLETED = "item/completed"
TASK_PLAN_UPDATED = "turn/plan/updated"

_STATUS_COMPLETED = "completed"
_STATUS_FAILED = "failed"

# Tool name -> Codex item ``type``. Anything unlisted is surfaced as a generic
# ``dynamicToolCall`` so unfamiliar tools still appear in the timeline.
_WRITE_TOOLS = frozenset({"Write", "Edit", "MultiEdit", "NotebookEdit"})
# Tools that run a shell command via a ``command`` input and so render as a
# command execution: ``Bash``; ``Monitor`` (a background script under Bash rules);
# and ``PowerShell`` (host commands natively). The worker gates all three as
# command execution, so the transcript must surface their command text too.
_COMMAND_INPUT_TOOLS = frozenset({"Bash", "Monitor", "PowerShell"})
_FILE_PATH_KEYS = ("file_path", "path", "notebook_path")


class EventTranslator:
    """Stateful per-session translator from SDK messages to Codex events."""

    def __init__(self) -> None:
        # Open tool items keyed by tool-use id, awaiting their result block.
        self._open_tools: dict[str, dict[str, Any]] = {}
        # Monotonic per-turn counter to keep assistant item ids unique across
        # messages that share a ``message_id`` (see ``_translate_assistant``).
        self._assistant_message_seq = 0
        # Number of upcoming user-prompt echoes to drop because the worker already
        # emitted that turn's ``userMessage`` itself (see
        # ``suppress_next_user_prompt``).
        self._suppress_user_prompts = 0

    def suppress_next_user_prompt(self) -> None:
        """Drop the next echoed user prompt from the SDK stream.

        The SDK's message parser silently drops image content blocks, so an
        image-bearing user turn echoes back with no trace of the attachment (an
        image-only turn even echoes empty). The worker -- which holds the
        attachment paths -- records such a turn's ``userMessage`` itself (text
        plus an ``[image]`` marker) and calls this so the matching SDK echo is
        skipped, keeping exactly one user entry (and an accurate auto-QA count).
        """
        self._suppress_user_prompts += 1

    def cancel_pending_user_prompt_suppression(self) -> None:
        """Release one armed suppression whose echo will never arrive.

        A steered image turn arms suppression *before* awaiting ``client.query``
        (so it is set before the concurrent receive loop can translate the echo).
        If that query is then rejected, no echo comes to consume the suppression,
        so the worker releases it here -- otherwise it would swallow a later real
        user prompt's echo.
        """
        if self._suppress_user_prompts > 0:
            self._suppress_user_prompts -= 1

    def translate(self, message: Any) -> list[Event]:
        if isinstance(message, AssistantMessage):
            return self._translate_assistant(message)
        if isinstance(message, UserMessage):
            return self._translate_user(message)
        if isinstance(message, ResultMessage):
            return self._translate_result(message)
        return []

    def _translate_result(self, message: ResultMessage) -> list[Event]:
        # With an ``output_schema`` the SDK puts the validated JSON on
        # ``ResultMessage.structured_output`` rather than in an ``agentMessage``
        # text block, so a structured-output turn (QA/spec/autonomous subagents)
        # would otherwise leave no agent text for ``_final_agent_text`` to read.
        # Emit the JSON as a final ``agentMessage`` so the existing
        # events-file parsers recover the structured verdict unchanged.
        structured = message.structured_output
        if structured is None:
            return []
        try:
            text = json.dumps(structured)
        except (TypeError, ValueError):
            return []
        return _complete_text_item("result:structured_output", "agentMessage", text)

    # -- assistant content -------------------------------------------------

    def _translate_assistant(self, message: AssistantMessage) -> list[Event]:
        events: list[Event] = []
        # ``message_id``/``uuid`` exist on the current SDK dataclass; guard with
        # getattr so a differing SDK shape can't crash the whole turn translation.
        # Multiple ``AssistantMessage``s in one turn can share a ``message_id``
        # (per the SDK), so fold in a per-message counter -- otherwise a later
        # message's text/thinking ids would collide with an earlier one's and the
        # live renderer (which updates the existing item on completion) would
        # overwrite pre-tool assistant text with the final response until reload.
        self._assistant_message_seq += 1
        base_id = (
            message.message_id
            or getattr(message, "uuid", None)
            or "assistant"
        )
        base_id = f"{base_id}:{self._assistant_message_seq}"
        for index, block in enumerate(message.content):
            item_id = f"{base_id}:{index}"
            if isinstance(block, TextBlock):
                events.extend(_complete_text_item(item_id, "agentMessage", block.text))
            elif isinstance(block, ThinkingBlock):
                events.extend(
                    _complete_text_item(
                        item_id, "reasoning", block.thinking, phase="commentary"
                    )
                )
            elif isinstance(block, ToolUseBlock):
                events.extend(self._open_tool(block))
        return events

    def _open_tool(self, block: ToolUseBlock) -> list[Event]:
        name = block.name
        if name == "TodoWrite":
            return [_plan_event(block.input)]
        if name == "ExitPlanMode":
            text = _string(block.input.get("plan"))
            return _complete_text_item(block.id, "plan", text)
        item = self._tool_item(block)
        self._open_tools[block.id] = item
        return [(ITEM_STARTED, {"item": dict(item)})]

    def _tool_item(self, block: ToolUseBlock) -> dict[str, Any]:
        name = block.name
        item: dict[str, Any] = {"id": block.id, "status": "inProgress"}
        if name in _COMMAND_INPUT_TOOLS:
            # ``Monitor`` runs a background shell script under Bash permission
            # rules with the same ``command`` input, so surface it (like Bash) as
            # a command execution rather than a generic tool call -- otherwise the
            # transcript records only "Tool call" and the actual command run is
            # not auditable in session history.
            item["type"] = "commandExecution"
            item["command"] = _string(block.input.get("command"))
        elif name in _WRITE_TOOLS:
            item["type"] = "fileChange"
            item["changes"] = [{"path": path} for path in _file_paths(block.input)]
        elif name == "WebSearch":
            item["type"] = "webSearch"
            item["query"] = _string(block.input.get("query"))
        elif name.startswith("mcp__"):
            server, tool = _split_mcp_name(name)
            item["type"] = "mcpToolCall"
            item["server"] = server
            item["tool"] = tool
            item["arguments"] = block.input
        else:
            item["type"] = "dynamicToolCall"
            item["tool"] = name
            item["arguments"] = block.input
        return item

    # -- user content (tool results, echoed prompts) -----------------------

    def _translate_user(self, message: UserMessage) -> list[Event]:
        content = message.content
        # A user-prompt echo carries text (or, for an image-stripped turn, no
        # blocks); tool results arrive as a ``UserMessage`` of ``ToolResultBlock``s
        # and must never be suppressed. When the worker already emitted this turn's
        # ``userMessage`` (an image turn), drop the matching prompt echo.
        is_tool_result = isinstance(content, list) and any(
            isinstance(block, ToolResultBlock) for block in content
        )
        if not is_tool_result and self._suppress_user_prompts > 0:
            self._suppress_user_prompts -= 1
            return []
        if isinstance(content, str):
            return _complete_text_item(_user_item_id(message, 0), "userMessage", content)
        events: list[Event] = []
        for index, block in enumerate(content):
            if isinstance(block, ToolResultBlock):
                events.extend(self._close_tool(block))
            elif isinstance(block, TextBlock):
                events.extend(
                    _complete_text_item(
                        _user_item_id(message, index), "userMessage", block.text
                    )
                )
        return events

    def _close_tool(self, block: ToolResultBlock) -> list[Event]:
        item = self._open_tools.pop(block.tool_use_id, None)
        if item is None:
            return []
        item = dict(item)
        item["status"] = _STATUS_FAILED if block.is_error else _STATUS_COMPLETED
        item["result"] = _result_text(block.content)
        return [(ITEM_COMPLETED, {"item": item})]


def user_message_events(item_id: str, text: str) -> list[Event]:
    """Build the started/completed events for a worker-emitted ``userMessage``.

    Used when the worker records a user turn itself (an image-bearing turn the
    SDK echo can't represent) rather than translating it from the SDK stream.
    """
    return _complete_text_item(item_id, "userMessage", text)


def _complete_text_item(
    item_id: str, item_type: str, text: str, *, phase: str | None = None
) -> list[Event]:
    started: dict[str, Any] = {"id": item_id, "type": item_type}
    completed: dict[str, Any] = {"id": item_id, "type": item_type, "text": text}
    if phase is not None:
        started["phase"] = phase
        completed["phase"] = phase
    return [
        (ITEM_STARTED, {"item": started}),
        (ITEM_COMPLETED, {"item": completed}),
    ]


def _plan_event(tool_input: dict[str, Any]) -> Event:
    todos = tool_input.get("todos")
    plan: list[dict[str, str]] = []
    if isinstance(todos, list):
        for todo in todos:
            if not isinstance(todo, dict):
                continue
            step = _string(todo.get("content"))
            if not step:
                continue
            plan.append({"step": step, "status": _plan_status(todo.get("status"))})
    return (TASK_PLAN_UPDATED, {"explanation": "", "plan": plan})


def _plan_status(status: Any) -> str:
    if status == "completed":
        return "completed"
    if status in {"in_progress", "inProgress"}:
        return "inProgress"
    return "pending"


def _file_paths(tool_input: dict[str, Any]) -> list[str]:
    for key in _FILE_PATH_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return [value]
    return []


def _split_mcp_name(name: str) -> tuple[str, str]:
    # ``mcp__<server>__<tool>`` -> ("<server>", "<tool>"). A tool name may itself
    # contain ``__`` (rare), so split the server off and keep the remainder.
    parts = name.split("__", 2)
    if len(parts) == 3:
        return parts[1], parts[2]
    if len(parts) == 2:
        return parts[1], ""
    return "", name


def _result_text(content: str | list[dict[str, Any]] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def _user_item_id(message: UserMessage, index: int) -> str:
    base = message.uuid or "user"
    return f"{base}:{index}"


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""
