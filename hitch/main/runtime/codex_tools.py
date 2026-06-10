"""Hitch dynamic tools exposed to Codex app-server sessions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from django.db import connection

from hitch.main.goals.proposed_sessions import (
    ProposedSessionError,
    ProposedSessionInput,
    create_proposed_session,
)
from hitch.main.models import AutonomousGoal

_TOOL_CALL_METHOD = "item/tool/call"
_HITCH_NAMESPACE = "hitch"
_PROPOSE_SESSION_TOOL = "propose_session"


@dataclass(frozen=True)
class ToolContext:
    cwd: str
    thread_id: str


@dataclass(frozen=True)
class HitchTool:
    namespace: str
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any], ToolContext], str]


def is_dynamic_tool_call(method: str) -> bool:
    return method == _TOOL_CALL_METHOD


def registered_dynamic_tool_specs() -> list[dict[str, Any]]:
    return [
        {
            "namespace": tool.namespace,
            "name": tool.name,
            "description": tool.description,
            "inputSchema": tool.input_schema,
            "deferLoading": False,
        }
        for tool in _TOOLS.values()
    ]


def handle_dynamic_tool_call(
    params: dict[str, Any] | None, context: ToolContext
) -> dict[str, Any]:
    if not isinstance(params, dict):
        return _tool_response("tool call params are required", success=False)
    namespace = params.get("namespace")
    tool_name = params.get("tool")
    if namespace is None:
        namespace = _HITCH_NAMESPACE
    if not isinstance(namespace, str) or not isinstance(tool_name, str):
        return _tool_response("tool namespace and name are required", success=False)
    tool = _TOOLS.get((namespace, tool_name))
    if tool is None:
        return _tool_response(f"unknown Hitch tool: {namespace}.{tool_name}", success=False)
    arguments = params.get("arguments")
    if not isinstance(arguments, dict):
        return _tool_response("tool arguments must be an object", success=False)
    try:
        try:
            message = tool.handler(arguments, context)
        finally:
            connection.close()
    except ProposedSessionError as exc:
        return _tool_response(str(exc), success=False)
    return _tool_response(message, success=True)


def _handle_propose_session(arguments: dict[str, Any], context: ToolContext) -> str:
    title = _string_arg(arguments, "title")
    summary = _string_arg(arguments, "summary")
    prompt = _string_arg(arguments, "prompt")
    relevant_files = arguments.get("relevant_files", [])
    if not isinstance(relevant_files, list):
        raise ProposedSessionError("relevant_files must be a list")
    proposal = create_proposed_session(
        ProposedSessionInput(
            title=title,
            summary=summary,
            prompt=prompt,
            cwd=context.cwd,
            relevant_files=[item for item in relevant_files if isinstance(item, str)],
            confidence=_string_arg(
                arguments, "confidence", default=AutonomousGoal.CONFIDENCE_MEDIUM
            ),
            source_thread_id=context.thread_id,
        )
    )
    return f"Created proposed session #{proposal.pk}: {proposal.title}"


def _string_arg(arguments: dict[str, Any], name: str, *, default: str = "") -> str:
    value = arguments.get(name, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ProposedSessionError(f"{name} must be a string")
    return value


def _tool_response(text: str, *, success: bool) -> dict[str, Any]:
    return {
        "contentItems": [{"type": "inputText", "text": text}],
        "success": success,
    }


_TOOLS: dict[tuple[str, str], HitchTool] = {
    (_HITCH_NAMESPACE, _PROPOSE_SESSION_TOOL): HitchTool(
        namespace=_HITCH_NAMESPACE,
        name=_PROPOSE_SESSION_TOOL,
        description=(
            "Create a Hitch inbox proposal for a follow-up coding session. Use this "
            "only when the user asks you to create proposed sessions or session "
            "instructions explicitly authorize creating proposals."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Concise title for the proposed session.",
                },
                "summary": {
                    "type": "string",
                    "description": "User-facing summary of why this session is useful.",
                },
                "prompt": {
                    "type": "string",
                    "description": "Concrete prompt to start the proposed session.",
                },
                "relevant_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Repository files likely relevant to the session.",
                },
                "confidence": {
                    "type": "string",
                    "enum": [
                        AutonomousGoal.CONFIDENCE_MEDIUM,
                        AutonomousGoal.CONFIDENCE_HIGH,
                        AutonomousGoal.CONFIDENCE_VERY_HIGH,
                    ],
                },
            },
            "required": ["title", "summary", "prompt"],
            "additionalProperties": False,
        },
        handler=_handle_propose_session,
    )
}
