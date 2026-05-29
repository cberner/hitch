"""In-process MCP tools exposed to Claude Code worker sessions.

This mirrors ``hitch.main.codex_tools`` for the Claude backend. The Codex
app-server takes a list of ``dynamicTools`` specs and routes calls back over
its JSON-RPC transport; the Claude Agent SDK instead hosts tools in an
in-process MCP server (``create_sdk_mcp_server``) whose handlers run inside the
worker's asyncio loop. Both expose the same ``hitch.propose_session`` behaviour
backed by :func:`hitch.main.proposed_sessions.create_proposed_session`.
"""

from __future__ import annotations

import asyncio
from typing import Any

from claude_agent_sdk import McpSdkServerConfig, create_sdk_mcp_server, tool

from hitch.main.models import AutonomousGoal
from hitch.main.proposed_sessions import (
    ProposedSessionError,
    ProposedSessionInput,
    create_proposed_session,
)

HITCH_MCP_SERVER_NAME = "hitch"
PROPOSE_SESSION_TOOL = "propose_session"
# Claude addresses in-process MCP tools as ``mcp__<server>__<tool>``; callers
# add this to ``allowed_tools`` so the model may invoke it without an approval
# prompt.
PROPOSE_SESSION_TOOL_NAME = f"mcp__{HITCH_MCP_SERVER_NAME}__{PROPOSE_SESSION_TOOL}"

_PROPOSE_SESSION_DESCRIPTION = (
    "Create a Hitch inbox proposal for a follow-up coding session. Use this "
    "only when the user asks you to create proposed sessions or session "
    "instructions explicitly authorize creating proposals."
)
_PROPOSE_SESSION_SCHEMA: dict[str, Any] = {
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
}


def build_hitch_mcp_server(*, cwd: str, thread_id: str) -> McpSdkServerConfig:
    """Return an SDK MCP server bound to one session's cwd and thread id.

    The cwd/thread id are captured here (the SDK tool handler only receives
    the call arguments) so proposals land in the right project and link back to
    the originating session.
    """

    @tool(PROPOSE_SESSION_TOOL, _PROPOSE_SESSION_DESCRIPTION, _PROPOSE_SESSION_SCHEMA)
    async def _propose_session(args: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(_run_propose_session, args, cwd, thread_id)

    return create_sdk_mcp_server(HITCH_MCP_SERVER_NAME, tools=[_propose_session])


def _run_propose_session(args: dict[str, Any], cwd: str, thread_id: str) -> dict[str, Any]:
    """Synchronous ORM body, run in a worker thread off the event loop."""
    from django.db import connection

    try:
        relevant_files = args.get("relevant_files", [])
        if not isinstance(relevant_files, list):
            return _tool_response("relevant_files must be a list", is_error=True)
        proposal = create_proposed_session(
            ProposedSessionInput(
                title=_string_arg(args, "title"),
                summary=_string_arg(args, "summary"),
                prompt=_string_arg(args, "prompt"),
                cwd=cwd,
                relevant_files=[item for item in relevant_files if isinstance(item, str)],
                confidence=_string_arg(
                    args, "confidence", default=AutonomousGoal.CONFIDENCE_MEDIUM
                ),
                source_thread_id=thread_id,
            )
        )
    except ProposedSessionError as exc:
        return _tool_response(str(exc), is_error=True)
    finally:
        connection.close()
    return _tool_response(
        f"Created proposed session #{proposal.pk}: {proposal.title}", is_error=False
    )


def _string_arg(args: dict[str, Any], name: str, *, default: str = "") -> str:
    value = args.get(name, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ProposedSessionError(f"{name} must be a string")
    return value


def _tool_response(text: str, *, is_error: bool) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "is_error": is_error}
