"""Adapt Codex MCP tool confirmations to Hitch's one-call approvals."""

from typing import Any

MCP_ELICITATION_METHOD = "mcpServer/elicitation/request"


def is_tool_confirmation(params: dict[str, Any]) -> bool:
    meta = params.get("_meta")
    schema = params.get("requestedSchema")
    # Data-entry and authentication elicitations need their own UI. Never
    # mistake those for permission to execute a tool, even in approve-all mode.
    return (
        params.get("mode") == "form"
        and isinstance(meta, dict)
        and meta.get("codex_approval_kind") == "mcp_tool_call"
        and isinstance(schema, dict)
        and schema.get("type") == "object"
        and schema.get("properties") == {}
        and not schema.get("required")
    )


def elicitation_response(decision: Any) -> dict[str, Any]:
    action = decision if decision in ("accept", "decline", "cancel") else "decline"
    return {"action": action, "content": {} if action == "accept" else None, "_meta": None}
