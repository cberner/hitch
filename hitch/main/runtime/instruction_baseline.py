"""Recover explicitly identified developer instructions without guessing prose."""

import json
from pathlib import Path

from openai_codex import Codex
from openai_codex.generated.v2_all import ConfigReadResponse

_MAX_BASELINE_BYTES = 16 * 1024 * 1024


def configured_developer_instructions(codex: Codex, cwd: str) -> str:
    result = codex._client.request(
        "config/read", {"cwd": cwd, "includeLayers": False},
        response_model=ConfigReadResponse,
    )
    value = result.config.developer_instructions
    return value if isinstance(value, str) else ""


def recorded_developer_instructions(path: Path | None) -> str | None:
    """None means unknown, while an explicitly tagged empty string is known."""
    if path is None:
        return None
    baseline: str | None = None
    try:
        with path.open("rb") as stream:
            remaining = _MAX_BASELINE_BYTES
            while line := stream.readline(remaining + 1):
                remaining -= len(line)
                if remaining < 0:
                    return None
                event = json.loads(line)
                if not isinstance(event, dict):
                    return None
                payload = event.get("payload")
                if not isinstance(payload, dict):
                    continue
                if event.get("type") == "compacted" or payload.get("type") in {
                    "thread_rolled_back", "thread_rollback", "context_compacted",
                }:
                    baseline = None
                if event.get("type") != "response_item" or payload.get("role") != "developer":
                    continue
                if payload.get("type") != "message":
                    baseline = None
                    continue
                metadata = payload.get("internal_chat_message_metadata_passthrough")
                kinds = metadata.get("content_item_kinds") if isinstance(metadata, dict) else None
                content = payload.get("content")
                if not isinstance(kinds, list) or not isinstance(content, list) or len(kinds) != len(content):
                    baseline = None
                    continue
                parts: list[str] = []
                for kind, part in zip(kinds, content, strict=True):
                    if kind == "generic.developer_instructions":
                        if (
                            not isinstance(part, dict) or part.get("type") != "input_text"
                            or not isinstance(part.get("text"), str)
                        ):
                            return None
                        parts.append(part["text"])
                if parts:
                    baseline = "\n\n".join(parts)
    except (OSError, UnicodeError, ValueError):
        return None
    return baseline
