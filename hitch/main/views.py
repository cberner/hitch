import shutil
from collections import Counter
from collections.abc import Iterator
from typing import Any

from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from openai_codex import AppServerConfig, Codex

# Friendly labels for non-message thread item types, surfaced in the per-session
# tool-call summary. Anything not in this map falls back to the raw type tag.
_NON_MESSAGE_LABELS = {
    "commandExecution": "Command execution",
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


def index(request: HttpRequest) -> HttpResponse:
    config = AppServerConfig(codex_bin=shutil.which("codex"))
    with Codex(config=config) as codex:
        sessions = codex.thread_list().data
    sessions = sorted(sessions, key=lambda s: s.updated_at, reverse=True)
    return render(request, "index.html", {"sessions": sessions})


def session(request: HttpRequest, session_id: str) -> HttpResponse:
    config = AppServerConfig(codex_bin=shutil.which("codex"))
    with Codex(config=config) as codex:
        thread = codex._client.thread_read(session_id, include_turns=True).thread
    entries = list(_render_entries(thread))
    return render(request, "session.html", {"thread": thread, "entries": entries})


def _render_entries(thread: Any) -> Iterator[dict[str, Any]]:
    """Walk every turn's items in order, emitting user/agent messages individually
    and collapsing runs of non-message items into a single tool-call summary.
    """
    pending: Counter[str] = Counter()

    def flush() -> dict[str, Any] | None:
        if not pending:
            return None
        entry = {
            "kind": "tool_calls",
            "total": sum(pending.values()),
            "summary": [
                {"label": label, "count": count}
                for label, count in sorted(pending.items())
            ],
        }
        pending.clear()
        return entry

    for turn in thread.turns:
        for thread_item in turn.items:
            item = thread_item.root
            item_type = item.type
            if item_type == "userMessage":
                flushed = flush()
                if flushed is not None:
                    yield flushed
                yield {"kind": "user", "text": _user_message_text(item)}
            elif item_type == "agentMessage":
                flushed = flush()
                if flushed is not None:
                    yield flushed
                yield {"kind": "agent", "text": item.text}
            else:
                pending[_NON_MESSAGE_LABELS.get(item_type, item_type)] += 1
    flushed = flush()
    if flushed is not None:
        yield flushed


def _user_message_text(item: Any) -> str:
    parts: list[str] = []
    for input_item in item.content:
        inner = input_item.root
        match inner.type:
            case "text":
                parts.append(inner.text)
            case "mention":
                parts.append(f"@{inner.name}")
            case "skill":
                parts.append(f"/{inner.name}")
            case "image":
                parts.append("[image]")
            case "localImage":
                parts.append(f"[image: {inner.path}]")
    return "\n".join(parts)
